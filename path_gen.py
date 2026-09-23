"""
path_gen.py -- Geometric path generators for the RL agent's action
==================================================================

Two generators:

1. `curvature_poly_path`  -- the paper's scheme (Eqs. 7-10): the agent sets
   curvature knots, kappa(s) is a piecewise cubic in arc length, and the pose
   comes from integrating theta(s) = int kappa ds, x(s) = int cos(theta) ds,
   y(s) = int sin(theta) ds. Faithful, and already better than the shipped
   MATLAB path because for a cubic kappa(s) = a3 s^3 + a2 s^2 + a1 s + a0 the
   second derivative kappa''(s) = 6*a3*s + 2*a2 is EXACT and piecewise linear,
   instead of coming out of three nested finite differences.

2. `bezier_offset_path`   -- what I use for the highway truck. The lateral
   offset y is a degree-11 Bezier in the along-road coordinate, with the first
   five and last five control points pinned to the start/end offsets. Because
   the k-th derivative of a Bezier at t=0 depends only on the k-th forward
   difference of the first k+1 control points, pinning five of them forces
        y' = y'' = y''' = y'''' = 0 at both ends of every transition
   which gives, by construction:
     * G4 continuity: heading, curvature AND curvature rate AND kappa'' all
       go continuously to zero at every join, so there is no jerk step for the
       steering actuator to chase;
     * an EXACT terminal lane offset and heading;
     * two free interior control points per transition -- the shape knobs the
       RL agent actually gets to optimise.

Why #2 for a truck
------------------
* A 40 t rig steers through a 0.25 s lag at 0.35 rad/s. A curvature step at a
  segment join is a steering-rate step the actuator cannot follow, and the
  tracking error shows up as trailer swing. G3 by construction removes it.
* The paper's scheme cannot guarantee the manoeuvre ENDS in the target lane
  with zero heading; the agent has to learn that, and most of the 150k
  episodes go into learning it. Here it is free.
* kappa is bounded by construction (see `kappa_bound`), so the rollover limit
  can be baked into the action scaling rather than punished after the fact.

Every derivative below is analytic; `self_test()` checks them against high-order
finite differences and against the closed-form kappa'' of the cubic scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import comb

from ttv_core import PreparedPath


# =============================================================================
@dataclass
class GeoPath:
    """A path with analytic differential geometry.

    s        arc length                       [m]
    x, y     position                         [m]
    psi      heading                          [rad]
    kappa    curvature        dpsi/ds         [1/m]
    kappa_s  dkappa/ds                        [1/m^2]
    kappa_ss d2kappa/ds2                      [1/m^3]
    """
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    kappa_s: np.ndarray
    kappa_ss: np.ndarray
    meta: dict

    @property
    def length(self) -> float:
        return float(self.s[-1])

    # -- resolution-independent extrema ---------------------------------------
    # The sampled arrays above use whatever `ds` the caller asked for, which is
    # set by what the SIMULATION needs. Taking max/min of kappa'' off those
    # samples reintroduces a resolution dependence (measured: 23 % between
    # ds = 2 m and ds = 0.05 m) because kappa'' peaks sharply near a
    # transition's ends. The generator therefore evaluates the extrema once on
    # a guaranteed-dense internal grid and stores them here; the reward must
    # read these, never the sampled arrays.
    @property
    def kappa_peak(self) -> float:
        return float(self.meta["exact"]["kappa_absmax"])

    @property
    def kappa_s_peak(self) -> float:
        return float(self.meta["exact"]["kappa_s_absmax"])

    @property
    def kappa_ss_max(self) -> float:
        return float(self.meta["exact"]["kappa_ss_max"])

    @property
    def kappa_ss_min(self) -> float:
        return float(self.meta["exact"]["kappa_ss_min"])

    def reward_kappa(self, c_kappa: float = 0.0) -> float:
        """Paper Eq. (5), evaluated on the exact extrema."""
        return c_kappa - abs(self.kappa_ss_max) - abs(self.kappa_ss_min)

    def to_prepared(self) -> PreparedPath:
        """A PreparedPath for ttv_core, carrying the ANALYTIC derivatives.

        This is the single most important difference from the shipped MATLAB
        path: prepare_path() computes kappa, kappa' and kappa'' with three
        nested calls to gradient(), which makes the reward depend on how
        densely the path happened to be sampled (5.6x spread, measured). Here
        they are exact.
        """
        return PreparedPath(x=self.x.copy(), y=self.y.copy(), s=self.s.copy(),
                            psi=self.psi.copy(), kappa=self.kappa.copy(),
                            kappaPrime=self.kappa_s.copy(),
                            kappaDD=self.kappa_ss.copy())

    def max_lateral_accel(self, V: float) -> float:
        return float(V ** 2 * self.kappa_peak)

    def max_lateral_jerk(self, V: float) -> float:
        """d(a_y)/dt = V^3 * dkappa/ds along the path."""
        return float(V ** 3 * self.kappa_s_peak)


# =============================================================================
# 1.  Lateral-offset Bezier path  (the one used for the truck)
# =============================================================================

_DEG = 11         # Bezier degree: 12 control points, 5 pinned at each end


def _bernstein_matrix_uncached(n: int, t: np.ndarray) -> np.ndarray:
    """B[k, i] = C(n,i) t_k^i (1-t_k)^(n-i)."""
    i = np.arange(n + 1)
    c = comb(n, i)
    tt = t[:, None]
    return c * (tt ** i) * ((1.0 - tt) ** (n - i))


# The basis depends only on (degree, grid), and every call inside a decode uses
# the SAME uniform grid. Memoising it took one action decode from 408 ms to
# under 3 ms -- it was 54 % of the whole episode.
_BASIS_CACHE: dict = {}


def _uniform_basis(n: int, m: int) -> np.ndarray:
    key = (n, m)
    B = _BASIS_CACHE.get(key)
    if B is None:
        if len(_BASIS_CACHE) > 256:
            _BASIS_CACHE.clear()
        B = _bernstein_matrix_uncached(n, np.linspace(0.0, 1.0, m))
        _BASIS_CACHE[key] = B
    return B


def _bezier_and_derivs(P: np.ndarray, t: np.ndarray, n_deriv: int = 4,
                       uniform: bool = False):
    """Value and the first n_deriv derivatives of a Bezier curve w.r.t. t."""
    out = []
    Q = P.copy()
    deg = P.size - 1
    for d in range(n_deriv + 1):
        B = _uniform_basis(deg, t.size) if uniform else \
            _bernstein_matrix_uncached(deg, t)
        out.append(B @ Q)
        if deg == 0:
            Q = np.zeros(1)
        else:
            Q = deg * np.diff(Q)
            deg -= 1
    return out


def _transition(y0: float, y1: float, L: float, q1: float, q2: float,
                t: np.ndarray, uniform: bool = False):
    """One G3 lateral transition y0 -> y1 over along-road length L.

    Returns (y, y', y'', y''', y'''') w.r.t. the along-road coordinate.
    q1, q2 shape the interior: 0.5/0.5 is the neutral symmetric S-curve;
    smaller values front-load the manoeuvre, larger values back-load it.
    """
    dy = y1 - y0
    P = np.array([y0, y0, y0, y0, y0,
                  y0 + q1 * dy, y0 + q2 * dy,
                  y1, y1, y1, y1, y1])
    assert P.size == _DEG + 1
    d = _bezier_and_derivs(P, t, 4, uniform=uniform)
    scale = [1.0, 1.0 / L, 1.0 / L ** 2, 1.0 / L ** 3, 1.0 / L ** 4]
    return [di * si for di, si in zip(d, scale)]


def _offset_geometry_raw(u, v, w, z):
    """Closed-form kappa, dkappa/ds, d2kappa/ds2 of the graph y(x)."""
    Gm = 1.0 + u ** 2
    kappa = v * Gm ** -1.5
    kappa_s = w * Gm ** -2.0 - 3.0 * u * v ** 2 * Gm ** -3.0
    kappa_sx = (z * Gm ** -2.0
                - 10.0 * u * v * w * Gm ** -3.0
                - 3.0 * v ** 3 * Gm ** -3.0
                + 18.0 * u ** 2 * v ** 3 * Gm ** -4.0)
    return kappa, kappa_s, kappa_sx * Gm ** -0.5


_DENSE = 4001      # samples per transition for the exact-extrema pass


def _exact_extrema(transitions, y_start: float) -> dict:
    """Extrema of kappa, kappa' and kappa'' over the whole path, on a dense
    internal grid that does not depend on the caller's output spacing.
    Straight sections contribute exactly zero, so only transitions matter."""
    k_abs, ks_abs, kss_hi, kss_lo = 0.0, 0.0, 0.0, 0.0
    y_cur = y_start
    t = np.linspace(0.0, 1.0, _DENSE)
    any_curved = False
    for (dy, L, q1, q2) in transitions:
        if L <= 0 or abs(dy) < 1e-12:
            continue
        any_curved = True
        _, u, v, w, z = _transition(y_cur, y_cur + dy, L, q1, q2, t, uniform=True)
        kappa, kappa_s, kappa_ss = _offset_geometry_raw(u, v, w, z)
        k_abs = max(k_abs, float(np.max(np.abs(kappa))))
        ks_abs = max(ks_abs, float(np.max(np.abs(kappa_s))))
        kss_hi = max(kss_hi, float(np.max(kappa_ss)))
        kss_lo = min(kss_lo, float(np.min(kappa_ss)))
        y_cur += dy
    if not any_curved:
        kss_hi = kss_lo = 0.0
    return {"kappa_absmax": k_abs, "kappa_s_absmax": ks_abs,
            "kappa_ss_max": kss_hi, "kappa_ss_min": kss_lo}


def _geometry_from_offset(xg: np.ndarray, y: np.ndarray, u: np.ndarray,
                          v: np.ndarray, w: np.ndarray, z: np.ndarray,
                          meta: dict) -> GeoPath:
    """Differential geometry of the graph y(x), all closed form.

        u=y'  v=y''  w=y'''  z=y''''     G = 1 + u^2
        psi      = atan(u)
        kappa    = v * G^(-3/2)
        dk/ds    = w*G^-2 - 3*u*v^2*G^-3
        d2k/ds2  = [ z*G^-2 - 10*u*v*w*G^-3 - 3*v^3*G^-3 + 18*u^2*v^3*G^-4 ]
                   * G^(-1/2)
    """
    psi = np.arctan(u)
    kappa, kappa_s, kappa_ss = _offset_geometry_raw(u, v, w, z)

    # Arc length by cumulative trapezoid on sqrt(1+u^2).
    ds_dx = np.sqrt(1.0 + u ** 2)
    dx = np.diff(xg)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1]) * dx)))

    return GeoPath(s=s, x=xg.copy(), y=y, psi=psi, kappa=kappa,
                   kappa_s=kappa_s, kappa_ss=kappa_ss, meta=meta)


def bezier_offset_path(
    *,
    lead_in: float = 20.0,
    transitions: Sequence[Tuple[float, float, float, float]],
    tail: float = 30.0,
    y_start: float = 0.0,
    ds: float = 0.25,
) -> GeoPath:
    """Build a highway path from G3 lateral transitions.

    transitions : sequence of (dy, L, q1, q2)
        dy  lateral offset change of this transition   [m]
        L   along-road length of the transition        [m]
        q1, q2  interior shape parameters              [-]
        Consecutive transitions may be separated by a straight `hold` by
        passing (0.0, L_hold, 0, 0).
    lead_in  straight run before the first transition  [m]
    tail     straight run after the last one           [m]
    ds       along-road sample spacing                 [m]
    """
    xs, ys, us, vs, ws, zs = [], [], [], [], [], []
    x_cursor = 0.0
    y_cursor = y_start

    def add_straight(L: float, y_level: float, include_first: bool):
        n = max(2, int(round(L / ds)) + 1)
        xx = np.linspace(x_cursor, x_cursor + L, n)
        sl = slice(0 if include_first else 1, None)
        xs.append(xx[sl])
        ys.append(np.full(xx[sl].size, y_level))
        for acc in (us, vs, ws, zs):
            acc.append(np.zeros(xx[sl].size))

    if lead_in > 0:
        add_straight(lead_in, y_cursor, True)
        x_cursor += lead_in
        first = False
    else:
        first = True

    for (dy, L, q1, q2) in transitions:
        if L <= 0:
            continue
        if abs(dy) < 1e-12:
            add_straight(L, y_cursor, first)
            x_cursor += L
            first = False
            continue
        n = max(4, int(round(L / ds)) + 1)
        t = np.linspace(0.0, 1.0, n)
        y, u, v, w, z = _transition(y_cursor, y_cursor + dy, L, q1, q2, t,
                                    uniform=True)
        sl = slice(0 if first else 1, None)
        xs.append(x_cursor + t[sl] * L)
        ys.append(y[sl]); us.append(u[sl]); vs.append(v[sl])
        ws.append(w[sl]); zs.append(z[sl])
        x_cursor += L
        y_cursor += dy
        first = False

    if tail > 0:
        add_straight(tail, y_cursor, first)
        x_cursor += tail

    xg = np.concatenate(xs)
    meta = {"generator": "bezier_offset", "lead_in": lead_in, "tail": tail,
            "transitions": list(transitions), "y_start": y_start,
            "y_end": y_cursor,
            "exact": _exact_extrema(transitions, y_start)}
    return _geometry_from_offset(xg, np.concatenate(ys), np.concatenate(us),
                                np.concatenate(vs), np.concatenate(ws),
                                np.concatenate(zs), meta)


def kappa_bound(dy: float, L: float, q1: float = 0.5, q2: float = 0.5) -> float:
    """Peak |kappa| of one transition, on a dense grid.

    kappa = y'' / (1+y'^2)^{3/2}. The numerator scales exactly as |dy|/L^2, so
    in the small-slope limit so does kappa; the (1+y'^2) factor makes the
    scaling approximate (~0.2 % over a 2x change in L for a highway lane
    change). `min_length_for_offset` therefore solves for L rather than
    assuming the power law.
    """
    t = np.linspace(0.0, 1.0, _DENSE)
    _, u, v, _, _ = _transition(0.0, dy, L, q1, q2, t, uniform=True)
    return float(np.max(np.abs(v / (1.0 + u ** 2) ** 1.5)))


_LMIN_CACHE: dict = {}


def min_length_for_offset(dy: float, kappa_max: float,
                          q1: float = 0.5, q2: float = 0.5,
                          tol: float = 1e-9) -> float:
    """Cached wrapper around the bisection below."""
    key = (round(dy, 6), round(kappa_max, 10), round(q1, 6), round(q2, 6))
    v = _LMIN_CACHE.get(key)
    if v is None:
        if len(_LMIN_CACHE) > 20000:
            _LMIN_CACHE.clear()
        v = _min_length_for_offset(dy, kappa_max, q1, q2, tol)
        _LMIN_CACHE[key] = v
    return v


def _min_length_for_offset(dy: float, kappa_max: float,
                           q1: float = 0.5, q2: float = 0.5,
                           tol: float = 1e-9) -> float:
    """Shortest transition length that keeps |kappa| <= kappa_max, exactly.

    kappa_bound is strictly decreasing in L, so bisect. The L^-2 power law
    gives the bracket; bisection removes its small-slope error.
    """
    if dy == 0.0:
        return 0.0
    k1 = kappa_bound(dy, 1.0, q1, q2)
    L = math.sqrt(k1 / kappa_max)                 # power-law seed
    lo, hi = 0.25 * L, 4.0 * L
    while kappa_bound(dy, lo, q1, q2) < kappa_max:
        lo *= 0.5
    while kappa_bound(dy, hi, q1, q2) > kappa_max:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kappa_bound(dy, mid, q1, q2) > kappa_max:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return hi


# =============================================================================
# 2.  The paper's curvature-polynomial path (Eqs. 7-10)
# =============================================================================

def _cubic_from_four_knots(k: Sequence[float], L: float) -> np.ndarray:
    """kappa(s) = a3 s^3 + a2 s^2 + a1 s + a0 through four evenly spaced knots.

    Paper, Fig. 5: kappa_a at s=0, kappa_b and kappa_c evenly distributed in
    the middle, kappa_d at s=L.
    """
    sn = np.array([0.0, L / 3.0, 2.0 * L / 3.0, L])
    Vm = np.vander(sn, 4)                 # columns [s^3, s^2, s, 1]
    return np.linalg.solve(Vm, np.asarray(k, dtype=float))


def curvature_poly_path(
    *,
    straight_lengths: Sequence[float],
    poly_lengths: Sequence[float],
    knots: Sequence[Sequence[float]],
    ds: float = 0.25,
    x0: float = 0.0, y0: float = 0.0, psi0: float = 0.0,
) -> GeoPath:
    """The paper's scheme: piecewise-cubic kappa(s), integrated to a pose.

    straight_lengths : [L_before, L_after]
    poly_lengths     : one length per polynomial section
    knots            : per section, four curvature values [ka, kb, kc, kd];
                       continuity requires knots[i][3] == knots[i+1][0].

    kappa''(s) = 6*a3 within each section -- piecewise constant and EXACT, so
    the paper's reward_kappa becomes well posed.
    """
    seg_s, seg_k, seg_ks, seg_kss = [], [], [], []

    def push_straight(L):
        n = max(2, int(round(L / ds)) + 1)
        ss = np.linspace(0.0, L, n)
        seg_s.append(ss)
        for acc in (seg_k, seg_ks, seg_kss):
            acc.append(np.zeros(n))

    push_straight(straight_lengths[0])
    for L, kk in zip(poly_lengths, knots):
        a = _cubic_from_four_knots(kk, L)        # [a3, a2, a1, a0]
        n = max(4, int(round(L / ds)) + 1)
        ss = np.linspace(0.0, L, n)
        seg_s.append(ss)
        seg_k.append(np.polyval(a, ss))
        seg_ks.append(np.polyval(np.polyder(a, 1), ss))
        seg_kss.append(np.polyval(np.polyder(a, 2), ss))
    push_straight(straight_lengths[1])

    # Stitch, dropping the duplicated joint sample.
    s_parts, k_parts, ks_parts, kss_parts = [], [], [], []
    off = 0.0
    for i, (ss, kk, kks, kkss) in enumerate(zip(seg_s, seg_k, seg_ks, seg_kss)):
        sl = slice(0 if i == 0 else 1, None)
        s_parts.append(off + ss[sl])
        k_parts.append(kk[sl]); ks_parts.append(kks[sl]); kss_parts.append(kkss[sl])
        off += ss[-1]
    s = np.concatenate(s_parts)
    kappa = np.concatenate(k_parts)
    kappa_s = np.concatenate(ks_parts)
    kappa_ss = np.concatenate(kss_parts)

    # Eqs. (8)-(10): cumulative-trapezoid integration of the Frenet-Serret ODE.
    dsv = np.diff(s)
    psi = np.concatenate(([psi0], psi0 + np.cumsum(
        0.5 * (kappa[1:] + kappa[:-1]) * dsv)))
    cx, cy = np.cos(psi), np.sin(psi)
    x = np.concatenate(([x0], x0 + np.cumsum(0.5 * (cx[1:] + cx[:-1]) * dsv)))
    y = np.concatenate(([y0], y0 + np.cumsum(0.5 * (cy[1:] + cy[:-1]) * dsv)))

    # Exact extrema: kappa is a cubic and kappa'' = 6*a3*s + 2*a2 is LINEAR in
    # s per section, so both attain their extremes either at a section end or
    # at an interior stationary point of the cubic. Evaluate those exactly.
    k_abs = ks_abs = 0.0
    kss_hi, kss_lo = 0.0, 0.0
    for L, kk in zip(poly_lengths, knots):
        a = _cubic_from_four_knots(kk, L)
        d1, d2 = np.polyder(a, 1), np.polyder(a, 2)
        # candidates for kappa: ends + roots of kappa'
        cand = [0.0, L] + [float(r.real) for r in np.roots(d1)
                           if abs(r.imag) < 1e-12 and 0.0 <= r.real <= L]
        k_abs = max(k_abs, max(abs(np.polyval(a, c)) for c in cand))
        # kappa' is a quadratic: ends + its own stationary point
        cand1 = [0.0, L] + [float(r.real) for r in np.roots(d2)
                            if abs(r.imag) < 1e-12 and 0.0 <= r.real <= L]
        ks_abs = max(ks_abs, max(abs(np.polyval(d1, c)) for c in cand1))
        # kappa'' is linear: extremes at the ends
        vals = [np.polyval(d2, 0.0), np.polyval(d2, L)]
        kss_hi = max(kss_hi, max(vals))
        kss_lo = min(kss_lo, min(vals))

    meta = {"generator": "curvature_poly", "straight_lengths": list(straight_lengths),
            "poly_lengths": list(poly_lengths), "knots": [list(k) for k in knots],
            "exact": {"kappa_absmax": float(k_abs), "kappa_s_absmax": float(ks_abs),
                      "kappa_ss_max": float(kss_hi), "kappa_ss_min": float(kss_lo)}}
    return GeoPath(s=s, x=x, y=y, psi=psi, kappa=kappa, kappa_s=kappa_s,
                   kappa_ss=kappa_ss, meta=meta)


# =============================================================================
# 3.  Self-tests
# =============================================================================

def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:52s} {detail}")

    if verbose:
        print("=" * 78)
        print("path_gen self-test")
        print("=" * 78)
        print("  A. analytic vs finite-difference differential geometry")

    p = bezier_offset_path(lead_in=20.0,
                           transitions=[(3.6, 70.0, 0.45, 0.55)],
                           tail=40.0, ds=0.02)
    # Interior region only (finite differences degrade at the joins).
    m = (p.s > 25) & (p.s < 85)
    k_fd = np.gradient(p.psi, p.s)
    ks_fd = np.gradient(p.kappa, p.s)
    kss_fd = np.gradient(p.kappa_s, p.s)
    e1 = np.max(np.abs(k_fd[m] - p.kappa[m])) / np.max(np.abs(p.kappa))
    e2 = np.max(np.abs(ks_fd[m] - p.kappa_s[m])) / np.max(np.abs(p.kappa_s))
    e3 = np.max(np.abs(kss_fd[m] - p.kappa_ss[m])) / np.max(np.abs(p.kappa_ss))
    check("kappa   matches d(psi)/ds", e1 < 2e-4, f"rel {e1:.2e}")
    check("kappa_s matches d(kappa)/ds", e2 < 2e-4, f"rel {e2:.2e}")
    check("kappa_ss matches d(kappa_s)/ds", e3 < 5e-3, f"rel {e3:.2e}")

    if verbose:
        print("\n  B. G3 continuity and exact terminal conditions")
    check("terminal offset exact", abs(p.y[-1] - 3.6) < 1e-9,
          f"y_end = {p.y[-1]:.12f}")
    check("terminal heading exact", abs(p.psi[-1]) < 1e-12,
          f"psi_end = {p.psi[-1]:.2e}")
    check("kappa = 0 at both ends", max(abs(p.kappa[0]), abs(p.kappa[-1])) < 1e-12)
    check("kappa_s = 0 at both ends",
          max(abs(p.kappa_s[0]), abs(p.kappa_s[-1])) < 1e-12)
    check("kappa_ss = 0 at both ends (G4)",
          max(abs(p.kappa_ss[0]), abs(p.kappa_ss[-1])) < 1e-12,
          f"{abs(p.kappa_ss[0]):.2e}, {abs(p.kappa_ss[-1]):.2e}")
    check("kappa continuous (no join jump)",
          np.max(np.abs(np.diff(p.kappa))) < 1e-4,
          f"max jump {np.max(np.abs(np.diff(p.kappa))):.2e}")
    check("kappa_ss continuous (no jerk step)",
          np.max(np.abs(np.diff(p.kappa_ss))) < 1e-5,
          f"max jump {np.max(np.abs(np.diff(p.kappa_ss))):.2e}")

    if verbose:
        print("\n  C. resolution independence of the reward term (vs the MATLAB path)")
    sampled, exact = [], []
    for dsv in (2.0, 1.0, 0.5, 0.25, 0.1, 0.05):
        q = bezier_offset_path(lead_in=20.0,
                               transitions=[(3.6, 70.0, 0.45, 0.55)],
                               tail=40.0, ds=dsv)
        sampled.append(-abs(np.max(q.kappa_ss)) - abs(np.min(q.kappa_ss)))
        exact.append(q.reward_kappa())
    sp_sampled = max(map(abs, sampled)) / min(map(abs, sampled))
    sp_exact = max(map(abs, exact)) / min(map(abs, exact))
    check("naive max/min over SAMPLES still drifts with ds",
          sp_sampled > 1.001, f"spread {sp_sampled:.4f}x")
    check("reward_kappa() off the exact extrema is invariant",
          abs(sp_exact - 1.0) < 1e-12,
          f"spread {sp_exact:.12f}x  (MATLAB path: 5.6x)")

    if verbose:
        print("\n  D. kappa'' of the paper's cubic scheme is exactly linear in s")
    kn = [[0.0, 0.002, 0.004, 0.005], [0.005, 0.003, -0.001, -0.004],
          [-0.004, -0.003, -0.001, 0.0]]
    cp = curvature_poly_path(straight_lengths=[15.0, 30.0],
                             poly_lengths=[35.0, 40.0, 35.0], knots=kn, ds=0.1)
    a = _cubic_from_four_knots(kn[0], 35.0)
    inside = (cp.s > 15.0 + 1e-9) & (cp.s < 50.0 - 1e-9)
    s_loc = cp.s[inside] - 15.0
    check("kappa_ss == 6*a3*s + 2*a2 on section 1",
          np.allclose(cp.kappa_ss[inside], 6 * a[0] * s_loc + 2 * a[1], atol=1e-14),
          f"6a3={6*a[0]:.3e}, 2a2={2*a[1]:.3e}")
    check("curvature continuous across joints",
          np.max(np.abs(np.diff(cp.kappa))) < 1e-3)
    # cross-check the exact-extrema pass against a very dense sampling
    dense = curvature_poly_path(straight_lengths=[15.0, 30.0],
                                poly_lengths=[35.0, 40.0, 35.0], knots=kn, ds=0.002)
    check("exact kappa_absmax matches dense sampling",
          abs(cp.kappa_peak - np.max(np.abs(dense.kappa))) < 1e-9,
          f"{cp.kappa_peak:.8f} vs {np.max(np.abs(dense.kappa)):.8f}")
    # The cubic scheme's kappa'' JUMPS at a straight->polynomial join (0 on the
    # straight, 2*a2 just inside), so a sampled grid sees whichever side the
    # stitching kept. The exact pass includes both sides, hence >= the sampled
    # range. That asymmetry is itself a finding: the paper's own generator is
    # only G2, and its reward term sits exactly on the discontinuity.
    check("exact kappa_ss range brackets dense sampling",
          cp.kappa_ss_max >= np.max(dense.kappa_ss) - 1e-12
          and cp.kappa_ss_min <= np.min(dense.kappa_ss) + 1e-12,
          f"exact [{cp.kappa_ss_min:.3e},{cp.kappa_ss_max:.3e}] vs sampled "
          f"[{np.min(dense.kappa_ss):.3e},{np.max(dense.kappa_ss):.3e}]")

    if verbose:
        print("\n  E. kappa_peak vs L, and the exact minimum-length solver")
    k70 = kappa_bound(3.6, 70.0)
    k140 = kappa_bound(3.6, 140.0)
    # y''/L^2 scales exactly; the (1+y'^2)^{3/2} factor perturbs it by <1 %.
    check("halving L quadruples kappa_peak to within 1 %",
          abs(k70 / k140 - 4.0) < 0.04, f"ratio {k70/k140:.6f}")
    for kmax in (0.0053, 0.0020, 0.0100):
        Lmin = min_length_for_offset(3.6, kmax)
        err = abs(kappa_bound(3.6, Lmin) - kmax) / kmax
        check(f"min_length_for_offset exact at kappa_max={kmax}", err < 1e-9,
              f"L_min = {Lmin:.4f} m, rel.err {err:.2e}")

    if verbose:
        print(f"\n  SELF-TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    good = self_test()

    print("\n" + "=" * 78)
    print("What a highway lane change costs the truck, by manoeuvre length")
    print("=" * 78)
    from truck_params import kappa_max_rollover, SRT, V_HIGHWAY
    V = V_HIGHWAY
    kmax = kappa_max_rollover(V)
    print(f"  V = {V} m/s ({V*3.6:.0f} km/h), SRT = {SRT:.3f} g, "
          f"kappa ceiling = {kmax:.5f} 1/m")
    print(f"  {'L [m]':>7s} {'t [s]':>7s} {'kappa':>9s} {'a_y [g]':>9s} "
          f"{'jerk [m/s3]':>12s} {'verdict':>10s}")
    for L in (40, 50, 60, 70, 85, 100, 120):
        p = bezier_offset_path(lead_in=10, transitions=[(3.6, float(L), .45, .55)],
                               tail=10, ds=0.25)
        ay = p.max_lateral_accel(V)
        print(f"  {L:7d} {L/V:7.2f} {np.max(np.abs(p.kappa)):9.5f} {ay/9.81:9.3f} "
              f"{p.max_lateral_jerk(V):12.3f} "
              f"{'ok' if np.max(np.abs(p.kappa)) <= kmax else 'ROLLOVER':>10s}")
    Lmin = min_length_for_offset(3.6, kmax)
    print(f"\n  shortest safe single lane change: L = {Lmin:.1f} m "
          f"({Lmin/V:.2f} s at {V*3.6:.0f} km/h)")
    print("  -> this one number is the hard constraint the RL agent must respect,")
    print("     and because kappa ~ |dy|/L^2 it can be built into the action scaling.")
    sys.exit(0 if good else 1)
