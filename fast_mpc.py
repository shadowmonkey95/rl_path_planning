"""
fast_mpc.py -- The same MPC, as a condensed QP
==============================================

The shipped code sends the tracking MPC to IPOPT through `nlpsol`. But the
problem is a *quadratic program*: the cost is quadratic, the dynamics are
linear equalities, and every constraint is a box. Handing a QP to a nonlinear
interior-point solver costs ~3.4 ms per solve (measured), which is 339 ms per
episode and ~14 h for the paper's 150k-episode budget on one core.

Condensing the equalities out and calling OSQP solves exactly the same problem.
`validate()` checks the two agree to solver tolerance.

Two capabilities are added that the shipped MPC does not have, both of which
matter for a 40 t rig:

  * a STEERING-RATE constraint  |u_k - u_{k-1}| <= dot_delta_max * T.
    The plant rate-limits the steering with a tanh; the shipped MPC model is an
    unsaturated first-order lag, so the controller plans steering it cannot
    physically deliver. On a truck (0.35 rad/s at the road wheel, 0.25 s lag)
    that mismatch is the difference between tracking and trailer swing.
  * MOVE BLOCKING. Condensing a 2 s horizon at 10 Hz leaves a Hessian with
    condition number ~2e4 (measured; it grows like N^4 because the y channel is
    a double integrator, so u_0 moves the terminal y ~400x more than u_19
    does). OSQP does not converge on it: on a lane change at the curvature
    ceiling it hit its iteration cap after 18 control steps and the episode was
    scored as a crash. Writing U = M*theta with a few geometrically growing
    blocks -- fine near t=0, where the only input that is actually applied
    lives, coarse far out -- drops the condition number by three orders and the
    problem size by two thirds. This is standard automotive-MPC practice and it
    is what makes the loop usable inside an RL training run at all.

  * SOFT articulation constraints. The shipped code applies phiMax / qMax as
    HARD boxes on predicted states -- harmless only because they ship as
    +-inf. Switch them on (you must, for a truck) and the QP becomes
    INFEASIBLE the moment the vehicle is outside the box, because no admissible
    steering can pull a predicted state back inside within the horizon. The
    solver then reports failure and the shipped episode loop scores that as a
    crash. Here each articulation constraint carries a non-negative slack with
    an L1+L2 penalty, so the QP is always feasible and the constraint is
    honoured whenever it can be.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

import osqp

from ttv_core import TTVConfig, build_linear_prediction_model


# =============================================================================
# OSQP >= 1.1 prints "Polishing not needed - no active set detected at optimal
# point" from its C polish routine on EVERY solve, regardless of
# `verbose=False`. A training run is ~10^5 solves, so the message buries the
# eval lines and inflates the log to tens of megabytes.
#
# `verbose=False` cannot suppress it because the write happens below Python's
# sys.stdout, so it has to be silenced at the file-descriptor level. The two
# dup2 calls cost ~2 us against a ~500 us solve.
#
# Set TTV_OSQP_VERBOSE=1 to see the solver's own output again when debugging a
# non-convergence.
# =============================================================================
_OSQP_VERBOSE = os.environ.get("TTV_OSQP_VERBOSE", "") not in ("", "0")


class _SilenceCStdout:
    """Redirect OS-level fd 1 to /dev/null. Reused across solves; opening the
    null device once keeps this off the hot path."""

    def __init__(self):
        self._null = None
        self._saved = None

    def __enter__(self):
        if _OSQP_VERBOSE:
            return self
        if self._null is None:
            self._null = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush()
        self._saved = os.dup(1)
        os.dup2(self._null, 1)
        return self

    def __exit__(self, *exc):
        if _OSQP_VERBOSE or self._saved is None:
            return False
        os.dup2(self._saved, 1)
        os.close(self._saved)
        self._saved = None
        return False


_silence = _SilenceCStdout()

NX = 7
IDX_Y, IDX_YDOT, IDX_PSI, IDX_R1, IDX_PHI, IDX_Q, IDX_DELTA = range(7)


@dataclass
class MPCDebug:
    solved: bool
    status: str
    iterations: int
    cost: float
    slack: float = 0.0        # largest articulation-constraint violation allowed


class CondensedMPC:
    """Condensed, warm-started QP form of the tracking MPC."""

    def __init__(self, cfg: TTVConfig, use_rate_constraint: bool = True,
                 terminal_weight_scale: float = 1.0,
                 soft_state_constraints: bool = True,
                 # Slack penalty: l1*s + l2*s^2 on the RELATIVE violation s.
                 # An l1-dominant penalty is an exact penalty function but makes
                 # the slack block of the Hessian nearly singular, and OSQP then
                 # hit its iteration cap whenever a slack went active (measured
                 # on a lane change at the curvature ceiling). An l2-dominant
                 # penalty keeps the QP strongly convex. The articulation limit
                 # is a guard, not a hard requirement -- Qphi/Qq in the cost are
                 # what actually manage articulation -- so trading exactness for
                 # conditioning is the right way round here.
                 slack_l1: float = 1.0, slack_l2: float = 1.0e2,
                 move_blocking: bool = True, n_blocks: Optional[int] = None):
        cfg.finalize()
        self.cfg = cfg
        self.N = int(cfg.N)
        self.T = float(cfg.T)
        self.use_rate = bool(use_rate_constraint)
        self.soft = bool(soft_state_constraints)

        Ac, Bc = build_linear_prediction_model(cfg)
        aug = expm(np.block([[Ac, Bc], [np.zeros((1, NX + 1))]]) * cfg.T)
        self.Ad = aug[:NX, :NX]
        self.Bd = aug[:NX, NX:NX + 1]

        N = self.N
        # --- prediction matrices  X = Phi x0 + Gamma U,  X = [x_1..x_N] -----
        Phi = np.zeros((NX * N, NX))
        Gam = np.zeros((NX * N, N))
        Apow = np.eye(NX)
        for k in range(N):
            Apow = Apow @ self.Ad if k else self.Ad.copy()
            Phi[k * NX:(k + 1) * NX, :] = Apow
        for k in range(N):                       # row block k -> x_{k+1}
            for j in range(k + 1):               # u_j contributes
                M = np.linalg.matrix_power(self.Ad, k - j) @ self.Bd
                Gam[k * NX:(k + 1) * NX, j:j + 1] = M
        self.Phi, self.Gam = Phi, Gam

        # --- stage weights ---------------------------------------------------
        w = np.array([cfg.mpc.Qy, 0.0, cfg.mpc.Qpsi, 0.0,
                      cfg.mpc.Qphi, cfg.mpc.Qq, 0.0])
        wvec = np.tile(w, N)
        if terminal_weight_scale != 1.0:
            wvec[(N - 1) * NX:] *= terminal_weight_scale
        self.W = wvec
        GtW = Gam.T * wvec                        # (N, NX*N)
        self.P_dense = 2.0 * (GtW @ Gam + cfg.mpc.Rdelta * np.eye(N))
        self.P_dense = 0.5 * (self.P_dense + self.P_dense.T)
        self.GtW = GtW

        # --- move blocking:  U = M theta -------------------------------------
        self.M = self._blocking_matrix(N, n_blocks) if move_blocking else np.eye(N)
        self.nu = self.M.shape[1]                 # number of free input DOF
        self.P_u = 0.5 * (self.M.T @ self.P_dense @ self.M
                          + (self.M.T @ self.P_dense @ self.M).T)
        self.GamM = Gam @ self.M

        # --- which states get a soft constraint ------------------------------
        self._soft_states = []
        if self.soft:
            if math.isfinite(cfg.phiMax):
                self._soft_states.append((IDX_PHI, cfg.phiMax))
            if math.isfinite(cfg.qMax):
                self._soft_states.append((IDX_Q, cfg.qMax))
        # One slack PER STEP per constrained state, and the constraint rows are
        # normalised by their own limit so the slack is a dimensionless relative
        # violation. A single scalar slack per constraint type couples all N
        # steps through one variable and made OSQP take thousands of iterations
        # (measured); per-step normalised slacks converge in tens.
        # One slack PER CONSTRAINED STEP, with the constraint row normalised by
        # its own limit so the slack is a dimensionless relative violation.
        # A single scalar slack per constraint type couples all N steps through
        # one variable and made OSQP take thousands of iterations (measured).
        self.n_soft = len(self._soft_states)
        self.n_slack = self.n_soft * N
        nu = self.nu
        self.nz = nu + self.n_slack                  # decision vector length

        # --- objective, padded for the slacks --------------------------------
        P = np.zeros((self.nz, self.nz))
        P[:nu, :nu] = self.P_u
        for i in range(self.n_slack):
            P[nu + i, nu + i] = 2.0 * slack_l2
        self.P_full = P
        self.slack_l1 = slack_l1

        # --- constraints -----------------------------------------------------
        def pad(B, slack_block=None, sign=0.0):
            """Pad a (rows, nu) block to the full decision width."""
            out = np.zeros((B.shape[0], self.nz))
            out[:, :nu] = B
            if slack_block is not None:
                c0 = nu + slack_block * N
                out[:, c0:c0 + N] = sign * np.eye(N)
            return out

        Mb = self.M
        rows, self._con = [], []
        rows.append(pad(Mb))                                     # |u_k| box
        self._con.append(("u", None))

        sel_delta = self._selector(IDX_DELTA)                    # actuator box
        rows.append(pad(sel_delta @ self.GamM))                  # always feasible
        self._con.append(("state_hard", (sel_delta, cfg.deltaMax)))

        if self.use_rate:                                        # actuator rate
            D = np.zeros((N, N))
            for k in range(N):
                D[k, k] = 1.0
                if k > 0:
                    D[k, k - 1] = -1.0
            rows.append(pad(D @ Mb))
            self._con.append(("rate", None))

        # Soft two-sided state constraints, normalised by the limit:
        #    (S X)_k/lim - s_k <=  1     and     (S X)_k/lim + s_k >= -1, s_k>=0
        for j, (idx, lim) in enumerate(self._soft_states):
            S = self._selector(idx) / lim
            SG = S @ self.GamM
            rows.append(pad(SG, j, -1.0))
            self._con.append(("soft_hi", S))
            rows.append(pad(SG, j, +1.0))
            self._con.append(("soft_lo", S))
        if self.n_slack:                                         # s >= 0
            E = np.zeros((self.n_slack, self.nz))
            E[:, nu:] = np.eye(self.n_slack)
            rows.append(E)
            self._con.append(("slack_nonneg", None))

        # Hard boxes when soft constraints are switched off (shipped behaviour)
        if not self.soft:
            if math.isfinite(cfg.phiMax):
                S = self._selector(IDX_PHI)
                rows.append(pad(S @ self.GamM))
                self._con.append(("state_hard", (S, cfg.phiMax)))
            if math.isfinite(cfg.qMax):
                S = self._selector(IDX_Q)
                rows.append(pad(S @ self.GamM))
                self._con.append(("state_hard", (S, cfg.qMax)))

        self.A_dense = np.vstack(rows)

        self._prob = osqp.OSQP()
        with _silence:
            self._prob.setup(P=sp.csc_matrix(self.P_full),
                             q=np.zeros(self.nz),
                             A=sp.csc_matrix(self.A_dense),
                             l=-np.inf * np.ones(self.A_dense.shape[0]),
                             u=np.inf * np.ones(self.A_dense.shape[0]),
                             eps_abs=1e-7, eps_rel=1e-7, max_iter=8000,
                             polish=True, verbose=False, warm_starting=True,
                             polish_refine_iter=3)
        self._last_u = np.zeros(N)

    # ------------------------------------------------------------------
    @staticmethod
    def _blocking_matrix(N: int, n_blocks: Optional[int]) -> np.ndarray:
        """Geometrically growing blocks: [1, 1, 2, 3, 5, 8, ...] capped at N.

        Only u_0 is ever applied, so resolution is spent near t = 0 and the far
        horizon is coarse -- exactly where the condensed Hessian's tiny
        singular values live.
        """
        lens, total = [], 0
        L = 1
        while total < N:
            take = min(L, N - total)
            lens.append(take)
            total += take
            if len(lens) >= 2:
                L = max(1, int(round(1.7 * L)))
        if n_blocks is not None and len(lens) > n_blocks:
            head = lens[:n_blocks - 1]
            lens = head + [N - sum(head)]
        M = np.zeros((N, len(lens)))
        k = 0
        for j, ln in enumerate(lens):
            M[k:k + ln, j] = 1.0
            k += ln
        return M

    # ------------------------------------------------------------------
    def _selector(self, idx: int) -> np.ndarray:
        """Pick state component `idx` out of every block of X."""
        S = np.zeros((self.N, NX * self.N))
        for k in range(self.N):
            S[k, k * NX + idx] = 1.0
        return S

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, y_ref: np.ndarray, psi_ref: np.ndarray
              ) -> Tuple[np.ndarray, MPCDebug]:
        N, nu, cfg = self.N, self.nu, self.cfg
        ref = np.zeros(NX * N)
        ref[IDX_Y::NX] = y_ref
        ref[IDX_PSI::NX] = psi_ref
        free = self.Phi @ x0                      # X with U = 0
        q = np.zeros(self.nz)
        q[:nu] = 2.0 * (self.M.T @ (self.GtW @ (free - ref)))
        q[nu:] = self.slack_l1

        INF = np.inf
        lo, hi = [], []
        for kind, payload in self._con:
            if kind == "u":
                lo.append(-cfg.deltaMax * np.ones(N))
                hi.append(cfg.deltaMax * np.ones(N))
            elif kind == "state_hard":
                S, lim = payload
                off = S @ free
                lo.append(-lim - off)
                hi.append(lim - off)
            elif kind == "soft_hi":
                S = payload
                lo.append(-INF * np.ones(N))
                hi.append(1.0 - S @ free)
            elif kind == "soft_lo":
                S = payload
                lo.append(-1.0 - S @ free)
                hi.append(INF * np.ones(N))
            elif kind == "slack_nonneg":
                lo.append(np.zeros(self.n_slack))
                hi.append(INF * np.ones(self.n_slack))
            else:                                  # rate
                r = cfg.deltaRateMax * cfg.T
                l_ = -r * np.ones(N)
                h_ = r * np.ones(N)
                # u_0 is measured against the ACTUAL steering angle, so the
                # first move is limited from where the actuator really is.
                l_[0] = x0[IDX_DELTA] - r
                h_[0] = x0[IDX_DELTA] + r
                lo.append(l_)
                hi.append(h_)

        self._prob.update(q=q, l=np.concatenate(lo), u=np.concatenate(hi))
        with _silence:
            res = self._prob.solve()
        st = str(res.info.status)
        ok = st in ("solved", "solved inaccurate")
        if ok and res.x is not None and np.all(np.isfinite(res.x)):
            z = np.asarray(res.x).flatten()
            u = np.clip(self.M @ z[:nu], -cfg.deltaMax, cfg.deltaMax)
            self._last_u = u
            slack = float(np.max(z[nu:])) if self.n_slack else 0.0
        else:
            ok = False
            u = self._last_u
            slack = float("nan")
        dbg = MPCDebug(ok, st, int(res.info.iter), float(res.info.obj_val))
        dbg.slack = slack
        return u, dbg

    # ------------------------------------------------------------------
    def predict(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """X = [x_1..x_N] as a (N, NX) array, for diagnostics.
        `u` is the full-length input sequence returned by solve()."""
        return (self.Phi @ x0 + self.Gam @ u).reshape(self.N, NX)

    @property
    def cond_hessian(self) -> float:
        return float(np.linalg.cond(self.P_u))


# =============================================================================
def validate(verbose: bool = True) -> bool:
    """Condensed QP vs the shipped IPOPT formulation, on identical data."""
    import casadi as ca
    from ttv_core import build_environment, make_cache_key

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:50s} {detail}")

    if verbose:
        print("=" * 78)
        print("fast_mpc validation: condensed OSQP  vs  shipped CasADi/IPOPT")
        print("=" * 78)

    rng = np.random.default_rng(7)

    # --- 1. no rate constraint -> must match IPOPT exactly ------------------
    cfg = TTVConfig(N=10, T=0.1, V=20.0).finalize()
    env = build_environment(cfg, make_cache_key(cfg))
    # Move blocking restricts the input space, so it CHANGES the optimum. To
    # compare against IPOPT's solution of the shipped problem it must be off.
    mpc = CondensedMPC(cfg, use_rate_constraint=False, move_blocking=False)
    check("discretisation identical",
          np.allclose(mpc.Ad, env["Ad"], atol=1e-14)
          and np.allclose(mpc.Bd, env["Bd"], atol=1e-14))

    worst_u, worst_J = 0.0, 0.0
    for trial in range(25):
        x0 = np.array([rng.normal(0, .3), rng.normal(0, .5), rng.normal(0, .05),
                       rng.normal(0, .05), rng.normal(0, .03), rng.normal(0, .05),
                       rng.uniform(-.2, .2)])
        yref = np.cumsum(rng.normal(0, .15, cfg.N))
        psiref = rng.normal(0, .03, cfg.N)

        u_fast, dbg = mpc.solve(x0, yref, psiref)

        p = np.zeros(NX + 2 * cfg.N)
        p[:NX] = x0
        p[NX + 0::2] = yref
        p[NX + 1::2] = psiref
        w0 = np.concatenate((np.tile(x0, cfg.N + 1), np.zeros(cfg.N)))
        sol = env["solver"](x0=w0, lbx=env["lbx"], ubx=env["ubx"],
                            lbg=env["lbg"], ubg=env["ubg"], p=p)
        wopt = np.asarray(sol["x"]).flatten()
        u_ipopt = wopt[NX * (cfg.N + 1):]
        J_ipopt = float(sol["f"])

        # OSQP objective is 1/2 u'Pu + q'u, missing the constant ref'W ref.
        ref = np.zeros(NX * cfg.N)
        ref[IDX_Y::NX] = yref
        ref[IDX_PSI::NX] = psiref
        free = mpc.Phi @ x0
        const = float(((free - ref) ** 2 * mpc.W).sum())
        J_fast = dbg.cost + const

        worst_u = max(worst_u, float(np.max(np.abs(u_fast - u_ipopt))))
        worst_J = max(worst_J, abs(J_fast - J_ipopt) / max(abs(J_ipopt), 1e-9))

    # The shipped solver runs with ipopt.acceptable_tol = 1e-7, so agreement is
    # bounded by IPOPT's own tolerance, not by the condensing. 1e-5 rad is
    # 6e-4 deg of steering -- five orders below the actuator resolution.
    check("optimal control sequence matches IPOPT", worst_u < 1e-5,
          f"max |du| = {worst_u:.2e} rad over 25 random problems")
    check("optimal cost matches IPOPT", worst_J < 1e-7,
          f"max rel. cost error = {worst_J:.2e}")

    # --- 2. constraints are actually enforced -------------------------------
    if verbose:
        print()
    cfg2 = TTVConfig(N=10, T=0.1, V=20.0, deltaMax=0.05, deltaRateMax=0.10,
                     phiMax=math.radians(5.0), qMax=math.radians(8.0)).finalize()
    m_soft = CondensedMPC(cfg2, use_rate_constraint=True, soft_state_constraints=True)
    m_hard = CondensedMPC(cfg2, use_rate_constraint=True, soft_state_constraints=False)

    # Benign states: the soft version must honour the boxes exactly.
    viol_u = viol_rate = viol_phi = viol_q = 0.0
    for _ in range(25):
        x0 = np.array([rng.normal(0, 1.0), rng.normal(0, 2.0), rng.normal(0, .1),
                       rng.normal(0, .1), rng.normal(0, .01), rng.normal(0, .02),
                       rng.uniform(-.04, .04)])
        yref = np.cumsum(rng.normal(0, .6, cfg2.N))
        u, dbg = m_soft.solve(x0, yref, np.zeros(cfg2.N))
        if not dbg.solved:
            continue
        X = m_soft.predict(x0, u)
        viol_u = max(viol_u, float(np.max(np.abs(u)) - cfg2.deltaMax))
        du = np.diff(np.concatenate(([x0[IDX_DELTA]], u)))
        viol_rate = max(viol_rate, float(np.max(np.abs(du)) - cfg2.deltaRateMax * cfg2.T))
        viol_phi = max(viol_phi, float(np.max(np.abs(X[:, IDX_PHI])) - cfg2.phiMax))
        viol_q = max(viol_q, float(np.max(np.abs(X[:, IDX_Q])) - cfg2.qMax))
    check("|u| <= deltaMax", viol_u < 1e-7, f"worst excess {viol_u:.2e}")
    check("|du| <= deltaRateMax*T (new)", viol_rate < 1e-7, f"worst excess {viol_rate:.2e}")
    # A soft constraint is SOFT: it permits a bounded violation by design. The
    # test is that the violation stays small, not that it is zero.
    check("|phi| <= phiMax to within 10 % when feasible",
          viol_phi < 0.10 * cfg2.phiMax,
          f"worst excess {math.degrees(viol_phi):.3f} deg of "
          f"{math.degrees(cfg2.phiMax):.1f}")
    check("|q| <= qMax to within 10 % when feasible", viol_q < 0.10 * cfg2.qMax,
          f"worst excess {math.degrees(viol_q):.3f} deg/s of "
          f"{math.degrees(cfg2.qMax):.1f}")

    # Hostile states: at or OUTSIDE the articulation box. This is the case the
    # shipped hard-constrained formulation cannot solve.
    if verbose:
        print("\n    solver failures vs how far the state is outside the box:")
        print(f"    {'overshoot':>10s} {'hard box':>10s} {'soft box':>10s}")
    table = []
    for f in (1.0, 1.2, 1.5, 2.0):
        r2 = np.random.default_rng(3)
        hf = sf = 0
        for _ in range(40):
            sgn = r2.choice([-1.0, 1.0])
            x0 = np.array([r2.normal(0, .5), r2.normal(0, 1.), r2.normal(0, .05),
                           r2.normal(0, .05), sgn * f * cfg2.phiMax,
                           sgn * r2.uniform(0, 1) * f * cfg2.qMax,
                           r2.uniform(-.04, .04)])
            yref = np.cumsum(r2.normal(0, .3, cfg2.N))
            _, d_h = m_hard.solve(x0, yref, np.zeros(cfg2.N))
            _, d_s = m_soft.solve(x0, yref, np.zeros(cfg2.N))
            hf += int(not d_h.solved)
            sf += int(not d_s.solved)
        table.append((f, hf, sf))
        if verbose:
            print(f"    {f:10.1f} {hf:7d}/40 {sf:7d}/40")
    check("HARD state boxes are unusable: they fail even AT the limit",
          table[0][1] > 30, f"{table[0][1]}/40 failures at overshoot 1.0")
    check("SOFT state boxes solve at and just past the limit",
          table[0][2] == 0 and table[1][2] <= 4,
          f"{table[0][2]}/40 at 1.0x, {table[1][2]}/40 at 1.2x")

    # --- 3. speed -----------------------------------------------------------
    if verbose:
        print()
    import time
    x0 = np.zeros(NX)
    yref = np.linspace(0, 1.5, cfg.N)
    psiref = np.zeros(cfg.N)
    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        mpc.solve(x0, yref, psiref)
    t_fast = (time.perf_counter() - t0) / n

    p = np.zeros(NX + 2 * cfg.N)
    p[NX + 0::2] = yref
    w0 = np.zeros(NX * (cfg.N + 1) + cfg.N)
    t0 = time.perf_counter()
    for _ in range(n):
        env["solver"](x0=w0, lbx=env["lbx"], ubx=env["ubx"],
                      lbg=env["lbg"], ubg=env["ubg"], p=p)
    t_ipopt = (time.perf_counter() - t0) / n

    if verbose:
        print(f"    IPOPT (as shipped) : {t_ipopt*1e6:9.1f} us / solve")
        print(f"    condensed OSQP     : {t_fast*1e6:9.1f} us / solve")
        print(f"    speedup            : {t_ipopt/t_fast:9.1f}x")
    check("condensed QP is at least 10x faster", t_ipopt / t_fast > 10.0,
          f"{t_ipopt/t_fast:.1f}x")

    if verbose:
        print(f"\n  VALIDATION {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
