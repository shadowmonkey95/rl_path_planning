"""
paper_metrics.py -- The physical trajectory evaluation of Karimyan et al. (2024)
================================================================================

Karimyan, Rahmani Hanzaki & Azadi, "An integrated longitudinal and lateral
control strategy for low friction conditions of tractor semi-trailer vehicles",
Mechanics Based Design of Structures and Machines 52:7 (2024) 4361-4398.
DOI 10.1080/15397734.2023.2227703

WHY THIS MODULE EXISTS
----------------------

The previous benchmark had a structural flaw: the reward function WAS the RL
objective. The policy was trained to maximise exactly the quantity used to
score every method, while the planning QP minimised something else (jerk). A
margin measured that way is close to tautological.

This module removes that flaw by scoring trajectories with quantities from the
published literature that NEITHER method is defined in terms of:

    RCOF  (Eq. 62)  the coefficient of friction a trajectory REQUIRES
    LTR   (Eq. 61)  lateral load transfer ratio, i.e. rollover proximity
    J_T   (Eq. 53)  tyre workload, how hard the tyres are being asked to work

The paper's own trajectory-selection rule (Fig. 9 and Table 4) is then
available to both planners:

    1. reject any trajectory whose peak LTR exceeds 0.6     (unstable)
    2. among the survivors, choose the one with the SMALLEST peak RCOF

That rule is physical, is published, and is not something either method was
built around -- so it can referee between them.

Table 4 of the paper, reproduced here because it is the validation target:

    traj  t_m[s]  a_x[m/s^2]  a_y[m/s^2]   mu_f   mu_r   mu_s   LTR_max
      1    2.5       1.1         3.46      0.59   0.53   0.64    1.00   reject
      2    3.4       1.0         1.98      0.47   0.43   0.51    0.89   reject
      3    3.5       0.5         1.76      0.41   0.40   0.48    0.79   reject
      4    3.8       0.0         1.58      0.38   0.36   0.38    0.65   reject
      5    4.0      -1.0         1.36      0.33   0.37   0.32    0.49
      6    4.2      -1.5         1.23      0.31   0.28   0.30    0.43
      7    4.4      -2.0         1.12      0.29   0.25   0.27    0.35   BEST
      8    4.6      -2.5         1.02      0.31   0.28   0.29    0.33
      9    5.2      -3.5         0.80      0.34   0.31   0.32    0.30

Note trajectories 8 and 9: braking harder keeps reducing LTR but RCOF turns
back UP, because the longitudinal force the braking demands eventually costs
more friction than the lateral force it saves. The optimum is interior, which
is what makes this a real optimisation rather than a monotone rule.

HONEST DEVIATIONS FROM THE PAPER
--------------------------------

1. LTR. The paper's Eq. (61) uses a roll degree of freedom (roll angle, roll
   stiffness, roll damping). The plant in this project is a yaw-plane model
   with no roll DOF, so Eq. (61) cannot be evaluated on it. This module uses
   the standard quasi-static load-transfer ratio instead,

       LTR = 2 * h * a_y / (g * track)

   derated by a roll-compliance factor, which is the same approximation
   `truck_params.py` already uses for the Static Rollover Threshold. It
   captures the quantity the paper's threshold is about -- how close the inside
   wheels are to lifting -- but it cannot represent roll oscillation or the
   transient overshoot a roll DOF would give, so it reads LOW during rapid
   transients. Treat the 0.6 threshold as correspondingly optimistic.

2. Longitudinal tyre force. The plant here runs at constant forward speed and
   generates no longitudinal tyre force. The paper's a_x is a PLANNING variable
   (Eq. 15: constant longitudinal acceleration through the manoeuvre), and the
   paper itself evaluates candidate trajectories on a simplified model before
   any controller exists (Section 3.4, Eqs. 51-52). This module does the same:
   the planner declares a_x, the required longitudinal force is distributed
   across axles by static load share, and it is combined with the lateral force
   the full nonlinear simulation actually produced. The simulated vehicle is
   untouched -- the manoeuvre layer stays frozen, as intended.

3. Vehicle. The paper's own rig (Table B1) is not this project's vehicle. All
   geometry here comes from the existing `TTVConfig`, so absolute RCOF values
   will not match the paper's table. The METHOD transfers; the numbers do not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

G = 9.81

# -----------------------------------------------------------------------------
# Roll geometry. Shared with `truck_params.py` so one project does not carry two
# different rollover models.
# -----------------------------------------------------------------------------
TRACK_WIDTH = 2.04              # [m] dual-tyre centre-to-centre
CG_HEIGHT_TRACTOR = 1.15        # [m]
CG_HEIGHT_TRAILER = 1.95        # [m] laden box trailer
ROLL_COMPLIANCE_FACTOR = 0.70   # suspension + tyre + fifth-wheel compliance

# The paper's thresholds (Section 3.4 and Fig. 9).
LTR_UNSTABLE = 0.60             # above this the paper rejects the trajectory
LTR_CRITICAL = 0.80             # Dong et al., quoted by the paper as the limit

# Tyre-workload weights, Eq. (53). The paper reports beta_f = 1, beta_r = 5,
# beta_s = 2 -- the drive axle is weighted most because losing it is what
# initiates jack-knife.
BETA_F, BETA_R, BETA_S = 1.0, 5.0, 2.0

# The quintic's peak lateral acceleration coefficient, Eq. (44).
# y(t) = W*(10*tau^3 - 15*tau^4 + 6*tau^5), tau = t/t_m
# y''(t) is extremal at tau = (1 +- 1/sqrt(3))/2, giving |y''|max = 10/sqrt(3)
# times W/t_m^2. The paper rounds this to 5.77.
QUINTIC_AY_COEFF = 10.0 / math.sqrt(3.0)        # = 5.7735...


# =============================================================================
# 1.  The paper's trajectory family: the quintic polynomial, Eq. (14)
# =============================================================================

def quintic_y(t: np.ndarray, W: float, t_m: float) -> np.ndarray:
    """Lateral displacement of Eq. (14), clamped outside [0, t_m].

    Zero position, velocity AND acceleration at both ends, which is why a
    fifth-order polynomial is the lowest order that works (You et al. 2015).
    """
    tau = np.clip(np.asarray(t, dtype=float) / t_m, 0.0, 1.0)
    return W * (10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5)


def quintic_dy(t: np.ndarray, W: float, t_m: float) -> np.ndarray:
    """Eq. (20)."""
    tau = np.asarray(t, dtype=float) / t_m
    inside = (tau >= 0.0) & (tau <= 1.0)
    tau = np.clip(tau, 0.0, 1.0)
    return np.where(inside,
                    W / t_m * (30.0 * tau ** 2 - 60.0 * tau ** 3
                               + 30.0 * tau ** 4), 0.0)


def quintic_ddy(t: np.ndarray, W: float, t_m: float) -> np.ndarray:
    """Eq. (21)."""
    tau = np.asarray(t, dtype=float) / t_m
    inside = (tau >= 0.0) & (tau <= 1.0)
    tau = np.clip(tau, 0.0, 1.0)
    return np.where(inside,
                    W / t_m ** 2 * (60.0 * tau - 180.0 * tau ** 2
                                    + 120.0 * tau ** 3), 0.0)


def quintic_peak_ay(W: float, t_m: float) -> float:
    """Eq. (44): |a_y,max| = 5.77 * W / t_m^2.

    EXACT for this family, unlike the tanh closed form the previous action
    space used, which under-predicted by up to 97 % once the two transitions
    overlapped. A quintic transition is finite in support, so two of them
    separated by any dwell at all cannot interact.
    """
    return QUINTIC_AY_COEFF * W / (t_m ** 2)


def quintic_min_time(W: float, a_y_max: float) -> float:
    """Eq. (50) rearranged: the fastest transition whose peak a_y fits."""
    return math.sqrt(QUINTIC_AY_COEFF * W / max(a_y_max, 1e-9))


# =============================================================================
# 2.  Friction circle, Eq. (41) and (42)
# =============================================================================

def friction_circle_demand(a_x: float | np.ndarray,
                           a_y: float | np.ndarray) -> np.ndarray:
    """mu_required = sqrt(a_x^2 + a_y^2) / g, Eq. (41).

    The whole-vehicle version. The per-axle version (RCOF, below) is stricter
    because load is not shared equally.
    """
    return np.hypot(np.asarray(a_x, dtype=float),
                    np.asarray(a_y, dtype=float)) / G


def max_longitudinal_accel(mu: float, a_y: float) -> float:
    """Eq. (42): what is left for braking/driving once a_y is spent."""
    budget = (mu * G) ** 2 - float(a_y) ** 2
    return math.sqrt(budget) if budget > 0.0 else 0.0


# =============================================================================
# 3.  RCOF, Eq. (62) -- the paper's primary selection criterion
# =============================================================================

def rcof_per_axle(Fy: np.ndarray, Fz: Sequence[float],
                  a_x: float = 0.0,
                  m_total: Optional[float] = None) -> np.ndarray:
    """mu_i(t) = sqrt(Fx_i^2 + Fy_i^2) / Fz_i for each axle, Eq. (62).

    `Fy` is (3, n) as the simulation produced it. `Fz` is the three static
    vertical loads. `a_x` is the planner's commanded longitudinal acceleration;
    the longitudinal force it demands is split across axles in proportion to
    their vertical load, which is the equal-friction-utilisation split and the
    one the paper's braking-distribution optimisation (Eq. 53-56) converges
    toward.

    Returns (3, n).
    """
    Fy = np.atleast_2d(np.asarray(Fy, dtype=float))
    Fz = np.asarray(Fz, dtype=float).reshape(-1, 1)
    if m_total is None:
        m_total = float(Fz.sum()) / G
    share = Fz / Fz.sum()                       # (3, 1)
    Fx = share * (m_total * float(a_x))         # broadcast over time
    return np.hypot(Fx, Fy) / Fz


def peak_rcof(Fy: np.ndarray, Fz: Sequence[float], a_x: float = 0.0
              ) -> Dict[str, float]:
    """Peak RCOF per axle and overall. The paper selects on the overall peak."""
    mu = rcof_per_axle(Fy, Fz, a_x)
    return {"mu_front": float(np.max(mu[0])),
            "mu_rear": float(np.max(mu[1])),
            "mu_trailer": float(np.max(mu[2])),
            "mu_max": float(np.max(mu))}


# =============================================================================
# 4.  LTR -- quasi-static stand-in for Eq. (61); see the module docstring
# =============================================================================

def ltr_quasi_static(a_y: np.ndarray, h_cg: float,
                     track: float = TRACK_WIDTH,
                     compliance: float = ROLL_COMPLIANCE_FACTOR) -> np.ndarray:
    """|LTR| = 2*h*a_y / (g*track), divided by the roll-compliance factor.

    Dividing (rather than multiplying) by the compliance factor is deliberate:
    compliance makes a real vehicle reach a given load transfer at a LOWER
    lateral acceleration than rigid-axle geometry predicts, so it raises LTR
    for a given a_y. The same factor lowers the Static Rollover Threshold in
    `truck_params.py`, which is the same statement seen from the other side.
    """
    return (2.0 * h_cg * np.abs(np.asarray(a_y, dtype=float))
            / (G * track * compliance))


def a_y_for_ltr(ltr_limit: float = LTR_UNSTABLE,
                h_cg: float = CG_HEIGHT_TRAILER,
                track: float = TRACK_WIDTH,
                compliance: float = ROLL_COMPLIANCE_FACTOR) -> float:
    """Invert `ltr_quasi_static`: the lateral acceleration at which LTR hits
    its limit.

    This matters more than it looks. For a laden artic the ROLLOVER limit bites
    well before the friction limit: with this geometry LTR = 0.6 is reached at
    about 2.2 m/s^2 (0.22 g), while mu = 0.5 would allow 4.9 m/s^2. The paper
    reports the same ordering -- a rollover threshold of 0.36 g against a
    friction capacity of 0.4 g at mu = 0.4, and an optimal trajectory that uses
    only 0.11 g.

    So the binding lateral-acceleration limit is the MINIMUM of the friction
    circle and this. Figure 9 of the paper encodes exactly that: estimate LTR,
    and if it exceeds the threshold, increase the manoeuvre time and try again.
    """
    return ltr_limit * G * track * compliance / (2.0 * h_cg)


def a_y_limit(mu: float, ltr_limit: float = LTR_UNSTABLE) -> float:
    """The binding lateral-acceleration capacity: friction circle AND rollover."""
    return min(mu * G, a_y_for_ltr(ltr_limit))


def ltr_from_run(a_y_tractor: np.ndarray, a_y_trailer: np.ndarray
                 ) -> Dict[str, float]:
    """Peak LTR for each unit, and the governing (larger) one."""
    ltr_t = ltr_quasi_static(a_y_tractor, CG_HEIGHT_TRACTOR)
    ltr_s = ltr_quasi_static(a_y_trailer, CG_HEIGHT_TRAILER)
    return {"ltr_tractor": float(np.max(ltr_t)),
            "ltr_trailer": float(np.max(ltr_s)),
            "ltr_max": float(max(np.max(ltr_t), np.max(ltr_s)))}


# =============================================================================
# 5.  Tyre workload, Eq. (53)
# =============================================================================

def tyre_workload(Fy: np.ndarray, Fz: Sequence[float], a_x: float = 0.0,
                  betas: Sequence[float] = (BETA_F, BETA_R, BETA_S)
                  ) -> float:
    """J_T, Eq. (53): a weighted sum of squared normalised axle force.

    This is the objective the paper minimises when distributing braking force.
    Reported here as a scalar (its peak over the run) so a trajectory can be
    compared on how hard it works the tyres, not only on whether it fits.
    """
    Fy = np.atleast_2d(np.asarray(Fy, dtype=float))
    Fz = np.asarray(Fz, dtype=float).reshape(-1, 1)
    m_total = float(Fz.sum()) / G
    share = Fz / Fz.sum()
    Fx = share * (m_total * float(a_x))
    terms = (Fx ** 2 + Fy ** 2) / (Fz ** 2)
    w = np.asarray(betas, dtype=float).reshape(-1, 1)
    return float(np.max((w * terms).sum(axis=0)))


# =============================================================================
# 6.  The paper's evaluation, as one call
# =============================================================================

@dataclass
class TrajectoryEvaluation:
    """One row of the paper's Table 4, for a trajectory this project produced."""
    t_m: float
    a_x: float
    a_y_peak: float
    mu_front: float
    mu_rear: float
    mu_trailer: float
    mu_max: float
    ltr_tractor: float
    ltr_trailer: float
    ltr_max: float
    workload: float
    stable: bool
    feasible: bool
    mu_available: float

    def as_row(self) -> str:
        flag = "" if self.stable else "  REJECT (LTR)"
        if self.stable and not self.feasible:
            flag = "  REJECT (mu)"
        return (f"{self.t_m:6.2f} {self.a_x:11.2f} {self.a_y_peak:11.2f} "
                f"{self.mu_front:7.3f} {self.mu_rear:7.3f} "
                f"{self.mu_trailer:7.3f} {self.mu_max:7.3f} "
                f"{self.ltr_max:8.3f}{flag}")


def evaluate_trajectory(Fy: np.ndarray, a_y_tractor: np.ndarray,
                        a_y_trailer: np.ndarray, Fz: Sequence[float],
                        mu_available: float, t_m: float = float("nan"),
                        a_x: float = 0.0) -> TrajectoryEvaluation:
    """Score one closed-loop run exactly as the paper scores a candidate path.

    `Fy` (3, n) and the two lateral-acceleration traces come from the
    simulation; `a_x` is what the planner asked for longitudinally.
    """
    r = peak_rcof(Fy, Fz, a_x)
    l = ltr_from_run(a_y_tractor, a_y_trailer)
    return TrajectoryEvaluation(
        t_m=float(t_m), a_x=float(a_x),
        a_y_peak=float(np.max(np.abs(a_y_tractor))),
        mu_front=r["mu_front"], mu_rear=r["mu_rear"],
        mu_trailer=r["mu_trailer"], mu_max=r["mu_max"],
        ltr_tractor=l["ltr_tractor"], ltr_trailer=l["ltr_trailer"],
        ltr_max=l["ltr_max"],
        workload=tyre_workload(Fy, Fz, a_x),
        stable=l["ltr_max"] <= LTR_UNSTABLE,
        feasible=r["mu_max"] <= mu_available,
        mu_available=float(mu_available))


def select_best(evals: Sequence[TrajectoryEvaluation]
                ) -> Optional[TrajectoryEvaluation]:
    """The paper's rule, Fig. 9: drop the unstable, then minimise peak RCOF."""
    ok = [e for e in evals if e.stable and e.feasible]
    return min(ok, key=lambda e: e.mu_max) if ok else None


# =============================================================================
def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:54s} {detail}")

    if verbose:
        print("=" * 92)
        print("paper_metrics self-test -- against the paper's own stated values")
        print("=" * 92)
        print("  A. the quintic, Eqs. (14), (21), (44)")

    # Eq. (44)'s coefficient, checked numerically rather than assumed.
    W, t_m = 3.735, 4.4
    t = np.linspace(0.0, t_m, 200001)
    num = float(np.max(np.abs(quintic_ddy(t, W, t_m))))
    cf = quintic_peak_ay(W, t_m)
    check("Eq. (44) closed form matches the numerical peak",
          abs(num - cf) / cf < 1e-6, f"{cf:.6f} vs {num:.6f} m/s^2")
    check("the coefficient is the paper's 5.77",
          abs(QUINTIC_AY_COEFF - 5.77) < 0.005,
          f"10/sqrt(3) = {QUINTIC_AY_COEFF:.4f}")

    # Boundary conditions: the reason a quintic and not a cubic.
    check("y(0)=0, y(t_m)=W", abs(quintic_y(0.0, W, t_m)) < 1e-12
          and abs(quintic_y(t_m, W, t_m) - W) < 1e-9)
    check("y'(0)=y'(t_m)=0", abs(quintic_dy(0.0, W, t_m)) < 1e-9
          and abs(quintic_dy(t_m, W, t_m)) < 1e-9)
    check("y''(0)=y''(t_m)=0", abs(quintic_ddy(0.0, W, t_m)) < 1e-9
          and abs(quintic_ddy(t_m, W, t_m)) < 1e-9)

    # Eq. (50): the paper's t_min = sqrt(21.56/a_y_max) with 21.56 = 5.77*W
    check("Eq. (50) round-trips against Eq. (44)",
          abs(quintic_min_time(W, cf) - t_m) < 1e-9,
          f"t_min({cf:.3f}) = {quintic_min_time(W, cf):.4f} s")
    # The paper's worked example, Eq. (50): t_min = sqrt(21.56/3.92) "= 2.3 s".
    # The exact value is 2.345; the paper rounds. Checked against the exact
    # arithmetic rather than against the rounded figure printed in the text,
    # because matching the rounding would mean hiding a real 2 % difference.
    t23 = quintic_min_time(3.735, 3.92)
    check("the paper's worked example, Eq. (50)",
          abs(t23 - math.sqrt(21.56 / 3.92)) < 0.01,
          f"{t23:.3f} s; the paper prints 2.3, exact is "
          f"{math.sqrt(21.56/3.92):.3f}")

    if verbose:
        print("\n  B. the friction circle, Eqs. (41) and (42)")
    check("mu = sqrt(ax^2+ay^2)/g",
          abs(float(friction_circle_demand(3.0, 4.0)) - 5.0 / G) < 1e-12)
    # Paper: mu = 0.4 -> a_yf_max = 3.92 m/s^2
    check("mu = 0.4 gives a_y capacity 3.92 m/s^2",
          abs(0.4 * G - 3.92) < 0.01, f"{0.4*G:.3f}")
    check("Eq. (42) leaves nothing when a_y uses the whole circle",
          max_longitudinal_accel(0.4, 0.4 * G) < 1e-6)
    check("Eq. (42) at zero a_y returns the full circle",
          abs(max_longitudinal_accel(0.4, 0.0) - 0.4 * G) < 1e-9)

    if verbose:
        print("\n  C. RCOF and LTR behave as the paper's Table 4 implies")

    Fz = (44103.8, 37567.1, 39973.0)
    n = 400
    Fy_small = np.vstack([np.full(n, 0.10 * Fz[0]),
                          np.full(n, 0.10 * Fz[1]),
                          np.full(n, 0.10 * Fz[2])])
    r0 = peak_rcof(Fy_small, Fz, a_x=0.0)
    check("pure lateral: RCOF = Fy/Fz", abs(r0["mu_max"] - 0.10) < 1e-9,
          f"{r0['mu_max']:.4f}")
    r1 = peak_rcof(Fy_small, Fz, a_x=-2.0)
    check("adding braking RAISES RCOF (the Table-4 turn-up)",
          r1["mu_max"] > r0["mu_max"],
          f"{r0['mu_max']:.4f} -> {r1['mu_max']:.4f} at a_x = -2 m/s^2")

    # Table 4's shape: braking buys a longer manoeuvre and so less lateral
    # demand, but costs longitudinal friction, and past some point the second
    # effect wins. Reproduce that shape using the paper's own a_x / t_m pairs
    # from Table 4 rather than an invented scaling, so this tests the RCOF
    # formula and not my choice of sweep.
    table4 = [(1.1, 2.5), (1.0, 3.4), (0.5, 3.5), (0.0, 3.8), (-1.0, 4.0),
              (-1.5, 4.2), (-2.0, 4.4), (-2.5, 4.6), (-3.5, 5.2)]
    # Scaled so the zero-braking row sits at mu ~ 0.38, which is the paper's
    # own trajectory 4. At the 0.10 level used above the lateral demand is too
    # small for the trade to bite and the optimum sits at a_x = 0, which says
    # nothing. In the paper's regime the U-shape appears.
    Fy_ref = np.vstack([np.full(n, 0.38 * fz) for fz in Fz])
    base_ay = quintic_peak_ay(3.735, 3.8)
    sweep = []
    for a_x, t_m_i in table4:
        Fy_i = Fy_ref * (quintic_peak_ay(3.735, t_m_i) / base_ay)
        sweep.append((a_x, peak_rcof(Fy_i, Fz, a_x)["mu_max"]))
    best_i = int(np.argmin([m for _, m in sweep]))
    check("RCOF over Table 4's own a_x/t_m pairs is minimised INTERIOR",
          0 < best_i < len(sweep) - 1,
          f"best at a_x = {sweep[best_i][0]:+.1f} m/s^2, "
          f"mu = {sweep[best_i][1]:.3f} (the paper picks -2.0, mu = 0.289)")

    ltr = ltr_from_run(np.array([0.36 * G]), np.array([0.36 * G]))
    check("LTR at the rig's SRT is close to 1",
          0.85 <= ltr["ltr_max"] <= 1.15, f"{ltr['ltr_max']:.3f} at 0.36 g")
    check("the trailer governs LTR, not the tractor",
          ltr["ltr_trailer"] > ltr["ltr_tractor"],
          f"{ltr['ltr_trailer']:.3f} vs {ltr['ltr_tractor']:.3f}")

    if verbose:
        print("\n  D. the selection rule, Fig. 9")
    mk = lambda mu, lt: TrajectoryEvaluation(
        4.0, 0.0, 1.0, mu, mu, mu, mu, lt, lt, lt, 0.0,
        lt <= LTR_UNSTABLE, True, 0.9)
    cands = [mk(0.20, 0.95), mk(0.29, 0.35), mk(0.31, 0.33)]
    best = select_best(cands)
    check("an unstable trajectory is rejected even with the lowest RCOF",
          best is not None and abs(best.mu_max - 0.29) < 1e-9,
          f"chose mu_max = {best.mu_max:.3f}" if best else "chose none")
    check("all-unstable returns None", select_best([mk(0.2, 0.95)]) is None)

    if verbose:
        print(f"\n  SELF-TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
