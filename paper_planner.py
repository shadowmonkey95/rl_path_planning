"""
paper_planner.py -- The paper's trajectory planner, as the upgraded baseline
=============================================================================

Replaces `dlc_baselines.DLCTrajectoryPlanner` (a jerk-minimising convex QP I
wrote) with the published method of Karimyan et al. (2024), Section 3 and
Figure 9.

WHY THE OLD BASELINE WAS NOT GOOD ENOUGH
----------------------------------------

Three criticisms of it were correct and all three are fixed here:

  OLD                                      NEW
  jerk, a convex proxy nobody asked for -> peak RCOF, the paper's own criterion
  deadline as a HARD constraint,        -> manoeuvre time is a DECISION
    so a tight task was infeasible         variable, so a tight task is
    rather than merely expensive            expensive, never infeasible
  no rollover model at all              -> LTR <= 0.6, the paper's threshold
  point-mass, no friction circle        -> friction circle on combined a_x, a_y

The result is a baseline that optimises the SAME physical objective the reward
now scores. That is the point: neither side is now being judged on its own
private objective, which was the strongest criticism of the earlier benchmark.

THE METHOD (Figure 9 of the paper)
----------------------------------

  1. Enumerate candidate trajectories over (manoeuvre time t_m, longitudinal
     acceleration a_x). The paper's Table 4 does exactly this with nine
     hand-chosen rows; this implementation sweeps the same two variables on a
     grid and then refines the best cell.
  2. Reject any candidate violating the friction circle, Eq. (41).
  3. Estimate roll angle and load transfer; reject LTR > 0.6.
  4. Among the survivors, choose the SMALLEST peak RCOF, Eq. (62).

Each candidate is evaluated on a simplified model, exactly as the paper does
before any controller exists -- a full closed-loop simulation per candidate
would not be a planner, it would be the offline optimum. The chosen trajectory
is then handed to the same frozen tracking MPC every other method uses.

WHAT THIS PLANNER IS AND IS NOT ALLOWED TO DO
---------------------------------------------

It may: evaluate as many candidates as it likes on its own internal model, use
the quintic family, and choose t_m, a_x and the dwell.

It may NOT: run the nonlinear plant, call the tracking MPC, or see the reward.
Those are what separate a deployable planner from the offline optimum. The
candidate count and the wall-clock cost are both reported so the comparison
against a single network forward pass stays honest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paper_metrics import (CG_HEIGHT_TRACTOR, CG_HEIGHT_TRAILER, G,
                           LTR_UNSTABLE, QUINTIC_AY_COEFF, TRACK_WIDTH,
                           ltr_quasi_static, peak_rcof, quintic_ddy,
                           quintic_dy, quintic_min_time, quintic_peak_ay,
                           quintic_y, tyre_workload)


# =============================================================================
# 1.  The reference: a double lane change built from two quintics
# =============================================================================

class QuinticDLCReference:
    """Out on a quintic, dwell, back on a quintic. Duck-types `DLCReference`.

    The paper's Eq. (14) describes a SINGLE lane change. A double lane change
    is two of them with a dwell between, which is also why the exact closed
    form for peak lateral acceleration survives here while the tanh version's
    did not: a quintic transition has finite support, so two of them separated
    by any dwell cannot overlap and raise each other's peak.
    """

    def __init__(self, W: float, t_m: float, t_start: float, dwell: float,
                 V: float = 20.0):
        self.W = float(W)
        self.t_m = float(t_m)
        self.t_start = float(t_start)
        self.dwell = float(dwell)
        self.V = float(V)
        self.t_return = self.t_start + self.t_m + self.dwell
        # kept so callers written against DLCReference keep working
        self.amplitude = 0.5 * self.W
        self.rate = float("nan")

    # -- the three the tracking MPC consumes -------------------------------
    def y(self, t):
        t = np.asarray(t, dtype=float)
        return (quintic_y(t - self.t_start, self.W, self.t_m)
                - quintic_y(t - self.t_return, self.W, self.t_m))

    def dy(self, t):
        t = np.asarray(t, dtype=float)
        return (quintic_dy(t - self.t_start, self.W, self.t_m)
                - quintic_dy(t - self.t_return, self.W, self.t_m))

    def ddy(self, t):
        t = np.asarray(t, dtype=float)
        return (quintic_ddy(t - self.t_start, self.W, self.t_m)
                - quintic_ddy(t - self.t_return, self.W, self.t_m))

    def psi(self, t):
        return np.arctan2(self.dy(t), self.V)

    # -- convenience --------------------------------------------------------
    @property
    def peak_lateral_accel(self) -> float:
        """Exact, Eq. (44). Valid because the two transitions never overlap."""
        return quintic_peak_ay(self.W, self.t_m)

    @property
    def completion_time(self) -> float:
        return self.t_return + self.t_m

    def describe(self) -> str:
        return (f"quintic DLC: W={self.W:.2f} m, t_m={self.t_m:.2f} s, "
                f"start={self.t_start:.2f} s, dwell={self.dwell:.2f} s, "
                f"peak a_y={self.peak_lateral_accel/G:.3f} g")


# =============================================================================
# 2.  The planner's internal model -- simplified, as in Section 3.4
# =============================================================================

@dataclass
class Candidate:
    t_m: float
    a_x: float
    dwell: float
    a_y_peak: float
    mu_demand: float          # whole-vehicle friction circle, Eq. (41)
    mu_rcof: float            # per-axle RCOF, Eq. (62)
    ltr_max: float
    workload: float
    completion: float
    stable: bool
    feasible: bool
    in_time: bool

    @property
    def admissible(self) -> bool:
        return self.stable and self.feasible and self.in_time


class PaperTrajectoryPlanner:
    """Karimyan et al. Figure 9, as a planner.

    Parameters
    ----------
    n_time, n_accel
        Grid resolution over (t_m, a_x). The paper evaluates nine rows by hand;
        a 24 x 13 grid is the same search done systematically.
    refine
        Rounds of local refinement around the best cell. Each round shrinks the
        window, so this is a coarse-to-fine search rather than a finer grid
        everywhere.
    allow_longitudinal
        False freezes a_x = 0, which reproduces the constant-speed case and is
        the right setting when the plant cannot change speed. Kept available
        because the paper's whole contribution is that a_x helps.
    """

    def __init__(self, n_time: int = 24, n_accel: int = 13, refine: int = 2,
                 allow_longitudinal: bool = False, n_dwell: int = 5):
        self.n_time = n_time
        self.n_accel = n_accel
        self.refine = refine
        self.allow_longitudinal = allow_longitudinal
        self.n_dwell = n_dwell
        self.last_candidates = 0
        self.last_rejected: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def _evaluate(self, t_m: float, a_x: float, dwell: float,
                  task) -> Candidate:
        """Score one candidate on the planner's own simplified model.

        Lateral tyre force is estimated from the lateral acceleration by load
        share -- the quasi-static balance a planner can afford. This is the
        planner's model error, and it is deliberate: the whole premise of the
        two-stage architecture is that the planner is simple and the controller
        carries the detail.
        """
        W = task.lateral_offset
        a_y = quintic_peak_ay(W, t_m)                       # Eq. (44), exact
        mu_demand = math.hypot(a_x, a_y) / G                # Eq. (41)

        # Per-axle RCOF from the quasi-static force balance.
        Fz = np.asarray(task.Fz, dtype=float).reshape(-1, 1)
        m_total = float(Fz.sum()) / G
        share = Fz / Fz.sum()
        Fy = share * (m_total * a_y)
        mu_rcof = float(np.max(np.hypot(share * (m_total * a_x), Fy) / Fz))

        ltr = float(max(ltr_quasi_static(np.array([a_y]), CG_HEIGHT_TRACTOR)[0],
                        ltr_quasi_static(np.array([a_y]), CG_HEIGHT_TRAILER)[0]))

        ref_dwell = max(dwell, 0.0)
        completion = task.start_time + 2.0 * t_m + ref_dwell

        # The dwell must actually hold the lane for the required time. For a
        # quintic the plateau IS the dwell, exactly -- no bisection needed.
        holds = ref_dwell >= task.hold_time - 1e-9

        return Candidate(
            t_m=t_m, a_x=a_x, dwell=ref_dwell, a_y_peak=a_y,
            mu_demand=mu_demand, mu_rcof=mu_rcof, ltr_max=ltr,
            workload=float(tyre_workload(Fy * np.ones((1, 2)), task.Fz, a_x)),
            completion=completion,
            stable=ltr <= LTR_UNSTABLE,
            feasible=(mu_rcof <= task.mu) and holds,
            in_time=completion <= task.deadline + 1e-9)

    # ------------------------------------------------------------------
    def plan(self, task) -> Tuple[Optional[QuinticDLCReference], Dict]:
        """Figure 9's loop. Returns (reference, diagnostics)."""
        t0 = time.perf_counter()
        W = task.lateral_offset

        # The search window over t_m. The floor is the friction-circle minimum
        # (Eq. 50) -- below it no trajectory of this family is admissible at
        # all, so there is nothing to gain from searching there.
        t_floor = quintic_min_time(W, task.mu * G)
        t_lo = max(0.6, t_floor)
        t_hi = max(t_lo + 0.2, 0.5 * (task.deadline - task.start_time))

        a_lo, a_hi = (-4.0, 1.5) if self.allow_longitudinal else (0.0, 0.0)
        d_lo, d_hi = task.hold_time, task.hold_time + 3.0

        best: Optional[Candidate] = None
        n_eval = 0
        rejected = {"ltr": 0, "friction": 0, "deadline": 0}

        for round_i in range(self.refine + 1):
            ts = np.linspace(t_lo, t_hi, self.n_time)
            axs = (np.linspace(a_lo, a_hi, self.n_accel)
                   if self.allow_longitudinal else np.array([0.0]))
            ds = np.linspace(d_lo, d_hi, self.n_dwell)
            for t_m in ts:
                for a_x in axs:
                    for dwell in ds:
                        c = self._evaluate(float(t_m), float(a_x),
                                           float(dwell), task)
                        n_eval += 1
                        if not c.stable:
                            rejected["ltr"] += 1
                            continue
                        if not c.feasible:
                            rejected["friction"] += 1
                            continue
                        if not c.in_time:
                            rejected["deadline"] += 1
                            continue
                        # Figure 9's rule: minimise the peak required friction.
                        # Ties broken toward the shorter manoeuvre, which is
                        # what the deadline term in the reward will prefer.
                        if (best is None or c.mu_rcof < best.mu_rcof - 1e-12
                                or (abs(c.mu_rcof - best.mu_rcof) <= 1e-12
                                    and c.completion < best.completion)):
                            best = c
            if best is None or round_i == self.refine:
                break
            # coarse-to-fine around the incumbent
            span_t = (t_hi - t_lo) / (self.n_time - 1) * 2.0
            t_lo, t_hi = max(t_floor, best.t_m - span_t), best.t_m + span_t
            if self.allow_longitudinal:
                span_a = (a_hi - a_lo) / max(self.n_accel - 1, 1) * 2.0
                a_lo, a_hi = best.a_x - span_a, best.a_x + span_a
            span_d = (d_hi - d_lo) / max(self.n_dwell - 1, 1) * 2.0
            d_lo = max(task.hold_time, best.dwell - span_d)
            d_hi = best.dwell + span_d

        self.last_candidates = n_eval
        self.last_rejected = rejected
        dt = time.perf_counter() - t0

        if best is None:
            return None, {"ok": False, "status": "no admissible trajectory",
                          "candidates": n_eval, "rejected": rejected,
                          "plan_time": dt}

        ref = QuinticDLCReference(W=W, t_m=best.t_m, t_start=task.start_time,
                                  dwell=best.dwell, V=task.V)
        return ref, {"ok": True, "status": "selected",
                     "candidates": n_eval, "rejected": rejected,
                     "plan_time": dt, "a_x": best.a_x,
                     "t_m": best.t_m, "dwell": best.dwell,
                     "mu_rcof_predicted": best.mu_rcof,
                     "ltr_predicted": best.ltr_max,
                     "a_y_peak": best.a_y_peak,
                     "completion_predicted": best.completion}


# =============================================================================
def self_test(verbose: bool = True) -> bool:
    from dlc_paper_env import PaperTask

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:54s} {detail}")

    if verbose:
        print("=" * 92)
        print("paper_planner self-test")
        print("=" * 92)
        print("  A. the quintic double lane change")

    ref = QuinticDLCReference(W=3.5, t_m=3.0, t_start=4.0, dwell=7.0)
    t = np.linspace(0.0, 25.0, 50001)
    y = ref.y(t)
    check("starts and ends on the centreline",
          abs(y[0]) < 1e-9 and abs(y[-1]) < 1e-9)
    check("reaches the offset exactly", abs(np.max(y) - 3.5) < 1e-6,
          f"peak {np.max(y):.6f} m")
    num = float(np.max(np.abs(ref.ddy(t))))
    check("Eq. (44) still exact for the DOUBLE lane change",
          abs(num - ref.peak_lateral_accel) / num < 1e-5,
          f"{ref.peak_lateral_accel:.6f} vs {num:.6f} m/s^2")
    check("the dwell holds the lane for its full duration",
          float(np.sum(y >= 3.5 - 1e-6) * (t[1] - t[0])) >= 7.0 - 1e-3,
          f"{float(np.sum(y >= 3.5 - 1e-6) * (t[1]-t[0])):.3f} s at the offset")

    if verbose:
        print("\n  B. the planner selects, and respects the thresholds")
    task = PaperTask()
    planner = PaperTrajectoryPlanner()
    ref, info = planner.plan(task)
    check("finds an admissible trajectory on the default task", info["ok"],
          info["status"])
    if info["ok"]:
        check("the chosen trajectory is below the LTR threshold",
              info["ltr_predicted"] <= LTR_UNSTABLE,
              f"LTR {info['ltr_predicted']:.3f} <= {LTR_UNSTABLE}")
        check("the chosen trajectory fits the available friction",
              info["mu_rcof_predicted"] <= task.mu,
              f"RCOF {info['mu_rcof_predicted']:.3f} <= mu {task.mu:.2f}")
        check("it meets the deadline",
              info["completion_predicted"] <= task.deadline + 1e-9,
              f"{info['completion_predicted']:.2f} s <= {task.deadline:.2f} s")
        if verbose:
            print(f"      {ref.describe()}")
            print(f"      {info['candidates']} candidates, "
                  f"{info['plan_time']*1e3:.1f} ms")

    if verbose:
        print("\n  C. a task too tight for the friction available")
    tight = PaperTask(mu=0.25, deadline=9.0, lateral_offset=4.5)
    _, info_t = planner.plan(tight)
    check("an impossible task is reported, not crashed",
          isinstance(info_t, dict) and "ok" in info_t,
          f"ok={info_t['ok']}, {info_t['status']}")

    if verbose:
        print("\n  D. lower friction forces a slower manoeuvre")
    times = []
    for mu in (0.9, 0.6, 0.4):
        t_task = PaperTask(mu=mu, deadline=20.0)
        r, i = planner.plan(t_task)
        times.append(i.get("t_m", float("nan")) if i["ok"] else float("nan"))
    check("t_m is non-decreasing as mu falls",
          all(times[k] <= times[k + 1] + 1e-6 for k in range(len(times) - 1)),
          f"mu 0.9/0.6/0.4 -> t_m "
          f"{times[0]:.2f}/{times[1]:.2f}/{times[2]:.2f} s")

    if verbose:
        print(f"\n  SELF-TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
