"""
dlc_baselines.py -- What the RL planner has to beat, in the simple DLC setting
==============================================================================

Three baselines, all driven through the SAME tracking MPC and the SAME
nonlinear plant as the RL planner. Only the planner differs.

1. `heuristic_action`  -- a fixed rule: use 75 % of the allowed transition
   rate. No learning, no optimisation. The "is any of this worth it?" bar.

2. `DLCTrajectoryPlanner` -- the two-MPC baseline, i.e. the method the paper
   compares against and the one your current approach uses. A planning
   optimisation produces the reference trajectory, then the tracking MPC
   follows it. Two optimisations in series.

   Formulated as a convex QP over the manoeuvre, in TIME:

       states   [y, y_dot, a]        a = lateral acceleration
       control  j = da/dt            (lateral jerk)
       minimise sum( j^2 + w_a*a^2 )
       s.t.     |a| <= a_y_max                     (the physical budget)
                y(0) = y_dot(0) = a(0) = 0
                y(t) >= offset - tol   during the hold window
                |y(t)| <= tol          after the deadline
                y_dot, a -> 0 at the end

   This is the honest analogue of the arc-length planning MPC in
   `baselines.py`: it optimises a point-mass trajectory subject to the same
   acceleration budget the RL action space is scaled by. Like that one, it has
   no trailer in its model.

3. `offline_optimum` -- a (1+lambda) evolution strategy over the 4-D action,
   ~80 closed-loop simulations per task. Not deployable; it bounds what the
   action space can reach, so the RL score can be read as a fraction of the
   attainable optimum rather than as a bare number.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ttv_dlc_tracking import FailureLimits, dlc_config, run_dlc_tracking
from dlc_rl_env import (ACTION_DIM, DLCRewardWeights, DLCTask, DLCPlannerEnv,
                        compute_reward, peak_lateral_accel)

G = 9.81


# =============================================================================
# 1.  A planned trajectory, wrapped so the tracking MPC can consume it
# =============================================================================

class TrajectoryReference:
    """Duck-types `DLCReference`: .y(t), .dy(t), .ddy(t), .psi(t).

    Holds a numerically planned trajectory instead of a tanh formula, so the
    same `run_dlc_tracking` drives it without modification.
    """

    def __init__(self, t: np.ndarray, y: np.ndarray, dy: np.ndarray,
                 ddy: np.ndarray, V: float):
        self.t_grid, self._y, self._dy, self._ddy, self.V = t, y, dy, ddy, V
        # kept so `describe`-style callers still work
        self.amplitude = float(np.max(np.abs(y)) / 2.0)
        self.rate = float("nan")

    def _i(self, t, arr):
        return np.interp(np.asarray(t, dtype=float), self.t_grid, arr,
                         left=arr[0], right=arr[-1])

    def y(self, t):
        return self._i(t, self._y)

    def dy(self, t):
        return self._i(t, self._dy)

    def ddy(self, t):
        return self._i(t, self._ddy)

    def psi(self, t):
        return np.arctan2(self.dy(t), self.V)


# =============================================================================
# 2.  The two-MPC baseline: a planning QP, then the tracking MPC
# =============================================================================

@dataclass
class PlanResult:
    ref: Optional[TrajectoryReference]
    solve_time: float
    status: str
    ok: bool


class DLCTrajectoryPlanner:
    """Convex QP trajectory planner. Built once, re-solved per task."""

    def __init__(self, dt: float = 0.2, horizon: float = 20.0,
                 w_a: float = 0.05, tol: float = 0.03,
                 max_iter: int = 400):
        # `tol` MUST be tighter than the tolerance `completion_time` uses to
        # decide the manoeuvre is finished (0.05 m). At tol = 0.15 the planner
        # legitimately satisfied its own corridor while the reward still
        # counted the manoeuvre as running, and the baseline scored a spurious
        # "missed deadline". That was a measurement mismatch in this file, not
        # a property of the planner.
        self.dt = dt
        self.n = int(round(horizon / dt))
        self.horizon = self.n * dt
        self.w_a = w_a
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    def plan(self, task: DLCTask) -> PlanResult:
        import casadi as ca
        n, dt = self.n, self.dt
        t = np.arange(n + 1) * dt

        Y = ca.SX.sym("Y", n + 1)
        D = ca.SX.sym("D", n + 1)      # y_dot
        A = ca.SX.sym("A", n + 1)      # lateral acceleration
        J = ca.SX.sym("J", n)          # jerk

        obj = 0
        g = [Y[0], D[0], A[0]]         # start from rest on the lane centre
        for k in range(n):
            obj += dt * (J[k] ** 2 + self.w_a * A[k + 1] ** 2)
            # trapezoidal integration of the triple integrator
            g.append(Y[k + 1] - (Y[k] + dt * D[k] + 0.5 * dt ** 2 * A[k]))
            g.append(D[k + 1] - (D[k] + dt * A[k] + 0.5 * dt ** 2 * J[k]))
            g.append(A[k + 1] - (A[k] + dt * J[k]))
        n_eq = len(g)

        # path constraints on y, as inequality rows
        for k in range(n + 1):
            g.append(Y[k])
        gv = ca.vertcat(*g)

        w = ca.vertcat(Y, D, A, J)
        nlp = {"f": obj, "x": w, "g": gv}
        opts = {"ipopt.max_iter": self.max_iter, "ipopt.print_level": 0,
                "ipopt.sb": "yes", "print_time": False,
                "ipopt.acceptable_tol": 1e-8}
        solver = ca.nlpsol("dlc_planner", "ipopt", nlp, opts)

        nw = 3 * (n + 1) + n
        lbx = -np.inf * np.ones(nw)
        ubx = np.inf * np.ones(nw)
        a_max = task.a_y_max
        o_a = 2 * (n + 1)
        lbx[o_a:o_a + n + 1] = -a_max          # THE physical budget
        ubx[o_a:o_a + n + 1] = a_max

        # terminal: at rest, on the centreline
        lbx[n] = ubx[n] = 0.0                              # Y[n]
        lbx[(n + 1) + n] = ubx[(n + 1) + n] = 0.0          # D[n]
        lbx[o_a + n] = ubx[o_a + n] = 0.0                  # A[n]

        lbg = np.zeros(n_eq + n + 1)
        ubg = np.zeros(n_eq + n + 1)

        # --- the task, as a corridor on y(t) --------------------------------
        off = task.lateral_offset
        hold_lo = task.start_time
        hold_hi = task.start_time + task.hold_time
        for k in range(n + 1):
            lo, hi = -self.tol, off + 0.6
            if hold_lo <= t[k] <= hold_hi:
                lo = off - self.tol                 # must be in the other lane
            if t[k] >= task.deadline:
                hi = self.tol                       # must be back by the deadline
                lo = -self.tol
            lbg[n_eq + k] = lo
            ubg[n_eq + k] = hi

        w0 = np.zeros(nw)
        t0 = time.perf_counter()
        try:
            sol = solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            st = solver.stats()
            ok = bool(st.get("success", False))
            status = str(st.get("return_status", "?"))
        except Exception as exc:                       # pragma: no cover
            return PlanResult(None, time.perf_counter() - t0, str(exc), False)
        dtsolve = time.perf_counter() - t0
        if not ok:
            return PlanResult(None, dtsolve, status, False)

        wo = np.asarray(sol["x"]).flatten()
        y = wo[:n + 1]
        d = wo[n + 1:2 * (n + 1)]
        a = wo[o_a:o_a + n + 1]
        return PlanResult(TrajectoryReference(t, y, d, a, task.V),
                          dtsolve, status, True)


def run_mpc_planner(task: DLCTask, planner: DLCTrajectoryPlanner,
                    w: DLCRewardWeights = DLCRewardWeights(),
                    limits: Optional[FailureLimits] = None
                    ) -> Tuple[float, Dict]:
    pr = planner.plan(task)
    if not pr.ok or pr.ref is None:
        return (w.fail_base, {"metrics": {},
                              "parts": {"cause": "planner: " + pr.status},
                              "plan_time": pr.solve_time, "planner_ok": False})
    cfg = dlc_config(V=task.V, mu=task.mu, Qpsi=1.0, Qphi=25.0, Qq=8.0)
    res = run_dlc_tracking(cfg, pr.ref, task.sim_time, backend="fast",
                           limits=limits or FailureLimits(enabled=True))
    reward, parts = compute_reward(res, task, w)
    return reward, {"metrics": res.metrics, "parts": parts, "result": res,
                    "ref": pr.ref, "plan_time": pr.solve_time,
                    "planner_ok": True,
                    "peak_a_y_g": peak_lateral_accel(pr.ref,
                                                     task.sim_time + 5.0) / G}


# =============================================================================
# 3.  Offline optimum over the RL action space
# =============================================================================

def offline_optimum(task: DLCTask, budget: int = 80, seed: int = 0,
                    w: DLCRewardWeights = DLCRewardWeights()
                    ) -> Tuple[float, np.ndarray]:
    env = DLCPlannerEnv(randomise=False, weights=w)
    rng = np.random.default_rng(seed)

    def f(a):
        env.reset(task)
        return env.step(a)[1]

    best_a = np.zeros(ACTION_DIM)
    best_r = f(best_a)
    sigma, lam, used = 0.6, 8, 1
    while used < budget:
        cand = np.clip(best_a + sigma * rng.normal(size=(lam, ACTION_DIM)), -1, 1)
        vals = [f(c) for c in cand]
        used += lam
        i = int(np.argmax(vals))
        if vals[i] > best_r:
            best_r, best_a = vals[i], cand[i]
            sigma *= 1.15
        else:
            sigma *= 0.82
        sigma = float(np.clip(sigma, 0.02, 1.0))
    return best_r, best_a


# =============================================================================
if __name__ == "__main__":
    import json
    np.set_printoptions(precision=3, suppress=True)
    from dlc_rl_env import heuristic_action

    print("=" * 92)
    print("DLC baselines on one task")
    print("=" * 92)
    task = DLCTask()
    print(f"  V={task.V} m/s, offset {task.lateral_offset} m, "
          f"hold {task.hold_time} s from t={task.start_time} s, "
          f"deadline {task.deadline} s, a_y budget {task.a_y_budget} g")

    planner = DLCTrajectoryPlanner()
    env = DLCPlannerEnv(randomise=False)

    env.reset(task)
    r_h, i_h = env.step(heuristic_action(task))[1:4:2]
    r_m, i_m = run_mpc_planner(task, planner)
    t0 = time.perf_counter()
    r_o, a_o = offline_optimum(task, budget=80)
    t_o = time.perf_counter() - t0

    print(f"\n  {'method':>20s} {'reward':>8s} {'a_y [g]':>8s} {'RMS e_y':>9s} "
          f"{'peak phi':>9s} {'t_done':>7s} {'plan ms':>9s} {'outcome':>16s}")
    for name, rw, info, tp in (("heuristic", r_h, i_h, 0.0),
                               ("2-MPC planner", r_m, i_m,
                                i_m.get("plan_time", 0) * 1e3),
                               ("offline optimum", r_o, None, t_o * 1e3)):
        if info is None:
            env.reset(task)
            info = env.step(a_o)[3]
        m = info.get("metrics", {})
        p = info.get("parts", {})
        print(f"  {name:>20s} {rw:8.4f} {info.get('peak_a_y_g', float('nan')):8.3f} "
              f"{m.get('rms_lateral_error', float('nan')):9.4f} "
              f"{math.degrees(m.get('peak_articulation_angle', float('nan'))):8.2f}d "
              f"{p.get('completion_time', float('nan')):7.2f} {tp:8.1f}m "
              f"{str(p.get('cause'))[:16]:>16s}")
    print(f"\n  offline-optimum action: {a_o}")
    json.dump({"heuristic": r_h, "mpc2": r_m, "optimum": r_o},
              open("out/dlc_baselines_single.json", "w"), indent=2)
    print("  wrote out/dlc_baselines_single.json")
