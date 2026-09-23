"""
baselines.py -- What the RL planner has to beat
===============================================

Three baselines, all driven through the SAME tracking MPC and the SAME
nonlinear plant, so only the PLANNER differs.

1. `MPCPlanner` -- the paper's comparison method: an "Adaptive MPC-based path
   planner" using a nonlinear kinematic model with the dynamic limits imposed
   as constraints, followed by the tracking MPC. Two MPCs in series. This is
   the method the user's current approach uses and the one to beat.

   Formulated in arc length with states [y, theta, kappa] and control
   sigma = dkappa/ds:

       y'      = sin(theta)
       theta'  = kappa
       kappa'  = sigma

   subject to the rollover curvature ceiling, a curvature-rate limit derived
   from the steering actuator, the obstacle as a one-sided corridor, the lane
   corridor, and terminal conditions y=theta=kappa=0. Solved with IPOPT.

2. `heuristic_action` -- a fixed engineering rule: use the shortest
   rollover-safe manoeuvre that fits before the obstacle, with a safety factor.
   No learning, no optimisation. This is the bar that says whether learning is
   worth anything at all.

3. `offline_optimum` -- CMA-ES over the 9-D action for ONE scenario, given a
   large budget. Not deployable (it needs hundreds of closed-loop simulations
   per scenario), but it bounds what the action space can achieve, so the RL
   result can be read as a fraction of the attainable optimum rather than as a
   bare number.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import casadi as ca

from highway_env import (Scenario, action_to_path, rollout, compute_reward,
                         RewardWeights, decode_action)
from path_gen import GeoPath, min_length_for_offset, _offset_geometry_raw
from fast_mpc import CondensedMPC
import truck_params as TP

G = 9.81


# =============================================================================
# 1.  The 2-MPC baseline: nonlinear kinematic planner + tracking MPC
# =============================================================================

@dataclass
class PlannerResult:
    path: Optional[GeoPath]
    solve_time: float
    status: str
    ok: bool
    iterations: int


class MPCPlanner:
    """Nonlinear kinematic path planner, solved as an NLP over arc length.

    The NLP structure is fixed for a given (ds, horizon), so it is built once
    and re-solved per scenario with new parameters -- which is the fairest
    possible version of this baseline, since rebuilding would make it look
    artificially slow.
    """

    def __init__(self, ds: float = 2.0, horizon_m: float = 400.0,
                 w_y: float = 1.0, w_kappa: float = 2.0e5,
                 w_sigma: float = 2.0e8, w_terminal: float = 1.0e3,
                 max_iter: int = 500):
        self.ds = ds
        self.n = int(round(horizon_m / ds))
        self.horizon_m = self.n * ds
        n = self.n

        Y = ca.SX.sym("Y", n + 1)
        TH = ca.SX.sym("TH", n + 1)
        K = ca.SX.sym("K", n + 1)
        SG = ca.SX.sym("SG", n)

        # parameters: kappa_max, sigma_max, y_lo, y_hi, and per-node y_ref and
        # per-node lower bound on y (the obstacle corridor)
        P = ca.SX.sym("P", 4 + 2 * (n + 1))
        kmax, smax, y_lo, y_hi = P[0], P[1], P[2], P[3]
        y_ref = P[4:4 + (n + 1)]
        y_obs_lo = P[4 + (n + 1):]

        g, obj = [], 0
        g.append(Y[0]); g.append(TH[0]); g.append(K[0])         # start on the lane
        for k in range(n):
            # RK2 in arc length on the kinematic model
            obj += (w_y * (Y[k + 1] - y_ref[k + 1]) ** 2
                    + w_kappa * K[k + 1] ** 2
                    + w_sigma * SG[k] ** 2)
            th_m = TH[k] + 0.5 * self.ds * K[k]
            k_m = K[k] + 0.5 * self.ds * SG[k]
            g.append(Y[k + 1] - (Y[k] + self.ds * ca.sin(th_m)))
            g.append(TH[k + 1] - (TH[k] + self.ds * k_m))
            g.append(K[k + 1] - (K[k] + self.ds * SG[k]))
        obj += w_terminal * (Y[n] ** 2 + 1e3 * TH[n] ** 2 + 1e6 * K[n] ** 2)

        self._n_eq = 3 + 3 * n
        # corridor + obstacle, as inequality rows on y
        for k in range(n + 1):
            g.append(Y[k])
        self._g = ca.vertcat(*g)

        w = ca.vertcat(Y, TH, K, SG)
        nlp = {"f": obj, "x": w, "g": self._g, "p": P}
        opts = {"ipopt.max_iter": max_iter, "ipopt.print_level": 0,
                "ipopt.sb": "yes", "ipopt.tol": 1e-8, "print_time": False,
                "ipopt.acceptable_tol": 1e-6}
        self.solver = ca.nlpsol("mpc_planner", "ipopt", nlp, opts)
        self.nw = 3 * (n + 1) + n

    # ------------------------------------------------------------------
    def plan(self, scen: Scenario) -> PlannerResult:
        n, ds = self.n, self.ds
        s_nodes = np.arange(n + 1) * ds

        kmax = scen.kappa_ceiling
        # |dkappa/ds| limit from the steering actuator:
        #   kappa ~ delta / L_eff  =>  dkappa/dt ~ delta_dot / L_eff
        #   dkappa/ds = (dkappa/dt)/V
        L_eff = TP.L1
        smax = TP.DELTA_RATE_MAX / (L_eff * scen.V)

        y_lo_corridor = -scen.lane_width / 2.0 + TP.VEHICLE_WIDTH / 2.0
        y_hi_corridor = scen.lane_width * (scen.n_lanes - 0.5) - TP.VEHICLE_WIDTH / 2.0

        # reference: stay at 0 except alongside the obstacle, where the target
        # is the adjacent lane centre
        x0, x1 = scen.obstacle_x, scen.obstacle_x + scen.obstacle_len
        y_need = scen.obstacle_half_width + TP.VEHICLE_WIDTH / 2.0 + 0.40
        along = (s_nodes >= x0 - 5.0) & (s_nodes <= x1 + 5.0)
        y_ref = np.zeros(n + 1)
        y_ref[along] = max(y_need, scen.lane_width)
        obs_lo = np.full(n + 1, y_lo_corridor)
        obs_lo[along] = y_need

        p = np.concatenate(([kmax, smax, y_lo_corridor, y_hi_corridor],
                            y_ref, obs_lo))

        lbx = -np.inf * np.ones(self.nw)
        ubx = np.inf * np.ones(self.nw)
        o_k = 2 * (n + 1)
        lbx[o_k:o_k + n + 1] = -kmax
        ubx[o_k:o_k + n + 1] = kmax
        o_s = 3 * (n + 1)
        lbx[o_s:] = -smax
        ubx[o_s:] = smax

        lbg = np.zeros(self._n_eq + n + 1)
        ubg = np.zeros(self._n_eq + n + 1)
        lbg[self._n_eq:] = obs_lo
        ubg[self._n_eq:] = y_hi_corridor

        w0 = np.zeros(self.nw)
        w0[:n + 1] = y_ref                      # warm start on the reference

        t0 = time.perf_counter()
        try:
            sol = self.solver(x0=w0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)
            st = self.solver.stats()
            ok = bool(st.get("success", False))
            status = str(st.get("return_status", "?"))
            iters = int(st.get("iter_count", -1))
        except Exception as exc:
            return PlannerResult(None, time.perf_counter() - t0, str(exc), False, -1)
        dt = time.perf_counter() - t0
        if not ok:
            return PlannerResult(None, dt, status, False, iters)

        wopt = np.asarray(sol["x"]).flatten()
        y = wopt[:n + 1]
        th = wopt[n + 1:2 * (n + 1)]
        kap = wopt[2 * (n + 1):3 * (n + 1)]

        # Resample to the simulation grid and rebuild the differential geometry
        # from y(x) so the tracking MPC and the reward see the same quantities
        # as for an RL path.
        path = self._to_geopath(s_nodes, y, scen)
        return PlannerResult(path, dt, status, True, iters)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_geopath(x_nodes: np.ndarray, y_nodes: np.ndarray,
                    scen: Scenario, ds_out: float = 0.5) -> GeoPath:
        """Fit a smoothing spline through the planner's nodes and differentiate
        it analytically, so kappa'' is exact here too."""
        from scipy.interpolate import make_smoothing_spline
        xs = np.arange(x_nodes[0], x_nodes[-1] + 1e-9, ds_out)
        spl = make_smoothing_spline(x_nodes, y_nodes, lam=1e-2)
        y = spl(xs)
        u = spl.derivative(1)(xs)
        v = spl.derivative(2)(xs)
        w = spl.derivative(3)(xs)
        z = np.gradient(w, xs)          # the spline is cubic: 4th deriv via FD
        kappa, kappa_s, kappa_ss = _offset_geometry_raw(u, v, w, z)
        psi = np.arctan(u)
        ds_dx = np.sqrt(1.0 + u ** 2)
        s = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dx[1:] + ds_dx[:-1])
                                             * np.diff(xs))))
        meta = {"generator": "mpc_planner",
                "exact": {"kappa_absmax": float(np.max(np.abs(kappa))),
                          "kappa_s_absmax": float(np.max(np.abs(kappa_s))),
                          "kappa_ss_max": float(np.max(kappa_ss)),
                          "kappa_ss_min": float(np.min(kappa_ss))}}
        return GeoPath(s=s, x=xs, y=y, psi=psi, kappa=kappa, kappa_s=kappa_s,
                       kappa_ss=kappa_ss, meta=meta)


def run_mpc_planner(scen: Scenario, planner: MPCPlanner,
                    w: RewardWeights = RewardWeights(),
                    mpc: Optional[CondensedMPC] = None
                    ) -> Tuple[float, Dict[str, Any]]:
    pr = planner.plan(scen)
    if not pr.ok or pr.path is None:
        return (w.fail_base, {"metrics": {}, "parts": {"cause": "planner: " + pr.status},
                              "plan_time": pr.solve_time, "planner_ok": False})
    cfg = scen.config()
    budget = 1.3 * pr.path.s[-1] / scen.V + 2.0
    res = rollout(pr.path, cfg, scen, max_time=budget,
                  mpc=mpc or CondensedMPC(cfg, use_rate_constraint=True))
    r, parts = compute_reward(res, scen, pr.path, w)
    return r, {"metrics": res.metrics, "parts": parts, "path": pr.path,
               "rollout": res, "plan_time": pr.solve_time, "planner_ok": True,
               "planner_iters": pr.iterations}


# =============================================================================
# 2.  Fixed engineering heuristic
# =============================================================================

def heuristic_action(scen: Scenario, safety_factor: float = 1.35) -> np.ndarray:
    """No learning: pick the shortest rollover-safe manoeuvre that FITS.

    L_target = safety_factor * L_min(rollover), clipped so the out-transition
    finishes before the obstacle. Everything else neutral.
    """
    kcap = scen.kappa_ceiling
    dy = scen.dy_required
    L_min = min_length_for_offset(dy, kcap, 0.5, 0.5)
    L_room = max(L_min, scen.obstacle_x - 10.0)
    L_target = min(safety_factor * L_min, L_room)
    L_max = 2.4 * L_min
    u1 = np.clip((L_target - L_min) / max(L_max - L_min, 1e-9), 0.0, 1.0)

    lead_max = max(5.0, scen.obstacle_x - L_target - 5.0)
    lead_target = max(5.0, min(lead_max, scen.obstacle_x - L_target - 5.0))
    u0 = np.clip((lead_target - 5.0) / max(lead_max - 5.0, 1e-9), 0.0, 1.0)

    a = np.zeros(9)
    a[0] = 2 * u0 - 1
    a[1] = 2 * u1 - 1
    a[5] = 2 * u1 - 1
    a[8] = -1.0                       # no extra lateral trim
    return a


# =============================================================================
# 3.  Offline optimum per scenario (CMA-ES)
# =============================================================================

def offline_optimum(scen: Scenario, budget: int = 240, seed: int = 0,
                    w: RewardWeights = RewardWeights(),
                    mpc: Optional[CondensedMPC] = None) -> Tuple[float, np.ndarray]:
    """(1+lambda)-CMA-ES-lite: isotropic Gaussian search with step adaptation.

    Deliberately simple and dependency-free; with a 240-simulation budget on a
    9-D box it gets close enough to serve as a reference ceiling.
    """
    cfg = scen.config()
    mpc = mpc or CondensedMPC(cfg, use_rate_constraint=True)
    rng = np.random.default_rng(seed)

    def f(a):
        path = action_to_path(a, scen, ds=0.5)
        res = rollout(path, cfg, scen,
                      max_time=1.3 * path.s[-1] / scen.V + 2.0, mpc=mpc)
        return compute_reward(res, scen, path, w)[0]

    best_a = np.zeros(9)
    best_r = f(best_a)
    sigma = 0.6
    lam = 8
    used = 1
    while used < budget:
        cand = np.clip(best_a + sigma * rng.normal(size=(lam, 9)), -1, 1)
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

    print("=" * 84)
    print("Baselines on a fixed scenario")
    print("=" * 84)
    scen = Scenario(V=24.0, mu=0.85, payload_fraction=1.0, obstacle_x=170.0,
                    must_return=True)
    cfg = scen.config()
    mpc = CondensedMPC(cfg, use_rate_constraint=True)

    print("\n  Building the planner NLP (done once, not timed per scenario)...")
    t0 = time.perf_counter()
    planner = MPCPlanner(ds=2.0, horizon_m=400.0)
    print(f"  build: {time.perf_counter()-t0:.2f} s, {planner.nw} variables, "
          f"{planner._g.shape[0]} constraint rows")

    rows = []
    r_h, info_h = None, None
    a_h = heuristic_action(scen)
    from highway_env import TruckHighwayEnv
    env = TruckHighwayEnv(randomise=False)
    env.reset(scen)
    _, r_h, _, info_h = env.step(a_h)
    rows.append(("heuristic", r_h, info_h, 0.0))

    r_m, info_m = run_mpc_planner(scen, planner, mpc=mpc)
    rows.append(("2-MPC planner", r_m, info_m, info_m["plan_time"]))

    t0 = time.perf_counter()
    r_o, a_o = offline_optimum(scen, budget=120, mpc=mpc)
    t_o = time.perf_counter() - t0
    env.reset(scen)
    _, _, _, info_o = env.step(a_o)
    rows.append(("offline optimum", r_o, info_o, t_o))

    print(f"\n  {'method':>18s} {'reward':>8s} {'LTR':>6s} {'RWA':>6s} "
          f"{'swept':>7s} {'jerk':>6s} {'dep':>6s} {'L[m]':>6s} "
          f"{'plan time':>10s} {'outcome':>10s}")
    for name, r, info, tplan in rows:
        m = info.get("metrics", {})
        print(f"  {name:>18s} {r:8.4f} {m.get('peak_LTR', float('nan')):6.3f} "
              f"{m.get('RWA', float('nan')):6.3f} "
              f"{m.get('swept_width', float('nan')):7.3f} "
              f"{m.get('peak_lateral_jerk', float('nan')):6.2f} "
              f"{m.get('lane_departure', float('nan')):6.3f} "
              f"{m.get('manoeuvre_length', float('nan')):6.0f} "
              f"{tplan*1e3:9.1f}ms {str(info['parts'].get('cause'))[:10]:>10s}")

    print(f"\n  planner IPOPT iterations: {info_m.get('planner_iters')}")
    print(f"  heuristic action:        {a_h}")
    print(f"  offline-optimum action: {a_o}")

    json.dump({n: {"reward": r, "metrics": i.get("metrics", {}),
                   "plan_time_s": t} for n, r, i, t in rows},
              open("out/baselines_single.json", "w"), indent=2, default=str)
    print("\n  wrote out/baselines_single.json")
