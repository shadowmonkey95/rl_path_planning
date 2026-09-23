"""
hard_cases.py -- Where does the 2-MPC planner's model run out?
==============================================================

On the nominal scenario distribution the 2-MPC baseline reaches ~93 % of the
attainable optimum (benchmark.py). That is a strong baseline and it should be
said plainly. So the useful question is not "is RL better" but "where is the
planner's MODEL wrong", because that is the only place a learned planner can
have a durable advantage.

The planner optimises a kinematic point-mass path subject to

    |kappa| <= margin * SRT * g / V^2

i.e. it limits the TRACTOR's steady-state lateral acceleration. Two things it
structurally cannot see:

  1. REARWARD AMPLIFICATION. The trailer's lateral acceleration exceeds the
     tractor's at the manoeuvre frequency -- peak RWA 1.26 at 0.30 Hz for this
     rig (truck_params.py). A path that holds the tractor at 0.85*SRT can put
     the TRAILER over SRT, and the trailer is the unit that rolls.
  2. LOAD STATE. SRT is a function of CoG height, which is a function of
     payload. The planner is given kappa_ceiling, so it gets the static part
     right -- but RWA and the yaw-mode damping also move with load, and those
     are not in its model at all.

This script looks for the region of scenario space where that matters, so the
comparison is run somewhere the question is live rather than somewhere both
methods are comfortably inside the envelope.
"""

from __future__ import annotations

import json
import numpy as np

from highway_env import (Scenario, TruckHighwayEnv, RewardWeights,
                         action_to_path, rollout, compute_reward)
from fast_mpc import CondensedMPC
from baselines import MPCPlanner, run_mpc_planner, heuristic_action
from path_gen import min_length_for_offset
import truck_params as TP

W = RewardWeights()


def ood_scenario_set(n: int, seed: int = 999):
    """Scenarios OUTSIDE the training ranges, to measure extrapolation.

    A learned planner is only as good as its training distribution, and the
    honest way to say so is to measure it. Everything here is beyond what the
    randomiser ever sampled:
      * V 25-29 m/s, above the 25 m/s cap (90-104 km/h: over the EU limiter,
        but a downhill overspeed or a non-EU market);
      * mu 0.22-0.34, below the 0.35 floor (standing water, packed snow);
      * obstacle as close as 55 m, below the 70 m floor;
      * initial disturbance 3x the training sigma.
    The optimisation-based planner should degrade gracefully here because its
    constraints are explicit functions of V and mu; the policy has to
    extrapolate.
    """
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(Scenario(
            V=float(r.uniform(25.0, 29.0)),
            mu=float(r.uniform(0.22, 0.34)),
            lane_width=float(r.uniform(3.10, 3.30)),
            obstacle_x=float(r.uniform(55.0, 110.0)),
            obstacle_len=float(r.uniform(10.0, 20.0)),
            obstacle_half_width=float(r.uniform(1.2, 1.7)),
            must_return=True,
            payload_fraction=float(r.uniform(0.80, 1.0)),
            e_y0=float(r.normal(0.0, 0.60)),
            e_psi0=float(r.normal(0.0, 0.024)),
            phi0=float(r.normal(0.0, 0.024)),
        ))
    return out


def hard_scenario_set(n: int, seed: int = 777):
    """Scenarios that force a SHORT, FAST manoeuvre, where RWA bites.

    Differences from the nominal randomiser:
      * obstacle 70-140 m ahead instead of 110-240, so the manoeuvre has to be
        near the curvature ceiling instead of comfortably long;
      * always must_return, so the rig is excited twice at the trailer's yaw
        frequency;
      * payload 0.6-1.0, i.e. always near the worst rollover threshold;
      * a real initial disturbance (the rig is not perfectly centred and
        straight when the obstacle appears).
    """
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(Scenario(
            V=float(r.uniform(21.0, 25.0)),
            mu=float(r.uniform(0.50, 0.90)),
            lane_width=float(r.uniform(3.25, 3.60)),
            obstacle_x=float(r.uniform(70.0, 140.0)),
            obstacle_len=float(r.uniform(8.0, 18.0)),
            obstacle_half_width=float(r.uniform(1.0, 1.6)),
            must_return=True,
            payload_fraction=float(r.uniform(0.60, 1.0)),
            e_y0=float(r.normal(0.0, 0.25)),
            e_psi0=float(r.normal(0.0, 0.010)),
            phi0=float(r.normal(0.0, 0.010)),
        ))
    return out


# =============================================================================
def rwa_gap_study(verbose: bool = True):
    """How much lateral acceleration does the TRAILER actually see, relative to
    the ceiling the planner imposes on the TRACTOR?"""
    if verbose:
        print("=" * 92)
        print("A.  The gap the planner's model cannot see: trailer vs tractor")
        print("=" * 92)
        print(f"  {'V':>5s} {'payload':>8s} {'SRT[g]':>7s} {'L/Lmin':>7s} "
              f"{'path ay[g]':>11s} {'tractor':>8s} {'trailer':>8s} "
              f"{'LTR':>6s} {'RWA':>6s} {'verdict':>10s}")
    rows = []
    for V in (22.0, 25.0):
        for payload in (0.7, 1.0):
            sc = Scenario(V=V, mu=0.85, payload_fraction=payload,
                          obstacle_x=110.0, must_return=True)
            cfg = sc.config()
            mpc = CondensedMPC(cfg, use_rate_constraint=True)
            kcap = sc.kappa_ceiling
            Lmin = min_length_for_offset(sc.dy_required, kcap, 0.5, 0.5)
            for ratio in (1.0, 1.15, 1.4, 1.8):
                from path_gen import bezier_offset_path
                L = ratio * Lmin
                path = bezier_offset_path(
                    lead_in=10.0,
                    transitions=[(sc.dy_required, L, 0.5, 0.5),
                                 (0.0, 30.0, 0.0, 0.0),
                                 (-sc.dy_required, L, 0.5, 0.5)],
                    tail=30.0, ds=0.5)
                res = rollout(path, cfg, sc,
                              max_time=1.3 * path.s[-1] / V + 2.0, mpc=mpc)
                m = res.metrics
                if not m or "peak_LTR" not in m:
                    continue
                row = {"V": V, "payload": payload, "srt": sc.srt,
                       "ratio": ratio,
                       "path_ay_g": path.max_lateral_accel(V) / 9.81,
                       "tractor_g": m["peak_ay_tractor_g"],
                       "trailer_g": m["peak_ay_trailer_g"],
                       "ltr": m["peak_LTR"], "rwa": m["RWA"],
                       "rollover": m["rollover"]}
                rows.append(row)
                if verbose:
                    print(f"  {V:5.1f} {payload:8.2f} {sc.srt:7.3f} {ratio:7.2f} "
                          f"{row['path_ay_g']:11.3f} {row['tractor_g']:8.3f} "
                          f"{row['trailer_g']:8.3f} {row['ltr']:6.3f} "
                          f"{row['rwa']:6.3f} "
                          f"{('ROLLOVER' if row['rollover'] else 'ok'):>10s}")
    if verbose:
        worst = max(rows, key=lambda r: r["ltr"])
        print(f"\n  worst LTR seen: {worst['ltr']:.3f} at V={worst['V']}, "
              f"payload={worst['payload']}, L/Lmin={worst['ratio']}")
        print("  -> the path is inside the curvature ceiling in EVERY row, yet the")
        print("     trailer's lateral acceleration is consistently above the")
        print("     tractor's. A planner that constrains the tractor is not")
        print("     constraining the thing that rolls.")
    return rows


# =============================================================================
def planner_on_hard_set(n: int = 40, verbose: bool = True):
    scens = hard_scenario_set(n)
    planner = MPCPlanner(ds=2.0, horizon_m=400.0)
    env = TruckHighwayEnv(randomise=False)
    cache = {}

    def mpc_for(cfg):
        k = (round(cfg.V, 4), round(cfg.m2, 2), round(cfg.mu, 4))
        if k not in cache:
            cache[k] = CondensedMPC(cfg, use_rate_constraint=True)
        return cache[k]

    out = {"mpc2": [], "heuristic": []}
    for sc in scens:
        tm = mpc_for(sc.config())
        r2, i2 = run_mpc_planner(sc, planner, W, mpc=tm)
        out["mpc2"].append((r2, i2))
        env.reset(sc)
        _, rh, _, ih = env.step(heuristic_action(sc))
        out["heuristic"].append((rh, ih))

    if verbose:
        print("\n" + "=" * 92)
        print(f"B.  The baselines on {n} HARD scenarios "
              f"(obstacle 70-140 m, laden, disturbed start)")
        print("=" * 92)
        print(f"  {'method':>16s} {'reward':>8s} {'worst':>8s} {'succ':>6s} "
              f"{'coll':>5s} {'roll':>5s} {'meanLTR':>8s} {'worstLTR':>9s} "
              f"{'dep':>6s} {'worst dep':>10s}")
        for k, rows in out.items():
            rw = np.array([r for r, _ in rows])
            ms = [i.get("metrics", {}) or {} for _, i in rows]
            succ = np.mean([i["parts"].get("cause") == "success" for _, i in rows])
            coll = np.mean([m.get("collision", False) for m in ms])
            roll = np.mean([m.get("rollover", False) for m in ms])
            ltr = np.array([m.get("peak_LTR", np.nan) for m in ms], dtype=float)
            dep = np.array([m.get("lane_departure", np.nan) for m in ms], dtype=float)
            print(f"  {k:>16s} {rw.mean():8.4f} {rw.min():8.4f} {succ*100:5.1f}% "
                  f"{coll*100:4.0f}% {roll*100:4.0f}% {np.nanmean(ltr):8.3f} "
                  f"{np.nanmax(ltr):9.3f} {np.nanmean(dep):6.3f} "
                  f"{np.nanmax(dep):10.3f}")
    return out


if __name__ == "__main__":
    rows = rwa_gap_study()
    hard = planner_on_hard_set(40)
    json.dump({"rwa_gap": rows}, open("out/hard_cases.json", "w"),
              indent=2, default=str)
    print("\n  wrote out/hard_cases.json")
