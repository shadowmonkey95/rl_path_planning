"""bench_hard.py -- the same three planners on the HARD scenario set.

The nominal set is where both methods are comfortably inside the envelope.
This is the set that forces a short, fast manoeuvre near the curvature ceiling,
with a disturbed initial condition -- where a planner working off a kinematic
point-mass surrogate should start to come apart.
"""
import json, time
import numpy as np, torch
from highway_env import TruckHighwayEnv, RewardWeights
from fast_mpc import CondensedMPC
from baselines import MPCPlanner, run_mpc_planner, heuristic_action
from hard_cases import hard_scenario_set, ood_scenario_set
from train_rl import load_agent
from benchmark import _row, _agg

import sys
W = RewardWeights()
WHICH = sys.argv[1] if len(sys.argv) > 1 else "hard"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
scens = hard_scenario_set(N) if WHICH == "hard" else ood_scenario_set(N)
TITLE = ("HARD scenarios (in-distribution tail): obstacle 70-140 m, laden, "
         "mu 0.50-0.90, disturbed start" if WHICH == "hard" else
         "OUT-OF-DISTRIBUTION: V 25-29 m/s, mu 0.22-0.34, obstacle 55-110 m "
         "-- all beyond the training ranges")
env = TruckHighwayEnv(randomise=False)
planner = MPCPlanner(ds=2.0, horizon_m=400.0)
actor, norm = load_agent("out/rl_agent.pt")
cache = {}
def mpc_for(cfg):
    k = (round(cfg.V,4), round(cfg.m2,2), round(cfg.mu,4))
    if k not in cache: cache[k] = CondensedMPC(cfg, use_rate_constraint=True)
    return cache[k]

res = {"rl": [], "mpc2": [], "heuristic": []}
for i, sc in enumerate(scens):
    tm = mpc_for(sc.config())
    obs = env.reset(sc)
    t0 = time.perf_counter()
    with torch.no_grad():
        a, _ = actor(torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0),
                     deterministic=True, with_logp=False)
    t_rl = (time.perf_counter()-t0)*1e3
    _, r, _, info = env.step(a.squeeze(0).numpy())
    res["rl"].append(_row(r, info, t_rl))
    r2, i2 = run_mpc_planner(sc, planner, W, mpc=tm)
    res["mpc2"].append(_row(r2, i2, i2["plan_time"]*1e3))
    env.reset(sc); _, rh, _, ih = env.step(heuristic_action(sc))
    res["heuristic"].append(_row(rh, ih, 0.0))
    if (i+1) % 15 == 0: print(f"  {i+1}/{N}", flush=True)

summ = {k: _agg(v) for k, v in res.items()}
print("\n" + "="*104)
print(TITLE)
print("="*104)
lab = {"heuristic":"fixed heuristic","mpc2":"2-MPC planner","rl":"RL planner (ours)"}
print(f"  {'method':>20s} {'reward':>8s} {'sd':>6s} {'p10':>7s} {'worst':>7s} "
      f"{'succ':>6s} {'coll':>5s} {'roll':>5s} {'LTR':>6s} {'wLTR':>6s} "
      f"{'swept':>6s} {'offtr':>6s} {'dep':>6s} {'wdep':>6s}")
for k in ("heuristic","mpc2","rl"):
    s = summ[k]
    print(f"  {lab[k]:>20s} {s['reward_mean']:8.4f} {s['reward_std']:6.3f} "
          f"{s['reward_p10']:7.3f} {s['reward_worst']:7.3f} "
          f"{s['success_rate']*100:5.1f}% {s['collision_rate']*100:4.0f}% "
          f"{s['rollover_rate']*100:4.0f}% {s['mean_peak_LTR']:6.3f} "
          f"{s['worst_peak_LTR']:6.3f} {s['mean_swept']:6.3f} "
          f"{s['mean_offtrack']:6.3f} {s['mean_lane_departure']:6.3f} "
          f"{s['worst_lane_departure']:6.3f}")
print("\n  paired, RL minus baseline:")
for k in ("mpc2","heuristic"):
    d = np.array([res["rl"][i]["reward"]-res[k][i]["reward"] for i in range(N)])
    print(f"    vs {k:12s} mean {d.mean():+.4f}  median {np.median(d):+.4f}  "
          f"win {np.mean(d>0)*100:5.1f}%  t {d.mean()/(d.std(ddof=1)/np.sqrt(N)):+.2f}")
json.dump(summ, open(f"out/benchmark_{WHICH}.json","w"), indent=2, default=str)
print(f"\n  wrote out/benchmark_{WHICH}.json")
