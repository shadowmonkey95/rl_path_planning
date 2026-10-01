"""
dlc_benchmark.py -- RL planner vs the 2-MPC planner vs a heuristic, simple DLC
==============================================================================

The same benchmark shape as `benchmark.py`, on the simple setting. All four
methods produce a REFERENCE TRAJECTORY; each is then driven by the same
tracking MPC through the same nonlinear tractor-trailer plant and scored by the
same reward. Only the planner differs.

    heuristic        a fixed rule, no optimisation          -- the "is it worth it" bar
    2-MPC planner    planning QP + tracking MPC             -- the method to beat
    RL planner       trained policy + tracking MPC          -- ours
    offline optimum  ~80 closed-loop sims per task          -- the attainable ceiling

Run on tasks the agent never trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch

from dlc_rl_env import (DLCPlannerEnv, DLCRewardWeights, DLCTask,
                        heuristic_action, ACTION_DIM)
from dlc_baselines import DLCTrajectoryPlanner, run_mpc_planner, offline_optimum

W = DLCRewardWeights()


def _row(reward: float, info: Dict, plan_ms: float) -> Dict:
    m = info.get("metrics", {}) or {}
    p = info.get("parts", {}) or {}
    return {
        "reward": float(reward),
        "success": p.get("cause") == "success",
        "plan_time_ms": plan_ms,
        "peak_a_y_g": info.get("peak_a_y_g", np.nan),
        "completion_time": p.get("completion_time", np.nan),
        "deadline_margin": p.get("deadline_margin", np.nan),
        "rms_lateral_error": m.get("rms_lateral_error", np.nan),
        "peak_lateral_error": m.get("peak_lateral_error", np.nan),
        "peak_articulation_deg": math.degrees(m.get("peak_articulation_angle", np.nan))
        if m else np.nan,
        "max_front_slip_deg": math.degrees(m.get("max_front_slip", np.nan))
        if m else np.nan,
        "peak_steer_deg": math.degrees(m.get("peak_steering_command", np.nan))
        if m else np.nan,
        "completed_steps": m.get("completed_steps", np.nan),
    }


def _agg(rows: List[Dict]) -> Dict[str, float]:
    def col(k):
        v = [r[k] for r in rows
             if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
        return np.asarray(v) if v else np.array([np.nan])
    return {
        "n": len(rows),
        "reward_mean": float(np.mean(col("reward"))),
        "reward_std": float(np.std(col("reward"))),
        "reward_p10": float(np.percentile(col("reward"), 10)),
        "reward_worst": float(np.min(col("reward"))),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "mean_a_y_g": float(np.mean(col("peak_a_y_g"))),
        "worst_a_y_g": float(np.max(col("peak_a_y_g"))),
        "mean_rms_e_y": float(np.mean(col("rms_lateral_error"))),
        "mean_phi_deg": float(np.mean(col("peak_articulation_deg"))),
        "worst_phi_deg": float(np.max(col("peak_articulation_deg"))),
        "mean_slip_deg": float(np.mean(col("max_front_slip_deg"))),
        "mean_completion_s": float(np.mean(col("completion_time"))),
        # The MEDIAN is the honest headline for planning time. PyTorch's first
        # forward pass in a process costs ~650 ms of lazy initialisation and
        # kernel selection, which is a property of the runtime rather than of
        # the planner, and over 40 tasks that single outlier moved the RL mean
        # from 1.7 ms to 18.9 ms -- reversing the comparison against a QP that
        # takes ~15 ms every time. The actor is warmed up below so this no
        # longer contaminates the sample, and both statistics are reported so
        # the distinction stays visible.
        "plan_ms_median": float(np.median(col("plan_time_ms"))),
        "plan_ms_mean": float(np.mean(col("plan_time_ms"))),
        "plan_ms_max": float(np.max(col("plan_time_ms"))),
    }


# The training loop's own eval set is `make_eval_set(40, 12345)`. It is never
# TRAINED on -- training draws from a separate RNG stream -- but it IS the set
# the learning curve is read off, and therefore the set any human watching the
# run selects a checkpoint against. Reporting the final benchmark on that same
# set would be selection on the test set.
#
# So the benchmark uses a DIFFERENT seed by default. Pass --seed 12345 to
# reproduce the training curve's number and compare the two: a large gap
# between them is evidence of exactly the selection effect this avoids.
TRAIN_EVAL_SEED = 12345
BENCH_SEED = 777


def run(n_tasks: int = 40, agent: str = "out/rl_dlc_agent.pt",
        optimum_budget: int = 80, seed: int = BENCH_SEED) -> Dict:
    from train_rl import load_agent, make_eval_set
    if seed == TRAIN_EVAL_SEED:
        print("  NOTE: this is the training loop's own eval set, not a "
              "fresh one.", flush=True)
    tasks = make_eval_set(n_tasks, seed)
    env = DLCPlannerEnv(randomise=False, weights=W)
    planner = DLCTrajectoryPlanner()
    actor, norm = load_agent(agent)

    # Warm up the actor and the planner BEFORE timing anything. PyTorch's first
    # forward pass pays a one-off ~650 ms initialisation cost, and IPOPT's
    # first solve pays for its own setup; timing those as if they were planning
    # cost measures the runtime, not the method.
    warm_obs = env.reset(tasks[0])
    with torch.no_grad():
        for _ in range(3):
            actor(torch.as_tensor(norm(warm_obs),
                                  dtype=torch.float32).unsqueeze(0),
                  deterministic=True, with_logp=False)
    planner.plan(tasks[0])

    res: Dict[str, List[Dict]] = {k: [] for k in
                                  ("heuristic", "mpc2", "rl", "optimum")}
    for i, task in enumerate(tasks):
        # ---- heuristic ----
        env.reset(task)
        t0 = time.perf_counter()
        ah = heuristic_action(task)
        th = (time.perf_counter() - t0) * 1e3
        _, rh, _, ih = env.step(ah)
        res["heuristic"].append(_row(rh, ih, th))

        # ---- 2-MPC ----
        rm, im = run_mpc_planner(task, planner, W)
        res["mpc2"].append(_row(rm, im, im.get("plan_time", 0.0) * 1e3))

        # ---- RL ----
        o = env.reset(task)
        t0 = time.perf_counter()
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0),
                         deterministic=True, with_logp=False)
        tr = (time.perf_counter() - t0) * 1e3
        _, rr, _, ir = env.step(a.squeeze(0).numpy())
        res["rl"].append(_row(rr, ir, tr))

        # ---- offline optimum ----
        t0 = time.perf_counter()
        ro, ao = offline_optimum(task, budget=optimum_budget, seed=i, w=W)
        to = (time.perf_counter() - t0) * 1e3
        env.reset(task)
        _, _, _, io = env.step(ao)
        res["optimum"].append(_row(ro, io, to))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_tasks} tasks", flush=True)

    summary = {k: _agg(v) for k, v in res.items()}
    paired = {}
    for k in ("mpc2", "heuristic"):
        d = np.array([res["rl"][i]["reward"] - res[k][i]["reward"]
                      for i in range(n_tasks)])
        paired[f"rl_minus_{k}"] = {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "win_rate": float(np.mean(d > 0)),
            "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            if d.std(ddof=1) > 0 else float("nan")}
    opt = summary["optimum"]["reward_mean"]
    for k in ("rl", "mpc2", "heuristic"):
        paired[f"{k}_fraction_of_optimum"] = summary[k]["reward_mean"] / opt
    return {"summary": summary, "paired": paired, "per_task": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--agent", default="out/rl_dlc_agent.pt")
    # The default output name carries the SEED. It used to be a fixed
    # "out/dlc_benchmark.json", which meant the two documented commands (seed
    # 777, then seed 12345) both wrote to the same file and the second silently
    # destroyed the first -- and `make_report.py`, which looks for the
    # seed-named files, then found neither and built a report with no results
    # in it. Deriving the name from the seed makes the documented commands do
    # the right thing with no extra flags.
    ap.add_argument("--out", default=None,
                    help="output json (default: out/dlc_benchmark_s<seed>.json)")
    ap.add_argument("--seed", type=int, default=BENCH_SEED,
                    help=f"task seed; {TRAIN_EVAL_SEED} reproduces the "
                         f"training curve's eval set")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"out/dlc_benchmark_s{args.seed}.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not os.path.exists(args.agent):
        raise SystemExit(
            f"\n  {args.agent} does not exist.\n"
            f"  Train an agent first:\n"
            f"    TTV_ENV=dlc python3 train_rl.py --env dlc --episodes 16000 \\\n"
            f"        --warmup 400 --eval-every 1000 --out out/rl_dlc\n")

    print("=" * 108)
    print(f"DLC benchmark on {args.n} tasks, seed {args.seed} "
          f"(never trained on; seed {BENCH_SEED} was never even watched)")
    print(f"agent {args.agent}  ->  {args.out}")
    print("=" * 108)
    r = run(args.n, args.agent, seed=args.seed)

    order = ["heuristic", "mpc2", "rl", "optimum"]
    label = {"heuristic": "fixed heuristic", "mpc2": "2-MPC planner",
             "rl": "RL planner (ours)", "optimum": "offline optimum*"}
    print(f"\n  {'method':>20s} {'reward':>8s} {'sd':>6s} {'p10':>7s} {'worst':>7s} "
          f"{'succ':>6s} {'a_y[g]':>7s} {'RMS e_y':>8s} {'phi':>7s} {'slip':>7s} "
          f"{'t_done':>7s} {'plan ms':>9s}")
    for k in order:
        s = r["summary"][k]
        print(f"  {label[k]:>20s} {s['reward_mean']:8.4f} {s['reward_std']:6.3f} "
              f"{s['reward_p10']:7.3f} {s['reward_worst']:7.3f} "
              f"{s['success_rate']*100:5.1f}% {s['mean_a_y_g']:7.3f} "
              f"{s['mean_rms_e_y']:8.4f} {s['mean_phi_deg']:6.2f}d "
              f"{s['mean_slip_deg']:6.2f}d {s['mean_completion_s']:7.2f} "
              f"{s['plan_ms_median']:9.2f}")
    print("  * not deployable: ~80 closed-loop simulations per task")
    print("  plan ms = MEDIAN over tasks; see `_agg` for why not the mean")

    print("\n  paired differences (same tasks, RL minus baseline):")
    for k, v in r["paired"].items():
        if isinstance(v, dict):
            print(f"    {k:28s} mean {v['mean']:+.4f}  median {v['median']:+.4f}  "
                  f"win {v['win_rate']*100:5.1f}%  t = {v['t']:+.2f}")
        else:
            print(f"    {k:28s} {v*100:.1f}%")

    # per_task is kept: `make_report.py` draws the paired-difference and
    # distribution figures from it, and a summary alone cannot support those.
    json.dump({"summary": r["summary"], "paired": r["paired"],
               "per_task": r["per_task"], "seed": args.seed,
               "n_tasks": args.n, "agent": args.agent},
              open(args.out, "w"), indent=2, default=str)
    print(f"\n  wrote {args.out}")