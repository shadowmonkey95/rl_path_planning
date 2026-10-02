"""
paper_benchmark.py -- The fair comparison, on the paper's physics
==================================================================

Same shape as `dlc_benchmark.py`, with the one change that matters: every
method is scored on a criterion from Karimyan et al. (2024) that none of them
was built around, and the planning baseline optimises that same criterion.

    heuristic        midpoint transition time, no optimisation
    paper planner    Karimyan et al. Fig. 9: reject LTR > 0.6, then minimise
                     peak RCOF. The published method, not a straw man.
    RL planner       the trained policy
    offline optimum  ~80 closed-loop simulations per task; the ceiling of the
                     action space

WHAT CHANGED FROM `dlc_benchmark.py`, AND WHY IT MATTERS
--------------------------------------------------------

Old: the reward was five hand-weighted terms, the policy was trained on it, and
the baseline minimised jerk instead. "RL scores higher on the RL objective" was
nearly tautological.

New: the score is peak required friction (Eq. 62), gated on the rollover
threshold (Fig. 9). The baseline minimises peak required friction. The policy
maximises grip margin, which is the same thing. Both aim at the identical
physical target, so the margin measures SEARCH QUALITY rather than whose
objective was used.

The contest that remains is real and worth stating: the planner evaluates
candidates on a simplified quasi-static model, while the reward is computed
from the full nonlinear closed-loop simulation. Whatever the planner's model
gets wrong is what a learned planner can exploit. That gap is the hypothesis.
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

from dlc_paper_env import (ACTION_DIM, PaperPlannerEnv, PaperRewardWeights,
                           PaperTask, compute_reward, heuristic_action)
from paper_planner import PaperTrajectoryPlanner
from paper_metrics import G, LTR_UNSTABLE
from ttv_dlc_tracking import FailureLimits, dlc_config, run_dlc_tracking

W = PaperRewardWeights()

TRAIN_EVAL_SEED = 12345
BENCH_SEED = 777


def _row(reward: float, info: Dict, plan_ms: float) -> Dict:
    m = info.get("metrics", {}) or {}
    p = info.get("parts", {}) or {}
    return {
        "reward": float(reward),
        "success": p.get("cause") == "success",
        "plan_time_ms": plan_ms,
        "mu_max": p.get("mu_max", np.nan),
        "grip_margin": p.get("grip_margin", np.nan),
        "ltr_max": p.get("ltr_max", np.nan),
        "workload": p.get("workload", np.nan),
        "peak_a_y_g": info.get("peak_a_y_g", np.nan),
        "completion_time": p.get("completion_time", np.nan),
        "deadline_margin": p.get("deadline_margin", np.nan),
        "rms_lateral_error": m.get("rms_lateral_error", np.nan),
        "peak_articulation_deg": math.degrees(m["peak_articulation_angle"])
        if m.get("peak_articulation_angle") is not None else np.nan,
        "max_front_slip_deg": math.degrees(m["max_front_slip"])
        if m.get("max_front_slip") is not None else np.nan,
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
        "mu_max_mean": float(np.mean(col("mu_max"))),
        "mu_max_worst": float(np.max(col("mu_max"))),
        "grip_margin_mean": float(np.mean(col("grip_margin"))),
        "ltr_mean": float(np.mean(col("ltr_max"))),
        "ltr_worst": float(np.max(col("ltr_max"))),
        "mean_a_y_g": float(np.mean(col("peak_a_y_g"))),
        "mean_rms_e_y": float(np.mean(col("rms_lateral_error"))),
        "mean_phi_deg": float(np.mean(col("peak_articulation_deg"))),
        "mean_completion_s": float(np.mean(col("completion_time"))),
        "plan_ms_median": float(np.median(col("plan_time_ms"))),
        "plan_ms_mean": float(np.mean(col("plan_time_ms"))),
    }


def make_eval_set(n: int, seed: int) -> List[PaperTask]:
    sampler = PaperPlannerEnv(randomise=True, seed=seed)
    return [sampler.sample_task() for _ in range(n)]


def run_paper_planner(task: PaperTask, planner: PaperTrajectoryPlanner,
                      env: PaperPlannerEnv):
    """Plan, then drive the plan through the SAME frozen tracking MPC."""
    ref, info = planner.plan(task)
    if not info["ok"] or ref is None:
        return (W.fail_base, {"metrics": {},
                              "parts": {"cause": "planner: " + info["status"]},
                              "plan_time": info["plan_time"]})
    cfg = dlc_config(V=task.V, mu=task.mu, Qpsi=1.0, Qphi=25.0, Qq=8.0)
    res = run_dlc_tracking(cfg, ref, task.sim_time, backend="fast",
                           limits=FailureLimits(enabled=True))
    reward, parts = compute_reward(res, task, env.w)
    return reward, {"metrics": res.metrics, "parts": parts, "ref": ref,
                    "result": res, "plan_time": info["plan_time"],
                    "candidates": info["candidates"],
                    "peak_a_y_g": ref.peak_lateral_accel / G}


def offline_optimum(task: PaperTask, env: PaperPlannerEnv, budget: int = 80,
                    seed: int = 0):
    rng = np.random.default_rng(seed)

    def f(a):
        env.reset(task)
        return env.step(a)[1]

    best_a = np.zeros(ACTION_DIM)
    best_r = f(best_a)
    sigma, lam, used = 0.6, 8, 1
    while used < budget:
        cand = np.clip(best_a + sigma * rng.normal(size=(lam, ACTION_DIM)),
                       -1, 1)
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


def run(n_tasks: int = 40, agent: str = "out/rl_paper_agent.pt",
        optimum_budget: int = 80, seed: int = BENCH_SEED) -> Dict:
    from train_rl import load_agent
    tasks = make_eval_set(n_tasks, seed)
    env = PaperPlannerEnv(randomise=False, weights=W)
    planner = PaperTrajectoryPlanner()
    actor, norm = load_agent(agent)

    warm = env.reset(tasks[0])
    with torch.no_grad():
        for _ in range(3):
            actor(torch.as_tensor(norm(warm),
                                  dtype=torch.float32).unsqueeze(0),
                  deterministic=True, with_logp=False)
    planner.plan(tasks[0])

    res: Dict[str, List[Dict]] = {k: [] for k in
                                  ("heuristic", "planner", "rl", "optimum")}
    for i, task in enumerate(tasks):
        env.reset(task)
        t0 = time.perf_counter()
        ah = heuristic_action(task)
        th = (time.perf_counter() - t0) * 1e3
        _, rh, _, ih = env.step(ah)
        res["heuristic"].append(_row(rh, ih, th))

        rp, ip = run_paper_planner(task, planner, env)
        res["planner"].append(_row(rp, ip, ip.get("plan_time", 0.0) * 1e3))

        o = env.reset(task)
        t0 = time.perf_counter()
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(norm(o),
                                         dtype=torch.float32).unsqueeze(0),
                         deterministic=True, with_logp=False)
        tr = (time.perf_counter() - t0) * 1e3
        _, rr, _, ir = env.step(a.squeeze(0).numpy())
        res["rl"].append(_row(rr, ir, tr))

        t0 = time.perf_counter()
        ro, ao = offline_optimum(task, env, budget=optimum_budget, seed=i)
        to = (time.perf_counter() - t0) * 1e3
        env.reset(task)
        _, _, _, io = env.step(ao)
        res["optimum"].append(_row(ro, io, to))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_tasks} tasks", flush=True)

    summary = {k: _agg(v) for k, v in res.items()}
    paired = {}
    for k in ("planner", "heuristic"):
        d = np.array([res["rl"][i]["reward"] - res[k][i]["reward"]
                      for i in range(n_tasks)])
        paired[f"rl_minus_{k}"] = {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "win_rate": float(np.mean(d > 0)),
            "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            if d.std(ddof=1) > 0 else float("nan")}
    opt = summary["optimum"]["reward_mean"]
    for k in ("rl", "planner", "heuristic"):
        paired[f"{k}_fraction_of_optimum"] = summary[k]["reward_mean"] / opt
    return {"summary": summary, "paired": paired, "per_task": res}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--agent", default="out/rl_paper_agent.pt")
    ap.add_argument("--seed", type=int, default=BENCH_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"out/paper_benchmark_s{args.seed}.json"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not os.path.exists(args.agent):
        raise SystemExit(
            f"\n  {args.agent} does not exist. Train one:\n"
            f"    TTV_ENV=paper python3 train_rl.py --env paper "
            f"--episodes 16000 \\\n"
            f"        --warmup 400 --eval-every 1000 --out out/rl_paper\n")

    print("=" * 112)
    print(f"Paper-physics benchmark on {args.n} tasks, seed {args.seed}")
    print(f"scored on peak required friction (Karimyan et al. 2024, Eq. 62), "
          f"gated on LTR <= {LTR_UNSTABLE}")
    print(f"agent {args.agent}  ->  {args.out}")
    print("=" * 112)
    r = run(args.n, args.agent, seed=args.seed)

    order = ["heuristic", "planner", "rl", "optimum"]
    label = {"heuristic": "fixed heuristic", "planner": "paper planner",
             "rl": "RL planner (ours)", "optimum": "offline optimum*"}
    print(f"\n  {'method':>20s} {'reward':>8s} {'sd':>6s} {'p10':>7s} "
          f"{'worst':>7s} {'succ':>6s} {'mu_max':>7s} {'mu_wst':>7s} "
          f"{'LTR':>6s} {'a_y[g]':>7s} {'t_done':>7s} {'plan ms':>9s}")
    for k in order:
        s = r["summary"][k]
        print(f"  {label[k]:>20s} {s['reward_mean']:8.4f} {s['reward_std']:6.3f} "
              f"{s['reward_p10']:7.3f} {s['reward_worst']:7.3f} "
              f"{s['success_rate']*100:5.1f}% {s['mu_max_mean']:7.3f} "
              f"{s['mu_max_worst']:7.3f} {s['ltr_mean']:6.3f} "
              f"{s['mean_a_y_g']:7.3f} {s['mean_completion_s']:7.2f} "
              f"{s['plan_ms_median']:9.2f}")
    print("  * not deployable: ~80 closed-loop simulations per task")
    print("  mu_max = peak required friction (lower is better); "
          "plan ms = median")

    print("\n  paired differences (same tasks, RL minus baseline):")
    for k, v in r["paired"].items():
        if isinstance(v, dict):
            print(f"    {k:28s} mean {v['mean']:+.4f}  median {v['median']:+.4f}"
                  f"  win {v['win_rate']*100:5.1f}%  t = {v['t']:+.2f}")
        else:
            print(f"    {k:28s} {v*100:.1f}%")

    json.dump({"summary": r["summary"], "paired": r["paired"],
               "per_task": r["per_task"], "seed": args.seed,
               "n_tasks": args.n, "agent": args.agent},
              open(args.out, "w"), indent=2, default=str)
    print(f"\n  wrote {args.out}")
