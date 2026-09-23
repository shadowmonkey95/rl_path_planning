"""
benchmark.py -- RL planner vs the 2-MPC planner vs a heuristic
==============================================================

All four methods produce a PATH. Each path is then driven by the same tracking
MPC through the same nonlinear 40 t plant, and scored by the same reward and
the same truck metrics. Only the planner differs.

Run on the held-out scenario set that training never saw.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import numpy as np
import torch

from highway_env import (TruckHighwayEnv, Scenario, RewardWeights,
                         action_to_path, rollout, compute_reward)
from fast_mpc import CondensedMPC
from baselines import MPCPlanner, run_mpc_planner, heuristic_action, offline_optimum
from train_rl import load_agent, make_eval_set, RunningNorm
import truck_params as TP

W = RewardWeights()


def _agg(rows: List[Dict]) -> Dict[str, float]:
    def col(k, default=np.nan):
        v = [r.get(k, default) for r in rows]
        v = [x for x in v if isinstance(x, (int, float)) and np.isfinite(x)]
        return np.asarray(v) if v else np.array([np.nan])
    n = len(rows)
    return {
        "n": n,
        "reward_mean": float(np.mean(col("reward"))),
        "reward_std": float(np.std(col("reward"))),
        "reward_p10": float(np.percentile(col("reward"), 10)),
        "reward_worst": float(np.min(col("reward"))),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collision"] for r in rows])),
        "rollover_rate": float(np.mean([r["rollover"] for r in rows])),
        "mean_peak_LTR": float(np.mean(col("peak_LTR"))),
        "worst_peak_LTR": float(np.max(col("peak_LTR"))),
        "mean_RWA": float(np.mean(col("RWA"))),
        "worst_RWA": float(np.max(col("RWA"))),
        "mean_swept": float(np.mean(col("swept_width"))),
        "mean_jerk": float(np.mean(col("peak_lateral_jerk"))),
        "mean_offtrack": float(np.mean(col("dynamic_offtracking"))),
        "mean_lane_departure": float(np.mean(col("lane_departure"))),
        "worst_lane_departure": float(np.max(col("lane_departure"))),
        "mean_length": float(np.mean(col("manoeuvre_length"))),
        "plan_time_ms_mean": float(np.mean(col("plan_time_ms"))),
        "plan_time_ms_p95": float(np.percentile(col("plan_time_ms"), 95)),
    }


def _row(reward, info, plan_ms) -> Dict:
    m = info.get("metrics", {}) or {}
    return {"reward": reward,
            "success": info["parts"].get("cause") == "success",
            "collision": bool(m.get("collision", False)),
            "rollover": bool(m.get("rollover", False)),
            "plan_time_ms": plan_ms,
            **{k: m.get(k, np.nan) for k in
               ("peak_LTR", "RWA", "swept_width", "peak_lateral_jerk",
                "dynamic_offtracking", "lane_departure", "manoeuvre_length",
                "peak_articulation_deg", "rms_lateral_error",
                "max_slip_front_deg", "peak_ay_trailer_g")}}


def run_benchmark(n_scen: int = 60, agent_path: str = "out/rl_agent.pt",
                  with_optimum: bool = True, optimum_budget: int = 96,
                  optimum_subset: int = 20) -> Dict:
    scens = make_eval_set(n_scen)
    env = TruckHighwayEnv(randomise=False)
    planner = MPCPlanner(ds=2.0, horizon_m=400.0)
    actor, norm = load_agent(agent_path)

    results: Dict[str, List[Dict]] = {k: [] for k in
                                      ("rl", "mpc2", "heuristic", "optimum")}
    mpc_cache: Dict[tuple, CondensedMPC] = {}

    def mpc_for(cfg):
        key = (round(cfg.V, 4), round(cfg.m2, 2), round(cfg.mu, 4))
        if key not in mpc_cache:
            mpc_cache[key] = CondensedMPC(cfg, use_rate_constraint=True)
        return mpc_cache[key]

    for i, sc in enumerate(scens):
        cfg = sc.config()
        tmpc = mpc_for(cfg)

        # ---- RL ------------------------------------------------------------
        obs = env.reset(sc)
        t0 = time.perf_counter()
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(norm(obs), dtype=torch.float32
                                         ).unsqueeze(0),
                         deterministic=True, with_logp=False)
        a = a.squeeze(0).numpy()
        t_rl = (time.perf_counter() - t0) * 1e3
        _, r, _, info = env.step(a)
        results["rl"].append(_row(r, info, t_rl))

        # ---- 2-MPC ---------------------------------------------------------
        r2, info2 = run_mpc_planner(sc, planner, W, mpc=tmpc)
        results["mpc2"].append(_row(r2, info2, info2["plan_time"] * 1e3))

        # ---- heuristic -----------------------------------------------------
        t0 = time.perf_counter()
        ah = heuristic_action(sc)
        t_h = (time.perf_counter() - t0) * 1e3
        env.reset(sc)
        _, rh, _, infoh = env.step(ah)
        results["heuristic"].append(_row(rh, infoh, t_h))

        # ---- offline optimum (subset only: it is ~100x the cost) -----------
        if with_optimum and i < optimum_subset:
            t0 = time.perf_counter()
            ro, ao = offline_optimum(sc, budget=optimum_budget, seed=i,
                                     w=W, mpc=tmpc)
            t_o = (time.perf_counter() - t0) * 1e3
            env.reset(sc)
            _, _, _, infoo = env.step(ao)
            results["optimum"].append(_row(ro, infoo, t_o))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_scen} scenarios", flush=True)

    summary = {k: _agg(v) for k, v in results.items() if v}

    # Paired comparison on the common subset
    n_opt = len(results["optimum"])
    paired = {}
    for k in ("mpc2", "heuristic"):
        d = np.array([results["rl"][i]["reward"] - results[k][i]["reward"]
                      for i in range(n_scen)])
        paired[f"rl_minus_{k}"] = {
            "mean": float(d.mean()), "std": float(d.std()),
            "win_rate": float(np.mean(d > 0)),
            "median": float(np.median(d)),
            # paired t-like statistic
            "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
            if d.std(ddof=1) > 0 else float("nan")}
    if n_opt:
        for k in ("rl", "mpc2", "heuristic"):
            num = np.array([results[k][i]["reward"] for i in range(n_opt)])
            den = np.array([results["optimum"][i]["reward"] for i in range(n_opt)])
            paired[f"{k}_fraction_of_optimum"] = float(np.mean(num) / np.mean(den))

    return {"summary": summary, "paired": paired, "per_scenario": results,
            "n_scenarios": n_scen, "n_optimum": n_opt}


# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--agent", type=str, default="out/rl_agent.pt")
    ap.add_argument("--optimum-subset", type=int, default=20)
    ap.add_argument("--out", type=str, default="out/benchmark.json")
    args = ap.parse_args()

    print("=" * 104)
    print(f"Benchmark on {args.n} held-out scenarios "
          f"(training never saw these)")
    print("=" * 104)
    res = run_benchmark(args.n, args.agent, optimum_subset=args.optimum_subset)

    order = ["heuristic", "mpc2", "rl", "optimum"]
    label = {"heuristic": "fixed heuristic", "mpc2": "2-MPC planner",
             "rl": "RL planner (ours)", "optimum": "offline optimum*"}
    print(f"\n  {'method':>20s} {'n':>3s} {'reward':>8s} {'sd':>6s} {'p10':>7s} "
          f"{'worst':>7s} {'succ':>6s} {'coll':>5s} {'roll':>5s} "
          f"{'LTR':>6s} {'wLTR':>6s} {'RWA':>6s} {'swept':>6s} {'jerk':>6s} "
          f"{'offtr':>6s} {'dep':>6s} {'plan ms':>8s}")
    for k in order:
        if k not in res["summary"]:
            continue
        s = res["summary"][k]
        print(f"  {label[k]:>20s} {s['n']:3d} {s['reward_mean']:8.4f} "
              f"{s['reward_std']:6.3f} {s['reward_p10']:7.3f} "
              f"{s['reward_worst']:7.3f} {s['success_rate']*100:5.1f}% "
              f"{s['collision_rate']*100:4.0f}% {s['rollover_rate']*100:4.0f}% "
              f"{s['mean_peak_LTR']:6.3f} {s['worst_peak_LTR']:6.3f} "
              f"{s['mean_RWA']:6.3f} {s['mean_swept']:6.3f} "
              f"{s['mean_jerk']:6.2f} {s['mean_offtrack']:6.3f} "
              f"{s['mean_lane_departure']:6.3f} {s['plan_time_ms_mean']:8.2f}")
    print("  * offline optimum is not deployable: it needs ~100 closed-loop")
    print("    simulations per scenario. It bounds what the action space can do.")

    print("\n  paired differences (same scenarios, RL minus baseline):")
    for k, v in res["paired"].items():
        if isinstance(v, dict):
            print(f"    {k:26s} mean {v['mean']:+.4f}  median {v['median']:+.4f}  "
                  f"win rate {v['win_rate']*100:5.1f}%  t = {v['t']:+.2f}")
        else:
            print(f"    {k:26s} {v:.4f}")

    json.dump({"summary": res["summary"], "paired": res["paired"],
               "n_scenarios": res["n_scenarios"], "n_optimum": res["n_optimum"]},
              open(args.out, "w"), indent=2, default=str)
    np.save("out/benchmark_per_scenario.npy",
            np.array(res["per_scenario"], dtype=object), allow_pickle=True)
    print(f"\n  wrote {args.out}")
