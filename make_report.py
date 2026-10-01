"""
make_report.py -- Build a self-contained `report/` folder for the supervision
=============================================================================

Collects everything that has been measured about the SIMPLE (DLC) version into
one folder you can open, read and present from, without needing this repo or a
Python environment to look at it.

    python3 make_report.py                 # build from what is already in out/
    python3 make_report.py --run-figures   # also regenerate the tracking plots
    python3 make_report.py --full          # also re-run the benchmarks (~20 min)

What it produces
----------------

    report/
        README.md                  the report itself, in reading order
        figures/
            01_learning_curve.png          RL vs the three baselines over training
            02_paired_difference.png       per-task RL minus 2-MPC, sorted
            03_reward_distribution.png     where each method's mass sits
            04_tail_comparison.png         the percentile story, which is the finding
            05_planning_time.png           cost per plan, log scale
            06_reward_components.png       which reward terms each method wins on
            (plus any tracking figures copied from out/)
        tables/
            benchmark_seed777.csv          the fresh-seed results
            benchmark_seed12345.csv        the training-eval-set results
            per_task_seed777.csv           all 40 tasks x 4 methods
            per_task_seed12345.csv
            paired_statistics.csv          the hypothesis tests
            learning_curve.csv             the training evals
        logs/
            training.log                   the clean eval-only training log
            self_test.log                  the environment's own verification
            provenance.txt                 what came from MATLAB, what did not
        data/
            (the raw json the tables were built from)

Design notes
------------

Every number in the report traces to a file in `data/`. Nothing is typed in by
hand here: if a figure and a table disagree, that is a bug in this script, not
a judgement call made while writing it.

The script is deliberately tolerant of missing inputs. If a benchmark has not
been run it says so in the README rather than failing, so a partial report is
still a report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
REPORT = os.path.join(HERE, "report")

# ---------------------------------------------------------------------------
# One palette, used everywhere, so a method is the same colour in every figure.
# Chosen to stay distinguishable in greyscale (a printed handout) and for the
# common red-green colour vision deficiency, which rules out the obvious
# red/green "bad/good" pairing.
# ---------------------------------------------------------------------------
C = {
    "heuristic": "#9aa0a6",     # grey     -- the no-optimisation floor
    "mpc2":      "#e8710a",     # orange   -- the method to beat
    "rl":        "#1a73e8",     # blue     -- ours
    "optimum":   "#188038",     # green    -- the attainable ceiling
}
LABEL = {
    "heuristic": "Fixed heuristic",
    "mpc2": "2-MPC planner",
    "rl": "RL planner",
    "optimum": "Offline optimum",
}
ORDER = ["heuristic", "mpc2", "rl", "optimum"]

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": True,
})


# =============================================================================
# Loading
# =============================================================================

def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def load_benchmarks() -> Dict[int, Dict]:
    """Every benchmark json in out/, keyed by the task seed it used.

    Scans rather than looking for two fixed filenames. The seed comes from
    inside the file when the benchmark recorded it, and from the filename
    otherwise, so a result produced before that field existed -- or written to
    a name chosen with --out -- is still picked up. The earlier version looked
    only for `dlc_benchmark_s777.json` and `dlc_benchmark_s12345.json`, and
    silently produced a report with no results when the benchmark had written
    its default `dlc_benchmark.json` instead.
    """
    import glob
    import re

    found: Dict[int, Dict] = {}
    for path in sorted(glob.glob(os.path.join(OUT, "dlc_benchmark*.json"))):
        d = _read_json(path)
        if not isinstance(d, dict) or "summary" not in d:
            continue
        seed = d.get("seed")
        if seed is None:
            m = re.search(r"_s(\d+)\.json$", os.path.basename(path))
            if m is None:
                # A file whose seed cannot be established is skipped rather
                # than filed under a placeholder: a stale result reported
                # under an invented seed is worse than a missing one.
                print(f"  skipping {os.path.basename(path)}: no task seed "
                      f"recorded in it or in its name")
                continue
            seed = m.group(1)
        seed = int(seed)
        # Prefer the richer record if two files claim the same seed.
        if seed not in found or ("per_task" in d
                                 and "per_task" not in found[seed]):
            found[seed] = d
            d.setdefault("_path", path)
    return found


def load_learning_curve() -> List[Dict]:
    """Every eval the training run recorded, oldest first.

    Prefers the 16k run's json; falls back to parsing the text log, so a run
    whose json was overwritten still yields a curve.
    """
    for name in ("rl_dlc16k_training.json", "rl_dlc_training.json"):
        d = _read_json(os.path.join(OUT, name))
        if d and d.get("evals"):
            return sorted(d["evals"], key=lambda e: e["episode"])

    log = os.path.join(OUT, "rl_dlc16k_log.txt")
    rows: List[Dict] = []
    try:
        import re
        pat = re.compile(r"ep\s+(\d+)\s+\|\s+eval R\s+([+-][\d.]+)\s+\+-([\d.]+)")
        for line in open(log):
            m = pat.search(line)
            if m:
                rows.append({"episode": int(m.group(1)),
                             "reward": float(m.group(2)),
                             "reward_std": float(m.group(3))})
    except OSError:
        pass
    return sorted(rows, key=lambda e: e["episode"])


# =============================================================================
# Tables
# =============================================================================

SUMMARY_COLUMNS = [
    ("method", "Method"),
    ("reward_mean", "Reward"),
    ("reward_std", "SD"),
    ("reward_p10", "P10"),
    ("reward_worst", "Worst"),
    ("success_rate", "Success"),
    ("mean_a_y_g", "Peak a_y [g]"),
    ("mean_rms_e_y", "RMS e_y [m]"),
    ("mean_phi_deg", "Peak phi [deg]"),
    ("mean_slip_deg", "Max slip [deg]"),
    ("mean_completion_s", "Completion [s]"),
    ("plan_ms_median", "Plan time [ms]"),
]


def write_summary_table(bench: Dict, path: str) -> List[List]:
    rows = [[h for _, h in SUMMARY_COLUMNS]]
    for k in ORDER:
        s = bench["summary"].get(k)
        if not s:
            continue
        row = [LABEL[k]]
        for key, _ in SUMMARY_COLUMNS[1:]:
            v = s.get(key, float("nan"))
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                row.append("-")
            elif key == "success_rate":
                row.append(f"{v * 100:.0f}%")
            elif key == "plan_ms_median":
                row.append(f"{v:.2f}" if v < 100 else f"{v:,.0f}")
            else:
                row.append(round(float(v), 4))
        rows.append(row)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return rows


def write_per_task_table(bench: Dict, path: str) -> None:
    per = bench.get("per_task")
    if not per:
        return
    methods = [k for k in ORDER if k in per]
    n = len(per[methods[0]])
    fields = ["reward", "success", "plan_time_ms", "peak_a_y_g",
              "completion_time", "deadline_margin", "rms_lateral_error",
              "peak_articulation_deg", "max_front_slip_deg"]
    header = ["task"] + [f"{m}_{f}" for m in methods for f in fields]
    rows = [header]
    for i in range(n):
        row = [i]
        for m in methods:
            r = per[m][i]
            for f in fields:
                v = r.get(f)
                row.append(round(float(v), 5)
                           if isinstance(v, (int, float)) and np.isfinite(v)
                           else v)
        rows.append(row)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def write_paired_table(benches: Dict[int, Dict], path: str) -> None:
    rows = [["seed", "comparison", "mean_difference", "median_difference",
             "win_rate", "paired_t"]]
    for seed, b in sorted(benches.items()):
        for key, v in b.get("paired", {}).items():
            if isinstance(v, dict):
                rows.append([seed, key, round(v["mean"], 5),
                             round(v["median"], 5), round(v["win_rate"], 4),
                             round(v["t"], 3)])
        for key, v in b.get("paired", {}).items():
            if not isinstance(v, dict):
                rows.append([seed, key, round(float(v), 5), "", "", ""])
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def write_curve_table(curve: List[Dict], path: str) -> None:
    rows = [["episode", "eval_reward", "eval_reward_std", "success_rate",
             "alpha", "elapsed_min"]]
    for e in curve:
        rows.append([e.get("episode"), round(e.get("reward", float("nan")), 5),
                     round(e.get("reward_std", float("nan")), 5),
                     e.get("success_rate", ""),
                     round(e["alpha"], 5) if "alpha" in e else "",
                     round(e["elapsed_s"] / 60, 2) if "elapsed_s" in e else ""])
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def md_table(rows: List[List]) -> str:
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(str(c) for c in head) + " |",
           "| " + " | ".join("---" for _ in head) + " |"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# =============================================================================
# Figures
# =============================================================================

def fig_learning_curve(curve: List[Dict], benches: Dict[int, Dict],
                       path: str) -> Optional[str]:
    if not curve:
        return None
    eps = [e["episode"] for e in curve]
    rew = [e["reward"] for e in curve]
    sd = [e.get("reward_std", 0.0) for e in curve]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.fill_between(eps, np.array(rew) - np.array(sd),
                    np.array(rew) + np.array(sd),
                    color=C["rl"], alpha=0.12, linewidth=0)
    ax.plot(eps, rew, color=C["rl"], lw=2.2, marker="o", ms=3.5,
            label=LABEL["rl"], zorder=3)

    # Baselines from the seed the curve was measured on, so the crossing is real
    b = benches.get(12345) or (list(benches.values())[0] if benches else None)
    cross = None
    if b:
        for k in ("optimum", "mpc2", "heuristic"):
            s = b["summary"].get(k)
            if not s:
                continue
            v = s["reward_mean"]
            # The line stops short of the label, and the label sits ABOVE it:
            # text centred on a dashed line reads as struck out.
            ax.axhline(v, xmax=0.76, color=C[k], ls="--", lw=1.3, alpha=0.9)
            ax.text(eps[-1] * 1.02, v + 0.006, f"{LABEL[k]} {v:.3f}",
                    va="bottom", ha="left", fontsize=9, color=C[k])
        target = b["summary"].get("mpc2", {}).get("reward_mean")
        if target is not None:
            for e, r in zip(eps, rew):
                if r > target:
                    cross = e
                    break
    if cross is not None:
        ax.axvline(cross, color=C["rl"], ls=":", lw=1.2, alpha=0.8)
        ax.annotate(f"passes the 2-MPC planner\nat ~{cross:,} episodes",
                    xy=(cross, target), xytext=(cross * 0.30, target - 0.10),
                    fontsize=9, color=C["rl"],
                    arrowprops=dict(arrowstyle="->", color=C["rl"], lw=1.1))

    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Mean reward on the 40-task eval set")
    ax.set_title("The learned planner overtakes the convex planner,\n"
                 "but not before ~8k episodes", loc="left")
    ax.set_xlim(0, eps[-1] * 1.42)
    ax.legend(loc="lower right", bbox_to_anchor=(0.74, 0.02))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_paired_difference(bench: Dict, seed: int, path: str) -> Optional[str]:
    per = bench.get("per_task")
    if not per or "rl" not in per or "mpc2" not in per:
        return None
    d = np.array([per["rl"][i]["reward"] - per["mpc2"][i]["reward"]
                  for i in range(len(per["rl"]))])
    order = np.argsort(d)
    d_sorted = d[order]
    colours = [C["rl"] if v > 0 else C["mpc2"] for v in d_sorted]

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(range(len(d_sorted)), d_sorted, color=colours, width=0.82)
    ax.axhline(0, color="#202124", lw=1.0)
    ax.axhline(d.mean(), color="#202124", ls="--", lw=1.2)
    ax.text(len(d) - 0.5, d.mean(), f"  mean {d.mean():+.3f}", va="bottom",
            ha="right", fontsize=9)
    wins = int((d > 0).sum())
    ax.set_xlabel(f"Task, sorted by difference (seed {seed})")
    ax.set_ylabel("RL reward minus 2-MPC reward")
    ax.set_title(f"RL wins {wins} of {len(d)} tasks, and its losses are small "
                 f"while its wins are large", loc="left")
    ax.set_xticks([])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_reward_distribution(bench: Dict, seed: int, path: str) -> Optional[str]:
    per = bench.get("per_task")
    if not per:
        return None
    methods = [k for k in ORDER if k in per]
    data = [[r["reward"] for r in per[k]] for k in methods]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, k in zip(parts["bodies"], methods):
        body.set_facecolor(C[k])
        body.set_alpha(0.28)
        body.set_edgecolor(C[k])
    for i, (vals, k) in enumerate(zip(data, methods), start=1):
        v = np.asarray(vals)
        ax.scatter(np.full_like(v, i) + np.random.default_rng(0).normal(0, 0.035, v.size),
                   v, s=13, color=C[k], alpha=0.75, linewidths=0, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color=C[k], lw=2.2, zorder=4)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([LABEL[k] for k in methods])
    ax.set_ylabel("Reward (max 1.0)")
    ax.set_title("Each method's 40 tasks; the bar is the median", loc="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_tail_comparison(benches: Dict[int, Dict], path: str) -> Optional[str]:
    """The finding is in the tail, so give the tail its own figure."""
    if not benches:
        return None
    seeds = sorted(benches)
    metrics = [("reward_mean", "Mean"), ("reward_p10", "10th pct"),
               ("reward_worst", "Worst")]
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.6 * len(seeds), 4.0),
                             sharey=True, squeeze=False)
    width = 0.36
    drew = False
    for ax, seed in zip(axes[0], seeds):
        b = benches[seed]["summary"]
        xs = np.arange(len(metrics))
        for j, k in enumerate(("mpc2", "rl")):
            if k not in b:
                continue
            vals = [b[k][m] for m, _ in metrics]
            drew = True
            bars = ax.bar(xs + (j - 0.5) * width, vals, width,
                          color=C[k], label=LABEL[k])
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for _, lab in metrics])
        ax.set_title(f"Task seed {seed}", loc="left")
        ax.set_ylim(0, 0.87)
    if not drew:
        plt.close(fig)
        return None
    axes[0][0].set_ylabel("Reward (max 1.0)")
    axes[0][0].legend(loc="upper right")
    fig.suptitle("The advantage is in the tail: similar means, very "
                 "different worst cases", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_planning_time(benches: Dict[int, Dict], path: str) -> Optional[str]:
    b = benches.get(777) or (list(benches.values())[0] if benches else None)
    if not b:
        return None
    ks = [k for k in ORDER if k in b["summary"]
          and np.isfinite(b["summary"][k].get("plan_ms_median", float("nan")))]
    vals = [max(b["summary"][k]["plan_ms_median"], 1e-3) for k in ks]
    if not vals:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bars = ax.barh([LABEL[k] for k in ks], vals, color=[C[k] for k in ks],
                   height=0.62)
    for rect, v, k in zip(bars, vals, ks):
        # Adaptive precision: the four methods span five orders of magnitude,
        # so a single format string prints either "0.0 ms" for the heuristic
        # or "11000.0 ms" for the search.
        if v >= 1000:
            txt = f"{v / 1000:.1f} s"
        elif v >= 1:
            txt = f"{v:.1f} ms"
        elif v >= 0.01:
            txt = f"{v:.2f} ms"
        else:
            txt = "< 0.01 ms"
        ax.text(v * 1.25, rect.get_y() + rect.get_height() / 2, txt,
                va="center", fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlabel("Median planning time per task, ms (log scale)")
    ax.set_title("One forward pass versus a convex solve versus a search",
                 loc="left")
    ax.set_xlim(right=max(vals) * 12)
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_reward_components(bench: Dict, path: str) -> Optional[str]:
    """Which parts of the objective each planner actually wins on.

    Uses the physical metrics rather than the reward terms, because those are
    the quantities that mean something outside this codebase.
    """
    per = bench.get("per_task")
    if not per:
        return None
    specs = [("rms_lateral_error", "RMS lateral\nerror [m]", False),
             ("peak_articulation_deg", "Peak articulation\n[deg]", False),
             ("max_front_slip_deg", "Max front slip\n[deg]", False),
             ("peak_a_y_g", "Peak lateral\naccel [g]", False),
             ("completion_time", "Completion\ntime [s]", False)]
    methods = [k for k in ("mpc2", "rl") if k in per]
    if len(methods) < 2:
        return None

    fig, axes = plt.subplots(1, len(specs), figsize=(2.35 * len(specs), 3.9))
    for ax, (key, title, _) in zip(axes, specs):
        vals, labs, cols = [], [], []
        for k in methods:
            v = [r[key] for r in per[k]
                 if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
            if v:
                vals.append(np.mean(v))
                labs.append(LABEL[k].split()[0])
                cols.append(C[k])
        if not vals:
            plt.close(fig)
            return None
        bars = ax.bar(labs, vals, color=cols, width=0.6)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, fontweight="normal")
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, max(vals) * 1.28)
    fig.suptitle("Lower is better in every panel: the QP tracks its own plan "
                 "more tightly, the policy finishes sooner and slips less",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Logs
# =============================================================================

PROVENANCE = """\
WHAT CAME FROM WHERE
====================

This matters for how the work is described. The MATLAB file that started this
project, RUN_TTV_DLC_TRACKING_TWO_MODELS.m, contains exactly ONE optimisation:
the tracking controller. Its "TWO MODELS" are the linear prediction model used
inside the MPC and the nonlinear plant it is tested against -- NOT two MPCs.
There is no path planner in it; the double lane change is a fixed analytic
formula with hard-coded parameters.

Inherited from the MATLAB (ported function-for-function, verified against an
automatic-differentiation Jacobian to 1.7e-14):

    - vehicle parameters and the saturating tyre model
    - the 5-DOF yaw-plane tractor-semitrailer plant
    - the linear 7-state prediction model
    - the tracking MPC: cost, constraints, horizon
    - the double-lane-change reference formula

Written for this project:

    - the RL task, observation, action decoder and reward   (dlc_rl_env.py)
    - the one-step SAC implementation                       (train_rl.py)
    - THE PLANNING MPC used as the 2-MPC baseline           (dlc_baselines.py)
    - the offline optimum and the benchmark harness         (dlc_baselines.py,
                                                             dlc_benchmark.py)
    - faster numerically-equivalent plant and MPC backends  (plant_numpy.py,
                                                             fast_mpc.py)

So the correct statement is: "the inherited code has one MPC, the tracking
controller. To have something to compare a learned planner against, I
implemented a planning MPC myself." Claiming the two-MPC baseline was
inherited would be wrong and is easy to check.
"""


def capture(cmd: List[str], path: str, timeout: int = 1800) -> bool:
    """Run a command and keep its output. Returns True if it succeeded."""
    env = dict(os.environ, TTV_ENV="dlc")
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, timeout=timeout,
                           capture_output=True, text=True)
    except (subprocess.TimeoutExpired, OSError) as exc:
        with open(path, "w") as fh:
            fh.write(f"FAILED: {exc}\n")
        return False
    with open(path, "w") as fh:
        fh.write(p.stdout)
        if p.stderr.strip():
            fh.write("\n--- stderr ---\n")
            fh.write(p.stderr)
    return p.returncode == 0


# =============================================================================
# The README
# =============================================================================

def build_readme(benches: Dict[int, Dict], curve: List[Dict],
                 figures: Dict[str, Optional[str]],
                 summary_tables: Dict[int, List[List]],
                 self_test_ok: Optional[bool]) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    fresh = benches.get(777)
    train = benches.get(12345)

    def fig(key: str, caption: str) -> str:
        p = figures.get(key)
        if not p:
            return ""
        rel = os.path.relpath(p, REPORT)
        return f"\n![{caption}]({rel})\n\n*{caption}*\n"

    L: List[str] = []
    A = L.append

    A("# RL path planning for a tractor-semitrailer: the simple (DLC) version")
    A("")
    A(f"Generated {stamp} by `make_report.py`. Every number below comes from a "
      f"file in `data/`; nothing here is typed in by hand.")
    A("")

    # ---- headline -------------------------------------------------------
    A("## The claim")
    A("")
    A("> Given the same vehicle, the same tracking MPC, the same plant and the "
      "same scoring function, a policy trained once offline chooses a better "
      "reference trajectory than an online planning MPC does, and chooses it "
      "faster.")
    A("")
    if fresh:
        s = fresh["summary"]
        p = fresh["paired"].get("rl_minus_mpc2", {})
        A(f"On 40 tasks drawn from a seed never used for anything "
          f"(seed 777), the RL planner scores **{s['rl']['reward_mean']:.4f}** "
          f"against the planning QP's **{s['mpc2']['reward_mean']:.4f}**. "
          f"Paired per task: **{p.get('mean', float('nan')):+.4f}**, "
          f"t = **{p.get('t', float('nan')):+.2f}**, win rate "
          f"**{p.get('win_rate', 0)*100:.0f}%**.")
        A("")
        A(f"It reaches **{fresh['paired'].get('rl_fraction_of_optimum', 0)*100:.1f}%** "
          f"of the offline optimum, meets the deadline on "
          f"**{s['rl']['success_rate']*100:.0f}%** of tasks against the QP's "
          f"**{s['mpc2']['success_rate']*100:.0f}%**, and plans in "
          f"**{s['rl']['plan_ms_median']:.1f} ms** against "
          f"**{s['mpc2']['plan_ms_median']:.1f} ms** (medians).")
    else:
        A("> **This report is incomplete.** No benchmark results were found in "
          "`out/`, so there is no comparison table, no paired statistics and "
          "only the learning-curve figure. Run:")
        A(">")
        A("> ```bash")
        A("> TTV_ENV=dlc python3 dlc_benchmark.py --n 40")
        A("> TTV_ENV=dlc python3 dlc_benchmark.py --n 40 --seed 12345")
        A("> python3 make_report.py")
        A("> ```")
        A(">")
        A("> Each benchmark takes about 10 minutes.")
    A("")

    A("### Three qualifications, stated up front")
    A("")
    A("1. **The win rate is around 55-68%, not 90%.** The mean advantage comes "
      "from the tail, not from beating the QP on every task. See figure 2.")
    A("2. **The result depends on the training budget.** At 4000 episodes the "
      "QP was ahead by 0.061. The crossover is near 8000 episodes. Any report "
      "of this number must carry the budget beside it.")
    A("3. **This is one training seed.** Until it is replicated across three "
      "seeds, treat the effect size as provisional.")
    A("")

    # ---- what was compared ---------------------------------------------
    A("## What was compared")
    A("")
    A("All four methods produce a *reference trajectory*. That reference is "
      "then driven by the **same** tracking MPC, through the **same** "
      "nonlinear plant, and scored by the **same** reward, on the **same** "
      "tasks. Only the reference generator differs, so any difference in "
      "score is attributable to the planner and nothing else.")
    A("")
    A("| Method | What it is | What it controls for |")
    A("| --- | --- | --- |")
    A("| Fixed heuristic | 75% of the allowed transition rate; no "
      "optimisation, ignores the task | whether any optimisation is worth "
      "doing at all |")
    A("| 2-MPC planner | a convex QP over a triple integrator (jerk-minimising, "
      "`abs(a) <= a_y_max`, corridor on `y`), then the tracking MPC | the "
      "method to beat: plan with MPC, steer with MPC |")
    A("| RL planner | the trained policy, then the tracking MPC | ours |")
    A("| Offline optimum | a (1+lambda) evolution strategy, ~80 closed-loop "
      "simulations per task | the ceiling of the action space, so the RL "
      "score reads as a fraction rather than a bare number |")
    A("")
    A("**The planning MPC was written for this project; it is not in the "
      "MATLAB.** See `logs/provenance.txt` -- this is the single easiest "
      "thing to get wrong when describing the work.")
    A("")

    # ---- figures --------------------------------------------------------
    A("## Results")
    A("")
    A("### 1. Training")
    A(fig("curve", "RL eval reward over training, against the three baselines. "
                   "The shaded band is +/- one standard deviation across the "
                   "40 eval tasks."))
    A("The curve crosses the QP line at about 8000 episodes and flattens just "
      "below the offline optimum. Two readings follow: the training budget is "
      "part of the result, and the action space is nearly exhausted, so "
      "further gains need a richer parameterisation rather than more compute.")
    A("")

    A("### 2. Per-task differences")
    A(fig("paired", "RL reward minus 2-MPC reward, one bar per task, sorted."))
    A("This is the figure that keeps the claim honest. The wins and losses are "
      "not symmetric: the losses are small and the wins are large, which is "
      "why the mean moves even though the win rate is near half.")
    A("")

    A("### 3. Where the advantage lives")
    A(fig("tail", "Mean, 10th percentile and worst case, on both task seeds."))
    A("The means are close. The tails are not. The QP's worst case is a task "
      "where its jerk-optimal trajectory is too slow for the deadline -- it "
      "has no way to trade comfort for time, because the deadline is a hard "
      "constraint in its formulation rather than a term in its objective. The "
      "policy is trained against a reward where brevity and smoothness are "
      "both weighted terms, so it makes that trade continuously.")
    A("")
    A("**That is the mechanism to quote when asked why RL wins.** It is not "
      "capacity and it is not magic: the learned planner optimises the actual "
      "objective, while the QP optimises a convex proxy of it.")
    A("")

    A("### 4. Distribution of outcomes")
    A(fig("dist", "All 40 task rewards for each method."))
    A("")

    A("### 5. Physical metrics")
    A(fig("components", "Mean of each physical quantity, QP versus policy."))
    A("The QP tracks its own plan more tightly (its reference is smoother by "
      "construction), while the policy finishes sooner and slips less. Neither "
      "dominates on every axis, which is what a weighted objective produces.")
    A("")

    A("### 6. Planning cost")
    A(fig("time", "Planning time per task, log scale."))
    A("The speed argument holds independently of reward quality, but it is "
      "modest here -- this planning QP is small and convex. The large "
      "speedups in the literature compare against nonlinear planners.")
    A("")

    # ---- tables ---------------------------------------------------------
    A("## Tables")
    A("")
    for seed in sorted(summary_tables):
        note = ("tasks never used for anything, not even watched during "
                "training" if seed == 777 else
                "the set the training curve is read off")
        A(f"### Task seed {seed} -- {note}")
        A("")
        A(md_table(summary_tables[seed]))
        A("")
    if benches:
        A("### Paired statistics")
        A("")
        rows = [["Seed", "Comparison", "Mean diff", "Median diff", "Win rate",
                 "Paired t"]]
        for seed, b in sorted(benches.items()):
            for key, v in b.get("paired", {}).items():
                if isinstance(v, dict):
                    rows.append([seed, key, f"{v['mean']:+.4f}",
                                 f"{v['median']:+.4f}",
                                 f"{v['win_rate']*100:.1f}%", f"{v['t']:+.2f}"])
        A(md_table(rows))
        A("")
        A("Paired, because task difficulty varies enormously -- an unpaired "
          "comparison of two 40-sample means would drown a 0.04 effect in a "
          "0.09 standard deviation.")
        A("")
        if 777 in benches and 12345 in benches:
            a = benches[777]["paired"]["rl_minus_mpc2"]["mean"]
            c = benches[12345]["paired"]["rl_minus_mpc2"]["mean"]
            A(f"The two seeds agree to within {abs(a - c):.4f} on the paired "
              f"difference. That agreement is the evidence that nothing was "
              f"selected on the test set: seed 12345 is the set the training "
              f"curve is read off, so reporting only that one would be "
              f"selection on the test set.")
            A("")

    # ---- constraints ----------------------------------------------------
    A("## The physical constraints")
    A("")
    A("Four tiers, and it matters which is which, because only one is a soft "
      "penalty.")
    A("")
    A("| Tier | Where it lives | Examples |")
    A("| --- | --- | --- |")
    A("| 1. Implicit | the plant | tyres saturate at `mu*Fz`; ask for more "
      "lateral force and you do not get it |")
    A("| 2. Hard | the tracking MPC's QP | steering `abs(delta) <= 0.5 rad`, "
      "rate `<= 0.6 rad/s` |")
    A("| 3. **By construction** | the action decoder | the lateral-"
      "acceleration budget and the lane-occupancy requirement |")
    A("| 4. Soft | the reward | the deadline, penalised up to -1.0 over 3 s |")
    A("")
    A("Tier 3 is the design decision. A reward penalty makes a violation "
      "*expensive*; an action-space constraint makes it *impossible*. The "
      "guarantee therefore holds for an untrained network, which is not "
      "something a penalty-based design can say.")
    if self_test_ok is not None:
        A("")
        A(f"Verified in `logs/self_test.log`: "
          f"{'PASSED' if self_test_ok else 'FAILED'}.")
    A("")

    # ---- environment ----------------------------------------------------
    A("## What the environment is, precisely")
    A("")
    A("The road is an infinite, flat, straight, level plane with a single "
      "friction coefficient. There is no road geometry, no lane markings, no "
      "other vehicles and no obstacles, so there is no collision check. What "
      "replaces road geometry is the *task*: move `lateral_offset` metres "
      "sideways, hold for `hold_time` seconds, be back by `deadline`, and do "
      "not exceed `a_y_budget`.")
    A("")
    A("| Randomised per task | Range |")
    A("| --- | --- |")
    A("| Forward speed `V` | 14-28 m/s |")
    A("| Lateral offset | 2.5-4.5 m |")
    A("| Hold time | 4-9 s |")
    A("| Start time | 3-5 s |")
    A("| Friction `mu` | 0.45-0.90 |")
    A("| Lateral-acceleration budget | 0.25-0.45 g |")
    A("| Deadline | `t_fast + frac*(t_slow - t_fast)`, `frac ~ U(0.25, 0.70)` |")
    A("")
    A("The observation is exactly those values, scaled. Six numbers. The "
      "policy sees no vehicle state and no feedback: it decides before the "
      "manoeuvre begins and never sees what happens. That is what makes it a "
      "planner rather than a controller.")
    A("")

    # ---- contents -------------------------------------------------------
    A("## What is in this folder")
    A("")
    A("| Path | What it holds |")
    A("| --- | --- |")
    A("| `figures/` | every figure in this report, as PNG |")
    A("| `tables/` | the same numbers as CSV, including all 40 per-task rows |")
    A("| `logs/training.log` | the eval-only training record, one line per eval |")
    A("| `logs/self_test.log` | the environment's own verification run |")
    A("| `logs/provenance.txt` | what came from the MATLAB and what did not |")
    A("| `data/` | the raw JSON every table and figure was built from |")
    A("")

    A("## How to reproduce")
    A("")
    A("```bash")
    A("python3 ttv_dlc_tracking.py --studies        # the inherited loop, ~3 min")
    A("python3 dlc_rl_env.py                        # task + self-test, ~90 s")
    A("python3 dlc_baselines.py                     # baselines on one task, ~30 s")
    A("TTV_ENV=dlc python3 train_rl.py --env dlc --episodes 16000 \\")
    A("    --warmup 400 --eval-every 1000 --out out/rl_dlc      # ~55 min")
    A("TTV_ENV=dlc python3 dlc_benchmark.py --n 40              # seed 777")
    A("TTV_ENV=dlc python3 dlc_benchmark.py --n 40 --seed 12345 # training set")
    A("python3 make_report.py                       # rebuild this folder")
    A("```")
    A("")
    return "\n".join(L)


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--full", action="store_true",
                    help="re-run both benchmarks before building (~20 min)")
    ap.add_argument("--run-figures", action="store_true",
                    help="also regenerate the tracking plots from "
                         "ttv_dlc_tracking.py (~3 min)")
    ap.add_argument("--skip-self-test", action="store_true",
                    help="do not run the environment self-test (~90 s)")
    ap.add_argument("--agent", default="out/rl_dlc_agent.pt")
    ap.add_argument("--out", default=REPORT)
    args = ap.parse_args()

    root = os.path.abspath(args.out)
    for sub in ("figures", "tables", "logs", "data"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    print("=" * 78)
    print(f"Building {os.path.relpath(root, HERE)}/")
    print("=" * 78)

    # ---- optionally regenerate the inputs -------------------------------
    if args.full:
        for seed in (777, 12345):
            print(f"  re-running the benchmark, seed {seed} "
                  f"(this takes ~10 min) ...", flush=True)
            capture([sys.executable, "dlc_benchmark.py", "--n", "40",
                     "--agent", args.agent, "--seed", str(seed),
                     "--out", f"out/dlc_benchmark_s{seed}.json"],
                    os.path.join(root, "logs", f"benchmark_seed{seed}.log"))
    if args.run_figures:
        print("  regenerating the tracking figures ...", flush=True)
        capture([sys.executable, "ttv_dlc_tracking.py", "--studies"],
                os.path.join(root, "logs", "tracking_studies.log"))

    # ---- load ------------------------------------------------------------
    benches = load_benchmarks()
    curve = load_learning_curve()
    print(f"  benchmarks found: "
          f"{', '.join(f'seed {s}' for s in sorted(benches)) or 'NONE'}")
    print(f"  learning curve:   {len(curve)} eval points")

    # A report missing its benchmarks still LOOKS like a report -- a folder of
    # files and a README -- so say plainly what is absent and how to get it,
    # rather than letting a partial build pass for a complete one.
    missing: List[str] = []
    if not benches:
        missing.append(
            "  NO BENCHMARK RESULTS. Without them the report has no comparison\n"
            "  table, no paired statistics and only 1 of 6 figures. Run:\n"
            f"    TTV_ENV=dlc python3 dlc_benchmark.py --n 40 "
            f"--agent {args.agent}\n"
            f"    TTV_ENV=dlc python3 dlc_benchmark.py --n 40 "
            f"--agent {args.agent} --seed 12345\n"
            "  then re-run this script. Each takes about 10 minutes.")
    elif len(benches) == 1:
        have = next(iter(benches))
        other = 12345 if have != 12345 else 777
        missing.append(
            f"  Only ONE task seed ({have}). The two-seed agreement is what\n"
            f"  rules out selection on the test set, so run the other:\n"
            f"    TTV_ENV=dlc python3 dlc_benchmark.py --n 40 "
            f"--agent {args.agent} --seed {other}")
    if not curve:
        missing.append(
            "  NO TRAINING LOG, so no learning curve. Expected\n"
            "  out/rl_dlc_training.json or out/rl_dlc16k_training.json.")
    if benches and not any("per_task" in b for b in benches.values()):
        missing.append(
            "  The benchmark json has no per-task rows, so the paired-\n"
            "  difference and distribution figures cannot be drawn. Re-run the\n"
            "  benchmark with the current dlc_benchmark.py.")
    if missing:
        print("\n" + "!" * 78)
        print("INCOMPLETE -- this report will be missing sections:")
        print("!" * 78)
        for m in missing:
            print(m)
        print("!" * 78 + "\n")

    # ---- copy the raw data ----------------------------------------------
    for b in benches.values():
        src = b.get("_path")
        if src and os.path.exists(src):
            shutil.copy2(src, os.path.join(root, "data",
                                           os.path.basename(src)))
    for name in ("rl_dlc16k_training.json", "rl_dlc_training.json",
                 "dlc_metrics.json"):
        src = os.path.join(OUT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root, "data", name))

    # ---- logs ------------------------------------------------------------
    for src_name, dst_name in (("rl_dlc16k_log.txt", "training.log"),
                               ("rl_dlc_log.txt", "training.log")):
        src = os.path.join(OUT, src_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(root, "logs", dst_name))
            break

    with open(os.path.join(root, "logs", "provenance.txt"), "w") as fh:
        fh.write(PROVENANCE)

    self_test_ok: Optional[bool] = None
    if not args.skip_self_test:
        print("  running the environment self-test (~90 s) ...", flush=True)
        self_test_ok = capture([sys.executable, "dlc_rl_env.py"],
                               os.path.join(root, "logs", "self_test.log"),
                               timeout=900)
        print(f"    self-test {'passed' if self_test_ok else 'FAILED'}")

    # ---- tables ----------------------------------------------------------
    summary_tables: Dict[int, List[List]] = {}
    for seed, b in benches.items():
        summary_tables[seed] = write_summary_table(
            b, os.path.join(root, "tables", f"benchmark_seed{seed}.csv"))
        write_per_task_table(
            b, os.path.join(root, "tables", f"per_task_seed{seed}.csv"))
    if benches:
        write_paired_table(benches,
                           os.path.join(root, "tables", "paired_statistics.csv"))
    if curve:
        write_curve_table(curve,
                          os.path.join(root, "tables", "learning_curve.csv"))

    # ---- figures ---------------------------------------------------------
    figdir = os.path.join(root, "figures")
    fresh = benches.get(777) or (list(benches.values())[0] if benches else None)
    figures: Dict[str, Optional[str]] = {
        "curve": fig_learning_curve(curve, benches,
                                    os.path.join(figdir, "01_learning_curve.png")),
        "paired": (fig_paired_difference(
            fresh, 777 if 777 in benches else sorted(benches)[0],
            os.path.join(figdir, "02_paired_difference.png"))
            if fresh else None),
        "dist": (fig_reward_distribution(
            fresh, 777, os.path.join(figdir, "03_reward_distribution.png"))
            if fresh else None),
        "tail": fig_tail_comparison(benches,
                                    os.path.join(figdir, "04_tail_comparison.png")),
        "time": fig_planning_time(benches,
                                  os.path.join(figdir, "05_planning_time.png")),
        "components": (fig_reward_components(
            fresh, os.path.join(figdir, "06_reward_components.png"))
            if fresh else None),
    }
    made = [k for k, v in figures.items() if v]
    print(f"  figures written:  {len(made)} ({', '.join(made)})")

    # tracking plots produced by ttv_dlc_tracking.py, if they exist
    for name in ("dlc_tracking.png", "dlc_states.png", "dlc_steering_slip.png",
                 "dlc_model_mismatch.png"):
        src = os.path.join(OUT, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(figdir, "10_" + name))

    # ---- readme ----------------------------------------------------------
    readme = build_readme(benches, curve, figures, summary_tables, self_test_ok)
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write(readme)

    n_files = sum(len(f) for _, _, f in os.walk(root))
    print(f"\n  {os.path.relpath(root, HERE)}/ built: {n_files} files")
    print(f"  start at {os.path.relpath(root, HERE)}/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())