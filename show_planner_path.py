"""
show_planner_path.py -- Look at the 2-MPC planner's trajectory on its own
=========================================================================

Runs ONLY `DLCTrajectoryPlanner` (the planning QP in `dlc_baselines.py`), with
no RL, no training and no benchmark, and plots what it produced.

    python3 show_planner_path.py                    # the default task
    python3 show_planner_path.py --compare          # overlay the tanh reference
    python3 show_planner_path.py --track            # also drive it through the
                                                    # tracking MPC and the plant
    python3 show_planner_path.py --V 25 --offset 4.0 --deadline 13

Prints the solver status and the trajectory's own numbers, writes
`out/planner_path.png`, and -- with `--track` -- reports what the vehicle
actually did when the tracking MPC followed that plan.

Why this exists
---------------

The planner is a point-mass QP: states [y, y_dot, a], control jerk, minimise
`sum(j^2 + w_a*a^2)` subject to `|a| <= a_y_max`, a corridor on y and terminal
rest on the centreline. Looking at its output in isolation is the quickest way
to see the two things that decide the benchmark:

  1. the acceleration trace pressed against +/- the budget, which is where the
     manoeuvre time comes from, and
  2. the deadline entering as a HARD corridor constraint rather than as a cost
     -- so a task that is slightly too tight is INFEASIBLE rather than merely
     expensive. That is the structural reason the QP misses the deadline on a
     quarter of the benchmark tasks.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dlc_baselines import DLCTrajectoryPlanner
from dlc_rl_env import (DLCTask, decode_action, lane_occupancy,
                        peak_lateral_accel)

G = 9.81


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=float, default=None, help="speed [m/s]")
    ap.add_argument("--offset", type=float, default=None,
                    help="lateral offset [m]")
    ap.add_argument("--hold", type=float, default=None, help="hold time [s]")
    ap.add_argument("--start", type=float, default=None, help="start time [s]")
    ap.add_argument("--deadline", type=float, default=None, help="deadline [s]")
    ap.add_argument("--budget", type=float, default=None,
                    help="lateral acceleration budget [g]")
    ap.add_argument("--mu", type=float, default=None, help="road friction")
    ap.add_argument("--compare", action="store_true",
                    help="overlay the tanh reference the RL action space uses")
    ap.add_argument("--track", action="store_true",
                    help="also drive the plan through the tracking MPC")
    ap.add_argument("--out", default="out/planner_path.png")
    args = ap.parse_args()

    task = DLCTask()
    for name, value in (("V", args.V), ("lateral_offset", args.offset),
                        ("hold_time", args.hold), ("start_time", args.start),
                        ("deadline", args.deadline),
                        ("a_y_budget", args.budget), ("mu", args.mu)):
        if value is not None:
            setattr(task, name, value)

    print("=" * 78)
    print("2-MPC planner (DLCTrajectoryPlanner) on one task")
    print("=" * 78)
    print(f"  V {task.V} m/s | offset {task.lateral_offset} m | "
          f"hold {task.hold_time} s from t={task.start_time} s")
    print(f"  deadline {task.deadline} s | a_y budget {task.a_y_budget} g "
          f"= {task.a_y_max:.2f} m/s^2 | mu {task.mu}")

    planner = DLCTrajectoryPlanner()
    pr = planner.plan(task)

    print(f"\n  solver status : {pr.status}")
    print(f"  solve time    : {pr.solve_time*1e3:.1f} ms")
    if not pr.ok or pr.ref is None:
        print("\n  INFEASIBLE. The deadline and the hold window are a HARD")
        print("  corridor in this formulation, so a task the acceleration")
        print("  budget cannot reach in time has no solution at all -- the")
        print("  planner cannot return a slightly-late trajectory. Try a")
        print("  later --deadline or a larger --budget.")
        return 1

    ref = pr.ref
    t = ref.t_grid
    y, dy, ddy = ref._y, ref._dy, ref._ddy
    peak_a = float(np.max(np.abs(ddy)))
    at_limit = float(np.mean(np.abs(ddy) > 0.98 * task.a_y_max) * 100)
    settled = np.nonzero(np.abs(y) > 0.05)[0]
    t_done = float(t[settled[-1]]) if settled.size else 0.0
    # Call the reward's OWN measurement rather than reimplementing it. The
    # test is one-sided (`y >= offset - tol`), because sitting further into
    # the adjacent lane than asked still counts as occupying it. A two-sided
    # `abs(y - offset) <= tol` check reported 1.4 s instead of 7.0 s here,
    # purely because the QP's jerk-optimal path overshoots to 4.1 m.
    occ = lane_occupancy(ref, task.lateral_offset, t_end=task.sim_time + 5.0)

    print(f"\n  peak |a_y|    : {peak_a:.3f} m/s^2 = {peak_a/G:.3f} g "
          f"({100*peak_a/task.a_y_max:.1f} % of the budget)")
    print(f"  at the limit  : {at_limit:.0f} % of the horizon")
    print(f"  peak |y_dot|  : {float(np.max(np.abs(dy))):.3f} m/s")
    print(f"  max lateral   : {float(np.max(y)):.3f} m "
          f"(asked for {task.lateral_offset})")
    print(f"  lane occupancy: {occ:.2f} s (asked for {task.hold_time})")
    print(f"  completed at  : {t_done:.2f} s "
          f"(deadline {task.deadline:.2f} s, "
          f"margin {task.deadline - t_done:+.2f} s)")

    # ---- optional: the tanh shape the RL action space is restricted to -----
    tanh_ref = None
    if args.compare:
        tanh_ref = decode_action(np.zeros(4), task)
        print(f"\n  tanh reference (RL action space, neutral action):")
        print(f"    rate {tanh_ref.rate:.3f} | amplitude {tanh_ref.amplitude:.3f} m")
        print(f"    peak a_y {peak_lateral_accel(tanh_ref, task.sim_time + 5)/G:.3f} g")

    # ---- optional: what the vehicle actually did --------------------------
    res = None
    if args.track:
        from ttv_dlc_tracking import dlc_config, run_dlc_tracking, FailureLimits
        cfg = dlc_config(V=task.V, mu=task.mu, Qpsi=1.0, Qphi=25.0, Qq=8.0)
        res = run_dlc_tracking(cfg, ref, task.sim_time, backend="fast",
                               limits=FailureLimits(enabled=True))
        m = res.metrics
        print(f"\n  driven through the tracking MPC and the nonlinear plant:")
        print(f"    ok              : {res.ok}  ({res.reason})")
        print(f"    RMS lateral err : {m['rms_lateral_error']:.4f} m")
        print(f"    peak articulation: {math.degrees(m['peak_articulation_angle']):.2f} deg")
        print(f"    max front slip  : {math.degrees(m['max_front_slip']):.2f} deg")
        print(f"    peak steering   : {math.degrees(m['peak_steering_command']):.2f} deg")

    # ---- plot --------------------------------------------------------------
    n = 4 if args.track else 3
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 2.4 * n), sharex=True)
    ink, hot, ref_c = "#202124", "#e8710a", "#1a73e8"

    ax = axes[0]
    ax.plot(t, y, color=hot, lw=2.2, label="2-MPC planner")
    if tanh_ref is not None:
        ax.plot(t, tanh_ref.y(t), color=ref_c, lw=1.8, ls="--",
                label="tanh reference (RL action space)")
    ax.axhline(task.lateral_offset, color=ink, ls=":", lw=1.0)
    ax.axhspan(task.lateral_offset - 0.25, task.lateral_offset + 0.25,
               color=ink, alpha=0.06)
    ax.axvspan(task.start_time, task.start_time + task.hold_time,
               color=hot, alpha=0.07)
    ax.axvline(task.deadline, color="#c5221f", ls="--", lw=1.2)
    ax.text(task.deadline, ax.get_ylim()[1], " deadline", color="#c5221f",
            va="top", fontsize=9)
    ax.set_ylabel("lateral y [m]")
    ax.set_title("What the planning QP produced", loc="left", fontweight="bold")
    ax.legend(loc="center right", fontsize=9, frameon=False)

    ax = axes[1]
    ax.plot(t, dy, color=hot, lw=2.0)
    if tanh_ref is not None:
        ax.plot(t, tanh_ref.dy(t), color=ref_c, lw=1.6, ls="--")
    ax.set_ylabel("lateral rate [m/s]")

    ax = axes[2]
    ax.plot(t, ddy, color=hot, lw=2.0)
    if tanh_ref is not None:
        ax.plot(t, tanh_ref.ddy(t), color=ref_c, lw=1.6, ls="--")
    for s in (+1, -1):
        ax.axhline(s * task.a_y_max, color="#c5221f", ls="--", lw=1.2)
    ax.text(t[-1], task.a_y_max, " budget", color="#c5221f", va="bottom",
            ha="right", fontsize=9)
    ax.set_ylabel("lateral accel [m/s$^2$]")

    if args.track and res is not None:
        ax = axes[3]
        ax.plot(res.t_control, np.degrees(res.delta_cmd), color="#188038",
                lw=1.8, label="steering command")
        ax.set_ylabel("steering [deg]")
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    axes[-1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())