"""
make_figures.py -- The pictures that carry the argument.

fig3  the truck vs the car: which constraint binds, and the SRT/load story
fig4  one scenario, three planners, with the rig's actual footprint
fig5  learning curves and the benchmark comparison
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import truck_params as TP
from highway_env import (Scenario, TruckHighwayEnv, action_to_path, rollout,
                         truck_geometry, RewardWeights)
from fast_mpc import CondensedMPC
from baselines import MPCPlanner, run_mpc_planner, heuristic_action
from path_gen import min_length_for_offset

G = 9.81
C = {"rl": "#2563eb", "mpc2": "#d97706", "heur": "#6b7280", "obs": "#dc2626"}


# =============================================================================
def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # -- which constraint binds, vs speed ---------------------------------
    V = np.linspace(15, 28, 200)
    a = ax[0]
    a.plot(V * 3.6, TP.kappa_max_rollover(V) * 1e3, lw=2.5, color="#dc2626",
           label="rollover (laden, SRT 0.37 g)")
    for mu, ls in ((0.9, "-"), (0.6, "--"), (0.35, ":")):
        a.plot(V * 3.6, TP.kappa_max_friction(V, mu) * 1e3, ls=ls, lw=1.8,
               color="#2563eb", label=f"friction, mu = {mu}")
    a.set_xlabel("speed [km/h]")
    a.set_ylabel(r"curvature ceiling [1/km]")
    a.set_title("A 40 t rig rolls before it slides\n(lower curve is the binding limit)")
    a.legend(fontsize=7.5)
    a.grid(alpha=.3)

    # -- SRT vs load ------------------------------------------------------
    a = ax[1]
    pf = np.linspace(0, 1, 100)
    srt = [Scenario(payload_fraction=float(p)).srt for p in pf]
    a.plot(pf * 100, np.array(srt), lw=2.5, color="#dc2626")
    a.fill_between(pf * 100, 0, srt, alpha=.12, color="#dc2626")
    a.set_xlabel("payload [% of maximum]")
    a.set_ylabel("Static Rollover Threshold [g]")
    a.set_title("SRT falls 1.7x from empty to laden\n"
                "a policy blind to load cannot be safe in both")
    a.grid(alpha=.3)
    for p in (0.0, 0.5, 1.0):
        s = Scenario(payload_fraction=p).srt
        a.annotate(f"{s:.2f} g", (p * 100, s), textcoords="offset points",
                   xytext=(6 if p < 1 else -34, 6), fontsize=8)

    # -- rearward amplification -------------------------------------------
    # RWA turns out to be almost independent of payload here (1.2580 empty vs
    # 1.2570 laden) because tyre cornering stiffness scales with vertical load,
    # so the trailer's mass-to-stiffness ratios barely move. The load story is
    # entirely in SRT (middle panel); RWA is a roughly constant multiplier on
    # top of it. Saying that is more useful than three curves on top of
    # each other.
    a = ax[2]
    f = np.linspace(0.05, 1.0, 300)
    cfg = Scenario(payload_fraction=1.0).config()
    r = np.array([TP.rearward_amplification(cfg, fi) for fi in f])
    a.plot(f, r, lw=2.5, color="#dc2626", label="laden (empty differs by 0.001)")
    a.axhline(1.0, color="k", lw=.9, alpha=.6)
    # The band a highway lane change actually excites. ONE lateral transition
    # is a FULL cycle of steering -- in, then out -- over its length L, so
    # f = V/L, not V/(2L). Cross-check: the shortest rollover-safe transition is
    # 74.8 m, giving 0.32 Hz and RWA 1.26; a long 250 m transition gives
    # 0.10 Hz and RWA 1.03. The rollouts measured 1.04-1.27, which brackets
    # exactly that, so the estimate is the right one.
    V0 = 24.0
    f_lo, f_hi = V0 / 250.0, V0 / 74.8
    a.axvspan(f_lo, f_hi, color="#2563eb", alpha=.15,
              label=f"band a truck lane change excites\n"
                    f"({f_lo:.2f}-{f_hi:.2f} Hz at {V0*3.6:.0f} km/h)")
    i = int(np.argmax(r))
    a.annotate(f"peak {r[i]:.2f}", (f[i], r[i]), textcoords="offset points",
               xytext=(8, -2), fontsize=8, color="#dc2626")
    a.set_xlabel("manoeuvre frequency [Hz]")
    a.set_ylabel("rearward amplification [-]")
    a.set_ylim(0, 1.45)
    a.set_title("The trailer sees 10-26% more lateral g than the tractor\n"
                "a planner that constrains the tractor misses this")
    a.legend(fontsize=7, loc="lower left")
    a.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("out/fig3_truck_physics.png", dpi=130)
    print("  wrote out/fig3_truck_physics.png")


# =============================================================================
def _draw_rig(ax, cfg, xp, color, alpha=0.5):
    """The rig's actual footprint: tractor body and trailer body as polygons."""
    g = truck_geometry(cfg, xp)
    hw = TP.VEHICLE_WIDTH / 2

    def body(xa, ya, xb, yb):
        dx, dy = xb - xa, yb - ya
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * hw, dx / L * hw
        ax.fill([xa + nx, xb + nx, xb - nx, xa - nx],
                [ya + ny, yb + ny, yb - ny, ya - ny],
                color=color, alpha=alpha, lw=0.8, edgecolor=color, zorder=4)

    body(g["x_front"], g["y_front"], g["x_drive"], g["y_drive"])
    body(g["x_hitch"], g["y_hitch"], g["x_rear"], g["y_rear"])


def fig4(agent_path="out/rl_agent.pt"):
    import torch
    from train_rl import load_agent

    # A demanding scenario, not a comfortable one: the obstacle is close enough
    # that the manoeuvre has to be short, the rig is laden, and the road is
    # damp. On an easy scenario all three planners look alike.
    scen = Scenario(V=24.0, mu=0.60, payload_fraction=1.0, obstacle_x=95.0,
                    obstacle_len=16.0, obstacle_half_width=1.5,
                    must_return=True, e_y0=0.25)
    cfg = scen.config()
    tmpc = CondensedMPC(cfg, use_rate_constraint=True)
    env = TruckHighwayEnv(randomise=False)
    planner = MPCPlanner(ds=2.0, horizon_m=400.0)

    runs = {}
    actor, norm = load_agent(agent_path)
    obs = env.reset(scen)
    with torch.no_grad():
        a, _ = actor(torch.as_tensor(norm(obs), dtype=torch.float32).unsqueeze(0),
                     deterministic=True, with_logp=False)
    _, r_rl, _, i_rl = env.step(a.squeeze(0).numpy())
    runs["RL planner"] = (r_rl, i_rl["path"], i_rl["rollout"], i_rl["metrics"], C["rl"])

    r_m, i_m = run_mpc_planner(scen, planner, RewardWeights(), mpc=tmpc)
    runs["2-MPC planner"] = (r_m, i_m["path"], i_m["rollout"], i_m["metrics"], C["mpc2"])

    env.reset(scen)
    _, r_h, _, i_h = env.step(heuristic_action(scen))
    runs["heuristic"] = (r_h, i_h["path"], i_h["rollout"], i_h["metrics"], C["heur"])

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.62)
    lw_ = scen.lane_width

    # --- plan view -------------------------------------------------------
    a = fig.add_subplot(gs[0])
    a.axhspan(-lw_ / 2, lw_ / 2, color="#f3f4f6")
    a.axhspan(lw_ / 2, lw_ * 1.5, color="#e5e7eb")
    for yl in (-lw_ / 2, lw_ / 2, lw_ * 1.5):
        a.axhline(yl, color="k", lw=1.2, ls="--" if abs(yl - lw_ / 2) < 1e-9 else "-")
    a.add_patch(Rectangle((scen.obstacle_x, -scen.obstacle_half_width),
                          scen.obstacle_len, 2 * scen.obstacle_half_width,
                          color=C["obs"], alpha=.8, zorder=5))
    a.annotate("obstacle", (scen.obstacle_x + scen.obstacle_len / 2, 1.7),
               ha="center", color=C["obs"], fontsize=9, zorder=6)

    for name, (rw, path, res, m, col) in runs.items():
        a.plot(path.x, path.y, color=col, lw=1.1, ls=":", alpha=.8)
        a.plot(res.x_front, res.y_front, color=col, lw=2.0,
               label=f"{name}  (R={rw:+.3f}, LTR={m['peak_LTR']:.2f})")
        a.plot(res.x_tridem, res.y_tridem, color=col, lw=1.0, alpha=.55)
    # the rig, drawn at its widest moment, for the winning method
    rw, path, res, m, col = runs["RL planner"]
    k = int(np.argmax(np.abs(res.y_front)))
    _draw_rig(a, cfg, res.xp[:, k], col, .35)
    a.set_xlim(0, 340)
    a.set_ylim(-2.4, lw_ * 1.5 + 1.2)
    a.set_xlabel("along-road distance [m]")
    a.set_ylabel("lateral [m]")
    a.set_title(f"A demanding highway obstacle-avoidance scenario, three planners\n"
                f"V = {scen.V*3.6:.0f} km/h, mu = {scen.mu:.2f}, laden, obstacle at "
                f"{scen.obstacle_x:.0f} m, 0.25 m off-centre at the start\n"
                "dotted = commanded path, thick = steer axle, thin = trailer tridem, "
                "shaded = the rig at its widest",
                fontsize=10.5, pad=10)
    a.legend(fontsize=8, loc="upper right")

    # --- LTR -------------------------------------------------------------
    a = fig.add_subplot(gs[1])
    for name, (rw, path, res, m, col) in runs.items():
        a.plot(res.t, res.ltr, color=col, lw=1.8, label=name)
    a.axhline(1.0, color=C["obs"], ls="--", lw=1.5, label="wheel lift (LTR=1)")
    a.set_ylabel("Load Transfer\nRatio [-]")
    a.set_title("rollover margin -- the binding truck constraint", fontsize=9, pad=6)
    a.set_xlabel("")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=.3)

    # --- lateral accel, tractor vs trailer -------------------------------
    a = fig.add_subplot(gs[2])
    for name, (rw, path, res, m, col) in runs.items():
        a.plot(res.t, res.a_y1 / G, color=col, lw=1.6, alpha=.55)
        a.plot(res.t, res.a_y2 / G, color=col, lw=1.8)
    a.axhline(scen.srt, color=C["obs"], ls="--", lw=1.2)
    a.axhline(-scen.srt, color=C["obs"], ls="--", lw=1.2)
    a.annotate(f"SRT = {scen.srt:.2f} g", (0.5, scen.srt), fontsize=7,
               color=C["obs"], va="bottom")
    a.set_ylabel("lateral accel [g]")
    a.set_title("faint = tractor, bold = trailer: the trailer is always worse",
                fontsize=9, pad=6)
    a.grid(alpha=.3)

    # --- articulation ----------------------------------------------------
    a = fig.add_subplot(gs[3])
    for name, (rw, path, res, m, col) in runs.items():
        a.plot(res.t, np.degrees(res.xp[5, :]), color=col, lw=1.8)
    a.axhline(math.degrees(cfg.phiMax), color=C["obs"], ls="--", lw=1.2,
              label=f"jack-knife guard {math.degrees(cfg.phiMax):.0f} deg")
    a.axhline(-math.degrees(cfg.phiMax), color=C["obs"], ls="--", lw=1.2)
    a.set_xlabel("time [s]")
    a.set_ylabel("articulation [deg]")
    a.legend(fontsize=7)
    a.grid(alpha=.3)

    plt.savefig("out/fig4_scenario_comparison.png", dpi=130, bbox_inches="tight")
    print("  wrote out/fig4_scenario_comparison.png")
    return {k: (v[0], v[3]) for k, v in runs.items()}


# =============================================================================
def fig5():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- learning curves --------------------------------------------------
    a = ax[0]
    # Only the current configuration's curve. The second seed I ran was on the
    # PRE-FIX action space and observation vector (13 obs, margin 0.85), so its
    # curve is not comparable and plotting it beside this one would mislead.
    # The reproducibility evidence lives in the text instead.
    for path, lab, col in (("out/rl_training.json", "held-out eval", "#2563eb"),):
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            continue
        ev = d["evals"]
        a.plot([e["episode"] for e in ev], [e["reward"] for e in ev],
               "o-", ms=3.5, lw=1.8, color=col, label=lab)
    try:
        b = json.load(open("out/benchmark.json"))["summary"]
        a.axhline(b["mpc2"]["reward_mean"], color=C["mpc2"], ls="--", lw=1.8,
                  label="2-MPC planner")
        a.axhline(b["heuristic"]["reward_mean"], color=C["heur"], ls=":", lw=1.8,
                  label="fixed heuristic")
        if "optimum" in b:
            a.axhline(b["optimum"]["reward_mean"], color="#16a34a", ls="-.",
                      lw=1.8, label="offline optimum")
    except (FileNotFoundError, KeyError):
        pass
    a.set_xlabel("episodes")
    a.set_ylabel("mean reward, held-out scenarios")
    a.set_title("Learning curve\n(the paper used 150 000 episodes)")
    a.legend(fontsize=7.5)
    a.grid(alpha=.3)

    # --- benchmark bars ---------------------------------------------------
    try:
        b = json.load(open("out/benchmark.json"))["summary"]
        order = [k for k in ("heuristic", "mpc2", "rl", "optimum") if k in b]
        lab = {"heuristic": "heuristic", "mpc2": "2-MPC", "rl": "RL",
               "optimum": "offline\noptimum"}
        cols = {"heuristic": C["heur"], "mpc2": C["mpc2"], "rl": C["rl"],
                "optimum": "#16a34a"}
        a = ax[1]
        xs = np.arange(len(order))
        a.bar(xs, [b[k]["reward_mean"] for k in order],
              yerr=[b[k]["reward_std"] for k in order],
              color=[cols[k] for k in order], capsize=4, alpha=.85)
        a.set_xticks(xs)
        a.set_xticklabels([lab[k] for k in order], fontsize=8)
        a.set_ylabel("mean reward")
        a.set_title("Held-out benchmark\n(error bars = 1 sd across scenarios)")
        a.grid(alpha=.3, axis="y")

        a = ax[2]
        keys = ["mean_peak_LTR", "mean_swept", "mean_jerk", "mean_offtrack"]
        names = ["peak LTR", "swept width\n/3 [m]", "jerk /8\n[m/s3]",
                 "off-track\n[m]"]
        scale = [1.0, 1 / 3, 1 / 8, 1.0]
        w = 0.8 / len(order)
        xs = np.arange(len(keys))
        for i, k in enumerate(order):
            vals = [b[k][kk] * s for kk, s in zip(keys, scale)]
            a.bar(xs + i * w, vals, w, label=lab[k].replace("\n", " "),
                  color=cols[k], alpha=.85)
        a.set_xticks(xs + 0.4 - w / 2)
        a.set_xticklabels(names, fontsize=7.5)
        a.set_title("Truck safety metrics (lower is better)")
        a.legend(fontsize=7)
        a.grid(alpha=.3, axis="y")
    except (FileNotFoundError, KeyError) as e:
        print(f"  (benchmark.json not ready: {e})")

    plt.tight_layout()
    plt.savefig("out/fig5_results.png", dpi=130)
    print("  wrote out/fig5_results.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "3"):
        fig3()
    if which in ("all", "4"):
        print(fig4())
    if which in ("all", "5"):
        fig5()
