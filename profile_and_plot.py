"""
profile_and_plot.py -- (1) how expensive is the shipped loop for RL training,
                       (2) a picture of what the shipped example actually does.
"""
import json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ttv_core import (TTVConfig, ttv_rl_episode, build_environment,
                      make_cache_key, example_path, prepare_path, _ENV_CACHE)

OUT = {}

# ---------------------------------------------------------------- profiling
print("=" * 78)
print("H.  Cost of the shipped loop, per RL episode")
print("=" * 78)

cfg = TTVConfig(V=12.0)
cfg.finalize()

t0 = time.perf_counter()
_ = build_environment(cfg, make_cache_key(cfg))
t_build = time.perf_counter() - t0

_ENV_CACHE.clear()
t0 = time.perf_counter()
ttv_rl_episode(example_path(), TTVConfig(V=12.0))
t_cold = time.perf_counter() - t0

ts = []
for _ in range(5):
    t0 = time.perf_counter()
    ttv_rl_episode(example_path(), TTVConfig(V=12.0))
    ts.append(time.perf_counter() - t0)
t_warm = float(np.mean(ts))

print(f"  build_environment (IPOPT codegen + plant build) : {t_build*1e3:8.1f} ms")
print(f"  first episode, cold cache                       : {t_cold*1e3:8.1f} ms")
print(f"  episode with warm cache (mean of 5)             : {t_warm*1e3:8.1f} ms")
print(f"  rebuild overhead                                : {t_build/t_warm:8.1f}x one episode")

EP = 150_000  # the paper trained for ~150k episodes
print(f"\n  Paper's budget: {EP:,} episodes.")
print(f"    warm cache, one core      : {EP*t_warm/3600:7.2f} h")
print(f"    rebuilt every episode     : {EP*(t_warm+t_build)/3600:7.2f} h   <-- what happens")
print(f"                                          when mu or V is randomised per")
print(f"                                          episode (paper Table 2), because")
print(f"                                          the cache key contains mu and V.")
OUT["H_profile"] = {"build_ms": t_build*1e3, "cold_ms": t_cold*1e3,
                    "warm_ms": t_warm*1e3, "rebuild_x": t_build/t_warm,
                    "hours_warm": EP*t_warm/3600,
                    "hours_rebuild": EP*(t_warm+t_build)/3600}

# ---------------------------------------------------------------- plots
cfgA = TTVConfig(V=20.0)
rA, pA, lA = ttv_rl_episode(example_path(), cfgA)
cfgB = TTVConfig(V=12.0)
rB, pB, lB = ttv_rl_episode(example_path(), cfgB)
path = prepare_path(example_path())

fig, ax = plt.subplots(3, 2, figsize=(13, 10.5))
fig.suptitle("Shipped ttv_rl_episode.m on its own docstring example\n"
             "left: V = 20 m/s (as shipped, FAILS)   |   right: V = 12 m/s (passes)",
             fontsize=12)

for col, (lg, pr, cf, ttl) in enumerate([(lA, pA, cfgA, "V = 20 m/s"),
                                         (lB, pB, cfgB, "V = 12 m/s")]):
    ps = lg["plantState"]
    t = lg["time"]

    a = ax[0, col]
    a.plot(path.x, path.y, "k--", lw=1.2, label="reference path")
    a.plot(ps[0, :], ps[1, :], lw=2.0, label="tractor CoG")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    a.set_title(f"{ttl} - {'PASS' if pr['passed'] else 'FAIL: ' + pr['failureReason']}"
                f"  (reward {pr['total']:.3f})")
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_xlim(0, 125)

    a = ax[1, col]
    a.plot(t, np.degrees(lg["slipAngles"][0, :]), label=r"$\alpha_1$ tractor front")
    a.plot(t, np.degrees(lg["slipAngles"][1, :]), label=r"$\alpha_2$ tractor rear")
    a.plot(t, np.degrees(lg["slipAngles"][2, :]), label=r"$\alpha_3$ trailer")
    a.axhline(math.degrees(cf.reward.maxLateralSlip), color="r", ls=":", lw=1.5,
              label="0.2 rad fail limit")
    a.axhline(-math.degrees(cf.reward.maxLateralSlip), color="r", ls=":", lw=1.5)
    a.set_xlabel("t [s]"); a.set_ylabel("slip angle [deg]")
    a.set_title("tyre slip angles"); a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[2, col]
    a.plot(t, np.degrees(ps[5, :]), label=r"articulation $\varphi$ [deg]")
    a.plot(t, np.degrees(ps[7, :]), label=r"steer $\delta_{act}$ [deg]")
    a.plot(t, lg["lateralError"] * 100, label="lateral error [cm]")
    a.set_xlabel("t [s]"); a.set_title("articulation, steering, tracking")
    a.legend(fontsize=7); a.grid(alpha=.3)

plt.tight_layout()
plt.savefig("out/fig1_shipped_behaviour.png", dpi=130)
print("\n  wrote out/fig1_shipped_behaviour.png")

# --- reward pathology figure
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
v = np.linspace(30, 95, 300)
mu = 0.0037 * np.exp(0.0693 * v)
ax[0].plot(v, np.degrees(mu), lw=2)
ax[0].axvspan(40, 60, alpha=.15, color="g", label="paper's fit range")
ax[0].axhline(math.degrees(0.2), color="r", ls=":", label="0.2 rad failure limit")
ax[0].axvline(72, color="k", ls="--", label="shipped V = 72 km/h")
ax[0].set_xlabel(r"$v_0$ [km/h]"); ax[0].set_ylabel(r"$\mu_{max}$ [deg]")
ax[0].set_title("Eq.(3) slip reference, extrapolated")
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_ylim(0, 80)

res = json.load(open("out/validation_results.json"))["F_resolution"]["rows"]
ax[1].semilogx([r["n"] for r in res], [-r["reward_curv"] for r in res], "o-", lw=2)
ax[1].set_xlabel("path points (same geometry)")
ax[1].set_ylabel(r"$-$reward$_\kappa$")
ax[1].set_title("Eq.(5) via 3 nested gradients:\nresolution-dependent")
ax[1].grid(alpha=.3, which="both")

sw = json.load(open("out/validation_results.json"))["D_speed_sweep"]
vs = [r["V"] for r in sw]
ax[2].plot(vs, [r["maxSlipFront_deg"] for r in sw], "o-", lw=2, label="max front slip")
ax[2].axhline(math.degrees(0.2), color="r", ls=":", label="fail limit")
ax[2].set_xlabel("V [m/s]"); ax[2].set_ylabel("max front slip [deg]")
for r in sw:
    ax[2].annotate("pass" if r["pass"] else "fail", (r["V"], r["maxSlipFront_deg"]),
                   textcoords="offset points", xytext=(0, 8), fontsize=7,
                   ha="center", color="g" if r["pass"] else "r")
ax[2].set_title("shipped example path vs speed")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
plt.tight_layout()
plt.savefig("out/fig2_reward_pathologies.png", dpi=130)
print("  wrote out/fig2_reward_pathologies.png")

json.dump(OUT, open("out/profile_results.json", "w"), indent=2)
