"""
calibrate_margin.py -- Turning a path guarantee into a closed-loop guarantee
===========================================================================

`min_length_for_offset` guarantees that the COMMANDED path's peak curvature is
at or below the ceiling. That is a statement about geometry. It does not
guarantee the closed-loop Load Transfer Ratio stays below 1, and at the first
margin I tried (0.85) it did not: a 400-action random sweep produced 2
rollovers, worst LTR 1.30.

Two things live in the gap:

  1. REARWARD AMPLIFICATION. The ceiling limits the tractor; the trailer rolls.
     `kappa_ceiling_for_length` now divides SRT by RWA at the frequency the
     manoeuvre excites, which removes 3-25% of it.

  2. THE CORRECTION TRANSIENT. The rig does not start perfectly centred. At low
     friction the MPC's linear model over-commands into saturated tyres, and
     the trailer is then dragged laterally through the kingpin -- which is not
     limited by the trailer's own tyre grip. Both measured rollovers were at
     mu = 0.46 and 0.52 with 0.37-0.39 m of initial lateral offset.

The second one has no closed form here, so the margin is calibrated: sweep it
and take the largest value with no rollover across a large random-action sweep,
with headroom. This is the honest way round -- a guarantee you have measured
beats a bound you have asserted.
"""

import json
import numpy as np

from highway_env import TruckHighwayEnv, Scenario


def sweep(margins=(0.85, 0.80, 0.75, 0.70, 0.66, 0.62, 0.58),
          n: int = 400, seed: int = 5) -> list:
    rows = []
    print(f"  {'margin':>7s} {'roll':>6s} {'coll':>6s} {'succ':>6s} "
          f"{'worst LTR':>10s} {'mean LTR':>9s} {'mean R':>8s} "
          f"{'mean L_out':>11s} {'worst k/cap':>12s}")
    for mg in margins:
        Scenario.KAPPA_MARGIN = mg
        env = TruckHighwayEnv(randomise=True, seed=seed)
        rng = np.random.default_rng(seed + 1)
        nroll = ncoll = nok = 0
        ltrs, rws, lens, kr = [], [], [], 0.0
        for _ in range(n):
            env.reset()
            a = rng.uniform(-1, 1, 9)
            _, r, _, info = env.step(a)
            m = info["metrics"]
            nroll += int(m["rollover"])
            ncoll += int(m["collision"])
            nok += int(info["parts"].get("cause") == "success")
            ltrs.append(m["peak_LTR"])
            rws.append(r)
            lens.append(info["decoded"]["L_out"])
            kr = max(kr, m["path_kappa_peak"] / info["decoded"]["kappa_ceiling"])
        row = {"margin": mg, "rollovers": nroll, "collisions": ncoll,
               "successes": nok, "worst_LTR": float(np.max(ltrs)),
               "mean_LTR": float(np.mean(ltrs)), "mean_reward": float(np.mean(rws)),
               "mean_L_out": float(np.mean(lens)), "worst_kappa_ratio": float(kr)}
        rows.append(row)
        print(f"  {mg:7.2f} {nroll:4d}/{n} {ncoll:4d}/{n} {nok:4d}/{n} "
              f"{row['worst_LTR']:10.3f} {row['mean_LTR']:9.3f} "
              f"{row['mean_reward']:8.4f} {row['mean_L_out']:11.1f} "
              f"{row['worst_kappa_ratio']:12.4f}")
    return rows


if __name__ == "__main__":
    print("=" * 96)
    print("Calibrating the curvature-ceiling margin against closed-loop rollouts")
    print("=" * 96)
    rows = sweep()
    safe = [r for r in rows if r["rollovers"] == 0]
    if safe:
        best = max(safe, key=lambda r: r["margin"])
        print(f"\n  largest margin with zero rollovers: {best['margin']:.2f} "
              f"(worst LTR {best['worst_LTR']:.3f})")
        print(f"  the code ships one step below it, for headroom.")
    else:
        print("\n  no margin in the sweep was clean -- widen the sweep")
    json.dump(rows, open("out/margin_calibration.json", "w"), indent=2)
    print("\n  wrote out/margin_calibration.json")
