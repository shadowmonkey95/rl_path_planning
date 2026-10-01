"""
speed_ablation.py -- Getting the shipped example past 60 km/h.

The shipped configuration fails above 12 m/s (43 km/h). Diagnosis first:

    at 60 km/h the path demands 0.534 g, which needs 6.95 deg of front slip
    quasi-statically. The failure limit is 11.46 deg. The closed loop produces
    14.37 deg -- a 2.07x transient overshoot.

So the path is NOT the problem at 60 km/h; the controller is. This sweeps one
controller parameter at a time and reports what each is worth, so the fix is
chosen by measurement rather than by taste.

For reference, the hard ceiling for THIS path, whatever the controller:
    70 km/h needs 11.40 deg quasi-static (limit 11.46) -- no margin at all
    80 km/h needs 105 % of available grip -- impossible
"""
import math
import json
import numpy as np

from ttv_core import TTVConfig, MPCConfig, ttv_rl_episode, example_path

V60 = 60.0 / 3.6
LIMIT_DEG = math.degrees(0.2)
QUASI_STATIC_DEG = 6.95


def run(label, **kw):
    mpc_kw = {k: kw.pop(k) for k in list(kw)
              if k in ("Qy", "Qpsi", "Qphi", "Qq", "Rdelta")}
    kw.setdefault("V", V60)
    cfg = TTVConfig(**kw)
    for k, v in mpc_kw.items():
        setattr(cfg.mpc, k, v)
    try:
        r, p, l = ttv_rl_episode(example_path(), cfg)
    except Exception as e:
        return {"label": label, "pass": False, "slip": float("nan"),
                "reason": f"EXC {e}"[:30], "ey": float("nan"),
                "phi": float("nan"), "steer": float("nan")}
    return {"label": label, "pass": p["passed"],
            "slip": math.degrees(p["maxSlipFront"]),
            "reason": p["failureReason"][:28],
            "ey": l["metrics"]["peakLateralError"],
            "phi": math.degrees(l["metrics"]["peakArticulationAngle"]),
            "steer": math.degrees(l["metrics"]["peakSteeringCommand"])}


def show(title, rows):
    print(f"\n{title}")
    print(f"  {'configuration':>34s} {'pass':>6s} {'peak slip':>10s} "
          f"{'x q-s':>6s} {'peak e_y':>9s} {'phi':>6s} {'steer':>6s} {'why it stopped':>20s}")
    for r in rows:
        ov = r["slip"] / QUASI_STATIC_DEG if r["slip"] == r["slip"] else float("nan")
        print(f"  {r['label']:>34s} {('PASS' if r['pass'] else 'fail'):>6s} "
              f"{r['slip']:9.2f}d {ov:6.2f} {r['ey']:9.3f} {r['phi']:6.2f} "
              f"{r['steer']:6.2f} {r['reason']:>20s}")


if __name__ == "__main__":
    print("=" * 118)
    print(f"Shipped example at 60 km/h.  quasi-static demand {QUASI_STATIC_DEG:.2f} deg, "
          f"failure limit {LIMIT_DEG:.2f} deg")
    print("=" * 118)

    out = {}

    base = run("shipped defaults")
    show("BASELINE", [base])

    out["horizon"] = [run(f"N={n} ({n*0.1:.1f} s preview)", N=n)
                      for n in (10, 15, 20, 30, 40, 60)]
    show("A. MPC horizon (T = 0.1 s)", out["horizon"])

    out["artic"] = [run(f"Qphi={a}, Qq={b}", Qphi=a, Qq=b)
                    for a, b in ((0, 0), (5, 2), (25, 8), (100, 25), (400, 100))]
    show("B. articulation damping in the MPC cost", out["artic"])

    out["reff"] = [run(f"Rdelta={r}", Rdelta=r)
                   for r in (0.02, 0.1, 0.5, 2.0, 8.0)]
    show("C. steering-effort weight", out["reff"])

    out["track"] = [run(f"Qy={a}, Qpsi={b}", Qy=a, Qpsi=b)
                    for a, b in ((2, 1), (4, 1), (2, 4), (1, 4), (0.5, 8))]
    show("D. tracking weights (lateral vs heading)", out["track"])

    out["act"] = [run(f"tau={t}, rate={d}", steeringTimeConstant=t, deltaRateMax=d)
                  for t, d in ((0.15, 0.6), (0.10, 0.6), (0.05, 0.6),
                               (0.15, 1.2), (0.15, 0.3))]
    show("E. steering actuator (a hardware change, not a tuning change)", out["act"])

    out["numerics"] = [run(f"T={t}, substeps={s}", T=t, plantSubsteps=s)
                       for t, s in ((0.1, 5), (0.05, 5), (0.02, 5), (0.1, 20))]
    show("F. numerics (sanity: is any of this an integration artefact?)",
         out["numerics"])

    # ---- the combination suggested by the single-lever results --------------
    combos = [
        run("N=30", N=30),
        run("N=30 + Qphi/Qq 25/8", N=30, Qphi=25, Qq=8),
        run("N=30 + Qphi/Qq 25/8 + Rdelta 0.5", N=30, Qphi=25, Qq=8, Rdelta=0.5),
        run("N=30 + Qphi/Qq + R + Qpsi 4", N=30, Qphi=25, Qq=8, Rdelta=0.5,
            Qy=2, Qpsi=4),
        run("all of the above, T=0.05", N=60, T=0.05, Qphi=25, Qq=8,
            Rdelta=0.5, Qy=2, Qpsi=4),
    ]
    show("G. combinations", combos)
    out["combos"] = combos

    # ---- how fast can the best configuration go? ---------------------------
    best_kw = dict(N=30, Qphi=25, Qq=8, Rdelta=0.5, Qy=2, Qpsi=4)
    print("\nH. top speed of the best configuration found, on this path")
    print(f"  {'km/h':>6s} {'shipped':>9s} {'tuned':>9s} {'tuned slip':>11s} "
          f"{'quasi-static':>13s}")
    sweep = []
    k = 0.018851
    tot = None
    for kmh in (43, 50, 55, 60, 65, 70, 75):
        V = kmh / 3.6
        a = run("s", V=V)
        b = run("t", V=V, **best_kw)
        cfg = TTVConfig().finalize()
        tot = cfg.Fz1 + cfg.Fz2 + cfg.Fz3
        ay = V * V * k
        Fy1 = (cfg.Fz1 / tot) * ((cfg.m1 + cfg.m2) * ay)
        sat = cfg.mu * cfg.Fz1
        z = Fy1 / sat
        qs = math.degrees(math.atanh(min(z, 0.999)) * sat / cfg.C1) if z < 1 else float("inf")
        sweep.append({"kmh": kmh, "shipped": a["pass"], "tuned": b["pass"],
                      "tuned_slip": b["slip"], "quasi_static": qs})
        print(f"  {kmh:6.0f} {('PASS' if a['pass'] else 'fail'):>9s} "
              f"{('PASS' if b['pass'] else 'fail'):>9s} {b['slip']:10.2f}d "
              f"{qs:12.2f}d")
    out["speed_sweep"] = sweep

    json.dump(out, open("out/speed_ablation.json", "w"), indent=2, default=str)
    print("\n  wrote out/speed_ablation.json")
