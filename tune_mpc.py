"""
tune_mpc.py -- Pick the tracking-MPC horizon and weights by measurement.

The shipped values (T = 0.1 s, N = 10 -> 1.0 s preview; Qphi = Qq = 0) are a
car's. Rather than assert a truck needs more, sweep and look.
"""
import copy
import json
import math
import numpy as np

import truck_params as TP
from fast_mpc import CondensedMPC
from highway_env import Scenario, rollout, action_to_path, decode_action

np.set_printoptions(precision=4, suppress=True)

scen = Scenario(V=24.0, mu=0.85, payload_fraction=1.0, obstacle_x=170.0,
                must_return=True)

# Tune on a DEMANDING reference, not the neutral one. The neutral action
# decodes to a 132 m transition at 0.11 g -- far inside the envelope, where
# every horizon and every weight looks fine. a[1] = a[5] = -1 asks for the
# shortest transition the action space allows, i.e. curvature at the ceiling.
a_ref = np.array([-0.6, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
path = action_to_path(a_ref, scen, ds=0.5)
print("=" * 90)
print("Reference manoeuvre for tuning (at the curvature ceiling)")
print("=" * 90)
d = decode_action(a_ref, scen)
print(f"  dy = {d['dy']:.2f} m, L_out = {d['L_out']:.1f} m, "
      f"L_hold = {d['L_hold']:.1f} m, L_back = {d['L_back']:.1f} m")
print(f"  path length {path.s[-1]:.0f} m, kappa_peak = {path.kappa_peak:.5f} "
      f"(ceiling {scen.kappa_ceiling:.5f}), a_y peak = "
      f"{path.max_lateral_accel(scen.V)/9.81:.3f} g")

# =============================================================================
print("\n" + "=" * 90)
print("A.  Horizon sweep  (Qphi/Qq at their truck values)")
print("=" * 90)
print(f"  {'T':>5s} {'N':>4s} {'preview':>8s} {'nz':>4s} {'cond(P)':>9s} "
      f"{'rms e_y':>9s} {'peak e_y':>9s} {'LTR':>6s} {'phi':>6s} "
      f"{'us/solve':>9s} {'iters':>7s} {'ok':>5s}")
rows = []
for T, N in [(0.10, 10), (0.10, 15), (0.10, 20), (0.10, 30),
             (0.05, 20), (0.05, 30), (0.20, 10), (0.15, 12), (0.20, 15)]:
    cfg = scen.config()
    cfg.T, cfg.N = T, N
    cfg.plantSubsteps = max(2, int(round(T / 0.02)))
    cfg.finalize()
    mpc = CondensedMPC(cfg, use_rate_constraint=True)
    res = rollout(path, cfg, scen, max_time=1.3 * path.s[-1] / scen.V + 2.0, mpc=mpc)
    m = res.metrics
    row = {"T": T, "N": N, "preview": T * N, "nz": mpc.nz,
           "cond": float(np.linalg.cond(mpc.P_dense)),
           "rms_ey": m.get("rms_lateral_error", float("nan")),
           "peak_ey": m.get("peak_lateral_error", float("nan")),
           "ltr": m.get("peak_LTR", float("nan")),
           "phi": m.get("peak_articulation_deg", float("nan")),
           "us": m.get("mpc_solve_us", float("nan")),
           "ok": res.ok, "reason": res.reason}
    rows.append(row)
    print(f"  {T:5.2f} {N:4d} {T*N:8.2f} {mpc.nz:4d} {row['cond']:9.1e} "
          f"{row['rms_ey']:9.4f} {row['peak_ey']:9.4f} {row['ltr']:6.3f} "
          f"{row['phi']:6.2f} {row['us']:9.0f} {'-':>7s} "
          f"{('ok' if res.ok else res.reason[:5]):>5s}")

good = [r for r in rows if r["ok"]]
best = min(good, key=lambda r: (r["rms_ey"], r["us"])) if good else None
print(f"\n  lowest tracking error among feasible: T={best['T']}, N={best['N']} "
      f"({best['preview']:.1f} s preview), {best['us']:.0f} us/solve")

# =============================================================================
print("\n" + "=" * 90)
print("B.  Does penalising articulation in the MPC actually help?")
print("=" * 90)
print("  (shipped default is Qphi = Qq = 0: articulation is logged, never damped)")
print(f"  {'Qphi':>6s} {'Qq':>6s} {'rms e_y':>9s} {'peak e_y':>9s} "
      f"{'peak phi':>9s} {'peak q':>8s} {'LTR':>6s} {'RWA':>6s} "
      f"{'offtrack':>9s} {'swept':>7s} {'ok':>5s}")
wrows = []
for Qphi, Qq in [(0.0, 0.0), (5.0, 0.0), (25.0, 0.0), (25.0, 8.0),
                 (100.0, 25.0), (400.0, 100.0)]:
    cfg = scen.config()
    cfg.T, cfg.N = best["T"], best["N"]
    cfg.plantSubsteps = max(2, int(round(cfg.T / 0.02)))
    cfg.mpc.Qphi, cfg.mpc.Qq = Qphi, Qq
    cfg.finalize()
    res = rollout(path, cfg, scen, max_time=1.3 * path.s[-1] / scen.V + 2.0,
                  mpc=CondensedMPC(cfg, use_rate_constraint=True))
    m = res.metrics
    r = {"Qphi": Qphi, "Qq": Qq, "rms_ey": m.get("rms_lateral_error", np.nan),
         "peak_ey": m.get("peak_lateral_error", np.nan),
         "phi": m.get("peak_articulation_deg", np.nan),
         "q": m.get("peak_artic_rate_deg_s", np.nan),
         "ltr": m.get("peak_LTR", np.nan), "rwa": m.get("RWA", np.nan),
         "offtrack": m.get("dynamic_offtracking", np.nan),
         "swept": m.get("swept_width", np.nan), "ok": res.ok}
    wrows.append(r)
    print(f"  {Qphi:6.1f} {Qq:6.1f} {r['rms_ey']:9.4f} {r['peak_ey']:9.4f} "
          f"{r['phi']:9.2f} {r['q']:8.2f} {r['ltr']:6.3f} {r['rwa']:6.3f} "
          f"{r['offtrack']:9.4f} {r['swept']:7.3f} "
          f"{('ok' if r['ok'] else 'fail'):>5s}")

# =============================================================================
print("\n" + "=" * 90)
print("C.  Does the new steering-RATE constraint matter?")
print("=" * 90)
print("  (the plant rate-limits the ACTUAL angle; the command is a reference to")
print("   a lagged actuator, so only the actual rate is physically bounded)")
print(f"  {'rate con':>9s} {'rms e_y':>9s} {'peak e_y':>9s} {'cmd rate':>9s} "
      f"{'act rate':>9s} {'limit':>7s} {'LTR':>6s} {'ok':>5s}")
for use_rate in (False, True):
    cfg = scen.config()
    cfg.T, cfg.N = best["T"], best["N"]
    cfg.plantSubsteps = max(2, int(round(cfg.T / 0.02)))
    cfg.finalize()
    res = rollout(path, cfg, scen, max_time=1.3 * path.s[-1] / scen.V + 2.0,
                  mpc=CondensedMPC(cfg, use_rate_constraint=use_rate))
    m = res.metrics
    print(f"  {str(use_rate):>9s} {m.get('rms_lateral_error', np.nan):9.4f} "
          f"{m.get('peak_lateral_error', np.nan):9.4f} "
          f"{m.get('peak_cmd_rate_deg_s', np.nan):9.2f} "
          f"{m.get('peak_actual_steer_rate_deg_s', np.nan):9.2f} "
          f"{math.degrees(cfg.deltaRateMax):7.2f} "
          f"{m.get('peak_LTR', np.nan):6.3f} "
          f"{('ok' if res.ok else 'fail'):>5s}")

json.dump({"horizon": rows, "weights": wrows,
           "chosen": {"T": best["T"], "N": best["N"]}},
          open("out/mpc_tuning.json", "w"), indent=2, default=str)
print("\n  wrote out/mpc_tuning.json")
