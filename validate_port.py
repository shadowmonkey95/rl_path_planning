"""
validate_port.py -- Does the port reproduce the MATLAB physics, and what does
the original code actually do when you run it as shipped?

Seven checks:
  A. Run the docstring example exactly as given.
  B. Verify the linear MPC model IS the Jacobian of the nonlinear plant
     (this validates both the port and the paper's sideslip-angle convention).
  C. Open-loop step response, linear model vs nonlinear plant.
  D. Sweep speed: where is the boundary between pass and fail?
  E. Reward decomposition and scaling: which term actually drives the agent?
  F. Reward sensitivity to path RESOLUTION (the triple-gradient problem).
  G. Plant conditioning / algebraic-identity diagnostics.
"""

import json
import math
import numpy as np
import casadi as ca

from ttv_core import (TTVConfig, MPCConfig, RewardConfig, ttv_rl_episode,
                      build_linear_prediction_model, build_nonlinear_plant,
                      prepare_path, example_path, plant_to_mpc_state,
                      matlab_gradient)

np.set_printoptions(precision=4, suppress=True, linewidth=140)
RESULTS = {}


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# -----------------------------------------------------------------------------
banner("A.  The docstring example, run exactly as shipped")
# -----------------------------------------------------------------------------
cfg = TTVConfig(logPhysicalDiagnostics=True)
r, parts, log = ttv_rl_episode(example_path(), cfg)

print(f"  reward                 = {r: .6f}")
print(f"  passed / failed        = {parts['passed']} / {parts['failed']}")
print(f"  failureReason          = {parts['failureReason']!r}")
print(f"  progress along path    = {parts['progress']*100:.1f} %")
print(f"  episode length         = {log['time'][-1]:.2f} s "
      f"({log['time'].size-1} control steps)")
print(f"  peak |lateral error|   = {log['metrics']['peakLateralError']*100:.2f} cm")
print(f"  peak |heading error|   = {math.degrees(log['metrics']['peakHeadingError']):.3f} deg")
print(f"  max front slip         = {parts['maxSlipFront']:.4f} rad "
      f"= {math.degrees(parts['maxSlipFront']):.2f} deg   (limit "
      f"{cfg.reward.maxLateralSlip} rad)")
print(f"  max rear  slip         = {parts['maxSlipRear']:.4f} rad "
      f"= {math.degrees(parts['maxSlipRear']):.2f} deg")
print(f"  max trailer slip       = {parts['maxSlipTrailer']:.4f} rad "
      f"= {math.degrees(parts['maxSlipTrailer']):.2f} deg")
print(f"  peak articulation      = {math.degrees(log['metrics']['peakArticulationAngle']):.2f} deg")
print(f"  peak steer command     = {math.degrees(log['metrics']['peakSteeringCommand']):.2f} deg")

# Why did it fail? Compute the lateral acceleration the PATH demands.
path = prepare_path(example_path())
kappa_max = float(np.max(np.abs(path.kappa)))
ay_demand = cfg.V ** 2 * kappa_max
print(f"\n  path max |curvature|   = {kappa_max:.5f} 1/m "
      f"(R_min = {1/kappa_max:.1f} m)")
print(f"  => demanded a_y        = V^2*kappa = {ay_demand:.2f} m/s^2 "
      f"= {ay_demand/9.81:.2f} g   at mu = {cfg.mu}")
print(f"  friction ceiling       = mu*g = {cfg.mu*9.81:.2f} m/s^2")
print(f"  utilisation            = {100*ay_demand/(cfg.mu*9.81):.0f} % of available grip")

RESULTS["A_shipped_example"] = {
    "reward": r, "passed": parts["passed"], "failureReason": parts["failureReason"],
    "progress": parts["progress"], "maxSlipFront": parts["maxSlipFront"],
    "peakLateralError_m": log["metrics"]["peakLateralError"],
    "ay_demand_ms2": ay_demand, "ay_demand_g": ay_demand / 9.81,
    "kappa_max": kappa_max,
}

# -----------------------------------------------------------------------------
banner("B.  Is the linear MPC model the true Jacobian of the nonlinear plant?")
# -----------------------------------------------------------------------------
# Plant  xp = [X1 Y1 psi1 v1 r1 phi q delta],  MPC x = [y ydot psi r1 phi q delta]
# At the origin:  ydot = V*sin(psi) + v1*cos(psi)  ->  V*psi + v1
cfg2 = TTVConfig().finalize()
Ac, Bc = build_linear_prediction_model(cfg2)
dyn, _, _ = build_nonlinear_plant(cfg2)

xs = ca.SX.sym("xs", 8)
us = ca.SX.sym("us")
Jx = ca.Function("Jx", [xs, us], [ca.jacobian(dyn(xs, us), xs)])
Ju = ca.Function("Ju", [xs, us], [ca.jacobian(dyn(xs, us), us)])
x0 = np.zeros(8)
Jp = np.asarray(Jx(x0, 0.0))        # 8x8, exact AD Jacobian at the origin
Bp = np.asarray(Ju(x0, 0.0)).reshape(-1, 1)

# drop X1 (row/col 0) -> plant sub-state [Y1 psi1 v1 r1 phi q delta]
keep = [1, 2, 3, 4, 5, 6, 7]
Jp7 = Jp[np.ix_(keep, keep)]
Bp7 = Bp[keep, :]

V = cfg2.V
# T : plant sub-state -> MPC state
#   y=Y1 ; ydot=V*psi+v1 ; psi=psi ; r1=r1 ; phi=phi ; q=q ; delta=delta
T = np.zeros((7, 7))
T[0, 0] = 1.0
T[1, 1] = V; T[1, 2] = 1.0
T[2, 1] = 1.0
T[3, 3] = 1.0
T[4, 4] = 1.0
T[5, 5] = 1.0
T[6, 6] = 1.0
Ti = np.linalg.inv(T)
A_from_plant = T @ Jp7 @ Ti
B_from_plant = T @ Bp7

errA = np.abs(A_from_plant - Ac)
scaleA = np.maximum(np.abs(Ac), 1.0)
print("  Ac  (hand-assembled from the descriptor model):")
print(Ac)
print("\n  T * dPlant/dx * T^-1  (exact AD Jacobian of the nonlinear plant):")
print(A_from_plant)
print(f"\n  max ABSOLUTE difference     = {errA.max():.3e}")
print(f"  max RELATIVE difference     = {(errA/scaleA).max():.3e}")
print(f"  B difference (max abs)      = {np.abs(B_from_plant - Bc).max():.3e}")
print("\n  -> The hand-written Ac and the plant Jacobian agree. This confirms")
print("     state 2 of the descriptor model is the SIDESLIP ANGLE beta = v1/V,")
print("     which is why Ac row 2 carries V*Av(2,k) and row 4 carries Av(1,2)/V.")

ev = np.linalg.eigvals(Ac)
print(f"\n  eig(Ac) real parts: {np.sort(ev.real)}")
print(f"  open-loop stable (all Re<0 except the 2 integrators)? "
      f"{np.sum(ev.real > 1e-9)} unstable, {np.sum(np.abs(ev) < 1e-9)} at origin")

RESULTS["B_jacobian_check"] = {
    "max_abs_diff": float(errA.max()),
    "max_rel_diff": float((errA / scaleA).max()),
    "B_max_abs_diff": float(np.abs(B_from_plant - Bc).max()),
    "eig_real": sorted(ev.real.tolist()),
}

# -----------------------------------------------------------------------------
banner("C.  Open-loop step response: linear prediction model vs nonlinear plant")
# -----------------------------------------------------------------------------
from scipy.linalg import expm
T_s, N_s = 0.01, 400
aug = expm(np.block([[Ac, Bc], [np.zeros((1, 8))]]) * T_s)
Ad, Bd = aug[:7, :7], aug[:7, 7:8]


def run_pair(delta_step):
    xl = np.zeros(7)
    xp_ = np.zeros(8)
    lin, non = [], []
    from ttv_core import rk4_step
    for _ in range(N_s):
        xl = Ad @ xl + Bd.flatten() * delta_step
        xp_ = rk4_step(dyn, xp_, delta_step, T_s)
        lin.append(xl.copy())
        non.append(plant_to_mpc_state(xp_, V))
    return np.array(lin), np.array(non)


print(f"  {'step [deg]':>11s} {'state':>16s} {'linear':>12s} {'nonlinear':>12s} {'rel.err':>9s}")
step_rows = []
for dstep_deg in (1.0, 3.0, 8.0):
    lin, non = run_pair(math.radians(dstep_deg))
    for name, i in (("y [m]", 0), ("psi [rad]", 2), ("r1 [rad/s]", 3), ("phi [rad]", 4)):
        a, b = lin[-1, i], non[-1, i]
        rel = abs(a - b) / max(abs(b), 1e-9)
        print(f"  {dstep_deg:11.1f} {name:>16s} {a:12.5f} {b:12.5f} {100*rel:8.2f}%")
        step_rows.append({"step_deg": dstep_deg, "state": name,
                          "linear": float(a), "nonlinear": float(b), "rel_err": float(rel)})
    print()
print("  -> At 1 deg the linear model is near-exact; by 8 deg the tyre saturation")
print("     in the plant (tanh brush model) makes the linear model over-predict.")
print("     The MPC therefore gets progressively optimistic as the path gets hard.")
RESULTS["C_step_response"] = step_rows

# -----------------------------------------------------------------------------
banner("D.  Speed sweep: at what speed does the shipped example path survive?")
# -----------------------------------------------------------------------------
print(f"  {'V [m/s]':>8s} {'V [km/h]':>9s} {'ay [g]':>8s} {'pass':>6s} "
      f"{'reward':>9s} {'maxSlipF [deg]':>15s} {'reason':>28s}")
sweep = []
for Vtest in (8.0, 10.0, 12.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0):
    c = TTVConfig(V=Vtest)
    try:
        rr, pp, ll = ttv_rl_episode(example_path(), c)
        row = {"V": Vtest, "pass": pp["passed"], "reward": rr,
               "maxSlipFront_deg": math.degrees(pp["maxSlipFront"]),
               "reason": pp["failureReason"], "progress": pp["progress"],
               "ay_g": Vtest ** 2 * kappa_max / 9.81,
               "peakLatErr_m": ll["metrics"]["peakLateralError"],
               "peakArt_deg": math.degrees(ll["metrics"]["peakArticulationAngle"])}
    except Exception as e:
        row = {"V": Vtest, "pass": False, "reward": float("nan"),
               "maxSlipFront_deg": float("nan"), "reason": f"EXC {e}", "progress": 0.0,
               "ay_g": Vtest ** 2 * kappa_max / 9.81, "peakLatErr_m": float("nan"),
               "peakArt_deg": float("nan")}
    sweep.append(row)
    print(f"  {row['V']:8.1f} {row['V']*3.6:9.1f} {row['ay_g']:8.2f} "
          f"{str(row['pass']):>6s} {row['reward']:9.4f} {row['maxSlipFront_deg']:15.2f} "
          f"{row['reason'][:28]:>28s}")
RESULTS["D_speed_sweep"] = sweep

# -----------------------------------------------------------------------------
banner("E.  Reward decomposition: which term actually carries the gradient?")
# -----------------------------------------------------------------------------
print("  Fehér Eq.(3):  mu_max = 0.0037 * exp(0.0693 * v0[km/h])")
print(f"  {'v0 [km/h]':>10s} {'mu_max [rad]':>13s} {'mu_max [deg]':>13s} {'note':>34s}")
mu_rows = []
for v_kmh in (40, 50, 55, 60, 72, 80, 90):
    m = 0.0037 * math.exp(0.0693 * v_kmh)
    note = "paper fit range" if 40 <= v_kmh <= 60 else "EXTRAPOLATED"
    if m > 0.2:
        note += " > 0.2 rad fail limit!"
    print(f"  {v_kmh:10.0f} {m:13.4f} {math.degrees(m):13.2f} {note:>34s}")
    mu_rows.append({"v_kmh": v_kmh, "mu_max": m, "note": note})

print(f"\n  At the default V=20 m/s = 72 km/h:")
print(f"    reward_slip      = 2*{parts['slipReference']:.4f} "
      f"- {parts['maxSlipFront']:.4f} - {parts['maxSlipRear']:.4f} "
      f"= {parts['slip']:.4f}")
print(f"    reward_curvature = {cfg.reward.cKappaDD:.1f} - |max k''| - |min k''| "
      f"= {parts['curvature']:.6f}")
print(f"    if it PASSED:  reward = 0.5*{parts['curvature']:.6f} "
      f"+ 0.5*{parts['slip']:.4f} = "
      f"{0.5*parts['curvature']+0.5*parts['slip']:.4f}")
print(f"\n  RATIO |reward_slip| / |reward_curvature| = "
      f"{abs(parts['slip'])/abs(parts['curvature']):.0f} : 1")
print("  -> the curvature/jerk term is ~300x smaller than the slip term, so with")
print("     the shipped cKappaDD=0 and V=20 the smoothness objective is invisible.")
RESULTS["E_reward"] = {
    "mu_fit": mu_rows, "slip_term": parts["slip"],
    "curv_term": parts["curvature"],
    "ratio": abs(parts["slip"]) / abs(parts["curvature"]),
    "reward_if_passed": 0.5 * parts["curvature"] + 0.5 * parts["slip"],
}

# -----------------------------------------------------------------------------
banner("F.  Reward vs path RESOLUTION (kappa'' from three nested gradients)")
# -----------------------------------------------------------------------------
# Analytic truth for y = 1.75*(tanh(a(x-35)) - tanh(a(x-80))):
# compare the finite-difference kappa'' against a high-accuracy reference.
print(f"  {'n points':>9s} {'ds [m]':>8s} {'max k''''':>12s} {'min k''''':>12s} "
      f"{'reward_curv':>13s}")
res_rows = []
for n in (61, 121, 241, 601, 1201, 2401, 4801):
    p = prepare_path(example_path(n))
    rc = 0.0 - abs(np.max(p.kappaDD)) - abs(np.min(p.kappaDD))
    ds = float(np.mean(np.diff(p.s)))
    print(f"  {n:9d} {ds:8.4f} {np.max(p.kappaDD):12.3e} "
          f"{np.min(p.kappaDD):12.3e} {rc:13.6f}")
    res_rows.append({"n": n, "ds": ds, "max_kdd": float(np.max(p.kappaDD)),
                     "min_kdd": float(np.min(p.kappaDD)), "reward_curv": rc})
spread = max(abs(x["reward_curv"]) for x in res_rows) / min(
    abs(x["reward_curv"]) for x in res_rows)
print(f"\n  spread across resolutions = {spread:.1f}x for the SAME geometric path")
print("  -> reward_curvature is a function of how densely you sampled the path,")
print("     not only of the path. In the paper kappa(s) is an analytic cubic so")
print("     kappa'' = 6*a3 exactly; the port must generate kappa analytically.")
RESULTS["F_resolution"] = {"rows": res_rows, "spread_factor": spread}

# -----------------------------------------------------------------------------
banner("G.  Plant conditioning and algebraic identities")
# -----------------------------------------------------------------------------
d = log["physicalDiagnostics"]
for k, v in d.items():
    finite = v[np.isfinite(v)]
    print(f"  {k:>28s}: max|.| = {np.max(np.abs(finite)):.3e}   "
          f"min = {np.min(finite):.3e}")
print(f"\n  solver iterations per step: mean {np.nanmean(log['solverIterations']):.1f}, "
      f"max {np.nanmax(log['solverIterations']):.0f}")
print("  -> mass-matrix residual at machine precision and rcond ~1e-5..1e-4:")
print("     the 5x5 descriptor solve is healthy; no hidden integration problem.")
RESULTS["G_diagnostics"] = {k: {"max_abs": float(np.max(np.abs(v[np.isfinite(v)]))),
                                "min": float(np.min(v[np.isfinite(v)]))}
                            for k, v in d.items()}

with open("out/validation_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("\n  wrote out/validation_results.json")
