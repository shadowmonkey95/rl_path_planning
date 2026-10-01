"""
ttv_dlc_tracking.py -- Port of RUN_TTV_DLC_TRACKING_TWO_MODELS.m
================================================================

Two models, no RL:

    Model 1   linear 3-DOF tractor-trailer model, inside the MPC
    Model 2   nonlinear 3-DOF tractor-trailer plant, as the simulated vehicle

A double-lane-change reference is generated internally and tracked. There is no
path input, no reward, no failure test. The point is to look at tracking,
articulation, and the gap between the prediction model and the plant before any
of it is connected to RL.

Relationship to the rest of this package
----------------------------------------
This shares its vehicle, its linear model, its nonlinear plant and its MPC
formulation with `ttv_core.py` -- they are the same equations, and this file
imports them rather than restating them. Exactly three things are new:

1. THE REFERENCE IS A FUNCTION OF TIME, not of arc length. `Yref(t)` is
   evaluated directly at `t + k*T`. That removes `prepare_path`, the
   nearest-point search and the whole arc-length machinery, and with them the
   two findings that came from it (the resolution-dependent reward and the
   non-monotonic progress match). For a fixed-speed study this is the simpler
   and better-posed choice.

2. THE ONE-STEP PREDICTION ERROR is logged: `x_measured(k+1) - (Ad*x(k) +
   Bd*u(k))`. This is the direct measurement of Model 1 vs Model 2 mismatch,
   per state, per step. `ttv_rl_episode.m` has no equivalent, and it is the
   most useful diagnostic in this script.

3. SOLVER TIME per step.

Everything else -- parameters, `Ac`/`Bc` assembly, the 5x5 descriptor plant,
`plant_to_mpc_state`, RK4, the multiple-shooting NLP -- is identical to the
MATLAB and to `ttv_core`.

Differences from `ttv_rl_episode.m`'s defaults, taken from the new script:
    N       = 20   (was 10)
    Qpsi    = 0.0  (was 1.0)
    max_iter = 500 (was 300),  acceptable_tol = 1e-8 (was 1e-7)
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ttv_core import (TTVConfig, MPCConfig, build_environment, make_cache_key,
                      plant_to_mpc_state)
from plant_numpy import PlantNP

G = 9.81


# =============================================================================
# 1.  The internal double-lane-change reference  (<-> section 4 of the MATLAB)
# =============================================================================

@dataclass
class DLCReference:
    """Yref(t) = A*(tanh(rate*t - t1) - tanh(rate*t - t2)).

    Defaults reproduce the MATLAB exactly:
        Yref  = 1.75*(tanh(1.5t - 6) - tanh(1.5t - 16.5))
        dYref = 2.625*(sech^2(1.5t - 6) - sech^2(1.5t - 16.5))
        Psiref = atan2(dYref, V)

    Unlike the MATLAB, the second and third derivatives are also provided in
    closed form, because they are what tell you whether the reference is
    physically reasonable before you run anything.
    """
    amplitude: float = 1.75
    rate: float = 1.5
    t1: float = 6.0
    t2: float = 16.5
    V: float = 20.0

    # -- value and derivatives w.r.t. TIME -----------------------------------
    def y(self, t):
        t = np.asarray(t, dtype=float)
        return self.amplitude * (np.tanh(self.rate * t - self.t1)
                                 - np.tanh(self.rate * t - self.t2))

    def dy(self, t):
        t = np.asarray(t, dtype=float)
        u1 = self.rate * t - self.t1
        u2 = self.rate * t - self.t2
        return self.amplitude * self.rate * (np.cosh(u1) ** -2 - np.cosh(u2) ** -2)

    def ddy(self, t):
        t = np.asarray(t, dtype=float)
        u1 = self.rate * t - self.t1
        u2 = self.rate * t - self.t2
        return self.amplitude * self.rate ** 2 * (
            -2.0 * np.tanh(u1) * np.cosh(u1) ** -2
            + 2.0 * np.tanh(u2) * np.cosh(u2) ** -2)

    def psi(self, t):
        """Heading reference, from dy/dx = (dy/dt)/V."""
        return np.arctan2(self.dy(t), self.V)

    # -- what the reference asks of the vehicle ------------------------------
    def peak_lateral_accel(self, n: int = 200001, t_end: float = 20.0) -> float:
        t = np.linspace(0.0, t_end, n)
        return float(np.max(np.abs(self.ddy(t))))

    def peak_curvature(self, **kw) -> float:
        return self.peak_lateral_accel(**kw) / self.V ** 2

    def describe(self, t_end: float = 20.0) -> Dict[str, float]:
        ay = self.peak_lateral_accel(t_end=t_end)
        t = np.linspace(0.0, t_end, 20001)
        return {
            "peak_y": float(np.max(self.y(t))),
            "peak_slope_deg": float(np.degrees(np.arctan(np.max(np.abs(self.dy(t))) / self.V))),
            "peak_lateral_accel": ay,
            "peak_lateral_accel_g": ay / G,
            "peak_curvature": ay / self.V ** 2,
            "min_radius": self.V ** 2 / ay,
            "transition_width_s": 1.0 / self.rate,
            "transition_width_m": self.V / self.rate,
            "road_length_m": self.V * t_end,
        }


# =============================================================================
# 2.  Configuration  (<-> sections 2 and 3 of the MATLAB)
# =============================================================================

def dlc_config(**overrides) -> TTVConfig:
    """The MATLAB script's settings, as a TTVConfig.

    The vehicle is identical to `ttv_core`'s defaults, so only the MPC
    settings actually differ.
    """
    cfg = TTVConfig(
        T=0.1, N=20, V=20.0,
        maxEpisodeTime=20.0, plantSubsteps=5,
        deltaMax=0.5, deltaRateMax=0.6, steeringTimeConstant=0.15,
        mpc=MPCConfig(Qy=2.0, Qpsi=0.0, Qphi=0.0, Qq=0.0, Rdelta=0.1,
                      maxIterations=500, acceptableTolerance=1e-8),
    )
    for k, v in overrides.items():
        if k in ("Qy", "Qpsi", "Qphi", "Qq", "Rdelta",
                 "maxIterations", "acceptableTolerance"):
            setattr(cfg.mpc, k, v)
        else:
            setattr(cfg, k, v)
    return cfg.finalize()


# =============================================================================
# 2b.  Optional failure limits  (from ttv_rl_episode.m; OFF by default)
# =============================================================================

@dataclass
class FailureLimits:
    """Episode-termination tests, off by default so the MATLAB behaviour is
    the default.

    The MATLAB script runs all 200 steps whatever happens, which is fine for a
    tracking study but hides divergence: the speed sweep completes 200/200
    steps at V = 30 m/s while the rig jack-knifes to 152 deg. Switching these
    on makes a bad run stop and say why, which is what you need once a reward
    is attached.

    The first four reproduce `ttv_rl_episode.m`. `maxArticulation` is new, as a
    jack-knife guard.

    Measured: on every diverging run in the speed sweep (8, 10, 12, 30, 35 m/s)
    it is the TYRE-SLIP test that fires first, not the articulation test. The
    jack-knife develops *after* the tyres saturate, not before it -- the
    articulation limit never actually triggers here. Worth knowing, because it
    means the slip test alone is enough to catch these, and phi is the symptom
    rather than the cause.
    """
    enabled: bool = False
    maxLateralSlip: float = 0.2                  # [rad] front/rear, as MATLAB
    maxLateralError: float = 1.0                 # [m]
    maxHeadingError: float = math.radians(20.0)  # [rad]
    maxArticulation: float = math.radians(45.0)  # [rad]  jack-knife guard
    checkFinite: bool = True

    def check(self, xp: np.ndarray, alpha: np.ndarray, e_lat: float,
              e_psi: float) -> str:
        """Return a failure reason, or '' if the state is acceptable."""
        if not self.enabled:
            return ""
        if self.checkFinite and not np.all(np.isfinite(xp)):
            return "non-finite plant state"
        if np.max(np.abs(alpha[0:2])) > self.maxLateralSlip:
            return "tractor lateral-slip limit"
        if abs(e_lat) > self.maxLateralError:
            return "lateral-error limit"
        if abs(e_psi) > self.maxHeadingError:
            return "heading-error limit"
        if abs(xp[5]) > self.maxArticulation:
            return "articulation (jack-knife) limit"
        return ""


# =============================================================================
# 3.  Result container
# =============================================================================

@dataclass
class TrackingResult:
    t_state: np.ndarray
    t_control: np.ndarray
    plant: np.ndarray              # (8, n)  [X1 Y1 psi1 v1 r1 phi q delta]
    mpc_state: np.ndarray          # (7, n)  [Y Ydot psi1 r1 phi q delta]
    delta_cmd: np.ndarray          # (n-1,)
    y_ref: np.ndarray
    psi_ref: np.ndarray
    slip: np.ndarray               # (3, n)
    force: np.ndarray              # (3, n)
    pred_err: np.ndarray           # (7, n-1)  plant minus linear one-step
    solve_time: np.ndarray         # (n-1,)  seconds
    completed_steps: int
    n_steps: int
    ok: bool
    reason: str
    cfg: TTVConfig
    ref: DLCReference
    metrics: Dict[str, float] = field(default_factory=dict)

    # convenience views
    @property
    def r1(self):
        return self.plant[4, :]

    @property
    def q(self):
        return self.plant[6, :]

    @property
    def r2(self):
        return self.plant[4, :] + self.plant[6, :]

    @property
    def beta1(self):
        return np.arctan2(self.plant[3, :], self.cfg.V)

    @property
    def tracking_error(self):
        return self.plant[1, :] - self.y_ref


# =============================================================================
# 4.  The closed loop  (<-> sections 8 and 9 of the MATLAB)
# =============================================================================

def run_dlc_tracking(cfg: Optional[TTVConfig] = None,
                     ref: Optional[DLCReference] = None,
                     sim_time: float = 20.0,
                     backend: str = "ipopt",
                     limits: Optional[FailureLimits] = None,
                     verbose: bool = False) -> TrackingResult:
    """Track the internal DLC reference with the linear MPC and the nonlinear plant.

    backend = "ipopt"  the shipped CasADi multiple-shooting NLP, as in MATLAB
    backend = "fast"   the condensed QP from fast_mpc.py, configured to solve
                       the identical problem (no move blocking, no rate
                       constraint, no soft constraints)
    limits             optional episode-termination tests; None or
                       FailureLimits(enabled=False) reproduces the MATLAB,
                       which runs every step regardless
    """
    cfg = (cfg or dlc_config())
    cfg.finalize()
    ref = ref or DLCReference(V=cfg.V)
    limits = limits or FailureLimits()

    nx, N, T, V = 7, int(cfg.N), float(cfg.T), float(cfg.V)
    n_steps = int(round(sim_time / T))
    t_grid = np.arange(n_steps + 1) * T

    plant = PlantNP.from_cfg(cfg)          # validated identical to CasADi

    if backend == "ipopt":
        env = build_environment(cfg, make_cache_key(cfg))
        solver = env["solver"]
        Ad, Bd = env["Ad"], env["Bd"]
        Xguess = None
        Uguess = None
    elif backend == "fast":
        from fast_mpc import CondensedMPC
        mpc = CondensedMPC(cfg, use_rate_constraint=False,
                           soft_state_constraints=False, move_blocking=False)
        Ad, Bd = mpc.Ad, mpc.Bd
    else:
        raise ValueError("backend must be 'ipopt' or 'fast'")

    # ---- section 8: initial conditions and histories ----------------------
    xp = np.zeros(8)
    xm = plant_to_mpc_state(xp, V)

    plant_hist = np.zeros((8, n_steps + 1))
    mpc_hist = np.zeros((nx, n_steps + 1))
    u_hist = np.zeros(n_steps)
    yref_hist = np.zeros(n_steps + 1)
    psiref_hist = np.zeros(n_steps + 1)
    slip_hist = np.zeros((3, n_steps + 1))
    force_hist = np.zeros((3, n_steps + 1))
    pred_err_hist = np.zeros((nx, n_steps))
    solve_t_hist = np.zeros(n_steps)

    plant_hist[:, 0] = xp
    mpc_hist[:, 0] = xm
    yref_hist[0] = float(ref.y(0.0))
    psiref_hist[0] = float(ref.psi(0.0))
    a0, f0 = _plant_outputs(plant, xp)
    slip_hist[:, 0] = a0
    force_hist[:, 0] = f0

    if backend == "ipopt":
        Xguess = np.tile(xm.reshape(-1, 1), (1, N + 1))
        Uguess = np.zeros((1, N))

    completed = n_steps
    ok, reason = True, ""

    # ---- section 9: the loop ----------------------------------------------
    for it in range(1, n_steps + 1):
        t_now = (it - 1) * T
        xm = plant_to_mpc_state(xp, V)

        t_prev = t_now + T * np.arange(1, N + 1)
        y_ref_h = np.asarray(ref.y(t_prev), dtype=float)
        psi_ref_h = np.asarray(ref.psi(t_prev), dtype=float)

        t0 = _time.perf_counter()
        if backend == "ipopt":
            p = np.zeros(nx + 2 * N)
            p[:nx] = xm
            p[nx + 0::2] = y_ref_h
            p[nx + 1::2] = psi_ref_h
            w0 = np.concatenate((Xguess.flatten(order="F"),
                                 Uguess.flatten(order="F")))
            try:
                sol = solver(x0=w0, lbx=env["lbx"], ubx=env["ubx"],
                             lbg=env["lbg"], ubg=env["ubg"], p=p)
                st = solver.stats()
                good = bool(st.get("success", False))
                status = str(st.get("return_status", "?"))
            except Exception as exc:                       # pragma: no cover
                good, status = False, str(exc)
            if not good:
                ok, reason, completed = False, f"MPC solver: {status}", it - 1
                if verbose:
                    print(f"  MPC solver failed at step {it}: {status}")
                break
            w = np.asarray(sol["x"]).flatten()
            Xopt = w[:nx * (N + 1)].reshape((nx, N + 1), order="F")
            Uopt = w[nx * (N + 1):].reshape((1, N), order="F")
            delta_cmd = float(Uopt[0, 0])
        else:
            u_seq, dbg = mpc.solve(xm, y_ref_h, psi_ref_h)
            if not dbg.solved:
                ok, reason, completed = False, f"MPC solver: {dbg.status}", it - 1
                break
            delta_cmd = float(u_seq[0])
        solve_t_hist[it - 1] = _time.perf_counter() - t0

        # one-step linear prediction, kept for the model-mismatch plot
        one_step = Ad @ xm + Bd.flatten() * delta_cmd

        # apply to the nonlinear plant
        xp = plant.step(xp, delta_cmd, T, cfg.plantSubsteps)

        xm_next = plant_to_mpc_state(xp, V)
        alpha, Fy = _plant_outputs(plant, xp)

        plant_hist[:, it] = xp
        mpc_hist[:, it] = xm_next
        u_hist[it - 1] = delta_cmd
        yref_hist[it] = float(ref.y(t_now + T))
        psiref_hist[it] = float(ref.psi(t_now + T))
        slip_hist[:, it] = alpha
        force_hist[:, it] = Fy
        pred_err_hist[:, it - 1] = xm_next - one_step

        # ---- optional failure tests (off by default) ----------------------
        why = limits.check(xp, alpha,
                           e_lat=xp[1] - yref_hist[it],
                           e_psi=float(np.arctan2(
                               math.sin(xp[2] - psiref_hist[it]),
                               math.cos(xp[2] - psiref_hist[it]))))
        if why:
            ok, reason, completed = False, why, it
            if verbose:
                print(f"  stopped at step {it} (t = {it*T:.2f} s): {why}")
            break

        if backend == "ipopt":
            Xguess = np.hstack((Xopt[:, 1:], Xopt[:, -1:]))
            Xguess[:, 0] = xm_next
            Uguess = np.hstack((Uopt[:, 1:], Uopt[:, -1:]))

    # ---- section 10: trim --------------------------------------------------
    s = slice(0, completed + 1)
    c = slice(0, completed)
    res = TrackingResult(
        t_state=t_grid[s], t_control=t_grid[c],
        plant=plant_hist[:, s], mpc_state=mpc_hist[:, s], delta_cmd=u_hist[c],
        y_ref=yref_hist[s], psi_ref=psiref_hist[s],
        slip=slip_hist[:, s], force=force_hist[:, s],
        pred_err=pred_err_hist[:, c], solve_time=solve_t_hist[c],
        completed_steps=completed, n_steps=n_steps, ok=ok, reason=reason,
        cfg=cfg, ref=ref)
    res.metrics = make_metrics(res)
    return res


def _plant_outputs(plant: PlantNP, xp: np.ndarray):
    (a1, a2, a3), (F1, F2, F3), _ = plant.slip_and_forces(xp)
    return np.array([a1, a2, a3]), np.array([F1, F2, F3])


# =============================================================================
# 5.  Summary  (<-> section 11 of the MATLAB)
# =============================================================================

def make_metrics(r: TrackingResult) -> Dict[str, float]:
    e = r.tracking_error
    n = r.delta_cmd.size
    m = {
        "completed_steps": r.completed_steps,
        "n_steps": r.n_steps,
        "rms_lateral_error": float(np.sqrt(np.mean(e ** 2))),
        "peak_lateral_error": float(np.max(np.abs(e))),
        "peak_articulation_angle": float(np.max(np.abs(r.plant[5, :]))),
        "peak_articulation_rate": float(np.max(np.abs(r.q))),
        "peak_steering_command": float(np.max(np.abs(r.delta_cmd))) if n else float("nan"),
        "peak_actual_steering": float(np.max(np.abs(r.plant[7, :]))),
        "max_front_slip": float(np.max(np.abs(r.slip[0, :]))),
        "max_rear_slip": float(np.max(np.abs(r.slip[1, :]))),
        "max_trailer_slip": float(np.max(np.abs(r.slip[2, :]))),
        "mean_solve_time": float(np.mean(r.solve_time)) if n else float("nan"),
        "max_solve_time": float(np.max(r.solve_time)) if n else float("nan"),
    }
    # the diagnostic the MATLAB adds and ttv_rl_episode does not have
    if n:
        m["pred_err_rms"] = float(np.sqrt(np.mean(r.pred_err ** 2)))
        for i, lab in enumerate(("Y", "Ydot", "psi1", "r1", "phi", "q", "delta")):
            m[f"pred_err_max_{lab}"] = float(np.max(np.abs(r.pred_err[i, :])))
    return m


def print_summary(r: TrackingResult) -> None:
    m = r.metrics
    print("\n===== TTV DOUBLE-LANE-CHANGE TRACKING =====")
    print(f"Completed MPC steps      : {m['completed_steps']} / {m['n_steps']}")
    if not r.ok:
        print(f"Stopped because          : {r.reason}")
    print(f"RMS lateral error        : {m['rms_lateral_error']:.6f} m")
    print(f"Peak lateral error       : {m['peak_lateral_error']:.6f} m")
    print(f"Peak articulation angle  : {m['peak_articulation_angle']:.6f} rad"
          f"  ({math.degrees(m['peak_articulation_angle']):.3f} deg)")
    print(f"Peak articulation rate   : {m['peak_articulation_rate']:.6f} rad/s")
    print(f"Peak steering command    : {m['peak_steering_command']:.6f} rad"
          f"  ({math.degrees(m['peak_steering_command']):.3f} deg)")
    print(f"Peak actual steering     : {m['peak_actual_steering']:.6f} rad")
    print(f"Max front slip           : {m['max_front_slip']:.6f} rad"
          f"  ({math.degrees(m['max_front_slip']):.3f} deg)")
    print(f"Max rear slip            : {m['max_rear_slip']:.6f} rad")
    print(f"Max trailer slip         : {m['max_trailer_slip']:.6f} rad")
    print(f"Mean MPC solve time      : {m['mean_solve_time']:.6f} s")
    print(f"Maximum MPC solve time   : {m['max_solve_time']:.6f} s")
    if "pred_err_rms" in m:
        print("--- Model 1 vs Model 2, one-step prediction error ---")
        print(f"RMS over all states      : {m['pred_err_rms']:.3e}")
        for lab in ("Y", "Ydot", "psi1", "r1", "phi", "q", "delta"):
            print(f"  max |e_pred({lab:5s})|    : {m[f'pred_err_max_{lab}']:.3e}")


# =============================================================================
# 6.  Plots  (<-> section 12 of the MATLAB)
# =============================================================================

def plot_all(r: TrackingResult, prefix: str = "out/dlc") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg, ts, tc = r.cfg, r.t_state, r.t_control
    P = r.plant

    # ---------- figure 1: tracking ------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(ts, r.y_ref, "b--", lw=1.8, label="Y reference")
    ax[0].plot(ts, P[1, :], "r", lw=1.5, label="nonlinear plant")
    ax[0].set_ylabel("Y (m)"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_title("Double-lane-change lateral tracking")

    ax[1].plot(ts, r.tracking_error * 100, "k", lw=1.5)
    ax[1].set_ylabel(r"$Y-Y_{ref}$ (cm)"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"Lateral tracking error  "
                    f"(RMS {r.metrics['rms_lateral_error']*100:.2f} cm, "
                    f"peak {r.metrics['peak_lateral_error']*100:.2f} cm)")

    ax[2].plot(cfg.V * ts, r.y_ref, "b--", lw=1.8, label="reference")
    ax[2].plot(P[0, :], P[1, :], "r", lw=1.5, label="tractor CG")
    ax[2].set_xlabel("X (m)"); ax[2].set_ylabel("Y (m)")
    ax[2].grid(alpha=.3); ax[2].legend()
    ax[2].set_title("Global trajectory")
    plt.tight_layout(); plt.savefig(f"{prefix}_tracking.png", dpi=130); plt.close()

    # ---------- figure 2: vehicle states ------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(12, 9))
    ax[0, 0].plot(ts, r.beta1, lw=1.4)
    ax[0, 0].set_ylabel(r"$\beta_1$ (rad)"); ax[0, 0].set_title("Tractor sideslip")

    ax[0, 1].plot(ts, r.r1, lw=1.4, label=r"$r_1$")
    ax[0, 1].plot(ts, r.r2, "--", lw=1.4, label=r"$r_2$")
    ax[0, 1].set_ylabel("yaw rate (rad/s)"); ax[0, 1].legend()
    ax[0, 1].set_title("Tractor and trailer yaw rate")

    ax[1, 0].plot(ts, P[5, :], lw=1.4)
    ax[1, 0].set_ylabel(r"$\varphi$ (rad)"); ax[1, 0].set_title("Articulation angle")

    ax[1, 1].plot(ts, r.q, lw=1.4)
    ax[1, 1].set_ylabel(r"$q=\dot\varphi$ (rad/s)"); ax[1, 1].set_title("Articulation rate")

    ax[2, 0].plot(ts, P[2, :], lw=1.4, label=r"$\psi_1$")
    ax[2, 0].plot(ts, r.psi_ref, "--", lw=1.4, label=r"$\psi_{ref}$")
    ax[2, 0].set_xlabel("time (s)"); ax[2, 0].set_ylabel(r"$\psi_1$ (rad)")
    ax[2, 0].legend(); ax[2, 0].set_title("Heading vs reference")

    ax[2, 1].plot(ts, P[3, :], lw=1.4)
    ax[2, 1].set_xlabel("time (s)"); ax[2, 1].set_ylabel(r"$v_1$ (m/s)")
    ax[2, 1].set_title("Tractor lateral velocity")
    for a in ax.ravel():
        a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{prefix}_states.png", dpi=130); plt.close()

    # ---------- figure 3: steering and slip ---------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    if tc.size:
        ax[0].step(tc, r.delta_cmd, "b", where="post", lw=1.4,
                   label=r"$\delta_{cmd}$")
    ax[0].plot(ts, P[7, :], "r", lw=1.4, label=r"$\delta_{actual}$")
    for s_ in (cfg.deltaMax, -cfg.deltaMax):
        ax[0].axhline(s_, color="k", ls="--", lw=1)
    ax[0].set_ylabel("steering (rad)"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[0].set_title("MPC command and nonlinear steering actuator "
                    f"(limit {cfg.deltaMax} rad)")

    for i, lab in enumerate((r"$\alpha_1$ front", r"$\alpha_2$ rear",
                             r"$\alpha_3$ trailer")):
        ax[1].plot(ts, r.slip[i, :], lw=1.4, label=lab)
    for s_ in (0.2, -0.2):
        ax[1].axhline(s_, color="k", ls="--", lw=1)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("slip angle (rad)")
    ax[1].grid(alpha=.3); ax[1].legend()
    ax[1].set_title("Nonlinear-plant tyre slip angles (dashed: 0.2 rad)")
    plt.tight_layout(); plt.savefig(f"{prefix}_steering_slip.png", dpi=130); plt.close()

    # ---------- figure 4: model mismatch and solver time --------------------
    labels = ["Y", r"$\dot Y$", r"$\psi_1$", r"$r_1$", r"$\varphi$", "q", r"$\delta$"]
    units = ["m", "m/s", "rad", "rad/s", "rad", "rad/s", "rad"]
    fig, ax = plt.subplots(4, 2, figsize=(12, 11))
    for i in range(7):
        a = ax.ravel()[i]
        if tc.size:
            a.plot(tc, r.pred_err[i, :], lw=1.3)
        a.set_ylabel(f"$e_{{pred}}$({labels[i]}) [{units[i]}]", fontsize=9)
        a.grid(alpha=.3)
        a.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if i >= 5:
            a.set_xlabel("time (s)")
    a = ax.ravel()[7]
    if tc.size:
        a.plot(tc, 1e3 * r.solve_time, lw=1.3, color="#d97706")
    a.set_xlabel("time (s)"); a.set_ylabel("solve time (ms)")
    a.grid(alpha=.3); a.set_title("MPC solver time", fontsize=9)
    fig.suptitle("One-step error: nonlinear plant minus linear prediction\n"
                 "(Model 2 $-$ Model 1, the quantity this script exists to show)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"{prefix}_model_mismatch.png", dpi=130); plt.close()

    print(f"  wrote {prefix}_tracking.png, {prefix}_states.png,")
    print(f"        {prefix}_steering_slip.png, {prefix}_model_mismatch.png")


# =============================================================================
# 7.  Studies the MATLAB comments point at
# =============================================================================

def weight_study(ref: Optional[DLCReference] = None,
                 sim_time: float = 20.0) -> list:
    """"Qpsi, Qphi and Qq are initially zero ... They can be activated later
    for controlled comparisons." -- this is that comparison."""
    rows = []
    combos = [
        ("shipped: Qy only", dict()),
        ("+ heading", dict(Qpsi=1.0)),
        ("+ articulation angle", dict(Qpsi=1.0, Qphi=25.0)),
        ("+ articulation rate", dict(Qpsi=1.0, Qphi=25.0, Qq=8.0)),
        ("articulation only", dict(Qphi=25.0, Qq=8.0)),
        ("heavy steering penalty", dict(Rdelta=2.0)),
    ]
    for label, kw in combos:
        cfg = dlc_config(**kw)
        r = run_dlc_tracking(cfg, ref, sim_time, backend="fast")
        m = r.metrics
        rows.append({"label": label, **kw, **m})
    return rows


DAMPED_WEIGHTS = dict(Qpsi=1.0, Qphi=25.0, Qq=8.0)


def speed_study(speeds=(8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 28.0, 30.0, 32.0, 35.0),
                sim_time: float = 20.0) -> list:
    """How wide is the speed range this controller actually works over?

    Two things to notice.

    FIRST, the reference is parameterised in TIME, so raising V stretches the
    manoeuvre in space: kappa falls as 1/V^2 exactly as fast as V^2 grows, and
    peak lateral acceleration is therefore INDEPENDENT of speed (0.309 g at
    every V). This is the opposite of the arc-length path in
    `ttv_rl_episode.m`, where a_y grows as V^2 and the example becomes
    infeasible above 12 m/s. A time-parameterised reference cannot become
    infeasible by going faster.

    SECOND, and despite that, the shipped weights only work between 15 and
    28 m/s. Outside that band the steering command sits on its +-0.5 rad limit
    for 30-74 % of the run and the rig jack-knifes (articulation to 152 deg).
    Nothing is wrong with the reference; the MPC is saturating.

    Turning on the weights that ship at zero removes the failure at both ends:
    zero saturation at every speed from 8 to 35 m/s. The cost is tracking
    error rising from ~0.005 m to ~0.10 m inside the old band. That is the
    trade -- a narrow window with excellent tracking, or a 4x wider window with
    10 cm of error and no saturation anywhere.
    """
    rows = []
    for V in speeds:
        ref = DLCReference(V=V)
        row = {"V": V, **ref.describe(sim_time)}
        for tag, kw in (("shipped", dict()), ("damped", DAMPED_WEIGHTS)):
            r = run_dlc_tracking(dlc_config(V=V, **kw), ref, sim_time,
                                 backend="fast")
            sat = float(np.mean(np.abs(r.delta_cmd) >= r.cfg.deltaMax - 1e-9))
            row[f"{tag}_rms"] = r.metrics["rms_lateral_error"]
            row[f"{tag}_phi_deg"] = math.degrees(r.metrics["peak_articulation_angle"])
            row[f"{tag}_saturation"] = 100.0 * sat
        rows.append(row)
    return rows


# =============================================================================
if __name__ == "__main__":
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ipopt", choices=["ipopt", "fast"],
                    help="ipopt = the MATLAB formulation; fast = condensed QP")
    ap.add_argument("--sim-time", type=float, default=20.0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--studies", action="store_true",
                    help="also run the weight and speed studies")
    ap.add_argument("--limits", action="store_true",
                    help="enable the episode-termination tests "
                         "(off by default, matching the MATLAB)")
    args = ap.parse_args()

    ref = DLCReference()
    cfg = dlc_config()

    print("=" * 86)
    print("TTV double-lane-change tracking: linear MPC model vs nonlinear plant")
    print("=" * 86)
    d = ref.describe(args.sim_time)
    print(f"  reference : Y = {ref.amplitude}*(tanh({ref.rate}t-{ref.t1}) "
          f"- tanh({ref.rate}t-{ref.t2})),  V = {cfg.V} m/s")
    print(f"  peak Y                 : {d['peak_y']:.4f} m")
    print(f"  peak slope             : {d['peak_slope_deg']:.2f} deg")
    print(f"  peak lateral accel     : {d['peak_lateral_accel']:.4f} m/s^2 "
          f"= {d['peak_lateral_accel_g']:.3f} g")
    print(f"  implied curvature      : {d['peak_curvature']:.5f} 1/m "
          f"(R_min {d['min_radius']:.0f} m)")
    print(f"  transition width       : {d['transition_width_s']:.3f} s "
          f"= {d['transition_width_m']:.1f} m")
    print(f"  road length            : {d['road_length_m']:.0f} m")
    print(f"\n  MPC: T={cfg.T}, N={cfg.N} ({cfg.T*cfg.N:.1f} s preview), "
          f"Qy={cfg.mpc.Qy}, Qpsi={cfg.mpc.Qpsi}, Qphi={cfg.mpc.Qphi}, "
          f"Qq={cfg.mpc.Qq}, Rdelta={cfg.mpc.Rdelta}")
    print(f"  backend: {args.backend}")

    lim = FailureLimits(enabled=args.limits)
    if args.limits:
        print(f"  failure limits ON: slip {lim.maxLateralSlip} rad, "
              f"e_y {lim.maxLateralError} m, "
              f"e_psi {math.degrees(lim.maxHeadingError):.0f} deg, "
              f"phi {math.degrees(lim.maxArticulation):.0f} deg")
    r = run_dlc_tracking(cfg, ref, args.sim_time, backend=args.backend,
                         limits=lim, verbose=True)
    print_summary(r)

    if not args.no_plots:
        plot_all(r)
    json.dump(r.metrics, open("out/dlc_metrics.json", "w"), indent=2, default=str)
    print("\n  wrote out/dlc_metrics.json")

    if args.studies:
        print("\n" + "=" * 86)
        print("A.  Activating the weights that ship at zero")
        print("=" * 86)
        rows = weight_study(ref, args.sim_time)
        print(f"  {'configuration':>24s} {'RMS e_y':>9s} {'peak e_y':>9s} "
              f"{'peak phi':>9s} {'peak q':>8s} {'peak del':>9s} {'max slip':>9s}")
        for w in rows:
            print(f"  {w['label']:>24s} {w['rms_lateral_error']:9.4f} "
                  f"{w['peak_lateral_error']:9.4f} "
                  f"{math.degrees(w['peak_articulation_angle']):8.3f}d "
                  f"{w['peak_articulation_rate']:8.4f} "
                  f"{math.degrees(w['peak_steering_command']):8.3f}d "
                  f"{math.degrees(w['max_front_slip']):8.3f}d")

        print("\n" + "=" * 86)
        print("B.  Speed sweep: how wide is the working range?")
        print("=" * 86)
        rows = speed_study(sim_time=args.sim_time)
        print(f"  {'V m/s':>6s} {'a_y [g]':>8s} {'kappa':>9s} {'R_min':>7s} | "
              f"{'SHIPPED Qphi=Qq=0':>27s} | {'+ Qpsi=1 Qphi=25 Qq=8':>27s}")
        print(f"  {'':>6s} {'':>8s} {'':>9s} {'':>7s} | "
              f"{'RMS e_y':>9s} {'peak phi':>9s} {'sat':>6s} | "
              f"{'RMS e_y':>9s} {'peak phi':>9s} {'sat':>6s}")
        for w in rows:
            print(f"  {w['V']:6.1f} {w['peak_lateral_accel_g']:8.3f} "
                  f"{w['peak_curvature']:9.5f} {w['min_radius']:7.0f} | "
                  f"{w['shipped_rms']:9.4f} {w['shipped_phi_deg']:8.2f}d "
                  f"{w['shipped_saturation']:5.1f}% | "
                  f"{w['damped_rms']:9.4f} {w['damped_phi_deg']:8.2f}d "
                  f"{w['damped_saturation']:5.1f}%")
        print("\n  a_y is the SAME at every speed -- a time-parameterised")
        print("  reference cannot become infeasible by going faster (kappa falls")
        print("  as 1/V^2 exactly as fast as V^2 grows). The failures at 8-12 and")
        print("  30-35 m/s are therefore NOT the reference being too hard. They")
        print("  are the MPC saturating its steering: with Qphi = Qq = 0 the")
        print("  command sits on its +-0.5 rad limit for 30-74 % of the run and")
        print("  the rig jack-knifes. Turning those weights on removes the")
        print("  saturation entirely at every speed tested.")
