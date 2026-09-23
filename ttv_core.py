"""
ttv_core.py -- Faithful Python port of ttv_rl_episode.m
=======================================================

One RL path-evaluation episode for a tractor-trailer vehicle (TTV):

        path  ->  linear MPC  ->  nonlinear tractor-trailer plant  ->  reward

This module is a deliberately *literal* translation of the MATLAB file so that
the two can be compared number-for-number. Every MATLAB local function has a
Python counterpart with the same name (snake_case kept as in the original).

Deviations from MATLAB are limited to things that have no exact Python twin,
and every one of them is marked with a `PORT NOTE:` comment:

  * `persistent cache`            -> module-level dict keyed by make_cache_key()
  * `gradient(F,X)`               -> matlab_gradient(), which reproduces
                                     MATLAB's non-uniform central difference
                                     (numpy.gradient uses a *different*,
                                     higher-order non-uniform formula)
  * `rcond`                       -> 1/cond(M,1)
  * `interp1(...,'pchip')`        -> scipy PchipInterpolator
  * `expm`                        -> scipy.linalg.expm

Reference
---------
A. Fehér, A. Domina, A. Bárdos, S. Aradi, T. Bécsi,
"Path planning via reinforcement learning with closed-loop motion control and
field tests", Engineering Applications of Artificial Intelligence 142 (2025)
109870.  (Fig. 2 training architecture; Eqs. 3-6 reward.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any

import numpy as np
import casadi as ca
from scipy.linalg import expm
from scipy.interpolate import PchipInterpolator


# =============================================================================
# 1.  Configuration  (<-> ttv_default_config)
# =============================================================================

@dataclass
class MPCConfig:
    """Weights of the linear MPC cost  (Eqs. 20-25 of the paper)."""
    Qy: float = 2.0              # lateral-error weight        [-]
    Qpsi: float = 1.0            # heading-error weight        [-]
    Qphi: float = 0.0            # articulation-angle weight   [-]  (OFF by default)
    Qq: float = 0.0              # articulation-rate weight    [-]  (OFF by default)
    Rdelta: float = 0.1          # steering-effort weight      [-]
    maxIterations: int = 300
    acceptableTolerance: float = 1e-7


@dataclass
class RewardConfig:
    """Fehér et al. reward (Eqs. 3-6) plus the episode-termination limits."""
    wCurvature: float = 0.5           # w_kappa   (must sum to 1 with wSlip)
    wSlip: float = 0.5                # w_mu
    cKappaDD: float = 0.0             # c_kappa   in Eq. (5)
    maxLateralSlip: float = 0.2       # [rad]  paper: 0.2 rad front/rear
    maxDistanceError: float = 1.0     # [m]    paper: 1 m
    maxHeadingError: float = math.radians(20.0)   # paper: 20 deg
    failurePenalty: float = -1.0
    failurePenaltyLastSection: float = -0.75
    lastSectionFraction: float = 0.80


@dataclass
class TTVConfig:
    # --- simulation / MPC horizon -------------------------------------------
    T: float = 0.1                 # [s]   control period
    N: int = 10                    # [-]   prediction horizon (paper: Np=Nc=10)
    V: float = 20.0                # [m/s] constant longitudinal speed
    maxEpisodeTime: float = 30.0   # [s]
    plantSubsteps: int = 5         # RK4 substeps per control period
    minPathLength: float = 5.0     # [m]
    finishTolerance: float = 0.5   # [m]
    initialPlantState: Optional[np.ndarray] = None
    logPhysicalDiagnostics: bool = False

    # --- steering actuator ---------------------------------------------------
    deltaMax: float = 0.5          # [rad]  +-0.5 rad, as in Eq. (27)
    deltaRateMax: float = 0.6      # [rad/s]
    steeringTimeConstant: float = 0.15   # [s]   T_st in Eq. (12)
    phiMax: float = math.inf       # [rad]  articulation-angle box (MPC only)
    qMax: float = math.inf         # [rad/s]

    # --- tractor-trailer geometry / inertia ---------------------------------
    m1: float = 5760.0             # [kg]    tractor mass
    m2: float = 6640.0             # [kg]    trailer mass
    a1: float = 1.10               # [m]     tractor CoG -> front axle
    a2: float = 5.21               # [m]     hitch -> trailer CoG
    b1: float = 2.39               # [m]     tractor CoG -> rear axle
    b2: float = 3.28               # [m]     trailer CoG -> trailer axle
    c: float = 1.64                # [m]     tractor CoG -> hitch (behind CoG)
    Iz1: float = 34823.0           # [kg m^2]
    Iz2: float = 179992.0          # [kg m^2]
    C1: float = 223281.0           # [N/rad] tractor front axle
    C2: float = 223281.0           # [N/rad] tractor rear axle
    C3: float = 223281.0           # [N/rad] trailer axle
    mu: float = 0.90               # [-]     friction coefficient
    g: float = 9.81                # [m/s^2]
    Fz1: Optional[float] = None    # [N] filled in by finalize() if not given
    Fz2: Optional[float] = None
    Fz3: Optional[float] = None

    mpc: MPCConfig = field(default_factory=MPCConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # ---------------------------------------------------------------------
    def finalize(self) -> "TTVConfig":
        """<-> the static-load block and validateattributes() of MATLAB."""
        if self.Fz1 is None or self.Fz2 is None or self.Fz3 is None:
            hitch_load = self.m2 * self.g * self.b2 / (self.a2 + self.b2)
            trailer_axle_load = self.m2 * self.g * self.a2 / (self.a2 + self.b2)
            tractor_front_load = (self.b1 * self.m1 * self.g
                                  + (self.b1 - self.c) * hitch_load) / (self.a1 + self.b1)
            tractor_rear_load = self.m1 * self.g + hitch_load - tractor_front_load
            self.Fz1 = tractor_front_load
            self.Fz2 = tractor_rear_load
            self.Fz3 = trailer_axle_load

        if not (self.T > 0 and math.isfinite(self.T)):
            raise ValueError("cfg.T must be a positive finite scalar.")
        if not (isinstance(self.N, (int, np.integer)) and self.N >= 2):
            raise ValueError("cfg.N must be an integer >= 2.")
        if not (self.V > 0 and math.isfinite(self.V)):
            raise ValueError("cfg.V must be a positive finite scalar.")
        if self.plantSubsteps < 1:
            raise ValueError("cfg.plantSubsteps must be an integer >= 1.")
        if abs(self.reward.wCurvature + self.reward.wSlip - 1.0) > 1e-10:
            raise ValueError("cfg.reward.wCurvature + cfg.reward.wSlip must equal one.")
        return self


# =============================================================================
# 2.  MATLAB compatibility helpers
# =============================================================================

def matlab_gradient(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """MATLAB's gradient(F, X) for a non-uniform grid X.

    PORT NOTE: MATLAB uses the *simple* central difference
        g[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    while numpy.gradient uses a second-order-accurate non-uniform stencil.
    On a uniform grid the two agree; on a non-uniform grid they do not, and the
    reward of this episode depends on three nested gradients, so the difference
    is not academic. This function reproduces MATLAB exactly.
    """
    f = np.asarray(f, dtype=float)
    x = np.asarray(x, dtype=float)
    n = f.size
    g = np.empty_like(f)
    if n == 1:
        return np.zeros_like(f)
    g[0] = (f[1] - f[0]) / (x[1] - x[0])
    g[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
    if n > 2:
        g[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    return g


def wrap_angle(angle: float) -> float:
    """<-> wrap_angle: atan2(sin,cos), i.e. wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def matlab_rcond(M: np.ndarray) -> float:
    """PORT NOTE: MATLAB rcond() is a 1-norm condition *estimate*; we use the
    exact 1-norm condition number, which is what rcond estimates."""
    try:
        return 1.0 / np.linalg.cond(M, 1)
    except np.linalg.LinAlgError:
        return 0.0


# =============================================================================
# 3.  Prediction model  (<-> build_linear_prediction_model)
# =============================================================================

def build_linear_prediction_model(cfg: TTVConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Linear 7-state lateral model of the articulated vehicle.

    The descriptor system underneath is  M z_dot + K z = G * delta  with
        z = [r1 ; beta ; q ; phi]
    r1   tractor yaw rate          [rad/s]
    beta tractor sideslip ANGLE    [rad]   (= v1 / V, dimensionless!)
    q    articulation rate         [rad/s]
    phi  articulation angle        [rad]

    That `beta` (and not the lateral velocity v1) is state 2 is the reason for
    every factor of V and 1/V in the Ac assembly below; see the derivation in
    the accompanying guide.

    MPC state:  x = [y ; y_dot ; psi ; r1 ; phi ; q ; delta_actual]
    MPC input:  u = delta_reference
    """
    # Sign convention of the source AHV paper: Fy_i = C_i * alpha_i, C_i < 0.
    C1s = -abs(cfg.C1)
    C2s = -abs(cfg.C2)
    C3s = -abs(cfg.C3)
    V = cfg.V

    M = np.array([
        [cfg.Iz1, cfg.m1 * cfg.c * V, 0.0, 0.0],
        [-cfg.m2 * (cfg.a2 + cfg.c), (cfg.m1 + cfg.m2) * V, -cfg.m2 * cfg.a2,
         C3s * (cfg.a2 + cfg.b2) / V],
        [cfg.Iz2, cfg.m1 * cfg.a2 * V, cfg.Iz2,
         -C3s * cfg.b2 * (cfg.a2 + cfg.b2) / V],
        [0.0, 0.0, 0.0, 1.0],
    ])

    K = np.array([
        [cfg.m1 * cfg.c * V
         - (C1s * cfg.a1 * (cfg.a1 + cfg.c) + C2s * cfg.b1 * (cfg.b1 - cfg.c)) / V,
         -C1s * (cfg.a1 + cfg.c) + C2s * (cfg.b1 - cfg.c), 0.0, 0.0],
        [(cfg.m1 + cfg.m2) * V
         + (-C1s * cfg.a1 + C2s * cfg.b1 + C3s * (cfg.a2 + cfg.b2 + cfg.c)) / V,
         -(C1s + C2s + C3s), 0.0, C3s],
        [cfg.m1 * cfg.a2 * V
         + (-C1s * cfg.a1 * cfg.a2 + C2s * cfg.b1 * cfg.a2
            - C3s * cfg.b2 * (cfg.a2 + cfg.b2 + cfg.c)) / V,
         -(C1s * cfg.a2 + C2s * cfg.a2 - C3s * cfg.b2), 0.0, -C3s * cfg.b2],
        [0.0, 0.0, -1.0, 0.0],
    ])

    G = np.array([-C1s * (cfg.a1 + cfg.c), -C1s, -C1s * cfg.a2, 0.0])

    Av = -np.linalg.solve(M, K)          # z_dot = Av z + Bv delta
    Bv = np.linalg.solve(M, G)

    Ac = np.zeros((7, 7))
    Ac[0, 1] = 1.0                                        # y_dot        = state 2
    Ac[1, :] = [0.0, Av[1, 1], -V * Av[1, 1], V * (Av[1, 0] + 1.0),
                V * Av[1, 3], V * Av[1, 2], V * Bv[1]]    # y_ddot
    Ac[2, 3] = 1.0                                        # psi_dot      = r1
    Ac[3, :] = [0.0, Av[0, 1] / V, -Av[0, 1], Av[0, 0],
                Av[0, 3], Av[0, 2], Bv[0]]                # r1_dot
    Ac[4, 5] = 1.0                                        # phi_dot      = q
    Ac[5, :] = [0.0, Av[2, 1] / V, -Av[2, 1], Av[2, 0],
                Av[2, 3], Av[2, 2], Bv[2]]                # q_dot
    Ac[6, 6] = -1.0 / cfg.steeringTimeConstant            # steering lag, Eq.(12)

    Bc = np.zeros((7, 1))
    Bc[6, 0] = 1.0 / cfg.steeringTimeConstant
    return Ac, Bc


# =============================================================================
# 4.  Nonlinear plant  (<-> build_nonlinear_plant)
# =============================================================================

def build_nonlinear_plant(cfg: TTVConfig):
    """5-DOF yaw-plane tractor-semitrailer with saturating tyres.

    Plant state  xp = [X1 ; Y1 ; psi1 ; v1 ; r1 ; phi ; q ; delta_actual]
      X1,Y1  tractor CoG position in the road frame   [m]
      psi1   tractor heading                          [rad]
      v1     tractor lateral velocity                 [m/s]   (NOT sideslip)
      r1     tractor yaw rate                         [rad/s]
      phi    articulation angle                       [rad]
      q      articulation rate                        [rad/s]
      delta_actual  steering angle at the road wheel  [rad]
    """
    xp = ca.SX.sym("xp", 8, 1)
    delta_cmd = ca.SX.sym("delta_cmd")

    psi1 = xp[2]
    v1 = xp[3]
    r1 = xp[4]
    phi = xp[5]
    q = xp[6]
    delta = xp[7]
    r2 = r1 + q
    V = cfg.V

    hitch_lateral_velocity = v1 - cfg.c * r1
    V2 = V * ca.cos(phi) + hitch_lateral_velocity * ca.sin(phi)
    v2 = -V * ca.sin(phi) + hitch_lateral_velocity * ca.cos(phi) - cfg.a2 * r2

    alpha1 = delta - ca.atan2(v1 + cfg.a1 * r1, V)
    alpha2 = -ca.atan2(v1 - cfg.b1 * r1, V)
    alpha3 = -ca.atan2(v2 - cfg.b2 * r2, V2)
    alpha = ca.vertcat(alpha1, alpha2, alpha3)

    Fy1 = cfg.mu * cfg.Fz1 * ca.tanh(cfg.C1 * alpha1 / (cfg.mu * cfg.Fz1))
    Fy2 = cfg.mu * cfg.Fz2 * ca.tanh(cfg.C2 * alpha2 / (cfg.mu * cfg.Fz2))
    Fy3 = cfg.mu * cfg.Fz3 * ca.tanh(cfg.C3 * alpha3 / (cfg.mu * cfg.Fz3))
    Fy = ca.vertcat(Fy1, Fy2, Fy3)

    # Unknown vector zeta = [v1_dot ; r1_dot ; q_dot ; Fhx ; Fhy]
    # (the last two are the hitch constraint forces).
    Mplant = ca.SX.zeros(5, 5)
    Mplant[0, :] = ca.horzcat(cfg.m1, 0, 0, 0, 1)
    Mplant[1, :] = ca.horzcat(0, cfg.Iz1, 0, 0, -cfg.c)
    Mplant[2, :] = ca.horzcat(cfg.m2 * ca.sin(phi), -cfg.m2 * cfg.c * ca.sin(phi),
                              0, -ca.cos(phi), -ca.sin(phi))
    Mplant[3, :] = ca.horzcat(cfg.m2 * ca.cos(phi),
                              -cfg.m2 * (cfg.c * ca.cos(phi) + cfg.a2),
                              -cfg.m2 * cfg.a2, ca.sin(phi), -ca.cos(phi))
    Mplant[4, :] = ca.horzcat(0, cfg.Iz2, cfg.Iz2,
                              cfg.a2 * ca.sin(phi), -cfg.a2 * ca.cos(phi))

    bplant = ca.vertcat(
        Fy1 * ca.cos(delta) + Fy2 - cfg.m1 * V * r1,
        cfg.a1 * Fy1 * ca.cos(delta) - cfg.b1 * Fy2,
        cfg.m2 * (v2 * r1 - cfg.a2 * r2 * q),
        Fy3 - cfg.m2 * V2 * r1,
        -cfg.b2 * Fy3,
    )

    zeta = ca.solve(Mplant, bplant)
    v1_dot = zeta[0]
    r1_dot = zeta[1]
    q_dot = zeta[2]

    raw_delta_rate = (delta_cmd - delta) / cfg.steeringTimeConstant
    delta_dot = cfg.deltaRateMax * ca.tanh(raw_delta_rate / cfg.deltaRateMax)

    x_dot = ca.vertcat(
        V * ca.cos(psi1) - v1 * ca.sin(psi1),
        V * ca.sin(psi1) + v1 * ca.cos(psi1),
        r1,
        v1_dot,
        r1_dot,
        q,
        q_dot,
        delta_dot,
    )

    plant_dynamics = ca.Function("ttv_plant_dynamics", [xp, delta_cmd], [x_dot])
    plant_output = ca.Function("ttv_plant_output", [xp, delta_cmd],
                               [alpha, Fy, ca.vertcat(V2, v2, zeta[3], zeta[4])])
    plant_diagnostics = ca.Function("ttv_plant_diagnostics", [xp, delta_cmd],
                                    [Mplant, zeta, bplant, x_dot])
    return plant_dynamics, plant_output, plant_diagnostics


def rk4_step(dynamics, state: np.ndarray, u: float, h: float) -> np.ndarray:
    """Classical RK4, identical to the MATLAB helper."""
    k1 = np.asarray(dynamics(state, u)).flatten()
    k2 = np.asarray(dynamics(state + 0.5 * h * k1, u)).flatten()
    k3 = np.asarray(dynamics(state + 0.5 * h * k2, u)).flatten()
    k4 = np.asarray(dynamics(state + h * k3, u)).flatten()
    return state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def plant_to_mpc_state(xp: np.ndarray, V: float) -> np.ndarray:
    """xp (8) -> MPC state (7). Note y_dot is built exactly, not linearised."""
    psi = xp[2]
    v1 = xp[3]
    y_dot = V * math.sin(psi) + v1 * math.cos(psi)
    return np.array([xp[1], y_dot, psi, xp[4], xp[5], xp[6], xp[7]])


# =============================================================================
# 5.  Environment cache  (<-> persistent cache + build_environment)
# =============================================================================

_ENV_CACHE: Dict[str, Any] = {}


def make_cache_key(cfg: TTVConfig) -> str:
    values = [cfg.T, cfg.N, cfg.V, cfg.plantSubsteps, cfg.deltaMax,
              cfg.deltaRateMax, cfg.steeringTimeConstant, cfg.phiMax, cfg.qMax,
              cfg.m1, cfg.m2, cfg.a1, cfg.a2, cfg.b1, cfg.b2, cfg.c,
              cfg.Iz1, cfg.Iz2, cfg.C1, cfg.C2, cfg.C3, cfg.mu,
              cfg.Fz1, cfg.Fz2, cfg.Fz3,
              cfg.mpc.Qy, cfg.mpc.Qpsi, cfg.mpc.Qphi, cfg.mpc.Qq,
              cfg.mpc.Rdelta, cfg.mpc.maxIterations, cfg.mpc.acceptableTolerance]
    return ",".join(f"{v:.16g}" for v in values)


def build_environment(cfg: TTVConfig, key: str) -> Dict[str, Any]:
    Ac, Bc = build_linear_prediction_model(cfg)
    nx = Ac.shape[0]

    # Exact zero-order-hold discretisation via the matrix exponential.
    aug = expm(np.block([[Ac, Bc], [np.zeros((1, nx + 1))]]) * cfg.T)
    Ad = aug[:nx, :nx]
    Bd = aug[:nx, nx:nx + 1]

    N = cfg.N
    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", 1, N)
    P = ca.SX.sym("P", nx + 2 * N, 1)

    objective = 0
    constraints = [X[:, 0] - P[0:nx]]
    for k in range(N):
        y_ref = P[nx + 2 * k]
        psi_ref = P[nx + 2 * k + 1]
        eY = X[0, k + 1] - y_ref
        ePsi = X[2, k + 1] - psi_ref
        objective = (objective
                     + cfg.mpc.Qy * eY ** 2
                     + cfg.mpc.Qpsi * ePsi ** 2
                     + cfg.mpc.Qphi * X[4, k + 1] ** 2
                     + cfg.mpc.Qq * X[5, k + 1] ** 2
                     + cfg.mpc.Rdelta * U[0, k] ** 2)
        constraints.append(X[:, k + 1] - (ca.DM(Ad) @ X[:, k] + ca.DM(Bd) @ U[0, k]))

    # PORT NOTE: MATLAB's X(:) / U(:) is column-major; ca.reshape(X, -1, 1) on a
    # CasADi matrix is column-major too, so the decision ordering is identical.
    decision = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {"f": objective, "x": decision, "g": ca.vertcat(*constraints), "p": P}
    opts = {
        "ipopt.max_iter": cfg.mpc.maxIterations,
        "ipopt.print_level": 0,
        "ipopt.acceptable_tol": cfg.mpc.acceptableTolerance,
        "ipopt.acceptable_obj_change_tol": 1e-6,
        "ipopt.sb": "yes",
        "print_time": False,
    }
    solver = ca.nlpsol("ttv_mpc_solver", "ipopt", nlp, opts)

    nw = nx * (N + 1) + N
    lbx = -np.inf * np.ones(nw)
    ubx = np.inf * np.ones(nw)
    for k in range(N + 1):
        off = k * nx
        lbx[off + 6] = -cfg.deltaMax          # delta_actual  (MATLAB index 7)
        ubx[off + 6] = cfg.deltaMax
        if math.isfinite(cfg.phiMax):
            lbx[off + 4] = -cfg.phiMax        # phi           (MATLAB index 5)
            ubx[off + 4] = cfg.phiMax
        if math.isfinite(cfg.qMax):
            lbx[off + 5] = -cfg.qMax          # q             (MATLAB index 6)
            ubx[off + 5] = cfg.qMax
    control_start = nx * (N + 1)
    lbx[control_start:] = -cfg.deltaMax
    ubx[control_start:] = cfg.deltaMax

    ng = nx * (N + 1)
    plant_dynamics, plant_output, plant_diagnostics = build_nonlinear_plant(cfg)

    return {
        "key": key, "Ad": Ad, "Bd": Bd, "solver": solver,
        "lbx": lbx, "ubx": ubx, "lbg": np.zeros(ng), "ubg": np.zeros(ng),
        "plantDynamics": plant_dynamics, "plantOutput": plant_output,
        "plantDiagnostics": plant_diagnostics, "nx": nx,
    }


def get_environment(cfg: TTVConfig) -> Dict[str, Any]:
    key = make_cache_key(cfg)
    cached = _ENV_CACHE.get("env")
    if cached is None or cached["key"] != key:
        cached = build_environment(cfg, key)
        _ENV_CACHE["env"] = cached
    return cached


# =============================================================================
# 6.  Path preparation  (<-> prepare_path)
# =============================================================================

@dataclass
class PreparedPath:
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    psi: np.ndarray
    kappa: np.ndarray
    kappaPrime: np.ndarray
    kappaDD: np.ndarray


def prepare_path(path_input) -> PreparedPath:
    if isinstance(path_input, PreparedPath):
        return path_input
    if isinstance(path_input, dict):
        x = np.asarray(path_input["x"], dtype=float).ravel()
        y = np.asarray(path_input["y"], dtype=float).ravel()
    else:
        arr = np.asarray(path_input, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("A numeric path must be an N-by-2 array [x,y].")
        x = arr[:, 0].copy()
        y = arr[:, 1].copy()

    if x.size != y.size or x.size < 7:
        raise ValueError("The path must contain at least seven paired x-y points.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Path coordinates must be finite.")

    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate(([True], seg > 1e-8))
    x, y = x[keep], y[keep]
    if x.size < 7:
        raise ValueError("The path contains too few distinct points.")

    # Normalise to the initial path pose: origin and heading of the first point.
    psi0 = math.atan2(y[1] - y[0], x[1] - x[0])
    R = np.array([[math.cos(psi0), math.sin(psi0)],
                  [-math.sin(psi0), math.cos(psi0)]])
    local = R @ np.vstack((x - x[0], y - y[0]))
    x, y = local[0, :], local[1, :]

    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    dxds = matlab_gradient(x, s)
    dyds = matlab_gradient(y, s)
    psi = np.unwrap(np.arctan2(dyds, dxds))
    kappa = matlab_gradient(psi, s)
    kappa_prime = matlab_gradient(kappa, s)
    kappa_dd = matlab_gradient(kappa_prime, s)

    return PreparedPath(x=x, y=y, s=s, psi=psi, kappa=kappa,
                        kappaPrime=kappa_prime, kappaDD=kappa_dd)


def path_errors(xp: np.ndarray, path: PreparedPath):
    """<-> path_errors. Global nearest-sample search, exactly as in MATLAB."""
    d2 = (path.x - xp[0]) ** 2 + (path.y - xp[1]) ** 2
    idx = int(np.argmin(d2))
    s_now = path.s[idx]
    dx = xp[0] - path.x[idx]
    dy = xp[1] - path.y[idx]
    e_lat = -math.sin(path.psi[idx]) * dx + math.cos(path.psi[idx]) * dy
    e_psi = wrap_angle(xp[2] - path.psi[idx])
    dist = math.sqrt(d2[idx])
    return s_now, e_lat, e_psi, dist


# =============================================================================
# 7.  The episode  (<-> ttv_rl_episode)
# =============================================================================

def ttv_rl_episode(path_input, cfg: Optional[TTVConfig] = None):
    """Run one closed-loop episode.

    Returns (reward, rewards, log) exactly like the MATLAB function.
    """
    cfg = (cfg or TTVConfig())
    cfg.finalize()

    path = prepare_path(path_input)
    if path.s[-1] <= cfg.minPathLength:
        raise ValueError(
            f"The usable path length must exceed {cfg.minPathLength:.3f} m.")

    cache = get_environment(cfg)
    nx = cache["nx"]
    N = cfg.N

    if cfg.initialPlantState is None:
        xp = np.zeros(8)
        xp[0] = path.x[0]
        xp[1] = path.y[0]
        xp[2] = path.psi[0]
    else:
        xp = np.asarray(cfg.initialPlantState, dtype=float).ravel().copy()
        if xp.size != 8:
            raise ValueError("cfg.initialPlantState must contain eight elements.")

    max_steps = max(1, math.ceil(cfg.maxEpisodeTime / cfg.T))

    xm = plant_to_mpc_state(xp, cfg.V)
    Xguess = np.tile(xm.reshape(-1, 1), (1, N + 1))
    Uguess = np.zeros((1, N))

    nan = np.nan
    tHist = np.full(max_steps + 1, nan)
    plantHist = np.full((8, max_steps + 1), nan)
    mpcHist = np.full((nx, max_steps + 1), nan)
    uHist = np.full(max_steps, nan)
    refYHist = np.full(max_steps, nan)
    refPsiHist = np.full(max_steps, nan)
    eLatHist = np.full(max_steps + 1, nan)
    ePsiHist = np.full(max_steps + 1, nan)
    distHist = np.full(max_steps + 1, nan)
    alphaHist = np.full((3, max_steps + 1), nan)
    forceHist = np.full((3, max_steps + 1), nan)
    sHist = np.full(max_steps + 1, nan)
    diag_hists = {k: np.full(max_steps + 1, nan) for k in
                  ("massRelativeResidual", "massRcond", "yawRateIdentity",
                   "articulationRateIdentity")} if cfg.logPhysicalDiagnostics else None

    s_now, e_lat, e_psi, dist = path_errors(xp, path)
    alpha, Fy = plant_outputs(cache, xp, 0.0)
    tHist[0] = 0.0
    plantHist[:, 0] = xp
    mpcHist[:, 0] = xm
    eLatHist[0] = e_lat
    ePsiHist[0] = e_psi
    distHist[0] = dist
    alphaHist[:, 0] = alpha
    forceHist[:, 0] = Fy
    sHist[0] = s_now
    if diag_hists is not None:
        d = plant_diagnostics(cache, xp, 0.0)
        for k in diag_hists:
            diag_hists[k][0] = d[k]

    s_interp = PchipInterpolator(path.s, path.y, extrapolate=True)

    failed = False
    passed = False
    failure_reason = ""
    final_step = 0
    solver_iterations = []

    for step in range(1, max_steps + 1):
        xm = plant_to_mpc_state(xp, cfg.V)

        # Preview the geometry by arc length. Path information enters the MPC
        # cost, never the state equation x_dot = A x + B u.
        s_preview = s_now + cfg.V * cfg.T * np.arange(1, N + 1)
        s_preview = np.clip(s_preview, path.s[0], path.s[-1])
        y_ref = s_interp(s_preview)
        psi_ref = np.interp(s_preview, path.s, path.psi)

        p = np.zeros(nx + 2 * N)
        p[:nx] = xm
        p[nx + 0::2] = y_ref
        p[nx + 1::2] = psi_ref

        w0 = np.concatenate((Xguess.flatten(order="F"), Uguess.flatten(order="F")))

        try:
            sol = cache["solver"](x0=w0, lbx=cache["lbx"], ubx=cache["ubx"],
                                  lbg=cache["lbg"], ubg=cache["ubg"], p=p)
            stats = cache["solver"].stats()
            solver_ok = bool(stats.get("success", False))
            solver_iterations.append(stats.get("iter_count", np.nan))
        except Exception as exc:                       # pragma: no cover
            solver_ok = False
            stats = {"return_status": str(exc)}

        if not solver_ok:
            failed = True
            failure_reason = "MPC solver: " + str(stats.get("return_status", "unknown failure"))
            final_step = step - 1
            break

        w_opt = np.asarray(sol["x"]).flatten()
        Xopt = w_opt[:nx * (N + 1)].reshape((nx, N + 1), order="F")
        Uopt = w_opt[nx * (N + 1):].reshape((1, N), order="F")
        delta_cmd = float(np.clip(Uopt[0, 0], -cfg.deltaMax, cfg.deltaMax))

        h = cfg.T / cfg.plantSubsteps
        for _ in range(cfg.plantSubsteps):
            xp = rk4_step(cache["plantDynamics"], xp, delta_cmd, h)
        xp[7] = float(np.clip(xp[7], -cfg.deltaMax, cfg.deltaMax))

        s_now, e_lat, e_psi, dist = path_errors(xp, path)
        alpha, Fy = plant_outputs(cache, xp, delta_cmd)
        xm = plant_to_mpc_state(xp, cfg.V)

        tHist[step] = step * cfg.T
        plantHist[:, step] = xp
        mpcHist[:, step] = xm
        uHist[step - 1] = delta_cmd
        refYHist[step - 1] = y_ref[0]
        refPsiHist[step - 1] = psi_ref[0]
        eLatHist[step] = e_lat
        ePsiHist[step] = e_psi
        distHist[step] = dist
        alphaHist[:, step] = alpha
        forceHist[:, step] = Fy
        sHist[step] = s_now
        if diag_hists is not None:
            d = plant_diagnostics(cache, xp, delta_cmd)
            for k in diag_hists:
                diag_hists[k][step] = d[k]
        final_step = step

        Xguess = np.hstack((Xopt[:, 1:], Xopt[:, -1:]))
        Uguess = np.hstack((Uopt[:, 1:], Uopt[:, -1:]))
        Xguess[:, 0] = xm

        # Failure checks that are meaningful with a path-only input.
        if np.max(np.abs(alpha[0:2])) > cfg.reward.maxLateralSlip:
            failed, failure_reason = True, "tractor lateral-slip limit"
        elif dist > cfg.reward.maxDistanceError:
            failed, failure_reason = True, "path-distance limit"
        elif abs(e_psi) > cfg.reward.maxHeadingError:
            failed, failure_reason = True, "path-heading limit"
        elif not np.all(np.isfinite(xp)):
            failed, failure_reason = True, "non-finite plant state"

        if failed:
            break
        if s_now >= path.s[-1] - cfg.finishTolerance:
            passed = True
            break

    if not failed and not passed:
        failed = True
        failure_reason = "episode time limit"

    n_state = final_step + 1
    sr = slice(0, n_state)
    cr = slice(0, final_step)

    alpha_used = alphaHist[:, sr]
    max_slip_front = float(np.max(np.abs(alpha_used[0, :])))
    max_slip_rear = float(np.max(np.abs(alpha_used[1, :])))
    max_slip_trailer = float(np.max(np.abs(alpha_used[2, :])))

    # Fehér et al., Eqs. (3)-(4): the slip reference is fitted with v0 in km/h.
    speed_kmh = 3.6 * cfg.V
    slip_reference = 0.0037 * math.exp(0.0693 * speed_kmh)
    reward_slip = 2.0 * slip_reference - max_slip_front - max_slip_rear

    # Eq. (5). NOTE: this depends only on the *input path*, not on the run.
    kappa_dd = path.kappaDD
    reward_curvature = (cfg.reward.cKappaDD
                        - abs(np.max(kappa_dd)) - abs(np.min(kappa_dd)))

    progress = float(min(max(sHist[final_step] / path.s[-1], 0.0), 1.0))
    if passed:
        reward_penalty = 0.0
        reward = (cfg.reward.wCurvature * reward_curvature
                  + cfg.reward.wSlip * reward_slip)
    else:
        reward_penalty = (cfg.reward.failurePenaltyLastSection
                          if progress >= cfg.reward.lastSectionFraction
                          else cfg.reward.failurePenalty)
        reward = reward_penalty

    rewards = {
        "total": float(reward), "slip": float(reward_slip),
        "curvature": float(reward_curvature), "penalty": float(reward_penalty),
        "slipReference": float(slip_reference), "passed": bool(passed),
        "failed": bool(failed), "failureReason": failure_reason,
        "progress": progress, "maxSlipFront": max_slip_front,
        "maxSlipRear": max_slip_rear, "maxSlipTrailer": max_slip_trailer,
    }

    log = {
        "time": tHist[sr], "plantState": plantHist[:, sr],
        "mpcState": mpcHist[:, sr], "deltaCommand": uHist[cr],
        "referenceY": refYHist[cr], "referencePsi": refPsiHist[cr],
        "lateralError": eLatHist[sr], "headingError": ePsiHist[sr],
        "distanceError": distHist[sr], "slipAngles": alphaHist[:, sr],
        "lateralForces": forceHist[:, sr], "pathProgress": sHist[sr],
        "path": path, "rewards": rewards, "config": cfg,
        "solverIterations": np.asarray(solver_iterations, dtype=float),
    }
    if diag_hists is not None:
        log["physicalDiagnostics"] = {k: v[sr] for k, v in diag_hists.items()}
    log["metrics"] = make_metrics(log)
    return float(reward), rewards, log


def plant_outputs(cache, xp, u):
    a, f, _ = cache["plantOutput"](xp, u)
    return np.asarray(a).flatten(), np.asarray(f).flatten()


def plant_diagnostics(cache, xp, u):
    Mc, zc, bc, xdc = cache["plantDiagnostics"](xp, u)
    M = np.asarray(Mc)
    zeta = np.asarray(zc).flatten()
    b = np.asarray(bc).flatten()
    x_dot = np.asarray(xdc).flatten()
    residual = M @ zeta - b
    scale = max(1.0, np.linalg.norm(M @ zeta, 2), np.linalg.norm(b, 2))
    return {
        "massRelativeResidual": float(np.linalg.norm(residual, 2) / scale),
        "massRcond": matlab_rcond(M),
        "yawRateIdentity": float((xp[4] + xp[6]) - xp[4] - xp[6]),
        "articulationRateIdentity": float(x_dot[5] - xp[6]),
    }


def make_metrics(log) -> Dict[str, float]:
    ps = log["plantState"]
    return {
        "rmsLateralError": float(np.sqrt(np.mean(log["lateralError"] ** 2))),
        "peakLateralError": float(np.max(np.abs(log["lateralError"]))),
        "rmsHeadingError": float(np.sqrt(np.mean(log["headingError"] ** 2))),
        "peakHeadingError": float(np.max(np.abs(log["headingError"]))),
        "peakArticulationAngle": float(np.max(np.abs(ps[5, :]))),
        "peakArticulationRate": float(np.max(np.abs(ps[6, :]))),
        "rmsArticulationRate": float(np.sqrt(np.mean(ps[6, :] ** 2))),
        "peakSteeringCommand": float(np.max(np.abs(log["deltaCommand"])))
        if log["deltaCommand"].size else float("nan"),
        "peakActualSteering": float(np.max(np.abs(ps[7, :]))),
        "maxSlipFront": float(np.max(np.abs(log["slipAngles"][0, :]))),
        "maxSlipRear": float(np.max(np.abs(log["slipAngles"][1, :]))),
        "maxSlipTrailer": float(np.max(np.abs(log["slipAngles"][2, :]))),
    }


# =============================================================================
# 8.  The docstring example
# =============================================================================

def example_path(n: int = 601) -> np.ndarray:
    """The double-lane-change path from the MATLAB header comment."""
    x = np.linspace(0.0, 120.0, n)
    y = 1.75 * (np.tanh(0.12 * (x - 35.0)) - np.tanh(0.12 * (x - 80.0)))
    return np.column_stack((x, y))


if __name__ == "__main__":
    r, parts, data = ttv_rl_episode(example_path(), TTVConfig(logPhysicalDiagnostics=True))
    print(f"reward = {r:.6f}")
    for k, v in parts.items():
        print(f"  {k:>20s} : {v}")
    print("metrics:")
    for k, v in data["metrics"].items():
        print(f"  {k:>24s} : {v: .6g}")
