"""
highway_env.py -- Highway obstacle avoidance for a laden tractor-semitrailer
============================================================================

Fig. 2 of the paper, re-targeted:

    scenario -> observation -> RL agent -> action -> PATH GENERATOR -> path
                                                                        |
                       reward <- truck metrics <- PLANT <- TRACKING MPC <+

What is different from the shipped MATLAB loop, and why
-------------------------------------------------------
1. ROLLOVER, NOT SLIP, IS THE BINDING LIMIT.  Measured in truck_params.py: on
   dry or damp asphalt a 40 t rig reaches its rollover threshold (0.37 g) well
   before its friction limit (0.9 g). The shipped failure test is tyre slip
   > 0.2 rad, which a laden artic can only reach *after* it has rolled. The
   primary safety metric here is the Load Transfer Ratio.

2. THE ACTION CANNOT COMMAND A ROLLOVER.  Peak curvature of a lateral
   transition scales as |dy|/L^2, so `min_length_for_offset` inverts it and the
   action's length range starts at the shortest rollover-safe manoeuvre. The
   constraint lives in the action scaling, not in a penalty the agent must
   discover.

3. THE TRAILER IS SCORED, NOT JUST LOGGED.  The shipped code computes
   articulation and then says "Articulation is logged but is not yet included
   in the reward". Rearward amplification and trailer off-tracking are what
   actually put a trailer into the next lane.

4. MONOTONIC PROGRESS.  path_errors() in the original does a global argmin over
   every path sample at every step: O(N) per step, and the matched point can
   jump backwards. Here the search is a forward-only window.

5. RECEDING-HORIZON OPTION.  The paper's episode is a single step (a bandit).
   That cannot react to anything after the path is committed. `mode="receding"`
   re-plans every `replan_period` seconds from the measured state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

from ttv_core import TTVConfig, plant_to_mpc_state
from plant_numpy import PlantNP
from fast_mpc import CondensedMPC, NX, IDX_DELTA
from path_gen import (GeoPath, bezier_offset_path, min_length_for_offset,
                      kappa_bound)
import truck_params as TP

G = 9.81


# =============================================================================
# 1.  Scenario
# =============================================================================

@dataclass
class Scenario:
    """One highway obstacle-avoidance task."""
    V: float = TP.V_HIGHWAY          # [m/s]  constant speed (cruise/limiter)
    mu: float = 0.85                 # [-]    road friction
    lane_width: float = TP.LANE_WIDTH
    n_lanes: int = 2
    # obstacle: a stopped vehicle / roadworks in the ego lane
    obstacle_x: float = 160.0        # [m] along-road position of its near face
    obstacle_len: float = 12.0       # [m]
    obstacle_half_width: float = 1.2  # [m] from the ego-lane centreline
    must_return: bool = True         # return to the original lane afterwards
    # load state -- the same rig runs laden and empty, and SRT nearly doubles
    payload_fraction: float = 1.0    # 1.0 = fully laden, 0.0 = empty trailer
    # initial condition
    e_y0: float = 0.0                # [m]   initial lateral offset
    e_psi0: float = 0.0              # [rad] initial heading error
    phi0: float = 0.0                # [rad] initial articulation
    road_length: float = 420.0       # [m]

    # ---- derived ----------------------------------------------------------
    @property
    def m2(self) -> float:
        """Trailer mass: tare 7.5 t plus payload up to 24.5 t."""
        return 7500.0 + self.payload_fraction * (TP.M2 - 7500.0)

    @property
    def cg_height(self) -> float:
        """Laden CoG rises with payload: empty deck 1.15 m, full box 1.95 m."""
        return 1.15 + self.payload_fraction * (TP.CG_HEIGHT_TRAILER - 1.15)

    @property
    def srt(self) -> float:
        """Static Rollover Threshold for THIS load state, in g."""
        return TP.ROLL_COMPLIANCE_FACTOR * TP.TRACK_WIDTH / (2.0 * self.cg_height)

    # The safety margin on the curvature ceiling. NOT a round number: it was
    # calibrated against closed-loop simulation. A ceiling derived from
    # steady-state lateral acceleration is a guarantee about the COMMANDED
    # PATH, and it does not transfer to a closed-loop guarantee. Two things sit
    # in the gap -- rearward amplification, which `kappa_ceiling_for_length`
    # now divides out, and the transient from correcting an initial
    # disturbance, which at low friction makes the MPC's linear model
    # over-command into saturated tyres. At margin 0.85 a random-action sweep
    # produced 2 rollovers in 400, worst LTR 1.30, all in low-mu scenarios with
    # a displaced start. See `calibrate_margin.py`.
    # NOTE: deliberately NOT annotated. An annotation inside a dataclass makes
    # this a FIELD, whose default is baked into __init__ at class-creation
    # time, so mutating Scenario.KAPPA_MARGIN would silently do nothing -- the
    # margin sweep returned seven identical rows before I caught it.
    KAPPA_MARGIN = 0.62

    @property
    def kappa_ceiling(self) -> float:
        """Steady-state ceiling, used for observations and reporting."""
        return min(TP.kappa_max_rollover(self.V, self.srt, self.KAPPA_MARGIN),
                   TP.kappa_max_friction(self.V, self.mu, self.KAPPA_MARGIN))

    def kappa_ceiling_for_length(self, L: float) -> float:
        """RWA-aware ceiling for a transition of along-road length L."""
        return TP.kappa_ceiling_for_length(self._cfg_cached(), self.srt, L,
                                           self.KAPPA_MARGIN)

    _cfg_memo = None

    def _cfg_cached(self) -> TTVConfig:
        if self._cfg_memo is None:
            object.__setattr__(self, "_cfg_memo", self.config())
        return self._cfg_memo

    @property
    def dy_min(self) -> float:
        """Smallest lateral offset that clears the obstacle with a margin.

        This is the number that makes a close obstacle survivable: clearing a
        1.2 m-wide obstacle needs 2.47 m of offset, not a full 3.6 m lane
        change, and 2.47 m fits in 61 m of road where 3.6 m needs 73 m. An
        action space that can only command a full lane change cannot solve a
        70 m obstacle at all -- measured: it collided in 50 % of close-obstacle
        scenarios until this was separated from `dy_preferred`.
        """
        return self.obstacle_half_width + TP.VEHICLE_WIDTH / 2.0 + 0.40

    @property
    def dy_preferred(self) -> float:
        """The tidy answer when there is room: a full lane change."""
        return max(self.dy_min, self.lane_width)

    @property
    def dy_required(self) -> float:          # kept for compatibility
        return self.dy_preferred

    def config(self) -> TTVConfig:
        cfg = TP.truck_config(V=self.V, mu=self.mu)
        cfg.m2 = self.m2
        cfg.Iz2 = self.m2 * (13.0 ** 2) / 12.0
        # Re-derive the static loads and hence the cornering stiffnesses.
        cfg.Fz1 = cfg.Fz2 = cfg.Fz3 = None
        cfg.finalize()
        cfg.C1 = TP.STEER_LOAD_TRANSFER_DERATE * TP.CN_STEER_FLAT * cfg.Fz1
        cfg.C2 = TP.CN_DUAL * cfg.Fz2
        cfg.C3 = TP.CN_TRIDEM * cfg.Fz3
        return cfg


# =============================================================================
# 2.  Truck kinematics and safety metrics
# =============================================================================

def truck_geometry(cfg: TTVConfig, xp: np.ndarray) -> Dict[str, float]:
    """Positions of the points that decide whether the rig stays in its lane.

    Sign convention, verified against the plant's own velocity transform:
        phi = psi2 - psi1      (so psi2 = psi1 + phi, and phi_dot = q)
    """
    X1, Y1, psi1, phi = xp[0], xp[1], xp[2], xp[5]
    psi2 = psi1 + phi
    Xh = X1 - cfg.c * math.cos(psi1)          # hitch, c behind the tractor CoG
    Yh = Y1 - cfg.c * math.sin(psi1)
    return {
        "psi2": psi2,
        "x_front": X1 + cfg.a1 * math.cos(psi1),
        "y_front": Y1 + cfg.a1 * math.sin(psi1),
        "x_drive": X1 - cfg.b1 * math.cos(psi1),
        "y_drive": Y1 - cfg.b1 * math.sin(psi1),
        "x_hitch": Xh, "y_hitch": Yh,
        "x_trailer_cg": Xh - cfg.a2 * math.cos(psi2),
        "y_trailer_cg": Yh - cfg.a2 * math.sin(psi2),
        "x_tridem": Xh - (cfg.a2 + cfg.b2) * math.cos(psi2),
        "y_tridem": Yh - (cfg.a2 + cfg.b2) * math.sin(psi2),
        "x_rear": Xh - 13.6 * math.cos(psi2),     # rear of the trailer body
        "y_rear": Yh - 13.6 * math.sin(psi2),
    }


def lateral_accelerations(cfg: TTVConfig, xp: np.ndarray, xdot: np.ndarray
                          ) -> Tuple[float, float]:
    """Body-frame lateral acceleration of the tractor and the trailer CoG.

    Tractor:  a_y1 = v1_dot + V*r1
    Trailer:  a_y2 = v2_dot + V2*r2,  with v2 differentiated in closed form
              from  v2 = -V sin(phi) + (v1 - c r1) cos(phi) - a2 r2.
    """
    V = cfg.V
    v1, r1, phi, q = xp[3], xp[4], xp[5], xp[6]
    v1d, r1d, qd = xdot[3], xdot[4], xdot[6]
    r2 = r1 + q
    r2d = r1d + qd
    sp, cp = math.sin(phi), math.cos(phi)

    V2 = V * cp + (v1 - cfg.c * r1) * sp
    a_y1 = v1d + V * r1
    v2d = (-V * cp * q
           + (v1d - cfg.c * r1d) * cp
           - (v1 - cfg.c * r1) * sp * q
           - cfg.a2 * r2d)
    a_y2 = v2d + V2 * r2
    return a_y1, a_y2


# =============================================================================
# 3.  Closed-loop rollout
# =============================================================================

@dataclass
class RolloutResult:
    ok: bool
    reason: str
    progress: float
    t: np.ndarray
    xp: np.ndarray                  # (8, n)
    delta_cmd: np.ndarray
    e_lat: np.ndarray
    e_psi: np.ndarray
    alpha: np.ndarray               # (3, n)
    a_y1: np.ndarray
    a_y2: np.ndarray
    ltr: np.ndarray
    y_front: np.ndarray
    y_tridem: np.ndarray
    y_rear: np.ndarray
    x_front: np.ndarray
    x_tridem: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    mpc_solve_us: float = 0.0


class _ForwardMatcher:
    """Forward-only nearest-point search on the path.

    Replaces the shipped global argmin, which is O(N) per step and whose match
    can jump backwards when the path doubles back near itself -- exactly what
    an out-and-back obstacle manoeuvre does.
    """

    def __init__(self, path: GeoPath, window: int = 400):
        self.p = path
        self.i = 0
        self.window = window

    def match(self, x: float, y: float) -> int:
        lo = self.i
        hi = min(self.p.s.size, self.i + self.window)
        d2 = (self.p.x[lo:hi] - x) ** 2 + (self.p.y[lo:hi] - y) ** 2
        self.i = lo + int(np.argmin(d2))
        return self.i


def rollout(path: GeoPath, cfg: TTVConfig, scen: Scenario,
            max_time: Optional[float] = None,
            mpc: Optional[CondensedMPC] = None) -> RolloutResult:
    """Drive the path with the tracking MPC and the nonlinear plant."""
    import time
    cfg.finalize()
    if mpc is None:
        mpc = CondensedMPC(cfg, use_rate_constraint=True)
    plant = PlantNP.from_cfg(cfg)          # validated == the CasADi plant

    T, N, V = cfg.T, cfg.N, cfg.V
    max_time = max_time if max_time is not None else cfg.maxEpisodeTime
    n_steps = max(1, int(round(max_time / T)))

    xp = np.zeros(8)
    xp[0] = path.x[0]
    xp[1] = path.y[0] + scen.e_y0
    xp[2] = path.psi[0] + scen.e_psi0
    xp[5] = scen.phi0

    matcher = _ForwardMatcher(path)
    srt_g = scen.srt * G

    rec = {k: [] for k in ("t", "delta", "e_lat", "e_psi", "a_y1", "a_y2",
                           "ltr", "y_front", "y_tridem", "y_rear",
                           "x_front", "x_tridem")}
    xps, alphas = [], []
    ok, reason = False, "time limit"
    solve_time = 0.0
    n_solves = 0
    s_now = path.s[0]

    for step in range(n_steps + 1):
        idx = matcher.match(xp[0], xp[1])
        s_now = path.s[idx]
        e_lat = (-math.sin(path.psi[idx]) * (xp[0] - path.x[idx])
                 + math.cos(path.psi[idx]) * (xp[1] - path.y[idx]))
        e_psi = math.atan2(math.sin(xp[2] - path.psi[idx]),
                           math.cos(xp[2] - path.psi[idx]))

        alpha = plant.slip_angles(xp)
        xdot = plant.dynamics(xp, xp[7])
        a_y1, a_y2 = plant.lateral_accels(xp, xdot)
        geo = truck_geometry(cfg, xp)

        rec["t"].append(step * T)
        rec["e_lat"].append(e_lat)
        rec["e_psi"].append(e_psi)
        rec["a_y1"].append(a_y1)
        rec["a_y2"].append(a_y2)
        rec["ltr"].append(abs(a_y2) / srt_g)
        for k in ("y_front", "y_tridem", "y_rear", "x_front", "x_tridem"):
            rec[k].append(geo[k])
        xps.append(xp.copy())
        alphas.append(alpha)

        if not np.all(np.isfinite(xp)):
            reason = "non-finite plant state"
            break
        if s_now >= path.s[-1] - cfg.finishTolerance:
            ok, reason = True, ""
            break
        if step == n_steps:
            reason = "time limit"
            break

        # ---- tracking MPC ------------------------------------------------
        s_prev = np.clip(s_now + V * T * np.arange(1, N + 1),
                         path.s[0], path.s[-1])
        y_ref = np.interp(s_prev, path.s, path.y)
        psi_ref = np.interp(s_prev, path.s, path.psi)
        xm = plant_to_mpc_state(xp, V)
        t0 = time.perf_counter()
        u, dbg = mpc.solve(xm, y_ref, psi_ref)
        solve_time += time.perf_counter() - t0
        n_solves += 1
        if not dbg.solved:
            reason = f"MPC solver: {dbg.status}"
            break
        delta_cmd = float(np.clip(u[0], -cfg.deltaMax, cfg.deltaMax))
        rec["delta"].append(delta_cmd)
        xp = plant.step(xp, delta_cmd, T, cfg.plantSubsteps)

    A = {k: np.asarray(v, dtype=float) for k, v in rec.items()}
    xp_arr = np.asarray(xps).T if xps else np.zeros((8, 0))
    al_arr = np.asarray(alphas).T if alphas else np.zeros((3, 0))
    progress = float(min(max(s_now / path.s[-1], 0.0), 1.0))

    res = RolloutResult(
        ok=ok, reason=reason, progress=progress,
        t=A["t"], xp=xp_arr, delta_cmd=A["delta"], e_lat=A["e_lat"],
        e_psi=A["e_psi"], alpha=al_arr, a_y1=A["a_y1"], a_y2=A["a_y2"],
        ltr=A["ltr"], y_front=A["y_front"], y_tridem=A["y_tridem"],
        y_rear=A["y_rear"], x_front=A["x_front"], x_tridem=A["x_tridem"],
        mpc_solve_us=1e6 * solve_time / max(n_solves, 1))
    res.metrics = truck_metrics(res, cfg, scen, path)
    return res


def truck_metrics(r: RolloutResult, cfg: TTVConfig, scen: Scenario,
                  path: GeoPath) -> Dict[str, float]:
    """The numbers a truck operator and a PBS assessor actually care about."""
    if r.t.size < 3:
        return {"empty": 1.0}
    peak1 = float(np.max(np.abs(r.a_y1)))
    peak2 = float(np.max(np.abs(r.a_y2)))
    dt = float(np.mean(np.diff(r.t))) if r.t.size > 1 else cfg.T

    # Swept path. Two different quantities, and conflating them is easy:
    #   swept_width   = INSTANTANEOUS lateral footprint of the rig, maximised
    #                   over time. This is the lane width the combination needs
    #                   and it is what off-tracking degrades.
    #   corridor_*    = the total lateral band the rig occupies over the whole
    #                   manoeuvre, which necessarily includes the intended
    #                   lane change and so is NOT a measure of tidiness.
    half = TP.VEHICLE_WIDTH / 2.0
    ys = np.vstack([r.y_front, r.y_tridem, r.y_rear])       # (3, n)
    inst_hi = ys.max(axis=0) + half
    inst_lo = ys.min(axis=0) - half
    swept_width = float(np.max(inst_hi - inst_lo))
    corridor_hi = float(np.max(inst_hi))
    corridor_lo = float(np.min(inst_lo))

    # Dynamic off-tracking: how far the tridem tracks outside the steer axle.
    # Compared at equal along-road position, so the trailer's lag is removed
    # and what is left is true off-tracking.
    if r.x_front.size > 4:
        order = np.argsort(r.x_tridem)
        y_tri_at_x = np.interp(r.x_front, r.x_tridem[order], r.y_tridem[order])
        offtrack = float(np.max(np.abs(y_tri_at_x - r.y_front)))
    else:
        offtrack = float("nan")

    jerk = float(np.max(np.abs(np.gradient(r.a_y1, dt)))) if r.t.size > 2 else 0.0

    m = {
        "peak_ay_tractor_g": peak1 / G,
        "peak_ay_trailer_g": peak2 / G,
        "peak_LTR": float(np.max(r.ltr)),
        "rollover": bool(np.max(r.ltr) >= 1.0),
        "RWA": peak2 / peak1 if peak1 > 1e-9 else float("nan"),
        "peak_lateral_jerk": jerk,
        "peak_articulation_deg": float(np.degrees(np.max(np.abs(r.xp[5, :])))),
        "peak_artic_rate_deg_s": float(np.degrees(np.max(np.abs(r.xp[6, :])))),
        "rms_lateral_error": float(np.sqrt(np.mean(r.e_lat ** 2))),
        "peak_lateral_error": float(np.max(np.abs(r.e_lat))),
        "peak_heading_error_deg": float(np.degrees(np.max(np.abs(r.e_psi)))),
        "max_slip_front_deg": float(np.degrees(np.max(np.abs(r.alpha[0, :])))),
        "max_slip_rear_deg": float(np.degrees(np.max(np.abs(r.alpha[1, :])))),
        "max_slip_trailer_deg": float(np.degrees(np.max(np.abs(r.alpha[2, :])))),
        "swept_width": swept_width,
        "corridor_hi": corridor_hi, "corridor_lo": corridor_lo,
        "dynamic_offtracking": offtrack,
        "peak_steer_deg": float(np.degrees(np.max(np.abs(r.delta_cmd))))
        if r.delta_cmd.size else 0.0,
        # Command rate and ACTUAL rate are different things. Only the actual
        # rate is physically limited (the plant rate-limits delta_dot with a
        # tanh); the command is a reference to a lagged actuator, so it may and
        # should step faster than delta_rate_max.
        "peak_cmd_rate_deg_s": float(np.degrees(
            np.max(np.abs(np.diff(r.delta_cmd) / cfg.T)))) if r.delta_cmd.size > 1 else 0.0,
        "peak_actual_steer_rate_deg_s": float(np.degrees(
            np.max(np.abs(np.diff(r.xp[7, :]) / cfg.T)))) if r.xp.shape[1] > 1 else 0.0,
        "manoeuvre_length": float(path.s[-1]),
        "path_kappa_peak": path.kappa_peak,
        "path_reward_kappa": path.reward_kappa(),
        "mpc_solve_us": r.mpc_solve_us,
    }

    # --- corridor and obstacle checks --------------------------------------
    y_min_allowed = -scen.lane_width / 2.0
    y_max_allowed = scen.lane_width * (scen.n_lanes - 0.5)
    m["lane_departure"] = float(max(0.0, corridor_hi - y_max_allowed,
                                    y_min_allowed - corridor_lo))

    # Obstacle: does any part of the rig enter its footprint?
    x0, x1 = scen.obstacle_x, scen.obstacle_x + scen.obstacle_len
    pen = 0.0
    for xa, ya in ((r.x_front, r.y_front), (r.x_tridem, r.y_tridem),
                   (r.x_tridem, r.y_rear)):
        inside = (xa >= x0) & (xa <= x1)
        if np.any(inside):
            pen = max(pen, float(np.max(scen.obstacle_half_width + half
                                        - ya[inside])))
    m["obstacle_incursion"] = max(0.0, pen)
    m["obstacle_clearance"] = -pen if pen < 0 else 0.0
    m["collision"] = bool(pen > 0.0)
    return m


# =============================================================================
# 4.  Action <-> path
# =============================================================================

ACTION_DIM = 9
ACTION_NAMES = ("t_start", "L_out", "q1_out", "q2_out",
                "L_hold", "L_back", "q1_back", "q2_back", "dy_trim")


def decode_action(a: np.ndarray, scen: Scenario) -> Dict[str, Any]:
    """Map a in [-1,1]^9 to path parameters, with the safety limit BAKED IN.

    The length of every transition is scaled from `min_length_for_offset`
    upwards, so the peak curvature of any decodable action is at or below the
    rollover / friction ceiling for this speed, friction and load state. The
    agent cannot emit a path that rolls the truck; it can only choose how much
    margin to leave and how to shape the manoeuvre.
    """
    a = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
    u = 0.5 * (a + 1.0)                                    # -> [0,1]

    LEAD_MIN = 5.0

    q1o = 0.20 + 0.60 * u[2]
    q2o = 0.20 + 0.60 * u[3]
    q1b = 0.20 + 0.60 * u[6]
    q2b = 0.20 + 0.60 * u[7]

    # ---- how much road is there before the obstacle? ----------------------
    room = max(1.0, scen.obstacle_x - LEAD_MIN)

    # ---- choose the offset, then the length, so that BOTH the rollover and
    #      the collision constraint are satisfied by construction ------------
    # dy in [dy_min, dy_preferred + 0.6]; u[8] picks where.
    dy_lo, dy_hi = scen.dy_min, scen.dy_preferred + 0.6
    dy = dy_lo + (dy_hi - dy_lo) * u[8]

    # The ceiling depends on the manoeuvre length (through RWA) and the length
    # depends on the ceiling, so this is a fixed point. It converges in two or
    # three passes because RWA varies slowly with frequency.
    kcap = scen.kappa_ceiling
    for _ in range(3):
        L_try = min_length_for_offset(dy, kcap, q1o, q2o)
        kcap_new = scen.kappa_ceiling_for_length(L_try)
        if abs(kcap_new - kcap) < 1e-7 * max(kcap, 1e-9):
            kcap = kcap_new
            break
        kcap = kcap_new
    L_min_out = min_length_for_offset(dy, kcap, q1o, q2o)

    if L_min_out > room:
        # The requested offset cannot be reached inside the rollover ceiling
        # before the obstacle. Peak kappa scales as |dy|/L^2, so the largest
        # offset that DOES fit is dy * (room/L_min)^2. Shrink to it, floored at
        # the minimum clearance -- if even that does not fit the scenario is
        # geometrically impossible and the reward will say so.
        dy_fit = max(dy_lo, dy * (room / L_min_out) ** 2)
        dy = dy_fit
        L_min_out = min_length_for_offset(dy, kcap, q1o, q2o)

    L_max_out = max(L_min_out, min(2.4 * L_min_out, room))
    L_out = L_min_out + (L_max_out - L_min_out) * u[1]

    L_min_back = min_length_for_offset(-dy, kcap, q1b, q2b)
    L_back = L_min_back + 1.4 * L_min_back * u[5]

    # Start: u[0] chooses how much of the remaining room to use as lead-in.
    lead_max = max(LEAD_MIN, scen.obstacle_x - L_out - LEAD_MIN)
    lead_in = LEAD_MIN + (lead_max - LEAD_MIN) * u[0]

    hold_min = max(0.0, (scen.obstacle_x + scen.obstacle_len + 5.0)
                   - (lead_in + L_out))
    L_hold = hold_min + 60.0 * u[4]

    trans: List[Tuple[float, float, float, float]] = [(dy, L_out, q1o, q2o)]
    if scen.must_return:
        trans.append((0.0, L_hold, 0.0, 0.0))
        trans.append((-dy, L_back, q1b, q2b))
    return {"lead_in": lead_in, "transitions": trans, "dy": dy,
            "L_out": L_out, "L_hold": L_hold, "L_back": L_back,
            "kappa_ceiling": kcap, "room": room,
            "L_min_out": L_min_out, "L_min_back": L_min_back,
            "dy_shrunk": dy < scen.dy_preferred - 1e-9}


def action_to_path(a: np.ndarray, scen: Scenario, ds: float = 0.5) -> GeoPath:
    d = decode_action(a, scen)
    used = d["lead_in"] + sum(t[1] for t in d["transitions"])
    tail = max(20.0, min(60.0, scen.road_length - used))
    return bezier_offset_path(lead_in=d["lead_in"], transitions=d["transitions"],
                              tail=tail, y_start=0.0, ds=ds)


# =============================================================================
# 5.  Observation
# =============================================================================

OBS_DIM = 14
OBS_NAMES = ("V_norm", "mu_norm", "obstacle_dist", "obstacle_half_width",
             "dy_required", "lane_width", "n_lanes", "must_return",
             "e_y0", "e_psi0", "phi0", "kappa_ceiling_norm", "payload",
             "room_ratio")


def observe(scen: Scenario) -> np.ndarray:
    """13 numbers, each scaled to roughly [0,1] as the paper does.

    Two are not in the paper's 11 and are the important additions:
      * `payload`  -- the same truck's SRT nearly doubles between laden and
        empty, so a policy that cannot see the load state cannot be safe in
        both. This is the single most valuable truck-specific observation.
      * `kappa_ceiling_norm` -- the curvature limit implied by (V, mu, load).
        Handing the agent the constraint directly, instead of making it infer
        the constraint from three other inputs, is what lets a small network
        generalise over the randomisation ranges.
    """
    kc = scen.kappa_ceiling
    # room_ratio: how much of the shortest rollover-safe FULL lane change fits
    # before the obstacle. Below 1.0 the manoeuvre must be partial, and this is
    # the one number that says so directly, instead of leaving the agent to
    # infer it from obstacle distance, speed, friction and load at once.
    L_full = min_length_for_offset(scen.dy_preferred, kc, 0.5, 0.5)
    room_ratio = np.clip(max(1.0, scen.obstacle_x - 5.0) / L_full, 0.0, 3.0) / 3.0
    return np.array([
        (scen.V - TP.V_RANGE[0]) / (TP.V_RANGE[1] - TP.V_RANGE[0]),
        (scen.mu - TP.MU_RANGE[0]) / (TP.MU_RANGE[1] - TP.MU_RANGE[0]),
        np.clip(scen.obstacle_x / 300.0, 0, 1),
        np.clip(scen.obstacle_half_width / 2.0, 0, 1),
        np.clip(scen.dy_required / 6.0, 0, 1),
        np.clip((scen.lane_width - 3.0) / 1.2, 0, 1),
        (scen.n_lanes - 1) / 2.0,
        1.0 if scen.must_return else 0.0,
        np.clip((scen.e_y0 + 1.0) / 2.0, 0, 1),
        np.clip((scen.e_psi0 + 0.05) / 0.10, 0, 1),
        np.clip((scen.phi0 + 0.05) / 0.10, 0, 1),
        np.clip(kc / 0.015, 0, 1),
        scen.payload_fraction,
        float(room_ratio),
    ], dtype=np.float64)


# =============================================================================
# 6.  Reward
# =============================================================================

@dataclass
class RewardWeights:
    """Every term is normalised to [0,1] BEFORE weighting, so the weights are
    the actual relative priorities. In the shipped reward the slip term is 300x
    the curvature term (measured), so the two nominal 0.5 weights do not mean
    what they look like.
    """
    rollover: float = 0.30      # LTR margin -- the binding truck constraint
    rwa: float = 0.15           # rearward amplification (PBS <= 2.0)
    swept: float = 0.15         # lane width actually consumed by the rig
    jerk: float = 0.10          # lateral jerk: cargo shift + driver comfort
    tracking: float = 0.10      # can the MPC actually follow this path
    brevity: float = 0.10       # shorter manoeuvre = earlier return to lane
    slip: float = 0.10          # keep the paper's tyre-slip term, rescaled

    # failure shaping
    fail_base: float = -1.0
    fail_progress_bonus: float = 0.35   # -1.0 .. -0.65 with progress
    collision_penalty: float = -1.5
    rollover_penalty: float = -2.0

    # references
    jerk_ref: float = 4.0       # [m/s^3]
    swept_ref: float = 3.2      # [m]
    track_ref: float = 0.25     # [m] RMS lateral error
    length_ref: float = 320.0   # [m] manoeuvre-length budget

    def sum_success(self) -> float:
        return (self.rollover + self.rwa + self.swept + self.jerk
                + self.tracking + self.brevity + self.slip)


def slip_reference_truck(V: float, mu: float) -> float:
    """Replacement for Eq. (3).

    The paper fits  mu_max = 0.0037*exp(0.0693*v0[km/h])  to car data over
    40-60 km/h. Extrapolated to 86 km/h it returns 0.54 rad = 31 deg, which is
    (a) past the 0.2 rad failure limit it is compared against and (b) far past
    where a truck tyre saturates. Measured in validate_port.py.

    A truck tyre's peak-force slip angle is set by grip, not by speed: it is
    where C*alpha reaches mu*Fz, i.e. alpha_peak = mu*Fz/C = mu/CN. With
    CN ~ 4-6 1/rad that is 0.15-0.22 rad of *saturation*; the usable reference
    is a fraction of it. This form is speed-independent and friction-dependent,
    which is the right way round.
    """
    return 0.55 * mu / TP.CN_STEER_FLAT


def compute_reward(res: RolloutResult, scen: Scenario, path: GeoPath,
                   w: RewardWeights = RewardWeights()) -> Tuple[float, Dict[str, float]]:
    m = res.metrics
    parts: Dict[str, float] = {}

    if m.get("rollover", False):
        total = w.rollover_penalty
        parts["terminal"] = total
        parts["cause"] = "rollover"
        return total, parts
    if m.get("collision", False):
        total = w.collision_penalty
        parts["terminal"] = total
        parts["cause"] = "collision"
        return total, parts
    if not res.ok:
        total = w.fail_base + w.fail_progress_bonus * res.progress
        parts["terminal"] = total
        parts["cause"] = res.reason
        parts["progress"] = res.progress
        return total, parts

    # ---- success: every component in [0,1] --------------------------------
    r_roll = float(np.clip(1.0 - m["peak_LTR"], 0.0, 1.0))
    r_rwa = float(np.clip(2.0 - m["RWA"], 0.0, 1.0))
    r_swept = float(np.clip(1.0 - (m["swept_width"] - TP.VEHICLE_WIDTH)
                            / (w.swept_ref - TP.VEHICLE_WIDTH), 0.0, 1.0))
    r_jerk = float(math.exp(-m["peak_lateral_jerk"] / w.jerk_ref))
    r_track = float(math.exp(-m["rms_lateral_error"] / w.track_ref))
    r_brev = float(np.clip(1.0 - m["manoeuvre_length"] / w.length_ref, 0.0, 1.0))

    a_ref = slip_reference_truck(scen.V, scen.mu)
    slip_worst = math.radians(max(m["max_slip_front_deg"], m["max_slip_rear_deg"]))
    r_slip = float(np.clip(1.0 - slip_worst / a_ref, 0.0, 1.0))

    # lane departure is a hard-ish penalty even on an otherwise clean run
    dep = m["lane_departure"]
    pen_dep = -2.0 * min(dep, 0.5) if dep > 0 else 0.0

    total = (w.rollover * r_roll + w.rwa * r_rwa + w.swept * r_swept
             + w.jerk * r_jerk + w.tracking * r_track + w.brevity * r_brev
             + w.slip * r_slip + pen_dep)

    parts.update({"r_rollover": r_roll, "r_rwa": r_rwa, "r_swept": r_swept,
                  "r_jerk": r_jerk, "r_tracking": r_track, "r_brevity": r_brev,
                  "r_slip": r_slip, "pen_lane_departure": pen_dep,
                  "cause": "success"})
    return float(total), parts


# =============================================================================
# 7.  The environment
# =============================================================================

class TruckHighwayEnv:
    """Gymnasium-style, but dependency-free.

    mode = "bandit"   : one action -> one whole path -> one reward (the paper)
    mode = "receding" : re-plan every `replan_period` s from the measured state
    """

    def __init__(self, mode: str = "bandit", seed: int = 0,
                 weights: RewardWeights = RewardWeights(),
                 randomise: bool = True, ds: float = 0.5,
                 replan_period: float = 1.0):
        assert mode in ("bandit", "receding")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.w = weights
        self.randomise = randomise
        self.ds = ds
        self.replan_period = replan_period
        self._mpc_cache: Dict[tuple, CondensedMPC] = {}
        self.scen: Optional[Scenario] = None

    # ------------------------------------------------------------------
    def sample_scenario(self) -> Scenario:
        """Domain randomisation, the truck analogue of the paper's Table 2."""
        if not self.randomise:
            return Scenario()
        r = self.rng
        # obstacle_x reaches down to 70 m. The first version of this sampler
        # started at 110 m, and the resulting policy collided in 50 % of
        # scenarios with the obstacle at 70-90 m: it had never seen a case
        # where the manoeuvre had to be short, so it always chose a long, gentle
        # one. Widening the range is half the fix; letting the action space
        # express a partial lateral offset (see decode_action) is the other half.
        return Scenario(
            V=float(r.uniform(*TP.V_RANGE)),
            mu=float(r.uniform(*TP.MU_RANGE)),
            lane_width=float(r.uniform(3.25, 3.75)),
            obstacle_x=float(r.uniform(70.0, 240.0)),
            obstacle_len=float(r.uniform(6.0, 18.0)),
            obstacle_half_width=float(r.uniform(0.8, 1.6)),
            must_return=bool(r.random() < 0.7),
            payload_fraction=float(r.uniform(0.0, 1.0)),
            e_y0=float(r.normal(0.0, 0.20)),
            e_psi0=float(r.normal(0.0, 0.008)),
            phi0=float(r.normal(0.0, 0.008)),
        )

    def _mpc_for(self, cfg: TTVConfig) -> CondensedMPC:
        key = (round(cfg.V, 4), round(cfg.m2, 2), round(cfg.mu, 4))
        if key not in self._mpc_cache:
            if len(self._mpc_cache) > 64:
                self._mpc_cache.clear()
            self._mpc_cache[key] = CondensedMPC(cfg, use_rate_constraint=True)
        return self._mpc_cache[key]

    # ------------------------------------------------------------------
    def reset(self, scen: Optional[Scenario] = None) -> np.ndarray:
        self.scen = scen if scen is not None else self.sample_scenario()
        return observe(self.scen)

    def step(self, action: np.ndarray):
        assert self.scen is not None, "call reset() first"
        scen = self.scen
        cfg = scen.config()
        path = action_to_path(action, scen, ds=self.ds)
        budget = 1.3 * path.s[-1] / scen.V + 2.0
        res = rollout(path, cfg, scen, max_time=budget, mpc=self._mpc_for(cfg))
        reward, parts = compute_reward(res, scen, path, self.w)
        info = {"metrics": res.metrics, "parts": parts, "path": path,
                "rollout": res, "scenario": scen,
                "decoded": decode_action(action, scen)}
        return observe(scen), float(reward), True, info


# =============================================================================
if __name__ == "__main__":
    import json
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("Highway truck environment: smoke test")
    print("=" * 78)

    scen = Scenario()
    cfg = scen.config()
    print(f"  scenario: V={scen.V} m/s, mu={scen.mu}, payload={scen.payload_fraction}")
    print(f"  SRT = {scen.srt:.3f} g, kappa ceiling = {scen.kappa_ceiling:.5f} 1/m")
    print(f"  dy required = {scen.dy_required:.2f} m")
    print(f"  shortest safe out-transition = "
          f"{min_length_for_offset(scen.dy_required, scen.kappa_ceiling, .5, .5):.1f} m")

    env = TruckHighwayEnv(randomise=False)
    obs = env.reset(scen)
    print(f"\n  obs ({len(obs)}) = {obs}")

    print(f"\n  {'action':>26s} {'reward':>8s} {'LTR':>6s} {'RWA':>6s} "
          f"{'swept':>7s} {'ay2[g]':>7s} {'jerk':>6s} {'L[m]':>6s} {'ok':>6s}")
    for name, a in [("neutral (all 0)", np.zeros(9)),
                    ("shortest manoeuvre", np.array([-1, -1, 0, 0, -1, -1, 0, 0, -1.])),
                    ("longest manoeuvre", np.array([0, 1, 0, 0, 0, 1, 0, 0, 0.])),
                    ("late + short", np.array([1, -1, 0, 0, -1, -1, 0, 0, 0.])),
                    ("max clearance", np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.]))]:
        _, rw, _, info = env.step(a)
        m = info["metrics"]
        print(f"  {name:>26s} {rw:8.4f} {m['peak_LTR']:6.3f} {m['RWA']:6.3f} "
              f"{m['swept_width']:7.3f} {m['peak_ay_trailer_g']:7.3f} "
              f"{m['peak_lateral_jerk']:6.2f} {m['manoeuvre_length']:6.0f} "
              f"{str(info['parts']['cause'])[:6]:>6s}")

    print(f"\n  MPC solve time: {info['metrics']['mpc_solve_us']:.0f} us/step")

    print("\n  Can any action roll the truck over? (400 random actions, "
          "randomised scenarios)")
    env2 = TruckHighwayEnv(randomise=True, seed=1)
    rng = np.random.default_rng(2)
    worst_ltr, worst_kappa_ratio, n_roll, n_ok = 0.0, 0.0, 0, 0
    causes: Dict[str, int] = {}
    rewards = []
    for i in range(400):
        env2.reset()
        a = rng.uniform(-1, 1, 9)
        _, rw, _, info = env2.step(a)
        m = info["metrics"]
        worst_ltr = max(worst_ltr, m["peak_LTR"])
        worst_kappa_ratio = max(worst_kappa_ratio,
                                m["path_kappa_peak"] / info["decoded"]["kappa_ceiling"])
        n_roll += int(m["rollover"])
        n_ok += int(info["parts"]["cause"] == "success")
        causes[str(info["parts"]["cause"])[:28]] = causes.get(
            str(info["parts"]["cause"])[:28], 0) + 1
        rewards.append(rw)
    print(f"    worst peak LTR over 400 random actions : {worst_ltr:.3f}")
    print(f"    worst  kappa_peak / kappa_ceiling       : {worst_kappa_ratio:.4f}")
    print(f"    rollovers                               : {n_roll} / 400")
    print(f"    successes                               : {n_ok} / 400")
    print(f"    reward  mean {np.mean(rewards):+.4f}  min {np.min(rewards):+.4f} "
          f"max {np.max(rewards):+.4f}")
    print("    outcome breakdown:")
    for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
        print(f"      {k:30s} {v:5d}")

    json.dump({"worst_ltr": worst_ltr, "worst_kappa_ratio": worst_kappa_ratio,
               "n_rollover": n_roll, "n_success": n_ok,
               "reward_mean": float(np.mean(rewards)),
               "reward_min": float(np.min(rewards)),
               "reward_max": float(np.max(rewards)),
               "causes": causes},
              open("out/env_smoke.json", "w"), indent=2)
    print("\n  wrote out/env_smoke.json")
