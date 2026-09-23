"""
truck_params.py -- A highway tractor-semitrailer, with every number justified
============================================================================

The MATLAB file ships an articulated vehicle of 12.4 t total. A European
highway tractor-semitrailer at legal maximum is 40 t. That is not a tweak: it
changes the mass by 3.2x, the trailer yaw inertia by 2.5x, and -- most
importantly -- it moves the binding safety constraint from TYRE SLIP to
ROLLOVER. Every parameter below is derived from a legal/physical constraint
rather than copied, and `audit()` checks the derivation.

Legal envelope used (EU Directive 96/53/EC, 5-axle articulated combination):
    steer axle           <=  7.5 t
    drive axle (dual)    <= 11.5 t
    trailer tridem       <= 24.0 t
    gross combination    <= 40.0 t
    vehicle width        <=  2.55 m
    trailer length       <= 13.60 m (kingpin to rear)

Design targets chosen inside that envelope (a typical fully-laden rig):
    steer  7.10 t | drive 11.40 t | tridem 21.50 t | GCW 40.00 t
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ttv_core import TTVConfig, MPCConfig, RewardConfig, build_linear_prediction_model

G = 9.81

# =============================================================================
# Geometry / mass, derived from the axle-load targets
# =============================================================================
#
# Tractor 4x2:   wheelbase L1 = 3.80 m
# Semitrailer:   kingpin -> tridem centre  L2 = 7.70 m, overall 13.6 m
#
# Unknowns: a1,b1 (tractor CoG to front/rear axle), c (tractor CoG to kingpin),
#           a2,b2 (kingpin to trailer CoG, trailer CoG to tridem centre).
#
# Trailer split from the tridem-load target:
#     tridem = m2*g*a2/(a2+b2)      ->  a2 = L2 * (tridem / m2)
# Tractor split from the steer-load target, with the kingpin load W_k applied
# at distance c behind the tractor CoG:
#     Fz_steer = (b1*m1*g + (b1-c)*W_k) / L1
#
# We fix the *bobtail* front/rear split of the tractor alone at 62 % front
# (engine + cab forward; real bobtail tractors sit at 60-70 % front), which
# pins b1, and then solve for c. The resulting kingpin position is reported by
# audit() and must land 0-0.8 m ahead of the drive axle to be realistic.

L1 = 3.80          # tractor wheelbase                      [m]
L2 = 7.70          # kingpin to tridem centre                [m]
BOBTAIL_FRONT_SHARE = 0.62

M_STEER = 7100.0   # target static steer-axle mass           [kg]
M_DRIVE = 11400.0  # target static drive-axle mass           [kg]
M_TRIDEM = 21500.0  # target static tridem mass              [kg]
M_GCW = M_STEER + M_DRIVE + M_TRIDEM          # 40 000 kg

M2 = 32000.0                   # semitrailer + payload        [kg]
M1 = M_GCW - M2                # tractor                      [kg]
W_KINGPIN = (M2 - M_TRIDEM) * G   # vertical kingpin load     [N]

B1 = BOBTAIL_FRONT_SHARE * L1                      # CoG -> rear axle
A1 = L1 - B1                                       # CoG -> front axle
C_HITCH = (B1 * (M1 * G + W_KINGPIN) - M_STEER * G * L1) / W_KINGPIN

A2 = L2 * (M_TRIDEM / M2)      # kingpin -> trailer CoG
B2 = L2 - A2                   # trailer CoG -> tridem centre

# --- yaw inertias -----------------------------------------------------------
# Tractor: radius of gyration ~ 0.5 * wheelbase for a cab-over tractor, which
# puts Iz1 in the 25-35e3 kg m^2 band reported for 4x2 tractors.
IZ1 = M1 * (0.50 * L1) ** 2                       # ~ 28 900 kg m^2
# Laden 13.6 m box semitrailer, payload roughly uniform over 13.0 m:
#   I_cg = m * Lbox^2 / 12
IZ2 = M2 * (13.0 ** 2) / 12.0                      # ~ 450 000 kg m^2

# --- cornering stiffness ----------------------------------------------------
# Heavy-truck radials: normalised cornering stiffness C_alpha/Fz is 5-7 1/rad
# on flat ground (passenger cars are 10-14). Dual assemblies scrub against each
# other and lose ~15 %; a lumped tridem loses more, because the outer axles run
# at slip angles set by their distance from the turn centre.
CN_STEER_FLAT = 6.2     # [1/rad] 385/65R22.5 single, at ~35 kN per tyre
CN_DUAL = 5.2           # [1/rad] drive axle, duals, incl. scrub derate
CN_TRIDEM = 4.6         # [1/rad] lumped tridem, incl. scrub derate
#
# The steer axle needs an extra EFFECTIVE derate that a yaw-plane model cannot
# generate by itself. Under lateral acceleration the combination transfers load
# laterally, and because Fy(Fz) is concave, an axle loses cornering stiffness
# in proportion to how much transfer it sees. The steer axle has by far the
# highest roll stiffness per unit load (narrow track, stiff front suspension,
# no air bags to soften it), so it sheds the most. Together with front
# roll-steer this is the documented reason a laden artic measures as
# UNDERSTEERING even though flat-ground tyre data alone predicts near-neutral.
# Folding it into C1 is the standard way to keep a 2-D model honest; the
# alternative is to add a roll degree of freedom, which this model does not have.
STEER_LOAD_TRANSFER_DERATE = 0.65

FZ1 = M_STEER * G
FZ2 = M_DRIVE * G
FZ3 = M_TRIDEM * G
C1 = STEER_LOAD_TRANSFER_DERATE * CN_STEER_FLAT * FZ1
C2 = CN_DUAL * FZ2
C3 = CN_TRIDEM * FZ3

# --- roll / rollover --------------------------------------------------------
# Static Rollover Threshold. Rigid-axle geometry gives SRT = t/(2h); real rigs
# lose 25-35 % to suspension + tyre + fifth-wheel roll compliance.
TRACK_WIDTH = 2.04          # [m] dual-tyre centre-to-centre track
CG_HEIGHT_TRAILER = 1.95    # [m] laden box trailer CoG height
ROLL_COMPLIANCE_FACTOR = 0.70
SRT_RIGID = TRACK_WIDTH / (2.0 * CG_HEIGHT_TRAILER)          # ~0.523 g
SRT = ROLL_COMPLIANCE_FACTOR * SRT_RIGID                      # ~0.366 g

# --- steering actuator ------------------------------------------------------
# Truck steering box ratio ~ 20:1, hydraulically assisted. A driver turning the
# handwheel at 400 deg/s gives 20 deg/s = 0.35 rad/s at the road wheel. The
# hydraulic circuit is slower than a sports car's EPS, hence 0.25 s not 0.15 s.
DELTA_MAX = 0.30           # [rad] 17 deg road wheel (evasive)
DELTA_RATE_MAX = 0.35      # [rad/s]
STEER_TAU = 0.25           # [s]

# --- highway operating point ------------------------------------------------
V_HIGHWAY = 24.0           # [m/s] 86 km/h (EU trucks are limited to 90 km/h)
V_RANGE = (19.0, 25.0)     # [m/s] 68-90 km/h
MU_RANGE = (0.35, 0.90)    # wet worn asphalt .. dry asphalt
LANE_WIDTH = 3.60          # [m] EU motorway lane
VEHICLE_WIDTH = 2.55       # [m]


# =============================================================================
def truck_config(V: float = V_HIGHWAY, mu: float = 0.85, **overrides) -> TTVConfig:
    """A TTVConfig for the laden highway rig.

    The MPC weights and reward limits are re-tuned for a truck; see the guide
    for the reason behind each change.
    """
    cfg = TTVConfig(
        # --- horizon ---------------------------------------------------------
        # A 40 t rig needs ~3 s of preview: the trailer yaw mode has a period of
        # ~2.5 s, so a 1 s horizon (the shipped N*T) cannot see the trailer
        # swing it is about to provoke.
        # T and N chosen by measurement (tune_mpc.py): 2.0 s of preview is where
        # tracking error stops improving on a manoeuvre at the curvature
        # ceiling; 1.0 s (the shipped N*T) leaves 0.26 m of peak lateral error
        # and 0.82 peak LTR, 2.0 s gives 0.19 m and 0.69. A 0.1 s control period
        # sits comfortably inside the 0.25 s steering lag.
        T=0.1, N=20, V=V, maxEpisodeTime=25.0, plantSubsteps=5,
        minPathLength=20.0, finishTolerance=1.0,
        # --- actuator --------------------------------------------------------
        deltaMax=DELTA_MAX, deltaRateMax=DELTA_RATE_MAX,
        steeringTimeConstant=STEER_TAU,
        # Jack-knife guard. The ANGLE is the safety limit; the rate limit is a
        # precursor guard, set from what a highway lane change actually needs
        # (measured peak ~9 deg/s) with margin, not from a round number.
        phiMax=math.radians(8.0), qMax=math.radians(20.0),
        # --- mass / geometry -------------------------------------------------
        m1=M1, m2=M2, a1=A1, a2=A2, b1=B1, b2=B2, c=C_HITCH,
        Iz1=IZ1, Iz2=IZ2, C1=C1, C2=C2, C3=C3, mu=mu, g=G,
        Fz1=FZ1, Fz2=FZ2, Fz3=FZ3,
        mpc=MPCConfig(
            Qy=4.0,        # 3.6 m lane, 2.55 m body -> 0.5 m margin: track hard
            Qpsi=2.0,
            # Was 0.0 -- the shipped MPC never damps articulation, it only logs
            # it. Sweeping these (tune_mpc.py) on a manoeuvre at the curvature
            # ceiling: 0/0 leaves the QP unable to converge at all once the
            # articulation-rate guard activates; 25/8 converges and trades
            # 0.09 m of RMS tracking error for a materially lower LTR. Since
            # 0.09 m is nothing on a 3.6 m lane and LTR is the binding limit,
            # the trade is worth making.
            Qphi=25.0,
            Qq=8.0,
            Rdelta=0.5,    # slow steering: penalise effort more than a car
            maxIterations=300, acceptableTolerance=1e-7,
        ),
        reward=RewardConfig(
            # Truck tyres saturate at 4-6 deg, not 11.5 deg. A 0.2 rad limit is
            # deep in saturation and would be reached only after the rig has
            # already rolled over, so it can never bind. Use 6 deg.
            maxLateralSlip=math.radians(6.0),
            maxDistanceError=0.5,                 # (3.60-2.55)/2 = 0.525 m
            maxHeadingError=math.radians(10.0),   # 20 deg is a jack-knife
            wCurvature=0.5, wSlip=0.5, cKappaDD=0.0,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg.finalize()


# =============================================================================
def kappa_max_rollover(V: float, srt: float = SRT, margin: float = 0.85) -> float:
    """Largest path curvature that keeps the rig below the rollover threshold.

        a_y = V^2 * kappa <= margin * SRT * g     ->   kappa <= margin*SRT*g/V^2

    This is the single most useful number for the RL action space: bound the
    commanded curvature by it and NO action the agent can emit will roll the
    truck over. The safety constraint moves from the reward into the geometry.
    """
    return margin * srt * G / (V ** 2)


def kappa_max_friction(V: float, mu: float, margin: float = 0.85) -> float:
    """Curvature ceiling from tyre grip (the constraint a car would hit first)."""
    return margin * mu * G / (V ** 2)


_RWA_CACHE: Dict[tuple, tuple] = {}


def rwa_lookup(cfg: TTVConfig, n: int = 64):
    """Cached RWA(f) table for this vehicle, 0.02-1.2 Hz, for interpolation."""
    key = (round(cfg.V, 4), round(cfg.m2, 2), round(cfg.Iz2, 2),
           round(cfg.C3, 2), round(cfg.C1, 2))
    t = _RWA_CACHE.get(key)
    if t is None:
        if len(_RWA_CACHE) > 512:
            _RWA_CACHE.clear()
        f = np.linspace(0.02, 1.2, n)
        r = np.array([rearward_amplification(cfg, fi) for fi in f])
        t = (f, r)
        _RWA_CACHE[key] = t
    return t


def kappa_ceiling_for_length(cfg: TTVConfig, srt: float, L: float,
                             margin: float) -> float:
    """Curvature ceiling that accounts for REARWARD AMPLIFICATION.

    The steady-state ceiling `margin*SRT*g/V^2` limits the TRACTOR. The trailer
    is the unit that rolls, and it sees more: at the frequency a transition of
    length L excites, f = V/L (one full steer-in/steer-out cycle per
    transition), the amplification is RWA(f). So the tractor must be held to
    SRT/RWA, not SRT.

    Measured in hard_cases.py: RWA runs 1.03 at L = 250 m to 1.25 at L = 75 m,
    and the closed-loop rollouts came out at 1.04-1.27, which brackets exactly
    that -- so the estimate is the right one.
    """
    f = cfg.V / max(L, 1.0)
    fs, rs = rwa_lookup(cfg)
    rwa = float(np.interp(f, fs, rs))
    rwa = max(rwa, 1.0)                      # never claim the trailer helps
    eff_srt = srt / rwa
    return margin * min(eff_srt, cfg.mu) * G / (cfg.V ** 2)


def binding_constraint(V: float, mu: float) -> Dict[str, float]:
    kr = kappa_max_rollover(V)
    kf = kappa_max_friction(V, mu)
    return {"kappa_rollover": kr, "kappa_friction": kf,
            "binding": "rollover" if kr < kf else "friction",
            "ratio": kf / kr}


# =============================================================================
def _descriptor_model(cfg: TTVConfig):
    """Return (Av, Bv) of the BODY-FRAME descriptor model  z_dot = Av z + Bv d
    with z = [r1 ; beta ; q ; phi].

    Handling and rearward-amplification numbers must be computed here, NOT from
    the 7-state Ac. Ac is written in a ROAD-FIXED frame where y and psi are
    absolute, so it has two eigenvalues at the origin and cannot represent
    steady-state cornering (psi grows without bound). Extracting an understeer
    gradient from Ac silently returns zero.
    """
    C1s, C2s, C3s = -abs(cfg.C1), -abs(cfg.C2), -abs(cfg.C3)
    V = cfg.V
    M = np.array([
        [cfg.Iz1, cfg.m1 * cfg.c * V, 0.0, 0.0],
        [-cfg.m2 * (cfg.a2 + cfg.c), (cfg.m1 + cfg.m2) * V, -cfg.m2 * cfg.a2,
         C3s * (cfg.a2 + cfg.b2) / V],
        [cfg.Iz2, cfg.m1 * cfg.a2 * V, cfg.Iz2,
         -C3s * cfg.b2 * (cfg.a2 + cfg.b2) / V],
        [0.0, 0.0, 0.0, 1.0]])
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
        [0.0, 0.0, -1.0, 0.0]])
    G4 = np.array([-C1s * (cfg.a1 + cfg.c), -C1s, -C1s * cfg.a2, 0.0])
    return -np.linalg.solve(M, K), np.linalg.solve(M, G4)


def handling_metrics(cfg: TTVConfig) -> Dict[str, float]:
    """Linear-model handling numbers: are these parameters a sane vehicle?"""
    Av, Bv = _descriptor_model(cfg)
    ev = np.linalg.eigvals(Av)
    V = cfg.V

    # Steady state at delta = 1 rad:  Av z + Bv = 0
    z_ss = np.linalg.solve(Av, -Bv)          # [r1, beta, q, phi]
    r_gain = float(z_ss[0])                  # yaw rate per rad steer  [1/s]
    ay_gain = V * r_gain                     # lateral accel per rad   [m/s^2]
    phi_ss = float(z_ss[3])

    # Understeer gradient K from  delta = L*kappa + K*a_y   (delta = 1 rad here)
    L = cfg.a1 + cfg.b1
    kappa = r_gain / V
    understeer = (1.0 - L * kappa) / ay_gain if abs(ay_gain) > 0 else float("nan")

    damped = [e for e in ev if abs(e.imag) > 1e-6]
    if damped:
        slowest = min(damped, key=lambda e: abs(e.real))
        zeta = -slowest.real / abs(slowest)
        wn = abs(slowest)
    else:
        zeta, wn = float("nan"), float("nan")

    return {
        "eig_real_min": float(np.min(ev.real)),
        "eig_real_max": float(np.max(ev.real)),
        "stable": bool(np.all(ev.real < -1e-9)),
        "slowest_time_constant_s": float(-1.0 / np.max(ev.real))
        if np.max(ev.real) < 0 else float("inf"),
        "trailer_mode_damping_ratio": float(zeta),
        "trailer_mode_natural_freq_rad_s": float(wn),
        "trailer_mode_period_s": float(2 * math.pi / wn)
        if wn == wn and wn > 0 else float("nan"),
        "yaw_rate_gain_per_rad": r_gain,
        "lat_accel_gain_ms2_per_rad": ay_gain,
        "articulation_per_rad_steer_deg": math.degrees(phi_ss),
        "understeer_gradient_rad_per_ms2": float(understeer),
        "understeer_gradient_deg_per_g": float(math.degrees(understeer * G)),
        "delta_for_0.2g_deg": float(math.degrees(0.2 * G / ay_gain)),
    }


def stability_vs_speed(cfg: TTVConfig,
                       speeds=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)):
    """Max real eigenvalue of the body-frame model at each speed.

    An articulated vehicle can be stable at its design speed and unstable
    higher up (divergent or oscillatory jack-knife). Checking only the
    operating point hides that.
    """
    import copy
    rows = []
    for V in speeds:
        c = copy.deepcopy(cfg)
        c.V = V
        Av, _ = _descriptor_model(c)
        ev = np.linalg.eigvals(Av)
        damped = [e for e in ev if abs(e.imag) > 1e-6]
        zeta = min((-e.real / abs(e) for e in damped), default=float("nan"))
        rows.append({"V": V, "max_real": float(np.max(ev.real)),
                     "zeta": float(zeta),
                     "stable": bool(np.all(ev.real < -1e-9))})
    return rows


def rearward_amplification(cfg: TTVConfig, f_hz: float = 0.4) -> float:
    """Rearward amplification: |a_y,trailer| / |a_y,tractor| at frequency f.

    PBS (Performance Based Standards) limit is RWA <= 2.0. Evaluated on the
    body-frame descriptor model:
        a_y1 = V*(beta_dot + r1)
        a_y2 = a_y1 - c*r1_dot - a2*r2_dot,    r2 = r1 + q   (small phi)
    """
    Av, Bv = _descriptor_model(cfg)
    s = 2j * math.pi * f_hz
    z = np.linalg.solve(s * np.eye(4) - Av, Bv)      # [r1, beta, q, phi](s)
    r1, beta, q = z[0], z[1], z[2]
    ay1 = cfg.V * (s * beta + r1)
    ay2 = ay1 - cfg.c * (s * r1) - cfg.a2 * (s * (r1 + q))
    return float(abs(ay2) / abs(ay1))


def rwa_peak(cfg: TTVConfig) -> Dict[str, float]:
    """Peak rearward amplification over the 0.05-1.0 Hz band, and where."""
    f = np.linspace(0.05, 1.0, 400)
    r = np.array([rearward_amplification(cfg, fi) for fi in f])
    i = int(np.argmax(r))
    return {"rwa_peak": float(r[i]), "rwa_peak_f_hz": float(f[i])}


# =============================================================================
def audit(verbose: bool = True) -> Dict[str, float]:
    """Check every derived number against the constraint it came from."""
    cfg = truck_config()
    hitch = cfg.m2 * G * cfg.b2 / (cfg.a2 + cfg.b2)
    tridem = cfg.m2 * G * cfg.a2 / (cfg.a2 + cfg.b2)
    steer = (cfg.b1 * cfg.m1 * G + (cfg.b1 - cfg.c) * hitch) / (cfg.a1 + cfg.b1)
    drive = cfg.m1 * G + hitch - steer
    kingpin_from_front = cfg.a1 + cfg.c
    kingpin_ahead_of_drive = (cfg.a1 + cfg.b1) - kingpin_from_front

    res = {
        "m1_kg": cfg.m1, "m2_kg": cfg.m2, "GCW_t": (cfg.m1 + cfg.m2) / 1000,
        "a1_m": cfg.a1, "b1_m": cfg.b1, "c_m": cfg.c,
        "a2_m": cfg.a2, "b2_m": cfg.b2,
        "Iz1": cfg.Iz1, "Iz2": cfg.Iz2,
        "C1_kN_rad": cfg.C1 / 1e3, "C2_kN_rad": cfg.C2 / 1e3, "C3_kN_rad": cfg.C3 / 1e3,
        "steer_t": steer / G / 1000, "drive_t": drive / G / 1000,
        "tridem_t": tridem / G / 1000, "kingpin_t": hitch / G / 1000,
        "kingpin_ahead_of_drive_m": kingpin_ahead_of_drive,
        "SRT_g": SRT, "SRT_rigid_g": SRT_RIGID,
    }
    res.update(handling_metrics(cfg))
    res.update(rwa_peak(cfg))
    res["RWA_0.4Hz"] = rearward_amplification(cfg, 0.4)
    res["RWA_0.3Hz"] = rearward_amplification(cfg, 0.3)
    svs = stability_vs_speed(cfg)
    res["stability_vs_speed"] = svs
    res["stable_over_speed_range"] = 1.0 if all(r["stable"] for r in svs) else 0.0

    checks = [
        ("steer axle <= 7.5 t", res["steer_t"], "<=", 7.5),
        ("drive axle <= 11.5 t", res["drive_t"], "<=", 11.5),
        ("tridem <= 24.0 t", res["tridem_t"], "<=", 24.0),
        ("GCW <= 40.0 t", res["GCW_t"], "<=", 40.0),
        ("kingpin 0..0.8 m ahead of drive", res["kingpin_ahead_of_drive_m"], "in", (0.0, 0.8)),
        ("Iz1 in 20-40e3 kg m^2", res["Iz1"] / 1e3, "in", (20.0, 40.0)),
        ("Iz2 in 300-550e3 kg m^2", res["Iz2"] / 1e3, "in", (300.0, 550.0)),
        ("linear model stable", 1.0 if res["stable"] else 0.0, "==", 1.0),
        ("peak RWA <= 2.0 (PBS)", res["rwa_peak"], "<=", 2.0),
        ("peak RWA >= 1.0 (trailer amplifies)", res["rwa_peak"], ">", 1.0),
        ("understeer gradient 1-6 deg/g (laden artic)",
         res["understeer_gradient_deg_per_g"], "in", (1.0, 6.0)),
        ("steer for 0.2 g in 0.8-3.0 deg", res["delta_for_0.2g_deg"], "in", (0.8, 3.0)),
        ("yaw damping zeta >= 0.15 (PBS)",
         res["trailer_mode_damping_ratio"], ">", 0.15),
        ("stable over the whole speed range", res["stable_over_speed_range"], "==", 1.0),
    ]

    if verbose:
        print("=" * 78)
        print("Highway tractor-semitrailer: derived parameters and their audit")
        print("=" * 78)
        print(f"  tractor m1            {cfg.m1:9.0f} kg      trailer m2  {cfg.m2:9.0f} kg")
        print(f"  a1 / b1 / c           {cfg.a1:6.3f} / {cfg.b1:6.3f} / {cfg.c:6.3f} m")
        print(f"  a2 / b2               {cfg.a2:6.3f} / {cfg.b2:6.3f} m")
        print(f"  Iz1 / Iz2             {cfg.Iz1:9.0f} / {cfg.Iz2:9.0f} kg m^2")
        print(f"  C1 / C2 / C3          {cfg.C1/1e3:7.1f} / {cfg.C2/1e3:7.1f} / {cfg.C3/1e3:7.1f} kN/rad")
        print(f"\n  static axle loads     steer {res['steer_t']:5.2f} t | drive "
              f"{res['drive_t']:5.2f} t | tridem {res['tridem_t']:5.2f} t "
              f"| kingpin {res['kingpin_t']:5.2f} t")
        print(f"  kingpin position      {res['kingpin_ahead_of_drive_m']:.3f} m ahead of drive axle")
        print(f"\n  SRT (rigid / real)    {SRT_RIGID:.3f} g / {SRT:.3f} g")
        print(f"  trailer yaw mode      zeta = {res['trailer_mode_damping_ratio']:.3f}, "
              f"period = {res['trailer_mode_period_s']:.2f} s")
        print(f"  slowest time constant {res['slowest_time_constant_s']:.2f} s")
        print(f"  lat-accel gain        {res['lat_accel_gain_ms2_per_rad']:.1f} m/s^2 per rad steer")
        print(f"  steer for 0.2 g       {res['delta_for_0.2g_deg']:.2f} deg at the road wheel")
        print(f"  articulation gain     {res['articulation_per_rad_steer_deg']:.1f} deg phi per rad steer")
        print(f"  understeer gradient   {res['understeer_gradient_deg_per_g']:.2f} deg/g")
        print(f"  RWA @ 0.3 / 0.4 Hz    {res['RWA_0.3Hz']:.3f} / {res['RWA_0.4Hz']:.3f}")
        print(f"  RWA peak              {res['rwa_peak']:.3f} at {res['rwa_peak_f_hz']:.2f} Hz")

        print("\n  audit:")
        ok_all = True
        for name, val, op, lim in checks:
            if op == "<=":
                ok = val <= lim
            elif op == ">":
                ok = val > lim
            elif op == "==":
                ok = abs(val - lim) < 1e-9
            else:
                ok = lim[0] <= val <= lim[1]
            ok_all &= ok
            print(f"    [{'ok ' if ok else 'FAIL'}] {name:42s} : {val:.4g}")
        print(f"\n  ALL CHECKS {'PASSED' if ok_all else 'DID NOT PASS'}")

        print("\n  stability vs speed (body-frame model):")
        print(f"  {'V [km/h]':>9s} {'max Re(eig)':>12s} {'zeta':>7s} {'stable':>7s}")
        for r in svs:
            print(f"  {r['V']*3.6:9.0f} {r['max_real']:12.4f} {r['zeta']:7.3f} "
                  f"{str(r['stable']):>7s}")

        print("\n  Which constraint binds on a highway?")
        print(f"  {'V [km/h]':>9s} {'mu':>5s} {'k_roll':>10s} {'k_fric':>10s} "
              f"{'binding':>10s} {'R_min roll [m]':>15s}")
        for Vt in (19.0, 22.0, 24.0, 25.0):
            for mut in (0.35, 0.60, 0.90):
                b = binding_constraint(Vt, mut)
                print(f"  {Vt*3.6:9.0f} {mut:5.2f} {b['kappa_rollover']:10.5f} "
                      f"{b['kappa_friction']:10.5f} {b['binding']:>10s} "
                      f"{1/b['kappa_rollover']:15.0f}")
        print("\n  -> On dry and even damp asphalt the rig ROLLS before it SLIDES.")
        print("     That is the opposite of the car the paper was written for, and")
        print("     it is why the reward and the action bounds both have to change.")
        res["all_checks_passed"] = ok_all
    return res


if __name__ == "__main__":
    import json
    r = audit()
    with open("out/truck_params_audit.json", "w") as f:
        json.dump(r, f, indent=2, default=str)
    print("\n  wrote out/truck_params_audit.json")
