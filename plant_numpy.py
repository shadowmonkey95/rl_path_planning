"""
plant_numpy.py -- The same nonlinear plant, in NumPy
====================================================

`build_nonlinear_plant` in ttv_core builds CasADi Functions, which is right for
a MATLAB-faithful port and for getting exact Jacobians. But inside an RL
training loop the plant is evaluated four times per RK4 step, five substeps per
control step, ~130 control steps per episode: ~2600 calls. CasADi's Python
marshalling costs ~60 us a call, which was 65 % of the episode time after the
other hot spots were fixed.

This is the same model, evaluated in NumPy. `validate()` checks it against the
CasADi version to machine precision over random states, so the faithful version
stays the reference and this one is only a faster path to the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ttv_core import TTVConfig


@dataclass
class PlantNP:
    """Frozen scalars, so the hot loop touches no Python attributes of cfg."""
    V: float
    m1: float
    m2: float
    a1: float
    a2: float
    b1: float
    b2: float
    c: float
    Iz1: float
    Iz2: float
    C1: float
    C2: float
    C3: float
    mu: float
    Fz1: float
    Fz2: float
    Fz3: float
    tau: float
    delta_rate_max: float
    delta_max: float

    @classmethod
    def from_cfg(cls, cfg: TTVConfig) -> "PlantNP":
        cfg.finalize()
        return cls(V=cfg.V, m1=cfg.m1, m2=cfg.m2, a1=cfg.a1, a2=cfg.a2,
                   b1=cfg.b1, b2=cfg.b2, c=cfg.c, Iz1=cfg.Iz1, Iz2=cfg.Iz2,
                   C1=cfg.C1, C2=cfg.C2, C3=cfg.C3, mu=cfg.mu,
                   Fz1=cfg.Fz1, Fz2=cfg.Fz2, Fz3=cfg.Fz3,
                   tau=cfg.steeringTimeConstant,
                   delta_rate_max=cfg.deltaRateMax, delta_max=cfg.deltaMax)

    # ------------------------------------------------------------------
    def slip_and_forces(self, xp: np.ndarray):
        V = self.V
        v1, r1, phi, q, delta = xp[3], xp[4], xp[5], xp[6], xp[7]
        r2 = r1 + q
        hitch = v1 - self.c * r1
        sp, cp = math.sin(phi), math.cos(phi)
        V2 = V * cp + hitch * sp
        v2 = -V * sp + hitch * cp - self.a2 * r2

        a1s = delta - math.atan2(v1 + self.a1 * r1, V)
        a2s = -math.atan2(v1 - self.b1 * r1, V)
        a3s = -math.atan2(v2 - self.b2 * r2, V2)

        s1 = self.mu * self.Fz1
        s2 = self.mu * self.Fz2
        s3 = self.mu * self.Fz3
        Fy1 = s1 * math.tanh(self.C1 * a1s / s1)
        Fy2 = s2 * math.tanh(self.C2 * a2s / s2)
        Fy3 = s3 * math.tanh(self.C3 * a3s / s3)
        return (a1s, a2s, a3s), (Fy1, Fy2, Fy3), (V2, v2, r2, sp, cp)

    # ------------------------------------------------------------------
    def dynamics(self, xp: np.ndarray, delta_cmd: float) -> np.ndarray:
        V = self.V
        psi1, v1, r1, phi, q, delta = xp[2], xp[3], xp[4], xp[5], xp[6], xp[7]
        (a1s, a2s, a3s), (Fy1, Fy2, Fy3), (V2, v2, r2, sp, cp) = \
            self.slip_and_forces(xp)
        cd = math.cos(delta)

        m1, m2, a2, c, Iz1, Iz2 = self.m1, self.m2, self.a2, self.c, self.Iz1, self.Iz2
        M = np.array([
            [m1, 0.0, 0.0, 0.0, 1.0],
            [0.0, Iz1, 0.0, 0.0, -c],
            [m2 * sp, -m2 * c * sp, 0.0, -cp, -sp],
            [m2 * cp, -m2 * (c * cp + a2), -m2 * a2, sp, -cp],
            [0.0, Iz2, Iz2, a2 * sp, -a2 * cp],
        ])
        b = np.array([
            Fy1 * cd + Fy2 - m1 * V * r1,
            self.a1 * Fy1 * cd - self.b1 * Fy2,
            m2 * (v2 * r1 - a2 * r2 * q),
            Fy3 - m2 * V2 * r1,
            -self.b2 * Fy3,
        ])
        zeta = np.linalg.solve(M, b)

        raw = (delta_cmd - delta) / self.tau
        ddot = self.delta_rate_max * math.tanh(raw / self.delta_rate_max)

        return np.array([
            V * math.cos(psi1) - v1 * math.sin(psi1),
            V * math.sin(psi1) + v1 * math.cos(psi1),
            r1,
            zeta[0],
            zeta[1],
            q,
            zeta[2],
            ddot,
        ])

    # ------------------------------------------------------------------
    def rk4(self, xp: np.ndarray, delta_cmd: float, h: float) -> np.ndarray:
        k1 = self.dynamics(xp, delta_cmd)
        k2 = self.dynamics(xp + 0.5 * h * k1, delta_cmd)
        k3 = self.dynamics(xp + 0.5 * h * k2, delta_cmd)
        k4 = self.dynamics(xp + h * k3, delta_cmd)
        return xp + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def step(self, xp: np.ndarray, delta_cmd: float, T: float,
             substeps: int) -> np.ndarray:
        h = T / substeps
        for _ in range(substeps):
            xp = self.rk4(xp, delta_cmd, h)
        xp[7] = min(max(xp[7], -self.delta_max), self.delta_max)
        return xp

    # ------------------------------------------------------------------
    def slip_angles(self, xp: np.ndarray) -> np.ndarray:
        a, _, _ = self.slip_and_forces(xp)
        return np.array(a)

    def lateral_accels(self, xp: np.ndarray, xdot: np.ndarray) -> Tuple[float, float]:
        """Body-frame lateral acceleration of tractor and trailer CoG."""
        V = self.V
        v1, r1, phi, q = xp[3], xp[4], xp[5], xp[6]
        v1d, r1d, qd = xdot[3], xdot[4], xdot[6]
        r2 = r1 + q
        r2d = r1d + qd
        sp, cp = math.sin(phi), math.cos(phi)
        V2 = V * cp + (v1 - self.c * r1) * sp
        a_y1 = v1d + V * r1
        v2d = (-V * cp * q + (v1d - self.c * r1d) * cp
               - (v1 - self.c * r1) * sp * q - self.a2 * r2d)
        return a_y1, v2d + V2 * r2


# =============================================================================
def validate(verbose: bool = True) -> bool:
    """NumPy plant vs the CasADi plant, over random states."""
    from ttv_core import build_nonlinear_plant, rk4_step
    import truck_params as TP
    import time

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:46s} {detail}")

    if verbose:
        print("=" * 78)
        print("plant_numpy validation: NumPy plant vs the CasADi/MATLAB plant")
        print("=" * 78)

    rng = np.random.default_rng(11)
    for label, cfg in (("shipped AHV params", TTVConfig().finalize()),
                       ("laden highway truck", TP.truck_config())):
        dyn, out, _ = build_nonlinear_plant(cfg)
        p = PlantNP.from_cfg(cfg)
        e_dyn = e_alpha = e_rk4 = 0.0
        for _ in range(300):
            xp = np.array([rng.normal(0, 30), rng.normal(0, 3), rng.normal(0, .2),
                           rng.normal(0, 1.5), rng.normal(0, .15),
                           rng.normal(0, .10), rng.normal(0, .15),
                           rng.uniform(-cfg.deltaMax, cfg.deltaMax)])
            u = float(rng.uniform(-cfg.deltaMax, cfg.deltaMax))
            d_ca = np.asarray(dyn(xp, u)).flatten()
            d_np = p.dynamics(xp, u)
            sc = np.maximum(np.abs(d_ca), 1.0)
            e_dyn = max(e_dyn, float(np.max(np.abs(d_ca - d_np) / sc)))

            a_ca = np.asarray(out(xp, u)[0]).flatten()
            a_np = p.slip_angles(xp)
            e_alpha = max(e_alpha, float(np.max(np.abs(a_ca - a_np))))

            x_ca = rk4_step(dyn, xp.copy(), u, 0.02)
            x_np = p.rk4(xp.copy(), u, 0.02)
            sc2 = np.maximum(np.abs(x_ca), 1.0)
            e_rk4 = max(e_rk4, float(np.max(np.abs(x_ca - x_np) / sc2)))

        check(f"{label}: x_dot matches", e_dyn < 1e-12, f"max rel {e_dyn:.2e}")
        check(f"{label}: slip angles match", e_alpha < 1e-13, f"max abs {e_alpha:.2e}")
        check(f"{label}: one RK4 step matches", e_rk4 < 1e-12, f"max rel {e_rk4:.2e}")

    # ---- speed -------------------------------------------------------------
    cfg = TP.truck_config()
    dyn, _, _ = build_nonlinear_plant(cfg)
    p = PlantNP.from_cfg(cfg)
    xp = np.zeros(8)
    n = 3000
    t0 = time.perf_counter()
    for _ in range(n):
        rk4_step(dyn, xp, 0.01, 0.02)
    t_ca = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        p.rk4(xp, 0.01, 0.02)
    t_np = (time.perf_counter() - t0) / n
    if verbose:
        print(f"\n    CasADi RK4 step : {t_ca*1e6:8.1f} us")
        print(f"    NumPy  RK4 step : {t_np*1e6:8.1f} us")
        print(f"    speedup         : {t_ca/t_np:8.1f}x")
    check("NumPy plant is faster", t_np < t_ca, f"{t_ca/t_np:.1f}x")

    if verbose:
        print(f"\n  VALIDATION {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
