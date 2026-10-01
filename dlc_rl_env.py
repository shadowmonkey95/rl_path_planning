"""
dlc_rl_env.py -- The same RL strategy, on the simple DLC setting
===============================================================

This exists to answer one question: does the architecture from `highway_env.py`
still work when the task is the simple double-lane-change tracking study
instead of the highway truck?

It does, and this file is the demonstration rather than the claim. It exposes
exactly the interface `train_rl.py` already consumes --

    OBS_DIM, ACTION_DIM, reset() -> obs, step(action) -> (obs, reward, done, info)

-- so `train_rl.py` trains on it with a one-line import change and nothing else.

    reference params  ->  MPC  ->  nonlinear plant  ->  reward
            ^                                             |
            +------------------ RL agent -----------------+

What changes from the highway version, and what does not
--------------------------------------------------------
UNCHANGED: the closed-loop evaluation strategy, the plant, the linear model,
`plant_to_mpc_state`, RK4, the tracking MPC, the one-step SAC, the benchmark
shape (RL vs optimiser vs heuristic vs offline optimum).

CHANGED: the action is the SHAPE OF THE REFERENCE, not a Bezier path; the
observation describes the manoeuvre rather than a highway scenario; and the
safety constraint is embedded differently -- see below, it gets simpler.

The constraint, and the closed form that does NOT work
------------------------------------------------------
For the tanh pair  y(t) = A*(tanh(rate*t - t1) - tanh(rate*t - t2))  with the
two transitions WELL SEPARATED,

    peak |d2y/dt2| = 0.7698 * A * rate^2

and that reproduces the shipped reference exactly (0.7698*1.75*1.5^2 = 3.031
m/s^2). It is tempting to invert it for a one-line safety bound:

    rate <= sqrt( a_y_max / (0.7698 * A) )        <-- DO NOT USE THIS

Measured over a grid of shapes, that formula UNDER-predicts the true peak by up
to 97 %. Once `rate * hold` falls below about 4 the two tanh terms overlap, y
never reaches full amplitude, and the resulting short sharp pulse has a larger
second derivative than either transition alone. A bound that under-predicts is
worse than no bound, because it looks like a guarantee.

`max_rate_for_accel` therefore bisects on the REAL reference, exactly as
`min_length_for_offset` does in the arc-length version. The closed form only
seeds the bracket. This is the same lesson twice: take extrema off the actual
function, never off a formula for a special case.

One genuine simplification does survive: peak lateral acceleration is
INDEPENDENT of speed here, because the reference is parameterised in time. A
time-parameterised reference cannot be made infeasible by driving faster,
which is the opposite of the arc-length path in `ttv_core`.

Why there is a deadline
------------------------
Without one this task is degenerate. Reward is monotone decreasing in the
transition rate -- gentler is always better on tracking, articulation, slip and
steering -- so the optimal action is "go as slowly as possible" and there is
nothing to learn. The deadline is the time-domain analogue of the highway
obstacle: it makes slow manoeuvres fail and fast ones cost lateral
acceleration, which is a real trade-off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ttv_dlc_tracking import (DLCReference, FailureLimits, TrackingResult,
                              dlc_config, run_dlc_tracking)

G = 9.81

# peak |2 tanh(u) sech^2(u)|, the shape constant of the tanh pair
TANH_SHAPE_CONST = 0.769800358919501


_DENSE_T = 6001


def peak_lateral_accel_isolated(amplitude: float, rate: float) -> float:
    """Closed form, VALID ONLY when the two transitions are well separated.

    Derivation: peak |d2y/dt2| = A*rate^2 * max|2 tanh(u) sech^2(u)| for a
    single tanh transition. Reproduces the shipped reference exactly
    (0.7698 * 1.75 * 1.5^2 = 3.031 m/s^2).

    DO NOT use it as a safety bound. Measured over a grid of shapes, it
    UNDER-predicts the true peak by up to 97 % once `rate * hold` drops below
    about 4: the two tanh terms then overlap, y never reaches full amplitude,
    and the resulting short sharp pulse has a larger second derivative than
    either transition alone. Use `peak_lateral_accel` below, which evaluates
    the real reference. This is the same lesson as `path_gen.py`: take extrema
    off the actual function, not off a formula for a special case.
    """
    return TANH_SHAPE_CONST * amplitude * rate ** 2


def peak_lateral_accel(ref: "DLCReference", t_end: float = 25.0) -> float:
    """True peak |d2y/dt2| of a reference, on a dense grid."""
    t = np.linspace(0.0, t_end, _DENSE_T)
    return float(np.max(np.abs(ref.ddy(t))))


def max_rate_for_accel(amplitude: float, a_y_max: float,
                       start: float, hold: float, V: float = 20.0,
                       t_end: float = 25.0, tol: float = 1e-9) -> float:
    """Largest transition rate whose TRUE peak lateral acceleration fits.

    The analogue of `min_length_for_offset`. The closed form above seeds the
    bracket; bisection on the real reference removes its overlap error.
    """
    def peak(rate: float) -> float:
        return peak_lateral_accel(
            DLCReference(amplitude=amplitude, rate=rate,
                         t1=rate * start, t2=rate * (start + hold), V=V),
            t_end)

    lo = 0.05
    hi = math.sqrt(a_y_max / (TANH_SHAPE_CONST * max(amplitude, 1e-9)))
    hi = max(hi, lo * 2.0)
    # widen until hi genuinely violates, then bisect
    for _ in range(60):
        if peak(hi) > a_y_max:
            break
        hi *= 1.3
    else:
        return hi
    if peak(lo) > a_y_max:          # even the slowest shape is over budget
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if peak(mid) > a_y_max:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return lo


# =============================================================================
@dataclass
class DLCTask:
    """One double-lane-change task. The analogue of `Scenario`."""
    V: float = 20.0
    lateral_offset: float = 3.5        # total lateral move, m (2 x amplitude)
    hold_time: float = 7.0             # time spent in the adjacent lane, s
    start_time: float = 4.0            # when the manoeuvre begins, s
    mu: float = 0.90
    a_y_budget: float = 0.35           # [g] the lateral-acceleration ceiling
    sim_time: float = 20.0
    # The manoeuvre must be COMPLETE (back in the original lane) by this time.
    # This is the time-domain analogue of the highway obstacle, and without it
    # the task is degenerate: reward is monotone decreasing in transition rate,
    # so the optimal action is always "go as slowly as possible" and there is
    # nothing for an agent to learn. The deadline makes slow manoeuvres fail
    # and fast ones cost lateral acceleration, which is a real trade-off.
    deadline: float = 15.0             # [s]

    @property
    def amplitude(self) -> float:
        return 0.5 * self.lateral_offset

    @property
    def a_y_max(self) -> float:
        return self.a_y_budget * G

    @property
    def rate_min(self) -> float:
        """Slowest sensible transition: the manoeuvre must fit in sim_time."""
        return 6.0 / max(self.sim_time - self.start_time - self.hold_time, 1.0)

    def rate_max(self, start: float, hold: float, amplitude: float) -> float:
        """Fastest transition whose TRUE peak lateral accel fits the budget."""
        return max_rate_for_accel(amplitude, self.a_y_max, start, hold, self.V,
                                  self.sim_time + 5.0)


# =============================================================================
ACTION_DIM = 4
ACTION_NAMES = ("rate", "hold", "start", "amplitude_trim")

OBS_DIM = 6
OBS_NAMES = ("V_norm", "mu_norm", "offset_norm", "hold_norm",
             "budget_norm", "deadline_norm")


def decode_action(a: np.ndarray, task: DLCTask) -> DLCReference:
    """a in [-1,1]^4 -> a DLCReference, with the a_y budget built in.

    `rate` is scaled from `rate_min` to `rate_max`, and `rate_max` is the
    closed-form bound above. So no decodable action exceeds the lateral
    acceleration budget -- the same guarantee the highway action space gives,
    obtained with a square root instead of a bisection.
    """
    a = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
    u = 0.5 * (a + 1.0)

    amp = task.amplitude * (0.9 + 0.2 * u[3])          # +-10 % trim
    hold = 0.5 * task.hold_time + 1.0 * task.hold_time * u[1]
    start = 2.0 + (task.start_time + 1.0) * u[2]

    # The rate ceiling depends on the shape (start, hold, amplitude), because
    # overlapping transitions raise the true peak. So it is solved here, after
    # the shape is known, on the real reference.
    r_max = task.rate_max(start, hold, amp)
    r_min = min(task.rate_min, 0.95 * r_max)
    rate = r_min + (r_max - r_min) * u[0]

    # tanh(rate*t - t1): the transition centre sits at t = t1/rate
    return DLCReference(amplitude=amp, rate=rate,
                        t1=rate * start, t2=rate * (start + hold), V=task.V)


def completion_time(ref: DLCReference, tol: float = 0.05,
                    t_end: float = 25.0) -> float:
    """When the reference is finally back within `tol` of zero, and stays."""
    t = np.linspace(0.0, t_end, _DENSE_T)
    y = np.abs(ref.y(t))
    outside = np.nonzero(y > tol)[0]
    return float(t[outside[-1]]) if outside.size else 0.0


def observe(task: DLCTask) -> np.ndarray:
    return np.array([
        (task.V - 12.0) / 16.0,
        (task.mu - 0.35) / 0.55,
        task.lateral_offset / 5.0,
        task.hold_time / 12.0,
        task.a_y_budget / 0.6,
        (task.deadline - 10.0) / 10.0,
    ], dtype=np.float64)


# =============================================================================
@dataclass
class DLCRewardWeights:
    """Every term normalised to [0,1] before weighting, as in the highway env,
    so a weight is the actual priority."""
    tracking: float = 0.30
    articulation: float = 0.25
    slip: float = 0.20
    steering: float = 0.15
    brevity: float = 0.10

    track_ref: float = 0.05        # [m]   RMS lateral error reference
    phi_ref: float = math.radians(12.0)
    slip_ref: float = 0.10         # [rad]
    steer_ref: float = 0.20        # [rad]
    fail_base: float = -1.0
    fail_progress_bonus: float = 0.5
    # missing the deadline is a failure, scaled by how badly
    deadline_penalty: float = -1.0
    deadline_slack: float = 3.0    # [s] over which the penalty saturates


def compute_reward(r: TrackingResult, task: DLCTask,
                   w: DLCRewardWeights = DLCRewardWeights()
                   ) -> Tuple[float, Dict[str, float]]:
    m = r.metrics
    if not r.ok:
        progress = r.completed_steps / max(r.n_steps, 1)
        return (w.fail_base + w.fail_progress_bonus * progress,
                {"cause": r.reason, "progress": progress})

    r_track = math.exp(-m["rms_lateral_error"] / w.track_ref)
    r_phi = float(np.clip(1.0 - m["peak_articulation_angle"] / w.phi_ref, 0, 1))
    r_slip = float(np.clip(1.0 - m["max_front_slip"] / w.slip_ref, 0, 1))
    r_steer = float(np.clip(1.0 - m["peak_steering_command"] / w.steer_ref, 0, 1))

    # brevity is measured against the DEADLINE, not against an arbitrary rate
    t_done = completion_time(r.ref, t_end=task.sim_time + 5.0)
    margin = task.deadline - t_done
    r_brev = float(np.clip(margin / w.deadline_slack, 0.0, 1.0))

    total = (w.tracking * r_track + w.articulation * r_phi + w.slip * r_slip
             + w.steering * r_steer + w.brevity * r_brev)

    pen = 0.0
    if margin < 0.0:
        pen = w.deadline_penalty * min(-margin / w.deadline_slack, 1.0)
        total += pen

    return float(total), {"r_tracking": r_track, "r_articulation": r_phi,
                          "r_slip": r_slip, "r_steering": r_steer,
                          "r_brevity": r_brev, "completion_time": t_done,
                          "deadline_margin": margin, "pen_deadline": pen,
                          "cause": "success" if margin >= 0 else "missed deadline"}


# =============================================================================
class DLCPlannerEnv:
    """Same interface as TruckHighwayEnv, so train_rl.py needs no changes."""

    def __init__(self, seed: int = 0, randomise: bool = True,
                 weights: DLCRewardWeights = DLCRewardWeights(),
                 limits: Optional[FailureLimits] = None,
                 backend: str = "fast"):
        self.rng = np.random.default_rng(seed)
        self.randomise = randomise
        self.w = weights
        self.limits = limits or FailureLimits(enabled=True)
        self.backend = backend
        self.task: Optional[DLCTask] = None
        self._mpc_cache: Dict[tuple, object] = {}

    def sample_task(self) -> DLCTask:
        if not self.randomise:
            return DLCTask()
        r = self.rng
        return DLCTask(
            V=float(r.uniform(14.0, 28.0)),
            lateral_offset=float(r.uniform(2.5, 4.5)),
            hold_time=float(r.uniform(4.0, 9.0)),
            start_time=float(r.uniform(3.0, 5.0)),
            mu=float(r.uniform(0.45, 0.90)),
            a_y_budget=float(r.uniform(0.25, 0.45)),
            deadline=float(r.uniform(12.0, 17.0)),
        )

    def reset(self, task: Optional[DLCTask] = None) -> np.ndarray:
        self.task = task if task is not None else self.sample_task()
        return observe(self.task)

    def step(self, action: np.ndarray):
        assert self.task is not None, "call reset() first"
        task = self.task
        ref = decode_action(action, task)
        cfg = dlc_config(V=task.V, mu=task.mu,
                         Qpsi=1.0, Qphi=25.0, Qq=8.0)
        res = run_dlc_tracking(cfg, ref, task.sim_time, backend=self.backend,
                               limits=self.limits)
        reward, parts = compute_reward(res, task, self.w)
        info = {"metrics": res.metrics, "parts": parts, "ref": ref,
                "result": res, "task": task,
                "peak_a_y_g": peak_lateral_accel(ref, task.sim_time + 5.0) / G}
        return observe(task), float(reward), True, info


# =============================================================================
def heuristic_action(task: DLCTask) -> np.ndarray:
    """No learning: use 75 % of the allowed transition rate, neutral otherwise."""
    return np.array([0.5, 0.0, 0.0, 0.0])


def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:52s} {detail}")

    if verbose:
        print("=" * 84)
        print("dlc_rl_env self-test")
        print("=" * 84)
        print("  A. the closed-form lateral-acceleration bound")

    # The closed form is only valid for well-separated transitions. Measure
    # where it breaks rather than asserting it is fine.
    worst_under = 0.0
    for A in (1.0, 1.75, 2.5):
        for rate in (0.5, 1.0, 1.5, 2.2):
            for hold in (3.0, 5.0, 9.0, 12.0):
                ref = DLCReference(amplitude=A, rate=rate, t1=rate * 4.0,
                                   t2=rate * (4.0 + hold))
                num = peak_lateral_accel(ref)
                cf = peak_lateral_accel_isolated(A, rate)
                worst_under = max(worst_under, (num - cf) / cf)
    check("closed form UNDER-predicts once transitions overlap",
          worst_under > 0.5, f"by up to {100*worst_under:.0f} % -- so it is "
                             f"NOT a safe bound")
    check("closed form still exact for the shipped reference",
          abs(peak_lateral_accel_isolated(1.75, 1.5) - 3.0311) < 1e-3,
          f"{peak_lateral_accel_isolated(1.75, 1.5):.4f} m/s^2")

    # the numerical bound is what the action space actually uses
    worst_err = 0.0
    for A in (1.2, 1.75, 2.2):
        for ay in (2.5, 3.5, 4.5):
            for hold in (3.0, 6.0, 10.0):
                rm = max_rate_for_accel(A, ay, 4.0, hold, 20.0)
                ref = DLCReference(amplitude=A, rate=rm, t1=rm * 4.0,
                                   t2=rm * (4.0 + hold))
                worst_err = max(worst_err, peak_lateral_accel(ref) / ay - 1.0)
    check("max_rate_for_accel never exceeds the budget", worst_err < 1e-6,
          f"worst overshoot {worst_err:.2e}")

    if verbose:
        print("\n  B. no action can exceed the lateral-acceleration budget")
    env = DLCPlannerEnv(seed=3, randomise=True)
    rng = np.random.default_rng(4)
    worst_ratio, n_fail, n_ok = 0.0, 0, 0
    causes: Dict[str, int] = {}
    rewards = []
    for _ in range(120):
        env.reset()
        a = rng.uniform(-1, 1, ACTION_DIM)
        _, rw, _, info = env.step(a)
        worst_ratio = max(worst_ratio,
                          info["peak_a_y_g"] * G / env.task.a_y_max)
        c = str(info["parts"]["cause"])[:28]
        causes[c] = causes.get(c, 0) + 1
        n_ok += int(c == "success")
        n_fail += int(c != "success")
        rewards.append(rw)
    check("worst peak a_y / budget over 120 random actions",
          worst_ratio <= 1.0 + 1e-9, f"{worst_ratio:.6f}")
    # A good RL task is one where random actions sometimes fail. If almost
    # everything succeeds there is nothing to learn; if almost nothing does,
    # there is no gradient. 30-80 % is the healthy band.
    check("task is non-degenerate: random actions both succeed and fail",
          0.30 <= n_ok / 120 <= 0.80,
          f"{n_ok}/120 succeeded ({100*n_ok/120:.0f} %)")
    if verbose:
        print(f"      reward mean {np.mean(rewards):+.4f}, "
              f"min {np.min(rewards):+.4f}, max {np.max(rewards):+.4f}")
        for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"      {k:30s} {v}")

    if verbose:
        print("\n  C. the interface train_rl.py expects")
    o = env.reset()
    check("reset() returns OBS_DIM floats", o.shape == (OBS_DIM,), str(o.shape))
    o2, rw, done, info = env.step(np.zeros(ACTION_DIM))
    check("step() returns (obs, float, done, info)",
          o2.shape == (OBS_DIM,) and isinstance(rw, float) and done is True)
    check("info carries metrics and parts",
          "metrics" in info and "parts" in info)

    if verbose:
        print(f"\n  SELF-TEST {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    good = self_test()

    print("\n" + "=" * 84)
    print("What the agent can choose, and what it costs")
    print("=" * 84)
    task = DLCTask()
    print(f"  task: V={task.V} m/s, offset {task.lateral_offset} m, "
          f"a_y budget {task.a_y_budget} g")
    rmax = task.rate_max(task.start_time, task.hold_time, task.amplitude)
    print(f"  deadline {task.deadline} s, a_y budget {task.a_y_budget} g")
    print(f"  rate allowed: {task.rate_min:.3f} .. {rmax:.3f} "
          f"(shipped reference uses 1.5)")
    env = DLCPlannerEnv(randomise=False)
    print(f"\n  {'action':>28s} {'rate':>6s} {'a_y [g]':>8s} {'t_done':>7s} "
          f"{'RMS e_y':>9s} {'peak phi':>9s} {'reward':>8s} {'outcome':>15s}")
    for name, a in (("slowest transition", np.array([-1., 0, 0, 0])),
                    ("25 % rate", np.array([-0.5, 0, 0, 0])),
                    ("heuristic (75 % rate)", heuristic_action(task)),
                    ("fastest allowed", np.array([1., 0, 0, 0])),
                    ("fastest + short hold", np.array([1., -1., 0, 0])),
                    ("fastest + early start", np.array([1., -1., -1., 0]))):
        env.reset(task)
        _, rw, _, info = env.step(a)
        m = info["metrics"]
        pr = info["parts"]
        print(f"  {name:>28s} {info['ref'].rate:6.3f} {info['peak_a_y_g']:8.3f} "
              f"{pr.get('completion_time', float('nan')):7.2f} "
              f"{m['rms_lateral_error']:9.4f} "
              f"{math.degrees(m['peak_articulation_angle']):8.2f}d {rw:8.4f} "
              f"{str(pr['cause'])[:15]:>15s}")
    sys.exit(0 if good else 1)
