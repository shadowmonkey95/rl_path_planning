"""
dlc_paper_env.py -- The RL task, with the reward grounded in Karimyan et al.
=============================================================================

Replaces `dlc_rl_env.py`'s hand-weighted reward with one built from the
physical criteria of Karimyan, Rahmani Hanzaki & Azadi (2024).

THE PROBLEM THIS SOLVES
-----------------------

The previous reward was five hand-chosen terms with hand-chosen weights and
hand-chosen reference scales. It had two defects, and the second is fatal to
the comparison:

  1. the weights were arbitrary -- "articulation at 0.25" answers to nothing;
  2. the RL policy was trained to maximise exactly the reward used to score
     every method, while the baseline planner minimised something else. A
     margin measured that way is close to tautological.

The reward here is instead the paper's own trajectory-evaluation criterion:

      PRIMARY   minimise the peak required coefficient of friction (Eq. 62)
      GATE      reject LTR > 0.6                           (Fig. 9, Table 4)
      GATE      reject RCOF > the friction actually available
      SECONDARY finish inside the deadline

RCOF is not a term anyone invented for this project. It is the fraction of
available grip a manoeuvre consumes, it is what the paper selects on, and
crucially the UPGRADED PLANNER IN `paper_planner.py` OPTIMISES THE SAME THING.
Neither method is now scored on its own private objective, which was the
strongest criticism of the earlier benchmark.

WHAT IS STILL HAND-CHOSEN, STATED PLAINLY
-----------------------------------------

Turning "minimise peak RCOF subject to gates" into a scalar in [-1, 1] needs a
mapping, and that mapping is mine, not the paper's. It is deliberately thin:

    reward = 1 - mu_max / mu_available          (the grip margin, normalised)
             - deadline penalty if late
             gates return a large negative

There is one free parameter (how hard a late finish is punished) against five
weights and four reference scales before. The grip-margin form is the standard
way to read RCOF -- "what fraction of the available friction did this consume"
-- and the paper's own selection rule is recovered exactly when the deadline is
slack: among admissible trajectories, highest reward == lowest peak RCOF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from paper_metrics import (CG_HEIGHT_TRACTOR, CG_HEIGHT_TRAILER, G,
                           LTR_UNSTABLE, a_y_for_ltr, a_y_limit,
                           evaluate_trajectory, ltr_from_run, peak_rcof,
                           quintic_min_time, quintic_peak_ay)
from paper_planner import QuinticDLCReference
from ttv_dlc_tracking import FailureLimits, dlc_config, run_dlc_tracking


# =============================================================================
# 1.  The task
# =============================================================================

@dataclass
class PaperTask:
    """One double-lane-change task, with the friction circle as the budget.

    The lateral-acceleration budget of the previous task is GONE. It was a
    stand-in for the real constraint, which is the friction circle: how much
    grip the road offers and how much of it the manoeuvre consumes. `mu` now
    does that job directly, which is both more physical and one fewer knob.
    """
    V: float = 20.0
    lateral_offset: float = 3.5        # [m] the paper's W
    hold_time: float = 7.0             # [s] dwell required in the other lane
    start_time: float = 4.0            # [s]
    mu: float = 0.50                   # road friction ACTUALLY available
    sim_time: float = 22.0
    # Left as None so it is DERIVED from what the physics allows. A hard-coded
    # deadline was unreachable once the rollover limit was applied: it forces a
    # slower manoeuvre than the friction circle alone would.
    deadline: Optional[float] = None    # [s] back in lane by here
    # static axle loads, filled from the vehicle config
    Fz: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    def __post_init__(self):
        if self.Fz == (0.0, 0.0, 0.0):
            cfg = dlc_config(V=self.V, mu=self.mu)
            self.Fz = (cfg.Fz1, cfg.Fz2, cfg.Fz3)
        if self.deadline is None:
            self.deadline = self.t_fastest + 0.35 * (self.t_slowest
                                                     - self.t_fastest)

    @property
    def a_y_circle(self) -> float:
        """The friction circle's lateral capacity, Eq. (41) with a_x = 0."""
        return self.mu * G

    @property
    def a_y_cap(self) -> float:
        """What actually binds: the friction circle OR the rollover threshold.

        For a laden artic the second is usually the smaller, which is the
        paper's own finding and the reason Fig. 9 loops on LTR before anything
        else.
        """
        return a_y_limit(self.mu)

    @property
    def rollover_binds(self) -> bool:
        return a_y_for_ltr() < self.a_y_circle

    @property
    def t_m_min(self) -> float:
        """Eq. (50), against whichever limit binds."""
        return quintic_min_time(self.lateral_offset, self.a_y_cap)

    @property
    def t_fastest(self) -> float:
        return self.start_time + 2.0 * self.t_m_min + self.hold_time

    @property
    def t_slowest(self) -> float:
        return self.start_time + 2.0 * self.t_m_max + self.hold_time + 3.0

    @property
    def t_m_max(self) -> float:
        """Slowest transition that still fits two of them before sim_time."""
        return max(self.t_m_min + 0.1,
                   0.5 * (self.sim_time - self.start_time - self.hold_time))


# =============================================================================
# 2.  Action space -- the paper's parameters, constraints by construction
# =============================================================================

ACTION_DIM = 3
ACTION_NAMES = ("t_m", "dwell", "start")

OBS_DIM = 6
OBS_NAMES = ("V_norm", "mu_norm", "offset_norm", "hold_norm",
             "deadline_norm", "t_m_min_norm")


def observe(task: PaperTask) -> np.ndarray:
    return np.array([
        (task.V - 12.0) / 16.0,
        (task.mu - 0.25) / 0.65,
        task.lateral_offset / 5.0,
        task.hold_time / 12.0,
        (task.deadline - 10.0) / 10.0,
        task.t_m_min / 5.0,
    ], dtype=np.float64)


def decode_action(a: np.ndarray, task: PaperTask) -> QuinticDLCReference:
    """a in [-1,1]^3 -> a quintic double lane change.

    The friction circle is enforced BY CONSTRUCTION: `t_m` is scaled from
    `t_m_min` (Eq. 50, the fastest transition the available grip permits)
    upward, so no decodable action can demand more lateral acceleration than
    the road can supply. This is the same structural guarantee the previous
    action space had, but now the bound is EXACT (Eq. 44 for a quintic) rather
    than found by bisection, because two quintic transitions separated by a
    dwell cannot overlap.

    The dwell is likewise scaled from `hold_time` upward, so the lane-occupancy
    requirement is satisfied exactly rather than approximately -- for a quintic
    the plateau IS the dwell.
    """
    a = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
    u = 0.5 * (a + 1.0)

    t_m = task.t_m_min + (task.t_m_max - task.t_m_min) * u[0]
    dwell = task.hold_time + 3.0 * u[1]
    start = 2.0 + (task.start_time + 1.0) * u[2]
    return QuinticDLCReference(W=task.lateral_offset, t_m=t_m,
                               t_start=start, dwell=dwell, V=task.V)


# =============================================================================
# 3.  The reward
# =============================================================================

@dataclass
class PaperRewardWeights:
    """One free parameter, against nine in the previous reward."""
    deadline_penalty: float = 1.0      # cost of being a full slack late
    deadline_slack: float = 3.0        # [s] over which that penalty saturates
    fail_base: float = -1.0
    fail_progress_bonus: float = 0.5
    unstable_penalty: float = -1.0     # LTR above the paper's threshold


def compute_reward(res, task: PaperTask,
                   w: PaperRewardWeights = PaperRewardWeights()
                   ) -> Tuple[float, Dict[str, float]]:
    """The paper's criterion, mapped to a scalar.

    Gates first (they are the paper's "reject" rows), then the grip margin,
    then the deadline.
    """
    if not res.ok:
        progress = res.completed_steps / max(res.n_steps, 1)
        return (w.fail_base + w.fail_progress_bonus * progress,
                {"cause": res.reason, "progress": progress})

    # Lateral acceleration of each unit, from what the plant actually did.
    a_y_tractor = np.asarray(res.ref.ddy(res.t_state), dtype=float)
    a_y_trailer = a_y_tractor            # no roll DOF; see paper_metrics
    ev = evaluate_trajectory(res.force, a_y_tractor, a_y_trailer, task.Fz,
                             mu_available=task.mu,
                             t_m=getattr(res.ref, "t_m", float("nan")))

    parts: Dict[str, float] = {
        "mu_max": ev.mu_max, "mu_front": ev.mu_front, "mu_rear": ev.mu_rear,
        "mu_trailer": ev.mu_trailer, "ltr_max": ev.ltr_max,
        "workload": ev.workload, "a_y_peak": ev.a_y_peak,
        "grip_margin": 1.0 - ev.mu_max / max(task.mu, 1e-9),
    }

    # GATE 1: the paper's rollover rejection.
    if not ev.stable:
        parts["cause"] = "LTR above threshold"
        over = (ev.ltr_max - LTR_UNSTABLE) / LTR_UNSTABLE
        return float(w.unstable_penalty * min(1.0, 0.5 + 0.5 * over)), parts

    # GATE 2: the trajectory demands more grip than the road has.
    if not ev.feasible:
        parts["cause"] = "RCOF exceeds available friction"
        over = (ev.mu_max - task.mu) / max(task.mu, 1e-9)
        return float(w.unstable_penalty * min(1.0, 0.5 + 0.5 * over)), parts

    # PRIMARY: the grip margin. Maximising this IS minimising peak RCOF, so an
    # agent that maximises reward is running the paper's selection rule.
    total = parts["grip_margin"]

    # SECONDARY: the deadline.
    t_done = float(res.ref.completion_time)
    margin = task.deadline - t_done
    parts["completion_time"] = t_done
    parts["deadline_margin"] = margin
    pen = 0.0
    if margin < 0.0:
        pen = -w.deadline_penalty * min(-margin / w.deadline_slack, 1.0)
        total += pen
    parts["pen_deadline"] = pen
    parts["cause"] = "success" if margin >= 0 else "missed deadline"
    return float(total), parts


# =============================================================================
# 4.  The environment
# =============================================================================

class PaperPlannerEnv:
    """Same interface `train_rl.py` already expects."""

    def __init__(self, randomise: bool = True, seed: int = 0,
                 weights: Optional[PaperRewardWeights] = None,
                 backend: str = "fast",
                 limits: Optional[FailureLimits] = None):
        self.rng = np.random.default_rng(seed)
        self.randomise = randomise
        self.w = weights or PaperRewardWeights()
        self.backend = backend
        self.limits = limits or FailureLimits(enabled=True)
        self.task: Optional[PaperTask] = None

    def sample_task(self) -> PaperTask:
        if not self.randomise:
            return PaperTask()
        r = self.rng
        base = PaperTask(
            V=float(r.uniform(14.0, 28.0)),
            lateral_offset=float(r.uniform(2.5, 4.5)),
            hold_time=float(r.uniform(4.0, 9.0)),
            start_time=float(r.uniform(3.0, 5.0)),
            mu=float(r.uniform(0.30, 0.85)),
            sim_time=28.0,
        )
        # The deadline sits a fraction of the way across what this friction
        # makes achievable, so tightness is what varies between tasks and the
        # agent has something to read off the observation. Same construction as
        # the previous task, now driven by the friction circle instead of an
        # invented acceleration budget.
        frac = float(r.uniform(0.15, 0.60))
        base.deadline = float(base.t_fastest
                              + frac * (base.t_slowest - base.t_fastest))
        return base

    def reset(self, task: Optional[PaperTask] = None) -> np.ndarray:
        self.task = task if task is not None else self.sample_task()
        return observe(self.task)

    def step(self, action: np.ndarray):
        assert self.task is not None, "call reset() first"
        task = self.task
        ref = decode_action(action, task)
        cfg = dlc_config(V=task.V, mu=task.mu, Qpsi=1.0, Qphi=25.0, Qq=8.0)
        res = run_dlc_tracking(cfg, ref, task.sim_time, backend=self.backend,
                               limits=self.limits)
        reward, parts = compute_reward(res, task, self.w)
        info = {"metrics": res.metrics, "parts": parts, "ref": ref,
                "result": res, "task": task,
                "peak_a_y_g": ref.peak_lateral_accel / G}
        return observe(task), float(reward), True, info


def heuristic_action(task: PaperTask) -> np.ndarray:
    """No learning: the midpoint of the admissible transition-time range."""
    return np.array([0.0, 0.0, 0.0])


# =============================================================================
def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"    [{'ok ' if cond else 'FAIL'}] {name:54s} {detail}")

    if verbose:
        print("=" * 92)
        print("dlc_paper_env self-test")
        print("=" * 92)
        print("  A. the friction circle is enforced BY CONSTRUCTION")

    env = PaperPlannerEnv(seed=3, randomise=True)
    rng = np.random.default_rng(4)
    worst, n_ok, rewards = 0.0, 0, []
    causes: Dict[str, int] = {}
    for _ in range(120):
        env.reset()
        a = rng.uniform(-1, 1, ACTION_DIM)
        _, rw, _, info = env.step(a)
        ref = info["ref"]
        worst = max(worst, ref.peak_lateral_accel / env.task.a_y_cap)
        c = str(info["parts"].get("cause"))[:34]
        causes[c] = causes.get(c, 0) + 1
        n_ok += int(c == "success")
        rewards.append(rw)
    check("worst peak a_y / binding capacity over 120 actions",
          worst <= 1.0 + 1e-9, f"{worst:.6f}")
    check("task is non-degenerate: random actions both succeed and fail",
          0.20 <= n_ok / 120 <= 0.85,
          f"{n_ok}/120 succeeded ({100*n_ok/120:.0f} %)")
    if verbose:
        print(f"      reward mean {np.mean(rewards):+.4f}, "
              f"min {np.min(rewards):+.4f}, max {np.max(rewards):+.4f}")
        for k, v in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"      {k:36s} {v}")

    if verbose:
        print("\n  B. the reward recovers the paper's selection rule")
    # With a slack deadline, higher reward must mean lower peak RCOF.
    task = PaperTask(deadline=40.0, sim_time=40.0)
    env2 = PaperPlannerEnv(randomise=False)
    pairs = []
    for u in np.linspace(-1.0, 1.0, 9):
        env2.reset(task)
        _, rw, _, info = env2.step(np.array([u, 0.0, 0.0]))
        p = info["parts"]
        if p.get("cause") == "success":
            pairs.append((rw, p["mu_max"]))
    mono = all(pairs[i][0] >= pairs[j][0] or pairs[i][1] >= pairs[j][1]
               for i in range(len(pairs)) for j in range(len(pairs)))
    corr = (np.corrcoef([r for r, _ in pairs], [m for _, m in pairs])[0, 1]
            if len(pairs) > 2 else float("nan"))
    check("reward is a decreasing function of peak RCOF",
          len(pairs) > 2 and corr < -0.99,
          f"corr(reward, mu_max) = {corr:.4f} over {len(pairs)} points")

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

    print("\n" + "=" * 92)
    print("What the agent can choose, and what it costs (paper reward)")
    print("=" * 92)
    task = PaperTask()
    print(f"  V={task.V} m/s, offset {task.lateral_offset} m, mu {task.mu}, "
          f"deadline {task.deadline} s")
    print(f"  friction circle gives a_y <= {task.a_y_circle:.2f} m/s^2; "
          f"LTR = {LTR_UNSTABLE} gives a_y <= {a_y_for_ltr():.2f} m/s^2")
    print(f"  the binding cap is {task.a_y_cap:.2f} m/s^2 "
          f"({'ROLLOVER' if task.rollover_binds else 'friction'}), "
          f"so t_m >= {task.t_m_min:.2f} s (Eq. 50)")
    env = PaperPlannerEnv(randomise=False)
    print(f"\n  {'action':>26s} {'t_m':>6s} {'a_y[g]':>7s} {'mu_max':>7s} "
          f"{'LTR':>6s} {'t_done':>7s} {'reward':>8s} {'outcome':>22s}")
    for name, a in (("fastest allowed", np.array([-1., 0, 0])),
                    ("25 %", np.array([-0.5, 0, 0])),
                    ("midpoint", np.array([0., 0, 0])),
                    ("slowest", np.array([1., 0, 0])),
                    ("fast + early start", np.array([-1., 0, -1.])),
                    ("fast + long dwell", np.array([-1., 1., 0]))):
        env.reset(task)
        _, rw, _, info = env.step(a)
        p = info["parts"]
        print(f"  {name:>26s} {info['ref'].t_m:6.2f} {info['peak_a_y_g']:7.3f} "
              f"{p.get('mu_max', float('nan')):7.3f} "
              f"{p.get('ltr_max', float('nan')):6.3f} "
              f"{p.get('completion_time', float('nan')):7.2f} {rw:8.4f} "
              f"{str(p.get('cause'))[:22]:>22s}")
    sys.exit(0 if good else 1)
