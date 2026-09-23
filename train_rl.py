"""
train_rl.py -- The RL planner
=============================

The paper uses TD3 with gamma = 0.99 on an episode that is ONE STEP long. That
is a contradiction worth naming: with a single-step episode there is no next
state, so the Bellman target reduces to r and the discount factor is
irrelevant. TD3's three signature mechanisms -- delayed actor updates, twin
critics with a clipped-min target, and target-policy smoothing -- all exist to
control BOOTSTRAPPING error, and with no bootstrapping there is none to
control. Fig. 11 of the paper shows DDPG failing to train at all on the
polynomial action space while TD3 succeeds, which is consistent with the
instability being in the critic, not in the task.

So this is a one-step soft actor-critic, i.e. a contextual bandit:

    critic:  Q_i(s,a)  regresses the observed reward directly   (no target net)
    actor:   maximise  min_i Q_i(s, a) + alpha * H[pi(.|s)]
    alpha:   auto-tuned to a target entropy

Twin critics are kept -- not for the min-target, which no longer exists, but
because the actor exploits a single critic's errors and the pessimistic min
over two independently initialised critics suppresses that.

Also here and not in the paper:
  * observation whitening from a running estimate, which matters because the
    13 observation components have very different scales;
  * a held-out evaluation set of scenarios, fixed once, so the learning curve
    measures generalisation rather than memorisation of the randomiser.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from highway_env import (TruckHighwayEnv, Scenario, RewardWeights,
                         OBS_DIM, ACTION_DIM)

torch.set_num_threads(2)
DEV = torch.device("cpu")


# =============================================================================
def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """tanh-squashed diagonal Gaussian, so actions live in [-1,1]^9 natively."""

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.body = mlp([obs_dim, *hidden], act=nn.ReLU)
        self.body = nn.Sequential(self.body, nn.ReLU())
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        # log-prob with the tanh change of variables
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std
                - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1], act=nn.ReLU)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class RunningNorm:
    """Welford mean/variance for observation whitening."""

    def __init__(self, dim):
        self.n = 1e-4
        self.mean = np.zeros(dim)
        self.M2 = np.ones(dim)

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def __call__(self, x):
        std = np.sqrt(np.maximum(self.M2 / self.n, 1e-8))
        return np.clip((x - self.mean) / std, -8.0, 8.0)

    def state(self):
        return {"n": self.n, "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load(self, d):
        self.n = d["n"]
        self.mean = np.asarray(d["mean"])
        self.M2 = np.asarray(d["M2"])


# =============================================================================
@dataclass
class TrainConfig:
    episodes: int = 6000
    warmup: int = 400              # uniform-random actions before learning
    batch: int = 256
    updates_per_step: int = 2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_alpha: float = 3e-4
    hidden: Tuple[int, ...] = (256, 256)
    buffer: int = 60000
    target_entropy_scale: float = 0.6   # * (-act_dim)
    eval_every: int = 250
    eval_scenarios: int = 40
    seed: int = 0


def make_eval_set(n: int, seed: int = 12345) -> List[Scenario]:
    """A fixed held-out scenario set, sampled once and never trained on."""
    sampler = TruckHighwayEnv(randomise=True, seed=seed)
    return [sampler.sample_scenario() for _ in range(n)]


def evaluate(actor: Actor, norm: RunningNorm, scens: List[Scenario],
             env: TruckHighwayEnv) -> dict:
    rewards, ltr, rwa, swept, dep, ok, coll, roll = [], [], [], [], [], 0, 0, 0
    for sc in scens:
        obs = env.reset(sc)
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(norm(obs), dtype=torch.float32,
                                         device=DEV).unsqueeze(0),
                         deterministic=True, with_logp=False)
        _, r, _, info = env.step(a.squeeze(0).numpy())
        m = info["metrics"]
        rewards.append(r)
        ok += int(info["parts"].get("cause") == "success")
        coll += int(m.get("collision", False))
        roll += int(m.get("rollover", False))
        if m:
            ltr.append(m.get("peak_LTR", np.nan))
            rwa.append(m.get("RWA", np.nan))
            swept.append(m.get("swept_width", np.nan))
            dep.append(m.get("lane_departure", np.nan))
    return {"reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "success_rate": ok / len(scens),
            "collisions": coll, "rollovers": roll,
            "mean_peak_LTR": float(np.nanmean(ltr)) if ltr else float("nan"),
            "mean_RWA": float(np.nanmean(rwa)) if rwa else float("nan"),
            "mean_swept": float(np.nanmean(swept)) if swept else float("nan"),
            "mean_lane_departure": float(np.nanmean(dep)) if dep else float("nan")}


# =============================================================================
def train(tc: TrainConfig, out_prefix: str = "out/rl"):
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    env = TruckHighwayEnv(randomise=True, seed=tc.seed, weights=RewardWeights())
    eval_env = TruckHighwayEnv(randomise=False)
    eval_set = make_eval_set(tc.eval_scenarios)

    actor = Actor(OBS_DIM, ACTION_DIM, tc.hidden).to(DEV)
    q1 = Critic(OBS_DIM, ACTION_DIM, tc.hidden).to(DEV)
    q2 = Critic(OBS_DIM, ACTION_DIM, tc.hidden).to(DEV)
    opt_a = torch.optim.Adam(actor.parameters(), lr=tc.lr_actor)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                             lr=tc.lr_critic)
    log_alpha = torch.zeros(1, requires_grad=True, device=DEV)
    opt_al = torch.optim.Adam([log_alpha], lr=tc.lr_alpha)
    target_H = -tc.target_entropy_scale * ACTION_DIM

    norm = RunningNorm(OBS_DIM)
    buf_o = np.zeros((tc.buffer, OBS_DIM), dtype=np.float32)
    buf_a = np.zeros((tc.buffer, ACTION_DIM), dtype=np.float32)
    buf_r = np.zeros(tc.buffer, dtype=np.float32)
    n_buf = 0
    ptr = 0

    history, evals = [], []
    t_start = time.perf_counter()

    for ep in range(1, tc.episodes + 1):
        obs = env.reset()
        norm.update(obs)
        on = norm(obs)
        if ep <= tc.warmup:
            act = rng.uniform(-1, 1, ACTION_DIM)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(on, dtype=torch.float32,
                                             device=DEV).unsqueeze(0),
                             with_logp=False)
            act = a.squeeze(0).numpy()
        _, rew, _, info = env.step(act)

        buf_o[ptr] = on
        buf_a[ptr] = act
        buf_r[ptr] = rew
        ptr = (ptr + 1) % tc.buffer
        n_buf = min(n_buf + 1, tc.buffer)
        history.append({"ep": ep, "reward": rew,
                        "cause": str(info["parts"].get("cause"))})

        if ep > tc.warmup and n_buf >= tc.batch:
            for _ in range(tc.updates_per_step):
                idx = rng.integers(0, n_buf, tc.batch)
                o = torch.as_tensor(buf_o[idx], device=DEV)
                a_b = torch.as_tensor(buf_a[idx], device=DEV)
                r_b = torch.as_tensor(buf_r[idx], device=DEV)

                # --- critics: one-step, so the target IS the reward ---------
                loss_q = F.mse_loss(q1(o, a_b), r_b) + F.mse_loss(q2(o, a_b), r_b)
                opt_q.zero_grad(set_to_none=True)
                loss_q.backward()
                nn.utils.clip_grad_norm_(list(q1.parameters())
                                         + list(q2.parameters()), 10.0)
                opt_q.step()

                # --- actor --------------------------------------------------
                a_pi, logp = actor(o)
                qpi = torch.min(q1(o, a_pi), q2(o, a_pi))
                alpha = log_alpha.exp().detach()
                loss_a = (alpha * logp - qpi).mean()
                opt_a.zero_grad(set_to_none=True)
                loss_a.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                opt_a.step()

                # --- temperature --------------------------------------------
                loss_al = -(log_alpha.exp() * (logp.detach() + target_H)).mean()
                opt_al.zero_grad(set_to_none=True)
                loss_al.backward()
                opt_al.step()

        if ep % tc.eval_every == 0 or ep == tc.episodes:
            ev = evaluate(actor, norm, eval_set, eval_env)
            ev["episode"] = ep
            ev["alpha"] = float(log_alpha.exp().item())
            ev["train_reward_500"] = float(np.mean(
                [h["reward"] for h in history[-500:]]))
            ev["elapsed_s"] = time.perf_counter() - t_start
            evals.append(ev)
            print(f"  ep {ep:6d} | eval R {ev['reward']:+.4f} "
                  f"+-{ev['reward_std']:.3f} | train R {ev['train_reward_500']:+.4f} "
                  f"| succ {ev['success_rate']*100:5.1f}% | coll {ev['collisions']:2d} "
                  f"| LTR {ev['mean_peak_LTR']:.3f} | dep {ev['mean_lane_departure']:.3f} "
                  f"| alpha {ev['alpha']:.3f} | {ev['elapsed_s']/60:.1f} min",
                  flush=True)

    torch.save({"actor": actor.state_dict(), "q1": q1.state_dict(),
                "q2": q2.state_dict(), "norm": norm.state(),
                "cfg": asdict(tc)}, f"{out_prefix}_agent.pt")
    json.dump({"config": asdict(tc), "evals": evals,
               "history": history[::5]},
              open(f"{out_prefix}_training.json", "w"), indent=2, default=str)
    print(f"\n  saved {out_prefix}_agent.pt and {out_prefix}_training.json")
    return actor, norm, evals


def load_agent(path: str = "out/rl_agent.pt"):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    tc = TrainConfig(**ck["cfg"])
    actor = Actor(OBS_DIM, ACTION_DIM, tuple(tc.hidden)).to(DEV)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    norm = RunningNorm(OBS_DIM)
    norm.load(ck["norm"])
    return actor, norm


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="out/rl")
    args = ap.parse_args()

    tc = TrainConfig(episodes=args.episodes, warmup=args.warmup,
                     eval_every=args.eval_every, seed=args.seed)
    print("=" * 96)
    print(f"Training the one-step SAC planner: {tc.episodes} episodes, "
          f"obs {OBS_DIM}, act {ACTION_DIM}")
    print("=" * 96)
    train(tc, args.out)
