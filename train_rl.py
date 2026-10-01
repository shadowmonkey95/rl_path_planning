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
import os
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The environment is selected at run time. Both expose the same interface --
# OBS_DIM, ACTION_DIM, reset() -> obs, step(a) -> (obs, reward, done, info) --
# so nothing below this line knows or cares which one is in use.
ENVS = {
    "highway": ("highway_env", "TruckHighwayEnv"),
    "dlc": ("dlc_rl_env", "DLCPlannerEnv"),
}


def load_env(name: str):
    import importlib
    mod_name, cls_name = ENVS[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name), mod.OBS_DIM, mod.ACTION_DIM


ENV_NAME = os.environ.get("TTV_ENV", "highway")
EnvClass, OBS_DIM, ACTION_DIM = load_env(ENV_NAME)

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


def make_eval_set(n: int, seed: int = 12345) -> list:
    """A fixed held-out scenario set, sampled once and never trained on."""
    sampler = EnvClass(randomise=True, seed=seed)
    sample = (sampler.sample_scenario if hasattr(sampler, "sample_scenario")
              else sampler.sample_task)
    return [sample() for _ in range(n)]


# =============================================================================
# Logging
# -----------------------------------------------------------------------------
# `evaluate` returns the same four metric slots whatever the environment, but
# they MEAN different things, so the eval line labels them per environment.
# Reading "LTR 0.052" on a DLC run was actively misleading: that slot carries
# the peak articulation angle in radians (3.0 deg), not a load transfer ratio.
#
# Every eval line is also appended to `<out_prefix>_log.txt`. Solver chatter
# goes to stdout and can bury the run; the log file holds only the header, the
# configuration and the eval lines, so a finished run leaves something short
# enough to read.
# =============================================================================
_EVAL_COLUMNS = {
    "highway": (("coll", "collisions", "{:2d}"),
                ("LTR", "mean_peak_LTR", "{:.3f}"),
                ("dep[m]", "mean_lane_departure", "{:.3f}")),
    "dlc": (("phi[deg]", "mean_peak_LTR", "{:.2f}", math.degrees),
            ("slip[deg]", "mean_RWA", "{:.2f}", math.degrees),
            ("RMSe[m]", "mean_swept", "{:.4f}")),
}


def _eval_line(ev: dict, env_name: str) -> str:
    cols = _EVAL_COLUMNS.get(env_name, _EVAL_COLUMNS["highway"])
    tail = ""
    for spec in cols:
        label, key, fmt = spec[0], spec[1], spec[2]
        v = ev.get(key, float("nan"))
        if len(spec) > 3 and isinstance(v, float) and math.isfinite(v):
            v = spec[3](v)
        try:
            shown = fmt.format(v)
        except (ValueError, TypeError):
            shown = "   nan"
        tail += f" | {label} {shown}"
    return (f"  ep {ev['episode']:6d} | eval R {ev['reward']:+.4f} "
            f"+-{ev['reward_std']:.3f} | train R {ev['train_reward_500']:+.4f}"
            f" | succ {ev['success_rate']*100:5.1f}%{tail}"
            f" | alpha {ev['alpha']:.3f} | {ev['elapsed_s']/60:.1f} min")


class RunLog:
    """Writes the run's readable record to `<prefix>_log.txt` and to stdout."""

    def __init__(self, prefix: str):
        self.path = f"{prefix}_log.txt"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def evaluate(actor: Actor, norm: RunningNorm, scens: list, env) -> dict:
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
            # highway metrics; the DLC env reports different ones, and the
            # nanmean below simply yields nan for whichever are absent
            ltr.append(m.get("peak_LTR", m.get("peak_articulation_angle", np.nan)))
            rwa.append(m.get("RWA", m.get("max_front_slip", np.nan)))
            swept.append(m.get("swept_width", m.get("rms_lateral_error", np.nan)))
            dep.append(m.get("lane_departure", m.get("peak_lateral_error", np.nan)))
    return {"reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "success_rate": ok / len(scens),
            "collisions": coll, "rollovers": roll,
            "mean_peak_LTR": float(np.nanmean(ltr)) if ltr else float("nan"),
            "mean_RWA": float(np.nanmean(rwa)) if rwa else float("nan"),
            "mean_swept": float(np.nanmean(swept)) if swept else float("nan"),
            "mean_lane_departure": float(np.nanmean(dep)) if dep else float("nan")}


# =============================================================================
def _save(prefix, actor, q1, q2, norm, tc, ep, evals, history,
          opt_a, opt_q, opt_al, log_alpha, buf_o, buf_a, buf_r, n_buf, ptr):
    """Checkpoint after every evaluation.

    The first version of this file saved only at the end. A 3-hour run that
    dies at hour 2 then loses everything, which is exactly what makes long
    training runs painful. The checkpoint carries the replay buffer and the
    optimiser states too, so `--resume` continues rather than restarts.
    """
    torch.save({
        "actor": actor.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
        "opt_a": opt_a.state_dict(), "opt_q": opt_q.state_dict(),
        "opt_al": opt_al.state_dict(), "log_alpha": log_alpha.detach().clone(),
        "norm": norm.state(), "cfg": asdict(tc), "episode": ep,
        "evals": evals, "history": history[::5],
        "buf": {"o": buf_o[:n_buf], "a": buf_a[:n_buf], "r": buf_r[:n_buf],
                "n": n_buf, "ptr": ptr},
    }, f"{prefix}_agent.pt")
    json.dump({"config": asdict(tc), "episode": ep, "evals": evals,
               "history": history[::5]},
              open(f"{prefix}_training.json", "w"), indent=2, default=str)


def train(tc: TrainConfig, out_prefix: str = "out/rl", resume: bool = False):
    log = RunLog(out_prefix)
    log(f"\n{'=' * 96}")
    log(f"one-step SAC on '{ENV_NAME}' | {tc.episodes:,} episodes | "
        f"obs {OBS_DIM} | act {ACTION_DIM} | seed {tc.seed} | "
        f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 96)

    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)
    env = EnvClass(randomise=True, seed=tc.seed)
    eval_env = EnvClass(randomise=False)
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
    start_ep = 0
    if resume:
        import os
        ck_path = f"{out_prefix}_agent.pt"
        if not os.path.exists(ck_path):
            raise FileNotFoundError(f"--resume given but {ck_path} does not exist")
        ck = torch.load(ck_path, map_location=DEV, weights_only=False)
        # asdict() keeps `hidden` a tuple; compare as lists on both sides.
        if (list(ck["cfg"]["hidden"]) != list(tc.hidden)
                or ck["cfg"]["seed"] != tc.seed):
            raise ValueError("checkpoint was trained with a different network "
                             "or seed; use a different --out")
        actor.load_state_dict(ck["actor"])
        q1.load_state_dict(ck["q1"]); q2.load_state_dict(ck["q2"])
        opt_a.load_state_dict(ck["opt_a"]); opt_q.load_state_dict(ck["opt_q"])
        opt_al.load_state_dict(ck["opt_al"])
        with torch.no_grad():
            log_alpha.copy_(ck["log_alpha"])
        norm.load(ck["norm"])
        b = ck["buf"]
        n_buf = int(b["n"]); ptr = int(b["ptr"])
        buf_o[:n_buf] = b["o"]; buf_a[:n_buf] = b["a"]; buf_r[:n_buf] = b["r"]
        evals = ck.get("evals", [])
        start_ep = int(ck["episode"])
        log(f"  resumed from {ck_path} at episode {start_ep:,} "
            f"(buffer {n_buf:,}, last eval R {evals[-1]['reward']:+.4f})"
            if evals else f"  resumed from {ck_path} at episode {start_ep:,}")
        if start_ep >= tc.episodes:
            log(f"  checkpoint is already at {start_ep:,} episodes; "
                f"asked for {tc.episodes:,}. Nothing to do.")
            return actor, norm, evals

    t_start = time.perf_counter()

    for ep in range(start_ep + 1, tc.episodes + 1):
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
            _save(out_prefix, actor, q1, q2, norm, tc, ep, evals, history,
                  opt_a, opt_q, opt_al, log_alpha, buf_o, buf_a, buf_r,
                  n_buf, ptr)
            ev = evaluate(actor, norm, eval_set, eval_env)
            ev["episode"] = ep
            ev["alpha"] = float(log_alpha.exp().item())
            ev["train_reward_500"] = float(np.mean(
                [h["reward"] for h in history[-500:]]))
            ev["elapsed_s"] = time.perf_counter() - t_start
            evals.append(ev)
            log(_eval_line(ev, ENV_NAME))

    _save(out_prefix, actor, q1, q2, norm, tc, tc.episodes, evals, history,
          opt_a, opt_q, opt_al, log_alpha, buf_o, buf_a, buf_r, n_buf, ptr)
    best = max(evals, key=lambda e: e["reward"]) if evals else None
    if best is not None:
        log(f"\n  best eval R {best['reward']:+.4f} at episode {best['episode']:,}"
            f"; final {evals[-1]['reward']:+.4f} at {evals[-1]['episode']:,}")
    log(f"  saved {out_prefix}_agent.pt, {out_prefix}_training.json "
        f"and {log.path}")
    log.close()
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
    ap.add_argument("--env", default=ENV_NAME, choices=sorted(ENVS),
                    help="which environment to train on")
    ap.add_argument("--out", type=str, default="out/rl",
                    help="prefix for <prefix>_agent.pt and <prefix>_training.json")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>_agent.pt instead of starting over")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing agent at --out")
    args = ap.parse_args()

    if args.env != ENV_NAME:
        raise SystemExit(
            f"\n  Set the environment before import:\n"
            f"    TTV_ENV={args.env} python train_rl.py ...\n"
            f"  (the network sizes are fixed at import time from OBS_DIM/ACTION_DIM)\n")

    ck = f"{args.out}_agent.pt"
    if os.path.exists(ck) and not (args.resume or args.force):
        raise SystemExit(
            f"\n  {ck} already exists.\n"
            f"  Pick one:\n"
            f"    --resume            continue training that agent\n"
            f"    --out out/rl_25k    train a new agent alongside it\n"
            f"    --force             overwrite it\n")

    tc = TrainConfig(episodes=args.episodes, warmup=args.warmup,
                     eval_every=args.eval_every, seed=args.seed)
    train(tc, args.out, resume=args.resume)
    print(f"\n  The readable record of this run is {args.out}_log.txt "
          f"-- solver chatter never reaches it.")
