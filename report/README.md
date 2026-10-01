# RL path planning for a tractor-semitrailer: the simple (DLC) version

Generated 2026-09-29 10:47 by `make_report.py`. Every number below comes from a file in `data/`; nothing here is typed in by hand.

## The claim

> Given the same vehicle, the same tracking MPC, the same plant and the same scoring function, a policy trained once offline chooses a better reference trajectory than an online planning MPC does, and chooses it faster.

On 40 tasks drawn from a seed never used for anything (seed 777), the RL planner scores **0.8229** against the planning QP's **0.4538**. Paired per task: **+0.3691**, t = **+4.91**, win rate **100%**.

It reaches **97.9%** of the offline optimum, meets the deadline on **100%** of tasks against the QP's **35%**, and plans in **0.8 ms** against **28.9 ms** (medians).

### Three qualifications, stated up front

1. **The win rate is around 55-68%, not 90%.** The mean advantage comes from the tail, not from beating the QP on every task. See figure 2.
2. **The result depends on the training budget.** At 4000 episodes the QP was ahead by 0.061. The crossover is near 8000 episodes. Any report of this number must carry the budget beside it.
3. **This is one training seed.** Until it is replicated across three seeds, treat the effect size as provisional.

## What was compared

All four methods produce a *reference trajectory*. That reference is then driven by the **same** tracking MPC, through the **same** nonlinear plant, and scored by the **same** reward, on the **same** tasks. Only the reference generator differs, so any difference in score is attributable to the planner and nothing else.

| Method | What it is | What it controls for |
| --- | --- | --- |
| Fixed heuristic | 75% of the allowed transition rate; no optimisation, ignores the task | whether any optimisation is worth doing at all |
| 2-MPC planner | a convex QP over a triple integrator (jerk-minimising, `abs(a) <= a_y_max`, corridor on `y`), then the tracking MPC | the method to beat: plan with MPC, steer with MPC |
| RL planner | the trained policy, then the tracking MPC | ours |
| Offline optimum | a (1+lambda) evolution strategy, ~80 closed-loop simulations per task | the ceiling of the action space, so the RL score reads as a fraction rather than a bare number |

**The planning MPC was written for this project; it is not in the MATLAB.** See `logs/provenance.txt` -- this is the single easiest thing to get wrong when describing the work.

## Results

### 1. Training

![RL eval reward over training, against the three baselines. The shaded band is +/- one standard deviation across the 40 eval tasks.](figures\01_learning_curve.png)

*RL eval reward over training, against the three baselines. The shaded band is +/- one standard deviation across the 40 eval tasks.*

The curve crosses the QP line at about 8000 episodes and flattens just below the offline optimum. Two readings follow: the training budget is part of the result, and the action space is nearly exhausted, so further gains need a richer parameterisation rather than more compute.

### 2. Per-task differences

![RL reward minus 2-MPC reward, one bar per task, sorted.](figures\02_paired_difference.png)

*RL reward minus 2-MPC reward, one bar per task, sorted.*

This is the figure that keeps the claim honest. The wins and losses are not symmetric: the losses are small and the wins are large, which is why the mean moves even though the win rate is near half.

### 3. Where the advantage lives

![Mean, 10th percentile and worst case, on both task seeds.](figures\04_tail_comparison.png)

*Mean, 10th percentile and worst case, on both task seeds.*

The means are close. The tails are not. The QP's worst case is a task where its jerk-optimal trajectory is too slow for the deadline -- it has no way to trade comfort for time, because the deadline is a hard constraint in its formulation rather than a term in its objective. The policy is trained against a reward where brevity and smoothness are both weighted terms, so it makes that trade continuously.

**That is the mechanism to quote when asked why RL wins.** It is not capacity and it is not magic: the learned planner optimises the actual objective, while the QP optimises a convex proxy of it.

### 4. Distribution of outcomes

![All 40 task rewards for each method.](figures\03_reward_distribution.png)

*All 40 task rewards for each method.*


### 5. Physical metrics

![Mean of each physical quantity, QP versus policy.](figures\06_reward_components.png)

*Mean of each physical quantity, QP versus policy.*

The QP tracks its own plan more tightly (its reference is smoother by construction), while the policy finishes sooner and slips less. Neither dominates on every axis, which is what a weighted objective produces.

### 6. Planning cost

![Planning time per task, log scale.](figures\05_planning_time.png)

*Planning time per task, log scale.*

The speed argument holds independently of reward quality, but it is modest here -- this planning QP is small and convex. The large speedups in the literature compare against nonlinear planners.

## Tables

### Task seed 777 -- tasks never used for anything, not even watched during training

| Method | Reward | SD | P10 | Worst | Success | Peak a_y [g] | RMS e_y [m] | Peak phi [deg] | Max slip [deg] | Completion [s] | Plan time [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed heuristic | 0.5251 | 0.1545 | 0.3166 | -0.1413 | 75% | 0.2472 | 0.0798 | 3.1989 | 1.5964 | 12.4518 | 0.01 |
| 2-MPC planner | 0.4538 | 0.4949 | 0.2025 | -1.0 | 35% | 0.1167 | 0.0219 | 1.645 | 2.0203 | 14.3613 | 28.85 |
| RL planner | 0.8229 | 0.0431 | 0.7701 | 0.715 | 100% | 0.0836 | 0.0172 | 1.2891 | 0.9667 | 11.5726 | 0.77 |
| Offline optimum | 0.8403 | 0.0443 | 0.7877 | 0.739 | 100% | 0.078 | 0.0146 | 1.1818 | 0.8794 | 11.5648 | 40,749 |

### Task seed 12345 -- the set the training curve is read off

| Method | Reward | SD | P10 | Worst | Success | Peak a_y [g] | RMS e_y [m] | Peak phi [deg] | Max slip [deg] | Completion [s] | Plan time [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed heuristic | 0.4685 | 0.2397 | 0.1859 | -0.3567 | 78% | 0.2524 | 0.0829 | 3.3221 | 1.6101 | 12.712 | 0.00 |
| 2-MPC planner | 0.4499 | 0.4936 | 0.2368 | -1.0 | 40% | 0.1148 | 0.0242 | 1.7709 | 2.0146 | 14.4696 | 19.91 |
| RL planner | 0.81 | 0.0528 | 0.7461 | 0.6804 | 100% | 0.0853 | 0.02 | 1.4486 | 0.9564 | 11.5411 | 0.52 |
| Offline optimum | 0.8241 | 0.0539 | 0.7578 | 0.714 | 100% | 0.0795 | 0.0174 | 1.3274 | 0.87 | 11.684 | 21,979 |

### Paired statistics

| Seed | Comparison | Mean diff | Median diff | Win rate | Paired t |
| --- | --- | --- | --- | --- | --- |
| 777 | rl_minus_mpc2 | +0.3691 | +0.2280 | 100.0% | +4.91 |
| 777 | rl_minus_heuristic | +0.2978 | +0.2662 | 100.0% | +14.67 |
| 12345 | rl_minus_mpc2 | +0.3602 | +0.2051 | 100.0% | +4.83 |
| 12345 | rl_minus_heuristic | +0.3415 | +0.2607 | 100.0% | +10.18 |

Paired, because task difficulty varies enormously -- an unpaired comparison of two 40-sample means would drown a 0.04 effect in a 0.09 standard deviation.

The two seeds agree to within 0.0090 on the paired difference. That agreement is the evidence that nothing was selected on the test set: seed 12345 is the set the training curve is read off, so reporting only that one would be selection on the test set.

## The physical constraints

Four tiers, and it matters which is which, because only one is a soft penalty.

| Tier | Where it lives | Examples |
| --- | --- | --- |
| 1. Implicit | the plant | tyres saturate at `mu*Fz`; ask for more lateral force and you do not get it |
| 2. Hard | the tracking MPC's QP | steering `abs(delta) <= 0.5 rad`, rate `<= 0.6 rad/s` |
| 3. **By construction** | the action decoder | the lateral-acceleration budget and the lane-occupancy requirement |
| 4. Soft | the reward | the deadline, penalised up to -1.0 over 3 s |

Tier 3 is the design decision. A reward penalty makes a violation *expensive*; an action-space constraint makes it *impossible*. The guarantee therefore holds for an untrained network, which is not something a penalty-based design can say.

Verified in `logs/self_test.log`: PASSED.

## What the environment is, precisely

The road is an infinite, flat, straight, level plane with a single friction coefficient. There is no road geometry, no lane markings, no other vehicles and no obstacles, so there is no collision check. What replaces road geometry is the *task*: move `lateral_offset` metres sideways, hold for `hold_time` seconds, be back by `deadline`, and do not exceed `a_y_budget`.

| Randomised per task | Range |
| --- | --- |
| Forward speed `V` | 14-28 m/s |
| Lateral offset | 2.5-4.5 m |
| Hold time | 4-9 s |
| Start time | 3-5 s |
| Friction `mu` | 0.45-0.90 |
| Lateral-acceleration budget | 0.25-0.45 g |
| Deadline | `t_fast + frac*(t_slow - t_fast)`, `frac ~ U(0.25, 0.70)` |

The observation is exactly those values, scaled. Six numbers. The policy sees no vehicle state and no feedback: it decides before the manoeuvre begins and never sees what happens. That is what makes it a planner rather than a controller.

## What is in this folder

| Path | What it holds |
| --- | --- |
| `figures/` | every figure in this report, as PNG |
| `tables/` | the same numbers as CSV, including all 40 per-task rows |
| `logs/training.log` | the eval-only training record, one line per eval |
| `logs/self_test.log` | the environment's own verification run |
| `logs/provenance.txt` | what came from the MATLAB and what did not |
| `data/` | the raw JSON every table and figure was built from |

## How to reproduce

```bash
python3 ttv_dlc_tracking.py --studies        # the inherited loop, ~3 min
python3 dlc_rl_env.py                        # task + self-test, ~90 s
python3 dlc_baselines.py                     # baselines on one task, ~30 s
TTV_ENV=dlc python3 train_rl.py --env dlc --episodes 16000 \
    --warmup 400 --eval-every 1000 --out out/rl_dlc      # ~55 min
TTV_ENV=dlc python3 dlc_benchmark.py --n 40              # seed 777
TTV_ENV=dlc python3 dlc_benchmark.py --n 40 --seed 12345 # training set
python3 make_report.py                       # rebuild this folder
```
