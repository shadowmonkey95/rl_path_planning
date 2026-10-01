# Tractor-Trailer RL Path Planning

A faithful Python port of `ttv_rl_episode.m`, an audit of it, and a highway
tractor-semitrailer RL planner built on top.

Reference: Á. Fehér, Á. Domina, Á. Bárdos, S. Aradi, T. Bécsi, *"Path planning
via reinforcement learning with closed-loop motion control and field tests"*,
Engineering Applications of Artificial Intelligence 142 (2025) 109870.

## Install

```bash
pip install numpy scipy matplotlib casadi osqp torch
```

Python 3.11. CasADi is needed only for the faithful port and the 2-MPC
baseline; the fast path uses OSQP and NumPy. `torch` is needed only for
training.

## Layer 0 — two models, no RL  ← start here

Port of `RUN_TTV_DLC_TRACKING_TWO_MODELS.m`. A double-lane-change reference
generated internally, tracked by the linear MPC and simulated on the nonlinear
plant. No path input, no reward, no RL. This is the simplest complete loop in
the package and the right place to build from.

```bash
python ttv_dlc_tracking.py                      # the MATLAB run, IPOPT backend
python ttv_dlc_tracking.py --backend fast       # same answer, 17x faster per solve
python ttv_dlc_tracking.py --studies            # weight activation + speed sweep

python dlc_rl_env.py                            # the simple RL task + self-test
python dlc_baselines.py                          # heuristic / 2-MPC / optimum, one task
TTV_ENV=dlc python train_rl.py --env dlc --episodes 4000 \
    --warmup 400 --eval-every 400 --out out/rl_dlc
TTV_ENV=dlc python dlc_benchmark.py --n 40       # the four-way comparison
```

Shares its vehicle, linear model, plant and MPC with `ttv_core`; it imports
them rather than restating them. Three things are new: the reference is a
function of **time** rather than arc length, the **one-step prediction error**
(Model 2 − Model 1) is logged per state, and solver time is recorded.

## Layer 1 — the faithful port of the RL episode

Reproduces `ttv_rl_episode.m` function-for-function. Deviations are marked
`PORT NOTE:` in the source.

```bash
python ttv_core.py          # run the docstring example (it fails; that is the finding)
python validate_port.py     # 7 checks, incl. Ac == the plant's AD Jacobian to 1.7e-14
python profile_and_plot.py  # cost per episode + fig1, fig2
```

## Layer 2 — the highway truck

```bash
python truck_params.py      # derive the 40 t rig and audit all 15 constraints
python path_gen.py          # path generators + self-test
python fast_mpc.py          # condensed QP vs IPOPT: same answer, 48x faster
python plant_numpy.py       # NumPy plant vs CasADi plant: 1e-13
python highway_env.py       # environment smoke test + the safety guarantee
python tune_mpc.py          # horizon and articulation weights, by measurement
```

## Layer 3 — planners

```bash
python baselines.py                  # heuristic, 2-MPC planner, offline optimum
python hard_cases.py                 # where the 2-MPC planner's model runs out
python train_rl.py --episodes 9000   # one-step SAC  (~45 min, 1 core)
python benchmark.py --n 60           # RL vs 2-MPC vs heuristic, held out
python make_figures.py               # fig3, fig4, fig5
```

## File map

| File | Lines | What it is |
| --- | --- | --- |
| `ttv_dlc_tracking.py` | ~560 | **Layer 0**: port of `RUN_TTV_DLC_TRACKING_TWO_MODELS.m` — two models, no RL |
| `dlc_rl_env.py` | ~600 | **the simple RL layer**: DLC task, 6-D obs, 4-D action, reward, self-test |
| `dlc_baselines.py` | ~300 | simple-version 2-MPC planning QP + offline optimum |
| `dlc_benchmark.py` | ~180 | simple-version four-way comparison |
| `ttv_core.py` | ~810 | the faithful port: config, plant, linear model, MPC, episode, reward |
| `validate_port.py` | ~300 | the 7-check verification suite |
| `profile_and_plot.py` | ~170 | episode cost and the two audit figures |
| `truck_params.py` | ~400 | the 40 t rig, derived from legal/physical constraints, with `audit()` |
| `path_gen.py` | ~500 | G4 Bezier offset paths and the paper's curvature-polynomial scheme |
| `fast_mpc.py` | ~400 | condensed QP, move blocking, soft state constraints, steering-rate limit |
| `plant_numpy.py` | ~230 | the plant in NumPy, validated against CasADi |
| `highway_env.py` | ~780 | scenario, rollout, truck metrics, action/observation/reward, the env |
| `tune_mpc.py` | ~140 | horizon and weight sweeps |
| `baselines.py` | ~340 | 2-MPC planner, heuristic, offline optimum |
| `hard_cases.py` | ~190 | the RWA gap study and the hard scenario set |
| `train_rl.py` | ~330 | one-step SAC |
| `benchmark.py` | ~200 | held-out comparison |
| `make_figures.py` | ~260 | fig3, fig4, fig5 |

Everything in `out/` is generated.

## Suggested build-up order

1. `ttv_dlc_tracking.py` — two models, one reference, no RL. Establish that the
   MPC tracks and see the Model 1 / Model 2 gap.
2. `ttv_dlc_tracking.py --studies` — activate `Qpsi`/`Qphi`/`Qq` and sweep
   speed. This is where the shipped weights turn out to work only between
   15 and 28 m/s.
3. `dlc_rl_env.py` → `dlc_baselines.py` → `train_rl.py --env dlc` →
   `dlc_benchmark.py` — the **simple version**: same vehicle, same tracking
   MPC, only the reference generator is swapped. Four-way benchmark on 40
   held-out tasks. Measured result: the 2-MPC planner wins on reward
   (0.678 vs 0.616, RL wins 8/40), while RL is 5.5× faster to plan and hits
   the deadline on 100% of tasks vs 75%. The offline optimum (0.718) searches
   the same action space, so the gap is a learning gap, not a
   parameterisation gap.
4. `ttv_core.py` + `validate_port.py` — swap the time reference for an
   arc-length path and add the reward. Now `a_y` grows as V², and the
   documented example fails above 12 m/s.
5. `highway_env.py` + `train_rl.py` — add the truck, the scenarios and the
   agent.

## The three things to know before you build on this

1. **The shipped example fails as configured.** `reward = -1.0` at 23% of the
   path, front tyre slip 0.232 rad against a 0.200 rad limit. The path demands
   0.77 g at the default 20 m/s. It only passes at V ≤ 12 m/s. Fix the speed
   or the path before you train anything.

2. **`cfg.mpc.Qphi = cfg.mpc.Qq = 0` by default.** The MPC never damps
   articulation. On a 40 t rig, on a lane change at the rollover curvature
   ceiling, that alone takes peak Load Transfer Ratio to 1.20 — a rollover.
   With articulation damping, 0.69. Same path, same vehicle.

3. **`cfg.phiMax`/`qMax` are hard state constraints.** Harmless at their
   default `inf`. Set them and the QP becomes infeasible whenever the vehicle
   is outside the box (36/40 solver failures measured with the state exactly
   at the limit), and `ttv_rl_episode` scores a solver failure as a crash.
   Use the soft formulation in `fast_mpc.py`.
