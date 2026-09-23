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

## Layer 1 — the faithful port

Reproduces the MATLAB file function-for-function. Deviations are marked
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
| `ttv_core.py` | ~700 | the faithful port: config, plant, linear model, MPC, episode, reward |
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
