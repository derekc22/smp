# Object-Goal mjlab/SMP Integration

This note documents the first real mjlab/SMP object-goal task path for the
two-stage `g1-diffusion` object-goal prior.

## Task

The task is registered as:

```text
Smp-ObjectGoal-G1
```

Registration lives in:

```text
src/smp/rl/tasks/object_goal/__init__.py
src/smp/rl/tasks/object_goal/object_goal_env_cfg.py
```

The task instantiates:

- Unitree G1 from the existing SMP G1 config.
- A real HF-BPS object mesh from
  `../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl`.
- The frozen two-stage object-goal prior through `init_object_goal_prior`.
- The existing object-goal SDS reward wrapper through
  `object_goal_task_smp_product`.

## Object Semantics

The first supported object is the real HF-BPS `largebox` sample:

```text
object_name: largebox
mesh_file: /home/learning/Documents/omomo_release/data/captured_objects/largebox_cleaned_simplified.obj
mesh_scale: 0.34874529050165676
```

The simulator object body is authored at the original mesh-origin frame. The
HF-BPS prior, however, expects object position to mean mesh centroid. Therefore
runtime conversion uses:

```text
p_centroid_world = p_body_world + R_body_world * p_body_to_centroid_body
```

For the checked `largebox` sample, the body-origin-to-centroid offset is:

```text
[0.04526461, 0.00815981, -0.12768450]
```

This offset is computed from the scaled OBJ vertices and validated against the
preprocessed HF-BPS `object_verts`. It is not assumed to be zero.

## Runtime Context

`src/smp/rl/object_goal_assets.py` provides the runtime object layer:

- loads HF-BPS PKLs and OBJ vertices
- computes scaled mesh vertices
- computes the body-origin-to-centroid offset
- builds a MuJoCo mesh `EntityCfg` spec
- converts object body pose to HF-BPS centroid pose
- reconstructs object vertices from runtime body pose
- reconstructs BPS with the same rule as g1 preprocessing:

```text
bps = basis_world - nearest_object_vertex
basis_world = bps_basis * bps_radius + object_centroid
```

The task stores an `ObjectGoalRuntimeContextBuilder` on the env. The reward
wrapper uses it to build:

- per-frame Stage 1 BPS encoding
- repeated static Stage 2 BPS context
- object centroid trajectory
- object vertices
- object rotation matrices

## Goal Source

The first integration uses a deterministic goal from the real HF-BPS sample:

```text
g = final object pose from the 300-frame HF-BPS window
```

The prior wrapper, not task code, normalizes this goal using checkpoint
statistics.

## Reset

`object_goal_sample_reset` replays one real HF-BPS window:

- writes the tail robot frame to the simulator
- writes the tail object body-origin pose to the simulator
- primes the 47D `[R, O]` object-goal reward buffer with the full HF-BPS window

This is a deterministic first-stage reset, not full object-goal GSI.

## Reward

The object task reward is intentionally simple:

```text
0.8 * object centroid tracking + 0.2 * object orientation tracking
```

The final reward follows the existing SMP product convention:

```text
r = task_reward * r_object_goal_smp
```

The SDS reward path still runs:

```text
Stage 1 hands -> contact rectification -> Stage 2 SDS over [R, O]
```

No generic free-box shortcut is used.

## Validation Scripts

Asset/context validation:

```bash
python scripts/check_object_goal_asset_context.py \
  --g1-diffusion-root ../g1-diffusion \
  --input-pkl ../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl \
  --device cpu \
  --window-size 300 \
  --num-envs 1
```

Reward-in-env smoke test:

```bash
python scripts/smoke_object_goal_env_reward.py \
  --g1-diffusion-root ../g1-diffusion \
  --stage1-ckpt /path/to/object_goal_stage1.pt \
  --stage2-ckpt /path/to/object_goal_stage2.pt \
  --input-pkl ../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl \
  --device cpu \
  --fixed-timesteps 1,3,5 \
  --ws 0.001
```

Tiny mjlab task step check, once `mjlab` is available:

```bash
uv run python scripts/check_object_goal_task_step.py \
  Smp-ObjectGoal-G1 \
  --num-envs 1 \
  --steps 3 \
  --fixed-timesteps 1,3,5
```

Tiny PPO dry launch, once task stepping works in the synced environment:

```bash
uv run scripts/train.py \
  Smp-ObjectGoal-G1 \
  --env.scene.num-envs=1 \
  --runner.max-iterations=1
```

## Current Runtime Limitation

This shell does not have `uv` or `mjlab` installed/importable, so the actual
mjlab reset/step and PPO launch were not run here. The code path is registered
and ready for the synced SMP environment.
