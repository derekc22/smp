# SMP

Score-Matching Motion Priors for humanoid motion tracking.  Trains a small
diffusion model on motion windows; the frozen score is reused as an SDS
reward during PPO training.  Reproduction of Mu et al. 2025
(arXiv:2512.03028).

## Status

The simplest downstream task is the **walk-jog-run forward** task: three
forward-motion clips (walk, jog, run) serve as the pretraining dataset;
the policy receives a uniformly-sampled target speed in `[0.5, 3.5]` m/s
and must modulate gait to match it, with a fixed `+x` heading.  It's
registered as `Smp-Forward-G1` — a fixed-direction specialization of the
more general `Smp-Steering-G1`.

The end-to-end pipeline:

1. **CSV → NPZ** — slice motion clips into fixed-length feature windows.
2. **Normalization stats** — compute per-feature q01/q99 quantiles.
3. **Diffusion pretraining** — train the DDPM ε-predictor on the windowed data.
4. **RL** — PPO with the frozen denoiser as an SMP guidance reward plus a
   steering task reward.

## Motion feature representation

Each window frame is a 59-dim vector

```
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15),
 root_lin_vel(3), root_ang_vel(3)]
```

All spatial quantities are anchored to the LAST window frame's yaw-only
local frame (origin at `pelvis_T`, x-axis = `heading_T`):

- `root_pos` — xy heading-inv relative to `pelvis_T`; z in world.
- `root_rot` — 6D tan-norm of `heading_inv(T) ⊗ root_quat[t]`.
- `joint_pos` — raw joint angles (frame-invariant).
- `ee_pos` — per-frame root offset rotated into the last-frame heading-inv frame.
- `root_lin_vel`, `root_ang_vel` — last-frame heading-inv.

Tracked end-effectors: `left_ankle_roll_link`, `right_ankle_roll_link`,
`torso_link` (proxy for head), `left_wrist_yaw_link`, `right_wrist_yaw_link`.

## 1. CSV → NPZ

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/loco \
  --output-dir datasets/npz/loco
```

Each input CSV holds base pose + DoF trajectories for a motion clip.  The
output NPZ stores `(N, W, 59)` windows; the online feature buffer
(`smp.rl.utils.MotionFeatureBuffer`) reproduces the same computation at
RL time.

## 2. Normalization stats

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/loco \
  --output datasets/norm_stats.npz
```

Computes per-feature q01 / q99 for mapping to `[-1, 1]`.  If you have a
larger motion database available (e.g. LAFAN), fit the stats on that to
give a wider normalization range — the RL policy drifts outside the narrow
walk/jog/run distribution during training, and a too-tight normalizer
makes those states look OOD.

## 3. Diffusion pretraining

DDPM ε-prediction with a cosine-β schedule (50 timesteps), optional EMA on
weights, and multi-noise-sample L1 loss for variance reduction.  Each run
is identified by `--name`; the final checkpoint lands at
`logs/pretrain/<name>/<timestamp>/pretrained.pt`.

`--num-noise-samples` controls the number of `(t, ε)` draws per data point
in the loss.  Larger values give lower-variance gradients at the cost of
more compute per step; use a higher value when the dataset is small (the
gradient noise dominates) and a lower value when the dataset is large.

### Forward task (walk / jog / run)

Three forward-motion clips → small score field over a narrow gait
distribution.  Pretrain on `datasets/npz/loco/`:

```bash
uv run scripts/pretrain.py \
  --data-dir datasets/npz/loco/ \
  --num-layers 2 --d-model 128 --no-use-ema \
  --num-noise-samples 50 \
  --num-epochs 10000 --save-interval 5000 \
  --train-split 1.0 \
  --name pretrain-forward
```

### Steering / Location tasks (LAFAN run)

A larger LAFAN run subset → broader score field that covers turning,
non-axis-aligned headings, and longer locomotion.  Pretrain on
`datasets/npz/lafan_run/`:

```bash
uv run scripts/pretrain.py \
  --data-dir datasets/npz/lafan_run/ \
  --num-layers 2 --d-model 128 --no-use-ema \
  --num-noise-samples 10 \
  --num-epochs 5000 --save-interval 1000 \
  --train-split 1.0 \
  --name pretrain-lafan-run
```

### Visualize unconditional samples

```bash
uv run scripts/generate_viz.py \
  --ckpt-path logs/pretrain/<name>/<timestamp>/pretrained.pt
```

Runs unconditional DDPM ancestral sampling and plays back the resulting
window in a viser viewer.  The pelvis trajectory is reconstructed directly
from the `root_pos` / `root_rot` features anchored at the robot's default
standing pose.

## 4. RL

Three downstream tasks are registered:

| Task                  | Env cfg                                              | Pretrained ckpt to use   |
| --------------------- | ---------------------------------------------------- | ------------------------ |
| `Smp-Forward-G1`      | `src/smp/rl/tasks/steering/forward_env_cfg.py`       | `pretrain-forward`       |
| `Smp-Steering-G1`     | `src/smp/rl/tasks/steering/steering_env_cfg.py`      | `pretrain-lafan-run`     |
| `Smp-Location-G1`     | `src/smp/rl/tasks/location/location_env_cfg.py`      | `pretrain-lafan-run`     |

**Before training, edit the per-task env_cfg.py and point the
`init_smp_state` event's `ckpt_path` param at the `.pt` file you produced
in step 3.**  For example, in `forward_env_cfg.py`:

```python
cfg.events["init_smp_state"].params["ckpt_path"] = (
  "logs/pretrain/pretrain-forward/<timestamp>/pretrained.pt"
)
```

Then:

```bash
# Train (replace task id as needed)
uv run scripts/train.py Smp-Forward-G1

# Play
uv run scripts/play.py Smp-Forward-G1 --wandb-run-path <org>/<project>/<run>
```

Common reward terms (defined on `g1_smp_env_cfg` and overlaid per task):

- **SMP guidance reward** (weight `1.0`): ensemble SDS at diffusion
  timesteps `K = (8, 15, 22)` with `w_s = 4.0` (forward) or `6.0`
  (steering / location), per-timestep normalization via `DiffNormalizer`.
- **Velocity tracking** (steering / forward): `exp(-vel_err_scale ·
  ‖tar_speed·tar_dir − v_xy‖²)`, zeroed when the velocity projects
  negatively onto the target.
- **Face direction** (steering only): clipped dot product between the
  commanded face direction and the robot's heading direction.
- **Position tracking** (location only): `exp(-pos_err_scale · ‖xy_goal −
  xy_robot‖²)` toward a periodically resampled world-frame xy goal.
- **GSI reset**: every episode reset draws a full window from the frozen
  denoiser and uses it to prime the feature buffer and set the initial
  joint / velocity state on sim.
