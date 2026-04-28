"""Startup + reset events for SMP RL.

These run from mjlab's event manager so the task can be a plain
``ManagerBasedRlEnv`` and we can use mjlab's built-in train/play scripts.

Per-frame motion features encode velocities + joint angles + EE positions
(see ``scripts/csv_to_npz.py``), so they do NOT directly carry an absolute
root pose.  On GSI we therefore write a DEFAULT root pose (origin xy,
default standing height, identity yaw) to sim and prime the feature buffer
with kinematics consistent with that pose.  This gives the policy a varied
joint / velocity distribution at reset time even though the root frame
itself is always the same.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import (
  EE_BODY_NAMES,
  NUM_EE,
  rot6d_to_quat,
  slice_features,
)

NUM_JOINTS = 29


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
) -> None:
  """Startup-mode event: load frozen denoiser, allocate buffer, prime buffer.

  Stashes the denoiser bundle, feature buffer, and ``DiffNormalizer`` on the
  env instance so the stock mjlab env class stays unsubclassed.
  """
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  env._smp_bundle = load_denoiser(ckpt_path, env.device)  # type: ignore[attr-defined]
  window_size = env._smp_bundle[5]  # type: ignore[attr-defined]
  robot = env.scene["robot"]
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
  )
  num_timesteps = env._smp_bundle[1].num_timesteps  # type: ignore[attr-defined]
  env._smp_normalizer = DiffNormalizer(num_timesteps, env.device)  # type: ignore[attr-defined]

  gsi_reset(env)


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
) -> None:
  """Common GSI tail: write last frame to sim, fill the feature buffer.

  Features carry root_pos (xy heading-inv + world z) and root_rot
  (heading-inv relative quat), so per-frame world pose is reconstructable
  directly.  The sim write anchors xy/yaw at the default root_pos / yaw
  (origin + identity yaw); per-frame heights and pitch/roll come from the
  features so the buffer's compute_features round-trip reproduces the
  stored window.

  Joint velocities are reconstructed from the joint-angle trajectory via
  finite differences, since joint_vel is not part of the feature window.
  """
  n, W, _ = window.shape
  E = NUM_EE
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(n, W, 4)

  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  last_root_state = torch.cat(
    [pelvis_pos_w[:, -1], pelvis_quat_w[:, -1], lin_vel_w[:, -1], ang_vel_w[:, -1]],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
  )


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization.

  Sample a full window from the SMP denoiser via DDPM ancestral sampling,
  write the last frame's velocities and joint state to sim (with a default
  root pose), and fill the feature buffer with the entire trajectory.
  Must run AFTER mjlab's ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle  # type: ignore[attr-defined]
  W = window_size

  x_t = torch.randn(n, W, feature_dim, device=env.device)
  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t)
    x_t = scheduler.step(eps, x_t, t_int)

  window = (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low
  _prime_sim_and_buffer(env, env_ids, window)
