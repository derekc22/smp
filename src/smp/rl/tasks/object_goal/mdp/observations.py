"""Observation terms for the HF-BPS object-goal task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.object_goal_assets import body_pos_to_mesh_centroid
from smp.rl.object_goal_events import ensure_object_goal_hf_bps_context

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _expand_env_batch(value: torch.Tensor, num_envs: int) -> torch.Tensor:
  if value.ndim == 1:
    return value.view(1, -1).expand(num_envs, -1)
  if value.shape[0] == num_envs:
    return value
  if value.shape[0] == 1:
    return value.expand(num_envs, *value.shape[1:])
  msg = f"Cannot expand object-goal tensor shape {tuple(value.shape)} to {num_envs} envs"
  raise ValueError(msg)


def object_goal_observation(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """Expose current object pose, final goal, and centroid delta.

  Positions are env-origin-relative to match the 47D prior window. Rotations use
  MuJoCo/mjlab ``wxyz`` quaternions here; the reward prior converts to g1 rot6d
  inside the object-goal feature builder.
  """
  ensure_object_goal_hf_bps_context(env)
  obj = env.scene["object"]
  offset = torch.as_tensor(
    env._object_goal_mesh_centroid_offset_local,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  offset = _expand_env_batch(offset, env.num_envs)
  centroid = body_pos_to_mesh_centroid(
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
    offset,
  )
  centroid_local = centroid - env.scene.env_origins
  quat = obj.data.root_link_quat_w
  goal = torch.as_tensor(
    env._object_goal_final_object_pose_raw,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  goal = _expand_env_batch(goal, env.num_envs)
  return torch.cat(
    [
      centroid_local,
      quat,
      goal,
      goal[:, :3] - centroid_local,
    ],
    dim=-1,
  )
