"""Task reward components for the HF-BPS object-goal task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.object_goal_assets import body_pos_to_mesh_centroid

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


def _current_object_centroid_local(env: "ManagerBasedRlEnv") -> torch.Tensor:
  obj = env.scene["object"]
  offset = torch.as_tensor(
    env._object_goal_mesh_centroid_offset_local,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  offset = _expand_env_batch(offset, env.num_envs)
  centroid_w = body_pos_to_mesh_centroid(
    obj.data.root_link_pos_w,
    obj.data.root_link_quat_w,
    offset,
  )
  return centroid_w - env.scene.env_origins


def _goal_raw(env: "ManagerBasedRlEnv") -> torch.Tensor:
  goal = torch.as_tensor(
    env._object_goal_final_object_pose_raw,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  return _expand_env_batch(goal, env.num_envs)


def object_goal_position(
  env: "ManagerBasedRlEnv",
  pos_err_scale: float = 8.0,
) -> torch.Tensor:
  """``exp(-scale * ||object_centroid - goal_centroid||^2)``."""
  pos_err = _current_object_centroid_local(env) - _goal_raw(env)[:, :3]
  return torch.exp(-float(pos_err_scale) * (pos_err * pos_err).sum(dim=-1))


def object_goal_orientation(
  env: "ManagerBasedRlEnv",
  ori_err_scale: float = 4.0,
) -> torch.Tensor:
  """Quaternion-dot orientation tracking against the HF-BPS final pose."""
  if not hasattr(env, "_object_goal_final_object_quat_wxyz"):
    return torch.ones(env.num_envs, device=env.device)
  obj = env.scene["object"]
  cur = obj.data.root_link_quat_w
  cur = cur / torch.linalg.vector_norm(cur, dim=-1, keepdim=True).clamp_min(1e-8)
  goal = torch.as_tensor(
    env._object_goal_final_object_quat_wxyz,  # type: ignore[attr-defined]
    dtype=torch.float32,
    device=env.device,
  )
  goal = _expand_env_batch(goal, env.num_envs)
  goal = goal / torch.linalg.vector_norm(goal, dim=-1, keepdim=True).clamp_min(1e-8)
  dot = torch.sum(cur * goal, dim=-1).abs().clamp(max=1.0)
  err = 1.0 - dot
  return torch.exp(-float(ori_err_scale) * err * err)
