"""Reward hooks for the object-goal two-stage SDS prior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.object_goal_assets import (
  ObjectGoalRuntimeContextBuilder,
  body_pos_to_mesh_centroid,
)
from smp.rl.object_goal_features import ObjectGoalMotionBuffer
from smp.rl.object_goal_prior import DEFAULT_SDS_TIMESTEPS, ObjectGoalTwoStagePrior

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _env_tensor(env: "ManagerBasedRlEnv", name: str) -> torch.Tensor:
  if not hasattr(env, name):
    msg = f"Object-goal reward requires env.{name}"
    raise RuntimeError(msg)
  value = getattr(env, name)
  return torch.as_tensor(value, device=env.device, dtype=torch.float32)


def _expand_env_batch(value: torch.Tensor, num_envs: int) -> torch.Tensor:
  if value.ndim == 1:
    return value.view(1, -1).expand(num_envs, -1)
  if value.shape[0] == num_envs:
    return value
  if value.shape[0] == 1:
    return value.expand(num_envs, *value.shape[1:])
  msg = f"Cannot expand object-goal tensor shape {tuple(value.shape)} to {num_envs} envs"
  raise ValueError(msg)


def _current_object_pose_from_env(
  env: "ManagerBasedRlEnv",
) -> tuple[torch.Tensor, torch.Tensor]:
  if hasattr(env, "_object_goal_object_centroid_pos_w") and hasattr(
    env,
    "_object_goal_object_quat_w",
  ):
    centroid_w = _env_tensor(env, "_object_goal_object_centroid_pos_w")
    quat_w = _env_tensor(env, "_object_goal_object_quat_w")
    return centroid_w, quat_w

  if "object" not in env.scene:
    msg = (
      "Object-goal reward requires a real object entity or explicit "
      "env._object_goal_object_centroid_pos_w/env._object_goal_object_quat_w. "
      "A generic free box is not a valid main integration."
    )
    raise RuntimeError(msg)

  if not hasattr(env, "_object_goal_mesh_centroid_offset_local"):
    msg = (
      "Object entity is present, but env._object_goal_mesh_centroid_offset_local "
      "is missing. The reward needs the HF-BPS mesh centroid, not the simulator "
      "body-frame origin."
    )
    raise RuntimeError(msg)

  obj = env.scene["object"]
  root_pos = obj.data.root_link_pos_w
  quat_w = obj.data.root_link_quat_w
  offset = _env_tensor(env, "_object_goal_mesh_centroid_offset_local")
  offset = _expand_env_batch(offset, env.num_envs)
  centroid_w = body_pos_to_mesh_centroid(root_pos, quat_w, offset)
  return centroid_w, quat_w


def _prime_or_update_buffer(env: "ManagerBasedRlEnv") -> ObjectGoalMotionBuffer:
  if not hasattr(env, "_object_goal_prior"):
    msg = "Object-goal reward requires env._object_goal_prior"
    raise RuntimeError(msg)
  prior: ObjectGoalTwoStagePrior = env._object_goal_prior  # type: ignore[attr-defined]
  robot = env.scene["robot"]
  origins = env.scene.env_origins
  object_centroid_w, object_quat_w = _current_object_pose_from_env(env)
  root_pos = robot.data.root_link_pos_w - origins
  object_centroid = object_centroid_w - origins
  root_quat = robot.data.root_link_quat_w
  dof_pos = robot.data.joint_pos

  if not hasattr(env, "_object_goal_buffer"):
    buffer = ObjectGoalMotionBuffer(
      num_envs=env.num_envs,
      window_size=prior.window_size,
      g1_diffusion_root=prior.g1_diffusion_root,
      device=env.device,
    )
    env_ids = torch.arange(env.num_envs, device=env.device)

    def repeat_frame(x: torch.Tensor) -> torch.Tensor:
      return x[:, None, ...].expand(-1, prior.window_size, *x.shape[1:]).clone()

    buffer.reset(
      env_ids,
      repeat_frame(root_pos),
      repeat_frame(root_quat),
      repeat_frame(dof_pos),
      repeat_frame(object_centroid),
      repeat_frame(object_quat_w),
    )
    env._object_goal_buffer = buffer  # type: ignore[attr-defined]
    return buffer

  buffer = env._object_goal_buffer  # type: ignore[attr-defined]
  buffer.update(
    root_pos=root_pos,
    root_quat_wxyz=root_quat,
    dof_pos=dof_pos,
    object_centroid_pos=object_centroid,
    object_quat_wxyz=object_quat_w,
  )
  return buffer


def _get_object_context(
  env: "ManagerBasedRlEnv",
  buffer: ObjectGoalMotionBuffer,
) -> dict[str, torch.Tensor | None]:
  goal_raw = _env_tensor(env, "_object_goal_final_object_pose_raw")

  if hasattr(env, "_object_goal_runtime_context_builder"):
    builder: ObjectGoalRuntimeContextBuilder = env._object_goal_runtime_context_builder  # type: ignore[attr-defined]
    runtime = builder.build_context_from_centroid_window(
      buffer.object_centroid_pos,
      buffer.object_quat_wxyz,
    )
    bps_encoding = runtime["bps_encoding"]
    static_bps = runtime["static_bps_context"]
    object_verts = runtime["object_verts"]
    object_rotations = runtime["object_rotations"]
  else:
    bps_encoding = _env_tensor(env, "_object_goal_bps_encoding")
    static_bps = _env_tensor(env, "_object_goal_static_bps_context")
    object_verts = (
      _env_tensor(env, "_object_goal_object_verts")
      if hasattr(env, "_object_goal_object_verts")
      else None
    )
    object_rotations = (
      _env_tensor(env, "_object_goal_object_rotations")
      if hasattr(env, "_object_goal_object_rotations")
      else None
    )

  contact_labels = (
    _env_tensor(env, "_object_goal_contact_labels")
    if hasattr(env, "_object_goal_contact_labels")
    else None
  )
  return {
    "bps_encoding": _expand_env_batch(bps_encoding, env.num_envs),
    "static_bps_context": _expand_env_batch(static_bps, env.num_envs),
    "goal_raw": _expand_env_batch(goal_raw, env.num_envs),
    "object_verts": (
      None if object_verts is None else _expand_env_batch(object_verts, env.num_envs)
    ),
    "object_rotations": (
      None if object_rotations is None else _expand_env_batch(object_rotations, env.num_envs)
    ),
    "contact_labels": (
      None if contact_labels is None else _expand_env_batch(contact_labels, env.num_envs)
    ),
  }


def object_goal_smp_guidance_reward(
  env: "ManagerBasedRlEnv",
  fixed_timesteps: tuple[int, ...] = DEFAULT_SDS_TIMESTEPS,
  ws: float = 6.0,
  normalize: bool = True,
  diagnostic_skip_rectification: bool = False,
) -> torch.Tensor:
  """Compute Stage 2 SDS reward after online Stage 1 hand conditioning."""
  prior: ObjectGoalTwoStagePrior = env._object_goal_prior  # type: ignore[attr-defined]
  buffer = _prime_or_update_buffer(env)
  x0_raw = buffer.compute_features()
  object_pose = x0_raw[..., 38:47]
  object_centroid = object_pose[..., :3]
  context = _get_object_context(env, buffer)

  normalizer = None
  if normalize:
    if not hasattr(env, "_object_goal_smp_normalizer"):
      from smp.rl.utils import DiffNormalizer

      env._object_goal_smp_normalizer = DiffNormalizer(  # type: ignore[attr-defined]
        prior.stage2_num_timesteps,
        torch.device(env.device),
      )
    normalizer = env._object_goal_smp_normalizer  # type: ignore[attr-defined]

  reward, diagnostics = prior.compute_reward_from_raw_inputs(
    x0_raw=x0_raw,
    bps_encoding=context["bps_encoding"],
    object_centroid=object_centroid,
    object_pose=object_pose,
    static_bps_context=context["static_bps_context"],
    goal_raw=context["goal_raw"],
    object_verts=context["object_verts"],
    object_rotations=context["object_rotations"],
    contact_labels=context["contact_labels"],
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    normalizer=normalizer,
    normalize=normalize,
    diagnostic_skip_rectification=diagnostic_skip_rectification,
  )
  env._object_goal_smp_diagnostics = diagnostics  # type: ignore[attr-defined]
  env._object_goal_smp_raw_err = diagnostics["sds_error"]  # type: ignore[attr-defined]
  env._object_goal_smp_reward = reward  # type: ignore[attr-defined]
  return reward


def object_goal_task_smp_product(
  env: "ManagerBasedRlEnv",
  task_terms: tuple["TaskTerm", ...],
  fixed_timesteps: tuple[int, ...] = DEFAULT_SDS_TIMESTEPS,
  ws: float = 6.0,
  normalize: bool = True,
  diagnostic_skip_rectification: bool = False,
) -> torch.Tensor:
  """``(sum_i w_i task_i(env)) * object_goal_smp``."""
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * object_goal_smp_guidance_reward(
    env,
    fixed_timesteps=fixed_timesteps,
    ws=ws,
    normalize=normalize,
    diagnostic_skip_rectification=diagnostic_skip_rectification,
  )
