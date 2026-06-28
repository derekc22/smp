"""Object-goal feature utilities for the two-stage g1-diffusion prior.

This module is intentionally separate from ``MotionFeatureBuffer``.  The
object-goal prior scores the corrected 47D g1-diffusion layout:

  [root_pos(3), root_rot_6d(6), dof_pos(29), object_centroid_pos(3),
   object_rot_6d(6)]

Rot6D conversion is delegated to the configured ``g1-diffusion`` checkout so the
SMP bridge follows the exact checkpoint convention.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROBOT_STATE_DIM = 38
OBJECT_POSE_DIM = 9
ROBOT_OBJECT_STATE_DIM = ROBOT_STATE_DIM + OBJECT_POSE_DIM
STAGE2_COND_DIM = 3078
STATIC_BPS_DIM = 3072
HAND_DIM = 6
DEFAULT_WINDOW_SIZE = 300
NUM_G1_DOFS = 29


def _ensure_g1_root(g1_diffusion_root: str | Path) -> Path:
  root = Path(g1_diffusion_root).expanduser().resolve()
  if not root.is_dir():
    msg = f"g1_diffusion_root does not exist or is not a directory: {root}"
    raise FileNotFoundError(msg)
  root_str = str(root)
  if root_str not in sys.path:
    sys.path.insert(0, root_str)
  return root


def quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
  """Convert MuJoCo/mjlab ``wxyz`` quaternions to g1-diffusion ``xyzw``."""
  if quat.shape[-1] != 4:
    msg = f"Expected quaternion last dim 4, got {tuple(quat.shape)}"
    raise ValueError(msg)
  return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)


def _as_float_tensor(
  value: np.ndarray | torch.Tensor,
  device: torch.device,
) -> torch.Tensor:
  return torch.as_tensor(value, device=device, dtype=torch.float32)


def _check_last_dim(name: str, value: torch.Tensor, dim: int) -> None:
  if value.shape[-1] != dim:
    msg = f"{name} last dim must be {dim}, got shape {tuple(value.shape)}"
    raise ValueError(msg)


@dataclass(frozen=True)
class G1DiffusionFeatureImports:
  quat_to_rot6d_xyzw: Callable[[torch.Tensor], torch.Tensor]
  mat_to_quat_xyzw: Callable[[torch.Tensor], torch.Tensor]
  object_pose_from_data: Callable[..., np.ndarray]
  robot_object_motion_from_data: Callable[..., np.ndarray]


def load_g1_feature_imports(g1_diffusion_root: str | Path) -> G1DiffusionFeatureImports:
  _ensure_g1_root(g1_diffusion_root)
  from utils.object_goal_features import (  # type: ignore[import-not-found]
    object_pose_from_data,
    robot_object_motion_from_data,
  )
  from utils.rotation import (  # type: ignore[import-not-found]
    mat_to_quat_xyzw,
    quat_to_rot6d_xyzw,
  )

  return G1DiffusionFeatureImports(
    quat_to_rot6d_xyzw=quat_to_rot6d_xyzw,
    mat_to_quat_xyzw=mat_to_quat_xyzw,
    object_pose_from_data=object_pose_from_data,
    robot_object_motion_from_data=robot_object_motion_from_data,
  )


class ObjectGoalFeatureBuilder:
  """Build g1-diffusion object-goal features from HF-BPS data or tensors."""

  def __init__(
    self,
    g1_diffusion_root: str | Path,
    device: torch.device | str = "cpu",
  ) -> None:
    self.g1_diffusion_root = _ensure_g1_root(g1_diffusion_root)
    self.device = torch.device(device)
    self.imports = load_g1_feature_imports(self.g1_diffusion_root)

  def root_rot6d_from_xyzw(self, quat_xyzw: torch.Tensor) -> torch.Tensor:
    _check_last_dim("quat_xyzw", quat_xyzw, 4)
    return self.imports.quat_to_rot6d_xyzw(quat_xyzw)

  def rot6d_from_matrix(self, rot_mat: torch.Tensor) -> torch.Tensor:
    if rot_mat.shape[-2:] != (3, 3):
      msg = f"Expected rotation matrix shape (..., 3, 3), got {tuple(rot_mat.shape)}"
      raise ValueError(msg)
    quat = self.imports.mat_to_quat_xyzw(rot_mat)
    return self.imports.quat_to_rot6d_xyzw(quat)

  def object_rot6d_from_rotation(self, rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-1] == 6:
      return rotation
    if rotation.shape[-1] == 4:
      return self.root_rot6d_from_xyzw(rotation)
    if rotation.shape[-2:] == (3, 3):
      return self.rot6d_from_matrix(rotation)
    msg = f"Unsupported object rotation shape {tuple(rotation.shape)}"
    raise ValueError(msg)

  def build_robot_object_from_tensors(
    self,
    root_pos: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    dof_pos: torch.Tensor,
    object_centroid_pos: torch.Tensor,
    object_rotation_xyzw_or_mat_or_6d: torch.Tensor,
  ) -> torch.Tensor:
    root_pos = root_pos.to(device=self.device, dtype=torch.float32)
    root_quat_xyzw = root_quat_xyzw.to(device=self.device, dtype=torch.float32)
    dof_pos = dof_pos.to(device=self.device, dtype=torch.float32)
    object_centroid_pos = object_centroid_pos.to(device=self.device, dtype=torch.float32)
    object_rotation = object_rotation_xyzw_or_mat_or_6d.to(
      device=self.device,
      dtype=torch.float32,
    )

    _check_last_dim("root_pos", root_pos, 3)
    _check_last_dim("root_quat_xyzw", root_quat_xyzw, 4)
    _check_last_dim("dof_pos", dof_pos, NUM_G1_DOFS)
    _check_last_dim("object_centroid_pos", object_centroid_pos, 3)

    root_rot6d = self.root_rot6d_from_xyzw(root_quat_xyzw)
    object_rot6d = self.object_rot6d_from_rotation(object_rotation)
    robot_state = torch.cat([root_pos, root_rot6d, dof_pos], dim=-1)
    object_pose = torch.cat([object_centroid_pos, object_rot6d], dim=-1)
    motion = torch.cat([robot_state, object_pose], dim=-1)
    _check_last_dim("robot_object_motion", motion, ROBOT_OBJECT_STATE_DIM)
    return motion

  def build_robot_object_from_wxyz_tensors(
    self,
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    dof_pos: torch.Tensor,
    object_centroid_pos: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
  ) -> torch.Tensor:
    return self.build_robot_object_from_tensors(
      root_pos=root_pos,
      root_quat_xyzw=quat_wxyz_to_xyzw(root_quat_wxyz),
      dof_pos=dof_pos,
      object_centroid_pos=object_centroid_pos,
      object_rotation_xyzw_or_mat_or_6d=quat_wxyz_to_xyzw(object_quat_wxyz),
    )

  def hf_bps_window_to_tensors(
    self,
    data: dict[str, Any],
    start: int = 0,
    window_size: int = DEFAULT_WINDOW_SIZE,
  ) -> dict[str, torch.Tensor]:
    """Build a batched object-goal reward sample from one HF-BPS PKL dict."""
    end = start + window_size
    required = (
      "root_pos",
      "root_rot",
      "dof_pos",
      "object_pos",
      "object_rot",
      "bps_encoding",
      "object_centroid",
    )
    missing = [name for name in required if name not in data or data[name] is None]
    if missing:
      msg = f"HF-BPS sample missing required fields: {', '.join(missing)}"
      raise KeyError(msg)

    total_frames = min(
      np.asarray(data["root_pos"]).shape[0],
      np.asarray(data["root_rot"]).shape[0],
      np.asarray(data["dof_pos"]).shape[0],
      np.asarray(data["object_centroid"]).shape[0],
      np.asarray(data["bps_encoding"]).shape[0],
    )
    if start < 0 or end > total_frames:
      msg = (
        f"Requested window [{start}:{end}] but sample only has "
        f"{total_frames} aligned frames"
      )
      raise ValueError(msg)

    window_data: dict[str, Any] = {}
    for key, value in data.items():
      if isinstance(value, np.ndarray) and value.shape[:1] and value.shape[0] >= end:
        window_data[key] = value[start:end]
      else:
        window_data[key] = value

    motion_np = self.imports.robot_object_motion_from_data(
      window_data,
      length=window_size,
    )
    object_pose_np = self.imports.object_pose_from_data(
      window_data,
      length=window_size,
    )
    bps_np = np.asarray(window_data["bps_encoding"], dtype=np.float32)
    if bps_np.ndim == 3:
      bps_np = bps_np.reshape(window_size, -1)
    if bps_np.shape != (window_size, STATIC_BPS_DIM):
      msg = f"Expected flattened BPS shape {(window_size, STATIC_BPS_DIM)}, got {bps_np.shape}"
      raise ValueError(msg)
    static_bps_np = np.repeat(bps_np[:1], window_size, axis=0)
    centroid_np = np.asarray(window_data["object_centroid"], dtype=np.float32)

    out = {
      "x0_raw": _as_float_tensor(motion_np, self.device).unsqueeze(0),
      "bps": _as_float_tensor(bps_np, self.device).unsqueeze(0),
      "object_centroid": _as_float_tensor(centroid_np, self.device).unsqueeze(0),
      "object_pose": _as_float_tensor(object_pose_np, self.device).unsqueeze(0),
      "static_bps_context": _as_float_tensor(static_bps_np, self.device).unsqueeze(0),
      "goal_raw": _as_float_tensor(object_pose_np[-1], self.device).unsqueeze(0),
    }

    if data.get("object_verts") is not None:
      out["object_verts"] = _as_float_tensor(
        np.asarray(data["object_verts"], dtype=np.float32)[start:end],
        self.device,
      ).unsqueeze(0)
    if data.get("object_rotation") is not None:
      out["object_rotation"] = _as_float_tensor(
        np.asarray(data["object_rotation"], dtype=np.float32)[start:end],
        self.device,
      ).unsqueeze(0)
    if data.get("contact") is not None:
      out["contact_labels"] = _as_float_tensor(
        np.asarray(data["contact"], dtype=np.float32)[start:end],
        self.device,
      ).unsqueeze(0)
    return out


class ObjectGoalMotionBuffer:
  """Rolling object-goal buffer for the 47D prior.

  The caller is responsible for feeding positions in the same frame for the
  robot and object.  For mjlab envs, that should normally mean env-origin
  relative positions for both root and object centroid.
  """

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    g1_diffusion_root: str | Path,
    device: torch.device | str,
  ) -> None:
    self.num_envs = int(num_envs)
    self.window_size = int(window_size)
    self.device = torch.device(device)
    self.builder = ObjectGoalFeatureBuilder(g1_diffusion_root, self.device)

    self.root_pos = torch.zeros(self.num_envs, self.window_size, 3, device=self.device)
    self.root_quat_wxyz = torch.zeros(
      self.num_envs,
      self.window_size,
      4,
      device=self.device,
    )
    self.root_quat_wxyz[..., 0] = 1.0
    self.dof_pos = torch.zeros(
      self.num_envs,
      self.window_size,
      NUM_G1_DOFS,
      device=self.device,
    )
    self.object_centroid_pos = torch.zeros(
      self.num_envs,
      self.window_size,
      3,
      device=self.device,
    )
    self.object_quat_wxyz = torch.zeros(
      self.num_envs,
      self.window_size,
      4,
      device=self.device,
    )
    self.object_quat_wxyz[..., 0] = 1.0

  def reset(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    dof_pos: torch.Tensor,
    object_centroid_pos: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
  ) -> None:
    if env_ids.numel() == 0:
      return
    self.root_pos[env_ids] = root_pos.to(self.device)
    self.root_quat_wxyz[env_ids] = root_quat_wxyz.to(self.device)
    self.dof_pos[env_ids] = dof_pos.to(self.device)
    self.object_centroid_pos[env_ids] = object_centroid_pos.to(self.device)
    self.object_quat_wxyz[env_ids] = object_quat_wxyz.to(self.device)

  def update(
    self,
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    dof_pos: torch.Tensor,
    object_centroid_pos: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
  ) -> None:
    self.root_pos = torch.roll(self.root_pos, shifts=-1, dims=1)
    self.root_quat_wxyz = torch.roll(self.root_quat_wxyz, shifts=-1, dims=1)
    self.dof_pos = torch.roll(self.dof_pos, shifts=-1, dims=1)
    self.object_centroid_pos = torch.roll(self.object_centroid_pos, shifts=-1, dims=1)
    self.object_quat_wxyz = torch.roll(self.object_quat_wxyz, shifts=-1, dims=1)
    self.root_pos[:, -1] = root_pos.to(self.device)
    self.root_quat_wxyz[:, -1] = root_quat_wxyz.to(self.device)
    self.dof_pos[:, -1] = dof_pos.to(self.device)
    self.object_centroid_pos[:, -1] = object_centroid_pos.to(self.device)
    self.object_quat_wxyz[:, -1] = object_quat_wxyz.to(self.device)

  def compute_features(self) -> torch.Tensor:
    return self.builder.build_robot_object_from_wxyz_tensors(
      root_pos=self.root_pos,
      root_quat_wxyz=self.root_quat_wxyz,
      dof_pos=self.dof_pos,
      object_centroid_pos=self.object_centroid_pos,
      object_quat_wxyz=self.object_quat_wxyz,
    )


def validate_runtime_object_metadata(metadata: dict[str, Any]) -> None:
  """Fail loudly when runtime object data is only a generic free body."""
  required = (
    "object_mesh_vertices",
    "object_mesh_scale",
    "bps_basis",
    "mesh_centroid_offset_local",
  )
  missing = [name for name in required if name not in metadata or metadata[name] is None]
  if missing:
    msg = (
      "Object-goal runtime metadata is incomplete. A real HF-BPS object asset "
      "must provide " + ", ".join(required) + f"; missing {', '.join(missing)}."
    )
    raise RuntimeError(msg)
