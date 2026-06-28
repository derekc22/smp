"""Runtime HF-BPS object asset and context helpers.

The object-goal prior uses HF-BPS semantics: object position is the mesh
centroid, while a MuJoCo floating body may be authored at the original mesh
origin.  This module keeps that mapping explicit instead of letting task code
silently treat body origin as centroid.
"""

from __future__ import annotations

import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smp.rl.object_goal_features import DEFAULT_WINDOW_SIZE, STATIC_BPS_DIM


DEFAULT_BPS_RADIUS = 1.0
DEFAULT_HF_BPS_PKL = (
  "../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl"
)


def _install_numpy_pickle_aliases() -> None:
  if "numpy._core" not in sys.modules:
    core_pkg = types.ModuleType("numpy._core")
    core_pkg.__path__ = []
    sys.modules["numpy._core"] = core_pkg
  if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
  if "numpy._core.numerictypes" not in sys.modules:
    sys.modules["numpy._core.numerictypes"] = np.core.numerictypes
  if "numpy._core.umath" not in sys.modules:
    sys.modules["numpy._core.umath"] = np.core.umath


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
  out = Path(path).expanduser()
  if not out.is_absolute() and base is not None:
    out = Path(base).expanduser() / out
  return out.resolve()


def load_hf_bps_pickle(path: str | Path) -> dict[str, Any]:
  input_path = resolve_path(path)
  if not input_path.exists():
    raise FileNotFoundError(f"HF-BPS PKL not found: {input_path}")
  _install_numpy_pickle_aliases()
  with open(input_path, "rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    msg = f"Expected HF-BPS PKL to contain a dict, got {type(data).__name__}"
    raise TypeError(msg)
  return data


def load_obj_vertices(path: str | Path) -> np.ndarray:
  mesh_path = resolve_path(path)
  if not mesh_path.exists():
    raise FileNotFoundError(f"Object mesh file not found: {mesh_path}")
  verts: list[tuple[float, float, float]] = []
  with open(mesh_path, "r", errors="replace") as f:
    for line in f:
      if line.startswith("v "):
        parts = line.split()
        if len(parts) >= 4:
          verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
  if not verts:
    raise ValueError(f"No OBJ vertices found in {mesh_path}")
  return np.asarray(verts, dtype=np.float32)


def quat_xyzw_to_wxyz_np(quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float32)
  if quat.shape[-1] != 4:
    msg = f"Expected xyzw quaternion last dim 4, got {quat.shape}"
    raise ValueError(msg)
  return quat[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw_np(quat: np.ndarray) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float32)
  if quat.shape[-1] != 4:
    msg = f"Expected wxyz quaternion last dim 4, got {quat.shape}"
    raise ValueError(msg)
  return quat[..., [1, 2, 3, 0]]


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
  """Convert normalized or unnormalized ``wxyz`` quaternions to matrices."""
  if quat.shape[-1] != 4:
    msg = f"Expected wxyz quaternion last dim 4, got {tuple(quat.shape)}"
    raise ValueError(msg)
  q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
  w, x, y, z = q.unbind(dim=-1)
  two = torch.as_tensor(2.0, dtype=q.dtype, device=q.device)
  mat = torch.empty(*q.shape[:-1], 3, 3, dtype=q.dtype, device=q.device)
  mat[..., 0, 0] = 1 - two * (y * y + z * z)
  mat[..., 0, 1] = two * (x * y - z * w)
  mat[..., 0, 2] = two * (x * z + y * w)
  mat[..., 1, 0] = two * (x * y + z * w)
  mat[..., 1, 1] = 1 - two * (x * x + z * z)
  mat[..., 1, 2] = two * (y * z - x * w)
  mat[..., 2, 0] = two * (x * z - y * w)
  mat[..., 2, 1] = two * (y * z + x * w)
  mat[..., 2, 2] = 1 - two * (x * x + y * y)
  return mat


def quat_wxyz_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
  rot = quat_wxyz_to_matrix(quat)
  return torch.einsum("...ij,...j->...i", rot, vec)


def rotate_points_wxyz(quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
  rot = quat_wxyz_to_matrix(quat)
  return torch.einsum("...ij,...vj->...vi", rot, points)


def body_pos_to_mesh_centroid(
  body_pos_w: torch.Tensor,
  body_quat_wxyz: torch.Tensor,
  body_to_centroid_offset_local: torch.Tensor,
) -> torch.Tensor:
  """Map MuJoCo body-origin pose to HF-BPS mesh-centroid position."""
  body_pos_w = torch.as_tensor(body_pos_w, dtype=torch.float32)
  body_quat_wxyz = torch.as_tensor(
    body_quat_wxyz,
    dtype=torch.float32,
    device=body_pos_w.device,
  )
  offset = torch.as_tensor(
    body_to_centroid_offset_local,
    dtype=torch.float32,
    device=body_pos_w.device,
  )
  while offset.ndim < body_pos_w.ndim:
    offset = offset.unsqueeze(0)
  return body_pos_w + quat_wxyz_apply(body_quat_wxyz, offset)


def mesh_centroid_to_body_pos(
  centroid_w: torch.Tensor,
  body_quat_wxyz: torch.Tensor,
  body_to_centroid_offset_local: torch.Tensor,
) -> torch.Tensor:
  centroid_w = torch.as_tensor(centroid_w, dtype=torch.float32)
  body_quat_wxyz = torch.as_tensor(
    body_quat_wxyz,
    dtype=torch.float32,
    device=centroid_w.device,
  )
  offset = torch.as_tensor(
    body_to_centroid_offset_local,
    dtype=torch.float32,
    device=centroid_w.device,
  )
  while offset.ndim < centroid_w.ndim:
    offset = offset.unsqueeze(0)
  return centroid_w - quat_wxyz_apply(body_quat_wxyz, offset)


def world_vertices_from_body_pose(
  body_pos_w: torch.Tensor,
  body_quat_wxyz: torch.Tensor,
  mesh_vertices_local: torch.Tensor,
) -> torch.Tensor:
  """Transform canonical mesh-origin vertices into world/env-relative frame."""
  body_pos_w = torch.as_tensor(body_pos_w, dtype=torch.float32)
  body_quat_wxyz = torch.as_tensor(
    body_quat_wxyz,
    dtype=torch.float32,
    device=body_pos_w.device,
  )
  verts = torch.as_tensor(
    mesh_vertices_local,
    dtype=torch.float32,
    device=body_pos_w.device,
  )
  while verts.ndim < body_pos_w.ndim + 1:
    verts = verts.unsqueeze(0)
  rotated = rotate_points_wxyz(body_quat_wxyz, verts)
  return rotated + body_pos_w.unsqueeze(-2)


def _ensure_batched_window(
  value: torch.Tensor,
  last_dim: int,
  name: str,
) -> torch.Tensor:
  value = torch.as_tensor(value, dtype=torch.float32)
  if value.shape[-1] != last_dim:
    msg = f"{name} last dim must be {last_dim}, got {tuple(value.shape)}"
    raise ValueError(msg)
  if value.ndim == 2:
    return value[:, None, :]
  if value.ndim == 3:
    return value
  msg = f"{name} must have shape (B, {last_dim}) or (B, T, {last_dim}), got {tuple(value.shape)}"
  raise ValueError(msg)


def compute_bps_encoding(
  object_verts: torch.Tensor,
  object_centroid: torch.Tensor,
  bps_basis: torch.Tensor,
  bps_radius: float = DEFAULT_BPS_RADIUS,
  torch_chunk_size: int = 128,
) -> torch.Tensor:
  """Mirror g1 preprocessing BPS: basis_world - nearest_object_vertex."""
  verts = torch.as_tensor(object_verts, dtype=torch.float32)
  centroid = torch.as_tensor(object_centroid, dtype=torch.float32, device=verts.device)
  basis = torch.as_tensor(bps_basis, dtype=torch.float32, device=verts.device)
  if verts.ndim != 4 or verts.shape[-1] != 3:
    msg = f"object_verts must be (B, T, V, 3), got {tuple(verts.shape)}"
    raise ValueError(msg)
  if centroid.shape != verts.shape[:2] + (3,):
    msg = f"object_centroid shape {tuple(centroid.shape)} incompatible with {tuple(verts.shape)}"
    raise ValueError(msg)
  if basis.ndim != 2 or basis.shape[-1] != 3:
    msg = f"bps_basis must be (P, 3), got {tuple(basis.shape)}"
    raise ValueError(msg)

  basis_world = basis.view(1, 1, -1, 3) * float(bps_radius) + centroid[:, :, None, :]
  bsz, horizon, num_points, _ = basis_world.shape
  flat_basis = basis_world.reshape(bsz * horizon, num_points, 3)
  flat_verts = verts.reshape(bsz * horizon, verts.shape[-2], 3)
  flat_out = torch.empty_like(flat_basis)

  try:
    from scipy.spatial import cKDTree  # type: ignore[import-not-found]
  except ImportError:
    cKDTree = None

  if cKDTree is not None and flat_basis.device.type == "cpu":
    for idx in range(flat_basis.shape[0]):
      tree = cKDTree(flat_verts[idx].numpy())
      _, nearest = tree.query(flat_basis[idx].numpy())
      flat_out[idx] = flat_basis[idx] - flat_verts[idx, torch.from_numpy(nearest)]
  else:
    for idx in range(flat_basis.shape[0]):
      for start in range(0, num_points, torch_chunk_size):
        end = min(start + torch_chunk_size, num_points)
        dists = torch.cdist(flat_basis[idx, start:end], flat_verts[idx])
        nearest = torch.argmin(dists, dim=-1)
        flat_out[idx, start:end] = flat_basis[idx, start:end] - flat_verts[idx, nearest]
  return flat_out.reshape(bsz, horizon, num_points * 3)


@dataclass(frozen=True)
class HFObjectAssetMetadata:
  """Real HF-BPS object metadata used by the mjlab object-goal task."""

  object_name: str
  sequence_name: str
  input_pkl: Path
  mesh_file: Path
  mesh_scale: float
  mesh_vertices_original: np.ndarray
  mesh_vertices_local: np.ndarray
  mesh_centroid_offset_local: np.ndarray
  bps_basis: np.ndarray
  bps_radius: float
  sample: dict[str, Any]
  local_vertex_validation_error: float

  @property
  def num_vertices(self) -> int:
    return int(self.mesh_vertices_local.shape[0])

  def initial_body_pose(
    self,
    frame: int = 0,
  ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    centroid = np.asarray(self.sample["object_centroid"], dtype=np.float32)[frame]
    quat_wxyz = quat_xyzw_to_wxyz_np(np.asarray(self.sample["object_rot"], dtype=np.float32))[frame]
    centroid_t = torch.from_numpy(centroid).view(1, 3)
    quat_t = torch.from_numpy(quat_wxyz).view(1, 4)
    offset_t = torch.from_numpy(self.mesh_centroid_offset_local.astype(np.float32))
    body_pos = mesh_centroid_to_body_pos(centroid_t, quat_t, offset_t)[0].numpy()
    return tuple(float(x) for x in body_pos), tuple(float(x) for x in quat_wxyz)

  def make_mujoco_spec(
    self,
    body_name: str = "prop",
    mass: float = 1.0,
    rgba: tuple[float, float, float, float] = (0.55, 0.72, 0.95, 1.0),
  ):
    """Build a floating MuJoCo mesh body at the original mesh-origin frame."""
    import mujoco

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh()
    mesh.name = f"{self.object_name}_mesh"
    mesh.file = str(self.mesh_file)
    mesh.scale[:] = (self.mesh_scale, self.mesh_scale, self.mesh_scale)

    body = spec.worldbody.add_body(name=body_name)
    body.add_freejoint(name=f"{body_name}_freejoint")
    body.add_geom(
      name=f"{self.object_name}_mesh_geom",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=mesh.name,
      mass=float(mass),
      rgba=rgba,
      friction=(0.8, 0.01, 0.001),
    )
    return spec


def load_hf_object_asset_metadata(
  input_pkl: str | Path = DEFAULT_HF_BPS_PKL,
  g1_diffusion_root: str | Path | None = None,
  bps_basis_path: str | Path | None = None,
  bps_radius: float = DEFAULT_BPS_RADIUS,
) -> HFObjectAssetMetadata:
  base = resolve_path(g1_diffusion_root) if g1_diffusion_root is not None else None
  input_path = resolve_path(input_pkl, base=base)
  data = load_hf_bps_pickle(input_path)
  required = (
    "object_name",
    "mesh_file",
    "object_mesh_scale",
    "object_verts",
    "object_centroid",
    "object_rotation",
    "object_rot",
    "bps_encoding",
  )
  missing = [name for name in required if name not in data or data[name] is None]
  if missing:
    msg = f"HF-BPS object metadata missing required fields: {', '.join(missing)}"
    raise KeyError(msg)

  mesh_file = resolve_path(str(data["mesh_file"]))
  mesh_scale = float(data["object_mesh_scale"])
  if not np.isfinite(mesh_scale) or mesh_scale <= 0.0:
    raise ValueError(f"object_mesh_scale must be positive finite, got {mesh_scale}")

  original_vertices = load_obj_vertices(mesh_file).astype(np.float32)
  mesh_vertices_local = original_vertices * mesh_scale
  centroid_offset = mesh_vertices_local.mean(axis=0).astype(np.float32)

  if bps_basis_path is None:
    if base is None:
      base = input_path.parents[2]
    bps_basis_path = base / "data" / "hf_bps_preprocessed" / "bps_basis_points.npy"
  bps_basis_file = resolve_path(bps_basis_path)
  if not bps_basis_file.exists():
    raise FileNotFoundError(f"BPS basis file not found: {bps_basis_file}")
  bps_basis = np.load(bps_basis_file).astype(np.float32)
  if bps_basis.ndim != 2 or bps_basis.shape[1] != 3:
    msg = f"BPS basis must be (P, 3), got {bps_basis.shape}"
    raise ValueError(msg)

  object_verts = np.asarray(data["object_verts"], dtype=np.float64)
  object_centroid = np.asarray(data["object_centroid"], dtype=np.float64)
  object_rotation = np.asarray(data["object_rotation"], dtype=np.float64)
  local_from_sample = (
    object_rotation[0].T @ (object_verts[0] - object_centroid[0]).T
  ).T.astype(np.float32)
  centered_vertices = mesh_vertices_local - centroid_offset
  validation_error = float(np.max(np.abs(local_from_sample - centered_vertices)))
  if validation_error > 5e-4:
    msg = (
      "HF-BPS mesh vertex convention mismatch: transformed sample vertices do "
      f"not match scaled OBJ vertices centered at centroid (max error {validation_error:.6g})."
    )
    raise ValueError(msg)

  return HFObjectAssetMetadata(
    object_name=str(data["object_name"]),
    sequence_name=str(data.get("seq_name", input_path.stem)),
    input_pkl=input_path,
    mesh_file=mesh_file,
    mesh_scale=mesh_scale,
    mesh_vertices_original=original_vertices,
    mesh_vertices_local=mesh_vertices_local,
    mesh_centroid_offset_local=centroid_offset,
    bps_basis=bps_basis,
    bps_radius=float(bps_radius),
    sample=data,
    local_vertex_validation_error=validation_error,
  )


class ObjectGoalRuntimeContextBuilder:
  """Build prior inputs from runtime MuJoCo object body poses."""

  def __init__(
    self,
    metadata: HFObjectAssetMetadata,
    device: torch.device | str = "cpu",
  ) -> None:
    self.metadata = metadata
    self.device = torch.device(device)
    self.mesh_vertices_local = torch.as_tensor(
      metadata.mesh_vertices_local,
      dtype=torch.float32,
      device=self.device,
    )
    self.centroid_offset_local = torch.as_tensor(
      metadata.mesh_centroid_offset_local,
      dtype=torch.float32,
      device=self.device,
    )
    self.bps_basis = torch.as_tensor(
      metadata.bps_basis,
      dtype=torch.float32,
      device=self.device,
    )

  def object_centroid_from_body_pose(
    self,
    body_pos_w: torch.Tensor,
    body_quat_wxyz: torch.Tensor,
  ) -> torch.Tensor:
    return body_pos_to_mesh_centroid(
      body_pos_w.to(self.device),
      body_quat_wxyz.to(self.device),
      self.centroid_offset_local,
    )

  def build_context_from_body_window(
    self,
    body_pos_w: torch.Tensor,
    body_quat_wxyz: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    body_pos = _ensure_batched_window(body_pos_w, 3, "body_pos_w").to(self.device)
    body_quat = _ensure_batched_window(
      body_quat_wxyz,
      4,
      "body_quat_wxyz",
    ).to(self.device)
    if body_pos.shape[:2] != body_quat.shape[:2]:
      msg = f"body pose window shapes mismatch: {tuple(body_pos.shape)} vs {tuple(body_quat.shape)}"
      raise ValueError(msg)

    centroid = self.object_centroid_from_body_pose(body_pos, body_quat)
    verts = world_vertices_from_body_pose(body_pos, body_quat, self.mesh_vertices_local)
    rotations = quat_wxyz_to_matrix(body_quat)
    bps = compute_bps_encoding(
      verts,
      centroid,
      self.bps_basis,
      bps_radius=self.metadata.bps_radius,
    )
    if bps.shape[-1] != STATIC_BPS_DIM:
      msg = f"Expected flattened BPS dim {STATIC_BPS_DIM}, got {tuple(bps.shape)}"
      raise ValueError(msg)
    static_bps = bps[:, :1].expand(-1, bps.shape[1], -1).clone()
    return {
      "object_centroid": centroid,
      "object_verts": verts,
      "object_rotations": rotations,
      "bps_encoding": bps,
      "static_bps_context": static_bps,
    }

  def build_context_from_centroid_window(
    self,
    object_centroid: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    centroid = _ensure_batched_window(object_centroid, 3, "object_centroid").to(
      self.device
    )
    quat = _ensure_batched_window(object_quat_wxyz, 4, "object_quat_wxyz").to(
      self.device
    )
    body_pos = mesh_centroid_to_body_pos(centroid, quat, self.centroid_offset_local)
    return self.build_context_from_body_window(body_pos, quat)


def hf_sample_window_tensors(
  metadata: HFObjectAssetMetadata,
  start: int = 0,
  window_size: int = DEFAULT_WINDOW_SIZE,
  device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
  data = metadata.sample
  end = start + window_size
  total = min(
    np.asarray(data["root_pos"]).shape[0],
    np.asarray(data["root_rot"]).shape[0],
    np.asarray(data["dof_pos"]).shape[0],
    np.asarray(data["object_centroid"]).shape[0],
    np.asarray(data["object_rot"]).shape[0],
  )
  if start < 0 or end > total:
    msg = f"Requested HF-BPS window [{start}:{end}] but only {total} frames are aligned"
    raise ValueError(msg)
  dev = torch.device(device)
  root_quat_wxyz = quat_xyzw_to_wxyz_np(np.asarray(data["root_rot"], dtype=np.float32))
  object_quat_wxyz = quat_xyzw_to_wxyz_np(
    np.asarray(data["object_rot"], dtype=np.float32)
  )
  tensors = {
    "root_pos": torch.as_tensor(data["root_pos"][start:end], dtype=torch.float32, device=dev).unsqueeze(0),
    "root_quat_wxyz": torch.as_tensor(root_quat_wxyz[start:end], dtype=torch.float32, device=dev).unsqueeze(0),
    "dof_pos": torch.as_tensor(data["dof_pos"][start:end], dtype=torch.float32, device=dev).unsqueeze(0),
    "object_centroid": torch.as_tensor(data["object_centroid"][start:end], dtype=torch.float32, device=dev).unsqueeze(0),
    "object_quat_wxyz": torch.as_tensor(object_quat_wxyz[start:end], dtype=torch.float32, device=dev).unsqueeze(0),
  }
  return tensors
