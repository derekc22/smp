"""Validate HF-BPS object metadata and runtime context construction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from smp.rl.object_goal_assets import (  # noqa: E402
  DEFAULT_HF_BPS_PKL,
  ObjectGoalRuntimeContextBuilder,
  body_pos_to_mesh_centroid,
  hf_sample_window_tensors,
  load_hf_object_asset_metadata,
  mesh_centroid_to_body_pos,
)


def _resolve(path: str | Path) -> Path:
  return Path(path).expanduser().resolve()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--g1-diffusion-root", default="../g1-diffusion")
  parser.add_argument("--input-pkl", default=DEFAULT_HF_BPS_PKL)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--window-size", type=int, default=300)
  parser.add_argument("--num-envs", type=int, default=1)
  args = parser.parse_args()

  device = torch.device(args.device)
  g1_root = _resolve(args.g1_diffusion_root)
  input_pkl = _resolve(args.input_pkl)
  metadata = load_hf_object_asset_metadata(
    input_pkl=input_pkl,
    g1_diffusion_root=g1_root,
  )
  sample = hf_sample_window_tensors(
    metadata,
    start=0,
    window_size=args.window_size,
    device=device,
  )
  builder = ObjectGoalRuntimeContextBuilder(metadata, device=device)

  centroid = sample["object_centroid"].expand(args.num_envs, -1, -1).clone()
  quat = sample["object_quat_wxyz"].expand(args.num_envs, -1, -1).clone()
  offset = torch.as_tensor(
    metadata.mesh_centroid_offset_local,
    dtype=torch.float32,
    device=device,
  )
  body_pos = mesh_centroid_to_body_pos(centroid, quat, offset)
  reconstructed = body_pos_to_mesh_centroid(body_pos, quat, offset)
  torch.testing.assert_close(reconstructed, centroid, atol=1e-5, rtol=1e-5)

  context = builder.build_context_from_body_window(body_pos, quat)
  expected_bps_shape = (args.num_envs, args.window_size, 3072)
  expected_verts_shape = (
    args.num_envs,
    args.window_size,
    metadata.num_vertices,
    3,
  )
  assert context["bps_encoding"].shape == expected_bps_shape
  assert context["static_bps_context"].shape == expected_bps_shape
  assert context["object_verts"].shape == expected_verts_shape
  assert context["object_rotations"].shape == (args.num_envs, args.window_size, 3, 3)
  for name, value in context.items():
    if not torch.isfinite(value).all():
      raise AssertionError(f"{name} contains non-finite values")

  print("Object-goal asset/context check passed")
  print(f"  object: {metadata.object_name}")
  print(f"  sequence: {metadata.sequence_name}")
  print(f"  input_pkl: {metadata.input_pkl}")
  print(f"  mesh_file: {metadata.mesh_file}")
  print(f"  mesh_scale: {metadata.mesh_scale:.9g}")
  print(f"  mesh_vertices: {metadata.mesh_vertices_local.shape}")
  print(f"  body_origin_to_centroid_offset: {metadata.mesh_centroid_offset_local.tolist()}")
  print(f"  local_vertex_validation_error: {metadata.local_vertex_validation_error:.6g}")
  print(f"  bps_basis: {metadata.bps_basis.shape}")
  print(f"  bps_encoding: {tuple(context['bps_encoding'].shape)}")
  print(f"  object_verts: {tuple(context['object_verts'].shape)}")


if __name__ == "__main__":
  main()
