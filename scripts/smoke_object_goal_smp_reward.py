"""Reward-only smoke test for the two-stage object-goal SMP bridge."""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

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

from smp.rl.object_goal_features import DEFAULT_WINDOW_SIZE, ObjectGoalFeatureBuilder
from smp.rl.object_goal_prior import DEFAULT_SDS_TIMESTEPS, ObjectGoalTwoStagePrior


def _parse_timesteps(value: str) -> tuple[int, ...]:
  parts = [part.strip() for part in value.replace(" ", ",").split(",")]
  timesteps = tuple(int(part) for part in parts if part)
  if not timesteps:
    raise argparse.ArgumentTypeError("--fixed-timesteps must contain at least one integer")
  return timesteps


def _resolve(path: str | Path) -> Path:
  return Path(path).expanduser().resolve()


def _format_tensor(value: torch.Tensor) -> str:
  if value.numel() == 1:
    return f"{float(value.detach().cpu().reshape(-1)[0]):.6g}"
  flat = value.detach().cpu().reshape(-1)
  return (
    f"shape={tuple(value.shape)} min={float(flat.min()):.6g} "
    f"mean={float(flat.mean()):.6g} max={float(flat.max()):.6g}"
  )


def _assert_finite_diagnostics(diagnostics: dict[str, Any]) -> None:
  for name, value in diagnostics.items():
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
      msg = f"diagnostic {name!r} contains non-finite values"
      raise AssertionError(msg)


def _contact_summary(contact_metadata: dict[str, Any]) -> str:
  left = contact_metadata.get("left_contact_frames")
  right = contact_metadata.get("right_contact_frames")
  if left is None or right is None:
    return str(contact_metadata.get("rectification", "metadata unavailable"))
  left_count = int(np.asarray(left).sum())
  right_count = int(np.asarray(right).sum())
  return (
    f"left_contact_frames={left_count} right_contact_frames={right_count} "
    f"used_fallback={contact_metadata.get('used_fallback')}"
  )


def main() -> None:
  default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
  parser = argparse.ArgumentParser(
    description="Compute a finite Stage 2 SDS reward from real HF-BPS inputs."
  )
  parser.add_argument("--g1-diffusion-root", default="../g1-diffusion")
  parser.add_argument("--stage1-ckpt", required=True)
  parser.add_argument("--stage2-ckpt", required=True)
  parser.add_argument(
    "--input-pkl",
    default="../g1-diffusion/data/hf_bps_preprocessed/omomo_sub3_largebox_003_sample1.pkl",
  )
  parser.add_argument("--device", default=default_device)
  parser.add_argument(
    "--ws",
    type=float,
    default=None,
    help="Optional SDS reward scale override for smoke tests.",
  )
  parser.add_argument(
    "--fixed-timesteps",
    type=_parse_timesteps,
    default=DEFAULT_SDS_TIMESTEPS,
    help="Comma-separated diffusion timesteps, e.g. 160,300,440.",
  )
  parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
  parser.add_argument("--diagnostic-skip-rectification", action="store_true")
  args = parser.parse_args()

  g1_root = _resolve(args.g1_diffusion_root)
  input_pkl = _resolve(args.input_pkl)
  device = torch.device(args.device)
  if not input_pkl.exists():
    raise FileNotFoundError(f"input PKL not found: {input_pkl}")

  with open(input_pkl, "rb") as f:
    data = pickle.load(f)

  builder = ObjectGoalFeatureBuilder(g1_root, device=device)
  sample = builder.hf_bps_window_to_tensors(
    data,
    start=0,
    window_size=int(args.window_size),
  )

  prior = ObjectGoalTwoStagePrior(
    g1_diffusion_root=g1_root,
    stage1_ckpt_path=args.stage1_ckpt,
    stage2_ckpt_path=args.stage2_ckpt,
    device=device,
    fixed_timesteps=args.fixed_timesteps,
  )

  reward, diagnostics = prior.compute_reward_from_raw_inputs(
    x0_raw=sample["x0_raw"],
    bps_encoding=sample["bps"],
    object_centroid=sample["object_centroid"],
    object_pose=sample["object_pose"],
    static_bps_context=sample["static_bps_context"],
    goal_raw=sample["goal_raw"],
    object_verts=sample.get("object_verts"),
    object_rotations=sample.get("object_rotation"),
    contact_labels=sample.get("contact_labels"),
    fixed_timesteps=args.fixed_timesteps,
    ws=args.ws,
    diagnostic_skip_rectification=bool(args.diagnostic_skip_rectification),
  )

  if reward.shape != (1,):
    msg = f"Expected reward shape (1,), got {tuple(reward.shape)}"
    raise AssertionError(msg)
  if not torch.isfinite(reward).all():
    raise AssertionError("reward contains non-finite values")
  _assert_finite_diagnostics(diagnostics)

  print("Object-goal SMP reward smoke test passed")
  print(f"  input_pkl: {input_pkl}")
  print(f"  stage1_ckpt: {Path(args.stage1_ckpt)}")
  print(f"  stage2_ckpt: {Path(args.stage2_ckpt)}")
  print(f"  x0_raw: {tuple(sample['x0_raw'].shape)}")
  print(f"  stage2_condition: (1, {args.window_size}, 3078)")
  print(f"  reward: {_format_tensor(reward)}")
  for name in (
    "sds_error",
    "sds_error_normalized",
    "epsilon_norm",
    "epsilon_hat_norm",
    "x0_norm",
    "condition_norm",
    "goal_norm",
  ):
    print(f"  {name}: {_format_tensor(diagnostics[name])}")
  print(f"  timesteps: {diagnostics['timesteps']}")
  print(f"  rectification: {_contact_summary(diagnostics['contact_metadata'])}")


if __name__ == "__main__":
  main()
