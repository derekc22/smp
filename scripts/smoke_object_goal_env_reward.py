"""Reward-in-env smoke test for the object-goal SMP wrapper.

This uses a minimal env-like wrapper when mjlab is unavailable.  The object is
still represented as a MuJoCo-style floating body at the original mesh origin,
and the reward hook must recover the HF-BPS mesh centroid through the stored
body-origin-to-centroid offset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

from smp.rl.object_goal_assets import (  # noqa: E402
  DEFAULT_HF_BPS_PKL,
  hf_sample_window_tensors,
  load_hf_object_asset_metadata,
  mesh_centroid_to_body_pos,
)
from smp.rl.object_goal_events import (  # noqa: E402
  init_object_goal_prior,
  object_goal_sample_reset,
)
from smp.rl.object_goal_rewards import object_goal_smp_guidance_reward  # noqa: E402


class _FakeEntity:
  def __init__(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    joint_pos: torch.Tensor | None = None,
  ) -> None:
    self.data = SimpleNamespace(
      root_link_pos_w=root_pos_w,
      root_link_quat_w=root_quat_w,
      joint_pos=joint_pos,
    )

  def write_root_state_to_sim(
    self,
    root_state: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    if env_ids is None:
      self.data.root_link_pos_w = root_state[:, :3].clone()
      self.data.root_link_quat_w = root_state[:, 3:7].clone()
    else:
      self.data.root_link_pos_w[env_ids] = root_state[:, :3]
      self.data.root_link_quat_w[env_ids] = root_state[:, 3:7]

  def write_joint_state_to_sim(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
  ) -> None:
    del velocity
    if self.data.joint_pos is None:
      self.data.joint_pos = position.clone()
    elif env_ids is None:
      self.data.joint_pos = position.clone()
    else:
      self.data.joint_pos[env_ids] = position


class _FakeScene(dict[str, Any]):
  def __init__(self, robot: _FakeEntity, obj: _FakeEntity, origins: torch.Tensor) -> None:
    super().__init__({"robot": robot, "object": obj})
    self.env_origins = origins


class _FakeEnv:
  def __init__(
    self,
    num_envs: int,
    device: torch.device,
    robot: _FakeEntity,
    obj: _FakeEntity,
  ) -> None:
    self.num_envs = int(num_envs)
    self.device = str(device)
    origins = torch.zeros(num_envs, 3, device=device)
    self.scene = _FakeScene(robot, obj, origins)
    self.common_step_counter = 0


def _resolve(path: str | Path) -> Path:
  return Path(path).expanduser().resolve()


def _latest_checkpoint(g1_root: Path, pattern: str) -> str:
  matches = sorted(g1_root.glob(pattern), key=lambda path: path.stat().st_mtime)
  return str(matches[-1]) if matches else ""


def _parse_timesteps(value: str) -> tuple[int, ...]:
  out = tuple(int(part.strip()) for part in value.replace(" ", ",").split(",") if part.strip())
  if not out:
    raise argparse.ArgumentTypeError("fixed timesteps must be non-empty")
  return out


def _assert_finite(diagnostics: dict[str, Any]) -> None:
  for key, value in diagnostics.items():
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
      raise AssertionError(f"diagnostic {key!r} has non-finite values")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--g1-diffusion-root", default="../g1-diffusion")
  parser.add_argument("--stage1-ckpt", default="")
  parser.add_argument("--stage2-ckpt", default="")
  parser.add_argument("--input-pkl", default=DEFAULT_HF_BPS_PKL)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--fixed-timesteps", type=_parse_timesteps, default=(1, 3, 5))
  parser.add_argument("--window-size", type=int, default=300)
  parser.add_argument("--ws", type=float, default=0.001)
  parser.add_argument("--normalize", action="store_true")
  args = parser.parse_args()

  device = torch.device(args.device)
  g1_root = _resolve(args.g1_diffusion_root)
  input_pkl = _resolve(args.input_pkl)
  stage1_ckpt = args.stage1_ckpt or _latest_checkpoint(
    g1_root,
    "logs/object_goal_stage1_hf_bps*/checkpoints/object_goal_stage1_epoch_*.pt",
  )
  stage2_ckpt = args.stage2_ckpt or _latest_checkpoint(
    g1_root,
    "logs/object_goal_stage2_hf_bps*/checkpoints/object_goal_stage2_epoch_*.pt",
  )
  if not stage1_ckpt or not stage2_ckpt:
    raise RuntimeError("Provide --stage1-ckpt and --stage2-ckpt corrected checkpoints")

  metadata = load_hf_object_asset_metadata(input_pkl=input_pkl, g1_diffusion_root=g1_root)
  sample = hf_sample_window_tensors(
    metadata,
    start=0,
    window_size=args.window_size,
    device=device,
  )
  root_pos = sample["root_pos"][:, -1].expand(args.num_envs, -1).clone()
  root_quat = sample["root_quat_wxyz"][:, -1].expand(args.num_envs, -1).clone()
  dof_pos = sample["dof_pos"][:, -1].expand(args.num_envs, -1).clone()
  object_centroid = sample["object_centroid"][:, -1].expand(args.num_envs, -1).clone()
  object_quat = sample["object_quat_wxyz"][:, -1].expand(args.num_envs, -1).clone()
  offset = torch.as_tensor(
    metadata.mesh_centroid_offset_local,
    dtype=torch.float32,
    device=device,
  )
  object_body_pos = mesh_centroid_to_body_pos(object_centroid, object_quat, offset)

  env = _FakeEnv(
    num_envs=args.num_envs,
    device=device,
    robot=_FakeEntity(root_pos, root_quat, dof_pos),
    obj=_FakeEntity(object_body_pos, object_quat),
  )
  init_object_goal_prior(
    env,
    g1_diffusion_root=str(g1_root),
    stage1_ckpt_path=stage1_ckpt,
    stage2_ckpt_path=stage2_ckpt,
    fixed_timesteps=args.fixed_timesteps,
    ws=args.ws,
    hf_bps_context_pkl=str(input_pkl),
    context_start=0,
  )
  object_goal_sample_reset(env)
  reward = object_goal_smp_guidance_reward(
    env,
    fixed_timesteps=args.fixed_timesteps,
    ws=args.ws,
    normalize=bool(args.normalize),
  )
  if reward.shape != (args.num_envs,):
    raise AssertionError(f"Expected reward shape {(args.num_envs,)}, got {tuple(reward.shape)}")
  if not torch.isfinite(reward).all():
    raise AssertionError("reward contains non-finite values")
  diagnostics = env._object_goal_smp_diagnostics
  _assert_finite(diagnostics)

  print("Object-goal env reward smoke test passed")
  print(f"  object: {metadata.object_name}")
  print(f"  stage1_ckpt: {stage1_ckpt}")
  print(f"  stage2_ckpt: {stage2_ckpt}")
  print(f"  reward_shape: {tuple(reward.shape)}")
  print(f"  reward_mean: {float(reward.mean().detach().cpu()):.6g}")
  print(f"  sds_error_mean: {float(diagnostics['sds_error'].mean().detach().cpu()):.6g}")
  print(f"  timesteps: {diagnostics['timesteps']}")


if __name__ == "__main__":
  main()
