"""Instantiate and step the registered object-goal mjlab task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task", nargs="?", default="Smp-ObjectGoal-G1")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--stage1-ckpt", default="")
  parser.add_argument("--stage2-ckpt", default="")
  parser.add_argument("--fixed-timesteps", default="")
  args = parser.parse_args()

  try:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, list_tasks
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "mjlab is not importable. Run this under the SMP uv environment."
    ) from exc

  import smp.rl.tasks  # noqa: F401

  if args.task not in list_tasks():
    raise RuntimeError(f"Task {args.task!r} is not registered; tasks={list_tasks()}")
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = int(args.num_envs)
  if args.stage1_ckpt:
    cfg.events["init_object_goal_prior"].params["stage1_ckpt_path"] = args.stage1_ckpt
  if args.stage2_ckpt:
    cfg.events["init_object_goal_prior"].params["stage2_ckpt_path"] = args.stage2_ckpt
  if args.fixed_timesteps:
    timesteps = tuple(int(p.strip()) for p in args.fixed_timesteps.split(",") if p.strip())
    cfg.events["init_object_goal_prior"].params["fixed_timesteps"] = timesteps
    cfg.rewards["object_goal_task_smp_product"].params["fixed_timesteps"] = timesteps

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset()
    actions = torch.zeros(
      env.num_envs,
      env.action_manager.total_action_dim,
      device=env.device,
    )
    rew = torch.zeros(env.num_envs, device=env.device)
    for _ in range(int(args.steps)):
      obs, rew, terminated, truncated, info = env.step(actions)
      del obs, terminated, truncated, info
    if not torch.isfinite(rew).all():
      raise AssertionError("step reward contains non-finite values")
    diagnostics = getattr(env, "_object_goal_smp_diagnostics", None)
    if diagnostics is None:
      raise AssertionError("env did not produce object-goal SMP diagnostics")
    for key, value in diagnostics.items():
      if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise AssertionError(f"diagnostic {key!r} contains non-finite values")
    print("Object-goal mjlab task step check passed")
    print(f"  task: {args.task}")
    print(f"  num_envs: {env.num_envs}")
    print(f"  steps: {args.steps}")
    print(f"  reward_shape: {tuple(rew.shape)}")
    print(f"  reward_mean: {float(rew.mean().detach().cpu()):.6g}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
