"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push the current sim kinematics onto the ``MotionFeatureBuffer`` tail.

  Root/EE positions are made env-origin-relative (matching the GSI prime in
  ``_prime_sim_and_buffer``), so the feature window — and thus the SMP reward —
  is invariant to where the env sits in the world grid.
  """
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  buffer.update(
    robot.data.root_link_pos_w - origins,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :],
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over a fixed timestep set ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.

  ``normalize`` divides each timestep's MSE by a ``DiffNormalizer`` running mean
  (value relative to the policy's average) vs. raw MSE (stable absolute scale).
  Always stashes the per-env mean raw MSE on ``env._smp_raw_err``.  Bundle,
  buffer, and normalizer are owned by the env (``init_smp_state``).
  """
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · r_smp`` — generic multiplicative SMP gating.

  ``task_terms`` is a tuple of ``(func, weight, kwargs)`` task-reward components.
  Requires both a positive task reward and staying on the motion manifold;
  neither factor can be farmed alone.  Calls ``smp_guidance_reward`` once (the
  sole SMP-buffer update), so it must be the task's only SMP reward term.
  """
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  return task * smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
