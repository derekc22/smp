"""Reward components for the getup task: head-height + up-velocity.

Combined and SMP-gated via the generic ``smp.rl.rewards.smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "track_head_height",
  "upward_velocity",
]


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
) -> torch.Tensor:
  """Reward the ``head`` site for reaching ``target_height``.

  ``r = exp(-scale * max(target_height - head_z, 0)^2)`` — saturates at 1 once
  the head is at/above target, decays below (no penalty for overshoot).
  Requires the ``head`` site added by ``getup_env_cfg.get_g1_spec_with_head``.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.25,
  head_height_threshold: float = 0.6,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity until the head clears ``head_height_threshold``.

  ``r = exp(-scale * max(target_velocity - head_vz, 0)^2)`` below the threshold,
  else ``1``.  Uses the head site's world-frame velocity (``site_lin_vel_w``,
  which includes the ω×r term from the torso pitching up), so it drives the head
  rising rather than the pelvis.  Requires the ``head`` site added by
  ``getup_env_cfg.get_g1_spec_with_head``.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(
    head_z < head_height_threshold,
    shaped,
    torch.ones_like(shaped),
  )
