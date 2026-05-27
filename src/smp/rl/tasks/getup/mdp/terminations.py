"""Termination terms for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["smp_too_low", "stood_up"]


def stood_up(
  env: ManagerBasedRlEnv,
  head_height: float = 1.2,
  max_speed: float = 0.5,
  hold_steps: int = 10,
) -> torch.Tensor:
  """Truncate the episode once the robot is STABLY standing (success).

  Standing = head at/above ``head_height`` AND base speed below ``max_speed``,
  sustained for ``hold_steps`` consecutive steps (a per-env counter, zeroed on
  reset by ``reset_stand_counter``) so a transient bob doesn't count.  Wire with
  ``time_out=True`` so it is a TRUNCATION: the value bootstraps from the
  standing state instead of being zeroed, otherwise ending on success would
  make standing look worthless and the policy would avoid it.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)
  is_standing = (z >= head_height) & (speed < max_speed)
  cnt = getattr(env, "_getup_stand_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cnt = torch.where(is_standing, cnt + 1, torch.zeros_like(cnt))
  env._getup_stand_count = cnt  # type: ignore[attr-defined]
  return cnt >= hold_steps


def smp_too_low(
  env: ManagerBasedRlEnv,
  threshold: float = 0.02,
  ws: float = 6.0,
  grace_steps: int = 15,
) -> torch.Tensor:
  """Terminate when the SMP guidance score has collapsed (off-manifold).

  Scores ``exp(-ws · env._smp_raw_err)`` — the RAW (un-normalized) MSE stashed by
  ``smp_guidance_reward``, a stable absolute realism scale, so a fixed threshold
  is meaningful.  Removes the "violent get-up" shortcut: leaving the motion
  manifold drives the score toward 0 and ends the episode.  ``ws`` must match the
  reward's; ``grace_steps`` skips the first steps after reset.
  """
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  raw_smp = torch.exp(-ws * raw_err)
  past_grace = env.episode_length_buf >= grace_steps
  return (raw_smp < threshold) & past_grace
