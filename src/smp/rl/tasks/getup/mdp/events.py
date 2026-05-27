"""Reset events for the getup task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["reset_stand_counter"]


@torch.no_grad()
def reset_stand_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Zero the ``stood_up`` standing-hold counter for the reset envs.

  Kept separate from the shared ``gsi_reset`` (which only primes state) so that
  can be reused unchanged.  No-op until ``stood_up`` lazily creates the counter.
  """
  if not hasattr(env, "_getup_stand_count"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._getup_stand_count[env_ids] = 0  # type: ignore[attr-defined]
