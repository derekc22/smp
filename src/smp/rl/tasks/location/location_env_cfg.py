"""G1 location task with SMP guidance.

Each env gets a world-frame xy goal sampled at a random angle and distance
from the character's current xy position, periodically resampled.
Reward = position-tracking (always on) + optional velocity / face-direction
shaping, on top of the SMP guidance reward inherited from
``g1_smp_env_cfg``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.tasks.location import mdp


def g1_location_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 location env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["location"] = mdp.LocationCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    tar_dist_min=1.0,
    tar_dist_max=10.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "location"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["location_position"] = RewardTermCfg(
    func=mdp.location_position,
    weight=1.0,
    params={"command_name": "location", "pos_err_scale": 0.5},
  )
  cfg.rewards["smp_guidance"].params["ws"] = 6.0

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
