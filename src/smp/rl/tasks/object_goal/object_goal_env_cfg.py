"""G1 object-goal task backed by real HF-BPS object metadata."""

from __future__ import annotations

from pathlib import Path

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.object_goal_assets import load_hf_object_asset_metadata
from smp.rl.object_goal_events import init_object_goal_prior, object_goal_sample_reset
from smp.rl.object_goal_prior import DEFAULT_SDS_TIMESTEPS
from smp.rl.object_goal_rewards import object_goal_task_smp_product
from smp.rl.tasks.object_goal.mdp import (
  object_goal_observation,
  object_goal_orientation,
  object_goal_position,
)


SMP_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_G1_DIFFUSION_ROOT = (SMP_ROOT.parent / "g1-diffusion").resolve()
DEFAULT_HF_BPS_SAMPLE = (
  DEFAULT_G1_DIFFUSION_ROOT
  / "data"
  / "hf_bps_preprocessed"
  / "omomo_sub3_largebox_003_sample1.pkl"
)


def _latest_checkpoint(pattern: str) -> str:
  matches = sorted(
    DEFAULT_G1_DIFFUSION_ROOT.glob(pattern),
    key=lambda path: path.stat().st_mtime,
  )
  return str(matches[-1]) if matches else ""


def _default_stage1_checkpoint() -> str:
  return _latest_checkpoint(
    "logs/object_goal_stage1_hf_bps*/checkpoints/object_goal_stage1_epoch_*.pt"
  )


def _default_stage2_checkpoint() -> str:
  return _latest_checkpoint(
    "logs/object_goal_stage2_hf_bps*/checkpoints/object_goal_stage2_epoch_*.pt"
  )


def _default_timesteps(stage2_ckpt: str) -> tuple[int, ...]:
  return (1, 3, 5) if "smoke" in stage2_ckpt else DEFAULT_SDS_TIMESTEPS


def g1_object_goal_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 object-goal env cfg with two-stage object-goal SDS reward."""
  cfg = g1_smp_env_cfg(play=play)
  for key in ("init_smp_state", "gsi_reset", "gsi_refresh", "push_robot"):
    cfg.events.pop(key, None)

  metadata = load_hf_object_asset_metadata(
    input_pkl=DEFAULT_HF_BPS_SAMPLE,
    g1_diffusion_root=DEFAULT_G1_DIFFUSION_ROOT,
  )
  init_pos, init_rot = metadata.initial_body_pose(frame=0)
  cfg.scene.entities["object"] = EntityCfg(
    spec_fn=metadata.make_mujoco_spec,
    init_state=EntityCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
  )
  cfg.scene.extent = max(float(cfg.scene.extent), 3.0)

  stage1_ckpt = _default_stage1_checkpoint()
  stage2_ckpt = _default_stage2_checkpoint()
  fixed_timesteps = _default_timesteps(stage2_ckpt)

  # --- Events --------------------------------------------------------------
  cfg.events["init_object_goal_prior"] = EventTermCfg(
    func=init_object_goal_prior,
    mode="startup",
    params={
      "g1_diffusion_root": str(DEFAULT_G1_DIFFUSION_ROOT),
      "stage1_ckpt_path": stage1_ckpt,
      "stage2_ckpt_path": stage2_ckpt,
      "fixed_timesteps": fixed_timesteps,
      "ws": 6.0,
      "hf_bps_context_pkl": str(DEFAULT_HF_BPS_SAMPLE),
      "context_start": 0,
    },
  )
  cfg.events["object_goal_sample_reset"] = EventTermCfg(
    func=object_goal_sample_reset,
    mode="reset",
    params={},
  )

  # --- Observations --------------------------------------------------------
  object_obs = ObservationTermCfg(func=object_goal_observation)
  cfg.observations["actor"].terms["object_goal"] = object_obs
  cfg.observations["critic"].terms["object_goal"] = object_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["object_goal_task_smp_product"] = RewardTermCfg(
    func=object_goal_task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (object_goal_position, 0.8, {"pos_err_scale": 8.0}),
        (object_goal_orientation, 0.2, {"ori_err_scale": 4.0}),
      ),
      "fixed_timesteps": fixed_timesteps,
      "ws": 6.0,
    },
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  if play:
    cfg.events["init_object_goal_prior"].params["fixed_timesteps"] = fixed_timesteps
    cfg.rewards["object_goal_task_smp_product"].params["fixed_timesteps"] = fixed_timesteps

  return cfg
