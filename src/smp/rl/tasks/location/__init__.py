"""SMP location task — registers ``Smp-Location-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

from smp.rl.tasks.location.location_env_cfg import g1_location_smp_env_cfg

_location_rl = unitree_g1_tracking_ppo_runner_cfg()
_location_rl.experiment_name = "smp_location_g1"
_location_rl.run_name = "smp_location_g1"
_location_rl.wandb_project = "smp"
_location_rl.actor.distribution_cfg = {
  "class_name": "GaussianDistribution",
  "init_std": 0.30,
  "std_type": "scalar",
  "learn_std": False,
}
_location_rl.algorithm.entropy_coef = 0.0

register_mjlab_task(
  task_id="Smp-Location-G1",
  env_cfg=g1_location_smp_env_cfg(play=False),
  play_env_cfg=g1_location_smp_env_cfg(play=True),
  rl_cfg=_location_rl,
)

__all__ = ["g1_location_smp_env_cfg"]
