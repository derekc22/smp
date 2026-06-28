"""SMP object-goal task registration."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_g1_smp_ppo_runner_cfg
from smp.rl.tasks.object_goal.object_goal_env_cfg import g1_object_goal_smp_env_cfg

_object_goal_rl = unitree_g1_smp_ppo_runner_cfg()
_object_goal_rl.experiment_name = "smp_object_goal_g1"
_object_goal_rl.run_name = "smp_object_goal_g1"

register_mjlab_task(
  task_id="Smp-ObjectGoal-G1",
  env_cfg=g1_object_goal_smp_env_cfg(play=False),
  play_env_cfg=g1_object_goal_smp_env_cfg(play=True),
  rl_cfg=_object_goal_rl,
)

__all__ = ["g1_object_goal_smp_env_cfg"]
