"""MDP terms for the object-goal SMP task."""

from smp.rl.tasks.object_goal.mdp.observations import object_goal_observation
from smp.rl.tasks.object_goal.mdp.rewards import (
  object_goal_orientation,
  object_goal_position,
)

__all__ = [
  "object_goal_observation",
  "object_goal_orientation",
  "object_goal_position",
]
