"""SMP downstream RL tasks.

Importing this package registers all SMP tasks in ``mjlab.tasks.registry``
via side-effect imports of each task sub-package.
"""

from smp.rl.tasks import (
  steering,  # noqa: F401  # registers Smp-Steering-G1, Smp-Forward-G1
)
