"""Pipeline-style model splitting across GPUs.

Activated by `parallel_mode = model_split` in the config. See plan-on-model-parallelism.
"""

from parallelism.partition import partition_stages
from parallelism.profile import BlockEstimate, profile_activation_memory

__all__ = ['partition_stages', 'profile_activation_memory', 'BlockEstimate']
