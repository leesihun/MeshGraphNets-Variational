"""State_dict merge for pipeline-split MeshGraphNets.

The saved checkpoint always stores a FULL merged state_dict keyed exactly as if
the model had been built on a single device. This keeps inference/rollout
unchanged: they call `model.load_state_dict(checkpoint['model_state_dict'])`
on a full single-GPU model with no awareness that training was split.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist


def merge_stage_state_dicts_to_rank0(
    stage_sd: Dict[str, torch.Tensor],
    group=None,
) -> Dict[str, torch.Tensor]:
    """Gather per-stage state_dicts to rank 0 and merge into a single dict.

    Each rank passes its stage_module.state_dict() (already broken into the
    keys assigned to that stage). The result is meaningful only on rank 0;
    other ranks receive an empty dict.
    """
    if not dist.is_available() or not dist.is_initialized():
        return dict(stage_sd)

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    # Move tensors to CPU for gather to avoid blowing up rank 0 GPU memory with K-1 extra stages.
    cpu_sd = {k: v.detach().cpu() for k, v in stage_sd.items()}

    gathered: List[Dict[str, torch.Tensor]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, cpu_sd, group=group)

    if rank != 0:
        return {}

    merged: Dict[str, torch.Tensor] = {}
    for sd in gathered:
        for k, v in sd.items():
            if k in merged:
                # Should not happen if slicing was clean; warn loudly.
                print(f"  [merge_state_dict] WARNING: duplicate key '{k}' across stages — last writer wins")
            merged[k] = v
    return merged
