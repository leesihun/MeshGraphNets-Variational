"""State_dict merge/slice for pipeline-split MeshGraphNets.

The saved checkpoint always stores a FULL merged state_dict keyed exactly as if
the model had been built on a single device. This keeps inference/rollout
unchanged: they call `model.load_state_dict(checkpoint['model_state_dict'])`
on a full single-GPU model with no awareness that training was split.

A `split_stage_assignment` and `split_num_stages` are also stored so a future
multi-GPU resume can re-shard the merged state_dict.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.distributed as dist


# Parameter-key prefixes that belong to specific components, used during
# slicing. Keys not under any of these prefixes are treated as "shared" and
# placed on stage 0 (they include the VAE encoder, aux_decoder, and any
# non-block top-level modules).
_ENCODER_PREFIXES = ('model.encoder.',)
_DECODER_PREFIXES = ('model.decoder.',)


def _processor_key_prefix(block_index: int, is_multiscale: bool, multiscale_meta: dict = None) -> List[str]:
    """Return the list of state_dict-key prefixes for a single processor block index.

    For flat mode, block i lives under `model.processer_list.{i}.`.
    Multiscale prefixes are derived from the meta dict {'L', 'mp_per_level'}.
    """
    if not is_multiscale:
        return [f"model.processer_list.{block_index}."]

    if multiscale_meta is None:
        raise ValueError("multiscale_meta required when is_multiscale=True")
    L = int(multiscale_meta['L'])
    mp_per_level = [int(x) for x in multiscale_meta['mp_per_level']]

    # Layout: pre[0] (mp[0]) | pre[1] (mp[1]) | ... | coarsest (mp[L]) | post[L-1] (mp[2L-(L-1)]) | ... | post[0] (mp[2L])
    # i.e. pre is forward order over levels, post is reverse order over levels.
    counts = []
    names = []
    for i in range(L):
        counts.append(mp_per_level[i])
        names.append(('pre', i))
    counts.append(mp_per_level[L])
    names.append(('coarsest', None))
    for i in range(L - 1, -1, -1):
        counts.append(mp_per_level[2 * L - i])
        names.append(('post', i))

    cumulative = 0
    for (kind, level), c in zip(names, counts):
        if block_index < cumulative + c:
            local_idx = block_index - cumulative
            if kind == 'pre':
                # z_fusers (when use_vae) follow the same per-level layout.
                return [
                    f"model.pre_blocks.{level}.{local_idx}.",
                    f"model.ms_z_fusers_pre.{level}.{local_idx}.",
                ]
            elif kind == 'post':
                return [
                    f"model.post_blocks.{level}.{local_idx}.",
                    f"model.ms_z_fusers_post.{level}.{local_idx}.",
                ]
            else:  # coarsest
                return [
                    f"model.coarsest_blocks.{local_idx}.",
                    f"model.ms_z_fusers_coarsest.{local_idx}.",
                ]
        cumulative += c

    raise IndexError(f"block_index {block_index} out of range for multiscale layout")


def slice_state_dict_for_stage(
    full_sd: Dict[str, torch.Tensor],
    stage_idx: int,
    num_stages: int,
    assignment: Sequence[Sequence[int]],
    is_multiscale: bool,
    multiscale_meta: dict = None,
) -> Dict[str, torch.Tensor]:
    """Extract the subset of keys from `full_sd` that belong to stage `stage_idx`.

    Stage 0 owns: encoder + vae_encoder + aux_decoder + its assigned blocks.
    Last stage owns: decoder + its assigned blocks.
    All stages own: z_fusers / ms_z_fusers for their assigned blocks.
    Multiscale pool/unpool modules (coarse_eb_encoders, skip_projs, unpool_blocks)
    are co-located with the stage that owns the corresponding boundary block.
    """
    stage_blocks = list(assignment[stage_idx])

    allowed_prefixes: List[str] = []

    # Processor block prefixes (flat and multiscale)
    for b in stage_blocks:
        allowed_prefixes.extend(_processor_key_prefix(b, is_multiscale, multiscale_meta))

    # VAE fusers — every stage owns fusers for its assigned blocks
    for b in stage_blocks:
        if not is_multiscale:
            allowed_prefixes.append(f'model.z_fusers.{b}.')
        else:
            if multiscale_meta is None:
                allowed_prefixes.append(f'model.z_fusers.{b}.')
            else:
                L = int(multiscale_meta['L'])
                mp = [int(x) for x in multiscale_meta['mp_per_level']]
                prefixes = _processor_key_prefix(b, True, multiscale_meta)
                for px in prefixes:
                    # Derive the fuser key from the block key by replacing the module name
                    # e.g. model.pre_blocks.0.2.  → model.ms_z_fusers_pre.0.2.
                    if 'pre_blocks' in px:
                        allowed_prefixes.append(px.replace('model.pre_blocks.', 'model.ms_z_fusers_pre.'))
                    elif 'coarsest_blocks' in px:
                        allowed_prefixes.append(px.replace('model.coarsest_blocks.', 'model.ms_z_fusers_coarsest.'))
                    elif 'post_blocks' in px:
                        allowed_prefixes.append(px.replace('model.post_blocks.', 'model.ms_z_fusers_post.'))

    # Multiscale pool/unpool modules — owned by stage that has the corresponding boundary block
    if is_multiscale and multiscale_meta is not None:
        L = int(multiscale_meta['L'])
        mp = [int(x) for x in multiscale_meta['mp_per_level']]
        # coarse_eb_encoders[level] — owned by stage that has the last pre[level] block
        # skip_projs[level] / unpool_blocks[level] — owned by stage that has the first post[level] block
        cumulative_pre = 0
        for level in range(L):
            last_pre_b = cumulative_pre + mp[level] - 1
            if last_pre_b in stage_blocks:
                allowed_prefixes.append(f'model.coarse_eb_encoders.{level}.')
            cumulative_pre += mp[level]
        # Post blocks in reverse order: post[L-1], post[L-2], ..., post[0]
        cumulative_post = sum(mp[:L + 1])  # skip pre + coarsest
        for rev_i, level in enumerate(range(L - 1, -1, -1)):
            first_post_b = cumulative_post
            if first_post_b in stage_blocks:
                allowed_prefixes.append(f'model.skip_projs.{level}.')
                allowed_prefixes.append(f'model.unpool_blocks.{level}.')
            cumulative_post += mp[2 * L - level]

    is_first = (stage_idx == 0)
    is_last  = (stage_idx == num_stages - 1)
    if is_first:
        allowed_prefixes.extend(_ENCODER_PREFIXES)
        allowed_prefixes.extend(['model.vae_encoder.', 'model.aux_decoder.'])
    if is_last:
        allowed_prefixes.extend(_DECODER_PREFIXES)

    result: Dict[str, torch.Tensor] = {}
    for key, tensor in full_sd.items():
        for prefix in allowed_prefixes:
            if key.startswith(prefix):
                result[key] = tensor
                break
    return result


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
