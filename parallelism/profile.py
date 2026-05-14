"""Activation-memory profiler for MeshGraphNets processor blocks.

Runs a single forward pass on a probe sample, recording peak CUDA memory
delta around each top-level processor block. Output feeds the partitioner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class BlockEstimate:
    name: str            # e.g. "processer_list.3"
    peak_bytes: int      # peak activation memory delta during this block's forward
    params_bytes: int    # parameter bytes resident in this block (for sanity logging)


def _params_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters(recurse=True):
        total += p.numel() * p.element_size()
    return total


def _enumerate_processor_blocks(model: nn.Module):
    """Return a list of (name, module) for the contiguous processor blocks.

    Order is the same order the forward pass executes them. For multiscale
    we treat each block within a level-segment as its own unit but emit
    them in V-cycle traversal order, so the partitioner can cut on any
    block boundary (multiscale stage routing in model_split.py only cuts
    between segments, but the profiler stays neutral).
    """
    # Walk into the inner EncoderProcessorDecoder
    inner = getattr(model, 'model', model)

    use_multiscale = getattr(inner, 'use_multiscale', False)
    if not use_multiscale:
        pl = getattr(inner, 'processer_list', None)
        if pl is None:
            raise RuntimeError("profile_activation_memory: model has no 'processer_list' and is not multiscale")
        return [(f"processer_list.{i}", m) for i, m in enumerate(pl)]

    # Multiscale: pre[0..L-1] -> coarsest -> post[L-1..0]
    L = inner.multiscale_levels
    blocks = []
    for i in range(L):
        for j, m in enumerate(inner.pre_blocks[i]):
            blocks.append((f"pre_blocks.{i}.{j}", m))
    for j, m in enumerate(inner.coarsest_blocks):
        blocks.append((f"coarsest_blocks.{j}", m))
    for i in range(L - 1, -1, -1):
        for j, m in enumerate(inner.post_blocks[i]):
            blocks.append((f"post_blocks.{i}.{j}", m))
    return blocks


def profile_activation_memory(
    model: nn.Module,
    probe_batch,
    device: torch.device,
) -> List[BlockEstimate]:
    """Run one forward pass and record peak memory delta per processor block.

    Args:
        model: an instantiated MeshGraphNets (already moved to `device`).
        probe_batch: a single PyG `Data` / `Batch` already on `device`. Use
            the first batch from the train loader.
        device: CUDA device. Profiling is meaningful only on CUDA.

    Returns:
        BlockEstimate list, in forward execution order.
    """
    if device.type != 'cuda':
        # CPU profiling: emit zeros so the partitioner falls back to equal splits.
        blocks = _enumerate_processor_blocks(model)
        return [
            BlockEstimate(name=name, peak_bytes=0, params_bytes=_params_bytes(m))
            for name, m in blocks
        ]

    blocks = _enumerate_processor_blocks(model)
    estimates: List[BlockEstimate] = []

    # Per-block recorded peaks via forward hooks. We rely on the order in which
    # forwards fire and on torch.cuda.max_memory_allocated being reset between
    # blocks. CUDA stream is single (sync between blocks via .item() barriers).
    peak_record = {name: 0 for name, _ in blocks}

    def _make_pre(name):
        def pre(_module, _inputs):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        return pre

    def _make_post(name):
        def post(_module, _inputs, _output):
            torch.cuda.synchronize(device)
            peak_record[name] = int(torch.cuda.max_memory_allocated(device))
        return post

    handles = []
    for name, m in blocks:
        handles.append(m.register_forward_pre_hook(_make_pre(name)))
        handles.append(m.register_forward_hook(_make_post(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            try:
                _ = model(probe_batch)
            except TypeError:
                # Top-level MeshGraphNets.forward signature is (graph, debug, add_noise, use_posterior, fixed_z)
                _ = model(probe_batch, add_noise=False, use_posterior=False)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    for name, m in blocks:
        estimates.append(BlockEstimate(
            name=name,
            peak_bytes=peak_record.get(name, 0),
            params_bytes=_params_bytes(m),
        ))
    return estimates
