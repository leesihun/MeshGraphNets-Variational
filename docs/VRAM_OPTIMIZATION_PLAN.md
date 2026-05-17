# VRAM And Throughput Controls

This document describes the memory and speed controls currently implemented in
the training runtime.

## Current Controls

| Config key | Runtime effect |
| --- | --- |
| `use_checkpointing` | Recomputes processor activations during backward to reduce activation memory. |
| `use_amp` | Enables bfloat16 autocast in the training and validation loops. |
| `grad_accum_steps` | Trades optimizer-step frequency for lower physical batch size. |
| `num_workers` | Controls DataLoader worker count. |
| `use_ema` | Maintains an EMA copy for evaluation/inference; costs extra model memory. |
| `use_compile` | Applies `torch.compile(dynamic=True)` to the simulator. |
| `parallel_mode model_split` | Experimental model split across GPUs instead of DDP replication. |

## Activation Checkpointing

Activation checkpointing is implemented in `model/checkpointing.py` and wired
through `model/MeshGraphNets.py`.

Flat processor:

- `process_with_checkpointing` checkpoints each `GnBlock`.
- Optional VAE `z` fusion is applied before the checkpointed block.
- World-edge tensors are preserved across the wrapper.

Multiscale processor:

- the same helper is used for pre, coarsest, and post block lists
- per-arm VAE fusers are supported
- world-edge tensors are only present on the fine-level blocks

Use:

```text
use_checkpointing True
```

Expected tradeoff: lower activation memory, higher compute time because forward
work is recomputed during backward.

## AMP

The live training code uses bfloat16 autocast:

```python
torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp)
```

There is no `GradScaler` in the current path. This is deliberate: bfloat16 has a
larger exponent range than fp16 and is less prone to overflow in graph scatter
operations on large meshes.

Use:

```text
use_amp True
```

If CUDA is unavailable, autocast is effectively not useful. The code still uses
the configured flag when entering the autocast context.

## Gradient Accumulation

`training_profiles/training_loop.py` supports:

| Value | Meaning |
| --- | --- |
| `grad_accum_steps 1` | optimizer step every batch |
| `grad_accum_steps N` | optimizer step every N batches |
| `grad_accum_steps 0` | one optimizer step for the whole epoch |

The loss is divided by the actual accumulation-window size so accumulated
gradients approximate the mean over that window.

Gradient accumulation does not reduce per-sample memory. It lets you lower
physical `batch_size` while keeping a larger effective batch:

```text
effective_batch = batch_size * grad_accum_steps
```

For `grad_accum_steps 0`, the effective batch is the full epoch.

## EMA Memory Cost

`use_ema True` creates an `AveragedModel` copy of the simulator. This improves
evaluation stability but stores an additional set of model parameters.

Use EMA when evaluation quality matters and the extra parameter memory is
acceptable. Disable it first when trying to fit a model that is near the memory
limit.

## DataLoader Pressure

The loader can be CPU and host-memory heavy because it computes positional
features, edge attributes, optional world edges, and optional multiscale
hierarchies.

Important knobs:

- `num_workers`
- `use_parallel_stats`
- `coarse_cache_per_worker`
- `prefetch_factor` is hardcoded to `1` in current training DataLoaders
- `pin_memory` is enabled when CUDA is available

If workers crash or host RAM grows too much, reduce `num_workers` and
`coarse_cache_per_worker` before changing the model.

## DDP Versus Model Split

Multiple `gpu_ids` normally launch DDP. DDP replicates the full model on every
GPU, so it improves throughput but does not solve a single-sample model-fit
problem.

`parallel_mode model_split` routes to `parallelism/launcher.py` and splits model
stages across GPUs. This is the path intended for models that do not fit on one
GPU. It is experimental and has explicit caveats in `parallelism/launcher.py`,
including detached reconstruction gradients across some stage boundaries.

## Recommended Fit Order

When a run does not fit:

1. Lower physical `batch_size`.
2. Enable `use_checkpointing True`.
3. Keep `use_amp True`.
4. Reduce or disable EMA.
5. Reduce DataLoader pressure if host memory or worker stability is the problem.
6. Use `grad_accum_steps` to recover effective batch size.
7. For true single-sample model-fit failures, evaluate `parallel_mode model_split`
   on the target multi-GPU machine.

## Common Failure Signals

| Symptom | Likely area |
| --- | --- |
| CUDA OOM during forward/backward | model activations, batch size, EMA, or DDP replication |
| CUDA OOM only during validation/test visualization | rendering/test batch count, EMA copy, stored output tensors |
| DataLoader worker exits | host RAM, HDF5 access, coarse cache, worker count |
| GPU idle between batches | preprocessing or DataLoader throughput |
| NaN/Inf under AMP | inspect input normalization and loss scale; current AMP is bfloat16 |
