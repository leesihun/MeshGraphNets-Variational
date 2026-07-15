"""Launcher and training loop for parallel_mode=model_split.

Spawns one process per GPU in `gpu_ids`, builds a pipeline stage per rank,
profiles activation memory on rank 0, broadcasts the assignment, and runs a
1F1B pipeline schedule (`pipeline_microbatches` batches per optimizer step,
default 2 * num_stages; set 1 for the legacy sequential loop) with cross-rank
autograd bridging. In-flight activations per stage stay bounded at num_stages
regardless of the micro-batch count.

Supported configs:
  - flat processor, multiscale V-cycle
  - VAE (use_vae=True)
  - world edges (use_world_edges=True)
  - EMA (use_ema=True)
  - torch.compile (use_compile=True)

Notes:
  - Same dataloader on every rank (shuffle=False) so the last rank has graph.y
    for loss computation, and all ranks have graph topology for multiscale pool/unpool.
  - VAE losses (MMD + aux) are added to stage-0's sync_loss only.
    The reconstruction gradient does NOT flow back through z to vae_encoder across
    stage boundaries (z_per_node is detached); vae_encoder trains via MMD/aux only.
"""

from __future__ import annotations

import datetime
import itertools
import os
import socket
import time
import traceback
from collections import deque

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.multiprocessing.spawn import ProcessExitedException
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch_geometric.loader import DataLoader

from model.MeshGraphNets import MeshGraphNets
from parallelism.checkpoint_io import merge_stage_state_dicts_to_rank0
from parallelism.model_split import (
    ModelSplitStage,
    drain_pending_sends,
    set_pipeline_process_groups,
)
from parallelism.partition import partition_stages, partition_summary
from parallelism.profile import profile_activation_memory
from training_profiles.setup import (
    build_dataset_splits,
    build_model_config,
    build_normalization_dict,
    build_optimizer_scheduler,
    cleanup_dataloaders,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_model_split(config: dict, config_filename: str = 'config.txt') -> None:
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if len(gpu_ids) < 2:
        print("[model_split] only one GPU — falling back to single-GPU training.")
        from training_profiles.single_training import single_worker
        single_worker(config, config_filename)
        return

    num_stages = len(gpu_ids)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        config['_ddp_port'] = str(s.getsockname()[1])

    if max(1, int(config.get('pipeline_microbatches', 2 * len(gpu_ids)))) == 1:
        print("[model_split] note: pipeline_microbatches=1 disables 1F1B overlap; "
              "stages run sequentially (legacy behavior).")

    print(f"[model_split] spawning {num_stages} processes on GPUs {gpu_ids} "
          f"(port {config['_ddp_port']})...")
    try:
        mp.spawn(
            _split_worker,
            args=(num_stages, config, gpu_ids, config_filename),
            nprocs=num_stages,
            join=True,
        )
        print("[model_split] training completed.")
    except (KeyboardInterrupt, ProcessExitedException):
        print("\n[model_split] training interrupted.")
    except Exception as e:
        print(f"\n[model_split] training failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _split_worker(rank: int, num_stages: int, config: dict, gpu_ids: list,
                  config_filename: str) -> None:
    try:
        _split_worker_inner(rank, num_stages, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _split_worker_inner(rank: int, num_stages: int, config: dict, gpu_ids: list,
                        config_filename: str) -> None:
    gpu_id = gpu_ids[rank]
    port   = config['_ddp_port']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=num_stages,
        timeout=datetime.timedelta(minutes=60),
    )

    # Separate communicators for downstream activations and upstream gradients:
    # 1F1B interleaves the two directions, which deadlocks on a single
    # communicator (kernel-order serialization on NCCL, blocking sends on gloo).
    pg_data = dist.new_group(list(range(num_stages)))
    pg_grad = dist.new_group(list(range(num_stages)))
    set_pipeline_process_groups(pg_data, pg_grad)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"[model_split rank=0] using device {device}")

    # ---- Dataset (same on every rank) ----
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)

    if rank == 0:
        print("[model_split rank=0] writing normalization stats to HDF5...")
        train_dataset.write_preprocessing_to_hdf5(split_seed)
    dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

    pin_memory  = torch.cuda.is_available()
    num_workers = int(config.get('num_workers', 0))
    mp_context  = 'spawn' if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset, batch_size=int(config['batch_size']), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=int(config.get('prefetch_factor', 2)) if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )

    # ---- Profile + partition ----
    L = int(config['message_passing_num'])
    assignment = _profile_and_partition(rank, config, train_loader, device, L, num_stages)
    if rank == 0:
        print(f"[model_split rank=0] stage assignment: {assignment}")

    # ---- Build stage ----
    stage = ModelSplitStage(config, rank, num_stages, assignment, device)
    if rank == 0:
        total_params = sum(p.numel() for p in stage.parameters())
        print(f"[model_split rank=0] stage {rank} built: "
              f"blocks={stage.my_block_indices}, params={total_params:,}")

    # ---- torch.compile ----
    use_compile = bool(config.get('use_compile', False))
    if use_compile:
        try:
            stage = torch.compile(stage, dynamic=True)
            if rank == 0:
                print("[model_split rank=0] torch.compile applied.")
        except Exception as e:
            if rank == 0:
                print(f"[model_split rank=0] torch.compile failed ({e}); running eager.")

    # ---- EMA ----
    use_ema   = bool(config.get('use_ema', False))
    ema_decay = float(config.get('ema_decay', 0.999))
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(stage, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if rank == 0:
            print(f"[model_split rank=0] EMA enabled (decay={ema_decay}).")

    # ---- Optimizer ----
    total_epochs = int(config['training_epochs'])
    raw_params   = (stage.parameters() if not use_compile
                    else [p for p in stage.parameters()])
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, raw_params, total_epochs)
    if rank == 0:
        print(f"[model_split rank=0] optimizer ready; warmup={warmup_epochs}, cosine_T0={cosine_T0}")

    # ---- Training loop ----
    use_amp   = bool(config.get('use_amp', True))
    amp_dtype = torch.bfloat16
    modelpath = config.get('modelpath')

    microbatches = max(1, int(config.get('pipeline_microbatches', 2 * num_stages)))
    if rank == 0:
        eff_batch = int(config['batch_size']) * microbatches
        print(
            f"[model_split rank=0] 1F1B pipeline: microbatches={microbatches} "
            f"(one optimizer step per {microbatches} batches, effective batch={eff_batch}; "
            f"in-flight activations per stage <= {num_stages})"
        )

    start_time     = time.time()
    last_train_loss = float('inf')
    last_saved_epoch = -1

    for epoch in range(total_epochs):
        train_loss = _train_one_epoch(
            stage=stage,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            ema_model=ema_model,
            microbatches=microbatches,
        )
        scheduler.step()
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[model_split] epoch {epoch}/{total_epochs} "
                  f"train_loss={train_loss:.2e} lr={current_lr:.2e} "
                  f"elapsed={time.time()-start_time:.1f}s")

        last_train_loss = train_loss
        last_saved_epoch = epoch
        _save_checkpoint(
            stage=stage, ema_model=ema_model,
            optimizer=optimizer, scheduler=scheduler,
            assignment=assignment, num_stages=num_stages,
            epoch=epoch, train_loss=train_loss,
            config=config, train_dataset=train_dataset,
            modelpath=modelpath, rank=rank,
        )

    if rank == 0:
        print(f"[model_split] training finished. last model saved at epoch {last_saved_epoch} with train_loss={last_train_loss:.2e}")

    cleanup_dataloaders(train_loader)


# ---------------------------------------------------------------------------
# Profile + partition
# ---------------------------------------------------------------------------

def _profile_and_partition(rank, config, train_loader, device, L, num_stages):
    if rank == 0:
        print("[model_split rank=0] building full model briefly for profiling...")
        assignment = None
        try:
            full_model = MeshGraphNets(config, str(device)).to(device)
            probe = next(iter(train_loader)).to(device)
            estimates = profile_activation_memory(full_model, probe, device)
            costs     = [max(e.peak_bytes, 1) for e in estimates]
            assignment = partition_stages(costs, num_stages)
            print("[model_split rank=0] " + partition_summary(costs, assignment))
            del full_model, probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("[model_split rank=0] WARNING: OOM during profiling. Falling back to equal split.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            costs      = [1] * L
            assignment = partition_stages(costs, num_stages)
        payload = [assignment]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


# ---------------------------------------------------------------------------
# One-batch forward step (dispatches flat vs multiscale)
# ---------------------------------------------------------------------------

def _forward_step(stage: ModelSplitStage, graph, device, config, loss_scale: float = 1.0):
    """Execute one forward pass for a single batch.

    Returns sync_loss (scalar) for .backward(), and (loss, count) for last stage.
    `loss_scale` (1/microbatches) makes the accumulated gradient the mean over
    the micro-batch group; it scales every real loss term (reconstruction on the
    last stage, MMD/aux on stage 0), while logging uses the unscaled loss.
    """
    rank       = stage.stage_idx
    is_first   = stage.is_first
    is_last    = stage.is_last
    use_vae    = stage.use_vae
    use_ms     = stage.use_multiscale
    use_we     = stage.use_world_edges

    lambda_mmd = float(config.get('lambda_mmd', 1.0))
    beta_aux   = float(config.get('beta_aux', 0.0))

    # ---------- First stage ----------
    if is_first:
        stage.apply_input_noise(graph)
        x, ea, ei, wea, wei = stage.encode(graph)

        # VAE: encode z on stage 0
        z_per_node = None
        z_full = None
        vae_sync_loss = torch.zeros((), device=device, dtype=x.dtype)
        if use_vae:
            z_per_node, z_full, vae_losses, aux_loss = stage.encode_vae(graph)
            vae_sync_loss = (lambda_mmd * vae_losses['mmd'].to(dtype=x.dtype)
                             + beta_aux * aux_loss.to(dtype=x.dtype))

        # Run local blocks
        if not use_ms:
            x, ea, ei, wea, wei = stage.run_local_blocks_flat(
                x, ea, ei, wea, wei, z_per_node)
            skip_stack, cur_level = [], 0
        else:
            x, ea, ei, skip_stack, wea, wei, z_per_node, cur_level = (
                stage.run_local_blocks_multiscale(
                    x, ea, ei, [], wea, wei, z_per_node, 0, graph,
                    z_full=z_full))

        sentinel = stage.send_to_next(x, ea, ei, skip_stack, wea, wei,
                                       z_per_node, cur_level, z_full=z_full)
        sync_loss = sentinel.sum() + vae_sync_loss * loss_scale
        return sync_loss, None, None

    # ---------- Last stage ----------
    elif is_last:
        (x, ea, ei, wea, wei, skip_stack,
         z_per_node, z_full, cur_level) = stage.recv_from_prev()

        if not use_ms:
            x, ea, ei, wea, wei = stage.run_local_blocks_flat(
                x, ea, ei, wea, wei, z_per_node)
        else:
            x, ea, ei, skip_stack, wea, wei, z_per_node, cur_level = (
                stage.run_local_blocks_multiscale(
                    x, ea, ei, list(skip_stack), wea, wei, z_per_node, cur_level, graph,
                    z_full=z_full))

        predicted = stage.decode(x, ea, ei)
        target    = graph.y
        errors    = F.huber_loss(predicted, target, reduction='none', delta=1.0)
        loss      = errors.mean()
        return loss * loss_scale, loss.detach(), predicted.numel()

    # ---------- Middle stage ----------
    else:
        (x, ea, ei, wea, wei, skip_stack,
         z_per_node, z_full, cur_level) = stage.recv_from_prev()

        if not use_ms:
            x, ea, ei, wea, wei = stage.run_local_blocks_flat(
                x, ea, ei, wea, wei, z_per_node)
        else:
            x, ea, ei, skip_stack, wea, wei, z_per_node, cur_level = (
                stage.run_local_blocks_multiscale(
                    x, ea, ei, list(skip_stack), wea, wei, z_per_node, cur_level, graph,
                    z_full=z_full))

        sentinel = stage.send_to_next(x, ea, ei, skip_stack, wea, wei,
                                       z_per_node, cur_level, z_full=z_full)
        return sentinel.sum(), None, None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _clip_grads_global(params, max_norm: float, device) -> None:
    """Clip by the global grad norm across all stages (matches the single-GPU path).

    Per-stage clip_grad_norm_ would clip each parameter subset to max_norm
    independently, which is a different (stricter) operation.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if grads:
        total_sq = torch.stack(
            [torch.linalg.vector_norm(g, 2) for g in grads]
        ).square().sum().to(dtype=torch.float32)
    else:
        total_sq = torch.zeros((), device=device, dtype=torch.float32)
    dist.all_reduce(total_sq)
    clip_coef = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    if grads:
        torch._foreach_mul_(grads, clip_coef)


def _train_one_epoch(
    *, stage: ModelSplitStage, loader, optimizer, device, config,
    epoch: int, use_amp: bool, amp_dtype, ema_model, microbatches: int,
) -> float:
    """1F1B pipeline schedule over groups of `microbatches` loader batches.

    Each group is one optimizer step (gradient accumulation). Stage i runs
    (num_stages - 1 - i) warmup forwards, then alternates one-forward/one-backward,
    then drains — so at most (num_stages - i) micro-batch activation sets are
    resident per stage, independent of `microbatches`. All stages issue forwards
    and backwards in the same FIFO micro-batch order, which keeps the P2P
    send/recv sequence matched pairwise between neighbors.
    """
    stage.train()

    num_stages = stage.num_stages
    is_last    = stage.is_last
    use_ms     = stage.use_multiscale
    rank_warmup = max(num_stages - 1 - stage.stage_idx, 0)

    params = [p for p in stage.parameters() if p.requires_grad]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    loss_count = 0
    in_flight = deque()

    def _forward(graph, loss_scale):
        # All stages move graph when multiscale (pool/unpool need topology) or first/last
        if use_ms or stage.is_first or is_last:
            graph = graph.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            return _forward_step(stage, graph, device, config, loss_scale)

    def _backward_oldest():
        nonlocal loss_count
        sync_loss, batch_loss, batch_count = in_flight.popleft()
        sync_loss.backward()
        if is_last and batch_loss is not None:
            loss_sum.add_(batch_loss.double() * batch_count)
            loss_count += batch_count

    data_iter = iter(loader)
    while True:
        group = deque(itertools.islice(data_iter, microbatches))
        if not group:
            break
        loss_scale = 1.0 / len(group)
        optimizer.zero_grad(set_to_none=True)

        for _ in range(min(rank_warmup, len(group))):
            in_flight.append(_forward(group.popleft(), loss_scale))
        while group:
            in_flight.append(_forward(group.popleft(), loss_scale))
            _backward_oldest()
        while in_flight:
            _backward_oldest()

        drain_pending_sends()
        _clip_grads_global(params, 3.0, device)
        optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(stage)

    # Reduce last-rank loss back to rank 0 for logging
    loss_tensor = torch.zeros(2, device=device, dtype=torch.float64)
    if is_last:
        loss_tensor[0] = loss_sum
        loss_tensor[1] = float(loss_count)
    dist.broadcast(loss_tensor, src=num_stages - 1)
    if loss_tensor[1].item() > 0:
        return float(loss_tensor[0].item() / loss_tensor[1].item())
    return 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    *, stage, ema_model, optimizer, scheduler, assignment, num_stages,
    epoch: int, train_loss: float, config, train_dataset, modelpath: str, rank: int,
) -> None:
    # Strip compiled wrapper to get raw state_dict
    raw_stage = getattr(stage, '_orig_mod', stage)
    stage_sd  = raw_stage.state_dict()
    merged    = merge_stage_state_dicts_to_rank0(stage_sd, group=None)

    # EMA state_dict
    ema_merged = None
    if ema_model is not None:
        ema_sd     = ema_model.state_dict()
        ema_merged = merge_stage_state_dicts_to_rank0(ema_sd, group=None)

    if rank != 0:
        return

    save_dict = {
        'epoch':               epoch,
        'model_state_dict':    merged,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss':          train_loss,
        'valid_loss':          train_loss,
        'normalization':       build_normalization_dict(train_dataset),
        'model_config':        build_model_config(config),
        'split_stage_assignment': [list(s) for s in assignment],
        'split_num_stages':    int(num_stages),
    }
    if ema_merged is not None:
        save_dict['ema_state_dict'] = ema_merged

    torch.save(save_dict, modelpath)
    print(f"  -> saved checkpoint at epoch {epoch} ({modelpath})")
