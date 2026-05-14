"""Launcher and training loop for parallel_mode=model_split.

Spawns one process per GPU in `gpu_ids`, builds a pipeline stage per rank,
profiles activation memory on rank 0, broadcasts the assignment, and runs
a synchronous (no-microbatching) training loop with cross-rank autograd
bridging via `SendActivation`/`RecvActivation`.

MVP scope:
  - flat processor only (no multiscale, no VAE, no world edges)
  - microbatches=1 (memory savings only; no throughput uplift)
  - no validation / test inside training loop (only periodic checkpoint saves)
  - same dataloader on every rank (shuffle=False) so the last rank has the
    correct local `graph.y` for the loss.
"""

from __future__ import annotations

import datetime
import os
import socket
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.multiprocessing.spawn import ProcessExitedException
from torch_geometric.loader import DataLoader

from model.MeshGraphNets import MeshGraphNets
from parallelism.checkpoint_io import merge_stage_state_dicts_to_rank0
from parallelism.model_split import ModelSplitStage, recv_from_prev, send_to_next
from parallelism.partition import partition_stages, partition_summary
from parallelism.profile import profile_activation_memory
from training_profiles.setup import (
    build_dataset_splits,
    build_model_config,
    build_normalization_dict,
    build_optimizer_scheduler,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def launch_model_split(config: dict, config_filename: str = 'config.txt') -> None:
    """Top-level entry; spawns one process per GPU in `gpu_ids` and runs training."""
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if len(gpu_ids) < 2:
        print("[model_split] only one GPU configured — falling back to single-GPU training.")
        from training_profiles.single_training import single_worker
        single_worker(config, config_filename)
        return

    # Warn about flags that aren't honored under model_split MVP
    if config.get('use_ema', False):
        print("[model_split] WARNING: use_ema=True is not yet honored under parallel_mode=model_split; ignoring.")
    if config.get('use_compile', False):
        print("[model_split] WARNING: use_compile=True is not yet honored under parallel_mode=model_split; ignoring.")
    if int(config.get('batch_size', 1)) < len(gpu_ids):
        print(f"[model_split] note: batch_size={config.get('batch_size')} < num_stages={len(gpu_ids)}. "
              f"Pipeline runs sequentially (memory savings only, no throughput uplift). "
              f"Set batch_size >= num_stages for pipeline throughput once microbatching lands.")

    num_stages = len(gpu_ids)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        config['_ddp_port'] = str(s.getsockname()[1])

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
        print("\n[model_split] training interrupted; all workers terminated.")
    except Exception as e:
        print(f"\n[model_split] training failed: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _split_worker(rank: int, num_stages: int, config: dict, gpu_ids: list, config_filename: str) -> None:
    try:
        _split_worker_inner(rank, num_stages, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _split_worker_inner(rank: int, num_stages: int, config: dict, gpu_ids: list, config_filename: str) -> None:
    gpu_id = gpu_ids[rank]
    port = config['_ddp_port']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=num_stages,
        timeout=datetime.timedelta(minutes=60),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    if rank == 0:
        print(f"[model_split rank={rank}] using device {device}")

    # ---- Dataset (same on every rank — see module docstring) ----
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)

    if rank == 0:
        print("[model_split rank=0] writing normalization stats to HDF5...")
        train_dataset.write_preprocessing_to_hdf5(split_seed)
    dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory
    num_workers = int(config.get('num_workers', 0))
    mp_context = 'spawn' if num_workers > 0 else None

    # Same data, same order on every rank (shuffle=False for now; see TODO).
    train_loader = DataLoader(
        train_dataset, batch_size=int(config['batch_size']), shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=1 if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )

    # ---- Profile + partition (rank 0 only, then broadcast) ----
    L = int(config['message_passing_num'])
    assignment = _profile_and_partition(rank, config, train_loader, device, L, num_stages)
    if rank == 0:
        print(f"[model_split rank=0] stage assignment: {assignment}")

    # ---- Build this stage ----
    stage = ModelSplitStage(config, rank, num_stages, assignment, device)
    if rank == 0:
        print(f"[model_split rank=0] built stage {rank} with blocks {stage.my_block_indices}")
        total_params = sum(p.numel() for p in stage.parameters())
        print(f"  stage param count: {total_params:,}")

    # ---- Optimizer ----
    total_epochs = int(config['training_epochs'])
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, stage.parameters(), total_epochs,
    )
    if rank == 0:
        print(f"[model_split rank=0] optimizer ready; warmup={warmup_epochs}, cosine_T0={cosine_T0}")

    # ---- Training loop ----
    use_amp = bool(config.get('use_amp', True))
    amp_dtype = torch.bfloat16
    modelpath = config.get('modelpath')

    start_time = time.time()
    best_train_loss = float('inf')
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
        )
        scheduler.step()
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[model_split] epoch {epoch}/{total_epochs} "
                  f"train_loss={train_loss:.2e} lr={current_lr:.2e} "
                  f"elapsed={time.time()-start_time:.1f}s")

        # Checkpoint on improvement (rank 0 collates from all ranks)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            _save_checkpoint(
                stage=stage, optimizer=optimizer, scheduler=scheduler,
                assignment=assignment, num_stages=num_stages,
                epoch=epoch, train_loss=train_loss,
                config=config, train_dataset=train_dataset,
                modelpath=modelpath, rank=rank,
            )

    if rank == 0:
        print(f"[model_split] training finished. best_train_loss={best_train_loss:.2e}")


# ---------------------------------------------------------------------------
# Profile + partition
# ---------------------------------------------------------------------------

def _profile_and_partition(rank: int, config: dict, train_loader, device, L: int, num_stages: int):
    """Rank 0 profiles the full model and broadcasts the stage assignment.

    If the full model OOMs at probe time, fall back to an equal-block split.
    """
    if rank == 0:
        print("[model_split rank=0] building full model briefly for profiling...")
        assignment = None
        try:
            full_model = MeshGraphNets(config, str(device)).to(device)
            probe = next(iter(train_loader)).to(device)
            estimates = profile_activation_memory(full_model, probe, device)
            costs = [max(e.peak_bytes, 1) for e in estimates]
            assignment = partition_stages(costs, num_stages)
            print("[model_split rank=0] " + partition_summary(costs, assignment))
            del full_model, probe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print("[model_split rank=0] WARNING: full model did not fit for profiling. "
                  "Falling back to equal-block split.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            costs = [1] * L
            assignment = partition_stages(costs, num_stages)
        payload = [assignment]
    else:
        payload = [None]

    dist.broadcast_object_list(payload, src=0)
    return payload[0]


# ---------------------------------------------------------------------------
# Training step (one batch)
# ---------------------------------------------------------------------------

def _train_one_epoch(
    *, stage: ModelSplitStage, loader, optimizer, device, config,
    epoch: int, use_amp: bool, amp_dtype,
) -> float:
    stage.train()
    total_loss_sum = 0.0
    total_loss_count = 0

    is_first = stage.is_first
    is_last = stage.is_last
    rank = stage.stage_idx
    num_stages = stage.num_stages

    for batch_idx, graph in enumerate(loader):
        if is_first or is_last:
            graph = graph.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            if is_first:
                stage.apply_input_noise(graph)
                encoded = stage.encode(graph)
                x, edge_attr, edge_index = encoded.x, encoded.edge_attr, encoded.edge_index
                x, edge_attr, edge_index = stage.run_local_blocks(x, edge_attr, edge_index)
                sync_loss = send_to_next(x, edge_attr, edge_index, dst=rank + 1).sum()
            elif is_last:
                x, edge_attr, edge_index = recv_from_prev(src=rank - 1, device=device)
                x, edge_attr, edge_index = stage.run_local_blocks(x, edge_attr, edge_index)
                predicted = stage.decode(x, edge_attr, edge_index)
                target = graph.y
                errors = F.huber_loss(predicted, target, reduction='none', delta=1.0)
                loss = errors.mean()
                sync_loss = loss
            else:
                x, edge_attr, edge_index = recv_from_prev(src=rank - 1, device=device)
                x, edge_attr, edge_index = stage.run_local_blocks(x, edge_attr, edge_index)
                sync_loss = send_to_next(x, edge_attr, edge_index, dst=rank + 1).sum()

        sync_loss.backward()
        torch.nn.utils.clip_grad_norm_(stage.parameters(), max_norm=3.0)
        optimizer.step()

        if is_last:
            batch_count = predicted.numel()
            total_loss_sum += float(loss.item()) * batch_count
            total_loss_count += batch_count

    # Reduce last-rank's loss back to rank 0 for logging
    loss_tensor = torch.tensor([total_loss_sum, float(total_loss_count)], device=device, dtype=torch.float64)
    dist.broadcast(loss_tensor, src=num_stages - 1)
    if loss_tensor[1].item() > 0:
        return float(loss_tensor[0].item() / loss_tensor[1].item())
    return 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save_checkpoint(
    *, stage, optimizer, scheduler, assignment, num_stages,
    epoch: int, train_loss: float, config, train_dataset, modelpath: str, rank: int,
) -> None:
    """Gather all stage state_dicts to rank 0; merge and save."""
    stage_sd = stage.state_dict()
    # Strip the leading "model." that ModelSplitStage adds — the saved checkpoint
    # uses the same key namespace as the full MeshGraphNets model.
    merged = merge_stage_state_dicts_to_rank0(stage_sd, group=None)
    if rank != 0:
        return

    save_dict = {
        'epoch': epoch,
        'model_state_dict': merged,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'valid_loss': train_loss,  # placeholder; MVP has no validation loop yet
        'normalization': build_normalization_dict(train_dataset),
        'model_config': build_model_config(config),
        'split_stage_assignment': [list(s) for s in assignment],
        'split_num_stages': int(num_stages),
    }
    torch.save(save_dict, modelpath)
    print(f"  -> saved checkpoint at epoch {epoch} ({modelpath})")
