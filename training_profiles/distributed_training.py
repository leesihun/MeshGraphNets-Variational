import os
import signal
import threading
import time
import datetime
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from training_profiles.setup import (
    analyze_debug_files,
    build_dataset_splits,
    build_model_and_ema,
    build_optimizer_scheduler,
    init_log_file,
    log_model_summary,
    save_checkpoint,
)
from training_profiles.training_loop import (
    evaluate_vae_posterior_epoch,
    evaluate_vae_prior_epoch,
    log_training_config,
    test_model,
    train_epoch,
    validate_epoch,
)

# Per-process shutdown flag, set by signal handler
_stop_event = threading.Event()

_FORCED_EXIT_DELAY_SECONDS = 10


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by setting the stop flag and scheduling a forced exit."""
    _stop_event.set()
    # Start a daemon thread that force-kills the process after a grace period.
    # This covers the case where the main thread is stuck inside a blocking C++
    # NCCL call and cannot check _stop_event.
    def _force_exit():
        time.sleep(_FORCED_EXIT_DELAY_SECONDS)
        os._exit(1)
    t = threading.Thread(target=_force_exit, daemon=True)
    t.start()

def train_worker(rank, world_size, config, gpu_ids, config_filename='config.txt'):
    """Training worker for distributed training.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Configuration dictionary
        gpu_ids: List of GPU IDs to use
        config_filename: Path to the config file (default: config.txt)
    """
    try:
        _train_worker_inner(rank, world_size, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_worker_inner(rank, world_size, config, gpu_ids, config_filename):
    """Actual training logic, called inside the error-handling wrapper."""
    # Register signal handlers so Ctrl+C (SIGINT on main → SIGTERM on workers)
    # sets the stop flag instead of killing the process mid-collective.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Enable NCCL flight recorder for debugging collective mismatches
    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '1000')

    # Get the physical GPU ID for this rank
    gpu_id = gpu_ids[rank]
    port = config['_ddp_port']
    setup_distributed(rank, world_size, gpu_id, port)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        if rank == 0:
            print(f'[Rank {rank}] Using physical GPU {gpu_id}, device: {device}')
            print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        if rank == 0:
            print(f'Using device: {device}')

    # ---- Dataset ----
    if rank == 0:
        print("\nLoading dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        print("Writing train-derived normalization stats to HDF5...")
        train_dataset.write_preprocessing_to_hdf5(split_seed)
        if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
            print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")
    dist.barrier()

    # Create distributed samplers
    if rank == 0:
        print("\nCreating dataloaders (distributed train, rank-0 eval)...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create dataloaders
    num_workers = config['num_workers']
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        val_loader = None

    # Test loader only needed on rank 0 (no DDP forward, uses unwrapped model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    if rank == 0:
        train_eval_subset_size = min(len(train_dataset), int(config.get('train_eval_subset_size', 128)))
        train_eval_rng = np.random.default_rng(split_seed)
    else:
        train_eval_subset_size = None
        train_eval_rng = None
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    if rank == 0:
        print("\nInitializing model...")
    model, ema_model = build_model_and_ema(config, device)

    # Wrap with DistributedDataParallel
    if torch.cuda.is_available():
        ddp_model = DDP(
            model,
            device_ids=[gpu_id],
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    else:
        ddp_model = DDP(
            model,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

    if torch.cuda.is_available() and rank == 0:
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_model_summary(ddp_model, config, ema_model)

    best_valid_loss = float('inf')
    best_epoch = -1

    # ---- Optimizer / Scheduler ----
    if rank == 0:
        print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, ddp_model.parameters(), total_epochs
    )
    use_fused = torch.cuda.is_available()
    if rank == 0:
        print(f"Optimizer: Adam (fused={use_fused})")
        print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
              f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available() and rank == 0:
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_training_config(config)
        print("\n" + "=" * 60)
        print("Starting training loop...")
        print("=" * 60 + "\n")

    start_time = time.time()

    log_file, log_dir = None, None
    if rank == 0:
        log_file, log_dir = init_log_file(config, config_filename)

    # Synchronize all processes before starting training
    dist.barrier(device_ids=[gpu_id])

    modelname = config.get('modelpath')
    use_vae = config.get('use_vae', False)
    vae_valid_prior_samples = int(config.get('vae_valid_prior_samples', 8)) if use_vae else 1
    if vae_valid_prior_samples < 1:
        raise ValueError("vae_valid_prior_samples must be >= 1")

    interrupted = False
    for epoch in range(config.get('training_epochs')):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model)

        # Synchronize stop decision across all ranks — if ANY rank wants to stop,
        # ALL ranks must stop together to avoid NCCL collective mismatches.
        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after train_epoch).")
            break

        train_totals = torch.tensor(
            [train_metrics['sum'], float(train_metrics['count'])],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = (train_totals[0] / train_totals[1]).item()

        if use_vae and 'mmd_mean' in train_metrics:
            mmd_tensor = torch.tensor([train_metrics['mmd_mean']], device=device, dtype=torch.float64)
            dist.all_reduce(mmd_tensor, op=dist.ReduceOp.SUM)
            train_metrics['mmd_mean'] = (mmd_tensor[0] / world_size).item()
        if use_vae and 'total_mean' in train_metrics:
            total_tensor = torch.tensor(
                [train_metrics['total_mean'] * train_metrics['count']],
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            train_metrics['total_mean'] = (total_tensor[0] / train_totals[1]).item()

        if rank == 0:
            eval_model = ema_model.module if ema_model is not None else model

            # Resample train_eval subset each epoch for an unbiased training loss estimate
            train_eval_indices = train_eval_rng.choice(
                len(train_dataset), size=train_eval_subset_size, replace=False
            ).tolist()
            train_eval_loader = DataLoader(
                Subset(train_dataset, train_eval_indices),
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            if use_vae:
                train_eval_metrics = evaluate_vae_posterior_epoch(
                    eval_model, train_eval_loader, device, config, epoch,
                    progress_name='TrainEvalQ'
                )
                valid_metrics = evaluate_vae_posterior_epoch(
                    eval_model, val_loader, device, config, epoch,
                    progress_name='ValidQ'
                )
                valid_prior_metrics = evaluate_vae_prior_epoch(
                    eval_model, val_loader, device, config, epoch,
                    num_prior_samples=vae_valid_prior_samples,
                    progress_name=f'ValidPrior@{vae_valid_prior_samples}'
                )
            else:
                train_eval_metrics = validate_epoch(model, train_eval_loader, device, config, epoch)
                valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
                valid_prior_metrics = None
            train_eval_loss = train_eval_metrics['mean']
            valid_loss = valid_metrics['mean']
        else:
            train_eval_loss = 0.0
            valid_loss = 0.0
        train_eval_loss_tensor = torch.tensor([train_eval_loss], device=device)
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.broadcast(train_eval_loss_tensor, src=0)
        dist.broadcast(valid_loss_tensor, src=0)
        train_eval_loss = train_eval_loss_tensor.item()
        valid_loss = valid_loss_tensor.item()

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after validate_epoch).")
            break

        # Step scheduler on all ranks (valid_loss is identical after all_reduce)
        scheduler.step()

        # Per epoch, node-weighted optimization and evaluation losses.
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            if use_vae:
                train_eval_mmd = train_eval_metrics.get('mmd_mean', 0.0)
                train_eval_total = train_eval_metrics.get('total_mean', train_eval_loss)
                valid_prior_loss = valid_prior_metrics['mean']
                prior_gap = valid_prior_loss - valid_loss
                print(
                    f"Epoch {epoch}/{config['training_epochs']} LR: {current_lr:.2e} | "
                    f"TrainOpt  recon={train_loss:.2e} mmd={train_metrics.get('mmd_mean', 0.0):.2e} total={train_metrics.get('total_mean', train_loss):.2e} | "
                    f"TrainEvalQ recon={train_eval_loss:.2e} mmd={train_eval_mmd:.2e} total={train_eval_total:.2e} | "
                    f"ValidQ    recon={valid_loss:.2e} mmd={valid_metrics.get('mmd_mean', 0.0):.2e} total={valid_metrics.get('total_mean', valid_loss):.2e} | "
                    f"ValidPrior@{vae_valid_prior_samples} recon={valid_prior_loss:.2e} gap={prior_gap:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch}/{config['training_epochs']} "
                    f"TrainOpt: {train_loss:.2e} "
                    f"TrainEval: {train_eval_loss:.2e} "
                    f"Valid: {valid_loss:.2e} "
                    f"LR: {current_lr:.2e}"
                )

        # Only rank 0 saves checkpoints
        if valid_loss < best_valid_loss and rank == 0:
            best_valid_loss = valid_loss
            best_epoch = epoch
            save_checkpoint(
                epoch, ddp_model.module, ema_model, optimizer, scheduler,
                train_loss, valid_loss, config, train_dataset, modelname,
                use_vae=use_vae,
                valid_prior_loss=valid_prior_loss if use_vae else None,
                vae_valid_prior_samples=vae_valid_prior_samples if use_vae else None,
            )
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

        if log_file and rank == 0: 
            with open(log_file, 'a') as f:
                if use_vae:
                    f.write(
                        f"Elapsed: {time.time() - start_time:.2f}s "
                        f"Epoch {epoch} LR: {current_lr:.4e} | "
                        f"TrainOpt recon={train_loss:.4e} mmd={train_metrics.get('mmd_mean', 0.0):.4e} total={train_metrics.get('total_mean', train_loss):.4e} | "
                        f"TrainEvalQ recon={train_eval_loss:.4e} mmd={train_eval_metrics.get('mmd_mean', 0.0):.4e} total={train_eval_metrics.get('total_mean', train_eval_loss):.4e} | "
                        f"ValidQ recon={valid_loss:.4e} mmd={valid_metrics.get('mmd_mean', 0.0):.4e} total={valid_metrics.get('total_mean', valid_loss):.4e} | "
                        f"ValidPrior@{vae_valid_prior_samples} recon={valid_prior_loss:.4e} gap={prior_gap:.4e}\n"
                    )
                else:
                    f.write(
                        f"Elapsed: {time.time() - start_time:.2f}s "
                        f"Epoch {epoch} TrainOpt {train_loss:.4e} "
                        f"TrainEval {train_eval_loss:.4e} "
                        f"Valid {valid_loss:.4e} LR: {current_lr:.4e}\n"
                    )

        # Periodically test the model on the test set
        # Use unwrapped model to avoid DDP deadlock (only rank 0 runs this)
        # Barrier ensures all ranks wait so rank 1+ don't race into next epoch's
        # DDP forward/backward while rank 0 is still running test_model
        test_interval = int(config.get('test_interval', 10))
        last_epoch = epoch == config.get('training_epochs') - 1
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                _test_start = time.time()
                test_loss = test_model(eval_model, test_loader, device, config, epoch, train_dataset)
                print(f"  Test completed in {time.time() - _test_start:.1f}s")

                # Optionally visualize training set reconstruction (same batch indices)
                if config.get('display_trainset', True):
                    train_viz_indices = config.get('test_batch_idx', [0, 1, 2, 3, 4, 5, 6, 7])
                    train_viz_indices = [i for i in train_viz_indices if i < len(train_dataset)]
                    if train_viz_indices:
                        train_viz_loader = DataLoader(
                            Subset(train_dataset, train_viz_indices),
                            batch_size=1, shuffle=False, pin_memory=True
                        )
                        viz_config = dict(config)
                        viz_config['test_batch_idx'] = list(range(len(train_viz_indices)))
                        train_viz_loss = test_model(eval_model, train_viz_loader, device, viz_config, epoch, train_dataset, output_prefix='train')
                        print(f"  Train reconstruction loss: {train_viz_loss:.2e}")
            dist.barrier(device_ids=[gpu_id])

    if rank == 0:
        if interrupted:
            print(f"\nTraining interrupted. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")
        else:
            print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Post-hoc GMM fitting on VAE latent codes (rank 0 only)
    if rank == 0 and config.get('use_vae', False) and config.get('fit_latent_gmm', False):
        from model.latent_gmm import run_posthoc_gmm_fitting
        gmm_model = ema_model.module if ema_model is not None else model
        run_posthoc_gmm_fitting(gmm_model, train_dataset, config, device, modelname)

    # Analyze debug files if they exist
    if rank == 0:
        analyze_debug_files(log_dir)

def setup_distributed(rank, world_size, gpu_id, port):
    """Initialize distributed training process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        gpu_id: Physical GPU ID to use for this rank
        port: TCP port for the rendezvous store
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=60)
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
