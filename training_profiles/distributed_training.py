import glob
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

from general_modules.data_loader import load_data
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch, test_model, log_training_config, build_ema_model

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

    # Disable HDF5 file locking — multiple ranks + workers open the same file read-only
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

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

    # Generate dataloader from dataset
    if rank == 0:
        print("\nLoading dataset...")
    dataset = load_data(config)
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Divide the dataset into training, validation, and test sets
    if rank == 0:
        print("\nSplitting dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1, seed=split_seed)
    if rank == 0:
        print("Writing train-derived normalization stats to HDF5...")
        train_dataset.write_preprocessing_to_hdf5(split_seed)
    dist.barrier()

    # Pass dataset metadata to config for model construction
    config['num_timesteps'] = train_dataset.num_timesteps
    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        config['num_node_types'] = train_dataset.num_node_types
        if rank == 0:
            print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")

    # Compute noise target correction ratio (node_std / delta_std) from train split stats
    if train_dataset.node_std is not None and train_dataset.delta_std is not None:
        output_var = config['output_var']
        config['noise_std_ratio'] = (
            train_dataset.node_std[:output_var] / np.maximum(train_dataset.delta_std, 1e-8)
        ).tolist()

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
        prefetch_factor=8 if num_workers > 0 else None,
    )

    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=8 if num_workers > 0 else None,
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

    # Generate MeshGraphNets model
    if rank == 0:
        print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)

    # EMA shadow model (created before torch.compile/DDP so it holds the raw Module)
    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    # Optional torch.compile for kernel fusion (10-30% speedup, requires PyTorch 2.0+)
    if config.get('use_compile', False):
        if rank == 0:
            print("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)

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
        print('\n'*2)
        print("Model initialized successfully")
        if config.get('use_checkpointing', False):
            print("Gradient checkpointing: ENABLED")
        if config.get('use_amp', True):
            print("Mixed precision (AMP): ENABLED (bfloat16)")
        if config.get('use_compile', False):
            print("torch.compile: ENABLED (dynamic=True)")
        if ema_model is not None:
            print(f"EMA: ENABLED (decay={config.get('ema_decay', 0.999)})")
        if config.get('use_vae', False):
            print(f"VAE: ENABLED (z_dim={config.get('vae_latent_dim', 32)}, beta_kl={config.get('beta_kl', 0.001)})")
        else:
            print("VAE: disabled")
        total_params = sum(p.numel() for p in ddp_model.parameters())
        trainable_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    best_valid_loss = float('inf')
    best_epoch = -1

    # Initialize optimizer
    if rank == 0:
        print("\nInitializing optimizer...")
    learning_rate = config.get('learningr')
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate, fused=use_fused)
    if rank == 0:
        print(f"Optimizer: Adam (fused={use_fused})")

    # Initialize learning rate scheduler: linear warmup + cosine warm restarts
    total_epochs = config.get('training_epochs')
    warmup_epochs = 3
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    # T_0 sized so 2 full cosine cycles fit in remaining epochs (T_0 + 2*T_0 = 3*T_0)
    cosine_T0 = max(remaining_epochs // 3, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_T0, T_mult=2, eta_min=1e-8
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    if rank == 0:
        print(f"Learning rate scheduler: LinearLR warmup ({warmup_epochs} epochs, start=0.01x) -> "
              f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available() and rank == 0:
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_training_config(config)

        print("\n" + "="*60)
        print("Starting training loop...")
        print("="*60 + "\n")

    start_time = time.time()

    log_file = None
    log_dir = None
    log_file_dir = config.get('log_file_dir')
    if log_file_dir and rank == 0:
        log_file = 'outputs/' + log_file_dir
        log_dir = os.path.dirname(log_file)
        # if log_file doesn't exist, create it
        if not os.path.exists(log_file):
            os.makedirs(log_dir, exist_ok=True)

        # Pass log directory to config for debug output
        config['log_dir'] = log_dir
        with open(log_file, 'w') as f:
            f.write(f"Training epoch log file\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log file absolute path: {os.path.abspath(log_file)}\n")
            # Write the whole config file here:
            with open(config_filename, 'r') as fc:
                f.write(fc.read())
            fc.close()

    # Synchronize all processes before starting training
    dist.barrier(device_ids=[gpu_id])

    modelname = config.get('modelpath')

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

        use_vae = config.get('use_vae', False)
        if use_vae and 'kl_mean' in train_metrics:
            kl_tensor = torch.tensor([train_metrics['kl_mean']], device=device, dtype=torch.float64)
            dist.all_reduce(kl_tensor, op=dist.ReduceOp.SUM)
            train_metrics['kl_mean'] = (kl_tensor[0] / world_size).item()
        if use_vae and 'total_mean' in train_metrics:
            total_tensor = torch.tensor(
                [train_metrics['total_mean'] * train_metrics['count']],
                device=device, dtype=torch.float64,
            )
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            train_metrics['total_mean'] = (total_tensor[0] / train_totals[1]).item()

        if rank == 0:
            # traineval uses the real model (sanity-check against trainopt);
            # valid/test use EMA for smoother generalization estimates.
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
            train_eval_metrics = validate_epoch(model, train_eval_loader, device, config, epoch)
            valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
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

        # Per epoch, node-weighted optimization, clean-train, and validation losses.
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            if use_vae:
                print(
                    f"Epoch {epoch}/{config['training_epochs']} LR: {current_lr:.2e} | "
                    f"TrainOpt  recon={train_loss:.2e} kl={train_metrics.get('kl_mean', 0.0):.2e} total={train_metrics.get('total_mean', train_loss):.2e} | "
                    f"TrainEval recon={train_eval_loss:.2e} kl={train_eval_metrics.get('kl_mean', 0.0):.2e} total={train_eval_metrics.get('total_mean', train_eval_loss):.2e} | "
                    f"Valid     recon={valid_loss:.2e} kl={valid_metrics.get('kl_mean', 0.0):.2e} total={valid_metrics.get('total_mean', valid_loss):.2e}"
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
            checkpoint_path = modelname
            normalization = {
                'node_mean': train_dataset.node_mean,
                'node_std': train_dataset.node_std,
                'edge_mean': train_dataset.edge_mean,
                'edge_std': train_dataset.edge_std,
                'delta_mean': train_dataset.delta_mean,
                'delta_std': train_dataset.delta_std,
            }
            if train_dataset.use_node_types and train_dataset.node_type_to_idx is not None:
                normalization['node_type_to_idx'] = train_dataset.node_type_to_idx
                normalization['num_node_types'] = train_dataset.num_node_types
            if train_dataset.use_world_edges and train_dataset.world_edge_radius is not None:
                normalization['world_edge_radius'] = train_dataset.world_edge_radius
            if train_dataset.use_multiscale and len(train_dataset.coarse_edge_means) > 0:
                normalization['coarse_edge_means'] = train_dataset.coarse_edge_means
                normalization['coarse_edge_stds']  = train_dataset.coarse_edge_stds
            model_config = {
                'input_var': config.get('input_var'),
                'output_var': config.get('output_var'),
                'edge_var': config.get('edge_var'),
                'latent_dim': config.get('latent_dim'),
                'message_passing_num': config.get('message_passing_num'),
                'use_node_types': config.get('use_node_types', False),
                'num_node_types': config.get('num_node_types', 0),
                'use_world_edges': config.get('use_world_edges', False),
                'use_checkpointing': config.get('use_checkpointing', False),
                'use_multiscale': config.get('use_multiscale', False),
                'multiscale_levels': config.get('multiscale_levels', 1),
                'mp_per_level': config.get('mp_per_level', None),
                'fine_mp_pre': config.get('fine_mp_pre', 5),
                'coarse_mp_num': config.get('coarse_mp_num', 5),
                'fine_mp_post': config.get('fine_mp_post', 5),
                'coarsening_type': config.get('coarsening_type', 'bfs'),
                'voronoi_clusters': config.get('voronoi_clusters', None),
                'use_vae': config.get('use_vae', False),
                'vae_latent_dim': config.get('vae_latent_dim', 32),
            }
            save_dict = {
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),  # Save unwrapped model
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'normalization': normalization,
                'model_config': model_config,
            }
            if ema_model is not None:
                save_dict['ema_state_dict'] = ema_model.state_dict()
            torch.save(save_dict, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

        if log_file_dir and rank == 0:
            with open(log_file, 'a') as f:
                if use_vae:
                    f.write(
                        f"Elapsed: {time.time() - start_time:.2f}s "
                        f"Epoch {epoch} LR: {current_lr:.4e} | "
                        f"TrainOpt recon={train_loss:.4e} kl={train_metrics.get('kl_mean', 0.0):.4e} total={train_metrics.get('total_mean', train_loss):.4e} | "
                        f"TrainEval recon={train_eval_loss:.4e} kl={train_eval_metrics.get('kl_mean', 0.0):.4e} total={train_eval_metrics.get('total_mean', train_eval_loss):.4e} | "
                        f"Valid recon={valid_loss:.4e} kl={valid_metrics.get('kl_mean', 0.0):.4e} total={valid_metrics.get('total_mean', valid_loss):.4e}\n"
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

    # Analyze debug files if they exist
    if rank == 0 and log_dir:
        debug_files = sorted(glob.glob(os.path.join(log_dir, 'debug_*.npz')))

        if debug_files:
            print("\n" + "="*60)
            print("DEBUG OUTPUT ANALYSIS (first 5 epochs)")
            print("="*60)
            for f in debug_files[:5]:
                try:
                    data = np.load(f)
                    fname = os.path.basename(f)
                    print(f"\n{fname}")
                    print(f"  Input (x):")
                    print(f"    mean={data['x_mean']}")
                    print(f"    std={data['x_std']}")
                    print(f"  Target (y):")
                    print(f"    mean={data['y_mean']}")
                    print(f"    std={data['y_std']}")
                    print(f"  Prediction (pred):")
                    print(f"    mean={data['pred_mean']}")
                    print(f"    std={data['pred_std']}")
                    pred_target_ratio = data['pred_std'] / (data['y_std'] + 1e-8)
                    print(f"  Pred/Target std ratio: {pred_target_ratio}")
                    if np.any(pred_target_ratio < 0.1):
                        print(f"    ^ WARNING: Pred much smaller than target!")
                except Exception as e:
                    print(f"  Error reading {f}: {e}")

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
