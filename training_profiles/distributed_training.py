import os
import signal
import threading
import time
import datetime
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch, test_model

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

    # Pass num_node_types to config for model to compute input dimension
    if config.get('use_node_types', False) and dataset.num_node_types is not None:
        config['num_node_types'] = dataset.num_node_types
        if rank == 0:
            print(f"  Node types enabled: {dataset.num_node_types} types will be added to input")

    # Divide the dataset into training, validation, and test sets
    if rank == 0:
        print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

    # Create distributed samplers
    if rank == 0:
        print("\nCreating dataloaders with distributed samplers...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Test loader only needed on rank 0 (no DDP forward, uses unwrapped model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Generate MeshGraphNets model
    if rank == 0:
        print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)

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
        if config.get('use_amp', False):
            print("Mixed precision (AMP): ENABLED (bfloat16)")
        if config.get('use_compile', False):
            print("torch.compile: ENABLED (dynamic=True)")
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
    weight_decay = float(config.get('weight_decay', 1e-4))
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=use_fused)
    if rank == 0:
        print(f"Optimizer: AdamW (weight_decay={weight_decay}, fused={use_fused})")

    # Initialize learning rate scheduler (OneCycleLR: warmup + cosine decay, stepped per batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.get('training_epochs'),
        pct_start=0.1,
    )
    if rank == 0:
        print(f"Learning rate scheduler: OneCycleLR (max_lr={learning_rate:.2e}, warmup=10%)")

    if torch.cuda.is_available() and rank == 0:
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        # Log per-feature loss weights if configured
        loss_weights_cfg = config.get('feature_loss_weights', None)
        if loss_weights_cfg is not None:
            if not isinstance(loss_weights_cfg, list):
                loss_weights_cfg = [loss_weights_cfg]
            w = torch.tensor(loss_weights_cfg, dtype=torch.float32)
            w_normalized = (w * len(w) / w.sum()).tolist()
            w_proportional = (w / w.sum()).tolist()
            print(f"Per-feature loss weights (raw):         {loss_weights_cfg}")
            print(f"Per-feature loss weights (normalized):  {[f'{v:.3f}' for v in w_normalized]}")
            print(f"Per-feature loss weights (proportional):{[f'{v:.4f}' for v in w_proportional]}")
        else:
            print("Per-feature loss weights: equal (default)")

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

        train_loss = train_epoch(ddp_model, train_loader, optimizer, device, config, epoch, scheduler=scheduler)

        # Synchronize stop decision across all ranks — if ANY rank wants to stop,
        # ALL ranks must stop together to avoid NCCL collective mismatches.
        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after train_epoch).")
            break

        valid_loss = validate_epoch(ddp_model, val_loader, device, config, epoch)

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after validate_epoch).")
            break

        # All-reduce losses across ranks for accurate logging and checkpoint decisions
        train_loss_tensor = torch.tensor([train_loss], device=device)
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(valid_loss_tensor, op=dist.ReduceOp.AVG)
        train_loss = train_loss_tensor.item()
        valid_loss = valid_loss_tensor.item()

        # Per epoch, batch-averaged train, validation losses.
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"Epoch {epoch}/{config['training_epochs']} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e} LR: {current_lr:.2e}")

        # Only rank 0 saves checkpoints
        if valid_loss < best_valid_loss and rank == 0:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = modelname
            normalization = {
                'node_mean': dataset.node_mean,
                'node_std': dataset.node_std,
                'edge_mean': dataset.edge_mean,
                'edge_std': dataset.edge_std,
                'delta_mean': dataset.delta_mean,
                'delta_std': dataset.delta_std,
            }
            if dataset.use_node_types and dataset.node_type_to_idx is not None:
                normalization['node_type_to_idx'] = dataset.node_type_to_idx
                normalization['num_node_types'] = dataset.num_node_types
            if dataset.use_world_edges and dataset.world_edge_radius is not None:
                normalization['world_edge_radius'] = dataset.world_edge_radius
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
                'use_vae': config.get('use_vae', False),
                'vae_latent_dim': config.get('vae_latent_dim', 32),
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),  # Save unwrapped model
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'normalization': normalization,
                'model_config': model_config,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

        if log_file_dir and rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"Elapsed time: {time.time() - start_time:.2f}s Epoch {epoch} Train Loss: {train_loss:.4e} Valid Loss: {valid_loss:.4e} LR: {current_lr:.4e}\n")

        # Periodically test the model on the test set
        # Use unwrapped model to avoid DDP deadlock (only rank 0 runs this)
        # Barrier ensures all ranks wait so rank 1+ don't race into next epoch's
        # DDP forward/backward while rank 0 is still running test_model
        test_interval = int(config.get('test_interval', 10))
        last_epoch = epoch == config.get('training_epochs') - 1
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                test_loss = test_model(model, test_loader, device, config, epoch, dataset)
            dist.barrier(device_ids=[gpu_id])

    if rank == 0:
        if interrupted:
            print(f"\nTraining interrupted. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")
        else:
            print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Analyze debug files if they exist
    if rank == 0 and log_dir:
        import glob
        import numpy as np
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
        timeout=datetime.timedelta(minutes=5)
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
