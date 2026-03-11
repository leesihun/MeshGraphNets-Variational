import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch, test_model

# Disable HDF5 file locking for persistent SWMR handles with multiple DataLoader workers
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def single_worker(config, config_filename='config.txt'):
    # Single GPU/CPU training
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")

    # Set device using the first (and only) GPU from gpu_ids
    if torch.cuda.is_available():
        gpu_id = gpu_ids
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using physical GPU {gpu_id}, device: {device}')
        print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # Generate dataloader from dataset
    print("\nLoading dataset...")
    dataset = load_data(config)
    if torch.cuda.is_available():
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Pass num_node_types to config for model to compute input dimension
    if config.get('use_node_types', False) and dataset.num_node_types is not None:
        config['num_node_types'] = dataset.num_node_types
        print(f"  Node types enabled: {dataset.num_node_types} types will be added to input")

    # Divide the dataset into training, validation, and test sets
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

    # Create dataloaders (no distributed samplers for single process)
    print("\nCreating dataloaders...")
    num_workers = config['num_workers']
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True
    )
    if torch.cuda.is_available():
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Generate MeshGraphNets model
    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Optional torch.compile for kernel fusion (10-30% speedup, requires PyTorch 2.0+)
    if config.get('use_compile', False):
        print("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)

    print('\n'*2)
    print("Model initialized successfully")
    if config.get('use_checkpointing', False):
        print("Gradient checkpointing: ENABLED")
    if config.get('use_amp', False):
        print("Mixed precision (AMP): ENABLED (bfloat16)")
    if config.get('use_compile', False):
        print("torch.compile: ENABLED (dynamic=True)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_valid_loss = float('inf')
    best_epoch = -1

    # Initialize optimizer
    print("\nInitializing optimizer...")
    learning_rate = config.get('learningr')
    weight_decay = float(config.get('weight_decay', 1e-4))
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=use_fused)
    print(f"Optimizer: AdamW (weight_decay={weight_decay}, fused={use_fused})")

    # Initialize learning rate scheduler (OneCycleLR: warmup + cosine decay, stepped per batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.get('training_epochs'),
        pct_start=0.1,
    )
    print(f"Learning rate scheduler: OneCycleLR (max_lr={learning_rate:.2e}, warmup=10%)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    # Log per-feature loss weights if configured
    loss_weights_cfg = config.get('feature_loss_weights', None)
    if loss_weights_cfg is not None:
        if not isinstance(loss_weights_cfg, list):
            loss_weights_cfg = [loss_weights_cfg]
        import torch as _torch
        w = _torch.tensor(loss_weights_cfg, dtype=_torch.float32)
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

    log_file_dir = config.get('log_file_dir')
    log_dir = None
    if log_file_dir:
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

    modelname = config.get('modelpath')

    try:
        for epoch in range(config.get('training_epochs')):

            train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, scheduler=scheduler)
            valid_loss = validate_epoch(model, val_loader, device, config, epoch)

            # Per epoch, batch-averaged train, validation losses.
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{config['training_epochs']} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e} LR: {current_lr:.2e}")

            if valid_loss < best_valid_loss:
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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'normalization': normalization,
                    'model_config': model_config,
                }, checkpoint_path)
                print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

            if log_file_dir:
                with open(log_file, 'a') as f:
                    f.write(f"Elapsed time: {time.time() - start_time:.2f}s Epoch {epoch} Train Loss: {train_loss:.4e} Valid Loss: {valid_loss:.4e} LR: {current_lr:.4e}\n")

            # Periodically test the model on the test set and save results with ground truth
            test_interval = int(config.get('test_interval', 10))
            last_epoch = epoch == config.get('training_epochs') - 1
            if epoch % test_interval == 0 or last_epoch:
                test_loss = test_model(model, test_loader, device, config, epoch, dataset)

        print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Analyze debug files if they exist
    if log_dir:
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