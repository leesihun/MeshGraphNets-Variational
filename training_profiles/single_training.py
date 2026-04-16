import glob
import os
import time

import numpy as np
import torch

from general_modules.data_loader import load_data
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import (
    build_ema_model,
    evaluate_vae_posterior_epoch,
    evaluate_vae_prior_epoch,
    log_training_config,
    test_model,
    train_epoch,
    validate_epoch,
)

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

    # Divide the dataset into training, validation, and test sets
    print("\nSplitting dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1, seed=split_seed)
    print("Writing train-derived normalization stats to HDF5...")
    train_dataset.write_preprocessing_to_hdf5(split_seed)

    # Pass dataset metadata to config for model construction
    config['num_timesteps'] = train_dataset.num_timesteps
    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        config['num_node_types'] = train_dataset.num_node_types
        print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")

    # Compute noise target correction ratio (node_std / delta_std) from train split stats
    if train_dataset.node_std is not None and train_dataset.delta_std is not None:
        output_var = config['output_var']
        config['noise_std_ratio'] = (
            train_dataset.node_std[:output_var] / np.maximum(train_dataset.delta_std, 1e-8)
        ).tolist()

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
        prefetch_factor=8 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
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

    # EMA shadow model (created before torch.compile so it holds the raw Module)
    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    # Optional torch.compile for kernel fusion (10-30% speedup, requires PyTorch 2.0+)
    if config.get('use_compile', False):
        print("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)

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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    best_valid_loss = float('inf')
    best_epoch = -1

    # Initialize optimizer
    print("\nInitializing optimizer...")
    learning_rate = config.get('learningr')
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, fused=use_fused)
    print(f"Optimizer: Adam (fused={use_fused})")

    # Initialize learning rate scheduler: linear warmup + cosine warm restarts
    total_epochs = config.get('training_epochs')
    warmup_epochs = 3
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    # T_0 sized so 2 full cosine cycles fit in remaining epochs (T_0 + 2*T_0 = 3*T_0)
    cosine_T0 = max(remaining_epochs // 2, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_T0, T_mult=2, eta_min=1e-8
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    print(f"Learning rate scheduler: LinearLR warmup ({warmup_epochs} epochs, start=0.01x) -> "
          f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    log_training_config(config)

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
    use_vae = config.get('use_vae', False)
    vae_valid_prior_samples = int(config.get('vae_valid_prior_samples', 8)) if use_vae else 1
    if vae_valid_prior_samples < 1:
        raise ValueError("vae_valid_prior_samples must be >= 1")

    try:
        for epoch in range(config.get('training_epochs')):

            train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch, ema_model=ema_model)

            eval_model = ema_model.module if ema_model is not None else model
            if use_vae:
                train_eval_metrics = evaluate_vae_posterior_epoch(
                    eval_model, train_loader, device, config, epoch,
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
                train_eval_metrics = validate_epoch(model, train_loader, device, config, epoch)
                valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
                valid_prior_metrics = None
            train_loss = train_metrics['mean']
            train_eval_loss = train_eval_metrics['mean']
            valid_loss = valid_metrics['mean']
            scheduler.step()

            # Per epoch, node-weighted optimization and evaluation losses.
            current_lr = optimizer.param_groups[0]['lr']
            if use_vae:
                train_kl    = train_metrics.get('kl_mean', 0.0)
                train_aux   = train_metrics.get('aux_mean', 0.0)
                train_total = train_metrics.get('total_mean', train_loss)
                train_eval_kl    = train_eval_metrics.get('kl_mean', 0.0)
                train_eval_total = train_eval_metrics.get('total_mean', train_eval_loss)
                valid_kl    = valid_metrics.get('kl_mean', 0.0)
                valid_total = valid_metrics.get('total_mean', valid_loss)
                valid_prior_loss = valid_prior_metrics['mean']
                prior_gap = valid_prior_loss - valid_loss
                print(
                    f"Epoch {epoch}/{config['training_epochs']} LR: {current_lr:.2e} | "
                    f"TrainOpt  recon={train_loss:.2e} kl={train_kl:.2e} aux={train_aux:.2e} total={train_total:.2e} | "
                    f"TrainEvalQ recon={train_eval_loss:.2e} kl={train_eval_kl:.2e} total={train_eval_total:.2e} | "
                    f"ValidQ    recon={valid_loss:.2e} kl={valid_kl:.2e} total={valid_total:.2e} | "
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

            if valid_loss < best_valid_loss:
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
                    'vae_mp_layers': config.get('vae_mp_layers', 2),
                    'beta_aux': config.get('beta_aux', 1.0),
                }
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'normalization': normalization,
                    'model_config': model_config,
                }
                if use_vae:
                    save_dict['valid_prior_loss'] = valid_prior_loss
                    save_dict['valid_prior_samples'] = vae_valid_prior_samples
                if ema_model is not None:
                    save_dict['ema_state_dict'] = ema_model.state_dict()
                torch.save(save_dict, checkpoint_path)
                print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

            if log_file_dir:
                with open(log_file, 'a') as f:
                    if use_vae:
                        f.write(
                            f"Elapsed: {time.time() - start_time:.2f}s "
                            f"Epoch {epoch} LR: {current_lr:.4e} | "
                            f"TrainOpt recon={train_loss:.4e} kl={train_metrics.get('kl_mean', 0.0):.4e} total={train_metrics.get('total_mean', train_loss):.4e} | "
                            f"TrainEvalQ recon={train_eval_loss:.4e} kl={train_eval_metrics.get('kl_mean', 0.0):.4e} total={train_eval_metrics.get('total_mean', train_eval_loss):.4e} | "
                            f"ValidQ recon={valid_loss:.4e} kl={valid_metrics.get('kl_mean', 0.0):.4e} total={valid_metrics.get('total_mean', valid_loss):.4e} | "
                            f"ValidPrior@{vae_valid_prior_samples} recon={valid_prior_loss:.4e} gap={prior_gap:.4e}\n"
                        )
                    else:
                        f.write(
                            f"Elapsed: {time.time() - start_time:.2f}s "
                            f"Epoch {epoch} TrainOpt {train_loss:.4e} "
                            f"TrainEval {train_eval_loss:.4e} "
                            f"Valid {valid_loss:.4e} LR: {current_lr:.4e}\n"
                        )

            # Periodically test the model on the test set and save results with ground truth
            test_interval = int(config.get('test_interval', 10))
            last_epoch = epoch == config.get('training_epochs') - 1
            if epoch % test_interval == 0 or last_epoch:
                test_loss = test_model(eval_model, test_loader, device, config, epoch, train_dataset)
                print(f"  Test loss: {test_loss:.2e}")

                # Optionally visualize training set reconstruction (same batch indices)
                if config.get('display_trainset', True):
                    train_viz_indices = config.get('test_batch_idx', [0, 1, 2, 3, 4, 5, 6, 7])
                    # Clamp indices to train dataset size
                    train_viz_indices = [i for i in train_viz_indices if i < len(train_dataset)]
                    if train_viz_indices:
                        train_viz_loader = DataLoader(
                            Subset(train_dataset, train_viz_indices),
                            batch_size=1, shuffle=False, pin_memory=True
                        )
                        # Map batch indices to 0..N so all subset samples get visualized
                        viz_config = dict(config)
                        viz_config['test_batch_idx'] = list(range(len(train_viz_indices)))
                        train_viz_loss = test_model(eval_model, train_viz_loader, device, viz_config, epoch, train_dataset, output_prefix='train')
                        print(f"  Train reconstruction loss: {train_viz_loss:.2e}")

        print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Analyze debug files if they exist
    if log_dir:
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