import os
import time

import numpy as np
import torch
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


def single_worker(config, config_filename='config.txt'):
    """Single GPU/CPU training entry point."""
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")

    if torch.cuda.is_available():
        gpu_id = gpu_ids
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using physical GPU {gpu_id}, device: {device}')
        print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # ---- Dataset ----
    print("\nLoading dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    if torch.cuda.is_available():
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    print("Writing train-derived normalization stats to HDF5...")
    train_dataset.write_preprocessing_to_hdf5(split_seed)

    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")

    # ---- DataLoaders ----
    print("\nCreating dataloaders...")
    num_workers = config['num_workers']
    pin_memory = torch.cuda.is_available()
    mp_context = 'spawn' if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory)

    if torch.cuda.is_available():
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    print("\nInitializing model...")
    model, ema_model = build_model_and_ema(config, device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    log_model_summary(model, config, ema_model)

    # ---- Optimizer / Scheduler ----
    print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, model.parameters(), total_epochs
    )
    use_fused = torch.cuda.is_available()
    print(f"Optimizer: Adam (fused={use_fused})")
    print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
          f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    log_training_config(config)
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    start_time = time.time()

    # ---- Logging ----
    log_file, log_dir = init_log_file(config, config_filename)

    modelname = config.get('modelpath')
    use_vae = config.get('use_vae', False)
    vae_valid_prior_samples = int(config.get('vae_valid_prior_samples', 8)) if use_vae else 1
    if vae_valid_prior_samples < 1:
        raise ValueError("vae_valid_prior_samples must be >= 1")

    best_valid_loss = float('inf')
    best_epoch = -1
    val_interval = int(config.get('val_interval', 1))

    try:
        # Pre-create first iterator so workers begin filling the buffer immediately.
        train_iter = iter(train_loader)
        for epoch in range(total_epochs):
            current_iter = train_iter
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
                _iter=current_iter,
            )
            # Start loading next epoch's data NOW — workers prefetch in the background
            # while val eval (or just logging) occupies the CPU/GPU.
            train_iter = iter(train_loader)

            train_loss = train_metrics['mean']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)

            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                if use_vae:
                    valid_metrics = evaluate_vae_posterior_epoch(
                        eval_model, val_loader, device, config, epoch, progress_name='ValidQ'
                    )
                    valid_prior_metrics = evaluate_vae_prior_epoch(
                        eval_model, val_loader, device, config, epoch,
                        num_prior_samples=vae_valid_prior_samples,
                        progress_name=f'ValidPrior@{vae_valid_prior_samples}'
                    )
                else:
                    valid_metrics      = validate_epoch(eval_model, val_loader, device, config, epoch)
                    valid_prior_metrics = None
                valid_loss = valid_metrics['mean']
            else:
                valid_loss = best_valid_loss  # reuse last known for checkpoint gating
                valid_metrics = {}
                valid_prior_metrics = None

            if use_vae:
                train_mmd   = train_metrics.get('mmd_mean', 0.0)
                train_aux   = train_metrics.get('aux_mean', 0.0)
                train_total = train_metrics.get('total_mean', train_loss)
                if do_val:
                    valid_mmd   = valid_metrics.get('mmd_mean', 0.0)
                    valid_total = valid_metrics.get('total_mean', valid_loss)
                    valid_prior_loss = valid_prior_metrics['mean']
                    prior_gap = valid_prior_loss - valid_loss
                    print(
                        f"Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                        f"TrainOpt  recon={train_loss:.2e} mmd={train_mmd:.2e} aux={train_aux:.2e} total={train_total:.2e} | "
                        f"ValidQ    recon={valid_loss:.2e} mmd={valid_mmd:.2e} total={valid_total:.2e} | "
                        f"ValidPrior@{vae_valid_prior_samples} recon={valid_prior_loss:.2e} gap={prior_gap:.2e}"
                    )
                else:
                    valid_prior_loss = None
                    print(
                        f"Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                        f"TrainOpt  recon={train_loss:.2e} mmd={train_mmd:.2e} aux={train_aux:.2e} total={train_total:.2e}"
                    )
            else:
                valid_prior_loss = None
                if do_val:
                    print(
                        f"Epoch {epoch}/{total_epochs} "
                        f"TrainOpt: {train_loss:.2e} "
                        f"Valid: {valid_loss:.2e} LR: {current_lr:.2e}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{total_epochs} "
                        f"TrainOpt: {train_loss:.2e} LR: {current_lr:.2e}"
                    )

            if do_val and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_checkpoint(
                    epoch, model, ema_model, optimizer, scheduler,
                    train_loss, valid_loss, config, train_dataset, modelname,
                    use_vae=use_vae,
                    valid_prior_loss=valid_prior_loss if use_vae else None,
                    vae_valid_prior_samples=vae_valid_prior_samples if use_vae else None,
                )
                print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    if use_vae:
                        val_str = (
                            f"ValidQ recon={valid_loss:.4e} mmd={valid_metrics.get('mmd_mean',0):.4e} "
                            f"total={valid_metrics.get('total_mean',valid_loss):.4e} | "
                            f"ValidPrior@{vae_valid_prior_samples} recon={valid_prior_loss:.4e} "
                            f"gap={prior_gap:.4e}"
                        ) if do_val else "ValidQ skipped"
                        f.write(
                            f"Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                            f"TrainOpt recon={train_loss:.4e} mmd={train_metrics.get('mmd_mean',0):.4e} "
                            f"total={train_metrics.get('total_mean',train_loss):.4e} | "
                            f"{val_str}\n"
                        )
                    else:
                        val_str = f"Valid {valid_loss:.4e}" if do_val else "Valid skipped"
                        f.write(
                            f"Elapsed: {elapsed:.2f}s "
                            f"Epoch {epoch} TrainOpt {train_loss:.4e} "
                            f"{val_str} LR: {current_lr:.4e}\n"
                        )

            test_interval = int(config.get('test_interval', 10))
            last_epoch = epoch == total_epochs - 1
            if epoch % test_interval == 0 or last_epoch:
                test_loss = test_model(eval_model, test_loader, device, config, epoch, train_dataset)
                print(f"  Test loss: {test_loss:.2e}")

                if config.get('display_trainset', True):
                    train_viz_indices = config.get('test_batch_idx', [0, 1, 2, 3, 4, 5, 6, 7])
                    train_viz_indices = [i for i in train_viz_indices if i < len(train_dataset)]
                    if train_viz_indices:
                        train_viz_loader = DataLoader(
                            Subset(train_dataset, train_viz_indices),
                            batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available()
                        )
                        viz_config = dict(config)
                        viz_config['test_batch_idx'] = list(range(len(train_viz_indices)))
                        train_viz_loss = test_model(
                            eval_model, train_viz_loader, device, viz_config, epoch,
                            train_dataset, output_prefix='train'
                        )
                        print(f"  Train reconstruction loss: {train_viz_loss:.2e}")

        print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Post-hoc GMM fitting on VAE latent codes
    if use_vae and config.get('fit_latent_gmm', False):
        from model.latent_gmm import run_posthoc_gmm_fitting
        gmm_model = ema_model.module if ema_model is not None else model
        run_posthoc_gmm_fitting(gmm_model, train_dataset, config, device, modelname)

    analyze_debug_files(log_dir)
