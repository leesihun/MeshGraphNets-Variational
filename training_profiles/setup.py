"""
Shared setup helpers for training launchers.

Both `single_training.py` and `distributed_training.py` use these builders to
avoid maintaining duplicate dataset/model/optimizer/checkpoint logic.
"""

import os
import time

import numpy as np
import torch

from general_modules.data_loader import load_data
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import build_ema_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset_splits(config, split_seed: int):
    """
    Load dataset, split 80/10/10, inject metadata into config, and return the
    three split datasets.  Writing normalization stats to HDF5 and barrier
    synchronization (for DDP) are left to the caller.
    """
    dataset = load_data(config)

    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1, seed=split_seed)

    # Inject dataset-derived metadata so the model can be constructed correctly
    config['num_timesteps'] = train_dataset.num_timesteps
    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        config['num_node_types'] = train_dataset.num_node_types

    # Noise target-correction ratio (node_std / delta_std) — used in forward pass
    if train_dataset.node_std is not None and train_dataset.delta_std is not None:
        output_var = config['output_var']
        config['noise_std_ratio'] = (
            train_dataset.node_std[:output_var] / np.maximum(train_dataset.delta_std, 1e-8)
        ).tolist()

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_ema(config, device):
    """
    Instantiate MeshGraphNets, wrap with EMA if configured, and optionally
    compile with torch.compile.  Returns (model, ema_model); ema_model is None
    when `use_ema` is not set.
    """
    model = MeshGraphNets(config, str(device)).to(device)

    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    if config.get('use_compile', False):
        model = torch.compile(model, dynamic=True)

    return model, ema_model


def log_model_summary(model, config, ema_model=None):
    """Print a one-time summary of enabled model features and parameter counts."""
    print('\n' * 2)
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
        print(f"VAE (MMD): ENABLED (z_dim={config.get('vae_latent_dim', 32)}, lambda_mmd={config.get('lambda_mmd', 100.0)})")
    else:
        print("VAE: disabled")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(config, params, total_epochs: int):
    """
    Build fused Adam and a SequentialLR: linear warmup then cosine warm restarts.

    Scheduler hyper-parameters:
        warmup_epochs  (config key, default 3)
        cosine_T0 = total_epochs - warmup_epochs  (one full cycle over remaining epochs)
        cosine_T_mult = 1
        eta_min = 1e-8
    """
    learning_rate = config.get('learningr')
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.Adam(params, lr=learning_rate, fused=use_fused)

    warmup_epochs = int(config.get('warmup_epochs', 3))
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_T0 = remaining_epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_T0, T_mult=1, eta_min=1e-8
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    return optimizer, scheduler, warmup_epochs, cosine_T0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def build_normalization_dict(train_dataset) -> dict:
    """Collect normalization stats and optional extras into a serialisable dict."""
    norm = {
        'node_mean': train_dataset.node_mean,
        'node_std':  train_dataset.node_std,
        'edge_mean': train_dataset.edge_mean,
        'edge_std':  train_dataset.edge_std,
        'delta_mean': train_dataset.delta_mean,
        'delta_std':  train_dataset.delta_std,
    }
    if train_dataset.use_node_types and train_dataset.node_type_to_idx is not None:
        norm['node_type_to_idx'] = train_dataset.node_type_to_idx
        norm['num_node_types']   = train_dataset.num_node_types
    if train_dataset.use_world_edges and train_dataset.world_edge_radius is not None:
        norm['world_edge_radius'] = train_dataset.world_edge_radius
    if train_dataset.use_multiscale and len(train_dataset.coarse_edge_means) > 0:
        norm['coarse_edge_means'] = train_dataset.coarse_edge_means
        norm['coarse_edge_stds']  = train_dataset.coarse_edge_stds
    return norm


def resolve_prior_type(config) -> str:
    """Normalize `prior_type` in config and return it.

    'gnn_e2e' (the default) trains a graph-conditional prior jointly with the
    VAE; its density family is selected by `prior_family` ('fm' or 'gmm').
    The legacy post-hoc sklearn GMM path (`prior_type gmm`) has been removed.

    Writes the resolved value back into `config['prior_type']` so downstream code
    (including MeshGraphNets.__init__) reads a consistent string.
    """
    pt_norm = str(config.get('prior_type') or 'gnn_e2e').lower().strip()
    if pt_norm == 'gmm':
        raise ValueError(
            "prior_type 'gmm' (post-hoc sklearn GMM) has been removed. Use "
            "prior_type 'gnn_e2e' with prior_family 'gmm' for a graph-"
            "conditional mixture prior."
        )
    config['prior_type'] = pt_norm
    return pt_norm


def build_model_config(config) -> dict:
    """Collect architecture hyper-parameters into a serialisable dict."""
    return {
        'input_var':         config.get('input_var'),
        'output_var':        config.get('output_var'),
        'edge_var':          config.get('edge_var'),
        'latent_dim':        config.get('latent_dim'),
        'message_passing_num': config.get('message_passing_num'),
        'use_node_types':    config.get('use_node_types', False),
        'num_node_types':    config.get('num_node_types', 0),
        'positional_features': config.get('positional_features', 0),
        'use_world_edges':   config.get('use_world_edges', False),
        'use_checkpointing': config.get('use_checkpointing', False),
        'use_multiscale':    config.get('use_multiscale', False),
        'multiscale_levels': config.get('multiscale_levels', 1),
        'mp_per_level':      config.get('mp_per_level', None),
        'coarsening_type':   config.get('coarsening_type', 'bfs'),
        'voronoi_clusters':  config.get('voronoi_clusters', None),
        'use_vae':           config.get('use_vae', False),
        'vae_latent_dim':    config.get('vae_latent_dim', 32),
        'vae_mp_layers':     config.get('vae_mp_layers', 5),
        'vae_graph_aware':   config.get('vae_graph_aware', False),
        'posterior_min_std': config.get('posterior_min_std', 0),
        'num_z':             config.get('num_z',
                                (config.get('multiscale_levels', 1) + 1)
                                if config.get('use_multiscale', False) else 1),
        'beta_aux':          config.get('beta_aux', 1.0),
        # Prior selection + joint GNN E2E hyperparams. `prior_type` is what
        # MeshGraphNets reads at load time to decide whether to instantiate
        # `self.prior` — must be persisted so inference reconstructs the model
        # with the same architecture.
        'prior_type':              config.get('prior_type', ''),
        'use_conditional_prior':   config.get('use_conditional_prior', False),
        # Defaults below MUST match build_prior_config (model/conditional_prior.py),
        # which is what actually sizes the prior at construction. A mismatch makes
        # the saved model_config rebuild a differently-shaped prior at inference
        # and load_state_dict fails.
        'prior_family':             str(config.get('prior_family', 'fm')).lower().strip(),
        'prior_hidden_dim':         config.get('prior_hidden_dim', config.get('latent_dim')),
        'prior_mp_layers':          config.get('prior_mp_layers', 10),
        # fm family only:
        'prior_fm_steps':           config.get('prior_fm_steps', 20),
        # gmm family only:
        'prior_mixture_components': config.get('prior_mixture_components', 50),
        'prior_min_std':            config.get('prior_min_std', 0.1),
        'prior_cov_rank':           config.get('prior_cov_rank', 0),
        'prior_nll_weight':         config.get('prior_nll_weight', 1.0),
        'prior_kl_reg_weight':      config.get('prior_kl_reg_weight', 0.02),
    }


def save_checkpoint(
    epoch: int,
    bare_model,          # unwrapped model (no DDP wrapper)
    ema_model,
    optimizer,
    scheduler,
    train_loss: float,
    valid_loss: float,
    config,
    train_dataset,
    modelpath: str,
) -> None:
    """Build and write a checkpoint dict to `modelpath`."""
    save_dict = {
        'epoch':               epoch,
        'model_state_dict':    bare_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss':          train_loss,
        'valid_loss':          valid_loss,
        'normalization':       build_normalization_dict(train_dataset),
        'model_config':        build_model_config(config),
    }
    if ema_model is not None:
        save_dict['ema_state_dict'] = ema_model.state_dict()
    torch.save(save_dict, modelpath)


# ---------------------------------------------------------------------------
# Post-training helpers
# ---------------------------------------------------------------------------

def cleanup_dataloaders(*loaders) -> None:
    """Explicitly shut down DataLoader persistent workers before process exit.

    Without this, `persistent_workers=True` keeps worker processes alive until the
    Python interpreter tears down, at which point `multiprocessing.resource_tracker`
    reports "There appear to be N leaked semaphore objects to clean up at shutdown"
    (one semaphore per worker × loader).

    Call this at the end of every training worker function (single, DDP, model_split).
    """
    import gc
    for loader in loaders:
        if loader is None:
            continue
        it = getattr(loader, '_iterator', None)
        if it is not None and hasattr(it, '_shutdown_workers'):
            try:
                it._shutdown_workers()
            except Exception:
                pass
        try:
            loader._iterator = None
        except Exception:
            pass
    gc.collect()


def init_log_file(config, config_filename: str):
    """Create the epoch log file (with the config embedded) and return its path, or None."""
    log_file_dir = config.get('log_file_dir')
    if not log_file_dir:
        return None

    log_file = 'outputs/' + log_file_dir
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, 'w') as f:
        f.write("Training epoch log file\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file absolute path: {os.path.abspath(log_file)}\n")
        with open(config_filename, 'r') as fc:
            f.write(fc.read())
    return log_file
