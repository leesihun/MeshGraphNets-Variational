import os
import time

import torch
from torch_geometric.loader import DataLoader

from model.MeshGraphNets import MeshGraphNets
from model.conditional_prior import (
    ConditionalMixturePrior,
    build_prior_config,
    mixture_nll,
)
from training_profiles.setup import build_dataset_splits


def _select_device(config):
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        torch.cuda.set_device(gpu_ids[0])
        return torch.device(f'cuda:{gpu_ids[0]}')
    return torch.device('cpu')


def _override_config_from_checkpoint(config, checkpoint):
    model_config = checkpoint.get('model_config', {})
    for key, value in model_config.items():
        config[key] = value
    return config


def _apply_checkpoint_preprocessing(dataset, checkpoint):
    norm = checkpoint['normalization']
    dataset.node_mean = norm['node_mean']
    dataset.node_std = norm['node_std']
    dataset.edge_mean = norm['edge_mean']
    dataset.edge_std = norm['edge_std']
    dataset.delta_mean = norm['delta_mean']
    dataset.delta_std = norm['delta_std']
    dataset.node_type_to_idx = norm.get('node_type_to_idx')
    dataset.num_node_types = norm.get('num_node_types')
    dataset.world_edge_radius = norm.get('world_edge_radius')
    dataset.coarse_edge_means = [m.copy() for m in norm.get('coarse_edge_means', [])]
    dataset.coarse_edge_stds = [s.copy() for s in norm.get('coarse_edge_stds', [])]


def _load_frozen_simulator(config, checkpoint, device):
    model = MeshGraphNets(config, str(device)).to(device)
    if 'ema_state_dict' in checkpoint:
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        model.load_state_dict(model_sd)
        print("  Loaded frozen EMA simulator weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded frozen simulator weights")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _posterior_samples(simulator, graph, output_dim, mc_samples):
    inner = getattr(simulator, 'model', simulator)
    y = graph.y[:, :output_dim]
    _, mu, logvar = inner.vae_encoder(y, graph.edge_index, graph.edge_attr, graph.batch)
    if mc_samples <= 1:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mc_samples, *std.shape, device=std.device, dtype=std.dtype)
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def _evaluate_prior(prior, simulator, loader, device, config):
    prior.eval()
    output_dim = int(config.get('output_var'))
    mc_samples = int(config.get('prior_mc_samples', 4))
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            target_z = _posterior_samples(simulator, graph, output_dim, mc_samples)
            params = prior(graph)
            loss = mixture_nll(params, target_z)
            bsz = int(graph.batch.max().item()) + 1 if hasattr(graph, 'batch') else 1
            total_loss += loss.item() * bsz
            total_count += bsz
    prior.train()
    return total_loss / max(total_count, 1)


def train_posthoc_prior(config, config_filename='config.txt'):
    if not config.get('use_vae', False):
        raise ValueError("Post-hoc prior training requires use_vae True")

    model_path = config.get('modelpath')
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = _select_device(config)
    print("\n" + "=" * 60)
    print("POST-HOC CONDITIONAL PRIOR TRAINING")
    print("=" * 60)
    print(f"Checkpoint: {model_path}")
    print(f"Device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'normalization' not in checkpoint:
        raise KeyError("Checkpoint is missing normalization stats")
    config = _override_config_from_checkpoint(config, checkpoint)
    config['use_conditional_prior'] = True

    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, _ = build_dataset_splits(config, split_seed)
    for dataset in (train_dataset, val_dataset):
        _apply_checkpoint_preprocessing(dataset, checkpoint)

    simulator = _load_frozen_simulator(config, checkpoint, device)
    prior = ConditionalMixturePrior(config).to(device)

    if 'conditional_prior_state_dict' in checkpoint and config.get('resume_prior', True):
        prior.load_state_dict(checkpoint['conditional_prior_state_dict'])
        print("  Resumed existing conditional prior weights from checkpoint")

    prior_epochs = int(config.get('prior_epochs', 200))
    prior_lr = float(config.get('prior_learningr', config.get('learningr', 1e-4)))
    prior_batch_size = int(config.get('prior_batch_size', config.get('batch_size', 4)))
    prior_num_workers = int(config.get('prior_num_workers', 0))
    prior_val_interval = int(config.get('prior_val_interval', 10))
    output_dim = int(config.get('output_var'))
    mc_samples = int(config.get('prior_mc_samples', 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=prior_batch_size,
        shuffle=True,
        num_workers=prior_num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=prior_batch_size,
        shuffle=False,
        num_workers=prior_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(prior.parameters(), lr=prior_lr)
    best_val = float('inf')
    best_state = None
    start_time = time.time()

    for epoch in range(prior_epochs):
        prior.train()
        total_loss = 0.0
        total_count = 0
        for graph in train_loader:
            graph = graph.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                target_z = _posterior_samples(simulator, graph, output_dim, mc_samples)
            params = prior(graph)
            loss = mixture_nll(params, target_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=3.0)
            optimizer.step()

            bsz = int(graph.batch.max().item()) + 1 if hasattr(graph, 'batch') else 1
            total_loss += loss.item() * bsz
            total_count += bsz

        train_loss = total_loss / max(total_count, 1)
        do_val = (epoch % prior_val_interval == 0) or (epoch == prior_epochs - 1)
        if do_val and len(val_dataset) > 0:
            val_loss = _evaluate_prior(prior, simulator, val_loader, device, config)
        else:
            val_loss = train_loss

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}

        print(
            f"Prior epoch {epoch}/{prior_epochs} "
            f"train_nll={train_loss:.4e} val_nll={val_loss:.4e}"
        )

    if best_state is not None:
        prior.load_state_dict(best_state)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    checkpoint['conditional_prior_state_dict'] = {
        k: v.detach().cpu() for k, v in prior.state_dict().items()
    }
    checkpoint['conditional_prior_config'] = build_prior_config(config)
    checkpoint['conditional_prior_metrics'] = {
        'best_val_nll': float(best_val),
        'epochs': prior_epochs,
        'training_time_s': time.time() - start_time,
        'config_file': config_filename,
    }
    torch.save(checkpoint, model_path)
    print(f"Saved conditional prior into checkpoint: {model_path}")
    return checkpoint['conditional_prior_metrics']
