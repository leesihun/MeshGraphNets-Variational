import os
import time

import torch
from torch_geometric.loader import DataLoader

from model.MeshGraphNets import MeshGraphNets
from model.conditional_prior import (
    ConditionalMixturePrior,
    analytical_prior_kl_loss,
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


def _posterior_params(simulator, graph, output_dim):
    """Return analytical posterior params (mu, logvar). One forward pass."""
    inner = getattr(simulator, 'model', simulator)
    y = graph.y[:, :output_dim]
    _, mu, logvar = inner.vae_encoder(y, graph.edge_index, graph.edge_attr, graph.batch)
    return mu, logvar


def _posterior_samples_from(mu, logvar, mc_samples):
    if mc_samples <= 1:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mc_samples, *std.shape, device=std.device, dtype=std.dtype)
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def _prior_loss(prior_params, q_mu, q_logvar, loss_type, mc_samples):
    """Dispatch on loss_type. Returns scalar loss."""
    if loss_type == 'analytical_kl':
        return analytical_prior_kl_loss(prior_params, q_mu, q_logvar)
    target_z = _posterior_samples_from(q_mu, q_logvar, mc_samples)
    return mixture_nll(prior_params, target_z)


def _evaluate_prior(prior, simulator, loader, device, config, diagnose=False):
    prior.eval()
    output_dim = int(config.get('output_var'))
    mc_samples = int(config.get('prior_mc_samples', 4))
    loss_type = str(config.get('prior_loss_type', 'mc_nll')).lower()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_idx, graph in enumerate(loader):
            graph = graph.to(device)
            q_mu, q_logvar = _posterior_params(simulator, graph, output_dim)
            params = prior(graph)
            loss = _prior_loss(params, q_mu, q_logvar, loss_type, mc_samples)
            bsz = int(graph.batch.max().item()) + 1 if hasattr(graph, 'batch') else 1
            total_loss += loss.item() * bsz
            total_count += bsz

            if diagnose:
                q_sigma = torch.exp(0.5 * q_logvar)
                prior_sigma = torch.exp(params['log_std'])
                log_pi = torch.log_softmax(params['logits'], dim=-1)
                mix_entropy = -(log_pi.exp() * log_pi).sum(dim=-1)
                sample_ids = getattr(graph, 'sample_id', None)
                tag = f"sid={sample_ids}" if sample_ids is not None else f"batch={batch_idx}"
                print(
                    f"    [diag] {tag}  "
                    f"q_sigma_mean={q_sigma.mean().item():.4f}  "
                    f"q_sigma_max={q_sigma.max().item():.4f}  "
                    f"q_mu_abs_mean={q_mu.abs().mean().item():.4f}  "
                    f"prior_sigma_mean={prior_sigma.mean().item():.4f}  "
                    f"prior_sigma_min={prior_sigma.min().item():.4f}  "
                    f"mix_entropy={mix_entropy.mean().item():.4f}"
                )
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
    prior_diagnose_interval = int(config.get('prior_diagnose_interval', 0))
    output_dim = int(config.get('output_var'))
    mc_samples = int(config.get('prior_mc_samples', 4))
    loss_type = str(config.get('prior_loss_type', 'mc_nll')).lower()
    if loss_type not in ('mc_nll', 'analytical_kl'):
        raise ValueError(f"prior_loss_type must be 'mc_nll' or 'analytical_kl', got {loss_type}")
    print(f"  Prior loss type: {loss_type}")
    print(f"  Prior min std:   {config.get('prior_min_std', 1e-3)}")

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
                q_mu, q_logvar = _posterior_params(simulator, graph, output_dim)
            params = prior(graph)
            loss = _prior_loss(params, q_mu, q_logvar, loss_type, mc_samples)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=3.0)
            optimizer.step()

            bsz = int(graph.batch.max().item()) + 1 if hasattr(graph, 'batch') else 1
            total_loss += loss.item() * bsz
            total_count += bsz

        train_loss = total_loss / max(total_count, 1)
        do_val = (epoch % prior_val_interval == 0) or (epoch == prior_epochs - 1)
        if do_val and len(val_dataset) > 0:
            diagnose = (prior_diagnose_interval > 0
                        and epoch % prior_diagnose_interval == 0)
            val_loss = _evaluate_prior(
                prior, simulator, val_loader, device, config, diagnose=diagnose
            )
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
