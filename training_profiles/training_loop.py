import contextlib
import time

import tqdm
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from general_modules.mesh_utils_fast import (
    save_inference_results_fast,
    render_plot_data,
    edges_to_triangles_gpu,
    edges_to_triangles_optimized,
)
from model.conditional_prior import (
    analytical_prior_kl_loss,
    mixture_nll,
    sample_from_mixture,
)


def _unwrap_for_submodule(model):
    """Return the underlying MeshGraphNets module, peeling DDP / EMA / compile wrappers.

    The joint GNN E2E objective needs access to `model.prior` (a submodule on
    `MeshGraphNets`). DDP wraps it under `.module`; AveragedModel under `.module`;
    `torch.compile` under `._orig_mod`. Try each.
    """
    m = model
    for attr in ('module', '_orig_mod'):
        inner = getattr(m, attr, None)
        if inner is not None:
            m = inner
    return m


def build_ema_model(model, config):
    """Create an EMA shadow model if use_ema is enabled.

    Returns the AveragedModel or None if EMA is disabled.
    The source model should be the raw nn.Module (before torch.compile or DDP wrapping).
    """
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=decay))
    # Disable gradients on shadow parameters
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _build_loss_weights(config, device):
    """Build per-feature loss weights normalized to sum to 1 (weighted mean)."""
    loss_weights = config.get('feature_loss_weights', None)
    if loss_weights is not None:
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights]
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)
        loss_weights = loss_weights / loss_weights.sum()
    return loss_weights


def _per_node_loss(errors, loss_weights):
    """Reduce feature errors to one scalar per node (mean over features)."""
    if loss_weights is not None:
        return torch.sum(errors * loss_weights, dim=-1)
    return torch.mean(errors, dim=-1)


def _loss_from_errors(errors, loss_weights):
    """Return mean loss used for backprop plus exact aggregation stats."""
    per_node = _per_node_loss(errors, loss_weights)
    loss_sum = per_node.sum()
    loss_count = per_node.numel()
    return loss_sum / loss_count, loss_sum.item(), loss_count


def _recon_errors(predicted, target, recon_loss='huber'):
    """Element-wise reconstruction errors. 'mse' (squared error) penalizes large,
    extreme-node errors far more than Huber(delta=1), which caps them — important
    for reproducing the warpage amplitude (peak-to-valley), an extreme-value
    statistic Huber systematically smooths. Default 'huber' keeps old behavior."""
    if str(recon_loss).lower().strip() == 'mse':
        return (predicted - target).pow(2)
    return torch.nn.functional.huber_loss(predicted, target, reduction='none', delta=1.0)


def _crps_from_samples(samples, target):
    """Fair (unbiased) CRPS estimator, per element.

    CRPS is the standard proper scoring rule for a probabilistic forecast scored
    against a single observation: it is minimized exactly when the generated
    distribution equals the true one, penalizing both a wrong mean AND wrong
    spread (too narrow OR too wide). This makes it the right validation metric
    for spread modeling, where recon-only metrics reward mean collapse.

        CRPS ≈ mean_s |x_s - y|  −  1/(2 S (S-1)) Σ_s Σ_j |x_s - x_j|

    The second (fair) normalization removes the finite-sample bias of the naive
    1/(2 S²) form, which otherwise rewards under-dispersion at small S.

    samples: [S, N, F] prior-sampled predictions. target: [N, F]. Returns [N, F].
    Uses absolute (L1) distance by definition, independent of `recon_loss`.
    """
    S = samples.shape[0]
    accuracy = (samples - target.unsqueeze(0)).abs().mean(dim=0)  # [N, F]
    if S < 2:
        return accuracy
    # Vectorized pairwise mean abs difference: [S, S, N, F] → sum dims 0,1 → [N, F].
    # Peak extra memory ≈ S² × N × F floats (e.g. 8²×96k×3 ≈ 74 MB). No loops.
    spread = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().sum(dim=[0, 1]) / (2.0 * S * (S - 1))
    return accuracy - spread


def _delta_stats(dataloader, device):
    """Fetch (delta_mean, delta_std) tensors from the loader's dataset (handles a
    Subset wrapper). Returns (None, None) if unavailable — used to denormalize
    predictions for the generated-amplitude tail metric."""
    ds = getattr(dataloader, 'dataset', None)
    for obj in (ds, getattr(ds, 'dataset', None)):
        dm = getattr(obj, 'delta_mean', None)
        st = getattr(obj, 'delta_std', None)
        if dm is not None and st is not None:
            return (torch.as_tensor(dm, dtype=torch.float32, device=device),
                    torch.as_tensor(st, dtype=torch.float32, device=device))
    return None, None


def _move_graph_to_device(graph, device, config):
    non_blocking = bool(config.get('_pin_memory', False)) and getattr(device, 'type', None) == 'cuda'
    return graph.to(device, non_blocking=non_blocking)


def _accum_window_size(batch_idx, total_batches, actual_accum):
    """Return the number of batches in the current accumulation window."""
    window_start = (batch_idx // actual_accum) * actual_accum
    window_end = min(window_start + actual_accum, total_batches)
    return window_end - window_start


def log_training_config(config):
    """Log per-feature loss weights and multiscale architecture to stdout."""
    loss_weights_cfg = config.get('feature_loss_weights', None)
    if loss_weights_cfg is not None:
        if not isinstance(loss_weights_cfg, list):
            loss_weights_cfg = [loss_weights_cfg]
        w = torch.tensor(loss_weights_cfg, dtype=torch.float32)
        w_normalized = (w / w.sum()).tolist()
        print(f"Per-feature loss weights (raw):         {loss_weights_cfg}")
        print(f"Per-feature loss weights (normalized):  {[f'{v:.4f}' for v in w_normalized]}")
    else:
        print("Per-feature loss weights: equal (default)")

    if config.get('use_vae', False):
        z_dim   = config.get('vae_latent_dim', 32)
        mp_enc  = config.get('vae_mp_layers', 5)
        lam     = config.get('lambda_mmd', 1.0)
        b_aux   = config.get('beta_aux', 1.0)
        alpha   = config.get('alpha_recon', 1.0)
        print(f"VAE (MMD): ENABLED (z_dim={z_dim}, vae_mp_layers={mp_enc}, "
              f"lambda_mmd={lam}, beta_aux={b_aux}, alpha_recon={alpha})")
        if str(config.get('prior_type', '')).lower().strip() == 'gnn_e2e':
            p_family = str(config.get('prior_family', 'fm')).lower().strip()
            p_w = config.get('prior_nll_weight', 1.0)
            if p_family == 'gmm':
                p_kl = config.get('prior_kl_reg_weight', 0.02)
                print(f"Prior (gnn_e2e, family=gmm): nll_weight={p_w} kl_anchor={p_kl}")
            else:
                print(f"Prior (gnn_e2e, family=fm): fm_weight={p_w} "
                      f"euler_steps={config.get('prior_fm_steps', 20)}")
    else:
        print("VAE: disabled")

    if config.get('use_multiscale', False):
        _L = int(config.get('multiscale_levels', 1))
        _mp = config.get('mp_per_level', [])
        if not isinstance(_mp, list):
            _mp = [int(_mp)]
        print(f"Multi-Scale: ENABLED (V-cycle, {_L} coarsening levels, {sum(int(x) for x in _mp)} total GnBlocks)")
        for i in range(_L):
            print(f"  Level {i} pre:  {_mp[i]} blocks")
        print(f"  Coarsest:    {_mp[_L]} blocks")
        for i in range(_L - 1, -1, -1):
            print(f"  Level {i} post: {_mp[2 * _L - i]} blocks")
        print(f"  [message_passing_num is IGNORED when use_multiscale=True]")
    else:
        print(f"Multi-Scale: disabled (flat GNN, message_passing_num={config.get('message_passing_num')})")


def train_epoch(model, dataloader, optimizer, device, config, epoch, ema_model=None):
    model.train()
    total_loss_sum = 0.0
    total_loss_count = 0
    mmd_count = 0

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    recon_loss_kind = config.get('recon_loss', 'huber')

    # VAE config (MMD objective: InfoVAE-style aggregate-posterior matching)
    use_vae = config.get('use_vae', False)
    alpha_recon = float(config.get('alpha_recon', 1.0))
    lambda_mmd = float(config.get('lambda_mmd', 1.0))
    beta_aux = float(config.get('beta_aux', 1.0))

    # Joint GNN E2E prior config — only active when prior_type='gnn_e2e' and the
    # model carries a prior submodule.
    #
    # The prior's job is density matching: fit p(z|graph) to the cloud of
    # posterior latents so inference samples from the territory the decoder was
    # trained on. The matching objective depends on prior_family:
    #   - fm (default): conditional flow-matching MSE on a FRESH reparameterized
    #     posterior sample each step. The regression target averages over
    #     samples by construction, so there is no component-collapse mode and
    #     no KL anchor is needed.
    #   - gmm: mixture NLL of the fresh sample plus a small analytical-KL
    #     Jensen-bound anchor (prior_kl_reg_weight) for stability.
    # Both weights share the prior_nll_weight key. Fresh resampling makes the
    # target the smoothed posterior cloud rather than fixed points.
    prior_type = str(config.get('prior_type', '')).lower().strip()
    inner_model = _unwrap_for_submodule(model)
    has_gnn_prior = (
        use_vae and prior_type == 'gnn_e2e'
        and getattr(inner_model, 'prior', None) is not None
    )
    prior_family = getattr(getattr(inner_model, 'prior', None), 'family', 'gmm')
    prior_loss_weight = float(config.get('prior_nll_weight', 1.0)) if has_gnn_prior else 0.0
    prior_kl_reg_weight = (
        float(config.get('prior_kl_reg_weight', 0.02))
        if (has_gnn_prior and prior_family == 'gmm') else 0.0
    )
    compute_prior_path = has_gnn_prior and (
        prior_loss_weight > 0.0 or prior_kl_reg_weight > 0.0
    )
    # GPU accumulators: avoid per-batch .item() syncs (sync once per epoch at result).
    total_opt_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
    total_mmd_gpu     = torch.zeros((), device=device, dtype=torch.float32) if use_vae else None
    total_aux_gpu     = torch.zeros((), device=device, dtype=torch.float32) if use_vae else None
    total_kl_anchor_gpu  = torch.zeros((), device=device, dtype=torch.float32) if has_gnn_prior else None
    total_prior_loss_gpu = torch.zeros((), device=device, dtype=torch.float32) if has_gnn_prior else None

    # Clip the prior's gradients separately from the simulator's. The prior
    # density loss can be large early in training (posteriors far from the
    # freshly initialized prior); under a shared global clip that gradient
    # would scale down the simulator's updates.
    if has_gnn_prior:
        prior_param_ids = {id(p) for p in inner_model.prior.parameters()}
        sim_params = [p for p in model.parameters() if id(p) not in prior_param_ids]
        prior_params_list = [p for p in model.parameters() if id(p) in prior_param_ids]
    else:
        sim_params = list(model.parameters())
        prior_params_list = []

    # Gradient accumulation: 0 = full epoch (1 step/epoch), 1 = per-batch (default), N = every N batches
    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm.tqdm(dataloader, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        graph = _move_graph_to_device(graph, device, config)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            predicted_acc, target_acc, vae_losses, aux_loss_val, prior_outputs = model(
                graph, compute_prior_path=compute_prior_path,
            )
            mmd_loss_val = vae_losses['mmd']

        # KL anchor for the gmm prior family. Computed in fp32 outside the
        # autocast region because analytical_prior_kl_loss does exp(log_var_k)
        # and 1/var_k which underflow in bfloat16. .detach() on μ, logvar so the
        # encoder isn't dragged by the anchor — only the prior is constrained.
        kl_anchor_val = torch.zeros((), device=predicted_acc.device, dtype=torch.float32)
        if (prior_outputs is not None
                and prior_outputs.get('prior_params') is not None
                and prior_kl_reg_weight > 0.0
                and vae_losses.get('mu') is not None):
            kl_anchor_val = analytical_prior_kl_loss(
                {k: v.float() for k, v in prior_outputs['prior_params'].items()},
                vae_losses['mu'].detach().float(),
                vae_losses['logvar'].detach().float(),
            )

        # Prior density-matching loss on a FRESH posterior sample (detached,
        # fp32). Resampling z ~ q(z|y) every step makes the target the smoothed
        # posterior cloud rather than a fixed set of points. Gradient reaches
        # only the prior — q is detached; for fm it also flows into the prior's
        # trunk via the pooled conditioning vector from the forward pass.
        prior_loss_val = torch.zeros((), device=predicted_acc.device, dtype=torch.float32)
        if (prior_outputs is not None
                and prior_loss_weight > 0.0
                and vae_losses.get('mu') is not None):
            mu_q = vae_losses['mu'].detach().float()
            logvar_q = vae_losses['logvar'].detach().float()
            z_fresh = mu_q + torch.exp(0.5 * logvar_q) * torch.randn_like(mu_q)
            if prior_outputs.get('pooled') is not None:
                prior_loss_val = inner_model.prior.fm_loss(
                    prior_outputs['pooled'], z_fresh,
                )
            else:
                prior_loss_val = mixture_nll(
                    {k: v.float() for k, v in prior_outputs['prior_params'].items()},
                    z_fresh,
                )

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            errors = _recon_errors(predicted_acc, target_acc, recon_loss_kind)
            recon_loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            if use_vae:
                loss = alpha_recon * recon_loss + lambda_mmd * mmd_loss_val + beta_aux * aux_loss_val
                # Joint GNN E2E prior contribution: density matching (mixture
                # NLL or flow-matching MSE; + small KL anchor for gmm).
                if prior_outputs is not None and prior_kl_reg_weight > 0.0:
                    loss = loss + prior_kl_reg_weight * kl_anchor_val
                if prior_outputs is not None and prior_loss_weight > 0.0:
                    loss = loss + prior_loss_weight * prior_loss_val
            else:
                loss = recon_loss
            # Scale loss so accumulated gradients equal the mean within each accumulation window.
            scaled_loss = loss / _accum_window_size(batch_idx, total_batches, actual_accum)

        # Suppress DDP gradient all-reduce on non-step batches (accumulation window).
        # no_sync() is a no-op on non-DDP models; hasattr check keeps single-GPU path clean.
        is_step_batch = (batch_idx + 1) % actual_accum == 0 or (batch_idx == total_batches - 1)
        sync_ctx = (contextlib.nullcontext() if (is_step_batch or not hasattr(model, 'no_sync'))
                    else model.no_sync())
        with sync_ctx:
            scaled_loss.backward()

        loss_val = batch_loss_sum / batch_loss_count
        total_loss_sum += batch_loss_sum
        total_loss_count += batch_loss_count
        # Accumulate on GPU; .item() deferred to epoch end (one sync per metric).
        total_opt_loss_gpu += loss.detach().float() * batch_loss_count
        if use_vae:
            total_mmd_gpu += mmd_loss_val.detach().float()
            total_aux_gpu += aux_loss_val.detach().float()
            mmd_count += 1
            if prior_outputs is not None:
                total_kl_anchor_gpu  += kl_anchor_val.detach()
                total_prior_loss_gpu += prior_loss_val.detach()

        # Step optimizer at end of accumulation window or final batch
        if is_step_batch:
            torch.nn.utils.clip_grad_norm_(sim_params, max_norm=3.0)
            if prior_params_list:
                torch.nn.utils.clip_grad_norm_(prior_params_list, max_norm=3.0)
            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar every 10 batches to reduce memory query overhead
        if batch_idx % 10 == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            postfix = {'rec': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'}
            if use_vae:
                postfix['mmd'] = f'{float(mmd_loss_val):.2e}'
                postfix['aux'] = f'{float(aux_loss_val):.2e}'
                postfix['λ']   = f'{lambda_mmd:.1e}'
                if prior_outputs is not None:
                    p_key = 'fm_p' if prior_outputs.get('pooled') is not None else 'nll_p'
                    postfix[p_key] = f'{prior_loss_val.item():.2e}'
                    if prior_kl_reg_weight > 0.0:
                        postfix['kl_a'] = f'{kl_anchor_val.item():.2e}'
                postfix['total'] = f'{loss.item():.2e}'
            pbar.set_postfix(postfix)

    # Sync GPU accumulators once per epoch (vs once per batch previously).
    result = {
        'mean': total_loss_sum / total_loss_count,
        'total_mean': total_opt_loss_gpu.item() / total_loss_count,
        'sum': total_loss_sum,
        'count': total_loss_count,
    }
    if use_vae and mmd_count > 0:
        result['mmd_mean'] = total_mmd_gpu.item() / mmd_count
        result['aux_mean'] = total_aux_gpu.item() / mmd_count
        if has_gnn_prior:
            result['prior_loss_mean'] = total_prior_loss_gpu.item() / mmd_count
            if prior_kl_reg_weight > 0.0:
                result['kl_anchor_mean'] = total_kl_anchor_gpu.item() / mmd_count
    return result


def _evaluate_epoch(model, dataloader, device, config, epoch=0, *,
                    use_posterior=None, progress_name='Validation'):
    model.eval()

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    recon_loss_kind = config.get('recon_loss', 'huber')

    use_vae = config.get('use_vae', False)
    alpha_recon = float(config.get('alpha_recon', 1.0))
    lambda_mmd = float(config.get('lambda_mmd', 1.0))

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0
        # GPU accumulators: avoid per-batch .item() syncs; sync once at epoch end.
        total_opt_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
        total_mmd_gpu = torch.zeros((), device=device, dtype=torch.float32) if use_vae else None
        mmd_count = 0

        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for batch_idx, graph in enumerate(pbar):
            graph = _move_graph_to_device(graph, device, config)

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target, vae_losses, _, _ = model(
                    graph, add_noise=False, use_posterior=use_posterior,
                )
                errors = _recon_errors(predicted, target, recon_loss_kind)
            mmd_loss_val = vae_losses['mmd']

            recon_loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            if use_vae:
                opt_loss = alpha_recon * recon_loss + lambda_mmd * mmd_loss_val
                total_opt_loss_gpu += opt_loss.detach() * batch_loss_count
                total_mmd_gpu += mmd_loss_val.detach().float()
                mmd_count += 1
            else:
                total_opt_loss_gpu += recon_loss.detach() * batch_loss_count

            total_loss_sum += batch_loss_sum
            total_loss_count += batch_loss_count

            # Throttle display to every 10 batches — avoids per-batch .item() GPU syncs.
            if batch_idx % 10 == 0:
                mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                postfix = {'rec': f'{batch_loss_sum / max(batch_loss_count, 1):.2e}',
                           'mem': f'{mem_gb:.1f}GB'}
                if use_vae:
                    postfix['mmd'] = f'{float(mmd_loss_val):.2e}'
                    postfix['total'] = f'{float(opt_loss):.2e}'
                pbar.set_postfix(postfix)

    result = {
        'mean': total_loss_sum / total_loss_count,
        'total_mean': total_opt_loss_gpu.item() / max(total_loss_count, 1),
        'sum': total_loss_sum,
        'count': total_loss_count,
    }
    if use_vae and mmd_count > 0:
        result['mmd_mean'] = total_mmd_gpu.item() / mmd_count
    return result


def validate_epoch(model, dataloader, device, config, epoch=0):
    return _evaluate_epoch(
        model, dataloader, device, config, epoch,
        use_posterior=None, progress_name='Validation',
    )


def evaluate_vae_posterior_epoch(model, dataloader, device, config, epoch=0, progress_name='ValidationQ'):
    return _evaluate_epoch(
        model, dataloader, device, config, epoch,
        use_posterior=True, progress_name=progress_name,
    )


def evaluate_vae_learned_prior_epoch(model, dataloader, device, config, epoch=0,
                                     progress_name='ValidationLearnedPrior'):
    """Evaluate reconstruction quality using the joint-trained GNN prior.

    Mirrors inference: sample z from p(z|graph) with the hard (non-reparameterized)
    sampler, then run the simulator with that z. This is the most informative
    validation signal for spread modeling — it answers "if I sampled the way I
    sample at inference, how good would the reconstruction be?".

    Also collects the prior-collapse diagnostic: per z-slot, the width of the
    posterior cloud (std of μ_q across samples plus mean posterior σ) vs the
    prior's own std. `spread_ratio` (prior/posterior) near 1 is healthy; near 0
    means the prior has collapsed and generated samples will have far too
    little spread.

    No-op when there is no learned prior (returns None).
    """
    inner = _unwrap_for_submodule(model)
    if getattr(inner, 'prior', None) is None:
        return None
    vae_encoder = getattr(getattr(inner, 'model', None), 'vae_encoder', None)
    is_fm_prior = getattr(inner.prior, 'family', 'gmm') == 'fm'
    # Prior samples drawn per graph, used for (a) the CRPS estimate and (b) the
    # prior-spread diagnostic (fm has no closed-form variance, so it is estimated
    # from this sample cloud). Sample 0 doubles as the recon eval z.
    num_prior_samples = int(config.get('vae_valid_prior_samples', 8))

    model.eval()
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16
    prior_temperature = float(config.get('prior_temperature', 1.0))

    total_loss_sum = 0.0
    total_loss_count = 0
    total_crps_sum = 0.0
    total_crps_count = 0
    mu_q_all = []          # posterior means per sample [B, num_z, D]
    var_q_all = []         # posterior variances per sample [B, num_z, D]
    prior_var_all = []     # prior total variance per graph [B, num_z, D]

    # Generated-amplitude tail tracking (matches compare_histograms' metric):
    # per-graph max-min of denormalized z_disp from prior-sampled predictions.
    amp_gen_all, amp_gt_all, _track_tail, _zc = [], [], False, 2
    try:
        from torch_geometric.utils import scatter as _scatter
        _dm, _st = _delta_stats(dataloader, device)
        _track_tail = _dm is not None and _st is not None
    except Exception:
        _scatter = None

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for graph in pbar:
            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                if is_fm_prior:
                    z_cloud = inner.prior.sample_n(
                        graph, num_prior_samples, temperature=prior_temperature,
                    )  # [B, S, num_z, D]
                else:
                    prior_params = inner.prior(graph)
                    z_cloud = torch.stack(
                        [sample_from_mixture(prior_params, temperature=prior_temperature)
                         for _ in range(num_prior_samples)],
                        dim=1,
                    )  # [B, S, num_z, D]
                # Forward every prior sample: sample 0 is the recon eval z, and
                # all S feed the CRPS estimate (mirrors inference sampling).
                pred_samples = []
                target = None
                for s in range(num_prior_samples):
                    predicted, target, _, _, _ = model(
                        graph, add_noise=False, use_posterior=False,
                        fixed_z=z_cloud[:, s],
                    )
                    pred_samples.append(predicted.float())
                predicted = pred_samples[0]
                errors = _recon_errors(predicted, target, config.get('recon_loss', 'huber'))

                # Posterior cloud sample (encoder sees the true target y).
                if vae_encoder is not None and getattr(graph, 'y', None) is not None:
                    batch = getattr(graph, 'batch', None)
                    if batch is None:
                        batch = torch.zeros(graph.x.shape[0], dtype=torch.long,
                                            device=graph.x.device)
                    x_in = graph.x if vae_encoder.graph_aware else None
                    _, mu_q, logvar_q = vae_encoder(
                        graph.y, graph.edge_index, graph.edge_attr, batch, x=x_in,
                    )
                    mu_q_all.append(mu_q.float().cpu())
                    var_q_all.append(logvar_q.float().exp().cpu())

                if is_fm_prior:
                    # Per-graph prior variance from the sample cloud [B, S, num_z, D].
                    prior_var_all.append(
                        z_cloud.float().var(dim=1, unbiased=False).cpu())
                else:
                    # Prior mixture total variance per dim:
                    #   Var = Σ_k π_k (σ_k² + μ_k²) − (Σ_k π_k μ_k)²
                    pi = torch.softmax(prior_params['logits'].float(), dim=-1).unsqueeze(-1)
                    mu_p = prior_params['mu'].float()
                    var_p = (2.0 * prior_params['log_std'].float()).exp()
                    if prior_params.get('cov_factor') is not None:
                        # per-dim marginal variance: diag(L L^T) + diag part
                        var_p = var_p + prior_params['cov_factor'].float().pow(2).sum(dim=-1)
                    mean_mix = (pi * mu_p).sum(dim=-2)
                    var_mix = (pi * (var_p + mu_p.pow(2))).sum(dim=-2) - mean_mix.pow(2)
                    prior_var_all.append(var_mix.clamp(min=0).cpu())

            _, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            total_loss_sum += batch_loss_sum
            total_loss_count += batch_loss_count

            crps_elem = _crps_from_samples(torch.stack(pred_samples, dim=0), target.float())
            _, crps_sum, crps_count = _loss_from_errors(crps_elem, loss_weights)
            total_crps_sum += crps_sum
            total_crps_count += crps_count

            if _track_tail:
                try:
                    if predicted.shape[1] > _zc:
                        b = (graph.batch if getattr(graph, 'batch', None) is not None
                             else torch.zeros(predicted.shape[0], dtype=torch.long, device=device))
                        zg = predicted[:, _zc].float() * _st[_zc] + _dm[_zc]
                        amp_gen_all.append((_scatter(zg, b, dim=0, reduce='max')
                                            - _scatter(zg, b, dim=0, reduce='min')).cpu())
                        if getattr(graph, 'y', None) is not None:
                            zt = target[:, _zc].float() * _st[_zc] + _dm[_zc]
                            amp_gt_all.append((_scatter(zt, b, dim=0, reduce='max')
                                               - _scatter(zt, b, dim=0, reduce='min')).cpu())
                except Exception:
                    _track_tail = False

            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'rec': f'{batch_loss_sum / max(batch_loss_count, 1):.2e}',
                              'mem': f'{mem_gb:.1f}GB'})

    result = {
        'mean': total_loss_sum / max(total_loss_count, 1),
        'sum': total_loss_sum,
        'count': total_loss_count,
        'crps': total_crps_sum / max(total_crps_count, 1),
    }

    if mu_q_all and prior_var_all:
        mu_q = torch.cat(mu_q_all, dim=0)        # [N, num_z, D]
        var_q = torch.cat(var_q_all, dim=0)      # [N, num_z, D]
        prior_var = torch.cat(prior_var_all, dim=0)
        # Aggregate posterior variance per dim: Var(μ_q) + E[σ_q²]
        posterior_var = mu_q.var(dim=0, unbiased=False) + var_q.mean(dim=0)  # [num_z, D]
        posterior_std = posterior_var.mean(dim=-1).sqrt()                    # [num_z]
        prior_std = prior_var.mean(dim=0).mean(dim=-1).sqrt()                # [num_z]
        ratio = prior_std / posterior_std.clamp(min=1e-8)
        result['posterior_std'] = [round(float(v), 4) for v in posterior_std]
        result['prior_std'] = [round(float(v), 4) for v in prior_std]
        result['spread_ratio'] = [round(float(v), 4) for v in ratio]
        ratio_str = ', '.join(
            f"slot{i}: q={result['posterior_std'][i]:.3f} p={result['prior_std'][i]:.3f} "
            f"ratio={result['spread_ratio'][i]:.2f}"
            for i in range(len(result['spread_ratio']))
        )
        tqdm.tqdm.write(f"  [PriorDiag] {ratio_str}")
        if min(result['spread_ratio']) < 0.5:
            tqdm.tqdm.write("  [PriorDiag] ⚠️  prior is narrower than half the posterior "
                            "cloud — generated spread will be too small.")

    if amp_gen_all:
        try:
            ag = torch.cat(amp_gen_all)
            gp99, gmax = float(torch.quantile(ag, 0.99)), float(ag.max())
            gkurt = float((ag - ag.mean()).pow(4).mean() / (ag.std().pow(4) + 1e-12) - 3.0)
            result['gen_amp_p99'] = round(gp99, 2)
            result['gen_amp_max'] = round(gmax, 2)
            result['gen_amp_kurt'] = round(gkurt, 2)
            line = f"  [PriorTail] gen amp  p99={gp99:.3e}  max={gmax:.3e}  kurt={gkurt:+.2f}"
            if amp_gt_all:
                tp99 = float(torch.quantile(torch.cat(amp_gt_all), 0.99))
                cov = gp99 / tp99 if tp99 > 1e-8 else float('nan')
                result['amp_p99_cov'] = round(cov, 3)
                line += f"  p99_cov={cov:.2f}  (1.0 = gen reaches GT extremes)"
            tqdm.tqdm.write(line)
        except Exception:
            pass

    return result


def run_periodic_test(model, test_loader, device, config, epoch, train_dataset):
    """Test-set evaluation plus optional train-set reconstruction visualization.

    Shared by the single-GPU and DDP launchers at `test_interval` cadence.
    Returns the test loss.
    """
    start = time.time()
    test_loss = test_model(model, test_loader, device, config, epoch, train_dataset)
    print(f"  Test loss: {test_loss:.2e} ({time.time() - start:.1f}s)")

    if config.get('display_trainset', True):
        viz_indices = config.get('test_batch_idx', [0, 1, 2, 3, 4, 5, 6, 7])
        viz_indices = [i for i in viz_indices if i < len(train_dataset)]
        if viz_indices:
            viz_loader = DataLoader(
                Subset(train_dataset, viz_indices),
                batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
            )
            viz_config = dict(config)
            viz_config['test_batch_idx'] = list(range(len(viz_indices)))
            viz_loss = test_model(model, viz_loader, device, viz_config, epoch,
                                  train_dataset, output_prefix='train')
            print(f"  Train reconstruction loss: {viz_loss:.2e}")
    return test_loss


def _scalar_attr(graph, name):
    """Extract a scalar graph attribute that may be a tensor, list, or scalar."""
    value = getattr(graph, name, None)
    if value is None:
        return None
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, '__getitem__') and len(value) > 0:
        return int(value[0])
    return int(value)


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    model.eval()

    loss_weights = _build_loss_weights(config, device)

    # Use GPU for triangle reconstruction if available
    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else (device != 'cpu')
    mesh_device = device if use_gpu else 'cpu'

    # Cache reconstructed faces by sample_id (topology is constant across timesteps)
    faces_cache = {}

    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    # Cap test batches to avoid NCCL timeout in DDP (test runs only on rank 0;
    # other ranks wait at a barrier whose timeout is the process-group timeout).
    total_test = len(dataloader)
    max_test_batches = int(config.get('test_max_batches', 200))
    effective_total = min(max_test_batches, total_test)
    if effective_total < total_test:
        print(f"  Test: evaluating {effective_total}/{total_test} samples "
              f"(set test_max_batches in config to change)")

    # Get denormalization parameters from dataset
    delta_mean = None
    delta_std = None
    if dataset is not None:
        delta_mean = dataset.delta_mean
        delta_std = dataset.delta_std
        if delta_mean is not None and delta_std is not None:
            print(f"Using denormalization: delta_mean={delta_mean}, delta_std={delta_std}")

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0

        # Collect plot data, then render sequentially at the end
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break

            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target, _, _, _ = model(graph, use_posterior=True)
                errors = _recon_errors(predicted, target, config.get('recon_loss', 'huber'))
                loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss_sum += batch_loss_sum
            total_loss_count += batch_loss_count

            # Save results with GPU-accelerated mesh reconstruction
            if batch_idx in config.get('test_batch_idx', [0, 1, 2, 3]):
                gpu_ids = str(config.get('gpu_ids'))

                sample_id = _scalar_attr(graph, 'sample_id')
                time_idx = _scalar_attr(graph, 'time_idx')

                if sample_id is not None and time_idx is not None:
                    filename = f'sample{sample_id}_t{time_idx}'
                elif sample_id is not None:
                    filename = f'sample{sample_id}'
                else:
                    filename = f'batch{batch_idx}'

                output_path = f'outputs/{output_prefix}/{gpu_ids}/{str(epoch)}/{filename}.h5'

                # Convert to numpy
                predicted_np = predicted.float().cpu().numpy() if hasattr(predicted, 'cpu') else predicted
                target_np = target.float().cpu().numpy() if hasattr(target, 'cpu') else target

                # DENORMALIZE: Convert normalized deltas to actual physical deltas
                if delta_mean is not None and delta_std is not None:
                    predicted_denorm = predicted_np * delta_std + delta_mean
                    target_denorm = target_np * delta_std + delta_mean
                else:
                    predicted_denorm = predicted_np
                    target_denorm = target_np

                # Look up or reconstruct faces for this sample
                cached_faces = faces_cache.get(sample_id)
                if cached_faces is None and sample_id is not None:
                    if use_gpu and torch.cuda.is_available():
                        edge_index_gpu = graph.edge_index.to(mesh_device)
                        cached_faces = edges_to_triangles_gpu(edge_index_gpu, device=mesh_device)
                    else:
                        ei_np = (graph.edge_index.cpu().numpy()
                                 if hasattr(graph.edge_index, 'cpu')
                                 else np.array(graph.edge_index))
                        cached_faces = edges_to_triangles_optimized(ei_np)
                    faces_cache[sample_id] = cached_faces

                # Save HDF5 and collect plot data
                display_testset = config.get('display_testset', True)
                plot_feature_idx = config.get('plot_feature_idx', -1)
                plot_data = save_inference_results_fast(
                    output_path, graph,
                    predicted_norm=predicted_np, target_norm=target_np,
                    predicted_denorm=predicted_denorm, target_denorm=target_denorm,
                    skip_visualization=not display_testset,
                    device=mesh_device,
                    feature_idx=plot_feature_idx,
                    precomputed_faces=cached_faces,
                )

                if plot_data:
                    plot_data_queue.append(plot_data)

        # Render all collected visualizations sequentially (PyVista is fast enough)
        if plot_data_queue:
            print(f"\nRendering {len(plot_data_queue)} visualizations...")
            failed = 0
            for pd in plot_data_queue:
                if not render_plot_data(pd):
                    failed += 1
            if failed:
                print(f"Visualization done with {failed}/{len(plot_data_queue)} failures.")
            else:
                print("All visualizations complete!")

    return total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
