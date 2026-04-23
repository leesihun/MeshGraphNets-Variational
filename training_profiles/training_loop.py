import os
import tqdm
import torch
import numpy as np
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from general_modules.mesh_utils_fast import (
    save_inference_results_fast,
    render_plot_data,
    edges_to_triangles_gpu,
    edges_to_triangles_optimized,
)


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

def save_debug_batch(epoch, batch_idx, graph, predicted, target, log_dir):
    """Save actual input/output values to debug file for inspection."""
    try:
        debug_file = os.path.join(log_dir, f'debug_epoch{epoch:03d}_batch{batch_idx:03d}.npz')

        # Convert to numpy
        x_np = graph.x.cpu().numpy()
        y_np = target.cpu().numpy()
        pred_np = predicted.cpu().numpy()

        np.savez(debug_file,
                 x=x_np,
                 y=y_np,
                 pred=pred_np,
                 x_mean=x_np.mean(axis=0),
                 x_std=x_np.std(axis=0),
                 y_mean=y_np.mean(axis=0),
                 y_std=y_np.std(axis=0),
                 pred_mean=pred_np.mean(axis=0),
                 pred_std=pred_np.std(axis=0))

        tqdm.tqdm.write(f"  Saved debug data to {debug_file}")
    except Exception as e:
        tqdm.tqdm.write(f"  Warning: Could not save debug data: {e}")

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
        mp_enc  = config.get('vae_mp_layers', 2)
        lam     = config.get('lambda_mmd', 100.0)
        b_aux   = config.get('beta_aux', 0.1)
        alpha   = config.get('alpha_recon', 1.0)
        print(f"VAE (MMD): ENABLED (z_dim={z_dim}, vae_mp_layers={mp_enc}, "
              f"lambda_mmd={lam}, beta_aux={b_aux}, alpha_recon={alpha})")
    else:
        print("VAE: disabled")

    if config.get('use_multiscale', False):
        _L = int(config.get('multiscale_levels', 1))
        _mp = config.get('mp_per_level', None)
        if _mp is None:
            _mp = [int(config.get('fine_mp_pre', 5)), int(config.get('coarse_mp_num', 5)), int(config.get('fine_mp_post', 5))]
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


def train_epoch(model, dataloader, optimizer, device, config, epoch, scheduler=None, ema_model=None):
    model.train()
    total_loss_sum = 0.0
    total_loss_count = 0
    total_opt_loss_sum = 0.0  # actual optimization loss (recon + lambda_mmd*mmd when VAE)
    total_grad_norm = 0.0
    total_mmd_sum = 0.0
    total_aux_sum = 0.0
    mmd_count = 0
    num_batches = 0
    num_steps = 0  # number of optimizer steps taken

    verbose = config.get('verbose', False)
    monitor_gradients = config.get('monitor_gradients', False)
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    use_compile = config.get('use_compile', False)
    amp_dtype = torch.bfloat16

    # VAE config (MMD objective: InfoVAE-style aggregate-posterior matching)
    use_vae = config.get('use_vae', False)
    alpha_recon = float(config.get('alpha_recon', 1.0))
    lambda_mmd = float(config.get('lambda_mmd', 100.0))
    beta_aux = float(config.get('beta_aux', 1.0))

    # Gradient accumulation: 0 = full epoch (1 step/epoch), 1 = per-batch (default), N = every N batches
    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    optimizer.zero_grad(set_to_none=True)
    grad_norm = torch.tensor(0.0)

    pbar = tqdm.tqdm(dataloader)
    for batch_idx, graph in enumerate(pbar):
        # DEBUG: disabled when use_compile=True (.item() causes graph breaks)
        debug_internal = (not use_compile) and (batch_idx == 0 and (epoch < 5 or epoch % 10 == 0))

        graph = graph.to(device)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            predicted_acc, target_acc, mmd_loss_val, aux_loss_val = model(graph, debug=debug_internal)

            if batch_idx == 0 and verbose:
                tqdm.tqdm.write(f"\n=== DEBUG Epoch {epoch} Batch 0 ===")
                tqdm.tqdm.write(f"  Pred:   mean={predicted_acc.mean().item():.6f}, std={predicted_acc.std().item():.6f}, min={predicted_acc.min().item():.4f}, max={predicted_acc.max().item():.4f}")
                tqdm.tqdm.write(f"  Target: mean={target_acc.mean().item():.6f}, std={target_acc.std().item():.6f}, min={target_acc.min().item():.4f}, max={target_acc.max().item():.4f}")
                if predicted_acc.std().item() < 0.01:
                    tqdm.tqdm.write(f"  *** WARNING: Pred std < 0.01 - model outputting near-constant values! ***")
                if epoch >= 5:
                    log_dir = config.get('log_dir', '.')
                    save_debug_batch(epoch, batch_idx, graph, predicted_acc, target_acc, log_dir)

            errors = torch.nn.functional.huber_loss(predicted_acc, target_acc, reduction='none', delta=1.0)
            recon_loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            if use_vae:
                loss = alpha_recon * recon_loss + lambda_mmd * mmd_loss_val + beta_aux * aux_loss_val
            else:
                loss = recon_loss
            # Scale loss so accumulated gradients equal the mean within each accumulation window.
            scaled_loss = loss / _accum_window_size(batch_idx, total_batches, actual_accum)

        scaled_loss.backward()

        # Per-feature loss breakdown for diagnostics
        if batch_idx == 0 and epoch % 10 == 0 and verbose:
            per_feature_loss_mean = torch.mean(errors, dim=0)
            per_feature_loss_max = torch.max(errors, dim=0)[0]
            per_feature_loss_min = torch.min(errors, dim=0)[0]
            per_feature_loss_std = torch.std(errors, dim=0)
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature MSE Loss (Epoch {epoch}, Batch {batch_idx}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(per_feature_loss_mean)]):
                tqdm.tqdm.write(f"  {feat_name}: mean={per_feature_loss_mean[feat_idx].item():.2e}, max={per_feature_loss_max[feat_idx].item():.2e}, min={per_feature_loss_min[feat_idx].item():.2e}, std={per_feature_loss_std[feat_idx].item():.2e}")

        loss_val = batch_loss_sum / batch_loss_count
        total_loss_sum += batch_loss_sum
        total_loss_count += batch_loss_count
        total_opt_loss_sum += loss.item() * batch_loss_count
        num_batches += 1
        if use_vae:
            mmd_scalar = mmd_loss_val.item() if hasattr(mmd_loss_val, 'item') else float(mmd_loss_val)
            aux_scalar = aux_loss_val.item() if hasattr(aux_loss_val, 'item') else float(aux_loss_val)
            total_mmd_sum += mmd_scalar
            total_aux_sum += aux_scalar
            mmd_count += 1

        # Step optimizer at end of accumulation window or final batch
        is_last_batch = (batch_idx == total_batches - 1)
        if (batch_idx + 1) % actual_accum == 0 or is_last_batch:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            if monitor_gradients:
                total_grad_norm += grad_norm.item()
                num_steps += 1

                if verbose:
                    layer_grad_stats = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_mean = param.grad.abs().mean().item()
                            grad_max = param.grad.abs().max().item()
                            if grad_mean > 1e-10:
                                layer_grad_stats.append(f"{name}: mean={grad_mean:.2e}, max={grad_max:.2e}")
                    if layer_grad_stats:
                        tqdm.tqdm.write(f"\n=== Gradient Stats (Step after batch {batch_idx}) ===")
                        tqdm.tqdm.write(f"Total grad norm: {grad_norm.item():.2e}")
                        tqdm.tqdm.write("\nPer-layer gradients (top 5):")
                        for stat in layer_grad_stats[:5]:
                            tqdm.tqdm.write(f"  {stat}")
                        tqdm.tqdm.write("")

            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # Update progress bar every 10 batches to reduce memory query overhead
        if batch_idx % 10 == 0:
            mem_gb = torch.cuda.memory_allocated() / 1e9
            postfix = {'rec': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'}
            if use_vae:
                mmd_val = mmd_loss_val.item() if hasattr(mmd_loss_val, 'item') else float(mmd_loss_val)
                aux_val = aux_loss_val.item() if hasattr(aux_loss_val, 'item') else float(aux_loss_val)
                postfix['mmd'] = f'{mmd_val:.2e}'
                postfix['aux'] = f'{aux_val:.2e}'
                postfix['λ']   = f'{lambda_mmd:.1e}'
                postfix['total'] = f'{loss.item():.2e}'
            if monitor_gradients:
                postfix['grad'] = f'{grad_norm.item():.2e}'
            pbar.set_postfix(postfix)

    avg_grad_norm = total_grad_norm / num_steps if num_steps > 0 else 0.0

    # Print gradient summary for the epoch
    if monitor_gradients and num_steps > 0:
        tqdm.tqdm.write(f"Epoch {epoch} avg gradient norm: {avg_grad_norm:.2e} ({num_steps} optimizer steps)")
        if avg_grad_norm < 1e-6:
            tqdm.tqdm.write("  ⚠️  WARNING: Very small gradients detected (< 1e-6) - possible vanishing gradient problem!")
        elif avg_grad_norm > 1e2:
            tqdm.tqdm.write("  ⚠️  WARNING: Very large gradients detected (> 100) - possible exploding gradient problem!")

    result = {
        'mean': total_loss_sum / total_loss_count,
        'total_mean': total_opt_loss_sum / total_loss_count,
        'sum': total_loss_sum,
        'count': total_loss_count,
    }
    if use_vae and mmd_count > 0:
        result['mmd_mean'] = total_mmd_sum / mmd_count
        result['aux_mean'] = total_aux_sum / mmd_count
    return result

def _eval_forward_errors(model, graph, use_amp, amp_dtype, use_posterior):
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        predicted, target, mmd_loss_val, _ = model(
            graph,
            add_noise=False,
            use_posterior=use_posterior,
        )
        errors = torch.nn.functional.huber_loss(
            predicted, target, reduction='none', delta=1.0
        )
    return errors, mmd_loss_val


def _evaluate_epoch(model, dataloader, device, config, epoch=0, *,
                    use_posterior=None, num_prior_samples=1, progress_name='Validation'):
    if num_prior_samples < 1:
        raise ValueError("num_prior_samples must be >= 1")
    if use_posterior is not False and num_prior_samples != 1:
        raise ValueError("num_prior_samples > 1 is only valid for prior evaluation")

    model.eval()

    verbose = config.get('verbose', False)
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    use_vae = config.get('use_vae', False)
    alpha_recon = float(config.get('alpha_recon', 1.0))
    lambda_mmd = float(config.get('lambda_mmd', 100.0))

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0
        total_opt_loss_sum = 0.0
        total_mmd_sum = 0.0
        mmd_count = 0

        # Accumulate per-feature losses across all batches
        accumulated_per_feature_loss = None
        accumulated_per_feature_count = 0

        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for batch_idx, graph in enumerate(pbar):
            # Memory tracking for first 3 validation batches
            if batch_idx < 3 and verbose:
                mem_before = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"\n=== {progress_name} Batch {batch_idx} ===")
                tqdm.tqdm.write(f"Before: {mem_before:.2f}GB")

            graph = graph.to(device)

            if use_vae and use_posterior is False and num_prior_samples > 1:
                errors_sum = None
                for _ in range(num_prior_samples):
                    sample_errors, _ = _eval_forward_errors(
                        model, graph, use_amp, amp_dtype, use_posterior=False
                    )
                    if errors_sum is None:
                        errors_sum = sample_errors
                    else:
                        errors_sum += sample_errors
                errors = errors_sum / num_prior_samples
                mmd_loss_val = errors.new_zeros(())
            else:
                errors, mmd_loss_val = _eval_forward_errors(
                    model, graph, use_amp, amp_dtype, use_posterior=use_posterior
                )

            recon_loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            if use_vae:
                opt_loss = alpha_recon * recon_loss + lambda_mmd * mmd_loss_val
            else:
                opt_loss = recon_loss

            # Accumulate per-feature losses
            per_feature_loss = torch.sum(errors, dim=0)
            if accumulated_per_feature_loss is None:
                accumulated_per_feature_loss = per_feature_loss
            else:
                accumulated_per_feature_loss += per_feature_loss
            accumulated_per_feature_count += errors.shape[0]

            if batch_idx < 3 and verbose:
                mem_after = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"After: {mem_after:.2f}GB (+{mem_after-mem_before:.2f}GB)")
                tqdm.tqdm.write(f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9
            postfix = {'rec': f'{recon_loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'}
            if use_vae:
                mmd_scalar = mmd_loss_val.item() if hasattr(mmd_loss_val, 'item') else float(mmd_loss_val)
                postfix['mmd'] = f'{mmd_scalar:.2e}'
                postfix['total'] = f'{opt_loss.item():.2e}'
            pbar.set_postfix(postfix)

            total_loss_sum += batch_loss_sum
            total_loss_count += batch_loss_count
            total_opt_loss_sum += opt_loss.item() * batch_loss_count
            if use_vae:
                mmd_scalar = mmd_loss_val.item() if hasattr(mmd_loss_val, 'item') else float(mmd_loss_val)
                total_mmd_sum += mmd_scalar
                mmd_count += 1

        # Print per-feature validation loss breakdown
        if verbose and accumulated_per_feature_loss is not None and accumulated_per_feature_count > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / accumulated_per_feature_count
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature {progress_name} Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                tqdm.tqdm.write(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            tqdm.tqdm.write("")

    result = {
        'mean': total_loss_sum / total_loss_count,
        'total_mean': total_opt_loss_sum / total_loss_count,
        'sum': total_loss_sum,
        'count': total_loss_count,
    }
    if use_vae and mmd_count > 0:
        result['mmd_mean'] = total_mmd_sum / mmd_count
    return result


def validate_epoch(model, dataloader, device, config, epoch=0):
    return _evaluate_epoch(
        model,
        dataloader,
        device,
        config,
        epoch,
        use_posterior=None,
        num_prior_samples=1,
        progress_name='Validation',
    )


def evaluate_vae_posterior_epoch(model, dataloader, device, config, epoch=0, progress_name='ValidationQ'):
    return _evaluate_epoch(
        model,
        dataloader,
        device,
        config,
        epoch,
        use_posterior=True,
        num_prior_samples=1,
        progress_name=progress_name,
    )


def evaluate_vae_prior_epoch(model, dataloader, device, config, epoch=0, *,
                             num_prior_samples, progress_name=None):
    if progress_name is None:
        progress_name = f'ValidationPrior@{num_prior_samples}'
    return _evaluate_epoch(
        model,
        dataloader,
        device,
        config,
        epoch,
        use_posterior=False,
        num_prior_samples=num_prior_samples,
        progress_name=progress_name,
    )


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    model.eval()

    verbose = config.get('verbose', False)
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
        num_batches = 0

        # Accumulate per-feature losses across all test batches
        accumulated_per_feature_loss = None
        accumulated_per_feature_count = 0

        # Collect plot data for parallel processing
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break

            graph = graph.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target, _, _ = model(graph, use_posterior=True)
                errors = torch.nn.functional.huber_loss(predicted, target, reduction='none', delta=1.0)
                loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            # Accumulate per-feature losses
            per_feature_loss = torch.sum(errors, dim=0)
            if accumulated_per_feature_loss is None:
                accumulated_per_feature_loss = per_feature_loss
            else:
                accumulated_per_feature_loss += per_feature_loss
            accumulated_per_feature_count += errors.shape[0]

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss_sum += batch_loss_sum
            total_loss_count += batch_loss_count
            num_batches += 1

            # Save results with GPU-accelerated mesh reconstruction
            if batch_idx in config.get('test_batch_idx', [0, 1, 2, 3]):
                gpu_ids = str(config.get('gpu_ids'))

                # Build filename with sample_id and time_idx for clarity
                # Extract sample_id and time_idx from graph (handle tensor or scalar)
                sample_id = None
                time_idx = None
                if hasattr(graph, 'sample_id') and graph.sample_id is not None:
                    sid = graph.sample_id
                    if hasattr(sid, 'cpu'):
                        sid = sid.cpu()
                    if hasattr(sid, 'item'):
                        sample_id = sid.item()
                    elif hasattr(sid, '__getitem__') and len(sid) > 0:
                        sample_id = int(sid[0])
                    else:
                        sample_id = int(sid)

                if hasattr(graph, 'time_idx') and graph.time_idx is not None:
                    tid = graph.time_idx
                    if hasattr(tid, 'cpu'):
                        tid = tid.cpu()
                    if hasattr(tid, 'item'):
                        time_idx = tid.item()
                    elif hasattr(tid, '__getitem__') and len(tid) > 0:
                        time_idx = int(tid[0])
                    else:
                        time_idx = int(tid)

                # Build descriptive filename
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
                    # Fallback: use normalized values
                    predicted_denorm = predicted_np
                    target_denorm = target_np

                # Look up or reconstruct faces for this sample
                cached_faces = faces_cache.get(sample_id)
                if cached_faces is None and sample_id is not None:
                    # Reconstruct once and cache for this sample_id
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
            for i, pd in enumerate(plot_data_queue):
                if not render_plot_data(pd):
                    failed += 1
            if failed:
                print(f"Visualization done with {failed}/{len(plot_data_queue)} failures.")
            else:
                print("All visualizations complete!")
        
        # Print per-feature test loss breakdown
        if verbose and accumulated_per_feature_loss is not None and accumulated_per_feature_count > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / accumulated_per_feature_count
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            print(f"\n=== Per-Feature Test MSE Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                print(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            print("")

    return total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0