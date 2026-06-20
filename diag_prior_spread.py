"""Diagnostic: is the generated under-dispersion a PRIOR problem (fixable by a
better prior) or a CEILING problem (encoder/decoder)?




python diag_prior_spread.py --config _b8_all_warpage_input/config_infer3_main.txt \
    --gt_dataset dataset/infer_b8_main.h5 --gt_mu 701 --gt_sigma 280 --prior_only

# 대조군 (sec는 rollout 0.86과 일치해야 정상)
python diag_prior_spread.py --config _b8_all_warpage_input/config_infer3_sec.txt \
    --gt_dataset dataset/infer_b8_secondary.h5 --gt_mu 749 --gt_sigma 360 --prior_only




"""
import argparse
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from general_modules.load_config import load_config
from general_modules.data_loader import load_data
from model.MeshGraphNets import MeshGraphNets
from model.conditional_prior import sample_from_mixture
from training_profiles.training_loop import _move_graph_to_device

# Output channel order is (x_disp, y_disp, z_disp); z_disp is index 2.
Z_DISP_OUT_IDX = 2


def _amplitude(predicted_norm, delta_mean, delta_std):
    """max-min of the denormalized z_disp field over nodes (one scalar)."""
    pred = predicted_norm.detach().float().cpu().numpy()      # [N, output_dim]
    delta = pred * delta_std + delta_mean                     # denormalize
    z = delta[:, Z_DISP_OUT_IDX]                              # [N]
    return float(z.max() - z.min())


def _get_batch(graph):
    batch = getattr(graph, 'batch', None)
    if batch is None:
        batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
    return batch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Infer config (for model arch + modelpath).')
    ap.add_argument('--gt_dataset', required=True, help='GT HDF5 with true fields (has y), one part type.')
    ap.add_argument('--n_samples', type=int, default=2000, help='# z samples for prior/fullcov decode.')
    ap.add_argument('--max_objects', type=int, default=300, help='# objects for the posterior/mu_q pass.')
    ap.add_argument('--temperature', type=float, default=1.0, help='prior sampling temperature.')
    ap.add_argument('--gt_mu', type=float, default=None, help='(optional) GT mean for the verdict line.')
    ap.add_argument('--gt_sigma', type=float, default=None, help='(optional) GT sigma for the verdict line.')
    ap.add_argument('--prior_only', action='store_true',
                    help='Skip posterior/fullcov/spectrum (which need a real target y); run only '
                         'the prior passes. Use on initial-condition datasets like infer_b8_*.h5.')
    args = ap.parse_args()

    config = load_config(args.config)
    gpu_ids = config.get('gpu_ids', [0])
    gpu = gpu_ids[0] if isinstance(gpu_ids, (list, tuple)) else gpu_ids
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- checkpoint: weights + normalization + model_config (mirror rollout) ---
    ckpt = torch.load(config['modelpath'], map_location=device, weights_only=False)
    norm = ckpt['normalization']
    delta_mean = np.asarray(norm['delta_mean'], dtype=np.float64)
    delta_std = np.asarray(norm['delta_std'], dtype=np.float64)
    if 'model_config' in ckpt:
        for k, v in ckpt['model_config'].items():
            config[k] = v

    # --- dataset (GT fields -> graphs with y), normalized with the CHECKPOINT stats ---
    config['dataset_dir'] = args.gt_dataset
    config['augment_geometry'] = False
    dataset = load_data(config)
    # Override the dataset's freshly-computed normalization with the checkpoint's,
    # so the encoder sees exactly training-consistent inputs.
    dataset.node_mean = np.asarray(norm['node_mean']); dataset.node_std = np.asarray(norm['node_std'])
    dataset.edge_mean = np.asarray(norm['edge_mean']); dataset.edge_std = np.asarray(norm['edge_std'])
    dataset.delta_mean = delta_mean.astype(np.float32); dataset.delta_std = delta_std.astype(np.float32)
    n_obj = len(dataset)
    print(f"Objects in {args.gt_dataset}: {n_obj}")
    if getattr(dataset, 'num_timesteps', 1) != 1:
        print(f"  WARNING: dataset.num_timesteps={dataset.num_timesteps} (expected 1 for static "
              f"warpage). y is then state_t+1 - state_t, and the amplitude comparison may not "
              f"match compare_histograms.py. Proceed only if this is the intended setup.")
    # PyG DataLoader adds .batch and collates the multiscale hierarchy exactly as
    # validation does — raw dataset[i] would lack .batch and break pooling.
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- model ---
    model = MeshGraphNets(config, str(device)).to(device)
    if 'ema_state_dict' in ckpt:
        sd = {k[len('module.'):]: v for k, v in ckpt['ema_state_dict'].items() if k.startswith('module.')}
        model.load_state_dict(sd); print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model_state_dict']); print("Loaded training weights")
    model.eval()

    vae_encoder = getattr(getattr(model, 'model', None), 'vae_encoder', None)
    prior = getattr(model, 'prior', None)
    if vae_encoder is None:
        raise RuntimeError("No vae_encoder on model — is this a VAE checkpoint?")
    if prior is None:
        raise RuntimeError("No conditional prior on model — checkpoint has no learned prior.")
    graph_aware = bool(getattr(vae_encoder, 'graph_aware', False))

    amp_post, mu_list, amp_prior_multi = [], [], []
    ref_graph = None

    # ---------- (a) posterior + (b*) multi-graph prior, in one pass ----------
    # (a)  z = mu_q on each graph (ceiling, collects mu_q)
    # (b*) z ~ p(z|graph) drawn on EACH graph (many graphs x samples) — the
    #      apples-to-apples match to the rollout histogram. The single-graph (b)
    #      below can be unrepresentative if the prior collapses per-graph.
    n_pass = min(args.max_objects, n_obj)
    samples_per_graph = max(1, args.n_samples // n_pass)
    print(f"\n[a] posterior + [b*] multi-graph prior over {n_pass} objects "
          f"({samples_per_graph} prior draw(s)/graph) ...")
    with torch.no_grad():
        for i, graph in enumerate(loader):
            if i >= n_pass:
                break
            graph = _move_graph_to_device(graph, device, config)
            if ref_graph is None:
                ref_graph = graph  # reuse one graph for the 1-graph prior/fullcov decode
            batch = _get_batch(graph)
            if not args.prior_only:
                x_in = graph.x if graph_aware else None
                _, mu_q, _ = vae_encoder(graph.y, graph.edge_index, graph.edge_attr, batch, x=x_in)
                out = model(graph, add_noise=False, use_posterior=False, fixed_z=mu_q)
                amp_post.append(_amplitude(out[0], delta_mean, delta_std))
                mu_list.append(mu_q.squeeze(0).float().cpu().numpy())  # [num_z, D]
            # prior conditioned on THIS graph (mirrors the rollout: many graphs)
            prior_params_g = prior(graph)
            for _ in range(samples_per_graph):
                zp = sample_from_mixture(prior_params_g, temperature=args.temperature)
                outp = model(graph, add_noise=False, use_posterior=False, fixed_z=zp)
                amp_prior_multi.append(_amplitude(outp[0], delta_mean, delta_std))
            if (i + 1) % max(1, n_pass // 10) == 0:
                print(f"  {i + 1}/{n_pass}")

    # ---------- (b) prior pass: z ~ p(z|graph) on ONE reference graph ----------
    print(f"\n[b] prior pass: {args.n_samples} samples on one reference graph ...")
    amp_prior = []
    with torch.no_grad():
        prior_params = prior(ref_graph)
        for j in range(args.n_samples):
            z = sample_from_mixture(prior_params, temperature=args.temperature)
            out = model(ref_graph, add_noise=False, use_posterior=False, fixed_z=z)
            amp_prior.append(_amplitude(out[0], delta_mean, delta_std))

    # ---------- (c) fullcov pass + spectrum (need mu_q; skipped in --prior_only) ----------
    amp_fullcov = []
    if not args.prior_only and mu_list:
        try:
            mus = np.stack(mu_list, axis=0)                  # [M, num_z, D]
            M, num_z, D = mus.shape
            flat = mus.reshape(M, -1).astype(np.float64)     # [M, num_z*D]
            if not np.all(np.isfinite(flat)):
                raise np.linalg.LinAlgError("mu_q contains NaN/Inf")
            mean_v = flat.mean(axis=0)
            cov_raw = np.atleast_2d(np.cov(flat, rowvar=False))

            # Eigen-spectrum of the mu_q covariance: how many latent directions
            # carry the spread. This sizes prior_cov_rank.
            evals = np.clip(np.sort(np.linalg.eigvalsh(cov_raw))[::-1], 0, None)
            cum = np.cumsum(evals) / max(evals.sum(), 1e-12)
            rank_for = lambda frac: int(np.searchsorted(cum, frac) + 1)
            print(f"\n[spectrum] mu_q covariance ({flat.shape[1]} dims) "
                  f"top eigvals = {np.array2string(evals[:8], precision=3)}")
            print(f"[spectrum] directions for 90 / 95 / 99% variance: "
                  f"{rank_for(0.90)} / {rank_for(0.95)} / {rank_for(0.99)}  "
                  f"=> suggested prior_cov_rank ~ {rank_for(0.95)}")

            print(f"\n[c] fullcov pass: {args.n_samples} samples from full-cov Gaussian over mu_q ...")
            cov = cov_raw + 1e-4 * np.eye(flat.shape[1])     # ridge for PD sampling
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(cov)
                L = V @ np.diag(np.sqrt(np.clip(w, 1e-8, None)))
            with torch.no_grad():
                for j in range(args.n_samples):
                    zf = mean_v + L @ np.random.randn(flat.shape[1])
                    z = torch.from_numpy(zf.reshape(1, num_z, D).astype(np.float32)).to(device)
                    out = model(ref_graph, add_noise=False, use_posterior=False, fixed_z=z)
                    amp_fullcov.append(_amplitude(out[0], delta_mean, delta_std))
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"\n[c/spectrum] SKIPPED: {e}\n  (mu_q degenerate/non-finite — expected on "
                  f"initial-condition datasets without real targets; (b*) prior is still valid.)")
            amp_fullcov = []

    # ---------- report ----------
    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        a = a[np.isfinite(a)]               # drop NaN/Inf decodes
        if a.size == 0:
            return None
        return dict(mean=a.mean(), std=a.std(), p99=float(np.quantile(a, 0.99)),
                    mx=a.max(), n=a.size)

    print("\n" + "=" * 78)
    print(f"GT dataset: {args.gt_dataset}")
    if args.gt_mu is not None and args.gt_sigma is not None:
        print(f"  GT           mu={args.gt_mu:.3e}  sigma={args.gt_sigma:.3e}")
    for name, a in [("(a)  posterior     ", amp_post),
                    ("(b*) prior  Ngraphs", amp_prior_multi),
                    ("(b)  prior  1graph ", amp_prior),
                    ("(c)  fullcov 1graph", amp_fullcov)]:
        s = stats(a)
        if s is None:
            continue
        frac = (f"  sd/GT={s['std'] / args.gt_sigma:.2f}" if args.gt_sigma else "")
        print(f"  {name} mu={s['mean']:.3e} sd={s['std']:.3e} "
              f"p99={s['p99']:.3e} max={s['mx']:.3e}{frac}  (n={s['n']})")
    print("=" * 78)
    print("Read:  (b*) is the apples-to-apples match to the rollout histogram.\n"
          "       (b*)<<(b)  -> prior COLLAPSES per-graph; the 1-graph (b) was a\n"
          "                     lucky non-collapsed graph. Fix the prior, not the decoder.\n"
          "       (b*)~(b)~GT but rollout<<  -> the infer dataset graphs differ from eval.\n"
          "       (a)<<GT    -> ceiling (encoder/decoder); only then alpha/mmd/loss matter.")


if __name__ == '__main__':
    main()
