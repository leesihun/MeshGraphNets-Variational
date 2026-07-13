import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features
from general_modules.world_edges import HAS_TORCH_CLUSTER, compute_world_edges
from model.MeshGraphNets import MeshGraphNets

# Multiscale coarsening (only needed when use_multiscale=True)
try:
    from model.coarsening import MultiscaleData
    from general_modules.multiscale_helpers import (
        attach_coarse_levels_to_graph,
        build_multiscale_hierarchy,
    )
    HAS_COARSENING = True
except ImportError:
    HAS_COARSENING = False


# ---------------------------------------------------------------------------
# z_disp spread-histogram (generated vs ground-truth eval dataset)
#
# Spread metric (one scalar per realization):
#     spread = max(z_disp_at_nodes) - min(z_disp_at_nodes)   at the final timestep.
#
# nodal_data layout is [x, y, z, x_disp, y_disp, z_disp, ...], so the z_disp
# channel is index 5 for both the ground-truth eval datasets and the rollout
# output (whose layout is [x, y, z, <output channels>, part_number]). This keeps
# the inline plot identical to the standalone compare_histograms.py script.
# ---------------------------------------------------------------------------
Z_DISP_CHANNEL = 5


def _format_bytes(num_bytes):
    """Compact binary-size formatter for CUDA memory logs."""
    num_bytes = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0


def _is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _spread_max_minus_min(field_1d):
    """max - min of a 1-D node field (the spread of a single realization)."""
    v = np.asarray(field_1d, dtype=np.float64)
    if v.size == 0:
        return float('nan')
    return float(v.max() - v.min())


def _eval_dataset_spreads(h5_path, channel=Z_DISP_CHANNEL):
    """One ground-truth spread value per sample in the eval HDF5."""
    spreads = []
    with h5py.File(h5_path, 'r') as f:
        if 'data' not in f:
            raise RuntimeError(f"No /data group in {h5_path}")
        for sample_id in f['data'].keys():
            spreads.append(
                _spread_max_minus_min(f[f'data/{sample_id}/nodal_data'][channel, -1, :])
            )
    return np.asarray(spreads, dtype=np.float64)


def _open_in_viewer(path):
    """Best-effort: open a saved image in the OS default viewer (for display)."""
    import sys
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['open', path])
        else:
            import subprocess
            subprocess.Popen(['xdg-open', path])
        return True
    except Exception as exc:  # headless box / no viewer — never fatal
        print(f"  (could not open image viewer: {exc})")
        return False


def _plot_spread_histogram(gt, gen, out_path, eval_path=None, rollout_dir=None,
                           bins=60, clip_quantile=0.0, show=False, dpi=150):
    """Overlay GT vs generated spread histograms; save PNG, optionally display.

    Renders headlessly (Agg) so it works on a display-less box; when show=True
    the saved PNG is additionally opened in the OS default viewer.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gt = np.asarray(gt, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)

    if clip_quantile > 0:
        q = clip_quantile
        lo = float(min(np.quantile(gt, q), np.quantile(gen, q)))
        hi = float(max(np.quantile(gt, 1 - q), np.quantile(gen, 1 - q)))
    else:
        lo = float(min(gt.min(), gen.min()))
        hi = float(max(gt.max(), gen.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1e-8
    bin_edges = np.linspace(lo, hi, bins + 1)

    def _stats(col):
        return (float(col.mean()), float(col.std()), float(col.min()), float(col.max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt, bins=bin_edges, density=True, alpha=0.55,
            label=f"eval dataset (n={gt.size:,})", color="steelblue")
    ax.hist(gen, bins=bin_edges, density=True, alpha=0.55,
            label=f"generated (n={gen.size:,})", color="darkorange")
    g_mu, g_sd, g_lo, g_hi = _stats(gt)
    p_mu, p_sd, p_lo, p_hi = _stats(gen)
    ax.set_title(
        "z_disp spread (max - min) per realization, final timestep\n"
        f"GT  mu={g_mu:.3e}  sigma={g_sd:.3e}  [{g_lo:.3e}, {g_hi:.3e}]\n"
        f"Gen mu={p_mu:.3e}  sigma={p_sd:.3e}  [{p_lo:.3e}, {p_hi:.3e}]"
    )
    ax.set_xlabel("max(z_disp) - min(z_disp)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    if eval_path is not None or rollout_dir is not None:
        fig.suptitle(f"eval:     {eval_path}\nrollouts: {rollout_dir}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved histogram figure: {os.path.abspath(out_path)}")
    if show:
        _open_in_viewer(os.path.abspath(out_path))
    return out_path


class _SampleContext:
    """Static per-scene data shared by every rollout step and z-sample.

    Bundles the mesh (topology, reference geometry, positional features,
    optional node-type one-hot, optional coarsening hierarchy) plus the
    normalization stats needed to build a normalized step graph.
    """

    def __init__(self, config, checkpoint_norm, ref_pos, edge_index, part_ids, device):
        self.device = device
        self.ref_pos = ref_pos                  # [N, 3]
        self.edge_index = edge_index            # [2, 2M] bidirectional
        self.num_nodes = ref_pos.shape[0]

        norm = checkpoint_norm
        self.node_mean, self.node_std = norm['node_mean'], norm['node_std']
        self.edge_mean, self.edge_std = norm['edge_mean'], norm['edge_std']
        self.delta_mean, self.delta_std = norm['delta_mean'], norm['delta_std']
        if 'coarse_edge_means' in norm:
            self.coarse_edge_means = norm['coarse_edge_means']
            self.coarse_edge_stds = norm['coarse_edge_stds']
        else:
            self.coarse_edge_means = [self.edge_mean]
            self.coarse_edge_stds = [self.edge_std]

        # Positional features (geometry + topology, static across steps)
        num_pos_features = int(config.get('positional_features', 0))
        self.pos_features = None
        if num_pos_features > 0:
            self.pos_features = compute_positional_features(
                ref_pos, edge_index, num_pos_features
            )
            print(f"  Positional features: {self.pos_features.shape}")

        # Node-type one-hot (static across steps and z-samples)
        self.node_type_onehot = None
        if config.get('use_node_types') and part_ids is not None:
            node_type_to_idx = norm.get('node_type_to_idx')
            num_node_types = norm.get('num_node_types')
            if node_type_to_idx is not None and num_node_types:
                indices = np.array([node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32)
                onehot = np.zeros((self.num_nodes, num_node_types), dtype=np.float32)
                onehot[np.arange(self.num_nodes), indices] = 1.0
                self.node_type_onehot = onehot

        # World edges
        self.use_world_edges = bool(config.get('use_world_edges'))
        self.world_edge_radius = norm.get('world_edge_radius')
        self.world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
        requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
        self.world_edge_backend = (
            'torch_cluster' if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER
            else 'scipy_kdtree'
        )
        if self.use_world_edges:
            print(f"  World edges: radius={self.world_edge_radius}, backend={self.world_edge_backend}")

        # Coarsening hierarchy (static — same for all rollout steps)
        self.use_multiscale = bool(config.get('use_multiscale', False))
        self.use_coarse_world_edges = bool(config.get('coarse_world_edges', False))
        self.hierarchy = None
        if self.use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
            multiscale_levels = int(config.get('multiscale_levels', 1))
            raw_ct = config.get('coarsening_type', 'bfs')
            if isinstance(raw_ct, list):
                coarsening_types = [str(t).strip().lower() for t in raw_ct]
            else:
                coarsening_types = [str(raw_ct).strip().lower()]
            if len(coarsening_types) == 1 and multiscale_levels > 1:
                coarsening_types = coarsening_types * multiscale_levels
            raw_vc = config.get('voronoi_clusters', None)
            if raw_vc is None:
                voronoi_clusters = [0] * multiscale_levels
            elif isinstance(raw_vc, list):
                voronoi_clusters = [int(v) for v in raw_vc]
            else:
                voronoi_clusters = [int(raw_vc)]
            if len(voronoi_clusters) == 1 and multiscale_levels > 1:
                voronoi_clusters = voronoi_clusters * multiscale_levels

            self.hierarchy = build_multiscale_hierarchy(
                edge_index, self.num_nodes, ref_pos,
                multiscale_levels, coarsening_types, voronoi_clusters,
            )
            current_n = self.num_nodes
            for level, entry in enumerate(self.hierarchy):
                method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
                n_c = entry['n_c']
                print(f"  Coarsening level {level} ({method}): {current_n} → {n_c} nodes "
                      f"({n_c / current_n * 100:.1f}%)")
                current_n = n_c

    def build_step_graph(self, current_state):
        """Build one normalized PyG graph for the given physical state [N, input_dim]."""
        device = self.device
        if self.pos_features is not None:
            x_raw = np.concatenate([current_state, self.pos_features], axis=1)
        else:
            x_raw = current_state
        x_norm = (x_raw - self.node_mean) / self.node_std
        if self.node_type_onehot is not None:
            x_norm = np.concatenate([x_norm, self.node_type_onehot], axis=1)

        deformed_pos = self.ref_pos + current_state[:, :3]
        edge_attr = (compute_edge_attr(self.ref_pos, deformed_pos, self.edge_index)
                     - self.edge_mean) / self.edge_std

        DataClass = MultiscaleData if self.use_multiscale else Data
        graph = DataClass(
            x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
            edge_index=torch.from_numpy(self.edge_index).long().to(device),
            edge_attr=torch.from_numpy(edge_attr.astype(np.float32)).to(device),
            pos=torch.from_numpy(self.ref_pos.astype(np.float32)).to(device),
        )

        if self.use_world_edges and self.world_edge_radius is not None:
            world_ei, world_ea = compute_world_edges(
                self.ref_pos, deformed_pos, self.edge_index,
                radius=self.world_edge_radius,
                max_num_neighbors=self.world_max_num_neighbors,
                backend=self.world_edge_backend,
                device=device,
                edge_mean=self.edge_mean, edge_std=self.edge_std,
            )
            graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
            graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
        else:
            graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            graph.world_edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

        if self.use_multiscale and self.hierarchy is not None:
            world_ei_for_coarse = (
                graph.world_edge_index.cpu().numpy()
                if self.use_world_edges and self.use_coarse_world_edges else None
            )
            attach_coarse_levels_to_graph(
                graph, self.hierarchy, self.ref_pos, deformed_pos,
                self.coarse_edge_means, self.coarse_edge_stds,
                device=device, world_edge_index=world_ei_for_coarse,
            )

        return graph


def _save_rollout_h5(output_path, sample_id, all_states, ctx, part_ids, output_dim,
                     num_steps, model_path, config_filename, rollout_time_s,
                     vae_sample_idx=None):
    """Write one trajectory to HDF5 following DATASET_FORMAT.md.

    all_states: [num_steps + 1, N, output_dim] predicted physical states.
    """
    num_nodes = ctx.num_nodes
    num_save_features = 3 + output_dim + 1  # xyz + outputs + part number

    # nodal_data layout: [x, y, z, <output channels>, part_number]
    nodal_data = np.zeros((num_save_features, num_steps + 1, num_nodes), dtype=np.float32)
    nodal_data[0, :, :] = ctx.ref_pos[:, 0]
    nodal_data[1, :, :] = ctx.ref_pos[:, 1]
    nodal_data[2, :, :] = ctx.ref_pos[:, 2]
    for ch in range(output_dim):
        nodal_data[3 + ch, :, :] = all_states[:, :, ch]
    if part_ids is not None:
        nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]

    # mesh_edge stored unidirectional (first half of the bidirectional index)
    mesh_edge = ctx.edge_index[:, :ctx.edge_index.shape[1] // 2]

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = num_save_features
        f.attrs['num_timesteps'] = num_steps + 1

        sample_grp = f.create_group('data').create_group(str(sample_id))
        sample_grp.create_dataset('nodal_data', data=nodal_data,
                                  compression='gzip', compression_opts=4)
        sample_grp.create_dataset('mesh_edge', data=mesh_edge)

        meta_grp = sample_grp.create_group('metadata')
        meta_grp.attrs['sample_id'] = sample_id
        meta_grp.attrs['num_nodes'] = num_nodes
        meta_grp.attrs['num_edges'] = mesh_edge.shape[1]
        meta_grp.attrs['num_timesteps'] = num_steps + 1
        meta_grp.attrs['model_path'] = model_path
        meta_grp.attrs['config_file'] = config_filename
        meta_grp.attrs['total_rollout_time_s'] = rollout_time_s
        if vae_sample_idx is not None:
            meta_grp.attrs['vae_sample_idx'] = vae_sample_idx

        _all_feature_names = [
            b'x_coord', b'y_coord', b'z_coord',
            b'x_disp(mm)', b'y_disp(mm)', b'z_disp(mm)',
            b'stress(MPa)', b'Part No.'
        ]
        feature_names = np.array(_all_feature_names[:3 + output_dim] + [b'Part No.'])
        meta_grp.create_dataset('feature_min',  data=np.array([nodal_data[i].min()  for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_max',  data=np.array([nodal_data[i].max()  for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_mean', data=np.array([nodal_data[i].mean() for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_std',  data=np.array([nodal_data[i].std()  for i in range(num_save_features)], dtype=np.float32))

        global_meta = f.create_group('metadata')
        global_meta.create_dataset('feature_names', data=feature_names)
        norm_grp = global_meta.create_group('normalization_params')
        norm_grp.create_dataset('node_mean',  data=ctx.node_mean)
        norm_grp.create_dataset('node_std',   data=ctx.node_std)
        norm_grp.create_dataset('edge_mean',  data=ctx.edge_mean)
        norm_grp.create_dataset('edge_std',   data=ctx.edge_std)
        norm_grp.create_dataset('delta_mean', data=ctx.delta_mean)
        norm_grp.create_dataset('delta_std',  data=ctx.delta_std)
        f.flush()


def _load_model_from_checkpoint(config, checkpoint, device):
    """Rebuild MeshGraphNets from a checkpoint and load (EMA-preferred) weights."""
    model = MeshGraphNets(config, str(device)).to(device)

    if 'ema_state_dict' in checkpoint:
        # AveragedModel saves keys as "module.<key>" + "n_averaged" buffer.
        # Strip the "module." prefix to get the raw model state dict directly,
        # avoiding a fragile AveragedModel reconstruction at inference.
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint valid loss: {checkpoint.get('valid_loss', 'unknown')}")
    return model


def _load_conditional_prior(config, checkpoint, model, device):
    """Return the conditional prior module to sample z from, or None.

    The joint-trained prior is a submodule of MeshGraphNets and was already
    loaded by load_state_dict. Legacy checkpoints stored a separately-trained
    ConditionalMixturePrior state dict instead.
    """
    if getattr(model, 'prior', None) is not None:
        model.prior.eval()
        return model.prior
    if 'conditional_prior_state_dict' in checkpoint:
        from model.conditional_prior import ConditionalMixturePrior
        prior_config = dict(config)
        prior_config.update(checkpoint.get('conditional_prior_config', {}))
        prior = ConditionalMixturePrior(prior_config).to(device)
        prior.load_state_dict(checkpoint['conditional_prior_state_dict'])
        prior.eval()
        return prior
    return None


def run_rollout(config, config_filename='config.txt'):
    """
    Perform autoregressive rollout inference.

    Given an initial condition from an HDF5 dataset and a pretrained model
    checkpoint, iteratively predicts the next state from the current state:

        1. Load initial state at t=0
        2. For each step t -> t+1:
            a. Normalize current state -> build graph
            b. Forward pass -> predicted normalized delta
            c. Denormalize delta;  state_{t+1} = state_t + delta
        3. Save all predicted timesteps to HDF5

    With use_vae=True, `num_vae_samples` independent trajectories are generated
    per scene, each with its own z drawn from the conditional prior (fallback:
    N(0, I)). `vae_batch_size` trajectories are advanced together in one
    batched forward pass.
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE ROLLOUT INFERENCE")
    print("=" * 60)

    # ---- Device --------------------------------------------------------
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ---- Checkpoint (weights + normalization + model_config) ------------
    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'normalization' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain normalization statistics. "
            "Re-train or re-save the model with the updated training code that "
            "includes normalization stats in the checkpoint."
        )
    norm = checkpoint['normalization']
    print(f"  Normalization stats loaded from checkpoint")

    # Override config with model_config from checkpoint if available
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        # Back-compat: checkpoints saved before prior_family existed are all
        # Gaussian-mixture priors. Without this shim the new 'fm' default
        # would rebuild them as FM priors and state_dict loading would fail.
        if ('prior_family' not in model_config
                and str(model_config.get('prior_type', '')).lower().strip() == 'gnn_e2e'):
            config['prior_family'] = 'gmm'
            print("\n  prior_family: gmm (implied by pre-FM checkpoint)")
        print(f"\n  Model config loaded from checkpoint:")
        for k, v in model_config.items():
            old_val = config.get(k)
            config[k] = v
            if old_val is not None and old_val != v:
                print(f"    {k}: {old_val} -> {v} (overridden by checkpoint)")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"\n  WARNING: No model_config in checkpoint, using config file values")

    if config.get('use_node_types') and norm.get('num_node_types'):
        config['num_node_types'] = norm['num_node_types']
        print(f"  Node types: {norm['num_node_types']} types, mapping: {norm.get('node_type_to_idx')}")

    # ---- Model + prior ---------------------------------------------------
    print("\nInitializing model...")
    model = _load_model_from_checkpoint(config, checkpoint, device)

    use_vae = config.get('use_vae', False)
    vae_latent_dim = int(config.get('vae_latent_dim', 8))
    use_conditional_prior = bool(config.get('use_conditional_prior', True))
    prior_temperature = float(config.get('prior_temperature', 1.0))
    num_vae_samples = int(config.get('num_vae_samples', 1)) if use_vae else 1
    raw_vae_batch_size = config.get('vae_batch_size', 1)
    auto_vae_batch_size = (
        use_vae
        and isinstance(raw_vae_batch_size, str)
        and raw_vae_batch_size.strip().lower() == 'auto'
    )
    if use_vae and not auto_vae_batch_size:
        vae_batch_size = max(1, int(raw_vae_batch_size))
    else:
        vae_batch_size = 1
    vae_batch_vram_fraction = float(config.get('vae_batch_vram_fraction', 0.70))
    vae_batch_vram_fraction = min(max(vae_batch_vram_fraction, 0.01), 1.0)
    vae_batch_size_min = max(1, int(config.get('vae_batch_size_min', 1)))
    vae_batch_size_max = max(0, int(config.get('vae_batch_size_max', 0)))

    conditional_prior = None
    if use_vae and use_conditional_prior:
        conditional_prior = _load_conditional_prior(config, checkpoint, model, device)

    if use_vae:
        if conditional_prior is not None:
            if getattr(conditional_prior, 'family', 'gmm') == 'fm':
                sampler_desc = (f"conditional flow-matching prior "
                                f"({conditional_prior.num_steps} Euler steps, temp={prior_temperature:g})")
            else:
                sampler_desc = (f"conditional mixture prior "
                                f"({conditional_prior.num_components} components, temp={prior_temperature:g})")
        else:
            sampler_desc = "N(0, I)"
        batch_desc = (
            f"auto, target={vae_batch_vram_fraction:.0%} free VRAM"
            if auto_vae_batch_size else str(vae_batch_size)
        )
        if auto_vae_batch_size and device.type != 'cuda':
            batch_desc = "auto requested, using 1 on CPU"
        print(f"  VAE sampling: {num_vae_samples} sample(s) per scene "
              f"(z_dim={vae_latent_dim}, batch_size={batch_desc}, prior={sampler_desc})")

    # ---- Rollout inputs --------------------------------------------------
    dataset_dir = config.get('infer_dataset')
    num_rollout_steps = config.get('infer_timesteps')
    input_dim = config.get('input_var')
    output_dim = config.get('output_var')

    # Inline z_disp spread-histogram (generated vs ground-truth eval dataset).
    # Enabled when an `eval_dataset` is given in the config (no GT -> no compare).
    output_dir = config.get('inference_output_dir', 'outputs/rollout')
    eval_dataset = config.get('eval_dataset')
    make_histogram = bool(config.get('make_histogram', use_vae and eval_dataset is not None))
    histogram_bins = int(config.get('histogram_bins', 60))
    histogram_clip_quantile = float(config.get('histogram_clip_quantile', 0.0))
    show_histogram = bool(config.get('show_histogram', True))
    z_gen_idx = Z_DISP_CHANNEL - 3  # z_disp is the 3rd output channel (index 2)
    generated_spreads = []  # one spread scalar per generated rollout trajectory

    print(f"\nLoading initial conditions...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Rollout steps: {num_rollout_steps}")

    with h5py.File(dataset_dir, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
    print(f"  Found {len(sample_ids)} samples: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    os.makedirs(output_dir, exist_ok=True)

    for sample_id in sample_ids:
        with h5py.File(dataset_dir, 'r') as f:
            nodal_data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]    # [2, M]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_id}: {num_nodes} nodes, {mesh_edge.shape[1]} edges, "
              f"{num_timesteps} dataset timestep(s)")

        # Resolve rollout length for this sample
        steps = num_rollout_steps
        if steps is None:
            if num_timesteps > 1:
                steps = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps} (full trajectory)")
            else:
                raise ValueError(
                    f"infer_timesteps not specified and dataset has only {num_timesteps} "
                    f"timestep(s). Please set infer_timesteps in the config."
                )

        # nodal_data layout: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
        ref_pos = nodal_data[:3, 0, :].T                     # [N, 3]
        initial_state = nodal_data[3:3 + input_dim, 0, :].T  # [N, input_dim]
        part_ids = (nodal_data[-1, 0, :].astype(np.int32)
                    if config.get('use_node_types') and num_features > 7 else None)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        ctx = _SampleContext(config, norm, ref_pos, edge_index, part_ids, device)

        def _run_batch(batch_start, requested_batch_size):
            B = min(requested_batch_size, num_vae_samples - batch_start)
            states = [initial_state.copy() for _ in range(B)]
            all_states = np.zeros((B, steps + 1, num_nodes, output_dim), dtype=np.float32)
            for b in range(B):
                all_states[b, 0] = initial_state[:, :output_dim]

            z_batch = None  # sampled once per trajectory batch at step 0
            rollout_start = time.time()

            with torch.no_grad():
                for step in range(steps):
                    # Step 0: all B states are identical; build one graph and
                    # replicate. Step 1+: states have diverged, so build each graph.
                    if step == 0:
                        graphs = [ctx.build_step_graph(states[0])] * B
                        if use_vae:
                            if conditional_prior is not None:
                                prior_batch = Batch.from_data_list(graphs)
                                z_batch = conditional_prior.sample(
                                    prior_batch, temperature=prior_temperature,
                                ).to(device)
                            else:
                                z_batch = torch.randn(B, vae_latent_dim, device=device)
                    else:
                        graphs = [ctx.build_step_graph(states[b]) for b in range(B)]

                    batch_graph = Batch.from_data_list(graphs)
                    predicted, _, _, _, _ = model(batch_graph, fixed_z=z_batch)
                    predicted = predicted.view(B, num_nodes, output_dim).cpu().numpy()

                    for b in range(B):
                        delta = predicted[b] * ctx.delta_std + ctx.delta_mean
                        states[b][:, :output_dim] += delta
                        all_states[b, step + 1] = states[b][:, :output_dim]

            return B, all_states, time.time() - rollout_start

        def _save_batch(batch_start, B, all_states, rollout_time):
            # Save one HDF5 per trajectory
            for b in range(B):
                vae_idx = batch_start + b

                if make_histogram and output_dim > z_gen_idx:
                    generated_spreads.append(
                        _spread_max_minus_min(all_states[b, -1, :, z_gen_idx])
                    )

                if num_vae_samples > 1:
                    filename = f"rollout_sample{sample_id}_vaesample{vae_idx}_steps{steps}.h5"
                else:
                    filename = f"rollout_sample{sample_id}_steps{steps}.h5"
                output_path = os.path.join(output_dir, filename)

                _save_rollout_h5(
                    output_path, sample_id, all_states[b], ctx, part_ids, output_dim,
                    steps, model_path, config_filename, rollout_time,
                    vae_sample_idx=vae_idx if use_vae else None,
                )
                if b == B - 1 or B <= 4:
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  Saved {filename}  ({size_mb:.1f} MB)")

        active_vae_batch_size = vae_batch_size
        batch_start = 0

        if use_vae and auto_vae_batch_size and device.type == 'cuda' and num_vae_samples > 0:
            print(f"\nRollout: {num_vae_samples} z-sample(s) x {steps} step(s), "
                  f"batch_size=auto (target {vae_batch_vram_fraction:.0%} of free VRAM)")
            print("  Auto batch probe: running z-sample 0 with batch_size=1")
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            free_before, total_bytes = torch.cuda.mem_get_info(device)
            allocated_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

            B, all_states, rollout_time = _run_batch(0, 1)

            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            probe_extra_allocated = max(0, peak_allocated - allocated_before)
            probe_extra_reserved = max(0, peak_reserved - reserved_before)
            probe_extra = max(1, probe_extra_allocated, probe_extra_reserved)
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info(device)

            _save_batch(0, B, all_states, rollout_time)
            batch_start = B

            remaining = num_vae_samples - batch_start
            target_extra = int(free_after * vae_batch_vram_fraction)
            raw_choice = max(1, int(target_extra // probe_extra))
            if vae_batch_size_max:
                raw_choice = min(raw_choice, vae_batch_size_max)
            if remaining > 0:
                active_vae_batch_size = min(
                    remaining,
                    max(vae_batch_size_min, raw_choice),
                )
            else:
                active_vae_batch_size = 1

            print("  Auto batch memory:")
            print(f"    free before probe: {_format_bytes(free_before)} / {_format_bytes(total_bytes)}")
            print(f"    free after probe:  {_format_bytes(free_after)}")
            print(f"    probe peak extra:  {_format_bytes(probe_extra)} "
                  f"(allocated {_format_bytes(probe_extra_allocated)}, "
                  f"reserved {_format_bytes(probe_extra_reserved)})")
            print(f"    target extra:      {_format_bytes(target_extra)} "
                  f"({vae_batch_vram_fraction:.0%} of free)")
            print(f"    selected batch:    {active_vae_batch_size}")
        else:
            if use_vae:
                if auto_vae_batch_size:
                    print("\nRollout: auto VAE batch sizing requested, but CUDA is unavailable; "
                          "using batch_size=1")
                num_batches = (num_vae_samples + active_vae_batch_size - 1) // active_vae_batch_size
                print(f"\nRollout: {num_vae_samples} z-sample(s) x {steps} step(s), "
                      f"batch_size={active_vae_batch_size} -> {num_batches} forward batch(es)")

        batch_counter = 0
        while batch_start < num_vae_samples:
            requested_batch_size = min(active_vae_batch_size, num_vae_samples - batch_start)
            if use_vae:
                batch_counter += 1
                print(f"  batch {batch_counter}: z-samples "
                      f"{batch_start}-{batch_start + requested_batch_size - 1} "
                      f"(B={requested_batch_size})")
            try:
                B, all_states, rollout_time = _run_batch(batch_start, requested_batch_size)
            except RuntimeError as exc:
                if (auto_vae_batch_size and device.type == 'cuda'
                        and requested_batch_size > 1 and _is_cuda_oom(exc)):
                    new_batch_size = max(1, requested_batch_size // 2)
                    print(f"  CUDA OOM at batch_size={requested_batch_size}; "
                          f"retrying z-sample {batch_start} with batch_size={new_batch_size}")
                    active_vae_batch_size = new_batch_size
                    if use_vae:
                        batch_counter -= 1
                    torch.cuda.empty_cache()
                    continue
                raise
            _save_batch(batch_start, B, all_states, rollout_time)
            batch_start += B

    total_outputs = len(sample_ids) * num_vae_samples
    print(f"\nRollout inference complete. Processed {len(sample_ids)} scene(s) x "
          f"{num_vae_samples} VAE sample(s) = {total_outputs} output file(s).")

    # ---- z_disp spread histogram: generated vs ground-truth eval set ------
    if make_histogram:
        print("\n" + "=" * 60)
        print("HISTOGRAM COMPARE (z_disp spread, final timestep)")
        print("=" * 60)
        if not eval_dataset:
            print("  Skipped: no `eval_dataset` set in config "
                  "(needed for the ground-truth comparison).")
        elif not os.path.exists(str(eval_dataset)):
            print(f"  Skipped: eval_dataset not found: {eval_dataset}")
        elif not generated_spreads:
            print("  Skipped: no generated spread values were collected "
                  f"(output_var={output_dim} has no z_disp channel).")
        else:
            try:
                gt = _eval_dataset_spreads(str(eval_dataset))
                gen = np.asarray(generated_spreads, dtype=np.float64)
                print(f"  GT spread values  (1 per eval sample): {gt.size:,}")
                print(f"  Gen spread values (1 per rollout):     {gen.size:,}")
                hist_path = os.path.join(output_dir, 'histogram_compare.png')
                _plot_spread_histogram(
                    gt, gen, hist_path,
                    eval_path=str(eval_dataset), rollout_dir=output_dir,
                    bins=histogram_bins, clip_quantile=histogram_clip_quantile,
                    show=show_histogram,
                )
            except Exception as exc:  # never let plotting break a finished rollout
                print(f"  Histogram generation failed: {exc}")
