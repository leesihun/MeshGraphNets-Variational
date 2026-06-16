import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.mesh_dataset import _compute_positional_features
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


def run_rollout(config, config_filename='config.txt'):
    """
    Perform autoregressive time-transient rollout inference.

    Given an initial condition from an HDF5 dataset and a pretrained model checkpoint,
    this function iteratively predicts the next state from the current state, building
    up a full time-transient trajectory.

    The rollout loop:
        1. Load initial state at t=rollout_start_step
        2. For each step t -> t+1:
            a. Normalize current state -> build graph
            b. Forward pass -> predicted normalized delta
            c. Denormalize delta
            d. Update state: state_{t+1} = state_t + delta
        3. Save all predicted timesteps to HDF5

    Args:
        config: Configuration dictionary (from load_config)
        config_filename: Path to the config file
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE ROLLOUT INFERENCE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup device
    # -------------------------------------------------------------------------
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        gpu_id = gpu_ids[0]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}, device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Load checkpoint (model weights + normalization statistics)
    # -------------------------------------------------------------------------
    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract normalization statistics from checkpoint
    if 'normalization' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain normalization statistics. "
            "Re-train or re-save the model with the updated training code that "
            "includes normalization stats in the checkpoint."
        )

    norm = checkpoint['normalization']
    node_mean = norm['node_mean']  # np.ndarray [input_var]
    node_std = norm['node_std']    # np.ndarray [input_var]
    edge_mean = norm['edge_mean']  # np.ndarray [4]
    edge_std = norm['edge_std']    # np.ndarray [4]
    delta_mean = norm['delta_mean']  # np.ndarray [output_var]
    delta_std = norm['delta_std']    # np.ndarray [output_var]
    # Multiscale coarse edge normalization stats (per-level lists)
    if 'coarse_edge_means' in norm:
        # New format: list of arrays per level
        coarse_edge_means = norm['coarse_edge_means']
        coarse_edge_stds  = norm['coarse_edge_stds']
    elif 'coarse_edge_mean' in norm:
        # Legacy format: single array → wrap in list
        coarse_edge_means = [norm['coarse_edge_mean']]
        coarse_edge_stds  = [norm['coarse_edge_std']]
    else:
        coarse_edge_means = [edge_mean]
        coarse_edge_stds  = [edge_std]

    print(f"  Normalization stats loaded from checkpoint")
    print(f"    node_mean:  {node_mean}")
    print(f"    node_std:   {node_std}")
    print(f"    delta_mean: {delta_mean}")
    print(f"    delta_std:  {delta_std}")

    # Override config with model_config from checkpoint if available
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
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

    # Node type info
    use_node_types = config.get('use_node_types')
    node_type_to_idx = norm.get('node_type_to_idx')
    num_node_types = norm.get('num_node_types')

    if use_node_types and num_node_types is not None and num_node_types > 0:
        config['num_node_types'] = num_node_types
        print(f"  Node types: {num_node_types} types, mapping: {node_type_to_idx}")

    # World edge info
    use_world_edges = config.get('use_world_edges')
    world_edge_radius = norm.get('world_edge_radius')
    world_max_num_neighbors = config.get('world_max_num_neighbors', 64)

    # Determine world edge backend
    requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
    if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
        world_edge_backend = 'torch_cluster'
    else:
        world_edge_backend = 'scipy_kdtree'

    if use_world_edges:
        print(f"  World edges: radius={world_edge_radius}, backend={world_edge_backend}")

    # -------------------------------------------------------------------------
    # 3. Initialize model
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)

    # Prefer EMA weights (better generalization), fall back to training weights
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

    # -------------------------------------------------------------------------
    # 4. Load initial condition from HDF5
    # -------------------------------------------------------------------------
    dataset_dir = config.get('infer_dataset')
    num_rollout_steps = config.get('infer_timesteps')
    input_dim = config.get('input_var')
    output_dim = config.get('output_var')
    num_pos_features = int(config.get('positional_features', 0))
    positional_encoding = str(config.get('positional_encoding', 'rwpe')).lower().strip()
    use_vae = config.get('use_vae', False)
    vae_latent_dim = int(config.get('vae_latent_dim', 8))
    use_conditional_prior = bool(config.get('use_conditional_prior', True))
    prior_temperature = float(config.get('prior_temperature', 1.0))

    num_vae_samples = int(config.get('num_vae_samples', 1)) if use_vae else 1

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

    conditional_prior = None
    if use_vae and use_conditional_prior:
        # New: joint-trained GNN prior is a submodule of MeshGraphNets and was
        # already loaded by model.load_state_dict above. Use it directly.
        if getattr(model, 'prior', None) is not None:
            conditional_prior = model.prior
            conditional_prior.eval()
        # Legacy back-compat: previously the prior was saved as a separate state
        # dict by the (now removed) post-hoc training pass.
        elif 'conditional_prior_state_dict' in checkpoint:
            from model.conditional_prior import ConditionalMixturePrior
            prior_config = dict(config)
            prior_config.update(checkpoint.get('conditional_prior_config', {}))
            conditional_prior = ConditionalMixturePrior(prior_config).to(device)
            conditional_prior.load_state_dict(checkpoint['conditional_prior_state_dict'])
            conditional_prior.eval()

    # Load legacy GMM prior if present in checkpoint (falls back to N(0,I) if absent)
    gmm_params = None
    if use_vae and conditional_prior is None:
        from model.latent_gmm import load_latent_gmm
        gmm_params = load_latent_gmm(checkpoint)

    print(f"\nLoading initial condition...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Rollout steps: {num_rollout_steps}")
    if use_vae:
        if conditional_prior is not None:
            sampler_desc = (
                f"conditional mixture prior "
                f"({conditional_prior.num_components} components, temp={prior_temperature:g})"
            )
        else:
            sampler_desc = (f"legacy GMM ({gmm_params['n_components']} components, "
                            f"{gmm_params['covariance_type']} cov)"
                            if gmm_params is not None else "N(0, I)")
        print(f"  VAE sampling: {num_vae_samples} sample(s) per scene "
              f"(z_dim={vae_latent_dim}, prior={sampler_desc})")

    # We need to infer all samples in the dataset
    # Gather sample IDs (may not be sequential 0..N-1)
    with h5py.File(dataset_dir, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])

    print(f"  Found {len(sample_ids)} samples: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    for sample_idx, sample_id in enumerate(sample_ids):

        with h5py.File(dataset_dir, 'r') as f:
            nodal_data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]    # [2, M]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"  Data shape: {nodal_data.shape} (features, timesteps, nodes)")
        print(f"  Mesh edges: {mesh_edge.shape[1]} (unidirectional)")

        # Validate rollout range (use local copy so config value isn't mutated across samples)
        steps_this_sample = num_rollout_steps
        if steps_this_sample is None:
            # If dataset has multiple timesteps, use them; otherwise use requested rollout steps
            if num_timesteps > 1:
                steps_this_sample = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps_this_sample} (full trajectory)")
            else:
                raise ValueError(
                    f"infer_timesteps not specified and dataset has only {num_timesteps} timestep(s). "
                    f"Please set infer_timesteps in config.txt"
                )

        # For inference, we typically want to generate MORE timesteps than in the dataset
        # Only warn if explicitly requesting fewer steps than available
        if steps_this_sample > num_timesteps and num_timesteps > 1:
            print(f"  INFO: Requested {steps_this_sample} steps, dataset has {num_timesteps} timesteps. "
                  f"Will generate {steps_this_sample} new predictions beyond the dataset.")

        # Extract initial state (always start from timestep 0)
        # nodal_data layout: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
        ref_pos = nodal_data[:3, 0, :].T  # [N, 3] reference position
        initial_state = nodal_data[3:3+input_dim, 0, :].T  # [N, input_dim]

        # Extract node types if enabled
        if use_node_types and num_features > 7:
            part_ids = nodal_data[-1, 0, :].astype(np.int32)  # [N]
        else:
            part_ids = None

        # Make edges bidirectional
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        # Pre-compute static positional features (geometry + topology, rotation-invariant)
        if num_pos_features > 0:
            pos_features = _compute_positional_features(
                ref_pos, edge_index, num_pos_features, positional_encoding
            )  # [N, num_pos_features]
            print(f"  Positional features: {pos_features.shape} ({positional_encoding})")
        else:
            pos_features = None

        # Pre-compute per-level coarsening topology (static — same for all rollout steps)
        use_multiscale = config.get('use_multiscale', False)
        multiscale_levels = int(config.get('multiscale_levels', 1))
        # Per-level coarsening config (from checkpoint model_config or config file)
        raw_ct = config.get('coarsening_type', 'bfs')
        if isinstance(raw_ct, list):
            coarsening_types = [str(t).strip().lower() for t in raw_ct]
        else:
            coarsening_types = [str(raw_ct).strip().lower()] * multiscale_levels
        if len(coarsening_types) == 1 and multiscale_levels > 1:
            coarsening_types = coarsening_types * multiscale_levels
        raw_vc = config.get('voronoi_clusters', None)
        if raw_vc is None:
            voronoi_clusters = [0] * multiscale_levels
        elif isinstance(raw_vc, list):
            voronoi_clusters = [int(v) for v in raw_vc]
        else:
            voronoi_clusters = [int(raw_vc)] * multiscale_levels
        if len(voronoi_clusters) == 1 and multiscale_levels > 1:
            voronoi_clusters = voronoi_clusters * multiscale_levels

        bipartite_unpool = config.get('bipartite_unpool', False)

        coarse_hierarchy = None  # list of {'ftc', 'c_ei', 'n_c'[, 'up_ei']} per level
        if use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
            coarse_hierarchy = build_multiscale_hierarchy(
                edge_index, num_nodes, ref_pos,
                multiscale_levels, coarsening_types, voronoi_clusters,
                bipartite_unpool=bipartite_unpool,
            )
            current_n_report = num_nodes
            for level, entry in enumerate(coarse_hierarchy):
                method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
                n_c = entry['n_c']
                print(f"  Coarsening level {level} ({method}): {current_n_report} → {n_c} nodes ({n_c/current_n_report*100:.1f}%)")
                current_n_report = n_c

        print(f"  Reference positions: {ref_pos.shape}")
        print(f"  Initial state: {initial_state.shape}")
        print(f"  Bidirectional edges: {edge_index.shape[1]}")

        # -------------------------------------------------------------------------
        # 5. Autoregressive rollout  (looped over VAE samples when use_vae=True)
        # -------------------------------------------------------------------------
        for vae_sample_idx in range(num_vae_samples):

            if use_vae and num_vae_samples > 1:
                print(f"\n{'=' * 60}")
                print(f"VAE sample {vae_sample_idx + 1}/{num_vae_samples}")

            # Sample a fixed z for this entire trajectory
            fixed_z = None
            if use_vae:
                if conditional_prior is not None:
                    # Sample after the first graph is built so z is conditioned on this mesh.
                    fixed_z = None
                elif gmm_params is not None:
                    from model.latent_gmm import sample_from_gmm
                    fixed_z = sample_from_gmm(gmm_params, 1, device)
                else:
                    fixed_z = torch.randn(1, vae_latent_dim, device=device)

            print(f"\n{'=' * 60}")
            print(f"Starting rollout: {steps_this_sample} steps")
            print(f"{'=' * 60}")

            # Storage for all predicted states: [num_steps+1, N, output_dim]
            all_states = np.zeros((steps_this_sample + 1, num_nodes, output_dim), dtype=np.float32)
            all_states[0] = initial_state[:, :output_dim]

            current_state = initial_state.copy()  # [N, input_dim], mutable
            rollout_start_time = time.time()

            with torch.no_grad():
                for step in range(steps_this_sample):
                    step_start = time.time()

                    # Nodal feature: physical state + positional features (must match training)
                    if pos_features is not None:
                        x_raw = np.concatenate([current_state, pos_features], axis=1)  # [N, input_dim + pos]
                    else:
                        x_raw = current_state  # [N, input_dim]
                    x_norm = (x_raw - node_mean) / node_std

                    # --- c. Add node type one-hot if enabled ---
                    if use_node_types and part_ids is not None and node_type_to_idx is not None:
                        node_type_indices = np.array(
                            [node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32
                        )
                        node_type_onehot = np.zeros((num_nodes, num_node_types), dtype=np.float32)
                        node_type_onehot[np.arange(num_nodes), node_type_indices] = 1.0
                        x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

                    # --- d. Compute deformed position ---
                    displacement = current_state[:, :3]  # [N, 3]
                    deformed_pos = ref_pos + displacement  # [N, 3]

                    # --- e. Compute 8-D edge features from current and reference geometry ---
                    edge_attr_raw = compute_edge_attr(ref_pos, deformed_pos, edge_index)

                    # --- f. Normalize edge features ---
                    edge_attr_norm = (edge_attr_raw - edge_mean) / edge_std

                    # --- g. Build graph Data object ---
                    DataClass = MultiscaleData if use_multiscale else Data
                    graph = DataClass(
                        x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                        edge_index=torch.from_numpy(edge_index).long().to(device),
                        edge_attr=torch.from_numpy(edge_attr_norm.astype(np.float32)).to(device),
                        pos=torch.from_numpy(ref_pos.astype(np.float32)).to(device),
                    )

                    # --- h. Compute world edges if enabled ---
                    if use_world_edges and world_edge_radius is not None:
                        world_ei, world_ea = compute_world_edges(
                            ref_pos, deformed_pos, edge_index,
                            radius=world_edge_radius,
                            max_num_neighbors=world_max_num_neighbors,
                            backend=world_edge_backend,
                            device=device,
                            edge_mean=edge_mean, edge_std=edge_std,
                        )
                        graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
                        graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
                    else:
                        graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        graph.world_edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

                    # --- h2. Attach per-level coarse graph data if multiscale ---
                    if use_multiscale and coarse_hierarchy is not None:
                        use_cwe = bool(config.get('coarse_world_edges', False))
                        world_ei_for_coarse = (
                            graph.world_edge_index.cpu().numpy()
                            if use_world_edges and use_cwe else None
                        )
                        attach_coarse_levels_to_graph(
                            graph, coarse_hierarchy,
                            ref_pos, deformed_pos,
                            coarse_edge_means, coarse_edge_stds,
                            device=device,
                            world_edge_index=world_ei_for_coarse,
                        )

                    if use_vae and conditional_prior is not None and fixed_z is None:
                        fixed_z = conditional_prior.sample(
                            graph, temperature=prior_temperature
                        ).to(device)

                    # --- i. Forward pass ---
                    predicted_delta_norm, _, _, _, _ = model(graph, fixed_z=fixed_z)  # [N, output_var]

                    # --- j. Denormalize delta ---
                    predicted_delta_norm_np = predicted_delta_norm.cpu().numpy()
                    predicted_delta = predicted_delta_norm_np * delta_std + delta_mean  # [N, output_var]

                    # --- k. Update state ---
                    current_state[:, :output_dim] = current_state[:, :output_dim] + predicted_delta

                    # --- l. Store result ---
                    all_states[step + 1] = current_state[:, :output_dim]

                    step_time = time.time() - step_start
                    if step % max(1, steps_this_sample // 20) == 0 or step == steps_this_sample - 1:
                        disp_mag = np.linalg.norm(current_state[:, :3], axis=1)
                        print(
                            f"  Step {step+1:>4d}/{steps_this_sample} | "
                            f"time: {step_time:.3f}s | "
                            f"disp range: [{disp_mag.min():.4e}, {disp_mag.max():.4e}]"
                        )

            total_rollout_time = time.time() - rollout_start_time
            if steps_this_sample > 0:
                print(f"\nRollout completed in {total_rollout_time:.2f}s ({total_rollout_time/steps_this_sample:.3f}s/step)")
            else:
                print(f"\nRollout completed in {total_rollout_time:.2f}s (no steps executed)")

            # Record this trajectory's z_disp spread for the histogram compare.
            # Computed from the same field that gets written to the .h5 (channel 5).
            if make_histogram and output_dim > z_gen_idx:
                generated_spreads.append(
                    _spread_max_minus_min(all_states[-1, :, z_gen_idx])
                )

            # -------------------------------------------------------------------------
            # 6. Save results to HDF5 (DATASET_FORMAT.md structure)
            # -------------------------------------------------------------------------
            output_dir = config.get('inference_output_dir', 'outputs/rollout')
            os.makedirs(output_dir, exist_ok=True)

            if use_vae and num_vae_samples > 1:
                output_filename = f"rollout_sample{sample_id}_vaesample{vae_sample_idx}_steps{steps_this_sample}.h5"
            else:
                output_filename = f"rollout_sample{sample_id}_steps{steps_this_sample}.h5"
            output_path = os.path.join(output_dir, output_filename)
            output_path_abs = os.path.abspath(output_path)

            print(f"\nSaving results to: {output_path_abs}")

            with h5py.File(output_path, 'w') as f:
                # Root attributes (mimics DATASET_FORMAT.md)
                f.attrs['num_samples'] = 1
                f.attrs['num_features'] = 3 + output_dim + 1
                f.attrs['num_timesteps'] = steps_this_sample + 1

                # ========================================
                # data/{sample_id}/ group
                # ========================================
                data_grp = f.create_group('data')
                sample_grp = data_grp.create_group(str(sample_id))

                # Build nodal_data: [3 + output_dim + 1, timesteps, nodes]
                # Layout: [x, y, z, <output_dim predicted channels>, part_number]
                num_save_features = 3 + output_dim + 1
                nodal_data = np.zeros((num_save_features, steps_this_sample + 1, num_nodes), dtype=np.float32)

                # Reference position (constant across timesteps)
                nodal_data[0, :, :] = ref_pos[:, 0]
                nodal_data[1, :, :] = ref_pos[:, 1]
                nodal_data[2, :, :] = ref_pos[:, 2]

                # Predicted state channels (all_states shape: [steps+1, nodes, output_dim])
                for ch in range(output_dim):
                    nodal_data[3 + ch, :, :] = all_states[:, :, ch]

                # Part number (constant across timesteps)
                if part_ids is not None:
                    nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]
                else:
                    nodal_data[3 + output_dim, :, :] = 0

                sample_grp.create_dataset(
                    'nodal_data', data=nodal_data,
                    compression='gzip', compression_opts=4
                )

                # mesh_edge: [2, M] unidirectional
                sample_grp.create_dataset('mesh_edge', data=mesh_edge)

                # ========================================
                # Metadata group (per-sample)
                # ========================================
                meta_grp = sample_grp.create_group('metadata')

                # Attributes
                meta_grp.attrs['sample_id'] = sample_id
                meta_grp.attrs['num_nodes'] = num_nodes
                meta_grp.attrs['num_edges'] = mesh_edge.shape[1]
                meta_grp.attrs['num_timesteps'] = steps_this_sample + 1
                meta_grp.attrs['model_path'] = model_path
                meta_grp.attrs['config_file'] = config_filename
                meta_grp.attrs['total_rollout_time_s'] = total_rollout_time
                if use_vae:
                    meta_grp.attrs['vae_sample_idx'] = vae_sample_idx

                # Feature statistics (per-feature, computed from predicted data)
                _all_feature_names = [
                    b'x_coord', b'y_coord', b'z_coord',
                    b'x_disp(mm)', b'y_disp(mm)', b'z_disp(mm)',
                    b'stress(MPa)', b'Part No.'
                ]
                feature_names = np.array(_all_feature_names[:3 + output_dim] + [b'Part No.'])
                feature_min  = np.array([nodal_data[i].min()  for i in range(num_save_features)], dtype=np.float32)
                feature_max  = np.array([nodal_data[i].max()  for i in range(num_save_features)], dtype=np.float32)
                feature_mean = np.array([nodal_data[i].mean() for i in range(num_save_features)], dtype=np.float32)
                feature_std  = np.array([nodal_data[i].std()  for i in range(num_save_features)], dtype=np.float32)

                meta_grp.create_dataset('feature_min', data=feature_min)
                meta_grp.create_dataset('feature_max', data=feature_max)
                meta_grp.create_dataset('feature_mean', data=feature_mean)
                meta_grp.create_dataset('feature_std', data=feature_std)

                # ========================================
                # Global metadata
                # ========================================
                global_meta = f.create_group('metadata')

                # Feature names
                global_meta.create_dataset('feature_names', data=feature_names)

                # Normalization parameters used for inference
                norm_grp = global_meta.create_group('normalization_params')
                norm_grp.create_dataset('node_mean', data=node_mean)
                norm_grp.create_dataset('node_std', data=node_std)
                norm_grp.create_dataset('edge_mean', data=edge_mean)
                norm_grp.create_dataset('edge_std', data=edge_std)
                norm_grp.create_dataset('delta_mean', data=delta_mean)
                norm_grp.create_dataset('delta_std', data=delta_std)

                f.flush()

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Saved ({file_size_mb:.1f} MB)")
            print(f"  File is now closed and ready to read.")

    total_outputs = len(sample_ids) * num_vae_samples
    print(f"\nRollout inference complete. Processed {len(sample_ids)} scene(s) × {num_vae_samples} VAE sample(s) = {total_outputs} output file(s).")

    # -------------------------------------------------------------------------
    # 7. z_disp spread histogram: generated rollouts vs ground-truth eval set
    # -------------------------------------------------------------------------
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


