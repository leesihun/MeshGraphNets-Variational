import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from torch_geometric.data import Data

from general_modules.dataset_stats import compute_normalization_stats, finalize_moments
from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features
from general_modules.world_edges import HAS_TORCH_CLUSTER, compute_world_edges

# Multiscale coarsening (optional — only imported when use_multiscale=True)
try:
    from model.coarsening import MultiscaleData, compute_coarse_centroids
    from general_modules.multiscale_helpers import attach_coarse_levels_to_graph
    HAS_COARSENING = True
except ImportError:
    HAS_COARSENING = False


class MeshGraphDataset(Dataset):

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        # Graph and feature parameters
        self.input_dim = config.get('input_var')  # Physical features only (4)
        self.output_dim = config.get('output_var')  # Physical features only (4)
        self.num_pos_features = int(config.get('positional_features', 0))
        configured_edge_dim = int(config.get('edge_var', EDGE_FEATURE_DIM))
        if configured_edge_dim != EDGE_FEATURE_DIM:
            raise ValueError(
                f"edge_var must be {EDGE_FEATURE_DIM} "
                f"([deformed_dx, deformed_dy, deformed_dz, deformed_dist, "
                f"ref_dx, ref_dy, ref_dz, ref_dist]), got {configured_edge_dim}"
            )
        self.edge_dim = EDGE_FEATURE_DIM

        # Node type parameters
        self.use_node_types = config.get('use_node_types', False)
        self.num_node_types = None  # Will be computed from dataset if node types exist
        self.node_type_to_idx = None  # Mapping from node type values to contiguous indices

        # World edge parameters
        self.use_world_edges = config.get('use_world_edges', False)
        self.coarse_world_edges = config.get('coarse_world_edges', False)
        self.world_radius_multiplier = config.get('world_radius_multiplier', 1.5)
        self.world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
        self.world_edge_radius = None  # Computed from mesh statistics
        self.min_edge_length = None    # Computed from first sample

        # Determine which world edge backend to use
        requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
        if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'torch_cluster'
        elif requested_backend == 'scipy_kdtree' or not HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'scipy_kdtree'
        else:
            # Invalid backend requested, default to available option
            self.world_edge_backend = 'torch_cluster' if HAS_TORCH_CLUSTER else 'scipy_kdtree'

        # Multiscale / coarsening parameters
        self.use_multiscale = config.get('use_multiscale', False)
        self.multiscale_levels = int(config.get('multiscale_levels', 1))
        self.coarse_edge_means: List = []   # per-level: [mean_level_0, mean_level_1, ...]
        self.coarse_edge_stds: List = []    # per-level: [std_level_0, std_level_1, ...]
        # Small bounded per-worker cache for positional features (used only when
        # the on-disk cache is absent, i.e. non-multiscale runs).
        self._static_cache_max = int(config.get('static_cache_per_worker', 64))
        self._static_cache: Dict = {}  # {sample_id: x_pos}
        # On-disk hierarchy/positional cache (built once, streamed by all workers).
        self._ms_cache_path = None
        self._ms_reader = None

        # Per-level coarsening method.
        from model.coarsening import ACCEPTED_COARSEN_METHODS
        raw_ct = config.get('coarsening_type', 'bfs')
        if isinstance(raw_ct, list):
            self.coarsening_types = [str(t).strip().lower() for t in raw_ct]
        else:
            self.coarsening_types = [str(raw_ct).strip().lower()]
        for t in self.coarsening_types:
            if t not in ACCEPTED_COARSEN_METHODS:
                raise ValueError(
                    f"Unknown coarsening method '{t}' in coarsening_type. "
                    f"Accepted: {ACCEPTED_COARSEN_METHODS}."
                )
        # Expand single value to all levels
        if len(self.coarsening_types) == 1 and self.multiscale_levels > 1:
            self.coarsening_types = self.coarsening_types * self.multiscale_levels

        # Per-level cluster counts for voronoi levels
        raw_vc = config.get('voronoi_clusters', None)
        if raw_vc is None:
            self.voronoi_clusters: List[int] = [0] * self.multiscale_levels
        elif isinstance(raw_vc, list):
            self.voronoi_clusters = [int(v) for v in raw_vc]
        else:
            self.voronoi_clusters = [int(raw_vc)]
        # Expand single value to all levels
        if len(self.voronoi_clusters) == 1 and self.multiscale_levels > 1:
            self.voronoi_clusters = self.voronoi_clusters * self.multiscale_levels

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}, edge_dim: {self.edge_dim}")
        print(f"  use_node_types: {self.use_node_types}")
        print(f"  use_world_edges: {self.use_world_edges}")
        print(f"  use_multiscale: {self.use_multiscale}" + (f" (levels={self.multiscale_levels})" if self.use_multiscale else ""))
        if self.use_multiscale:
            print(f"  coarsening_types: {self.coarsening_types}")
            if any(t.startswith('voronoi') for t in self.coarsening_types):
                print(f"  voronoi_clusters: {self.voronoi_clusters}")
        if self.use_world_edges:
            print(f"  world_radius_multiplier: {self.world_radius_multiplier}")
            print(f"  world_max_num_neighbors: {self.world_max_num_neighbors}")
            print(f"  world_edge_backend: {self.world_edge_backend}")

        # Load sample IDs, timesteps, and normalization params
        with h5py.File(h5_file, 'r') as f:
            if 'data' not in f:
                raise ValueError(f"HDF5 file missing 'data' group")

            self.sample_ids = sorted([int(k) for k in f['data'].keys()])

            # Check number of timesteps from first sample
            sample_id = self.sample_ids[0]
            data_shape = f[f'data/{sample_id}/nodal_data'].shape
            self.num_timesteps = data_shape[1]  # Shape: (features, time, nodes)

            self.delta_mean = None
            self.delta_std = None
            self.node_mean = None
            self.node_std = None
            self.edge_mean = None
            self.edge_std = None
            self._h5_handle = None
            self._static_cache = {}
            self.is_training = False
            self.augment_geometry = False

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}")

        self._sanity_check_mesh_topology()

        # Build (or locate) the shared on-disk hierarchy cache for ALL samples.
        # Hierarchies are pure functions of mesh topology + reference geometry, so
        # one shared file serves every split, worker, and concurrent job — keeping
        # FPS off the hot path and per-worker RAM flat.
        if self.use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True requires model/coarsening.py (import failed)")
            from general_modules.multiscale_cache import ensure_cache
            self._ms_cache_path = ensure_cache(self.h5_file, self.sample_ids, config)

    def _sanity_check_mesh_topology(self) -> None:
        """Validate mesh topology across all samples.

        Reports node/edge/cell sizes, edge/node ratio distribution, infers
        likely element type from cell/node ratios, flags outliers, and checks
        edge index validity.
        """
        print('\n  === Mesh Topology Sanity Check ===')
        n_samples = len(self.sample_ids)

        node_counts = np.empty(n_samples, dtype=np.int64)
        edge_counts = np.empty(n_samples, dtype=np.int64)
        cell_counts = np.empty(n_samples, dtype=np.int64)
        has_cells = False
        ev_ratios = np.empty(n_samples, dtype=np.float64)
        issues = []

        with h5py.File(self.h5_file, 'r') as f:
            for i, sid in enumerate(self.sample_ids):
                grp = f[f'data/{sid}']
                nodal_shape = grp['nodal_data'].shape   # (features, time, nodes)
                edge_data = grp['mesh_edge'][:]          # (2, edges)

                n_nodes = nodal_shape[2]
                n_edges = edge_data.shape[1]
                node_counts[i] = n_nodes
                edge_counts[i] = n_edges
                ev_ratios[i] = n_edges / n_nodes if n_nodes > 0 else 0.0

                # Read cell count from metadata if available
                meta = grp.get('metadata')
                if meta is not None and 'num_cells' in meta.attrs:
                    cell_counts[i] = int(meta.attrs['num_cells'])
                    has_cells = True
                else:
                    cell_counts[i] = 0

                # Edge index bounds
                if n_edges > 0:
                    emin, emax = int(edge_data.min()), int(edge_data.max())
                    if emin < 0 or emax >= n_nodes:
                        issues.append(f"  ERROR: sample {sid}: edge index out of range "
                                      f"[{emin}, {emax}] for {n_nodes} nodes")
                    self_loops = int(np.sum(edge_data[0] == edge_data[1]))
                    if self_loops > 0:
                        issues.append(f"  ERROR: sample {sid}: {self_loops} self-loops")

                if n_nodes == 0:
                    issues.append(f"  ERROR: sample {sid}: 0 nodes")
                if n_edges == 0:
                    issues.append(f"  ERROR: sample {sid}: 0 edges")

        # --- Summary statistics ---
        ev_mean = float(np.mean(ev_ratios))
        ev_std = float(np.std(ev_ratios))

        print(f'  Nodes      - min: {int(np.min(node_counts)):,}  max: {int(np.max(node_counts)):,}  '
              f'mean: {int(np.mean(node_counts)):,}')
        print(f'  Edges      - min: {int(np.min(edge_counts)):,}  max: {int(np.max(edge_counts)):,}  '
              f'mean: {int(np.mean(edge_counts)):,}')
        if has_cells:
            valid_cells = cell_counts[cell_counts > 0]
            print(f'  Cells      - min: {int(np.min(valid_cells)):,}  max: {int(np.max(valid_cells)):,}  '
                  f'mean: {int(np.mean(valid_cells)):,}')
        print(f'  Edge/Node  - min: {float(np.min(ev_ratios)):.4f}  max: {float(np.max(ev_ratios)):.4f}  '
              f'mean: {ev_mean:.4f}  std: {ev_std:.4f}')

        # --- Element type inference from cell metadata ---
        if has_cells:
            valid_mask = (cell_counts > 0) & (node_counts > 0) & (edge_counts > 0)
            if np.any(valid_mask):
                cv_ratios = cell_counts[valid_mask].astype(np.float64) / node_counts[valid_mask]
                ec_ratios = edge_counts[valid_mask].astype(np.float64) / cell_counts[valid_mask]
                cv_mean = float(np.mean(cv_ratios))
                ec_mean = float(np.mean(ec_ratios))

                print(f'  Cell/Node  - mean: {cv_mean:.4f}')
                print(f'  Edge/Cell  - mean: {ec_mean:.4f}')

                # Classify element type using (Cell/Node, Edge/Node) signature:
                #   Tri shell:   C/V ~ 2.0, E/V ~ 3.0  (cells = triangular faces)
                #   Tet volume:  C/V ~ 3-6, E/V ~ 4.5-7 (cells = tetrahedra)
                #   Quad shell:  C/V ~ 1.0, E/V ~ 2.0  (cells = quad faces)
                #   Hex volume:  C/V ~ 1.0, E/V ~ 3-4  (cells = hexahedra)
                if 1.7 <= cv_mean <= 2.3 and 2.7 <= ev_mean <= 3.3:
                    elem_type = 'triangulated surface (shell)'
                elif cv_mean > 2.5 and ev_mean > 4.0:
                    elem_type = 'tetrahedral volume'
                elif 0.7 <= cv_mean <= 1.3 and 1.7 <= ev_mean <= 2.3:
                    elem_type = 'quad surface (shell)'
                elif 0.7 <= cv_mean <= 1.3 and ev_mean > 2.3:
                    elem_type = 'hexahedral volume'
                else:
                    elem_type = f'unknown (C/V={cv_mean:.2f}, E/V={ev_mean:.2f})'
                print(f'  Likely element type: {elem_type}')

        # --- Edge/Node ratio distribution (histogram for wide spreads) ---
        if ev_std > 0.01:
            hist, bin_edges = np.histogram(ev_ratios, bins=min(15, n_samples))
            print(f'  Edge/Node ratio distribution:')
            max_bar = max(hist) if max(hist) > 0 else 1
            for j in range(len(hist)):
                if hist[j] == 0:
                    continue
                lo, hi = bin_edges[j], bin_edges[j + 1]
                bar = '#' * max(1, int(40 * hist[j] / max_bar))
                print(f'    [{lo:.3f}, {hi:.3f}): {hist[j]:>5}  {bar}')

        # --- Outliers (>3 sigma from mean) ---
        if ev_std > 1e-6:
            outlier_mask = np.abs(ev_ratios - ev_mean) > 3 * ev_std
            n_outliers = int(np.sum(outlier_mask))
            if n_outliers > 0:
                print(f'  WARNING: {n_outliers} sample(s) with outlier edge/node ratio (>3 sigma):')
                for j in np.where(outlier_mask)[0][:10]:
                    sid = self.sample_ids[j]
                    print(f'    sample {sid}: nodes={node_counts[j]:,}  '
                          f'edges={edge_counts[j]:,}  edge/node={ev_ratios[j]:.4f}')

        # --- Issues ---
        if issues:
            for issue in issues[:20]:
                print(issue)
        else:
            print('  All samples passed topology checks')

    def prepare_preprocessing(self) -> None:
        """Fit preprocessing statistics using this dataset's sample_ids only."""
        if self.use_node_types:
            self._compute_node_type_info()

        self._compute_zscore_stats()

        if self.use_world_edges:
            self._compute_world_edge_radius()

        if self.use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True requires model/coarsening.py (import failed)")
            self._compute_coarse_edge_stats()

    def inherit_preprocessing_from(self, source_dataset) -> None:
        """Reuse preprocessing fit on another dataset, typically the train split."""
        self.node_mean = source_dataset.node_mean.copy()
        self.node_std = source_dataset.node_std.copy()
        self.edge_mean = source_dataset.edge_mean.copy()
        self.edge_std = source_dataset.edge_std.copy()
        self.delta_mean = source_dataset.delta_mean.copy()
        self.delta_std = source_dataset.delta_std.copy()
        self.num_node_types = source_dataset.num_node_types
        self.node_type_to_idx = (
            dict(source_dataset.node_type_to_idx)
            if source_dataset.node_type_to_idx is not None
            else None
        )
        self.world_edge_radius = source_dataset.world_edge_radius
        self.min_edge_length = source_dataset.min_edge_length
        self.coarse_edge_means = [m.copy() for m in source_dataset.coarse_edge_means]
        self.coarse_edge_stds = [s.copy() for s in source_dataset.coarse_edge_stds]

    def write_preprocessing_to_hdf5(self, split_seed: int) -> None:
        """Persist train-derived preprocessing statistics to the HDF5 dataset."""
        if any(value is None for value in (
            self.node_mean, self.node_std,
            self.edge_mean, self.edge_std,
            self.delta_mean, self.delta_std,
        )):
            raise RuntimeError("Cannot write preprocessing stats before prepare_preprocessing()")

        with h5py.File(self.h5_file, 'r+') as f:
            metadata = f.require_group('metadata')
            norm_group = metadata.require_group('normalization_params')

            def _write_array(name: str, value: np.ndarray) -> None:
                if name in norm_group:
                    del norm_group[name]
                norm_group.create_dataset(name, data=value.astype(np.float32))

            _write_array('node_mean', self.node_mean)
            _write_array('node_std', self.node_std)
            _write_array('edge_mean', self.edge_mean)
            _write_array('edge_std', self.edge_std)
            _write_array('delta_mean', self.delta_mean)
            _write_array('delta_std', self.delta_std)

            norm_group.attrs['edge_feature_layout'] = (
                'deformed_dx,deformed_dy,deformed_dz,deformed_dist,'
                'ref_dx,ref_dy,ref_dz,ref_dist'
            )
            norm_group.attrs['edge_var'] = self.edge_dim
            norm_group.attrs['normalization_source'] = 'train_split'
            norm_group.attrs['split_seed'] = int(split_seed)

    def _create_subset(self, sample_ids: List[int], is_training: bool = False):
        subset = MeshGraphDataset.__new__(MeshGraphDataset)
        subset.h5_file = self.h5_file
        subset.config = self.config
        subset.input_dim = self.input_dim
        subset.output_dim = self.output_dim
        subset.num_pos_features = self.num_pos_features
        subset.edge_dim = self.edge_dim
        subset.sample_ids = list(sample_ids)
        subset.num_timesteps = self.num_timesteps
        subset.use_node_types = self.use_node_types
        subset.num_node_types = None
        subset.node_type_to_idx = None
        subset.use_world_edges = self.use_world_edges
        subset.coarse_world_edges = self.coarse_world_edges
        subset.world_radius_multiplier = self.world_radius_multiplier
        subset.world_max_num_neighbors = self.world_max_num_neighbors
        subset.world_edge_backend = self.world_edge_backend
        subset.world_edge_radius = None
        subset.min_edge_length = None
        subset.delta_mean = None
        subset.delta_std = None
        subset.node_mean = None
        subset.node_std = None
        subset.edge_mean = None
        subset.edge_std = None
        subset.use_multiscale = self.use_multiscale
        subset.multiscale_levels = self.multiscale_levels
        subset.coarsening_types = self.coarsening_types
        subset.voronoi_clusters = self.voronoi_clusters
        subset.coarse_edge_means = []
        subset.coarse_edge_stds = []
        subset._static_cache = {}
        subset._static_cache_max = self._static_cache_max
        # Share the parent's on-disk cache (covers all sample_ids); reader is
        # process-local and opened lazily per worker.
        subset._ms_cache_path = self._ms_cache_path
        subset._ms_reader = None
        subset._h5_handle = None
        subset.is_training = is_training
        subset.augment_geometry = self.config.get('augment_geometry', False) and is_training
        return subset

    def _resolve_split_ids(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
        """Always generate a deterministic seeded split."""
        rng = np.random.default_rng(seed)
        shuffled_ids = self.sample_ids.copy()
        rng.shuffle(shuffled_ids)
        n_samples = len(shuffled_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val:]
        print(f"Using seeded random split (seed={seed}).")
        return train_ids, val_ids, test_ids

    def _compute_zscore_stats(self) -> None:
        """Compute train-split z-score statistics for node, edge, and delta features."""
        print('Computing z-score normalization statistics...')

        stats = compute_normalization_stats(
            self.h5_file, self.sample_ids, self.input_dim, self.output_dim,
            self.num_timesteps, self.num_pos_features,
            use_parallel=self.config.get('use_parallel_stats', True),
        )

        self.node_mean, self.node_std = finalize_moments(
            stats['node_sum'], stats['node_sumsq'], stats['node_count'])
        self.edge_mean, self.edge_std = finalize_moments(
            stats['edge_sum'], stats['edge_sumsq'], stats['edge_count'])
        self.delta_mean, self.delta_std = finalize_moments(
            stats['delta_sum'], stats['delta_sumsq'], stats['delta_count'])

        print(f'  Node features - mean: {self.node_mean}, std: {self.node_std}')
        print(f'  Edge features - mean: {self.edge_mean}, std: {self.edge_std}')
        print(f'  Delta features - mean: {self.delta_mean}, std: {self.delta_std}')
        print(f"  Delta features - min: {stats['delta_min'].astype(np.float32)}, "
              f"max: {stats['delta_max'].astype(np.float32)}")

        self._warn_on_degenerate_stats()

    def _warn_on_degenerate_stats(self) -> None:
        """Print warnings for suspicious normalization statistics."""
        print('\n  === Normalization Sanity Checks ===')
        warnings = []
        if np.any(self.node_std > 100):
            warnings.append(f"  WARNING: Very large node std (> 100): {self.node_std[self.node_std > 100]}")
        if np.any(self.node_std < 0.01):
            warnings.append(f"  WARNING: Very small node std (< 0.01): {self.node_std[self.node_std < 0.01]}")
        if np.any(self.delta_std > 100):
            warnings.append(f"  WARNING: Very large delta std (> 100): {self.delta_std[self.delta_std > 100]}")
        if np.any(self.delta_std < 0.01):
            warnings.append(f"  WARNING: Very small delta std (< 0.01): {self.delta_std[self.delta_std < 0.01]}")
        if np.any(np.abs(self.node_mean) > 10):
            warnings.append(f"  WARNING: Large node mean (|mean| > 10): {self.node_mean[np.abs(self.node_mean) > 10]}")
        if np.any(np.abs(self.delta_mean) > 10):
            warnings.append(f"  WARNING: Large delta mean (|mean| > 10): {self.delta_mean[np.abs(self.delta_mean) > 10]}")
        if np.any(self.node_std < 1e-6):
            warnings.append("  CRITICAL: Near-zero node variance - feature is constant!")
        if np.any(self.delta_std < 1e-6):
            warnings.append("  CRITICAL: Near-zero delta variance - targets are constant!")
        for w in warnings:
            print(w)
        if not warnings:
            print('  All normalization statistics look reasonable')

    def _compute_coarse_edge_stats(self) -> None:
        """
        Compute z-score normalization stats for coarse edge features and pre-populate
        the coarsening cache with per-level topology.

        Coarsening is topology-based (BFS bi-stride on mesh_edge), so the same mapping
        applies to all timesteps of a given sample. This pass runs once at startup.

        For multiscale_levels=L, stores per-level mappings:
            fine_to_coarse_{i}: level i → level i+1 (NOT composed)
            coarse_edge_index_{i}: edge topology at level i+1
            num_coarse_{i}: node count at level i+1
        """
        import time as _time
        n_samples = len(self.sample_ids)
        L = self.multiscale_levels
        print(f'Computing coarse edge normalization statistics ({n_samples} samples, {L} levels)...')

        # Per-level accumulators for edge feature stats
        level_sum = [np.zeros(self.edge_dim, dtype=np.float64) for _ in range(L)]
        level_sumsq = [np.zeros(self.edge_dim, dtype=np.float64) for _ in range(L)]
        level_count = [0] * L
        t_start = _time.time()

        with h5py.File(self.h5_file, 'r') as f:
            for i_sample, sid in enumerate(self.sample_ids):
                if i_sample % max(1, n_samples // 10) == 0 or i_sample == n_samples - 1:
                    elapsed = _time.time() - t_start
                    print(f'  Coarsening sample {i_sample+1}/{n_samples} ({elapsed:.1f}s)')

                try:
                    mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]
                except Exception as e:
                    print(f"  Warning: could not load sample {sid}: {e}")
                    continue

                data_h5 = f[f'data/{sid}/nodal_data']
                num_nodes = data_h5.shape[2]
                edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

                # Iterative coarsening: level 0 → 1 → 2 → ...
                # Get reference positions for Euclidean FPS (voronoi levels)
                first_pos_data = data_h5[:3, 0, :]  # [3, nodes] — ref xyz
                level_ref_pos = first_pos_data.T     # [N, 3]

                # Hierarchies come from the shared on-disk cache, built for all
                # sample_ids at dataset construction time.
                hierarchy = self._get_ms_reader().get_hierarchy(sid)
                actual_levels = len(hierarchy)

                # Collect coarse edge stats per level — 10 timesteps is sufficient
                max_t = min(10, self.num_timesteps)
                if self.num_timesteps > 1:
                    timesteps = np.linspace(0, self.num_timesteps - 1, max_t, dtype=int)
                else:
                    timesteps = [0]

                for t in timesteps:
                    pos_data = data_h5[:6, t, :]  # [6, nodes] — ref_pos + disp
                    ref_pos = pos_data[:3, :].T         # [N, 3]
                    deformed_pos = (pos_data[:3, :] + pos_data[3:6, :]).T  # [N, 3]

                    cur_ref, cur_def = ref_pos, deformed_pos
                    for level, entry in enumerate(hierarchy):
                        ftc_l = entry['ftc']
                        c_ei_l = entry['c_ei']
                        n_c_l = entry['n_c']
                        mode_l = entry.get('mode', 'centroid')
                        seeds_l = entry.get('seeds')

                        if mode_l == 'inherit':
                            coarse_ref = cur_ref[seeds_l]
                            coarse_def = cur_def[seeds_l]
                        else:
                            coarse_ref = compute_coarse_centroids(cur_ref, ftc_l, n_c_l)
                            coarse_def = compute_coarse_centroids(cur_def, ftc_l, n_c_l)

                        if c_ei_l.shape[1] > 0:
                            c_edge_attr = compute_edge_attr(
                                coarse_ref.astype(np.float32),
                                coarse_def.astype(np.float32),
                                c_ei_l
                            )
                            level_sum[level] += np.sum(c_edge_attr, axis=0)
                            level_sumsq[level] += np.sum(c_edge_attr ** 2, axis=0)
                            level_count[level] += c_edge_attr.shape[0]

                        cur_ref, cur_def = coarse_ref, coarse_def

        elapsed = _time.time() - t_start
        self.coarse_edge_means = []
        self.coarse_edge_stds = []
        for level in range(L):
            if level_count[level] == 0:
                print(f'  Level {level}: no coarse edge stats; falling back to fine edge stats')
                self.coarse_edge_means.append(self.edge_mean.copy())
                self.coarse_edge_stds.append(self.edge_std.copy())
            else:
                mean, std = finalize_moments(level_sum[level], level_sumsq[level], level_count[level])
                self.coarse_edge_means.append(mean)
                self.coarse_edge_stds.append(std)
                print(f'  Level {level} coarse edge stats: mean={mean}, std={std}')
        print(f'  Coarse edge stats done in {elapsed:.1f}s')

    def _compute_node_type_info(self) -> None:
        """Compute the number of unique node types from the dataset."""
        print('Computing node type information...')
        with h5py.File(self.h5_file, 'r') as f:
            # Collect unique node types from the entire current split.
            unique_types = set()
            for sid in self.sample_ids:
                nodal_data = f[f'data/{sid}/nodal_data'][:]
                node_types = nodal_data[-1, 0, :].astype(np.int32)  # Last feature, first timestep
                unique_types.update(node_types)

            sorted_types = sorted(unique_types)
            self.node_type_to_idx = {t: i for i, t in enumerate(sorted_types)}
            self.num_node_types = len(unique_types)
            print(f'  Found {self.num_node_types} unique node types: {sorted_types}')
            print(f'  Node type mapping: {self.node_type_to_idx}')

    def _compute_world_edge_radius(self) -> None:
        print('Computing world edge radius...')
        num_samples = min(10, len(self.sample_ids))
        min_lengths = []
        with h5py.File(self.h5_file, 'r') as f:
            for i in range(num_samples):
                sid = self.sample_ids[i]
                nd = f[f'data/{sid}/nodal_data'][:]
                me = f[f'data/{sid}/mesh_edge'][:]
                pos = nd[:3, 0, :].T
                lens = np.linalg.norm(pos[me[1]] - pos[me[0]], axis=1)
                min_lengths.append(np.min(lens))
        self.min_edge_length = np.min(min_lengths)
        self.world_edge_radius = self.world_radius_multiplier * self.min_edge_length
        print(f'  min_edge_length: {self.min_edge_length:.6f}')
        print(f'  world_edge_radius: {self.world_edge_radius:.6f}')

    def __len__(self) -> int:
        """
        Calculate total number of samples.

        For multi-timestep data, each sample can produce (num_timesteps - 1)
        training pairs: (t_0→t_1), (t_1→t_2), ..., (t_n-1→t_n)

        For single timestep data, returns the number of samples.
        """
        if self.num_timesteps > 1:
            # Multiple timesteps: each sample generates (T-1) training pairs
            return len(self.sample_ids) * (self.num_timesteps - 1)
        else:
            # Single timestep: static data
            return len(self.sample_ids)

    def _get_h5_handle(self):
        """Get or create persistent HDF5 file handle (one per DataLoader worker process).

        Avoids opening/closing the HDF5 file on every __getitem__ call, which is
        a major I/O bottleneck with thousands of samples and multiple workers.
        Uses SWMR (Single Writer Multiple Reader) mode for safe multi-worker access.
        """
        if not hasattr(self, '_h5_handle') or self._h5_handle is None:
            self._h5_handle = h5py.File(self.h5_file, 'r', swmr=True)
        return self._h5_handle

    def _get_ms_reader(self):
        """Lazily open the per-process on-disk hierarchy cache reader (or None)."""
        if getattr(self, '_ms_cache_path', None) is None:
            return None
        if getattr(self, '_ms_reader', None) is None:
            from general_modules.multiscale_cache import HierarchyCacheReader
            self._ms_reader = HierarchyCacheReader(self._ms_cache_path)
        return self._ms_reader

    def _get_static_sample_data(self, sample_id: int, h5_handle, data: np.ndarray):
        """Return (edge_index, x_pos, node_types) for a sample.

        edge_index is cheap (read + mirror), so it is recomputed each call rather
        than cached — caching it cost ~10-20 MB/sample × workers. Positional
        features come from the shared on-disk cache when available, else are
        computed and held in a small bounded per-worker cache.
        """
        mesh_edge = h5_handle[f'data/{sample_id}/mesh_edge'][:]  # [2, M]
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        node_types = data[-1, 0, :].astype(np.int32) if self.use_node_types else None

        x_pos = None
        if self.num_pos_features > 0:
            x_pos = self._get_positional_features(sample_id, data, edge_index)

        return edge_index, x_pos, node_types

    def _get_positional_features(self, sample_id: int, data: np.ndarray, edge_index):
        """Positional features from the on-disk cache, else compute (bounded RAM)."""
        reader = self._get_ms_reader()
        if reader is not None:
            xp = reader.get_pos(sample_id)
            if xp is not None:
                return xp

        cache = self._static_cache
        cached = cache.get(sample_id)
        if cached is not None:
            cache.pop(sample_id)
            cache[sample_id] = cached  # LRU bump
            return cached

        ref_pos_0 = data[:3, 0, :].T
        xp = compute_positional_features(ref_pos_0, edge_index, self.num_pos_features)
        if self._static_cache_max > 0:
            if len(cache) >= self._static_cache_max:
                cache.pop(next(iter(cache)))
            cache[sample_id] = xp
        return xp

    def __getstate__(self):
        """Exclude unpicklable HDF5 handles when pickling (for DataLoader workers)."""
        state = self.__dict__.copy()
        state['_h5_handle'] = None
        state['_ms_reader'] = None      # h5 handle is not picklable; workers reopen lazily
        state['_static_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        """Close persistent HDF5 handles on cleanup."""
        if hasattr(self, '_h5_handle') and self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            self._h5_handle = None
        reader = getattr(self, '_ms_reader', None)
        if reader is not None:
            reader.close()
            self._ms_reader = None

    def _random_augmentation_matrix(self) -> np.ndarray:
        """Generate a random Z-axis rotation + optional x/y reflection matrix [3, 3].

        Gravity-independent: full Z-rotation (0-360) and axis reflections are valid.
        Translation is skipped because edge features use relative positions only.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        # Independent 50% chance of x-reflection and y-reflection
        if np.random.random() < 0.5:
            R[0, :] *= -1
        if np.random.random() < 0.5:
            R[1, :] *= -1
        return R

    def __getitem__(self, idx: int) -> Data:
        """
        Load a single graph sample with optional temporal prediction.

        Dataset structure:
            data/{sample_id}/nodal_data: [7 or 8, time, N]
                Features: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
                Part number (index 7) is optional and used for visualization
            data/{sample_id}/mesh_edge: [2, M]

        Single timestep (T=1):
            x: [N, 4] normalized physical features (all zeros -> zeros)
            pos: [N, 3] positions (unnormalized)
            y: [N, 4] normalized target delta (target - input)

        Multi-timestep (T>1):
            x: [N, 4] normalized physical features at time t
            pos: [N, 3] positions at time t
            y: [N, 4] normalized target delta (state_t+1 - state_t)

        All features are normalized using statistics fit on the training split.

        Args:
            idx: Sample index

        Returns:
            Data object with normalized x, y, edge_attr, plus pos, edge_index,
            sample_id, time_idx, and optionally part_ids
        """
        # Calculate sample and timestep indices
        if self.num_timesteps > 1:
            sample_idx = idx // (self.num_timesteps - 1)
            time_idx = idx % (self.num_timesteps - 1)
        else:
            sample_idx = idx
            time_idx = 0

        sample_id = self.sample_ids[sample_idx]

        # Load data from HDF5 (persistent handle for performance — avoids open/close per sample)
        f = self._get_h5_handle()
        data = f[f'data/{sample_id}/nodal_data'][:]  # [7 or 8, time, nodes]
        edge_index, x_pos, node_types = self._get_static_sample_data(sample_id, f, data)
        part_ids = node_types  # same raw IDs, stored separately in graph for visualization

        # Transpose to [nodes, time, 7]
        data = np.transpose(data, (2, 1, 0))
        # Data shape: [nodes, time, features]

        # Extract data based on timesteps
        if self.num_timesteps == 1:  # Static case
            data_t = data[:, 0, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_phys = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)  # [N, input_var] zeros
            y_raw = data_t[:, 3:3+self.output_dim]  # [N, output_var]
            target_delta = y_raw.copy()  # Predict final displacement directly
        else:
            # Multi-timestep: state t → state t+1
            data_t = data[:, time_idx, :]  # [N, 7]
            data_t1 = data[:, time_idx + 1, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_phys = data_t[:, 3:3+self.input_dim]  # [N, input_var]
            y_raw = data_t1[:, 3:3+self.output_dim]  # [N, output_var]
            target_delta = y_raw - x_phys  # [N, output_var]

        if self.num_pos_features > 0:
            x_raw = np.concatenate([x_phys, x_pos], axis=1)  # [N, input_var + pos_features]
        else:
            x_raw = x_phys

        # Geometric augmentation: Z-axis rotation + reflection (training only)
        if getattr(self, 'augment_geometry', False):
            R = self._random_augmentation_matrix()  # [3, 3]
            pos = pos @ R.T                          # rotate reference positions
            x_raw[:, :3] = x_raw[:, :3] @ R.T       # rotate displacement components (zeros for T=1, no-op)
            target_delta[:, :3] = target_delta[:, :3] @ R.T  # rotate delta displacement
            # Scalar features (stress) and positional features are rotation-invariant, unchanged

        displacement = x_raw[:, :3]  # [N, 3] - displacement (zeros for T=1)
        deformed_pos = pos + displacement  # [N, 3] - actual mesh position at time t

        # Compute 8-D edge features from current and reference geometry.
        edge_attr_raw = compute_edge_attr(pos, deformed_pos, edge_index)

        # Apply z-score normalization to all features
        # Node features: z-score normalization

        if self.node_mean is None or self.node_std is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: node statistics are missing.")
        if self.edge_mean is None or self.edge_std is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: edge statistics are missing.")
        if self.delta_mean is None or self.delta_std is None:
            raise RuntimeError("Dataset preprocessing has not been prepared: delta statistics are missing.")

        x_norm = (x_raw - self.node_mean) / self.node_std

        # Add node types if enabled
        if self.use_node_types and node_types is not None:
            # Map node types to contiguous indices using the precomputed mapping
            # e.g., node type 3 -> index 2 if mapping is {0:0, 1:1, 3:2}
            node_type_indices = np.array([self.node_type_to_idx[t] for t in node_types], dtype=np.int32)
            # One-hot encode node types: [N] -> [N, num_node_types]
            node_type_onehot = np.zeros((len(node_types), self.num_node_types), dtype=np.float32)
            node_type_onehot[np.arange(len(node_types)), node_type_indices] = 1.0
            # Concatenate with physical features: [N, 4] + [N, num_node_types] = [N, 4+num_node_types]
            x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

        # Target features: z-score normalization using delta-specific parameters
        target_norm = (target_delta - self.delta_mean) / self.delta_std

        # Edge features: z-score normalization
        edge_attr_norm = (edge_attr_raw - self.edge_mean) / self.edge_std

        # Convert to tensors
        pos = torch.from_numpy(pos.astype(np.float32))
        x = torch.from_numpy(x_norm.astype(np.float32)) # nodal state at time t, dx, dy, dz, stress, nodal type, ...
        y = torch.from_numpy(target_norm.astype(np.float32)) # nodal state at time t+1, dx, dy, dz, stress, ...
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr_norm.astype(np.float32))

        # Convert part_ids to tensor if available
        if part_ids is not None:
            part_ids_tensor = torch.from_numpy(part_ids).long()
        else:
            part_ids_tensor = None

        # Create base Data object (MultiscaleData if multiscale enabled for proper batching)
        DataClass = MultiscaleData if self.use_multiscale else Data
        graph_data = DataClass(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sample_id=sample_id,
            time_idx=time_idx if self.num_timesteps > 1 else None,
            part_ids=part_ids_tensor
        )

        # Compute world edges if enabled
        # IMPORTANT: Use deformed_pos (reference + displacement) for collision detection
        if self.use_world_edges:
            mesh_ei_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
            world_edge_index, world_edge_attr_norm = compute_world_edges(
                pos, deformed_pos, mesh_ei_np,
                radius=self.world_edge_radius,
                max_num_neighbors=self.world_max_num_neighbors,
                backend=self.world_edge_backend,
                edge_mean=self.edge_mean, edge_std=self.edge_std,
            )
            graph_data.world_edge_index = torch.from_numpy(world_edge_index).long()
            graph_data.world_edge_attr = torch.from_numpy(world_edge_attr_norm.astype(np.float32))
        else:
            graph_data.world_edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_data.world_edge_attr = torch.zeros((0, self.edge_dim), dtype=torch.float32)

        # Attach per-level coarsening data (only when use_multiscale=True).
        # Hierarchies always come from the shared on-disk cache.
        if self.use_multiscale:
            hierarchy = self._get_ms_reader().get_hierarchy(sample_id)

            world_ei_for_coarse = (
                graph_data.world_edge_index.numpy()
                if self.use_world_edges and self.coarse_world_edges else None
            )
            attach_coarse_levels_to_graph(
                graph_data, hierarchy,
                pos.numpy(), deformed_pos.astype(np.float32),
                self.coarse_edge_means, self.coarse_edge_stds,
                world_edge_index=world_ei_for_coarse,
            )

        return graph_data

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 0):
        """
        Split dataset into train, validation, and test sets.

        Args:
            train_ratio: Fraction of data for training (e.g., 0.8)
            val_ratio: Fraction of data for validation (e.g., 0.1)
            test_ratio: Fraction of data for testing (e.g., 0.1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        train_ids, val_ids, test_ids = self._resolve_split_ids(train_ratio, val_ratio, test_ratio, seed)

        train_dataset = self._create_subset(train_ids, is_training=True)
        val_dataset = self._create_subset(val_ids, is_training=False)
        test_dataset = self._create_subset(test_ids, is_training=False)

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
        print("Fitting preprocessing on train split only...")
        train_dataset.prepare_preprocessing()
        val_dataset.inherit_preprocessing_from(train_dataset)
        test_dataset.inherit_preprocessing_from(train_dataset)

        # Diagnostic: check node count and delta magnitude across splits
        try:
            import h5py as _h5
            with _h5.File(self.h5_file, 'r') as _f:
                _train_nc = [_f[f'data/{s}/nodal_data'].shape[2] for s in train_ids[:100]]
                _val_nc = [_f[f'data/{s}/nodal_data'].shape[2] for s in val_ids[:100]]
                _test_nc = [_f[f'data/{s}/nodal_data'].shape[2] for s in test_ids[:100]]
                print(f"  Node counts (up to 100 samples):")
                print(f"    Train: min={min(_train_nc)}, max={max(_train_nc)}, mean={sum(_train_nc)/len(_train_nc):.0f}")
                print(f"    Val:   min={min(_val_nc)}, max={max(_val_nc)}, mean={sum(_val_nc)/len(_val_nc):.0f}")
                print(f"    Test:  min={min(_test_nc)}, max={max(_test_nc)}, mean={sum(_test_nc)/len(_test_nc):.0f}")
        except Exception:
            pass

        return train_dataset, val_dataset, test_dataset
