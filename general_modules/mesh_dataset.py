import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Set, Tuple, List
from torch_geometric.data import Data
from scipy.spatial import KDTree
import multiprocessing as mp

# Try to import torch_cluster for GPU acceleration; fall back to scipy.KDTree if unavailable
try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


def _welford_update(stats, batch):
    """Update running (n, mean, M2) with a new batch [B, D]. O(D) memory."""
    b = batch.shape[0]
    if b == 0:
        return stats
    b_mean = np.mean(batch, axis=0, dtype=np.float64)
    b_M2 = np.var(batch, axis=0, dtype=np.float64) * b
    if stats is None:
        return (b, b_mean, b_M2)
    n_a, mean_a, M2_a = stats
    n = n_a + b
    delta = b_mean - mean_a
    mean = mean_a + delta * (b / n)
    M2 = M2_a + b_M2 + delta ** 2 * (n_a * b / n)
    return (n, mean, M2)


def _welford_combine(stats_a, stats_b):
    """Merge two sets of Welford statistics from parallel workers."""
    if stats_a is None:
        return stats_b
    if stats_b is None:
        return stats_a
    n_a, mean_a, M2_a = stats_a
    n_b, mean_b, M2_b = stats_b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    M2 = M2_a + M2_b + delta ** 2 * (n_a * n_b / n)
    return (n, mean, M2)


def _welford_finalize(stats):
    """Extract (mean, std) as float32 from Welford stats. Clamps std >= 1e-8."""
    n, mean, M2 = stats
    std = np.sqrt(M2 / n)
    return mean.astype(np.float32), np.maximum(std.astype(np.float32), 1e-8)


def _process_sample_chunk(h5_file: str, sample_ids: List[int], input_dim: int,
                          output_dim: int, num_timesteps: int) -> Dict:
    """
    Worker function to process a chunk of samples using streaming statistics.

    Uses Welford's online algorithm so each worker only keeps O(features) memory
    instead of accumulating all raw data. Returns compact (n, mean, M2) tuples.

    Args:
        h5_file: Path to HDF5 dataset file
        sample_ids: List of sample IDs to process
        input_dim: Number of input features (typically 4)
        output_dim: Number of output features (typically 4)
        num_timesteps: Number of timesteps in the dataset

    Returns:
        Dictionary with Welford stats for node/edge/delta features plus delta min/max.
    """
    node_stats = None
    edge_stats = None
    delta_stats = [None for _ in range(output_dim)]
    delta_min = np.full(output_dim, np.inf, dtype=np.float64)
    delta_max = np.full(output_dim, -np.inf, dtype=np.float64)

    try:
        with h5py.File(h5_file, 'r') as f:
            for sid in sample_ids:
                try:
                    data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
                    mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]

                    max_timesteps_for_stats = 500
                    if num_timesteps > 1:
                        num_samples_t = min(max_timesteps_for_stats, num_timesteps)
                        timesteps = np.linspace(0, num_timesteps - 1, num_samples_t, dtype=int)
                    else:
                        timesteps = [0]

                    edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

                    for t in timesteps:
                        node_feat = data[3:3+input_dim, t, :].T  # [N, input_dim]
                        node_stats = _welford_update(node_stats, node_feat)

                        pos = (data[:3, t, :] + data[3:6, t, :]).T  # [N, 3]
                        rel_pos = pos[edge_idx[1]] - pos[edge_idx[0]]
                        dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
                        edge_feat = np.concatenate([rel_pos, dist], axis=1)  # [2E, 4]
                        edge_stats = _welford_update(edge_stats, edge_feat)

                    # Delta features
                    if num_timesteps > 1:
                        num_delta_samples = min(max_timesteps_for_stats, num_timesteps - 1)
                        delta_timesteps = np.linspace(0, num_timesteps - 2, num_delta_samples, dtype=int)
                        for t in delta_timesteps:
                            for feat_idx in range(output_dim):
                                delta = data[3 + feat_idx, t + 1, :] - data[3 + feat_idx, t, :]
                                delta_2d = delta.reshape(-1, 1)
                                delta_stats[feat_idx] = _welford_update(delta_stats[feat_idx], delta_2d)
                                delta_min[feat_idx] = min(delta_min[feat_idx], float(np.min(delta)))
                                delta_max[feat_idx] = max(delta_max[feat_idx], float(np.max(delta)))
                    else:
                        for feat_idx in range(output_dim):
                            feat = data[3 + feat_idx, 0, :]
                            feat_2d = feat.reshape(-1, 1)
                            delta_stats[feat_idx] = _welford_update(delta_stats[feat_idx], feat_2d)
                            delta_min[feat_idx] = min(delta_min[feat_idx], float(np.min(feat)))
                            delta_max[feat_idx] = max(delta_max[feat_idx], float(np.max(feat)))

                except Exception as e:
                    print(f"Warning: Failed to process sample {sid}: {e}")
                    continue

        return {
            'node_stats': node_stats,
            'edge_stats': edge_stats,
            'delta_stats': delta_stats,
            'delta_min': delta_min,
            'delta_max': delta_max,
            'num_samples_processed': len(sample_ids)
        }

    except Exception as e:
        print(f"Error in worker process: {e}")
        return {
            'node_stats': None,
            'edge_stats': None,
            'delta_stats': [None for _ in range(output_dim)],
            'delta_min': np.full(output_dim, np.inf, dtype=np.float64),
            'delta_max': np.full(output_dim, -np.inf, dtype=np.float64),
            'num_samples_processed': 0
        }


class MeshGraphDataset(Dataset):

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        # Graph and feature parameters
        self.input_dim = config.get('input_var')  # Physical features only (4)
        self.output_dim = config.get('output_var')  # Physical features only (4)

        # Node type parameters
        self.use_node_types = config.get('use_node_types', False)
        self.num_node_types = None  # Will be computed from dataset if node types exist
        self.node_type_to_idx = None  # Mapping from node type values to contiguous indices

        # World edge parameters
        self.use_world_edges = config.get('use_world_edges', False)
        self.world_radius_multiplier = config.get('world_radius_multiplier', 1.5)
        self.world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
        self.world_edge_radius = None  # Computed from mesh statistics
        self.min_edge_length = None    # Computed from first sample

        # Determine which world edge backend to use
        requested_backend = config.get('world_edge_backend', 'torch_cluster').lower()
        if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'torch_cluster'
        elif requested_backend == 'scipy_kdtree' or not HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'scipy_kdtree'
        else:
            # Invalid backend requested, default to available option
            self.world_edge_backend = 'torch_cluster' if HAS_TORCH_CLUSTER else 'scipy_kdtree'

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}")
        print(f"  use_node_types: {self.use_node_types}")
        print(f"  use_world_edges: {self.use_world_edges}")
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

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}")

        # Compute z-score normalization statistics for node and edge features
        self._compute_zscore_stats()

        if self.use_node_types:
            self._compute_node_type_info()

        if self.use_world_edges:
            self._compute_world_edge_radius()

    def _compute_zscore_stats(self) -> None:
        """Compute z-score normalization statistics (mean, std) for node, edge, and delta features.

        Also updates the HDF5 file with correct delta normalization parameters computed from
        actual data, fixing any incorrect pre-stored values.

        Supports parallel processing for large datasets via config option 'use_parallel_stats'.
        """
        print('Computing z-score normalization statistics...')

        num_samples = len(self.sample_ids)

        # Check if parallel processing is enabled and beneficial
        use_parallel = self.config.get('use_parallel_stats', True)  # Default: enabled
        min_samples_for_parallel = 10  # Don't parallelize for small datasets

        # Determine number of workers
        if use_parallel and num_samples >= min_samples_for_parallel:
            # Use 80% of available cores, minimum 1, maximum 8
            num_workers = max(1, min(64, int(mp.cpu_count() * 0.45)))
        else:
            num_workers = 1

        if num_workers > 1:
            print(f'  Using parallel processing: {num_workers} workers for {num_samples} samples')
            stats = self._compute_stats_parallel(num_workers, num_samples)
        else:
            if not use_parallel:
                print(f'  Parallel processing disabled (use_parallel_stats=False)')
            else:
                print(f'  Using serial processing ({num_samples} samples, threshold={min_samples_for_parallel})')
            stats = self._compute_stats_serial(num_samples)

        # Finalize node and edge statistics from Welford accumulators
        self.node_mean, self.node_std = _welford_finalize(stats['node_stats'])
        self.edge_mean, self.edge_std = _welford_finalize(stats['edge_stats'])

        print(f'  Node features - mean: {self.node_mean}, std: {self.node_std}')
        print(f'  Edge features - mean: {self.edge_mean}, std: {self.edge_std}')

        # Finalize delta statistics per feature
        self.delta_mean = np.zeros(self.output_dim, dtype=np.float32)
        self.delta_std = np.zeros(self.output_dim, dtype=np.float32)
        delta_min = stats['delta_min'].astype(np.float32)
        delta_max = stats['delta_max'].astype(np.float32)

        for feat_idx in range(self.output_dim):
            mean, std = _welford_finalize(stats['delta_stats'][feat_idx])
            self.delta_mean[feat_idx] = mean[0]
            self.delta_std[feat_idx] = std[0]

        print(f'  Delta features - mean: {self.delta_mean}, std: {self.delta_std}')
        print(f'  Delta features - min: {delta_min}, max: {delta_max}')

        # Sanity checks for normalization
        print('\n  === Normalization Sanity Checks ===')
        warnings = []

        # Check for extreme std values
        if np.any(self.node_std > 100):
            warnings.append(f"  ⚠️  WARNING: Very large node std detected (> 100): {self.node_std[self.node_std > 100]}")
        if np.any(self.node_std < 0.01):
            warnings.append(f"  ⚠️  WARNING: Very small node std detected (< 0.01): {self.node_std[self.node_std < 0.01]}")

        if np.any(self.delta_std > 100):
            warnings.append(f"  ⚠️  WARNING: Very large delta std detected (> 100): {self.delta_std[self.delta_std > 100]}")
        if np.any(self.delta_std < 0.01):
            warnings.append(f"  ⚠️  WARNING: Very small delta std detected (< 0.01): {self.delta_std[self.delta_std < 0.01]}")

        # Check for extreme mean values
        if np.any(np.abs(self.node_mean) > 10):
            warnings.append(f"  ⚠️  WARNING: Large node mean detected (|mean| > 10): {self.node_mean[np.abs(self.node_mean) > 10]}")
        if np.any(np.abs(self.delta_mean) > 10):
            warnings.append(f"  ⚠️  WARNING: Large delta mean detected (|mean| > 10): {self.delta_mean[np.abs(self.delta_mean) > 10]}")

        # Check for near-zero variance (constant features)
        if np.any(self.node_std < 1e-6):
            warnings.append(f"  ⚠️  CRITICAL: Near-zero node variance detected - feature is constant!")
        if np.any(self.delta_std < 1e-6):
            warnings.append(f"  ⚠️  CRITICAL: Near-zero delta variance detected - targets are constant!")

        if warnings:
            for w in warnings:
                print(w)
        else:
            print('  ✓ All normalization statistics look reasonable')

        # Update HDF5 file with computed normalization parameters
        self._update_hdf5_normalization_params(delta_min, delta_max)

    def _update_hdf5_normalization_params(self, delta_min: np.ndarray, delta_max: np.ndarray) -> None:
        """Update HDF5 file with computed delta normalization parameters.

        Uses pre-computed delta_min/delta_max from the streaming stats pass,
        avoiding a redundant full re-read of all samples.
        """
        try:
            with h5py.File(self.h5_file, 'r+') as f:
                if 'metadata/normalization_params' not in f:
                    f.create_group('metadata/normalization_params')

                norm_params = f['metadata/normalization_params']

                if 'delta_mean' in norm_params and 'delta_std' in norm_params:
                    norm_params['delta_mean'][...] = self.delta_mean
                    norm_params['delta_std'][...] = self.delta_std

                    if 'delta_max' in norm_params and 'delta_min' in norm_params:
                        norm_params['delta_max'][...] = delta_max
                        norm_params['delta_min'][...] = delta_min

                    print(f'  [OK] HDF5 delta normalization parameters updated successfully')
        except (OSError, BlockingIOError):
            print(f'  [INFO] Could not update HDF5 normalization params (file locked by another process)')

    def _compute_stats_serial(self, num_samples: int) -> Dict:
        """Serial streaming statistics using Welford's online algorithm. O(features) memory."""
        node_stats = None
        edge_stats = None
        delta_stats = [None for _ in range(self.output_dim)]
        delta_min = np.full(self.output_dim, np.inf, dtype=np.float64)
        delta_max = np.full(self.output_dim, -np.inf, dtype=np.float64)

        max_timesteps_for_stats = 500

        with h5py.File(self.h5_file, 'r') as f:
            for i in range(num_samples):
                sid = self.sample_ids[i]
                data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
                mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]

                if self.num_timesteps > 1:
                    num_samples_t = min(max_timesteps_for_stats, self.num_timesteps)
                    timesteps = np.linspace(0, self.num_timesteps - 1, num_samples_t, dtype=int)
                else:
                    timesteps = [0]

                edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

                for t in timesteps:
                    node_feat = data[3:3+self.input_dim, t, :].T  # [N, input_dim]
                    node_stats = _welford_update(node_stats, node_feat)

                    pos = (data[:3, t, :] + data[3:6, t, :]).T  # [N, 3]
                    rel_pos = pos[edge_idx[1]] - pos[edge_idx[0]]
                    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
                    edge_feat = np.concatenate([rel_pos, dist], axis=1)  # [2E, 4]
                    edge_stats = _welford_update(edge_stats, edge_feat)

                # Delta features
                if self.num_timesteps > 1:
                    num_delta_samples = min(max_timesteps_for_stats, self.num_timesteps - 1)
                    delta_timesteps = np.linspace(0, self.num_timesteps - 2, num_delta_samples, dtype=int)
                    for t in delta_timesteps:
                        for feat_idx in range(self.output_dim):
                            delta = data[3 + feat_idx, t + 1, :] - data[3 + feat_idx, t, :]
                            delta_2d = delta.reshape(-1, 1)
                            delta_stats[feat_idx] = _welford_update(delta_stats[feat_idx], delta_2d)
                            delta_min[feat_idx] = min(delta_min[feat_idx], float(np.min(delta)))
                            delta_max[feat_idx] = max(delta_max[feat_idx], float(np.max(delta)))
                else:
                    for feat_idx in range(self.output_dim):
                        feat = data[3 + feat_idx, 0, :]
                        feat_2d = feat.reshape(-1, 1)
                        delta_stats[feat_idx] = _welford_update(delta_stats[feat_idx], feat_2d)
                        delta_min[feat_idx] = min(delta_min[feat_idx], float(np.min(feat)))
                        delta_max[feat_idx] = max(delta_max[feat_idx], float(np.max(feat)))

        return {
            'node_stats': node_stats,
            'edge_stats': edge_stats,
            'delta_stats': delta_stats,
            'delta_min': delta_min,
            'delta_max': delta_max,
        }

    def _compute_stats_parallel(self, num_workers: int, num_samples: int) -> Dict:
        """Parallel streaming statistics — each worker returns compact Welford stats, master merges."""
        chunk_size = max(1, num_samples // num_workers)
        sample_chunks = []
        for i in range(0, num_samples, chunk_size):
            sample_chunks.append(self.sample_ids[i:i+chunk_size])

        try:
            with mp.Pool(num_workers) as pool:
                worker_args = [
                    (self.h5_file, chunk, self.input_dim, self.output_dim, self.num_timesteps)
                    for chunk in sample_chunks
                ]
                results = pool.starmap(_process_sample_chunk, worker_args)

            # Merge Welford stats from all workers — O(workers * features) memory
            node_stats = None
            edge_stats = None
            delta_stats = [None for _ in range(self.output_dim)]
            delta_min = np.full(self.output_dim, np.inf, dtype=np.float64)
            delta_max = np.full(self.output_dim, -np.inf, dtype=np.float64)

            total_processed = 0
            for result in results:
                if result['num_samples_processed'] > 0:
                    node_stats = _welford_combine(node_stats, result['node_stats'])
                    edge_stats = _welford_combine(edge_stats, result['edge_stats'])
                    for feat_idx in range(self.output_dim):
                        delta_stats[feat_idx] = _welford_combine(
                            delta_stats[feat_idx], result['delta_stats'][feat_idx])
                    np.minimum(delta_min, result['delta_min'], out=delta_min)
                    np.maximum(delta_max, result['delta_max'], out=delta_max)
                    total_processed += result['num_samples_processed']

            print(f'  Successfully processed {total_processed}/{num_samples} samples')

            if total_processed == 0:
                raise RuntimeError("No samples were successfully processed in parallel mode")

            return {
                'node_stats': node_stats,
                'edge_stats': edge_stats,
                'delta_stats': delta_stats,
                'delta_min': delta_min,
                'delta_max': delta_max,
            }

        except Exception as e:
            print(f'  Warning: Parallel processing failed ({e}), falling back to serial processing')
            return self._compute_stats_serial(num_samples)

    def _compute_node_type_info(self) -> None:
        """Compute the number of unique node types from the dataset."""
        print('Computing node type information...')
        with h5py.File(self.h5_file, 'r') as f:
            # Collect unique node types from first few samples
            # Node types are always the last feature
            unique_types = set()
            num_samples = min(10, len(self.sample_ids))
            for i in range(num_samples):
                sid = self.sample_ids[i]
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

    def _compute_world_edges(self, pos, mesh_edges):
        """
        Compute world edges using either torch_cluster (GPU) or scipy.KDTree (CPU).

        Supports two backends:
        - 'torch_cluster': GPU-accelerated (5-10x faster for 68k nodes)
        - 'scipy_kdtree': CPU-based fallback (original implementation)

        Args:
            pos: (N, 3) array of node positions
            mesh_edges: (2, E_mesh) array of existing mesh edge indices

        Returns:
            world_edge_index: (2, E_world) array of world edge indices
            world_edge_attr: (E_world, 4) array with [dx, dy, dz, distance]
        """
        if not self.world_edge_radius:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        if self.world_edge_backend == 'torch_cluster':
            return self._compute_world_edges_torch_cluster(pos, mesh_edges)
        else:
            return self._compute_world_edges_scipy_kdtree(pos, mesh_edges)

    def _compute_world_edges_torch_cluster(self, pos, mesh_edges):
        """
        Compute world edges using GPU-accelerated torch_cluster.radius_graph().
        Expected 5-10x speedup for 68k-node meshes compared to scipy.KDTree.
        """
        # Convert positions to GPU tensor
        pos_tensor = torch.from_numpy(pos).float().cuda()

        # GPU-accelerated radius query (torch_cluster)
        world_edges = radius_graph(
            x=pos_tensor,
            r=self.world_edge_radius,
            batch=None,                           # Single sample (no batch tensor)
            loop=False,                           # No self-loops
            max_num_neighbors=self.world_max_num_neighbors
        )

        # Convert back to numpy for edge filtering
        world_edges_np = world_edges.cpu().numpy()

        if world_edges_np.shape[1] == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Efficient filtering: remove edges that already exist in mesh topology
        mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i])) for i in range(mesh_edges.shape[1])}

        valid_mask = np.array([
            (world_edges_np[0, i], world_edges_np[1, i]) not in mesh_set
            for i in range(world_edges_np.shape[1])
        ])

        we = world_edges_np[:, valid_mask]

        if we.shape[1] == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Compute edge features: [relative_position, distance]
        rel = pos[we[1]] - pos[we[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)

        return we, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def _compute_world_edges_scipy_kdtree(self, pos, mesh_edges):
        """
        Compute world edges using scipy.spatial.KDTree (CPU fallback).
        Original implementation, slower but always available.
        """
        tree = KDTree(pos)
        pairs = tree.query_pairs(r=self.world_edge_radius, output_type='ndarray')

        if len(pairs) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Filter out existing mesh edges
        mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i])) for i in range(mesh_edges.shape[1])}
        we = []
        for s, r in pairs:
            if (s, r) not in mesh_set:
                we.append([s, r])
            if (r, s) not in mesh_set:
                we.append([r, s])

        if not we:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        wei = np.array(we, dtype=np.int64).T
        rel = pos[wei[1]] - pos[wei[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)

        return wei, np.concatenate([rel, dist], axis=1).astype(np.float32)

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

    def __del__(self):
        """Close persistent HDF5 handle on cleanup."""
        if hasattr(self, '_h5_handle') and self._h5_handle is not None:
            try:
                self._h5_handle.close()
            except Exception:
                pass
            self._h5_handle = None

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

        All features are normalized using precomputed global statistics.

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
        edge_index = f[f'data/{sample_id}/mesh_edge'][:]  # [2, M]

        if self.config['use_node_types']:
            has_part_info = True
        else:
            has_part_info = False

        if has_part_info:
            part_ids = data[-1, 0, :].astype(np.int32)  # [nodes]
        else:
            part_ids = None

        if self.use_node_types:
            node_types = data[-1, 0, :].astype(np.int32)  # [nodes]
        else:
            node_types = None

            # Essentially, node_types are part_ids

        # Make edges bidirectional (like DeepMind's MeshGraphNets implementation)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)  # [2, 2M]

        # Transpose to [nodes, time, 7]
        data = np.transpose(data, (2, 1, 0))
        # Data shape: [nodes, time, features]

        # Extract data based on timesteps
        if self.num_timesteps == 1: # Static case
            # Single timestep: geometry → physics
            data_t = data[:, 0, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)  # [N, 4] zeros
            y_raw = data_t[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: y - x (for single timestep, x is zeros so delta = y)
            target_delta = y_raw - x_raw  # [N, 4], not including node types
        else:
            # Multi-timestep: state t → state t+1
            data_t = data[:, time_idx, :]  # [N, 7]
            data_t1 = data[:, time_idx + 1, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = data_t[:, 3:3+self.input_dim]  # [N, 4]
            y_raw = data_t1[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: difference between next and current state
            target_delta = y_raw - x_raw  # [N, 4]

        displacement = x_raw[:, :3]  # [N, 3] - extract displacement components (x_disp, y_disp, z_disp)
        deformed_pos = pos + displacement  # [N, 3] - actual mesh position at time t

        # Compute edge features (before normalization)
        # Edge features are computed for all edges (including reverse edges)
        # Reverse edges naturally get negated relative_pos since src/dst are swapped
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        relative_pos = deformed_pos[dst_idx] - deformed_pos[src_idx]  # [2M, 3]
        distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # [2M, 1]
        edge_attr_raw = np.concatenate([relative_pos, distance], axis=1)  # [2M, 4]

        # Apply z-score normalization to all features
        # Node features: z-score normalization

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
        if self.delta_mean is not None and self.delta_std is not None:
            target_norm = (target_delta - self.delta_mean) / self.delta_std
        else:
            # Fallback: use node stats
            target_norm = (target_delta - self.node_mean) / self.node_std

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

        # Create base Data object
        graph_data = Data(
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
            world_edge_index, world_edge_attr_raw = self._compute_world_edges(
                deformed_pos,  # Use DEFORMED position (pos + displacement), not reference
                edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
            )
            # Normalize world edge features using the same statistics as mesh edges
            # This fixes the scale mismatch between normalized mesh edges and raw world edges
            if world_edge_attr_raw.shape[0] > 0:
                world_edge_attr_norm = (world_edge_attr_raw - self.edge_mean) / self.edge_std
            else:
                world_edge_attr_norm = world_edge_attr_raw
            graph_data.world_edge_index = torch.from_numpy(world_edge_index).long()
            graph_data.world_edge_attr = torch.from_numpy(world_edge_attr_norm.astype(np.float32))
        else:
            graph_data.world_edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_data.world_edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        return graph_data

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
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

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Shuffle sample IDs
        shuffled_ids = self.sample_ids.copy()
        np.random.shuffle(shuffled_ids)

        # Calculate split sizes
        n_samples = len(shuffled_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Split IDs
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val:]

        # Create subset datasets (copy all attributes including normalization params)
        train_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        train_dataset.h5_file = self.h5_file
        train_dataset.config = self.config
        train_dataset.input_dim = self.input_dim
        train_dataset.output_dim = self.output_dim
        train_dataset.sample_ids = train_ids
        train_dataset.num_timesteps = self.num_timesteps
        train_dataset.use_node_types = self.use_node_types
        train_dataset.num_node_types = self.num_node_types
        train_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        train_dataset.use_world_edges = self.use_world_edges
        train_dataset.world_radius_multiplier = self.world_radius_multiplier
        train_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        train_dataset.world_edge_backend = self.world_edge_backend
        train_dataset.world_edge_radius = self.world_edge_radius
        train_dataset.min_edge_length = self.min_edge_length
        train_dataset.delta_mean = self.delta_mean
        train_dataset.delta_std = self.delta_std
        train_dataset.node_mean = self.node_mean
        train_dataset.node_std = self.node_std
        train_dataset.edge_mean = self.edge_mean
        train_dataset.edge_std = self.edge_std

        val_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        val_dataset.h5_file = self.h5_file
        val_dataset.config = self.config
        val_dataset.input_dim = self.input_dim
        val_dataset.output_dim = self.output_dim
        val_dataset.sample_ids = val_ids
        val_dataset.num_timesteps = self.num_timesteps
        val_dataset.use_node_types = self.use_node_types
        val_dataset.num_node_types = self.num_node_types
        val_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        val_dataset.use_world_edges = self.use_world_edges
        val_dataset.world_radius_multiplier = self.world_radius_multiplier
        val_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        val_dataset.world_edge_backend = self.world_edge_backend
        val_dataset.world_edge_radius = self.world_edge_radius
        val_dataset.min_edge_length = self.min_edge_length
        val_dataset.delta_mean = self.delta_mean
        val_dataset.delta_std = self.delta_std
        val_dataset.node_mean = self.node_mean
        val_dataset.node_std = self.node_std
        val_dataset.edge_mean = self.edge_mean
        val_dataset.edge_std = self.edge_std

        test_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        test_dataset.h5_file = self.h5_file
        test_dataset.config = self.config
        test_dataset.input_dim = self.input_dim
        test_dataset.output_dim = self.output_dim
        test_dataset.sample_ids = test_ids
        test_dataset.num_timesteps = self.num_timesteps
        test_dataset.use_node_types = self.use_node_types
        test_dataset.num_node_types = self.num_node_types
        test_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        test_dataset.use_world_edges = self.use_world_edges
        test_dataset.world_radius_multiplier = self.world_radius_multiplier
        test_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        test_dataset.world_edge_backend = self.world_edge_backend
        test_dataset.world_edge_radius = self.world_edge_radius
        test_dataset.min_edge_length = self.min_edge_length
        test_dataset.delta_mean = self.delta_mean
        test_dataset.delta_std = self.delta_std
        test_dataset.node_mean = self.node_mean
        test_dataset.node_std = self.node_std
        test_dataset.edge_mean = self.edge_mean
        test_dataset.edge_std = self.edge_std

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

        return train_dataset, val_dataset, test_dataset
