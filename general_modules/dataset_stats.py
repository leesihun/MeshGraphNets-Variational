"""Normalization-statistics computation for MeshGraphDataset.

Accumulates element-weighted first/second moments over all nodes, edges, and
target deltas of a sample split, optionally in parallel across processes.
This matches MeshGraphNets' online training normalizers more closely than
per-sample averaging.
"""

import multiprocessing as mp
from typing import Dict, List

import h5py
import numpy as np

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features

# Sub-sample long trajectories: statistics converge well before 500 timesteps.
MAX_TIMESTEPS_FOR_STATS = 500

# Don't parallelize tiny datasets; spawn overhead dominates.
_MIN_SAMPLES_FOR_PARALLEL = 10


def finalize_moments(feature_sum, feature_sumsq, count):
    """Finalize element-weighted mean/std from sums and squared sums."""
    if count <= 0:
        raise ValueError("Cannot finalize normalization statistics with count <= 0")
    mean = (feature_sum / count).astype(np.float32)
    meansq = (feature_sumsq / count).astype(np.float32)
    var = np.maximum(meansq - mean ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    return mean, np.maximum(std, 1e-8)


def _empty_accumulators(node_dim: int, output_dim: int) -> Dict:
    return {
        'node_sum': np.zeros(node_dim, dtype=np.float64),
        'node_sumsq': np.zeros(node_dim, dtype=np.float64),
        'node_count': 0,
        'edge_sum': np.zeros(EDGE_FEATURE_DIM, dtype=np.float64),
        'edge_sumsq': np.zeros(EDGE_FEATURE_DIM, dtype=np.float64),
        'edge_count': 0,
        'delta_sum': np.zeros(output_dim, dtype=np.float64),
        'delta_sumsq': np.zeros(output_dim, dtype=np.float64),
        'delta_count': 0,
        'delta_min': np.full(output_dim, np.inf, dtype=np.float64),
        'delta_max': np.full(output_dim, -np.inf, dtype=np.float64),
        'num_samples_processed': 0,
    }


def _process_sample_chunk(h5_file: str, sample_ids: List[int], input_dim: int,
                          output_dim: int, num_timesteps: int,
                          num_pos_features: int = 0) -> Dict:
    """Accumulate stats over one chunk of samples. Runs in a pool worker."""
    acc = _empty_accumulators(input_dim + num_pos_features, output_dim)

    with h5py.File(h5_file, 'r') as f:
        for sid in sample_ids:
            try:
                data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
                mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]

                if num_timesteps > 1:
                    n_t = min(MAX_TIMESTEPS_FOR_STATS, num_timesteps)
                    timesteps = np.linspace(0, num_timesteps - 1, n_t, dtype=int)
                else:
                    timesteps = [0]

                edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

                # Positional features depend on topology only — compute once per sample
                pos_feat = None
                if num_pos_features > 0:
                    ref_pos_0 = data[:3, 0, :].T
                    pos_feat = compute_positional_features(ref_pos_0, edge_idx, num_pos_features)

                for t in timesteps:
                    ref_pos = data[:3, t, :].T  # [N, 3]

                    if num_timesteps == 1:
                        # Static: physical features are zeros (model sees zeros)
                        node_feat = np.zeros((data.shape[2], input_dim), dtype=np.float64)
                        deformed_pos = ref_pos  # no displacement
                    else:
                        node_feat = data[3:3 + input_dim, t, :].T  # [N, input_dim]
                        deformed_pos = ref_pos + data[3:6, t, :].T

                    if pos_feat is not None:
                        node_feat = np.concatenate([node_feat, pos_feat], axis=1)

                    acc['node_sum'] += np.sum(node_feat, axis=0)
                    acc['node_sumsq'] += np.sum(node_feat ** 2, axis=0)
                    acc['node_count'] += node_feat.shape[0]

                    edge_feat = compute_edge_attr(ref_pos, deformed_pos, edge_idx)
                    acc['edge_sum'] += np.sum(edge_feat, axis=0)
                    acc['edge_sumsq'] += np.sum(edge_feat ** 2, axis=0)
                    acc['edge_count'] += edge_feat.shape[0]

                # Target deltas
                if num_timesteps > 1:
                    n_d = min(MAX_TIMESTEPS_FOR_STATS, num_timesteps - 1)
                    delta_timesteps = np.linspace(0, num_timesteps - 2, n_d, dtype=int)
                    deltas = (
                        (data[3:3 + output_dim, t + 1, :] - data[3:3 + output_dim, t, :]).T
                        for t in delta_timesteps
                    )
                else:
                    deltas = (data[3:3 + output_dim, 0, :].T,)

                for delta in deltas:
                    acc['delta_sum'] += np.sum(delta, axis=0)
                    acc['delta_sumsq'] += np.sum(delta ** 2, axis=0)
                    acc['delta_min'] = np.minimum(acc['delta_min'], np.min(delta, axis=0))
                    acc['delta_max'] = np.maximum(acc['delta_max'], np.max(delta, axis=0))
                    acc['delta_count'] += delta.shape[0]

                acc['num_samples_processed'] += 1

            except Exception as e:
                print(f"Warning: Failed to process sample {sid}: {e}")
                continue

    return acc


def _merge_accumulators(results: List[Dict], node_dim: int, output_dim: int) -> Dict:
    merged = _empty_accumulators(node_dim, output_dim)
    for r in results:
        if r['num_samples_processed'] == 0:
            continue
        for key in ('node_sum', 'node_sumsq', 'edge_sum', 'edge_sumsq',
                    'delta_sum', 'delta_sumsq'):
            merged[key] += r[key]
        for key in ('node_count', 'edge_count', 'delta_count', 'num_samples_processed'):
            merged[key] += r[key]
        np.minimum(merged['delta_min'], r['delta_min'], out=merged['delta_min'])
        np.maximum(merged['delta_max'], r['delta_max'], out=merged['delta_max'])
    return merged


def compute_normalization_stats(h5_file: str, sample_ids: List[int], input_dim: int,
                                output_dim: int, num_timesteps: int,
                                num_pos_features: int, use_parallel: bool = True) -> Dict:
    """Compute raw stat sums over a sample split, in parallel when worthwhile.

    Returns the accumulator dict (see `_empty_accumulators`); use
    `finalize_moments` on the sum/sumsq/count triples to get mean/std.
    """
    n = len(sample_ids)
    if use_parallel and n >= _MIN_SAMPLES_FOR_PARALLEL:
        # Cap at 8: spawn-context pool workers each re-import torch, so a
        # large pool risks OOM-killed workers and a permanent starmap hang.
        num_workers = max(1, min(8, int(mp.cpu_count() * 0.45)))
    else:
        num_workers = 1

    if num_workers <= 1:
        reason = ('disabled (use_parallel_stats=False)' if not use_parallel
                  else f'serial ({n} samples < {_MIN_SAMPLES_FOR_PARALLEL})')
        print(f'  Normalization stats: {reason}')
        return _process_sample_chunk(h5_file, sample_ids, input_dim, output_dim,
                                     num_timesteps, num_pos_features)

    print(f'  Normalization stats: {num_workers} parallel workers for {n} samples')
    chunk_size = max(1, n // num_workers)
    chunks = [sample_ids[i:i + chunk_size] for i in range(0, n, chunk_size)]
    try:
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(_process_sample_chunk, [
                (h5_file, chunk, input_dim, output_dim, num_timesteps, num_pos_features)
                for chunk in chunks
            ])
        merged = _merge_accumulators(results, input_dim + num_pos_features, output_dim)
        print(f"  Successfully processed {merged['num_samples_processed']}/{n} samples")
        if merged['num_samples_processed'] == 0:
            raise RuntimeError("No samples were successfully processed in parallel mode")
        return merged
    except Exception as e:
        print(f'  Warning: Parallel processing failed ({e}), falling back to serial')
        return _process_sample_chunk(h5_file, sample_ids, input_dim, output_dim,
                                     num_timesteps, num_pos_features)
