"""
Shared helpers for multiscale graph coarsening used by both the dataset loader
(`general_modules/mesh_dataset.py`) and the rollout loop
(`inference_profiles/rollout.py`).

Factors out two duplicated blocks:
  1. The per-level coarsening loop that produces a list of
     {ftc, c_ei, n_c[, up_ei]} entries.
  2. The per-timestep centroid-chaining pass that attaches normalized
     coarse edge features, centroids, and bipartite unpool edges onto a PyG
     Data object.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from model.coarsening import (
    build_unpool_edges,
    coarsen_graph,
    compute_coarse_centroids,
)


def build_multiscale_hierarchy(
    edge_index: np.ndarray,
    num_nodes: int,
    ref_pos: np.ndarray,
    multiscale_levels: int,
    coarsening_types: Sequence[str],
    voronoi_clusters: Sequence[int],
    bipartite_unpool: bool = False,
) -> List[dict]:
    """
    Build coarsening topology for a single sample.

    Returns a list of per-level dicts — len(result) may be less than
    `multiscale_levels` when the coarsening saturates (n_c <= 1 or empty edges).

    Each entry has:
        'ftc':   [N_level]       fine-to-coarse mapping (np.int64)
        'c_ei':  [2, E_coarse]   coarse edge index (np.int64)
        'n_c':   int             number of coarse nodes
        'up_ei': [2, E_up]       bipartite unpool edges (only if bipartite_unpool)
    """
    hierarchy: List[dict] = []
    current_ei, current_n = edge_index, num_nodes
    level_ref_pos = ref_pos.astype(np.float32)

    for level in range(multiscale_levels):
        method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
        n_clusters = voronoi_clusters[level] if level < len(voronoi_clusters) else 0
        ftc, c_ei, n_c = coarsen_graph(
            current_ei, current_n, method=method,
            num_clusters=n_clusters, ref_pos=level_ref_pos,
        )
        entry = {'ftc': ftc, 'c_ei': c_ei, 'n_c': n_c}
        if bipartite_unpool:
            entry['up_ei'] = build_unpool_edges(ftc, c_ei, n_c)
        hierarchy.append(entry)

        if n_c <= 1 or c_ei.shape[1] == 0:
            break

        level_ref_pos = compute_coarse_centroids(level_ref_pos, ftc, n_c).astype(np.float32)
        current_ei, current_n = c_ei, n_c

    return hierarchy


def attach_coarse_levels_to_graph(
    graph,
    hierarchy: List[dict],
    ref_pos: np.ndarray,
    deformed_pos: np.ndarray,
    coarse_edge_means: Sequence[np.ndarray],
    coarse_edge_stds: Sequence[np.ndarray],
    device: Optional[torch.device] = None,
) -> None:
    """
    Compute per-level centroids and coarse edge features for a single timestep,
    normalize with the provided per-level stats, and attach as graph attributes.

    Mutates `graph` in place, setting for each level `i` in `hierarchy`:
        fine_to_coarse_{i}, coarse_edge_index_{i}, coarse_edge_attr_{i},
        num_coarse_{i}, coarse_centroid_{i},
        unpool_edge_index_{i} (only if the entry carries 'up_ei')

    If `device` is provided, tensors are moved to that device; otherwise they
    stay on CPU (the DataLoader / .to() will handle the transfer).
    """
    cur_ref = ref_pos.astype(np.float32)
    cur_def = deformed_pos.astype(np.float32)

    for level, entry in enumerate(hierarchy):
        ftc = entry['ftc']
        c_ei = entry['c_ei']
        n_c = entry['n_c']

        coarse_ref = compute_coarse_centroids(cur_ref, ftc, n_c)
        coarse_def = compute_coarse_centroids(cur_def, ftc, n_c)

        if c_ei.shape[1] > 0:
            c_ea_raw = compute_edge_attr(
                coarse_ref.astype(np.float32),
                coarse_def.astype(np.float32),
                c_ei,
            )
        else:
            c_ea_raw = np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)

        if c_ea_raw.shape[0] > 0 and level < len(coarse_edge_means):
            c_ea_norm = (c_ea_raw - coarse_edge_means[level]) / coarse_edge_stds[level]
        else:
            c_ea_norm = c_ea_raw

        ftc_t = torch.from_numpy(ftc.astype(np.int64))
        c_ei_t = torch.from_numpy(c_ei)
        c_ea_t = torch.from_numpy(c_ea_norm.astype(np.float32))
        n_c_t = torch.tensor([n_c], dtype=torch.long)
        cent_t = torch.from_numpy(coarse_ref.astype(np.float32))

        if device is not None:
            ftc_t = ftc_t.to(device)
            c_ei_t = c_ei_t.to(device)
            c_ea_t = c_ea_t.to(device)
            n_c_t = n_c_t.to(device)
            cent_t = cent_t.to(device)

        graph[f'fine_to_coarse_{level}']    = ftc_t
        graph[f'coarse_edge_index_{level}'] = c_ei_t
        graph[f'coarse_edge_attr_{level}']  = c_ea_t
        graph[f'num_coarse_{level}']        = n_c_t
        graph[f'coarse_centroid_{level}']   = cent_t

        if 'up_ei' in entry:
            up_t = torch.from_numpy(entry['up_ei'])
            if device is not None:
                up_t = up_t.to(device)
            graph[f'unpool_edge_index_{level}'] = up_t

        cur_ref, cur_def = coarse_ref, coarse_def
