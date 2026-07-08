"""Rotation-invariant positional node features (geometry + topology).

Appended to physical node features to give each node a unique identity while
preserving rotation/translation invariance. Used for both T=1 and T>1.

Feature order:
    1.  Distance from centroid     — global position context
    2.  Mean neighbor edge length  — local mesh density / feature size
    3+  RWPE: random walk return probabilities (k = 2, 4, 8, 16, 32)
"""

import numpy as np


def _compute_rwpe(edge_index, N, num_rwpe):
    """RWPE: random walk return probabilities (purely topological).

    Returns diagonals of RW^k for k = 2, 4, 8, 16, 32.
    """
    from scipy import sparse

    src, dst = edge_index[0], edge_index[1]
    degree = np.zeros(N, dtype=np.float64)
    np.add.at(degree, src, 1)
    degree_inv = 1.0 / np.maximum(degree, 1)

    vals = degree_inv[src].astype(np.float64)
    RW = sparse.csr_matrix((vals, (src, dst)), shape=(N, N))

    features = []
    k_schedule = [2, 4, 8, 16, 32]
    RW_power = RW @ RW  # start at k=2
    prev_k = 2
    for k in k_schedule:
        if len(features) >= num_rwpe:
            break
        for _ in range(k - prev_k):
            RW_power = RW_power @ RW
        prev_k = k
        diag = np.array(RW_power.diagonal()).flatten()
        features.append(diag.astype(np.float32))

    return features


def compute_positional_features(pos, edge_index, num_features):
    """Compute [N, num_features] rotation-invariant positional node features.

    Args:
        pos:          [N, 3] reference positions.
        edge_index:   [2, E] bidirectional mesh edges.
        num_features: number of features to produce (1: centroid distance,
                      2: + mean edge length, 3+: + RWPE).
    """
    N = pos.shape[0]
    src, dst = edge_index[0], edge_index[1]
    features = []

    # 1. Distance from centroid (rotation-invariant: ||R(p-c)|| = ||p-c||)
    centroid = pos.mean(axis=0)
    dist_centroid = np.linalg.norm(pos - centroid, axis=1)
    features.append(dist_centroid)

    if num_features >= 2:
        # 2. Mean neighbor edge length (rotation-invariant: ||R(p_j-p_i)|| = ||p_j-p_i||)
        edge_lengths = np.linalg.norm(pos[dst] - pos[src], axis=1)
        edge_len_sum = np.zeros(N, dtype=np.float64)
        edge_len_count = np.zeros(N, dtype=np.float64)
        np.add.at(edge_len_sum, dst, edge_lengths)
        np.add.at(edge_len_count, dst, 1)
        mean_edge_len = edge_len_sum / np.maximum(edge_len_count, 1)
        features.append(mean_edge_len.astype(np.float32))

    if num_features >= 3:
        features.extend(_compute_rwpe(edge_index, N, num_features - len(features)))

    return np.column_stack(features[:num_features]).astype(np.float32)
