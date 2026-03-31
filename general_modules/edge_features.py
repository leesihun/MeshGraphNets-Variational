import numpy as np


EDGE_FEATURE_DIM = 8


def compute_edge_attr(reference_pos: np.ndarray, deformed_pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Build 8-D edge features from reference and deformed positions.

    Feature order:
        [deformed_dx, deformed_dy, deformed_dz, deformed_dist,
         ref_dx,      ref_dy,      ref_dz,      ref_dist]
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]

    deformed_rel = deformed_pos[dst_idx] - deformed_pos[src_idx]
    deformed_dist = np.linalg.norm(deformed_rel, axis=1, keepdims=True)

    ref_rel = reference_pos[dst_idx] - reference_pos[src_idx]
    ref_dist = np.linalg.norm(ref_rel, axis=1, keepdims=True)

    return np.concatenate([deformed_rel, deformed_dist, ref_rel, ref_dist], axis=1).astype(np.float32)
