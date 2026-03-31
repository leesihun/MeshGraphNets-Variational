"""
Multi-Scale Coarsening for MeshGraphNets.

Supports two coarsening methods:

1. **BFS Bi-Stride** (Cao et al., ICML 2023):
   - BFS assigns each node a depth. Even-depth → coarse; odd-depth → fine-only.
   - ~4x reduction per level on triangular meshes.
   - Topology-based → no cross-boundary false edges.

2. **FPS-Voronoi** (new):
   - Farthest Point Sampling selects k seed nodes maximally spread across the mesh.
   - Multi-source BFS assigns each node to its nearest seed (Voronoi partition).
   - Configurable reduction ratio (e.g., 20k → 200 in one level).

Both methods produce the same output signature:
  (fine_to_coarse [N], coarse_edge_index [2, E_c], num_coarse int)

Methods can be mixed per level via ``coarsening_type`` config (e.g., ``bfs, voronoi``).
Coarse edges are always boundary edges: two coarse nodes connect iff any of their
fine members share a fine edge.

Multi-level support:
  Coarsening is applied iteratively: level 0 → level 1 → level 2 → ...
  Each level stores its own fine_to_coarse mapping (NOT composed).
  The model uses a V-cycle architecture with per-level GnBlocks.
"""

import re
from collections import deque

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import compute_edge_attr


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _build_coarse_edges(
    fine_to_coarse: np.ndarray,
    edge_index_np: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Build bidirectional coarse edges from fine mesh adjacency.

    Two coarse nodes are connected if any fine node in one cluster is adjacent
    to any fine node in the other cluster (boundary edge construction).

    Args:
        fine_to_coarse: [N] int32 array — coarse cluster index per fine node.
        edge_index_np:  [2, E] int array — bidirectional fine mesh edges.
        num_coarse:     int M — number of coarse nodes.

    Returns:
        coarse_edge_index: [2, E_c] int64 array — bidirectional coarse edges.
    """
    cu = fine_to_coarse[edge_index_np[0]].astype(np.int64)
    cv = fine_to_coarse[edge_index_np[1]].astype(np.int64)
    cross_mask = cu != cv
    if cross_mask.any():
        a = np.minimum(cu[cross_mask], cv[cross_mask])
        b = np.maximum(cu[cross_mask], cv[cross_mask])
        pairs_encoded = a * (num_coarse + 1) + b
        unique_encoded = np.unique(pairs_encoded)
        a_uniq = unique_encoded // (num_coarse + 1)
        b_uniq = unique_encoded % (num_coarse + 1)
        src = np.concatenate([a_uniq, b_uniq])
        dst = np.concatenate([b_uniq, a_uniq])
        coarse_edge_index = np.stack([src, dst], axis=0)
    else:
        coarse_edge_index = np.zeros((2, 0), dtype=np.int64)
    return coarse_edge_index


# ---------------------------------------------------------------------------
# BFS Bi-Stride Coarsening
# ---------------------------------------------------------------------------

def bfs_bistride_coarsen(edge_index_np: np.ndarray, num_nodes: int):
    """
    Compute BFS bi-stride coarsening of a mesh graph.

    Algorithm:
    1. Multi-source BFS assigns each node a depth (hops from seed).
    2. Even-depth nodes → coarse graph (kept); odd-depth → fine-only.
    3. Each fine-only node is assigned to its BFS parent (always even-depth).
    4. Coarse edges via 2nd-order adjacency: iterate fine edges; if src and dst
       map to different coarse clusters, add a coarse edge.

    Handles disconnected meshes (e.g., multi-part FEA with separate steel plate,
    PCB, chips) by restarting BFS at every unvisited seed.

    Uses scipy.sparse.csgraph for C-level BFS speed (~10-50x faster than Python).

    Args:
        edge_index_np: [2, E] int numpy array of bidirectional mesh edges.
                       Must already be bidirectional (as produced by mesh_dataset).
        num_nodes:     N — total number of fine nodes.

    Returns:
        fine_to_coarse:   [N] int32 numpy array. fine_to_coarse[i] is the coarse
                          cluster index (0 … M-1) of fine node i.
        coarse_edge_index:[2, E_c] int64 numpy array. Bidirectional coarse edges.
        num_coarse:       int M — number of coarse nodes.
    """
    # 1. Build CSR adjacency matrix (vectorized, no Python loop)
    row = edge_index_np[0].astype(np.int32)
    col = edge_index_np[1].astype(np.int32)
    data = np.ones(row.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # 2. Multi-source BFS using scipy C-level routines
    n_comp, comp_labels = connected_components(adj, directed=False)

    depth = np.full(num_nodes, -1, dtype=np.int32)
    bfs_parent = np.arange(num_nodes, dtype=np.int32)  # default: own parent (for coarse nodes)

    for comp_id in range(n_comp):
        # Find seed (first node in this component)
        comp_mask = (comp_labels == comp_id)
        seed = int(np.argmax(comp_mask))

        # Scipy BFS: returns nodes in BFS order + predecessor array (C-level speed)
        order, predecessors = breadth_first_order(adj, i_start=seed, directed=False)

        # Compute depth from BFS order (parents guaranteed processed first)
        depth[seed] = 0
        for i in range(1, len(order)):
            node = order[i]
            depth[node] = depth[predecessors[node]] + 1
            bfs_parent[node] = predecessors[node]

    # 3. Even-depth → coarse; odd-depth → fine-only (vectorized)
    coarse_mask = (depth % 2 == 0)
    coarse_nodes = np.where(coarse_mask)[0]  # original fine node IDs of coarse nodes
    num_coarse = int(len(coarse_nodes))

    # Map original fine node ID → contiguous coarse index [0 … M-1] (vectorized)
    coarse_idx_of = np.full(num_nodes, -1, dtype=np.int32)
    coarse_idx_of[coarse_nodes] = np.arange(num_coarse, dtype=np.int32)

    # 4. Build fine_to_coarse [N] (vectorized):
    #    Even-depth nodes → their own coarse index; odd-depth → their BFS parent's coarse index
    parent_or_self = np.where(coarse_mask, np.arange(num_nodes, dtype=np.int32), bfs_parent)
    fine_to_coarse = coarse_idx_of[parent_or_self]  # [N] int32, values in [0, M-1]

    # 5. Build coarse edges via boundary detection
    coarse_edge_index = _build_coarse_edges(fine_to_coarse, edge_index_np, num_coarse)

    return fine_to_coarse, coarse_edge_index, num_coarse


# ---------------------------------------------------------------------------
# FPS-Voronoi Coarsening
# ---------------------------------------------------------------------------

def _fps_euclidean(pos: np.ndarray, k: int) -> list:
    """
    Farthest Point Sampling using Euclidean distance.  O(N·k) with numpy.

    Greedily selects k points that are maximally spread apart.
    """
    N = pos.shape[0]
    if k >= N:
        return list(range(N))

    pos = np.asarray(pos, dtype=np.float64)
    seeds = np.empty(k, dtype=np.intp)
    seeds[0] = np.random.randint(N)

    min_sq_dists = np.full(N, np.inf, dtype=np.float64)

    for i in range(k):
        s = seeds[i]
        diff = pos - pos[s]
        sq_dists = np.einsum('ij,ij->i', diff, diff)
        np.minimum(min_sq_dists, sq_dists, out=min_sq_dists)
        min_sq_dists[s] = -1.0  # exclude selected
        if i < k - 1:
            seeds[i + 1] = np.argmax(min_sq_dists)

    return seeds.tolist()


def _bfs_distances(adj: csr_matrix, start: int, num_nodes: int) -> np.ndarray:
    """BFS distance from *start* using scipy C-level BFS + Python depth pass."""
    order, predecessors = breadth_first_order(adj, i_start=start, directed=False)
    dists = np.full(num_nodes, np.iinfo(np.int32).max, dtype=np.int32)
    dists[start] = 0
    for i in range(1, len(order)):
        node = order[i]
        dists[node] = dists[predecessors[node]] + 1
    return dists


def _fps_geodesic(adj: csr_matrix, num_nodes: int, k: int) -> list:
    """FPS using geodesic (hop) distance.  O(N·k) — slower fallback."""
    if k >= num_nodes:
        return list(range(num_nodes))

    seeds = [np.random.randint(num_nodes)]
    min_dists = _bfs_distances(adj, seeds[0], num_nodes)
    min_dists[seeds[0]] = -1

    for _ in range(k - 1):
        new_seed = int(np.argmax(min_dists))
        seeds.append(new_seed)
        dists = _bfs_distances(adj, new_seed, num_nodes)
        np.minimum(min_dists, dists, out=min_dists)
        for s in seeds:
            min_dists[s] = -1

    return seeds


def fps_voronoi_coarsen(
    edge_index_np: np.ndarray,
    num_nodes: int,
    num_clusters: int,
    ref_pos: np.ndarray | None = None,
):
    """
    FPS-Voronoi coarsening: Farthest Point Sampling + Voronoi partition.

    1. FPS selects *k* seed nodes maximally spread across the mesh.
    2. Multi-source BFS from seeds assigns each node to its nearest seed.
    3. Boundary edges connect clusters sharing fine-level adjacency.

    Handles disconnected components by ensuring at least one seed per component.

    Args:
        edge_index_np: [2, E] int numpy array — bidirectional mesh edges.
        num_nodes:     N — total number of fine nodes.
        num_clusters:  k — desired number of coarse nodes.
        ref_pos:       [N, 3] float array — reference positions (optional).
                       If provided, Euclidean FPS is used (fast).
                       Otherwise, geodesic BFS FPS (slower).

    Returns:
        fine_to_coarse:    [N] int32 numpy array.
        coarse_edge_index: [2, E_c] int64 numpy array.
        num_coarse:        int M.
    """
    k = min(num_clusters, num_nodes)
    if k <= 0:
        raise ValueError(f"num_clusters must be > 0, got {num_clusters}")

    # Build CSR adjacency
    row = edge_index_np[0].astype(np.int32)
    col = edge_index_np[1].astype(np.int32)
    data = np.ones(row.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # --- Select seeds via FPS ---
    if ref_pos is not None:
        seeds = _fps_euclidean(ref_pos, k)
    else:
        seeds = _fps_geodesic(adj, num_nodes, k)

    # --- Ensure every connected component has at least one seed ---
    n_comp, comp_labels = connected_components(adj, directed=False)
    comp_has_seed = set(int(comp_labels[s]) for s in seeds)
    for comp_id in range(n_comp):
        if comp_id not in comp_has_seed:
            comp_node = int(np.argmax(comp_labels == comp_id))
            seeds.append(comp_node)

    k_actual = len(seeds)

    # --- Multi-source BFS for Voronoi partition ---
    fine_to_coarse = np.full(num_nodes, -1, dtype=np.int32)
    dist = np.full(num_nodes, np.iinfo(np.int32).max, dtype=np.int32)

    queue = deque()
    for cluster_id, seed in enumerate(seeds):
        fine_to_coarse[seed] = cluster_id
        dist[seed] = 0
        queue.append(seed)

    while queue:
        node = queue.popleft()
        d = dist[node]
        start, end = adj.indptr[node], adj.indptr[node + 1]
        for i in range(start, end):
            nbr = int(adj.indices[i])
            if d + 1 < dist[nbr]:
                dist[nbr] = d + 1
                fine_to_coarse[nbr] = fine_to_coarse[node]
                queue.append(nbr)

    # --- Compact cluster IDs (remove any empty clusters) ---
    unique_clusters = np.unique(fine_to_coarse)
    if unique_clusters[0] == -1:
        unique_clusters = unique_clusters[1:]
    num_coarse = len(unique_clusters)
    if num_coarse < k_actual:
        remap = np.full(k_actual, -1, dtype=np.int32)
        remap[unique_clusters] = np.arange(num_coarse, dtype=np.int32)
        fine_to_coarse = remap[fine_to_coarse]

    # --- Build boundary edges ---
    coarse_edge_index = _build_coarse_edges(fine_to_coarse, edge_index_np, num_coarse)

    return fine_to_coarse, coarse_edge_index, num_coarse


# ---------------------------------------------------------------------------
# Coarsening Dispatcher
# ---------------------------------------------------------------------------

def coarsen_graph(
    edge_index_np: np.ndarray,
    num_nodes: int,
    method: str = 'bfs',
    num_clusters: int | None = None,
    ref_pos: np.ndarray | None = None,
):
    """
    Dispatch to the appropriate coarsening method.

    Args:
        edge_index_np: [2, E] int numpy array — bidirectional mesh edges.
        num_nodes:     N — total number of fine nodes.
        method:        'bfs' or 'voronoi'.
        num_clusters:  Required for voronoi — number of coarse nodes.
        ref_pos:       [N, 3] positions — optional, used by voronoi for Euclidean FPS.

    Returns:
        (fine_to_coarse [N], coarse_edge_index [2, E_c], num_coarse int)
    """
    method = method.strip().lower()
    if method == 'bfs':
        return bfs_bistride_coarsen(edge_index_np, num_nodes)
    elif method == 'voronoi':
        if num_clusters is None:
            raise ValueError("num_clusters is required for 'voronoi' coarsening")
        return fps_voronoi_coarsen(edge_index_np, num_nodes, num_clusters, ref_pos)
    else:
        raise ValueError(f"Unknown coarsening method: '{method}'. Use 'bfs' or 'voronoi'.")


# ---------------------------------------------------------------------------
# Coarse Edge Features
# ---------------------------------------------------------------------------

def compute_coarse_edge_attr(
    reference_pos: np.ndarray,
    deformed_pos: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_edge_index: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Compute 8-D coarse edge features between coarse node centroids.

    Coarse reference position = mean reference position of all fine nodes in the cluster.
    Coarse deformed position = mean deformed position of all fine nodes in the cluster.
    Edge features use the same 8-D format as fine mesh edges.

    Args:
        reference_pos:     [N, 3] float array — reference positions of fine nodes.
        deformed_pos:      [N, 3] float array — deformed positions of fine nodes
                           (reference pos + displacement, same as used for fine edges).
        fine_to_coarse:    [N] int32 array — coarse cluster index for each fine node.
        coarse_edge_index: [2, E_c] int64 array — coarse edge indices.
        num_coarse:        int M — number of coarse nodes.

    Returns:
        coarse_edge_attr: [E_c, 8] float32 array.
    """
    if coarse_edge_index.shape[1] == 0:
        return np.zeros((0, 8), dtype=np.float32)

    # Compute coarse centroid positions (mean of fine nodes per cluster)
    # Vectorized: use np.add.at for O(N) scatter-add without a Python loop
    coarse_ref_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    coarse_def_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    np.add.at(coarse_ref_pos, fine_to_coarse, reference_pos.astype(np.float64))
    np.add.at(coarse_def_pos, fine_to_coarse, deformed_pos.astype(np.float64))
    counts = np.bincount(fine_to_coarse, minlength=num_coarse).reshape(-1, 1)
    coarse_ref_pos /= np.maximum(counts, 1)
    coarse_def_pos /= np.maximum(counts, 1)

    return compute_edge_attr(coarse_ref_pos, coarse_def_pos, coarse_edge_index)


def compute_coarse_centroids(
    positions: np.ndarray,
    fine_to_coarse: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Compute mean centroid positions for coarse clusters.

    Args:
        positions:      [N, 3] float array — positions of fine/current-level nodes.
        fine_to_coarse: [N] int32 array — coarse cluster index for each node.
        num_coarse:     int M — number of coarse nodes.

    Returns:
        coarse_pos: [M, 3] float64 array — centroid positions.
    """
    coarse_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    np.add.at(coarse_pos, fine_to_coarse, positions.astype(np.float64))
    counts = np.bincount(fine_to_coarse, minlength=num_coarse).reshape(-1, 1)
    coarse_pos /= np.maximum(counts, 1)
    return coarse_pos


# ---------------------------------------------------------------------------
# Pool / Unpool Operators
# ---------------------------------------------------------------------------

def pool_features(
    h_fine: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    num_coarse: int,
) -> torch.Tensor:
    """
    Mean-aggregate fine node features to coarse nodes.

    Args:
        h_fine:         [N, D] fine node feature tensor.
        fine_to_coarse: [N] long tensor, coarse cluster index for each fine node.
        num_coarse:     int M — total number of coarse nodes (handles batching correctly
                        when fine_to_coarse values are already offset by batch).

    Returns:
        h_coarse: [M, D] coarse node features.
    """
    return scatter(h_fine, fine_to_coarse, dim=0, dim_size=num_coarse, reduce='mean')


def unpool_features(
    h_coarse: torch.Tensor,
    fine_to_coarse: torch.Tensor,
) -> torch.Tensor:
    """
    Broadcast coarse node features back to fine nodes via simple gather (no learned weights).

    Args:
        h_coarse:       [M, D] coarse node features.
        fine_to_coarse: [N] long tensor, coarse cluster index for each fine node.

    Returns:
        h_fine: [N, D] — each fine node receives its coarse cluster's features.
    """
    return h_coarse[fine_to_coarse]


def build_unpool_edges(
    fine_to_coarse: np.ndarray,
    coarse_edge_index: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Build bipartite edge_index [2, E_up] from coarse→fine for unpool message passing.

    Each fine node connects to: own cluster + all coarse neighbors of own cluster.
    Row 0 = coarse src indices, Row 1 = fine dst indices.

    Args:
        fine_to_coarse:    [N] int — cluster assignment per fine node.
        coarse_edge_index: [2, E_c] int — coarse-level edges (bidirectional).
        num_coarse:        int M — number of coarse nodes.

    Returns:
        [2, E_up] int64 array — bipartite edge index.
    """
    # Build coarse adjacency list
    adj = [set() for _ in range(num_coarse)]
    src, dst = coarse_edge_index[0], coarse_edge_index[1]
    for s, d in zip(src, dst):
        adj[int(s)].add(int(d))
        adj[int(d)].add(int(s))

    # For each cluster c: coarse targets = {c} ∪ neighbors(c)
    coarse_targets = [sorted({c} | adj[c]) for c in range(num_coarse)]

    # Build fine→coarse member lists
    members = [[] for _ in range(num_coarse)]
    for fine_i, coarse_c in enumerate(fine_to_coarse):
        members[int(coarse_c)].append(fine_i)

    # Build bipartite edges: for each fine node, connect to all coarse targets of its cluster
    src_list = []
    dst_list = []
    for c in range(num_coarse):
        targets = coarse_targets[c]
        fine_nodes = members[c]
        if len(fine_nodes) == 0:
            continue
        fine_arr = np.array(fine_nodes, dtype=np.int64)
        for t in targets:
            src_list.append(np.full(len(fine_arr), t, dtype=np.int64))
            dst_list.append(fine_arr)

    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    return np.stack([np.concatenate(src_list), np.concatenate(dst_list)], axis=0)


# ---------------------------------------------------------------------------
# Custom Data class for proper PyG batching of multiscale attributes
# ---------------------------------------------------------------------------

# Regex to extract level index from attribute names like "fine_to_coarse_0"
_LEVEL_RE = re.compile(
    r'^(fine_to_coarse|coarse_edge_index|coarse_edge_attr|coarse_centroid'
    r'|unpool_edge_index|num_coarse)_(\d+)$'
)


class MultiscaleData(Data):
    """
    PyTorch Geometric Data subclass that handles correct batching of N-level
    multiscale coarsening attributes.

    Per-level attributes (for level i = 0 .. L-1):
        fine_to_coarse_{i}   [N_i] long  — mapping from level i to level i+1
        coarse_edge_index_{i} [2, E_i]   — edge topology at level i+1
        coarse_edge_attr_{i}  [E_i, 8]   — edge features at level i+1
        num_coarse_{i}        [1] long    — node count at level i+1

    When Batch.from_data_list combines multiple MultiscaleData objects:
    - fine_to_coarse_{i} values are offset by cumulative num_coarse_{i} counts
    - coarse_edge_index_{i} values are offset by cumulative num_coarse_{i} counts
    - coarse_edge_attr_{i} is concatenated along dim 0 (no offset needed)
    - num_coarse_{i} values are concatenated → [B] tensor
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        m = _LEVEL_RE.match(key)
        if m:
            prefix, level = m.group(1), m.group(2)
            if prefix in ('fine_to_coarse', 'coarse_edge_index'):
                return int(self[f'num_coarse_{level}'])
            if prefix == 'unpool_edge_index':
                # Row 0 = coarse src (offset by num_coarse), Row 1 = fine dst (offset by fine node count)
                coarse_inc = int(self[f'num_coarse_{level}'])
                fine_inc = self.num_nodes if int(level) == 0 else int(self[f'num_coarse_{int(level) - 1}'])
                return torch.tensor([[coarse_inc], [fine_inc]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        m = _LEVEL_RE.match(key)
        if m:
            prefix = m.group(1)
            if prefix in ('coarse_edge_index', 'unpool_edge_index'):
                return 1   # [2, E] — concatenate along edge dimension
            if prefix in ('fine_to_coarse', 'coarse_edge_attr', 'coarse_centroid'):
                return 0   # [N, ...] — concatenate along node/edge dim
        return super().__cat_dim__(key, value, *args, **kwargs)
