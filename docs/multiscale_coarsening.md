# Multiscale Hierarchical Coarsening in MeshGraphNets

This document describes the two graph coarsening algorithms used to build hierarchical,
multi-resolution GNN representations of FEA meshes: **BFS Bi-Stride** and **FPS-Voronoi**.
Both are implemented in [`model/coarsening.py`](../model/coarsening.py) and integrate
into the V-cycle processor inside [`model/MeshGraphNets.py`](../model/MeshGraphNets.py).

---

## Table of Contents

1. [Why Coarsening?](#1-why-coarsening)
2. [Shared Output Contract](#2-shared-output-contract)
3. [BFS Bi-Stride Coarsening](#3-bfs-bi-stride-coarsening)
4. [FPS-Voronoi Coarsening](#4-fps-voronoi-coarsening)
5. [Shared Infrastructure](#5-shared-infrastructure)
6. [V-Cycle Architecture](#6-v-cycle-architecture)
7. [Data Loading and Batching](#7-data-loading-and-batching)
8. [Configuration Reference](#8-configuration-reference)
9. [Comparison and When to Use Each](#9-comparison-and-when-to-use-each)

---

## 1. Why Coarsening?

Standard GNNs propagate information one hop per layer. On a fine FEA mesh with
20,000+ nodes, reaching a node 100 edges away requires 100 message-passing layers —
far too expensive and prone to over-smoothing.

Hierarchical coarsening solves this by building a multi-resolution graph pyramid.
The network processes coarse levels with many fewer nodes, then unpools back to
full resolution, merging global structure with local detail. This is the graph
analogue of multigrid solvers in numerical PDE methods.

**Reduction rates:**
| Method   | Typical ratio per level | Controllable? |
|----------|------------------------|---------------|
| BFS Bi-Stride | ~4x (triangular mesh) | No (topology-driven) |
| FPS-Voronoi   | Any (e.g. 100x)       | Yes (`num_clusters`) |

---

## 2. Shared Output Contract

Both methods return the same three-tuple:

```
(fine_to_coarse [N], coarse_edge_index [2, E_c], num_coarse int)
```

| Output | Shape | Description |
|--------|-------|-------------|
| `fine_to_coarse` | `[N] int32` | Cluster index in `[0, M-1]` for each fine node |
| `coarse_edge_index` | `[2, E_c] int64` | Bidirectional edges between coarse nodes |
| `num_coarse` | `int M` | Total number of coarse nodes |

Coarse edge construction is **always boundary-based**: two coarse nodes are connected
iff at least one fine node in cluster A is a neighbor of some fine node in cluster B
in the original fine mesh. This preserves the topological connectivity structure
regardless of coarsening method.

---

## 3. BFS Bi-Stride Coarsening

### 3.1 Concept

BFS Bi-Stride (Cao et al., ICML 2023, "Efficient Graph Neural Network Inference at
Large Scale") coarsens by BFS depth parity. Even-depth nodes become the coarse graph;
odd-depth nodes are mapped to their BFS parent (always even).

On a regular triangular mesh, even-depth nodes form an approximately uniform
sub-sampling with roughly 1/4 the original count. The name "bi-stride" refers to the
fact that the coarse graph corresponds to every other BFS frontier.

### 3.2 Algorithm Step by Step

**Step 1 — Build CSR adjacency matrix**

```
edge_index_np [2, E]  →  CSR adjacency A [N, N]
```

The fine mesh edge list is converted to a scipy `csr_matrix` using vectorized
`np.ones` weights. This C-level representation is required for scipy's fast BFS.

**Step 2 — Detect connected components**

```
n_comp, comp_labels = connected_components(A, directed=False)
```

FEA assemblies often have multiple disconnected parts (e.g., PCB + chip + steel
plate). The algorithm handles each component independently to avoid false BFS
depth assignments across isolated regions.

**Step 3 — Multi-source BFS, one component at a time**

For each connected component:
1. Pick the component's first node as the seed.
2. Run `scipy.sparse.csgraph.breadth_first_order` from that seed — this is a
   C-level BFS that returns both the visit order and the predecessor array.
3. Assign depths from predecessors: `depth[node] = depth[predecessor[node]] + 1`.

After all components are processed, every node has a depth ≥ 0.

```
depth[seed_i] = 0
depth[node]   = depth[predecessor[node]] + 1   ∀ node ≠ seed
```

**Step 4 — Partition by depth parity**

```
coarse_mask  = (depth % 2 == 0)   →  True for coarse nodes
fine_only    = (depth % 2 == 1)   →  True for fine-only nodes
```

Coarse nodes are assigned contiguous indices `0 … M-1`.

**Step 5 — Build `fine_to_coarse` mapping**

```
fine_to_coarse[i] = coarse_idx_of[i]            if depth[i] is even
fine_to_coarse[i] = coarse_idx_of[parent[i]]    if depth[i] is odd
```

Every fine-only (odd) node maps to its BFS parent, which is always even-depth
(proven by BFS tree structure: parent of an odd-depth node is at depth−1, which
is even).

**Step 6 — Build coarse edges** (shared with Voronoi; see Section 5.1)

### 3.3 Illustrative Example

```
Fine graph:    0 — 1 — 2 — 3 — 4 — 5
BFS depth:     0   1   2   3   4   5
Parity:        C   F   C   F   C   F
Coarse nodes:  0       1       2
fine_to_coarse:[0, 0,  1, 1,   2, 2]
Coarse edges:  0—1, 1—2
```

In 2D triangular meshes the pattern is more complex, but the principle is identical:
even-parity BFS nodes form the coarse graph, with ~4x reduction per level.

### 3.4 Properties

| Property | Value |
|----------|-------|
| Reduction per level | ~4x (triangular mesh), ~2x (path/chain) |
| Deterministic? | Yes (given fixed seed selection — first node per component) |
| Handles disconnected graphs? | Yes — per-component BFS with independent seeding |
| Topology-aware? | Yes — never creates cross-boundary false edges |
| Requires node positions? | No — purely topological |
| Computational complexity | O(N + E) — C-level BFS via scipy |

### 3.5 Code Location

```python
# model/coarsening.py
def bfs_bistride_coarsen(edge_index_np, num_nodes):
    ...
```

---

## 4. FPS-Voronoi Coarsening

### 4.1 Concept

FPS-Voronoi is a two-stage algorithm:

1. **Farthest Point Sampling (FPS)** selects `k` seed nodes that are maximally
   spread apart across the mesh — ensuring the coarse graph covers the entire domain
   without clustering all seeds in one region.
2. **Multi-source BFS (Voronoi partition)** assigns each fine node to its nearest
   seed, forming `k` approximately equal-size clusters shaped by mesh topology.

The result is a graph-analogue of a Voronoi diagram, where "distance" is measured
in hops rather than Euclidean space (unless positions are available, in which case
Euclidean FPS is used as a faster approximation).

This method allows **precise control** over the coarse graph size: set
`num_clusters = 200` and you get exactly ~200 coarse nodes regardless of the mesh
size or topology.

### 4.2 Farthest Point Sampling

FPS is a greedy algorithm that builds a set of seeds maximally spread in the
space defined by the chosen distance metric.

**Euclidean FPS** (used when `ref_pos` is provided):

```
seeds = [random_start]
min_sq_dists[i] = inf for all i

for j in 1 .. k-1:
    for each node i:
        min_sq_dists[i] = min(min_sq_dists[i], ||pos[i] - pos[seeds[j-1]]||²)
    seeds[j] = argmax(min_sq_dists)
```

At each step, the next seed is the node farthest from all currently selected seeds.
The `min_sq_dists` array is updated incrementally (only comparing against the newest
seed), making the total cost O(N·k).

**Geodesic FPS** (fallback when positions are unavailable):

```
seeds = [random_start]
min_dists[i] = BFS_distance(seeds[0], i)  for all i

for j in 1 .. k-1:
    seeds[j] = argmax(min_dists)
    new_dists = BFS_distance(seeds[j], *)
    min_dists = element-wise min(min_dists, new_dists)
```

Same greedy logic but distance is measured in hops (number of graph edges). This
requires one full BFS per new seed, making it O(N·k·(N+E)) — significantly slower,
but topology-correct even for irregular meshes.

**FPS guarantee:** The minimum pairwise distance among selected seeds is maximized.
This ensures seeds are spread as uniformly as possible across the domain.

### 4.3 Multi-Source BFS (Voronoi Partition)

Once seeds are chosen, all seeds are inserted simultaneously into a BFS queue:

```
fine_to_coarse[seed_i] = i     ∀ seed i ∈ seeds
dist[seed_i] = 0               ∀ seed i ∈ seeds
queue = [seed_0, seed_1, ..., seed_{k-1}]

while queue not empty:
    node = queue.popleft()
    for each neighbor nbr of node:
        if dist[node] + 1 < dist[nbr]:
            dist[nbr] = dist[node] + 1
            fine_to_coarse[nbr] = fine_to_coarse[node]  # inherit cluster
            queue.append(nbr)
```

This is a standard multi-source BFS. Since all seeds start at distance 0, each node
is assigned to the seed it can reach in the fewest hops. On a uniform grid, this
produces Voronoi-like cells of approximately equal size.

**Voronoi correctness:** Any node that is equally close to two seeds will be
assigned to whichever seed's wavefront arrives first in BFS queue order (FIFO tie-
breaking). The cells are always connected and convex in the hop-distance metric.

### 4.4 Disconnected Component Handling

After FPS, the algorithm checks whether every connected component has at least one
seed:

```python
comp_has_seed = set(comp_labels[s] for s in seeds)
for comp_id in range(n_comp):
    if comp_id not in comp_has_seed:
        seeds.append(first_node_in_component(comp_id))
```

This guarantees that isolated parts (e.g., a chip separated from the PCB) are always
coarsened — even if FPS happened to skip them.

### 4.5 Cluster ID Compaction

After multi-source BFS, some seeds may not have attracted any nodes (possible if
`k > N` or in degenerate topologies). Empty clusters are removed and IDs are
remapped to contiguous integers `[0, M-1]` where `M ≤ k`.

### 4.6 Properties

| Property | Value |
|----------|-------|
| Reduction per level | User-controlled via `num_clusters` |
| Deterministic? | No — random initial seed (can fix with `np.random.seed`) |
| Handles disconnected graphs? | Yes — per-component seed injection |
| Topology-aware? | Yes — Voronoi partition via BFS preserves connectivity |
| Requires node positions? | Optional (Euclidean FPS is faster; geodesic FPS works without) |
| Complexity (Euclidean FPS) | O(N·k) FPS + O(N+E) BFS |
| Complexity (Geodesic FPS) | O(N·k·(N+E)) — slow for large k |

### 4.7 Code Location

```python
# model/coarsening.py
def fps_voronoi_coarsen(edge_index_np, num_nodes, num_clusters, ref_pos=None):
    ...
```

---

## 5. Shared Infrastructure

### 5.1 Boundary-Based Coarse Edge Construction

```python
def _build_coarse_edges(fine_to_coarse, edge_index_np, num_coarse):
```

Two coarse nodes `a` and `b` are connected iff the fine mesh contains at least one
edge `(u, v)` where `fine_to_coarse[u] = a` and `fine_to_coarse[v] = b`.

**Implementation:**
1. Map each fine edge endpoint to its coarse cluster: `cu = fine_to_coarse[src]`, `cv = fine_to_coarse[dst]`.
2. Keep only cross-cluster edges (`cu ≠ cv`).
3. Canonicalize by sorting each pair: `(min, max)`.
4. Deduplicate via integer encoding: `encoded = a*(M+1) + b`, then `np.unique`.
5. Both directions `(a, b)` and `(b, a)` are emitted to maintain bidirectionality.

This step is the same regardless of how `fine_to_coarse` was computed — it only
depends on the final cluster assignments and the original fine edge list.

### 5.2 Coarse Edge Features (8D)

```python
def compute_coarse_edge_attr(reference_pos, deformed_pos, fine_to_coarse,
                              coarse_edge_index, num_coarse):
```

Coarse nodes are positioned at the **centroid** of their constituent fine nodes:
```
coarse_ref_pos[c]  = mean(reference_pos[i]  for i where fine_to_coarse[i] = c)
coarse_def_pos[c]  = mean(deformed_pos[i]   for i where fine_to_coarse[i] = c)
```

Coarse edge features use the same 8D format as fine mesh edges:
```
[deformed_dx, deformed_dy, deformed_dz, deformed_dist,
 ref_dx,      ref_dy,      ref_dz,      ref_dist]
```
computed between the centroid positions of adjacent coarse clusters.

These edge features are **normalized** at training time. The normalization stats are
saved in the checkpoint under `coarse_edge_means` / `coarse_edge_stds` (one set
per level).

### 5.3 Pool and Unpool Operators

**Pool (fine → coarse): mean aggregation**
```python
def pool_features(h_fine, fine_to_coarse, num_coarse):
    return scatter(h_fine, fine_to_coarse, dim=0, dim_size=num_coarse, reduce='mean')
```
Each coarse node receives the average latent feature of all its fine-level children.
This is a *lossy* operation — fine-grained variation within a cluster is collapsed.

**Unpool (coarse → fine): broadcast**
```python
def unpool_features(h_coarse, fine_to_coarse):
    return h_coarse[fine_to_coarse]
```
Each fine node simply copies the latent feature of its coarse cluster. There are no
learned weights in the unpool step — information recovery happens through the
post-processing GnBlocks and skip connections.

### 5.4 MultiscaleData and PyG Batching

The `MultiscaleData` class (a `torch_geometric.data.Data` subclass) stores per-level
coarsening attributes with correct batching semantics:

```
fine_to_coarse_0    [N_0] long    # mapping from level-0 to level-1
coarse_edge_index_0 [2, E_0]      # edges at level 1
coarse_edge_attr_0  [E_0, 8]      # edge features at level 1
num_coarse_0        [1] long      # node count at level 1
...
fine_to_coarse_{L-1}  ...         # mapping to coarsest level
```

When `Batch.from_data_list` combines multiple samples into a mini-batch, the
`__inc__` override ensures `fine_to_coarse_i` and `coarse_edge_index_i` are offset
by the cumulative `num_coarse_i` from previous samples in the batch — exactly like
the standard `edge_index` offset in PyG.

---

## 6. V-Cycle Architecture

The V-cycle is the core execution pattern of the multiscale GNN. It mirrors
multigrid V-cycle solvers in numerical analysis.

### 6.1 V-Cycle Structure

For `L` coarsening levels, the `mp_per_level` config list has `2L+1` entries:

```
[pre_0, pre_1, ..., pre_{L-1},  coarsest,  post_{L-1}, ..., post_1, post_0]
```

Example with `L=2`, `mp_per_level = [3, 3, 5, 3, 3]`:

```
Level 0 (fine):     pre[0] = 3 GnBlocks
                    ↓  pool
Level 1:            pre[1] = 3 GnBlocks
                    ↓  pool
Level 2 (coarsest): coarsest = 5 GnBlocks
                    ↑  unpool
Level 1:            post[1] = 3 GnBlocks
                    ↑  unpool
Level 0 (fine):     post[0] = 3 GnBlocks
```

### 6.2 Forward Pass (Descending Arm)

```
1. Encode fine graph (node MLP + edge MLP → latent embeddings)
2. Save encoder output as encoder_x (for global skip connection)

For i = 0 to L-1:
    a. Run pre_blocks[i]  — GnBlocks at level i
    b. Save skip state:  (h_i, edge_attr_i, edge_index_i)
    c. Pool:  h_coarse = mean_scatter(h_i, fine_to_coarse_i)
    d. Encode coarse edges:  e_coarse = coarse_eb_encoders[i](coarse_edge_attr_i)
    e. Build coarse graph: Data(h_coarse, e_coarse, coarse_edge_index_i)
```

Note: world edges (long-range connectivity between mesh parts) are only active at
the finest level (i=0). Coarser levels use only the coarsened local mesh topology.

### 6.3 Coarsest Level

```
Run coarsest_blocks — GnBlocks operating on the smallest graph
```

At this level, even distant fine-mesh nodes are neighbors in the coarsened graph.
This is where long-range information exchange happens efficiently.

### 6.4 Forward Pass (Ascending Arm)

```
For i = L-1 down to 0:
    a. Unpool:  h_up = h_coarse[fine_to_coarse_i]
    b. Merge skip:  h_merged = skip_projs[i](cat([skip.h_i, h_up], dim=-1))
       (Linear(2D, D) projection of concatenated skip + unpooled features)
    c. Restore graph at level i with saved edge topology
    d. Run post_blocks[i] — GnBlocks at level i
```

The skip connection `Linear(2D → D)` is learned separately per level. It allows
the ascending arm to selectively combine fine-level detail preserved in the skip
state with the global context recovered from the coarser level.

### 6.5 Global Gated Skip Connection

After the V-cycle, a global skip from the *encoder output* to the *post-processor
output* is added:

```python
gate = sigmoid(Linear(2D → D)(cat([h_out, encoder_x])))
h_final = h_out + gate * encoder_x
```

This is a learned gate that controls how much of the original encoded node features
(before any message passing) is injected back at the end. It acts as a global
residual and stabilizes training.

### 6.6 Decode

```
output = decoder_mlp(h_final)  →  predicted normalized delta [N, output_var]
```

The decoder is a 2-hidden-layer MLP with no LayerNorm on the output.

### 6.7 Module Structure

| Module | Count | Description |
|--------|-------|-------------|
| `pre_blocks[i]` | `mp_per_level[i]` GnBlocks | Pre-pooling blocks at level i |
| `post_blocks[i]` | `mp_per_level[2L-i]` GnBlocks | Post-unpool blocks at level i |
| `coarsest_blocks` | `mp_per_level[L]` GnBlocks | Blocks at coarsest level |
| `coarse_eb_encoders[i]` | 1 MLP per level | Encodes coarse edge features |
| `skip_projs[i]` | `Linear(2D, D)` per level | Skip merge projections |
| `skip_gate` | 1 shared module | Global encoder→output gate |

---

## 7. Data Loading and Batching

Coarsening is computed **offline** at dataset load time, not during training. Each
sample in `MultiscaleDataset.__getitem__` triggers:

1. Build fine mesh `edge_index` from HDF5 connectivity.
2. For each level `i = 0 … L-1`:
   - Call `coarsen_graph(edge_index_i, num_nodes_i, method=coarsening_type[i], ...)`
   - Compute coarse edge attributes from centroid positions.
   - Store results as `MultiscaleData` attributes `fine_to_coarse_i`, etc.
3. Return the `MultiscaleData` object to the DataLoader.

The `MultiscaleData.__inc__` and `__cat_dim__` overrides handle batch index
offsetting automatically when PyG's `Batch.from_data_list` collates samples.

Coarsening is typically applied once (or cached) per sample, since the mesh topology
does not change across time steps. Only the node features (physical state) differ.

---

## 8. Configuration Reference

```ini
use_multiscale True           # Enable hierarchical V-cycle
multiscale_levels 2           # Number of coarsening levels (L)
mp_per_level 3, 3, 5, 3, 3   # GnBlocks per stage (2L+1 values)

# Coarsening method per level (comma-separated, one per level)
coarsening_type bfs           # 'bfs' or 'voronoi' (or mixed: 'bfs, voronoi')

# For voronoi levels — number of coarse nodes per level (comma-separated)
voronoi_clusters 500, 50      # e.g., level 0 → 500 nodes, level 1 → 50 nodes
```

**`mp_per_level` layout** for `L` levels:

```
Index:  0        1      ...   L-1      L         L+1    ...   2L
Role:   pre[0]  pre[1]  ...  pre[L-1]  coarsest  post[L-1] ...  post[0]
```

The sum `sum(mp_per_level)` is the total number of GnBlocks and should equal
`message_passing_num` for consistent bookkeeping.

---

## 9. Comparison and When to Use Each

### BFS Bi-Stride

**Strengths:**
- Completely deterministic (no randomness).
- Purely topology-based — no dependency on node positions.
- Natural ~4x reduction on triangular FEA meshes with no tuning.
- Coarse nodes are actual fine-mesh nodes (not virtual centroids), preserving
  physical meaning.
- O(N+E) time; fastest option.

**Weaknesses:**
- Reduction ratio is fixed by topology — cannot be specified precisely.
- Very irregular meshes (e.g., refined areas near stress concentrations) may
  produce unbalanced coarse graphs.
- Multiple levels may over-coarsen small components in multi-part assemblies.

**Best for:** Standard triangular/tetrahedral FEA meshes where ~4x reduction per
level is acceptable and speed is a priority.

### FPS-Voronoi

**Strengths:**
- Exact control over coarse graph size via `num_clusters`.
- Clusters tend to be geometrically compact and of uniform size (especially with
  Euclidean FPS).
- Aggressive coarsening in a single level is possible (e.g., 20,000 → 100 nodes).
- Works well for highly non-uniform meshes.

**Weaknesses:**
- Non-deterministic with random initial FPS seed (unless seeded).
- Euclidean FPS requires node positions (not always meaningful for abstract GNNs).
- Geodesic FPS is O(N·k·(N+E)) — slow for large k.
- Coarse nodes are virtual centroids; positional meaning is approximate.

**Best for:** Applications requiring a specific coarse graph size, or very
non-uniform meshes where BFS would produce poorly balanced partitions.

### Mixed Strategies

The `coarsening_type` config accepts per-level methods. For example:

```ini
coarsening_type bfs, voronoi
multiscale_levels 2
voronoi_clusters 100
```

This applies BFS at level 0 (fine → ~4x smaller) and Voronoi at level 1 (→ exactly
100 coarse nodes). This combines BFS's topological fidelity at fine resolution with
Voronoi's size control at the coarsest level.

---

## Summary Diagram

```
Fine mesh (N nodes)
        │
        ▼ BFS/Voronoi coarsening (level 0)
Coarse level 1 (~N/4 or k₁ nodes)
        │
        ▼ BFS/Voronoi coarsening (level 1)
Coarse level 2 (~N/16 or k₂ nodes)
        │
        ▼ (coarsest)
        ...

V-cycle forward pass:
                  fine
                 /    \
            pre[0]     post[0]
               |         |
         pool  ↓    skip  ↑ unpool+merge
            level 1      level 1
          pre[1]  post[1]
              |      |
        pool  ↓  ↑ unpool
            coarsest
          (mp_per_level[L] blocks)
```

The skip connections at each level preserve fine-scale detail while the ascending
arm restores spatial resolution with global context from the coarsest level.
