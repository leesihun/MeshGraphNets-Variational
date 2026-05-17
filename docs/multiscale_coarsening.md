# Multiscale Coarsening And V-Cycle

This document describes the current multiscale path implemented by:

- `model/coarsening.py`
- `general_modules/multiscale_helpers.py`
- `general_modules/mesh_dataset.py`
- `model/MeshGraphNets.py`

The multiscale implementation is optional and controlled by `use_multiscale`.
When enabled, the model replaces the flat processor with a V-cycle over one or
more coarsened graph levels.

## Configuration

| Key | Meaning |
| --- | --- |
| `use_multiscale` | Enables multiscale graph data and the V-cycle processor. |
| `multiscale_levels` | Number of coarse levels. |
| `coarsening_type` | `bfs`, `voronoi`, or a comma list with one method per level. |
| `voronoi_clusters` | Cluster count for Voronoi levels; scalar or comma list. |
| `mp_per_level` | Processor block counts. Must have `2 * multiscale_levels + 1` entries. |
| `bipartite_unpool` | Enables learned coarse-to-fine unpooling. |
| `coarse_cache_per_worker` | Limits each worker's hierarchy cache. |

For `multiscale_levels = L`, `mp_per_level` is interpreted as:

```text
[pre_0, pre_1, ..., pre_(L-1), coarsest, post_(L-1), ..., post_1, post_0]
```

If `mp_per_level` is omitted, the one-level fallback is:

```text
[fine_mp_pre, coarse_mp_num, fine_mp_post]
```

When multiscale is enabled, `message_passing_num` is ignored by the processor.

## Coarsening Output Contract

Both coarsening algorithms return:

```text
fine_to_coarse:    [N] cluster id for each fine node
coarse_edge_index: [2, E_c] directed coarse graph edges
num_coarse:        number of coarse nodes
```

`build_multiscale_hierarchy` extends that base contract with data needed by the
model and DataLoader, including:

- coarse edge attributes computed with the same 8-D edge layout as mesh edges
- coarse centroids for each level
- optional bipartite unpool edges from coarse nodes to fine nodes

The dataset attaches these tensors as `fine_to_coarse_i`,
`coarse_edge_index_i`, `coarse_edge_attr_i`, `num_coarse_i`, and, when
`bipartite_unpool True`, `unpool_edge_index_i` and `coarse_centroid_i`.

## BFS Bi-Stride

`bfs_bistride_coarsen` performs topology-driven coarsening:

1. Build adjacency from the current level's graph.
2. Traverse connected components with BFS.
3. Select even-depth nodes as coarse representatives.
4. Map odd-depth nodes to nearby selected representatives.
5. Build coarse edges from boundary crossings between fine clusters.

This method needs only graph topology. The reduction ratio depends on mesh
structure and is not controlled by a requested cluster count.

Use it when topology-preserving, deterministic coarsening is more important than
controlling the exact coarse node count.

## FPS-Voronoi

`fps_voronoi_coarsen` performs geometry-driven coarsening:

1. Select `num_clusters` seed nodes using farthest point sampling over positions.
2. Assign every fine node to the nearest seed.
3. Build coarse edges from boundary crossings between clusters.

This method requires reference positions. It gives direct control over the
coarse graph size through `voronoi_clusters`.

Use it when the model must fit a fixed or aggressive coarse resolution, for
example the `_b8_all_warpage_input` configs with `voronoi_clusters 200`.

## Iterative Hierarchy

For multiple levels, coarsening is iterative:

```text
fine graph -> level 0 coarse graph -> level 1 coarse graph -> ...
```

Each level's `fine_to_coarse_i` maps from that level's fine nodes to that
level's coarse nodes. At level 0 the fine nodes are original mesh nodes; at
later levels the fine nodes are the previous level's coarse nodes.

The model stops at the number of levels whose tensors are actually present on
the graph. This lets the forward path tolerate samples where fewer levels were
attached than requested, although the normal training path builds all requested
levels.

## Data Loading And Caching

`MeshGraphDataset.prepare_preprocessing()` computes normalization statistics on
the train split. When multiscale is enabled, it also computes coarse-edge
normalization statistics for each level.

During `__getitem__`:

1. The base graph is loaded from HDF5.
2. Node, edge, and delta features are normalized.
3. If the per-worker coarse cache does not contain the sample, the hierarchy is
   built and cached.
4. `attach_coarse_levels_to_graph` attaches normalized coarse edge attributes and
   mapping tensors to the PyG data object.

Coarse hierarchy data is not stored permanently in the dataset by default. It is
computed and cached by workers at runtime. The normalization arrays are stored in
checkpoints and, when explicitly written, under
`metadata/normalization_params` in the HDF5 file.

## V-Cycle Forward Pass

The multiscale processor in `model/MeshGraphNets.py` works as:

1. Encode node and edge features on the fine graph.
2. Descend through each level:
   - run that level's pre-blocks
   - save a skip state
   - pool node states with `pool_features`
   - encode normalized coarse edge attributes
3. Run coarsest graph blocks.
4. Ascend through the levels in reverse:
   - unpool coarse states to the finer level
   - merge with the saved skip state through `skip_projs[i]`
   - run that level's post-blocks
5. Decode the final fine-node states.

There is no global gated skip path in the current implementation. Skip merging is
a per-level linear projection from concatenated skip and upsampled features:

```text
Linear(2 * latent_dim -> latent_dim)
```

## Unpooling Modes

With `bipartite_unpool False`, `unpool_features` broadcasts each coarse node
state to fine nodes using `fine_to_coarse`.

With `bipartite_unpool True`, `UnpoolBlock` runs learned message passing over
`unpool_edge_index_i`:

```text
message = MLP(coarse_state, fine_skip_state, relative_position)
fine_state = MLP(fine_skip_state, summed_messages)
```

Relative position is computed from fine positions and coarse centroids at that
level.

## World Edges With Multiscale

By default (`coarse_world_edges False`), world-edge message passing is only used
on the original fine graph. Coarse processor blocks are built with
`use_world_edges False`. World-edge attributes are carried through the level-0
skip state so the ascending fine-level post blocks can still use them.

Set `coarse_world_edges True` to lift world edges to every coarse level. For each
fine world edge (u, v) where `fine_to_coarse[u] ≠ fine_to_coarse[v]`, a coarse
world edge is added between the cluster centroids. Lifting is applied iteratively
across levels so contact topology is consistent at all scales. Coarse GnBlocks
receive explicit contact-graph awareness, which matters for contact-rich FEA where
contact has large-scale structural consequences (e.g. multi-body assemblies).

Without `coarse_world_edges`, coarser message-passing stages that capture global
load redistribution have no direct awareness of contact topology — contact effects
are present only implicitly through the pooled fine-level node features.

## Failure Checks

Common multiscale errors are usually shape or config mismatches:

- `mp_per_level` length must be `2 * multiscale_levels + 1`.
- `edge_var` must be `8`.
- Voronoi levels need a valid `voronoi_clusters` value.
- Missing `fine_to_coarse_i` or `coarse_edge_attr_i` means hierarchy attachment
  did not run for that graph.
- Very large `coarse_cache_per_worker` values increase host RAM pressure.
