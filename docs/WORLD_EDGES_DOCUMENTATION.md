# World Edges

World edges are optional radius-based edges that connect geometrically nearby
nodes that are not already connected by the mesh topology. They are implemented
in:

- `general_modules/world_edges.py`
- `general_modules/mesh_dataset.py`
- `inference_profiles/rollout.py`
- `model/encoder_decoder.py`
- `model/blocks.py`

## Purpose

Mesh edges follow the finite-element connectivity. World edges add proximity
contacts in deformed space, which can help the model exchange information
between nearby parts that are not mesh neighbors.

The feature is controlled by:

```text
use_world_edges True
world_radius_multiplier 1.5
world_max_num_neighbors 64
world_edge_backend auto
```

`use_world_edges False` creates empty `world_edge_index` and `world_edge_attr`
tensors so the rest of the graph contract stays stable.

## Edge Construction

`compute_world_edges` takes:

- reference positions
- current deformed positions
- bidirectional mesh edges
- radius
- maximum neighbors
- backend
- optional edge normalization arrays

It returns:

```text
world_edge_index: [2, E_world]
world_edge_attr:  [E_world, 8]
```

Mesh edges are filtered out of the radius graph, so a pair should not appear in
both `edge_index` and `world_edge_index`.

World-edge attributes use the same 8-D layout as mesh edges:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

The world-edge attributes are normalized with the mesh-edge `edge_mean` and
`edge_std`. There is no separate `world_edge_mean` or `world_edge_std` in the
current runtime.

## Radius

During training, `MeshGraphDataset._compute_world_edge_radius()` estimates a
single radius from sampled mesh-edge lengths:

```text
world_edge_radius = world_radius_multiplier * min_edge_length
```

That value is stored in checkpoint normalization as `world_edge_radius`. During
inference, `rollout.py` reads the radius from the checkpoint, which keeps rollout
behavior aligned with training.

## Backends

`world_edge_backend` can be:

| Value | Behavior |
| --- | --- |
| `torch_cluster` | Uses `torch_cluster.radius_graph` when installed. |
| `scipy_kdtree` | Uses `scipy.spatial.KDTree.query_pairs` on CPU. |
| `auto` | Uses `torch_cluster` if available, otherwise SciPy. |

Any non-`torch_cluster` backend string falls back to the SciPy path inside
`compute_world_edges`.

The SciPy backend expands unordered pairs to directed edges in both directions.
The torch_cluster backend uses the directed output from `radius_graph`.

## Model Integration

When `use_world_edges True`, the simulator encoder builds a separate world-edge
encoder in addition to the mesh-edge encoder.

Each world-edge-enabled `GnBlock` then performs:

1. Mesh edge update with the normal mesh `EdgeBlock`.
2. World edge update with a separate world `EdgeBlock`.
3. Node update through `HybridNodeBlock`, which concatenates:
   - current node state
   - summed incoming mesh-edge messages
   - summed incoming world-edge messages

Node and edge updates remain residual. If no world edges are present for a graph,
the model uses empty world-edge tensors and the world aggregation is zero.

## Interaction With Multiscale

By default (`coarse_world_edges False`), world-edge message passing is used only
at the original fine level. Coarse-level processor blocks are constructed with
`use_world_edges False`, and coarse graph edges come from the multiscale
hierarchy, not the radius graph.

Set `coarse_world_edges True` to propagate contact information through all coarse
levels. For each fine world edge (u, v) where the two nodes belong to different
coarse clusters (`fine_to_coarse[u] ≠ fine_to_coarse[v]`), a coarse world edge is
added between the two cluster centroids. Edge features use the standard 8-D layout
computed from coarse centroid positions and normalized with the per-level coarse-
edge statistics. The lifting is applied iteratively so that level-i world edges are
derived from level-(i-1) world edges, keeping the contact graph consistent across
all scales. Coarse-level GnBlocks are then constructed with `use_world_edges True`
so contact messages are explicitly aggregated at every scale.

When only `use_world_edges True` (default `coarse_world_edges False`) is set, the
level-0 skip state preserves fine world-edge tensors so the ascending fine-level
post blocks can still use them. With `coarse_world_edges True`, every skip level
carries its own world-edge tensors so contact information is restored correctly at
each scale during the ascending arm.

## Inference

`rollout.py` recomputes world edges at every rollout step from the current
deformed position:

```text
deformed_pos = reference_pos + current_predicted_state[:, :3]
```

It uses checkpoint `world_edge_radius`, config `world_max_num_neighbors`, and
config/checkpoint `world_edge_backend` after model_config overrides.

If no radius is present in checkpoint normalization, rollout creates empty
world-edge tensors even when `use_world_edges` is enabled.

## Practical Checks

Use these checks when world edges look wrong:

- `edge_var` must be `8`.
- Training checkpoint normalization should contain `world_edge_radius`.
- `torch_cluster` is optional; SciPy fallback is expected when it is missing.
- Very small radius can produce zero world edges.
- Very large radius can create excessive edges and memory pressure.
- Mesh edges are intentionally filtered from the world-edge set.
