# Adaptive Remeshing for MeshGraphNets

## Context

The current MeshGraphNets implementation uses **fixed mesh topology** throughout simulation -- the edge connectivity loaded from HDF5 never changes during rollout. This means regions that develop high stress gradients or large deformations use the same resolution as quiescent regions, leading to:

1. **Accuracy loss** in high-gradient regions (insufficient resolution)
2. **Wasted computation** in low-gradient regions (unnecessary resolution)
3. **Inability to capture localized phenomena** like stress concentrations or contact zones

Adaptive remeshing addresses this by dynamically refining the mesh where error is high and coarsening where the solution is smooth, during the autoregressive rollout.

**Approach**: Error-indicator-based remeshing (no retraining required). The model's own predictions drive refinement/coarsening decisions. This works immediately with existing trained checkpoints.

**Key insight**: The model architecture (message passing GNN) already handles variable node/edge counts natively. The Z-score normalization stats are per-feature (not per-node), so they remain valid after topology changes. Only the rollout loop and output format need modification.

---

## Research References

- [Original MeshGraphNets (Pfaff et al., ICLR 2021)](https://openreview.net/pdf?id=roNqYL0_XP) -- sizing field approach with learned remeshing
- [Multiscale GNN with AMR](https://arxiv.org/html/2402.08863v1) -- multigrid-style coarsening/refinement
- [G-Adaptive mesh refinement](https://arxiv.org/html/2407.04516v1) -- GNN + differentiable FEM for node relocation
- [PyMesh documentation](https://pymesh.readthedocs.io/en/latest/mesh_processing.html) -- split_long_edges, collapse_short_edges API
- [GNNs for Mesh Generation and Adaptation](https://www.mdpi.com/2227-7390/12/18/2933) -- AdaptNet framework

---

## New Files to Create

```
adaptive_remeshing/
    __init__.py                # Package init, exports RemeshingManager
    error_indicators.py        # Per-edge error indicators from model predictions
    mesh_operations.py         # Edge split, collapse, flip, Laplacian smoothing
    state_interpolation.py     # Interpolate node state onto new mesh
    remeshing_manager.py       # Orchestrator: decide when/where to remesh
```

## Files to Modify

| File | Changes |
|------|---------|
| `inference_profiles/rollout.py` | Add RemeshingManager init + call in loop; variable-size output storage; per-timestep HDF5 output when remeshing active |

**No changes needed** to: model code, training pipeline, data loading, or config parser.

---

## Implementation Plan

### Step 1: Error Indicators (`adaptive_remeshing/error_indicators.py`)

Three complementary error indicators combined with configurable weights:

**A. Stress Gradient Indicator** (primary) -- For each edge (i,j): `|stress_i - stress_j| / edge_length_ij`. High values = stress discontinuity = needs refinement.

**B. Displacement Rate Indicator** -- Per-node `||delta_displacement||` from latest predicted delta, then compute gradient across edges. Large displacement changes = dynamic region.

**C. Edge Length Ratio Indicator** -- `deformed_edge_length / reference_edge_length`. Ratio >> 1 = mesh is stretched = refine. Ratio << 1 = mesh is compressed = coarsen candidate.

**Combined**: Normalize each to [0,1], weighted sum, aggregate to per-node via max of incident edges.

### Step 2: Mesh Operations (`adaptive_remeshing/mesh_operations.py`)

Pure numpy/scipy implementation (no external mesh library dependency -- avoids PyMesh installation issues on Windows).

**Operations** (standard local mesh modification pipeline):
1. **Edge Split**: Insert midpoint node, split adjacent triangles into 2 each
2. **Edge Collapse**: Merge two nodes, remove degenerate triangles. Constraint: never collapse edges between different node types (boundary preservation)
3. **Edge Flip**: Swap diagonal of two adjacent triangles (Delaunay quality improvement)
4. **Laplacian Smoothing**: Move interior nodes toward neighbor centroid (3 iterations, boundary nodes fixed)

**Batch function** `adaptive_remesh()`:
- Mark edges above `refine_threshold` for splitting
- Mark edges below `coarsen_threshold` for collapsing
- Cap operations at `max_refine_fraction` / `max_coarsen_fraction` of total edges
- Execute: splits first, then collapses, then quality flips, then smoothing
- Enforce min/max edge length bounds to prevent runaway refinement

**Triangle reconstruction**: Reuse pattern from existing `general_modules/mesh_utils_fast.py` `edges_to_triangles_optimized()` for face identification during split/collapse.

### Step 3: State Interpolation (`adaptive_remeshing/state_interpolation.py`)

- **New nodes** (from edge split): Linear interpolation of parent nodes' state: `state_new = 0.5 * state_a + 0.5 * state_b`
- **Merged nodes** (from edge collapse): Average of merged nodes' state
- **Reference positions**: Same interpolation scheme for ref_pos
- **Node types**: New nodes inherit parent type (only split within same type)

### Step 4: Remeshing Manager (`adaptive_remeshing/remeshing_manager.py`)

Orchestrator class `RemeshingManager`:
- Initialized with config, initial mesh topology, reference positions
- `should_remesh(step)`: Check if step is a remesh interval
- `remesh(step, ref_pos, current_state, mesh_edge, predicted_delta, deformed_pos, part_ids, output_dim)`:
  1. Compute combined error indicators
  2. Skip if max error is low (no remeshing needed)
  3. Call `adaptive_remesh()` with thresholds
  4. Enforce max node count
  5. Update internal face cache
  6. Return updated topology + state
- Tracks history (operations per step, node count evolution)

### Step 5: Integrate into `inference_profiles/rollout.py`

Surgical changes to `run_rollout()`:

**Before the rollout loop** (~line 221):
```python
from adaptive_remeshing.remeshing_manager import RemeshingManager
remesher = RemeshingManager(config, ref_pos, mesh_edge, part_ids)
```

**Change storage** (line 218): When remeshing enabled, use list of per-step snapshot dicts instead of fixed `all_states` array (node count can change).

**After state update** (after line 287): Call `remesher.remesh()`. If topology changed, update `ref_pos`, `current_state`, `mesh_edge`, `edge_index`, `part_ids`, `num_nodes`.

**HDF5 output** (lines 316-405): When remeshing was active, write per-timestep groups:
```
data/{sample_id}/
    timestep_0/nodal_data [features, nodes_t0], mesh_edge [2, edges_t0]
    timestep_1/nodal_data [features, nodes_t1], mesh_edge [2, edges_t1]
    ...
    metadata/remeshing_enabled=True, remeshing_history=...
```
When remeshing inactive, keep current format (backward compatible).

### Step 6: Config Parameters

All disabled by default (backward compatible). Added to `config.txt` as comments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_adaptive_remeshing` | False | Enable during inference |
| `remesh_interval` | 10 | Remesh every N steps |
| `remesh_refine_threshold` | 0.7 | Error threshold for splitting (0-1) |
| `remesh_coarsen_threshold` | 0.1 | Error threshold for collapsing (0-1) |
| `remesh_max_refine_fraction` | 0.1 | Max fraction of edges to split per step |
| `remesh_max_coarsen_fraction` | 0.05 | Max fraction of edges to collapse per step |
| `remesh_min_edge_length_factor` | 0.25 | Min edge = factor * initial_min_edge |
| `remesh_max_edge_length_factor` | 4.0 | Max edge = factor * initial_min_edge |
| `remesh_max_nodes` | 0 | Max nodes (0=unlimited) |
| `remesh_weight_stress` | 0.5 | Stress gradient indicator weight |
| `remesh_weight_displacement` | 0.3 | Displacement rate indicator weight |
| `remesh_weight_edge_length` | 0.2 | Edge length ratio indicator weight |
| `remesh_smoothing_iterations` | 3 | Laplacian smoothing iterations |

---

## Verification Plan

1. **Regression test**: Run rollout with `use_adaptive_remeshing=False` -- output must be identical to current code
2. **Unit tests**: Small synthetic mesh (4 nodes, 2 triangles) -- verify split produces 5 nodes/4 triangles, collapse produces 3 nodes/1 triangle, flip swaps diagonal
3. **State interpolation test**: Split edge on mesh with known linear field -- verify midpoint value is exact
4. **Boundary preservation test**: On multi-part mesh, verify no cross-type collapses occur
5. **Enable remeshing**: Run on inference dataset, verify:
   - No NaN values in output
   - Node count stays bounded (doesn't explode)
   - HDF5 output is readable and well-formed
   - Console shows remeshing statistics (splits/collapses per step)
6. **Accuracy comparison**: Compare remeshed vs fixed-mesh rollout error against ground truth (if available)

---

## Future Path: Learned Sizing Field

Once the error-indicator infrastructure is working, a learned sizing field can replace it:

1. Add `SizingFieldHead` decoder to `model/MeshGraphNets.py` -- predicts per-node target edge length
2. Train with L2 loss against ground truth sizing (from simulator mesh sequences)
3. Swap `compute_combined_error()` for learned sizing field predictions in `RemeshingManager`
4. The remeshing operations, state interpolation, and rollout integration remain identical
