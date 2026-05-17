# Dataset Format

The training and inference code expects HDF5 files with a `data/{sample_id}`
group per sample. The current loader is `general_modules/mesh_dataset.py`; the
dataset builders are `build_dataset.py`, `dataset/generate_inference_dataset.py`,
and `dataset/reduce_dataset.py`.

## File Layout

```text
dataset.h5
  attrs:
    num_samples
    num_features
    num_timesteps
  data/
    1/
      nodal_data
      mesh_edge
      metadata/
        attrs:
          source_filename
          filename_id
          num_nodes
          num_edges
          num_cells
          num_corner_nodes
          num_total_nodes
        feature_min
        feature_max
        feature_mean
        feature_std
    2/
      ...
  metadata/
    feature_names
    normalization_params/
      min
      max
      mean
      std
      optional train-derived arrays written by the loader:
        node_mean
        node_std
        edge_mean
        edge_std
        delta_mean
        delta_std
    splits/
      train
      val
      test
```

The builder initializes `metadata/splits/*`, but the current training loader
does not consume those HDF5 splits. Training calls `dataset.split(0.8, 0.1, 0.1,
seed=split_seed)` and creates a deterministic seeded split from sorted sample
ids.

## `nodal_data`

Path:

```text
data/{sample_id}/nodal_data
```

Shape:

```text
[num_features, num_timesteps, num_nodes]
```

The standard builder writes 8 features:

| Index | Meaning |
| --- | --- |
| `0` | x coordinate |
| `1` | y coordinate |
| `2` | z coordinate |
| `3` | x displacement |
| `4` | y displacement |
| `5` | z displacement |
| `6` | stress or other scalar field when present |
| `7` | part number |

Current `_b8_all_warpage_input` configs use `input_var 3` and `output_var 3`,
so they train on displacement channels only. The part-number channel is still
kept for visualization and optional node-type features.

For single-timestep datasets, the loader sets the physical input channels to
zeros and uses `nodal_data[3:3 + output_var]` as the target state. For
multi-timestep datasets, the loader uses:

```text
x_phys = nodal_data[3:3 + input_var, t]
target_delta = nodal_data[3:3 + output_var, t + 1] - x_phys
```

`pos` is always the reference coordinate slice `nodal_data[0:3, t]`.

## `mesh_edge`

Path:

```text
data/{sample_id}/mesh_edge
```

Shape:

```text
[2, E]
```

`build_dataset.py` extracts unique undirected edges from triangular elements and
writes them once. `MeshGraphDataset` converts loaded mesh edges to the
bidirectional graph representation used by PyG.

Runtime edge attributes are not stored in the dataset. They are recomputed from
reference and deformed positions as 8-D features:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

The same feature function is used for mesh edges, world edges, and coarse
multiscale edges.

## Metadata

Per-sample metadata contains attributes for source tracking and graph size plus
per-feature summary arrays. These arrays are useful for inspection but are not
the training normalizers used by the model.

Global metadata contains:

- `feature_names`
- builder-level `normalization_params/min`, `max`, `mean`, and `std`
- optional split datasets

The loader fits train-split z-score normalizers for nodes, edges, and deltas.
Those are the values saved into checkpoints under `checkpoint['normalization']`.
When `write_preprocessing_to_hdf5()` is called, it writes train-derived arrays
under `metadata/normalization_params`:

```text
node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std
```

The training setup writes preprocessing to HDF5 before training starts, so the
file may contain both builder-level summary stats and train-derived model
normalizers.

## Positional Features

`positional_features` appends rotation-invariant node features to the physical
input channels. The feature order is:

1. distance from the graph centroid
2. mean neighbor edge length
3. remaining features from `positional_encoding`

Supported encodings:

| Encoding | Meaning |
| --- | --- |
| `rwpe` | random-walk return probabilities at powers 2, 4, 8, 16, 32 |
| `lpe` | normalized Laplacian eigenvectors |
| `rwpe+lpe` | split the remaining slots between RWPE and LPE |

The model input size is:

```text
input_var + positional_features + optional num_node_types
```

## Node Types

If `use_node_types True`, the loader reads feature index 7 as a raw node type or
part id, maps observed values to contiguous indices, and appends one-hot node
type vectors to `x`.

The node-type mapping and count are saved in checkpoint normalization:

```text
node_type_to_idx
num_node_types
```

## World Edges

World edges are optional and are not stored in the source dataset. When
`use_world_edges True`, the loader computes radius edges from deformed positions
at each sample access and attaches:

```text
world_edge_index
world_edge_attr
```

World-edge attributes use the same 8-D layout and normalization as mesh-edge
attributes. The computed radius is stored in checkpoint normalization as
`world_edge_radius`.

## Multiscale Data

Multiscale hierarchy tensors are not stored permanently in the default dataset
builder. When `use_multiscale True`, the loader computes and caches them per
worker, then attaches per-level tensors to `MultiscaleData`:

```text
fine_to_coarse_i
coarse_edge_index_i
coarse_edge_attr_i
num_coarse_i
optional unpool_edge_index_i
optional coarse_centroid_i
```

Coarse edge normalizers are saved into checkpoints as:

```text
coarse_edge_means
coarse_edge_stds
```

## Inference Dataset

`dataset/generate_inference_dataset.py` copies selected samples from a source
dataset and keeps only the first timestep:

```text
nodal_data[:, 0:1, :]
mesh_edge
metadata attrs when present
```

Rollout uses this file for initial conditions, then writes a separate output
HDF5 per sample or per VAE sample.

## Rollout Output Format

`inference_profiles/rollout.py` writes:

```text
outputs/rollout/rollout_sample{sample_id}_steps{steps}.h5
outputs/rollout/rollout_sample{sample_id}_vaesample{idx}_steps{steps}.h5
```

Each output file has one sample and this `nodal_data` layout:

```text
x, y, z, predicted output channels..., Part No.
```

Root attributes:

```text
num_samples = 1
num_features = 3 + output_var + 1
num_timesteps = rollout_steps + 1
```

Per-sample metadata stores sample id, node and edge counts, rollout time, model
path, config file, and optional VAE sample index. Global metadata stores feature
names and the normalization arrays used for inference.

## Builder Notes

`build_dataset.py` assumes the local ANSYS/CSV source layout configured at the
top of that file. It writes:

- a last-step static dataset through `build_dataset_last_timestep()`
- a full temporal dataset through `build_dataset()`

`dataset/reduce_dataset.py` can copy a subset of samples, optionally create new
split metadata, and preserve global/per-sample metadata.

## Validation Checklist

Before training, check:

- each sample has `nodal_data` and `mesh_edge`
- `nodal_data` has at least `3 + max(input_var, output_var)` feature rows
- feature index 7 exists if `use_node_types True`
- `mesh_edge` uses valid node indices for each sample
- `edge_var` is `8`
- `num_timesteps` is consistent with static vs temporal training intent
