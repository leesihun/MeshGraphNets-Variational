# Configuration Reference

Complete reference for all keys accepted in MeshGraphNets config files.

**Format rules:**
- One `key value` pair per line (space or tab separated)
- `%` begins a full-line comment
- `#` begins an inline comment (rest of line ignored)
- Keys are case-insensitive (lowercased internally)
- CSV values (`a, b, c`) are parsed as Python lists
- Booleans: `True` / `False`
- Numeric types are auto-detected (int vs float)
- A line containing only `'` is treated as a blank separator (ignored)

---

## Mode & Execution

| Key | Type | Example | Description |
|---|---|---|---|
| `model` | str | `MeshGraphNets` | Model class name. Currently only `MeshGraphNets` is supported. |
| `mode` | str | `train` | Execution mode: `train` or `inference`. |
| `gpu_ids` | int or list | `0` / `0,1` / `-1` | GPU device IDs. Single int = single GPU. Comma-separated list = multi-GPU DDP. `-1` = CPU. |

**Multi-GPU:** `gpu_ids 0,1` auto-enables DDP via `torch.multiprocessing.spawn` with NCCL backend. `world_size` and `use_distributed` are computed automatically — do not set them manually.

---

## Paths & Logging

| Key | Type | Example | Description |
|---|---|---|---|
| `modelpath` | str | `./outputs/warpage5.pth` | Path to save/load the checkpoint `.pth` file. |
| `log_file_dir` | str | `train5.log` | Path for the training loss log file. |
| `dataset_dir` | str | `./dataset/warpage.h5` | HDF5 dataset for training/validation/test. |
| `infer_dataset` | str | `./dataset/warpage_infer.h5` | HDF5 dataset used as initial conditions for inference rollout. |

---

## Feature Dimensions

These must match the actual layout of your HDF5 `nodal_data` array.

| Key | Type | Default | Description |
|---|---|---|---|
| `input_var` | int | — | Number of physical node input features (e.g., `3` for x/y/z displacement). Excludes node types and positional features — those are added automatically. |
| `output_var` | int | — | Number of output features the model predicts (must match `input_var` unless a different subset is predicted). |
| `edge_var` | int | `8` | Edge feature dimensionality. Must be `8` when using both deformed and reference edge features `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. |
| `positional_features` | int | `0` | Number of rotation-invariant node features to append. Feature 0 = centroid distance, Feature 1 = mean neighbor edge length, Feature 2+ = positional encoding (see `positional_encoding`). Set `0` to disable. |
| `positional_encoding` | str | `rwpe` | Encoding type for `positional_features > 2`: `rwpe` (random-walk PE at k=2,4,8,16,32), `lpe` (Laplacian eigenvectors), `rwpe+lpe` (both concatenated). |

---

## Network Architecture

| Key | Type | Example | Description |
|---|---|---|---|
| `Latent_dim` | int | `256` | Hidden dimension for all MLPs and latent node/edge embeddings throughout the network. |
| `message_passing_num` | int | `15` | Total number of GnBlock message-passing iterations. When multiscale is enabled, this is overridden by the sum of `mp_per_level` entries. |
| `residual_scale` | float | `1` | Scale factor on residual connections. `1.0` = full residual (matches DeepMind/NVIDIA). |
| `use_pairnorm` | bool | `False` | Apply PairNorm on aggregated messages to counteract over-smoothing in deep GNNs. |

**Activation:** SiLU (Swish) is hardcoded throughout all MLPs and is not configurable.

---

## Training Hyperparameters

| Key | Type | Example | Description |
|---|---|---|---|
| `Training_epochs` | int | `500` | Total number of training epochs. |
| `Batch_size` | int | `2` | Mini-batch size per GPU (for DDP, effective batch = Batch_size × world_size). |
| `LearningR` | float | `0.0001` | Peak learning rate after warmup. Scale by `sqrt(effective_batch / base_batch)` when changing batch size. |
| `num_workers` | int | `8` | DataLoader worker processes per GPU. |
| `grad_accum_steps` | int | `0` | Gradient accumulation. `0` = accumulate full epoch then step, `1` = step every batch, `N` = step every N batches. |

**Loss function:** Huber loss (delta=0.1, per-element, not reduced). The reduction is a weighted mean controlled by `feature_loss_weights`.

**LR schedule:** LinearLR warmup over 3 epochs (0.01× → 1.0×), then CosineAnnealingWarmRestarts with T_0 = (total_epochs − 3) // 3, T_mult = 2, eta_min = 1e-8. Scheduler steps after each optimizer step.

**Optimizer:** Fused Adam with no weight decay (matches DeepMind original).

**Gradient clipping:** max_norm = 3.0 (not configurable).

---

## Loss Weighting

| Key | Type | Example | Description |
|---|---|---|---|
| `feature_loss_weights` | list | `0.1, 0.1, 1.0` | Per-output-feature loss weights. Length must equal `output_var`. Auto-normalized to sum to 1 (so this controls relative importance, not absolute scale). Applied during train, val, and test. |

Example: `0.1, 0.1, 1.0` for `[x_disp, y_disp, z_disp]` de-emphasizes planar displacement and focuses on z-axis.

---

## Data Augmentation & Noise

| Key | Type | Default | Description |
|---|---|---|---|
| `augment_geometry` | bool | `False` | Random Z-rotation + X/Y reflection applied to the mesh during training only. Makes the model rotation/reflection invariant. |
| `std_noise` | float | `0.0` | Standard deviation of Gaussian noise added to both node features and edge features during training. Set `0.0` to disable. |

When `std_noise > 0`, the training target is corrected: `y -= gamma * noise * ratio` to account for the injected perturbation.

---

## Performance & Memory

| Key | Type | Default | Description |
|---|---|---|---|
| `use_amp` | bool | `False` | Mixed-precision training with **bfloat16** (not float16 — float16 causes scatter_add overflow in GNNs). ~1.5–2× speedup on Ampere+ GPUs. |
| `use_compile` | bool | `False` | Wrap model in `torch.compile(dynamic=True)` for kernel fusion. Adds compilation overhead on first run. |
| `use_checkpointing` | bool | `False` | Gradient checkpointing (recompute activations on backward pass). Reduces VRAM at the cost of ~30% more compute. Useful for large meshes or deep networks. |
| `use_ema` | bool | `False` | Maintain an Exponential Moving Average shadow model. EMA weights are used for validation and inference. Effectively free improved generalization. |
| `ema_decay` | float | `0.999` | EMA decay factor. `0.999` ≈ 1000-step averaging window. Only used when `use_ema True`. |

---

## Evaluation & Visualization

| Key | Type | Default | Description |
|---|---|---|---|
| `test_interval` | int | `10` | Run test evaluation and save plots every N epochs. `1` = every epoch. |
| `test_batch_idx` | list | `0, 1, 2` | Indices of test-set samples to visualize. |
| `plot_feature_idx` | int | `-1` | Which output feature to visualize in mesh plots. `-1` = last feature (e.g., stress if `output_var` includes it). |

---

## Dataset Splitting

| Key | Type | Default | Description |
|---|---|---|---|
| `split_seed` | int | `42` | Random seed for deterministic 80/10/10 train/val/test split. Change to get a different split. |

---

## Node Types

| Key | Type | Default | Description |
|---|---|---|---|
| `use_node_types` | bool | `False` | Append one-hot encoded node type categories to node features. Types are read from the dataset and concatenated after Z-score normalization. |

---

## World Edges (Long-Range Connections)

| Key | Type | Default | Description |
|---|---|---|---|
| `use_world_edges` | bool | `False` | Add long-range "world" edges between nodes within a radius threshold. Uses `torch_cluster.radius_graph`. Only applied at the finest scale in multiscale mode. |
| `world_edge_radius` | float | — | Spatial radius for world edge construction. Required when `use_world_edges True`. |

See `docs/WORLD_EDGES_DOCUMENTATION.md` for details.

---

## Multiscale (BFS Bi-Stride V-Cycle)

Based on the ICML 2023 hierarchical GNN coarsening scheme.

| Key | Type | Default | Description |
|---|---|---|---|
| `use_multiscale` | bool | `False` | Enable V-cycle hierarchical message passing. |
| `multiscale_levels` | int | `1` | Number of coarsening levels L. Adds L coarser graph representations. |
| `mp_per_level` | list | `2, 10, 2` | Message-passing steps at each stage of the V-cycle. Must have exactly `2*L + 1` entries for L levels: `[pre_0, pre_1, ..., coarsest, post_{L-1}, ..., post_0]`. For `L=1`: `[pre_fine, coarsest, post_fine]`. |

**Coarsening method:** BFS Bi-Stride. Even-depth BFS nodes → coarse graph; odd-depth nodes → mapped to their BFS parent. Handles disconnected components (multi-part meshes).

**Pool:** Mean aggregation. **Unpool:** Broadcast (no learned weights). Skip connections merge skip state with unpooled features via `Linear(2 × Latent_dim, Latent_dim)`.

---

## Inference Only

| Key | Type | Example | Description |
|---|---|---|---|
| `infer_timesteps` | int | `34` | Number of autoregressive rollout steps to predict. |

**Checkpoint loading:** Normalization stats are read from `checkpoint['normalization']`. If `ema_state_dict` is present, it is preferred over the base model weights. Model architecture config in the checkpoint overrides the config file (a warning is printed).

**Output file:** `rollout_sample{id}_steps{N}.h5` with `nodal_data` of shape `[8, timesteps, nodes]`.

---

## Full Example Config (Training)

```
model               MeshGraphNets
mode                train
gpu_ids             0,1              # DDP on 2 GPUs

log_file_dir        my_run.log
modelpath           ./outputs/my_model.pth

% Datasets
dataset_dir         ./dataset/train_data.h5
infer_dataset       ./dataset/infer_data.h5
infer_timesteps     34

% Features
input_var           3                # x_disp, y_disp, z_disp
output_var          3
edge_var            8
positional_features 4
positional_encoding rwpe

% Network
Latent_dim          256
message_passing_num 15

% Training
Training_epochs     500
Batch_size          2
LearningR           0.0001
num_workers         8
std_noise           0.003
augment_geometry    True
grad_accum_steps    1
feature_loss_weights  0.1, 0.1, 1.0

% Performance
use_amp             True
use_ema             True
ema_decay           0.999
use_checkpointing   False
use_compile         False

% Evaluation
test_interval       5
test_batch_idx      0, 1, 2, 3
plot_feature_idx    -1

% Multiscale
use_multiscale      True
multiscale_levels   1
mp_per_level        2, 10, 2
```

## Full Example Config (Inference)

```
model               MeshGraphNets
mode                inference
gpu_ids             0

modelpath           ./outputs/my_model.pth
infer_dataset       ./dataset/infer_data.h5
infer_timesteps     34

% These must match the checkpoint's saved config
input_var           3
output_var          3
edge_var            8
positional_features 4
```

> **Note:** For inference, the model architecture (Latent_dim, message_passing_num, multiscale settings, etc.) is loaded from the checkpoint and overrides the config file. You only need to specify paths and `infer_timesteps`.
