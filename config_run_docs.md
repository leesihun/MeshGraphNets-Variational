# Config File Reference

Complete reference for all config keys used by MeshGraphNets-V.

**Format:** `key   value   # inline comment`  
Full-line comments use `%`. Lists use comma-separated values. Booleans: `true`/`false`. Keys are case-insensitive. Config files live in `_warpage_input/`.

---

## Core: Mode & I/O

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | str | — | Display name for the model (e.g. `MeshGraphNets-V`). Not functional. |
| `mode` | str | — | **Required.** `train` or `inference`. |
| `gpu_ids` | int or list[int] | 0 | GPU device ID(s). Comma-separated list enables DDP (e.g. `0,1`). `-1` forces CPU. |
| `modelpath` | str | — | **Required.** Path to checkpoint `.pth` for saving (train) or loading (inference). |
| `log_file_dir` | str | None | Log file path. Omit to disable file logging. |
| `dataset_dir` | str | — | Training HDF5 dataset path. Required for `mode train`. |
| `infer_dataset` | str | — | Inference HDF5 dataset path. Required for `mode inference`. |

---

## Data & Features

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_var` | int | — | **Required.** Number of physical node features used as model input. Typically `3` (x/y/z disp) or `4` (+ stress). Does **not** include reference x/y/z positions (extracted separately). |
| `output_var` | int | — | **Required.** Number of predicted delta features. Usually equals `input_var`. |
| `edge_var` | int | 8 | Edge feature dimension. Must be `8` (deformed dx/dy/dz/dist + reference dx/dy/dz/dist). |
| `feature_loss_weights` | list[float] or float | None | Per-output-feature Huber loss weights. Auto-normalized to sum=1. E.g. `1, 1, 1.0`. If omitted, uniform weights. |
| `positional_features` | int | 0 | Number of rotation-invariant positional features appended to node features. Uses `positional_encoding` type. |
| `positional_encoding` | str | `rwpe` | Type of positional encoding: `rwpe` (random walk return probabilities), `lpe` (Laplacian eigenvectors), or `rwpe+lpe` (both, split evenly). |

---

## Model Architecture — Core

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `latent_dim` | int | — | **Required.** Hidden dimension for all MLPs (encoder, processor, decoder). Typical: `128`. |
| `message_passing_num` | int | — | Number of GnBlocks in flat (non-multiscale) mode. Ignored when `use_multiscale True`. |
| `residual_scale` | float | `1.0` | Residual connection scale factor: `h_out = h_in + residual_scale × MLP_out`. `1.0` = full residual (DeepMind/NVIDIA default). |

---

## Model Architecture — Multiscale V-Cycle

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_multiscale` | bool | `false` | Enable BFS Bi-Stride or FPS-Voronoi V-cycle multiscale processor. |
| `multiscale_levels` | int | 1 | Number of coarsening levels L. `mp_per_level` must have `2L+1` entries. |
| `coarsening_type` | str or list[str] | `bfs` | Coarsening algorithm per level: `bfs` (BFS Bi-Stride, ~4× reduction) or `voronoi` (FPS-Voronoi, configurable). Scalar applies to all levels; list sets per level. |
| `voronoi_clusters` | int or list[int] | 0 | Target cluster count for Voronoi coarsening per level. List sets per level. `0` = disabled. |
| `mp_per_level` | list[int] | — | Message-passing blocks per V-cycle level: `[pre_0, pre_1, ..., coarsest, post_{L-1}, ..., post_0]`. Length must be `2L+1`. E.g. for `multiscale_levels 1`: `4, 12, 4`. |
| `bipartite_unpool` | bool | `false` | `true` = learned bipartite message-passing unpool (fine ← coarse with skip connection). `false` = simple broadcast (gather). |

---

## Node Types

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_node_types` | bool | `false` | Append one-hot encoded node type features to node embeddings. Features added **after** normalization. |

---

## World Edges (Collision / Radius Edges)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_world_edges` | bool | `false` | Enable radius-based collision edges beyond mesh topology. Only at finest level in multiscale. |
| `world_radius_multiplier` | float | — | World edge radius = `mean_edge_len × world_radius_multiplier`. Used to compute `world_edge_radius` at dataset build time. |
| `world_max_num_neighbors` | int | 64 | Maximum collision edge neighbors per node. Applies to `torch_cluster` backend only. |
| `world_edge_backend` | str | `scipy_kdtree` | `torch_cluster` (GPU-accelerated, requires `torch-cluster`) or `scipy_kdtree` (CPU fallback). |

---

## Variational Autoencoder (MMD-InfoVAE)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_vae` | bool | `false` | Enable conditional VAE. Encodes target `y` during training to produce latent `z`. |
| `vae_latent_dim` | int | 32 | Dimension of the VAE latent code `z`. |
| `vae_mp_layers` | int | 5 | Number of GnBlocks in `GNNVariationalEncoder`. |
| `alpha_recon` | float | `1.0` | Reconstruction loss weight: `α × Huber(ŷ, y)`. |
| `lambda_mmd` | float | `100.0` | MMD loss weight: `λ × MMD²(z, N(0,I))`. Controls aggregate posterior–prior matching. Typical range: `0.01–1.0`. |
| `beta_aux` | float | `1.0` | Auxiliary prediction loss weight: `β × MSE(aux(z), [y_mean, y_std])`. |
| `num_vae_samples` | int | 1 | Number of independent `z` samples drawn per inference rollout (stochastic trajectories). |
| `vae_valid_prior_samples` | int | 8 | Number of prior samples averaged during validation with `use_posterior False`. |

### Post-hoc GMM Prior

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit_latent_gmm` | bool | `false` | After training, fit a GMM on posterior means from the full training set and save to checkpoint. At inference, samples `z` from GMM instead of `N(0,I)`. |
| `gmm_components` | int | 10 | Number of GMM components (Gaussians). Capped to number of training samples if smaller. |
| `gmm_covariance_type` | str | `full` | GMM covariance structure: `full`, `diag`, `tied`, or `spherical`. |
| `gmm_reg_covar` | float | `1e-4` | Diagonal regularizer added to each covariance for numerical stability. On `LinAlgError` (singular covariance), the fitter auto-retries with `10×`, `100×`, `1000×` this value, then falls back to `diag`. |

---

## Training Hyperparameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `training_epochs` | int | — | **Required.** Total number of training epochs. |
| `batch_size` | int | — | **Required.** Batch size per GPU. |
| `learningr` | float | — | **Required.** Peak learning rate for Adam. |
| `warmup_epochs` | int | 3 | Linear LR warmup from 0% to 100% of `learningr` over N epochs. |
| `num_workers` | int | — | **Required.** DataLoader worker processes. `0` = main thread only. |
| `split_seed` | int | 42 | RNG seed for 80/10/10 train/val/test split. Change to produce different splits. |
| `grad_accum_steps` | int | 1 | Gradient accumulation. `1` = update each batch. `0` = accumulate full epoch (one update per epoch). `N` = update every N batches. |

---

## Data Augmentation & Noise

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `augment_geometry` | bool | `false` | Apply random Z-rotation + X/Y reflection during training. Disabled at validation/inference. |
| `std_noise` | float | 0.0 | Standard deviation of Gaussian noise added to node features and edge features during training. Trains robustness to observation noise. |

---

## Memory & Performance Optimization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_checkpointing` | bool | `false` | Activation/gradient checkpointing: recomputes forward activations during backward instead of storing them. Reduces memory ~60–70% at cost of ~20–30% compute. |
| `use_amp` | bool | `true` | Mixed precision with **bfloat16** (not float16). Requires Ampere+ GPU (A100, H100, RTX 30xx+). ~1.5–2× speedup. |
| `use_compile` | bool | `false` | `torch.compile(model, dynamic=True)` optimization. PyTorch 2.1+. Adds cold-start overhead but speeds up steady-state training. |
| `use_ema` | bool | `false` | Exponential Moving Average shadow model. Validation and inference always prefer EMA weights when available. |
| `ema_decay` | float | 0.999 | EMA decay factor. `0.999` ≈ 1000-step averaging window. `0.99` ≈ 100-step (responds faster). |
| `use_parallel_stats` | bool | `false` | Parallel preprocessing stats computation across dataset samples. |

---

## Validation, Testing & Logging

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `val_interval` | int | 1 | Run validation every N epochs. |
| `test_interval` | int | 10 | Run full test evaluation + visualizations every N epochs. |
| `test_batch_idx` | list[int] | `[0,1,2,3]` | Indices of test samples to visualize (create plots/HDF5). |
| `test_max_batches` | int | 200 | Max test batches evaluated in DDP mode (prevents NCCL timeout). |
| `display_testset` | bool | `true` | Create 3D PyVista mesh plots for test samples. |
| `display_trainset` | bool | `true` | Visualize training set reconstruction during test runs. |
| `plot_feature_idx` | int | -1 | Feature channel index to colorize in 3D plots. `-1` = last feature (typically stress). |
| `verbose` | bool | `false` | Print per-feature losses, gradient norms, memory usage. |
| `monitor_gradients` | bool | `false` | Log gradient statistics per layer to the log file. |

---

## Inference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `infer_timesteps` | int | — | Number of autoregressive rollout steps. If omitted, uses the dataset's timestep count. |
| `inference_output_dir` | str | `outputs/rollout` | Directory for rollout HDF5 output files. |
| `num_vae_samples` | int | 1 | Number of independent latent codes (`z`) to sample per rollout sample. Each produces a distinct trajectory. |

---

## Example Configs

### Minimal Training (Flat GNN, No VAE)

```
model       MeshGraphNets-V
mode        train
gpu_ids     0
modelpath   ./outputs/model.pth
dataset_dir ./dataset/data.h5

input_var               3
output_var              3
message_passing_num     10
latent_dim              128

training_epochs         1000
batch_size              16
learningr               0.0001
num_workers             4

use_amp                 true
use_ema                 true
ema_decay               0.99
```

### Full Training (Multiscale + VAE + DDP)

```
model       MeshGraphNets-V
mode        train
gpu_ids     0,1
modelpath   ./outputs/warpage1.pth
dataset_dir ./dataset/b8_train.h5

input_var               3
output_var              3
feature_loss_weights    1, 1, 1.0
positional_features     4

latent_dim              128
std_noise               0.01
residual_scale          1
augment_geometry        true
grad_accum_steps        1

use_checkpointing       true
use_amp                 true
use_ema                 true
ema_decay               0.99

training_epochs         2000
batch_size              16
learningr               0.0001
num_workers             8
test_interval           100
val_interval            1

use_multiscale          true
coarsening_type         voronoi
voronoi_clusters        200
multiscale_levels       1
mp_per_level            4, 12, 4
bipartite_unpool        true

use_vae                 true
vae_latent_dim          4
alpha_recon             1
lambda_mmd              0.1

fit_latent_gmm          true
gmm_components          10
gmm_covariance_type     full
```

### Inference (Stochastic Rollout)

```
model           MeshGraphNets-V
mode            inference
gpu_ids         0
modelpath       ./outputs/warpage1.pth
infer_dataset   ./dataset/infer_dataset_b8_test.h5
infer_timesteps 1
num_vae_samples 1000

input_var       3
output_var      3
```

---

## Config Keys Quick Lookup

| Category | Key(s) |
|----------|--------|
| **Mode/IO** | `mode`, `gpu_ids`, `modelpath`, `dataset_dir`, `infer_dataset`, `log_file_dir` |
| **Features** | `input_var`, `output_var`, `edge_var`, `feature_loss_weights`, `positional_features`, `positional_encoding` |
| **Architecture** | `latent_dim`, `message_passing_num`, `residual_scale` |
| **Multiscale** | `use_multiscale`, `multiscale_levels`, `coarsening_type`, `voronoi_clusters`, `mp_per_level`, `bipartite_unpool` |
| **Node types** | `use_node_types` |
| **World edges** | `use_world_edges`, `world_radius_multiplier`, `world_max_num_neighbors`, `world_edge_backend` |
| **VAE** | `use_vae`, `vae_latent_dim`, `vae_mp_layers`, `alpha_recon`, `lambda_mmd`, `beta_aux`, `num_vae_samples`, `vae_valid_prior_samples` |
| **GMM** | `fit_latent_gmm`, `gmm_components`, `gmm_covariance_type` |
| **Training** | `training_epochs`, `batch_size`, `learningr`, `warmup_epochs`, `num_workers`, `split_seed`, `grad_accum_steps` |
| **Augmentation** | `augment_geometry`, `std_noise` |
| **Optimization** | `use_checkpointing`, `use_amp`, `use_compile`, `use_ema`, `ema_decay`, `use_parallel_stats` |
| **Eval/Logging** | `val_interval`, `test_interval`, `test_batch_idx`, `test_max_batches`, `plot_feature_idx`, `verbose`, `monitor_gradients` |
| **Inference** | `infer_timesteps`, `inference_output_dir`, `num_vae_samples` |
