# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Running the Code

```bash
# Training (single GPU or CPU — set gpu_ids in config)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Training (multi-GPU DDP — set multiple gpu_ids, e.g. gpu_ids 0,1,2,3)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Inference / rollout
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

The `--config` flag is optional; defaults to `config.txt` in the working directory. Mode is controlled by the `mode` key (`train` or `inference`). Single vs. multi-GPU is auto-detected from `gpu_ids`. `gpu_ids -1` forces CPU.

There are no tests or linters configured in this project.

## Config File Format

Key-value pairs, one per line. `%` for full-line comments, `#` for inline comments. Keys are lowercased. CSV values become lists, space-separated become arrays, booleans are `true`/`false`. Config files live in `_warpage_input/`.

### Complete Config Key Reference

| Key | Type | Description |
|-----|------|-------------|
| `model` | str | Model name identifier (e.g. `MeshGraphNets-V`) |
| `mode` | str | `train` or `inference` |
| `gpu_ids` | int/list | GPU device IDs; comma-separated for DDP; `-1` for CPU |
| `log_file_dir` | str | Log output path |
| `modelpath` | str | Checkpoint save/load path |
| `dataset_dir` | str | Training HDF5 dataset path |
| `infer_dataset` | str | Inference HDF5 dataset path |
| `infer_timesteps` | int | Number of autoregressive rollout steps |
| `input_var` | int | Input node feature dimension (physical vars) |
| `output_var` | int | Output feature dimension (predicted delta vars) |
| `edge_var` | int | Edge feature dim — must be `8` for full deformed+reference set |
| `feature_loss_weights` | list | Per-output-feature Huber loss weights (auto-normalized) |
| `positional_features` | int | Rotation-invariant positional features appended to nodes |
| `positional_encoding` | str | `rwpe`, `lpe`, or `rwpe+lpe` |
| `message_passing_num` | int | Total GnBlocks (flat) or ignored when using `mp_per_level` (multiscale) |
| `latent_dim` | int | Hidden dimension throughout model |
| `training_epochs` | int | Total training epochs |
| `batch_size` | int | Batch size per GPU |
| `learningr` | float | Peak learning rate |
| `warmup_epochs` | int | Linear LR warmup epochs (default 3) |
| `num_workers` | int | DataLoader workers |
| `split_seed` | int | Deterministic train/val/test split seed (default 42) |
| `std_noise` | float | Gaussian noise std on nodes+edges during training |
| `residual_scale` | float | Residual connection scale factor |
| `augment_geometry` | bool | Z-rotation + X/Y reflection augmentation (train only) |
| `grad_accum_steps` | int | Gradient accumulation steps |
| `use_checkpointing` | bool | Activation checkpointing to save GPU memory |
| `use_amp` | bool | bfloat16 AMP — use bfloat16, **not** float16; float16 overflows scatter_add |
| `use_ema` | bool | Exponential moving average shadow model |
| `ema_decay` | float | EMA decay factor (default 0.99) |
| `use_compile` | bool | `torch.compile` optimization |
| `test_interval` | int | Epoch interval for validation evaluation |
| `use_node_types` | bool | Concatenate one-hot node type features after normalization |
| `use_world_edges` | bool | Radius-based non-mesh collision/contact edges |
| `world_radius_multiplier` | float | World edge radius = mean_edge_len × this |
| `world_max_num_neighbors` | int | Max world edge degree per node |
| `world_edge_backend` | str | `torch_cluster` or `scipy_kdtree` |
| `use_multiscale` | bool | BFS Bi-Stride V-cycle multiscale architecture |
| `multiscale_levels` | int | Number of coarsening levels L; `mp_per_level` needs 2L+1 entries |
| `coarsening_type` | str | `bfs` or `voronoi` |
| `voronoi_clusters` | int/list | Target cluster counts per Voronoi level |
| `mp_per_level` | list | MP blocks per level: `[pre, coarse_1, ..., coarse_L, post]` |
| `bipartite_unpool` | bool | Learned bipartite message-passing unpool (vs. broadcast) |
| `use_vae` | bool | Conditional VAE latent conditioning |
| `vae_latent_dim` | int | VAE latent code dimension |
| `vae_mp_layers` | int | GnBlocks in VAE encoder |
| `lambda_mmd` | float | Weight on MMD between aggregate posterior q(z) and N(0,I) |
| `alpha_recon` | float | Reconstruction loss weight |
| `beta_aux` | float | Auxiliary prediction loss weight |
| `fit_latent_gmm` | bool | Fit GMM on posterior means after training; stored in checkpoint |
| `gmm_components` | int | Number of GMM components (capped to n_train if smaller) |
| `gmm_covariance_type` | str | `full` \| `diag` \| `tied` \| `spherical` |
| `use_parallel_stats` | bool | Parallel preprocessing stats computation |

## Architecture Overview

**Encode–Process–Decode GNN** on FEA mesh graphs, with optional MMD-VAE (InfoVAE) conditioning and multiscale V-cycle.

- **Input:** Nodes carry physical state (displacements, stress) + optional positional features + optional one-hot node types. Edges are 8D: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. Always bidirectional.
- **Prediction target:** Normalized state deltas (`Δstate = state_{t+1} − state_t`). Rollout: `state_{t+1} = state_t + denormalize(delta_pred)`.
- **Encoder:** MLP per node, MLP per edge → latent embeddings.
- **Processor (flat):** Stack of `message_passing_num` GnBlocks. Each block: EdgeBlock (MLP on `[sender‖receiver‖edge]`) then NodeBlock (sum-aggregate msgs, MLP update). Residual on both nodes and edges.
- **Processor (multiscale):** BFS Bi-Stride V-cycle. Pool = mean, Unpool = broadcast or learned bipartite. Skip connections via `Linear(2·D, D)`. World edges only at finest level.
- **MMD-VAE conditioning (`use_vae True`):** `GNNVariationalEncoder` encodes target `y` → `(μ, log σ²)` → `z` via reparameterization; `z` is fused into each GnBlock. Training regularizer is a multi-scale RBF MMD² between the batch of reparameterized `z` samples and `N(0, I)` (Zhao et al. 2019, InfoVAE). This directly matches the aggregate posterior to the prior — avoiding posterior collapse and the prior/posterior mismatch that afflicts standard KL-VAE — so `z ∼ N(0, I)` sampling at inference lands in the decoder's training support. Optionally (`fit_latent_gmm True`), a GMM is fit on the training-set posterior means after training and saved to the checkpoint; rollout then samples from the GMM instead of `N(0, I)`, capturing the true shape of the aggregate posterior.
- **Decoder:** Single MLP, **no LayerNorm on output**. Last-layer weights ×0.01 when T>1.
- **Activation:** SiLU (Swish) hardcoded — not configurable.
- **Aggregation:** Sum (not mean) in NodeBlock.
- **Normalization:** Z-score from training split only, stored in checkpoint under `checkpoint['normalization']`. Node types concatenated *after* normalization.

## Training Details

| Aspect | Value |
|--------|-------|
| Loss | Huber (δ=0.1) with optional per-feature weights |
| Optimizer | Fused Adam, no weight decay |
| LR schedule | LinearLR warmup → CosineAnnealingWarmRestarts |
| Grad clip | max_norm=3.0 |
| AMP | bfloat16 via `torch.amp.autocast` |
| EMA | Optional (`use_ema True`), inference prefers EMA weights |
| Dataset split | 80/10/10 train/val/test, deterministic via `split_seed` (default 42) |
| Augmentation | Z-rotation + X/Y reflection (train only); Gaussian noise on nodes+edges |

## HDF5 Data Format

**Input dataset** — `nodal_data` shape `[features, time, nodes]`:
- Features: `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`

**Preprocessing writes to HDF5:** normalization stats, train/val/test splits, and multiscale coarse edge stats.

**Rollout output:** `rollout_sample{id}_steps{N}.h5`, `nodal_data` shape `[8, timesteps, nodes]`.

**Checkpoints:** model weights, optimizer state, `checkpoint['normalization']`, optionally `ema_state_dict`, `coarse_edge_means`/`coarse_edge_stds`.

## Key Files

| File | Role |
|------|------|
| `MeshGraphNets_main.py` | Entry point; routes to training or inference |
| `model/MeshGraphNets.py` | Full Encode-Process-Decode model class |
| `model/encoder_decoder.py` | Encoder, GnBlock, Decoder |
| `model/blocks.py` | EdgeBlock, NodeBlock, HybridNodeBlock, UnpoolBlock |
| `model/mlp.py` | `build_mlp` utility and weight initialization |
| `model/vae.py` | GNNVariationalEncoder (conditional MMD-VAE) |
| `model/latent_gmm.py` | Post-hoc GMM fitting + inference sampling on VAE latent space |
| `model/coarsening.py` | BFS Bi-Stride and Voronoi FPS coarsening, pool/unpool |
| `model/checkpointing.py` | Activation checkpointing wrapper |
| `training_profiles/setup.py` | Dataset/model/optimizer builders |
| `training_profiles/training_loop.py` | Loss, optimizer, epoch loop, EMA, VAE losses |
| `training_profiles/single_training.py` | Single-GPU launcher; saves normalization to checkpoint |
| `training_profiles/distributed_training.py` | DDP launcher with NCCL and signal handling |
| `inference_profiles/rollout.py` | Autoregressive rollout; outputs HDF5 |
| `general_modules/mesh_dataset.py` | Dataset class: normalization, positional encoding, augmentation |
| `general_modules/edge_features.py` | 8D edge feature computation (deformed + reference) |
| `general_modules/world_edges.py` | Radius-based collision edges (torch_cluster or scipy KDTree) |
| `general_modules/multiscale_helpers.py` | Multiscale hierarchy building and attachment |
| `general_modules/mesh_utils_fast.py` | GPU-accelerated mesh utilities, PyVista rendering |
| `general_modules/load_config.py` | Config file parser |
| `build_dataset.py` | HDF5 dataset builder from ANSYS/FEA export data |
| `dataset/generate_inference_dataset.py` | Extract initial conditions for inference |
| `dataset/reduce_dataset.py` | Dataset subsampling utility |
| `animate_h5.py` | Animated GIF generation from HDF5 rollout output |
| `misc/plot_loss.py` | Loss curve plotting |
| `misc/plot_loss_realtime.py` | Real-time loss monitoring via FastAPI |
