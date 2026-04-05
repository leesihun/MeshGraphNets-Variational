# MeshGraphNets Variational

A MeshGraphNets-based Graph Neural Network surrogate for FEA mesh simulations. The model predicts normalized node-state deltas and rolls them out autoregressively over time. This repository now supports both the original deterministic model and a variational version for manufacturing spread estimation.

Developed by SiHun Lee, Ph.D., MX, SEC. README updated for the April 2026 variational changes.

## Recent Changes

- Added an optional conditional VAE path with `use_vae`.
- Added a GNN variational encoder with message passing on target deltas and attention pooling to produce the global latent `z`.
- Added per-block latent injection so the processor receives `z` throughout message passing instead of only once.
- Added an auxiliary decoder loss (`beta_aux`) that makes `z` predict per-graph target-delta mean and std.
- Added VAE-aware evaluation logs: posterior reconstruction (`TrainEvalQ`, `ValidQ`) and prior-sampled validation (`ValidPrior@K`).
- Added `vae_valid_prior_samples` to control how many prior samples are averaged during VAE validation.
- Checkpoints now save `model_config`, VAE metadata, EMA weights, and per-level coarse-edge normalization statistics.
- Multiscale configs now support per-level `coarsening_type` (`bfs` or `voronoi`) and optional `bipartite_unpool`.

## What It Does

Given a mesh at time `t` (geometry plus physical state), the model predicts the normalized delta to time `t+1`. During rollout, the next state is recovered as:

```text
state_{t+1} = state_t + denormalize(delta_pred)
```

When `use_vae True`, training encodes `q(z | target_delta)` and inference samples `z ~ N(0, I)`. Re-running rollout with different latent samples gives different plausible outcomes for the same input geometry.

## Setup

Requires PyTorch and PyTorch Geometric. Key extra packages:

```bash
# HDF5 datasets
pip install h5py

# Optional: world edges
pip install torch-cluster
```

Set `HDF5_USE_FILE_LOCKING=FALSE` if you train from shared storage.

## Running

```bash
# Training (included example config: multiscale + VAE)
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt

# Inference / autoregressive rollout
python MeshGraphNets_main.py --config <your_inference_config.txt>
```

The `--config` flag defaults to `config.txt`.

- `mode` inside the config selects `train` or `inference`.
- `gpu_ids -1` forces CPU.
- One GPU ID runs single-GPU training.
- Multiple GPU IDs automatically enable DDP.
- For inference, set `modelpath`, `infer_dataset`, and `infer_timesteps` in the config.

See [config_run_docs.md](config_run_docs.md) for a full reference of configuration keys.

## Core Model

- Encode-Process-Decode MeshGraphNets with SiLU activations and sum aggregation.
- 8D edge features: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`.
- Gated encoder-to-decoder skip connection.
- Optional world edges on the finest level.
- Optional multiscale V-cycle with per-level `coarsening_type` (`bfs` or `voronoi`) and optional `bipartite_unpool`.

## Variational Model

- `GNNVariationalEncoder` runs message passing on the target delta field, then uses attention pooling to predict `mu` and `logvar`.
- The sampled global latent `z` is injected before every processor block in the flat model.
- In multiscale mode, `z` is injected at the finest-level pre and post blocks.
- The auxiliary decoder predicts per-graph target-delta mean and std from `z`.
- Training uses a weighted objective:

```text
alpha_recon * reconstruction + beta_kl * KL + beta_aux * auxiliary
```

- If `kl_anneal_epochs > 0`, `beta_kl` is linearly ramped during training.

## Key Config Groups

| Group | Keys |
|---|---|
| Mode / IO | `mode`, `gpu_ids`, `dataset_dir`, `infer_dataset`, `infer_timesteps`, `modelpath`, `log_file_dir` |
| Base model | `input_var`, `output_var`, `edge_var`, `Latent_dim`, `message_passing_num`, `feature_loss_weights` |
| Performance | `use_amp`, `use_ema`, `ema_decay`, `use_checkpointing`, `grad_accum_steps` |
| Optional features | `use_node_types`, `positional_features`, `positional_encoding`, `use_world_edges`, `world_edge_radius` |
| Multiscale | `use_multiscale`, `multiscale_levels`, `mp_per_level`, `coarsening_type`, `voronoi_clusters`, `bipartite_unpool` |
| VAE | `use_vae`, `vae_latent_dim`, `vae_mp_layers`, `beta_kl`, `beta_aux`, `alpha_recon`, `kl_anneal_epochs`, `vae_valid_prior_samples` |

## Dataset Format

Input HDF5 files contain a `nodal_data` dataset with shape **`[features, time, nodes]`**:

| Index | Feature |
|---|---|
| 0-2 | x, y, z (reference coordinates) |
| 3-5 | x_disp, y_disp, z_disp |
| 6 | stress (von Mises or equivalent) |
| 7 | part_number (optional) |

## Outputs

Training checkpoints are saved to the path specified by `modelpath`. A best checkpoint contains:

- `model_state_dict`
- `optimizer_state_dict` and scheduler state
- `normalization` stats for nodes, edges, deltas, and coarse edges when multiscale is enabled
- `model_config` for architecture reconstruction at inference time
- `ema_state_dict` when EMA is enabled
- `valid_prior_loss` and `valid_prior_samples` when `use_vae True`

For VAE runs, the best checkpoint is still selected by posterior validation reconstruction (`ValidQ`). Prior-sampled validation (`ValidPrior@K`) is logged and stored as extra context.

Inference outputs are saved as HDF5: `rollout_sample{id}_steps{N}.h5` with `nodal_data` shape `[8, timesteps, nodes]`.

Loss and validation curves are written to `log_file_dir`. Use `misc/plot_loss.py` or `misc/plot_loss_realtime.py` to visualize them.

## Docs

- [config_run_docs.md](config_run_docs.md): full config reference
- [VAE_IMPLEMENTATION_GUIDE.md](VAE_IMPLEMENTATION_GUIDE.md): detailed variational architecture notes
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): dataset conventions and examples
- [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md): multiscale coarsening details
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): world-edge construction and usage
