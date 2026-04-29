# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---

## Running the Code

```bash
# Training (single GPU or CPU — set gpu_ids in config)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Training (multi-GPU DDP — set gpu_ids 0,1 in config)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Inference / rollout
python MeshGraphNets_main.py --config _warpage_input/config_infer3.txt
```

`--config` defaults to `config.txt`. Mode (`train`/`inference`) is set inside the config. Single vs. multi-GPU is auto-detected from `gpu_ids`. `gpu_ids -1` forces CPU. There are no tests or linters in this project.

---

## Config File Format

Key-value pairs, one per line. `%` for full-line comments, `#` for inline comments. Keys are lowercased. CSV values become lists. Booleans: `true`/`false`. Config files live in `_warpage_input/`. See `config_run_docs.md` for the complete key reference.

---

## Key Files

| File | Role |
|------|------|
| `MeshGraphNets_main.py` | Entry point — routes to training or inference based on `mode` |
| `model/MeshGraphNets.py` | Top-level `MeshGraphNets` wrapper; applies noise, calls EncoderProcessorDecoder |
| `model/encoder_decoder.py` | `Encoder`, `GnBlock`, `Decoder`; flat and multiscale processor logic |
| `model/blocks.py` | `EdgeBlock`, `NodeBlock`, `HybridNodeBlock`, `UnpoolBlock` |
| `model/mlp.py` | `build_mlp` utility; Kaiming init + zero bias; SiLU activation (hardcoded) |
| `model/vae.py` | `GNNVariationalEncoder` — conditional MMD-VAE encoder + reparameterization |
| `model/latent_gmm.py` | Post-hoc GMM fitting on posterior means; inference sampling |
| `model/coarsening.py` | BFS Bi-Stride and FPS-Voronoi coarsening; pool/unpool operations |
| `model/checkpointing.py` | Gradient checkpointing wrapper (`use_reentrant=False`) |
| `training_profiles/setup.py` | Builds datasets, model, EMA, optimizer, scheduler |
| `training_profiles/training_loop.py` | `train_epoch`, `validate_epoch`, `test_model`, VAE evaluation variants |
| `training_profiles/single_training.py` | Single-GPU training entry; saves normalization to checkpoint |
| `training_profiles/distributed_training.py` | DDP launcher with NCCL; auto-detects free port |
| `inference_profiles/rollout.py` | Autoregressive rollout; outputs HDF5 |
| `general_modules/mesh_dataset.py` | `MeshGraphDataset`: HDF5 loading, normalization, positional encoding, augmentation |
| `general_modules/edge_features.py` | 8-D edge feature computation (deformed + reference geometry) |
| `general_modules/world_edges.py` | Radius-based collision edges (torch_cluster GPU or scipy KDTree fallback) |
| `general_modules/multiscale_helpers.py` | Builds and attaches multiscale coarsening hierarchy to PyG Data |
| `general_modules/mesh_utils_fast.py` | GPU-accelerated mesh utilities; PyVista 3D rendering |
| `general_modules/load_config.py` | Config parser |
| `build_dataset.py` | HDF5 dataset builder from ANSYS/FEA export |
| `dataset/generate_inference_dataset.py` | Extracts initial conditions for inference |
| `dataset/reduce_dataset.py` | Dataset subsampling utility |
| `animate_h5.py` | Animated GIF generation from HDF5 rollout output |
| `misc/plot_loss.py` | Loss curve plotting from log file |
| `misc/plot_loss_realtime.py` | Real-time loss monitoring via FastAPI |

---

## Architecture Overview

**Encode–Process–Decode GNN** on FEA mesh graphs, with optional MMD-VAE (InfoVAE) conditioning and multiscale V-cycle.

- **Input:** Nodes carry physical state (displacements, stress) + optional positional features + optional one-hot node types. Edges are 8-D: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. Always bidirectional.
- **Prediction target:** Normalized state deltas (`Δstate = state_{t+1} − state_t`). Rollout: `state_{t+1} = state_t + denormalize(delta_pred)`.
- **Encoder:** MLP per node, MLP per edge → latent embeddings.
- **Processor (flat):** Stack of `message_passing_num` GnBlocks. Each block: EdgeBlock (MLP on `[sender‖receiver‖edge]`) then NodeBlock (sum-aggregate msgs, MLP update). Residual on both nodes and edges.
- **Processor (multiscale):** BFS Bi-Stride V-cycle. Pool = mean, Unpool = broadcast or learned bipartite MP. Skip connections via `Linear(2·D, D)`. World edges only at finest level.
- **MMD-VAE conditioning (`use_vae True`):** `GNNVariationalEncoder` encodes target `y` → `(μ, log σ²)` → `z` via reparameterization; `z` is fused into each GnBlock. Regularizer is multi-scale RBF MMD² between the batch of `z` samples and `N(0, I)` (InfoVAE, Zhao et al. 2019). Avoids posterior collapse; `z ~ N(0, I)` at inference lands in the decoder's training support. Optionally (`fit_latent_gmm True`), a GMM is fit on the training-set posterior means after training and saved to checkpoint; rollout samples from GMM instead of N(0, I).
- **Decoder:** Single MLP, **no LayerNorm on output**. Last-layer weights ×0.01 when T>1.
- **Activation:** SiLU (Swish) hardcoded — not configurable.
- **Aggregation:** Sum (not mean) in NodeBlock.
- **Normalization:** Z-score from training split only, stored in checkpoint under `checkpoint['normalization']`. Node types concatenated *after* normalization.

### MLP structure

Every encoder/processor MLP: `Linear → SiLU → Linear → SiLU → Linear → LayerNorm`.  
Decoder MLP: same but **no final LayerNorm**.  
Init: Kaiming uniform (ReLU nonlinearity) + zero bias. Decoder final layer × 0.01.

---

## Training Details

| Aspect | Value |
|--------|-------|
| Loss | Huber (δ=1.0) with optional per-feature weights |
| Optimizer | Fused Adam, no weight decay |
| LR schedule | `LinearLR` warmup (default 3 epochs) → `CosineAnnealingWarmRestarts` |
| Grad clip | `max_norm=3.0` |
| AMP | `bfloat16` via `torch.amp.autocast` — use bfloat16, **not** float16 (float16 overflows `scatter_add`) |
| EMA | Optional (`use_ema True`), inference prefers EMA weights |
| Dataset split | 80/10/10 train/val/test, deterministic via `split_seed` (default 42) |
| Augmentation | Z-rotation + X/Y reflection (train only); Gaussian noise on nodes+edges |
| VAE total loss | `α·Huber + λ·MMD(z, N(0,I)) + β·MSE(aux(z), [y_mean, y_std])` |

### Gradient accumulation

`grad_accum_steps 1` = per-batch update. `grad_accum_steps 0` = full epoch accumulation (loss averaged over all batches before backward).

---

## HDF5 Data Format

**Training dataset** — per sample, `nodal_data` shape `[features, timesteps, nodes]`:
- Default layout: `[x, y, z, x_disp, y_disp, z_disp, stress, part_number]`
- `x, y, z` are reference positions (not counted in `input_var`)

**Preprocessing:** normalization stats, train/val/test split indices, and multiscale coarse edge stats are written back to HDF5.

**Rollout output:** `rollout_sample{id}_steps{N}.h5`, `nodal_data` shape `[3 + output_var + 1, timesteps, nodes]`.

**Checkpoints:** model weights, optimizer state, `checkpoint['normalization']`, optionally `ema_state_dict`, `coarse_edge_means/stds`, `gmm_params`.

---

## Important Implementation Details

- **`HDF5_USE_FILE_LOCKING=FALSE`**: set in `mesh_dataset.py` to avoid locking issues with multi-worker DataLoaders.
- **Decoder init ×0.01**: when `num_timesteps > 1`, final decoder layer weights are scaled down to bias the model toward "predict zero change" initially.
- **World edge normalization**: world edges are z-score normalized using separate stats computed from the training set, stored under `checkpoint['normalization']['world_edge_mean/std']`.
- **Fixed-z rollout**: a single `z` is sampled once per rollout trajectory and held constant across all time steps for temporal consistency.
- **Test batch capping in DDP**: `test_max_batches` (default 200) prevents NCCL timeout when rank 0 evaluates while others wait at `dist.barrier()`.
- **bfloat16 required over float16**: `scatter_add` overflows with float16 on typical FEA mesh sizes.
- **`use_compile True`**: uses `torch.compile(model, dynamic=True)` to handle variable batch sizes and graph sizes.
- **Multiscale topology is static**: coarsening is computed once per sample during dataset construction and stored in the HDF5; rollout rebuilds it from stored data, not re-computed each step.

---

## Checkpoint Structure

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'ema_state_dict': dict,           # optional
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'train_loss': float,
    'valid_loss': float,
    'normalization': {
        'node_mean': tensor[input_var],
        'node_std':  tensor[input_var],
        'edge_mean': tensor[8],
        'edge_std':  tensor[8],
        'delta_mean': tensor[output_var],
        'delta_std':  tensor[output_var],
        'coarse_edge_means': list[tensor],   # per level
        'coarse_edge_stds':  list[tensor],
        'world_edge_mean': tensor[8],        # optional
        'world_edge_std':  tensor[8],
        'node_type_to_idx': dict,            # optional
        'world_edge_radius': float,          # optional
    },
    'model_config': { ... },
    'gmm_params': {                   # optional, if fit_latent_gmm=True
        'weights', 'means', 'covariances', 'covariance_type', 'n_components'
    },
}
```
