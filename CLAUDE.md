# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Code

```bash
# Training (single GPU or CPU — set gpu_ids in config)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Training (multi-GPU DDP — set multiple gpu_ids, e.g. gpu_ids 0,1,2,3)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Inference / rollout
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

The `--config` flag is optional; it defaults to `config.txt` in the working directory. Mode is controlled by the `mode` key in the config file (`train` or `inference`). Single vs. multi-GPU is auto-detected based on how many IDs are listed in `gpu_ids`. `gpu_ids -1` forces CPU.

There are no tests or linters configured in this project.

## Config File Format

Key-value pairs, one per line. `%` for full-line comments, `#` for inline comments. Keys are lowercased. CSV values become lists, space-separated become arrays, booleans are `true`/`false`. Example configs live in `_warpage_input/`.

**Key config parameters:**
- `mode` — `train` or `inference`
- `gpu_ids` — GPU IDs (comma-separated for DDP, `-1` for CPU)
- `input_var`, `output_var` — number of physical node/output features
- `edge_var` — must be `8` for the full deformed+reference edge feature set
- `message_passing_num` — total GNN layers (or sum of `mp_per_level` for multiscale)
- `Latent_dim` — hidden dimension throughout
- `feature_loss_weights` — per-output-feature loss weighting (auto-normalized)
- `use_amp True` — enable bfloat16 AMP (use bfloat16, not float16; scatter_add overflows with float16)
- `use_multiscale True`, `multiscale_levels L`, `mp_per_level` — BFS V-cycle (requires 2L+1 entries in `mp_per_level`)
- `positional_features N` — appends N rotation-invariant features; `positional_encoding rwpe|lpe|rwpe+lpe`

**Inference-specific keys:** `modelpath`, `infer_dataset`, `infer_timesteps`

## Architecture Overview

**Encode–Process–Decode GNN** operating on FEA mesh graphs.

- **Input representation:** Nodes carry physical state (displacements, stress) + optional positional features + optional one-hot node types. Edges carry 8D features: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. Edges are always bidirectional.
- **Prediction target:** Normalized state *deltas* (`Δstate = state_{t+1} − state_t`). The model never predicts absolute values. During rollout, `state_{t+1} = state_t + denormalize(delta_pred)`.
- **Encoder:** MLP per node, MLP per edge → latent embeddings.
- **Processor:** Stack of `message_passing_num` GnBlocks. Each block: EdgeBlock (MLP on concatenated sender/receiver/edge) then NodeBlock (sum-aggregate edge msgs, MLP update). Residual connections on both nodes and edges. Gated skip from encoder output to post-processor (learnable sigmoid gate).
- **Decoder:** Single MLP, **no LayerNorm on output layer**. Last-layer weights scaled ×0.01 for T>1 (predict-no-change prior).
- **Activation:** SiLU (Swish) hardcoded everywhere — not configurable.
- **Aggregation:** Sum (not mean) in NodeBlock.

**Optional multiscale (BFS Bi-Stride V-cycle, `use_multiscale True`):** Coarsens graph topology using BFS; even-depth nodes form coarse graph. Pool = mean, Unpool = broadcast. Skip connections via `Linear(2·D, D)`. World edges only at finest level.

**Normalization:** Z-score, computed from training split only, saved in checkpoint under `checkpoint['normalization']`. Node types one-hot encoded and concatenated *after* normalization.

## Training Details

| Aspect | Value |
|---|---|
| Loss | Huber (delta=0.1) with optional per-feature weights |
| Optimizer | Fused Adam, no weight decay |
| LR schedule | LinearLR warmup (3 epochs) → CosineAnnealingWarmRestarts; steps per optimizer step |
| Grad clip | max_norm=3.0 |
| AMP | bfloat16 via `torch.amp.autocast` |
| EMA | Optional (`use_ema True`), inference prefers EMA weights if present |
| Dataset split | 80/10/10 train/val/test, deterministic via `split_seed` (default 42) |
| Augmentation | Geometry (Z-rotation + X/Y reflection, train only); noise on nodes+edges with target correction |

## Key Files

| File | Role |
|---|---|
| `MeshGraphNets_main.py` | Entry point; routes to training or inference |
| `model/MeshGraphNets.py` | Full Encode-Process-Decode model class |
| `model/blocks.py` | EdgeBlock, NodeBlock, HybridNodeBlock (world edges) |
| `model/coarsening.py` | BFS Bi-Stride coarsening, pool/unpool |
| `training_profiles/training_loop.py` | Loss, optimizer, epoch loop |
| `training_profiles/single_training.py` | Single-GPU launcher; saves normalization to checkpoint |
| `training_profiles/distributed_training.py` | DDP launcher with NCCL, signal handling |
| `inference_profiles/rollout.py` | Autoregressive rollout; outputs HDF5 |
| `general_modules/mesh_dataset.py` | Dataset class, normalization, positional encoding, augmentation |
| `general_modules/edge_features.py` | 8D edge feature computation (deformed + reference) |
| `general_modules/load_config.py` | Config parser |

## HDF5 Data Format

Input datasets: nodal_data shape `[features, time, nodes]` where features = `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`.

Rollout output: `rollout_sample{id}_steps{N}.h5` with nodal_data shape `[8, timesteps, nodes]`.

Checkpoints store model weights, optimizer state, normalization stats (`checkpoint['normalization']`), and optionally `ema_state_dict` and `coarse_edge_means`/`coarse_edge_stds` for multiscale.
