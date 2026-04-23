# MeshGraphNets — Variational

A Graph Neural Network surrogate for FEA (Finite Element Analysis) simulations implementing an **Encode–Process–Decode** architecture with optional **Conditional VAE**, **Multiscale V-cycle coarsening**, and **RealNVP Normalizing Flow** latent prior.

Designed for learning and autoregressively rolling out time-transient structural mechanics simulations (warpage, displacement, stress) on unstructured FEA meshes.

*Developed by SiHun Lee, Ph.D. — MX, Samsung Electronics.*

---

## Features

- **GNN surrogate for FEA** — predicts normalized state deltas autoregressively on mesh graphs
- **Conditional VAE** — encodes target trajectory into a global latent `z`; supports stochastic rollout at inference
- **RealNVP normalizing flow** — structured latent sampling at inference after post-training flow fitting
- **Multiscale V-cycle** — BFS Bi-Stride or Voronoi FPS coarsening with pool/unpool and skip connections
- **World edges** — radius-based collision/contact edges alongside mesh connectivity
- **Rotation-invariant positional features** — RWPE, LPE, or both
- **Multi-GPU DDP** training via PyTorch NCCL; auto-detected from `gpu_ids`
- **bfloat16 AMP**, **EMA**, **activation checkpointing**, **fused Adam**, **gradient accumulation**
- Single entry point (`MeshGraphNets_main.py`) for training and inference via plain-text config files

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and torch-geometric must match your CUDA version. Example for CUDA 12.1:

```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-cluster torch-scatter \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

If training from shared storage, set `HDF5_USE_FILE_LOCKING=FALSE`.

---

## Quick Start

### 1. Build dataset

```bash
python build_dataset.py
```

Input nodal data shape: `[features, time, nodes]` — features are `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`.

### 2. Write a config

Create a `.txt` config in `_warpage_input/`. Minimal example:

```
mode                train
gpu_ids             0
dataset_dir         ./dataset/train.h5
modelpath           ./outputs/model.pth
input_var           3
output_var          3
edge_var            8
latent_dim          128
message_passing_num 15
training_epochs     5000
batch_size          4
learningr           0.0001
use_amp             true
use_ema             true
```

### 3. Train

```bash
# Single GPU
python MeshGraphNets_main.py --config _warpage_input/config_train.txt

# Multi-GPU DDP (set gpu_ids 0,1,2,3 in config — DDP is auto-detected)
python MeshGraphNets_main.py --config _warpage_input/config_train.txt
```

### 4. Inference / rollout

```bash
python MeshGraphNets_main.py --config _warpage_input/config_infer.txt
```

Rollout outputs are saved as `rollout_sample{id}_steps{N}.h5`.

### 5. Animate

```bash
python animate_h5.py
```

---

## Architecture

```
Input Graph
  nodes: [state | positional_features | node_type_onehot]
  edges: [deformed_dx/dy/dz/dist | ref_dx/dy/dz/dist]  (8D, bidirectional)
        │
   ┌────▼──────┐
   │  Encoder  │   node MLP + edge MLP → latent embeddings (dim = latent_dim)
   └────┬──────┘
        │
   ┌────▼──────────────────────────────────────────────────┐
   │  Processor                                             │
   │                                                        │
   │  FLAT: N × GnBlock                                    │
   │                                                        │
   │  MULTISCALE V-cycle:                                  │
   │    pre-blocks (fine) → coarsen → coarse-level blocks  │
   │    → unpool → post-blocks (fine)                      │
   │    skip connections via Linear(2D, D)                 │
   │    world edges at finest level only                   │
   │                                                        │
   │  Each GnBlock:                                        │
   │    EdgeBlock: [src ‖ dst ‖ edge] → MLP + residual    │
   │    NodeBlock: [node ‖ Σ edge_msgs] → MLP + residual  │
   └────┬──────────────────────────────────────────────────┘
        │
   ┌────▼──────┐
   │  Decoder  │   MLP → predicted Δstate (normalized, no output LayerNorm)
   └────┬──────┘
        │
   state_{t+1} = state_t + denormalize(Δstate)
```

### VAE Conditioning (`use_vae true`)

- `GNNVariationalEncoder` runs GnBlocks on the **target** delta graph → GlobalAttention pool → `(μ, log σ²)` → reparameterized `z`
- `z` is fused into every GnBlock during training
- At inference, `z` is sampled from a **RealNVP** flow trained post-training on the aggregate posterior
- KL annealing: `linear` or `three_phase` schedule

### Key design choices

| Choice | Value |
|--------|-------|
| Activation | SiLU (Swish) — hardcoded |
| Aggregation | Sum (not mean) in NodeBlock |
| Normalization | Z-score from training split only |
| AMP dtype | bfloat16 (**not** float16 — float16 overflows `scatter_add`) |
| Node types | One-hot concatenated **after** normalization |
| Decoder init | Last-layer weights ×0.01 when T>1 (predict-no-change prior) |

---

## Config Reference

Config files are plain text: one `key    value` pair per line. `%` = full-line comment, `#` = inline comment.

### Mode / IO

| Key | Description |
|-----|-------------|
| `mode` | `train` or `inference` |
| `gpu_ids` | GPU IDs; comma-separated for DDP; `-1` for CPU |
| `modelpath` | Checkpoint save/load path |
| `log_file_dir` | Log output path |
| `dataset_dir` | Training HDF5 path |
| `infer_dataset` | Inference HDF5 path |
| `infer_timesteps` | Autoregressive rollout steps |

### Base Model

| Key | Description |
|-----|-------------|
| `input_var` | Input node feature count |
| `output_var` | Output feature count (predicted delta) |
| `edge_var` | Edge feature dim — must be `8` |
| `latent_dim` | Hidden dimension throughout |
| `message_passing_num` | GnBlocks in flat mode |
| `feature_loss_weights` | Per-feature Huber loss weights (auto-normalized) |
| `positional_features` | Rotation-invariant node features to append |
| `positional_encoding` | `rwpe`, `lpe`, or `rwpe+lpe` |
| `use_node_types` | Append one-hot node type features |

### Training

| Key | Description |
|-----|-------------|
| `training_epochs` | Total epochs |
| `batch_size` | Batch size per GPU |
| `learningr` | Peak learning rate |
| `warmup_epochs` | Linear LR warmup epochs (default 3) |
| `split_seed` | Train/val/test split seed (default 42) |
| `std_noise` | Gaussian noise std on nodes+edges |
| `augment_geometry` | Z-rotation + X/Y reflection augmentation |
| `grad_accum_steps` | Gradient accumulation steps |
| `use_amp` | bfloat16 mixed precision |
| `use_ema` | EMA shadow model |
| `ema_decay` | EMA decay (default 0.99) |
| `use_checkpointing` | Activation checkpointing (saves GPU memory) |
| `test_interval` | Epoch interval for validation |

### Multiscale

| Key | Description |
|-----|-------------|
| `use_multiscale` | Enable V-cycle |
| `multiscale_levels` | Levels L — `mp_per_level` needs 2L+1 entries |
| `coarsening_type` | `bfs` or `voronoi` |
| `mp_per_level` | MP blocks per level: `[pre, coarse_1…coarse_L, post]` |
| `voronoi_clusters` | Target cluster counts per Voronoi level |
| `bipartite_unpool` | Learned bipartite unpool (vs. broadcast) |

### VAE / Normalizing Flow

| Key | Description |
|-----|-------------|
| `use_vae` | Conditional VAE latent conditioning |
| `vae_latent_dim` | VAE latent code dimension |
| `vae_mp_layers` | GnBlocks in VAE encoder |
| `beta_kl` | KL divergence weight |
| `alpha_recon` | Reconstruction loss weight |
| `beta_aux` | Auxiliary prediction loss weight |
| `kl_anneal_schedule` | `linear` or `three_phase` |
| `kl_phase1_ratio` | Fraction of training in constant-low-β phase |
| `kl_phase2_ratio` | Fraction of training in β ramp phase |
| `kl_min_beta_ratio` | Phase 1 β = `beta_kl × kl_min_beta_ratio` |
| `train_latent_flow` | Train RealNVP flow post-training |
| `flow_hidden_dim` | Flow MLP hidden dim |
| `flow_num_layers` | RealNVP coupling layers |
| `flow_lr` | Flow learning rate |
| `flow_weight_decay` | Flow L2 regularization |

### World Edges

| Key | Description |
|-----|-------------|
| `use_world_edges` | Radius-based collision/contact edges |
| `world_radius_multiplier` | Radius = mean_edge_len × this |
| `world_max_num_neighbors` | Max degree per node |
| `world_edge_backend` | `torch_cluster` (fast) or `scipy_kdtree` (fallback) |

---

## HDF5 Data Format

**Input dataset** — `data/{id}/nodal_data` shape `[F, T, N]`:

| Index | Feature |
|-------|---------|
| 0–2 | x, y, z (reference coordinates) |
| 3–5 | x_disp, y_disp, z_disp |
| 6 | stress (von Mises or equivalent) |
| 7 | part_number (optional) |

**Mesh connectivity** — `data/{id}/mesh_edge` shape `[2, E]` (bidirectional node index pairs)

**Preprocessing** writes normalization stats and train/val/test splits back into `metadata/`.

**Rollout output** — `rollout_sample{id}_steps{N}.h5`, same layout, shape `[8, T, N]`

**Checkpoints** contain: model weights, optimizer/scheduler state, `checkpoint['normalization']`, optionally `ema_state_dict`, `coarse_edge_means`/`coarse_edge_stds`, `flow_state_dict`, `model_config`.

---

## Project Structure

```
MeshGraphNets_main.py              Entry point — routes train/inference
model/
  MeshGraphNets.py                 Top-level Encode-Process-Decode model
  encoder_decoder.py               Encoder, GnBlock, Decoder
  blocks.py                        EdgeBlock, NodeBlock, HybridNodeBlock, UnpoolBlock
  mlp.py                           build_mlp and weight initialization
  vae.py                           GNNVariationalEncoder
  coarsening.py                    BFS Bi-Stride + Voronoi FPS coarsening
  latent_flow.py                   RealNVP normalizing flow
  checkpointing.py                 Gradient checkpointing wrapper
training_profiles/
  setup.py                         Dataset / model / optimizer builders
  training_loop.py                 Epoch loop, loss, EMA, VAE losses
  single_training.py               Single-GPU entry; saves normalization
  distributed_training.py          DDP multi-GPU entry with NCCL
inference_profiles/
  rollout.py                       Autoregressive rollout → HDF5
general_modules/
  mesh_dataset.py                  MeshGraphDataset (normalization, augmentation, positional encoding)
  edge_features.py                 8D edge feature computation
  world_edges.py                   Collision/contact edge builder
  multiscale_helpers.py            Multiscale hierarchy builder
  mesh_utils_fast.py               GPU mesh utilities + PyVista rendering
  load_config.py                   Plain-text config parser
build_dataset.py                   HDF5 dataset builder from FEA exports
dataset/
  generate_inference_dataset.py    Extract initial conditions for inference
  reduce_dataset.py                Dataset subsampling utility
animate_h5.py                      GIF animation from rollout HDF5
misc/
  plot_loss.py                     Loss curve plotting
  plot_loss_realtime.py            FastAPI real-time loss monitor
  analyze_mesh_topology.py         Mesh topology analysis
  debug_model_output.py            Model output debugging
_warpage_input/                    Example config files
docs/                              Architecture and feature documentation
```

---

## Additional Docs

- [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md) — multiscale coarsening details
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md) — world-edge construction
- [VAE_IMPLEMENTATION_GUIDE.md](VAE_IMPLEMENTATION_GUIDE.md) — variational architecture notes
- [config_run_docs.md](config_run_docs.md) — extended config key reference
