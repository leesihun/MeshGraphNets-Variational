# MeshGraphNets — Variational

Graph Neural Network surrogate for FEA mesh simulation. Predicts time-transient deformation and stress fields on arbitrary mesh topologies using an Encode–Process–Decode GNN with optional multiscale V-cycle, conditional MMD-VAE, and world (collision) edges.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Data Format](#data-format)
6. [Training](#training)
7. [Inference & Rollout](#inference--rollout)
8. [Config File Reference](#config-file-reference)
9. [Helper Scripts](#helper-scripts)
10. [Outputs](#outputs)

---

## Overview

**Task:** Given the current mesh state at time *t* (node positions + physical fields), predict the state at time *t+1* by learning normalized state deltas `Δstate = state_{t+1} − state_t`. Repeated autoregressive application enables long-horizon rollout.

**Key capabilities:**

| Feature | Description |
|---------|-------------|
| Flat GNN | Stack of message-passing GnBlocks (NVIDIA/DeepMind MeshGraphNets) |
| Multiscale V-cycle | BFS Bi-Stride or FPS-Voronoi coarsening with learned unpooling |
| Conditional VAE | MMD-InfoVAE latent conditioning for stochastic output distributions |
| World edges | Radius-based collision/contact edges beyond mesh topology |
| EMA | Exponential Moving Average shadow model for stable inference |
| AMP | bfloat16 mixed precision (Ampere+ GPUs) |
| DDP | Multi-GPU distributed training via PyTorch DistributedDataParallel |
| GMM prior | Post-hoc GMM fitted to VAE posterior means; used at inference for realistic sampling |

---

## Architecture

### Encode–Process–Decode

```
Input mesh graph
  ├─ Node features: [x_disp, y_disp, z_disp, (stress), pos_features, node_type_onehot]
  └─ Edge features: [deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]  (8-D)

      │
      ▼
  ┌─────────┐
  │ Encoder │  MLP per node, MLP per edge → latent embeddings (dim = latent_dim)
  └─────────┘
      │
      ▼
  ┌───────────────────────────────────────────┐
  │ Processor                                 │
  │  Flat:  GnBlock × message_passing_num     │
  │  ─ OR ─                                   │
  │  Multiscale V-cycle:                      │
  │    Pre-descent: GnBlocks → Pool           │
  │    Coarsest level: GnBlocks               │
  │    Post-ascent:  Unpool → skip → GnBlocks │
  └───────────────────────────────────────────┘
      │
      ▼
  ┌─────────┐
  │ Decoder │  MLP → predicted Δstate (normalized)
  └─────────┘
      │
      ▼
  Denormalize → state_{t+1} = state_t + Δstate
```

### GnBlock (one message-passing iteration)

```
EdgeBlock:  MLP([sender ‖ receiver ‖ edge]) → updated edge features
NodeBlock:  scatter-sum(updated edges) → MLP([node ‖ aggregated]) → updated node
Residual:   h_out = h_in + residual_scale × MLP_out
```

All MLPs: `Linear → SiLU → Linear → SiLU → Linear → LayerNorm` (no LayerNorm on Decoder output).

### Conditional VAE (MMD-InfoVAE)

When `use_vae True`, a graph encoder processes the **target** `y` during training:

```
GNNVariationalEncoder(y) → (μ, log σ²) → z = μ + σ·ε
z is fused into every processor GnBlock via a Linear projection.

Loss = α·Huber(ŷ, y) + λ·MMD(z, N(0,I)) + β·MSE(aux(z), [y_mean, y_std])
```

MMD uses multi-scale RBF kernels (σ ∈ {0.5, 1, 2, 4, 8}) — the InfoVAE/MMD-VAE objective.  
At inference, `z ~ GMM` (if `fit_latent_gmm True`) or `z ~ N(0, I)`.

### Multiscale Coarsening

**BFS Bi-Stride** (default): Even-depth BFS nodes form the coarse graph; ~4× reduction per level.  
**FPS-Voronoi**: Farthest Point Sampling seeds + Voronoi assignment; configurable cluster count.

Upsampling: broadcast (simple gather) or learned bipartite message passing (`bipartite_unpool True`).

---

## Quick Start

```bash
# Train (single GPU)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Train (multi-GPU DDP — gpu_ids 0,1 in config)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Inference / autoregressive rollout
python MeshGraphNets_main.py --config _warpage_input/config_infer3.txt
```

The `--config` flag defaults to `config.txt` in the working directory. Single vs. multi-GPU is auto-detected from `gpu_ids`. Mode (`train`/`inference`) is set inside the config file.

---

## Installation

```bash
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-cluster
pip install -r requirements.txt
```

**requirements.txt:**

```
torch>=2.1.0
torch-geometric>=2.4.0
torch-cluster>=1.6.3        # optional: GPU radius_graph for world edges
torch-scatter>=2.1.2
numpy>=1.24.0
scipy>=1.10.0
h5py>=3.8.0
pandas>=1.5.0
tqdm>=4.65.0
matplotlib>=3.7.0
Pillow>=9.5.0
pyvista>=0.43.0
```

Optional (real-time loss server):
```bash
pip install fastapi uvicorn
```

---

## Data Format

### Training HDF5

```
data/
└── {sample_id}/
    ├── nodal_data   [num_features, num_timesteps, num_nodes]
    │   └── layout: [x, y, z, x_disp, y_disp, z_disp, stress, part_number]
    ├── mesh_edge    [2, num_edges]   (unidirectional; code makes bidirectional)
    └── metadata/   (num_nodes, num_edges, num_timesteps, feature_min/max/mean/std)
metadata/
└── feature_names   [num_features]
```

`input_var` counts the physical state features used as model input (e.g., 3 for x/y/z displacement; 4 to add stress). `x, y, z` reference positions are always the first 3 features in the HDF5 but are **not** counted in `input_var` — they are extracted separately as `ref_pos`.

### Rollout Output HDF5

```
data/{sample_id}/
├── nodal_data   [3 + output_var + 1, timesteps, nodes]
│   └── layout: [x, y, z, <output_var channels>, part_number]
└── mesh_edge    [2, E]
metadata/
└── normalization_params/ (node_mean, node_std, delta_mean, delta_std, ...)
```

### Dataset Builder

```bash
python build_dataset.py          # Build HDF5 from raw ANSYS .inp + .csv export
python dataset/reduce_dataset.py # Subsample an existing HDF5
python dataset/generate_inference_dataset.py  # Extract initial conditions for rollout
```

---

## Training

### Single GPU

Set `gpu_ids 0` in config:

```
mode        train
gpu_ids     0
dataset_dir ./dataset/b8_train.h5
modelpath   ./outputs/warpage.pth
```

### Multi-GPU DDP

Set `gpu_ids 0,1` (or more) in config. `mp.spawn` is used automatically — no `torchrun` needed:

```
gpu_ids     0,1
batch_size  16    # per GPU
```

### Training Loop

1. Build 80/10/10 train/val/test splits (deterministic by `split_seed`).
2. Compute and store Z-score normalization stats on training split.
3. For each epoch: forward → Huber loss (+ MMD + aux if VAE) → backward → grad clip (max_norm=3) → optimizer step → EMA update.
4. Validate every `val_interval` epochs; save checkpoint on best validation loss.
5. Test/visualize every `test_interval` epochs.
6. After training: optionally fit GMM on posterior means (`fit_latent_gmm True`).

### Optimizer & Schedule

```
Adam (fused on CUDA)
  └── LinearLR (warmup_epochs) → CosineAnnealingWarmRestarts
```

### Gradient Accumulation

`grad_accum_steps 1` = per-batch update. `grad_accum_steps 0` = full-epoch accumulation (gradient averaged over all batches).

---

## Inference & Rollout

Set `mode inference` in config. Rollout is **fully autoregressive**:

```
state_0 → model → Δstate_0 → state_1 → model → Δstate_1 → ... → state_T
```

Each step:
1. Normalize current state.
2. Compute 8-D deformed + reference edge features.
3. (Optional) compute world edges.
4. Attach multiscale coarsening data (topology is static and pre-built once).
5. Forward pass with fixed `z` sampled from GMM or N(0, I).
6. Denormalize predicted delta; update state.

Results saved to `inference_output_dir` as `rollout_sample{id}_steps{N}.h5`.

For stochastic rollout set `num_vae_samples N` — runs N independent latent codes and saves all trajectories.

---

## Config File Reference

See [config_run_docs.md](config_run_docs.md) for the complete, detailed reference.

**Format:** one `key  value  # comment` per line. `%` for full-line comments. Lists: comma-separated values. Booleans: `true`/`false`. Keys are case-insensitive.

**Minimal training config:**

```
model       MeshGraphNets-V
mode        train
gpu_ids     0
modelpath   ./outputs/model.pth
dataset_dir ./dataset/data.h5
input_var   3
output_var  3
message_passing_num 10
latent_dim  128
training_epochs 1000
batch_size  16
learningr   0.0001
num_workers 4
```

**Minimal inference config:**

```
mode            inference
gpu_ids         0
modelpath       ./outputs/model.pth
infer_dataset   ./dataset/infer_data.h5
infer_timesteps 10
```

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `animate_h5.py` | Generate animated GIFs from rollout HDF5 output |
| `build_dataset.py` | Build HDF5 training dataset from raw FEA export |
| `dataset/generate_inference_dataset.py` | Extract initial conditions for inference |
| `dataset/reduce_dataset.py` | Subsample / split existing HDF5 dataset |
| `misc/plot_loss.py` | Plot training loss from log file |
| `misc/plot_loss_realtime.py` | Real-time loss monitoring via FastAPI server |

---

## Outputs

| Path | Description |
|------|-------------|
| `outputs/*.pth` | Model checkpoints (weights + normalization + GMM params) |
| `outputs/rollout/rollout_sample*_steps*.h5` | Autoregressive rollout results |
| `outputs/test/` | Per-epoch test visualizations (.png / .h5) |
| `*.log` | Training log (loss per epoch, validation, test) |
