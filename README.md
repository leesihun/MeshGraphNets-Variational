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

### 1. Train

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt
```

Trains on `./dataset/b8_train.h5` using DDP on GPUs 0 and 1. Saves checkpoint to `./outputs/warpage1.pth`. After training, fits a GMM on VAE posterior means and embeds it in the checkpoint.

### 2. Infer (rollout)

```bash
# Against warpage1.pth  (matches config_train5)
python MeshGraphNets_main.py --config _warpage_input/config_infer4.txt

# Against warpage0.pth  (older checkpoint)
python MeshGraphNets_main.py --config _warpage_input/config_infer3.txt
```

Both configs run 1000 stochastic rollouts (`num_vae_samples 1000`) starting from `./dataset/infer_dataset_b8_test.h5`. Results are written to `outputs/rollout/` as `rollout_sample{id}_steps{N}.h5`.

The `--config` flag defaults to `config.txt`. Mode (`train`/`inference`) is set **inside** the config. Single vs. multi-GPU is auto-detected from `gpu_ids`. `gpu_ids -1` forces CPU.

---

### Current config_train5.txt — [`_warpage_input/config_train5.txt`](_warpage_input/config_train5.txt)

```
model               MeshGraphNets-V
mode                train
gpu_ids             0,1             # DDP on both GPUs
log_file_dir        train5.log
modelpath           ./outputs/warpage1.pth

% Datasets
dataset_dir         ./dataset/b8_train.h5
infer_dataset       ./dataset/infer_dataset_b8_test.h5
infer_timesteps     1
num_vae_samples     177

% Features
input_var           3               # x_disp, y_disp, z_disp
output_var          3
edge_var            8               # deformed+ref dx/dy/dz/dist
positional_features 4
feature_loss_weights 1, 1, 1.0

% Network
message_passing_num 15
latent_dim          128
training_epochs     2000
batch_size          16              # per GPU
learningr           0.0001
std_noise           0.01
residual_scale      1
num_workers         8
augment_geometry    True
grad_accum_steps    1

% Memory / Speed
use_checkpointing   True
use_amp             True            # bfloat16
use_ema             True
ema_decay           0.99
test_interval       100
val_interval        1

% Multiscale (Voronoi V-cycle)
use_multiscale      True
coarsening_type     voronoi
voronoi_clusters    200
multiscale_levels   1
mp_per_level        4, 12, 4        # [pre, coarsest, post]
bipartite_unpool    True

% VAE (MMD-InfoVAE)
use_vae             True
vae_latent_dim      4
alpha_recon         1
lambda_mmd          0.1

% Post-hoc GMM prior
fit_latent_gmm      True
gmm_components      10
gmm_covariance_type full
```

---

### Current config_infer4.txt — [`_warpage_input/config_infer4.txt`](_warpage_input/config_infer4.txt)

```
model               MeshGraphNets-V
mode                inference
gpu_ids             1               # single GPU
log_file_dir        train4.log
modelpath           ./outputs/warpage1.pth

% Datasets
dataset_dir         ./dataset/b8_train.h5
infer_dataset       ./dataset/infer_dataset_b8_test.h5
infer_timesteps     1
num_vae_samples     1000

% Features  (must match training checkpoint)
input_var           3
output_var          3
edge_var            8
positional_features 4
feature_loss_weights 1, 1, 1.0

% Network  (must match training checkpoint)
message_passing_num 15
latent_dim          128
use_checkpointing   True
use_amp             True
use_ema             True
ema_decay           0.99

% Multiscale
use_multiscale      True
coarsening_type     voronoi
voronoi_clusters    200
multiscale_levels   1
mp_per_level        4, 12, 4
bipartite_unpool    True

% VAE
use_vae             True
vae_latent_dim      256             # expanded latent for sampling diversity
alpha_recon         1
lambda_mmd          0.1

% GMM prior
fit_latent_gmm      True
gmm_components      10
gmm_covariance_type full
```

> **Note:** `vae_latent_dim` in an inference config does NOT need to match training — the checkpoint carries the trained VAE weights. The inference `vae_latent_dim` controls how many GMM components are sampled from when `fit_latent_gmm True` is set and the GMM is present in the checkpoint.

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

### Overview

All data is stored as HDF5 (`.h5`). There are three HDF5 types in this project:

| File | Purpose | Created by |
|------|---------|------------|
| Training dataset (`b8_train.h5`) | Full multi-step trajectories for training and test | `build_dataset.py` |
| Inference dataset (`infer_dataset_b8_test.h5`) | Initial conditions only (t=0) for rollout | `dataset/generate_inference_dataset.py` |
| Rollout output (`rollout_sample*_steps*.h5`) | Autoregressive predictions produced by the model | `inference_profiles/rollout.py` |

---

### Training HDF5 (`b8_train.h5`)

```
/
├── data/
│   ├── {sample_id}/                    # integer key, e.g. "0", "1", "42"
│   │   ├── nodal_data   float32  [num_features, num_timesteps, num_nodes]
│   │   ├── mesh_edge    int64    [2, num_edges]          (unidirectional; code mirrors to bidirectional)
│   │   └── metadata/
│   │       └── attrs:  num_nodes, num_edges, num_timesteps,
│   │                   feature_min, feature_max, feature_mean, feature_std
│   ├── {sample_id}/
│   │   └── ...
│   └── ...
└── metadata/
    └── feature_names   bytes[num_features]
```

#### nodal_data feature layout

Index | Name | Unit | Notes
------|------|------|------
0 | `x_coord` | mm | Reference (undeformed) X position
1 | `y_coord` | mm | Reference Y position
2 | `z_coord` | mm | Reference Z position
3 | `x_disp`  | mm | X displacement at each timestep
4 | `y_disp`  | mm | Y displacement
5 | `z_disp`  | mm | Z displacement
6 | `stress`  | MPa | von Mises equivalent stress
7 | `part_no` | — | Integer part/component ID

- Indices 0–2 (`x/y/z_coord`) are **reference positions** — static across timesteps, never counted in `input_var`.
- `input_var 3` → model reads features 3–5 (x/y/z displacement).
- `input_var 4` → model reads features 3–6 (displacement + stress).
- `part_no` (index 7) is used only when `use_node_types True`; otherwise ignored.

#### mesh_edge

Shape `[2, num_edges]` — unidirectional connectivity (row 0 = source, row 1 = target). The dataloader mirrors each edge to produce bidirectional pairs.

#### Normalization and split metadata written back to HDF5

After the first training run, the following are written back into the HDF5 by the dataset class:

```
/metadata/
    train_indices        int[]     sample IDs in training split
    val_indices          int[]     sample IDs in validation split
    test_indices         int[]     sample IDs in test split
    node_mean            float32[input_var]
    node_std             float32[input_var]
    edge_mean            float32[8]
    edge_std             float32[8]
    delta_mean           float32[output_var]
    delta_std            float32[output_var]
    coarse_edge_means    float32[levels, 8]
    coarse_edge_stds     float32[levels, 8]
```

---

### Inference Input HDF5 (`infer_dataset_b8_test.h5`)

Same schema as the training HDF5, but each sample has **only one timestep** (`num_timesteps = 1`) — the initial condition.

```
/
├── data/
│   └── {sample_id}/
│       ├── nodal_data   float32  [num_features, 1, num_nodes]   ← t=0 only
│       ├── mesh_edge    int64    [2, num_edges]
│       └── metadata/  (attrs)
└── metadata/
    └── feature_names   bytes[num_features]
```

Generated by:
```bash
python dataset/generate_inference_dataset.py
```

This script randomly selects `num_samples` (default 10) from the training HDF5 and copies only `nodal_data[:, 0:1, :]` (time index 0) plus `mesh_edge`.

---

### Rollout Output HDF5 (`rollout_sample{id}_steps{N}.h5`)

Produced by `inference_profiles/rollout.py`. One file per input sample per VAE sample draw.

```
/
├── data/
│   └── {sample_id}/
│       ├── nodal_data   float32  [3 + output_var + 1, num_timesteps, num_nodes]
│       │   └── layout: [x_coord, y_coord, z_coord, <output_var channels>, part_no]
│       └── mesh_edge    int64    [2, num_edges]
└── metadata/
    └── normalization_params/
        ├── node_mean, node_std
        ├── delta_mean, delta_std
        └── ...
```

For `output_var 3` the nodal_data channels are: `[x, y, z, x_disp_pred, y_disp_pred, z_disp_pred, part_no]` → shape `[7, T, N]`.

Visualize with:
```bash
python animate_h5.py
```

---

### Building a Dataset from Scratch

#### Step 1 — Build training HDF5 from ANSYS export

```bash
python build_dataset.py
```

Expects the following raw input structure (configure paths at top of `build_dataset.py`):

```
{device_code}/picked_inp/
├── nonfem/
│   └── sub{1..10}/
│       └── id{N}_*_mesh.inp          # ANSYS mesh file
└── Ansys_inp_path/
    └── sub{1..10}/
        └── Sim_data/
            └── {N}/
                ├── node_coordinates.csv
                ├── z_disp_step_1.csv   # one CSV per timestep
                ├── z_disp_step_2.csv
                └── ...
```

Each CSV has columns: node_id, x_disp, y_disp, z_disp, and one of `[SEQV, S_EQV, Stress, stress, VM_Stress, von_mises]` for von Mises stress.

#### Step 2 — Generate inference initial conditions

```bash
python dataset/generate_inference_dataset.py
```

Editable parameters at the bottom of the script: `source_dataset`, `output_path`, `num_samples`, `random_seed`.

#### Step 3 (optional) — Reduce dataset size

```bash
python dataset/reduce_dataset.py
```

Subsamples an existing HDF5 to a smaller set of samples.

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

1. Build 80/10/10 train/val/test splits (deterministic by `split_seed`, default 42).
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
model               MeshGraphNets-V
mode                train
gpu_ids             0
modelpath           ./outputs/model.pth
dataset_dir         ./dataset/data.h5
input_var           3
output_var          3
message_passing_num 15
latent_dim          128
training_epochs     2000
batch_size          16
learningr           0.0001
num_workers         8
```

**Minimal inference config:**

```
model           MeshGraphNets-V
mode            inference
gpu_ids         0
modelpath       ./outputs/model.pth
dataset_dir     ./dataset/data.h5
infer_dataset   ./dataset/infer_data.h5
infer_timesteps 1
num_vae_samples 1000
input_var       3
output_var      3
```

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `animate_h5.py` | Generate animated GIFs from rollout HDF5 output |
| `build_dataset.py` | Build HDF5 training dataset from raw ANSYS FEA export |
| `dataset/generate_inference_dataset.py` | Extract t=0 initial conditions for inference |
| `dataset/reduce_dataset.py` | Subsample / split an existing HDF5 dataset |
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
