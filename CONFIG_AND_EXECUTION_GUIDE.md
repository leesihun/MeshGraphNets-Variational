# Configuration and Execution Guide

Complete reference for configuring and running MeshGraphNets. Covers config file format, all parameters, execution modes, dataset structure, and real config file examples from the repository.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Layout: Config Folders](#project-layout-config-folders)
- [Config File Format](#config-file-format)
- [Parameter Reference](#parameter-reference)
  - [General Parameters](#general-parameters)
  - [Dataset Parameters](#dataset-parameters)
  - [Network Architecture Parameters](#network-architecture-parameters)
  - [Training Hyperparameters](#training-hyperparameters)
  - [Node Type Parameters](#node-type-parameters)
  - [World Edge Parameters](#world-edge-parameters)
  - [Memory Optimization Parameters](#memory-optimization-parameters)
  - [Performance Optimization Parameters](#performance-optimization-parameters)
  - [Diagnostics and Visualization Parameters](#diagnostics-and-visualization-parameters)
  - [Inference Parameters](#inference-parameters)
  - [Internally-Assigned Config Keys](#internally-assigned-config-keys)
- [Required Keys by Mode](#required-keys-by-mode)
- [Execution Modes](#execution-modes)
  - [Training (Single GPU)](#training-single-gpu)
  - [Training (Multi-GPU DDP)](#training-multi-gpu-ddp)
  - [Training (CPU)](#training-cpu)
  - [Inference (Autoregressive Rollout)](#inference-autoregressive-rollout)
- [Existing Config Files](#existing-config-files)
  - [Warpage Configs (_warpage_input/)](#warpage-configs-_warpage_input)
  - [Flag Simple Configs (_flag_input/)](#flag-simple-configs-_flag_input)
  - [Flag Dynamic Configs (_flag_input/)](#flag-dynamic-configs-_flag_input)
- [Writing New Config Files](#writing-new-config-files)
  - [Naming Convention](#naming-convention)
  - [Train Config Template](#train-config-template)
  - [Inference Config Template](#inference-config-template)
  - [Hyperparameter Sweep Setup](#hyperparameter-sweep-setup)
- [HDF5 Dataset Structure](#hdf5-dataset-structure)
- [Data Flow: How the Model Uses Data](#data-flow-how-the-model-uses-data)
- [Checkpoint Contents](#checkpoint-contents)
- [Output Directory Structure](#output-directory-structure)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Every run uses the same entry point with a `--config` flag pointing to a config file:

```bash
# Training
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt

# Inference (using the model trained above)
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

Config files live in underscore-prefixed folders (e.g., `_warpage_input/`, `_flag_input/`) organized by dataset or experiment. Each folder typically contains paired train and inference configs.

---

## Project Layout: Config Folders

Config files are organized into folders prefixed with `_` to keep them visually separated from code:

```
MeshGraphNets/
├── _warpage_input/                   # Warpage simulation configs
│   ├── config_train1.txt             # Train: GPU 0, Latent_dim=128, MP=15, BS=1
│   ├── config_train2.txt             # Train: GPU 1, Latent_dim=128, MP=5,  BS=1
│   ├── config_train3.txt             # Train: GPU 1, Latent_dim=128, MP=15, BS=10
│   ├── config_train4.txt             # Train: GPU 0, Latent_dim=128, MP=5,  BS=10
│   ├── config_infer1.txt             # Infer: warpage1.pth, MP=15
│   └── config_infer2.txt             # Infer: warpage1.pth, MP=5
│
├── _flag_input/                      # Flag simulation configs
│   ├── config_flag_simple1.txt       # Train: GPU 0, Latent_dim=256, node_types+world_edges
│   ├── config_flag_simple2.txt       # Train: GPU 1, Latent_dim=128, node_types+world_edges
│   ├── config_flag_simple_infer1.txt # Infer: flag_simple1.pth
│   ├── config_flag_simple_infer2.txt # Infer: flag_simple2.pth
│   ├── config_flag_dynamic1.txt      # Train: GPU 0, Latent_dim=256, dynamic
│   └── config_flag_dynamic2.txt      # Train: GPU 1, Latent_dim=128, dynamic
│
├── MeshGraphNets_main.py             # Entry point
└── ...
```

**Convention**: The folder name describes the dataset/experiment. Files inside use `config_train{N}.txt` for training and `config_infer{N}.txt` for inference. The number `{N}` typically corresponds to a specific hyperparameter variant or GPU assignment.

**Execution pattern**: Always run from the repository root:

```bash
python MeshGraphNets_main.py --config _<folder>/config_<type><N>.txt
```

---

## Config File Format

Configuration is stored in plain-text files. The path is specified via `--config` (default: `config.txt`):

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt
```

### Syntax Rules

| Rule | Description | Example |
|------|-------------|---------|
| **Key-value pairs** | Space or tab separated | `LearningR 0.0001` |
| **Comments** | Lines starting with `%` | `% This is a comment` |
| **Inline comments** | Text after `#` is stripped | `Batch_size 50  # per-GPU` |
| **Section separators** | Lines starting with `'` (ignored) | `'` |
| **Case-insensitive keys** | All keys lowercased internally | `LearningR` becomes `learningr` |
| **List values** | Comma-separated | `gpu_ids 0, 1, 2, 3` |
| **Boolean values** | `True` or `False` (case-insensitive) | `verbose False` |
| **Reserved keyword** | Lines with key `reserved` are skipped | `reserved ...` |

### Type Parsing Order

Values are auto-parsed in this order by `general_modules/load_config.py`:

1. **Comma-separated** -> list of int/float or strings
2. **Space-separated** (multi-token) -> list of int/float or strings
3. **Boolean** -> `True`/`False`
4. **Numeric** -> int (no decimal) or float (with decimal)
5. **String** -> lowercase string (fallback)

**Examples:**

| Config line | Stored as | Type |
|-------------|-----------|------|
| `LearningR 0.0001` | `config['learningr'] = 0.0001` | float |
| `gpu_ids 0, 1, 2, 3` | `config['gpu_ids'] = [0, 1, 2, 3]` | list[int] |
| `gpu_ids 0` | `config['gpu_ids'] = 0` | int |
| `verbose True` | `config['verbose'] = True` | bool |
| `mode Train` | `config['mode'] = 'train'` | str (lowercased) |
| `test_batch_idx 0, 1, 2, 3` | `config['test_batch_idx'] = [0, 1, 2, 3]` | list[int] |

---

## Parameter Reference

### General Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | str | Yes | Model name. Only `MeshGraphNets` is supported. |
| `mode` | str | Yes | `Train` or `Inference` (case-insensitive, stored lowercase). |
| `gpu_ids` | int/list | Yes | GPU device(s). `0` = single GPU 0, `0, 1, 2, 3` = multi-GPU DDP, `-1` = CPU. |
| `log_file_dir` | str | No | Log filename relative to `outputs/`. E.g., `train1.log` writes to `outputs/train1.log`. |
| `modelpath` | str | Yes | Path to save (training) or load (inference) the `.pth` checkpoint. Single-GPU training saves here. Multi-GPU DDP always saves to `outputs/best_model.pth` regardless of this value. |

**GPU routing logic in `MeshGraphNets_main.py`:**

| `gpu_ids` value | Route | Scheduler |
|-----------------|-------|-----------|
| Single int (e.g., `0`) | `single_training.py` | ExponentialLR(gamma=0.995) |
| `-1` | `single_training.py` on CPU | ExponentialLR(gamma=0.995) |
| Multiple (e.g., `0, 1, 2, 3`) | `distributed_training.py` (DDP) | ReduceLROnPlateau(factor=0.5, patience=2) |

---

### Dataset Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dataset_dir` | str | Train | Path to HDF5 training dataset. |
| `infer_dataset` | str | Inference | Path to HDF5 inference dataset. |

- Paths can be relative or absolute
- Dataset is auto-split 80/10/10 (train/val/test) with seed=42
- **All samples must have the same number of timesteps**

---

### Network Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_var` | int | - | Node input features (excluding node types). Typically `4`: [x_disp, y_disp, z_disp, stress]. |
| `output_var` | int | - | Node output features. Typically `4`: [x_disp, y_disp, z_disp, stress]. |
| `edge_var` | int | - | Edge features. Always `4`: [dx, dy, dz, distance]. |
| `Latent_dim` | int | 128 | Hidden dimension of all MLPs. VRAM scales **quadratically** with this. |
| `message_passing_num` | int | 15 | Number of GNN message passing blocks. More = larger receptive field, linear VRAM scaling. |

**Architecture notes:**
- Actual model input dim = `input_var + num_node_types` when `use_node_types=True`
- All MLPs: 2 hidden layers, ReLU, LayerNorm (except decoder has no LayerNorm)
- Residual connections on **nodes only** (not edges)
- Aggregation: **sum** (forces/stresses accumulate at nodes)
- Initialization: Kaiming/He uniform

---

### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Training_epochs` | int | - | Total training epochs. |
| `Batch_size` | int | - | Per-GPU batch size. For DDP: effective = `Batch_size * num_GPUs`. |
| `LearningR` | float | 0.0001 | Initial Adam learning rate. **Most sensitive hyperparameter.** |
| `num_workers` | int | 10 | DataLoader workers. Set 0 for debugging. |
| `std_noise` | float | 0.001 | Gaussian noise std added to inputs during training. 0 to disable. |
| `residual_scale` | float | 0.1 | Defined in config but **not currently used** in code (residual = scale 1.0). |
| `feature_loss_weights` | list[float] | None | Per-feature loss weights (comma-separated). Example: `1.0, 1.0, 5.0` emphasizes z_disp 5x. Auto-normalized to sum to `output_var`. |

**Training details:**
- **Optimizer**: Adam
- **Gradient clipping**: max_norm=5.0
- **Loss**: MSE on normalized deltas, optionally weighted per-feature
- **Weight init**: Kaiming/He uniform
- **Data split**: 80% train / 10% val / 10% test (seed=42)

---

### Node Type Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_node_types` | bool | False | Enable one-hot node type encoding from last feature in HDF5. |

When enabled:
1. Unique types auto-discovered from first 10 samples
2. Mapped to contiguous indices (e.g., `{0: 0, 1: 1, 3: 2}`)
3. One-hot vectors concatenated **after** z-score normalization
4. Mapping saved in checkpoint for inference

---

### World Edge Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_world_edges` | bool | False | Enable radius-based collision detection edges. |
| `world_radius_multiplier` | float | 1.5 | Collision radius = multiplier * min_mesh_edge_length. |
| `world_max_num_neighbors` | int | 64 | Max neighbors per node in radius query. |
| `world_edge_backend` | str | - | `torch_cluster` (GPU, fast) or `scipy_kdtree` (CPU, always available). **Must be in every config** -- rollout.py crashes without it. |

**Critical**: `world_edge_backend` must be explicitly set in **all** config files (both train and inference). The inference code (`rollout.py`) calls `.get('world_edge_backend').lower()` with no default and will crash with `AttributeError` if absent. All existing configs use `scipy_kdtree` for portability.

---

### Memory Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_checkpointing` | bool | False | Gradient checkpointing for message passing. Trades ~20-30% compute for ~60-70% VRAM savings. Training only. |

---

### Performance Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_parallel_stats` | bool | True | Parallel processing for normalization stat computation. Uses ~45% of CPU cores (max 64 workers). Auto-disabled for <10 samples. |

---

### Diagnostics and Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | False | Per-feature loss breakdowns, CUDA memory tracking, per-layer gradient stats. |
| `monitor_gradients` | bool | True | Gradient norm tracking. Warns about vanishing (<1e-6) or exploding (>100) gradients. |
| `display_testset` | bool | True | Generate PyVista visualizations during test eval (every 10 epochs). |
| `test_batch_idx` | list | [0] | Which test batch indices to visualize and save. All existing configs use `0, 1, 2, 3`. |
| `plot_feature_idx` | int | -1 | Feature index to plot. `-1` = stress, `-2` = z_disp. All existing configs use `-2`. |

---

### Inference Parameters

Only used when `mode=Inference`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelpath` | str | - | Path to trained checkpoint (`.pth`). Must contain `model_state_dict`, `normalization`, `model_config`. |
| `infer_dataset` | str | - | Path to HDF5 dataset with initial conditions. |
| `infer_timesteps` | int | - | Number of autoregressive rollout steps. If `None` and dataset has T>1, defaults to T-1. |
| `inference_output_dir` | str | `outputs/rollout` | Output directory for rollout HDF5 files. |

**Important**: Architecture parameters (`input_var`, `output_var`, `latent_dim`, etc.) are **overridden** by the checkpoint's `model_config` during inference. They must still be present in the config for parsing but their values don't matter.

---

### Internally-Assigned Config Keys

These are set by the code during execution. **Do not set them in config files.**

| Key | Assigned by | Purpose |
|-----|-------------|---------|
| `num_node_types` | `single_training.py`, `distributed_training.py`, `rollout.py` | Passes node type count to model constructor |
| `log_dir` | `single_training.py` | Directory for debug `.npz` files |

---

## Required Keys by Mode

### Train (single or multi-GPU)

```
model, mode, gpu_ids, modelpath, dataset_dir,
input_var, output_var, edge_var,
Latent_dim, message_passing_num,
Training_epochs, Batch_size, LearningR, num_workers,
world_edge_backend
```

### Inference

```
model, mode, gpu_ids, modelpath, infer_dataset,
infer_timesteps, world_edge_backend
```

`world_edge_backend` is required even if `use_world_edges=False`.

---

## Execution Modes

### Training (Single GPU)

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt
```

**Pipeline**: `MeshGraphNets_main.py` -> `single_training.py` -> `training_loop.py`

- Scheduler: ExponentialLR(gamma=0.995)
- Saves best model to `modelpath`
- Test evaluation every 10 epochs
- Logs per-feature loss weights (if configured) at startup

### Training (Multi-GPU DDP)

```bash
python MeshGraphNets_main.py --config _my_input/config_ddp.txt
```

Set `gpu_ids 0, 1, 2, 3` in the config to enable DDP.

**Pipeline**: `MeshGraphNets_main.py` -> `mp.spawn()` -> `distributed_training.py` -> `training_loop.py`

- Uses `torch.distributed` with NCCL backend (Gloo on CPU)
- `DistributedSampler` for data partitioning
- Effective batch size = `Batch_size * num_GPUs`
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-8)
- Only rank 0 saves checkpoints (to `outputs/best_model.pth`)
- Communication: `localhost:12355`

### Training (CPU)

Set `gpu_ids -1` in the config. Same pipeline as single GPU but on CPU.

### Inference (Autoregressive Rollout)

```bash
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

**Pipeline**: `MeshGraphNets_main.py` -> `rollout.py`

**Rollout loop:**
1. Load checkpoint (model weights + normalization stats + model_config)
2. Override config with `model_config` from checkpoint
3. For each sample in `infer_dataset`:
   a. Extract initial state at timestep 0
   b. For each step t -> t+1:
      - Normalize current state using checkpoint stats
      - Compute edge features from **deformed** positions (ref + displacement)
      - Build graph (including world edges if enabled)
      - Forward pass -> predicted normalized delta
      - Denormalize: `delta = delta_norm * delta_std + delta_mean`
      - Update state: `state_{t+1} = state_t + delta`
4. Save trajectory to `<inference_output_dir>/rollout_sample{id}_steps{N}.h5`

---

## Existing Config Files

### Warpage Configs (`_warpage_input/`)

Static (T=1) warpage simulation. Dataset: `./dataset/warpage_all.h5`.

All warpage configs share: `input_var=4`, `output_var=4`, `edge_var=4`, `LearningR=0.001`, `std_noise=0.02`, `use_node_types=False`, `use_world_edges=False`.

#### Training Configs

| File | GPU | MP | Batch | Latent | Modelpath | Run command |
|------|-----|-----|-------|--------|-----------|-------------|
| `config_train1.txt` | 0 | 15 | 1 | 128 | `./outputs/warpage1.pth` | `python MeshGraphNets_main.py --config _warpage_input/config_train1.txt` |
| `config_train2.txt` | 1 | 5 | 1 | 128 | `./outputs/warpage2.pth` | `python MeshGraphNets_main.py --config _warpage_input/config_train2.txt` |
| `config_train3.txt` | 1 | 15 | 10 | 128 | `./outputs/warpage3.pth` | `python MeshGraphNets_main.py --config _warpage_input/config_train3.txt` |
| `config_train4.txt` | 0 | 5 | 10 | 128 | `./outputs/warpage2.pth` | `python MeshGraphNets_main.py --config _warpage_input/config_train4.txt` |

These form a hyperparameter grid over **message_passing_num** (5 vs 15) and **Batch_size** (1 vs 10). Train1+Train4 run on GPU 0, Train2+Train3 run on GPU 1 -- they can run in parallel on a 2-GPU machine.

#### Inference Configs

| File | GPU | Modelpath | Infer Dataset | Steps | Run command |
|------|-----|-----------|---------------|-------|-------------|
| `config_infer1.txt` | 0 | `./outputs/warpage1.pth` | `./dataset/warpage_infer.h5` | 34 | `python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt` |
| `config_infer2.txt` | 0 | `./outputs/warpage1.pth` | `./dataset/warpage_infer.h5` | 34 | `python MeshGraphNets_main.py --config _warpage_input/config_infer2.txt` |

Both point to the same model (`warpage1.pth`). `config_infer2.txt` has `message_passing_num=5` (overridden by checkpoint anyway).

**Typical workflow:**

```bash
# 1. Train two variants in parallel (on separate GPUs)
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt   # terminal 1, GPU 0
python MeshGraphNets_main.py --config _warpage_input/config_train2.txt   # terminal 2, GPU 1

# 2. Run inference on the best model
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

---

### Flag Simple Configs (`_flag_input/`)

Static (T=1) flag with node types and world edges. Dataset: `./dataset/flag_simple.h5`.

All flag simple configs share: `input_var=4`, `output_var=4`, `edge_var=4`, `LearningR=0.0001`, `std_noise=0.001`, `Batch_size=50`, `Training_epochs=500`, `use_node_types=True`, `use_world_edges=True`.

#### Training Configs

| File | GPU | Latent | Modelpath | Run command |
|------|-----|--------|-----------|-------------|
| `config_flag_simple1.txt` | 0 | 256 | `./outputs/flag_simple1.pth` | `python MeshGraphNets_main.py --config _flag_input/config_flag_simple1.txt` |
| `config_flag_simple2.txt` | 1 | 128 | `./outputs/flag_simple2.pth` | `python MeshGraphNets_main.py --config _flag_input/config_flag_simple2.txt` |

Sweep over **Latent_dim** (256 vs 128) across GPUs 0 and 1.

#### Inference Configs

| File | GPU | Modelpath | Infer Dataset | Steps | Run command |
|------|-----|-----------|---------------|-------|-------------|
| `config_flag_simple_infer1.txt` | 0 | `./outputs/flag_simple1.pth` | `./infer/flag_inference.h5` | 1000 | `python MeshGraphNets_main.py --config _flag_input/config_flag_simple_infer1.txt` |
| `config_flag_simple_infer2.txt` | 1 | `./outputs/flag_simple2.pth` | `./infer/flag_inference.h5` | 1000 | `python MeshGraphNets_main.py --config _flag_input/config_flag_simple_infer2.txt` |

Each inference config points to the corresponding trained model.

---

### Flag Dynamic Configs (`_flag_input/`)

Multi-timestep (T>1) flag dynamics with node types and world edges. Dataset: `./dataset/flag_dynamic.h5`.

All flag dynamic configs share: `input_var=4`, `output_var=4`, `edge_var=4`, `LearningR=0.0001`, `std_noise=0.0001`, `Batch_size=1`, `Training_epochs=500`, `use_node_types=True`, `use_world_edges=True`.

Note: `Batch_size=1` because each sample already generates T-1 training pairs. `std_noise=0.0001` is smaller than static configs to preserve transient dynamics.

| File | GPU | Latent | Modelpath | Run command |
|------|-----|--------|-----------|-------------|
| `config_flag_dynamic1.txt` | 0 | 256 | `./outputs/flag_dynamic1.pth` | `python MeshGraphNets_main.py --config _flag_input/config_flag_dynamic1.txt` |
| `config_flag_dynamic2.txt` | 1 | 128 | `./outputs/flag_dynamic2.pth` | `python MeshGraphNets_main.py --config _flag_input/config_flag_dynamic2.txt` |

No separate inference configs exist for flag dynamic -- create one by copying a train config and changing `mode` to `inference`.

---

## Writing New Config Files

### Naming Convention

```
_<dataset_name>_input/
├── config_train1.txt          # Training variant 1
├── config_train2.txt          # Training variant 2
├── config_infer1.txt          # Inference for model 1
├── config_infer2.txt          # Inference for model 2
└── ...
```

- Folder: `_<name>_input/` (underscore prefix, `_input` suffix)
- Train files: `config_train{N}.txt`
- Inference files: `config_infer{N}.txt`
- Alternatively for named datasets: `config_<dataset>_<type>{N}.txt` (e.g., `config_flag_simple1.txt`)

### Train Config Template

```
model   MeshGraphNets
mode    train  # Train / Inference
gpu_ids 0      # -1 for CPU, GPU ids for multi-GPU training
log_file_dir    train1.log
modelpath   ./outputs/my_model1.pth
%   Datasets
dataset_dir ./dataset/my_dataset.h5
infer_dataset   ./dataset/my_infer.h5
infer_timesteps 100
%   Common params
input_var   4   # number of input variables: x_disp, y_disp, z_disp, stress (excluding node types)
output_var  4   # number of output variables: x_disp, y_disp, z_disp, stress (excluding node types)
edge_var    4   # dx, dy, dz, disp
'
%   Network parameters
message_passing_num 15
Training_epochs	500
Batch_size	50
LearningR	0.0001
Latent_dim	128	# MeshGraphNets latent dimension
num_workers 10
std_noise   0.001
residual_scale  0.1
verbose     False
monitor_gradients  False
'
% Memory Optimization
use_checkpointing   False
'
% Performance Optimization
use_parallel_stats  True
'
% Node Type Parameters
use_node_types  False
'
% World Edge Parameters
use_world_edges         False
world_radius_multiplier 1.5
world_max_num_neighbors 64
world_edge_backend      scipy_kdtree
% Test set control
display_testset True
test_batch_idx  0, 1, 2, 3
plot_feature_idx    -2
```

### Inference Config Template

The inference config is typically a copy of the train config with one change: `mode` set to `inference`.

```
model   MeshGraphNets
mode    inference  # <-- changed from train
gpu_ids 0
log_file_dir    train1.log
modelpath   ./outputs/my_model1.pth   # <-- must point to trained checkpoint
%   Datasets
dataset_dir ./dataset/my_dataset.h5
infer_dataset   ./dataset/my_infer.h5
infer_timesteps 100
%   Common params
input_var   4
output_var  4
edge_var    4
'
%   Network parameters (overridden by checkpoint, but must be present for parsing)
message_passing_num 15
Training_epochs	500
Batch_size	50
LearningR	0.0001
Latent_dim	128
num_workers 10
std_noise   0.001
residual_scale  0.1
verbose     False
monitor_gradients  False
'
% Memory Optimization
use_checkpointing   False
'
% Performance Optimization
use_parallel_stats  True
'
% Node Type Parameters
use_node_types  False
'
% World Edge Parameters
use_world_edges         False
world_radius_multiplier 1.5
world_max_num_neighbors 64
world_edge_backend      scipy_kdtree   # <-- REQUIRED, even if use_world_edges=False
% Test set control
display_testset True
test_batch_idx  0, 1, 2, 3
plot_feature_idx    -2
```

**Key difference from train config**: Only `mode` changes to `inference`. All other parameters remain (training params are ignored, architecture params are overridden by checkpoint). The simplest way to create an inference config is to copy the train config and change the `mode` line.

### Hyperparameter Sweep Setup

To run a grid search, create multiple config files varying the parameters of interest. Assign each to a different GPU so they run in parallel:

```
_my_experiment/
├── config_train1.txt    # GPU 0: Latent_dim=128, MP=15
├── config_train2.txt    # GPU 1: Latent_dim=128, MP=5
├── config_train3.txt    # GPU 0: Latent_dim=256, MP=15  (after train1 finishes)
├── config_train4.txt    # GPU 1: Latent_dim=256, MP=5   (after train2 finishes)
├── config_infer1.txt    # Inference for best model
└── config_infer2.txt    # Inference for second best
```

Run in parallel terminals:

```bash
# Terminal 1 (GPU 0)
python MeshGraphNets_main.py --config _my_experiment/config_train1.txt

# Terminal 2 (GPU 1)
python MeshGraphNets_main.py --config _my_experiment/config_train2.txt
```

Each config must use a different `modelpath`, `log_file_dir`, and `gpu_ids` to avoid conflicts.

---

## HDF5 Dataset Structure

### Complete Hierarchy

```
dataset.h5
├── [Attributes]
│   ├── num_samples: int
│   ├── num_features: int            # Typically 8
│   └── num_timesteps: int           # Must be equal across all samples
│
├── data/
│   └── {sample_id}/                 # Integer ID (not necessarily sequential)
│       ├── nodal_data               # (num_features, num_timesteps, num_nodes) float32
│       ├── mesh_edge                # (2, num_edges) int64
│       └── metadata/
│           ├── [Attributes]
│           │   ├── num_nodes: int
│           │   ├── num_edges: int
│           │   └── source_filename: str  (optional)
│           ├── feature_min          # (num_features,) float32, optional
│           ├── feature_max          # (num_features,) float32, optional
│           ├── feature_mean         # (num_features,) float32, optional
│           └── feature_std          # (num_features,) float32, optional
│
└── metadata/
    ├── feature_names                # (num_features,) byte strings
    ├── normalization_params/
    │   ├── min, max, mean, std      # (num_features,) float32
    │   ├── delta_mean               # (output_var,) float32
    │   └── delta_std                # (output_var,) float32
    └── splits/                      # Optional
        ├── train                    # (N_train,) int64
        ├── val                      # (N_val,) int64
        └── test                     # (N_test,) int64
```

### nodal_data

**Shape**: `(num_features, num_timesteps, num_nodes)` -- **features-first layout**

| Index | Name | Description | Used by model |
|-------|------|-------------|---------------|
| 0 | `x_coord` | X reference position | Reference geometry |
| 1 | `y_coord` | Y reference position | Reference geometry |
| 2 | `z_coord` | Z reference position | Reference geometry |
| 3 | `x_disp` | X displacement (mm) | Input/output feature |
| 4 | `y_disp` | Y displacement (mm) | Input/output feature |
| 5 | `z_disp` | Z displacement (mm) | Input/output feature |
| 6 | `stress` | Stress (MPa) | Input/output feature |
| 7 | `part_number` | Part number (integer) | Node types (if enabled) |

**How the model reads this:**
- **Reference position**: `nodal_data[:3, t, :]` -> `[x, y, z]`
- **Physical features**: `nodal_data[3:3+input_var, t, :]` -> typically `[x_disp, y_disp, z_disp, stress]`
- **Node types**: `nodal_data[-1, 0, :]` -> last feature, first timestep
- **Deformed position**: `nodal_data[:3, t, :] + nodal_data[3:6, t, :]`

### mesh_edge

**Shape**: `(2, num_edges)` -- stored **unidirectional**. Code automatically makes bidirectional: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`.

### Timestep Handling

| Scenario | `num_timesteps` | Training pairs per sample | Target computation |
|----------|-----------------|--------------------------|-------------------|
| **Static** | 1 | 1 | `delta = feature_values - 0` (from zero initial state) |
| **Transient** | T > 1 | T - 1 | `delta = state_{t+1} - state_t` |

---

## Data Flow: How the Model Uses Data

### 1. Loading and Feature Extraction

```
nodal_data shape: [features, timesteps, nodes]

For timestep t:
  reference_pos     = nodal_data[:3, t, :].T            -> [N, 3]
  physical_features = nodal_data[3:3+input_var, t, :].T  -> [N, input_var]

For T=1 (static):
  x_raw = zeros(N, input_var)          # input is all zeros
  target_delta = feature_values - 0    # delta equals the features themselves

For T>1 (transient):
  x_raw = nodal_data[3:3+input_var, t, :].T       # state at t
  y_raw = nodal_data[3:3+output_var, t+1, :].T    # state at t+1
  target_delta = y_raw - x_raw                     # delta between steps
```

### 2. Normalization (Z-score)

```
Node features:    x_norm      = (x_raw - node_mean) / node_std
Target deltas:    target_norm = (target_delta - delta_mean) / delta_std
Edge features:    edge_norm   = (edge_raw - edge_mean) / edge_std
```

Stats computed from **entire training dataset** during initialization. Minimum std clamped to `1e-8`.

### 3. Edge Feature Computation

```
deformed_pos  = reference_pos + displacement    # displacement = x_raw[:, :3]
relative_pos  = deformed_pos[dst] - deformed_pos[src]   -> [2M, 3]
distance      = ||relative_pos||                         -> [2M, 1]
edge_features = [relative_pos, distance]                 -> [2M, 4]
```

Edge features always computed from **deformed positions**, not reference.

### 4. Node Type Encoding (if enabled)

```
node_types = nodal_data[-1, 0, :]
one_hot    = to_one_hot(node_types)   -> [N, num_node_types]
x_final    = concat(x_norm, one_hot)  -> [N, input_var + num_node_types]
```

Concatenated **after** normalization.

### 5. Model Forward Pass

```
Input:  x_final [N, input_var (+node_types)], edge_attr [2M, 4]
  -> Encoder: project to latent_dim
  -> Processor: message_passing_num x (EdgeBlock + NodeBlock) with node residuals
  -> Decoder: project to output_var (no LayerNorm)
Output: predicted_delta_norm [N, output_var]
```

### 6. Denormalization (Inference)

```
predicted_delta = predicted_delta_norm * delta_std + delta_mean
state_{t+1}     = state_t + predicted_delta
```

---

## Checkpoint Contents

Checkpoints (`.pth` files) saved during training contain:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | int | Epoch number when saved |
| `model_state_dict` | OrderedDict | Model weights |
| `optimizer_state_dict` | dict | Adam optimizer state |
| `scheduler_state_dict` | dict | LR scheduler state |
| `train_loss` | float | Training loss at this epoch |
| `valid_loss` | float | Best validation loss so far |
| `normalization` | dict | Z-score normalization statistics |
| `model_config` | dict | Model architecture parameters |

### `normalization` dict

| Key | Shape | Description |
|-----|-------|-------------|
| `node_mean` | `[input_var]` | Per-feature mean of node features |
| `node_std` | `[input_var]` | Per-feature std (min 1e-8) |
| `edge_mean` | `[4]` | Per-feature mean of edge features |
| `edge_std` | `[4]` | Per-feature std (min 1e-8) |
| `delta_mean` | `[output_var]` | Per-feature mean of target deltas |
| `delta_std` | `[output_var]` | Per-feature std (min 1e-8) |
| `node_type_to_idx` | dict | *(optional)* Node type -> contiguous index |
| `num_node_types` | int | *(optional)* Number of unique node types |
| `world_edge_radius` | float | *(optional)* Computed world edge radius |

### `model_config` dict

| Key | Type | Description |
|-----|------|-------------|
| `input_var` | int | Node input features |
| `output_var` | int | Node output features |
| `edge_var` | int | Edge features |
| `latent_dim` | int | MLP hidden dimension |
| `message_passing_num` | int | Message passing blocks |
| `use_node_types` | bool | Whether node types were used |
| `num_node_types` | int | Number of node types (0 if not used) |
| `use_world_edges` | bool | Whether world edges were used |
| `use_checkpointing` | bool | Whether gradient checkpointing was used |

---

## Output Directory Structure

### Training

```
outputs/
├── <log_file_dir>              # e.g., train1.log
├── <modelpath>                 # e.g., warpage1.pth (single GPU)
├── best_model.pth              # (multi-GPU DDP only)
├── debug_epoch*.npz            # Debug data (from epoch 5+)
└── test/
    └── <gpu_ids>/
        └── <epoch>/
            └── sample{id}_t{time}.h5
```

### Inference

```
<inference_output_dir>/         # Default: outputs/rollout/
├── rollout_sample{id}_steps{N}.h5
├── rollout_sample{id}_steps{N}.h5
└── ...
```

### Training Log Format

Header includes timestamp and full config. Each epoch:

```
Elapsed time: <seconds>s Epoch <N> Train Loss: <X.XXXXe-XX> Valid Loss: <X.XXXXe-XX> LR: <X.XXXXe-XX>
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Insufficient VRAM | Reduce `Batch_size`, enable `use_checkpointing`, reduce `Latent_dim` |
| `FileNotFoundError: Model checkpoint not found` | Wrong `modelpath` | Verify path exists |
| `KeyError: 'normalization'` | Old checkpoint format | Re-train with current code |
| `ValueError: HDF5 file missing 'data' group` | Invalid dataset | Check HDF5 follows structure above |
| `AttributeError: 'NoneType' has no attribute 'lower'` | `world_edge_backend` missing | Add `world_edge_backend scipy_kdtree` to config |
| `torch.save` error with `None` path | `modelpath` missing | Add `modelpath ./outputs/my_model.pth` |
| Loss is `nan` | LR too high or bad data | Reduce `LearningR` to 1e-5, check data for NaN/Inf |
| Loss plateaus at 1e-1+ | Insufficient capacity, LR too high, or noise too high | Try: (1) Reduce `std_noise` to 0.001 or 0.0; (2) Reduce `LearningR` by 2-5x; (3) Check per-feature loss with `verbose True`; (4) Increase `Latent_dim` or `message_passing_num` |
| Loss doesn't drop after warm-up | Learning rate decays too fast or plateaus | Check scheduler: single GPU uses ExponentialLR(gamma=0.995); DDP uses ReduceLROnPlateau. Verify initial `LearningR` is appropriate |
| `Address already in use` (DDP) | Port 12355 in use | Kill lingering processes or wait |
| Near-zero delta std warning | Constant features | Check that features actually vary between timesteps |

### VRAM Estimates

Approximate VRAM for a single sample with ~10k nodes:

| Latent_dim | MP Blocks | Checkpointing | Approx. VRAM |
|------------|-----------|---------------|--------------|
| 128 | 15 | Off | ~2-4 GB |
| 256 | 15 | Off | ~4-8 GB |
| 256 | 15 | On | ~2-3 GB |
| 512 | 15 | Off | ~8-16 GB |
| 512 | 30 | On | ~6-10 GB |

Multiply by `Batch_size` for total VRAM.

### Hyperparameter Tuning Order

Tune in this order (most to least impactful):

1. **LearningR**: Start 1e-4, try 1e-3 and 1e-5. **Single-GPU note**: ExponentialLR(gamma=0.995) decays LR by ~26% every 50 epochs; if training plateaus early, reduce LR before training
2. **std_noise**: Start 0.001 (originally 0.02 was too high for normalized features). Reduce to 0.0 if loss is unstable
3. **Latent_dim**: Start 128, try 256 if underfitting
4. **feature_loss_weights**: If one feature dominates loss, down-weight it. Example: `1.0, 1.0, 5.0` emphasizes z_disp 5x
5. **Batch_size**: Largest that fits in VRAM
6. **message_passing_num**: Start 15, increase if loss plateaus
7. **Training_epochs**: Monitor validation loss for convergence

---

*Last updated: 2026-03-03*
