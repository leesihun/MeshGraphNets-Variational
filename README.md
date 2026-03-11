# MeshGraphNets

A Graph Neural Network (GNN) for simulating deformable mesh dynamics. Predicts node displacements and stresses using an Encoder-Processor-Decoder architecture with graph message passing. Based on NVIDIA PhysicsNeMo and DeepMind's MeshGraphNets paper.

The model predicts **normalized deltas** (state_{t+1} - state_t), not absolute values. This allows the same trained model to generalize across different geometry scales and simulation conditions.

## Quick Start

### Training

```bash
# Edit configuration (or use example configs from _flag_input/ or _warpage_input/)
vim config.txt  # Set mode=Train, dataset_dir, hyperparameters

# Run training (auto-detects single/multi-GPU based on gpu_ids)
python MeshGraphNets_main.py

# Or specify a config file explicitly
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt

# Monitor training in real-time
pip install -r misc/requirements_plotting.txt
python misc/plot_loss_realtime.py config.txt  # Visit http://localhost:5000
```

### Inference (Autoregressive Rollout)

```bash
# Set mode=Inference, modelpath, infer_dataset, infer_timesteps in config
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

### Utilities

```bash
python misc/plot_loss.py config.txt --output loss_plot.png   # Static loss plot
python misc/debug_model_output.py                            # Check for NaN/zero outputs
python generate_inference_dataset.py                         # Create inference dataset from training data
python animate_h5.py                                         # Animate mesh deformation from HDF5
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+ with CUDA support
- PyTorch Geometric
- h5py, numpy, scipy

### Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install h5py numpy scipy scikit-learn matplotlib pyvista

# Optional: real-time training dashboard
pip install -r misc/requirements_plotting.txt
```

## Configuration

Configuration is managed through plain-text config files. By default, `MeshGraphNets_main.py` reads `config.txt` from the repo root. Use `--config` to specify a different file.

Example configs are provided in `_flag_input/` and `_warpage_input/`.

### File Format

- `key value` pairs (one per line)
- `%` starts a comment line
- `'` marks section separators (optional, ignored)
- `#` inline comments are stripped from values
- Keys are case-insensitive (converted to lowercase internally)
- Comma-separated values become lists (e.g., `gpu_ids 0, 1, 2, 3`)

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **mode** | str | - | `Train` or `Inference` |
| **gpu_ids** | int/list | - | `-1` for CPU, single GPU (e.g., `0`), multi-GPU DDP (e.g., `0, 1, 2, 3`) |
| **dataset_dir** | str | - | Path to HDF5 training dataset |
| **modelpath** | str | - | Checkpoint save path (single-GPU) or load path (inference). DDP always saves to `outputs/best_model.pth` |
| **infer_dataset** | str | - | Path to HDF5 inference dataset |
| **infer_timesteps** | int | - | Number of rollout steps to predict |
| **input_var** | int | 4 | Node input features (excluding node types) |
| **output_var** | int | 4 | Node output features |
| **edge_var** | int | 4 | Edge features (always: `[dx, dy, dz, distance]`) |

### Network Hyperparameters

| Parameter | Type | Default | Range | Impact |
|-----------|------|---------|-------|--------|
| **LearningR** | float | 0.0001 | 1e-6 to 1e-2 | **High** - most sensitive parameter |
| **Latent_dim** | int | 128 | 32-512 | **High** - MLP hidden dimension, affects VRAM quadratically |
| **message_passing_num** | int | 15 | 1-30 | **Medium** - GNN depth |
| **Batch_size** | int | 50 | 1-128 | **High** - per-GPU batch size |
| **Training_epochs** | int | 500 | - | Total training epochs |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **use_checkpointing** | bool | False | Gradient checkpointing: trades 20-30% compute for 60-70% VRAM reduction |
| **use_node_types** | bool | False | One-hot encode node types from HDF5 metadata (last feature of nodal_data) |
| **use_world_edges** | bool | False | Radius-based collision detection edges |
| **use_parallel_stats** | bool | True | Parallel stat computation for large datasets |
| **world_radius_multiplier** | float | 1.5 | Collision radius = multiplier × min_mesh_edge_length |
| **world_max_num_neighbors** | int | 64 | Max neighbors per node in radius query |
| **world_edge_backend** | str | `torch_cluster` | `torch_cluster` (GPU, fast) or `scipy_kdtree` (CPU). **Must be explicitly set in every config** — inference has no fallback default |
| **std_noise** | float | 0.001 | Gaussian noise augmentation during training |
| **verbose** | bool | False | Per-feature loss breakdowns |
| **monitor_gradients** | bool | True | Gradient norm tracking with vanishing/exploding warnings |
| **display_testset** | bool | True | Visualize test set predictions during training |
| **plot_feature_idx** | int | -1 | Feature to visualize (-1 = last/stress, -2 = z_disp) |

For a complete parameter reference, see [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md).

## Architecture

### Execution Flow

1. **Entry point**: `MeshGraphNets_main.py` parses config via `--config` flag (default: `config.txt`)
2. Auto-detects GPU configuration from `gpu_ids` and routes:
   - Single GPU/CPU → `training_profiles/single_training.py`
   - Multi-GPU DDP → `training_profiles/distributed_training.py` (via `mp.spawn`)
   - Inference → `inference_profiles/rollout.py`
3. Both training pipelines use `training_profiles/training_loop.py` for epoch logic

### Core Model: Encoder-Processor-Decoder

```
Input (node + edge features)
        |
     Encoder (project to latent_dim via MLPs)
        |
     Processor (message_passing_num GnBlocks)
     |- EdgeBlock: [sender, receiver, edge] -> MLP -> updated edge
     |- NodeBlock: aggregate edges (sum) + node -> MLP -> updated node
     |- Residual connection on nodes only (not edges)
     (repeat message_passing_num times)
        |
     Decoder (project to output_dim, no LayerNorm)
        |
     Output (normalized deltas)
```

**Key details**:
- MLPs: 2 hidden layers, ReLU, LayerNorm on output (except decoder)
- Sum aggregation (forces/stresses accumulate at nodes)
- Kaiming/He initialization
- Optional gradient checkpointing reduces VRAM ~60-70%
- Optional world edges: HybridNodeBlock aggregates mesh + world edges separately

### Data Pipeline

- **HDF5 format**: `nodal_data [features, time, nodes]`, `mesh_edge [2, edges]`, `metadata`
  - Features: `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`
- **Edges**: Always bidirectional (`edge_index = [mesh_edge; mesh_edge[[1,0]]]`)
- **Edge features**: `[dx, dy, dz, distance]` computed from deformed positions (reference + displacement)
- **Normalization**: Z-score per feature, computed separately for nodes, edges, and deltas
- **Node types**: One-hot encoded and concatenated after normalization

See [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) for complete HDF5 specification.

### Training

- **Optimizer**: Adam with gradient clipping (max_norm=5.0)
- **LR Scheduler**:
  - Single GPU: ExponentialLR (gamma=0.995, decays every epoch)
  - Multi-GPU DDP: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-8)
- **Loss**: MSE on normalized deltas per feature
- **Data split**: 80/10/10 train/val/test (seed=42)
- **Checkpoints**:
  - Single GPU: saved to `modelpath` from config
  - Multi-GPU DDP: always saved to `outputs/best_model.pth`
  - Contains: model weights, optimizer state, scheduler state, normalization stats, model config

### Inference

Autoregressive rollout loop:
1. Load initial state from HDF5 (timestep 0)
2. For each step: normalize → build graph → forward pass → denormalize delta → update state
3. Save predicted trajectories to `<inference_output_dir>/rollout_sample{id}_steps{N}.h5`

## Project Structure

```
MeshGraphNets/
├── MeshGraphNets_main.py             # Entry point (--config flag)
├── model/
│   ├── MeshGraphNets.py              # EncoderProcessorDecoder architecture
│   ├── blocks.py                     # EdgeBlock, NodeBlock, HybridNodeBlock
│   └── checkpointing.py             # Gradient checkpointing utilities
├── general_modules/
│   ├── load_config.py                # Config file parser
│   ├── data_loader.py                # DataLoader creation
│   ├── mesh_dataset.py               # Dataset class, normalization, parallel stats
│   └── mesh_utils_fast.py            # Mesh utilities, GPU triangle reconstruction, PyVista rendering
├── training_profiles/
│   ├── single_training.py            # Single-GPU/CPU training pipeline
│   ├── distributed_training.py       # Multi-GPU DDP training
│   └── training_loop.py             # Epoch logic (train, validate, test)
├── inference_profiles/
│   └── rollout.py                    # Autoregressive inference
├── misc/
│   ├── plot_loss.py                  # Static loss plot
│   ├── plot_loss_realtime.py         # Real-time FastAPI dashboard
│   ├── debug_model_output.py         # Model debugging
│   └── requirements_plotting.txt     # Visualization dependencies
├── _flag_input/                      # Example configs for flag simulation
├── _warpage_input/                   # Example configs for warpage simulation
├── generate_inference_dataset.py     # Create inference dataset from training data
├── animate_h5.py                     # Mesh deformation animation
├── dataset/                          # HDF5 datasets and format spec
├── docs/                             # Architecture, world edges, VRAM optimization docs
└── outputs/                          # Training outputs (auto-created)
```

## Troubleshooting

| Issue | Solutions |
|-------|-----------|
| **CUDA OOM** | Reduce `Batch_size` (most effective), enable `use_checkpointing=True`, reduce `Latent_dim`, use multi-GPU |
| **Loss not decreasing** | Reduce `LearningR` by 10x, increase `message_passing_num`, verify dataset path |
| **NaN loss** | Reduce `LearningR` to 1e-5, reduce `std_noise`, check normalization stats |
| **Slow training** | Adjust `num_workers`, use multi-GPU, reduce dataset size for debugging |
| **`AttributeError: 'NoneType' has no attribute 'lower'` during inference** | `world_edge_backend` is missing from config. Add `world_edge_backend scipy_kdtree` |
| **Crash on first model save** | `modelpath` is missing from config. Add `modelpath ./outputs/my_model.pth` |

## Design Decisions

- **Sum aggregation** (not mean): Forces/stresses accumulate at nodes (matches NVIDIA PhysicsNeMo)
- **Normalized delta prediction**: Decouples geometry scale from learned dynamics
- **Bidirectional edges**: Graph is undirected; edges stored both directions
- **Separate delta normalization**: Delta statistics are independent from node statistics
- **Node residuals only**: Edge features have no residual connections (matches NVIDIA)
- **Gradient clipping at 5.0**: Stabilizes deep message passing networks

## References

- "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICML 2020, DeepMind)
- NVIDIA PhysicsNeMo (deforming_plate example)
- PyTorch + PyTorch Geometric

## Documentation

- [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md): Complete parameter reference with HDF5 dataset specification and data flow
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Architecture walkthrough
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection
- [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization
- [docs/VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization
- [docs/ADAPTIVE_REMESHING_PLAN.md](docs/ADAPTIVE_REMESHING_PLAN.md): Adaptive remeshing
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 structure
- [misc/README.md](misc/README.md): Visualization tools

---

Version 1.0.0, 2026-01-06 | Developed by SiHun Lee, Ph. D.
