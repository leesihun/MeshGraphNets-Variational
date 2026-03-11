# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MeshGraphNets** is a GNN for simulating deformable mesh dynamics. It predicts node displacements and stresses using an Encoder-Processor-Decoder architecture with graph message passing. Based on NVIDIA PhysicsNeMo and DeepMind's MeshGraphNets paper (Pfaff et al., ICML 2020).

The model predicts **normalized deltas** (state_{t+1} - state_t), not absolute values. This decouples geometry scale from learned dynamics.

## Commands

```bash
# Training (reads config.txt by default, or specify --config)
python MeshGraphNets_main.py
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt

# Inference (set mode=Inference in config)
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt

# Real-time loss dashboard
pip install -r misc/requirements_plotting.txt
python misc/plot_loss_realtime.py config.txt  # http://localhost:5000

# Static loss plot
python misc/plot_loss.py config.txt --output loss_plot.png

# Debug model outputs
python misc/debug_model_output.py

# Generate inference dataset from training data
python generate_inference_dataset.py

# Animate mesh deformation from HDF5
python animate_h5.py
```

No test suite exists. No linter is configured.

## Configuration (config.txt)

Plain text format: `key value`. Lines starting with `%` are comments. `'` marks section separators. `#` inline comments are stripped. Keys are **case-insensitive** (lowercased by [load_config.py](general_modules/load_config.py)). Comma-separated values become lists.

Example configs in [_flag_input/](\_flag_input/) and [_warpage_input/](\_warpage_input/).

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| mode | - | `Train` or `Inference` |
| gpu_ids | 0 | `-1`=CPU, `0`=single GPU, `0,1,2,3`=multi-GPU DDP |
| modelpath | - | Checkpoint save/load path |
| dataset_dir | - | HDF5 training dataset |
| infer_dataset | - | HDF5 inference dataset |
| infer_timesteps | - | Rollout steps for inference |
| input_var | 4 | Node input features (excluding node types) |
| output_var | 4 | Node output features |
| edge_var | 4 | Always `[dx, dy, dz, distance]` |
| Latent_dim | 128 | MLP hidden dimension |
| message_passing_num | 15 | GNN depth (number of processor blocks) |
| Batch_size | 50 | Per-GPU batch size |
| LearningR | 0.0001 | Adam learning rate |
| Training_epochs | 500 | |
| std_noise | 0.001 | Gaussian noise augmentation (training only) |
| feature_loss_weights | None | Per-feature loss weights (comma-separated). Example: `1.0, 1.0, 5.0` emphasizes z_disp. Auto-normalized. |
| use_checkpointing | False | Gradient checkpointing (saves ~60-70% VRAM) |
| use_amp | False | Mixed precision training with bfloat16 (1.5-2x speedup on Ampere+ GPUs). Uses bfloat16 not float16 due to scatter_add overflow issues in GNNs |
| use_compile | False | `torch.compile(dynamic=True)` for kernel fusion (10-30% speedup). First epoch slower due to JIT warmup |
| test_interval | 10 | Run test/visualization every N epochs. Previous default was 1 (every epoch) |
| use_node_types | False | One-hot encode node types from HDF5 metadata |
| use_world_edges | False | Radius-based collision detection edges |
| use_parallel_stats | True | Parallel normalization stat computation |
| verbose | False | Per-feature loss breakdowns |
| monitor_gradients | True | Gradient norm tracking |
| world_edge_backend | `torch_cluster` | `torch_cluster` (GPU) or `scipy_kdtree` (CPU). **Required in every config** — rollout.py has no default and crashes if absent |
| use_vae | False | Enable variational encoder (global stochastic latent z) |
| vae_latent_dim | 32 | Dimension of global latent z |
| beta_kl | 0.001 | KL divergence weight β in loss |
| alpha_recon | 1.0 | Reconstruction MSE weight α in loss |
| kl_anneal_epochs | 0 | Linearly ramp β from 0 over N epochs (0=disabled) |

Full reference: [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md)

## Architecture

### Execution Flow

```
MeshGraphNets_main.py (entry point, --config flag)
  ├── mode=Train + single GPU  → single_training.py → training_loop.py
  ├── mode=Train + multi GPU   → distributed_training.py (DDP) → training_loop.py
  └── mode=Inference            → rollout.py (autoregressive)
```

### Model: EncoderProcessorDecoder ([model/MeshGraphNets.py](model/MeshGraphNets.py))

```
Input → Encoder → [GnBlock × message_passing_num] → Decoder → Output
```

- **Encoder**: Projects node features and edge features to `latent_dim` via MLPs
- **GnBlock** ([model/blocks.py](model/blocks.py)): EdgeBlock updates edges, NodeBlock aggregates edges to nodes
  - **Residual connections on nodes only** (not edges), matching NVIDIA implementation
  - With `use_world_edges`: HybridNodeBlock aggregates mesh + world edges separately then concatenates
- **Decoder**: Projects `latent_dim` to `output_var` (no LayerNorm on output)
- **MLP**: 2 hidden layers, ReLU, LayerNorm on output (except decoder)
- **Initialization**: Kaiming/He uniform
- **Aggregation**: Sum (forces/stresses accumulate at nodes)

### Variational Extension (MGN-V, `use_vae=True`)

When `use_vae=True`, a **VariationalEncoder** adds a global stochastic latent `z` between the Encoder and Processor:

```
Training:  graph.y → VariationalEncoder → mean_pool → μ, log(σ²) → z = μ + σ·ε
Inference: z ~ N(0, I)

Encoder output [N, latent_dim] + z broadcast [N, vae_latent_dim]
  → concat → condition_proj (Linear) → [N, latent_dim] → Processor (unchanged)
```

- **Loss**: `α · MSE + β · KL(q(z|Δ) ∥ N(0,1))` where `KL = -½ · mean(1 + log(σ²) - μ² - exp(log(σ²)))`
- **`graph.y` is consumed by the VAE encoder** during training only; at inference `z` is sampled from the prior
- **`MeshGraphNets.forward` returns 3 values**: `(predicted, target, kl_loss)` — `target` is `None` during rollout, `kl_loss` is 0.0 when `use_vae=False` or at eval time
- **KL annealing**: optional linear warmup of β over `kl_anneal_epochs`

### Data Pipeline ([general_modules/](general_modules/))

- [mesh_dataset.py](general_modules/mesh_dataset.py): Loads HDF5, computes Z-score normalization stats, returns `torch_geometric.data.Data` graphs
- [data_loader.py](general_modules/data_loader.py): Creates DataLoaders (DistributedSampler for DDP)
- [mesh_utils_fast.py](general_modules/mesh_utils_fast.py): Edge features, world edge KDTree queries, GPU triangle reconstruction, PyVista rendering

### Training

- **Optimizer**: AdamW with `fused=True` on CUDA (fuses parameter update kernels for 10-15% speedup)
- **LR Scheduler**: ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-8) for both single-GPU and DDP
- **AMP**: Optional bfloat16 mixed precision via `use_amp` config (bfloat16 preferred over float16 for GNN scatter_add safety)
- **DataLoader**: Uses `persistent_workers=True` and `prefetch_factor=2` to avoid worker respawn overhead
- **Loss**: MSE on normalized deltas, with optional per-feature weighting (via `feature_loss_weights` config)
- **Gradient clipping**: max_norm=5.0
- **Checkpoint path**: single-GPU uses `modelpath` from config; DDP always saves to `outputs/best_model.pth` regardless of `modelpath`
- **Checkpoint contents**: model_state_dict, optimizer, scheduler, normalization stats, model_config

### Inference ([inference_profiles/rollout.py](inference_profiles/rollout.py))

Autoregressive rollout: normalize state → build graph → forward pass → denormalize delta → update state → repeat. Outputs HDF5 with predicted trajectories.

## Critical Invariants

These are easy to break if you don't know them:

1. **Edges are always bidirectional**: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`
2. **Node types are one-hot encoded AFTER normalization** and concatenated to normalized features
3. **Edge features are computed from deformed positions** (reference + displacement), not reference positions
4. **Decoder has NO LayerNorm** — this allows full output range for delta prediction
5. **Residual connections only on nodes**, not edges (matches NVIDIA PhysicsNeMo)
6. **Delta normalization is separate** from node normalization (delta_mean/delta_std vs node_mean/node_std)
7. **All samples must have equal timestep counts** (current limitation, printed at startup)
8. **HDF5 nodal_data shape**: `[num_features, num_timesteps, num_nodes]` — features-first, not nodes-first
9. **World edges exclude existing mesh edges** to avoid duplication
10. **Config keys are case-insensitive** — `LearningR` becomes `learningr` internally
11. **`world_edge_backend` is required in all configs** — `rollout.py` calls `.get('world_edge_backend').lower()` with no default; omitting it crashes inference with `AttributeError`
12. **For T=1 (static) datasets**, the model input `x` is all-zeros and the target delta equals the feature values themselves
13. **`num_node_types` is assigned to config by code** (not from config file) after dataset load; do not set it manually
14. **Per-feature loss weights are auto-normalized** — specified weights are scaled so they sum to `output_var`, preserving loss magnitude comparability across configs
15. **Loss weights apply to all three phases** (train, validation, test) for consistent optimization and reporting
16. **`MeshGraphNets.forward` returns 3 values** `(predicted, target, kl_loss)` — all callers must unpack 3 values. `target` is `None` during rollout; `kl_loss` is 0.0 when `use_vae=False` or eval mode
17. **`Encoder.forward` drops `graph.y` and `graph.batch`** — it creates a new `Data(x, edge_attr, edge_index)`. Must save these via `getattr` before calling the encoder
18. **VAE modules only exist when `use_vae=True`** — they are conditionally created in `__init__`, so no unused params in DDP

## HDF5 Dataset Format

Per-sample: `nodal_data` `[features, time, nodes]`, `mesh_edge` `[2, edges]`, `metadata` attributes.
Features: `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`.
Full spec: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

## Normalization

Z-score per feature, computed separately for three domains:
- **Node**: from physical features `nodal_data[3:3+input_var]` across all samples and sampled timesteps (up to 500 per sample)
- **Edge**: from deformed edge positions `[dx, dy, dz, distance]`
- **Delta**: from actual `state_{t+1} - state_t` transitions (for T=1 datasets, delta = feature value itself)

Stored in checkpoint: `checkpoint['normalization']` dict with `node_mean`, `node_std`, `edge_mean`, `edge_std`, `delta_mean`, `delta_std`, plus `node_type_to_idx` and `world_edge_radius` if applicable.

## Additional Documentation

- [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md): Complete parameter reference
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Architecture walkthrough
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection implementation
- [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization strategies
- [docs/VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization for visualization
- [docs/ADAPTIVE_REMESHING_PLAN.md](docs/ADAPTIVE_REMESHING_PLAN.md): Adaptive remeshing plan
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 dataset structure
- [misc/README.md](misc/README.md): Visualization tools
