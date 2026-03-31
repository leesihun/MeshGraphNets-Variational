# MeshGraphNets

A Graph Neural Network surrogate model for FEA (Finite Element Analysis) mesh simulations, based on the [DeepMind MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture with extensions for multiscale processing, positional encoding, mixed-precision training, and distributed training.

Developed by SiHun Lee, Ph.D., MX, SEC — Version 1.0.0 (2026-01-06).

## What it does

Given a mesh at time *t* (node positions + physical state), the model predicts the state at time *t+1* as a **delta**. At inference time it rolls out autoregressively over many timesteps. The primary application is warpage/deformation prediction in manufacturing FEA.

## Setup

Requires PyTorch + PyTorch Geometric. Key optional dependencies:

```bash
# For world edges (long-range connections)
pip install torch-cluster

# HDF5 datasets
pip install h5py
```

Set `HDF5_USE_FILE_LOCKING=FALSE` in your environment if running on shared storage.

## Running

```bash
# Training — single GPU
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Training — multi-GPU DDP (set gpu_ids 0,1 in config)
python MeshGraphNets_main.py --config _warpage_input/config_train3.txt

# Inference / autoregressive rollout
python MeshGraphNets_main.py --config <your_infer_config.txt>
```

The `--config` flag defaults to `config.txt`. Mode (`train` / `inference`) is set inside the config file. Single vs. multi-GPU is auto-detected from the number of IDs in `gpu_ids`.

See [config_run_docs.md](config_run_docs.md) for a full reference of all configuration keys.

## Dataset Format

HDF5 files with a `nodal_data` dataset of shape **`[features, time, nodes]`**:

| Index | Feature |
|---|---|
| 0–2 | x, y, z (reference coordinates) |
| 3–5 | x\_disp, y\_disp, z\_disp |
| 6 | stress (von Mises or equivalent) |
| 7 | part\_number (optional) |

## Outputs

Training checkpoints are saved to the path specified by `modelpath`. Each checkpoint contains:
- Model weights (and optionally `ema_state_dict`)
- Optimizer state
- Normalization statistics under `checkpoint['normalization']`
- Per-level coarse edge stats if multiscale is enabled

Inference outputs are saved as HDF5: `rollout_sample{id}_steps{N}.h5` with nodal_data shape `[8, timesteps, nodes]`.

Loss and validation curves are written to the file specified by `log_file_dir`. Use `misc/plot_loss.py` or `misc/plot_loss_realtime.py` to visualize.

## Architecture

**Encode–Process–Decode** GNN operating on FEA mesh graphs:

- **Encoder:** Independent MLPs encode node features and edge features to a shared latent dimension.
- **Processor:** Stack of GnBlocks (EdgeBlock → NodeBlock with residual connections). Supports a multiscale BFS Bi-Stride V-cycle (ICML 2023) for hierarchical processing.
- **Decoder:** MLP from latent to predicted normalized state delta. No LayerNorm on the output layer.

**Edge features** are 8D: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. Edges are always bidirectional.

**Prediction:** Normalized delta (`Δstate`). Denormalized and added to current state: `state_{t+1} = state_t + delta`.

## Key Config Sections

| Section | Purpose |
|---|---|
| Mode / GPU | `mode`, `gpu_ids` |
| Paths | `modelpath`, `dataset_dir`, `log_file_dir` |
| Model size | `Latent_dim`, `message_passing_num` |
| Training | `LearningR`, `Training_epochs`, `Batch_size`, `use_amp`, `use_ema` |
| Features | `input_var`, `output_var`, `edge_var`, `positional_features` |
| Multiscale | `use_multiscale`, `multiscale_levels`, `mp_per_level` |
| Inference | `infer_dataset`, `infer_timesteps` |

Full documentation: [config_run_docs.md](config_run_docs.md)
