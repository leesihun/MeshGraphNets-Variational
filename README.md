# MeshGraphNets - Variational

This repository trains and runs MeshGraphNets-style graph neural network
surrogates for FEA mesh data. The live code supports deterministic
encode-process-decode simulation, optional hierarchical V-cycle processing,
optional world edges, optional MMD-VAE latent conditioning, legacy GMM latent
sampling, and a newer mesh-conditioned post-hoc latent prior.

The executable entry point is [MeshGraphNets_main.py](MeshGraphNets_main.py).

## Current Runtime Truth

The code has four modes, selected inside the config file:

| Mode | What runs |
|------|-----------|
| `train` | Train the simulator. If `use_vae True` and `fit_latent_gmm True`, fit legacy GMM after training. If `train_conditional_prior True`, also train the conditional prior after training. |
| `train_with_prior` | Same simulator training path as `train`, but the launcher sets `train_conditional_prior True` before training. |
| `train_prior` | Load an existing VAE-MGN checkpoint and train only the mesh-conditioned prior into that checkpoint. |
| `inference` | Run autoregressive rollout from an HDF5 initial-condition dataset. |

Inference samples VAE latents in this priority order:

1. Mesh-conditioned prior, only when `use_vae True`, `use_conditional_prior True`,
   and the checkpoint contains `conditional_prior_state_dict`.
2. Legacy checkpoint GMM, when present.
3. Standard normal `N(0, I)`.

Important caveat: rollout loads `checkpoint['model_config']` and lets those keys
override the inference config before it decides whether `use_conditional_prior` is
enabled. If the checkpoint was saved with `use_conditional_prior False`, setting it
only in an inference config can be overwritten. The safest conditional-prior path is
to train or refresh the checkpoint with the conditional-prior keys present.

## Quick Start

Historical VAE/GMM workflow:

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt
python MeshGraphNets_main.py --config _warpage_input/config_infer4.txt
```

Current B8 all-warpage inference configs request the mesh-conditioned prior:

```bash
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer1.txt
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer2.txt
```

The paired `_b8_all_warpage_input/config_train1.txt` and `config_train2.txt` files
are currently `mode train` and still contain legacy GMM keys. To produce a checkpoint
that contains the conditional prior, use either `mode train_with_prior` for the
simulator training run or run a separate `mode train_prior` config against a trained
VAE checkpoint.

## Architecture

The model is implemented in [model/MeshGraphNets.py](model/MeshGraphNets.py).

Node input features are:

```text
physical state channels from nodal_data[3:3+input_var]
+ optional positional features
+ optional one-hot node types
```

Edge features are always 8-D and validated against `edge_var 8`:

```text
[deformed_dx, deformed_dy, deformed_dz, deformed_dist,
 ref_dx,      ref_dy,      ref_dz,      ref_dist]
```

The base model is:

```text
Encoder: node MLP + mesh edge MLP (+ world edge MLP when enabled)
Processor:
  flat: message_passing_num GnBlocks
  multiscale: pre blocks -> pool -> coarsest blocks -> unpool/skip -> post blocks
Decoder: node MLP to normalized delta
```

MLPs are built by [model/mlp.py](model/mlp.py): `Linear -> SiLU -> Linear -> SiLU
-> Linear`, with a final `LayerNorm` only when `layer_norm=True`. Decoder and prior
heads omit final LayerNorm.

When `use_vae True`, [model/vae.py](model/vae.py) encodes the training target
delta `y` into a graph-level latent `z`. The sampled `z` is broadcast to nodes and
fused into each processor block. Training uses reconstruction Huber loss plus MMD
and auxiliary latent losses. Optional `free_bits` adds a KL floor as a collapse
safeguard.

The post-hoc conditional prior is [model/conditional_prior.py](model/conditional_prior.py):
a graph encoder predicts mixture logits, means, and diagonal log-stds for
`p(z | graph)`. [training_profiles/posthoc_prior.py](training_profiles/posthoc_prior.py)
trains it from frozen VAE posterior samples and saves it into the same checkpoint.

## Data

HDF5 samples use:

```text
data/{sample_id}/nodal_data    float32 [features, timesteps, nodes]
data/{sample_id}/mesh_edge     int64   [2, edges]
```

Default feature layout:

| Index | Field | Used as |
|-------|-------|---------|
| `0:3` | reference `x,y,z` | Geometry only, never part of `input_var` |
| `3:6` | displacement `x,y,z` | Default physical input and output |
| `6` | stress | Used when `input_var/output_var` include it |
| `7` | part number | One-hot node type source when `use_node_types True` |

For multi-step data, each training item is `(sample_id, t)` and the target is
`state[t+1] - state[t]`. For single-step data, the input physical state is zero and
the target is the stored state.

The dataset class always creates a deterministic 80/10/10 split from sample IDs
using `split_seed`; it does not consume existing HDF5 split datasets. Training
statistics are fit on the train split and saved to the checkpoint under
`checkpoint['normalization']`.

More detail: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).

## Training

Single-process training is in
[training_profiles/single_training.py](training_profiles/single_training.py).
DDP training is in
[training_profiles/distributed_training.py](training_profiles/distributed_training.py).

`gpu_ids` controls the launcher:

| Value | Behavior |
|-------|----------|
| `-1` | CPU when CUDA is unavailable or not selected |
| `0` | Single GPU |
| `0,1` | PyTorch DDP via `mp.spawn` |

`parallel_mode model_split` activates the experimental pipeline split launcher in
[parallelism/launcher.py](parallelism/launcher.py). It slices the processor across
GPUs and saves a merged checkpoint for normal inference.

Training uses:

| Component | Current code |
|-----------|--------------|
| Loss | Huber on normalized deltas, optional normalized feature weights |
| Optimizer | Adam, fused on CUDA |
| Schedule | Linear warmup then `CosineAnnealingWarmRestarts` |
| Grad clip | `max_norm=3.0` |
| AMP | bfloat16 autocast when `use_amp True` |
| EMA | optional `AveragedModel`; rollout prefers EMA weights |
| DataLoader | PyG `DataLoader`, spawn workers, `prefetch_factor=1`, pinned memory on CUDA |

## Inference

[inference_profiles/rollout.py](inference_profiles/rollout.py) performs
autoregressive rollout:

```text
state_t -> normalize graph -> model predicts normalized delta
        -> denormalize delta -> state_{t+1} = state_t + delta
```

For each input scene and each VAE sample draw, rollout writes:

```text
{inference_output_dir}/rollout_sample{sample_id}_steps{N}.h5
{inference_output_dir}/rollout_sample{sample_id}_vaesample{idx}_steps{N}.h5
```

The latent `z` is sampled once per trajectory and held fixed across rollout steps.
When the conditional prior is active, the first graph is built before sampling so
`z` is conditioned on that mesh and initial state.

## Documentation Map

| File | Purpose |
|------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Short run guide and workflow warnings |
| [CLAUDE.md](CLAUDE.md) | Agent-facing engineering map |
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Current config keys and semantics |
| [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md) | Architecture details grounded in live code |
| [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md) | Hierarchical V-cycle and coarsening |
| [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md) | World-edge runtime path |
| [hierarchical_interpolation_mgn_comparison.md](hierarchical_interpolation_mgn_comparison.md) | Paper-style comparison for deterministic hierarchical MGN |

## Installation

Install a PyTorch build matching your CUDA environment, then install project deps:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`torch-cluster` is optional unless `world_edge_backend torch_cluster` is requested.
The code falls back to scipy KDTree world-edge construction when torch-cluster is not
available.
