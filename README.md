# MeshGraphNets - Variational (Hi-MGN-V)

This repository implements Hi-MGN-V, a probabilistic surrogate for
**manufacturing spread modeling**. The training dataset contains multiple
manufactured objects that share the same mesh topology but produce different
physical outputs (displacement, warpage, stress) due to real production
variability. Hi-MGN-V learns the spread of that output distribution and
generates realistic samples that follow the patterns of the training data.
At inference time the number of generated samples can exceed the training
set size, enabling extrapolation of spread structure across part variants.

The architecture is a MeshGraphNets-style graph neural network with optional
hierarchical V-cycle processing, optional world edges, and an MMD-VAE latent
branch that injects a graph-level stochastic code `z` into every processor
block. A mesh-conditioned prior `p(z|graph)`, trained jointly with the VAE,
maps each part type to its spread distribution at inference time.

The executable entry point is [MeshGraphNets_main.py](MeshGraphNets_main.py).

## Current Runtime Truth

The code has two modes, selected inside the config file:

| Mode | What runs |
|------|-----------|
| `train` | Train the simulator. With `use_vae True`, the mesh-conditioned prior (`prior_type gnn_e2e`) trains jointly in the same loop. |
| `inference` | Run autoregressive rollout from an HDF5 initial-condition dataset. |

Inference samples VAE latents in this priority order:

1. Mesh-conditioned prior, when `use_vae True`, `use_conditional_prior True`,
   and the checkpoint carries a prior (joint-trained submodule, or a legacy
   `conditional_prior_state_dict`).
2. Standard normal `N(0, I)`.

Important caveat: rollout loads `checkpoint['model_config']` and lets those keys
override the inference config before it decides whether `use_conditional_prior` is
enabled. If the checkpoint was saved with `use_conditional_prior False`, setting it
only in an inference config can be overwritten. The safest conditional-prior path is
to train or refresh the checkpoint with the conditional-prior keys present.

## Quick Start

```bash
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_train1.txt
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer1_main.txt
```

Training a config with `use_vae True` and `prior_type gnn_e2e` produces a
checkpoint that already contains the conditional prior; inference configs then
sample from it via `use_conditional_prior True`.

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
fused into each processor block. Training uses reconstruction loss (Huber or MSE
via `recon_loss`) plus MMD and the auxiliary latent loss.

For manufacturing spread modeling the key tuning levers are:
- `lambda_mmd 0.1` — keep low; z must retain structured spread, not collapse to N(0,I).
- `beta_aux 1.0` — anchors z to per-graph output statistics; prevents mode collapse.
- `posterior_min_std 0.05` — floors sigma_q so the prior cannot memorize point masses.
- `vae_graph_aware True` — posterior encoder sees graph inputs alongside target y,
  enabling type-conditional spread encoding across part variants.

The conditional prior is [model/conditional_prior.py](model/conditional_prior.py):
a shared graph trunk conditions either a flow-matching velocity field
(`prior_family fm`, default) or a Gaussian mixture head (`prior_family gmm`) for
`p(z | graph)`. It trains jointly with the VAE by density matching on fresh
posterior samples. For spread modeling this prior is essential: it routes each
part type to its own spread distribution at inference time rather than sampling
from a global N(0,I).

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
| Loss | Huber or MSE (`recon_loss`) on normalized deltas, optional normalized feature weights |
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
`z` is conditioned on that mesh and initial state. Set `num_vae_samples` larger
than the training set size to generate more spread samples than were observed;
the conditional prior extrapolates within the learned spread distribution for each
part type.

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
