# CLAUDE.md

Agent-facing notes for this repository. Keep answers and edits grounded in the
live code, not older prose in historical plan docs.

## Project Objective

Hi-MGN-V is a probabilistic surrogate for **manufacturing spread modeling**.
The training dataset contains multiple manufactured objects that share the same
mesh topology but produce different physical outputs (e.g. displacement/warpage)
due to real production variability. The model must learn the spread of that
distribution — not just the mean response — and generate realistic samples that
follow the patterns of the input data. At inference time the number of generated
samples exceeds the number of training items, so the model must extrapolate
spread structure, not merely memorize training outputs.

Implications for VAE design:

- **z must encode spread, not be forced toward N(0,I).** Keep `lambda_mmd` low
  (≈ 0.1). A residual MMD > 0 is acceptable because the true aggregate posterior
  reflects genuine spread structure, not noise.
- **beta_aux (≈ 1.0)** anchors z to per-graph output statistics and prevents
  posterior/mode collapse. Do not reduce it.
- **lambda_det must be 0.0.** The deterministic auxiliary loss (second forward
  with z=0) was introduced to fight posterior shortcuts in single-mode problems.
  It conflicts with the spread-modeling objective and must not be used.
- **vae_graph_aware True** lets the posterior encoder see graph input features
  alongside target y, enabling type-conditional spread encoding. Recommended
  when data contains multiple manufactured part types.
- **use_conditional_prior True** (with `train_with_prior` or `train_prior`) is
  the correct architectural fix for type-to-type generalization: the prior
  `p(z|graph)` maps each part type to its spread distribution at inference time.

## Run Commands

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt
python MeshGraphNets_main.py --config _warpage_input/config_infer4.txt
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer1.txt
```

`mode` is inside the config. `--config` only picks the file.

## Runtime Modes

| Mode | Code path |
|------|-----------|
| `train` | `single_worker` or `train_worker`; with `use_vae True`, runs post-hoc conditional prior at the end by default (set `train_conditional_prior False` to skip), then fits legacy GMM if `fit_latent_gmm True`. |
| `train_with_prior` | Backward-compat alias for `train`. Rewritten to `train` at dispatch. |
| `train_prior` | `training_profiles.posthoc_prior.train_posthoc_prior`. |
| `inference` | `inference_profiles.rollout.run_rollout`; loads conditional prior by default if present in the checkpoint (set `use_conditional_prior False` to fall back to GMM/N(0,I)). |

`gpu_ids` length controls single process vs DDP unless `parallel_mode model_split`
is set. `parallel_mode model_split` routes to `parallelism.launcher`.

## Key Files

| File | Role |
|------|------|
| [MeshGraphNets_main.py](MeshGraphNets_main.py) | Config load, mode dispatch, DDP/model-split launch. |
| [model/MeshGraphNets.py](model/MeshGraphNets.py) | Top-level model, flat/multiscale processor, VAE latent fusion. |
| [model/encoder_decoder.py](model/encoder_decoder.py) | Encoder, GnBlock, Decoder. |
| [model/blocks.py](model/blocks.py) | EdgeBlock, NodeBlock, HybridNodeBlock, UnpoolBlock. |
| [model/vae.py](model/vae.py) | Graph VAE encoder, MMD, optional free-bits KL. |
| [model/conditional_prior.py](model/conditional_prior.py) | Mesh-conditioned diagonal Gaussian mixture prior for `z`. |
| [model/latent_gmm.py](model/latent_gmm.py) | Legacy post-hoc GMM collection, fit, and sampling. |
| [model/coarsening.py](model/coarsening.py) | BFS and Voronoi coarsening, pool/unpool, `MultiscaleData`. |
| [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) | HDF5 loading, split, normalization, positional features, world/multiscale attrs. |
| [general_modules/edge_features.py](general_modules/edge_features.py) | 8-D reference/deformed edge features. |
| [general_modules/world_edges.py](general_modules/world_edges.py) | scipy KDTree or torch-cluster world-edge construction. |
| [general_modules/multiscale_helpers.py](general_modules/multiscale_helpers.py) | Shared hierarchy build/attach used by dataset and rollout. |
| [training_profiles/setup.py](training_profiles/setup.py) | Dataset/model/EMA/optimizer/checkpoint helpers. |
| [training_profiles/training_loop.py](training_profiles/training_loop.py) | Train, validate, VAE prior/posterior eval, test visualization. |
| [training_profiles/posthoc_prior.py](training_profiles/posthoc_prior.py) | Frozen-simulator conditional-prior training. |
| [inference_profiles/rollout.py](inference_profiles/rollout.py) | Autoregressive rollout and HDF5 output. |
| [parallelism/](parallelism) | Experimental model-split training across GPUs. |

## Architecture Facts

- Edge features are 8-D. `edge_var` must be `8`.
- Node input size is `input_var + positional_features + optional num_node_types`.
- MLPs use SiLU, not ReLU: `Linear -> SiLU -> Linear -> SiLU -> Linear`.
- LayerNorm is appended once at the MLP output when `layer_norm=True`.
- Decoder output has no LayerNorm. For time-transient runs, decoder final weights
  are scaled by `0.01` at initialization.
- Node aggregation is sum.
- World edges use a separate edge encoder/block and a hybrid node block that
  aggregates mesh and world messages separately.
- Multiscale V-cycle has no global gated encoder skip in the live code. It uses
  per-level skip states merged by `Linear(2 * latent_dim, latent_dim)`.

## Data Facts

- HDF5 stores `mesh_edge` as one-directional unique edges; loaders mirror it to
  bidirectional edges.
- `nodal_data[0:3]` are reference coordinates and are not part of `input_var`.
- Physical channels start at `nodal_data[3]`.
- Multi-step targets are `state[t+1] - state[t]`.
- Single-step targets are the stored state from zero input.
- The dataset class ignores HDF5 split datasets and creates a seeded 80/10/10 split.
- Training writes normalization arrays under `metadata/normalization_params`, but
  split IDs are not written by the current training path.

## VAE And Priors

Training with `use_vae True` uses posterior `q(z | y)` and fuses sampled `z` into
every processor block. Validation logs both posterior reconstruction and prior
sampling reconstruction.

For manufacturing spread modeling the recommended training loss weights are:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_mmd` | 0.1 | Low: z must encode structured spread, not collapse to N(0,I) |
| `beta_aux` | 1.0 | High: anchors z to graph output statistics, prevents collapse |
| `lambda_det` | 0.0 | Must be zero: det auxiliary loss conflicts with spread objective |
| `vae_latent_dim` | 32 | Full capacity needed to represent spread across part types |
| `vae_graph_aware` | True | Enables type-conditional spread encoding |

Post-training priors:

- Conditional prior: `conditional_prior_state_dict`,
  `conditional_prior_config`, `conditional_prior_metrics`.
- Legacy GMM: `gmm_params`.

Rollout tries conditional prior, then GMM, then `N(0, I)`. Because rollout applies
checkpoint `model_config` before reading `use_conditional_prior`, a checkpoint saved
with that key false can suppress the conditional prior even if the inference config
sets it true.

For spread modeling, the conditional prior is strongly preferred over GMM or N(0,I):
it maps each graph (part type + boundary conditions) to its own spread distribution
at inference time, enabling realistic extrapolation across part variants.

## Documentation Notes

The authoritative docs are:

- [README.md](README.md)
- [QUICKSTART.md](QUICKSTART.md)
- [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md)
- [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md)
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md)
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

Files named `*_PLAN.md` or `*_RESEARCH.md` can be historical and should not be
treated as current implementation truth without checking code.
