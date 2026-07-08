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
- **vae_graph_aware True** lets the posterior encoder see graph input features
  alongside target y, enabling type-conditional spread encoding. Recommended
  when data contains multiple manufactured part types.
- **The conditional prior `p(z|graph)`** is the architectural fix for
  type-to-type generalization: it maps each part type to its spread
  distribution at inference time. It trains jointly with the VAE
  (`prior_type gnn_e2e`, the default when `use_vae True`).
- **The prior is trained ONLY by density matching** on a fresh detached
  posterior sample each step, weighted by `prior_nll_weight`. `prior_family`
  selects the family: `fm` (default) is a conditional flow-matching prior —
  velocity-MSE regression, no components, no collapse machinery; `gmm` is a
  graph-conditional Gaussian mixture — mixture NLL plus a small
  `prior_kl_reg_weight` analytical-KL anchor. Variance-collapsing objectives
  (deterministic z=0 auxiliary loss `lambda_det`, single-sample prior
  reconstruction `alpha_prior_max`) were removed from the code entirely — do
  not reintroduce them.

## Run Commands

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer1_main.txt
```

`mode` is inside the config. `--config` only picks the file.

## Runtime Modes

| Mode | Code path |
|------|-----------|
| `train` | `single_worker` or `train_worker`; with `use_vae True` the conditional prior trains jointly inside the epoch loop. |
| `inference` | `inference_profiles.rollout.run_rollout`; samples z from the conditional prior in the checkpoint (set `use_conditional_prior False` to fall back to N(0,I)). |

`gpu_ids` length controls single process vs DDP unless `parallel_mode model_split`
is set. `parallel_mode model_split` routes to `parallelism.launcher`.

## Key Files

| File | Role |
|------|------|
| [MeshGraphNets_main.py](MeshGraphNets_main.py) | Config load, mode dispatch, DDP/model-split launch. |
| [model/MeshGraphNets.py](model/MeshGraphNets.py) | Top-level model, flat/multiscale processor, VAE latent fusion. |
| [model/encoder_decoder.py](model/encoder_decoder.py) | Encoder, GnBlock, Decoder. |
| [model/blocks.py](model/blocks.py) | EdgeBlock, NodeBlock, HybridNodeBlock, UnpoolBlock. |
| [model/vae.py](model/vae.py) | Graph VAE encoder and MMD regularizer. |
| [model/conditional_prior.py](model/conditional_prior.py) | Mesh-conditioned priors for `z`: flow matching (`fm`, default) and Gaussian mixture (`gmm`), shared GNN trunk. |
| [model/coarsening.py](model/coarsening.py) | BFS and Voronoi coarsening, pooling, `MultiscaleData`. |
| [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) | HDF5 loading, split, normalization, world/multiscale attrs. |
| [general_modules/positional_features.py](general_modules/positional_features.py) | Rotation-invariant positional node features (centroid dist, edge length, RWPE). |
| [general_modules/edge_features.py](general_modules/edge_features.py) | 8-D reference/deformed edge features. |
| [general_modules/world_edges.py](general_modules/world_edges.py) | scipy KDTree or torch-cluster world-edge construction. |
| [general_modules/multiscale_helpers.py](general_modules/multiscale_helpers.py) | Shared hierarchy build/attach used by dataset and rollout. |
| [general_modules/multiscale_cache.py](general_modules/multiscale_cache.py) | Shared on-disk hierarchy/positional cache (mandatory for multiscale). |
| [training_profiles/setup.py](training_profiles/setup.py) | Dataset/model/EMA/optimizer/checkpoint helpers. |
| [training_profiles/training_loop.py](training_profiles/training_loop.py) | Train, validate, VAE prior/posterior eval, test visualization. |
| [inference_profiles/rollout.py](inference_profiles/rollout.py) | Autoregressive rollout and HDF5 output. |
| [parallelism/](parallelism) | Experimental model-split training across GPUs. |

## Architecture Facts

- Edge features are 8-D. `edge_var` must be `8`.
- Node input size is `input_var + positional_features + optional num_node_types`.
- MLPs use SiLU, not ReLU: `Linear -> SiLU -> Linear -> SiLU -> Linear`.
- LayerNorm is appended once at the MLP output when `layer_norm=True`.
- Decoder output has no LayerNorm. For time-transient runs, decoder final weights
  are scaled by `0.01` at initialization.
- Node aggregation is sum. GnBlock residual connections are unscaled (x + Δx).
- World edges use a separate edge encoder/block and a hybrid node block that
  aggregates mesh and world messages separately.
- Multiscale requires `mp_per_level` (2·L+1 entries) and always uses the learned
  bipartite unpool (`UnpoolBlock`); the broadcast unpool was removed. Per-level
  skip states are merged by `Linear(2 * latent_dim, latent_dim)`.
- Multiscale hierarchies always come from the shared on-disk cache
  (`*.mscache.*.h5` next to the dataset); there is no in-RAM fallback.
- `coarsening_type` accepts `bfs`, `voronoi_centroid`, `voronoi_inherit`, and
  `voronoi_seedmean` per level (the bare `voronoi` alias was removed). Inherit
  mode pools by gathering the FPS seed feature (coarse node *is* the seed);
  centroid mode pools by scatter mean; seedmean uses FPS seed positions as
  coarse anchors (on-manifold geometry) but scatter-mean pool — it does **not**
  write `coarse_seed_idx_{i}`. Mixing modes per level is supported. Stats are
  mode-specific, so a checkpoint trained in one mode cannot be loaded into
  another.

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
every processor block. Validation logs posterior reconstruction plus a
learned-prior CRPS pass that mirrors inference sampling.

For manufacturing spread modeling the recommended training loss weights are:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lambda_mmd` | 0.1 | Low: z must encode structured spread, not collapse to N(0,I) |
| `beta_aux` | 1.0 | High: anchors z to graph output statistics, prevents collapse |
| `posterior_min_std` | 0.05 | Floor on σ_q; prevents the prior from memorizing point masses |
| `vae_latent_dim` | 32–64 | Full capacity needed to represent spread across part types |
| `vae_graph_aware` | True | Enables type-conditional spread encoding |
| `prior_family` | `fm` | Flow-matching prior (default): expressive density matching with no component-collapse modes; models all num_z slots jointly |
| `prior_nll_weight` | 1.0 | Weight of the prior objective (FM velocity-MSE, or mixture NLL for `gmm`) |
| `prior_fm_steps` | 20 | Euler ODE steps for FM prior sampling |
| `prior_kl_reg_weight` | 0.05 | gmm family only: small analytical-KL stability anchor (ignored for `fm`) |

Validation prints a per-slot `[PriorDiag]` line (posterior cloud std vs prior
std). A `spread_ratio` near 1 is healthy; near 0 means the prior collapsed and
generated samples will have far too little spread. The CRPS metric mirrors
inference (z sampled from the learned `p(z|graph)`).

Rollout samples z from the joint-trained conditional prior in the checkpoint
(legacy separately-saved `conditional_prior_state_dict` checkpoints still load),
falling back to `N(0, I)` when none exists. Because rollout applies checkpoint
`model_config` before reading `use_conditional_prior`, a checkpoint saved with
that key false can suppress the conditional prior even if the inference config
sets it true. The post-hoc sklearn GMM prior (`model/latent_gmm.py`,
`mode train_prior`, `fit_latent_gmm`) was removed.

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
