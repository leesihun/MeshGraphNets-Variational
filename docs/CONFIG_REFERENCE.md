# Configuration Reference

This file documents the configuration keys that are used by the current Python
runtime. Config files are loaded by `general_modules/load_config.py`; keys are
lowercased, comments start at `#`, blank lines and `%` section headers are
ignored, comma-separated values become lists, and `true` / `false` are parsed
case-insensitively.

Run every mode through the same launcher:

```powershell
python MeshGraphNets_main.py --config path\to\config.txt
```

## Required Execution Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `model` | launcher | Informational model name printed at startup. |
| `mode` | launcher | `train` or `inference`. |
| `gpu_ids` | launcher | `-1` for CPU, one GPU id for single process, or comma list for multi-GPU. |
| `parallel_mode` | launcher | Optional. `ddp` by default; `model_split` enables the experimental pipeline split path. |
| `pipeline_microbatches` | model_split | Optional. Batches pipelined per optimizer step under the 1F1B schedule (default `2 * num_stages`; `1` = legacy sequential). Effective batch = `batch_size * pipeline_microbatches`; VRAM does not grow with it. |
| `modelpath` | training, inference, prior | Checkpoint path to save to or load from. |
| `log_file_dir` | training | Relative path under `outputs/` for epoch logs. |

Mode behavior:

- `train`: trains the simulator. With `use_vae True`, the mesh-conditioned prior (`prior_type gnn_e2e`) trains jointly inside the same loop.
- `inference`: runs autoregressive rollout using the checkpoint and writes rollout HDF5 files. Samples z from the conditional prior in the checkpoint by default (set `use_conditional_prior False` to fall back to N(0,I)).

Important caveat: inference loads `checkpoint['model_config']` and overwrites many
architecture keys from the config file before it checks `use_conditional_prior`.
If the checkpoint's `model_config` says `use_conditional_prior False`, then a
config-file `use_conditional_prior True` can be overwritten. For conditional-prior
inference, prefer a checkpoint saved from a conditional-prior-aware run, or patch
that model_config field deliberately.

## Dataset Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `dataset_dir` | training, prior | Training HDF5 dataset. |
| `infer_dataset` | inference | HDF5 dataset used for rollout initial conditions. |
| `infer_timesteps` | inference | Number of autoregressive rollout steps. |
| `split_seed` | training, prior | Optional seeded 80/10/10 split. Defaults to `42` where used. |
| `use_parallel_stats` | data loader | Optional. Enables parallel normalization-stat computation. |

Training always creates its own deterministic 80/10/10 split from sorted
`data/{sample_id}` keys. The loader does not consume `metadata/splits/*` from the
HDF5 file for training.

## Input And Model Shape

| Key | Used by | Meaning |
| --- | --- | --- |
| `input_var` | model, data | Number of physical input channels after xyz coordinates. Current configs use `3` for displacement. |
| `output_var` | model, data | Number of predicted delta channels. Current configs use `3`. |
| `edge_var` | model, data | Must be `8`. The code validates this against `EDGE_FEATURE_DIM`. |
| `positional_features` | data, model | Number of extra node features appended to physical channels (centroid distance, mean edge length, then RWPE). |
| `use_node_types` | data, model | Adds one-hot node type features from feature index 7 when available. |
| `num_node_types` | model | Filled from the dataset when node types are enabled. |
| `latent_dim` | model | Processor hidden width. Config files may write `Latent_dim`; the loader lowercases it. |
| `message_passing_num` | flat model | Number of flat processor blocks when `use_multiscale False`. |

Node input size is:

```text
input_var + positional_features + optional num_node_types
```

Edge features are always:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

## Multiscale Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `use_multiscale` | data, model | Enables V-cycle processor and per-level graph data. |
| `multiscale_levels` | data, model | Number of coarsening levels. |
| `mp_per_level` | model | Required for multiscale. Must contain `2 * multiscale_levels + 1` integers: descending arm, coarsest, ascending arm. |
| `coarsening_type` | data | `bfs`, `voronoi_centroid`, `voronoi_inherit`, `voronoi_seedmean`, or a comma list per level. |
| `voronoi_clusters` | data | Required for Voronoi levels; scalar or comma list. |
| `hierarchy_cache_dir` | data | Optional directory for the shared on-disk hierarchy cache (defaults next to the dataset). |
| `hierarchy_cache_build_workers` | data | Optional worker count for the one-time cache build. |
| `hierarchy_cache_wait_timeout` | data | Seconds to wait for another job's cache build before failing. Default `36000`. |
| `static_cache_per_worker` | data | Per-worker LRU cap for positional features (non-multiscale runs only). Default `64`. |

When `use_multiscale True`, `message_passing_num` is not used by the processor;
the block counts come from `mp_per_level`. Unpooling always uses the learned
bipartite `UnpoolBlock`. Hierarchies are precomputed once into a shared on-disk
cache (`*.mscache.*.h5`) that all workers and jobs stream from.

## World Edge Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `use_world_edges` | data, model | Enables radius edges in addition to mesh edges. |
| `world_radius_multiplier` | data | Multiplies the minimum sampled mesh-edge length to get the radius. Default `1.5`. |
| `world_max_num_neighbors` | data, inference | Cap for `torch_cluster` radius graph. Default `64`. |
| `world_edge_backend` | data, inference | `torch_cluster`, `scipy_kdtree`, or `auto`. |
| `coarse_world_edges` | data, model | Lift world edges to all coarse V-cycle levels (default `False`). Requires `use_world_edges True` and `use_multiscale True`. See [WORLD_EDGES_DOCUMENTATION.md](WORLD_EDGES_DOCUMENTATION.md). |

World edges reuse the same 8-D edge feature layout and the mesh-edge
normalization statistics. Mesh edges are filtered out so the world-edge set stays
disjoint from the mesh topology. When `coarse_world_edges True`, coarse-level
world edges are normalized with the per-level coarse mesh-edge statistics.

## VAE And Prior Keys

The VAE branch exists to model manufacturing spread: `z` captures the
sample-to-sample variation in physical outputs for objects that share the
same mesh topology. For this objective, keep `lambda_mmd` low (≈ 0.1) and
`beta_aux` high (≈ 1.0). See
[MESHGRAPHNET_ARCHITECTURE.md](MESHGRAPHNET_ARCHITECTURE.md) for guidance.

| Key | Used by | Meaning |
| --- | --- | --- |
| `use_vae` | model, training, inference | Enables the graph VAE latent path. |
| `vae_latent_dim` | model | Global latent `z` dimension. 32 recommended for spread modeling. |
| `vae_mp_layers` | VAE encoder | Message-passing layers inside the posterior encoder. Default `5` in model construction. |
| `vae_graph_aware` | VAE encoder | Fuses graph input `x` with target `y` in the posterior encoder. Recommended `True` for multi-type datasets: enables type-conditional spread encoding. |
| `alpha_recon` | training | Reconstruction-loss weight. |
| `lambda_mmd` | training | MMD regularizer weight matching aggregate posterior to `N(0,I)`. Keep low (≈ 0.1) for spread modeling; residual MMD > 0 is acceptable and expected. |
| `beta_aux` | training | Weight for auxiliary latent decoder predicting per-graph output stats from `z`. Keep high (≈ 1.0) to prevent mode collapse. |
| `posterior_min_std` | VAE encoder | Floor on the posterior std. ≈ 0.05 recommended so the prior cannot memorize point masses. |
| `recon_loss` | training | `huber` (default) or `mse`. MSE preserves extreme-node amplitude better. |
| `num_vae_samples` | inference | Number of rollout samples per scene. Can exceed training set size to extrapolate spread. |
| `vae_batch_size` | inference | Trajectories advanced together per batched forward pass. |
| `vae_valid_prior_samples` | training | Prior samples per graph for the CRPS validation metric. Default `8`. |
| `num_z` | model, prior | Per-level z slots. Defaults to `multiscale_levels + 1` for multiscale, else `1`. |
| `use_conditional_prior` | inference | At inference, sample z from the joint-trained conditional prior (a model submodule). Default `True`; set `False` to force N(0,I). |
| `prior_type` | model, training | `gnn_e2e` (default when `use_vae True`) enables the joint graph-conditional prior submodule. |
| `prior_family` | prior | Density family of the gnn_e2e prior: `fm` (conditional flow matching, **default**) or `gmm` (Gaussian mixture, legacy). Persisted in the checkpoint; pre-FM checkpoints load as `gmm` automatically. |
| `prior_nll_weight` | training | Weight of the prior density-matching objective — flow-matching MSE (`fm`) or mixture NLL (`gmm`). Default `1.0`. |
| `prior_fm_steps` | prior (fm) | Euler ODE steps when sampling from the FM prior. Default `20`. |
| `prior_hidden_dim` | prior | Prior hidden width. Defaults to `latent_dim`. |
| `prior_mp_layers` | prior | Prior graph message-passing layers. Default `3`. |
| `prior_mixture_components` | prior (gmm) | Mixture components. Default `50`. |
| `prior_min_std` | prior (gmm) | Minimum predicted component std. Default `0.1`. |
| `prior_cov_rank` | prior (gmm) | Low-rank covariance rank per component. Default `0` (diagonal). |
| `prior_kl_reg_weight` | training (gmm) | Small analytical-KL stability anchor. Default `0.02`. Ignored for `fm` (no collapse mode to anchor). |
| `prior_temperature` | inference | `gmm`: divides mixture logits and scales sampled std by `sqrt(temperature)`. `fm`: scales the initial noise std by `sqrt(temperature)`. |

Removed legacy keys: `alpha_prior_max`, `alpha_prior_warmup_frac`,
`prior_loss_type`, `prior_gumbel_temp` (deleted with the FM prior introduction);
`lambda_det`, `free_bits`, `fit_latent_gmm`, `gmm_components`,
`gmm_covariance_type`, `gmm_reg_covar`, `train_conditional_prior`,
`bipartite_unpool`, `residual_scale`, `use_pairnorm`, `extreme_weight`,
`positional_encoding`, `fine_mp_pre`/`coarse_mp_num`/`fine_mp_post`
(deleted in the 2026-07 refactor — these keys are now ignored or rejected).

## Training And Performance Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `training_epochs` | training | Epoch count. Config files may write `Training_epochs`; the loader lowercases it. |
| `batch_size` | training | Physical DataLoader batch size. |
| `learningr` | training | Adam learning rate. Config files may write `LearningR`. |
| `warmup_epochs` | training | Linear warmup epochs before cosine restarts. Default `3`. |
| `num_workers` | DataLoader | Worker count for simulator training. |
| `std_noise` | model forward | Adds Gaussian noise to node and edge inputs during training. |
| `noise_gamma` | model forward | Target correction factor when input noise is applied. Default `0.1`. |
| `grad_accum_steps` | training | `1` steps per batch, `N` accumulates N batches, `0` accumulates the whole epoch. |
| `use_checkpointing` | model | Activation checkpointing in training. |
| `use_amp` | training | bfloat16 autocast. |
| `use_compile` | training | `torch.compile(dynamic=True)`. |
| `use_ema` | training | EMA shadow model for validation and inference. |
| `ema_decay` | training | EMA decay. |
| `test_interval` | training | Visualization/test output cadence. |
| `val_interval` | training | Validation cadence. |
| `test_batch_idx` | training | Test/visualization batch indices. |
| `display_testset` | visualization | Render test-set PNGs during `test_model`. Default `True`. |
| `display_trainset` | visualization | Also visualize train-set reconstructions at test cadence. Default `True`. |
| `plot_feature_idx` | visualization | Feature index to plot; `-1` selects the last feature. |
| `feature_loss_weights` | training | Optional per-output-channel loss weights, normalized to sum to 1. |
| `augment_geometry` | data | Training-only random z rotation and x/y reflection. |
| `test_max_batches` | training | Cap on test batches per visualization pass. Default `200`. |

Simulator loss is Huber or MSE (`recon_loss`) on normalized deltas. VAE training
adds MMD and auxiliary latent losses according to the configured weights. For
manufacturing spread modeling, set `lambda_mmd 0.1`, `beta_aux 1.0`.

## Inference Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `inference_output_dir` | inference | Output directory. Defaults to `outputs/rollout`. |
| `infer_timesteps` | inference | Rollout steps per selected scene. |
| `num_vae_samples` | inference | Number of stochastic samples per scene when VAE is enabled. |
| `prior_temperature` | inference | Conditional-prior sampling spread control. |
| `eval_dataset` | inference | Ground-truth eval HDF5 used for the inline z_disp spread-histogram comparison. When set (and VAE enabled), rollout writes `histogram_compare.png` next to the `.h5` outputs. No GT path → comparison skipped. |
| `make_histogram` | inference | Force the spread histogram on/off. Defaults to `True` when `use_vae` and `eval_dataset` are both set. |
| `show_histogram` | inference | Open the saved `histogram_compare.png` in the OS default viewer after rollout. Default `True`; best-effort (silently degrades on a headless box). |
| `histogram_bins` | inference | Histogram bin count (default `60`). |
| `histogram_clip_quantile` | inference | Symmetric quantile clip for the binning range, e.g. `0.001` trims the 0.1% tails (default `0` = no clip). |

Inference output files are named like:

```text
rollout_sample{sample_id}_steps{steps}.h5
rollout_sample{sample_id}_vaesample{idx}_steps{steps}.h5
```

When `eval_dataset` is set, rollout also writes `histogram_compare.png` in
`inference_output_dir` — the same z_disp spread (max − min, final timestep)
comparison produced by the standalone `_b8_all_warpage_input/compare_histograms.py`,
now generated automatically as part of the rollout.

## Checkpoint Contents

Training checkpoints contain:

- `model_state_dict`
- optional `ema_state_dict`
- optimizer and scheduler states
- `train_loss`, `valid_loss`
- `normalization` with node, edge, delta, optional world-edge radius, optional node-type mapping, and optional coarse-edge stats
- `model_config` with architecture-critical config values

The joint-trained conditional prior lives inside `model_state_dict` /
`ema_state_dict` (as the `prior.*` submodule). Legacy checkpoints with a
separately-saved `conditional_prior_state_dict` still load at inference.
