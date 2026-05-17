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
| `mode` | launcher | `train`, `inference`, `train_prior`, or `train_with_prior`. |
| `gpu_ids` | launcher | `-1` for CPU, one GPU id for single process, or comma list for multi-GPU. |
| `parallel_mode` | launcher | Optional. `ddp` by default; `model_split` enables the experimental pipeline split path. |
| `modelpath` | training, inference, prior | Checkpoint path to save to or load from. |
| `log_file_dir` | training | Relative path under `outputs/` for epoch logs. |

Mode behavior:

- `train`: trains the simulator. If `use_vae True` and `fit_latent_gmm True`, the legacy GMM prior is fit after simulator training.
- `train_prior`: freezes an existing VAE simulator checkpoint and trains a mesh-conditioned conditional prior into that checkpoint.
- `train_with_prior`: runs simulator training, then sets `train_conditional_prior True` so post-hoc conditional prior training runs before exit.
- `inference`: runs autoregressive rollout using the checkpoint and writes rollout HDF5 files.

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
| `positional_features` | data, model | Number of extra node features appended to physical channels. |
| `positional_encoding` | data | `rwpe` default, `lpe`, or `rwpe+lpe` for features after centroid distance and mean edge length. |
| `use_node_types` | data, model | Adds one-hot node type features from feature index 7 when available. |
| `num_node_types` | model | Filled from the dataset when node types are enabled. |
| `latent_dim` | model | Processor hidden width. Config files may write `Latent_dim`; the loader lowercases it. |
| `message_passing_num` | flat model | Number of flat processor blocks when `use_multiscale False`. |
| `residual_scale` | blocks | Multiplier on residual node and edge updates. |
| `use_pairnorm` | blocks | Optional PairNorm after node residual update. |

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
| `mp_per_level` | model | Must contain `2 * multiscale_levels + 1` integers: descending arm, coarsest, ascending arm. |
| `fine_mp_pre` | model | Fallback pre-block count when `mp_per_level` is absent. |
| `coarse_mp_num` | model | Fallback coarsest block count when `mp_per_level` is absent. |
| `fine_mp_post` | model | Fallback post-block count when `mp_per_level` is absent. |
| `coarsening_type` | data | `bfs`, `voronoi`, or a comma list per level. |
| `voronoi_clusters` | data | Required for Voronoi levels; scalar or comma list. |
| `bipartite_unpool` | data, model | `True` uses learned coarse-to-fine bipartite message passing; `False` broadcasts by cluster id. |
| `coarse_cache_per_worker` | data | Per-worker LRU cache size for hierarchy data. |

When `use_multiscale True`, `message_passing_num` is not used by the processor;
the block counts come from `mp_per_level` or the fallback triplet.

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

| Key | Used by | Meaning |
| --- | --- | --- |
| `use_vae` | model, training, inference | Enables the graph VAE latent path. |
| `vae_latent_dim` | model | Global latent `z` dimension. |
| `vae_mp_layers` | VAE encoder | Message-passing layers inside the posterior encoder. Default `5` in model construction. |
| `vae_graph_aware` | VAE encoder | If true, fuses graph input `x` with target delta `y` in the posterior encoder. |
| `alpha_recon` | training | Reconstruction-loss weight. |
| `lambda_mmd` | training | MMD regularizer weight matching aggregate posterior to `N(0,I)`. |
| `beta_aux` | training | Weight for auxiliary latent decoder loss. |
| `free_bits` | training | Optional per-dimension KL floor. `0.0` disables it. |
| `num_vae_samples` | inference | Number of rollout samples per scene. |
| `fit_latent_gmm` | training | Legacy post-hoc GMM fitting after VAE simulator training. |
| `gmm_components` | GMM | Number of legacy GMM components. |
| `gmm_covariance_type` | GMM | Sklearn covariance type, for example `full`. |
| `gmm_reg_covar` | GMM | Optional GMM regularization. |
| `train_conditional_prior` | training | Internal flag used by `train_with_prior`; can also be set directly. |
| `use_conditional_prior` | inference, prior | Enables conditional-prior construction when a checkpoint contains prior weights. |
| `prior_mixture_components` | prior | Mixture components predicted by the conditional prior. Default `10`. |
| `prior_hidden_dim` | prior | Prior hidden width. Defaults to `latent_dim`. |
| `prior_mp_layers` | prior | Prior graph message-passing layers. Default `3`. |
| `prior_min_std` | prior | Minimum predicted latent standard deviation. Default `1e-3`. |
| `prior_epochs` | prior | Post-hoc prior training epochs. Default `200`. |
| `prior_learningr` | prior | Prior optimizer LR. Defaults to `learningr`. |
| `prior_batch_size` | prior | Prior batch size. Defaults to `batch_size`. |
| `prior_num_workers` | prior | Prior DataLoader workers. Default `0`. |
| `prior_val_interval` | prior | Prior validation interval. Default `10`. |
| `prior_mc_samples` | prior | Number of posterior `z` samples used as prior-training targets. Default `4`. |
| `prior_temperature` | inference | Divides mixture logits and scales sampled std by `sqrt(temperature)`. |
| `resume_prior` | prior | Defaults to `True`; resumes prior weights already stored in checkpoint. |

Current `_b8_all_warpage_input/config_train*.txt` files say "Train VAE-MGN, then
train post-hoc conditional prior" in comments, but their live keys are `mode train`
and `fit_latent_gmm True`. That launch trains the simulator and fits the legacy
GMM. To train the conditional prior, use `mode train_with_prior` for the simulator
run or run `mode train_prior` against the finished checkpoint.

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
| `plot_feature_idx` | visualization | Feature index to plot; `-1` selects the last feature. |
| `feature_loss_weights` | training | Optional per-output-channel Huber weights, normalized to sum to 1. |
| `augment_geometry` | data | Training-only random z rotation and x/y reflection. |

Simulator loss is Huber loss on normalized deltas. VAE training adds MMD and
auxiliary latent losses according to the configured weights.

## Inference Keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `inference_output_dir` | inference | Output directory. Defaults to `outputs/rollout`. |
| `infer_timesteps` | inference | Rollout steps per selected scene. |
| `num_vae_samples` | inference | Number of stochastic samples per scene when VAE is enabled. |
| `prior_temperature` | inference | Conditional-prior sampling spread control. |

Inference output files are named like:

```text
rollout_sample{sample_id}_steps{steps}.h5
rollout_sample{sample_id}_vaesample{idx}_steps{steps}.h5
```

## Checkpoint Contents

Training checkpoints contain:

- `model_state_dict`
- optional `ema_state_dict`
- optimizer and scheduler states
- `train_loss`, `valid_loss`
- `normalization` with node, edge, delta, optional world-edge radius, optional node-type mapping, and optional coarse-edge stats
- `model_config` with architecture-critical config values
- optional VAE prior diagnostics

Conditional-prior training appends:

- `conditional_prior_state_dict`
- `conditional_prior_config`
- `conditional_prior_metrics`

Legacy GMM fitting appends a GMM artifact consumed by `model/latent_gmm.py`.
