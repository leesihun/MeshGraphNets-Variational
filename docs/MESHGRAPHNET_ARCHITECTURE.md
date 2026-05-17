# MeshGraphNets Architecture

This document describes the architecture implemented in the current codebase.
The authoritative modules are:

- `model/MeshGraphNets.py`
- `model/encoder_decoder.py`
- `model/blocks.py`
- `model/mlp.py`
- `model/vae.py`
- `model/conditional_prior.py`
- `training_profiles/training_loop.py`
- `training_profiles/posthoc_prior.py`

## Runtime Entry Points

`MeshGraphNets_main.py` loads a text config and dispatches by `mode`:

| Mode | Runtime path |
| --- | --- |
| `train` | simulator training through `single_training.py` or DDP in `distributed_training.py` |
| `train_with_prior` | simulator training, then post-hoc conditional-prior training |
| `train_prior` | post-hoc conditional-prior training against an existing checkpoint |
| `inference` | rollout through `inference_profiles/rollout.py` |

With multiple `gpu_ids`, the default path is DDP data parallelism. Setting
`parallel_mode model_split` uses the experimental pipeline split launcher under
`parallelism/`.

## Graph Contract

Each training sample becomes a PyG `Data` or `MultiscaleData` object with:

| Field | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N, input_var + positional_features + optional node_types]` | normalized node input |
| `y` | `[N, output_var]` | normalized target delta |
| `pos` | `[N, 3]` | unnormalized reference position |
| `edge_index` | `[2, E]` | bidirectional mesh edges |
| `edge_attr` | `[E, 8]` | normalized mesh edge features |
| `world_edge_index` | `[2, E_world]` | optional radius edges, empty when disabled |
| `world_edge_attr` | `[E_world, 8]` | optional normalized world-edge features |
| `part_ids` | `[N]` | optional raw part numbers for visualization |

`edge_var` must be `8`. The code validates it against `EDGE_FEATURE_DIM` in
`general_modules/edge_features.py`.

Edge feature order:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

For single-timestep datasets, the node input physical channels are zeros and the
target is the final displacement/state. For multi-timestep datasets, the node
input is state `t` and the target is `state[t + 1] - state[t]`.

## MLP Building Block

`model/mlp.py::build_mlp` builds:

```text
Linear -> SiLU -> Linear -> SiLU -> Linear -> optional LayerNorm
```

LayerNorm is enabled for encoders, processor blocks, coarse edge encoders,
unpool blocks, VAE encoders, and the conditional prior. It is disabled for the
final simulator decoder and the VAE auxiliary decoder.

Weights are initialized with Kaiming uniform for `nn.Linear` and zero bias.

## Flat Encoder-Processor-Decoder

The top-level wrapper `MeshGraphNets` owns an `EncoderProcessorDecoder` as
`model`.

### Encoder

`Encoder` maps raw node and edge features into `latent_dim`:

- node encoder: node input size to `latent_dim`
- mesh edge encoder: 8-D edge features to `latent_dim`
- optional world edge encoder: 8-D world-edge features to `latent_dim`

### Processor

When `use_multiscale False`, the processor is a `ModuleList` of
`message_passing_num` `GnBlock` layers.

Each `GnBlock` does:

1. Mesh edge update from sender node, receiver node, and current mesh edge state.
2. Optional world edge update with the same edge-block structure.
3. Node update from current node state plus summed incoming mesh edge messages.
4. If world edges are enabled, node update also receives summed incoming
   world-edge messages through `HybridNodeBlock`.
5. Residual node and edge updates scaled by `residual_scale`.
6. Optional PairNorm on node features when `use_pairnorm True`.

Aggregation is `sum`, matching the NVIDIA PhysicsNeMo deforming-plate style in
the local comments.

### Decoder

`Decoder` maps latent node states to `output_var` normalized delta channels. The
decoder MLP has no final LayerNorm. For delta prediction (`num_timesteps` absent
or greater than 1), the last decoder layer is scaled by `0.01` at construction
time so the initial prior is close to no change.

## Multiscale V-Cycle

When `use_multiscale True`, the flat `message_passing_num` processor is replaced
by a V-cycle built in `EncoderProcessorDecoder._build_multiscale_processor`.

For `L = multiscale_levels`, `mp_per_level` must contain `2 * L + 1` integers:

```text
[pre_0, pre_1, ..., pre_(L-1), coarsest, post_(L-1), ..., post_1, post_0]
```

The default fallback for one level is:

```text
[fine_mp_pre, coarse_mp_num, fine_mp_post]
```

The forward pass is:

1. Encode the fine graph.
2. Run pre-blocks on each level.
3. Save a fine-level skip state.
4. Pool node states by `fine_to_coarse_i`.
5. Encode the corresponding coarse edge attributes.
6. Run coarsest blocks.
7. Unpool back up each level.
8. Merge skip state and unpooled state with `skip_projs[i]`, a linear projection
   from `2 * latent_dim` to `latent_dim`.
9. Run post-blocks and decode.

If `bipartite_unpool True`, unpooling is a learned `UnpoolBlock` over
`unpool_edge_index_i` using coarse state, fine skip state, and relative position.
Otherwise, coarse node states are broadcast to fine nodes by cluster assignment.

There is no global gated skip module in the current code. Skip merging is the
per-level linear `skip_projs` described above.

World-edge message passing only applies on the original fine level. Coarse-level
blocks are constructed with `use_world_edges False`.

## VAE Branch

When `use_vae True`, the model adds a graph variational encoder and injects a
global latent `z` into the simulator processor.

### Posterior Encoder

`GNNVariationalEncoder` encodes target delta `y` into a graph-level latent:

1. Encode `y` with an MLP.
2. If `vae_graph_aware True`, encode graph input `x` with a second MLP and fuse
   it with the encoded `y`.
3. Encode 8-D mesh edge attributes.
4. Run `vae_mp_layers` `GnBlock` layers.
5. Pool with `GlobalAttention`.
6. Predict `mu` and `logvar`.
7. Reparameterize to sample `z`.

The posterior regularizer used in training is MMD between sampled posterior `z`
and `N(0,I)`. Optional free-bits KL can be enabled with `free_bits > 0`.

### Latent Injection

During training, the simulator uses posterior `z`. During inference, it uses a
fixed `z`, conditional-prior sample, legacy GMM sample, or standard normal sample
depending on checkpoint contents and config.

Flat processor:

- one `z_fusers[i]` linear layer per processor block
- each fuser maps `[node_latent, z]` back to `latent_dim`

Multiscale processor:

- separate fusers for each pre arm, coarsest arm, and post arm
- pooled batch assignments are used to map graph-level `z` to coarse nodes

The auxiliary decoder predicts per-graph target mean and standard deviation from
`z`, and its loss is weighted by `beta_aux`.

## Conditional Prior

`model/conditional_prior.py::ConditionalMixturePrior` is a post-hoc prior network
for VAE checkpoints.

It uses the same node input accounting as the simulator:

```text
input_var + positional_features + optional num_node_types
```

Its edge input is also `edge_var`, which must be 8. The prior:

1. Encodes node and edge features.
2. Runs `prior_mp_layers` graph blocks.
3. Pools with `GlobalAttention`.
4. Predicts mixture logits, means, and log standard deviations for
   `prior_mixture_components` Gaussian components in latent-z space.

`training_profiles/posthoc_prior.py` freezes the simulator, samples target
posterior latents from the simulator VAE encoder, optimizes mixture negative
log-likelihood, and appends these checkpoint keys:

- `conditional_prior_state_dict`
- `conditional_prior_config`
- `conditional_prior_metrics`

At inference, `rollout.py` uses the conditional prior only when all of these are
true after checkpoint model_config overrides:

- `use_vae True`
- `use_conditional_prior True`
- checkpoint contains `conditional_prior_state_dict`

If that path is not active, VAE inference falls back to the legacy GMM artifact
when present; otherwise it samples `N(0,I)`.

Sampling temperature divides mixture logits and scales component standard
deviations by `sqrt(temperature)`.

## Training Objective

The simulator predicts normalized target deltas. Reconstruction loss is Huber
loss with `delta=1.0`, optionally weighted per output channel by
`feature_loss_weights` after normalizing those weights to sum to 1.

Without VAE:

```text
loss = reconstruction_loss
```

With VAE:

```text
loss = alpha_recon * reconstruction_loss
     + lambda_mmd * mmd_loss
     + beta_aux * auxiliary_loss
     + optional free_bits_kl
```

Other training behavior:

- Adam optimizer, fused when CUDA is available.
- Linear warmup followed by cosine warm restarts.
- bfloat16 autocast when `use_amp True`.
- gradient clipping with max norm `3.0`.
- optional EMA shadow model.
- optional activation checkpointing during training.
- optional `torch.compile(dynamic=True)`.

## Checkpoint And Inference Behavior

Training checkpoints store:

- model, optimizer, scheduler states
- optional EMA state
- train and validation losses
- train-split normalization
- architecture-critical `model_config`
- optional VAE prior diagnostics

Inference first loads normalization, then applies checkpoint `model_config` over
the runtime config. This is intentional for shape safety, but it means changing
architecture or prior keys only in an inference config may not take effect if
the checkpoint stored different values.

Rollout writes HDF5 files under `inference_output_dir` or `outputs/rollout` by
default. The saved nodal layout is:

```text
x, y, z, predicted output channels..., Part No.
```
