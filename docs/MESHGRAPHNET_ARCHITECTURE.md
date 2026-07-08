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
5. Residual node and edge updates (unscaled: `x + delta`).

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

Unpooling is always a learned `UnpoolBlock` over
`unpool_edge_index_i` using coarse state, fine skip state, and relative position.
Otherwise, coarse node states are broadcast to fine nodes by cluster assignment.

There is no global gated skip module in the current code. Skip merging is the
per-level linear `skip_projs` described above.

World-edge message passing only applies on the original fine level. Coarse-level
blocks are constructed with `use_world_edges False`.

## VAE Branch

The VAE branch is the core mechanism for spread modeling. The latent `z` captures
the part of the output that varies across manufactured samples (the "spread") while
the graph processor captures the part that is determined by geometry and boundary
conditions. During inference, sampling different `z` values from the prior
`p(z|graph)` produces distinct but physically plausible output variants for the
same input mesh.

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
and `N(0,I)`.

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

`model/conditional_prior.py` provides the mesh-conditioned prior `p(z|graph)`,
trained **jointly** with the simulator when `use_vae True` and
`prior_type gnn_e2e`. It lives as the `prior` submodule of `MeshGraphNets`, so
DDP and EMA wrap it together with the simulator and it is saved inside the
normal `model_state_dict`.

A shared graph trunk uses the same node input accounting as the simulator:

```text
input_var + positional_features + optional num_node_types
```

Its edge input is also `edge_var`, which must be 8. The trunk encodes node and
edge features, runs `prior_mp_layers` graph blocks, and pools with attentional
aggregation into one conditioning vector per graph. `prior_family` selects the
density head:

- `fm` (default): a conditional flow-matching velocity field. Training is MSE
  regression on straight-line interpolation paths toward fresh detached
  posterior samples; sampling integrates an Euler ODE for `prior_fm_steps`
  steps. Temperature scales the initial noise std by `sqrt(temperature)`.
- `gmm`: mixture logits, means, and log-stds for `prior_mixture_components`
  Gaussian components (optionally low-rank covariance via `prior_cov_rank`).
  Training is mixture NLL plus a small `prior_kl_reg_weight` analytical-KL
  anchor. Temperature divides logits and scales stds by `sqrt(temperature)`.

At inference, `rollout.py` samples from the conditional prior when
`use_vae True` and `use_conditional_prior True` hold after checkpoint
model_config overrides, and the checkpoint carries a prior (the joint-trained
submodule, or a legacy separately-saved `conditional_prior_state_dict`).
Otherwise it samples `N(0,I)`.

## Training Objective

The model's goal is **manufacturing spread modeling**: given multiple manufactured
objects that share mesh topology but differ in physical outputs due to production
variability, the VAE must learn the spread of that output distribution and enable
generation of realistic new samples at inference time (potentially more samples
than the training set).

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
     + lambda_mmd * mmd_loss          # aggregate posterior ↔ N(0,I); keep LOW (≈0.1)
     + beta_aux * auxiliary_loss      # z → graph output stats anchor; keep HIGH (≈1.0)
```

**Spread modeling guidance for loss weights:**

- `lambda_mmd` should remain low (≈ 0.1). A non-zero residual MMD is acceptable
  and expected: the true aggregate posterior encodes real spread structure that does
  not match an isotropic Gaussian. Forcing MMD → 0 erases that structure.
- `beta_aux` should remain high (≈ 1.0). The auxiliary decoder forces `z` to
  predict per-graph output mean and standard deviation. Without this anchor the
  encoder collapses all spread into a small z subspace (mode collapse).
- The deterministic z=0 auxiliary pass (`lambda_det`) was removed from the code:
  it conflicts directly with the spread objective by punishing z for carrying
  information the graph cannot predict alone. Do not reintroduce it.

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
