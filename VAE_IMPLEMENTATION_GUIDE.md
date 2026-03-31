# VAE Implementation Guide — MeshGraphNets-Variational

## Architecture Overview

```
Time step = t                            Train only, Inference: z~N(0,1)
┌─────────────┐                         ┌──────────────────────────────────┐
│ Δ Real Node │─→ E ─→ z_Realnode ─→ Mean Pool ─→ MLP ─→ μ, log(σ)      │
│ (t+1 target)│                                           z = μ + σ × ε   │
└─────────────┘                         └────────────┬─────────────────────┘
                                                     │
┌─────────────┐                                      ▼
│  CAE Node   │─→ E ─→ z_CAEnode ──────────────→ z_concat ─┐
│ (input)     │                                              │
└─────────────┘                                              │
                                                             ▼
┌─────────────┐                                   ┌──────────────────┐
│    Edge     │─→ E ─→ z_edge ─────────────────→  │ Message Passing  │
│ (geometry)  │                                   │ Processor x15~20 │
└─────────────┘                                   │ (EdgeBlock +     │
                                                  │  NodeBlock)      │
┌─────────────┐                                   └────────┬─────────┘
│   World     │─→ E ─→ z_world ───────────────────────────→│
│ (contact)   │                                             │
└─────────────┘                                             ▼
                                                    ┌───────────────┐
                                                    │    Decoder    │
                                                    │   Δ Real Node │
                                                    │   (t+1 pred) │
                                                    └───────────────┘
```

## Loss Function

```
Loss = α · Huber(x, x̂) + β_eff · (-1/2) mean(1 + log(σ²) - μ² - e^{log(σ²)})
       ├── reconstruction ─┤   ├── KL divergence ─────────────────────────────┤
```

- `α` = `alpha_recon` (default 1.0) — reconstruction loss weight
- `β` = `beta_kl` (default 0.001) — KL divergence weight
- Reconstruction uses Huber loss (delta=1.0) with optional per-feature weights
- Optional KL annealing: `β_eff = β · min(1, epoch / kl_anneal_epochs)` linearly ramps from 0

## Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_vae` | `False` | Enable variational encoder |
| `vae_latent_dim` | `32` | Dimension of global latent z |
| `beta_kl` | `0.001` | KL divergence weight |
| `alpha_recon` | `1.0` | Reconstruction loss weight |
| `kl_anneal_epochs` | `0` | Linearly ramp beta from 0 (0 = disabled) |

## Files Modified

### model/MeshGraphNets.py
- **Added `from torch_geometric.nn import global_mean_pool`** import.

- **Added `VariationalEncoder` class** (after `build_mlp`): Encodes ground-truth target delta (`graph.y`) into a global stochastic latent `z` via:
  1. MLP encoder on per-node deltas → `[N, latent_dim]`
  2. `global_mean_pool` → `[B, latent_dim]`
  3. Linear heads → `μ [B, vae_latent_dim]`, `log(σ²) [B, vae_latent_dim]`
  4. Reparameterize: `z = μ + exp(0.5·log(σ²)) · ε`, where `ε ~ N(0,I)`
  5. Static method `kl_loss(μ, log(σ²))` computes KL(q||p)

- **Added `_vae_condition()` method** on `EncoderProcessorDecoder`:
  - Training: encodes `original_y` via `vae_encoder`, returns `(x_conditioned, kl)`
  - Inference (`model.eval()` or `original_y is None`): samples `z ~ N(0, I)`, returns `(x_conditioned, 0.0)`
  - Broadcasts `z` per-node via batch index, concatenates with encoded node features, projects back to `latent_dim`
  - Handles missing `batch` index (single-graph manual construction) by defaulting to `torch.zeros`

- **Added VAE components in `EncoderProcessorDecoder.__init__`** (conditional on `use_vae`):
  - `vae_encoder = VariationalEncoder(node_output_size, latent_dim, vae_latent_dim)`
  - `condition_proj = Linear(latent_dim + vae_latent_dim, latent_dim)`
  - When `use_vae=False`: no VAE modules are created, zero memory overhead.

- **Modified `EncoderProcessorDecoder.forward()`** (both flat and multiscale paths):
  - Saves `original_y` and `original_batch` via `getattr` before encoder (encoder creates new `Data`, dropping `.y` and `.batch`)
  - `encoder_x` is saved **before** VAE conditioning (preserves clean encoder output for gated skip connection, matching the architecture diagram where skip is from z_CAEnode, not z_concat)
  - Applies `_vae_condition()` after encoder, before processor
  - Returns `(output, kl)` instead of just `output`

- **Modified `MeshGraphNets.forward()`**: returns `(predicted, target, kl)` — always 3 values.

### training_profiles/training_loop.py
- **`train_epoch()`**:
  - Reads VAE config: `use_vae`, `alpha_recon`, `beta_kl`, `kl_anneal_epochs`.
  - Unpacks 3 return values: `predicted_acc, target_acc, kl_loss = model(graph, debug=...)`.
  - Computes combined loss with KL annealing: `loss = α·recon + β_eff·kl` where `β_eff = β·min(1, epoch/kl_anneal)`.
  - Progress bar when VAE active: `loss` (total combined), `rec` (recon only), `kl` (KL divergence).
  - Returns `kl_mean` in result dict when VAE is active.
- **`validate_epoch()`**: Unpacks 3 return values `(predicted, target, _)`, ignores KL.
- **`test_model()`**: Same 3-value unpack.
- **`log_training_config()`**: Prints VAE status (enabled/disabled with z_dim, beta_kl, alpha_recon, kl_anneal).

### training_profiles/single_training.py
- Saves `use_vae` and `vae_latent_dim` in `model_config` dict within checkpoints.
- Prints VAE status at model initialization (z_dim, beta_kl).
- Epoch summary line includes KL when VAE active.
- Log file includes KL when VAE active.

### training_profiles/distributed_training.py
- Same checkpoint `model_config` additions as single_training.
- Prints VAE status at model initialization (rank 0 only).
- Epoch summary line includes KL when VAE active.
- Log file includes KL when VAE active.

### inference_profiles/rollout.py
- Unpacks 3 return values: `predicted_delta_norm, _, _ = model(graph)`.
- During `model.eval()`, `_vae_condition` automatically samples `z ~ N(0, I)` — no ground-truth needed.

### Config files (_warpage_input/config_train{3,4,5}.txt)
- Added VAE parameter section at the end of each config file (disabled by default).

## How It Works

### Training
1. Model receives graph with `graph.y` = normalized target delta (after optional noise correction).
2. `EncoderProcessorDecoder.forward()` saves `original_y = graph.y` and `original_batch = graph.batch` before encoding.
3. Standard node/edge encoder runs, producing `z_CAEnode` and `z_edge`. `encoder_x` is saved here for the gated skip connection.
4. VAE branch (`_vae_condition`):
   - Encodes `original_y` through `VariationalEncoder`: MLP → `global_mean_pool` → `μ, log(σ²)` → reparameterize `z`.
   - Broadcasts `z` to all nodes via PyG batch index.
   - Concatenates `[z_CAEnode, z_broadcast]` → `condition_proj` → back to `latent_dim`.
5. Processor (message passing) runs on the z-conditioned node features.
6. Gated skip connection blends `encoder_x` (pre-VAE) back into processed output.
7. Decoder produces predicted delta.
8. Loss = `α · Huber_recon + β_eff · KL_divergence`.

### Inference / Rollout
1. `model.eval()` → no ground-truth target available → `_vae_condition` samples `z ~ N(0, I)`.
2. Everything else proceeds identically.
3. The model has learned to decode from prior samples during training via the KL constraint.

### Noise Interaction
When `std_noise > 0`, noise is added to `graph.x` and `graph.y` is corrected in `MeshGraphNets.forward()` **before** `EncoderProcessorDecoder` sees it. The VAE therefore encodes the noise-corrected target, which matches the target the model is trained to predict. This is correct — if the VAE encoded the uncorrected delta but the model predicted the corrected one, there would be a mismatch.

## Design Decisions

- **VAE encodes target delta, not input**: Following the architecture diagram, the VAE branch takes the ground-truth delta (what the model is trying to predict) and compresses it. This is a conditional VAE — the latent captures "how" the deformation happens.
- **z concatenated with z_CAEnode (not encoder_x skip)**: The latent is injected after the standard encoder but before message passing. The gated skip connection uses the pre-VAE `encoder_x`, so the skip path carries only deterministic features. This matches the diagram: skip is from z_CAEnode, VAE conditioning enters the processor path only.
- **Backward compatible**: When `use_vae=False` (default), no VAE modules are created, no code paths change, no extra computation occurs. The 3rd return value is always `kl=0.0`.
- **Works with multiscale**: VAE conditioning is applied at the fine level after encoding, before the V-cycle begins.
- **DDP compatible**: When `use_vae=True`, all VAE parameters (`vae_encoder`, `condition_proj`) participate in every forward pass during training, so `find_unused_parameters=False` is safe.
- **AMP compatible**: VAE components use standard Linear layers and `global_mean_pool` — all work with bfloat16 autocast. `torch.randn_like` inherits the dtype.
- **EMA compatible**: EMA is built from the raw model (before compile/DDP). VAE parameters are part of the model and get EMA-averaged automatically.
- **Null batch handling**: `_vae_condition` handles `original_batch=None` (single graph without PyG DataLoader) by defaulting to `torch.zeros(N, dtype=long)`.

## Compatibility Matrix

| Feature | Compatible | Notes |
|---|---|---|
| Flat GNN (`use_multiscale=False`) | Yes | VAE conditioning after encoder, before processor |
| Multiscale V-cycle (`use_multiscale=True`) | Yes | VAE conditioning at fine level before V-cycle |
| World edges (`use_world_edges=True`) | Yes | World edges unaffected by VAE |
| Gradient checkpointing (`use_checkpointing=True`) | Yes | Checkpointing in processor; VAE runs before it |
| Mixed precision (`use_amp=True`) | Yes | All VAE ops support bfloat16 |
| EMA (`use_ema=True`) | Yes | VAE params included in EMA averaging |
| DDP (multi-GPU) | Yes | No unused parameters when `use_vae=True` |
| `torch.compile` | Yes | Standard ops, no graph breaks |
| Noise injection (`std_noise > 0`) | Yes | VAE encodes noise-corrected target (correct) |
| Node types (`use_node_types=True`) | Yes | Node types are in input encoder, orthogonal to VAE |
