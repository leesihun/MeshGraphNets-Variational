# VAE Implementation Guide — MeshGraphNets-Variational

## Purpose

This VAE models **manufacturing variability**: the same nominal design produces different physical outcomes due to unknown process variables (material batch, temperatures, tolerances, etc.). `z` encodes the hidden manufacturing instance — at inference, sampling different `z` gives a distribution of plausible outcomes for the same design.

## Architecture Overview

```
                    ┌──────────────────────────────────────────────────────────┐
                    │              GNN VAE Encoder  (train only)               │
                    │  target_delta [N, output_var]                            │
                    │    → node_encoder MLP  → [N, latent_dim]                │
                    │    → edge_encoder MLP (raw 8D edges) → [E, latent_dim]  │
                    │    → vae_mp_layers GnBlocks (mesh message passing)       │
                    │      (nodes now see spatially correlated patterns)       │
                    │    → GlobalAttention pool → [B, latent_dim]             │
                    │    → mu_head, logvar_head → z [B, vae_latent_dim]       │
                    │      Inference: z ~ N(0, I)                              │
                    └────────────────────────┬─────────────────────────────────┘
                                             │ z [B, vae_latent_dim]
                                             │ broadcast → z_per_node [N, vae_latent_dim]
                                             │
   Input geometry ──→ Encoder ──────────────────────────────────────────────────────────→ (skip gate)
                          ↓                  │                                                 ↓
                    encoded_x [N, D]         ↓ inject at every GnBlock                  Decoder → Δ pred
                          ↓       ┌─── z_proj_i(cat[x, z_per_node]) → GnBlock_i ───┐
                          ↓       │  (repeated message_passing_num times)           │
                          └───────┴─────────────────────────────────────────────────┘

                    ┌──────────────────────────────┐
                    │    Auxiliary Decoder (train) │
                    │  z → MLP → (μ_y, σ_y)       │
                    │  per output feature, per graph│
                    │  MSE loss vs. actual stats   │
                    └──────────────────────────────┘
```

## Loss Function

```
Loss = α · Huber(x, x̂)          reconstruction
     + β_eff · KL(q||p)           KL divergence
     + β_aux · MSE(aux_pred, aux_target)   auxiliary: z → global stats
```

- `β_eff = β_kl · min(1, epoch / kl_anneal_epochs)` with KL annealing (optional)
- `aux_target`: per-graph mean and std of target delta per output feature `[B, 2·output_var]`
- Auxiliary loss only active during training (not eval/inference)

## Why This Architecture Works for Manufacturing Spread

| Problem (old design) | Solution (new design) |
|---|---|
| Per-node MLP → mean pool destroys spatial patterns | GNN message passing before pool: nodes see neighbors, encoder captures spatially correlated deformation patterns |
| z injected once → washed out by 15+ residual MP layers | z injected at **every** GnBlock via `cat([x, z_per_node])` → processor cannot forget z |
| No gradient forcing z to be informative | Auxiliary decoder: z must directly predict global delta stats, independent of the processor |
| Global mean treats all nodes equally | GlobalAttention pool focuses on high-variation nodes (stress concentrations, etc.) |

## Config Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_vae` | `False` | Enable variational encoder |
| `vae_latent_dim` | `32` | Dimension of global latent z |
| `vae_mp_layers` | `2` | GnBlock layers in the VAE encoder (before attention pool) |
| `beta_kl` | `0.001` | KL divergence weight |
| `beta_aux` | `0.1` | Auxiliary decoder loss weight |
| `alpha_recon` | `1.0` | Reconstruction loss weight |
| `kl_anneal_epochs` | `0` | Linearly ramp beta_kl from 0 (0 = disabled) |

## Key Files Modified

### model/MeshGraphNets.py
- **`GNNVariationalEncoder`** (replaces `VariationalEncoder`): GnBlock MP + GlobalAttention pool → mu/logvar
- **`EncoderProcessorDecoder.__init__`**: `z_projs` ModuleList (one Linear per GnBlock), `aux_decoder` MLP
- **`_encode_vae()`** (replaces `_vae_condition()`): returns `(z [B, D], kl)` — no longer injects directly
- **`_aux_loss()`**: computes auxiliary MSE loss from z → per-graph delta stats
- **`forward()` flat path**: saves `original_edge_attr/index`; per-layer z injection in processor loop; computes aux_loss; returns `(output, kl, aux_loss)`
- **`forward()` multiscale path**: same at fine level (level 0 pre/post blocks)
- **`MeshGraphNets.forward()`**: returns `(predicted, target, kl, aux_loss)`

### model/checkpointing.py
- `process_with_checkpointing()` accepts optional `z_projs` and `z_per_node` for VAE z injection during checkpointed forward pass

### training_profiles/training_loop.py
- Unpacks 4 return values: `predicted, target, kl, aux_loss`
- `beta_aux` config param; loss = `α·recon + β_eff·kl + β_aux·aux`
- Progress bar shows `rec`, `kl`, `aux`, `total`
- `train_epoch` result dict includes `kl_mean` and `aux_mean`

### training_profiles/single_training.py
- Epoch summary prints `aux` alongside `kl`
- Checkpoint `model_config` includes `vae_mp_layers`, `beta_aux`

## Training Flow

1. Graph arrives with `graph.y` = normalized target delta (manufacturing-specific deformation).
2. `original_y`, `original_edge_attr`, `original_edge_index`, `original_batch` saved before main encoder.
3. Main encoder encodes input features → `encoded_x`.
4. **VAE encoder**: runs `vae_mp_layers` GnBlocks on `(original_y, edge_index, edge_attr)` → GlobalAttention pool → `μ, logσ²` → sample `z [B, vae_latent_dim]`.
5. **Auxiliary loss**: `z → aux_decoder → predicted (μ_y, σ_y)`; MSE vs actual per-graph stats.
6. **Processor**: at each GnBlock, `z_projs[i](cat([graph.x, z_per_node]))` modifies node features before message passing.
7. Gated skip blends `encoded_x` back. Decoder produces predicted delta.
8. Total loss = `α·Huber + β_eff·KL + β_aux·aux_MSE`.

## Inference / Rollout

- `model.eval()` → `_encode_vae` samples `z ~ N(0, I)` (no ground-truth needed).
- Different `z` samples produce different plausible deformations for the same input geometry → manufacturing spread estimation.
- `rollout.py` unpacks `predicted_delta_norm, _, _, _ = model(graph)` — no changes needed.

## Compatibility Matrix

| Feature | Compatible | Notes |
|---|---|---|
| Flat GNN (`use_multiscale=False`) | Yes | z injection in `processer_list` loop |
| Multiscale V-cycle (`use_multiscale=True`) | Yes | z injection at fine level (level 0) pre/post blocks |
| Gradient checkpointing (`use_checkpointing=True`) | Yes | `process_with_checkpointing` accepts z_projs |
| Mixed precision (`use_amp=True`) | Yes | All ops support bfloat16 |
| EMA (`use_ema=True`) | Yes | VAE params included in EMA |
| DDP (multi-GPU) | Yes | No unused parameters when `use_vae=True` |
| World edges | Yes | World edges unaffected |
| `use_vae=False` | Yes | Zero overhead — no VAE modules created, returns `kl=0, aux=0` |
