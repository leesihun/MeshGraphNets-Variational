# Latent Normalizing Flow for cVAE

## Problem: Why N(0,1) Sampling Gives Squashed Distributions

Our cVAE encoder maps each training sample's target delta into a posterior
distribution `q(z|x)` with parameters `(mu, logvar)`.  The KL loss penalises
the difference between each `q(z|x)` and the standard normal `N(0,I)`.

However, the **aggregate posterior** -- the distribution you get by averaging
over all training samples -- is:

```
q(z) = (1/N) * sum_i  q(z | x_i)
```

This is a **mixture of small Gaussians**, each centered at `mu_i` with
variance `exp(logvar_i)`.  It is NOT the same as `N(0,I)`:

- It may have lower effective variance in some dimensions
- Dimensions may be correlated
- The shape may be multi-modal or non-Gaussian

When we sample `z ~ N(0,I)` at inference, some samples land in **holes** --
regions of latent space that no training data ever mapped to.  The decoder has
no training signal in those regions and regresses toward the mean, producing
outputs with correct shape and center but **compressed variance** (squashing).

## Solution: Learn the True Latent Distribution

Instead of sampling from `N(0,I)`, we learn the exact shape of `q(z)` from
the training data and sample from *that*.

**Normalizing flows** do this by learning an invertible function `f` that
warps `N(0,I)` into `q(z)`:

```
u ~ N(0, I)        simple base distribution (8-dim standard normal)
z = f(u)           flow transforms u into the real latent distribution
                   -> feed z into VAE decoder as usual
```

Because `f` is bijective (invertible), we can:
1. **Sample**: draw `u ~ N(0,I)`, compute `z = f(u)` -- every z follows `q(z)`
2. **Evaluate density**: compute exact `log p(z)` via the change-of-variables formula
3. **Generate unlimited samples**: each `u` gives a valid `z`, no squashing

## Change of Variables Formula

If `z = f(u)` where `u ~ p_base(u)`, then:

```
log p(z) = log p_base(f^{-1}(z)) + log |det(df^{-1}/dz)|
```

The second term -- the **log-determinant of the Jacobian** -- accounts for how
`f` stretches or compresses probability density.  A regular MLP would require
`O(D^3)` to compute this determinant.  RealNVP makes it `O(D)`.

## RealNVP: How Coupling Layers Work

### Single Coupling Layer

Split the D-dimensional input into two groups using a binary mask:

```
Input:  x = [x_A, x_B]       (e.g., first 4 dims, last 4 dims)

x_A  ---------------------------------------->  z_A  (unchanged)
  |                                               
  +--> MLP_s --> s (scale)                        
  |                  \                            
x_B  -----------------* exp(s) + t ----------->  z_B  (transformed)
  |                  /                            
  +--> MLP_t --> t (shift)                        

Output: z = [z_A, z_B]
```

- `x_A` passes through **unchanged** (identity)
- `x_A` is fed into two small MLPs producing scale `s` and shift `t`
- `x_B` is transformed: `z_B = x_B * exp(s(x_A)) + t(x_A)`

### Why This is Invertible

To invert, just rearrange -- no matrix inversion needed:

```
z_A = x_A                          (identity)
x_B = (z_B - t(z_A)) * exp(-s(z_A))   (algebraic rearrangement)
```

### Why the Jacobian is Cheap

Because `x_A` passes through unchanged, the Jacobian matrix is triangular.
The determinant of a triangular matrix is just the product of its diagonal:

```
log |det(J)| = sum(s(x_A))
```

Just the sum of the scale network outputs -- `O(D)`, not `O(D^3)`.

### Stacking Layers with Alternating Masks

One coupling layer only transforms half the dimensions.  By **alternating
which half is fixed**, every dimension gets transformed:

```
Layer 1:  mask = [1,1,1,1,0,0,0,0]   fix first 4, transform last 4
Layer 2:  mask = [0,0,0,0,1,1,1,1]   fix last 4, transform first 4
Layer 3:  mask = [1,1,1,1,0,0,0,0]   fix first 4, transform last 4
Layer 4:  mask = [0,0,0,0,1,1,1,1]   fix last 4, transform first 4
Layer 5:  mask = [1,1,1,1,0,0,0,0]   ...
Layer 6:  mask = [0,0,0,0,1,1,1,1]   ...
```

After 6 layers, every dimension has been transformed conditioned on every
other dimension multiple times.  The total log-det is simply the sum of all
individual layer log-dets.

## Training Objective

We train the flow by **maximum likelihood** -- minimize the negative
log-probability of the observed mu vectors under the flow model:

```
Loss = -E_z [ log p(z) ]
     = -E_z [ log p_base(f^{-1}(z)) + log |det(J^{-1})| ]
```

Where `z` are the collected mu vectors from the trained VAE encoder.

This requires only the **inverse** direction (`z -> u`, data to base) during
training. The **forward** direction (`u -> z`, base to data) is used at
inference for sampling.

## Architecture for vae_latent_dim = 8

```
Latent dimension:     8
Coupling layers:      6  (3 full passes through alternating masks)
Scale/shift MLPs:     8 -> 64 -> 64 -> 8  (SiLU activations)
Total parameters:     ~36K
Training time:        10-30 seconds
Training data:        mu vectors from all training samples
```

Each coupling layer has two MLPs (scale_net and shift_net), each with two
hidden layers of width 64.  The flow also stores `data_mean` and `data_std`
as registered buffers for normalizing the mu vectors before processing.

## Pipeline

```
TRAINING (post-hoc, after main training completes):

  1. Load best checkpoint (prefer EMA weights)
  2. Run VAE encoder over all training data -> collect mu vectors [N, 8]
  3. Train RealNVP flow to maximize log p(mu) 
  4. Save flow state_dict into the checkpoint

INFERENCE (automatic when flow is in checkpoint):

  u ~ N(0, I)  in R^8     (standard normal, same as before)
       |
  z = flow(u)  in R^8     (now follows the true training distribution)
       |
  decoder(z) -> mesh      (existing VAE decoder, unchanged)
```

## Config

Add to training config to enable:

```
train_latent_flow   True
```

No inference config needed -- if the flow is saved in the checkpoint, it is
loaded and used automatically.  Old checkpoints without a flow fall back to
raw `N(0,I)` sampling silently.

## Implementation

All code lives in `model/latent_flow.py`:

| Component | Description |
|---|---|
| `CouplingLayer` | Single affine coupling layer (forward + inverse) |
| `LatentRealNVP` | Full flow model (stack of coupling layers) |
| `collect_vae_mus()` | Collect mu vectors from trained VAE encoder |
| `train_latent_flow()` | Train the flow on collected mus |
| `run_posthoc_flow_training()` | Top-level orchestrator for training profiles |
| `load_latent_flow()` | Load flow from checkpoint for inference |
