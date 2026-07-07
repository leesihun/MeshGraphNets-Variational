# Distribution Modeling Research: Hi-MGN-V vs. VGAE vs. LDGN

> Research notes, 2026-07-07. Compares the current Hi-MGN-V architecture against the
> VGAE baseline and Latent Diffusion Graph Net (LDGN) from *Learning Distributions of
> Complex Fluid Simulations with Diffusion Graph Networks* (Lino et al., ICLR 2025 oral,
> [arXiv:2504.02843](https://arxiv.org/abs/2504.02843)), including exact hyperparameters
> extracted from the official code ([tum-pbs/dgn4cfd](https://github.com/tum-pbs/dgn4cfd)).
> Historical-doc caveat applies (see CLAUDE.md): verify against live code before acting.

## 1. Problem framing

All three models target the same object: a conditional generative model
`p(output field | mesh, conditions)` that produces *many plausible samples*, not one
mean answer. Every design splits this into two roles:

1. **Compressor** (VAE encoder/decoder): squeeze a full mesh field into a small latent
   code and unpack it back.
2. **Sampler** (prior): at inference, invent realistic new codes without seeing a target.

The generated distribution is only as good as the sampler — at inference it is the sole
source of randomness. The central empirical finding of the paper is that **which
component carries the distribution-matching burden decides success or collapse.**

## 2. What the paper's dataset actually is

Not manufacturing data. The "distribution" is the **temporal (ergodic) distribution of
unsteady flow at statistical equilibrium**, per system:

| Task | Nodes | Physics | Train states/system | Eval states |
|---|---|---|---|---|
| Ellipse | ~70 (surface) | Laminar vortex shedding, Re 500–1000 | 10 consecutive (26–48% of one period) | 60–100 |
| EllipseFlow | ~2,300 | Velocity+pressure fields around ellipse | 10 consecutive | 60–100 |
| Wing | ~7,000 (surface) | 3D DES turbulence (OpenFOAM PISO); wings vary thickness/taper/sweep/twist | 250 consecutive (~10% of variance-convergence window) | 2,500 |

Mapping to Hi-MGN-V's manufacturing-spread problem:

- **Same mathematical object**: conditional distribution over mesh fields with multiple
  observed realizations per condition.
- **Directly relevant headline result**: per-condition training coverage is deliberately
  incomplete, and the model reconstructs full per-condition distributions by pooling
  spread structure *across* conditions — exactly the "generate more spread than seen in
  training / extrapolate spread structure" requirement.
- **Transfer caveats**: (a) their conditions vary continuously (Re, geometry params),
  making cross-condition pooling easier than discrete part types; (b) 10–250 states per
  condition may exceed available replicates per part type; (c) authors note performance
  degrades as ground-truth variance grows and rare high-variance states can be missed;
  (d) they never test unseen topologies, only parameter variations of known geometries.

## 3. The three models

### 3.1 VGAE baseline (paper) — failed

- **Compressor**: multiscale graph autoencoder; latent = **4 channels per coarse node**
  at scale 4 (`depths [2,2,2,2]`, width 124), condition encoder feeds geometry into
  encoder and decoder at every scale.
- **Sampler**: none — draws latent from N(0,1) per coarse node. Validity requires the
  KL penalty (weight **1e-3**) to force codes toward a bell curve.
- **Failure**: mode collapse "despite careful selection of the latent space size and the
  KL penalty." Mechanism (paper's words): "training ... caused the decoder to become
  overly dependent on input conditions rather than latent features." Forcing codes to
  look like noise strips their information; the decoder learns to ignore them and emits
  one safe answer. Identical mechanism to this repo's documented KLD disaster /
  posterior-shortcut history (`beta_aux`, `posterior_min_std` exist to fight it).

### 3.2 Hi-MGN-V (current, per `_b8_all_warpage_input/config_train1.txt`) — works, two weak spots

- **Compressor**: graph VAE; latent = **global pooled vector** (GlobalAttention,
  `model/vae.py`), 3 slots × 32 dims = 96 total (multiscale_levels 2 → num_z 3).
  Gentle aggregate-level MMD (λ=0.1 vs alpha_recon=100) instead of per-sample KL;
  `beta_aux 1.0` anchors z to per-graph output stats; `posterior_min_std 0.05`.
- **Sampler**: **learned** graph-conditional Gaussian mixture
  (`model/conditional_prior.py`): K=30 components, low-rank covariance rank 8, trained
  jointly (`prior_type gnn_e2e`) by MC-NLL on fresh detached posterior samples plus a
  small analytical-KL anchor (0.05).
- **Weak spot 1 — sampler expressiveness**: a finite diagonal/low-rank mixture is a stiff
  density family; if the aggregate posterior is curved/multi-modal, components
  over-smooth or collapse onto samples. The repo's pathology history (alpha_prior
  variance collapse, min-std floors, cov_rank retrofit, KL anchor) is this limit
  surfacing repeatedly.
- **Weak spot 2 — sampler training budget**: joint training gives the prior exactly the
  simulator's epoch count; no cheap way to train it 10× longer.
- **Ceiling to note**: with a deterministic decoder, all samples for one part live on a
  ≤96-dim manifold. Likely acceptable for smooth warpage modes; a real cap for spatially
  independent local variability.

### 3.3 LDGN (paper) — winner

- **Compressor**: near-identical AE to the baseline but with the KL turned down to
  **1e-6** (~a plain autoencoder; latent free to be informative) and a thinner, spatially
  finer latent: **1 channel per coarse node** at scale 3 (`depths [2,2,1]`), plus
  `norm_latents: True` (batch norm on sampled latents → unit-scale codes).
- **Sampler**: **conditional diffusion GNN in latent space** (1000 train steps, linear
  schedule, hybrid loss with learnable variance, importance step sampling; ~50 inference
  steps). Flow-matching variant (FMGN/LFMGN) available; README: superior at ≤10 steps.
- **Training**: strictly two-stage. AE 5,000 epochs; prior alone **50,000 epochs** on
  the frozen latent space (10× the compressor's budget, cheap because latent is small).
- **Results**: best joint-distribution accuracy (W₂^graph) on every task; ~3× lower
  high-frequency error than field-space diffusion (the AE decoder filters denoising
  artifacts); 8× faster inference than field-space DGN; learns full distributions from
  incomplete per-system coverage. Baselines: GM-GNN (per-node mixtures) matched
  marginals (best W₂^node) but produced spatially discontinuous samples; Bayesian GNN
  inferior and 8× slower to train.

## 4. Side-by-side

| | VGAE baseline (failed) | Hi-MGN-V (current) | LDGN (winner) |
|---|---|---|---|
| Latent shape | Field, 4 ch/coarse node, scale 4 | Global vector, 3×32 | Field, **1 ch/coarse node**, scale 3 |
| Latent regularization | KL **1e-3** (fatal) | MMD 0.1 + beta_aux 1.0 | KL **1e-6** + norm_latents |
| Collapse control | none | posterior_min_std 0.05 | unnecessary (latent carries no matching burden) |
| Sampler | N(0,1) | GM K=30, cov_rank 8 | Diffusion / flow matching in latent space |
| Sampler training | — | joint, simulator's budget | separate, **10× epochs**, frozen codes |
| Conditioning | condition encoder per scale | z concat+Linear into every processor block; graph-aware posterior | condition encoder feeds AE decoder *and* denoiser |
| Sampling cost | 1 pass | 1 pass | ~50 steps (diffusion) / ≤10 (FM) on tiny graph |
| Shared params | width 124–128, lr 1e-4, sum aggr, MSE | width 128, lr 1e-4, sum aggr, MSE | width 126–128, lr 1e-4, sum aggr, MSE |
| Divergences | dropout 0.1, batch 64 | no dropout, batch 8, EMA 0.99 | dropout 0.1, batch 32 |

Empirical triangle: latent field + strong KL + dumb sampler → collapse; weak
regularization + learned-but-stiff sampler → works with patching (current); ~zero
regularization + expressive long-trained sampler → state of the art. **The decisive
variable is the sampler, not the latent field and not diffusion per se** — the collapsed
baseline already had a latent field.

## 5. Ranked causes and solutions (significance order)

1. **Prior family: Gaussian mixture → conditional flow-matching/diffusion prior.**
   Highest evidence (VGAE vs LDGN is a controlled comparison of exactly this), attacks
   the repo's most persistent pathology class, cheapest to implement: the FM prior
   consumes the identical training signal as `mc_nll` (fresh detached posterior sample
   per step, `training_profiles/training_loop.py`), so it is a drop-in at that site.
   Falls back to the mixture with zero architectural damage if it underperforms.
   Note on past "flow matching trained poorly" experience: FM on raw physics fields is
   hard; FM on a normalized, compact, richly conditioned latent is a far easier
   regression problem — and dgn4cfd ships working FMGN/LFMGN reference code and weights.
2. **Latent topology: global pooled z → small per-node latent at a coarse level.**
   Raises the representational ceiling and de-amplifies the decoder shortcut
   (over-compression contributed to the baseline's collapse). Second priority because
   the field alone demonstrably does not fix collapse, and warpage spread is plausibly
   dominated by smooth global modes. If adopted: latent at the existing 500-cluster
   Voronoi level, 1–4 ch/node, near-zero regularization, norm_latents-style batchnorm;
   `pool_features`/`unpool_features` and `ms_z_fusers_*` plumbing already exist.
   Do **not** pursue "very large global z": LDGN's capacity direction is thin channels
   with spatial structure, not a fatter pooled vector (which only makes the prior's
   density problem higher-dimensional and unstructured).
3. **Objective: nothing grades the generated ensemble → scoring-rule fine-tune +
   ensemble metrics.** Proper scoring rules (energy score; afCRPS per AIFS-CRPS,
   [arXiv:2412.15832](https://arxiv.org/abs/2412.15832)) calibrate whatever generator
   exists but cannot expand its family — do this after (1). Caveat from CRPS-LAM
   ([arXiv:2510.09484](https://arxiv.org/html/2510.09484)): per-node CRPS matches
   marginals only; use multivariate scores or rely on a correlated generator.
   Adopt the *measurement* half immediately: report W₂^node **and** a joint/graph-level
   metric, rank histograms, spread-skill ratio — GM-GNN's best-marginals/broken-joint
   result is the cautionary tale for single-metric validation.
4. **Per-node heteroscedastic residual head** — only as a small residual on top of a
   correlated generator (GM-GNN's discontinuous samples show why).

## 6. Actionable parameter takeaways (independent of architecture change)

- **Decouple and extend prior training**: add a post-hoc prior-only phase on the frozen
  simulator with ~10× the effective epochs of the joint run (LDGN ratio 50k:5k).
- **Normalize z** (batchnorm on sampled latents) before the prior consumes it.
- Once the sampler is expressive, **push latent regularization toward zero** and retire
  the collapse-prevention knobs (posterior_min_std, prior_min_std, prior_kl_reg_weight).
- Consider **dropout 0.1** in encoder/prior networks (uniform across all dgn4cfd models).
- Budget check: 50 diffusion / ≤10 FM steps on a 32–96-dim latent (or 500-node coarse
  graph) is negligible next to one simulator forward.

## 7. Recommended path

1. **Step 1 (days, low risk)**: conditional flow-matching prior over the existing global
   z, drop-in at the `mc_nll` site; keep beta_aux and graph-aware posterior; add
   decoupled long prior-training phase; add ensemble validation metrics.
2. **Step 2 (weeks, if Step 1's ceiling binds)**: thin per-node latent field at the
   coarsest Voronoi level + graph-FM/diffusion prior on that field (full LDGN
   adaptation), reusing multiscale pool/unpool and fuser infrastructure.
3. **Step 3**: energy-score/afCRPS fine-tune of the prior (optionally + fusion layers)
   for spread calibration.

Scrutinize distribution tails (rank histograms/coverage) — the paper's stated weakness
is exactly low-probability high-variance states, which for manufacturing means outlier
parts.

## Sources

- Paper: [arXiv:2504.02843](https://arxiv.org/abs/2504.02843) ·
  [HTML full text](https://arxiv.org/html/2504.02843v1) ·
  [OpenReview](https://openreview.net/forum?id=uKZdlihDDn) ·
  [ICLR 2025 oral](https://iclr.cc/virtual/2025/oral/31745) ·
  [Thuerey Group post](https://ge.in.tum.de/2025/01/27/diffusion-graph-nets-iclr25-paper-1-5/)
- Code: [tum-pbs/dgn4cfd](https://github.com/tum-pbs/dgn4cfd) —
  [Wing AE](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Wing/train_ae.py),
  [Wing LDGN](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Wing/train_ldgn.py),
  [Ellipse VGAE baseline](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Ellipse/train_vgae.py),
  [Ellipse AE](https://github.com/tum-pbs/dgn4cfd/blob/main/examples/Ellipse/train_ae.py),
  [VGAE source](https://github.com/tum-pbs/dgn4cfd/blob/main/dgn4cfd/nn/models/vgae.py)
- Scoring-rule training: [AIFS-CRPS, arXiv:2412.15832](https://arxiv.org/abs/2412.15832) ·
  [Pacchiardi et al., arXiv:2112.08217](https://arxiv.org/abs/2112.08217) ·
  [CRPS-LAM, arXiv:2510.09484](https://arxiv.org/html/2510.09484)
- Latent flow matching: [VinAI LFM](https://vinairesearch.github.io/LFM/) ·
  [Generative Latent Neural PDE Solver, arXiv:2503.22600](https://arxiv.org/abs/2503.22600)
