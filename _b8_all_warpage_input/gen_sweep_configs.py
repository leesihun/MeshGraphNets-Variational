"""Generate the network-capacity (width x depth) sweep around the previous
winner, 8 single-GPU cells.

=== WHY THIS SWEEP EXISTS (history) ===

Previous sweep (lambda-ladder + capacity, 2026-07-15) winner =
**cell 8 = z128_h256_median_ph256_l0p2**
(checkpoint preserved: outputs/b8_all/warpage_train8_z128_h256_median_ph256_l0p2.pth).
That winner was still slightly UNDER-dispersed (p99_cov 0.85/0.78 < 1), and the
decisive discriminator that round was prior_hidden_dim (cell 4 vs 8: same
z128/l0.2, only ph 192->256, cell 4 blew up to p99_cov 2.16-2.45 while cell 8
was calibrated). Conclusion: at higher vae_latent_dim the *network capacity* of
the sub-nets — not just lambda_mmd — gates whether spread is calibrated.

So this sweep freezes the winner's VAE-loss recipe (vae_latent_dim=128,
lambda_mmd=0.2, median bandwidth, beta_aux=1.0, posterior_min_std=0.05,
fm_steps=40) and probes the width x depth of the three sub-networks:

  - Main processor:   Latent_dim (width)    [mp_per_level = depth, left pinned]
  - Conditional prior: prior_hidden_dim (width) x prior_mp_layers (depth)
  - VAE posterior enc: vae_mp_layers (depth)  [never touched before]

NOTE: under use_multiscale=True the main-processor DEPTH is mp_per_level
(4,8,12,8,4), NOT message_passing_num (that key only builds the flat processor
and is dead here). This sweep keeps mp_per_level pinned and moves main width via
Latent_dim only. Latent_dim is NOT fully isolated to the processor — it also
sizes the VAE encoder trunk and the z-fusers — so read its cells as
"processor + VAE-encoder width" combined.

=== CELLS (anchor = winner recipe; each cell changes ONE thing vs cell 1) ===

  Focus 2x2 (user pick): Latent_dim {256,512} x prior_mp_layers {5,10}
    cell 1: Latent_dim 256, prior_mp_layers 5   -> ANCHOR (winner replicate)
    cell 2: Latent_dim 512, prior_mp_layers 5   -> main width up
    cell 3: Latent_dim 256, prior_mp_layers 10  -> prior depth up
    cell 4: Latent_dim 512, prior_mp_layers 10  -> both up (interaction)
  OFAT extras (one-axis probes vs the anchor):
    cell 5: Latent_dim 128                       -> main width down (lower bound)
    cell 6: prior_mp_layers 3                     -> prior depth down (lower bound)
    cell 7: prior_hidden_dim 384                  -> prior width up (depth's partner)
    cell 8: vae_mp_layers 8                       -> posterior-encoder depth up
                                                    (the untouched 3rd sub-net)

=== WHICH AXES ARE EXPECTED TO MOVE WHAT ===

- prior_mp_layers, prior_hidden_dim, vae_mp_layers act on the prior/posterior
  path -> they are the SPREAD (p99_cov / tail) knobs; weight them for the
  manufacturing-defect-tail objective.
- Latent_dim mostly affects mean/prediction fidelity (and could relieve a
  decoder bottleneck on wide outputs), not spread directly.

=== PINNED TO THE WINNER (so every delta is attributable) ===

vae_latent_dim 128, lambda_mmd 0.2, mmd_bandwidth median, prior_family fm,
prior_fm_steps 40, prior_temperature 1.0, alpha_recon 1000, beta_aux 1.0,
posterior_min_std 0.05, Training_epochs 1000, L=2 voronoi 2000/500, mse,
mp_per_level 4,8,12,8,4. use_checkpointing False on every cell (no OOM risk on
this box). NOTE: epochs 1000 (was 2000 for the old winner), so cell 1 is NOT a
bit-exact replicate of old train8 — the intra-sweep comparison (cells 1-8) is
still clean since all share 1000ep.

=== JUDGING ===

Pass gate: p99_cov ~ 1 AND gen_frac_gt_p99 ~ 1%. Tie-break: Wasserstein.
Gen_max is an outlier ALARM only — never rank by it. prior_temperature stays 1.0
in-config; it remains the free post-hoc scale knob. The old winner checkpoint
(warpage_train8_z128_h256_median_ph256_l0p2.pth) is NOT overwritten — new tags
embed pml/ph/vml so filenames never collide.

One GPU per cell (gpu_ids 0..7). config_train1..8 (+ main/sec infer per cell).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 1000
ALPHA_RECON = 1000
BW = "median"
FM_STEPS = 40      # sampling-only knob: tail-overshoot fix, no training effect
VAE_LD = 128       # vae_latent_dim, pinned to the winner
LAM = 0.2          # lambda_mmd, pinned to the winner

# n, gpu, Latent_dim (main width), prior_mp_layers (prior depth),
# prior_hidden_dim (prior width), vae_mp_layers (posterior depth).
# use_checkpointing is False on every cell (no OOM risk on this box), so the
# memory/speed axis is identical across cells too.
cells = [
    dict(n=1, gpu=0, ldm=256, pml=5,  phd=256, vml=5, ckpt=False),  # ANCHOR = winner replicate
    dict(n=2, gpu=1, ldm=512, pml=5,  phd=256, vml=5, ckpt=False),  # main width up
    dict(n=3, gpu=2, ldm=256, pml=10, phd=256, vml=5, ckpt=False),  # prior depth up
    dict(n=4, gpu=3, ldm=512, pml=10, phd=256, vml=5, ckpt=False),  # both up (interaction)
    dict(n=5, gpu=4, ldm=128, pml=5,  phd=256, vml=5, ckpt=False),  # main width down
    dict(n=6, gpu=5, ldm=256, pml=3,  phd=256, vml=5, ckpt=False),  # prior depth down
    dict(n=7, gpu=6, ldm=256, pml=5,  phd=384, vml=5, ckpt=False),  # prior width up
    dict(n=8, gpu=7, ldm=256, pml=5,  phd=256, vml=8, ckpt=False),  # posterior depth up
]


def make_tag(c):
    return f"z{VAE_LD}_h{c['ldm']}_pml{c['pml']}_ph{c['phd']}_vml{c['vml']}"


TRAIN_TMPL = """% ============================================================
% Network-capacity (width x depth) sweep cell {n}: {tag}
% Why: see gen_sweep_configs.py docstring. Anchored on the previous winner
% (z128_h256_median_ph256_l0p2). Focus 2x2 = Latent_dim {{256,512}} x
% prior_mp_layers {{5,10}} (cells 1-4); OFAT extras cells 5-8.
% Latent_dim={ldm} (main width), prior_mp_layers={pml} (prior depth),
% prior_hidden_dim={phd} (prior width), vae_mp_layers={vml} (posterior depth).
% Pinned: vae_latent_dim={ld}, lambda_mmd={lam}, mmd_bandwidth={bw},
% alpha_recon={alpha}, beta_aux=1.0, posterior_min_std=0.05, fm_steps={fm_steps},
% Training_epochs={epochs}, L=2 voronoi 2000/500, mse, mp_per_level 4,8,12,8,4.
% ============================================================
model   MeshGraphNets-V
mode    train
gpu_ids {gpu}
log_file_dir    b8_all/train{n}_{tag}.log
modelpath       ./outputs/b8_all/warpage_train{n}_{tag}.pth

% Datasets
dataset_dir     ./dataset/b8_main_sec_dataset_traincopy.h5
infer_dataset   ./dataset/b8_main_sec_dataset_infer.h5
infer_timesteps 1
num_vae_samples 10000

% Common params
input_var            3
output_var           3
feature_loss_weights 1, 1, 1
edge_var             8
positional_features  4

% Network parameters
message_passing_num  15
Training_epochs      {epochs}
Batch_size           8
LearningR            0.0001
Latent_dim           {ldm}
num_workers          4
std_noise            0.01
augment_geometry     True
grad_accum_steps     1

% Memory / performance
use_checkpointing    {ckpt}   # faster; set True if OOM
use_amp              True
use_ema              True
ema_decay            0.99
test_interval        100
val_interval         1

% Node types
use_node_types  False

% World edges
use_world_edges False

% Test set
test_batch_idx   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34
plot_feature_idx -1

% Multiscale (2-level Voronoi V-cycle)
use_multiscale      True
coarsening_type     voronoi_seedmean
voronoi_clusters    2000, 500
multiscale_levels   2
mp_per_level        4, 8, 12, 8, 4

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   {ld}
vae_mp_layers    {vml}
recon_loss       mse
alpha_recon      {alpha}
lambda_mmd       {lam}
mmd_bandwidth    {bw}
vae_graph_aware  True
beta_aux         1.0
posterior_min_std 0.05

% Prior - gnn_e2e joint training; prior_family fm = conditional flow matching.
% prior_fm_steps 40: sampling-only (validation CRPS + inference); halves Euler
% overshoot in the z tails vs the old 20-step default (Gen_max spike fix).
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_nll_weight         1.0
prior_fm_steps           {fm_steps}
prior_mp_layers          {pml}
prior_hidden_dim         {phd}
prior_temperature        1.0
"""

INFER_TMPL = """% Inference for cell {n}: {tag} ({which})
% Latent_dim={ldm}, prior_mp_layers={pml}, prior_hidden_dim={phd}, vae_mp_layers={vml}
% (all architecture keys MUST match training so the checkpoint loads).
model   MeshGraphNets-V
mode    inference
gpu_ids {gpu}
log_file_dir         b8_all/infer_train{n}_{which}.log
modelpath            ./outputs/b8_all/warpage_train{n}_{tag}.pth
inference_output_dir outputs/b8_all/infer_train{n}_{which}

% Datasets
dataset_dir     ./dataset/b8_main_sec_dataset_traincopy.h5
infer_dataset   {infer_ds}
infer_timesteps 1
num_vae_samples 5000

% Common params
input_var            3
output_var           3
feature_loss_weights 1, 1, 1
edge_var             8
positional_features  4

% Network parameters
message_passing_num  15
Batch_size           8
LearningR            0.0001
Latent_dim           {ldm}
num_workers          4
std_noise            0.01

% Memory / performance
use_checkpointing    False
use_amp              True
use_ema              True
ema_decay            0.99

% Node types
use_node_types  False

% World edges
use_world_edges False

% Test set
test_batch_idx   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34
plot_feature_idx -1

% Multiscale - MUST match training
use_multiscale      True
coarsening_type     voronoi_seedmean
voronoi_clusters    2000, 500
multiscale_levels   2
mp_per_level        4, 8, 12, 8, 4

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   {ld}
vae_mp_layers    {vml}
recon_loss       mse
vae_graph_aware  True

% Prior - sample z from the learned conditional prior p(z|graph)
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_fm_steps           {fm_steps}
prior_mp_layers          {pml}
prior_hidden_dim         {phd}
prior_temperature        1.0   # free post-hoc knob: calibrate spread scale at inference
"""

INFER_DATASETS = {
    "main": "./dataset/infer_b8_main.h5",
    "sec":  "./dataset/infer_b8_secondary.h5",
}


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


written = []
for c in cells:
    tag = make_tag(c)
    fields = dict(n=c["n"], gpu=c["gpu"], ldm=c["ldm"], pml=c["pml"],
                  phd=c["phd"], vml=c["vml"], ckpt=c["ckpt"], ld=VAE_LD,
                  lam=LAM, bw=BW, alpha=ALPHA_RECON, tag=tag, epochs=EPOCHS,
                  fm_steps=FM_STEPS)
    tpath = os.path.join(HERE, f"config_train{c['n']}.txt")
    write(tpath, TRAIN_TMPL.format(**fields))
    written.append(os.path.basename(tpath))
    for which, ds in INFER_DATASETS.items():
        ipath = os.path.join(HERE, f"config_infer{c['n']}_{which}.txt")
        write(ipath, INFER_TMPL.format(which=which, infer_ds=ds, **fields))
        written.append(os.path.basename(ipath))

print(f"Wrote {len(written)} files:")
for w in written:
    print("  " + w)
