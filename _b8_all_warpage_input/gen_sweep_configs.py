"""Generate the aggressive-lambda + capacity sweep around the previous winner,
8 single-GPU cells.

=== WHY THIS SWEEP EXISTS (history) ===

1. 4-factor FM sweep (z{64,256} x bandwidth{fixed,median} x ph{192,384} x
   lambda_mmd{0.05,0.2}, 2000 ep): winner = old train3 =
   **z64_median_ph192_l0p2** — best across metrics per user judgment
   (checkpoint preserved: outputs/b8_all/warpage_train3_z64_median_ph192_l0p2.pth).
   Two open issues it left behind:
     (a) it sat at the BOUNDARY of the tested lambda range (0.2 was the max),
         so the response above 0.2 is unmapped;
     (b) occasional Gen_max spikes — single outlier realizations, most likely
         the FM sampler throwing rare z outside the posterior support.
2. The 2026-07-13 capacity sweep (z{128,256} x Latent_dim{128,256} x
   lambda{0.1,1.0}, all-FIXED bandwidth, 1000 ep) was scrapped: it abandoned
   the winning recipe on every axis (dropped z64, dropped median bandwidth,
   dropped lambda 0.2, halved epochs), so nothing in it was comparable to the
   winner.

This sweep therefore keeps the winner as cell 1 (control) and changes ONE
question per cell group.

=== CELLS ===

  Cells 1-3:  z64,  ph192, Latent_dim 128, lambda_mmd {0.2, 1, 10}
  Cells 4-6:  z128, ph256, Latent_dim 128, lambda_mmd {0.2, 1, 10}
  Cell  7:    z64,  ph192, Latent_dim 256, lambda 0.2   (capacity probe vs cell 1)
  Cell  8:    z128, ph256, Latent_dim 256, lambda 0.2   (capacity probe vs cell 4)

=== WHY EACH CHOICE ===

- lambda_mmd {0.2, 1, 10}: 0.2 is the winner/control; 1 and 10 push the
  MMD shaping roughly one and two decades harder. Rationale: stronger shaping
  forces the aggregate posterior toward N(0,I), which is the config-level fix
  for the Gen_max tail spikes (an FM prior matching a tighter target throws
  fewer out-of-support samples) — at the risk of erasing per-type spread
  structure. The largest lambda that keeps [PriorDiag] spread_ratio ~ 1 and
  per-type histograms distinct is the operating point. (CLAUDE.md's
  keep-lambda-low default is intentionally overridden to map this collapse
  threshold; if 10 collapses spread, that is the answer, not a failure.)
- vae_latent_dim {64, 128}: 64 = the winner; 128 = the sweet spot predicted by
  the earlier z-dim sweep but never yet tested with median bandwidth. z256 is
  excluded — it overshot tails / destabilized in both earlier sweeps.
- Latent_dim {128 -> 256} (cells 7, 8): user hypothesis "just give the base
  MGN more width". Each probe differs from its partner (cell 1, cell 4) ONLY
  in Latent_dim, at the winning lambda, so the pair isolates the pure
  capacity effect. lambda variants of L256 are deliberately not spent cells:
  if width helps at 0.2 it can be crossed with lambda in the next sweep.
- mmd_bandwidth median (all cells): beat fixed at the winning cell, and the
  fixed sigmas are tuned for ~z64 — they weaken by z128, which would confound
  the lambda and z comparisons.
- prior_hidden_dim tied to z (64->192, 128->256): ph was a factor in the
  4-factor sweep and did not decide it; tying it saves cells.
- prior_fm_steps 40 (was 20): sampling-only knob — fm_loss never uses
  num_steps, so training is untouched; it only refines validation-CRPS and
  inference ODE integration, halving Euler overshoot in the z tails (the
  Gen_max fix that costs nothing). Since cell 1 is otherwise an exact
  weights-level anchor replicate, cell 1 vs the old train3 result isolates
  the steps effect.
- Training_epochs 2000, alpha_recon 1000, beta_aux 1.0, posterior_min_std
  0.05, L=2 voronoi 2000/500, mse: pinned to the winner recipe so every
  observed delta is attributable to the swept factor.

=== JUDGING ===

Pass gate: p99_cov ~ 1 AND gen_frac_gt_p99 ~ 1%. Tie-break: Wasserstein.
Gen_max is an outlier ALARM only — never rank by it (single-sample statistic).
prior_temperature stays 1.0 in-config; it remains the free post-hoc scale knob.

One GPU per cell (gpu_ids 0..7). config_train1..8 (+ main/sec infer per cell).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 2000
ALPHA_RECON = 1000
BW = "median"
FM_STEPS = 40    # sampling-only knob: tail-overshoot fix, no training effect

# n, gpu, vae_latent_dim, model Latent_dim, prior_hidden_dim, lambda_mmd
cells = [
    dict(n=1, gpu=0, ld=64,  ldm=128, phd=192, lam=0.2),   # anchor = old train3
    dict(n=2, gpu=1, ld=64,  ldm=128, phd=192, lam=1.0),
    dict(n=3, gpu=2, ld=64,  ldm=128, phd=192, lam=10.0),
    dict(n=4, gpu=3, ld=128, ldm=128, phd=256, lam=0.2),
    dict(n=5, gpu=4, ld=128, ldm=128, phd=256, lam=1.0),
    dict(n=6, gpu=5, ld=128, ldm=128, phd=256, lam=10.0),
    dict(n=7, gpu=6, ld=64,  ldm=256, phd=192, lam=0.2),   # capacity probe vs cell 1
    dict(n=8, gpu=7, ld=128, ldm=256, phd=256, lam=0.2),   # capacity probe vs cell 4
]


def make_tag(c):
    lam = f"{c['lam']:g}".replace('.', 'p')
    return f"z{c['ld']}_h{c['ldm']}_{BW}_ph{c['phd']}_l{lam}"


TRAIN_TMPL = """% ============================================================
% Aggressive-lambda + capacity sweep cell {n}: {tag}
% Why: see gen_sweep_configs.py docstring (lambda {{0.2,1,10}} maps the MMD
% collapse threshold above the previous winner; Latent_dim 256 cells 7-8 are
% pure width probes vs cells 1/4; everything else pinned to the winner).
% vae_latent_dim={ld}, mmd_bandwidth={bw}, prior_hidden_dim={phd}, lambda_mmd={lam}
% alpha_recon={alpha} (fixed), Latent_dim={ldm}, Training_epochs={epochs}, single-GPU.
% Anchored on the previous sweep winner (z64_median_ph192_l0p2).
% Base L=2, voronoi 2000/500, mse, ew=0, beta_aux=1.0, posterior_min_std=0.05.
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
use_checkpointing    False   # faster; set True if OOM
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
prior_mp_layers          5
prior_hidden_dim         {phd}
prior_temperature        1.0
"""

INFER_TMPL = """% Inference for cell {n}: {tag} ({which})
% vae_latent_dim={ld}, prior_hidden_dim={phd}, Latent_dim={ldm}
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
recon_loss       mse
vae_graph_aware  True

% Prior - sample z from the learned conditional prior p(z|graph)
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_fm_steps           {fm_steps}
prior_mp_layers          5
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
    fields = dict(n=c["n"], gpu=c["gpu"], ld=c["ld"], bw=BW, phd=c["phd"],
                  lam=c["lam"], ldm=c["ldm"], alpha=ALPHA_RECON, tag=tag, epochs=EPOCHS,
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
