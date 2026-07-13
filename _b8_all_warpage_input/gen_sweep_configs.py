"""Generate the FM capacity/MMD sweep, 8 single-GPU cells.

Strategy: prior_temperature (fixed 1.0 here) will calibrate the spread SCALE
post-hoc at inference, so this training sweep is judged on SCALE-INVARIANT
quality — centering (Gen_mu vs GT_mu), distribution shape (kurtosis + unit-scaled
Wasserstein/KS), and prior health ([PriorDiag] spread_ratio) — NOT raw Wasserstein.

Three 2-level factors in 8 runs = a full factorial around the train6 regime.

  A = vae_latent_dim    {128, 256}      stochastic global z capacity
  B = Latent_dim        {128, 256}      base MGN hidden/global capacity
  C = lambda_mmd        {0.1, 1.0}      aggregate-posterior shaping strength

prior_hidden_dim is tied to z capacity:
  z128 -> ph256, z256 -> ph384

mmd_bandwidth is fixed for this batch.

FM prior only. alpha_recon = 1000 (fixed): pushes recon fidelity hard; safe
because temperature restores spread downstream. Caveat: alpha_recon=1000 heavily
dominates the loss, so the MMD-related main effects (B, D) may read small simply
because MMD is swamped — interpret a null B/D effect in that light.

Base: L=2 Voronoi V-cycle, mp 4/8/12/8/4, voronoi 2000/500, mse, ew=0,
beta_aux=1.0, posterior_min_std=0.05, Training_epochs=1000.
One GPU per cell (gpu_ids 0..7). config_train1..8 (+ main/sec infer per cell).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 1000
ALPHA_RECON = 1000

# Full factorial:
#   A vae_latent_dim {128,256}
#   B Latent_dim {128,256}
#   C lambda_mmd {0.1,1.0}
# prior_hidden_dim scales with vae_latent_dim; mmd_bandwidth is fixed.
# n, gpu, vae_latent_dim, Latent_dim, prior_hidden_dim, lambda_mmd
cells = [
    dict(n=1, gpu=0, ld=128, ldm=128, bw="fixed", phd=256, lam=0.1),
    dict(n=2, gpu=1, ld=128, ldm=128, bw="fixed", phd=256, lam=1.0),
    dict(n=3, gpu=2, ld=128, ldm=256, bw="fixed", phd=256, lam=0.1),
    dict(n=4, gpu=3, ld=128, ldm=256, bw="fixed", phd=256, lam=1.0),
    dict(n=5, gpu=4, ld=256, ldm=128, bw="fixed", phd=384, lam=0.1),
    dict(n=6, gpu=5, ld=256, ldm=128, bw="fixed", phd=384, lam=1.0),
    dict(n=7, gpu=6, ld=256, ldm=256, bw="fixed", phd=384, lam=0.1),
    dict(n=8, gpu=7, ld=256, ldm=256, bw="fixed", phd=384, lam=1.0),
]


def make_tag(c):
    return f"z{c['ld']}_h{c['ldm']}_{c['bw']}_ph{c['phd']}_l{str(c['lam']).replace('.', 'p')}"


TRAIN_TMPL = """% ============================================================
% FM 4-factor sweep cell {n}: {tag}
% vae_latent_dim={ld}, mmd_bandwidth={bw}, prior_hidden_dim={phd}, lambda_mmd={lam}
% alpha_recon={alpha} (fixed), Latent_dim={ldm}, Training_epochs={epochs}, single-GPU.
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
use_checkpointing    False   # faster; set True if OOM (esp. z256 / prior_hidden_dim 384)
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
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_nll_weight         1.0
prior_fm_steps           20
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
prior_fm_steps           20
prior_mp_layers          5
prior_hidden_dim         {phd}
prior_temperature        1.0   # Phase 2: sweep this at inference to calibrate spread scale
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
    fields = dict(n=c["n"], gpu=c["gpu"], ld=c["ld"], bw=c["bw"], phd=c["phd"],
                  lam=c["lam"], ldm=c["ldm"], alpha=ALPHA_RECON, tag=tag, epochs=EPOCHS)
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
