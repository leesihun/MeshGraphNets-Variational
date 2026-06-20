"""Generate the prior_cov_rank x posterior_min_std sweep configs.

Post-diagnosis (diag_prior_spread.py) the bottleneck is TWO things:
  1. the diagonal conditional prior loses the correlated (amplitude) latent
     direction  -> fix: prior_cov_rank > 0 (low-rank covariance components);
  2. the decoder ceiling was smoothed by a high posterior_min_std noise floor +
     low alpha_recon  -> fix: lower posterior_min_std, keep alpha_recon high so
     reconstruction stays sharp (spread comes from Var(mu_q), not sigma_q noise).

Grid: prior_cov_rank in {4, 8, 16, 32} x posterior_min_std in {0.05, 0.10}
= 8 cells -> config_train1..8, single GPU each (0..7), all run concurrently.
alpha_recon is fixed at 100 (sharp reconstruction = high amplitude ceiling).

For each cell a matching pair of inference configs (config_inferN_main.txt /
_sec.txt) points at that cell's checkpoint. Re-run after editing the grid.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

RANKS = [4, 8, 16, 32]
PMINS = [0.05, 0.10]
ALPHA = 100  # fixed: sharp reconstruction -> high amplitude ceiling

# pmin-major (outer), rank inner; gpu = 0..7. train2 (pmin0.05, rank8) = primary.
cells = []
n = 1
gpu = 0
for pmin in PMINS:
    for rank in RANKS:
        cells.append({"n": n, "pmin": pmin, "rank": rank, "gpu": gpu})
        n += 1
        gpu += 1


def suffix(c):
    return f"r{c['rank']:02d}_p{int(round(c['pmin'] * 100)):03d}"


def primary_tag(c):
    return "   [PRIMARY: sharp ceiling + low-rank prior]" if (
        c["pmin"] == 0.05 and c["rank"] == 8) else ""


TRAIN_TMPL = """% ============================================================
% SWEEP cell: prior_cov_rank={rank}, posterior_min_std={pmin:.2f}, alpha_recon={alpha}{tag}
% Two-pronged fix for narrow generated spread (see diag_prior_spread.py):
%   prior_cov_rank>0  -> prior captures the correlated amplitude direction
%   low posterior_min_std + high alpha_recon + MSE loss -> sharp decoder (high
%   ceiling, extremes not smoothed); spread comes from Var(mu_q), not sigma_q noise.
% Single GPU per run; 8 cells fill GPUs 0-7 and run concurrently.
% ============================================================
model   MeshGraphNets-V
mode    train
gpu_ids {gpu}
log_file_dir    b8_all/train{n}_{sfx}.log
modelpath       ./outputs/b8_all/warpage_train{n}_{sfx}.pth

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
Training_epochs      1000
Batch_size           8
LearningR            0.0001
Latent_dim           128
num_workers          4
std_noise            0.01    # without noise the model ignores z and memorises x->y
residual_scale       1
augment_geometry     True
grad_accum_steps     1

% Memory / performance
use_checkpointing    True
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
bipartite_unpool    True

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   32
recon_loss       mse    # MSE penalizes extreme-node errors -> sharper amplitude (vs Huber)
alpha_recon      {alpha}    # high: sharp reconstruction -> high amplitude ceiling
lambda_mmd       0.1    # low: z encodes structured spread, not collapsed to N(0,I)
vae_graph_aware  True   # z encodes only y-residual unexplained by x
beta_aux         1.0    # anchors z to per-graph [y_mean, y_std]; teaches z to encode spread
posterior_min_std {pmin:.2f}  # SWEPT: low floor -> sharp decoder; spread comes from Var(mu_q)

% Prior - gnn_e2e: joint training, graph-conditional, low-rank covariance.
% Density matching only: the alpha_prior recon term is variance-collapsing and stays 0.
prior_type               gnn_e2e
use_conditional_prior    True    # CRITICAL: persist True so rollout uses the learned prior
prior_cov_rank           {rank}     # SWEPT: low-rank covariance per component (captures correlation)
alpha_prior_max          0.0     # MUST be 0 for spread modeling
prior_loss_type          mc_nll  # mixture NLL on fresh posterior samples
prior_nll_weight         1.0
prior_kl_reg_weight      0.05    # small analytical-KL stability anchor
prior_gumbel_temp        1.0
prior_mixture_components 30
prior_mp_layers          5
prior_hidden_dim         192
prior_min_std            0.1
prior_temperature        1.0
"""

INFER_TMPL = """% Inference for sweep cell train{n}: prior_cov_rank={rank}, posterior_min_std={pmin:.2f}{tag}
model   MeshGraphNets-V
mode    inference
gpu_ids {gpu}
log_file_dir         b8_all/infer_train{n}_{which}.log
modelpath            ./outputs/b8_all/warpage_train{n}_{sfx}.pth
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
Latent_dim           128
num_workers          4
std_noise            0.01
residual_scale       1

% Memory / performance
use_checkpointing    True
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

% Multiscale - MUST match training coarsening mode (voronoi_seedmean)
use_multiscale      True
coarsening_type     voronoi_seedmean
voronoi_clusters    2000, 500
multiscale_levels   2
mp_per_level        4, 8, 12, 8, 4
bipartite_unpool    True

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   32
alpha_recon      {alpha}
lambda_mmd       0.1
vae_graph_aware  True

% Prior - sample z from the learned conditional prior p(z|graph)
prior_type               gnn_e2e
use_conditional_prior    True
prior_cov_rank           {rank}
alpha_prior_max          0.0
prior_kl_reg_weight      0.05
prior_gumbel_temp        1.0
prior_mixture_components 30
prior_mp_layers          5
prior_hidden_dim         192
prior_min_std            0.1
prior_temperature        1.0   # raise (e.g. 1.3) to widen generated spread without retraining
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
    sfx = suffix(c)
    tag = primary_tag(c)
    tpath = os.path.join(HERE, f"config_train{c['n']}.txt")
    write(tpath, TRAIN_TMPL.format(sfx=sfx, tag=tag, alpha=ALPHA, **c))
    written.append(os.path.basename(tpath))
    for which, ds in INFER_DATASETS.items():
        ipath = os.path.join(HERE, f"config_infer{c['n']}_{which}.txt")
        write(ipath, INFER_TMPL.format(sfx=sfx, tag=tag, alpha=ALPHA, which=which,
                                       infer_ds=ds, **c))
        written.append(os.path.basename(ipath))

print(f"Wrote {len(written)} files:")
for w in written:
    print("  " + w)
