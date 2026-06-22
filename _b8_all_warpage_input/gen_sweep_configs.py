"""Generate the extreme_weight sweep configs.

Diagnosis (confirmed): the amplitude ceiling (decode(mu_q) ~ 0.74 of GT) is a
DECODER problem — per-node recon is good but PEAKS are smoothed, because the
per-node averaged loss under-weights the few extreme nodes that set the
amplitude (max-min). Structure (L / voronoi) and latent knobs (beta_aux,
lambda_mmd, cov_rank, std_noise, temperature) don't move it.

Fix under test: `extreme_weight` upweights extreme-z_disp nodes in the POSTERIOR
reconstruction only (one-to-one, z encodes the true target -> no variance
collapse). Everything else is held at the best known config (cell2 of the
cov_rank sweep: L=2, mp 4/8/12/8/4, voronoi 2000/500, cov_rank=8, pmin=0.05,
alpha_recon=100, mse, feature_loss_weights 1/1/1, beta_aux=1.0).

2 axes: extreme_weight {0,2,4,8} x feature_loss_weights {[1,1,1],[0,0,1]}.
[0,0,1] focuses all decoder capacity on z_disp (warpage) — synergizes with ew.
8 cells -> config_train1..8, single GPU each (0..7). cell1 (ew=0, flw111) is the
control (= current best). SCREENING budget: Training_epochs=500 (relative ranking
is valid early; full-train the winner to 1000). Re-run after editing the grid.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 500                              # screening budget (full-train winner to 1000)
EWS = [0, 2, 4, 8]                        # extreme_weight
FLWS = [("111", "1, 1, 1"), ("001", "0, 0, 1")]   # feature_loss_weights (tag, csv)

cells = []
n, gpu = 1, 0
for flw_tag, flw_csv in FLWS:             # flw-major: cells 1-4 = 111, 5-8 = 001
    for ew in EWS:
        cells.append(dict(n=n, gpu=gpu, ew=ew, flw_tag=flw_tag, flw=flw_csv))
        n += 1
        gpu += 1


def suffix(c):
    return f"ew{c['ew']}_flw{c['flw_tag']}"


def primary_tag(c):
    if c["ew"] == 0 and c["flw_tag"] == "111":
        return "   [control: ew=0, flw=1/1/1 == current best]"
    if c["ew"] == 0 and c["flw_tag"] == "001":
        return "   [isolates the z_disp-focus effect]"
    return ""


TRAIN_TMPL = """% ============================================================
% SWEEP cell: extreme_weight={ew}{tag}
% Raises the amplitude ceiling by upweighting extreme-z_disp nodes in the
% POSTERIOR reconstruction (one-to-one -> no variance collapse). Base = best
% known config (L=2, mp 4/8/12/8/4, voronoi 2000/500, cov_rank=8, pmin=0.05, mse).
% Single GPU per run; 8 cells fill GPUs 0-7.
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
feature_loss_weights {flw}    # SWEPT: 1/1/1 (all disp) vs 0/0/1 (z_disp/warpage only)
edge_var             8
positional_features  4

% Network parameters
message_passing_num  15
Training_epochs      {epochs}    # SCREENING budget; full-train the winner to 1000
Batch_size           8
LearningR            0.0001
Latent_dim           128
num_workers          4
std_noise            0.01    # without noise the model ignores z and memorises x->y
residual_scale       1
augment_geometry     True
grad_accum_steps     1

% Memory / performance
use_checkpointing    False
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

% Multiscale (2-level Voronoi V-cycle) -- fixed at best known
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
extreme_weight   {ew}      # SWEPT: upweight extreme-z_disp nodes in posterior recon (0 = off)
alpha_recon      100    # high: sharp reconstruction -> high amplitude ceiling
lambda_mmd       0.1    # low: z encodes structured spread, not collapsed to N(0,I)
vae_graph_aware  True   # z encodes only y-residual unexplained by x
beta_aux         1.0    # anchors z to per-graph [y_mean, y_std]; prevents posterior collapse
posterior_min_std 0.05  # low floor -> sharp decoder; spread comes from Var(mu_q)

% Prior - gnn_e2e: joint training, graph-conditional, low-rank covariance.
prior_type               gnn_e2e
use_conditional_prior    True    # CRITICAL: persist True so rollout uses the learned prior
prior_cov_rank           8       # low-rank covariance per component (captures correlation)
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

INFER_TMPL = """% Inference for sweep cell train{n}: extreme_weight={ew}{tag}
% (extreme_weight is training-only; architecture is identical across cells.)
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
bipartite_unpool    True

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   32
recon_loss       mse
alpha_recon      100
lambda_mmd       0.1
vae_graph_aware  True

% Prior - sample z from the learned conditional prior p(z|graph)
prior_type               gnn_e2e
use_conditional_prior    True
prior_cov_rank           8
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
    fields = dict(n=c["n"], gpu=c["gpu"], ew=c["ew"], sfx=sfx, tag=tag,
                  flw=c["flw"], epochs=EPOCHS)
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
