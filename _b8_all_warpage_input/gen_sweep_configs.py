"""Generate the multiscale-levels x voronoi-resolution sweep configs.

Diagnosis: the amplitude ceiling (decode(mu_q) ~ 0.74 of GT) is a DECODER problem
— per-node recon is good but peaks are smoothed. Latent knobs (beta_aux,
lambda_mmd, cov_rank, std_noise) are downstream and don't move it; the prior
already saturates the ceiling. So this sweep probes the DECODER's structural
smoothing, holding the total message-passing budget ~constant (~30) while varying:
  - multiscale_levels L in {2, 3, 4}  (how the budget is split across scales)
  - voronoi_clusters                  (how aggressively pooling discards fine
                                       detail where peaks live)
This isolates over-smoothing vs pooling-detail-loss vs capacity without changing
the loss. Everything else is held at the best known values.

8 cells -> config_train1..8, single GPU each (0..7). Re-run after editing cells.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# n, gpu, L, mp_per_level (sum ~30, symmetric V-cycle of length 2L+1), voronoi_clusters (L)
cells = [
    dict(n=1, gpu=0, L=2, mp=[4, 6, 10, 6, 4],              vc=[2000, 500]),
    dict(n=2, gpu=1, L=2, mp=[4, 6, 10, 6, 4],              vc=[4000, 1000]),
    dict(n=3, gpu=2, L=3, mp=[3, 4, 5, 6, 5, 4, 3],         vc=[2000, 500, 125]),
    dict(n=4, gpu=3, L=3, mp=[3, 4, 5, 6, 5, 4, 3],         vc=[4000, 1000, 250]),
    dict(n=5, gpu=4, L=3, mp=[3, 4, 5, 6, 5, 4, 3],         vc=[1000, 250, 60]),
    dict(n=6, gpu=5, L=4, mp=[2, 3, 3, 4, 6, 4, 3, 3, 2],   vc=[2000, 500, 125, 32]),
    dict(n=7, gpu=6, L=4, mp=[2, 3, 3, 4, 6, 4, 3, 3, 2],   vc=[4000, 1000, 250, 64]),
    dict(n=8, gpu=7, L=4, mp=[2, 3, 3, 4, 6, 4, 3, 3, 2],   vc=[1000, 250, 60, 16]),
]

# sanity: mp_per_level length must be 2L+1 and sum ~30
for c in cells:
    assert len(c["mp"]) == 2 * c["L"] + 1, (c["n"], len(c["mp"]), c["L"])
    assert len(c["vc"]) == c["L"], (c["n"], len(c["vc"]), c["L"])


def suffix(c):
    return f"L{c['L']}_vc{c['vc'][0]}"


def csv(xs):
    return ", ".join(str(x) for x in xs)


TRAIN_TMPL = """% ============================================================
% SWEEP cell: multiscale_levels={L}, mp_per_level=[{mp}] (sum={mpsum}), voronoi={vc}
% Decoder-ceiling probe (amplitude peaks are smoothed; (a) posterior ~0.74).
% Total MP budget held ~30; vary how it splits across scales (L) and how
% aggressively pooling discards fine detail (voronoi_clusters).
% Latent side held fixed (cov_rank=8, pmin=0.05, alpha_recon=100, mse).
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

% Multiscale (Voronoi V-cycle) -- SWEPT: L levels + voronoi resolution
use_multiscale      True
coarsening_type     {ctype}
voronoi_clusters    {vc}
multiscale_levels   {L}
mp_per_level        {mp}
bipartite_unpool    True

% VAE (MMD-InfoVAE)
use_vae          True
vae_latent_dim   32
recon_loss       mse    # MSE penalizes extreme-node errors -> sharper amplitude (vs Huber)
alpha_recon      100    # high: sharp reconstruction -> high amplitude ceiling
lambda_mmd       0.1    # low: z encodes structured spread, not collapsed to N(0,I)
vae_graph_aware  True   # z encodes only y-residual unexplained by x
beta_aux         1.0    # anchors z to per-graph [y_mean, y_std]; teaches z to encode spread
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

INFER_TMPL = """% Inference for sweep cell train{n}: multiscale_levels={L}, voronoi={vc}
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

% Multiscale - MUST match training (L levels + voronoi resolution + seedmean)
use_multiscale      True
coarsening_type     {ctype}
voronoi_clusters    {vc}
multiscale_levels   {L}
mp_per_level        {mp}
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
    ctype = csv(["voronoi_seedmean"] * c["L"])   # explicit per-level (length L)
    fields = dict(n=c["n"], gpu=c["gpu"], L=c["L"], sfx=sfx,
                  mp=csv(c["mp"]), mpsum=sum(c["mp"]), vc=csv(c["vc"]), ctype=ctype)
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
