"""Generate the anti-smoothing (capacity + sharp-posterior) sweep.

Generated warpage is uniformly ~0.75 of GT width AND the shape is off (trimming
tails didn't help -> VAE smoothing toward the conditional mean, not an artifact).
This sweep attacks smoothing with more capacity + an extremely sharp posterior:

  feature_loss_weights = 1/1/1, Training_epochs = 2000, 2-GPU DDP per run.
  Base = L=2, mp 4/8/12/8/4, voronoi 2000/500, cov_rank=8, alpha_recon=100, mse,
  extreme_weight=0, beta_aux=1.0.

2x2 factorial: capacity {32/128, 64/192} x posterior_min_std {0.05, 0.001}.
  cell1 control      : 32/128, pmin=0.05   (cap-, sharp-)
  cell2 bigcap       : 64/192, pmin=0.05   (cap+, sharp-)
  cell3 bigcap+sharp : 64/192, pmin=0.001  (cap+, sharp+)
  cell4 sharp        : 32/128, pmin=0.001  (cap-, sharp+)  -- completes the 2x2
pmin=0.001 is a near-zero sigma_q floor -> sharpest decoder; watch for the
sigma_q-collapse failure ([PriorDiag] spread_ratio / Valid).

GPU7 is down. cells 1-3 use pairs (0,1)/(2,3)/(4,5); cell4 reuses (0,1) in a
second wave (after cell1). config_train1..4 (+ infer).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 2000

# n, train gpus (2-GPU DDP), infer gpu, vae_latent_dim, Latent_dim(model), posterior_min_std, tag
cells = [
    dict(n=1, gpus="0,1", igpu=0, ld=32, ldm=128, pmin=0.05,  tag="control"),
    dict(n=2, gpus="2,3", igpu=2, ld=64, ldm=192, pmin=0.05,  tag="bigcap"),
    dict(n=3, gpus="4,5", igpu=4, ld=64, ldm=192, pmin=0.001, tag="bigcap_sharp"),
    dict(n=4, gpus="0,1", igpu=0, ld=32, ldm=128, pmin=0.001, tag="sharp"),  # wave 2 (after cell1)
]


TRAIN_TMPL = """% ============================================================
% ANTI-SMOOTHING sweep cell {n}: {tag}
% vae_latent_dim={ld}, Latent_dim={ldm}, posterior_min_std={pmin}
% feature_loss_weights 1/1/1, Training_epochs 2000, 2-GPU DDP. Base L=2,
% voronoi 2000/500, cov_rank=8, mse, ew=0.
% ============================================================
model   MeshGraphNets-V
mode    train
gpu_ids {gpus}
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
use_checkpointing    False   # faster; set True if OOM (esp. bigcap cells 2/3)
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
alpha_recon      100
lambda_mmd       0.1
vae_graph_aware  True
beta_aux         1.0
posterior_min_std {pmin}

% Prior - gnn_e2e joint training; prior_family fm = conditional flow matching (default).
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_nll_weight         1.0
prior_fm_steps           20
prior_mp_layers          5
prior_hidden_dim         192
prior_temperature        1.0
"""

INFER_TMPL = """% Inference for cell {n}: {tag} (vae_latent_dim={ld}, Latent_dim={ldm})
model   MeshGraphNets-V
mode    inference
gpu_ids {igpu}
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
alpha_recon      100
lambda_mmd       0.1
vae_graph_aware  True

% Prior - sample z from the learned conditional prior p(z|graph)
prior_type               gnn_e2e
use_conditional_prior    True
prior_family             fm
prior_fm_steps           20
prior_mp_layers          5
prior_hidden_dim         192
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
    fields = dict(n=c["n"], gpus=c["gpus"], igpu=c["igpu"], ld=c["ld"],
                  ldm=c["ldm"], pmin=c["pmin"], tag=c["tag"], epochs=EPOCHS)
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
