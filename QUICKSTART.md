# QUICKSTART

This is the short operational guide for this checkout. For details, read
[README.md](README.md), [CLAUDE.md](CLAUDE.md), and
[docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

**Goal**: Hi-MGN-V models manufacturing spread — generating realistic diverse
samples from a trained VAE surrogate. Training with `use_vae True` produces a
checkpoint containing the jointly-trained conditional prior; set
`num_vae_samples` larger than the training set size at inference time.

## Fast Commands

```bash
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_train1.txt
python MeshGraphNets_main.py --config _b8_all_warpage_input/config_infer1_main.txt
```

`mode` is read from the config file. The CLI only selects the config path.

## Before Running

Training needs:

- `dataset_dir` exists.
- Every GPU in `gpu_ids` is visible.
- `edge_var` is `8`; the model rejects any other edge feature count.
- `modelpath` parent directory exists or can be created by the checkpoint save path.

Inference needs:

- `modelpath` exists and contains `normalization`.
- `infer_dataset` exists and has `data/{id}/nodal_data` plus `mesh_edge`.
- Architecture keys in the checkpoint match the model being loaded. Rollout applies
  `checkpoint['model_config']` over the inference config before building the model.

Quick check:

```bash
dir dataset
dir outputs
```

## Modes

| Mode | Use it for |
|------|------------|
| `train` | Train the simulator; with `use_vae True` the mesh-conditioned prior trains jointly. |
| `inference` | Autoregressive rollout. |

## Spread Modeling Config Checklist

For manufacturing spread modeling, verify these keys before training:

| Key | Recommended | Why |
|-----|-------------|-----|
| `use_vae` | `True` | VAE is required for spread sampling |
| `vae_graph_aware` | `True` | Type-conditional spread encoding |
| `lambda_mmd` | `0.1` | Low: preserve structured spread in z |
| `beta_aux` | `1.0` | High: prevent mode/posterior collapse |
| `posterior_min_std` | `0.05` | Floors sigma_q; prevents point-mass memorization |
| `vae_latent_dim` | `32` | Full capacity for spread representation |
| `prior_type` | `gnn_e2e` | Trains simulator + conditional prior in one run |
| `num_vae_samples` | > dataset size | Extrapolates spread at inference time |

## Active Prior Warning

Rollout samples z from the mesh-conditioned prior in the checkpoint (the
joint-trained submodule, or a legacy `conditional_prior_state_dict`), falling
back to `N(0, I)` when none exists — but only when `use_conditional_prior` is
still true after checkpoint `model_config` overrides the config. If a checkpoint
was saved with `use_conditional_prior False`, an inference config can be
overwritten. When debugging stochastic rollout, inspect the startup log line:

```text
VAE sampling: ... prior=conditional flow-matching prior
```

If it says `N(0, I)`, the conditional prior is not active.

## Config Gotchas

- Keys are lowercased by the parser.
- Full-line comments use `%`; inline comments use `#`.
- Booleans are parsed case-insensitively (`True`, `true`, `False`, `false` all work).
- Comma lists are stripped and parsed as numbers when possible.
- `gpu_ids 0,1` triggers DDP unless `parallel_mode model_split` is set.
- `message_passing_num` is ignored by the model when `use_multiscale True`; the
  V-cycle uses `mp_per_level`.
- `mp_per_level` must have `2 * multiscale_levels + 1` entries.
- `positional_features` increases node input size before normalization.
- Node type one-hot features are appended after node normalization.

## Outputs

| Output | Source |
|--------|--------|
| `outputs/*.pth` | Checkpoints with weights, optimizer/scheduler state, normalization, model_config, optional EMA |
| `outputs/<log_file_dir>` | Epoch logs from training |
| `outputs/test/...` | Per-epoch test reconstruction HDF5/PNG |
| `outputs/train/...` | Optional train-set reconstruction HDF5/PNG |
| `outputs/rollout/...` | Autoregressive rollout HDF5 |

## First Failure Checks

| Symptom | Check |
|---------|-------|
| `FileNotFoundError` for HDF5 | `dataset_dir` or `infer_dataset` path is wrong or missing. |
| Checkpoint missing normalization | Re-train or re-save with current training code. |
| Size mismatch on load | Checkpoint `model_config` disagrees with the model architecture. |
| Conditional prior not used | Check rollout startup log and checkpoint `model_config['use_conditional_prior']`. |
| DDP hang | Drop to one GPU, then retry with visible `gpu_ids`; the launcher chooses a free port automatically. |
| NaN/Inf or fp16 overflow | Keep `use_amp True`; current AMP path uses bfloat16, not fp16. |
