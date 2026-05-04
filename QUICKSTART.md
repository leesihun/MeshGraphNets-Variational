# QUICKSTART (for AI agents)

This file is the single entry point for an AI/Claude session that needs to run training or inference in this repo. Read top-to-bottom; everything is action-first. For depth see [README.md](README.md), [CLAUDE.md](CLAUDE.md), [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md), [docs/](docs/).

---

## TL;DR

```bash
# Train (DDP on GPUs 0,1 → writes outputs/warpage1.pth)
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt

# Infer / autoregressive rollout (single GPU → writes outputs/rollout/*.h5)
python MeshGraphNets_main.py --config _warpage_input/config_infer4.txt
```

`mode` (`train` vs `inference`) is set **inside** the config, not on the CLI. Single vs multi-GPU is auto-detected from the `gpu_ids` field.

---

## Repository at a glance

Purpose: orient yourself before running anything.

| Item | Path | Status |
|------|------|--------|
| Entry point | [MeshGraphNets_main.py](MeshGraphNets_main.py) | in repo |
| Default training config | [_warpage_input/config_train5.txt](_warpage_input/config_train5.txt) | in repo |
| Default inference config (matches train5 checkpoint) | [_warpage_input/config_infer4.txt](_warpage_input/config_infer4.txt) | in repo |
| Older inference config (uses warpage0.pth) | [_warpage_input/config_infer3.txt](_warpage_input/config_infer3.txt) | in repo |
| Training dataset | `dataset/b8_train.h5` | **NOT in repo — must be supplied/built** |
| Inference initial-condition dataset | `dataset/infer_dataset_b8_test.h5` | **NOT in repo — built from training set** |
| Trained checkpoint | `outputs/warpage1.pth` | **produced by training** |
| Rollout output dir | `outputs/rollout/` | created on first inference run |

---

## Prerequisites checklist

Purpose: bail out early if a required artifact is missing.

Before training:
- [ ] `dataset/b8_train.h5` exists (otherwise build via [build_dataset.py](build_dataset.py))
- [ ] CUDA visible for every GPU listed in `gpu_ids` (e.g. `0,1` requires 2 GPUs)
- [ ] Python deps installed (see [README.md#installation](README.md#installation))

Before inference:
- [ ] `outputs/warpage1.pth` exists (or change `modelpath` in the config)
- [ ] `dataset/infer_dataset_b8_test.h5` exists (build via [dataset/generate_inference_dataset.py](dataset/generate_inference_dataset.py))
- [ ] `dataset/b8_train.h5` exists — infer configs reference it for normalization stat lookups
- [ ] CUDA visible for the GPU id listed in `gpu_ids` (default `1`)

Quick existence check:

```bash
ls dataset/b8_train.h5 dataset/infer_dataset_b8_test.h5 outputs/warpage1.pth
```

---

## Run training

Purpose: produce a checkpoint at `modelpath`.

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train5.txt
```

Produces:
- `outputs/warpage1.pth` (checkpoint with weights + EMA + normalization + GMM params)
- `train5.log` (per-epoch loss; tail with [misc/plot_loss.py](misc/plot_loss.py) or `misc/plot_loss_realtime.py`)
- `outputs/test/` (per-epoch test visualizations every `test_interval` epochs)

Knobs to tweak in [config_train5.txt](_warpage_input/config_train5.txt):

| Key | Effect |
|-----|--------|
| `gpu_ids` | `0` = single GPU; `0,1` = DDP via `mp.spawn`; `-1` = CPU |
| `batch_size` | per-GPU batch size |
| `training_epochs` | total epochs |
| `modelpath` | output checkpoint path |
| `dataset_dir` | training HDF5 path |
| `use_amp` | bfloat16 mixed precision (keep True on Ampere+) |
| `use_vae` | enable conditional MMD-VAE branch |
| `use_multiscale` | enable Voronoi V-cycle processor |

---

## Run inference

Purpose: autoregressive rollout from initial conditions, writing one HDF5 per (sample, latent draw).

```bash
python MeshGraphNets_main.py --config _warpage_input/config_infer4.txt
```

Produces:
- `outputs/rollout/rollout_sample{id}_steps{N}.h5` — one file per VAE draw per sample

Knobs:

| Key | Effect |
|-----|--------|
| `gpu_ids` | single GPU id; multi-GPU inference not supported |
| `infer_timesteps` | rollout horizon |
| `num_vae_samples` | number of independent latent draws (1000 = 1000 trajectories per input sample) |
| `modelpath` | which checkpoint to load |
| `infer_dataset` | HDF5 with initial-condition (t=0) samples |

To use the older `warpage0.pth` checkpoint, run with [config_infer3.txt](_warpage_input/config_infer3.txt) instead.

---

## Config gotchas

Purpose: things that bite agents who skim.

- `mode` is **inside** the config (`mode train` or `mode inference`), not a CLI flag.
- `--config` defaults to `config.txt` in CWD if omitted — almost certainly not what you want.
- `gpu_ids` accepts `-1` (CPU), a single int (`0`), or a comma list (`0,1`) for DDP. DDP launches via `torch.multiprocessing.spawn` — **do not** call `torchrun`.
- Comments: `%` for a full-line comment, `#` for inline. Keys are case-insensitive (`Latent_dim` == `latent_dim`).
- At inference, `input_var`, `output_var`, `latent_dim`, `message_passing_num`, multiscale settings must match the training checkpoint. `vae_latent_dim` and `num_vae_samples` may differ.
- Inference configs reference `dataset_dir` (the **training** HDF5) as well as `infer_dataset` — normalization stats and split metadata are read from the training file.
- bfloat16, not float16: `scatter_add` overflows in fp16 on typical FEA mesh sizes.
- `HDF5_USE_FILE_LOCKING=FALSE` is set in code — no env-var setup needed.

---

## Dataset format — minimum to know

Purpose: enough to load a sample without mistakes. Full schema in [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).

Per-sample HDF5 layout:

```
data/{sample_id}/nodal_data    float32  [num_features, num_timesteps, num_nodes]
data/{sample_id}/mesh_edge     int64    [2, num_edges]   (unidirectional; code mirrors it)
```

Default `nodal_data` feature indices:

| Index | Field | Notes |
|-------|-------|-------|
| 0–2 | `x_coord, y_coord, z_coord` | reference (undeformed) positions; **not** in `input_var` |
| 3–5 | `x_disp, y_disp, z_disp` | per-timestep displacement (mm) |
| 6 | `stress` | von Mises (MPa) |
| 7 | `part_no` | integer part ID; only used if `use_node_types True` |

So `input_var 3` → use channels 3–5; `input_var 4` → channels 3–6.

Inference HDF5 = same schema but `num_timesteps == 1` (initial condition only). Rollout output adds back `[x, y, z, ...predicted channels..., part_no]`. See [README.md#data-format](README.md#data-format) for the variants.

---

## Building a dataset from scratch

Purpose: when `b8_train.h5` is missing.

```bash
python build_dataset.py                            # ANSYS .inp + CSV → training HDF5 (edit paths at top of file)
python dataset/generate_inference_dataset.py       # training HDF5 → inference HDF5 (t=0 only, default 10 random samples)
python dataset/reduce_dataset.py                   # subsample an existing HDF5
```

---

## Outputs cheat sheet

| Path | Meaning |
|------|---------|
| `outputs/*.pth` | checkpoints (weights + EMA + normalization stats + optional GMM params) |
| `outputs/rollout/rollout_sample*_steps*.h5` | autoregressive rollout results |
| `outputs/test/` | per-epoch test visualizations (`.png`/`.h5`) |
| `train5.log`, `train4.log`, `train3.log` | per-epoch loss/val/test logs (one per config) |
| `outputs/train/` | DDP training scratch (created automatically) |

Visualize a rollout HDF5 as a GIF:

```bash
python animate_h5.py
```

---

## If it fails

Purpose: first thing to check before debugging deeper.

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError ... b8_train.h5` | dataset not built | run [build_dataset.py](build_dataset.py) or supply the HDF5 |
| `FileNotFoundError ... warpage1.pth` | no checkpoint yet | train first, or change `modelpath` to an existing checkpoint |
| `RuntimeError: NCCL ...` / DDP hang | port collision or partial GPU visibility | drop to single GPU (`gpu_ids 0`) and confirm before retrying DDP |
| Loss = NaN / Inf | float16 instead of bfloat16 | confirm `use_amp True` and **don't** patch to fp16 — `scatter_add` overflows |
| Inference mismatch (`size mismatch ...`) | config disagrees with checkpoint | match `input_var`, `output_var`, `latent_dim`, `message_passing_num`, `multiscale_levels`, `mp_per_level` to training config |
| OOM during training | mesh too large for batch | lower `batch_size`, keep `use_checkpointing True`, or reduce `latent_dim` |
| HDF5 multi-worker errors | (already handled) | confirm `HDF5_USE_FILE_LOCKING=FALSE` is exported by `mesh_dataset.py` — no action needed |

---

## Where to look for more

- [README.md](README.md) — full prose docs, installation, architecture diagrams.
- [CLAUDE.md](CLAUDE.md) — engineering conventions, file map, important implementation details.
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) — full HDF5 schema with code snippets for loading.
- [docs/](docs/) — deep dives: multiscale coarsening, world edges, VRAM optimization, architecture.
