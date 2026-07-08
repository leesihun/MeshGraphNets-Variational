# Performance Optimizations

Changes applied 2026-07-09. Documented here so the same improvements can be
ported to sibling repos (e.g. `MeshGraphNets`).

---

## 1 — EdgeBlock split-linear (`model/blocks.py`)

**What:** `EdgeBlock.forward` used to gather sender and receiver node features to
`[E, D]` each, concatenate with `edge_attr` to `[E, 3D]`, then run
`Linear(3D → D)` on the result.  
The new path projects node features at `[N, D]` scale *before* gathering:

```python
# Linear(cat(s, r, e)) = s @ W_s + r @ W_r + e @ W_e  — mathematically identical
first = self.net[0]   # nn.Linear(3D, D)
D = node_attr.shape[-1]
h_s = F.linear(node_attr, first.weight[:, :D])
h_r = F.linear(node_attr, first.weight[:, D:2*D])
h_e = F.linear(edge_attr, first.weight[:, 2*D:], first.bias)
edge_attr = self.net[1:](h_s[senders_idx] + h_r[receivers_idx] + h_e)
```

**Why it helps:**  
With `E ≈ 6 N` (bidirectional shell mesh), the old path allocated `[E, 3D]` in
every GnBlock forward *and* saved it for backward. With 36 processor blocks +
VAE + prior blocks, that is ~46 × `[E, 3D]` autograd saves per step — the
single largest activation memory consumer. The new path never materialises the
`[E, 3D]` tensor; the two node projections use `[N, D]` instead. VRAM savings
are typically ~40–60% of total activation memory.

**State-dict compatibility:** The weight key `net.0.weight` / `net.0.bias` are
unchanged. Existing checkpoints load without modification.

**Port notes:**  
- Requires `import torch.nn.functional as F` in `blocks.py`.  
- Only applicable to `EdgeBlock` (where `E ≫ N`). `NodeBlock` and
  `HybridNodeBlock` inputs are `[N, D]` so no savings there.  
- Assumes the first element of `self.net` is `nn.Linear(3D, D)`. This is always
  true for GnBlock-constructed EdgeBlocks; verify if EdgeBlock is used elsewhere
  with a different MLP.

---

## 2 — MMD cdist hoisted outside bandwidth loop (`model/vae.py`)

**What:** The 5-kernel RBF MMD loss computed three `torch.cdist` calls *inside*
the `for sigma in kernel_sigmas` loop — 15 cdist calls per batch. The three
distance matrices are now computed once outside the loop and reused:

```python
xx = torch.cdist(z_posterior, z_posterior).pow(2)
yy = torch.cdist(z_prior,     z_prior    ).pow(2)
xy = torch.cdist(z_posterior, z_prior    ).pow(2)
for sigma in kernel_sigmas:
    two_sigma_sq = 2.0 * sigma * sigma
    mmd_total += exp(-xx / two_sigma_sq).mean() + exp(-yy /...).mean() - 2*exp(-xy/...).mean()
```

**Port notes:** Change is entirely inside `GNNVariationalEncoder.mmd_loss`. The
cdist matrices are `[B, B]` (B = batch size) so memory impact is negligible.

---

## 3 — CRPS vectorized (`training_profiles/training_loop.py`)

**What:** The fair-CRPS pairwise term used a Python `for s in range(S)` loop:

```python
# Old — O(S) Python iterations, each with a [S, N, F] temporary
pair_sum = torch.zeros_like(accuracy)
for s in range(S):
    pair_sum += (samples[s].unsqueeze(0) - samples).abs().sum(dim=0)
spread = pair_sum / (2.0 * S * (S - 1))

# New — fully vectorized, one [S, S, N, F] allocation
spread = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().sum(dim=[0, 1]) / (2.0 * S * (S - 1))
```

**Port notes:** Memory peak is `S² × N × F` floats. With `S=8`, `N≈96k`, `F=3`
this is ~74 MB; fine for GPU eval under `torch.no_grad()`. If mesh is much
larger than the b8 panel (>200k nodes × 8 batch), consider a chunked loop.

---

## 4 — GPU loss accumulators in train_epoch (`training_profiles/training_loop.py`)

**What:** Per-batch `.item()` calls in `train_epoch` forced CPU↔GPU syncs after
every backward, preventing the CPU from pre-launching the next batch's H2D copy
and kernels. The opt-loss, MMD, aux, KL-anchor, and prior-loss scalars are now
accumulated as GPU tensors; `.item()` is called exactly once per metric at epoch
end.

**Key change pattern:**
```python
# Old (syncs every batch)
total_opt_loss_sum += loss.item() * batch_loss_count
total_mmd_sum += float(mmd_loss_val)

# New (no sync per batch)
total_opt_loss_gpu += loss.detach().float() * batch_loss_count
total_mmd_gpu      += mmd_loss_val.detach().float()

# End of epoch (one sync per metric)
result['total_mean'] = total_opt_loss_gpu.item() / total_loss_count
result['mmd_mean']   = total_mmd_gpu.item() / mmd_count
```

**Port notes:** Add GPU zero-tensor initialisation (`torch.zeros((), device=device,
dtype=torch.float32)`) before the batch loop for each accumulator. The
`total_loss_sum` and `total_loss_count` remain Python floats because
`_loss_from_errors` already calls `.item()` on `loss_sum` (one unavoidable sync
per batch for the recon loss used in `backward`).

---

## 5 — _evaluate_epoch display throttle + GPU accumulators (`training_profiles/training_loop.py`)

**What:** `_evaluate_epoch` (called by both `validate_epoch` and
`evaluate_vae_posterior_epoch`) called `.item()` on `recon_loss`, `opt_loss`,
and `mmd_loss_val` for the tqdm postfix on **every** batch. That is 3 GPU syncs
per validation batch — more than training, which already throttled to 10-batch
intervals.

Fix: (a) throttle `pbar.set_postfix()` to every 10 batches, and (b) accumulate
`opt_loss` and `mmd` as GPU tensors with one `.item()` at epoch end (same
pattern as §4 above).

**Port notes:** Add `for batch_idx, graph in enumerate(pbar)` (was `for graph in
pbar` without index). Guard `pbar.set_postfix(...)` with `if batch_idx % 10 == 0`.

---

## 6 — DDP no_sync() for gradient accumulation (`training_profiles/training_loop.py`)

**What:** With `grad_accum_steps > 1` (or `0` = full-epoch), DDP was
all-reducing gradients after every `backward()` call, even mid-accumulation
window. Added `model.no_sync()` context on non-step batches:

```python
import contextlib

is_step_batch = (batch_idx + 1) % actual_accum == 0 or (batch_idx == total_batches - 1)
sync_ctx = (contextlib.nullcontext()
            if is_step_batch or not hasattr(model, 'no_sync')
            else model.no_sync())
with sync_ctx:
    scaled_loss.backward()
```

**Port notes:**  
- `hasattr(model, 'no_sync')` is False for non-DDP models, so the single-GPU
  path is unchanged.  
- When `grad_accum_steps 1` (default), `is_step_batch` is always True, so
  `no_sync()` is never invoked — no change in default behavior.  
- The `is_last_batch` variable previously defined after the backward was folded
  into `is_step_batch` to avoid a forward-reference.

---

## 7 — Checkpoint saved only on best/last epoch

**What (single_training.py, distributed_training.py):** Checkpoint was written
every epoch regardless of validation result. This synchronously serialises model
+ optimizer + EMA (hundreds of MB) to disk, which can be a significant fraction
of wall time on fast-epoch runs.

New behaviour: checkpoint is written only when `valid_loss < best_valid_loss`
(new best) **or** on the last epoch.

```python
# single_training.py
best_valid_loss = float('inf')   # before epoch loop

# inside epoch loop
last_epoch = (epoch == total_epochs - 1)
is_best    = do_val and valid_loss < best_valid_loss
if is_best or last_epoch:
    if is_best:
        best_valid_loss = valid_loss
    save_checkpoint(...)
    print(f"  -> Model saved at epoch {epoch}: {', '.join(reason)}")
```

**Port notes:**  
- Add `best_valid_loss = float('inf')` before the epoch loop in both
  `single_worker` and `_train_worker_inner`.  
- In DDP, validation runs every epoch so `do_val` is always True; the logic
  simplifies to just `is_best or last_epoch`.  
- `last_valid_loss` (used in the interrupted-training message) is updated only
  when a checkpoint is actually saved.

---

## 8 — TF32 + expandable memory segments (`MeshGraphNets_main.py`)

**What:** Two global switches added at module level in `MeshGraphNets_main.py`:

```python
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**TF32:** Uses 10-bit mantissa for float32 matmuls (vs 23-bit), giving ~3×
throughput on Ampere/Hopper with negligible precision loss. Primarily benefits
the FM-prior velocity net and MMD loss (both run in fp32 outside autocast).
On A100/H100/RTX 3090+, enabling this is unambiguously correct for training.

**expandable_segments:** Prevents OOM-with-free-memory when the CUDA allocator
can't merge fragmented blocks. Particularly valuable with variable-size batched
graphs where the high-watermark reserved block is stranded after a large batch.

**Port notes:** Set these in the entry-point script before any CUDA tensor is
allocated. No other code changes needed.

---

## 9 — Validation DataLoader prefetch_factor 1 → 2

**What:** `prefetch_factor=1` on the val loader meant the data pipeline stalled
waiting for the next batch while the GPU was idle. Raised to 2 (matching the
stable-known-safe setting from the environment).

Changed in both `single_training.py` and `distributed_training.py`.

**Port notes:** Keep train at 4 (already there). Do not raise val above 2 on
this machine without testing — the prior safe setting was `prefetch_factor=2`
with `spawn` multiprocessing context.

---

## Summary table

| # | File | Change | Primary benefit |
|---|------|--------|-----------------|
| 1 | `model/blocks.py` | Split-linear EdgeBlock | VRAM −40–60% activation |
| 2 | `model/vae.py` | cdist hoisted out of σ loop | 5× fewer cdist calls |
| 3 | `training_loop.py` | CRPS vectorized | Eliminate Python loop |
| 4 | `training_loop.py` | GPU loss accumulators in train | Fewer CPU↔GPU syncs per batch |
| 5 | `training_loop.py` | Eval display throttle + GPU accum | 3× fewer syncs in validation |
| 6 | `training_loop.py` | DDP no_sync() during accumulation | No wasted all-reduces mid-window |
| 7 | `single_training.py`, `distributed_training.py` | Best+last checkpoint only | Eliminates per-epoch disk write |
| 8 | `MeshGraphNets_main.py` | TF32 + expandable_segments | Free fp32 throughput + fragmentation |
| 9 | `single_training.py`, `distributed_training.py` | val prefetch_factor 1→2 | Reduce loader stall |
