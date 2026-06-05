"""Compare histograms of z_disp spread (max - min) between the ground-truth
eval dataset and the generated rollout outputs.

Spread metric (one scalar per realization):
    spread = max(z_disp_at_nodes) - min(z_disp_at_nodes)   at the final timestep

For the ground-truth eval HDF5 (e.g. dataset/eval_b8_main.h5):
    one scalar per sample under /data/{sample_id}/nodal_data[5, -1, :].

For the generated rollouts (rollout_sample*.h5 inside <rollout_dir>):
    one scalar per file, computed from /data/{sample_id}/nodal_data[5, -1, :].
    With num_vae_samples=5000 there are 5000 scalars per ground-truth sample.

Usage:
    python compare_histograms.py \\
        --eval_dataset dataset/eval_b8_main.h5 \\
        --rollout_dir  outputs/b8_all/infer32_1_main \\
        --output       outputs/b8_all/infer32_1_main/histogram_compare.png
"""

import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_DISP_CHANNEL = 5  # nodal_data layout: [x,y,z, x_disp,y_disp,z_disp, ...]


def _sample_spread(nodal_data) -> float:
    """max-min of z_disp across all nodes at the final timestep."""
    z = nodal_data[Z_DISP_CHANNEL, -1, :]  # [nodes]
    z = np.asarray(z, dtype=np.float64)
    return float(z.max() - z.min())


def _eval_spreads(h5_path: str) -> np.ndarray:
    """Return one spread value per sample in the eval HDF5."""
    spreads = []
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"No /data group in {h5_path}")
        for sample_id in f["data"].keys():
            spreads.append(_sample_spread(f[f"data/{sample_id}/nodal_data"]))
    return np.asarray(spreads, dtype=np.float64)


def _rollout_spreads(rollout_dir: str) -> np.ndarray:
    """Return one spread value per generated rollout file."""
    files = sorted(glob.glob(os.path.join(rollout_dir, "rollout_sample*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No rollout_sample*.h5 files found in {rollout_dir}"
        )
    print(f"Found {len(files):,} rollout files in {rollout_dir}")
    spreads = np.empty(len(files), dtype=np.float64)
    report_every = max(1, len(files) // 20)
    for i, fp in enumerate(files):
        with h5py.File(fp, "r") as f:
            # Each rollout file contains exactly one sample under /data
            sample_id = next(iter(f["data"].keys()))
            spreads[i] = _sample_spread(f[f"data/{sample_id}/nodal_data"])
        if (i + 1) % report_every == 0 or i + 1 == len(files):
            print(f"  loaded {i + 1:,}/{len(files):,} files")
    return spreads


def _stats(col: np.ndarray) -> dict:
    return {
        "mean": float(col.mean()),
        "std": float(col.std()),
        "min": float(col.min()),
        "max": float(col.max()),
        "n": int(col.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval_dataset", required=True,
        help="Path to the ground-truth eval HDF5 (e.g. dataset/eval_b8_main.h5).",
    )
    parser.add_argument(
        "--rollout_dir", required=True,
        help="Directory containing generated rollout_sample*.h5 files.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output figure path. Defaults to <rollout_dir>/histogram_compare.png.",
    )
    parser.add_argument(
        "--bins", type=int, default=60,
        help="Number of histogram bins (default 60).",
    )
    parser.add_argument(
        "--clip_quantile", type=float, default=0.0,
        help="Symmetric quantile clip for the binning range, e.g. 0.001 trims "
             "the 0.1%% tails. 0 = no clipping (default).",
    )
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.rollout_dir, "histogram_compare.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground-truth eval dataset: {args.eval_dataset}")
    gt = _eval_spreads(args.eval_dataset)
    print(f"  GT spread values (1 per sample): {gt.size:,}")
    if gt.size == 0:
        raise RuntimeError("Eval dataset has no samples in /data.")

    print(f"Loading generated rollouts from: {args.rollout_dir}")
    gen = _rollout_spreads(args.rollout_dir)
    print(f"  generated spread values (1 per VAE sample): {gen.size:,}")

    if args.clip_quantile > 0:
        q = args.clip_quantile
        lo = float(min(np.quantile(gt, q), np.quantile(gen, q)))
        hi = float(max(np.quantile(gt, 1 - q), np.quantile(gen, 1 - q)))
    else:
        lo = float(min(gt.min(), gen.min()))
        hi = float(max(gt.max(), gen.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1e-8
    bins = np.linspace(lo, hi, args.bins + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        gt, bins=bins, density=True, alpha=0.55,
        label=f"eval dataset (n={gt.size:,})", color="steelblue",
    )
    ax.hist(
        gen, bins=bins, density=True, alpha=0.55,
        label=f"generated (n={gen.size:,})", color="darkorange",
    )
    gt_s = _stats(gt)
    gen_s = _stats(gen)
    ax.set_title(
        "z_disp spread (max - min) per realization, final timestep\n"
        f"GT  mu={gt_s['mean']:.3e}  sigma={gt_s['std']:.3e}  "
        f"[{gt_s['min']:.3e}, {gt_s['max']:.3e}]\n"
        f"Gen mu={gen_s['mean']:.3e}  sigma={gen_s['std']:.3e}  "
        f"[{gen_s['min']:.3e}, {gen_s['max']:.3e}]"
    )
    ax.set_xlabel("max(z_disp) - min(z_disp)")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"eval:     {args.eval_dataset}\n"
        f"rollouts: {args.rollout_dir}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
