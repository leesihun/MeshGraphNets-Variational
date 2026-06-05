"""Compare histograms of generated rollout outputs vs. the ground-truth inference dataset.

Pools per-node displacement values (x_disp, y_disp, z_disp) at the final
timestep across all samples and plots a histogram per feature for both the
ground-truth inference dataset and the generated rollout files. Useful for
verifying that the VAE samples reproduce the spread of the manufacturing
dataset.

Both HDF5 files follow the layout in dataset/DATASET_FORMAT.md:
    data/{sample_id}/nodal_data  shape [features, timesteps, nodes]
where channels 0..2 are reference coordinates and channels 3..5 are the
displacement (x_disp, y_disp, z_disp).

Usage:
    python compare_histograms.py \\
        --infer_dataset dataset/infer_b8_main.h5 \\
        --rollout_dir   outputs/b8_all/infer32_1_main \\
        --output        outputs/b8_all/infer32_1_main/histogram_compare.png
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

FEATURE_NAMES = ["x_disp", "y_disp", "z_disp"]
NUM_FEATURES = len(FEATURE_NAMES)


def _load_final_disp(h5_path: str) -> np.ndarray:
    """Return [num_samples * num_nodes, 3] of final-timestep displacement
    values from an HDF5 file in DATASET_FORMAT.md layout."""
    chunks = []
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32)
        for sample_id in f["data"].keys():
            nodal = f[f"data/{sample_id}/nodal_data"]
            arr = nodal[3:3 + NUM_FEATURES, -1, :]  # [3, nodes]
            chunks.append(arr.T.astype(np.float32))  # [nodes, 3]
    if not chunks:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _gather_rollouts(rollout_dir: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(rollout_dir, "rollout_sample*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No rollout_sample*.h5 files found in {rollout_dir}"
        )
    print(f"Found {len(files):,} rollout files in {rollout_dir}")
    chunks = []
    report_every = max(1, len(files) // 20)
    for i, fp in enumerate(files):
        chunks.append(_load_final_disp(fp))
        if (i + 1) % report_every == 0 or i + 1 == len(files):
            print(f"  loaded {i + 1:,}/{len(files):,} files")
    return np.concatenate(chunks, axis=0)


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
        "--infer_dataset", required=True,
        help="Path to the ground-truth inference dataset HDF5.",
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
        "--bins", type=int, default=100,
        help="Number of histogram bins per feature (default 100).",
    )
    parser.add_argument(
        "--clip_quantile", type=float, default=0.0,
        help="Symmetric quantile clip for the binning range, e.g. 0.001 trims "
             "the 0.1%% tails. 0 = no clipping (default).",
    )
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.rollout_dir, "histogram_compare.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading ground-truth infer dataset: {args.infer_dataset}")
    gt = _load_final_disp(args.infer_dataset)
    print(f"  ground-truth values per feature: {gt.shape[0]:,}")
    if gt.shape[0] == 0:
        raise RuntimeError("Ground-truth dataset has no samples in /data.")

    print(f"Loading generated rollouts from: {args.rollout_dir}")
    gen = _gather_rollouts(args.rollout_dir)
    print(f"  generated values per feature:   {gen.shape[0]:,}")

    fig, axes = plt.subplots(1, NUM_FEATURES, figsize=(5 * NUM_FEATURES, 4.5))
    if NUM_FEATURES == 1:
        axes = [axes]

    for ch, name in enumerate(FEATURE_NAMES):
        gt_col = gt[:, ch]
        gen_col = gen[:, ch]

        if args.clip_quantile > 0:
            q = args.clip_quantile
            lo = float(min(np.quantile(gt_col, q), np.quantile(gen_col, q)))
            hi = float(max(np.quantile(gt_col, 1 - q), np.quantile(gen_col, 1 - q)))
        else:
            lo = float(min(gt_col.min(), gen_col.min()))
            hi = float(max(gt_col.max(), gen_col.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            hi = lo + 1e-8
        bins = np.linspace(lo, hi, args.bins + 1)

        ax = axes[ch]
        ax.hist(
            gt_col, bins=bins, density=True, alpha=0.55,
            label=f"infer dataset (n={gt_col.size:,})", color="steelblue",
        )
        ax.hist(
            gen_col, bins=bins, density=True, alpha=0.55,
            label=f"generated (n={gen_col.size:,})", color="darkorange",
        )

        gt_s = _stats(gt_col)
        gen_s = _stats(gen_col)
        ax.set_title(
            f"{name}\n"
            f"GT  mu={gt_s['mean']:+.3e}  sigma={gt_s['std']:.3e}\n"
            f"Gen mu={gen_s['mean']:+.3e}  sigma={gen_s['std']:.3e}"
        )
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Histogram comparison (final timestep, all nodes pooled)\n"
        f"infer:    {args.infer_dataset}\n"
        f"rollouts: {args.rollout_dir}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()
