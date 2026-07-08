"""Plot 2-D scatter for every rollout_sample*.h5 file in a directory.

Each file is plotted as a 2-D scatter of node (x, y) reference positions
coloured by z_disp (channel 5 of nodal_data) at the final timestep.  The
output PNG is saved next to the .h5 with the same stem name.

Usage:
    python _b8_all_warpage_input/plot_rollout_meshes.py \\
        --rollout_dir outputs/b8_all/infer_train1_main

    # Parallel (8 workers):
    python _b8_all_warpage_input/plot_rollout_meshes.py \\
        --rollout_dir outputs/b8_all/infer_train1_main --jobs 8
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Z_DISP_CHANNEL = 5  # nodal_data layout: [x,y,z, x_disp,y_disp,z_disp, ...]


def _plot_one(h5_path: str, out_png: str, dpi: int) -> tuple:
    try:
        with h5py.File(h5_path, "r") as f:
            sample_id = next(iter(f["data"].keys()))
            nd = f[f"data/{sample_id}/nodal_data"][:]  # [features, timesteps, nodes]

        x = np.asarray(nd[0, 0, :], dtype=np.float32)         # reference x
        y = np.asarray(nd[1, 0, :], dtype=np.float32)         # reference y
        c = np.asarray(nd[Z_DISP_CHANNEL, -1, :], dtype=np.float32)  # z_disp at final step

        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(x, y, c=c, s=0.5, cmap="coolwarm", rasterized=True)
        plt.colorbar(sc, ax=ax, label="z_disp")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(Path(h5_path).stem, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        return (h5_path, True, None)
    except Exception as exc:
        return (h5_path, False, str(exc))


def _worker(args):
    matplotlib.use("Agg")
    return _plot_one(*args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rollout_dir", required=True,
        help="Directory containing rollout_sample*.h5 files.",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel worker count (default 1).",
    )
    parser.add_argument(
        "--dpi", type=int, default=80,
        help="Plot DPI — lower is faster/smaller (default 80).",
    )
    args = parser.parse_args()

    h5_files = sorted(glob.glob(os.path.join(args.rollout_dir, "rollout_sample*.h5")))
    if not h5_files:
        print(f"No rollout_sample*.h5 files found in {args.rollout_dir}")
        sys.exit(0)

    print(f"Found {len(h5_files):,} rollout files — plotting with {args.jobs} worker(s)...")

    tasks = [(fp, str(Path(fp).with_suffix(".png")), args.dpi) for fp in h5_files]

    if args.jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            results = pool.map(_worker, tasks)
    else:
        results = []
        report_every = max(1, len(tasks) // 20)
        for i, t in enumerate(tasks):
            results.append(_plot_one(*t))
            if (i + 1) % report_every == 0 or i + 1 == len(tasks):
                print(f"  {i + 1:,}/{len(tasks):,}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"Done: {n_ok:,} plotted, {n_fail:,} failed.")
    for fp, ok, err in results:
        if not ok:
            print(f"  FAILED {os.path.basename(fp)}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
