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


def _rollout_spread_one(h5_path: str) -> float:
    with h5py.File(h5_path, "r") as f:
        # Each rollout file contains exactly one sample under /data
        sample_id = next(iter(f["data"].keys()))
        return _sample_spread(f[f"data/{sample_id}/nodal_data"])


def _rollout_spreads(rollout_dir: str, jobs: int = 1) -> np.ndarray:
    """Return one spread value per generated rollout file."""
    files = sorted(glob.glob(os.path.join(rollout_dir, "rollout_sample*.h5")))
    if not files:
        raise FileNotFoundError(
            f"No rollout_sample*.h5 files found in {rollout_dir}"
        )
    jobs = max(1, int(jobs))
    print(f"Found {len(files):,} rollout files in {rollout_dir}")
    print(f"Loading rollout spreads with {jobs} worker(s)")
    spreads = np.empty(len(files), dtype=np.float64)
    report_every = max(1, len(files) // 20)
    if jobs > 1:
        import multiprocessing
        chunksize = max(1, len(files) // (jobs * 8))
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            for i, spread in enumerate(pool.imap(_rollout_spread_one, files, chunksize=chunksize)):
                spreads[i] = spread
                if (i + 1) % report_every == 0 or i + 1 == len(files):
                    print(f"  loaded {i + 1:,}/{len(files):,} files")
    else:
        for i, fp in enumerate(files):
            spreads[i] = _rollout_spread_one(fp)
            if (i + 1) % report_every == 0 or i + 1 == len(files):
                print(f"  loaded {i + 1:,}/{len(files):,} files")
    return spreads


def _skew(x: np.ndarray) -> float:
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / (s ** 3 + 1e-12))


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 = Gaussian; >0 = heavy tails / more outliers)."""
    m, s = x.mean(), x.std()
    return float(((x - m) ** 4).mean() / (s ** 4 + 1e-12) - 3.0)


def _stats(col: np.ndarray) -> dict:
    col = np.asarray(col, dtype=np.float64)
    p1, p5, p50, p95, p99 = np.quantile(col, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "mean": float(col.mean()), "std": float(col.std()),
        "min": float(col.min()), "max": float(col.max()),
        "p1": float(p1), "p5": float(p5), "p50": float(p50),
        "p95": float(p95), "p99": float(p99),
        "skew": _skew(col), "kurt": _kurtosis(col),
        "n": int(col.size),
    }


def _crps(forecast: np.ndarray, obs: np.ndarray) -> float:
    """CRPS of the generated ensemble vs GT observations (lower = better; a proper
    scoring rule that rewards both calibration and sharpness, tail-sensitive).
    Estimator  mean_y E|X-y| - 0.5 E|X-X'|  in O((n+m) log n)."""
    f = np.sort(np.asarray(forecast, dtype=np.float64))
    y = np.asarray(obs, dtype=np.float64)
    n = f.size
    if n == 0 or y.size == 0:
        return float("nan")
    csum = np.concatenate([[0.0], np.cumsum(f)])
    idx = np.searchsorted(f, y, side="right")            # # forecast <= y
    e_abs = (idx * y - csum[idx]) + ((csum[n] - csum[idx]) - (n - idx) * y)
    term1 = (e_abs / n).mean()                           # mean_y E|X-y|
    i = np.arange(n)
    term2 = (2.0 / n ** 2) * np.sum((2 * i - n + 1) * f)  # E|X-X'|
    return float(term1 - 0.5 * term2)


def _compare(gt: np.ndarray, gen: np.ndarray) -> dict:
    """Cross-distribution metrics, tail-focused — outliers (extreme warpage) matter."""
    gt = np.asarray(gt, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    gt_p99, gt_p1 = np.quantile(gt, 0.99), np.quantile(gt, 0.01)
    out = {
        # tail coverage ratios: 1.0 = gen reaches GT's extremes; <1 = too timid.
        "max_cov": float(gen.max() / gt.max()) if gt.max() else float("nan"),
        "min_cov": float(gen.min() / gt.min()) if gt.min() else float("nan"),
        "p99_cov": float(np.quantile(gen, 0.99) / gt_p99) if gt_p99 else float("nan"),
        "p1_cov":  float(np.quantile(gen, 0.01) / gt_p1) if gt_p1 else float("nan"),
        # outlier rate: ~1% each if gen matches GT's tails; <<1% = misses outliers.
        "gen_frac_above_gt_p99": float((gen > gt_p99).mean()),
        "gen_frac_below_gt_p1":  float((gen < gt_p1).mean()),
    }
    out["crps"] = _crps(gen, gt)        # gen ensemble vs GT observations
    try:
        from scipy import stats as sps
        out["wasserstein"] = float(sps.wasserstein_distance(gt, gen))
        out["ks"] = float(sps.ks_2samp(gt, gen).statistic)
    except Exception:
        out["wasserstein"] = float("nan")
        out["ks"] = float("nan")
    return out


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
        help="Symmetric quantile clip for the binning RANGE only (visual zoom), "
             "e.g. 0.001 trims the 0.1%% tails. 0 = no clipping (default).",
    )
    parser.add_argument(
        "--trim_quantile", type=float, default=0.0,
        help="EXCLUDE the bottom and top q fraction of EACH distribution before "
             "stats AND histogram (e.g. 0.02 = drop <2%% and >98%%). Compares the "
             "central bulk, ignoring artifact/spurious tails. 0 = off (default).",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel worker count for generated rollout HDF5 loading (default 1).",
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
    gen = _rollout_spreads(args.rollout_dir, jobs=args.jobs)
    print(f"  generated spread values (1 per VAE sample): {gen.size:,}")

    if args.trim_quantile > 0:
        q = args.trim_quantile

        def _trim(a):
            lo, hi = np.quantile(a, q), np.quantile(a, 1.0 - q)
            return a[(a >= lo) & (a <= hi)]

        n_gt0, n_gen0 = gt.size, gen.size
        gt, gen = _trim(gt), _trim(gen)
        print(f"  Trimmed to central {100 * (1 - 2 * q):.0f}% (drop <{100 * q:.0f}% / "
              f">{100 * (1 - q):.0f}% per distribution): GT {n_gt0:,}->{gt.size:,}, "
              f"gen {n_gen0:,}->{gen.size:,}")

    # Save generated warpage values to CSV alongside the histogram PNG
    csv_path = Path(out_path).with_suffix(".csv")
    with open(csv_path, "w", newline="") as _cf:
        _cf.write("vaesample,warpage\n")
        for _sample_idx, _v in enumerate(gen):
            _cf.write(f"{_sample_idx},{_v}\n")
    print(f"Saved spread CSV:   {csv_path}")

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
        f"GT  mu={gt_s['mean']:.2e} sd={gt_s['std']:.2e} p99={gt_s['p99']:.2e} "
        f"max={gt_s['max']:.2e} kurt={gt_s['kurt']:.1f}\n"
        f"Gen mu={gen_s['mean']:.2e} sd={gen_s['std']:.2e} p99={gen_s['p99']:.2e} "
        f"max={gen_s['max']:.2e} kurt={gen_s['kurt']:.1f}"
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

    cmp = _compare(gt, gen)

    def _row(tag, s):
        return (f"  {tag:>3}  mean={s['mean']:.3e}  std={s['std']:.3e}  min={s['min']:.3e}  "
                f"p1={s['p1']:.3e}  p50={s['p50']:.3e}  p99={s['p99']:.3e}  max={s['max']:.3e}  "
                f"skew={s['skew']:+.2f}  kurt={s['kurt']:+.2f}")

    print("\n" + "=" * 80)
    print("  spread = max(z_disp) - min(z_disp) per realization   (outliers = extreme warpage)")
    print(_row("GT", gt_s))
    print(_row("Gen", gen_s))
    print(f"\n  GT vs Gen     wasserstein={cmp['wasserstein']:.3e}   ks={cmp['ks']:.3f}   "
          f"crps={cmp['crps']:.3e}")
    print(f"  tail coverage p99_cov={cmp['p99_cov']:.2f}  max_cov={cmp['max_cov']:.2f}  "
          f"p1_cov={cmp['p1_cov']:.2f}  min_cov={cmp['min_cov']:.2f}   (1.00 = matches GT extremes)")
    print(f"  outlier rate  gen>GT_p99={cmp['gen_frac_above_gt_p99'] * 100:.2f}%   "
          f"gen<GT_p1={cmp['gen_frac_below_gt_p1'] * 100:.2f}%   (ideal ~1.00% each)")
    print("=" * 80)

    # machine-parseable (run_all.sh summary parses these)
    print(f"\nHIST_STATS  GT  mu={gt_s['mean']:.4e} sigma={gt_s['std']:.4e} "
          f"min={gt_s['min']:.4e} max={gt_s['max']:.4e} p99={gt_s['p99']:.4e} "
          f"kurt={gt_s['kurt']:.3f} n={gt_s['n']}")
    print(f"HIST_STATS  Gen mu={gen_s['mean']:.4e} sigma={gen_s['std']:.4e} "
          f"min={gen_s['min']:.4e} max={gen_s['max']:.4e} p99={gen_s['p99']:.4e} "
          f"kurt={gen_s['kurt']:.3f} n={gen_s['n']}")
    print(f"HIST_COMPARE wasserstein={cmp['wasserstein']:.4e} ks={cmp['ks']:.4f} "
          f"crps={cmp['crps']:.4e} p99_cov={cmp['p99_cov']:.4f} max_cov={cmp['max_cov']:.4f} "
          f"gen_frac_gt_p99={cmp['gen_frac_above_gt_p99']:.4f}")


if __name__ == "__main__":
    main()
