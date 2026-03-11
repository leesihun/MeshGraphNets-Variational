"""
reduce_dataset.py — Reduce the number of samples in an HDF5 dataset.

Usage examples:
    # Keep first 100 samples
    python reduce_dataset.py --input dataset.h5 --output small.h5 --n 100

    # Keep a random 10% subset (reproducible)
    python reduce_dataset.py --input dataset.h5 --output small.h5 --fraction 0.1 --seed 42

    # Keep specific sample IDs (1-based, matching HDF5 keys)
    python reduce_dataset.py --input dataset.h5 --output small.h5 --ids 1 5 10 42

    # Random subset, no shuffle of splits (just trim existing ones)
    python reduce_dataset.py --input dataset.h5 --output small.h5 --n 200 --no-shuffle
"""

import argparse
import sys
import time

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Reduce samples in an HDF5 GNN dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",    required=True,  help="Path to the source HDF5 file.")
    p.add_argument("--output",   required=True,  help="Path for the reduced output HDF5 file.")

    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--n",        type=int,   help="Keep exactly N samples (first N, or random if --shuffle).")
    sel.add_argument("--fraction", type=float, help="Keep this fraction of samples, e.g. 0.1 for 10%%.")
    sel.add_argument("--ids",      type=int, nargs="+", help="Explicit list of 1-based sample IDs to keep.")

    p.add_argument("--shuffle",    action="store_true", default=False,
                   help="When using --n or --fraction, pick samples randomly instead of the first N.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for reproducibility when shuffling (default: 42).")

    p.add_argument("--no-splits",  action="store_true", default=False,
                   help="Do not carry over split metadata (train/val/test) into the output.")
    p.add_argument("--new-splits", action="store_true", default=False,
                   help="Create fresh random 80/10/10 splits from the selected samples.")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Train fraction when --new-splits is used (default: 0.8).")
    p.add_argument("--val-ratio",   type=float, default=0.1,
                   help="Val fraction when --new-splits is used (default: 0.1).")

    p.add_argument("--compression", default="gzip",
                   choices=["gzip", "lzf", "none"],
                   help="HDF5 compression for copied datasets (default: gzip). 'none' disables compression.")
    p.add_argument("--compression-opts", type=int, default=4,
                   help="Compression level for gzip (1–9, default: 4). Ignored for other codecs.")

    p.add_argument("--verbose", action="store_true", default=False,
                   help="Print per-sample progress.")

    return p.parse_args()


def compress_kwargs(compression: str, compression_opts: int) -> dict:
    """Return h5py dataset creation kwargs for compression."""
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": compression_opts}
    return {"compression": compression}


def copy_group(src_grp, dst_grp, ckw: dict):
    """Recursively copy a group (datasets + attributes + sub-groups)."""
    for key in src_grp.attrs:
        dst_grp.attrs[key] = src_grp.attrs[key]
    for name, item in src_grp.items():
        if isinstance(item, h5py.Dataset):
            data = item[()]
            dst_grp.create_dataset(name, data=data, **ckw)
        elif isinstance(item, h5py.Group):
            sub = dst_grp.require_group(name)
            copy_group(item, sub, ckw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ckw = compress_kwargs(args.compression, args.compression_opts)

    # ---- Open source file -------------------------------------------------
    with h5py.File(args.input, "r") as src:
        total = int(src.attrs["num_samples"])
        all_ids = sorted(int(k) for k in src["data"].keys())

        if len(all_ids) != total:
            print(f"[warn] num_samples attribute ({total}) != actual groups ({len(all_ids)}); using actual count.")
            total = len(all_ids)
        all_ids = np.array(all_ids, dtype=np.int64)

        # ---- Determine which IDs to keep ----------------------------------
        if args.ids is not None:
            keep_ids = np.array(sorted(set(args.ids)), dtype=np.int64)
            missing = [i for i in keep_ids if i not in all_ids]
            if missing:
                print(f"[error] Requested IDs not found in dataset: {missing[:10]}{'...' if len(missing)>10 else ''}")
                sys.exit(1)

        else:
            n_keep = args.n if args.n is not None else max(1, round(total * args.fraction))
            if n_keep > total:
                print(f"[warn] Requested {n_keep} samples but only {total} exist; keeping all.")
                n_keep = total

            if args.shuffle:
                rng = np.random.default_rng(args.seed)
                chosen_idx = rng.choice(total, size=n_keep, replace=False)
                chosen_idx.sort()
                keep_ids = all_ids[chosen_idx]
            else:
                keep_ids = all_ids[:n_keep]

        n_out = len(keep_ids)
        print(f"Source : {args.input}  ({total} samples)")
        print(f"Output : {args.output}  ({n_out} samples)")
        if args.shuffle and args.ids is None:
            print(f"Seed   : {args.seed}")

        # ---- Create output file -------------------------------------------
        with h5py.File(args.output, "w") as dst:

            # File-level attributes
            for attr_key in src.attrs:
                if attr_key == "num_samples":
                    dst.attrs["num_samples"] = n_out
                else:
                    dst.attrs[attr_key] = src.attrs[attr_key]

            # Global metadata (feature names, normalization params, etc.)
            if "metadata" in src:
                meta_dst = dst.require_group("metadata")
                copy_group(src["metadata"], meta_dst, ckw)
                # Splits will be handled separately below

            # ---- Copy samples, renumbering 1..N ---------------------------
            data_grp = dst.require_group("data")
            t0 = time.time()

            # Build reverse mapping: old_id -> new_id (1-based)
            id_remap = {int(old): new_idx + 1 for new_idx, old in enumerate(keep_ids)}

            for new_idx, old_id in enumerate(keep_ids):
                new_id = new_idx + 1
                src_path = f"data/{old_id}"
                dst_sample = data_grp.require_group(str(new_id))

                sample_src = src[src_path]
                for ds_name in ("nodal_data", "mesh_edge"):
                    if ds_name in sample_src:
                        dst_sample.create_dataset(ds_name, data=sample_src[ds_name][()], **ckw)

                # metadata sub-group (attributes + optional stat datasets)
                if "metadata" in sample_src:
                    meta_src = sample_src["metadata"]
                    meta_dst_s = dst_sample.require_group("metadata")
                    for attr_key in meta_src.attrs:
                        meta_dst_s.attrs[attr_key] = meta_src.attrs[attr_key]
                    for ds_name, ds_val in meta_src.items():
                        if isinstance(ds_val, h5py.Dataset):
                            meta_dst_s.create_dataset(ds_name, data=ds_val[()], **ckw)

                if args.verbose or (new_id % 50 == 0):
                    elapsed = time.time() - t0
                    rate = new_id / elapsed
                    eta = (n_out - new_id) / rate if rate > 0 else 0
                    print(f"  [{new_id:>{len(str(n_out))}}/{n_out}]  old_id={old_id}  "
                          f"{elapsed:.1f}s elapsed  ETA {eta:.1f}s")

            # ---- Splits ---------------------------------------------------
            splits_dst = dst["metadata"].require_group("splits") if "metadata" in dst else None

            if splits_dst is not None:
                # Remove any splits copied blindly from global metadata
                for split_name in list(splits_dst.keys()):
                    del splits_dst[split_name]

                if args.no_splits:
                    pass  # leave splits empty

                elif args.new_splits:
                    rng = np.random.default_rng(args.seed)
                    new_ids_arr = np.arange(1, n_out + 1, dtype=np.int64)
                    shuffled = rng.permutation(new_ids_arr)

                    n_train = max(1, round(n_out * args.train_ratio))
                    n_val   = max(1, round(n_out * args.val_ratio))
                    n_test  = max(0, n_out - n_train - n_val)

                    splits_dst.create_dataset("train", data=shuffled[:n_train], **ckw)
                    splits_dst.create_dataset("val",   data=shuffled[n_train:n_train + n_val], **ckw)
                    splits_dst.create_dataset("test",  data=shuffled[n_train + n_val:], **ckw)
                    print(f"New splits: {n_train} train / {n_val} val / {n_test} test")

                else:
                    # Carry over existing splits, remapping old IDs → new IDs
                    if "metadata/splits" in src:
                        kept_set = set(int(i) for i in keep_ids)
                        for split_name in src["metadata/splits"]:
                            old_split_ids = src[f"metadata/splits/{split_name}"][()].tolist()
                            new_split_ids = np.array(
                                [id_remap[i] for i in old_split_ids if i in id_remap],
                                dtype=np.int64,
                            )
                            splits_dst.create_dataset(split_name, data=new_split_ids, **ckw)
                            print(f"Split '{split_name}': {len(old_split_ids)} → {len(new_split_ids)} samples")
                    else:
                        print("[info] No splits found in source; output will have no splits.")

            elapsed_total = time.time() - t0
            print(f"\nDone. {n_out} samples written in {elapsed_total:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
