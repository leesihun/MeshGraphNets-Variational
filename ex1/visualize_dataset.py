"""
Visualize all 20 samples in dataset/ex1.h5.

Output (one PNG per sample + one stats summary):
  ex1/sample_01.png … ex1/sample_20.png   — 5-panel per sample
  ex1/viz_stats.png                        — dataset-wide statistics
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

H5_PATH = "dataset/ex1.h5"
OUT_DIR  = "ex1"
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sample(f, sid):
    nd     = f[f"data/{sid}/nodal_data"][:, 0, :]   # [8, N]
    xy     = nd[0:2].T                               # [N, 2]
    disp   = nd[3:6].T                               # [N, 3]  x/y/z disp
    stress = nd[6]                                   # [N]
    part   = nd[7].astype(int)                       # [N]  part number
    edges  = f[f"data/{sid}/mesh_edge"][:]            # [2, E]
    meta   = dict(f[f"data/{sid}/metadata"].attrs)
    return xy, disp, stress, part, edges, meta


def make_lc(xy, edges, max_edges=60_000):
    E    = edges.shape[1]
    step = max(1, E // max_edges)
    sel  = edges[:, ::step]
    segs = np.stack([xy[sel[0]], xy[sel[1]]], axis=1)
    return LineCollection(segs, linewidths=0.25, colors="0.55", alpha=0.15)


def add_colorbar(fig, ax, sc, label=""):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.03)
    cb  = fig.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=6.5)
    if label:
        cb.set_label(label, fontsize=7)


def scatter_panel(fig, ax, xy, edges, vals, cmap, symmetric=False, label=""):
    if symmetric:
        vabs = np.abs(vals).max()
        vmin, vmax = (-vabs, vabs) if vabs > 0 else (-1, 1)
    else:
        vmin, vmax = vals.min(), vals.max()
        if vmin == vmax:
            vmin -= 1e-6; vmax += 1e-6

    ax.add_collection(make_lc(xy, edges))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap,
                    s=0.8, linewidths=0, rasterized=True,
                    vmin=vmin, vmax=vmax)
    add_colorbar(fig, ax, sc, label)
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


# ── per-sample figure (5 panels) ─────────────────────────────────────────────

def plot_part(fig, ax, xy, edges, part):
    """Scatter panel with discrete categorical colors for part number."""
    import matplotlib.cm as cm
    part_ids = np.unique(part)
    cmap     = plt.get_cmap("tab10")
    colors   = np.array([cmap(int(p) % 10) for p in part])

    ax.add_collection(make_lc(xy, edges))
    for pid in part_ids:
        mask = part == pid
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=[cmap(int(pid) % 10)], s=0.8, linewidths=0,
                   rasterized=True, label=f"Part {pid}")
    ax.legend(loc="upper right", fontsize=6.5, markerscale=4,
              framealpha=0.7, handlelength=1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_sample(f, sid):
    xy, disp, stress, part, edges, meta = load_sample(f, sid)
    mag = np.linalg.norm(disp[:, :2], axis=1)

    panels = [
        ("x_disp (mm)",     disp[:, 0], "RdBu_r",  True),
        ("y_disp (mm)",     disp[:, 1], "RdBu_r",  True),
        ("z_disp (mm)",     disp[:, 2], "RdBu_r",  True),
        ("stress (MPa)",    stress,     "hot",      False),
        ("|u_xy| (mm)",     mag,        "plasma",   False),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(26, 5))
    N, E = meta["num_nodes"], meta["num_edges"]
    src  = meta.get("source_filename", "")
    fig.suptitle(
        f"Sample {sid}  —  {N:,} nodes  |  {E:,} edges  |  {src}",
        fontsize=11, y=1.01
    )

    for ax, (title, vals, cmap, sym) in zip(axes[:5], panels):
        scatter_panel(fig, ax, xy, edges, vals, cmap, symmetric=sym)
        ax.set_title(title, fontsize=9)

    plot_part(fig, axes[5], xy, edges, part)
    axes[5].set_title("Part No.", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"sample_{sid:02d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── dataset-wide statistics ───────────────────────────────────────────────────

def plot_stats(f, sample_ids):
    records = []
    for sid in sample_ids:
        xy, disp, stress, part, edges, meta = load_sample(f, sid)
        mag = np.linalg.norm(disp[:, :2], axis=1)
        records.append({
            "sid":        sid,
            "nodes":      meta["num_nodes"],
            "edges":      meta["num_edges"],
            "disp_max":   mag.max(),
            "disp_mean":  mag.mean(),
            "xd_min":     disp[:, 0].min(),
            "xd_max":     disp[:, 0].max(),
            "stress_max": stress.max(),
            "stress_mean":stress.mean(),
        })

    sids      = [r["sid"]       for r in records]
    nodes     = [r["nodes"]     for r in records]
    dmax      = [r["disp_max"]  for r in records]
    smax      = [r["stress_max"]for r in records]
    dmean     = [r["disp_mean"] for r in records]

    import matplotlib.cm as cm
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Dataset statistics — ex1.h5 (20 samples)", fontsize=13)

    # 1. mesh size
    ax = axes[0, 0]
    colors = cm.viridis(np.linspace(0.15, 0.85, len(sids)))
    bars = ax.bar(sids, [n / 1000 for n in nodes], color=colors, edgecolor="0.3", lw=0.4)
    for bar, n in zip(bars, nodes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{n//1000}k", ha="center", va="bottom", fontsize=6)
    ax.set_title("Mesh size (nodes)"); ax.set_xlabel("Sample ID")
    ax.set_ylabel("Nodes (×10³)"); ax.set_xticks(sids)
    ax.tick_params(axis="x", labelsize=7)

    # 2. max displacement magnitude
    ax = axes[0, 1]
    norm_d = np.array(dmax) / max(dmax)
    ax.bar(sids, dmax, color=cm.plasma(norm_d), edgecolor="0.3", lw=0.4)
    ax.set_title("Peak displacement magnitude |u_xy|")
    ax.set_xlabel("Sample ID"); ax.set_ylabel("|u_xy| max (mm)")
    ax.set_xticks(sids); ax.tick_params(axis="x", labelsize=7)

    # 3. max stress
    ax = axes[1, 0]
    norm_s = np.array(smax) / max(smax) if max(smax) > 0 else np.zeros(len(smax))
    ax.bar(sids, smax, color=cm.hot(0.3 + 0.6 * norm_s), edgecolor="0.3", lw=0.4)
    ax.set_title("Peak stress (MPa)")
    ax.set_xlabel("Sample ID"); ax.set_ylabel("stress max (MPa)")
    ax.set_xticks(sids); ax.tick_params(axis="x", labelsize=7)

    # 4. nodes vs peak displacement scatter
    ax = axes[1, 1]
    sc = ax.scatter(nodes, dmax, c=sids, cmap="tab20", s=70, zorder=3)
    for r in records:
        ax.annotate(str(r["sid"]), (r["nodes"], r["disp_max"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=7)
    ax.set_title("Mesh size vs. peak displacement")
    ax.set_xlabel("Num nodes"); ax.set_ylabel("|u_xy| max (mm)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "viz_stats.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with h5py.File(H5_PATH, "r") as f:
        sample_ids = sorted(int(k) for k in f["data"].keys())
        print(f"Found {len(sample_ids)} samples.\n")

        for sid in sample_ids:
            print(f"Sample {sid:2d} / {len(sample_ids)} …", end=" ")
            plot_sample(f, sid)

        print("\nRendering stats figure …")
        plot_stats(f, sample_ids)

    print(f"\nDone: {len(sample_ids) + 1} PNGs written to {OUT_DIR}/")
