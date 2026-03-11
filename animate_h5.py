"""
Animated GIFs of transient response for a random flag_simple sample.
Color = nodal_data[-2] (stress, feature index 6).
Produces separate GIF files for XZ, XY, YZ, and 3D isometric views.

Native Windows file dialog (no external dependencies).
Fallback to CLI if not on Windows.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
import random
import sys
import os
import argparse
import ctypes
from pathlib import Path

# Use absolute path: repo_root/dataset/flag_simple.h5
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Feature indices in nodal_data (8, T, N)
IX, IY, IZ = 0, 1, 2
IDX, IDY, IDZ = 3, 4, 5
# nodal_data[-2] => index 6 (stress)
COLOR_FEAT = -2


def load_sample(h5_path, sample_id):
    with h5py.File(h5_path, "r") as f:
        nd = f[f"data/{sample_id}/nodal_data"][:]
        edges = f[f"data/{sample_id}/mesh_edge"][:]
        meta = dict(f[f"data/{sample_id}/metadata"].attrs)
        feat_names = list(f["metadata/feature_names"][:])
    return nd, edges, meta, feat_names


def world_positions(nd, t):
    """Return deformed (wx, wy, wz) at timestep t."""
    wx = nd[IX, 0, :] + nd[IDX, t, :]
    wy = nd[IY, 0, :] + nd[IDY, t, :]
    wz = nd[IZ, 0, :] + nd[IDZ, t, :]  # rest z is 0 for flag
    return wx, wy, wz


# ------------------------------------------------------------------ #
#  2-D view renderer
# ------------------------------------------------------------------ #
def render_frame_2d(nd, edges, t, norm, cmap, fig, ax,
                    axis_a, axis_b, label_a, label_b, lims_a, lims_b, dt):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    all_coords = {0: wx, 1: wy, 2: wz}
    ca = all_coords[axis_a]
    cb = all_coords[axis_b]
    coords_2d = np.stack([ca, cb], axis=1)

    color = nd[COLOR_FEAT, t, :]

    segments = np.stack([coords_2d[edges[0]], coords_2d[edges[1]]], axis=1)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = LineCollection(segments, colors=cmap(norm(edge_c)),
                        linewidths=0.35, alpha=0.85)
    ax.add_collection(lc)
    ax.scatter(ca, cb, c=color, cmap=cmap, s=0.3, norm=norm)

    ax.set_xlim(lims_a)
    ax.set_ylim(lims_b)
    ax.set_aspect("equal")
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(f"t = {t * dt:.2f} s  (step {t})", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  3-D view renderer
# ------------------------------------------------------------------ #
def render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim, dt):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    color = nd[COLOR_FEAT, t, :]

    # 3D edge segments
    starts = np.stack([wx[edges[0]], wy[edges[0]], wz[edges[0]]], axis=1)
    ends = np.stack([wx[edges[1]], wy[edges[1]], wz[edges[1]]], axis=1)
    segments = np.stack([starts, ends], axis=1)  # (E, 2, 3)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = Line3DCollection(segments, colors=cmap(norm(edge_c)),
                          linewidths=0.3, alpha=0.8)
    ax.add_collection3d(lc)
    ax.scatter(wx, wy, wz, c=color, cmap=cmap, s=0.3, norm=norm, depthshade=False)

    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-3.0, 3.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"t = {t * dt:.2f} s  (step {t})", fontsize=11)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  GIF builder
# ------------------------------------------------------------------ #
def build_gif(frames, path, gif_fps):
    gif_duration_ms = int(1000 / gif_fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=gif_duration_ms, loop=0)


def make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a, axis_b, label_a, label_b,
                lims_a, lims_b, view_tag, dt, gif_fps, progress_callback=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(color_label)
    fig.suptitle(f"Inference  sample {sample_id}  (N={num_nodes})  [{view_tag}]",
                 fontsize=12, fontweight="bold")

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_2d(nd, edges, t, norm, cmap, fig, ax,
                              axis_a, axis_b, label_a, label_b, lims_a, lims_b, dt)
        frames.append(img)
        msg = f"{view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)"
        if progress_callback:
            progress_callback(msg)
        else:
            sys.stdout.write(f"\r  {msg}")
            sys.stdout.flush()
    plt.close(fig)
    if not progress_callback:
        print()

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


def make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev, azim, view_tag, dt, gif_fps, progress_callback=None):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.08, shrink=0.7)
    cbar.set_label(color_label)
    fig.suptitle(f"Flag Simple  sample {sample_id}  (N={num_nodes})  [{view_tag}]",
                 fontsize=12, fontweight="bold")

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim, dt)
        frames.append(img)
        msg = f"{view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)"
        if progress_callback:
            progress_callback(msg)
        else:
            sys.stdout.write(f"\r  {msg}")
            sys.stdout.flush()
    plt.close(fig)
    if not progress_callback:
        print()

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


# ------------------------------------------------------------------ #
#  Main Animation Generator
# ------------------------------------------------------------------ #
def generate_animations(h5_path, dt=0.02, frame_skip=4, gif_fps=20, progress_callback=None):
    """
    Generate animated GIFs from HDF5 dataset.

    Args:
        h5_path: Path to HDF5 file
        dt: Time step in seconds
        frame_skip: Skip every N frames
        gif_fps: Frames per second for GIF
        progress_callback: Optional callback function for progress updates

    Returns:
        List of generated GIF filenames
    """
    try:
        with h5py.File(h5_path, "r") as f:
            sample_ids = list(f["data"].keys())

        if not sample_ids:
            raise ValueError("No samples found in dataset")

        sample_id = random.choice(sample_ids)

        nd, edges, meta, feat_names = load_sample(h5_path, sample_id)
        num_features, num_timesteps, num_nodes = nd.shape

        color_name = feat_names[COLOR_FEAT] if isinstance(feat_names[COLOR_FEAT], str) \
            else feat_names[COLOR_FEAT].decode()
        color_data = nd[COLOR_FEAT]  # (T, N)

        msg = f"Sample {sample_id}  |  nodes={num_nodes}  timesteps={num_timesteps}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        msg = f"Color feature: nodal_data[{COLOR_FEAT}] = '{color_name}'"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        msg = f"  range: [{color_data.min():.6f}, {color_data.max():.6f}]"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        if color_data.max() == color_data.min():
            msg = f"  WARNING: '{color_name}' is constant ({color_data.min():.4f}) for this dataset."
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Global axis limits from world positions across all timesteps
        wx_all = nd[IX, 0, :][None, :] + nd[IDX]   # (T, N)
        wy_all = nd[IY, 0, :][None, :] + nd[IDY]
        wz_all = nd[IZ, 0, :][None, :] + nd[IDZ]
        pad = 0.3
        xlims = (wx_all.min() - pad, wx_all.max() + pad)
        ylims = (wy_all.min() - pad, wy_all.max() + pad)
        zlims = (wz_all.min() - pad, wz_all.max() + pad)

        # Color normalization
        vmin, vmax = float(color_data.min()), float(color_data.max())
        if vmin == vmax:
            vmax = vmin + 1.0  # avoid degenerate norm
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.plasma
        color_label = color_name

        timesteps = list(range(0, num_timesteps, frame_skip))
        msg = f"Frames: {len(timesteps)} (every {frame_skip} steps)\n"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        gifs = []

        # --- View 1: X-Z (front) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=0, axis_b=2,
                    label_a="X (world)", label_b="Z (world)",
                    lims_a=xlims, lims_b=zlims, view_tag="XZ_front",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 2: X-Y (top-down) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=0, axis_b=1,
                    label_a="X (world)", label_b="Y (world)",
                    lims_a=xlims, lims_b=ylims, view_tag="XY_top",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 3: Y-Z (side) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=1, axis_b=2,
                    label_a="Y (world)", label_b="Z (world)",
                    lims_a=ylims, lims_b=zlims, view_tag="YZ_side",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 4: 3D isometric ---
        gif = make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, elev=25, azim=-60, view_tag="3D_iso",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 5: 3D top-down ---
        gif = make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, elev=80, azim=-60, view_tag="3D_top",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        msg = "\nDone."
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        return gifs

    except Exception as e:
        msg = f"ERROR: {str(e)}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        raise


# ================================================================== #
#  Cross-platform File Dialog
# ================================================================== #
def browse_for_file():
    """
    Try to open native file dialog based on platform.
    Falls back to CLI input if unavailable.
    """
    import subprocess
    import platform

    system = platform.system()

    # Try Windows native dialog
    if system == "Windows":
        try:
            import ctypes.wintypes as wintypes
            from ctypes import windll

            # Initialize COM
            windll.ole32.CoInitializeEx(None, 0)

            # Create file open dialog
            file_dialog = windll.comdlg32.GetOpenFileNameA
            file_dialog.argtypes = [wintypes.c_char_p]

            # File filter
            filter_str = b"HDF5 Files (*.h5)\0*.h5\0All Files (*.*)\0*.*\0"

            # Prepare buffer
            file_path = ctypes.create_string_buffer(260)

            # OPENFILENAME structure
            class OPENFILENAME(ctypes.Structure):
                pass

            OPENFILENAME._fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HANDLE),
                ("lpstrFilter", wintypes.LPCSTR),
                ("lpstrCustomFilter", wintypes.LPCSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCSTR),
                ("lpstrTitle", wintypes.LPCSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", wintypes.c_void_p),
                ("lpTemplateName", wintypes.LPCSTR),
            ]

            ofn = OPENFILENAME()
            ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
            ofn.lpstrFilter = filter_str
            ofn.lpstrFile = file_path
            ofn.nMaxFile = 260
            ofn.lpstrTitle = b"Select HDF5 File"
            ofn.Flags = 0x00000004  # OFN_FILEMUSTEXIST

            if windll.comdlg32.GetOpenFileNameA(ctypes.byref(ofn)):
                return file_path.value.decode("utf-8")
            return None
        except Exception as e:
            print(f"Warning: Could not open Windows file dialog: {e}\n")

    # Try Linux/Unix zenity
    elif system in ["Linux", "Darwin"]:
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--file-filter=HDF5 Files (*.h5) | *.h5",
                 "--file-filter=All Files | *"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass

        # Try kdialog as fallback for KDE
        try:
            result = subprocess.run(
                ["kdialog", "--getopenfilename", str(Path.home()),
                 "*.h5 | HDF5 Files"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass

    print("Note: Could not open native file dialog.")
    return None


# ================================================================== #
#  Main Entry Point
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="Generate animated GIFs from flag_simple HDF5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python animate_flag_simple.py                                    # Opens file browser
  python animate_flag_simple.py dataset/flag_simple.h5             # Use default parameters
  python animate_flag_simple.py data.h5 --dt 0.01 --frame-skip 2   # Custom parameters
        """
    )
    parser.add_argument("h5_file", nargs="?", default=None, help="Path to HDF5 file")
    parser.add_argument("--dt", type=float, default=1, help="Time step in seconds (default: 0.02)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Skip every N frames (default: 4)")
    parser.add_argument("--gif-fps", type=int, default=10, help="GIF frames per second (default: 20)")

    args = parser.parse_args()

    # Get file path
    h5_file = args.h5_file
    if not h5_file:
        print("No H5 file specified. Opening file browser...")
        h5_file = browse_for_file()
        if not h5_file:
            print("No file selected. Exiting.")
            sys.exit(1)

    # Validate file
    if not os.path.exists(h5_file):
        print(f"Error: File not found: {h5_file}")
        sys.exit(1)

    print(f"\nGenerating animations from: {h5_file}")
    print(f"  Time step (dt): {args.dt} s")
    print(f"  Frame skip: {args.frame_skip}")
    print(f"  GIF FPS: {args.gif_fps}\n")

    try:
        generate_animations(
            h5_file,
            dt=args.dt,
            frame_skip=args.frame_skip,
            gif_fps=args.gif_fps
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
