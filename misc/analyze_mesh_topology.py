"""Analyze node/edge/cell topology statistics across HDF5 dataset samples.

Reports mesh sizes, edge/node ratios, infers likely element type from
cell metadata, and flags outlier samples.

Usage:
    python misc/analyze_mesh_topology.py dataset/dataset.h5
    python misc/analyze_mesh_topology.py dataset/flag_simple.h5 dataset/deforming_plate.h5
"""
import sys
import h5py
import numpy as np


def _infer_element_type(cv_mean: float, ev_mean: float) -> str:
    """Classify likely element type from Cell/Node and Edge/Node ratios.

    Known signatures (from face-based edge extraction):
      Tri shell:   C/V ~ 2.0, E/V ~ 3.0  (cells = triangular faces)
      Tet volume:  C/V ~ 3-6, E/V ~ 4.5-7 (cells = tetrahedra)
      Quad shell:  C/V ~ 1.0, E/V ~ 2.0  (cells = quad faces)
      Hex volume:  C/V ~ 1.0, E/V ~ 3-4  (cells = hexahedra)
    """
    if 1.7 <= cv_mean <= 2.3 and 2.7 <= ev_mean <= 3.3:
        return 'triangulated surface (shell)'
    elif cv_mean > 2.5 and ev_mean > 4.0:
        return 'tetrahedral volume'
    elif 0.7 <= cv_mean <= 1.3 and 1.7 <= ev_mean <= 2.3:
        return 'quad surface (shell)'
    elif 0.7 <= cv_mean <= 1.3 and ev_mean > 2.3:
        return 'hexahedral volume'
    else:
        return f'unknown (C/V={cv_mean:.2f}, E/V={ev_mean:.2f})'


def analyze_dataset(h5_path: str) -> None:
    print(f"\n{'='*70}")
    print(f"Dataset: {h5_path}")
    print(f"{'='*70}")

    with h5py.File(h5_path, 'r') as f:
        if 'data' not in f:
            print("  ERROR: no 'data' group found")
            return

        sample_ids = sorted(int(k) for k in f['data'].keys())
        n = len(sample_ids)
        print(f"  Samples: {n}")

        node_counts = []
        edge_counts = []
        cell_counts = []
        has_cells = False
        ev_ratios = []
        issues = []

        for sid in sample_ids:
            grp = f[f'data/{sid}']
            nodal_shape = grp['nodal_data'].shape  # (features, time, nodes)
            edge_data = grp['mesh_edge'][:]         # (2, edges)

            n_nodes = nodal_shape[2]
            n_edges = edge_data.shape[1]
            ratio = n_edges / n_nodes if n_nodes > 0 else 0.0

            node_counts.append(n_nodes)
            edge_counts.append(n_edges)
            ev_ratios.append(ratio)

            # Read cell count from metadata if available
            meta = grp.get('metadata')
            if meta is not None and 'num_cells' in meta.attrs:
                cell_counts.append(int(meta.attrs['num_cells']))
                has_cells = True
            else:
                cell_counts.append(0)

            # Validity checks
            if n_edges > 0:
                emin, emax = int(edge_data.min()), int(edge_data.max())
                if emin < 0 or emax >= n_nodes:
                    issues.append(f"  sample {sid}: edge index out of range [{emin}, {emax}] for {n_nodes} nodes")
                self_loops = int(np.sum(edge_data[0] == edge_data[1]))
                if self_loops > 0:
                    issues.append(f"  sample {sid}: {self_loops} self-loops")
            if n_nodes == 0:
                issues.append(f"  sample {sid}: 0 nodes")
            if n_edges == 0:
                issues.append(f"  sample {sid}: 0 edges")

        node_counts = np.array(node_counts)
        edge_counts = np.array(edge_counts)
        cell_counts = np.array(cell_counts)
        ev_ratios = np.array(ev_ratios)

        print(f"\n  Nodes:")
        print(f"    min: {int(np.min(node_counts)):>10,}")
        print(f"    max: {int(np.max(node_counts)):>10,}")
        print(f"    mean: {np.mean(node_counts):>10,.1f}")
        print(f"    std:  {np.std(node_counts):>10,.1f}")

        print(f"\n  Edges (stored, one-directional):")
        print(f"    min: {int(np.min(edge_counts)):>10,}")
        print(f"    max: {int(np.max(edge_counts)):>10,}")
        print(f"    mean: {np.mean(edge_counts):>10,.1f}")
        print(f"    std:  {np.std(edge_counts):>10,.1f}")

        if has_cells:
            valid_cells = cell_counts[cell_counts > 0]
            print(f"\n  Cells:")
            print(f"    min: {int(np.min(valid_cells)):>10,}")
            print(f"    max: {int(np.max(valid_cells)):>10,}")
            print(f"    mean: {np.mean(valid_cells):>10,.1f}")
            print(f"    std:  {np.std(valid_cells):>10,.1f}")

        ev_mean = float(np.mean(ev_ratios))
        ev_std = float(np.std(ev_ratios))

        print(f"\n  Edge/Node ratio:")
        print(f"    min:  {np.min(ev_ratios):.6f}")
        print(f"    max:  {np.max(ev_ratios):.6f}")
        print(f"    mean: {ev_mean:.6f}")
        print(f"    std:  {ev_std:.6f}")

        # Element type inference from cell metadata
        if has_cells:
            valid_mask = (cell_counts > 0) & (node_counts > 0) & (edge_counts > 0)
            if np.any(valid_mask):
                cv_ratios = cell_counts[valid_mask].astype(np.float64) / node_counts[valid_mask]
                ec_ratios = edge_counts[valid_mask].astype(np.float64) / cell_counts[valid_mask]
                cv_mean = float(np.mean(cv_ratios))
                ec_mean = float(np.mean(ec_ratios))
                print(f"\n  Cell/Node ratio:  {cv_mean:.4f}")
                print(f"  Edge/Cell ratio:  {ec_mean:.4f}")
                print(f"  Likely element type: {_infer_element_type(cv_mean, ev_mean)}")

        # Show histogram of ratios
        unique_ratios, counts = np.unique(np.round(ev_ratios, 2), return_counts=True)
        if len(unique_ratios) <= 20:
            print(f"\n  Edge/Node ratio distribution:")
            for r, c in zip(unique_ratios, counts):
                bar = '#' * min(c, 60)
                print(f"    {r:.2f}: {c:>5} {bar}")
        else:
            print(f"\n  Edge/Node ratio distribution (binned):")
            hist, bin_edges = np.histogram(ev_ratios, bins=15)
            max_bar = max(hist) if max(hist) > 0 else 1
            for j in range(len(hist)):
                if hist[j] == 0:
                    continue
                lo, hi = bin_edges[j], bin_edges[j+1]
                bar = '#' * max(1, int(40 * hist[j] / max_bar))
                print(f"    [{lo:.3f}, {hi:.3f}): {hist[j]:>5} {bar}")

        # Outliers (>3 sigma)
        if ev_std > 1e-8:
            outlier_mask = np.abs(ev_ratios - ev_mean) > 3 * ev_std
            n_out = int(np.sum(outlier_mask))
            if n_out > 0:
                print(f"\n  Outliers (>3 sigma): {n_out}")
                for j in np.where(outlier_mask)[0][:10]:
                    sid = sample_ids[j]
                    print(f"    sample {sid}: nodes={node_counts[j]:,}  edges={edge_counts[j]:,}  Edge/Node={ev_ratios[j]:.4f}")

        # Per-sample details for small datasets
        if n <= 20:
            print(f"\n  Per-sample detail:")
            header = f"    {'ID':>6}  {'Nodes':>10}  {'Edges':>10}"
            if has_cells:
                header += f"  {'Cells':>10}"
            header += f"  {'Edge/Node':>10}"
            print(header)
            for i, sid in enumerate(sample_ids):
                line = f"    {sid:>6}  {node_counts[i]:>10,}  {edge_counts[i]:>10,}"
                if has_cells:
                    line += f"  {cell_counts[i]:>10,}"
                line += f"  {ev_ratios[i]:>10.4f}"
                print(line)

        if issues:
            print(f"\n  Issues found ({len(issues)}):")
            for issue in issues[:20]:
                print(issue)
        else:
            print(f"\n  No issues found")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python misc/analyze_mesh_topology.py <dataset.h5> [dataset2.h5 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        analyze_dataset(path)
