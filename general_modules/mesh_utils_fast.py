"""
Optimized mesh utilities with GPU acceleration and PyVista rendering.

Performance improvements:
1. GPU-accelerated triangle reconstruction using vectorized dense adjacency
2. PyVista (VTK) off-screen rendering for fast mesh visualization
3. Batched operations for face value computation
4. Optional visualization to avoid blocking inference
"""

import os
import h5py
import numpy as np
import torch
import pyvista as pv


def _triangles_from_edges_dict(edges_np, num_nodes):
    """
    Find triangles using adjacency sets.
    Fallback for meshes too large for the dense adjacency approach.

    Args:
        edges_np: (E, 2) numpy array of unique undirected edges (u < v)
        num_nodes: number of nodes in the mesh

    Returns:
        faces: (F, 3) numpy array of triangular face node indices
    """
    adj = [set() for _ in range(num_nodes)]
    for i in range(edges_np.shape[0]):
        u, v = int(edges_np[i, 0]), int(edges_np[i, 1])
        adj[u].add(v)
        adj[v].add(u)

    triangles = []
    for i in range(edges_np.shape[0]):
        u, v = int(edges_np[i, 0]), int(edges_np[i, 1])
        for w in adj[u] & adj[v]:
            if w > v:
                triangles.append((u, v, w))

    if not triangles:
        return np.array([], dtype=np.int64).reshape(0, 3)
    return np.array(triangles, dtype=np.int64)


# Maximum number of nodes for the dense adjacency approach.
# Above this, memory cost (N^2 bools) becomes prohibitive.
_DENSE_ADJ_NODE_LIMIT = 20_000


def edges_to_triangles_gpu(edge_index, device='cpu'):
    """
    Vectorized triangle reconstruction using dense adjacency on GPU.

    For each unique undirected edge (u, v) with u < v, finds all common
    neighbors w > v via batched boolean AND on adjacency rows. This avoids
    Python-level per-edge loops for meshes up to ~20k nodes. Larger meshes
    fall back to an adjacency-set approach.

    Args:
        edge_index: (2, E) tensor of edges (can be bidirectional)
        device: 'cpu' or 'cuda'

    Returns:
        faces: (F, 3) numpy array of triangular face node indices
    """
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.from_numpy(edge_index)
    edge_index = edge_index.to(device)

    # Build unique undirected edges (u < v)
    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    edges = torch.unique(torch.stack([u, v], dim=1), dim=0)

    num_nodes = int(edge_index.max()) + 1
    num_edges = edges.shape[0]

    if num_nodes > _DENSE_ADJ_NODE_LIMIT:
        return _triangles_from_edges_dict(edges.cpu().numpy(), num_nodes)

    # Build dense boolean adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True

    node_indices = torch.arange(num_nodes, device=device)

    # Process edges in batches to limit memory (B * N bools per batch)
    max_elements = 10**8  # ~100 MB for bool tensor
    batch_size = max(1, max_elements // num_nodes)

    all_triangles = []
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        batch_edges = edges[start:end]  # (B, 2)

        adj_u = adj[batch_edges[:, 0]]   # (B, N)
        adj_v = adj[batch_edges[:, 1]]   # (B, N)
        common = adj_u & adj_v           # (B, N)

        # Keep only w > v to produce each triangle exactly once
        mask = node_indices.unsqueeze(0) > batch_edges[:, 1].unsqueeze(1)
        common = common & mask

        edge_idx, w = torch.where(common)
        u_tri = batch_edges[edge_idx, 0]
        v_tri = batch_edges[edge_idx, 1]
        all_triangles.append(torch.stack([u_tri, v_tri, w], dim=1))

    if not all_triangles:
        return np.array([], dtype=np.int64).reshape(0, 3)

    return torch.cat(all_triangles, dim=0).cpu().numpy().astype(np.int64)


def edges_to_triangles_optimized(edge_index):
    """
    CPU triangle reconstruction using numpy dense adjacency.

    Same algorithm as the GPU version but operates on numpy arrays.
    Falls back to adjacency-set approach for large meshes.

    Args:
        edge_index: (2, E) array of edges

    Returns:
        faces: (F, 3) array of triangular face node indices
    """
    edges = edge_index.T  # (E, 2)

    # Build unique undirected edges (u < v)
    u = np.minimum(edges[:, 0], edges[:, 1])
    v = np.maximum(edges[:, 0], edges[:, 1])
    edge_pairs = np.stack([u, v], axis=1)
    unique_edges = np.unique(edge_pairs, axis=0)

    num_nodes = int(unique_edges.max()) + 1
    num_edges = unique_edges.shape[0]

    if num_nodes > _DENSE_ADJ_NODE_LIMIT:
        return _triangles_from_edges_dict(unique_edges, num_nodes)

    # Build dense boolean adjacency matrix
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
    adj[unique_edges[:, 0], unique_edges[:, 1]] = True
    adj[unique_edges[:, 1], unique_edges[:, 0]] = True

    node_indices = np.arange(num_nodes)

    # Process edges in batches
    batch_size = max(1, 10**8 // num_nodes)

    all_triangles = []
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        batch = unique_edges[start:end]  # (B, 2)

        adj_u = adj[batch[:, 0]]   # (B, N)
        adj_v = adj[batch[:, 1]]   # (B, N)
        common = adj_u & adj_v

        # Keep only w > v
        mask = node_indices[np.newaxis, :] > batch[:, 1:2]
        common = common & mask

        edge_idx, w = np.where(common)
        if len(edge_idx) > 0:
            u_tri = batch[edge_idx, 0]
            v_tri = batch[edge_idx, 1]
            all_triangles.append(np.stack([u_tri, v_tri, w], axis=1))

    if not all_triangles:
        return np.array([], dtype=np.int64).reshape(0, 3)

    return np.concatenate(all_triangles, axis=0).astype(np.int64)


def compute_face_values_gpu(faces, node_values, device='cpu'):
    """
    GPU-accelerated face value computation.

    Args:
        faces: (F, 3) array of face node indices
        node_values: (N, D) array of node feature values
        device: 'cpu' or 'cuda'

    Returns:
        face_values: (F, D) array of face-averaged values
    """
    if faces.shape[0] == 0:
        return np.array([], dtype=np.float32).reshape(0, node_values.shape[1])

    if not isinstance(faces, torch.Tensor):
        faces = torch.from_numpy(faces).to(device)
    if not isinstance(node_values, torch.Tensor):
        node_values = torch.from_numpy(node_values).to(device)
    else:
        faces = faces.to(device)
        node_values = node_values.to(device)

    # Vectorized face averaging
    v0 = node_values[faces[:, 0]]  # (F, D)
    v1 = node_values[faces[:, 1]]  # (F, D)
    v2 = node_values[faces[:, 2]]  # (F, D)

    face_values = (v0 + v1 + v2) / 3.0

    return face_values.cpu().numpy()


def save_inference_results_fast(output_path, graph,
                                  predicted_norm=None, target_norm=None,
                                  predicted_denorm=None, target_denorm=None,
                                  skip_visualization=False, device='cpu',
                                  feature_idx=-1, precomputed_faces=None):
    """
    Save inference results to HDF5 and optionally return data for visualization.

    Args:
        output_path: Path to save the HDF5 file
        graph: PyG Data object with pos, edge_index, edge_attr, sample_id, time_idx
               Optional: part_ids (N,) array of part assignments per node
        predicted_norm: (N, D) numpy array of predicted node features (normalized)
        target_norm: (N, D) numpy array of target node features (normalized)
        predicted_denorm: (N, D) numpy array of predicted node features (denormalized)
        target_denorm: (N, D) numpy array of target node features (denormalized)
        skip_visualization: If True, skip rendering (much faster)
        device: 'cpu' or 'cuda' for GPU acceleration
        feature_idx: Which feature to visualize (default -1 = last feature)
        precomputed_faces: (F, 3) numpy array of pre-computed triangle faces.
                           When provided, skips triangle reconstruction entirely.

    Returns:
        dict: Plot data for visualization, or None if skip_visualization=True
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy for HDF5
    pos = graph.pos.cpu().numpy() if hasattr(graph.pos, 'cpu') else np.array(graph.pos)
    edge_index_np = graph.edge_index.cpu().numpy() if hasattr(graph.edge_index, 'cpu') else np.array(graph.edge_index)
    edge_attr = graph.edge_attr.cpu().numpy() if hasattr(graph.edge_attr, 'cpu') else np.array(graph.edge_attr)

    # Extract sample_id and time_idx (handle both scalar and tensor cases for batch_size > 1)
    sample_id = None
    time_idx = None
    if hasattr(graph, 'sample_id') and graph.sample_id is not None:
        sid = graph.sample_id
        if hasattr(sid, 'cpu'):
            sid = sid.cpu()
        if hasattr(sid, 'numpy'):
            sid = sid.numpy()
        # For batch_size > 1, sample_id would be an array - take first element
        sample_id = int(sid) if np.isscalar(sid) or sid.ndim == 0 else int(sid[0])

    if hasattr(graph, 'time_idx') and graph.time_idx is not None:
        tid = graph.time_idx
        if hasattr(tid, 'cpu'):
            tid = tid.cpu()
        if hasattr(tid, 'numpy'):
            tid = tid.numpy()
        # For batch_size > 1, time_idx would be an array - take first element
        time_idx = int(tid) if np.isscalar(tid) or tid.ndim == 0 else int(tid[0])

    # Extract part_ids if available (for multi-part visualization)
    part_ids = None
    if hasattr(graph, 'part_ids') and graph.part_ids is not None:
        pid = graph.part_ids
        if hasattr(pid, 'cpu'):
            pid = pid.cpu()
        if hasattr(pid, 'numpy'):
            pid = pid.numpy()
        part_ids = np.array(pid).astype(np.int32)

    # Triangle reconstruction (skip if pre-computed faces are provided)
    if precomputed_faces is not None:
        faces = precomputed_faces
    elif device != 'cpu' and torch.cuda.is_available():
        edge_index_gpu = (graph.edge_index if hasattr(graph.edge_index, 'device')
                          else torch.from_numpy(edge_index_np).to(device))
        faces = edges_to_triangles_gpu(edge_index_gpu, device=device)
    else:
        faces = edges_to_triangles_optimized(edge_index_np)

    # Validate inputs before computing face values
    if predicted_norm.shape != target_norm.shape:
        print(f"Warning: Shape mismatch for sample_id={sample_id}, time_idx={time_idx}")
        print(f"  predicted_norm: {predicted_norm.shape}, target_norm: {target_norm.shape}")
    if predicted_denorm.shape != target_denorm.shape:
        print(f"Warning: Shape mismatch for sample_id={sample_id}, time_idx={time_idx}")
        print(f"  predicted_denorm: {predicted_denorm.shape}, target_denorm: {target_denorm.shape}")

    # GPU-accelerated face value computation for both normalized and denormalized
    pred_face_values_norm = compute_face_values_gpu(faces, predicted_norm, device=device)
    target_face_values_norm = compute_face_values_gpu(faces, target_norm, device=device)
    pred_face_values_denorm = compute_face_values_gpu(faces, predicted_denorm, device=device)
    target_face_values_denorm = compute_face_values_gpu(faces, target_denorm, device=device)

    # Validate face value shapes
    if pred_face_values_norm.shape[0] != faces.shape[0]:
        print(f"Error: Face value count mismatch for sample_id={sample_id}, time_idx={time_idx}")
        print(f"  faces: {faces.shape[0]}, pred_face_values: {pred_face_values_norm.shape[0]}")

    # Compute face-level part IDs if node-level part_ids are available
    face_part_ids = None
    if part_ids is not None and faces.shape[0] > 0:
        # Use majority vote of the 3 vertices for each face
        v0_parts = part_ids[faces[:, 0]]
        v1_parts = part_ids[faces[:, 1]]
        v2_parts = part_ids[faces[:, 2]]
        # Stack and find mode (most common) for each face
        face_parts_stack = np.stack([v0_parts, v1_parts, v2_parts], axis=1)
        # Simple approach: take the first vertex's part (they should all be the same for valid meshes)
        face_part_ids = v0_parts

    # Save to HDF5 (fast I/O) - save both normalized and denormalized
    with h5py.File(output_path, 'w') as f:
        # Node data
        nodes_grp = f.create_group('nodes')
        nodes_grp.create_dataset('pos', data=pos, dtype=np.float32)
        nodes_grp.create_dataset('predicted_norm', data=predicted_norm, dtype=np.float32)
        nodes_grp.create_dataset('target_norm', data=target_norm, dtype=np.float32)
        nodes_grp.create_dataset('predicted_denorm', data=predicted_denorm, dtype=np.float32)
        nodes_grp.create_dataset('target_denorm', data=target_denorm, dtype=np.float32)
        if part_ids is not None:
            nodes_grp.create_dataset('part_ids', data=part_ids, dtype=np.int32)

        # Edge data
        edges_grp = f.create_group('edges')
        edges_grp.create_dataset('index', data=edge_index_np, dtype=np.int64)
        edges_grp.create_dataset('attr', data=edge_attr, dtype=np.float32)

        # Face data
        faces_grp = f.create_group('faces')
        faces_grp.create_dataset('index', data=faces, dtype=np.int64)
        faces_grp.create_dataset('predicted_norm', data=pred_face_values_norm, dtype=np.float32)
        faces_grp.create_dataset('target_norm', data=target_face_values_norm, dtype=np.float32)
        faces_grp.create_dataset('predicted_denorm', data=pred_face_values_denorm, dtype=np.float32)
        faces_grp.create_dataset('target_denorm', data=target_face_values_denorm, dtype=np.float32)
        if face_part_ids is not None:
            faces_grp.create_dataset('part_ids', data=face_part_ids, dtype=np.int32)

        # Metadata
        f.attrs['num_nodes'] = pos.shape[0]
        f.attrs['num_edges'] = edge_index_np.shape[1]
        f.attrs['num_faces'] = faces.shape[0]
        f.attrs['num_features'] = predicted_norm.shape[1]

        if sample_id is not None:
            f.attrs['sample_id'] = sample_id
        if time_idx is not None:
            f.attrs['time_idx'] = time_idx
        if part_ids is not None:
            f.attrs['num_parts'] = len(np.unique(part_ids))

    # Optional visualization (can be deferred or skipped)
    if not skip_visualization:
        plot_path = output_path.replace('.h5', '.png')

        # Validate data before returning for visualization
        if pred_face_values_norm.shape[1] == 0:
            print(f"Warning: No features in face values for sample_id={sample_id}, time_idx={time_idx}. Skipping visualization.")
            return None

        # Debug: Print shape info for timestep==1
        if time_idx == 1:
            print(f"DEBUG timestep==1: sample_id={sample_id}")
            print(f"  faces: {faces.shape}")
            print(f"  pred_face_values_norm: {pred_face_values_norm.shape}")
            print(f"  target_face_values_norm: {target_face_values_norm.shape}")
            print(f"  feature_idx: {feature_idx}")

        # Return plot data for parallel processing with full metadata
        # Include both normalized and denormalized values for visualization
        return {
            'plot_path': plot_path,
            'pos': pos,
            'faces': faces,
            'pred_values_norm': pred_face_values_norm,
            'target_values_norm': target_face_values_norm,
            'pred_values_denorm': pred_face_values_denorm,
            'target_values_denorm': target_face_values_denorm,
            'sample_id': sample_id,
            'time_idx': time_idx,
            'face_part_ids': face_part_ids,
            'feature_idx': feature_idx
        }

    return None


def plot_mesh_comparison(pos, faces, pred_values_norm, target_values_norm,
                         pred_values_denorm, target_values_denorm, output_path,
                         feature_idx=-1, sample_id=None, time_idx=None, face_part_ids=None):
    """
    Create 2x2 mesh plots comparing normalized and denormalized predicted vs ground truth.
    Uses PyVista (VTK) for fast off-screen rendering.

    Args:
        pos: (N, 3) node positions
        faces: (F, 3) triangular face indices
        pred_values_norm: (F, D) predicted face values (normalized)
        target_values_norm: (F, D) ground truth face values (normalized)
        pred_values_denorm: (F, D) predicted face values (denormalized)
        target_values_denorm: (F, D) ground truth face values (denormalized)
        output_path: Path to save the PNG
        feature_idx: Which feature to visualize (default -1 = last, i.e. stress)
        sample_id: Sample ID for plot title (optional)
        time_idx: Timestep index for plot title (optional)
        face_part_ids: (F,) array of part IDs per face for edge coloring (optional)
    """
    if faces.shape[0] == 0:
        return

    # Validate feature_idx
    num_features = pred_values_norm.shape[1]
    if num_features == 0:
        print(f"Warning: No features to visualize for sample_id={sample_id}, time_idx={time_idx}")
        return

    actual_feature_idx = feature_idx if feature_idx >= 0 else num_features + feature_idx
    if actual_feature_idx < 0 or actual_feature_idx >= num_features:
        print(f"Error: feature_idx={feature_idx} (actual={actual_feature_idx}) out of bounds "
              f"for {num_features} features (sample_id={sample_id}, time_idx={time_idx})")
        return

    # Extract the selected feature for all four plots
    pred_colors_norm = pred_values_norm[:, feature_idx].astype(np.float64)
    target_colors_norm = target_values_norm[:, feature_idx].astype(np.float64)
    pred_colors_denorm = pred_values_denorm[:, feature_idx].astype(np.float64)
    target_colors_denorm = target_values_denorm[:, feature_idx].astype(np.float64)

    # Feature name and units
    feature_names = ['Delta Disp X', 'Delta Disp Y', 'Delta Disp Z', 'Delta Stress']
    feature_units = ['mm', 'mm', 'mm', 'MPa']
    feature_name = (feature_names[actual_feature_idx]
                    if actual_feature_idx < len(feature_names)
                    else f'Feature {actual_feature_idx}')
    feature_unit = (feature_units[actual_feature_idx]
                    if actual_feature_idx < len(feature_units)
                    else '')

    # Shared color ranges (same scale for pred vs target within each row)
    eps = 1e-12
    clim_norm = [
        min(float(pred_colors_norm.min()), float(target_colors_norm.min())),
        max(float(pred_colors_norm.max()), float(target_colors_norm.max())),
    ]
    if clim_norm[1] - clim_norm[0] < eps:
        clim_norm[1] = clim_norm[0] + eps

    clim_denorm = [
        min(float(pred_colors_denorm.min()), float(target_colors_denorm.min())),
        max(float(pred_colors_denorm.max()), float(target_colors_denorm.max())),
    ]
    if clim_denorm[1] - clim_denorm[0] < eps:
        clim_denorm[1] = clim_denorm[0] + eps

    # MAE for each row
    mae_norm = float(np.abs(pred_colors_norm - target_colors_norm).mean())
    mae_denorm = float(np.abs(pred_colors_denorm - target_colors_denorm).mean())

    # Build VTK-format faces: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = faces.shape[0]
    vtk_faces = np.column_stack([np.full(n_faces, 3, dtype=faces.dtype), faces]).ravel()
    mesh = pv.PolyData(pos.astype(np.float64), vtk_faces)

    # Show edges when there are multiple parts
    show_edges = (face_part_ids is not None and len(np.unique(face_part_ids)) > 1)

    # Build header text
    header_parts = []
    if sample_id is not None:
        header_parts.append(f'Sample {sample_id}')
    if time_idx is not None:
        header_parts.append(f'Timestep {time_idx}')
    if face_part_ids is not None:
        n_parts = len(np.unique(face_part_ids))
        if n_parts > 1:
            header_parts.append(f'{n_parts} Parts')

    mae_str_norm = f'MAE: {mae_norm:.4f}'
    mae_str_denorm = (f'MAE: {mae_denorm:.4f} {feature_unit}'.strip()
                      if feature_unit else f'MAE: {mae_denorm:.4f}')

    # Colorbar labels
    cbar_label_norm = f'{feature_name} (Normalized)'
    cbar_label_denorm = (f'{feature_name} ({feature_unit})'
                         if feature_unit else f'{feature_name} (Denormalized)')

    # Subplot definitions: (row, col, scalars, title, clim, show_cbar, cbar_title)
    subplot_configs = [
        (0, 0, pred_colors_norm,    'Normalized - Predicted',                         clim_norm,   False, ''),
        (0, 1, target_colors_norm,  f'Normalized - Ground Truth | {mae_str_norm}',    clim_norm,   True,  cbar_label_norm),
        (1, 0, pred_colors_denorm,  'Denormalized - Predicted',                       clim_denorm, False, ''),
        (1, 1, target_colors_denorm, f'Denormalized - Ground Truth | {mae_str_denorm}', clim_denorm, True,  cbar_label_denorm),
    ]

    # Create 2x2 off-screen plotter
    plotter = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(1800, 1400))

    for row, col, scalars, title, clim, show_cbar, cbar_title in subplot_configs:
        plotter.subplot(row, col)
        m = mesh.copy()
        m.cell_data['values'] = scalars

        sbar_args = dict(title=cbar_title, n_labels=5, fmt='%.4f', vertical=True)
        plotter.add_mesh(
            m,
            scalars='values',
            cmap='jet',
            clim=clim,
            show_edges=show_edges,
            edge_color='gray',
            line_width=0.3 if show_edges else 0,
            show_scalar_bar=show_cbar,
            scalar_bar_args=sbar_args,
        )
        plotter.add_text(title, font_size=10, position='upper_edge')
        plotter.camera_position = 'iso'

    # Add header with sample / timestep info in top-left subplot
    if header_parts:
        plotter.subplot(0, 0)
        plotter.add_text(', '.join(header_parts), font_size=12, position='upper_left')

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plotter.screenshot(output_path)
    plotter.close()


def render_plot_data(plot_data):
    """
    Render a single visualization from a plot_data dict.

    Args:
        plot_data: dict returned by save_inference_results_fast when
                   skip_visualization=False

    Returns:
        True on success, False on failure
    """
    sample_id = plot_data.get('sample_id', 'unknown')
    time_idx = plot_data.get('time_idx', 'unknown')

    if 'pred_values_norm' not in plot_data or 'target_values_norm' not in plot_data:
        print(f"Error: Missing required data for sample_id={sample_id}, time_idx={time_idx}")
        return False

    try:
        plot_mesh_comparison(
            plot_data['pos'],
            plot_data['faces'],
            plot_data['pred_values_norm'],
            plot_data['target_values_norm'],
            plot_data['pred_values_denorm'],
            plot_data['target_values_denorm'],
            plot_data['plot_path'],
            feature_idx=plot_data.get('feature_idx', -1),
            sample_id=sample_id,
            time_idx=time_idx,
            face_part_ids=plot_data.get('face_part_ids'),
        )
        return True
    except Exception as e:
        import traceback
        print(f"Error rendering sample_id={sample_id}, time_idx={time_idx}: {e}")
        traceback.print_exc()
        return False
