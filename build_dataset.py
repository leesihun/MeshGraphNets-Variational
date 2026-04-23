import os
import re
import glob
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Configuration ===
device_code = "sm-f766u_m"
max_sample = -1
NUM_WORKERS = 64
CHUNK_SIZE = NUM_WORKERS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "..", device_code, "picked_inp")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "dataset.h5")

ANSYS_INP_PATH = "Ansys_inp_path"
NONFEM_PATH = "nonfem"
SUBFOLDERS = [f"sub{i}" for i in range(1, 11)]

NUM_TIMESTEPS = 34
NUM_FEATURES = 8

FEATURE_NAMES = [
    b'x_coord',
    b'y_coord',
    b'z_coord',
    b'x_disp(mm)',
    b'y_disp(mm)',
    b'z_disp(mm)',
    b'stress(MPa)',
    b'Part No.'
]

# Possible stress column names in displacement CSVs (checked in order)
STRESS_COLUMN_CANDIDATES = ['SEQV', 'S_EQV', 'Stress', 'stress', 'VM_Stress', 'von_mises']


def find_all_samples():
    samples = []
    skipped_no_csv = 0

    for sub in SUBFOLDERS:
        nonfem_dir = os.path.join(DATA_ROOT, NONFEM_PATH, sub)
        ansys_dir = os.path.join(DATA_ROOT, ANSYS_INP_PATH, sub, "Sim_data")

        if not os.path.exists(nonfem_dir):
            continue

        mesh_files = sorted(glob.glob(os.path.join(nonfem_dir, "*_mesh.inp")))

        for mesh_file in mesh_files:
            filename = os.path.basename(mesh_file)
            match = re.match(r'id(\d+)_', filename)
            if not match:
                continue

            sample_id = int(match.group(1))
            sim_data_dir = os.path.join(ansys_dir, str(sample_id))

            coords_file = os.path.join(sim_data_dir, "node_coordinates.csv")
            if not os.path.exists(coords_file):
                continue

            first_disp_file = os.path.join(sim_data_dir, "z_disp_step_1.csv")
            if not os.path.exists(first_disp_file):
                skipped_no_csv += 1
                continue

            samples.append({
                'sample_id': sample_id,
                'sub': sub,
                'mesh_file': mesh_file,
                'coords_file': coords_file,
                'sim_data_dir': sim_data_dir
            })

    if skipped_no_csv > 0:
        print(f"Skipped {skipped_no_csv} samples due to missing CSV files")

    return samples


def load_coordinates(coords_file):
    df = pd.read_csv(coords_file, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['NodeID'] = df['NodeID'].astype(int)
    df = df.sort_values('NodeID').reset_index(drop=True)
    return df


def load_displacement_step(sim_data_dir, step):
    disp_file = os.path.join(sim_data_dir, f"z_disp_step_{step}.csv")
    if not os.path.exists(disp_file):
        return None

    with open(disp_file, 'r') as f:
        first_line = f.readline().strip()  # noqa: F841 - skip metadata line

    df = pd.read_csv(disp_file, skiprows=1, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['NodeID'] = df['NodeID'].astype(int)
    df = df.sort_values('NodeID').reset_index(drop=True)
    return df


def parse_mesh_inp(filepath):
    nodes = {}
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if line.startswith('*NODE'):
            section = 'nodes'
            continue
        elif line.startswith('*ELEMENT'):
            section = 'elements'
            continue
        elif line.startswith('*'):
            section = None
            continue

        if section == 'nodes' and line:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                node_id = int(float(parts[0]))
                nodes[node_id] = (float(parts[1]), float(parts[2]), float(parts[3]))

        elif section == 'elements' and line:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                n1, n2, n3 = int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                elements.append((n1, n2, n3))

    return nodes, elements


def extract_edges_from_elements(elements):
    edges_set = set()
    for n1, n2, n3 in elements:
        e1 = (min(n1, n2), max(n1, n2))
        e2 = (min(n2, n3), max(n2, n3))
        e3 = (min(n3, n1), max(n3, n1))
        edges_set.add(e1)
        edges_set.add(e2)
        edges_set.add(e3)

    edges = sorted(list(edges_set))
    return np.array(edges, dtype=np.int64).T


def get_corner_nodes_and_mapping(elements):
    all_nodes = set()
    for n1, n2, n3 in elements:
        all_nodes.add(n1)
        all_nodes.add(n2)
        all_nodes.add(n3)

    corner_nodes = sorted(list(all_nodes))
    node_to_idx = {n: i for i, n in enumerate(corner_nodes)}
    return corner_nodes, node_to_idx


def _remap_edges(edge_index_raw, node_to_corner):
    """Remap edges from original node IDs to compact corner indices (vectorized)."""
    max_orig_id = max(node_to_corner.keys())
    id_to_compact = np.full(max_orig_id + 1, -1, dtype=np.int64)
    for orig_id, compact_id in node_to_corner.items():
        id_to_compact[orig_id] = compact_id
    return id_to_compact[edge_index_raw]


def _fill_displacement(nodal_data, t_idx, disp_df, corner_ids):
    """Fill displacement (and optionally stress) for one timestep using NodeID lookup."""
    disp_by_id = disp_df.set_index('NodeID')
    disp_vals = disp_by_id.reindex(corner_ids).fillna(0.0)

    if 'UX' in disp_vals.columns:
        nodal_data[3, t_idx, :] = disp_vals['UX'].values.astype(np.float32)
    if 'UY' in disp_vals.columns:
        nodal_data[4, t_idx, :] = disp_vals['UY'].values.astype(np.float32)
    if 'UZ' in disp_vals.columns:
        nodal_data[5, t_idx, :] = disp_vals['UZ'].values.astype(np.float32)

    # Read stress if available in the CSV
    for col in STRESS_COLUMN_CANDIDATES:
        if col in disp_vals.columns:
            nodal_data[6, t_idx, :] = disp_vals[col].values.astype(np.float32)
            break


def _fill_coordinates(nodal_data, coords_df, corner_ids, num_timesteps):
    """Fill static coordinates at all timesteps using NodeID lookup (vectorized)."""
    coords_by_id = coords_df.set_index('NodeID')
    corner_coords = (coords_by_id.reindex(corner_ids)[['X', 'Y', 'Z']]
                     .fillna(0.0).values.astype(np.float32))
    # Broadcast [num_corner] to [num_timesteps, num_corner]
    nodal_data[0, :, :] = corner_coords[:, 0]
    nodal_data[1, :, :] = corner_coords[:, 1]
    nodal_data[2, :, :] = corner_coords[:, 2]

    n_missing = int(coords_by_id.reindex(corner_ids)['X'].isna().sum())
    return n_missing


def process_sample(sample, num_timesteps=NUM_TIMESTEPS):
    """Process a single sample: load mesh, coordinates, displacements."""
    coords_df = load_coordinates(sample['coords_file'])
    num_total_nodes = len(coords_df)

    nodes, elements = parse_mesh_inp(sample['mesh_file'])
    corner_nodes, node_to_corner = get_corner_nodes_and_mapping(elements)
    edge_index_raw = extract_edges_from_elements(elements)

    # Vectorized edge remapping (preserves sort order since mapping is monotonic)
    remapped_edges = _remap_edges(edge_index_raw, node_to_corner)

    num_corner = len(corner_nodes)
    num_edges = remapped_edges.shape[1]
    corner_ids = np.array(corner_nodes, dtype=np.int64)

    nodal_data = np.zeros((NUM_FEATURES, num_timesteps, num_corner), dtype=np.float32)

    # FIX: Fill coordinates unconditionally at ALL timesteps (not inside timestep loop)
    n_missing = _fill_coordinates(nodal_data, coords_df, corner_ids, num_timesteps)

    # FIX: Fill displacement per timestep using NodeID-based lookup (not positional iloc)
    for t in range(1, num_timesteps + 1):
        disp_df = load_displacement_step(sample['sim_data_dir'], t)
        if disp_df is None:
            continue
        t_idx = t - 1
        _fill_displacement(nodal_data, t_idx, disp_df, corner_ids)

    # FIX: Per-sample stats across ALL timesteps (not just timestep 0)
    flat_data = nodal_data.reshape(NUM_FEATURES, -1)
    feature_min = flat_data.min(axis=1)
    feature_max = flat_data.max(axis=1)
    feature_mean = flat_data.mean(axis=1)
    feature_std = flat_data.std(axis=1)

    return {
        'nodal_data': nodal_data,
        'mesh_edge': remapped_edges,
        'num_nodes': num_corner,
        'num_edges': num_edges,
        'num_cells': len(elements),
        'num_total_nodes': num_total_nodes,
        'num_data_points': num_corner * num_timesteps,
        'feature_min': feature_min,
        'feature_max': feature_max,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'source_filename': os.path.basename(sample['mesh_file']),
        'filename_id': str(sample['sample_id']),
        'n_missing_coords': n_missing,
    }


def process_sample_worker(sample_dict):
    try:
        result = process_sample(sample_dict)
        # Check if ANY timestep has displacement data
        has_disp = np.abs(result['nodal_data'][3:6, :, :]).sum() > 0
        if not has_disp:
            return {'status': 'skipped', 'sample': sample_dict, 'error': 'no displacement data'}
        return {'status': 'success', 'sample': sample_dict, 'result': result}
    except Exception as e:
        return {'status': 'error', 'sample': sample_dict, 'error': str(e)}


def process_last_timestep_worker(sample_dict):
    last_timestep = NUM_TIMESTEPS

    try:
        coords_df = load_coordinates(sample_dict['coords_file'])
        num_total_nodes = len(coords_df)

        nodes, elements = parse_mesh_inp(sample_dict['mesh_file'])
        corner_nodes, node_to_corner = get_corner_nodes_and_mapping(elements)
        edge_index_raw = extract_edges_from_elements(elements)

        remapped_edges = _remap_edges(edge_index_raw, node_to_corner)

        num_corner = len(corner_nodes)
        num_edges = remapped_edges.shape[1]
        corner_ids = np.array(corner_nodes, dtype=np.int64)

        nodal_data = np.zeros((NUM_FEATURES, 1, num_corner), dtype=np.float32)

        # FIX: Fill coordinates using NodeID-based lookup (vectorized)
        _fill_coordinates(nodal_data, coords_df, corner_ids, 1)

        disp_df = load_displacement_step(sample_dict['sim_data_dir'], last_timestep)
        if disp_df is None:
            return {'status': 'skipped', 'sample': sample_dict, 'error': 'no displacement data'}

        # FIX: Fill displacement using NodeID-based lookup (not positional iloc)
        _fill_displacement(nodal_data, 0, disp_df, corner_ids)

        if (nodal_data[3, 0, :].sum() == 0
                and nodal_data[4, 0, :].sum() == 0
                and nodal_data[5, 0, :].sum() == 0):
            return {'status': 'skipped', 'sample': sample_dict, 'error': 'no displacement data'}

        feature_min = nodal_data[:, 0, :].min(axis=1)
        feature_max = nodal_data[:, 0, :].max(axis=1)
        feature_mean = nodal_data[:, 0, :].mean(axis=1)
        feature_std = nodal_data[:, 0, :].std(axis=1)

        return {
            'status': 'success',
            'sample': sample_dict,
            'result': {
                'nodal_data': nodal_data,
                'mesh_edge': remapped_edges,
                'num_nodes': num_corner,
                'num_edges': num_edges,
                'num_cells': len(elements),
                'num_total_nodes': num_total_nodes,
                'num_data_points': num_corner,
                'feature_min': feature_min,
                'feature_max': feature_max,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'source_filename': os.path.basename(sample_dict['mesh_file']),
                'filename_id': str(sample_dict['sample_id'])
            }
        }
    except Exception as e:
        return {'status': 'error', 'sample': sample_dict, 'error': str(e)}


def _write_sample_to_hdf5(f, sample_id, res):
    """Write a single processed sample to the HDF5 file."""
    grp = f.create_group(f'data/{sample_id}')
    grp.create_dataset('nodal_data', data=res['nodal_data'])
    grp.create_dataset('mesh_edge', data=res['mesh_edge'])

    meta_grp = grp.create_group('metadata')
    meta_grp.attrs['source_filename'] = res['source_filename']
    meta_grp.attrs['filename_id'] = res['filename_id']
    meta_grp.attrs['num_nodes'] = res['num_nodes']
    meta_grp.attrs['num_edges'] = res['num_edges']
    meta_grp.attrs['num_cells'] = res['num_cells']
    meta_grp.attrs['num_corner_nodes'] = res['num_nodes']
    meta_grp.attrs['num_total_nodes'] = res['num_total_nodes']

    meta_grp.create_dataset('feature_min', data=res['feature_min'])
    meta_grp.create_dataset('feature_max', data=res['feature_max'])
    meta_grp.create_dataset('feature_mean', data=res['feature_mean'])
    meta_grp.create_dataset('feature_std', data=res['feature_std'])


def _init_hdf5(f, num_timesteps):
    """Initialize HDF5 file with metadata structure."""
    f.attrs['num_features'] = NUM_FEATURES
    f.attrs['num_timesteps'] = num_timesteps

    f.create_dataset('metadata/feature_names', data=FEATURE_NAMES)
    f.create_dataset('metadata/normalization_params/min',
                     data=np.full(NUM_FEATURES, np.inf, dtype=np.float32))
    f.create_dataset('metadata/normalization_params/max',
                     data=np.full(NUM_FEATURES, -np.inf, dtype=np.float32))
    f.create_dataset('metadata/normalization_params/mean',
                     data=np.zeros(NUM_FEATURES, dtype=np.float32))
    f.create_dataset('metadata/normalization_params/std',
                     data=np.ones(NUM_FEATURES, dtype=np.float32))
    f.create_dataset('metadata/splits/train', data=np.array([], dtype=np.int64))
    f.create_dataset('metadata/splits/val', data=np.array([], dtype=np.int64))
    f.create_dataset('metadata/splits/test', data=np.array([], dtype=np.int64))


def _finalize_global_stats(f, global_min, global_max, global_sum, global_sq_sum,
                           total_data_points):
    """Compute and write global normalization stats to HDF5."""
    if total_data_points > 0:
        global_mean = (global_sum / total_data_points).astype(np.float32)
        global_std = np.sqrt(
            np.maximum(global_sq_sum / total_data_points - global_mean ** 2, 0.0)
        ).astype(np.float32)
    else:
        global_mean = np.zeros(NUM_FEATURES, dtype=np.float32)
        global_std = np.ones(NUM_FEATURES, dtype=np.float32)
    global_std = np.maximum(global_std, 1e-8)

    f['metadata/normalization_params/min'][...] = global_min
    f['metadata/normalization_params/max'][...] = global_max
    f['metadata/normalization_params/mean'][...] = global_mean
    f['metadata/normalization_params/std'][...] = global_std


def _run_chunked_build(output_path, sample_dicts, worker_fn, num_timesteps,
                       max_samples=-1):
    """Core build loop shared by both builders."""
    print("Discovering samples...")
    if max_samples > 0:
        sample_dicts = sample_dicts[:max_samples]

    if len(sample_dicts) == 0:
        print("No samples found!")
        return

    global_min = np.full(NUM_FEATURES, np.inf, dtype=np.float32)
    global_max = np.full(NUM_FEATURES, -np.inf, dtype=np.float32)
    global_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    global_sq_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    total_data_points = 0

    skipped_no_disp = 0
    error_count = 0
    successful_count = 0

    total_chunks = (len(sample_dicts) + CHUNK_SIZE - 1) // CHUNK_SIZE

    with h5py.File(output_path, 'w') as f:
        _init_hdf5(f, num_timesteps)

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, len(sample_dicts))
            chunk_samples = sample_dicts[start_idx:end_idx]

            print(f"Processing chunk {chunk_idx + 1}/{total_chunks} "
                  f"({len(chunk_samples)} samples)...")

            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(worker_fn, sd): sd
                           for sd in chunk_samples}

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Chunk {chunk_idx + 1}"):
                    result = future.result()

                    if result['status'] == 'success':
                        res = result['result']
                        successful_count += 1

                        # FIX: weight by total data points (nodes * timesteps)
                        n_pts = res['num_data_points']
                        global_min = np.minimum(global_min, res['feature_min'])
                        global_max = np.maximum(global_max, res['feature_max'])
                        global_sum += res['feature_mean'].astype(np.float64) * n_pts
                        global_sq_sum += (
                            (res['feature_std'].astype(np.float64) ** 2
                             + res['feature_mean'].astype(np.float64) ** 2) * n_pts
                        )
                        total_data_points += n_pts

                        _write_sample_to_hdf5(f, successful_count, res)
                        del res

                    elif result['status'] == 'skipped':
                        skipped_no_disp += 1
                    else:
                        error_count += 1
                        print(f"Error processing "
                              f"{result['sample']['mesh_file']}: "
                              f"{result['error']}")

            f.attrs['num_samples'] = successful_count

        _finalize_global_stats(f, global_min, global_max, global_sum,
                               global_sq_sum, total_data_points)

    if skipped_no_disp > 0:
        print(f"Skipped {skipped_no_disp} samples due to missing displacement data")
    if error_count > 0:
        print(f"Errors processing {error_count} samples")

    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {successful_count}")


def build_dataset(output_path=None, max_samples=max_sample):
    if output_path is None:
        # FIX: was OUTPUT_FILE+'static' which produced 'dataset.h5static'
        output_path = OUTPUT_FILE.replace('.h5', '_static.h5')

    samples = find_all_samples()
    print(f"Found {len(samples)} samples")

    _run_chunked_build(
        output_path=output_path,
        sample_dicts=samples,
        worker_fn=process_sample_worker,
        num_timesteps=NUM_TIMESTEPS,
        max_samples=max_samples,
    )


def build_dataset_last_timestep(output_path=None, max_samples=max_sample):
    if output_path is None:
        output_path = OUTPUT_FILE.replace('.h5', '_last_step.h5')

    samples = find_all_samples()
    print(f"Found {len(samples)} samples")

    _run_chunked_build(
        output_path=output_path,
        sample_dicts=samples,
        worker_fn=process_last_timestep_worker,
        num_timesteps=1,
        max_samples=max_samples,
    )


if __name__ == "__main__":
    build_dataset_last_timestep()
    build_dataset()
