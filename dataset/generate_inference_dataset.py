#!/usr/bin/env python3
"""
Generate an inference dataset from flag_simple by selecting 10 random samples
and extracting only their initial conditions (t=0).

Output: ./infer/flag_inference.h5
"""

import h5py
import numpy as np
import os
from pathlib import Path

def generate_inference_dataset(
    source_dataset="dataset/warpage.h5",
    output_path="dataset/warpage_infer.h5",
    num_samples=10,
    random_seed=42
):
    """
    Extract initial conditions from random samples in the source dataset.

    Args:
        source_dataset: Path to source HDF5 file
        output_path: Path to output HDF5 file
        num_samples: Number of random samples to select
        random_seed: Random seed for reproducibility
    """

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load source dataset
    print(f"Loading source dataset: {source_dataset}")
    with h5py.File(source_dataset, 'r') as src_f:
        # Get all sample IDs
        sample_ids = sorted([int(k) for k in src_f['data'].keys()])
        total_samples = len(sample_ids)
        print(f"Total samples in source: {total_samples}")

        # Randomly select samples
        selected_indices = np.random.choice(total_samples, size=min(num_samples, total_samples), replace=False)
        selected_samples = [sample_ids[i] for i in selected_indices]
        print(f"Selected {len(selected_samples)} random samples: {selected_samples}")

        # Create output HDF5 file
        print(f"\nCreating output dataset: {output_path}")
        with h5py.File(output_path, 'w') as out_f:
            # Create data group
            data_grp = out_f.create_group('data')

            # Extract initial condition (t=0) for each selected sample
            for sample_id in selected_samples:
                sample_key = str(sample_id)
                print(f"\n  Processing sample {sample_id}...", end="")

                # Load source data
                nodal_data_full = src_f[f'data/{sample_key}/nodal_data'][:]  # [features, time, nodes]
                mesh_edge = src_f[f'data/{sample_key}/mesh_edge'][:]  # [2, M]

                # Extract only initial condition (t=0)
                nodal_data_initial = nodal_data_full[:, 0:1, :]  # [features, 1, nodes]

                num_features, num_timesteps, num_nodes = nodal_data_full.shape

                # Create group for this sample in output
                sample_grp = data_grp.create_group(sample_key)

                # Store initial condition
                sample_grp.create_dataset(
                    'nodal_data',
                    data=nodal_data_initial,
                    compression='gzip',
                    compression_opts=4
                )

                # Store mesh edge (same for all timesteps)
                sample_grp.create_dataset(
                    'mesh_edge',
                    data=mesh_edge
                )

                # Copy metadata if it exists
                if 'metadata' in src_f[f'data/{sample_key}']:
                    src_metadata = src_f[f'data/{sample_key}/metadata']
                    meta_grp = sample_grp.create_group('metadata')
                    for key, val in src_metadata.attrs.items():
                        meta_grp.attrs[key] = val

                print(f" OK ({num_features} features, {num_nodes} nodes)")

            # Copy global metadata if it exists
            if 'metadata' in src_f:
                src_global_meta = src_f['metadata']
                meta_grp = out_f.create_group('metadata')
                for key, val in src_global_meta.attrs.items():
                    meta_grp.attrs[key] = val
                print(f"\nGlobal metadata copied")

    # Print file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput file created: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Samples: {len(selected_samples)}")
    print(f"  Each sample contains only initial condition (t=0)")

    return output_path


if __name__ == "__main__":
    output_file = generate_inference_dataset()
    print(f"\nInference dataset ready at: {output_file}")
