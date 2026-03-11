#!/usr/bin/env python3

import torch

from general_modules.mesh_dataset import MeshGraphDataset


def load_data(config):
    """Create mesh graph dataset"""
    print("Loading mesh graph dataset...")

    data_file = config.get('dataset_dir')

    print(f"Creating MeshGraphDataset from: {data_file}")
    dataset = MeshGraphDataset(data_file, config=config)

    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset
