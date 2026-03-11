# GNN Dataset Format Specification

## Overview

This document describes the HDF5 dataset format for Graph Neural Network (GNN) training on finite element analysis (FEA) mesh data representation.

**File Format**: HDF5 (`.h5`)
**Primary Library**: `h5py` for Python ML pipelines
**Structure**: Single file containing all samples with global metadata
**Node Selection**: FEM mesh nodes
**Edge Extraction**: FEM mesh connectivity (geometrically accurate)

---

## Complete Hierarchy

```
dataset.h5
├── [Attributes]                           # File-level metadata
│   ├── num_samples: int                   # Total number of samples (2138)
│   ├── num_features: int                  # Number of features per node (8)
│   └── num_timesteps: int                 # Timesteps per sample (1)
│
├── data/                                  # Main data group
│   ├── 1/                                 # Sample 1 (sequential ID)
│   │   ├── nodal_data                     # Dataset: (8, timesteps, nodes) float32
│   │   │                                  # Only corner nodes included
│   │   │                                  # Features: [x, y, z, x_disp, y_disp, z_disp, stress, part number]
│   │   ├── mesh_edge                      # Dataset: (2, edges) int64
│   │   │                                  # Node indices remapped to compact numbering
│   │   └── metadata/                      # Sample-specific metadata group
│   │       ├── [Attributes]
│   │       │   ├── source_filename: str   # Original .h5 filename, Optional
│   │       │   ├── filename_id: str       # Filename stem (e.g., "634_421"), Optional
│   │       │   ├── num_nodes: int         # Number of corner nodes in graph, Required
│   │       │   ├── num_edges: int         # Number of unique edges, Required
│   │       │   ├── num_cells: int         # Number of cells in source mesh, Optional
│   │       │   ├── num_corner_nodes: int  # Same as num_nodes (corner nodes), Optional
│   │       │   └── num_total_nodes: int   # Total nodes in original FEA mesh, Optional
│   │       ├── feature_min                # Dataset: (8,) float32 - per-feature minimums, Optional
│   │       ├── feature_max                # Dataset: (8,) float32 - per-feature maximums, Optional
│   │       ├── feature_mean               # Dataset: (8,) float32 - per-feature means, Optional
│   │       └── feature_std                # Dataset: (8,) float32 - per-feature std devs, Optional
│   ├── 2/                                 # Sample 2
│   │   └── ...
│   └── 2138/                              # Sample 2138
│
└── metadata/                              # Global metadata group
    ├── feature_names                      # Dataset: (8,) variable-length string
    │                                      # ['x_coord', 'y_coord', 'z_coord',
    │                                      #  'x_disp(mm)', 'y_disp(mm)',
    │                                      #  'z_disp(mm)', 'stress(MPa)', 'Part No.']
    ├── normalization_params/              # Global normalization statistics
    │   ├── min                            # Dataset: (8,) float32 - global minimums
    │   ├── max                            # Dataset: (8,) float32 - global maximums
    │   ├── mean                           # Dataset: (8,) float32 - global means
    │   └── std                            # Dataset: (8,) float32 - global std devs
    └── splits/                            # Train/validation/test splits
        ├── train                          # Dataset: (N_train,) int64 - sample IDs, optional
        ├── val                            # Dataset: (N_val,) int64 - sample IDs, optional
        └── test                           # Dataset: (N_test,) int64 - sample IDs, optional
```

---

## Data Specifications

### 1. Nodal Data

**Path**: `data/{sample_id}/nodal_data`
**Shape**: `(num_features, num_timesteps, num_nodes)`
**Dtype**: `float32`

**Feature Order** (index-based):
- `[0]` x_coord - X coordinate in 3D space
- `[1]` y_coord - Y coordinate in 3D space
- `[2]` z_coord - Z coordinate in 3D space
- `[3]` x_disp(mm) -  displacement in X direction (mm)
- `[4]` y_disp(mm) -  displacement in Y direction (mm)
- `[5]` z_disp(mm) -  displacement in Z direction (mm)
- `[6]` stress(MPa) -  stress (MPa)
- `[7]` Part number -  Part number

**Typical Dimensions**:
- Features: 8
- Timesteps: 100
- Nodes: ~60,000-90,000 (varies by sample)



### 2. Mesh Edges

**Path**: `data/{sample_id}/mesh_edge`
**Shape**: `(2, num_edges)`
**Dtype**: `int64`

**Structure**:
- Row 0: Source node indices
- Row 1: Target node indices

**Properties**:
- Edges are **undirected** (sorted: source ≤ target)
- Edges extracted from **triangular face connectivity**
- **No spurious edges**: Only geometrically valid connections
- Node indices use **compact numbering** (0 to num_nodes-1)
- **No self-loops**: All edges connect distinct nodes
- **No duplicates**: Each edge appears once

**Extraction Method**:
1. Read triangular faces from source mesh (3 nodes per face)
2. Extract 3 edges per face: (n0,n1), (n1,n2), (n2,n0)
3. Deduplicate across all faces
4. Remap node indices to compact corner-node numbering


### 3. Sample Metadata

**Path**: `data/{sample_id}/metadata`
**Type**: HDF5 Group with attributes and datasets

**Attributes** (scalar values):
- `source_filename`: Original source .h5 file name
- `filename_id`: Filename stem (identifier from filename)
- `num_nodes`: Number of corner nodes in the graph
- `num_edges`: Number of unique edges
- `num_cells`: Number of cells in original mesh
- `num_corner_nodes`: Number of corner nodes (same as num_nodes)
- `num_total_nodes`: Total nodes in original FEA mesh (including mid-edge)

**Datasets** (all shape `(8,)`, dtype `float32`):
- `feature_min`: Per-feature minimum values for this sample
- `feature_max`: Per-feature maximum values for this sample
- `feature_mean`: Per-feature mean values for this sample
- `feature_std`: Per-feature standard deviations for this sample

**Example**:
```python
meta = f['data/1/metadata'].attrs
print(f"Corner nodes: {meta['num_corner_nodes']:,}")
print(f"Total nodes: {meta['num_total_nodes']:,}")
print(f"Reduction: {100 * (1 - meta['num_corner_nodes']/meta['num_total_nodes']):.1f}%")
```

### 4. Global Metadata

#### Feature Names
**Path**: `metadata/feature_names`
**Shape**: `(8,)`
**Dtype**: Variable-length byte string

**Usage**:
```python
feature_names = [name.decode() for name in f['metadata/feature_names'][:]]
# ['x_coord', 'y_coord', 'z_coord', 'tor_x_disp(mm)', 'tor_y_disp(mm)', 'tor_z_disp(mm)', 'tor_stress(MPa)']
```

#### Normalization Parameters
**Path**: `metadata/normalization_params/{min|max|mean|std}`
**Shape**: `(7,)` for each
**Dtype**: `float32`

Computed across **all samples** in the dataset for global normalization.

**Example values** (from actual dataset):
```
Feature: x_coord
  min: -39.040001, max: 71.459839
  mean: 13.405532, std: (varies)

Feature: tor_stress(MPa)
  min: -393.085388, max: 405.905365
  mean: 0.479371, std: (varies)
```

#### Data Splits
**Path**: `metadata/splits/{train|val|test}`
**Shape**: `(N_split,)` - variable per split
**Dtype**: `int64`

Contains sample IDs (1, 2, 3, ..., 2138) assigned to each split.

**Note**: Default splits are empty arrays. Use the provided utility to create splits.

---

## Building the Dataset

### Basic Usage

```python
from build_dataset import build_dataset

# Build from all .h5 files in FieldMesh directory
build_dataset(
    source_dir='FieldMesh',
    output_path='dataset.h5',
    num_timesteps=1
)
```

### Processing Pipeline

The `build_dataset.py` script performs:

1. **Discovery**: Finds all `.h5` files in source directory (2138 files)
2. **Face-based Edge Extraction**:
   - Reads triangular faces from source mesh
   - Extracts 3 edges per face
   - Deduplicates to get unique edges
3. **Corner Node Identification**:
   - Identifies nodes referenced in faces (corner nodes)
   - Creates mapping from old indices to compact numbering
4. **Feature Assembly**:
   - Extracts corner node coordinates and solution data
   - Remaps edge indices to compact numbering
5. **Statistics**: Computes per-sample and global normalization parameters
6. **Organization**: Writes structured HDF5 with proper hierarchy

**Processing Time**: ~5-10 minutes for 2138 samples
**Output Size**: ~10.3 GB

---

## Accessing Data for GNN Training

### 1. Basic Data Loading

```python
import h5py
import numpy as np

# Open dataset
f = h5py.File('dataset.h5', 'r')

# Load file-level info
num_samples = f.attrs['num_samples']  # 2138
num_features = f.attrs['num_features']  # 8
num_timesteps = f.attrs['num_timesteps']  # 1

# Load feature names
feature_names = [name.decode() for name in f['metadata/feature_names'][:]]

# Load normalization parameters
norm_min = f['metadata/normalization_params/min'][:]
norm_max = f['metadata/normalization_params/max'][:]
norm_mean = f['metadata/normalization_params/mean'][:]
norm_std = f['metadata/normalization_params/std'][:]

# Close when done
f.close()
```

### 2. PyTorch Geometric Dataset Class

```python
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import h5py

class FEAGraphDataset(Dataset):
    def __init__(self, h5_path, split='train', normalize=True):
        """
        Args:
            h5_path: Path to dataset.h5
            split: 'train', 'val', or 'test'
            normalize: Whether to apply normalization
        """
        self.h5_path = h5_path
        self.normalize = normalize

        with h5py.File(h5_path, 'r') as f:
            self.sample_ids = f[f'metadata/splits/{split}'][:].tolist()

            if normalize:
                self.norm_mean = torch.from_numpy(
                    f['metadata/normalization_params/mean'][:]
                )
                self.norm_std = torch.from_numpy(
                    f['metadata/normalization_params/std'][:]
                )

        # Keep file open for fast access
        self.file = h5py.File(h5_path, 'r')

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Load data
        nodal_data = self.file[f'data/{sample_id}/nodal_data'][:]  # (7, 1, N)
        edge_index = self.file[f'data/{sample_id}/mesh_edge'][:]    # (2, E)

        # Extract timestep 0
        x = nodal_data[:, 0, :].T  # (N, 7) - transpose to node-major

        # Normalize
        if self.normalize:
            x = (x - self.norm_mean) / (self.norm_std + 1e-8)

        # Convert to torch tensors
        x = torch.from_numpy(x).float()
        edge_index = torch.from_numpy(edge_index).long()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        return data

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()


# Usage
dataset = FEAGraphDataset('dataset.h5', split='train')
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    print(batch.x.shape)          # Node features: (N_batch, 7)
    print(batch.edge_index.shape)  # Edges: (2, E_batch)
```

### 3. Loading Single Sample

```python
def load_sample(h5_path, sample_id):
    """Load a single sample as PyG Data object."""
    with h5py.File(h5_path, 'r') as f:
        # Load data
        nodal_data = f[f'data/{sample_id}/nodal_data'][:, 0, :].T  # (N, 7)
        edge_index = f[f'data/{sample_id}/mesh_edge'][:]            # (2, E)

        # Load metadata
        metadata = dict(f[f'data/{sample_id}/metadata'].attrs)

        # Convert to tensors
        x = torch.from_numpy(nodal_data).float()
        edge_index = torch.from_numpy(edge_index).long()

    return Data(x=x, edge_index=edge_index), metadata


# Usage
data, meta = load_sample('dataset.h5', sample_id=1)
print(f"Corner nodes: {meta['num_corner_nodes']:,}")
print(f"Edges: {meta['num_edges']:,}")
print(f"Original total nodes: {meta['num_total_nodes']:,}")
```

### 4. Creating Train/Val/Test Splits

```python
def create_splits(h5_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create and write train/val/test splits to dataset.

    Args:
        h5_path: Path to dataset.h5
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    with h5py.File(h5_path, 'r+') as f:
        num_samples = f.attrs['num_samples']
        sample_ids = np.arange(1, num_samples + 1)

        # Shuffle
        np.random.shuffle(sample_ids)

        # Split
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)

        train_ids = sample_ids[:n_train]
        val_ids = sample_ids[n_train:n_train + n_val]
        test_ids = sample_ids[n_train + n_val:]

        # Write splits (delete old ones first)
        del f['metadata/splits/train']
        del f['metadata/splits/val']
        del f['metadata/splits/test']

        f['metadata/splits'].create_dataset('train', data=train_ids)
        f['metadata/splits'].create_dataset('val', data=val_ids)
        f['metadata/splits'].create_dataset('test', data=test_ids)

        print(f"Splits created: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")


# Usage
create_splits('dataset.h5', train_ratio=0.8, val_ratio=0.1, seed=42)
# Output: Splits created: 1710 train, 213 val, 215 test
```

---

## Dataset Statistics

**From actual built dataset (2138 samples):**

| Metric | Value |
|--------|-------|
| Total samples | 2,138 |
| Total file size | 10.3 GB |
| Avg per sample | 4.9 MB |
| Features per node | 8 |
| Timesteps | 1 |

**Per-sample statistics:**

| Metric | Min | Max | Average |
|--------|-----|-----|---------|
| Corner nodes | ~59,000 | ~86,000 | ~68,000 |
| Edges | ~178,000 | ~259,000 | ~206,000 |
| Edge/node ratio | 3.00 | 3.00 | 3.00 |

**Comparison to original (quadratic) mesh:**

| Metric | Original FEA | Corner-only | Reduction |
|--------|--------------|-------------|-----------|
| Nodes/sample | ~191,000 | ~68,000 | 64% |
| Data in dataset | Would be ~28 GB | 10.3 GB | 63% |

---

## Performance Considerations

### Memory Efficiency
- HDF5 supports **lazy loading** - only requested data is loaded into memory
- Use `h5py.File` context managers or keep file handles open for fast access
- Memory mapping enables working with datasets larger than RAM

### Access Patterns
- **Sequential access**: Iterate through samples in order (fastest)
- **Random access**: Direct indexing by sample ID (still efficient)
- **Batch loading**: Load multiple samples efficiently using DataLoader

### Parallel Loading
```python
from torch.utils.data import DataLoader

# PyTorch DataLoader handles parallel loading
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel loading processes
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive between epochs
)
```

---

## File Format Advantages

1. **Single File**: No file management overhead, atomic operations
2. **Hierarchical**: Logical organization matches data structure
3. **Metadata Included**: Self-documenting with feature names and statistics
4. **Random Access**: Fast indexed access to any sample
5. **Compression**: HDF5 supports transparent compression (not currently enabled)
6. **Cross-platform**: Works on Windows, Linux, macOS
7. **Standard Format**: Wide tool support, language bindings for C++, Java, etc.
8. **Efficient Storage**: Corner-only approach reduces size by 63%

---

## Implementation Notes

### Edge Extraction Algorithm

The edges are extracted from **triangular face connectivity** rather than cell connectivity:

```python
def extract_edges_from_faces(faces):
    """
    Extract unique edges from face connectivity.
    Each triangular face contributes 3 edges.
    """
    num_faces = faces.shape[0]
    edges = np.zeros((num_faces * 3, 2), dtype=np.int64)

    # Extract all 3 edges from each triangular face
    edges[0::3] = [faces[:, 0], faces[:, 1]]
    edges[1::3] = [faces[:, 1], faces[:, 2]]
    edges[2::3] = [faces[:, 2], faces[:, 0]]

    # Sort and deduplicate
    edges.sort(axis=1)
    edges_unique = np.unique(edges, axis=0)

    return edges_unique.T  # (2, num_edges)
```

**Why faces instead of cells?**
- ✅ Geometrically correct connectivity
- ✅ Element-type agnostic (works for TET10, HEX20, etc.)
- ✅ Avoids spurious edges from all-pairs connectivity
- ✅ 91% reduction in edges vs. naive all-pairs approach

### Node Filtering

Corner nodes are identified as those referenced in faces:

```python
def get_corner_nodes(faces, cells):
    """Identify corner nodes and create compact mapping."""
    corner_node_indices = np.unique(faces.flatten())
    node_mapping = {old_idx: new_idx
                    for new_idx, old_idx in enumerate(corner_node_indices)}
    return corner_node_indices, node_mapping
```

This works because:
- Quadratic elements store mid-edge nodes in cells
- Linear faces only reference corner vertices
- Mid-edge nodes don't appear in face connectivity

---

## Troubleshooting

### File is locked
```python
# Always close files properly
with h5py.File('dataset.h5', 'r') as f:
    data = f['data/1/nodal_data'][:]
# File auto-closed here
```

### Out of memory
```python
# Don't load everything at once
# BAD:
all_data = [f[f'data/{i}/nodal_data'][:] for i in range(1, 2139)]

# GOOD:
for i in range(1, 2139):
    data = f[f'data/{i}/nodal_data'][:]  # Process one at a time
    process(data)
```

### Slow loading
```python
# Keep file handle open during training
f = h5py.File('dataset.h5', 'r')
# ... use f repeatedly ...
f.close()

# Use DataLoader with num_workers > 0
```

---

## Version History

- **v2.0**: Corner-node-only approach with face-based edge extraction (Current)
  - 64% node reduction, 91% edge reduction
  - Geometrically accurate topology
  - Element-type agnostic

- **v1.0**: Initial format (deprecated)
  - Included all nodes (corner + mid-edge)
  - Incorrect all-pairs edge connectivity
  - Not recommended for use
