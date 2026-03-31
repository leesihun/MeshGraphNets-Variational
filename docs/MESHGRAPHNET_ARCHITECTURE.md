# MeshGraphNets Architecture Documentation

## Overview

MeshGraphNets is a Graph Neural Network architecture designed for learning mesh-based simulations. It uses **Graph Network (GN) blocks** to perform message passing on mesh structures, enabling predictions of physical field data from mesh geometry.

**Paper**: "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., 2020, ICML)
**Original Implementation**: DeepMind Research (TensorFlow)
**Our Implementation**: PyTorch + PyTorch Geometric

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Graph Representation](#graph-representation)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Hyperparameters](#hyperparameters)
6. [Implementation Details](#implementation-details)

---

## High-Level Architecture

MeshGraphNets follows an **Encode-Process-Decode** paradigm:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MESHGRAPHNET                                │
│                                                                     │
│  ┌──────────┐      ┌──────────────┐      ┌──────────┐            │
│  │          │      │              │      │          │            │
│  │ ENCODER  │ ───> │  PROCESSOR   │ ───> │ DECODER  │            │
│  │          │      │  (15 layers) │      │          │            │
│  └──────────┘      └──────────────┘      └──────────┘            │
│       │                    │                    │                 │
│   Embed to             Message              Map to               │
│   latent              Passing              output                │
│   space                                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

```
INPUT:
  Node Features: [num_nodes, node_input_dim]
  Edge Features: [num_edges, edge_input_dim]
  Graph Topology: edge_index [2, num_edges]

       ↓

┌─────────────────────────────────────────┐
│           ENCODER                       │
│  ┌────────────────┐  ┌───────────────┐ │
│  │  Node Encoder  │  │ Edge Encoder  │ │
│  │   (MLP + LN)   │  │  (MLP + LN)   │ │
│  └────────────────┘  └───────────────┘ │
│         │                    │          │
│    [N, 128]             [E, 128]       │
└─────────────────────────────────────────┘

       ↓

┌─────────────────────────────────────────┐
│          PROCESSOR                      │
│  ┌────────────────────────────────┐    │
│  │   Graph Network Block 1        │    │
│  │  (Edge Update + Node Update)   │    │
│  └────────────────────────────────┘    │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │   Graph Network Block 2        │    │
│  └────────────────────────────────┘    │
│              ↓                          │
│            ...                          │
│              ↓                          │
│  ┌────────────────────────────────┐    │
│  │   Graph Network Block 15       │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘

       ↓

┌─────────────────────────────────────────┐
│           DECODER                       │
│  ┌────────────────┐                    │
│  │  Node Decoder  │                    │
│  │     (MLP)      │                    │
│  └────────────────┘                    │
│         │                               │
│    [N, output_dim]                     │
└─────────────────────────────────────────┘

       ↓

OUTPUT:
  Predicted Fields: [num_nodes, node_output_dim]
```

---

## Graph Representation

### Mesh as Graph

A mesh structure is represented as a graph where:
- **Nodes** = mesh vertices/points
- **Edges** = mesh connectivity (edges, faces)

```
Mesh Example (2D):                 Graph Representation:

    v2 -------- v3                     n2 -------- n3
    |  \        |                      |  \        |
    |    \      |                      |    \      |
    |      \    |                      |      \    |
    v0 -------- v1                     n0 -------- n1

Physical Mesh                      Graph Structure
- Vertices                         - Nodes
- Edges/Faces                      - Edges
```

### Edge Features from Geometry

For each edge connecting nodes i and j:

```
Node i position: pi = [xi, yi, zi]
Node j position: pj = [xj, yj, zj]

Relative Position Vector:
  rij = pj - pi = [Δx, Δy, Δz]

Euclidean Distance:
  dij = ||rij||₂ = √(Δx² + Δy² + Δz²)

Edge Feature:
  edge_attr = [Δx, Δy, Δz, dij]  ← 4D vector
```

**Visualization:**

```
        pj (receiver)
         ●
        /│
       / │
      /  │ dij (distance)
     /   │
    /    │
   /     │
  ●──────┘
 pi (sender)

  rij = [Δx, Δy, Δz]  (relative position vector)
```

---

## Core Components

### 1. Graph Network Block

The fundamental building block is the **Graph Network (GN) Block**, which updates both edges and nodes.

```
┌───────────────────────────────────────────────────────────────┐
│                   GRAPH NETWORK BLOCK                         │
│                                                               │
│  Input: x [N, d], edge_index [2, E], edge_attr [E, d]       │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ STEP 1: EDGE UPDATE                                  │    │
│  │                                                       │    │
│  │  For each edge (i→j):                                │    │
│  │                                                       │    │
│  │    ┌─────┐   ┌─────┐   ┌─────┐                      │    │
│  │    │ x_i │   │ x_j │   │ e_ij│                      │    │
│  │    └──┬──┘   └──┬──┘   └──┬──┘                      │    │
│  │       │         │         │                          │    │
│  │       └─────────┴─────────┘                          │    │
│  │               │                                       │    │
│  │          Concatenate                                 │    │
│  │               │                                       │    │
│  │               ▼                                       │    │
│  │        ┌────────────┐                                │    │
│  │        │  Edge MLP  │                                │    │
│  │        └─────┬──────┘                                │    │
│  │              │                                        │    │
│  │              ▼                                        │    │
│  │         e'_ij = e_ij + MLP([x_i, x_j, e_ij])        │    │
│  │                         └──────────────┘             │    │
│  │                         Residual Connection          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ STEP 2: NODE UPDATE                                  │    │
│  │                                                       │    │
│  │  For each node i:                                    │    │
│  │                                                       │    │
│  │    Aggregate incoming edge messages:                 │    │
│  │                                                       │    │
│  │         e'_1i   e'_2i   e'_3i                        │    │
│  │           │       │       │                          │    │
│  │           └───────┼───────┘                          │    │
│  │                   │                                   │    │
│  │                  SUM                                 │    │
│  │                   │                                   │    │
│  │                   ▼                                   │    │
│  │             ┌──────────┐                             │    │
│  │        x_i  │  Aggr.   │                             │    │
│  │          │  │ Messages │                             │    │
│  │          │  └────┬─────┘                             │    │
│  │          └───────┘                                    │    │
│  │               │                                       │    │
│  │          Concatenate                                 │    │
│  │               │                                       │    │
│  │               ▼                                       │    │
│  │        ┌────────────┐                                │    │
│  │        │  Node MLP  │                                │    │
│  │        └─────┬──────┘                                │    │
│  │              │                                        │    │
│  │              ▼                                        │    │
│  │         x'_i = x_i + MLP([x_i, Σe'_ji])             │    │
│  │                      └────────────┘                  │    │
│  │                      Residual Connection             │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  Output: x' [N, d], edge_attr' [E, d]                       │
└───────────────────────────────────────────────────────────────┘
```

### 2. MLP Architecture

Each MLP (Multi-Layer Perceptron) follows this structure:

```
┌────────────────────────────────────────┐
│           MLP STRUCTURE                │
│                                        │
│  Input [dim_in]                        │
│      │                                 │
│      ▼                                 │
│  ┌────────────────┐                   │
│  │ Linear(hidden) │                   │
│  └────────┬───────┘                   │
│           ▼                            │
│  ┌────────────────┐                   │
│  │  LayerNorm     │  ← Encoder/Proc  │
│  └────────┬───────┘     only          │
│           ▼                            │
│  ┌────────────────┐                   │
│  │     ReLU       │                   │
│  └────────┬───────┘                   │
│           ▼                            │
│  ┌────────────────┐                   │
│  │ Linear(hidden) │                   │
│  └────────┬───────┘                   │
│           ▼                            │
│  ┌────────────────┐                   │
│  │  LayerNorm     │  ← Encoder/Proc  │
│  └────────┬───────┘     only          │
│           ▼                            │
│  ┌────────────────┐                   │
│  │     ReLU       │                   │
│  └────────┬───────┘                   │
│           ▼                            │
│  ┌────────────────┐                   │
│  │ Linear(output) │                   │
│  └────────┬───────┘                   │
│           ▼                            │
│  Output [dim_out]                     │
│                                        │
│  Note: Decoder MLP has NO LayerNorm   │
└────────────────────────────────────────┘
```

**MLP Specifications:**
- **Encoder/Processor MLPs**: 2 hidden layers + output layer, with LayerNorm after each Linear
- **Decoder MLP**: 2 hidden layers + output layer, NO LayerNorm
- **Activation**: ReLU
- **Hidden size**: 128 (default)

### 3. Message Passing Visualization

```
BEFORE Message Passing:              AFTER Message Passing:

    n2 -------- n3                      n2' ------- n3'
    |  \        |                       |  \        |
    |    \      |                       |    \      |
    |      \    |                       |      \    |
    n0 -------- n1                      n0' ------- n1'

Each node has:                      Each node updated with:
- Own features                      - Own features
                                    - Aggregated neighbor info

Example for node n1:

STEP 1 - Edge Updates:
  e(0→1) ← MLP([n0, n1, e(0→1)])
  e(2→1) ← MLP([n2, n1, e(2→1)])
  e(3→1) ← MLP([n3, n1, e(3→1)])

STEP 2 - Aggregate:
  messages_1 = e'(0→1) + e'(2→1) + e'(3→1)
                  │        │        │
                  └────────┼────────┘
                           │
                          SUM

STEP 3 - Node Update:
  n1' ← MLP([n1, messages_1])
```

---

## Data Flow

### Complete Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│                      FORWARD PASS                               │
└─────────────────────────────────────────────────────────────────┘

INPUT DATA:
  Mesh geometry:    Node positions [95008, 3]
  Graph topology:   edge_index [2, num_edges]
  Node features:    Initial conditions, BCs [95008, feat_dim]

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING: Compute Edge Features                           │
│                                                                 │
│  for each edge (i, j) in edge_index:                           │
│    relative_pos = pos[j] - pos[i]        # [Δx, Δy, Δz]       │
│    distance = ||relative_pos||           # scalar              │
│    edge_attr[i,j] = [relative_pos, distance]  # 4D            │
└─────────────────────────────────────────────────────────────────┘

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ ENCODER                                                         │
│                                                                 │
│  Node: [95008, feat_dim] → MLP → [95008, 128]                 │
│  Edge: [E, 4] → MLP → [E, 128]                                │
└─────────────────────────────────────────────────────────────────┘

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ PROCESSOR (15 iterations)                                       │
│                                                                 │
│  Layer 1:  GN Block → [95008, 128], [E, 128]                  │
│  Layer 2:  GN Block → [95008, 128], [E, 128]                  │
│  Layer 3:  GN Block → [95008, 128], [E, 128]                  │
│  ...                                                            │
│  Layer 15: GN Block → [95008, 128], [E, 128]                  │
│                                                                 │
│  Each layer performs:                                          │
│    1. Update all edges                                         │
│    2. Aggregate messages to nodes                              │
│    3. Update all nodes                                         │
└─────────────────────────────────────────────────────────────────┘

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ DECODER                                                         │
│                                                                 │
│  Node: [95008, 128] → MLP → [95008, output_dim]               │
│                                                                 │
│  (Edge features discarded)                                      │
└─────────────────────────────────────────────────────────────────┘

                            ↓

OUTPUT:
  Predicted field: [95008, output_dim]
  (e.g., velocity field, pressure, temperature, etc.)
```

---

## Hyperparameters

### Default Configuration (from DeepMind paper)

| Parameter                    | Value | Description                              |
|------------------------------|-------|------------------------------------------|
| **Latent Dimension**         | 128   | Hidden feature size for nodes/edges      |
| **MLP Hidden Layers**        | 2     | Number of hidden layers per MLP         |
| **MLP Hidden Size**          | 128   | Width of hidden layers                   |
| **Message Passing Steps**    | 15    | Number of GN blocks in processor         |
| **Aggregation Function**     | Sum   | How to combine incoming edge messages    |
| **Normalization**            | LayerNorm | Applied after encoder/processor MLPs |
| **Activation Function**      | ReLU  | Non-linearity in MLPs                   |
| **Residual Connections**     | Yes   | On both edge and node updates           |

### Architecture Summary Table

```
┌────────────────────────────────────────────────────────────┐
│ Component      │ Input         │ Output        │ Layers   │
├────────────────────────────────────────────────────────────┤
│ Node Encoder   │ [N, d_in]     │ [N, 128]      │ 3-layer  │
│                │               │               │ MLP+LN   │
├────────────────────────────────────────────────────────────┤
│ Edge Encoder   │ [E, 4]        │ [E, 128]      │ 3-layer  │
│                │               │               │ MLP+LN   │
├────────────────────────────────────────────────────────────┤
│ Edge MLP       │ [2×128+128]   │ [128]         │ 3-layer  │
│ (in GN block)  │ = [384]       │               │ MLP+LN   │
├────────────────────────────────────────────────────────────┤
│ Node MLP       │ [128+128]     │ [128]         │ 3-layer  │
│ (in GN block)  │ = [256]       │               │ MLP+LN   │
├────────────────────────────────────────────────────────────┤
│ Processor      │ [N,128],[E,128│ [N,128],[E,128│ 15 GN    │
│                │               │               │ blocks   │
├────────────────────────────────────────────────────────────┤
│ Decoder        │ [N, 128]      │ [N, d_out]    │ 3-layer  │
│                │               │               │ MLP only │
└────────────────────────────────────────────────────────────┘

Legend: N = num_nodes, E = num_edges
        d_in = input feature dim, d_out = output dim
        LN = LayerNorm
```

---

## Implementation Details

### 1. Residual Connections

Residual connections are crucial for training deep networks (15 layers):

```
WITHOUT Residuals:              WITH Residuals:

x → MLP → x'                    x → MLP ──┐
                                          │
Gradient vanishing                        │ +
issues after 15 layers                    │
                                          ↓
                                x ────→  x' = x + MLP(x)

                                Stable gradients,
                                easier optimization
```

**Mathematical Form:**
```
Edge update:
  e'_ij = e_ij + EdgeMLP([x_i, x_j, e_ij])
          └────┘
          Residual

Node update:
  x'_i = x_i + NodeMLP([x_i, Σe'_ji])
         └──┘
         Residual
```

### 2. Layer Normalization

LayerNorm normalizes features across the feature dimension:

```
Before LayerNorm:                After LayerNorm:

Feature values vary widely       Features normalized:
                                 mean ≈ 0, std ≈ 1
[10.2, -3.5, 100.4, -50.1]  →   [0.12, -0.45, 1.23, -0.90]

Benefits:
  ✓ Stable training
  ✓ Faster convergence
  ✓ Better gradient flow
```

**Applied to:**
- Encoder MLPs (after each layer)
- Processor MLPs (after each layer)
- NOT applied to Decoder (to allow arbitrary output scale)

### 3. Aggregation Function

Sum aggregation combines messages from neighboring edges:

```
Node i receives messages from neighbors:

        n_j         n_k         n_l
         │           │           │
      e'_ji       e'_ki       e'_li
         │           │           │
         └───────────┼───────────┘
                     │
                    SUM
                     │
                     ▼
              aggregated_msg_i
                     │
                     ▼
              NodeMLP([x_i, aggregated_msg_i])

Alternative aggregations:
  - Mean: average of messages
  - Max: maximum across messages

Sum is standard in MeshGraphNets
```

### 4. Memory Considerations

For your mesh (95,008 nodes):

```
┌─────────────────────────────────────────────────────┐
│ Memory Estimate (approximate)                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Nodes: 95,008                                       │
│ Edges: ~6 × 95,008 = 570,048 (assuming 6 neighbors)│
│                                                     │
│ Node features: 95,008 × 128 × 4 bytes ≈ 48 MB      │
│ Edge features: 570,048 × 128 × 4 bytes ≈ 292 MB    │
│                                                     │
│ Per GN block: ~340 MB                               │
│ 15 GN blocks: ~5.1 GB                              │
│                                                     │
│ Recommendation:                                     │
│   - Use gradient checkpointing                      │
│   - Batch size = 1 or 2                            │
│   - GPU with ≥16 GB VRAM                           │
└─────────────────────────────────────────────────────┘
```

---

## Example: Predicting Flow Around Cylinder

```
INPUT MESH:                    OUTPUT PREDICTION:

    Cylinder flow mesh             Velocity field

    ○ ○ ○ ○ ○ ○ ○               ← ← ← ← ← ← ←
    ○ ○ ● ● ○ ○ ○               ← ← ↑ ↓ → → →
    ○ ○ ● ● ○ ○ ○               ← ← ↓ ↑ → → →
    ○ ○ ○ ○ ○ ○ ○               ← ← ← ← ← ← ←

    ● = cylinder                   Flow vectors around
    ○ = fluid nodes               obstacle

Node Features:                 Predicted Fields:
- Position (x,y)               - Velocity (vx, vy)
- Boundary flag                - Pressure (p)
- Initial velocity

Edge Features:
- Relative position
- Distance
```

---

## Comparison with Other GNN Architectures

```
┌──────────────┬─────────────┬──────────────┬─────────────┐
│ Architecture │ Edge Update │ Aggregation  │ Best For    │
├──────────────┼─────────────┼──────────────┼─────────────┤
│ GCN          │ No          │ Mean         │ Node class. │
│ GAT          │ No          │ Attention    │ Citations   │
│ GraphSAGE    │ No          │ Mean/Max     │ Large graphs│
│ MeshGraphNet │ YES         │ Sum          │ Mesh/Physics│
└──────────────┴─────────────┴──────────────┴─────────────┘

Key Advantage of MeshGraphNets:

  Explicitly updates EDGE features, which encode
  spatial relationships critical for physical systems

  Standard GNNs only update nodes ✗
  MeshGraphNets updates both nodes AND edges ✓
```

---

## Training Considerations

### Loss Function

For physics simulations, typically use:

```
L2 Loss (MSE):
  L = Σ ||y_pred - y_true||²

For your config (Loss_type = 1 → MSE):

  loss = MSELoss(predicted_field, ground_truth_field)
```

### Normalization of Inputs/Outputs

```
┌─────────────────────────────────────────────────────┐
│ CRITICAL: Normalize your data!                      │
│                                                     │
│ Before Training:                                    │
│   1. Compute statistics on training set:           │
│      μ_x, σ_x (node features)                      │
│      μ_y, σ_y (output fields)                      │
│                                                     │
│   2. Normalize:                                     │
│      x_norm = (x - μ_x) / σ_x                      │
│      y_norm = (y - μ_y) / σ_y                      │
│                                                     │
│   3. Train on normalized data                       │
│                                                     │
│   4. Denormalize predictions:                       │
│      y_pred = y_pred_norm × σ_y + μ_y              │
└─────────────────────────────────────────────────────┘
```

### Autoregressive Rollout (Time-Series)

For multi-timestep prediction:

```
Timestep:    t=0      t=1      t=2      t=3
             │        │        │        │
Input:       x₀ ───>  ·        ·        ·
             │        │        │        │
             ▼        │        │        │
Predict:    MGN      MGN      MGN      MGN
             │        │        │        │
             ▼        ▼        ▼        ▼
Output:     x₁ ───> x₂ ───> x₃ ───> x₄
                     └─────>  └─────>

Each prediction becomes input for next timestep
(Errors accumulate!)
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  MESHGRAPHNET IN A NUTSHELL                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  What: GNN for mesh-based physical simulation              │
│  How:  Encode-Process-Decode with Graph Network blocks     │
│  Key:  Updates BOTH nodes AND edges with message passing   │
│                                                             │
│  Architecture:                                              │
│    • Encoder: Embed raw features → latent space (128D)     │
│    • Processor: 15 GN blocks with residual connections     │
│    • Decoder: Map latent → output predictions              │
│                                                             │
│  Core Innovation:                                           │
│    • Edge features encode spatial relationships            │
│    • Two-stage update: edges first, then nodes             │
│    • Residual connections for deep networks                │
│                                                             │
│  Use Cases:                                                 │
│    ✓ Computational Fluid Dynamics (CFD)                    │
│    ✓ Structural Mechanics                                  │
│    ✓ Cloth Simulation                                      │
│    ✓ Any mesh-based physics                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## References

1. **Paper**: Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. W. (2020).
   "Learning Mesh-Based Simulation with Graph Networks." ICML 2021.
   - ArXiv: https://arxiv.org/abs/2010.03409

2. **Official Implementation**:
   - https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

3. **PyTorch Implementations**:
   - https://github.com/echowve/meshGraphNets_pytorch
   - https://github.com/wwMark/meshgraphnets

4. **NVIDIA PhysicsNeMo Documentation**:
   - https://docs.nvidia.com/physicsnemo/latest/user-guide/model_architecture/meshgraphnet.html

---

## Next Steps for Implementation

1. **Understand your data format**
   - Mesh file format (VTK, FEM, etc.)
   - Node feature dimensions
   - Output field variables

2. **Build graph from mesh**
   - Extract connectivity → edge_index
   - Compute edge features from geometry

3. **Implement components**
   - GraphNetworkBlock class
   - MeshGraphNet class
   - Data preprocessing

4. **Training pipeline**
   - Data normalization
   - Loss function
   - Optimizer (Adam recommended)

5. **Evaluation**
   - Rollout error for time-series
   - Comparison with ground truth

---

*End of Documentation*
