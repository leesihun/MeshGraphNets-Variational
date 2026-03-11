"""
Gradient Checkpointing Utilities for MeshGraphNets

Reduces VRAM usage by recomputing activations during backward pass
instead of storing them. Trades ~20-30% more compute for ~60-70% less memory.

Usage:
    Set `use_checkpointing True` in config.txt to enable.
"""

import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data


def checkpoint_gn_block(block, x, edge_attr, edge_index, world_edge_attr=None, world_edge_index=None):
    """
    Run a GnBlock with gradient checkpointing.

    Args:
        block: GnBlock module
        x: Node features [N, latent_dim]
        edge_attr: Edge features [E, latent_dim]
        edge_index: Edge connectivity [2, E]
        world_edge_attr: World edge features [E_world, latent_dim] (optional)
        world_edge_index: World edge connectivity [2, E_world] (optional)

    Returns:
        Tuple of (updated_x, updated_edge_attr, updated_world_edge_attr, updated_world_edge_index)
    """
    def run_block(x, edge_attr, edge_index, world_edge_attr, world_edge_index):
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if world_edge_attr is not None and world_edge_index is not None:
            graph.world_edge_attr = world_edge_attr
            graph.world_edge_index = world_edge_index
        out = block(graph)
        out_world_edge_attr = out.world_edge_attr if hasattr(out, 'world_edge_attr') else None
        out_world_edge_index = out.world_edge_index if hasattr(out, 'world_edge_index') else None
        return out.x, out.edge_attr, out_world_edge_attr, out_world_edge_index

    return checkpoint(
        run_block,
        x,
        edge_attr,
        edge_index,
        world_edge_attr,
        world_edge_index,
        use_reentrant=False
    )


def process_with_checkpointing(processor_list, graph):
    """
    Run processor blocks with gradient checkpointing.

    Args:
        processor_list: nn.ModuleList of GnBlock modules
        graph: PyG Data object (after encoding)

    Returns:
        PyG Data object with updated features
    """
    x = graph.x
    edge_attr = graph.edge_attr
    edge_index = graph.edge_index
    world_edge_attr = graph.world_edge_attr if hasattr(graph, 'world_edge_attr') else None
    world_edge_index = graph.world_edge_index if hasattr(graph, 'world_edge_index') else None

    for block in processor_list:
        x, edge_attr, world_edge_attr, world_edge_index = checkpoint_gn_block(
            block, x, edge_attr, edge_index, world_edge_attr, world_edge_index
        )

    out = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    if world_edge_attr is not None and world_edge_index is not None:
        out.world_edge_attr = world_edge_attr
        out.world_edge_index = world_edge_index
    return out
