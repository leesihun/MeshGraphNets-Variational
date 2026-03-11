import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.data import Data


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr = graph.x 
        senders_idx, receivers_idx = graph.edge_index
        edge_attr = graph.edge_attr

        edges_to_collect = []

        senders_attr = node_attr[senders_idx]   # sender nodal features 
        receivers_attr = node_attr[receivers_idx]# Receiver nodal features

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr) # edge features

        # All three features are concatenated along the feature dimension

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_attr = self.net(collected_edges)   
        # Update edge features via Edge block w.r.t. sender, receiver nodal attribute and its edge attribute

        return Data(x=node_attr, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr # [E, 4] (3D)
        nodes_to_collect = []
        
        _, receivers_idx = graph.edge_index # [E, 2] (sender, receiver)
        num_nodes = graph.num_nodes
        # Use sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate implementation)
        # For physics: forces/stresses from neighbors should add up, not average
        agg_received_edges = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        
        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # Aggregate mesh edges
        mesh_edge_attr = graph.edge_attr
        _, mesh_receivers = graph.edge_index
        num_nodes = graph.num_nodes
        # Use sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate implementation)
        mesh_agg = scatter(mesh_edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')

        # Aggregate world edges (if present)
        if (hasattr(graph, 'world_edge_attr') and hasattr(graph, 'world_edge_index')
            and graph.world_edge_attr is not None and graph.world_edge_index.shape[1] > 0):
            world_edge_attr = graph.world_edge_attr
            _, world_receivers = graph.world_edge_index
            # Use sum aggregation for world edges as well
            world_agg = scatter(world_edge_attr, world_receivers, dim=0, dim_size=num_nodes, reduce='sum')
        else:
            world_agg = torch.zeros_like(mesh_agg)

        # Concatenate node features with both aggregations
        collected_nodes = torch.cat([graph.x, mesh_agg, world_agg], dim=-1)
        x = self.net(collected_nodes)

        return Data(
            x=x,
            edge_attr=mesh_edge_attr,
            edge_index=graph.edge_index,
            world_edge_attr=graph.world_edge_attr if hasattr(graph, 'world_edge_attr') else None,
            world_edge_index=graph.world_edge_index if hasattr(graph, 'world_edge_index') else None
        )
