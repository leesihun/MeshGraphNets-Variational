import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention

from model.encoder_decoder import GnBlock
from model.mlp import build_mlp


class GNNVariationalEncoder(nn.Module):
    """GNN-based cVAE encoder: encodes target delta into a global stochastic latent z.

    Runs message passing on the mesh graph before pooling, so z captures spatially
    correlated patterns in the deformation field rather than just per-node averages.

    Architecture:
        y [N, output_var]  → node_encoder MLP  → [N, latent_dim]
        edge_attr [E, 8]   → edge_encoder MLP  → [E, latent_dim]
        → num_mp_layers GnBlocks (message passing on mesh topology)
        → GlobalAttention pool → [B, latent_dim]
        → mu_head, logvar_head → z [B, vae_latent_dim]
    """

    def __init__(self, node_output_size, edge_input_size, latent_dim, vae_latent_dim, num_mp_layers=2):
        super().__init__()
        self.node_encoder = build_mlp(node_output_size, latent_dim, latent_dim)
        self.edge_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        minimal_config = {'residual_scale': 1.0, 'use_pairnorm': False}
        self.mp_layers = nn.ModuleList([
            GnBlock(minimal_config, latent_dim, use_world_edges=False)
            for _ in range(num_mp_layers)
        ])
        self.attention_pool = GlobalAttention(nn.Linear(latent_dim, 1))
        self.mu_head = nn.Linear(latent_dim, vae_latent_dim)
        self.logvar_head = nn.Linear(latent_dim, vae_latent_dim)

    def forward(self, y, edge_index, edge_attr, batch):
        """
        Args:
            y:          [N, node_output_size] per-node target delta
            edge_index: [2, E] mesh edge connectivity
            edge_attr:  [E, 8] raw edge features
            batch:      [N] PyG batch assignment index
        Returns:
            z:      [B, vae_latent_dim] reparameterized latent
            mu:     [B, vae_latent_dim]
            logvar: [B, vae_latent_dim]
        """
        h = self.node_encoder(y)
        e = self.edge_encoder(edge_attr)
        g = Data(x=h, edge_attr=e, edge_index=edge_index)
        for mp in self.mp_layers:
            g = mp(g)
        h_graph = self.attention_pool(g.x, batch)
        mu = self.mu_head(h_graph)
        logvar = self.logvar_head(h_graph)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        """KL(q||p) = -1/2 * mean(1 + logvar - mu^2 - exp(logvar))"""
        return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
