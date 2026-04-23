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

    Note: mu/logvar heads are retained for reparameterization; the training
    regularizer is MMD between the sampled z and N(0,I), NOT per-sample KL.
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
    def mmd_loss(z_posterior, kernel_sigmas=(0.5, 1.0, 2.0, 4.0, 8.0)):
        """Multi-scale RBF MMD² between z_posterior and N(0, I).

        Implements the InfoVAE objective (Zhao et al. 2019): match the aggregate
        posterior q(z) to the prior p(z) = N(0, I). This replaces the standard
        VAE per-sample KL term, preventing posterior collapse and ensuring that
        N(0, I) sampling at inference lands in the decoder's training support.

        A multi-bandwidth kernel avoids single-bandwidth sensitivity; the sum
        acts as a characteristic kernel across a range of scales.

        Args:
            z_posterior:    [B, D] reparameterized samples from q(z|x).
            kernel_sigmas:  RBF bandwidths to sum over.

        Returns:
            Scalar MMD² estimate (biased V-statistic). Non-negative.
        """
        z_prior = torch.randn_like(z_posterior)
        mmd_total = z_posterior.new_zeros(())
        for sigma in kernel_sigmas:
            two_sigma_sq = 2.0 * sigma * sigma
            xx = torch.cdist(z_posterior, z_posterior).pow(2)
            yy = torch.cdist(z_prior, z_prior).pow(2)
            xy = torch.cdist(z_posterior, z_prior).pow(2)
            k_xx = torch.exp(-xx / two_sigma_sq)
            k_yy = torch.exp(-yy / two_sigma_sq)
            k_xy = torch.exp(-xy / two_sigma_sq)
            mmd_total = mmd_total + (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        return mmd_total / len(kernel_sigmas)
