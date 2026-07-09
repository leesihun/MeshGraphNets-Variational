import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation

from model.encoder_decoder import GnBlock
from model.mlp import build_mlp


class GNNVariationalEncoder(nn.Module):
    """GNN-based cVAE encoder: encodes target delta into a global stochastic latent z.

    Runs message passing on the mesh graph before pooling, so z captures spatially
    correlated patterns in the deformation field rather than just per-node averages.

    Architecture (graph_aware=False):
        y [N, output_var]  → node_encoder MLP  → [N, latent_dim]
        edge_attr [E, 8]   → edge_encoder MLP  → [E, latent_dim]
        → num_mp_layers GnBlocks (message passing on mesh topology)
        → GlobalAttention pool → [B, latent_dim]
        → mu_head, logvar_head → z [B, num_z, vae_latent_dim]

    Architecture (graph_aware=True):
        y [N, output_var]  → node_encoder MLP        → h_y [N, latent_dim]
        x [N, node_input]  → node_input_encoder MLP  → h_x [N, latent_dim]
        cat(h_y, h_x)      → node_fuse MLP           → h   [N, latent_dim]
        ... (same MP, pool, heads as above)
    With graph awareness, z is incentivized to encode only the y-residual that
    the input graph does not already explain — keeping z mesh-type-orthogonal.

    Note: mu/logvar heads are retained for reparameterization; the training
    regularizer is MMD between the sampled z and N(0,I), NOT per-sample KL.
    """

    def __init__(self, node_output_size, edge_input_size, latent_dim, vae_latent_dim,
                 num_mp_layers=2, node_input_size=None, graph_aware=False,
                 posterior_min_std=0, num_z=1):
        super().__init__()
        self.graph_aware = bool(graph_aware)
        self.min_logvar = 2.0 * math.log(max(float(posterior_min_std), 1e-6))
        self.num_z = int(num_z)
        self.vae_latent_dim = int(vae_latent_dim)
        self.node_encoder = build_mlp(node_output_size, latent_dim, latent_dim)
        if self.graph_aware:
            if node_input_size is None:
                raise ValueError("graph_aware=True requires node_input_size")
            self.node_input_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
            self.node_fuse = build_mlp(2 * latent_dim, latent_dim, latent_dim)
        self.edge_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        self.mp_layers = nn.ModuleList([
            GnBlock(latent_dim, use_world_edges=False)
            for _ in range(num_mp_layers)
        ])
        # State-dict compatible with the deprecated GlobalAttention (same gate_nn keys).
        self.attention_pool = AttentionalAggregation(nn.Linear(latent_dim, 1))
        self.mu_head = nn.Linear(latent_dim, self.num_z * self.vae_latent_dim)
        self.logvar_head = nn.Linear(latent_dim, self.num_z * self.vae_latent_dim)

    def forward(self, y, edge_index, edge_attr, batch, x=None):
        """
        Args:
            y:          [N, node_output_size] per-node target delta
            edge_index: [2, E] mesh edge connectivity
            edge_attr:  [E, 8] raw edge features
            batch:      [N] PyG batch assignment index
            x:          [N, node_input_size] per-node input features.
                        Required when graph_aware=True; ignored otherwise.
        Returns:
            z:      [B, num_z, vae_latent_dim] reparameterized latent
            mu:     [B, num_z, vae_latent_dim]
            logvar: [B, num_z, vae_latent_dim]
        """
        h_y = self.node_encoder(y)
        if self.graph_aware:
            if x is None:
                raise ValueError("graph_aware=True but x is None")
            h_x = self.node_input_encoder(x)
            h = self.node_fuse(torch.cat([h_y, h_x], dim=-1))
        else:
            h = h_y
        e = self.edge_encoder(edge_attr)
        g = Data(x=h, edge_attr=e, edge_index=edge_index)
        for mp in self.mp_layers:
            g = mp(g)
        h_graph = self.attention_pool(g.x, batch)
        B = h_graph.shape[0]
        mu = self.mu_head(h_graph).view(B, self.num_z, self.vae_latent_dim)
        logvar = self.logvar_head(h_graph).view(B, self.num_z, self.vae_latent_dim).clamp(min=self.min_logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @staticmethod
    def mmd_loss(z_posterior, kernel_sigmas=(0.5, 1.0, 2.0, 4.0, 8.0), bandwidth='fixed'):
        """Multi-scale RBF MMD² between z_posterior and N(0, I).

        Implements the InfoVAE objective (Zhao et al. 2019): match the aggregate
        posterior q(z) to the prior p(z) = N(0, I). This replaces the standard
        VAE per-sample KL term, preventing posterior collapse and ensuring that
        N(0, I) sampling at inference lands in the decoder's training support.

        A multi-bandwidth kernel avoids single-bandwidth sensitivity; the sum
        acts as a characteristic kernel across a range of scales.

        Args:
            z_posterior:    [B, D] or [B, num_z, D] reparameterized samples from q(z|x).
                            For per-level z, MMD is computed independently per slot and
                            averaged so each slot is regularised against N(0,I) on its own.
            kernel_sigmas:  RBF bandwidths to sum over (used only for bandwidth='fixed').
            bandwidth:      'fixed'  — constant `kernel_sigmas`. These are tuned for
                              ~64-D z and SATURATE at high vae_latent_dim: pairwise
                              squared distances grow ~2D, so every RBF kernel → 0 and
                              the penalty (and its gradient) vanishes, silently
                              disabling the regularizer.
                            'median' — dimension-adaptive. Sets the base 2σ² from the
                              median pairwise squared distance each step and keeps the
                              multi-scale multipliers, so the kernel stays in its
                              sensitive regime at any latent dim.

        Returns:
            Scalar MMD² estimate (biased V-statistic). Non-negative.
        """
        if z_posterior.dim() == 3:
            mmd_acc = z_posterior.new_zeros(())
            num_z = z_posterior.shape[1]
            for i in range(num_z):
                mmd_acc = mmd_acc + GNNVariationalEncoder.mmd_loss(
                    z_posterior[:, i, :], kernel_sigmas=kernel_sigmas, bandwidth=bandwidth
                )
            return mmd_acc / num_z

        z_prior = torch.randn_like(z_posterior)
        # Compute squared pairwise distances once; reuse across all kernel bandwidths.
        # Reduces from 15 cdist calls (5 sigmas × 3) to 3 total.
        xx = torch.cdist(z_posterior, z_posterior).pow(2)
        yy = torch.cdist(z_prior, z_prior).pow(2)
        xy = torch.cdist(z_posterior, z_prior).pow(2)
        if str(bandwidth).lower().strip() == 'median':
            # Median-heuristic base scale (detached: the bandwidth is a constant,
            # not backpropped). The cross term xy has no trivial self-distances.
            med = torch.clamp(torch.median(xy.detach()), min=1e-6)
            two_sigma_sqs = [med * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
        else:
            two_sigma_sqs = [z_posterior.new_tensor(2.0 * s * s) for s in kernel_sigmas]
        mmd_total = z_posterior.new_zeros(())
        for two_sigma_sq in two_sigma_sqs:
            mmd_total = mmd_total + (
                torch.exp(-xx / two_sigma_sq).mean()
                + torch.exp(-yy / two_sigma_sq).mean()
                - 2.0 * torch.exp(-xy / two_sigma_sq).mean()
            )
        return mmd_total / len(two_sigma_sqs)
