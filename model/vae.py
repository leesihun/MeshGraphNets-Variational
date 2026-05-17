import math

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

    Architecture (graph_aware=False, default — backward compatible):
        y [N, output_var]  → node_encoder MLP  → [N, latent_dim]
        edge_attr [E, 8]   → edge_encoder MLP  → [E, latent_dim]
        → num_mp_layers GnBlocks (message passing on mesh topology)
        → GlobalAttention pool → [B, latent_dim]
        → mu_head, logvar_head → z [B, vae_latent_dim]

    Architecture (graph_aware=True):
        y [N, output_var]  → node_encoder MLP        → h_y [N, latent_dim]
        x [N, node_input]  → node_input_encoder MLP  → h_x [N, latent_dim]
        cat(h_y, h_x)      → node_fuse MLP           → h   [N, latent_dim]
        ... (same MP, pool, heads as above)
    With graph awareness, z is incentivized to encode only the y-residual that
    the input graph does not already explain — keeping z mesh-type-orthogonal.

    Note: mu/logvar heads are retained for reparameterization; the training
    regularizer is MMD between the sampled z and N(0,I), NOT per-sample KL.
    Free-bits KL (via the kl_loss static method) is available as an opt-in
    collapse safeguard with zero gradient on healthy posteriors.
    """

    def __init__(self, node_output_size, edge_input_size, latent_dim, vae_latent_dim,
                 num_mp_layers=2, node_input_size=None, graph_aware=False,
                 posterior_min_std=0.1):
        super().__init__()
        self.graph_aware = bool(graph_aware)
        self.min_logvar = 2.0 * math.log(max(float(posterior_min_std), 1e-6))
        self.node_encoder = build_mlp(node_output_size, latent_dim, latent_dim)
        if self.graph_aware:
            if node_input_size is None:
                raise ValueError("graph_aware=True requires node_input_size")
            self.node_input_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
            self.node_fuse = build_mlp(2 * latent_dim, latent_dim, latent_dim)
        self.edge_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        minimal_config = {'residual_scale': 1.0, 'use_pairnorm': False}
        self.mp_layers = nn.ModuleList([
            GnBlock(minimal_config, latent_dim, use_world_edges=False)
            for _ in range(num_mp_layers)
        ])
        self.attention_pool = GlobalAttention(nn.Linear(latent_dim, 1))
        self.mu_head = nn.Linear(latent_dim, vae_latent_dim)
        self.logvar_head = nn.Linear(latent_dim, vae_latent_dim)

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
            z:      [B, vae_latent_dim] reparameterized latent
            mu:     [B, vae_latent_dim]
            logvar: [B, vae_latent_dim]
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
        mu = self.mu_head(h_graph)
        logvar = self.logvar_head(h_graph).clamp(min=self.min_logvar)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar, free_bits=0.0):
        """Per-sample, per-dim KL[q(z|y) || N(0,I)] with optional free-bits floor.

        kl_per_dim = 0.5 * (mu**2 + exp(logvar) - 1 - logvar)  shape [B, D]

        With free_bits = 0: returns (zero, zero). No KL contribution to loss.
        With free_bits > 0: clamps each (sample, dim) entry to a floor before
        summing over D and averaging over B. The clamp is per-(sample, dim) —
        NOT after batch reduction — so a single highly-active sample cannot
        mask the collapse of other samples (e.g. minority mesh types).

        Why this is safe (no KL-disaster recurrence):
            For a healthy posterior μ≈0, σ≈1: kl_per_dim ≈ 0 → clamped to
            free_bits (a constant) → gradient w.r.t. mu/logvar is exactly 0.
            The floor activates only when something tries to collapse.

        Args:
            mu:        [B, D] posterior mean
            logvar:    [B, D] posterior log-variance
            free_bits: per-dim floor in nats. 0.0 disables.
        Returns:
            (kl_clamped, kl_raw): both scalar tensors.
            kl_clamped is what gets added to loss (if free_bits > 0).
            kl_raw is the un-clamped KL for diagnostics.
        """
        if free_bits <= 0.0:
            zero = mu.new_zeros(())
            return zero, zero
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_raw = kl_per_dim.sum(dim=-1).mean()
        kl_clamped = torch.clamp(kl_per_dim, min=free_bits).sum(dim=-1).mean()
        return kl_clamped, kl_raw

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
