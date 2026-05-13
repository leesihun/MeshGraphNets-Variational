import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention

from model.encoder_decoder import GnBlock
from model.mlp import build_mlp, init_weights


class ConditionalMixturePrior(nn.Module):
    """Graph-conditioned Gaussian mixture prior for VAE latent z.

    This module is trained post-hoc against the frozen VAE posterior.  At
    inference it replaces the global latent sampler with p(z | graph).
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.z_dim = int(config.get('vae_latent_dim', 32))
        self.num_components = int(config.get('prior_mixture_components', 10))
        self.hidden_dim = int(config.get('prior_hidden_dim', config.get('latent_dim', 128)))
        self.num_mp_layers = int(config.get('prior_mp_layers', 3))
        self.min_std = float(config.get('prior_min_std', 1e-3))

        base_input_size = int(config.get('input_var'))
        base_input_size += int(config.get('positional_features', 0))
        if config.get('use_node_types', False):
            base_input_size += int(config.get('num_node_types', 0))
        edge_input_size = int(config.get('edge_var'))

        self.node_encoder = build_mlp(base_input_size, self.hidden_dim, self.hidden_dim)
        self.edge_encoder = build_mlp(edge_input_size, self.hidden_dim, self.hidden_dim)
        block_config = {
            'residual_scale': config.get('residual_scale', 1.0),
            'use_pairnorm': config.get('prior_use_pairnorm', False),
        }
        self.mp_layers = nn.ModuleList([
            GnBlock(block_config, self.hidden_dim, use_world_edges=False)
            for _ in range(self.num_mp_layers)
        ])
        self.pool = GlobalAttention(nn.Linear(self.hidden_dim, 1))
        self.head = build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            self.num_components * (1 + 2 * self.z_dim),
            layer_norm=False,
        )
        self.apply(init_weights)

    def forward(self, graph):
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)

        h = self.node_encoder(graph.x)
        e = self.edge_encoder(graph.edge_attr)
        g = Data(x=h, edge_attr=e, edge_index=graph.edge_index)
        for block in self.mp_layers:
            g = block(g)

        pooled = self.pool(g.x, batch)
        raw = self.head(pooled)
        bsz = raw.shape[0]
        raw = raw.view(bsz, self.num_components, 1 + 2 * self.z_dim)

        logits = raw[:, :, 0]
        mu = raw[:, :, 1:1 + self.z_dim]
        log_std = raw[:, :, 1 + self.z_dim:]
        log_std = torch.clamp(log_std, min=math.log(self.min_std), max=5.0)
        return {'logits': logits, 'mu': mu, 'log_std': log_std}

    @torch.no_grad()
    def sample(self, graph, temperature=1.0):
        params = self.forward(graph)
        return sample_from_mixture(params, temperature=temperature)


def mixture_nll(params, target_z):
    """Negative log likelihood of target_z under a diagonal Gaussian mixture."""
    if target_z.dim() == 3:
        losses = [mixture_nll(params, target_z[i]) for i in range(target_z.shape[0])]
        return torch.stack(losses).mean()

    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']

    z = target_z.unsqueeze(1)
    var_term = ((z - mu) / torch.exp(log_std)).pow(2)
    comp_log_prob = -0.5 * (
        var_term.sum(dim=-1)
        + 2.0 * log_std.sum(dim=-1)
        + target_z.shape[-1] * math.log(2.0 * math.pi)
    )
    log_mix = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_mix + comp_log_prob, dim=-1).mean()


def sample_from_mixture(params, temperature=1.0):
    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']

    temp = max(float(temperature), 1e-6)
    cat = torch.distributions.Categorical(logits=logits / temp)
    component = cat.sample()

    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    chosen_mu = mu[batch_idx, component]
    chosen_std = torch.exp(log_std[batch_idx, component]) * math.sqrt(temp)
    return chosen_mu + chosen_std * torch.randn_like(chosen_std)


def build_prior_config(config):
    return {
        'input_var': config.get('input_var'),
        'edge_var': config.get('edge_var'),
        'latent_dim': config.get('latent_dim'),
        'vae_latent_dim': config.get('vae_latent_dim'),
        'positional_features': config.get('positional_features', 0),
        'use_node_types': config.get('use_node_types', False),
        'num_node_types': config.get('num_node_types', 0),
        'prior_mixture_components': config.get('prior_mixture_components', 10),
        'prior_hidden_dim': config.get('prior_hidden_dim', config.get('latent_dim')),
        'prior_mp_layers': config.get('prior_mp_layers', 3),
        'prior_min_std': config.get('prior_min_std', 1e-3),
        'residual_scale': config.get('residual_scale', 1.0),
    }
