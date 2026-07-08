import math
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation

from model.encoder_decoder import GnBlock
from model.mlp import build_mlp, init_weights


class _ConditionalPriorBase(nn.Module):
    """Shared graph trunk for conditional priors on the VAE latent z.

    Encodes the input graph (node/edge features + message passing + attention
    pooling) into one conditioning vector per graph. Subclasses attach either
    an explicit density head (Gaussian mixture) or a velocity head (flow
    matching). Attribute names (node_encoder / edge_encoder / mp_layers / pool)
    are load-bearing: mixture checkpoints saved before this refactor must keep
    identical state_dict keys.
    """

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.z_dim = int(config.get('vae_latent_dim', 32))
        self.hidden_dim = int(config.get('prior_hidden_dim', config.get('latent_dim', 128)))
        self.num_mp_layers = int(config.get('prior_mp_layers', 3))
        # Per-level z slots (matches MeshGraphNets.num_z). Default 1 for back-compat.
        if config.get('use_multiscale', False):
            default_num_z = int(config.get('multiscale_levels', 1)) + 1
        else:
            default_num_z = 1
        self.num_z = int(config.get('num_z', default_num_z))

        base_input_size = int(config.get('input_var'))
        base_input_size += int(config.get('positional_features', 0))
        if config.get('use_node_types', False):
            base_input_size += int(config.get('num_node_types', 0))
        edge_input_size = int(config.get('edge_var'))

        self.node_encoder = build_mlp(base_input_size, self.hidden_dim, self.hidden_dim)
        self.edge_encoder = build_mlp(edge_input_size, self.hidden_dim, self.hidden_dim)
        self.mp_layers = nn.ModuleList([
            GnBlock(self.hidden_dim, use_world_edges=False)
            for _ in range(self.num_mp_layers)
        ])
        # State-dict compatible with the deprecated GlobalAttention (same gate_nn keys).
        self.pool = AttentionalAggregation(nn.Linear(self.hidden_dim, 1))

    def condition(self, graph):
        """Encode graph → pooled conditioning vector [B, hidden_dim]."""
        batch = getattr(graph, 'batch', None)
        if batch is None:
            batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
        h = self.node_encoder(graph.x)
        e = self.edge_encoder(graph.edge_attr)
        g = Data(x=h, edge_attr=e, edge_index=graph.edge_index)
        for block in self.mp_layers:
            g = block(g)
        return self.pool(g.x, batch)


class ConditionalMixturePrior(_ConditionalPriorBase):
    """Graph-conditioned Gaussian mixture prior for VAE latent z (legacy family).

    Kept as the `prior_family gmm` fallback and for loading pre-FM checkpoints.
    At inference it replaces the global latent sampler with p(z | graph).
    """

    family = 'gmm'

    def __init__(self, config):
        super().__init__(config)
        self.num_components = int(config.get('prior_mixture_components', 10))
        self.min_std = float(config.get('prior_min_std', 0.05))
        # Low-rank covariance per mixture component: Sigma_k = L_k L_k^T + diag(psi_k).
        # 0 (default) = diagonal components (back-compat, no cov_factor emitted).
        # r > 0 lets a component capture correlated latent directions — the
        # manufacturing-spread axis a diagonal prior misses (see diag_prior_spread).
        self.cov_rank = int(config.get('prior_cov_rank', 0))
        # Start near-diagonal (weak correlations) and let training grow them.
        self.cov_factor_scale = 0.1
        # Per component: 1 logit + D mean + D diagonal log-std (+ D*rank cov factor).
        self.params_per_comp = 1 + (2 + self.cov_rank) * self.z_dim
        self.head = build_mlp(
            self.hidden_dim,
            self.hidden_dim,
            self.num_z * self.num_components * self.params_per_comp,
            layer_norm=False,
        )
        self.apply(init_weights)

    def forward(self, graph):
        pooled = self.condition(graph)
        raw = self.head(pooled)
        bsz = raw.shape[0]
        D = self.z_dim
        # [B, num_z, K, params_per_comp]
        raw = raw.view(bsz, self.num_z, self.num_components, self.params_per_comp)

        logits = raw[..., 0]                          # [B, num_z, K]
        mu = raw[..., 1:1 + D]                        # [B, num_z, K, D]
        log_std = raw[..., 1 + D:1 + 2 * D]           # [B, num_z, K, D]
        log_std = torch.clamp(log_std, min=math.log(self.min_std), max=5.0)
        out = {'logits': logits, 'mu': mu, 'log_std': log_std}
        if self.cov_rank > 0:
            # Low-rank covariance factor L_k: [B, num_z, K, D, rank]
            factor = raw[..., 1 + 2 * D:].view(
                bsz, self.num_z, self.num_components, D, self.cov_rank)
            out['cov_factor'] = factor * self.cov_factor_scale
        return out

    @torch.no_grad()
    def sample(self, graph, temperature=1.0):
        params = self.forward(graph)
        return sample_from_mixture(params, temperature=temperature)


class ConditionalFMPrior(_ConditionalPriorBase):
    """Graph-conditioned flow-matching prior for VAE latent z (default family).

    Instead of emitting an explicit density, learns a velocity field
    v(z_t, t, c) that transports N(0, I) onto the aggregate posterior of z for
    each graph (conditional flow matching, Lipman et al. 2023; same recipe as
    the LFMGN sampler in tum-pbs/dgn4cfd). Training is plain MSE regression on
    straight-line interpolation paths — no logsumexp, no components, hence none
    of the mixture's collapse machinery (min-std floors, KL anchor, Gumbel
    reparameterization). Sampling integrates the ODE with fixed Euler steps.

    The num_z slots are modeled jointly as one flat vector, so cross-level
    correlations between per-level latents are captured — the mixture treated
    each slot as an independent mixture.

    z is consumed at its native scale: the MMD regularizer already holds the
    aggregate posterior near unit scale, and avoiding running-stat buffers
    keeps EMA/DDP snapshots exact.
    """

    family = 'fm'
    # Small path-endpoint noise floor from the conditional-FM objective;
    # sigma_min = 0 would make t=1 a Dirac endpoint.
    sigma_min = 1e-4

    def __init__(self, config):
        super().__init__(config)
        self.num_steps = int(config.get('prior_fm_steps', 20))
        self.flat_dim = self.num_z * self.z_dim
        # Fourier features of t: [sin(2^k π t), cos(2^k π t)], k = 0..15.
        freqs = (2.0 ** torch.arange(16, dtype=torch.float32)) * math.pi
        self.register_buffer('t_freqs', freqs, persistent=False)
        t_emb_dim = 2 * freqs.numel()
        self.velocity_net = build_mlp(
            self.flat_dim + t_emb_dim + self.hidden_dim,
            self.hidden_dim,
            self.flat_dim,
            layer_norm=False,
        )
        self.apply(init_weights)

    def _t_embed(self, t):
        ang = t * self.t_freqs.view(1, -1)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def velocity(self, z_t, t, cond):
        """v_θ(z_t, t, c): [B, flat_dim] × [B, 1] × [B, hidden] → [B, flat_dim]."""
        return self.velocity_net(torch.cat([z_t, self._t_embed(t), cond], dim=-1))

    def fm_loss(self, cond, target_z):
        """Conditional flow-matching MSE on a detached posterior sample.

        Path: z_t = (1 − (1−σ_min)·t)·z0 + t·z1,  target v* = z1 − (1−σ_min)·z0.
        The regression optimum at (z_t, t) is the marginal velocity
        E[v* | z_t, c], whose ODE flow transports N(0,I) exactly onto the
        distribution of z1 — single-sample noise averages out instead of
        sculpting spurious sharp modes (the mixture-NLL failure).

        Args:
            cond:     [B, hidden_dim] pooled condition. Gradient flows through
                      it into the trunk, so the trunk trains jointly.
            target_z: [B, num_z, D] fresh detached posterior sample.
        Returns scalar loss (fp32).
        """
        with _autocast_disabled_for(cond):
            z1 = target_z.reshape(target_z.shape[0], -1).float()
            c = cond.float()
            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], 1, device=z1.device)
            s = 1.0 - self.sigma_min
            z_t = (1.0 - s * t) * z0 + t * z1
            target_v = z1 - s * z0
            pred_v = self.velocity(z_t, t, c)
            return torch.nn.functional.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def sample(self, graph, temperature=1.0):
        """Sample z of shape [B, num_z, D] — same interface as the mixture."""
        return self.sample_n(graph, 1, temperature=temperature)[:, 0]

    @torch.no_grad()
    def sample_n(self, graph, n, temperature=1.0):
        """Draw n z samples per graph via Euler ODE integration: [B, n, num_z, D].

        The trunk runs once per graph; only the (tiny) velocity net is called
        per step. Temperature scales the initial noise std by sqrt(temperature)
        — the nearest analog of mixture covariance scaling.
        """
        cond = self.condition(graph)
        with _autocast_disabled_for(cond):
            c = cond.float().repeat_interleave(n, dim=0)     # [B*n, hidden]
            bn = c.shape[0]
            z = torch.randn(bn, self.flat_dim, device=c.device)
            z = z * math.sqrt(max(float(temperature), 1e-6))
            dt = 1.0 / self.num_steps
            for k in range(self.num_steps):
                t = torch.full((bn, 1), k * dt, device=c.device)
                z = z + dt * self.velocity(z, t, c)
        B = cond.shape[0]
        return z.view(B, n, self.num_z, self.z_dim).to(cond.dtype)


def build_conditional_prior(config):
    """Instantiate the conditional prior selected by `prior_family`.

    'fm' (default) → ConditionalFMPrior; 'gmm' → ConditionalMixturePrior.
    """
    prior_config = build_prior_config(config)
    family = prior_config.get('prior_family', 'fm')
    if family == 'gmm':
        return ConditionalMixturePrior(prior_config)
    if family != 'fm':
        raise ValueError(f"Unknown prior_family '{family}' (expected 'fm' or 'gmm')")
    return ConditionalFMPrior(prior_config)


def _autocast_disabled_for(tensor):
    device_type = tensor.device.type
    if device_type in ('cuda', 'cpu'):
        return torch.amp.autocast(device_type, enabled=False)
    return nullcontext()


def _lowrank_mvn(mu, log_std, cov_factor):
    """Batched low-rank-plus-diagonal Gaussian: Sigma = L L^T + diag(exp(2*log_std)).

    Construct and consume this distribution with autocast disabled.
    LowRankMultivariateNormal uses a capacitance Cholesky internally, and CUDA
    does not implement Cholesky for bfloat16.
    """
    return torch.distributions.LowRankMultivariateNormal(
        loc=mu.float(),
        cov_factor=cov_factor.float(),
        cov_diag=torch.exp(2.0 * log_std.float()),
        validate_args=False,
    )


def _lowrank_log_prob(mu, log_std, cov_factor, value):
    with _autocast_disabled_for(mu):
        return _lowrank_mvn(mu, log_std, cov_factor).log_prob(value.float())


def _lowrank_sample(mu, log_std, cov_factor, *, temperature=1.0, reparameterized=False):
    temp = max(float(temperature), 1e-6)
    with _autocast_disabled_for(mu):
        mvn = torch.distributions.LowRankMultivariateNormal(
            loc=mu.float(),
            cov_factor=cov_factor.float() * math.sqrt(temp),
            cov_diag=torch.exp(2.0 * log_std.float()) * temp,
            validate_args=False,
        )
        return mvn.rsample() if reparameterized else mvn.sample()


def mixture_nll(params, target_z):
    """Negative log likelihood of target_z under a Gaussian mixture.

    Components are diagonal, or low-rank-plus-diagonal when params has
    'cov_factor' (Sigma_k = L_k L_k^T + diag(exp(2*log_std_k))).

    Shapes:
        target_z:  [B, num_z, D]            or [MC, B, num_z, D] for MC stacks
        logits:    [B, num_z, K]
        mu:        [B, num_z, K, D]
        log_std:   [B, num_z, K, D]
        cov_factor:[B, num_z, K, D, rank]   (optional)
    """
    if target_z.dim() == 4:
        losses = [mixture_nll(params, target_z[i]) for i in range(target_z.shape[0])]
        return torch.stack(losses).mean()

    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']
    cov_factor = params.get('cov_factor')

    z = target_z.unsqueeze(2)  # [B, num_z, 1, D]
    if cov_factor is not None:
        comp_log_prob = _lowrank_log_prob(
            mu, log_std, cov_factor, z,
        )  # [B, num_z, K]
    else:
        var_term = ((z - mu) / torch.exp(log_std)).pow(2)
        comp_log_prob = -0.5 * (
            var_term.sum(dim=-1)
            + 2.0 * log_std.sum(dim=-1)
            + target_z.shape[-1] * math.log(2.0 * math.pi)
        )  # [B, num_z, K]
    log_mix = torch.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_mix + comp_log_prob, dim=-1).mean()


def analytical_prior_kl_loss(params, q_mu, q_logvar):
    """Variational upper bound on KL(q(z|y) || prior_mixture(z|graph)).

    Uses Jensen's inequality on log Σ_k π_k N_k(z):
        log p(z) ≥ Σ_k π_k log N_k(z) + H(π)
    Hence H(q, p) = -E_q[log p] ≤ -Σ_k π_k E_q[log N_k(z)] - H(π)

    E_q[log N(z | μ_k, σ_k²)] is closed-form when q is diagonal Gaussian:
        = -½ [D log 2π + Σ_d log σ_k_d² + Σ_d (σ_q_d² + (μ_q_d - μ_k_d)²) / σ_k_d²]

    Critically, this loss is computed against the full posterior distribution
    (μ_q, σ_q), not against single Monte Carlo samples. That eliminates the
    component-overfitting failure of MC NLL where prior components collapse to
    individual posterior samples rather than covering the posterior's spread.

    Args:
        params:    dict with 'logits' [B, K], 'mu' [B, K, D], 'log_std' [B, K, D]
        q_mu:      posterior mean [B, D]
        q_logvar:  posterior log variance [B, D]

    Returns:
        Scalar loss (upper bound on KL(q || p) up to the constant H(q)).
    """
    # Per-level shapes:
    #   q_mu, q_logvar: [B, num_z, D]
    #   logits:         [B, num_z, K]
    #   mu, log_std:    [B, num_z, K, D]
    logits = params['logits']
    mu = params['mu']
    log_std = params['log_std']

    q_mu_b = q_mu.unsqueeze(2)                # [B, num_z, 1, D]
    q_var_b = torch.exp(q_logvar).unsqueeze(2)
    D = q_mu.shape[-1]

    var_k = torch.exp(2.0 * log_std)
    cov_factor = params.get('cov_factor')
    if cov_factor is not None:
        # Small stability anchor: match the prior's per-dim MARGINAL variance
        # (diagonal of L L^T + diag); cross-correlations are left to mc_nll.
        var_k = var_k + cov_factor.pow(2).sum(dim=-1)
    log_var_k = torch.log(var_k.clamp_min(1e-8))

    expected_log_pk = -0.5 * (
        D * math.log(2.0 * math.pi)
        + log_var_k.sum(dim=-1)
        + ((q_var_b + (q_mu_b - mu).pow(2)) / var_k).sum(dim=-1)
    )  # [B, num_z, K]

    log_pi = torch.log_softmax(logits, dim=-1)
    pi = log_pi.exp()

    weighted_term = -(pi * expected_log_pk).sum(dim=-1)   # [B, num_z]
    entropy_pi = -(pi * log_pi).sum(dim=-1)               # [B, num_z]

    return (weighted_term - entropy_pi).mean()


def sample_from_mixture(params, temperature=1.0):
    """Sample z of shape [B, num_z, D]."""
    logits = params['logits']        # [B, num_z, K]
    mu = params['mu']                # [B, num_z, K, D]
    log_std = params['log_std']      # [B, num_z, K, D]

    temp = max(float(temperature), 1e-6)
    cat = torch.distributions.Categorical(logits=logits / temp)
    component = cat.sample()         # [B, num_z]

    B, num_z = component.shape
    b_idx = torch.arange(B, device=logits.device).view(B, 1).expand(B, num_z)
    z_idx = torch.arange(num_z, device=logits.device).view(1, num_z).expand(B, num_z)

    chosen_mu = mu[b_idx, z_idx, component]                  # [B, num_z, D]
    chosen_log_std = log_std[b_idx, z_idx, component]        # [B, num_z, D]

    cov_factor = params.get('cov_factor')
    if cov_factor is not None:
        chosen_factor = cov_factor[b_idx, z_idx, component]  # [B, num_z, D, rank]
        # Temperature scales the covariance by `temp` (std by sqrt(temp)).
        return _lowrank_sample(
            chosen_mu, chosen_log_std, chosen_factor, temperature=temp,
        ).to(chosen_mu.dtype)

    chosen_std = torch.exp(chosen_log_std) * math.sqrt(temp)
    return chosen_mu + chosen_std * torch.randn_like(chosen_std)


def build_prior_config(config):
    use_multiscale = bool(config.get('use_multiscale', False))
    default_num_z = (int(config.get('multiscale_levels', 1)) + 1) if use_multiscale else 1
    return {
        'input_var': config.get('input_var'),
        'edge_var': config.get('edge_var'),
        'latent_dim': config.get('latent_dim'),
        'vae_latent_dim': config.get('vae_latent_dim'),
        'positional_features': config.get('positional_features', 0),
        'use_node_types': config.get('use_node_types', False),
        'num_node_types': config.get('num_node_types', 0),
        'use_multiscale': use_multiscale,
        'multiscale_levels': config.get('multiscale_levels', 1),
        'num_z': int(config.get('num_z', default_num_z)),
        'prior_family': str(config.get('prior_family', 'fm')).lower().strip(),
        'prior_hidden_dim': config.get('prior_hidden_dim', config.get('latent_dim')),
        'prior_mp_layers': config.get('prior_mp_layers', 10),
        # fm family only:
        'prior_fm_steps': config.get('prior_fm_steps', 20),
        # gmm family only:
        'prior_mixture_components': config.get('prior_mixture_components', 50),
        'prior_min_std': config.get('prior_min_std', 0.1),
        'prior_cov_rank': config.get('prior_cov_rank', 0),
    }
