"""
Post-hoc Normalizing Flow for cVAE Latent Space.

Trains a RealNVP normalizing flow on collected VAE encoder mu vectors,
learning the bijective mapping N(0,1) -> actual aggregate posterior.
At inference, replaces raw N(0,1) sampling with flow-transformed sampling
so generated z values follow the true training distribution.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# RealNVP Components
# ---------------------------------------------------------------------------

class CouplingLayer(nn.Module):
    """Single affine coupling layer for RealNVP.

    Splits input by a binary mask. The masked (fixed) dimensions condition
    two small MLPs that produce scale (s) and shift (t) for the remaining
    dimensions.  Forward = base -> data (sampling).  Inverse = data -> base
    (training).
    """

    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        """Forward: base -> data (sampling direction)."""
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.shift_net(x_masked) * (1 - self.mask)
        s = torch.clamp(s, -5.0, 5.0)
        z = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z):
        """Inverse: data -> base (training direction)."""
        z_masked = z * self.mask
        s = self.scale_net(z_masked) * (1 - self.mask)
        t = self.shift_net(z_masked) * (1 - self.mask)
        s = torch.clamp(s, -5.0, 5.0)
        x = z_masked + (1 - self.mask) * (z - t) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        return x, log_det


class LatentRealNVP(nn.Module):
    """RealNVP normalizing flow for the VAE latent space.

    Learns a bijection  f: R^D -> R^D  such that  f(N(0,I)) ≈ q(z),
    where q(z) is the aggregate posterior of the trained VAE encoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space (= vae_latent_dim).
    hidden_dim : int
        Hidden layer width inside scale/shift MLPs.
    num_coupling_layers : int
        Number of coupling layers (alternating masks).
    """

    def __init__(self, latent_dim=8, hidden_dim=64, num_coupling_layers=6):
        super().__init__()
        self.latent_dim = latent_dim

        # Registered buffers for data normalization (set during training)
        self.register_buffer('data_mean', torch.zeros(latent_dim))
        self.register_buffer('data_std', torch.ones(latent_dim))

        # Build alternating binary masks
        half = latent_dim // 2
        mask_a = torch.cat([torch.ones(half), torch.zeros(latent_dim - half)])
        mask_b = 1.0 - mask_a

        layers = []
        for i in range(num_coupling_layers):
            mask = mask_a if i % 2 == 0 else mask_b
            layers.append(CouplingLayer(latent_dim, hidden_dim, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, u):
        """Sampling direction: u ~ N(0,I) -> z (data space).

        Returns (z, total_log_det).
        """
        z = u
        total_log_det = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for layer in self.layers:
            z, log_det = layer.forward(z)
            total_log_det = total_log_det + log_det
        # Un-normalize: map from standardized space back to original mu scale
        z = z * self.data_std + self.data_mean
        return z, total_log_det

    def inverse(self, z):
        """Training direction: z (data space) -> u (base space).

        Returns (u, total_log_det).
        """
        # Normalize: map from original mu scale to standardized space
        z = (z - self.data_mean) / self.data_std
        u = z
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            u, log_det = layer.inverse(u)
            total_log_det = total_log_det + log_det
        return u, total_log_det

    def log_prob(self, z):
        """Exact log-likelihood: log p(z) = log p_base(f^{-1}(z)) + log|det(J^{-1})|."""
        u, log_det = self.inverse(z)
        log_p_u = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_p_u + log_det


# ---------------------------------------------------------------------------
# Mu Collection
# ---------------------------------------------------------------------------

def collect_vae_mus(model, dataloader, device, config):
    """Collect posterior mu vectors from the trained VAE encoder.

    Runs the encoder in eval mode over the entire dataloader, returning
    the deterministic mu (not sampled z) for each graph.

    Parameters
    ----------
    model : MeshGraphNets
        Trained model (raw nn.Module, not DDP-wrapped).
    dataloader : DataLoader
        Training set dataloader.
    device : torch.device
        Device to run on.
    config : dict
        Config dict (used for AMP settings).

    Returns
    -------
    mus : Tensor [N_samples, vae_latent_dim]  on CPU
    """
    # Unwrap torch.compile if present
    raw_model = model
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod
    vae_encoder = raw_model.model.vae_encoder

    use_amp = config.get('use_amp', False)
    amp_dtype = torch.bfloat16

    was_training = model.training
    model.eval()

    mus = []
    with torch.no_grad():
        for graph in dataloader:
            graph = graph.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                _, mu, _ = vae_encoder(
                    graph.y, graph.edge_index, graph.edge_attr,
                    graph.batch if hasattr(graph, 'batch') and graph.batch is not None
                    else torch.zeros(graph.y.shape[0], dtype=torch.long, device=device),
                )
            mus.append(mu.float().cpu())

    if was_training:
        model.train()

    return torch.cat(mus, dim=0)


# ---------------------------------------------------------------------------
# Flow Training
# ---------------------------------------------------------------------------

def train_latent_flow(mu_data, latent_dim=8, device='cpu',
                      hidden_dim=64, num_coupling_layers=6,
                      lr=1e-3, max_epochs=500, batch_size=256,
                      patience=50):
    """Train a RealNVP flow on collected mu vectors.

    Parameters
    ----------
    mu_data : Tensor [N, latent_dim]
        Collected mu vectors from the VAE encoder.
    latent_dim : int
        Latent space dimensionality.
    device : str or torch.device
        Device for training.
    hidden_dim, num_coupling_layers : int
        Flow architecture hyperparameters.
    lr, max_epochs, batch_size, patience : training hyperparameters.

    Returns
    -------
    flow : LatentRealNVP  (eval mode, on CPU)
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Compute normalization stats from the data
    data_mean = mu_data.mean(dim=0)
    data_std = mu_data.std(dim=0).clamp(min=1e-6)

    # Build flow
    flow = LatentRealNVP(latent_dim, hidden_dim, num_coupling_layers).to(device)
    flow.data_mean.copy_(data_mean)
    flow.data_std.copy_(data_std)

    # Train / val split (90/10)
    n_total = mu_data.shape[0]
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(
        TensorDataset(mu_data),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val_nll = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        flow.train()
        train_nll_sum = 0.0
        train_count = 0
        for (batch_z,) in train_loader:
            batch_z = batch_z.to(device)
            nll = -flow.log_prob(batch_z).mean()
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()
            train_nll_sum += nll.item() * batch_z.shape[0]
            train_count += batch_z.shape[0]

        # --- Validate ---
        flow.eval()
        val_nll_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch_z,) in val_loader:
                batch_z = batch_z.to(device)
                nll = -flow.log_prob(batch_z).mean()
                val_nll_sum += nll.item() * batch_z.shape[0]
                val_count += batch_z.shape[0]

        val_nll = val_nll_sum / val_count

        if epoch % 50 == 0 or epoch == 1:
            train_nll = train_nll_sum / train_count
            print(f"  [Flow] Epoch {epoch:4d}  train_nll={train_nll:.4f}  val_nll={val_nll:.4f}")

        # Early stopping
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [Flow] Early stopping at epoch {epoch} (patience={patience})")
                break

    # Load best model
    flow.load_state_dict(best_state)
    flow.eval()
    flow.cpu()
    print(f"  [Flow] Training complete. Best val NLL: {best_val_nll:.4f}")
    return flow


# ---------------------------------------------------------------------------
# Top-level Orchestrators
# ---------------------------------------------------------------------------

def run_posthoc_flow_training(checkpoint_path, train_dataset, config, device):
    """Train a normalizing flow on VAE latent codes and save to checkpoint.

    This is the single entry point called by training profiles.  It handles:
    loading the best checkpoint, collecting mu vectors, training the flow,
    and re-saving the checkpoint with the flow state.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved checkpoint file.
    train_dataset : Dataset
        Training dataset (PyG).
    config : dict
        Full config dict.
    device : torch.device or str
        Device to use.
    """
    from model.MeshGraphNets import MeshGraphNets

    print("\n" + "=" * 60)
    print("POST-HOC LATENT FLOW TRAINING")
    print("=" * 60)

    device = torch.device(device) if isinstance(device, str) else device

    # 1. Load best checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Create fresh model and load best weights (prefer EMA)
    flow_model = MeshGraphNets(config, str(device)).to(device)
    if 'ema_state_dict' in checkpoint:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema_tmp = AveragedModel(flow_model, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))
        ema_tmp.load_state_dict(checkpoint['ema_state_dict'])
        flow_model.load_state_dict(
            {k: v for k, v in ema_tmp.module.state_dict().items()}
        )
        del ema_tmp
        print("  Using EMA weights for mu collection")
    else:
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        print("  Using training weights for mu collection")
    flow_model.eval()

    # 3. Collect mu vectors
    num_workers = config.get('num_workers', 0)
    mu_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )
    mus = collect_vae_mus(flow_model, mu_loader, device, config)
    print(f"  Collected {mus.shape[0]} mu vectors, dim={mus.shape[1]}")
    print(f"  mu mean: {mus.mean(0).tolist()}")
    print(f"  mu std:  {mus.std(0).tolist()}")

    del flow_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Train the normalizing flow
    vae_latent_dim = int(config.get('vae_latent_dim', 8))
    flow = train_latent_flow(mus, latent_dim=vae_latent_dim, device=device)

    # 5. Save flow into checkpoint
    checkpoint['latent_flow_state_dict'] = flow.state_dict()
    checkpoint['latent_flow_config'] = {
        'latent_dim': vae_latent_dim,
        'hidden_dim': 64,
        'num_coupling_layers': 6,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  Flow saved to checkpoint: {checkpoint_path}")
    print("=" * 60)


def load_latent_flow(checkpoint, device):
    """Load a trained latent flow from a checkpoint, if present.

    Parameters
    ----------
    checkpoint : dict
        Loaded checkpoint dictionary.
    device : torch.device or str
        Device to place the flow on.

    Returns
    -------
    LatentRealNVP or None
        The trained flow model in eval mode, or None if not in checkpoint.
    """
    if 'latent_flow_state_dict' not in checkpoint:
        print("  Latent flow: not found in checkpoint (using raw N(0,I) sampling)")
        return None

    flow_cfg = checkpoint['latent_flow_config']
    flow = LatentRealNVP(
        latent_dim=flow_cfg['latent_dim'],
        hidden_dim=flow_cfg['hidden_dim'],
        num_coupling_layers=flow_cfg['num_coupling_layers'],
    )
    flow.load_state_dict(checkpoint['latent_flow_state_dict'])
    flow.eval()
    flow.to(device)
    print(f"  Latent flow: LOADED ({flow_cfg['num_coupling_layers']} coupling layers, "
          f"dim={flow_cfg['latent_dim']}, hidden={flow_cfg['hidden_dim']})")
    return flow
