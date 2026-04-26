import numpy as np
import torch
from torch_geometric.loader import DataLoader


def collect_posterior_means(model, train_dataset, config, device):
    """Run VAE encoder over training set; collect posterior means.

    Args:
        model: Trained model with .vae_encoder attribute (EMA preferred).
        train_dataset: Training MeshGraphDataset.
        config: Config dict.
        device: torch.device.

    Returns:
        mu_data: np.ndarray [N_train, vae_latent_dim], float32.
    """
    output_var = int(config.get('output_var', 3))
    batch_size = int(config.get('batch_size', 4))

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    # MeshGraphNets wraps EncoderProcessorDecoder as .model; vae_encoder lives there
    inner = getattr(model, 'model', model)

    model.eval()
    mus = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y = data.y[:, :output_var]
            _, mu, _ = inner.vae_encoder(y, data.edge_index, data.edge_attr, data.batch)
            mus.append(mu.float().cpu().numpy())

    return np.concatenate(mus, axis=0)  # [N_train, D]


def fit_gmm(mu_data, n_components=10, covariance_type='full', random_state=0):
    """Fit GMM on posterior means via sklearn; return storable param dict.

    Caps n_components to n_samples so sklearn never errors on small datasets.

    Args:
        mu_data: [N, D] float32 array.
        n_components: Number of mixture components.
        covariance_type: 'full' | 'diag' | 'tied' | 'spherical'.
        random_state: RNG seed.

    Returns:
        dict with keys: weights [K], means [K, D], covariances [*],
        covariance_type str, n_components int.
    """
    from sklearn.mixture import GaussianMixture

    n_components = min(n_components, len(mu_data))
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=300,
        n_init=5,
    )
    gmm.fit(mu_data)

    return {
        'weights': gmm.weights_.astype(np.float32),
        'means': gmm.means_.astype(np.float32),
        'covariances': gmm.covariances_.astype(np.float32),
        'covariance_type': covariance_type,
        'n_components': n_components,
    }


def run_posthoc_gmm_fitting(model, train_dataset, config, device, checkpoint_path):
    """Collect posterior means, fit GMM, append to checkpoint.

    Args:
        model: Trained model (EMA preferred), must have .vae_encoder.
        train_dataset: Training dataset.
        config: Config dict.
        device: torch.device.
        checkpoint_path: Path to existing .pth checkpoint; updated in-place.
    """
    n_components = int(config.get('gmm_components', 10))
    covariance_type = str(config.get('gmm_covariance_type', 'full'))
    n_train = len(train_dataset)

    print(f"\n[GMM] Collecting posterior means from {n_train} training samples...")
    mu_data = collect_posterior_means(model, train_dataset, config, device)
    per_dim_std = mu_data.std(0)
    print(f"[GMM] mu shape: {mu_data.shape}  per-dim std: {np.round(per_dim_std, 3)}")

    n_fit = min(n_components, n_train)
    print(f"[GMM] Fitting GMM (n_components={n_fit}, covariance_type={covariance_type})...")
    gmm_params = fit_gmm(mu_data, n_components=n_fit, covariance_type=covariance_type)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint['gmm_params'] = gmm_params
    torch.save(checkpoint, checkpoint_path)
    print(f"[GMM] Saved GMM ({n_fit} components, {covariance_type} cov) to {checkpoint_path}")


def load_latent_gmm(checkpoint):
    """Return GMM param dict from checkpoint, or None if not present."""
    return checkpoint.get('gmm_params', None)


def sample_from_gmm(gmm_params, n_samples, device):
    """Sample latent codes from a stored GMM without sklearn at inference.

    Args:
        gmm_params: dict from load_latent_gmm.
        n_samples: Number of z samples to draw.
        device: torch.device.

    Returns:
        z: torch.Tensor [n_samples, D], float32, on device.
    """
    weights = gmm_params['weights']
    means = gmm_params['means']
    covariances = gmm_params['covariances']
    cov_type = gmm_params['covariance_type']
    K, D = means.shape

    k_indices = np.random.choice(K, size=n_samples, p=weights.astype(np.float64))

    samples = np.empty((n_samples, D), dtype=np.float32)
    if cov_type == 'full':
        for i, k in enumerate(k_indices):
            samples[i] = np.random.multivariate_normal(means[k], covariances[k])
    elif cov_type == 'diag':
        for i, k in enumerate(k_indices):
            samples[i] = np.random.normal(means[k], np.sqrt(covariances[k]))
    elif cov_type == 'tied':
        # covariances is [D, D] shared across all components
        for i, k in enumerate(k_indices):
            samples[i] = np.random.multivariate_normal(means[k], covariances)
    elif cov_type == 'spherical':
        for i, k in enumerate(k_indices):
            samples[i] = np.random.normal(means[k], np.sqrt(covariances[k]))
    else:
        raise ValueError(f"Unknown covariance_type: {cov_type!r}")

    return torch.from_numpy(samples).to(device)
