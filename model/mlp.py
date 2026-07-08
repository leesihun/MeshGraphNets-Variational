import torch.nn.init as init
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def build_mlp(in_size, hidden_size, out_size, layer_norm=True):
    """Two-hidden-layer SiLU MLP following the MeshGraphNets architecture.

    LayerNorm is appended on the output when layer_norm=True (all non-decoder uses).
    """
    layers = [
        nn.Linear(in_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, out_size),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))
    return nn.Sequential(*layers)
