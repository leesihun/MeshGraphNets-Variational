import torch.nn.init as init
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)


def build_mlp(in_size, hidden_size, out_size, layer_norm=True, activation='silu'):
    """
    Two-hidden-layer MLP following the MeshGraphNets architecture.
    LayerNorm is appended on the output when layer_norm=True (all non-decoder uses).
    """
    if activation == 'relu':
        activation_fn = nn.ReLU
    elif activation == 'gelu':
        activation_fn = nn.GELU
    elif activation == 'silu':
        activation_fn = nn.SiLU
    elif activation == 'tanh':
        activation_fn = nn.Tanh
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid
    else:
        raise ValueError(f'Invalid activation function: {activation}')

    layers = [
        nn.Linear(in_size, hidden_size),
        activation_fn(),
        nn.Linear(hidden_size, hidden_size),
        activation_fn(),
        nn.Linear(hidden_size, out_size),
    ]
    if layer_norm:
        layers.append(nn.LayerNorm(normalized_shape=out_size))
    return nn.Sequential(*layers)
