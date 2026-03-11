"""
Debug: Check what the model actually outputs.
If model outputs all zeros or constants, we found the bug.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets


def debug_model():
    print("=" * 60)
    print("DEBUG: MODEL OUTPUT ANALYSIS")
    print("=" * 60)

    # Load config
    config = load_config('./config.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load dataset (just first few samples)
    dataset = MeshGraphDataset(config['dataset_dir'], config)

    # Update config with node type info
    if dataset.use_node_types and dataset.num_node_types:
        config['num_node_types'] = dataset.num_node_types

    # Create model
    model = MeshGraphNets(config, device)
    model.eval()

    # Get one batch
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)

    print(f"\nInput shapes:")
    print(f"  x: {batch.x.shape}")
    print(f"  y (target): {batch.y.shape}")
    print(f"  edge_attr: {batch.edge_attr.shape}")
    print(f"  edge_index: {batch.edge_index.shape}")

    print(f"\nInput statistics:")
    print(f"  x mean: {batch.x.mean().item():.6f}")
    print(f"  x std: {batch.x.std().item():.6f}")
    print(f"  x min: {batch.x.min().item():.6f}")
    print(f"  x max: {batch.x.max().item():.6f}")

    print(f"\nTarget statistics:")
    print(f"  y mean: {batch.y.mean().item():.6f}")
    print(f"  y std: {batch.y.std().item():.6f}")
    print(f"  y min: {batch.y.min().item():.6f}")
    print(f"  y max: {batch.y.max().item():.6f}")

    # Forward pass
    with torch.no_grad():
        predicted, target = model(batch)

    print(f"\nModel output statistics:")
    print(f"  pred mean: {predicted.mean().item():.6f}")
    print(f"  pred std: {predicted.std().item():.6f}")
    print(f"  pred min: {predicted.min().item():.6f}")
    print(f"  pred max: {predicted.max().item():.6f}")

    # Check if output is constant
    if predicted.std().item() < 1e-6:
        print("\n*** CRITICAL: Model output is CONSTANT (std < 1e-6)! ***")
        print("    The model is not producing varied outputs.")
        print("    This explains why loss doesn't decrease.")

    # Per-feature analysis
    print(f"\nPer-feature analysis:")
    feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
    for i, name in enumerate(feature_names):
        pred_feat = predicted[:, i]
        target_feat = target[:, i]
        print(f"\n  {name}:")
        print(f"    target: mean={target_feat.mean().item():.6f}, std={target_feat.std().item():.6f}")
        print(f"    pred:   mean={pred_feat.mean().item():.6f}, std={pred_feat.std().item():.6f}")

    # Compute loss
    mse = ((predicted - target) ** 2).mean().item()
    print(f"\nMSE Loss: {mse:.6f}")

    # Check what loss would be if predicting zeros
    zero_loss = (target ** 2).mean().item()
    print(f"Loss if predicting zeros: {zero_loss:.6f}")

    if abs(mse - zero_loss) < 0.1:
        print("\n*** Model is essentially predicting zeros! ***")

    # Check gradients
    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)

    model.train()
    batch = next(iter(loader)).to(device)
    predicted, target = model(batch)
    loss = ((predicted - target) ** 2).mean()
    loss.backward()

    # Check decoder gradients
    decoder_grads = []
    for name, param in model.named_parameters():
        if 'decoder' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            decoder_grads.append((name, grad_norm))

    print("\nDecoder gradient norms:")
    for name, grad in decoder_grads[:5]:
        print(f"  {name}: {grad:.6e}")

    if decoder_grads and all(g[1] < 1e-8 for g in decoder_grads):
        print("\n*** CRITICAL: Decoder gradients are near-zero! ***")
        print("    Gradients are not flowing to the decoder.")

    # Check encoder gradients
    encoder_grads = []
    for name, param in model.named_parameters():
        if 'encoder' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            encoder_grads.append((name, grad_norm))

    print("\nEncoder gradient norms:")
    for name, grad in encoder_grads[:5]:
        print(f"  {name}: {grad:.6e}")


if __name__ == '__main__':
    debug_model()
