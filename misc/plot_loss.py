#!/usr/bin/env python3
"""
Script to plot training and validation loss from log files and save as image.

Usage:
    python plot_loss.py [config_file] [--output OUTPUT_PATH]

Example:
    python plot_loss.py config.txt
    python plot_loss.py config.txt --output my_loss_plot.png
"""

import os
import sys
import re
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for load_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_modules.load_config import load_config


def parse_gpu_ids(gpu_ids_config):
    """Convert gpu_ids config to string format for path construction."""
    if isinstance(gpu_ids_config, list):
        return ", ".join(str(g) for g in gpu_ids_config)
    return str(gpu_ids_config)


def construct_log_path(config):
    """Construct the full path to the log file from config."""
    gpu_ids_str = parse_gpu_ids(config.get('gpu_ids', '0'))
    log_file_dir = config.get('log_file_dir', 'train.log')

    # Build path: outputs/<gpu_ids>/<log_file_dir>
    log_path = os.path.join('outputs', gpu_ids_str, log_file_dir)
    return log_path


def parse_log_file(log_path):
    """Parse log file and extract epoch, train loss, and validation loss."""
    epochs = []
    train_losses = []
    valid_losses = []

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return None, None, None

    print(f"Reading log file: {log_path}")

    with open(log_path, 'r') as f:
        for line in f:
            # Pattern: "Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X"
            # Extract epoch number, train loss, and valid loss
            match = re.search(
                r'Epoch\s+(\d+)\s+Train Loss:\s+([\de\.\-]+)\s+Valid Loss:\s+([\de\.\-]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                valid_loss = float(match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

    if not epochs:
        print("Warning: No loss data found in log file. Check log format.")
        return None, None, None

    return epochs, train_losses, valid_losses


def plot_loss(epochs, train_losses, valid_losses, config, output_path=None):
    """Plot training and validation loss and save to file."""
    if epochs is None:
        print("Cannot plot: no data available")
        return

    plt.figure(figsize=(12, 6))

    plt.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2.5, markersize=5, color='#667eea')
    plt.plot(epochs, valid_losses, 's-', label='Validation Loss', linewidth=2.5, markersize=5, color='#764ba2')

    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.title('Training and Validation Loss Over Epochs', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')  # Log scale for better visualization of loss values

    # Auto-generate output path if not provided
    if output_path is None:
        gpu_ids_str = parse_gpu_ids(config.get('gpu_ids', '0'))
        output_dir = os.path.join('outputs', gpu_ids_str)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loss_plot.png')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Plot training loss from log file')
    parser.add_argument('config_file', nargs='?', default='config.txt',
                       help='Path to config file (default: config.txt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: outputs/<gpu_ids>/loss_plot.png)')

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config_file)
    print()

    # Construct log file path
    log_path = construct_log_path(config)

    # Parse log file
    epochs, train_losses, valid_losses = parse_log_file(log_path)

    if epochs is None:
        sys.exit(1)

    # Print summary
    print(f"✓ Found {len(epochs)} epochs")
    print(f"  Training Loss  - Min: {min(train_losses):.4e}, Max: {max(train_losses):.4e}")
    print(f"  Validation Loss - Min: {min(valid_losses):.4e}, Max: {max(valid_losses):.4e}")
    print()

    # Create plot
    plot_loss(epochs, train_losses, valid_losses, config, args.output)


if __name__ == '__main__':
    main()
