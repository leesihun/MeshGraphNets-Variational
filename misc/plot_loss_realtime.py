#!/usr/bin/env python3
"""
Real-time training loss visualization dashboard using FastAPI.

Usage:
    python plot_loss_realtime.py [config_file] [--port PORT]

Example:
    python plot_loss_realtime.py config.txt
    python plot_loss_realtime.py config.txt --port 8080

Then open browser to: http://localhost:5000
API docs available at: http://localhost:5000/docs
"""

import os
import sys
import re
import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add parent directory to path for load_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general_modules.load_config import load_config


class LogReader:
    """Thread-safe log file reader with caching."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.epochs = []
        self.train_losses = []
        self.valid_losses = []
        self.last_position = 0
        self.last_modified = 0

    def parse_log_file(self):
        """Parse log file and extract epoch, train loss, and validation loss."""
        if not os.path.exists(self.log_path):
            return False

        try:
            current_modified = os.path.getmtime(self.log_path)

            # Only read if file was modified since last read
            if current_modified <= self.last_modified and self.epochs:
                return False

            self.last_modified = current_modified
            self.epochs = []
            self.train_losses = []
            self.valid_losses = []

            with open(self.log_path, 'r') as f:
                for line in f:
                    # Pattern: "Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X"
                    match = re.search(
                        r'Epoch\s+(\d+)\s+Train Loss:\s+([\de\.\-]+)\s+Valid Loss:\s+([\de\.\-]+)',
                        line
                    )
                    if match:
                        epoch = int(match.group(1))
                        train_loss = float(match.group(2))
                        valid_loss = float(match.group(3))

                        self.epochs.append(epoch)
                        self.train_losses.append(train_loss)
                        self.valid_losses.append(valid_loss)

            return True
        except Exception as e:
            print(f"Error reading log file: {e}")
            return False

    def get_data(self):
        """Get current loss data."""
        self.parse_log_file()
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'num_epochs': len(self.epochs)
        }


# Global instances
app = FastAPI(title="MeshGraphNets Training Dashboard", version="1.0")
log_reader = None
config = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main dashboard page."""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'dashboard.html')
    try:
        with open(template_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: dashboard.html not found</h1>"


@app.get("/api/loss-data")
async def get_loss_data():
    """API endpoint to get current loss data."""
    data = log_reader.get_data()
    return data


@app.get("/api/config")
async def get_config_info():
    """API endpoint to get config info."""
    return {
        'log_file_dir': config.get('log_file_dir', 'unknown'),
        'gpu_ids': str(config.get('gpu_ids', 'unknown')),
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Real-time loss visualization')
    parser.add_argument('config_file', nargs='?', default='config.txt',
                       help='Path to config file')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')

    args = parser.parse_args()

    global log_reader, config

    # Load configuration
    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)

    config = load_config(args.config_file)
    print()

    # Construct log file path
    gpu_ids = config.get('gpu_ids', '0')
    if isinstance(gpu_ids, list):
        gpu_ids_str = ", ".join(str(g) for g in gpu_ids)
    else:
        gpu_ids_str = str(gpu_ids)

    log_file_dir = config.get('log_file_dir', 'train.log')
    log_path = os.path.join('outputs', gpu_ids_str, log_file_dir)

    print(f"Log file path: {log_path}")

    # Create log reader
    log_reader = LogReader(log_path)

    # Print server info
    print(f"\n{'='*60}")
    print(f"Real-time Loss Visualization Dashboard")
    print(f"{'='*60}")
    print(f"Server running at: http://localhost:{args.port}")
    print(f"API Documentation: http://localhost:{args.port}/docs")
    print(f"Watching log file: {log_path}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    # Run Uvicorn server
    try:
        uvicorn.run(app, host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()
