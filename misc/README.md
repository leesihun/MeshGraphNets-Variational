# Training Loss Visualization Tools

This directory contains two tools for visualizing training loss from MeshGraphNets logs:

1. **`plot_loss.py`** - Static plotting (saves loss curve as PNG image)
2. **`plot_loss_realtime.py`** - Real-time dashboard (live visualization in web browser using FastAPI)

## Installation

Install required dependencies:

```bash
pip install -r requirements_plotting.txt
```

## Usage

### Option 1: Static Loss Plot

Generate a PNG image of the training loss curve:

```bash
# Using default config.txt
python misc/plot_loss.py

# Using custom config file
python misc/plot_loss.py config.txt

# Specify output path
python misc/plot_loss.py config.txt --output my_plot.png
```

**Output:**
- By default: `outputs/<gpu_ids>/loss_plot.png`
- Or specify with `--output` flag

**Features:**
- Reads `log_file_dir` from config file
- Parses training and validation loss from log
- Log scale for better visualization
- Shows loss statistics
- Fast execution (no server required)

---

### Option 2: Real-Time Dashboard (FastAPI)

Start a live training dashboard in your web browser:

```bash
# Using default config.txt on port 5000
python misc/plot_loss_realtime.py

# Using custom config file
python misc/plot_loss_realtime.py config.txt

# Specify port
python misc/plot_loss_realtime.py config.txt --port 8080
```

**Access:**
- Open browser to: `http://localhost:5000`
- Auto-refreshes every 2 seconds as training progresses
- API documentation: `http://localhost:5000/docs` (FastAPI auto-generated)

**Features:**
- Real-time loss curve updates
- Live statistics (best/latest losses)
- Training metadata (GPU config, log directory)
- Status indicator showing last update
- Responsive design (works on desktop and mobile)
- Interactive tooltips (hover over points)
- Log scale for loss visualization
- FastAPI-powered with auto-generated OpenAPI docs

**Stop the server:**
- Press `Ctrl+C` in terminal

---

## Configuration

Both scripts read from your training config file and look for these keys:

```
gpu_ids 0, 1, 2, 3      # or single GPU: gpu_ids 0
log_file_dir train.log
```

The scripts construct the log path as:
```
outputs/<gpu_ids>/<log_file_dir>
```

Example paths:
- Single GPU: `outputs/0/train.log`
- Multi-GPU: `outputs/0, 1, 2, 3/train.log`

---

## Log File Format

The scripts expect log files with this format:

```
Elapsed time: 123s Epoch 1 Train Loss: 1.2345e-02 Valid Loss: 1.5678e-02 LR: 1.0000e-04
Elapsed time: 456s Epoch 2 Train Loss: 9.8765e-03 Valid Loss: 1.3456e-02 LR: 1.0000e-04
...
```

Key regex pattern matched:
```
Epoch\s+(\d+)\s+Train Loss:\s+([\de\.\-]+)\s+Valid Loss:\s+([\de\.\-]+)
```

---

## Troubleshooting

### "Log file not found"
- Check that your training has started (log file should exist)
- Verify `gpu_ids` and `log_file_dir` in your config file
- Check the correct output directory exists: `outputs/<gpu_ids>/`

### Plot shows no data
- Ensure at least one epoch has completed
- Verify log file contains lines matching the expected format
- Check that loss values are valid numbers

### Real-time dashboard won't load
- Make sure dependencies are installed: `pip install -r misc/requirements_plotting.txt`
- Try a different port: `python misc/plot_loss_realtime.py --port 8080`
- Check firewall isn't blocking localhost

### Port already in use
```bash
# Use a different port
python misc/plot_loss_realtime.py --port 8080
```

### Import error from general_modules
- Run scripts from the project root directory (not from misc/)
- Make sure you're in the correct working directory

---

## Examples

### Generate plot after training completes
```bash
python misc/plot_loss.py config.txt
# Output: outputs/0/loss_plot.png
```

### Monitor live while training runs in another terminal
```bash
# Terminal 1: Start training
python MeshGraphNets_main.py

# Terminal 2: Start dashboard
python misc/plot_loss_realtime.py config.txt
# Visit: http://localhost:5000
```

### Use custom log file location
```bash
# In config.txt:
log_file_dir custom_logs/training_v2.log

# Then run:
python misc/plot_loss.py config.txt
```

### Access FastAPI documentation
```bash
# Start the dashboard
python misc/plot_loss_realtime.py config.txt

# In browser, visit:
http://localhost:5000/docs       # Interactive API docs (Swagger UI)
http://localhost:5000/redoc      # Alternative docs (ReDoc)
```

---

## File Structure

```
MeshGraphNets/
├── misc/
│   ├── plot_loss.py              # Static plotting script
│   ├── plot_loss_realtime.py     # Real-time dashboard (FastAPI)
│   ├── requirements_plotting.txt # Python dependencies
│   ├── README.md                 # This file
│   └── templates/
│       └── dashboard.html        # Web UI for real-time dashboard
├── outputs/
│   └── <gpu_ids>/
│       └── train.log             # Training log (read by scripts)
└── MeshGraphNets_main.py         # Main training script
```

---

## Technology Stack

### Static Plotting
- **Framework:** Matplotlib
- **Usage:** Batch plotting after training

### Real-Time Dashboard
- **Backend:** FastAPI (modern, high-performance async web framework)
- **Server:** Uvicorn (ASGI server)
- **Frontend:** HTML5 + JavaScript + Chart.js
- **Benefits:**
  - Auto-generated API documentation (/docs)
  - Better performance than Flask
  - Modern async support
  - Type hints and validation

---

## Notes

- Both scripts are read-only and don't modify your training logs
- Real-time dashboard auto-detects when new epochs are added to the log
- Loss values are plotted on log scale for better visualization
- Statistics update live on the dashboard as training progresses
- The static plot uses higher DPI (150) for better print quality
- FastAPI provides automatic OpenAPI documentation at `/docs` endpoint

---

## FastAPI Features

When running the real-time dashboard, you get additional benefits:

- **Auto-generated Swagger UI:** `/docs` endpoint for interactive API exploration
- **ReDoc documentation:** `/redoc` endpoint for alternative API documentation
- **Type validation:** Request/response validation and documentation
- **Async support:** Non-blocking I/O for better performance
- **JSON Schema:** Automatic schema generation for API responses
