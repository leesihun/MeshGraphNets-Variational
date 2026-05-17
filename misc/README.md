# Miscellaneous Utilities

This directory contains optional debugging and visualization helpers. They are
not part of the main training or rollout path.

## Files

| File | Purpose |
| --- | --- |
| `plot_loss.py` | Legacy static training-loss plotter. |
| `plot_loss_realtime.py` | Legacy FastAPI dashboard for live loss plotting. |
| `debug_model_output.py` | Loads a checkpoint and dataset sample to inspect model output magnitudes. |
| `analyze_mesh_topology.py` | Mesh topology inspection helper. |
| `requirements_plotting.txt` | Extra packages for the plotting dashboard. |
| `templates/dashboard.html` | HTML template used by `plot_loss_realtime.py`. |

## Main Runtime Logs

The current training setup creates the log file in:

```text
outputs/<log_file_dir>
```

This comes from `training_profiles/setup.py::init_log_file`. For example, a
config line:

```text
log_file_dir b8_all/train1.log
```

writes:

```text
outputs/b8_all/train1.log
```

Current VAE logs look like:

```text
Elapsed: 123.45s Epoch 10 LR: 1.0000e-04 | TrainOpt recon=... mmd=... total=... | ValidQ recon=... mmd=... total=... | ValidPrior@... recon=... gap=...
```

Current non-VAE logs look like:

```text
Elapsed: 123.45s Epoch 10 TrainOpt 1.2345e-02 Valid 1.5678e-02 LR: 1.0000e-04
```

## Loss Plotters Are Legacy

`plot_loss.py` and `plot_loss_realtime.py` still construct log paths as:

```text
outputs/<gpu_ids>/<log_file_dir>
```

and parse the older log pattern:

```text
Epoch N Train Loss: ... Valid Loss: ...
```

That does not match the current training log path or current VAE log format.
Use these scripts only for older logs with that format, or update the scripts
before using them for current training runs.

## Legacy Plotter Usage

Install optional plotting dependencies:

```bash
pip install -r misc/requirements_plotting.txt
```

Static plotter:

```bash
python misc/plot_loss.py config.txt
python misc/plot_loss.py config.txt --output outputs/loss_plot.png
```

Realtime dashboard:

```bash
python misc/plot_loss_realtime.py config.txt
python misc/plot_loss_realtime.py config.txt --port 8080
```

The dashboard serves:

```text
http://localhost:5000
http://localhost:5000/docs
```

or the port supplied with `--port`.

## Debug Output Helper

`debug_model_output.py` is a targeted inspection script. Use it when predictions
look collapsed or badly scaled and you need to compare normalized model output
against target tensors. It prints aggregate diagnostics rather than running a
full training loop.
