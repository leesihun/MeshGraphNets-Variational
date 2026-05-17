# Visualization And Denormalization

The model trains on normalized deltas. Visualization paths convert predictions
back to physical units when the required normalization arrays are available.

There are two different output schemas:

- training-time test visualization output from `training_profiles/training_loop.py`
- autoregressive rollout output from `inference_profiles/rollout.py`

## Training-Time Test Visualization

`test_model()` in `training_profiles/training_loop.py` is run at
`test_interval` and at the last epoch. It calls
`general_modules/mesh_utils_fast.py::save_inference_results_fast`.

For each selected test graph it saves an HDF5 file under:

```text
outputs/test_set/<gpu_ids>/<epoch>/
```

Training-set reconstruction visualizations, when enabled, use:

```text
outputs/train_set/<gpu_ids>/<epoch>/
```

The saved HDF5 groups include:

```text
nodes/
  pos
  predicted_norm
  target_norm
  predicted_denorm
  target_denorm
  optional part_ids
edges/
  index
  attr
faces/
  index
  predicted_norm
  target_norm
  predicted_denorm
  target_denorm
  optional part_ids
```

The corresponding PNG compares normalized and denormalized predicted/target face
values.

## Denormalization Formula

The test visualization path reads `dataset.delta_mean` and `dataset.delta_std`
from the train-split preprocessing state:

```python
predicted_denorm = predicted_norm * delta_std + delta_mean
target_denorm = target_norm * delta_std + delta_mean
```

If those arrays are missing, the code falls back to storing normalized values in
the denormalized slots. That fallback should be treated as a diagnostic problem,
not as physical output.

## Feature Names And Units

`plot_mesh_comparison()` uses a small fixed name table:

```text
x_disp, y_disp, z_disp, stress
```

For three-channel displacement configs, the plotted features are displacement
channels only. For four-channel configs, the last channel is treated as stress.

Units are assigned by feature name:

- displacement channels: `mm`
- stress channel: `MPa`

`plot_feature_idx -1` selects the last available predicted channel.

## Training Loss Versus Visualization Values

Loss is computed on normalized deltas:

```text
Huber(predicted_norm, target_norm)
```

Denormalization is only for saved inspection artifacts and plots. It does not
change training gradients, validation loss, checkpoint selection, or rollout
state updates.

## Rollout Output

Autoregressive rollout uses a different HDF5 schema. It writes one dataset-like
file per sample or per VAE sample under `inference_output_dir` or
`outputs/rollout`.

The rollout `nodal_data` layout is:

```text
x, y, z, predicted output channels..., Part No.
```

Rollout saves the predicted physical state over time, not the training-test
`nodes/predicted_norm` schema. See `dataset/DATASET_FORMAT.md` for the complete
rollout output contract.

## Practical Checks

If plots look wrong:

- confirm the checkpoint and dataset use the same normalization statistics
- check `plot_feature_idx`
- confirm `output_var` matches the intended channels
- inspect `delta_mean` and `delta_std` printed by `test_model()`
- remember that very small normalized values can still correspond to meaningful
  physical deltas when `delta_std` is large
