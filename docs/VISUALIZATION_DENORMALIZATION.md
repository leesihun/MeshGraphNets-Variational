# Visualization Denormalization

## Overview

The test set visualizations now display **denormalized delta values** (actual physical units) instead of normalized z-scores.

## What Changed

### Before
- Visualizations showed normalized values: -0.0002 to 0.0002 (meaningless z-scores)
- No unit labels
- Hard to interpret physical meaning

### After
- Visualizations show actual delta values:
  - **Stress deltas**: 5-30 MPa (typical stress changes between timesteps)
  - **Displacement deltas**: 0.001-0.1 mm (typical displacement changes)
- Proper unit labels: "(MPa)" for stress, "(mm)" for displacements
- Clear feature names: "Δ Stress", "Δ Disp X", "Δ Disp Y", "Δ Disp Z"

## Technical Details

### Normalization Statistics (from dataset)

```python
delta_mean = [4.96e-07, -7.68e-10, 1.22e-07, 8.79]  # Near zero for displacements, ~8.79 MPa for stress
delta_std  = [1.63e-04, 1.40e-04, 4.21e-04, 14919.6]  # Very large for stress (14,919 MPa)
```

### Why Values Appeared So Small

The normalized values were tiny (-0.0002 to 0.0002) because:
1. Model predicts **deltas** (state(t) - state(t-1))
2. These are z-score normalized using `delta_std`
3. For stress, `delta_std` = 14,919 MPa (very large)
4. Example: actual delta of 10 MPa → normalized = 10/14919 = **0.00067**

### Denormalization Formula

```python
actual_delta = normalized_delta * delta_std + delta_mean
```

For stress (feature index 3 or -1):
```
actual_stress_delta = norm_value * 14919.6 + 8.79
```

## Code Changes

### 1. `training_loop.py`
- Added `dataset` parameter to `infer_model()`
- Extract `delta_mean` and `delta_std` from dataset
- Denormalize predictions before saving: `predicted_denorm = predicted_np * delta_std + delta_mean`

### 2. `mesh_utils_fast.py`
- Updated `plot_mesh_comparison()` to show feature names and units
- Added colorbar labels: "Δ Stress (MPa)", "Δ Disp X (mm)", etc.
- Updated MAE in title to include units

### 3. `single_training.py` and `distributed_training.py`
- Pass `dataset` to `infer_model()` call

## Visualization Output

### Colorbar Label Examples
- `Δ Stress (MPa)` - for stress deltas
- `Δ Disp X (mm)` - for X displacement deltas
- `Δ Disp Y (mm)` - for Y displacement deltas
- `Δ Disp Z (mm)` - for Z displacement deltas

### Title Example
```
Sample 42, Timestep 15 | MAE: 2.34 MPa
```

## Expected Value Ranges

Based on dataset statistics:

| Feature | Typical Delta Range |
|---------|-------------------|
| Disp X | 0.0001 - 0.05 mm |
| Disp Y | 0.0001 - 0.05 mm |
| Disp Z | 0.0001 - 0.1 mm |
| Stress | 5 - 30 MPa |

## Notes

- The model internally works with **normalized deltas** (for training stability)
- Denormalization happens **only for visualization** (after inference)
- HDF5 output files contain both normalized and denormalized values
- Loss computation uses normalized values (unchanged)

## Usage

No changes required to your workflow. Just run training as usual:

```bash
python MeshGraphNets_main.py
```

Test visualizations (every 10 epochs) will automatically show denormalized values with proper units.

## Verification

Run the verification script to check normalization statistics:

```bash
python misc/check_normalization.py
```

This shows:
- Dataset normalization parameters
- Denormalization examples
- Expected value ranges
