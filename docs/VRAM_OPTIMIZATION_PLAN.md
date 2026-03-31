# VRAM Optimization Plan: Gradient Checkpointing + AMP

## Overview

Implement two VRAM optimization techniques for MeshGraphNets:
1. **Step 1: Gradient Checkpointing** - Trade compute for memory by recomputing activations during backward pass
2. **Step 2: AMP (Automatic Mixed Precision)** - Use float16 for forward/backward, float32 for optimizer

**Expected VRAM Savings:**
- Gradient checkpointing: ~60-70% reduction in activation memory
- AMP: ~50% reduction in model/gradient memory
- Combined: ~70-75% total VRAM reduction

---

## Design Decisions

### 1. How to handle PyG Data objects with torch.utils.checkpoint?

**Problem**: `torch.utils.checkpoint.checkpoint()` requires tensor inputs, but `GnBlock.forward()` takes and returns `Data` objects.

**Solution**: Create a wrapper function that unpacks `Data` to tensors, calls the checkpointed function, and repacks to `Data`.

### 2. Checkpoint each block individually or group them?

**Decision**: Checkpoint each block individually.
- 15 GN blocks with individual checkpointing provides fine-grained memory control
- Simpler to implement and debug than grouping

### 3. Where should autocast boundaries be?

**Decision**: Wrap only the forward pass and loss computation inside `autocast`.
- Backward pass happens outside autocast (handled automatically by GradScaler)
- LayerNorm is safe with AMP (uses FP32 accumulation internally)

### 4. How to handle gradient clipping with scaled gradients?

**Solution**: Use `scaler.unscale_(optimizer)` before `clip_grad_norm_`, then `scaler.step()`.

---

## Files to Modify

| File | Purpose |
|------|---------|
| `config.txt` | Add toggle options |
| `model/MeshGraphNets.py` | Add checkpointing to processor loop |
| `training_profiles/training_loop.py` | Add autocast + scaler handling |
| `training_profiles/single_training.py` | Initialize GradScaler, enable checkpointing |
| `training_profiles/distributed_training.py` | Same as single_training.py for DDP |

---

## Step 1: Gradient Checkpointing

### 1.1 Add Config Option

**File:** `config.txt`

```
use_checkpointing   True
```

### 1.2 Modify model/MeshGraphNets.py

**Add import at top:**
```python
from torch.utils.checkpoint import checkpoint
```

**Add to EncoderProcessorDecoder.__init__ (after line 66):**
```python
self.use_checkpointing = config.get('use_checkpointing', False)
```

**Add helper method to EncoderProcessorDecoder:**
```python
def _run_gn_block(self, block, x, edge_attr, edge_index):
    """Helper for checkpointing - converts tensors to Data and back."""
    graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    out_graph = block(graph)
    return out_graph.x, out_graph.edge_attr

def set_checkpointing(self, enabled: bool):
    """Enable or disable gradient checkpointing."""
    self.use_checkpointing = enabled
```

**Replace forward method (lines 77-84):**
```python
def forward(self, graph):
    graph = self.encoder(graph)

    if self.use_checkpointing and self.training:
        # Checkpointing requires tensors, not Data objects
        x = graph.x
        edge_attr = graph.edge_attr
        edge_index = graph.edge_index

        for block in self.processer_list:
            x, edge_attr = checkpoint(
                self._run_gn_block,
                block,
                x,
                edge_attr,
                edge_index,
                use_reentrant=False
            )

        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    else:
        for model in self.processer_list:
            graph = model(graph)

    output = self.decoder(graph)
    return output
```

**Add to MeshGraphNets class (after line 22):**
```python
def set_checkpointing(self, enabled: bool):
    """Enable or disable gradient checkpointing."""
    self.model.set_checkpointing(enabled)
```

---

## Step 2: AMP (Automatic Mixed Precision)

### 2.1 Add Config Option

**File:** `config.txt`

```
use_amp             True
```

### 2.2 Modify training_profiles/training_loop.py

**Add import at top:**
```python
from torch.amp import autocast, GradScaler
```

**Modify train_epoch signature (line 5):**
```python
def train_epoch(model, dataloader, optimizer, device, config, epoch, scaler=None):
```

**Replace forward/backward pass (lines 29-48):**
```python
        use_amp = config.get('use_amp', False) and device.type == 'cuda'

        # Forward pass with optional AMP
        with autocast(device_type='cuda', enabled=use_amp):
            predicted_acc, target_acc = model(graph)
            errors = ((predicted_acc - target_acc) ** 2)
            loss = torch.mean(errors)

        optimizer.zero_grad()

        if scaler is not None:
            # AMP backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
```

**Modify validate_epoch to add autocast:**
```python
def validate_epoch(model, dataloader, device, config):
    model.eval()
    use_amp = config.get('use_amp', False) and device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, graph in enumerate(dataloader):
            graph = graph.to(device)

            with autocast(device_type='cuda', enabled=use_amp):
                predicted, target = model(graph)
                errors = ((predicted - target) ** 2)
                loss = torch.mean(errors)
            # ... rest unchanged
```

### 2.3 Modify training_profiles/single_training.py

**Add import:**
```python
from torch.amp import GradScaler
```

**Initialize GradScaler and checkpointing (after model creation, ~line 72):**
```python
    # Enable gradient checkpointing if configured
    if config.get('use_checkpointing', False):
        model.set_checkpointing(True)
        print("Gradient checkpointing enabled")

    # Initialize GradScaler for AMP
    use_amp = config.get('use_amp', False) and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print("AMP enabled")
```

**Update training loop call (line 132):**
```python
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, scaler=scaler)
```

**Update checkpoint saving (lines 146-153):**
```python
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
```

### 2.4 Modify training_profiles/distributed_training.py

**Add import:**
```python
from torch.amp import GradScaler
```

**Initialize after DDP wrapping (~line 100):**
```python
    # Enable gradient checkpointing (on underlying module, not DDP wrapper)
    if config.get('use_checkpointing', False):
        model.module.set_checkpointing(True)
        if rank == 0:
            print("Gradient checkpointing enabled")

    # Initialize GradScaler for AMP
    use_amp = config.get('use_amp', False)
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if rank == 0 and use_amp:
        print("AMP enabled")
```

---

## Final config.txt Additions

Add at the end of `config.txt`:
```
% Memory Optimization
use_checkpointing   True
use_amp             True
```

---

## Verification Plan

### Test 1: Gradient Checkpointing Only
```
use_checkpointing True
use_amp           False
```
- Run `python MeshGraphNets_main.py`
- Verify training completes without errors
- Check memory logging shows reduced VRAM

### Test 2: AMP Only
```
use_checkpointing False
use_amp           True
```
- Run `python MeshGraphNets_main.py`
- Verify no NaN/Inf in loss
- Check memory reduction in logs

### Test 3: Both Combined
```
use_checkpointing True
use_amp           True
```
- Run `python MeshGraphNets_main.py`
- Verify maximum VRAM savings
- Model should still converge

### Expected Memory Usage
| Configuration | VRAM per sample |
|---------------|-----------------|
| Baseline | ~5-8 GB |
| Checkpointing only | ~2-3 GB |
| AMP only | ~3-4 GB |
| Both | ~1.5-2 GB |

---

## Implementation Order

1. `config.txt` - Add new options
2. `model/MeshGraphNets.py` - Add checkpointing logic
3. `training_profiles/training_loop.py` - Add autocast and scaler handling
4. `training_profiles/single_training.py` - Initialize scaler, enable checkpointing
5. `training_profiles/distributed_training.py` - Same changes for DDP
6. Test each feature independently, then combined

---

## Potential Issues

| Issue | Mitigation |
|-------|------------|
| NaN/Inf with AMP | GradScaler automatically skips bad updates |
| DDP + checkpointing | Use `model.module.set_checkpointing()` not `model.set_checkpointing()` |
| Slower training | Expected ~20-30% slowdown from recomputation (checkpointing) |
| edge_index dtype | autocast handles this automatically (keeps int64) |
