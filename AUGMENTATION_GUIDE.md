# Data Augmentation Guide for Jet Classification

## Overview

This guide documents the **Phase 1** data augmentation strategies implemented for your ParticleNet jet flavor tagging model. These augmentations are physics-preserving transformations that help improve model generalization and robustness.

## What Was Implemented

### ✅ Phase 1 Augmentations (Tier 1: Easy + High Impact)

1. **Azimuthal Rotations** (`aug_rotation`)
   - Rotates all particles around the beam axis by random angle θ ∈ [0, 2π)
   - Updates: `pfcand_phirel`, `pfcand_px`, `pfcand_py`, `pfcand_drrel`
   - **Why it works**: Physics is rotationally symmetric around beam axis
   - **Impact**: Infinite dataset variations, strongest physics motivation

2. **Reflections** (`aug_reflection`)
   - **Phi reflection**: `pfcand_phirel → -pfcand_phirel` (reflects across φ=0)
   - **Eta reflection**: `pfcand_thetarel → -pfcand_thetarel` (reflects across η=0)
   - Each applied independently with 50% probability (configurable)
   - Updates: `pfcand_phirel`, `pfcand_thetarel`, `pfcand_px`, `pfcand_py`, `pfcand_pz`, `pfcand_dz`, `pfcand_drrel`
   - **Why it works**: Detector is symmetric in φ, and often in η
   - **Impact**: 4x effective dataset size (no flip, φ flip, η flip, both flips)

3. **Soft Particle Dropout** (`aug_dropout`)
   - Preferentially removes low-pT particles (simulates detector inefficiency)
   - Dropout probability inversely proportional to particle pT
   - Updates: `pfcand_mask` and zeros out features for dropped particles
   - **Why it works**: Realistic simulation of tracking inefficiency, provides regularization
   - **Impact**: Improves robustness to missing particles

## Configuration

### Training Config Settings

Add these to your training config YAML (e.g., `training_config_part_augmented.yaml`):

```yaml
# Data Augmentation Settings
augment: true                    # Enable/disable all augmentation
aug_rotation: true               # Enable azimuthal rotation
aug_reflection: true             # Enable phi/eta reflections
aug_dropout: 0.1                 # Fraction of particles to drop (0.0-1.0)
aug_reflection_prob: 0.5         # Probability for each reflection (optional, default 0.5)
```

### Recommended Settings

**Conservative (start here):**
```yaml
augment: true
aug_rotation: true
aug_reflection: true
aug_dropout: 0.10
```

**Moderate (if conservative works well):**
```yaml
augment: true
aug_rotation: true
aug_reflection: true
aug_dropout: 0.15
```

**Aggressive (for maximum regularization):**
```yaml
augment: true
aug_rotation: true
aug_reflection: true
aug_dropout: 0.20
aug_reflection_prob: 0.7
```

## How It Works in Your Codebase

### Integration Points

1. **Config Loading**: Settings loaded from training YAML
2. **Data Pipeline**: Augmentations applied in `weaver/utils/dataset.py` line 110
3. **Timing**: Applied **before** standardization (raw data → augment → standardize → model)
4. **Training Only**: Only applied during training, not during evaluation/testing

### Code Flow

```
Training Loop
    ↓
Load Batch (raw data)
    ↓
_preprocess() in dataset.py
    ↓
  • Selection
  • Build new variables
  • Reweight indices
  • AUGMENT ← Your augmentations happen here!
  • Standardize & finalize inputs
    ↓
Send to Model
```

### Augmentation Order

1. **Rotation** (if enabled)
2. **Reflection** (if enabled)
3. **Dropout** (if enabled)

This order ensures:
- Geometric transformations (rotation, reflection) happen first
- Dropout last, so we drop particles from transformed geometry

## Implementation Details

### Azimuthal Rotation

```python
def _rotate(self, table):
    # Random angle per event
    theta = np.random.uniform(0, 2*pi, n_events)
    
    # Update phi
    pfcand_phirel_new = pfcand_phirel + theta
    pfcand_phirel_new = wrap_to_[-pi, pi]
    
    # Update cartesian
    px_new = px * cos(theta) - py * sin(theta)
    py_new = px * sin(theta) + py * cos(theta)
    
    # Update drrel
    drrel_new = sqrt(thetarel^2 + phirel_new^2)
```

### Reflections

```python
def _reflect(self, table):
    # Independent phi and eta reflections
    # Each with probability reflection_prob (default 0.5)
    
    # Phi reflection (50% chance per event)
    if random < 0.5:
        pfcand_phirel *= -1
        pfcand_py *= -1
    
    # Eta reflection (50% chance per event, independent)
    if random < 0.5:
        pfcand_thetarel *= -1
        pfcand_pz *= -1
        pfcand_dz *= -1
    
    # Update drrel
    drrel_new = sqrt(thetarel^2 + phirel^2)
```

### Soft Particle Dropout

```python
def _dropout(self, table):
    # Compute per-particle dropout probability
    # Lower pT → higher probability
    
    # Normalize pT per event
    pt_normalized = (pt_log - pt_min) / (pt_max - pt_min)
    
    # Invert (low pT → high weight)
    pt_weight = 1.0 - pt_normalized
    
    # Dropout prob: 0 for highest pT, 2*frac for lowest pT
    dropout_prob = 2 * frac_to_drop * pt_weight
    
    # Drop particles
    drop_mask = random < dropout_prob
    pfcand_mask[drop_mask] = 0
```

## Validation & Testing

### Unit Tests

Run tests to verify augmentations work correctly:

```bash
cd /users/tgillin/files/weaver-core
python -m pytest tests/test_augmentation.py -v
```

Or:

```bash
python tests/test_augmentation.py
```

### Visual Validation

To validate augmentations are working during training:

1. **Check logs**: Training script should show augmentation settings
2. **Monitor metrics**: Augmentation should improve validation performance
3. **Inspect data**: Add debug prints in `augmentation.py` to verify transformations

### Expected Behavior

**Good signs:**
- Training loss slightly higher than without augmentation (expected)
- Validation loss lower than without augmentation
- Better generalization to test set
- Model less sensitive to systematic variations

**Bad signs:**
- Training becomes unstable → reduce `aug_dropout`
- No improvement → check augmentation is actually applied
- Worse performance → verify data consistency (drrel, mask updates)

## Physics Validation

### Things That Should NOT Change

✅ **Jet kinematics preserved:**
- Sum of particle momenta = jet momentum
- Energy-momentum relation: E² = p² + m²

✅ **Particle properties preserved:**
- Particle IDs (isChargedHad, isNeutralHad, etc.)
- Charge
- Mass (implied by E and p)

✅ **Relative ordering preserved:**
- High-pT particles stay high-pT
- Spatial relationships maintained

### Things That WILL Change

✓ Angular coordinates: φ, θ (rotation, reflection)
✓ Cartesian coordinates: px, py, pz (rotation, reflection)
✓ Number of particles: (dropout)
✓ Mask values: (dropout)

## Advanced Configuration

### Disable Specific Augmentations

```yaml
# Only rotation
augment: true
aug_rotation: true
aug_reflection: false
aug_dropout: 0.0

# Only dropout
augment: true
aug_rotation: false
aug_reflection: false
aug_dropout: 0.15

# Only reflection
augment: true
aug_rotation: false
aug_reflection: true
aug_dropout: 0.0
```

### Per-Augmentation Probability Control

Currently only reflection has probability control. To add for others, modify `augmentation.py`:

```python
# In __init__:
self.params = {
    'rotation': self.options.get('aug_rotation', True),
    'rotation_prob': self.options.get('aug_rotation_prob', 1.0),  # Add this
    ...
}

# In _rotate:
if np.random.rand() < self.params['rotation_prob']:
    # do rotation
```

## Troubleshooting

### Issue: No augmentation happening

**Check:**
1. Is `augment: true` in your training config?
2. Are you using the augmented config file? (`training_config_part_augmented.yaml`)
3. Check logs for "Augmenter" initialization

### Issue: Training crashes with augmentation

**Possible causes:**
1. Awkward array version mismatch → check `awkward` version
2. Missing fields → add debug prints to see what fields exist
3. Shape mismatch → verify mask and features have same length

**Debug:**
```python
# Add to augmentation.py at start of augment():
print(f"Table fields: {table.fields}")
print(f"Table length: {len(table)}")
```

### Issue: Augmentation too aggressive

**Solution:**
Reduce intensity:
```yaml
aug_dropout: 0.05  # Reduce from 0.1
aug_reflection_prob: 0.3  # Reduce from 0.5
```

### Issue: Augmentation not helping

**Possible reasons:**
1. Model already well-regularized (dropout layers, etc.)
2. Dataset already large enough
3. Task doesn't benefit from these symmetries
4. Augmentation intensity too low

**Try:**
- Increase dropout: `aug_dropout: 0.15` or `0.20`
- Train longer (augmentation needs more epochs to show benefit)
- Check if model is overfitting without augmentation (if not, augmentation won't help much)

## Future Enhancements (Phase 2+)

Not yet implemented, but recommended if Phase 1 helps:

### Phase 2 (Moderate Effort)
- **Kinematic smearing**: Add Gaussian noise to pT, energy
- **Impact parameter smearing**: Smear dxy, dz, update Sip values
- **pT rescaling**: Scale all particle momenta by factor α ∈ [0.95, 1.05]

### Phase 3 (Advanced)
- **Full 3D rotations**: SO(3) rotations in momentum space
- **Collinear splitting**: Split high-pT particles into nearby pairs
- **CutMix**: Mix particles from different jets (same class)

See the original augmentation strategy document for details.

## References

- Implementation: `weaver/utils/data/augmentation.py`
- Integration: `weaver/utils/dataset.py` line 110
- Tests: `tests/test_augmentation.py`
- Config example: `weaver/configs/training_config_part_augmented.yaml`

## Questions?

If you encounter issues or want to add more augmentations, check:
1. The implementation in `augmentation.py` for details
2. The test file for usage examples
3. The dataset.py integration for how it's called



