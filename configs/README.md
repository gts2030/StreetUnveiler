# StreetUnveiler Configuration System

## Overview

The StreetUnveiler training system uses a flexible YAML-based configuration system that allows you to easily customize training parameters without modifying the source code.

## Quick Start

### Basic Usage
```bash
# Use default configuration
python train.py

# Use custom configuration  
python train.py --config configs/my_experiment.yaml

# Use simple template with overrides
python train.py --config configs/simple_config.yaml
```

### Creating Custom Configurations

1. **Start with the simple template:**
   ```bash
   cp configs/simple_config.yaml configs/my_experiment.yaml
   ```

2. **Edit only the values you want to change:**
   ```yaml
   # my_experiment.yaml
   loss_weights:
     uncertainty_lambda1: 2.0  # Increase uncertainty weight
   
   progress:
     training_title: "🔬 My Experiment"
   ```

3. **Run training:**
   ```bash
   python train.py --config configs/my_experiment.yaml
   ```

## Configuration Files

### `training_config.yaml`
Complete configuration with all available options and default values. Use as reference.

### `simple_config.yaml`  
Minimal template showing only commonly changed values. Use as starting point for experiments.

## Configuration Structure

### Core Sections

#### `feature_cfg`
Feature extraction and depth estimation configuration:
```yaml
feature_cfg:
  device: "cuda"
  mono_prior:
    feature_extractor: "dinov2_reg_small_fine"
    depth: "metric3d_vit_large"
  cam:
    fx: 1000.0
```

#### `uncertainty`
Uncertainty estimation parameters:
```yaml
uncertainty:
  ssim_median_filter_size: 5
  ssim_window_size: 7
  uncer_depth_mult: 0.2
  opacity_th_for_uncer_loss: 0.9
  fallback_loss_weight: 0.01
  beta_epsilon: 1e-8
```

#### `loss_weights`
Training loss weights and lambda parameters:
```yaml
loss_weights:
  semantic_ce_weights: [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]  # 6 classes
  render_lambda5: 1.0    # Color loss weight
  render_lambda6: 1.0    # Depth loss weight  
  render_lambda7: 0.01   # Isotropic regularization
  uncertainty_lambda1: 1.0   # Depth uncertainty
  uncertainty_lambda2: 0.01  # Variance regularization
  uncertainty_lambda3: 0.01  # Uncertainty regularization
```



## Advanced Usage

### Partial Configuration Override

You only need to specify the values you want to change from defaults:

```yaml
# minimal_config.yaml
loss_weights:
  uncertainty_lambda1: 2.0  # Only change this one value

progress:
  training_title: "🧪 Ablation Study"  # And this title
```

All other values will use defaults from the system.

### Experiment Organization

Organize experiments with descriptive config files:

```
configs/
├── experiments/
│   ├── baseline.yaml           # Baseline experiment
│   ├── high_uncertainty.yaml   # High uncertainty weights
│   ├── fast_training.yaml      # Faster training settings
│   └── ablation_study_1.yaml   # Specific ablation
├── training_config.yaml        # Full reference config
└── simple_config.yaml          # Simple template
```

### Validation and Debugging

The system automatically:
- ✅ Validates configuration completeness
- ✅ Shows configuration summary at startup
- ✅ Falls back to defaults for missing values
- ✅ Handles malformed YAML gracefully

## Configuration Utilities

### Python API

```python
from utils.config_utils import (
    load_training_config,
    save_config, 
    validate_config,
    print_config_summary
)

# Load and validate
config = load_training_config("my_config.yaml")
is_valid = validate_config(config)

# Print summary
print_config_summary(config)

# Save config
save_config(config, "output/used_config.yaml")
```

### Command Line Tools

```bash
# Test configuration loading
python -c "from utils.config_utils import load_training_config, print_config_summary; print_config_summary(load_training_config('configs/simple_config.yaml'))"
```

## Tips and Best Practices

### 🎯 Experiment Workflow
1. Start with `simple_config.yaml`
2. Change 1-3 parameters per experiment
3. Use descriptive config file names
4. Save configs alongside results

### 🔧 Parameter Tuning
- **Uncertainty weights:** Start with λ1=1.0, λ2=0.01, λ3=0.01
- **Loss weights:** Adjust semantic_ce_weights for class balance
- **Feature settings:** Try different depth estimators and feature extractors

### 📊 Reproducibility  
- Save exact config used: `--config` argument is logged
- Config summary shows all active parameters
- Version control your config files

### ⚡ Performance
- Use smaller models in feature_cfg for faster training
- Adjust uncertainty parameters for speed vs quality

## Troubleshooting

### Common Issues

**Config file not found:**
```
Configuration file not found: my_config.yaml
Using default configuration...
```
→ Check file path and spelling

**YAML syntax error:**
```
Error parsing YAML configuration: ...
Using default configuration...
```
→ Validate YAML syntax (use online YAML validator)

**Missing parameters:**
```
Missing configuration key: uncertainty.ssim_window_size
```
→ System will use defaults, or check for typos

### Getting Help

1. Check `configs/training_config.yaml` for all available options
2. Use `configs/simple_config.yaml` as starting template  
3. Run with default config first to ensure base system works
4. Check configuration summary output for active parameters

## Examples

### Reduce Uncertainty Regularization
```yaml
# configs/less_regularization.yaml
loss_weights:
  uncertainty_lambda2: 0.001  # Reduce from 0.01
  uncertainty_lambda3: 0.001  # Reduce from 0.01
```

### Custom Feature Extractor
```yaml
# configs/custom_features.yaml  
feature_cfg:
  mono_prior:
    feature_extractor: "dinov2_reg_large"  # Use larger model
    depth: "metric3d_vit_small"  # Use smaller depth model
  cam:
    fx: 800.0  # Different focal length
```

### High Uncertainty Settings
```yaml
# configs/high_uncertainty.yaml
uncertainty:
  ssim_window_size: 15  # Larger window
  uncer_depth_mult: 0.3  # Higher depth multiplier

loss_weights:
  uncertainty_lambda1: 2.0  # Higher uncertainty weight
```

---

For more details, see the source code in `utils/config_utils.py` and examples in `configs/`. 