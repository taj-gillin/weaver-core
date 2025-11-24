# Interpretability Analysis for Jet Flavor Tagging

This module provides tools to understand ParticleTransformer model decisions for b/c/udsg jet classification.

## Features

The `interpret_model.py` script implements four interpretability methods:

1. **Gradient-Based Feature Attribution** - Computes ∂(score)/∂(feature) to identify which features most influence each class prediction
2. **Permutation Importance** - Measures performance drop when shuffling each feature
3. **Feature Group Ablation** - Analyzes importance of physics-motivated feature groups (kinematic, impact parameters, particle ID, etc.)
4. **Attention Weight Statistics** - Analyzes which particles the model focuses on (TODO: requires model modification)

## Usage

### Running Locally

```bash
python weaver/evaluation/interpret_model.py \
    --model-path models/part/network_best_epoch_state.pt \
    --data-config weaver/configs/data_configs/data_config_part_standardized.yaml \
    --network-config weaver/configs/model_config_part.py \
    --test-data 'path/to/test/*.root' \
    --output-dir interpretability_results \
    --methods gradient permutation ablation \
    --num-jets 1000
```

### Running on SLURM Cluster

```bash
sbatch submit_interpret.sh \
    models/part/network_best_epoch_state.pt \
    weaver/configs/data_configs/data_config_part_standardized.yaml \
    'path/to/test/*.root'
```

Optional arguments for sbatch script:
```bash
sbatch submit_interpret.sh \
    <model_path> \
    <data_config> \
    <test_data> \
    [network_config] \
    [output_dir] \
    [methods] \
    [num_jets]
```

## Command Line Arguments

- `--model-path`: Path to trained model checkpoint (.pt file)
- `--data-config`: Path to data config YAML file
- `--network-config`: Path to network config Python file
- `--test-data`: Test data file patterns (supports wildcards and YAML sample lists)
- `--output-dir`: Output directory for results (default: `interpretability_results`)
- `--methods`: Which methods to run (choices: `gradient`, `permutation`, `ablation`, `attention`)
- `--num-jets`: Maximum number of jets to analyze (default: all)
- `--batch-size`: Batch size for inference (default: 512)
- `--cpu`: Use CPU instead of GPU
- `--n-repeats`: Number of repeats for permutation importance (default: 5)

## Output Files

The script generates the following files in the output directory:

### CSV Files
- `gradient_importance_per_class.csv` - Gradient-based importance scores per class
- `permutation_importance.csv` - Permutation importance scores with std
- `ablation_importance.csv` - Feature group ablation results

### Visualizations
- `gradient_importance_heatmap.png` - Heatmap of feature importance per class
- `permutation_importance_bars.png` - Bar plots of permutation importance
- `ablation_importance_bars.png` - Feature group comparison

## Feature Groups

Features are organized into physics-motivated groups:

- **Kinematic**: `pt_log`, `e_log`, `ptrel_log`, `erel_log`, `drrel`
- **Impact Parameters**: `dxy`, `dz`, `btagSip2dVal`, `btagSip2dSig`, `btagSip3dVal`, `btagSip3dSig`
- **Jet Distance**: `btagJetDistVal`, `btagJetDistSig`
- **Particle ID**: `isChargedHad`, `isNeutralHad`, `isGamma`, `isEl`, `isMu`
- **Angular**: `thetarel`, `phirel`
- **Charge**: `charge`

## Expected Results

Based on physics expectations:

- **b-jets**: Impact parameter features (SIP2D, SIP3D) should be highly important due to displaced vertices
- **c-jets**: Moderate impact parameters, lepton identification features
- **udsg-jets**: Kinematic features, particle multiplicity

## Requirements

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

All dependencies should already be installed in the weaver conda environment.

## Notes

- Gradient attribution requires gradients, so model is temporarily set to train mode
- Permutation importance can be slow for large datasets (use `--num-jets` to limit)
- Attention weight extraction is not yet implemented (requires model modification)
- Results are saved incrementally, so you can inspect outputs while the script runs
