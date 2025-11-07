# Variable Visualization Script

## Overview

`visualize_variables.py` generates comprehensive visualizations for all input variables in the dataset, including:
- Distribution histograms
- Before/after standardization comparisons
- Statistics summary plots

## Features

- ✅ Distribution plots for each variable (original and log scale)
- ✅ Before/after standardization comparison plots
- ✅ Summary statistics visualization
- ✅ Handles large datasets efficiently with sampling
- ✅ Saves all plots as PNG files

## Usage

### Basic Usage

```bash
cd /users/tgillin/files/weaver-core
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver
python exploration_temp/visualize_variables.py
```

### Custom Options

```bash
python exploration_temp/visualize_variables.py \
    --data-dir /HEP/data/share/aleph/aleph-data/ntuples/mc \
    --config configs/data_config_pnet.yaml \
    --output-dir exploration_temp/plots \
    --sample-fraction 0.05 \
    --max-events-per-file 20000
```

### Arguments

- `--data-dir`: Directory containing `output_*_train.root` files (default: `/HEP/data/share/aleph/aleph-data/ntuples/mc`)
- `--config`: Path to data config file relative to weaver dir (default: `configs/data_config_pnet.yaml`)
- `--output-dir`: Output directory for plots (default: `exploration_temp/plots`)
- `--sample-fraction`: Fraction of events to use (default: 0.05 = 5%)
- `--max-events-per-file`: Maximum events per file (default: 20000)

## Output Structure

The script generates plots in the following directory structure:

```
exploration_temp/plots/
├── distributions/
│   ├── pfcand_pt_log.png
│   ├── pfcand_e_log.png
│   └── ... (one plot per variable)
├── standardized/
│   ├── pfcand_pt_log_standardized.png
│   ├── pfcand_e_log_standardized.png
│   └── ... (comparison plots)
└── summary/
    └── statistics_summary.png
```

## Plot Types

### 1. Distribution Plots (`distributions/`)
- **Histogram**: Shows the distribution of values
- **Log scale histogram**: Shows distribution on log scale (if values are positive)
- Includes statistics overlay (mean, median, std, min, max)

### 2. Standardized Comparison Plots (`standardized/`)
- **Top panel**: Original distribution with center line marked
- **Bottom panel**: Standardized distribution (centered at 0, scaled)
- Shows clip bounds (-5, 5) used by weaver

### 3. Summary Plot (`summary/`)
- **Centers**: All standardization centers (medians)
- **Scales**: All standardization scales
- **Means**: All variable means
- **Standard deviations**: All variable standard deviations

## Requirements

The script requires:
- `matplotlib` for plotting
- `seaborn` for styling
- `numpy` and `awkward` for data handling
- `weaver` framework for data loading

## Notes

- The script uses a small sample (5% by default) for speed
- Uses first 10 training files by default
- Automatically finds standardized config from `standardized_configs/` directory
- Outliers are trimmed (1st-99th percentile) for better visualization
- All plots are saved as PNG files with 150 DPI

## Example Output

After running, you'll see:
```
Found 10 training files (using first 10 for visualization)

Loading data from 10 files...
Total events loaded: 15000
Events after selection: 15000

Generating plots for 26 variables...
  Distribution plots: 100%|████████| 26/26
  Standardized plots: 100%|████████| 26/26

Visualization Complete!

Plots saved to:
  Distributions: exploration_temp/plots/distributions
  Standardized: exploration_temp/plots/standardized
  Summary: exploration_temp/plots/summary
```



