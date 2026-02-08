#!/bin/bash
# Multi-seed training submission script
# Usage: ./submit_multi_seed.sh <config.yaml> [num_seeds] [base_output_dir]
#
# Examples:
#   ./submit_multi_seed.sh weaver/configs/training_config_part.yaml
#   ./submit_multi_seed.sh weaver/configs/training_config_part.yaml 5
#   ./submit_multi_seed.sh weaver/configs/training_config_part.yaml 5 /path/to/output

# Check if config file argument is provided
if [ -z "$1" ]; then
    echo "Error: Config file path is required"
    echo "Usage: $0 <config.yaml> [num_seeds] [base_output_dir]"
    exit 1
fi

CONFIG_FILE="$1"
NUM_SEEDS="${2:-5}"
BASE_OUTPUT_DIR="$3"

# Change to the weaver directory
cd /users/tgillin/files/weaver-core/weaver

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Build command
CMD="python run_multi_seed.py --config \"$CONFIG_FILE\" --num-seeds $NUM_SEEDS"

if [ -n "$BASE_OUTPUT_DIR" ]; then
    CMD="$CMD --base-output-dir \"$BASE_OUTPUT_DIR\""
fi

echo "Running: $CMD"
eval $CMD
