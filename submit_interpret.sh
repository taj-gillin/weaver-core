#!/bin/bash
#SBATCH --job-name=weaver_interpret
#SBATCH --output=logs/weaver_interpret_%j.out
#SBATCH --error=logs/weaver_interpret_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core/weaver

# Check if required arguments are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <model_path> <data_config> <test_data>"
    echo "Example: $0 models/part/network_best_epoch_state.pt configs/data_configs/data_config_part_standardized.yaml 'test_data/*.root'"
    exit 1
fi

MODEL_PATH="$1"
DATA_CONFIG="$2"
TEST_DATA="$3"
NETWORK_CONFIG="${4:-configs/model_config_part.py}"
OUTPUT_DIR="${5:-interpretability_results}"
METHODS="${6:-gradient permutation ablation attention}"
NUM_JETS="${7:-1000}"

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Run interpretability analysis
python evaluation/interpret_model.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --test-data $TEST_DATA \
    --output-dir "$OUTPUT_DIR" \
    --methods $METHODS \
    --num-jets $NUM_JETS \
    --batch-size 512 \
    --gradient-batch-size 64

echo "Interpretability analysis complete. Results saved to $OUTPUT_DIR"
