#!/bin/bash
#SBATCH --job-name=weaver_eval_avg
#SBATCH --output=logs/weaver_eval_avg_%j.out
#SBATCH --error=logs/weaver_eval_avg_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Averaged evaluation submission script
# Usage: sbatch submit_evaluate_averaged.sh <manifest.json>
#    OR: sbatch submit_evaluate_averaged.sh <output1.root> <output2.root> ...
#
# Examples:
#   sbatch submit_evaluate_averaged.sh /path/to/manifest.json
#   sbatch submit_evaluate_averaged.sh seed_0/output.root seed_1/output.root seed_2/output.root

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core/weaver

# Check if input is provided
if [ -z "$1" ]; then
    echo "Error: Input is required"
    echo "Usage: sbatch $0 <manifest.json>"
    echo "   OR: sbatch $0 <output1.root> <output2.root> ..."
    exit 1
fi

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Determine input type
if [[ "$1" == *.json ]]; then
    # Manifest file
    python evaluation/evaluate_averaged.py -m "$1"
else
    # List of root files
    python evaluation/evaluate_averaged.py -i "$@"
fi
