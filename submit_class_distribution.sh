#!/bin/bash
#SBATCH --job-name=class_dist
#SBATCH --output=logs/class_dist_%j.out
#SBATCH --error=logs/class_dist_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core

# Check if input file argument is provided
if [ -z "$1" ]; then
    echo "Error: Input file path is required"
    echo "Usage: $0 <yaml_sample_list_or_root_files>"
    echo ""
    echo "Examples:"
    echo "  sbatch submit_class_distribution.sh weaver/configs/samplelists/oscar/samples_testing.yaml"
    echo "  sbatch submit_class_distribution.sh /path/to/file1.root /path/to/file2.root"
    exit 1
fi

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Run the class distribution script with all provided arguments
python weaver/utils/class_distribution.py -i "$@" --treename tree


