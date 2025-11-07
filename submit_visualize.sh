#!/bin/bash
#SBATCH --job-name=weaver_visualize
#SBATCH --output=logs/weaver_visualize_%j.out
#SBATCH --error=logs/weaver_visualize_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

# Run visualization script
python exploration_temp/visualize_variables.py \
    --data-dir /HEP/data/share/aleph/aleph-data/ntuples/mc \
    --config configs/data_config_pnet.yaml \
    --output-dir exploration_temp/plots \
    --sample-fraction 0.05 \
    --max-events-per-file 20000

