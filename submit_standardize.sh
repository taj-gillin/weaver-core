#!/bin/bash
#SBATCH --job-name=weaver_standardize
#SBATCH --output=logs/weaver_standardize_%j.out
#SBATCH --error=logs/weaver_standardize_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Change to the weaver-core directory
cd /users/tgillin/files/weaver-core

# Activate conda environment
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver

python weaver/configs/compute_standardization.py \
--samples-file weaver/configs/samplelists/oscar/samples_training.yaml \
--config-pnet configs/data_configs/data_config_pnet.yaml \
--config-part configs/data_configs/data_config_part.yaml \
--sample-fraction 0.1 --max-events-per-file 100000 \
--output-dir weaver/configs/data_configs
