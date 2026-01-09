#!/bin/bash
#SBATCH --job-name=profile_adamw
#SBATCH --output=/users/tgillin/files/weaver-core/logs/profile_adamw_%j.out
#SBATCH --error=/users/tgillin/files/weaver-core/logs/profile_adamw_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver
cd /users/tgillin/files/weaver-core/weaver

# Run with AdamW optimizer for comparison
weaver \
    --data-train /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/sample_config_train.yaml \
    --data-config /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/data_config.yaml \
    --network-config /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/model_config.py \
    --num-epochs 5 \
    --steps-per-epoch 100 \
    --batch-size 512 \
    --model-prefix /oscar/data/lgouskos/training_runs/profile_adamw/network \
    --num-workers 0 \
    --gpus 0 \
    --optimizer adamW \
    --profile-train \
    --profile-steps 50 \
    --profile-dir /oscar/data/lgouskos/training_runs/profile_adamw




