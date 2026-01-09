#!/bin/bash
#SBATCH --job-name=sjob_weaver
#SBATCH --partition=gpu
#SBATCH --output /oscar/home/tgillin/files/weaver-core/logs/sjob_weaver_%j.out
#SBATCH --error /oscar/home/tgillin/files/weaver-core/logs/sjob_weaver_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=09:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
echo "current host:" $(hostname)
echo "current directory:" $(pwd)
echo "current time:" $(date)
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver
cd /oscar/home/tgillin/files/weaver-core/weaver
weaver --data-train /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/sample_config_train.yaml --data-config /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/data_config.yaml --network-config /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/model_config.py --num-epochs 100 --steps-per-epoch 300 --batch-size 512 --model-prefix /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/network --data-test /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/sample_config_test.yaml --predict-output /oscar/data/lgouskos/training_runs/output_part_augmented_no_drop/output.root --num-workers 0 --optimizer ranger --start-lr 0.005 --copy-inputs --augment --aug-rotation --aug-reflection --gpus 0 --use-wandb --wandb-project jet-tagging --wandb-entity aleph_project --wandb-name part_standardized_augmented_no_drop --wandb-tags part,standardized,augmented,no-dropout --wandb-notes "Baseline training with standardized data and augmentation without dropout"
