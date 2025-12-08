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
weaver --data-train /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/sample_config_train.yaml --data-config /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/data_config.yaml --network-config /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/model_config.py --num-epochs 50 --steps-per-epoch 300 --batch-size 512 --model-prefix /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/network --data-test /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/sample_config_test.yaml --predict-output /oscar/home/tgillin/files/weaver-core/weaver/output_part_augmented/output.root --num-workers 0 --copy-inputs --gpus 0
