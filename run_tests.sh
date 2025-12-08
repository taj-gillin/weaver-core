#!/bin/bash
source /users/tgillin/miniconda3/etc/profile.d/conda.sh
conda activate weaver
python tests/test_augmentation.py
