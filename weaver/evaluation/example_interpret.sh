#!/bin/bash
# Example script showing how to run interpretability analysis
# This is a template - update the paths to match your setup

# Set your paths here
MODEL_PATH="models/part/network_best_epoch_state.pt"
DATA_CONFIG="weaver/configs/data_configs/data_config_part_standardized.yaml"
NETWORK_CONFIG="weaver/configs/model_config_part.py"
TEST_DATA="path/to/your/test/data/*.root"

# Example 1: Run all methods on 1000 jets
echo "Example 1: Running all methods on 1000 jets"
python weaver/evaluation/interpret_model.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --test-data $TEST_DATA \
    --output-dir interpretability_results_example1 \
    --methods gradient permutation ablation \
    --num-jets 1000 \
    --batch-size 512

# Example 2: Run only gradient attribution on all jets
echo "Example 2: Running only gradient attribution"
python weaver/evaluation/interpret_model.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --test-data $TEST_DATA \
    --output-dir interpretability_results_gradient_only \
    --methods gradient \
    --batch-size 512

# Example 3: Run permutation importance with more repeats for robustness
echo "Example 3: Running permutation importance with 10 repeats"
python weaver/evaluation/interpret_model.py \
    --model-path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --network-config "$NETWORK_CONFIG" \
    --test-data $TEST_DATA \
    --output-dir interpretability_results_permutation \
    --methods permutation \
    --num-jets 500 \
    --n-repeats 10 \
    --batch-size 512

# Example 4: Submit to SLURM cluster
echo "Example 4: Submitting to SLURM cluster"
sbatch submit_interpret.sh \
    "$MODEL_PATH" \
    "$DATA_CONFIG" \
    "$TEST_DATA" \
    "$NETWORK_CONFIG" \
    "interpretability_results_slurm" \
    "gradient permutation ablation" \
    1000

echo "Done! Check the output directories for results."
