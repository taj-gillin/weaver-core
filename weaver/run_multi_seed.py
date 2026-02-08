#!/usr/bin/env python
"""
Multi-seed training orchestration script.
Trains multiple models with different random seeds and saves each to a unique directory.
Generates a manifest file for averaged evaluation later.
"""

import os
import sys
import argparse
import json
import yaml
import shutil
from datetime import datetime

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)

import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


def load_config(config_path):
    """Load and parse the training config YAML file."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file does not exist: {config_path}')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_seed_config(base_config_path, output_path, seed, output_dir):
    """Create a modified config file for a specific seed."""
    config = load_config(base_config_path)
    
    # Update seed and output directory
    config['split_seed'] = seed
    config['output_dir'] = output_dir
    
    # Update wandb name to include seed if wandb is enabled
    if config.get('use_wandb', False) and config.get('wandb_name'):
        config['wandb_name'] = f"{config['wandb_name']}_seed{seed}"
    
    # Write modified config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-seed training with averaged evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base training config YAML file')
    parser.add_argument('--num-seeds', type=int, default=5,
                        help='Number of seeds to train')
    parser.add_argument('--seed-start', type=int, default=0,
                        help='Starting seed number')
    parser.add_argument('--base-output-dir', type=str, default=None,
                        help='Override base output directory (each seed gets a subdirectory)')
    
    # Allow override of key settings (passed through to run_with_config.py)
    parser.add_argument('--runmode', type=str, default=None,
                        choices=['local', 'condor', 'slurm'],
                        help='Override runmode from config')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Override steps_per_epoch from config')
    
    args = parser.parse_args()
    
    # Load base config to get output directory
    base_config = load_config(args.config)
    
    # Determine base output directory
    if args.base_output_dir:
        base_output_dir = os.path.abspath(args.base_output_dir)
    elif base_config.get('output_dir'):
        base_output_dir = os.path.abspath(base_config['output_dir'])
    else:
        # Generate from data config name
        data_config = base_config.get('data_config', 'unknown')
        data_config_basename = os.path.splitext(os.path.basename(data_config))[0]
        base_output_dir = os.path.join('/oscar/data/lgouskos/training_runs', f'{data_config_basename}_multiseed')
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    print(f'Multi-seed output directory: {base_output_dir}')
    
    # Create configs subdirectory for seed-specific configs
    configs_dir = os.path.join(base_output_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    
    # Generate seed list
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    
    # Prepare manifest
    manifest = {
        'base_config': os.path.abspath(args.config),
        'num_seeds': args.num_seeds,
        'seed_start': args.seed_start,
        'created': datetime.now().isoformat(),
        'base_output_dir': base_output_dir,
        'runs': []
    }
    
    # Copy base config to base output directory
    base_config_copy = os.path.join(base_output_dir, 'base_training_config.yaml')
    shutil.copy2(args.config, base_config_copy)
    print(f'Copied base config to: {base_config_copy}')
    
    print(f'\nSubmitting {args.num_seeds} training runs with seeds: {seeds}')
    print('=' * 80)
    
    for seed in seeds:
        # Create seed-specific output directory
        seed_output_dir = os.path.join(base_output_dir, f'seed_{seed}')
        
        # Create seed-specific config
        seed_config_path = os.path.join(configs_dir, f'config_seed_{seed}.yaml')
        create_seed_config(args.config, seed_config_path, seed, seed_output_dir)
        
        # Expected output.root path
        test_output = base_config.get('test_output', 'output.root')
        if os.path.isabs(test_output):
            output_root = test_output.replace('.root', f'_seed{seed}.root')
        else:
            output_root = os.path.join(seed_output_dir, test_output)
        
        # Add to manifest
        manifest['runs'].append({
            'seed': seed,
            'config': seed_config_path,
            'output_dir': seed_output_dir,
            'output_root': output_root
        })
        
        # Build command for run_with_config.py
        cmd = f'python {os.path.join(thisdir, "run_with_config.py")} --config {seed_config_path}'
        
        # Add overrides if specified
        if args.runmode:
            cmd += f' --runmode {args.runmode}'
        if args.num_epochs:
            cmd += f' --num-epochs {args.num_epochs}'
        if args.batch_size:
            cmd += f' --batch-size {args.batch_size}'
        if args.steps_per_epoch:
            cmd += f' --steps-per-epoch {args.steps_per_epoch}'
        
        print(f'\n[Seed {seed}] Command: {cmd}')
        print(f'[Seed {seed}] Output dir: {seed_output_dir}')
        
        # Execute the command
        runmode = args.runmode or base_config.get('runmode', 'slurm')
        
        if runmode == 'local':
            print(f'[Seed {seed}] Running locally...')
            os.system(cmd)
        else:
            # For slurm/condor, we run the python script which handles submission
            print(f'[Seed {seed}] Submitting via run_with_config.py ({runmode})...')
            os.system(cmd)
    
    # Save manifest
    manifest_path = os.path.join(base_output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print('\n' + '=' * 80)
    print(f'Multi-seed training submitted!')
    print(f'Manifest saved to: {manifest_path}')
    print(f'\nAfter all jobs complete, run averaged evaluation with:')
    print(f'  python evaluation/evaluate_averaged.py -m {manifest_path}')


if __name__ == '__main__':
    main()
