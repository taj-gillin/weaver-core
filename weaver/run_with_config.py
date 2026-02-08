#!/usr/bin/env python
"""
Flexible training script that accepts a single training config file.
This config file references all other configs (data, model, samples) and training settings.
"""

import os
import sys
import argparse
import shutil
import yaml
import glob
import random

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


def expand_file_patterns(patterns):
    """Expand glob patterns to get list of actual files."""
    all_files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'Warning: No files matched pattern: {pattern}')
        all_files.extend(matches)
    return all_files


def split_files(files, train_ratio, seed=None):
    """Split files into train and test sets based on ratio.
    
    Args:
        files: List of file paths
        train_ratio: Fraction of files to use for training (0.0 to 1.0)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Tuple of (train_files, test_files)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle a copy of the list
    shuffled = files.copy()
    random.shuffle(shuffled)
    
    # Split based on ratio
    split_idx = int(len(shuffled) * train_ratio)
    train_files = shuffled[:split_idx]
    test_files = shuffled[split_idx:]
    
    return train_files, test_files


def write_sample_list(files, output_path):
    """Write a list of files to a sample list YAML file."""
    with open(output_path, 'w') as f:
        for file_path in files:
            f.write(f'- {file_path}\n')


def load_config(config_path):
    """Load and parse the training config YAML file."""
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file does not exist: {config_path}')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the directory containing the config file (for resolving relative paths)
    config_dir = os.path.dirname(config_path)
    
    # Resolve relative paths to absolute paths (relative to config file location)
    # If path doesn't exist relative to config file, try relative to weaver configs directory
    def resolve_path(path, default_base=config_dir):
        if os.path.isabs(path):
            return path
        # Try relative to config file first
        full_path = os.path.join(default_base, path)
        if os.path.exists(full_path):
            return os.path.abspath(full_path)
        # Try relative to weaver configs directory (handles nested paths like data_configs/standardized_configs/)
        weaver_configs_dir = os.path.join(thisdir, 'configs')
        alt_path = os.path.join(weaver_configs_dir, path)
        if os.path.exists(alt_path):
            return os.path.abspath(alt_path)
        # Also try just the basename in case it's a flat file in configs/
        basename_path = os.path.join(weaver_configs_dir, os.path.basename(path))
        if os.path.exists(basename_path):
            return os.path.abspath(basename_path)
        # Return absolute path even if doesn't exist (will be checked later)
        return os.path.abspath(full_path)
    
    # Resolve paths for config files
    config['data_config'] = resolve_path(config['data_config'])
    config['model_config'] = resolve_path(config['model_config'])
    
    # Check if using new data_files + train_test_split format or old sample_train/sample_test format
    if 'data_files' in config and 'train_test_split' in config:
        # New format: data_files with train/test split ratio
        config['use_split_mode'] = True
        # data_files can be a single pattern or a list
        data_files = config['data_files']
        if isinstance(data_files, str):
            config['data_files'] = [data_files]
    else:
        # Old format: separate sample_train and sample_test files
        config['use_split_mode'] = False
        config['sample_train'] = resolve_path(config['sample_train'])
        config['sample_test'] = resolve_path(config['sample_test'])
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run weaver training using a single training config file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config YAML file')
    
    # Allow override of key settings via CLI (optional)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory from config')
    parser.add_argument('--runmode', type=str, default=None,
                        choices=['local', 'condor', 'slurm'],
                        help='Override runmode from config')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Override steps_per_epoch from config')
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['adam', 'adamW', 'radam', 'ranger'],
                        help='Override optimizer from config')
    parser.add_argument('--start-lr', type=float, default=None,
                        help='Override start learning rate from config')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Enable torch.compile for faster training')
    parser.add_argument('--compile-mode', type=str, default=None,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='Override compile mode from config')
    parser.add_argument('--profile-train', action='store_true', default=False,
                        help='Enable real training-loop profiling for a short run')
    parser.add_argument('--profile-steps', type=int, default=None,
                        help='Number of training steps to capture when profiling')
    parser.add_argument('--profile-dir', type=str, default=None,
                        help='Directory for profiler traces (default: output directory)')
    parser.add_argument('--profile-train-full', action='store_true', default=False,
                        help='Profile the entire training run (all epochs/steps)')
    parser.add_argument('--profile-train-full-active', type=int, default=None,
                        help='Steps per profiler window for full-train profiling (overrides train default)')
    
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    
    # Override config with CLI arguments if provided
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.runmode is not None:
        config['runmode'] = args.runmode
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.steps_per_epoch is not None:
        config['steps_per_epoch'] = args.steps_per_epoch
    if args.optimizer is not None:
        config['optimizer'] = args.optimizer
    if args.start_lr is not None:
        config['start_lr'] = args.start_lr
    if args.compile:
        config['compile'] = True
    if args.compile_mode is not None:
        config['compile_mode'] = args.compile_mode
    if args.profile_train:
        config['profile_train'] = True
    if args.profile_steps is not None:
        config['profile_steps'] = args.profile_steps
    if args.profile_dir is not None:
        config['profile_dir'] = args.profile_dir
    if args.profile_train_full:
        config['profile_train_full'] = True
    if args.profile_train_full_active is not None:
        config['profile_train_full_active'] = args.profile_train_full_active
    
    # Extract config paths
    data_config = config['data_config']
    model_config = config['model_config']
    
    # Handle sample configs based on mode
    use_split_mode = config.get('use_split_mode', False)
    
    # Auto-generate output directory name if not provided or None
    output_dir_config = config.get('output_dir')
    if output_dir_config is None or output_dir_config == 'null':
        # Extract name from data config filename (remove extension)
        data_config_basename = os.path.splitext(os.path.basename(data_config))[0]
        # Save to group folder
        outputdir = os.path.join('/oscar/data/lgouskos/training_runs', data_config_basename)
    else:
        outputdir = os.path.abspath(output_dir_config)
    
    # Check if config files exist
    files_to_check = [data_config, model_config]
    if not use_split_mode:
        # Old format: check sample config files exist
        sample_config_train = config['sample_train']
        sample_config_test = config['sample_test']
        files_to_check.extend([sample_config_train, sample_config_test])
    
    for f in files_to_check:
        if not os.path.exists(f):
            raise FileNotFoundError(f'Config file does not exist: {f}')
    
    # Get settings with defaults
    keep_output_dir = config.get('keep_output_dir', False)
    model_prefix_name = config.get('model_prefix_name', 'network')
    test_output = config.get('test_output', 'output.root')
    num_epochs = config.get('num_epochs', 50)
    steps_per_epoch = config.get('steps_per_epoch', 300)
    batch_size = config.get('batch_size', 512)
    num_workers = config.get('num_workers', 0)
    copy_inputs = config.get('copy_inputs', True)
    gpus = config.get('gpus', '0')
    runmode = config.get('runmode', 'slurm')
    profile_train = config.get('profile_train', False)
    profile_steps = config.get('profile_steps', None)
    profile_dir_config = config.get('profile_dir', None)
    profile_train_full = config.get('profile_train_full', False)
    profile_train_full_active = config.get('profile_train_full_active', None)
    optimizer = config.get('optimizer', 'ranger')
    start_lr = config.get('start_lr', 5e-3)
    use_compile = config.get('compile', False)
    compile_mode = config.get('compile_mode', 'default')
    
    # Make output directory (remove if it already exists, unless keep_output_dir)
    if os.path.exists(outputdir):
        if keep_output_dir:
            print(f'Keeping existing output directory: {outputdir}')
        else:
            print(f'Removing existing output directory: {outputdir}')
            shutil.rmtree(outputdir)
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print(f'Created output directory: {outputdir}')
    
    # Copy the config files to the output directory
    this_data_config = os.path.join(outputdir, 'data_config.yaml')
    shutil.copy2(data_config, this_data_config)
    print(f'Copied data config to: {this_data_config}')
    
    this_model_config = os.path.join(outputdir, 'model_config.py')
    shutil.copy2(model_config, this_model_config)
    print(f'Copied model config to: {this_model_config}')
    
    # Handle sample configs based on mode
    this_sample_config_train = os.path.join(outputdir, 'sample_config_train.yaml')
    this_sample_config_test = os.path.join(outputdir, 'sample_config_test.yaml')
    
    if use_split_mode:
        # New format: expand patterns, split files, and generate sample lists
        data_files = config['data_files']
        train_ratio = config['train_test_split']
        split_seed = config.get('split_seed', None)
        
        # Expand glob patterns to get all files
        all_files = expand_file_patterns(data_files)
        
        if len(all_files) == 0:
            raise FileNotFoundError(f'No files found matching patterns: {data_files}')
        
        print(f'Found {len(all_files)} files matching data patterns')
        
        # Split into train and test sets
        train_files, test_files = split_files(all_files, train_ratio, seed=split_seed)
        
        print(f'Split into {len(train_files)} training files ({train_ratio*100:.0f}%) '
              f'and {len(test_files)} test files ({(1-train_ratio)*100:.0f}%)')
        
        if split_seed is not None:
            print(f'Using random seed: {split_seed}')
        
        # Write generated sample lists
        write_sample_list(train_files, this_sample_config_train)
        print(f'Generated training sample list: {this_sample_config_train}')
        
        write_sample_list(test_files, this_sample_config_test)
        print(f'Generated testing sample list: {this_sample_config_test}')
    else:
        # Old format: copy existing sample config files
        sample_config_train = config['sample_train']
        sample_config_test = config['sample_test']
        
        shutil.copy2(sample_config_train, this_sample_config_train)
        print(f'Copied training sample list to: {this_sample_config_train}')
        
        shutil.copy2(sample_config_test, this_sample_config_test)
        print(f'Copied testing sample list to: {this_sample_config_test}')
    
    # Also copy the training config file itself
    training_config_copy = os.path.join(outputdir, 'training_config.yaml')
    shutil.copy2(args.config, training_config_copy)
    print(f'Copied training config to: {training_config_copy}')
    
    # Set model prefix
    model_prefix = os.path.join(outputdir, model_prefix_name)
    
    # Set output file for test results
    if os.path.isabs(test_output):
        test_output_path = test_output
    else:
        test_output_path = os.path.join(outputdir, test_output)
    
    # Make the command
    cmd = 'weaver'
    cmd += f' --data-train {this_sample_config_train}'
    cmd += f' --data-config {this_data_config}'
    cmd += f' --network-config {this_model_config}'
    cmd += f' --num-epochs {num_epochs}'
    cmd += f' --steps-per-epoch {steps_per_epoch}'
    cmd += f' --batch-size {batch_size}'
    cmd += f' --model-prefix {model_prefix}'
    cmd += f' --data-test {this_sample_config_test}'
    cmd += f' --predict-output {test_output_path}'
    cmd += f' --num-workers {num_workers}'
    cmd += f' --optimizer {optimizer}'
    cmd += f' --start-lr {start_lr}'
    if use_compile:
        cmd += ' --compile'
        cmd += f' --compile-mode {compile_mode}'
    if copy_inputs:
        cmd += ' --copy-inputs'
    
    # Augmentation options
    if config.get('augment', False):
        cmd += ' --augment'
        if config.get('aug_rotation', False):
            cmd += ' --aug-rotation'
        if config.get('aug_reflection', False):
            cmd += ' --aug-reflection'
        if config.get('aug_dropout', 0) > 0:
            cmd += f' --aug-dropout {config["aug_dropout"]}'
        if config.get('aug_reflection_prob', 0.5) != 0.5:
            cmd += f' --aug-reflection-prob {config["aug_reflection_prob"]}'
    
    if gpus and gpus != '""':
        cmd += f' --gpus {gpus}'
    
    # Profiling options
    if profile_train:
        cmd += ' --profile-train'
        if profile_steps is not None:
            cmd += f' --profile-steps {profile_steps}'
        # Resolve profile dir; default to output directory
        if profile_dir_config is not None:
            if os.path.isabs(profile_dir_config):
                profile_dir_resolved = profile_dir_config
            else:
                profile_dir_resolved = os.path.abspath(os.path.join(config_dir, profile_dir_config))
        else:
            profile_dir_resolved = outputdir
        if profile_dir_resolved:
            cmd += f' --profile-dir {profile_dir_resolved}'
    if profile_train_full:
        cmd += ' --profile-train-full'
        if profile_train_full_active is not None:
            cmd += f' --profile-train-full-active {profile_train_full_active}'
    
    # Add wandb flags if enabled
    if config.get('use_wandb', False):
        cmd += ' --use-wandb'
        if config.get('wandb_project'):
            cmd += f' --wandb-project {config["wandb_project"]}'
        if config.get('wandb_entity'):
            cmd += f' --wandb-entity {config["wandb_entity"]}'
        if config.get('wandb_name'):
            cmd += f' --wandb-name {config["wandb_name"]}'
        if config.get('wandb_tags'):
            tags = ','.join(config['wandb_tags']) if isinstance(config['wandb_tags'], list) else config['wandb_tags']
            cmd += f' --wandb-tags {tags}'
        if config.get('wandb_notes'):
            cmd += f' --wandb-notes "{config["wandb_notes"]}"'
    
    print('\n' + '='*80)
    print('Generated weaver command:')
    print('='*80)
    print(cmd)
    print('='*80 + '\n')
    
    # Get slurm/condor specific settings
    slurm_memory = config.get('slurm_memory', '16G')
    slurm_time = config.get('slurm_time', '05:00:00')
    slurm_cpus = config.get('slurm_cpus', 1)
    conda_activate = config.get('conda_activate', 'source /users/tgillin/miniconda3/etc/profile.d/conda.sh')
    conda_env = config.get('conda_env', 'weaver')
    condor_conda_activate = config.get('condor_conda_activate', 'source /eos/user/l/llambrec/miniforge3/bin/activate')
    condor_jobflavour = config.get('condor_jobflavour', 'workday')
    
    # Run or submit commands
    if runmode == 'local':
        print('Running locally...')
        os.system(cmd)
    elif runmode == 'condor':
        print('Submitting to Condor...')
        ct.submitCommandAsCondorJob(
            'cjob_weaver', cmd,
            jobflavour=condor_jobflavour,
            conda_activate=condor_conda_activate,
            conda_env=conda_env
        )
    elif runmode == 'slurm':
        print('Submitting to Slurm...')
        slurmscript = 'sjob_weaver.sh'
        # Remove old slurm script if it exists
        if os.path.exists(slurmscript):
            os.remove(slurmscript)
        
        env_cmds = [
            conda_activate,
            f'conda activate {conda_env}',
            f'cd {thisdir}'
        ]
        
        # Set up logs directory
        logs_dir = os.path.join(weavercoredir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        job_name = os.path.splitext(slurmscript)[0]
        # Set log file paths to logs folder with job name and job ID pattern
        log_output = os.path.join(logs_dir, f'{job_name}_%j.out')
        log_error = os.path.join(logs_dir, f'{job_name}_%j.err')
        
        slurm_options = {
            'job_name': job_name,
            'env_cmds': env_cmds,
            'memory': slurm_memory,
            'time': slurm_time,
            'cpus_per_task': slurm_cpus,
            'output': log_output,
            'error': log_error
        }
        
        if gpus and gpus != '""':
            slurm_options['partition'] = 'gpu'
            slurm_options['gres'] = 'gpu:1'
            slurm_options['gpus'] = '1'
        
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
        print(f'Slurm job submitted. Script saved to: {slurmscript}')
        print(f'Logs will be saved to: {logs_dir}/')
    
    print(f'\nOutput directory: {outputdir}')
    print('Training will save models to:')
    print(f'  - {model_prefix}_epoch-{{N}}_state.pt (each epoch)')
    print(f'  - {model_prefix}_best_epoch_state.pt (best model)')
    print(f'\nTest results will be saved to: {test_output_path}')


if __name__ == '__main__':
    main()
