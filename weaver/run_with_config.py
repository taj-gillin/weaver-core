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
    
    args = parser.parse_args()
    
    # Load config file
    config = load_config(args.config)
    
    # Override config with CLI arguments if provided
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.runmode is not None:
        config['runmode'] = args.runmode
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Extract config paths
    data_config = config['data_config']
    model_config = config['model_config']
    sample_config_train = config['sample_train']
    sample_config_test = config['sample_test']
    
    # Auto-generate output directory name if not provided or None
    output_dir_config = config.get('output_dir')
    if output_dir_config is None or output_dir_config == 'null':
        # Extract name from data config filename (remove extension)
        data_config_basename = os.path.splitext(os.path.basename(data_config))[0]
        # Save to group folder
        outputdir = os.path.join('/oscar/data/lgouskos/training_runs', data_config_basename)
    else:
        outputdir = os.path.abspath(output_dir_config)
    
    # Check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test]
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
    
    this_sample_config_train = os.path.join(outputdir, 'sample_config_train.yaml')
    shutil.copy2(sample_config_train, this_sample_config_train)
    print(f'Copied training sample list to: {this_sample_config_train}')
    
    this_sample_config_test = os.path.join(outputdir, 'sample_config_test.yaml')
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
    if copy_inputs:
        cmd += ' --copy-inputs'
    if gpus and gpus != '""':
        cmd += f' --gpus {gpus}'
    
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
