import os
import sys
import json
import shutil
import numpy as np  

thisdir = os.path.abspath(os.path.dirname(__file__))
weavercoredir = os.path.abspath(os.path.join(thisdir, '../'))
weaverdir = thisdir  # Add this line
sys.path.append(weavercoredir)
import weaver.utils.jobsubmission.condortools as ct
import weaver.utils.jobsubmission.slurmtools as st


if __name__=='__main__':

    # load config
    # Default config file
    config_file = os.path.join(weaverdir, 'configs/training_config_part_augmented.yaml')
    
    # Allow overriding config file from command line
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    if not os.path.exists(config_file):
        raise Exception(f'Config file {config_file} does not exist.')

    with open(config_file, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    # Resolve paths (relative to config file location or absolute)
    config_dir = os.path.dirname(os.path.abspath(config_file))
    
    def resolve_path(path):
        if path is None: return None
        if os.path.isabs(path): return path
        return os.path.abspath(os.path.join(config_dir, path))

    data_config = resolve_path(config['data_config'])
    model_config = resolve_path(config['model_config'])
    sample_config_train = resolve_path(config['sample_train'])
    sample_config_test = resolve_path(config['sample_test'])
    
    # output dir
    if config.get('output_dir'):
        outputdir = resolve_path(config['output_dir'])
    else:
        # Default to config filename
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        outputdir = os.path.join(thisdir, f'output_{config_name}')

    # network settings
    num_epochs = config['num_epochs']
    steps_per_epoch = config['steps_per_epoch']
    batch_size = config['batch_size']
    
    # runmode and job settings
    runmode = config['runmode']
    gpus = config['gpus']

    # check if all config files exist
    files_to_check = [data_config, model_config, sample_config_train, sample_config_test]
    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception('File {} does not exist.'.format(f))

    # make output directory (remove if it already exists)
    if os.path.exists(outputdir):
        if config.get('keep_output_dir', False):
            print(f'Output directory {outputdir} already exists, keeping it.')
        else:
            shutil.rmtree(outputdir)
            os.makedirs(outputdir)
    else:
        os.makedirs(outputdir)

    # copy the config files to the output directory
    this_data_config = os.path.join(outputdir, 'data_config.yaml')
    os.system(f'cp {data_config} {this_data_config}')
    this_model_config = os.path.join(outputdir, 'model_config.py')
    os.system(f'cp {model_config} {this_model_config}')
    this_sample_config_train = os.path.join(outputdir, 'sample_config_train.yaml')
    os.system(f'cp {sample_config_train} {this_sample_config_train}')
    this_sample_config_test = os.path.join(outputdir, 'sample_config_test.yaml')
    os.system(f'cp {sample_config_test} {this_sample_config_test}')
    
    # Copy the run config itself
    os.system(f'cp {config_file} {os.path.join(outputdir, "run_config.yaml")}')

    # set model prefix
    model_prefix = os.path.join(outputdir, config['model_prefix_name'])

    # set output file for test results
    test_output = os.path.join(outputdir, config['test_output'])

    # make the command
    cmd = 'weaver'
    cmd += f' --data-train {this_sample_config_train}'
    cmd += f' --data-config {this_data_config}'
    cmd += f' --network-config {this_model_config}'
    cmd += f' --num-epochs {num_epochs}'
    cmd += f' --steps-per-epoch {steps_per_epoch}'
    cmd += f' --batch-size {batch_size}'
    cmd += f' --model-prefix {model_prefix}'
    cmd += f' --data-test {this_sample_config_test}'
    cmd += f' --predict-output {test_output}'
    
    # data loading options
    cmd += f' --num-workers {config.get("num_workers", 0)}'
    if config.get('copy_inputs', False):
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
            
    # compute options
    if gpus is not None and gpus != '': cmd += f' --gpus {gpus}'

    # run or submit commands
    if runmode == 'local':
        print(cmd)
        os.system(cmd)
    elif runmode=='condor':
        conda_activate = config.get('condor_conda_activate', 'source /eos/user/l/llambrec/miniforge3/bin/activate')
        conda_env = config.get('conda_env', 'weaver') # Default to weaver if not specified separately for condor?
        # Actually condor usually needs full path or specific activation
        
        ct.submitCommandAsCondorJob('cjob_weaver', cmd,
          jobflavour=config.get('condor_jobflavour', 'workday'), 
          conda_activate=conda_activate, conda_env=conda_env)
          
    elif runmode=='slurm':
        slurmscript = 'sjob_weaver.sh'
        # remove old slurm script if it exists
        if os.path.exists(slurmscript):
            os.remove(slurmscript)
            
        env_cmds = ([
          config.get('conda_activate', 'source /users/tgillin/miniconda3/etc/profile.d/conda.sh'),
          f'conda activate {config.get("conda_env", "weaver")}',
          f'cd {thisdir}'
        ])
        
        job_name = os.path.splitext(slurmscript)[0]
        slurm_options = {
          'job_name': job_name,
          'env_cmds': env_cmds,
          'memory': config.get('slurm_memory', '16G'),
          'time': config.get('slurm_time', '05:00:00')
        }
        
        if gpus is not None and gpus != '':
            slurm_options['partition'] = 'gpu'
            slurm_options['gres'] = 'gpu:1'
            slurm_options['gpus'] = '1'
            
        st.submitCommandAsSlurmJob(cmd, slurmscript, **slurm_options)
