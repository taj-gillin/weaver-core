#!/usr/bin/env python

import os
import ast
import sys
import shutil
import glob
import argparse
import functools
import numpy as np
import math
import copy
import torch
from weaver.utils.nn.tools import _flatten_preds

from torch.utils.data import DataLoader
from weaver.utils.logger import _logger, _configLogger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from weaver.utils.samplelisttools import read_sample_list

parser = argparse.ArgumentParser()
parser.add_argument('--regression-mode', action='store_true', default=False,
                    help='run in regression mode if this flag is set; otherwise run in classification mode')
parser.add_argument('-c', '--data-config', type=str,
                    help='data config YAML file')
parser.add_argument('--extra-selection', type=str, default=None,
                    help='Additional selection requirement, will modify `selection` to `(selection) & (extra)` on-the-fly')
parser.add_argument('--extra-test-selection', type=str, default=None,
                    help='Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly')
parser.add_argument('-i', '--data-train', nargs='*', default=[],
                    help='training files; supported syntax:'
                         ' (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;'
                         ' (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,'
                         ' the file splitting (for each dataloader worker) will be performed per group,'
                         ' and then mixed together, to ensure a uniform mixing from all groups for each worker.'
                    )
parser.add_argument('-l', '--data-val', nargs='*', default=[],
                    help='validation files; when not set, will use training files and split by `--train-val-split`')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test a%%10:/path/to/a/*`, will split per 10 input files')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('--fetch-by-files', action='store_true', default=False,
                    help='When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. '
                         'Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.')
parser.add_argument('--fetch-step', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--no-remake-weights', action='store_true', default=False,
                    help='do not remake weights for sampling (reweighting), use existing ones in the previous auto-generated data config YAML file')
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
# Wandb logging options
parser.add_argument('--use-wandb', action='store_true', default=False,
                    help='enable Weights & Biases logging')
parser.add_argument('--wandb-project', type=str, default='weaver-training',
                    help='wandb project name')
parser.add_argument('--wandb-entity', type=str, default=None,
                    help='wandb team/user name')
parser.add_argument('--wandb-name', type=str, default=None,
                    help='wandb run name (auto-generated if not provided)')
parser.add_argument('--wandb-tags', type=str, default=None,
                    help='comma-separated list of tags for the wandb run')
parser.add_argument('--wandb-notes', type=str, default=None,
                    help='notes for the wandb run')
parser.add_argument('-n', '--network-config', type=str,
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-m', '--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
parser.add_argument('--load-model-weights', type=str, default=None,
                    help='initialize model with pre-trained weights')
parser.add_argument('--exclude-model-weights', type=str, default=None,
                    help='comma-separated regex to exclude matched weights from being loaded, e.g., `a.fc..+,b.fc..+`')
parser.add_argument('--freeze-model-weights', type=str, default=None,
                    help='comma-separated regex to freeze matched weights from being updated in the training, e.g., `a.fc..+,b.fc..+`')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch-val', type=int, default=None,
                    help='number of steps (iterations) per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch', type=int, default=None,
                    help='number of samples per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch-val', type=int, default=None,
                    help='number of samples per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'radam', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--lr-scheduler', type=str, default='flat+decay',
                    choices=['none', 'steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle'],
                    help='learning rate scheduler')
parser.add_argument('--warmup-steps', type=int, default=0,
                    help='number of warm-up steps, only valid for `flat+linear` and `flat+cos` lr schedulers')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%%d_state.pt` and `epoch-%%d_optimizer.pt` files')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--predict-batch-size', type=int, default=-1,
                    help='Batch size used in testing/prediction (default: same as training batch size, set by args.batch_size).')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16)')
parser.add_argument('--gpus', type=str, default='',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--predict-gpus', type=str, default=None,
                    help='device for the testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`; if not set, use the same as `--gpus`')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.parquet` format')
parser.add_argument('--export-onnx', type=str, default=None,
                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--onnx-opset', type=int, default=15,
                    help='ONNX opset version.')
parser.add_argument('--io-test', action='store_true', default=False,
                    help='test throughput of the dataloader')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')
parser.add_argument('--log', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--print', action='store_true', default=False,
                    help='do not run training/prediction but only print model information, e.g., FLOPs and number of parameters of a model')
parser.add_argument('--profile', action='store_true', default=False,
                    help='run the profiler')
parser.add_argument('--profile-train', action='store_true', default=False,
                    help='profile a short slice of the real training loop (forward+backward+optimizer)')
parser.add_argument('--profile-steps', type=int, default=40,
                    help='number of training steps to capture when --profile-train is set')
parser.add_argument('--profile-dir', type=str, default=None,
                    help='where to write profiler traces; defaults to the model_prefix directory')
parser.add_argument('--profile-train-full', action='store_true', default=False,
                    help='profile the entire training run (all epochs/steps) and export a single trace')
parser.add_argument('--profile-train-full-active', type=int, default=10,
                    help='number of steps per profiler window when using --profile-train-full (smaller keeps memory down)')
parser.add_argument('--backend', type=str, choices=['gloo', 'nccl', 'mpi'], default=None,
                    help='backend for distributed training')
parser.add_argument('--cross-validation', type=str, default=None,
                    help='enable k-fold cross validation; input format: `variable_name%%k`')
parser.add_argument('--compile', action='store_true', default=False,
                    help='use torch.compile() to optimize the model for faster training (requires PyTorch 2.0+)')
parser.add_argument('--compile-mode', type=str, default='default',
                    choices=['default', 'reduce-overhead', 'max-autotune'],
                    help='torch.compile mode: default (balanced), reduce-overhead (faster compile), max-autotune (slower compile but faster runtime)')

# Data augmentation options
parser.add_argument('--augment', action='store_true', default=False,
                    help='enable data augmentation during training')
parser.add_argument('--aug-rotation', action='store_true', default=False,
                    help='enable azimuthal rotation augmentation')
parser.add_argument('--aug-reflection', action='store_true', default=False,
                    help='enable phi/eta reflection augmentation')
parser.add_argument('--aug-dropout', type=float, default=0.0,
                    help='fraction of particles to drop (0.0-1.0), biased toward low-pT particles')
parser.add_argument('--aug-reflection-prob', type=float, default=0.5,
                    help='probability to apply each reflection (phi and eta independently)')


def parse_file_patterns(file_patterns, local_rank=None, copy_inputs=False):
    """
    Get a valid file list from command line arguments
    Input arguments:
     - file_patterns: list of strings representing file patterns (potentially with wildcards).
       note: the file patterns may contain a ":" character as follows: <name>:<pattern>.
             the names provided in this way are the keys of the returned dict
             (the default name for patterns that do not have this structure is "", i.e. empty string).
       note: named file patterns may additionally contain a '%' character as follows: <name>%<split>:<pattern>.
             in this case, the files will be split in groups of <split> (integer).
             this is not supported for training/validation data, only for testing data!
     - local_rank: ?
     - copy_inputs: bool whether to copy the files to a local area.
    Returns:
     - file dict matching names (default name: "_") to corresponding files.
     - flat list of all files.
    """

    # make a dict matching names to lists of corresponding files
    # (files for which no name is provided on )
    file_dict = {}
    split_dict = {}
    for file_pattern in file_patterns:
        if ':' in file_pattern:
            name, file_pattern = file_pattern.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else: name = ''
        # find all files corresponding to pattern
        files = glob.glob(file_pattern)
        # append to dict
        if name in file_dict: file_dict[name] += files
        else: file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    # modify file dict based on local_rank
    # (what is this? probably irrelevant for now)
    if local_rank is not None:
        local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        new_file_dict = {}
        for name, files in file_dict.items():
            new_files = files[args.local_rank::local_world_size]
            assert(len(new_files) > 0)
            np.random.shuffle(new_files)
            new_file_dict[name] = new_files
        file_dict = new_file_dict

    # copy all input files to a temporary local directory
    if copy_inputs:
        # define a temporary directory
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir): shutil.rmtree(tmpdir)
        # copy all files
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip('/'))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info('Copied file %s to %s' % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error('Only %d/%d files copied for group %s',
                              len(new_file_dict[name]), len(files), name)
        file_dict = new_file_dict

    # check that all files are unique
    file_list = sum(file_dict.values(), [])
    assert(len(file_list) == len(set(file_list)))

    # return dict and list of files
    return file_dict, file_list


def train_load(args):
    """
    Loads the training data.
    Input arguments:
     - full set of command line args
    """

    # check if args.data_train was provided on the command line
    if args.data_train is None or len(args.data_train)==0:
        # this should not normally happen as this function is only called in training mode,
        # but add an explicit check anyway.
        raise Exception('Something went wrong: train_load(args) was called while no training data is provided.')

    # get the file patterns for training data in the case of a provided sample list
    print('HERE')
    print(len(args.data_train))
    if len(args.data_train)==1 and args.data_train[0].endswith('.yaml'):
        print('ALSO HERE')
        samplelist = args.data_train[0]
        _logger.info(f'Reading sample list {samplelist} for training data.')
        args.data_train = read_sample_list(samplelist)

    # get the files for training data
    train_file_dict, train_files = parse_file_patterns(args.data_train,
        copy_inputs=args.copy_inputs, local_rank=args.local_rank)

    # check if args.data_val was provided on the command line
    if args.data_val is None or len(args.data_val)==0:
        # use a fraction of the training data for validation
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)
    else:
        # get the file patterns for validation data in the case of a provided sample list
        if len(args.data_val)==1 and args.data_val[0].endswith('.yaml'):
            samplelist = args.data_val[0]
            _logger.info(f'Reading sample list {samplelist} for validation data.')
            args.data_val = read_sample_list(samplelist)
        # get the files for validation data
        val_file_dict, val_files = parse_file_patterns(args.data_val,
            copy_inputs=args.copy_inputs, local_rank=None)
        train_range = val_range = (0, 1)

    # print number of files for debugging
    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))
    
    # Log augmentation settings
    if args.augment:
        _logger.info('Data augmentation ENABLED:')
        _logger.info('  - Rotation: %s' % args.aug_rotation)
        _logger.info('  - Reflection: %s (prob=%.2f)' % (args.aug_reflection, args.aug_reflection_prob))
        _logger.info('  - Dropout: %.2f' % args.aug_dropout)
    else:
        _logger.info('Data augmentation DISABLED')

    # modify files and some data loading settings for small demo runs
    if args.demo:
        train_files = train_files[:20]
        val_files = val_files[:20]
        train_file_dict = {'_': train_files}
        val_file_dict = {'_': val_files}
        _logger.info(train_files)
        _logger.info(val_files)
        args.data_fraction = 0.1
        args.fetch_step = 0.002

    # check if correct arguments are set for in-memory running
    # (why must steps-per-epoch be set for in-memory running?)
    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')

    # make training data loader
    name = 'train' + ('' if args.local_rank is None else '_rank%d' % args.local_rank)
    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True,
                                   extra_selection=args.extra_selection,
                                   remake_weights=not args.no_remake_weights,
                                   load_range_and_fraction=(train_range, args.data_fraction),
                                   file_fraction=args.file_fraction,
                                   fetch_by_files=args.fetch_by_files,
                                   fetch_step=args.fetch_step,
                                   infinity_mode=args.steps_per_epoch is not None,
                                   in_memory=args.in_memory,
                                   name=name,
                                   # Data augmentation options (training only)
                                   augment=args.augment,
                                   aug_rotation=args.aug_rotation,
                                   aug_reflection=args.aug_reflection,
                                   aug_dropout=args.aug_dropout,
                                   aug_reflection_prob=args.aug_reflection_prob)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                     num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
                     persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None)

    # make validation data loader
    name = 'val' + ('' if args.local_rank is None else '_rank%d' % args.local_rank)
    val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True,
                                 extra_selection=args.extra_selection,
                                 load_range_and_fraction=(val_range, args.data_fraction),
                                 file_fraction=args.file_fraction,
                                 fetch_by_files=args.fetch_by_files,
                                 fetch_step=args.fetch_step,
                                 infinity_mode=args.steps_per_epoch_val is not None,
                                 in_memory=args.in_memory,
                                 name=name)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                   num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
                   persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None)
    
    # make a new reference to the data config
    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names

    # return data loaders and some other info
    return train_loader, val_loader, data_config, train_input_names, train_label_names


def test_load(args):
    """
    Loads the testing data.
    Input arguments:
     - full set of command line args
    """

    # check if args.data_test was provided on the command line
    if args.data_test is None:
        # this should not normally happen as this function is only called in testing mode,
        # but add an explicit check anyway.
        raise Exception('Something went wrong: test_load(args) was called while args.data_test is None.')

    # get the file patterns for testing data in the case of a provided sample list
    if len(args.data_test)==1 and args.data_test[0].endswith('.yaml'):
        samplelist = args.data_test[0]
        _logger.info(f'Reading sample list {samplelist} for testing data...')
        args.data_test = read_sample_list(samplelist)

    # get the files for testing data
    _logger.info(f'Finding files for testing data...')
    test_file_dict, test_files = parse_file_patterns(args.data_test,
        copy_inputs=args.copy_inputs, local_rank=None)
    _logger.info(f'Found following files:')
    _logger.info(test_file_dict)

    def get_test_loader(name):
        filelist = test_file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n  - %s', name, len(filelist),
                '\n  - '.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      extra_selection=args.extra_test_selection,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      #fetch_by_files=True, fetch_step=1,
                                      # update for inference on data
                                      # (since default approach tries to read the full data file)
                                      fetch_by_files=False, fetch_step=0.05,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.predict_batch_size,
                        drop_last=False, pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in test_file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config

def onnx(args):
    """
    Saving model as ONNX.
    :param args:
    :return:
    """
    assert (args.export_onnx.endswith('.onnx'))
    model_path = args.model_prefix
    _logger.info('Exporting model %s to ONNX' % model_path)

    from weaver.utils.dataset import DataConfig
    data_config = DataConfig.load(args.data_config, load_observers=False, load_reweight_info=False)
    model, model_info, _ = model_setup(args, data_config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.cpu()
    model.eval()

    if not os.path.dirname(args.export_onnx):
        args.export_onnx = os.path.join(os.path.dirname(model_path), args.export_onnx)
    os.makedirs(os.path.dirname(args.export_onnx), exist_ok=True)
    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32) for k in model_info['input_names'])
    torch.onnx.export(model, inputs, args.export_onnx,
                      input_names=model_info['input_names'],
                      output_names=model_info['output_names'],
                      dynamic_axes=model_info.get('dynamic_axes', None),
                      opset_version=args.onnx_opset)
    _logger.info('ONNX model saved to %s', args.export_onnx)

    preprocessing_json = os.path.join(os.path.dirname(args.export_onnx), 'preprocess.json')
    data_config.export_json(preprocessing_json)
    _logger.info('Preprocessing parameters saved to %s', preprocessing_json)


def flops(model, model_info, device='cpu'):
    """
    Count FLOPs and params.
    :param args:
    :param model:
    :param model_info:
    :return:
    """
    from weaver.utils.flops_counter import get_model_complexity_info
    import copy

    model = copy.deepcopy(model).to(device)
    model.eval()

    inputs = tuple(
        torch.ones(model_info['input_shapes'][k], dtype=torch.float32, device=device) for k in model_info['input_names'])

    macs, params = get_model_complexity_info(model, inputs, as_strings=True, print_per_layer_stat=True, verbose=True)
    _logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    _logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def profile(args, model, model_info, device):
    """
    Profile.
    :param model:
    :param model_info:
    :return:
    """
    import copy
    from torch.profiler import profile, record_function, ProfilerActivity

    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()

    inputs = tuple(
        torch.ones((args.batch_size,) + model_info['input_shapes'][k][1:],
                   dtype=torch.float32).to(device) for k in model_info['input_names'])
    for x in inputs:
        print(x.shape, x.device)

    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=2),
        on_trace_ready=trace_handler
    ) as p:
        for idx in range(100):
            model(*inputs)
            p.step()


def profile_training_loop(args, model, loss_func, opt, scheduler, train_loader, device, data_config, grad_scaler=None):
    """
    Profile a short slice of the real training loop (forward + backward + optimizer).
    Writes a Chrome trace to `profile_dir/profile_trace.json`.
    """
    import torch.profiler

    profile_dir = args.profile_dir
    if profile_dir is None or profile_dir == '':
        profile_dir = os.path.dirname(args.model_prefix) or '.'
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(profile_dir, 'profile_trace.json')

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    data_iter = iter(train_loader)
    model.train()
    steps = max(1, args.profile_steps)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True
    ) as prof:
        for step in range(steps):
            try:
                X, y, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                X, y, _ = next(data_iter)

            inputs = [X[k].to(device) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(device)
            mask = None
            mask_key = data_config.label_names[0] + '_mask'
            if mask_key in y:
                mask = y[mask_key].bool().to(device)

            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
                loss = loss_func(logits, label)

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            prof.step()

    prof.export_chrome_trace(trace_path)
    _logger.info('Profiler trace saved to %s', trace_path)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    _logger.info('Profiler summary (top 50 by self_cuda_time_total):\n%s', table)


def optim(args, model, device):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info('Optimizer options: %s' % str(optimizer_options))

    names_lr_mult = []
    if 'weight_decay' in optimizer_options or 'lr_mult' in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re
        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or (
                    hasattr(model, 'no_weight_decay') and name in model.no_weight_decay()):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if 'lr_mult' in optimizer_options:
            pattern, mult_factor = optimizer_options.pop('lr_mult')
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert(len(decay_1x) + len(decay_mult) == len(decay))
            assert(len(no_decay_1x) + len(no_decay_mult) == len(no_decay))
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop('weight_decay', 0.)
        parameters = [
            {'params': no_decay_1x, 'weight_decay': 0.},
            {'params': decay_1x, 'weight_decay': wd},
            {'params': no_decay_mult, 'weight_decay': 0., 'lr': args.start_lr * mult_factor},
            {'params': decay_mult, 'weight_decay': wd, 'lr': args.start_lr * mult_factor},
        ]
        _logger.info('Parameters excluded from weight decay:\n - %s', '\n - '.join(names_no_decay))
        if len(names_lr_mult):
            _logger.info('Parameters with lr multiplied by %s:\n - %s', mult_factor, '\n - '.join(names_lr_mult))
    else:
        parameters = model.parameters()

    if args.optimizer == 'ranger':
        from weaver.utils.nn.optimizer.ranger import Ranger
        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'adamW':
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == 'radam':
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info('Resume training from epoch %d' % args.load_epoch)
        model_state = torch.load(args.model_prefix + '_epoch-%d_state.pt' % args.load_epoch, map_location=device)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        opt_state_file = args.model_prefix + '_epoch-%d_optimizer.pt' % args.load_epoch
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            opt.load_state_dict(opt_state)
        else:
            _logger.warning('Optimizer state file %s NOT found!' % opt_state_file)

    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == 'steps':
            lr_step = round(args.num_epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=[lr_step, 2 * lr_step], gamma=0.1,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+decay':
            num_decay_epochs = max(1, int(args.num_epochs * 0.3))
            milestones = list(range(args.num_epochs - num_decay_epochs, args.num_epochs))
            gamma = 0.01 ** (1. / num_decay_epochs)
            if len(names_lr_mult):
                def get_lr(epoch): return gamma ** max(0, epoch - milestones[0] + 1)  # noqa
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt, (lambda _: 1, lambda _: 1, get_lr, get_lr),
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch, verbose=True)
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=milestones, gamma=gamma,
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch)
        elif args.lr_scheduler == 'flat+linear' or args.lr_scheduler == 'flat+cos':
            total_steps = args.num_epochs * args.steps_per_epoch
            warmup_steps = args.warmup_steps
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001

            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps))
                if step_num < warmup_steps:
                    return 1. * step_num / warmup_steps
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == 'flat+linear':
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_fn, last_epoch=-1 if args.load_epoch is None else args.load_epoch * args.steps_per_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
        elif args.lr_scheduler == 'one-cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.start_lr, epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch, pct_start=0.3,
                anneal_strategy='cos', div_factor=25.0, last_epoch=-1 if args.load_epoch is None else args.load_epoch)
            scheduler._update_per_step = True  # mark it to update the lr every step, instead of every epoch
    return opt, scheduler


def model_setup(args, data_config, device='cpu'):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.export_onnx:
        network_options['for_inference'] = True
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    if args.load_model_weights:
        model_state = torch.load(args.load_model_weights, map_location='cpu')
        if args.exclude_model_weights:
            import re
            exclude_patterns = args.exclude_model_weights.split(',')
            _logger.info('The following weights will not be loaded: %s' % str(exclude_patterns))
            key_state = {}
            for k in model_state.keys():
                key_state[k] = True
                for pattern in exclude_patterns:
                    if re.match(pattern, k):
                        key_state[k] = False
                        break
            model_state = {k: v for k, v in model_state.items() if key_state[k]}
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info('Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s' %
                     (args.load_model_weights, missing_keys, unexpected_keys))
        if args.freeze_model_weights:
            import re
            freeze_patterns = args.freeze_model_weights.split(',')
            for name, param in model.named_parameters():
                freeze = False
                for pattern in freeze_patterns:
                    if re.match(pattern, name):
                        freeze = True
                        break
                if freeze:
                    param.requires_grad = False
            _logger.info('The following weights has been frozen:\n - %s',
                         '\n - '.join([name for name, p in model.named_parameters() if not p.requires_grad]))
    # _logger.info(model)
    flops(model, model_info, device=device)
    # loss function
    try:
        loss_func = network_module.get_loss(data_config, **network_options)
        _logger.info('Using loss function %s with options %s' % (loss_func, network_options))
    except AttributeError:
        loss_func = torch.nn.CrossEntropyLoss()
        _logger.warning('Loss function not defined in %s. Will use `torch.nn.CrossEntropyLoss()` by default.',
                        args.network_config)
    return model, model_info, loss_func


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from weaver.utils.data.tools import _concat
    _logger.info('Start running IO test')
    monitor_info = defaultdict(list)

    for X, y, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v)
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = 'weaver_monitor_info.parquet'
        try:
            import awkward as ak
            ak.to_parquet(ak.Array(monitor_info), monitor_output_path, compression='LZ4', compression_level=4)
            _logger.info('Monitor info written to %s' % monitor_output_path, color='bold')
        except Exception as e:
            _logger.error('Error when writing output parquet file: \n' + str(e))


def save_root(args, output_path, data_config, scores, labels, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param labels
    :param observers
    :return:
    """
    import awkward as ak
    from weaver.utils.data.fileio import _write_root
    output = {}
    if data_config.label_type == 'simple':
        for idx, label_name in enumerate(data_config.label_value):
            output[label_name] = (labels[data_config.label_names[0]] == idx)
            output['score_' + label_name] = scores[:, idx]
    else:
        if scores.ndim <= 2:
            output['output'] = scores
        elif scores.ndim == 3:
            num_classes = len(scores[0, 0, :])
            try:
                names = data_config.labels['names']
                assert (len(names) == num_classes)
            except KeyError:
                names = [f'class_{idx}' for idx in range(num_classes)]
            for idx, label_name in enumerate(names):
                output[label_name] = (labels[data_config.label_names[0]] == idx)
                output['score_' + label_name] = scores[:, :, idx]
        else:
            output['output'] = scores
    output.update(labels)
    output.update(observers)

    try:
        _write_root(output_path, ak.Array(output))
        _logger.info('Written output to %s' % output_path, color='bold')
    except Exception as e:
        _logger.error('Error when writing output ROOT file: \n' + str(e))

    save_as_parquet = any(v.ndim > 2 for v in output.values())
    if save_as_parquet:
        try:
            ak.to_parquet(
                ak.Array(output),
                output_path.replace('.root', '.parquet'),
                compression='LZ4', compression_level=4)
            _logger.info('Written alternative output file to %s' %
                         output_path.replace('.root', '.parquet'), color='bold')
        except Exception as e:
            _logger.error('Error when writing output parquet file: \n' + str(e))


def save_parquet(args, output_path, scores, labels, observers):
    """
    Saves as parquet file
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    import awkward as ak
    output = {'scores': scores}
    output.update(labels)
    output.update(observers)
    try:
        ak.to_parquet(ak.Array(output), output_path, compression='LZ4', compression_level=4)
        _logger.info('Written output to %s' % output_path, color='bold')
    except Exception as e:
        _logger.error('Error when writing output parquet file: \n' + str(e))


def _main(args):
    _logger.info('args:\n - %s', '\n - '.join(str(it) for it in args.__dict__.items()))

    # export to ONNX
    if args.export_onnx:
        onnx(args)
        return

    if args.file_fraction < 1:
        _logger.warning('Use of `file-fraction` is not recommended in general -- prefer using `data-fraction` instead.')

    # classification/regression mode
    if args.regression_mode:
        _logger.info('Running in regression mode')
        from weaver.utils.nn.tools import train_regression as train
        from weaver.utils.nn.tools import evaluate_regression as evaluate
    else:
        _logger.info('Running in classification mode')
        from weaver.utils.nn.tools import train_classification as train
        from weaver.utils.nn.tools import evaluate_classification as evaluate

    # training/testing mode
    training_mode = not args.predict

    # device
    if args.gpus:
        # distributed training
        if args.backend is not None:
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            dev = torch.device(local_rank)
            torch.distributed.init_process_group(backend=args.backend)
            _logger.info(f'Using distributed PyTorch with {args.backend} backend')
        else:
            gpus = [int(i) for i in args.gpus.split(',')]
            dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')
        try:
            if torch.backends.mps.is_available():
                dev = torch.device('mps')
        except AttributeError:
            pass

    # load data
    if training_mode:
        train_loader, val_loader, data_config, train_input_names, train_label_names = train_load(args)
    else:
        test_loaders, data_config = test_load(args)

    if args.io_test:
        data_loader = train_loader if training_mode else list(test_loaders.values())[0]()
        iotest(args, data_loader)
        return

    model, model_info, loss_func = model_setup(args, data_config, device=dev)

    # TODO: load checkpoint
    # if args.backend is not None:
    #     load_checkpoint()

    if args.print:
        return

    if args.profile:
        profile(args, model, model_info, device=dev)
        return

    if args.profile_train:
        # Profile a short slice of the real training loop then exit.
        model = model.to(dev)
        if args.backend is not None:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=local_rank, find_unused_parameters=True)
        elif gpus is not None and len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpus)
        grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        opt, scheduler = optim(args, model, dev)
        profile_training_loop(args, model, loss_func, opt, scheduler, train_loader, dev, data_config, grad_scaler=grad_scaler)
        return

    if args.tensorboard:
        from weaver.utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    # Initialize wandb if enabled
    if args.use_wandb:
        from weaver.utils.nn.tools import WandbHelper
        wandb_config = {
            'model_config': args.network_config,
            'data_config': args.data_config,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.start_lr,
            'optimizer': args.optimizer,
            'scheduler': args.lr_scheduler,
            'regression_mode': args.regression_mode,
        }
        wb = WandbHelper(
            config=wandb_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes
        )
    else:
        wb = None

    # note: we should always save/load the state_dict of the original model, not the one wrapped by nn.DataParallel
    # so we do not convert it to nn.DataParallel now
    orig_model = model

    if training_mode:
        profiler = None
        profile_dir = args.profile_dir
        if profile_dir is None or profile_dir == '':
            profile_dir = os.path.dirname(args.model_prefix) or '.'

        if args.profile_train_full:
            os.makedirs(profile_dir, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if dev.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            trace_idx = {'i': 0}

            def trace_handler(p):
                trace_path = os.path.join(profile_dir, f'profile_train_full_{trace_idx["i"]}.json')
                p.export_chrome_trace(trace_path)
                trace_idx['i'] += 1

            active_len = max(1, args.profile_train_full_active)
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=active_len, repeat=1_000_000),
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
            )
            profiler.__enter__()

        model = orig_model.to(dev)

        # Apply torch.compile for faster training (PyTorch 2.0+)
        if args.compile:
            _logger.info('Compiling model with torch.compile (mode=%s)...' % args.compile_mode)
            model = torch.compile(model, mode=args.compile_mode)
            _logger.info('Model compiled successfully')

        # DistributedDataParallel
        if args.backend is not None:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus, output_device=local_rank, find_unused_parameters=True)

        # optimizer & learning rate
        opt, scheduler = optim(args, model, dev)

        # Watch model with wandb if enabled
        if wb is not None:
            wb.watch_model(model)

        # DataParallel
        if args.backend is None:
            if gpus is not None and len(gpus) > 1:
                # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
                model = torch.nn.DataParallel(model, device_ids=gpus)
            # model = model.to(dev)

        # lr finder: keep it after all other setups
        if args.lr_finder is not None:
            start_lr, end_lr, num_iter = args.lr_finder.replace(' ', '').split(',')
            from weaver.utils.lr_finder import LRFinder
            lr_finder = LRFinder(model, opt, loss_func, device=dev, input_names=train_input_names,
                                 label_names=train_label_names)
            lr_finder.range_test(train_loader, start_lr=float(start_lr), end_lr=float(end_lr), num_iter=int(num_iter))
            lr_finder.plot(output='lr_finder.png')  # to inspect the loss-learning rate graph
            if profiler is not None:
                profiler.__exit__(None, None, None)
            return

        # training loop
        best_valid_metric = np.inf if args.regression_mode else 0
        grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        try:
            for epoch in range(args.num_epochs):
                if args.load_epoch is not None:
                    if epoch <= args.load_epoch:
                        continue
                _logger.info('-' * 50)
                _logger.info('Epoch #%d training' % epoch)
                train(model, loss_func, opt, scheduler, train_loader, dev, epoch,
                      steps_per_epoch=args.steps_per_epoch, grad_scaler=grad_scaler, tb_helper=tb, wandb_helper=wb, profiler=profiler)
                if args.model_prefix and (args.backend is None or local_rank == 0):
                    dirname = os.path.dirname(args.model_prefix)
                    if dirname and not os.path.exists(dirname):
                        os.makedirs(dirname)
                    # Get state_dict from the original model to avoid torch.compile's _orig_mod prefix
                    # and DataParallel/DistributedDataParallel's module prefix
                    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                        state_dict = model.module.state_dict()
                    elif args.compile:
                        # For compiled models, save from orig_model which has correct keys
                        state_dict = orig_model.state_dict()
                    else:
                        state_dict = model.state_dict()
                    torch.save(state_dict, args.model_prefix + '_epoch-%d_state.pt' % epoch)
                    torch.save(opt.state_dict(), args.model_prefix + '_epoch-%d_optimizer.pt' % epoch)
                # if args.backend is not None and local_rank == 0:
                # TODO: save checkpoint
                #     save_checkpoint()

                _logger.info('Epoch #%d validating' % epoch)
                valid_metric = evaluate(model, val_loader, dev, epoch, loss_func=loss_func,
                                        steps_per_epoch=args.steps_per_epoch_val, tb_helper=tb, wandb_helper=wb)
                is_best_epoch = (
                    valid_metric < best_valid_metric) if args.regression_mode else(
                    valid_metric > best_valid_metric)
                if is_best_epoch:
                    best_valid_metric = valid_metric
                    if args.model_prefix and (args.backend is None or local_rank == 0):
                        shutil.copy2(args.model_prefix + '_epoch-%d_state.pt' %
                                     epoch, args.model_prefix + '_best_epoch_state.pt')
                        # torch.save(model, args.model_prefix + '_best_epoch_full.pt')
                _logger.info('Epoch #%d: Current validation metric: %.5f (best: %.5f)' %
                             (epoch, valid_metric, best_valid_metric), color='bold')
        finally:
            if args.profile_train_full and profiler is not None:
                profiler.__exit__(None, None, None)
                table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
                _logger.info('Profiler summary (top 50 by self_cuda_time_total):\n%s', table)

    if args.data_test:
        if args.backend is not None and local_rank != 0:
            return
        if training_mode:
            del train_loader, val_loader
            test_loaders, data_config = test_load(args)

        if not args.model_prefix.endswith('.onnx'):
            if args.predict_gpus:
                gpus = [int(i) for i in args.predict_gpus.split(',')]
                dev = torch.device(gpus[0])
            else:
                gpus = None
                dev = torch.device('cpu')
                try:
                    if torch.backends.mps.is_available():
                        dev = torch.device('mps')
                except AttributeError:
                    pass
            model = orig_model.to(dev)
            model_path = args.model_prefix if args.model_prefix.endswith(
                '.pt') else args.model_prefix + '_best_epoch_state.pt'
            _logger.info('Loading model %s for eval' % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
            if gpus is not None and len(gpus) > 1:
                model = torch.nn.DataParallel(model, device_ids=gpus)
            model = model.to(dev)

        for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()
            # run prediction
            if args.model_prefix.endswith('.onnx'):
                _logger.info('Loading model %s for eval' % args.model_prefix)
                from weaver.utils.nn.tools import evaluate_onnx
                test_metric, scores, labels, observers = evaluate_onnx(args.model_prefix, test_loader)
            else:
                test_metric, scores, labels, observers = evaluate(
                    model, test_loader, dev, epoch=None, for_training=False, tb_helper=tb)
            _logger.info('Test metric %.5f' % test_metric, color='bold')
            del test_loader

            if args.predict_output:
                if not os.path.dirname(args.predict_output):
                    predict_output = os.path.join(
                        os.path.dirname(args.model_prefix),
                        'predict_output', args.predict_output)
                else:
                    predict_output = args.predict_output
                os.makedirs(os.path.dirname(predict_output), exist_ok=True)
                if name == '':
                    output_path = predict_output
                else:
                    base, ext = os.path.splitext(predict_output)
                    output_path = base + '_' + name + ext
                if output_path.endswith('.root'):
                    save_root(args, output_path, data_config, scores, labels, observers)
                else:
                    save_parquet(args, output_path, scores, labels, observers)


def main():
    
       # parse command line args
    args = parser.parse_args()

    # set number of instances ('samples') or number of batches ('steps') per epoch for training and validation
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            msg = 'Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!'
            raise RuntimeError(msg)
    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            msg = 'Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!'
            raise RuntimeError(msg)

    # set batch size for testing
    if args.predict_batch_size <= 0: args.predict_batch_size = args.batch_size

    # set number of batches ('steps') per epoch for validation
    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(args.steps_per_epoch
          * (1 - args.train_val_split) / args.train_val_split)
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None

    # make auto-generated model name
    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%Y%m%d-%H%M%S') + "_" + os.path.basename(args.network_config).replace('.py', '')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)

    if args.predict_gpus is None:
        args.predict_gpus = args.gpus

    args.local_rank = None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))

    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += '.%03d' % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger('weaver', stdout=stdout, filename=args.log)

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        if args.predict_output:
            predict_output_base, predict_output_ext = os.path.splitext(args.predict_output)
        load_model = args.load_model_weights or None
        var_name, kfold = args.cross_validation.split('%')
        kfold = int(kfold)
        for i in range(kfold):
            _logger.info(f'\n=== Running cross validation, fold {i} of {kfold} ===')
            opts = copy.deepcopy(args)
            opts.model_prefix = os.path.join(f'{model_dir}_fold{i}', model_fn)
            if args.predict_output:
                opts.predict_output = f'{predict_output_base}_fold{i}' + predict_output_ext
            opts.extra_selection = f'{var_name}%{kfold}!={i}'
            opts.extra_test_selection = f'{var_name}%{kfold}=={i}'
            if load_model and '{fold}' in load_model:
                opts.load_model_weights = load_model.replace('{fold}', f'fold{i}')

            _main(opts)
    else:
        _main(args)


if __name__ == '__main__':
    main()
