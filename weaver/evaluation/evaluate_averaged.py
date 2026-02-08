#!/usr/bin/env python
"""
Averaged evaluation script for multi-seed training.
Loads results from multiple seed runs, computes averaged ROC curves and AUC with uncertainty.
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from tools import read_file
from plot_roc_multi import compute_roc_curve


def load_events_from_file(rootfile, treename='Events'):
    """Load events from a ROOT file with standard branch names."""
    signal_categories = {
        'b': {
            'label_branch': 'recojet_isB',
            'score_branch': 'score_recojet_isB',
            'color': 'red',
            'label': r'b'
        },
        'c': {
            'label_branch': 'recojet_isC',
            'score_branch': 'score_recojet_isC',
            'color': 'blue',
            'label': r'c'
        },
    }
    background_categories = {
        'udsg': {
            'label_branch': 'recojet_isUDSG',
            'score_branch': 'score_recojet_isUDSG',
            'color': 'green',
            'label': r'udsg'
        }
    }
    all_categories = {**signal_categories, **background_categories}
    
    branches_to_read = (
        [cat['label_branch'] for cat in all_categories.values()]
        + [cat['score_branch'] for cat in all_categories.values()]
    )
    
    events = read_file(rootfile, treename=treename, branches=branches_to_read)
    return events, signal_categories, background_categories, all_categories


def compute_all_roc_curves(events, all_categories):
    """Compute ROC curves for all pairwise category comparisons."""
    # Get masks for each category
    masks = {}
    for cat_name, cat_settings in all_categories.items():
        branch = cat_settings['label_branch']
        masks[cat_name] = events[branch].astype(bool)
    
    # Generate all pairs
    cat_names = list(all_categories.keys())
    pairs = []
    for i, sig_name in enumerate(cat_names):
        for j, bkg_name in enumerate(cat_names):
            if j > i:
                pairs.append((sig_name, bkg_name))
    
    # Compute ROC for each pair
    roc_results = {}
    for sig_name, bkg_name in pairs:
        sig_settings = all_categories[sig_name]
        bkg_settings = all_categories[bkg_name]
        
        sig_score_branch = sig_settings['score_branch']
        
        scores_sig = events[sig_score_branch][masks[sig_name]]
        scores_bkg = events[sig_score_branch][masks[bkg_name]]
        
        if len(scores_sig) == 0 or len(scores_bkg) == 0:
            continue
        
        eff_sig, eff_bkg, auc = compute_roc_curve(scores_sig, scores_bkg)
        
        pair_key = f"{sig_name}_vs_{bkg_name}"
        roc_results[pair_key] = {
            'eff_sig': eff_sig,
            'eff_bkg': eff_bkg,
            'auc': auc,
            'label': f"{sig_settings['label']} vs. {bkg_settings['label']}"
        }
    
    return roc_results


def interpolate_roc_curve(eff_bkg, eff_sig, fpr_points):
    """Interpolate ROC curve to common FPR points."""
    # Sort by eff_bkg (FPR) to ensure monotonicity for interpolation
    sort_idx = np.argsort(eff_bkg)
    eff_bkg_sorted = eff_bkg[sort_idx]
    eff_sig_sorted = eff_sig[sort_idx]
    
    # Remove duplicate FPR values (take max TPR for each)
    unique_fpr, unique_idx = np.unique(eff_bkg_sorted, return_index=True)
    unique_tpr = eff_sig_sorted[unique_idx]
    
    # Interpolate
    if len(unique_fpr) < 2:
        return np.ones_like(fpr_points) * 0.5
    
    interp_func = interpolate.interp1d(
        unique_fpr, unique_tpr, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(0, 1)
    )
    
    return interp_func(fpr_points)


def average_roc_curves(all_roc_data, fpr_points=None):
    """Average ROC curves across seeds and compute statistics."""
    if fpr_points is None:
        fpr_points = np.logspace(-6, 0, 500)
    
    # Group by pair key
    pair_keys = list(all_roc_data[0].keys())
    
    averaged_results = {}
    
    for pair_key in pair_keys:
        # Collect interpolated TPR for each seed
        tpr_values = []
        auc_values = []
        
        for seed_data in all_roc_data:
            if pair_key not in seed_data:
                continue
            
            roc = seed_data[pair_key]
            tpr_interp = interpolate_roc_curve(roc['eff_bkg'], roc['eff_sig'], fpr_points)
            tpr_values.append(tpr_interp)
            auc_values.append(roc['auc'])
        
        if len(tpr_values) == 0:
            continue
        
        tpr_array = np.array(tpr_values)
        
        # Compute mean and std
        tpr_mean = np.mean(tpr_array, axis=0)
        tpr_std = np.std(tpr_array, axis=0)
        auc_mean = np.mean(auc_values)
        auc_std = np.std(auc_values)
        
        averaged_results[pair_key] = {
            'fpr': fpr_points,
            'tpr_mean': tpr_mean,
            'tpr_std': tpr_std,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'auc_values': auc_values,
            'label': all_roc_data[0][pair_key]['label'],
            'n_seeds': len(tpr_values)
        }
    
    return averaged_results


def plot_averaged_roc(averaged_results, outputdir, colormap='cool'):
    """Plot averaged ROC curves with error bands."""
    fig_linear, ax_linear = plt.subplots(figsize=(10, 7))
    fig_log, ax_log = plt.subplots(figsize=(10, 7))
    
    n_curves = len(averaged_results)
    cmap = plt.get_cmap(colormap, max(n_curves, 1))
    
    for idx, (pair_key, data) in enumerate(averaged_results.items()):
        color = cmap(idx)
        fpr = data['fpr']
        tpr_mean = data['tpr_mean']
        tpr_std = data['tpr_std']
        
        label = f"{data['label']} (AUC: {data['auc_mean']:.3f} ± {data['auc_std']:.3f})"
        
        # Linear plot
        ax_linear.plot(fpr, tpr_mean, color=color, linewidth=2, label=label)
        ax_linear.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std, 
                               color=color, alpha=0.2)
        
        # Log plot
        ax_log.plot(fpr, tpr_mean, color=color, linewidth=2, label=label)
        ax_log.fill_between(fpr, tpr_mean - tpr_std, tpr_mean + tpr_std,
                           color=color, alpha=0.2)
    
    # Diagonal reference line
    ax_linear.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    ax_log.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    
    # Configure linear plot
    ax_linear.set_xlabel('Background pass-through (FPR)', fontsize=12)
    ax_linear.set_ylabel('Signal efficiency (TPR)', fontsize=12)
    ax_linear.set_title('Averaged ROC Curves (Mean ± Std)', fontsize=14)
    ax_linear.legend(loc='lower right', fontsize=9)
    ax_linear.grid(True, alpha=0.3)
    ax_linear.set_xlim(0, 1)
    ax_linear.set_ylim(0, 1)
    fig_linear.tight_layout()
    
    # Configure log plot
    ax_log.set_xlabel('Background pass-through (FPR)', fontsize=12)
    ax_log.set_ylabel('Signal efficiency (TPR)', fontsize=12)
    ax_log.set_title('Averaged ROC Curves - Log Scale (Mean ± Std)', fontsize=14)
    ax_log.legend(loc='lower right', fontsize=9)
    ax_log.grid(True, which='both', alpha=0.3)
    ax_log.set_xscale('log')
    ax_log.set_xlim(1e-5, 1)
    ax_log.set_ylim(0, 1)
    fig_log.tight_layout()
    
    # Save figures
    os.makedirs(outputdir, exist_ok=True)
    
    fig_linear.savefig(os.path.join(outputdir, 'roc_averaged.png'), dpi=150)
    fig_log.savefig(os.path.join(outputdir, 'roc_averaged_log.png'), dpi=150)
    
    print(f'Saved: {os.path.join(outputdir, "roc_averaged.png")}')
    print(f'Saved: {os.path.join(outputdir, "roc_averaged_log.png")}')
    
    plt.close(fig_linear)
    plt.close(fig_log)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and average results from multi-seed training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-i', '--inputfiles', nargs='+', default=None,
                        help='List of output.root files to average')
    parser.add_argument('-m', '--manifest', type=str, default=None,
                        help='Path to manifest.json from multi-seed training')
    parser.add_argument('-o', '--outputdir', type=str, default=None,
                        help='Output directory for averaged evaluation results')
    
    args = parser.parse_args()
    
    # Determine input files
    if args.manifest:
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
        
        input_files = [run['output_root'] for run in manifest['runs']]
        
        # Default output directory next to manifest
        if args.outputdir is None:
            args.outputdir = os.path.join(
                os.path.dirname(args.manifest), 
                'averaged_evaluation'
            )
    elif args.inputfiles:
        input_files = args.inputfiles
        if args.outputdir is None:
            args.outputdir = 'averaged_evaluation'
    else:
        parser.error('Either --inputfiles or --manifest must be provided')
    
    # Validate files exist
    existing_files = []
    for f in input_files:
        if os.path.exists(f):
            existing_files.append(f)
        else:
            print(f'Warning: File not found: {f}')
    
    if len(existing_files) == 0:
        print('Error: No valid input files found')
        sys.exit(1)
    
    print(f'Found {len(existing_files)} input files')
    print('=' * 80)
    
    # Load and compute ROC for each file
    all_roc_data = []
    
    for i, inputfile in enumerate(existing_files):
        print(f'[{i+1}/{len(existing_files)}] Processing: {inputfile}')
        
        events, sig_cats, bkg_cats, all_cats = load_events_from_file(inputfile)
        nevents = len(events[list(events.keys())[0]])
        print(f'  Loaded {nevents} events')
        
        roc_results = compute_all_roc_curves(events, all_cats)
        all_roc_data.append(roc_results)
        
        # Print individual AUC values
        for pair_key, roc in roc_results.items():
            print(f"  {roc['label']}: AUC = {roc['auc']:.4f}")
    
    print('\n' + '=' * 80)
    print('Computing averaged metrics...')
    
    # Average ROC curves
    averaged_results = average_roc_curves(all_roc_data)
    
    # Print summary
    print('\nAveraged Results:')
    summary = {}
    for pair_key, data in averaged_results.items():
        print(f"  {data['label']}: AUC = {data['auc_mean']:.4f} ± {data['auc_std']:.4f} (n={data['n_seeds']})")
        summary[pair_key] = {
            'label': data['label'],
            'auc_mean': data['auc_mean'],
            'auc_std': data['auc_std'],
            'auc_values': data['auc_values'],
            'n_seeds': data['n_seeds']
        }
    
    # Plot averaged ROC curves
    print(f'\nSaving plots to: {args.outputdir}')
    plot_averaged_roc(averaged_results, args.outputdir)
    
    # Save summary JSON
    summary_path = os.path.join(args.outputdir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary: {summary_path}')
    
    print('\nAveraged evaluation complete!')


if __name__ == '__main__':
    main()
