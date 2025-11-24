#!/usr/bin/env python
"""
Interpretability analysis for ParticleTransformer jet flavor tagging models.

This module provides four methods to understand model decisions:
1. Gradient-based feature attribution
2. Permutation importance
3. Feature group ablation
4. Attention weight statistics
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import copy

# Add weaver to path
thisdir = os.path.abspath(os.path.dirname(__file__))
weaverdir = os.path.abspath(os.path.join(thisdir, '../'))
sys.path.insert(0, weaverdir)

from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from weaver.utils.data.tools import _concat
from weaver.utils.samplelisttools import read_sample_list
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


# Feature groups based on physics meaning
FEATURE_GROUPS = {
    'kinematic': ['pfcand_pt_log', 'pfcand_e_log', 'pfcand_ptrel_log', 'pfcand_erel_log', 'pfcand_drrel'],
    'impact_parameters': ['pfcand_dxy', 'pfcand_dz', 'pfcand_btagSip2dVal', 'pfcand_btagSip2dSig', 
                          'pfcand_btagSip3dVal', 'pfcand_btagSip3dSig'],
    'jet_distance': ['pfcand_btagJetDistVal', 'pfcand_btagJetDistSig'],
    'particle_id': ['pfcand_isChargedHad', 'pfcand_isNeutralHad', 'pfcand_isGamma', 'pfcand_isEl', 'pfcand_isMu'],
    'angular': ['pfcand_thetarel', 'pfcand_phirel'],
    'charge': ['pfcand_charge'],
}


def load_model_and_data(args):
    """Load trained model, data config, and test dataset."""
    print("Loading model and data...")
    
    # Load data config
    from weaver.utils.dataset import DataConfig
    data_config = DataConfig.load(args.data_config, load_observers=True, load_reweight_info=False)
    
    # Load model
    network_module = import_module(args.network_config, name='_network_module')
    model, model_info = network_module.get_model(data_config)
    
    # Load weights
    model_state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(model_state)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load test data - handle YAML sample lists
    import glob
    
    # Check if test_data is a YAML sample list
    if len(args.test_data) == 1 and args.test_data[0].endswith('.yaml'):
        print(f"Reading sample list from {args.test_data[0]}")
        test_patterns = read_sample_list(args.test_data[0])
    else:
        test_patterns = args.test_data
    
    # Expand file patterns
    test_files = []
    for pattern in test_patterns:
        expanded = glob.glob(pattern)
        test_files.extend(expanded)
        print(f"Pattern '{pattern}' matched {len(expanded)} files")
    
    if len(test_files) == 0:
        raise ValueError(f"No test files found matching patterns: {test_patterns}")
    
    print(f"Found {len(test_files)} test files total")
    
    test_data = SimpleIterDataset(
        {'test': test_files}, 
        args.data_config, 
        for_training=False,
        fetch_by_files=False, 
        fetch_step=0.05,
        name='test'
    )
    
    test_loader = DataLoader(
        test_data, 
        num_workers=min(4, len(test_files)), 
        batch_size=args.batch_size,
        drop_last=False, 
        pin_memory=True
    )
    
    return model, model_info, data_config, test_loader, device



def get_predictions(model, data_loader, data_config, device, max_jets=None):
    """Get model predictions and collect data for analysis."""
    print("Getting model predictions...")
    
    all_scores = []
    all_labels = defaultdict(list)
    all_inputs = defaultdict(list)
    
    num_jets = 0
    
    with torch.no_grad():
        for X, y, Z in tqdm(data_loader, desc="Processing batches"):
            # Move inputs to device
            # Model expects: forward(points, features, lorentz_vectors, mask)
            pf_points = X['pf_points'].to(device)
            pf_features = X['pf_features'].to(device)
            pf_vectors = X['pf_vectors'].to(device)
            pf_mask = X['pf_mask'].to(device)
            
            # Get predictions
            outputs = model(pf_points, pf_features, pf_vectors, pf_mask)
            scores = torch.softmax(outputs, dim=1) if outputs.shape[1] > 1 else outputs
            
            # Store results
            all_scores.append(scores.cpu().numpy())
            for k in data_config.label_names:
                all_labels[k].append(y[k])
            for k in data_config.input_names:
                all_inputs[k].append(X[k].cpu().numpy())
            
            num_jets += scores.shape[0]
            if max_jets and num_jets >= max_jets:
                break
    
    # Concatenate results
    scores = np.concatenate(all_scores, axis=0)
    labels = {k: _concat(v) for k, v in all_labels.items()}
    inputs = {k: np.concatenate(v, axis=0) for k, v in all_inputs.items()}
    
    if max_jets:
        scores = scores[:max_jets]
        labels = {k: v[:max_jets] for k, v in labels.items()}
        inputs = {k: v[:max_jets] for k, v in inputs.items()}
    
    print(f"Collected predictions for {scores.shape[0]} jets")
    return scores, labels, inputs



def compute_gradient_attribution(model, data_loader, data_config, device, output_dir, max_jets=None, gradient_batch_size=None):
    """
    Compute gradient-based feature attribution.
    
    For each class, compute ∂(score_class)/∂(each feature) and aggregate statistics.
    """
    print("\n" + "="*80)
    print("Computing Gradient-Based Feature Attribution")
    print("="*80)
    
    model.train()  # Need gradients
    
    # Get feature names from pf_features
    feature_names = data_config.input_dicts['pf_features']
    num_classes = len(data_config.label_value)
    class_names = data_config.label_value
    
    # Store gradients for each class
    gradients_per_class = {cls: [] for cls in class_names}
    
    num_jets = 0
    
    # If gradient_batch_size is specified and different from data_loader batch size,
    # create a new data loader with smaller batches
    if gradient_batch_size is not None and gradient_batch_size != data_loader.batch_size:
        print(f"Creating new data loader with batch size {gradient_batch_size} for gradient computation")
        from torch.utils.data import DataLoader
        gradient_loader = DataLoader(
            data_loader.dataset,
            batch_size=gradient_batch_size,
            num_workers=data_loader.num_workers,
            drop_last=False,
            pin_memory=True
        )
    else:
        gradient_loader = data_loader
    
    for X, y, Z in tqdm(gradient_loader, desc="Computing gradients"):
        # Move inputs to device and enable gradients
        # Map data config names to model forward arguments
        # Model expects: forward(points, features, lorentz_vectors, mask)
        pf_points = X['pf_points'].to(device)
        pf_features = X['pf_features'].to(device)
        pf_vectors = X['pf_vectors'].to(device)
        pf_mask = X['pf_mask'].to(device)
        
        # Only compute gradients for pf_features
        pf_features.requires_grad = True
        
        # Forward pass
        outputs = model(pf_points, pf_features, pf_vectors, pf_mask)
        scores = torch.softmax(outputs, dim=1)
        
        # Compute gradients for each class separately to save memory
        for class_idx, class_name in enumerate(class_names):
            # Clear gradients
            if pf_features.grad is not None:
                pf_features.grad.zero_()
            model.zero_grad()
            
            # Backward for this class
            class_scores = scores[:, class_idx]
            class_scores.sum().backward(retain_graph=(class_idx < num_classes - 1))
            
            # Get gradients (batch, num_features, num_particles)
            # Check if gradients were computed (can be None if no gradient flow)
            if pf_features.grad is None:
                print(f"Warning: No gradients computed for class {class_name} in this batch, skipping...")
                continue
            
            grads = pf_features.grad.detach().cpu().numpy()
            
            # Average over particles dimension, take absolute value
            # Shape: (batch, num_features)
            grads_avg = np.abs(grads).mean(axis=2)
            
            gradients_per_class[class_name].append(grads_avg)
            
            # Clear CUDA cache to free memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        num_jets += scores.shape[0]
        if max_jets and num_jets >= max_jets:
            break

    
    # Aggregate gradients
    results = {}
    for class_name in class_names:
        grads = np.concatenate(gradients_per_class[class_name], axis=0)
        if max_jets:
            grads = grads[:max_jets]
        
        # Compute statistics
        results[class_name] = {
            'mean': grads.mean(axis=0),
            'std': grads.std(axis=0),
            'median': np.median(grads, axis=0),
            'p25': np.percentile(grads, 25, axis=0),
            'p75': np.percentile(grads, 75, axis=0),
        }
    
    # Save results to CSV
    df_list = []
    for class_name in class_names:
        df = pd.DataFrame({
            'feature': feature_names,
            'class': class_name,
            'mean_gradient': results[class_name]['mean'],
            'std_gradient': results[class_name]['std'],
            'median_gradient': results[class_name]['median'],
            'p25_gradient': results[class_name]['p25'],
            'p75_gradient': results[class_name]['p75'],
        })
        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    csv_path = os.path.join(output_dir, 'gradient_importance_per_class.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"Saved gradient importance to {csv_path}")
    
    # Create visualization
    plot_gradient_importance(df_all, output_dir, class_names, feature_names)
    
    model.eval()
    return results


def plot_gradient_importance(df, output_dir, class_names, feature_names):
    """Plot gradient-based importance as heatmap."""
    # Pivot for heatmap
    pivot_df = df.pivot(index='feature', columns='class', values='mean_gradient')
    
    # Normalize by row for better visualization
    pivot_df_norm = pivot_df.div(pivot_df.max(axis=1), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Absolute values
    sns.heatmap(pivot_df, annot=False, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Mean |Gradient|'})
    ax1.set_title('Gradient-Based Feature Importance (Absolute)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Feature', fontsize=12)
    
    # Normalized values
    sns.heatmap(pivot_df_norm, annot=False, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Normalized Importance'})
    ax2.set_title('Gradient-Based Feature Importance (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'gradient_importance_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved gradient importance heatmap to {plot_path}")


def compute_permutation_importance(model, data_loader, data_config, device, output_dir, 
                                   max_jets=None, n_repeats=5):
    """
    Compute permutation importance by shuffling each feature and measuring performance drop.
    """
    print("\n" + "="*80)
    print("Computing Permutation Importance")
    print("="*80)
    
    # Get baseline predictions
    scores_baseline, labels, inputs = get_predictions(model, data_loader, data_config, device, max_jets)
    
    # Get feature names
    feature_names = data_config.input_dicts['pf_features']
    num_features = len(feature_names)
    class_names = data_config.label_value
    num_classes = len(class_names)
    
    # Compute baseline ROC AUC for each class (one-vs-rest)
    baseline_aucs = {}
    for class_idx, class_name in enumerate(class_names):
        y_true = np.asarray(labels[data_config.label_names[0]] == class_idx).astype(int)
        y_score = scores_baseline[:, class_idx]
        baseline_aucs[class_name] = roc_auc_score(y_true, y_score)
    
    print(f"Baseline ROC AUCs: {baseline_aucs}")
    
    # Permutation importance
    importance_results = {class_name: np.zeros((num_features, n_repeats)) for class_name in class_names}
    
    for feat_idx, feat_name in enumerate(tqdm(feature_names, desc="Permuting features")):
        for repeat in range(n_repeats):
            # Create permuted dataset
            inputs_permuted = copy.deepcopy(inputs)
            
            # Permute this feature across all jets
            # Shape: (num_jets, num_features, num_particles)
            perm_indices = np.random.permutation(inputs_permuted['pf_features'].shape[0])
            inputs_permuted['pf_features'][:, feat_idx, :] = inputs_permuted['pf_features'][perm_indices, feat_idx, :]
            
            # Get predictions with permuted feature
            all_scores = []
            for i in range(0, len(inputs_permuted['pf_features']), data_loader.batch_size):
                batch_end = min(i + data_loader.batch_size, len(inputs_permuted['pf_features']))
                
                # Prepare batch inputs as positional arguments
                pf_points = torch.from_numpy(inputs_permuted['pf_points'][i:batch_end]).to(device)
                pf_features = torch.from_numpy(inputs_permuted['pf_features'][i:batch_end]).to(device)
                pf_vectors = torch.from_numpy(inputs_permuted['pf_vectors'][i:batch_end]).to(device)
                pf_mask = torch.from_numpy(inputs_permuted['pf_mask'][i:batch_end]).to(device)
                
                with torch.no_grad():
                    outputs = model(pf_points, pf_features, pf_vectors, pf_mask)
                    scores = torch.softmax(outputs, dim=1)
                    all_scores.append(scores.cpu().numpy())

            
            scores_permuted = np.concatenate(all_scores, axis=0)
            
            # Compute ROC AUC drop for each class
            for class_idx, class_name in enumerate(class_names):
                y_true = np.asarray(labels[data_config.label_names[0]] == class_idx).astype(int)
                y_score = scores_permuted[:, class_idx]
                auc_permuted = roc_auc_score(y_true, y_score)
                
                # Importance = drop in performance
                importance_results[class_name][feat_idx, repeat] = baseline_aucs[class_name] - auc_permuted
    
    # Save results
    df_list = []
    for class_name in class_names:
        df = pd.DataFrame({
            'feature': feature_names,
            'class': class_name,
            'importance_mean': importance_results[class_name].mean(axis=1),
            'importance_std': importance_results[class_name].std(axis=1),
        })
        df_list.append(df)
    
    df_all = pd.concat(df_list, ignore_index=True)
    csv_path = os.path.join(output_dir, 'permutation_importance.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"Saved permutation importance to {csv_path}")
    
    # Create visualization
    plot_permutation_importance(df_all, output_dir, class_names, feature_names)
    
    return importance_results


def plot_permutation_importance(df, output_dir, class_names, feature_names):
    """Plot permutation importance as bar plots."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(6*len(class_names), 10))
    if len(class_names) == 1:
        axes = [axes]
    
    for ax, class_name in zip(axes, class_names):
        df_class = df[df['class'] == class_name].sort_values('importance_mean', ascending=True)
        
        ax.barh(df_class['feature'], df_class['importance_mean'], 
                xerr=df_class['importance_std'], capsize=3, color='steelblue', alpha=0.7)
        ax.set_xlabel('Importance (ROC AUC Drop)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Permutation Importance: {class_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'permutation_importance_bars.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved permutation importance plot to {plot_path}")


def compute_ablation_importance(model, data_loader, data_config, device, output_dir, max_jets=None):
    """
    Compute feature ablation importance.
    
    Zero out each individual feature and measure performance drop.
    """
    print("\n" + "="*80)
    print("Computing Individual Feature Ablation")
    print("="*80)
    
    # Get baseline predictions
    scores_baseline, labels, inputs = get_predictions(model, data_loader, data_config, device, max_jets)
    
    # Get feature info
    feature_names = data_config.input_dicts['pf_features']
    class_names = data_config.label_value
    num_features = len(feature_names)
    
    # Compute baseline ROC AUC
    baseline_aucs = {}
    for class_idx, class_name in enumerate(class_names):
        y_true = np.asarray(labels[data_config.label_names[0]] == class_idx).astype(int)
        y_score = scores_baseline[:, class_idx]
        baseline_aucs[class_name] = roc_auc_score(y_true, y_score)
    
    print(f"Baseline ROC AUCs: {baseline_aucs}")
    
    # Ablation results
    ablation_results = {class_name: {} for class_name in class_names}
    
    # Test each individual feature
    for feat_idx, feat_name in enumerate(tqdm(feature_names, desc="Ablating individual features")):
        print(f"  Ablating feature: {feat_name}")
        
        # Create ablated dataset (zero out this feature)
        inputs_ablated = copy.deepcopy(inputs)
        inputs_ablated['pf_features'][:, feat_idx, :] = 0
        
        # Get predictions with ablated feature
        all_scores = []
        for i in range(0, len(inputs_ablated['pf_features']), data_loader.batch_size):
            batch_end = min(i + data_loader.batch_size, len(inputs_ablated['pf_features']))
            
            # Prepare batch inputs as positional arguments
            pf_points = torch.from_numpy(inputs_ablated['pf_points'][i:batch_end]).to(device)
            pf_features = torch.from_numpy(inputs_ablated['pf_features'][i:batch_end]).to(device)
            pf_vectors = torch.from_numpy(inputs_ablated['pf_vectors'][i:batch_end]).to(device)
            pf_mask = torch.from_numpy(inputs_ablated['pf_mask'][i:batch_end]).to(device)
            
            with torch.no_grad():
                outputs = model(pf_points, pf_features, pf_vectors, pf_mask)
                scores = torch.softmax(outputs, dim=1)
                all_scores.append(scores.cpu().numpy())

        
        scores_ablated = np.concatenate(all_scores, axis=0)
        
        # Compute performance drop for each class
        for class_idx, class_name in enumerate(class_names):
            y_true = np.asarray(labels[data_config.label_names[0]] == class_idx).astype(int)
            y_score = scores_ablated[:, class_idx]
            auc_ablated = roc_auc_score(y_true, y_score)
            
            # Importance = drop in performance
            ablation_results[class_name][feat_name] = baseline_aucs[class_name] - auc_ablated
    
    # Save results
    df_list = []
    for class_name in class_names:
        for feat_name, importance in ablation_results[class_name].items():
            df_list.append({
                'class': class_name,
                'feature': feat_name,
                'importance': importance,
            })
    
    df_all = pd.DataFrame(df_list)
    csv_path = os.path.join(output_dir, 'ablation_importance.csv')
    df_all.to_csv(csv_path, index=False)
    print(f"Saved ablation importance to {csv_path}")
    
    # Create visualization
    plot_ablation_importance(df_all, output_dir, class_names, feature_names)
    
    return ablation_results


def plot_ablation_importance(df, output_dir, class_names, feature_names):
    """Plot individual feature ablation importance."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(8*len(class_names), 10))
    if len(class_names) == 1:
        axes = [axes]
    
    for ax, class_name in zip(axes, class_names):
        df_class = df[df['class'] == class_name].sort_values('importance', ascending=True)
        
        ax.barh(df_class['feature'], df_class['importance'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Importance (ROC AUC Drop)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Feature Ablation: {class_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'ablation_importance_bars.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation importance plot to {plot_path}")


def extract_attention_weights(model, data_loader, data_config, device, output_dir, max_jets=None):
    """
    Extract and analyze attention weights from ParticleTransformer.
    
    Note: This requires the model to support returning attention weights.
    If not supported, this method will be skipped.
    """
    print("\n" + "="*80)
    print("Extracting Attention Weights")
    print("="*80)
    
    print("Note: Attention weight extraction requires model modification.")
    print("This feature will be implemented in a future update.")
    print("For now, skipping attention analysis.")
    
    # TODO: Implement attention weight extraction
    # This requires modifying ParticleTransformer to return attention weights
    # Will be added in the next iteration
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Interpretability analysis for jet flavor tagging')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--data-config', type=str, required=True,
                       help='Path to data config YAML file')
    parser.add_argument('--network-config', type=str, required=True,
                       help='Path to network config Python file')
    parser.add_argument('--test-data', nargs='+', required=True,
                       help='Test data file patterns (supports wildcards)')
    parser.add_argument('--output-dir', type=str, default='interpretability_results',
                       help='Output directory for results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['gradient', 'permutation', 'ablation', 'attention'],
                       default=['gradient', 'permutation', 'ablation'],
                       help='Interpretability methods to run')
    parser.add_argument('--num-jets', type=int, default=None,
                       help='Maximum number of jets to analyze (None = all)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for inference')
    parser.add_argument('--gradient-batch-size', type=int, default=128,
                       help='Batch size for gradient computation (smaller to avoid OOM)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--n-repeats', type=int, default=5,
                       help='Number of repeats for permutation importance')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Using batch size {args.batch_size} for inference, {args.gradient_batch_size} for gradients")
    
    # Load model and data
    model, model_info, data_config, test_loader, device = load_model_and_data(args)
    
    # Run selected methods
    if 'gradient' in args.methods:
        compute_gradient_attribution(model, test_loader, data_config, device, 
                                    args.output_dir, args.num_jets, args.gradient_batch_size)
    
    if 'permutation' in args.methods:
        compute_permutation_importance(model, test_loader, data_config, device, 
                                      args.output_dir, args.num_jets, args.n_repeats)
    
    if 'ablation' in args.methods:
        compute_ablation_importance(model, test_loader, data_config, device, 
                                   args.output_dir, args.num_jets)
    
    if 'attention' in args.methods:
        extract_attention_weights(model, test_loader, data_config, device, 
                                args.output_dir, args.num_jets)
    
    print("\n" + "="*80)
    print("Interpretability analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
