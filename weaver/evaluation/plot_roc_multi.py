# Plot score distribution and ROC for multiple processes
# This module provides shared ROC curve generation used by both:
# - Standalone evaluation scripts
# - Wandb logging during training

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def compute_roc_curve(scores_sig, scores_bkg, weights_sig=None, weights_bkg=None):
    """
    Compute ROC curve using sklearn's roc_curve for proper monotonic curves.
    
    Args:
        scores_sig: Discriminant scores for signal class
        scores_bkg: Discriminant scores for background class
        weights_sig: Optional weights for signal (default: uniform)
        weights_bkg: Optional weights for background (default: uniform)
    
    Returns:
        efficiency_sig: Signal efficiency (TPR) at each threshold - monotonic
        efficiency_bkg: Background efficiency (FPR/pass-through) at each threshold - monotonic
        auc: Area under the ROC curve
    """
    from sklearn.metrics import roc_curve as sklearn_roc_curve
    
    if weights_sig is None:
        weights_sig = np.ones(len(scores_sig))
    if weights_bkg is None:
        weights_bkg = np.ones(len(scores_bkg))
    
    # Safety check
    if len(scores_sig) == 0 or len(scores_bkg) == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    
    # Combine into standard format for sklearn
    all_scores = np.concatenate([scores_sig, scores_bkg])
    all_labels = np.concatenate([np.ones(len(scores_sig)), np.zeros(len(scores_bkg))])
    all_weights = np.concatenate([weights_sig, weights_bkg])
    
    # Use sklearn's roc_curve which guarantees monotonicity
    # Returns: fpr (background efficiency), tpr (signal efficiency), thresholds
    fpr, tpr, _ = sklearn_roc_curve(all_labels, all_scores, sample_weight=all_weights)
    
    # fpr = false positive rate = background pass-through
    # tpr = true positive rate = signal efficiency
    efficiency_bkg = fpr
    efficiency_sig = tpr
    
    # Calculate AUC
    try:
        auc = roc_auc_score(all_labels, all_scores, sample_weight=np.abs(all_weights))
    except:
        auc = 0.5
    
    return efficiency_sig, efficiency_bkg, auc


def generate_roc_figures(roc_data, title_suffix=''):
    """
    Generate ROC curve figures (linear and log scale) from precomputed data.
    
    Args:
        roc_data: List of dicts with keys:
            - 'eff_sig': signal efficiency array
            - 'eff_bkg': background efficiency array
            - 'auc': AUC value
            - 'label': label for legend (e.g., 'b vs c')
            - 'color': optional color
        title_suffix: Optional suffix for plot title (e.g., '(Epoch 10)')
    
    Returns:
        fig_linear: matplotlib figure with linear scale
        fig_log: matplotlib figure with log scale
    """
    fig_linear, ax_linear = plt.subplots(figsize=(8, 6))
    fig_log, ax_log = plt.subplots(figsize=(8, 6))
    
    # Colormap for lines
    n_curves = len(roc_data)
    cmap = plt.get_cmap('cool', max(n_curves, 1))
    
    for idx, curve in enumerate(roc_data):
        color = curve.get('color', cmap(idx))
        label = f"{curve['label']} (AUC: {curve['auc']:.3f})"
        
        ax_linear.plot(curve['eff_bkg'], curve['eff_sig'], 
                      color=color, linewidth=2, label=label)
        ax_log.plot(curve['eff_bkg'], curve['eff_sig'], 
                   color=color, linewidth=2, label=label)
    
    # Diagonal reference line
    ax_linear.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    ax_log.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    
    # Configure linear plot
    ax_linear.set_xlabel('Background pass-through', fontsize=12)
    ax_linear.set_ylabel('Signal efficiency', fontsize=12)
    ax_linear.set_title(f'ROC Curves {title_suffix}'.strip(), fontsize=14)
    ax_linear.legend(loc='lower right', fontsize=9)
    ax_linear.grid(True, alpha=0.3)
    ax_linear.set_xlim(0, 1)
    ax_linear.set_ylim(0, 1)
    fig_linear.tight_layout()
    
    # Configure log plot
    ax_log.set_xlabel('Background pass-through', fontsize=12)
    ax_log.set_ylabel('Signal efficiency', fontsize=12)
    ax_log.set_title(f'ROC Curves - Log Scale {title_suffix}'.strip(), fontsize=14)
    ax_log.legend(loc='lower right', fontsize=9)
    ax_log.grid(True, which='both', alpha=0.3)
    ax_log.set_xscale('log')
    ax_log.set_xlim(1e-5, 1)
    ax_log.set_ylim(0, 1)
    fig_log.tight_layout()
    
    return fig_linear, fig_log


def generate_score_figures(score_data, title_suffix=''):
    """
    Generate score distribution figures from precomputed data.
    
    Args:
        score_data: Dict mapping score_name -> list of dicts with keys:
            - 'values': score values for this class
            - 'label': class label for legend
            - 'color': optional color
        title_suffix: Optional suffix for plot title (e.g., '(Epoch 10)')
    
    Returns:
        Dict mapping score_name -> (fig_linear, fig_log) tuples
    """
    figures = {}
    
    for score_name, class_data in score_data.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        fig_log, ax_log = plt.subplots(figsize=(8, 6))
        
        bins = np.linspace(0, 1, num=41)
        
        for class_info in class_data:
            values = class_info['values']
            if len(values) == 0:
                continue
                
            label = class_info['label']
            color = class_info.get('color', None)
            
            hist, _ = np.histogram(values, bins=bins)
            norm = np.sum(hist * np.diff(bins))
            if norm == 0:
                continue
            
            staterrors = np.sqrt(hist)
            
            plot_kwargs = {'label': label, 'linewidth': 2}
            if color is not None:
                plot_kwargs['color'] = color
            
            ax.stairs(hist/norm, edges=bins, **plot_kwargs)
            ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                     color=plot_kwargs.get('color'), edges=bins, fill=True, alpha=0.15)
            
            ax_log.stairs(hist/norm, edges=bins, **plot_kwargs)
            ax_log.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                         color=plot_kwargs.get('color'), edges=bins, fill=True, alpha=0.15)
        
        # Configure linear plot
        ax.set_xlabel(f'Classifier output score ({score_name})', fontsize=12)
        ax.set_ylabel('Events (normalized)', fontsize=12)
        ax.set_title(f'Score Distribution {title_suffix}'.strip(), fontsize=12)
        ylim = ax.get_ylim()
        ax.set_ylim((0., ylim[1]*1.3))
        ax.legend(fontsize=10)
        fig.tight_layout()
        
        # Configure log plot
        ax_log.set_xlabel(f'Classifier output score ({score_name})', fontsize=12)
        ax_log.set_ylabel('Events (normalized)', fontsize=12)
        ax_log.set_title(f'Score Distribution {title_suffix}'.strip(), fontsize=12)
        ax_log.set_yscale('log')
        ax_log.legend(fontsize=10)
        fig_log.tight_layout()
        
        figures[score_name] = (fig, fig_log)
    
    return figures


def generate_roc_from_arrays(labels, scores, class_names, title_suffix=''):
    """
    Generate ROC figures from array-based labels and scores.
    Used by wandb logging during training.
    
    Args:
        labels: 1D array of true class labels (integers 0, 1, 2, ...)
        scores: 2D array of predicted probabilities, shape (n_samples, n_classes)
        class_names: List of class names corresponding to each class index
        title_suffix: Optional suffix for plot title (e.g., '(Epoch 10)')
    
    Returns:
        fig_linear: matplotlib figure with linear scale ROC
        fig_log: matplotlib figure with log scale ROC
        roc_data: List of ROC curve data dicts (for reuse)
    """
    n_classes = len(class_names)
    
    # Create masks for each class
    class_masks = {i: (labels == i) for i in range(n_classes)}
    
    # Compute ROC curves for all pairwise combinations
    roc_data = []
    
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Get samples belonging to class i or class j
            mask = class_masks[i] | class_masks[j]
            if mask.sum() == 0:
                continue
            
            # Get scores for these two classes
            scores_i = scores[mask, i]
            scores_j = scores[mask, j]
            binary_labels = labels[mask]
            
            # Use raw signal score as discriminant (no binarization)
            # For class i vs class j, use the score for class i
            disc_class_i = scores_i[binary_labels == i]
            disc_class_j = scores_i[binary_labels == j]
            
            if len(disc_class_i) == 0 or len(disc_class_j) == 0:
                continue
            
            # Compute ROC curve
            eff_sig, eff_bkg, auc = compute_roc_curve(disc_class_i, disc_class_j)
            
            roc_data.append({
                'eff_sig': eff_sig,
                'eff_bkg': eff_bkg,
                'auc': auc,
                'label': f'{class_names[i]} vs {class_names[j]}'
            })
    
    # Generate figures
    fig_linear, fig_log = generate_roc_figures(roc_data, title_suffix=title_suffix)
    
    return fig_linear, fig_log, roc_data


def generate_scores_from_arrays(labels, scores, class_names, title_suffix=''):
    """
    Generate score distribution figures from array-based labels and scores.
    Used by wandb logging during training.
    
    Args:
        labels: 1D array of true class labels (integers 0, 1, 2, ...)
        scores: 2D array of predicted probabilities, shape (n_samples, n_classes)
        class_names: List of class names corresponding to each class index
        title_suffix: Optional suffix for plot title (e.g., '(Epoch 10)')
    
    Returns:
        Dict mapping score_name -> (fig_linear, fig_log) tuples
    """
    n_classes = len(class_names)
    class_masks = {i: (labels == i) for i in range(n_classes)}
    
    # Build score data structure
    score_data = {}
    for score_idx, score_name in enumerate(class_names):
        class_data = []
        for class_idx, class_name in enumerate(class_names):
            class_scores = scores[class_masks[class_idx], score_idx]
            class_data.append({
                'values': class_scores,
                'label': class_name
            })
        score_data[score_name] = class_data
    
    return generate_score_figures(score_data, title_suffix=title_suffix)


# ============================================================================
# Legacy API - maintains backward compatibility with existing evaluation scripts
# ============================================================================

def plot_scores_multi(events, categories, outputdir=None):
    """
    Plot score distributions for multiple categories.
    Legacy API for evaluation scripts using dict-based events.
    
    Args:
        events: Dict mapping branch names to arrays
        categories: Dict mapping category name to settings dict with:
            - 'label_branch': branch name for true labels
            - 'score_branch': branch name for scores
            - 'color': color for plotting
            - 'label': display label
        outputdir: Directory to save figures (None to skip saving)
    """
    # Get mask for each category
    cat_masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        cat_masks[category_name] = mask

    # Make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

    # Loop over different scores to plot
    for score_name, score_settings in categories.items():
        score_branch = score_settings['score_branch']
        scores = events[score_branch]

        # Build score data for this score type
        class_data = []
        for category_name, category_settings in categories.items():
            cat_mask = cat_masks[category_name]
            class_data.append({
                'values': scores[cat_mask],
                'label': category_settings['label'],
                'color': category_settings['color']
            })
        
        # Generate figures
        score_data = {score_name: class_data}
        figures = generate_score_figures(score_data)
        fig, fig_log = figures[score_name]
        
        # Save figures
        if outputdir is not None:
            figname = os.path.join(outputdir, f'{score_branch}.png')
            fig.savefig(figname)
            print(f'Saved figure {figname}.')
            
            figname_log = os.path.join(outputdir, f'{score_branch}_log.png')
            fig_log.savefig(figname_log)
            print(f'Saved figure {figname_log}.')
        
        plt.close(fig)
        plt.close(fig_log)


def plot_roc_multi(events, signal_categories, background_categories, 
                   outputdir=None, all_pairwise=False):
    """
    Plot ROC curves for multiple category pairs.
    Legacy API for evaluation scripts using dict-based events.
    
    Args:
        events: Dict mapping branch names to arrays
        signal_categories: Dict of signal category settings
        background_categories: Dict of background category settings
        outputdir: Directory to save figures (None to skip saving)
        all_pairwise: If True, plot all pairwise combinations; 
                      if False, only signal vs background
    """
    all_categories = {**signal_categories, **background_categories}

    # Get mask for each category
    masks = {}
    for category_name, category_settings in all_categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        masks[category_name] = mask

    # Make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

    # Determine which pairs to plot
    if all_pairwise:
        pairs = [
            (sig_name, sig_settings, bkg_name, bkg_settings)
            for sidx, (sig_name, sig_settings) in enumerate(all_categories.items())
            for bidx, (bkg_name, bkg_settings) in enumerate(all_categories.items())
            if bidx > sidx
        ]
    else:
        pairs = [
            (sig_name, sig_settings, bkg_name, bkg_settings)
            for sig_name, sig_settings in signal_categories.items()
            for bkg_name, bkg_settings in background_categories.items()
        ]

    # Compute ROC curves for each pair
    roc_data = []
    cmap = plt.get_cmap('cool', max(len(pairs), 1))
    
    for idx, (sig_name, sig_settings, bkg_name, bkg_settings) in enumerate(pairs):
        # Use raw signal score as discriminant (no binarization)
        sig_score_branch = sig_settings['score_branch']
        
        scores_sig = events[sig_score_branch][masks[sig_name]]
        scores_bkg = events[sig_score_branch][masks[bkg_name]]
        
        if len(scores_sig) == 0 or len(scores_bkg) == 0:
            continue
        
        # Compute ROC curve using shared function
        eff_sig, eff_bkg, auc = compute_roc_curve(scores_sig, scores_bkg)
        
        roc_data.append({
            'eff_sig': eff_sig,
            'eff_bkg': eff_bkg,
            'auc': auc,
            'label': f"{sig_settings['label']} vs. {bkg_settings['label']}",
            'color': cmap(idx)
        })

    # Generate figures
    fig_linear, fig_log = generate_roc_figures(roc_data)

    # Save figures
    if outputdir is not None:
        figname = os.path.join(outputdir, 'roc.png')
        fig_linear.savefig(figname)
        print(f'Saved figure {figname}.')
        
        figname_log = os.path.join(outputdir, 'roc_log.png')
        fig_log.savefig(figname_log)
        print(f'Saved figure {figname_log}.')
    
    plt.close(fig_linear)
    plt.close(fig_log)
