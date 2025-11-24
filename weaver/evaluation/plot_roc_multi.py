# Plot score distribution and ROC for multiple processes

import os
import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def plot_scores_multi(events,
            categories,
            outputdir = None):

    # get mask for each category
    cat_masks = {}
    for category_name, category_settings in categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        cat_masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # loop over different scores to plot
    for score_name, score_branch in categories.items():
        score_branch = score_branch['score_branch']

        # retrieve score
        scores = events[score_branch]

        # initialize figure
        fig, ax = plt.subplots()

        # loop over categories
        for category_name, category_settings in categories.items():
            cat_mask = cat_masks[category_name]

            # get scores
            this_values = scores[cat_mask]
            
            # make a histogram
            label = category_settings['label']
            bins = np.linspace(0, 1, num=41)
            hist = np.histogram(this_values, bins=bins)[0]
            norm = np.sum( np.multiply(hist, np.diff(bins) ) )
            staterrors = np.sqrt(np.histogram(this_values, bins=bins)[0])
            ax.stairs(hist/norm, edges=bins,
                  color = category_settings['color'],
                  label = label,
                  linewidth=2)
            ax.stairs((hist+staterrors)/norm, baseline=(hist-staterrors)/norm,
                        color = category_settings['color'],
                        edges=bins, fill=True, alpha=0.15)
        
        ax.set_xlabel(f'Classifier output score ({score_name})', fontsize=12)
        ax.set_ylabel('Events (normalized)', fontsize=12)
        ax.set_title(f'Score distribution', fontsize=12)
        ylim_default= ax.get_ylim()
        ax.set_ylim((0., ylim_default[1]*1.3))
        leg = ax.legend(fontsize=10)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
            lh._sizes = [30]
        fig.tight_layout()
        figname = os.path.join(outputdir, f'{score_branch}.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
    
        # same with log scale
        ax.autoscale()
        ax.set_yscale('log')
        fig.tight_layout()
        figname = os.path.join(outputdir, f'{score_branch}_log.png')
        fig.savefig(figname)
        print(f'Saved figure {figname}.')
        plt.close()


def plot_roc_multi(events,
            signal_categories,
            background_categories,
            outputdir = None,
            all_pairwise = False):

    # check arguments
    all_categories = {**signal_categories, **background_categories}

    # get mask for each category
    masks = {}
    for category_name, category_settings in all_categories.items():
        branch = category_settings['label_branch']
        mask = events[branch].astype(bool)
        masks[category_name] = mask

    # make output directory
    if outputdir is not None:
        if not os.path.exists(outputdir): os.makedirs(outputdir)

    # initialize figure
    fig, ax = plt.subplots()
    
    # determine pairs to plot
    if all_pairwise:
        # plot all pairwise combinations
        category_list = list(all_categories.items())
        pairs = []
        for i in range(len(category_list)):
            for j in range(i+1, len(category_list)):
                pairs.append((category_list[i], category_list[j]))
        cmap = plt.get_cmap('cool', len(pairs))
    else:
        # plot only signal vs background pairs
        pairs = []
        for sig_item in signal_categories.items():
            for bkg_item in background_categories.items():
                pairs.append((sig_item, bkg_item))
        cmap = plt.get_cmap('cool', len(pairs))
    
    cidx = 0

    # loop over pairs of categories
    for pair in pairs:
        (cat1_name, cat1_settings), (cat2_name, cat2_settings) = pair

        # get scores for the two categories
        cat1_score_branch = cat1_settings['score_branch']
        cat2_score_branch = cat2_settings['score_branch']
        scores = np.divide(events[cat1_score_branch], events[cat1_score_branch] + events[cat2_score_branch])
        scores_cat1 = scores[masks[cat1_name]]
        scores_cat2 = scores[masks[cat2_name]]
        weights_cat1 = np.ones(len(scores_cat1))
        weights_cat2 = np.ones(len(scores_cat2))
        
        # safety for no passing events
        if len(scores_cat1)==0 or len(scores_cat2)==0:
            continue

        # calculate AUC
        this_scores = np.concatenate((scores_cat1, scores_cat2))
        this_weights = np.concatenate((weights_cat1, weights_cat2))
        this_labels = np.concatenate((np.ones(len(scores_cat1)), np.zeros(len(scores_cat2))))
        auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

        # calculate signal and background efficiency
        thresholds = np.linspace(np.amin(this_scores), np.amax(this_scores), num=100)
        efficiency_cat1 = np.zeros(len(thresholds))
        efficiency_cat2 = np.zeros(len(thresholds))
        for idx, threshold in enumerate(thresholds):
            eff_1 = np.sum(weights_cat1[scores_cat1 > threshold])
            efficiency_cat1[idx] = eff_1
            eff_2 = np.sum(weights_cat2[scores_cat2 > threshold])
            efficiency_cat2[idx] = eff_2
        efficiency_cat1 /= np.sum(weights_cat1)
        efficiency_cat2 /= np.sum(weights_cat2)

        # make a plot of the ROC curve
        label = cat1_settings['label'] + ' vs. '
        label += cat2_settings['label']
        label += ' (AUC: {:.2f})'.format(auc)
        ax.plot(efficiency_cat2, efficiency_cat1,
          color=cmap(cidx), linewidth=3, label=label)
        cidx += 1
    
    # other plot settings
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--', label='Baseline')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    ax.grid(which='both')
    leg = ax.legend()
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()
