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
    
    # determine which pairs to plot
    if all_pairwise:
        # plot all pairwise combinations (e.g., b vs c, b vs udsg, c vs udsg)
        nlines = int(len(all_categories)*(len(all_categories)-1)/2)
        categories_to_compare = all_categories
    else:
        # plot only signal vs background combinations
        nlines = len(signal_categories)*len(background_categories)
        categories_to_compare = None  # will use nested loops below
    
    cmap = plt.get_cmap('cool', nlines)
    cidx = 0

    # loop over pairs of categories
    if all_pairwise:
        # loop over all pairs
        for sidx, (signal_category_name, signal_category_settings) in enumerate(all_categories.items()):
            for bidx, (background_category_name, background_category_settings) in enumerate(all_categories.items()):
                if bidx <= sidx: continue

                # get scores for signal and background
                sig_score_branch = signal_category_settings['score_branch']
                bkg_score_branch = background_category_settings['score_branch']
                scores = np.divide(events[sig_score_branch], events[sig_score_branch] + events[bkg_score_branch])
                scores_sig = scores[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_sig = np.ones(len(scores_sig))
                weights_bkg = np.ones(len(scores_bkg))
                
                # safety for no passing events
                if len(scores_sig)==0 or len(scores_bkg)==0:
                    continue

                # calculate AUC
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_weights = np.concatenate((weights_sig, weights_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

                # calculate signal and background efficiency
                thresholds = np.linspace(np.amin(this_scores), np.amax(this_scores), num=100)
                efficiency_sig = np.zeros(len(thresholds))
                efficiency_bkg = np.zeros(len(thresholds))
                for idx, threshold in enumerate(thresholds):
                    eff_s = np.sum(weights_sig[scores_sig > threshold])
                    efficiency_sig[idx] = eff_s
                    eff_b = np.sum(weights_bkg[scores_bkg > threshold])
                    efficiency_bkg[idx] = eff_b
                efficiency_sig /= np.sum(weights_sig)
                efficiency_bkg /= np.sum(weights_bkg)

                # make a plot of the ROC curve
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                ax.plot(efficiency_bkg, efficiency_sig,
                  color=cmap(cidx), linewidth=3, label=label)
                cidx += 1
    else:
        # loop only over signal vs background pairs
        for signal_category_name, signal_category_settings in signal_categories.items():
            for background_category_name, background_category_settings in background_categories.items():

                # get scores for signal and background
                sig_score_branch = signal_category_settings['score_branch']
                bkg_score_branch = background_category_settings['score_branch']
                scores = np.divide(events[sig_score_branch], events[sig_score_branch] + events[bkg_score_branch])
                scores_sig = scores[masks[signal_category_name]]
                scores_bkg = scores[masks[background_category_name]]
                weights_sig = np.ones(len(scores_sig))
                weights_bkg = np.ones(len(scores_bkg))
                
                # safety for no passing events
                if len(scores_sig)==0 or len(scores_bkg)==0:
                    continue

                # calculate AUC
                this_scores = np.concatenate((scores_sig, scores_bkg))
                this_weights = np.concatenate((weights_sig, weights_bkg))
                this_labels = np.concatenate((np.ones(len(scores_sig)), np.zeros(len(scores_bkg))))
                auc = roc_auc_score(this_labels, this_scores, sample_weight=np.abs(this_weights))

                # calculate signal and background efficiency
                thresholds = np.linspace(np.amin(this_scores), np.amax(this_scores), num=100)
                efficiency_sig = np.zeros(len(thresholds))
                efficiency_bkg = np.zeros(len(thresholds))
                for idx, threshold in enumerate(thresholds):
                    eff_s = np.sum(weights_sig[scores_sig > threshold])
                    efficiency_sig[idx] = eff_s
                    eff_b = np.sum(weights_bkg[scores_bkg > threshold])
                    efficiency_bkg[idx] = eff_b
                efficiency_sig /= np.sum(weights_sig)
                efficiency_bkg /= np.sum(weights_bkg)

                # make a plot of the ROC curve
                label = signal_category_settings['label'] + ' vs. '
                label += background_category_settings['label']
                label += ' (AUC: {:.2f})'.format(auc)
                ax.plot(efficiency_bkg, efficiency_sig,
                  color=cmap(cidx), linewidth=3, label=label)
                cidx += 1
    
    # other plot settings
    dummy_efficiency = np.linspace(0, 1, num=101)
    ax.plot(dummy_efficiency, dummy_efficiency,
      color='darkblue', linewidth=3, linestyle='--')
    ax.set_xlabel('Background pass-through', fontsize=12)
    ax.set_ylabel('Signal efficiency', fontsize=12)
    ax.grid(which='both')
    leg = ax.legend()

    # save figure
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')

    # same with log scale on x-axis
    ax.set_xscale('log')
    ax.set_xlim((1e-4, 1))
    fig.tight_layout()
    figname = os.path.join(outputdir, 'roc_log.png')
    fig.savefig(figname)
    print(f'Saved figure {figname}.')
    plt.close()
