# Do model evaluation for jet flavour tagging

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

thisdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(thisdir)

from tools import read_file
from plot_roc_multi import plot_scores_multi
from plot_roc_multi import plot_roc_multi
from plot_ternary import plot_ternary


if __name__=='__main__':

    # read command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfiles', required=True, nargs='+')
    args = parser.parse_args()

    # hard coded settings
    treename = 'Events'
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

    # find all branches to read
    branches_to_read = (
        [cat['label_branch'] for cat in all_categories.values()]
        + [cat['score_branch'] for cat in all_categories.values()]
    )

    # loop over input files
    for inputfile in args.inputfiles:
        print(f'Running on input file {inputfile}...')

        # load events
        events = read_file(
                   inputfile,
                   treename = treename,
                   branches = branches_to_read)
        keys = list(events.keys())
        nevents = len(events[list(events.keys())[0]])
        print('Read events file with following properties:')
        print(f'  - Keys: {keys}.')
        print(f'  - Number of events: {nevents}.')

        # define output directory
        outputdir = inputfile.replace('.root', '_plots')

        # plot the score distributions and ROC curves
        print(f'    Plotting ROC...')
        plot_scores_multi(
            events,
            all_categories,
            outputdir = outputdir)
        plot_roc_multi(
            events,
            signal_categories,
            background_categories,
            outputdir = outputdir,
            all_pairwise = True)
        plt.close()

        # make a ternary scatter plot
        print(f'    Plotting scores in ternary scatter plot...')
        plot_ternary(
            events,
            all_categories,
            outputdir = outputdir)
