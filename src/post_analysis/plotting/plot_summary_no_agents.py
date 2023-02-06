#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:53:57 2021.

@author: floracharbonnier

plots summary of results for different number of agents
"""
# packages
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.initialisation.initialise_objects import initialise_objects
from src.utilities.userdeftools import (data_source, distr_learning,
                                        initialise_dict, reward_type)


def _get_prm(PATH, MAIN_DIR_NOT_SERVER, run, server, n_ag, run_mode=1):
    if os.path.exists(PATH + f'run{run}/inputData/prm.npy'):
        prm = np.load(
            PATH + f'run{run}/inputData/prm.npy',
            allow_pickle=True).item()
    else:
        prm = np.load(PATH + f'run{run}/inputData/syst.npy',
                      allow_pickle=True).item()
        prm['RL'] = np.load(PATH + f'run{run}/inputData/lp.npy',
                            allow_pickle=True).item()
        prm['syst'] = prm['prm']
    if 'paths' not in prm:
        with open("/Users/floracharbonnier/Documents/GitHub/dphil/"
                  "inputs/paths.yaml", "r") as f:
            prm['paths'] = yaml.safe_load(f)
    if not server:
        prm['paths']['main_dir'] = Path(MAIN_DIR_NOT_SERVER)
    prm['paths']['current_path'] = \
        Path('/Users/floracharbonnier/Documents/GitHub/dphil')
    prm['paths']['input_folder'] = 'Inputs'
    prm, _ = initialise_objects(
        prm, no_run=run, initialise_record=False, run_mode=run_mode
    )
    if n_ag == 1:
        prm['RL']['evaluation_methods'] = \
            [e for e in prm['RL']['evaluation_methods']
             if distr_learning(e) == 'd' or e in ['baseline', 'opt']]
    metrics = np.load(
        PATH + f'run{run}/figures/metrics.npy',
        allow_pickle=True).item()

    if run < 254:
        prm['RL']['evaluation_methods'] = [
            method for method in prm['RL']['evaluation_methods']
            if method == 'opt' or data_source[method] == 'opt'
        ]

    return prm, metrics


def _metrics_to_results(prm, n_ag, to_plot, res, res_entries, metrics):
    for evaluation_method in \
            [evaluation_method for evaluation_method in prm['RL']['evaluation_methods']
             if evaluation_method != 'baseline']:
        if n_ag > 1 or evaluation_method == 'opt':
            type_evals = [evaluation_method]
        else:
            type_evals = \
                [f"{data_source[evaluation_method]}_{reward_type(evaluation_method)}_{e}"
                 for e in ['c', 'd']]

        for t_ in type_evals:
            if t_ in to_plot:
                if t_ in res['xs'].keys():
                    res['xs'][t_].append(n_ag)
                    for key in res_entries[1:]:
                        res[key][t_].append(
                            metrics['end_bl'][key][evaluation_method])
                else:
                    res['xs'][t_] = [n_ag]
                    for key in res_entries[1:]:
                        res[key][t_] = [metrics['end_bl'][key][evaluation_method]]

    return res


def plot_results_vs_nag():
    """
    Plot comparison results across runs.

    e.g. varying number of agents on x-axis.
    """
    # inputs
    SMALL = True
    PERSONAL_PATH = '/Users/floracharbonnier'
    MAIN_DIR_NOT_SERVER = '/Users/floracharbonnier/OneDrive - Nexus365' \
                          '/DPhil/Python/Phase2'
    current_path = os.getcwd()
    save = True
    server = 0 if current_path[0: len(PERSONAL_PATH)] == PERSONAL_PATH else 1
    PATH0 = '/Users/floracharbonnier/OneDrive - Nexus365/DPhil/Python/Phase2/'
    PATH = PATH0 + 'results/results_EPGbeast/'
    res_entries = ['xs', 'ave', 'std', 'p25', 'p50', 'p75']
    res = initialise_dict(res_entries, 'empty_dict')

    plt.rcParams['font.size'] = '10'

    to_plot = ['opt', 'env_r_c', 'opt_d_d']

    if SMALL:
        fig = plt.figure(figsize=(4.600606418100001, 4.2))
    else:
        fig = plt.figure(figsize=(5.023, 2.953))
    for n_ag, run in zip(n_ags, runs):
        prm, metrics = _get_prm(PATH, MAIN_DIR_NOT_SERVER, run, server, n_ag, run_mode=2)
        res = _metrics_to_results(
            prm, n_ag, to_plot, res, res_entries, metrics
        )

    # 3 too shallow, 3.3 too deep, 3.15 slightly too deep
    # fig = plt.figure(figsize=(5.25*0.95**2*0.98*0.99*0.8*0.9*1.39,5.25*0.8))
    # for the poster
    # red = prm['save']['colourse']['opt']
    # purple = prm['save']['colourse']['opt_d_d']
    red = (234 / 255, 53 / 255, 37 / 255)
    prm['save']['colourse']['env_r_d'] = red
    # blue = prm['save']['colourse']['opt_r_d']
    prm['save']['colourse']['opt'] = 'grey'
    green = prm['save']['colourse']['env_d_d']
    prm['save']['colourse']['opt_d_d'] = green

    for evaluation_method in res['xs'].keys():
        order = np.argsort(res['xs'][evaluation_method])
        for key in res_entries:
            res[key][evaluation_method] = [res[key][evaluation_method][i] for i in order]
        line_style = 'dotted' if evaluation_method == 'opt' else '-'
        colour = prm['save']['colourse'][evaluation_method] \
            if evaluation_method == 'opt' \
            else prm['save']['colourse'][
            f"{data_source(evaluation_method)}_{reward_type(evaluation_method)}_d"]
        plt.plot(
            res['xs'][evaluation_method],
            res['p50'][evaluation_method],
            color=colour,
            ls=line_style,
            label=evaluation_method)
        plt.fill_between(
            res['xs'][evaluation_method],
            res['p25'][evaluation_method],
            res['p75'][evaluation_method],
            color=colour,
            alpha=0.3)
        print(f"evaluation_method {evaluation_method} colour {colour} "
              f"res['xs'][evaluation_method] {res['xs'][evaluation_method]} "
              f"res['p50'][evaluation_method] {res['p50'][evaluation_method]}")
    plt.gca().set_xscale('log')
    xmax = max(n_ags)
    plt.hlines(y=0, xmin=1, xmax=xmax, colors='k',
               linestyle='dotted', label='bla')
    # plt.legend(loc = 'right', bbox_to_anchor = (1.4, 0), fancybox = True)
    lower_bound, upper_bound = [np.load(PATH0 + e + '0.npy')
                                for e in ['lower_bound', 'upper_bound']]
    # plt.ylim([-0.025, upper_bound])
    plt.ylim([lower_bound, upper_bound])
    plt.gca().set_yticks(np.arange(-0.25, 0.2, 0.05))
    # plt.legend(loc = 'right', bbox_to_anchor = (1.4, 0), fancybox = True)
    if SMALL:
        TITLE += '_small'
    prm['save']['high_res'] = True
    TITLE = 'simplified_results_withlegend'
    if save:
        if 'high_res' in prm['save'] and prm['save']['high_res']:
            fig.savefig(PATH0 + TITLE + '.pdf', bbox_inches='tight',
                        format='pdf', dpi=1200)
        else:
            fig.savefig(PATH0 + TITLE, bbox_inches='tight')
    print('end plots results vs n_ag')
