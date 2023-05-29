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

from src.environment.initialisation.initialise_objects import \
    initialise_objects
from src.environment.utilities.utilities import (data_source, distr_learning,
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

    # i removed 10 opts True,  diff False largeQ False (run 232) as
    # there now is run 236 opts True diff True largeQ False
    # (they did not have the same values)
    # i removed 10 opts False, diff True  largeQ False (run 231) as
    # there now is run 236 opts True diff True largeQ False
    # (they did not have the same values)
    # i need to do 5 agents opts True diff True largeQ false as there
    # is a clash between run 229 and 230
    # i removed 229 ( T F F) and 230 (F T F) for 5 agents as there
    # is 241 (T T F)
    # i removed 225 to add new simulation for 1 agent 242
    # i removed 228 to add newer simulation 243 for 4 agents True True True
    # i removed 226 to add newer simulation 244 for 2 agents True True True
    # i removed 227 to add newer simulation 245 for 3 agents True True True
    # i removed 234 to add newer simulation 246 for 40 agents True True True
    # 242 (server T T T) / 247 (laptop T T F) ?
    # 244 (server T T T) / 248 (laptop T T F) ?
    # 245 (server T T T) / 250 (laptop T T F) ?
    # 243 (server T T T) / 251 (laptop T T F) ?
    # 252 newer run for 50 agents, replacing run 235
    # 246 newer run for 40 agents, replacing run 234
    # runs = [247,  248,  250,  251,  246,   252,   236,   237,   239,
    # 240,   241,   253,   254,   255,  256,    257,   258,   259,
    # 260, 262] #,   261]
    # n_ags = [1,    2,    3,    4,    40,    50,    10,    20,
    # 25,    60,    5,     12,    1,     2,     3,     4,     10,
    # 50,    30, 25]#,    5]
    # opts = [True, True, True, True, False, False, True,  False,
    # False, False, True,  True,  True,  False, False, False, False,
    # False, False, False]#, False]
    # diff = [True, True, True, True, False, False, True,  True,  True,
    # False, True,  True,  True,  True,  True,  True,  True,  False,
    # False, True]#, True]
    # larQ = [True, True, True, True, False, False, False, False, False,
    # False, False, False, False, False, False, False, False, False, False,
    # False]#, False]

    # new runs 50 epochs - for paper
    # runs = [263, 264, 265, 266, 267, 268, 269, 270, 271]
    # n_ags = [1,   2,   3,   4,   5,   15,  10,  20, 30]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]

    # fixed advantage - 20210308
    # runs = [272, 273, 274, 275, 276, 277, 278, 279]
    # n_ags = [1,   4,   10,  2,   3,   5,   20,  30]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]

    # 07 May 2021 - compare again, hysteretic
    # runs = [395, 392, 396, 397]
    # n_ags = [1,   5,   10,  30 ]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    # TITLE = 'results_vs_nag_hysteretic_20210507'

    # 8 May 2021 - compare again, NOT hysteretic
    # runs = [403, 404, 405, 406]
    # n_ags = [1,   5,   10,  30 ]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]

    # policy_gradient (DDPG) 09.05.21
    # runs = [428, 429, 430, 431]
    # n_ags = [1,   5,   10,  30 ]
    # opts = [False for _ in range(len(runs))]
    # diff = [False for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    # to_plot = ['env_r_c']

    # not policy gradient, 20 repeats, 12.05.21
    # runs = [433, 434, 435]
    # n_ags = [1,   5,   10]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]

    # policy gradient 10 repeats 11.06.21
    # runs = range(505, 509)
    # n_ags = [1,   5,   10,  30 ]
    # opts = [False for _ in range(len(runs))]
    # diff = [False for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    # to_plot = ['env_r_d']
    # TITLE = 'results_vs_nag_DDPG_110621'

    # runs = range(518,527) # for update paper applied energy
    # n_ags = [1, 2, 3, 4, 5, 15, 10, 20, 30]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    # TITLE = 'results_vs_nag_20210713'
    # type_learning = ['q_learning'] * len(n_ags)

    # 25 aug 2021 - new data as changed temperature control
    # - for update IEEEv2 - also for 1st submission applied energy
    # for update paper applied energy
    # runs = list(range(530, 537)) + list(range(538, 543))[:-3]
    # n_ags = [1, 2, 3, 5, 10, 20, 30, 4, 15, 7, 6, 50][:-3]

    # 22 may 2023- thesis q learning
    runs = [1917] + list(range(1956, 1965)) + [1919]
    n_ags = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    TITLE = 'results_vs_nag_20210826_tryagain_'
    # type_learning = ['q_learning'] * len(n_ags)

    # 07 feb 2022 - new data as changed temperature control
    # for resubmission applied energy
    # runs = [603, 604] + list(range(606, 613))
    # n_ags = [1, 2, 3, 4, 5, 10, 15, 20, 30]
    # opts = [True for _ in range(len(runs))]
    # diff = [True for _ in range(len(runs))]
    # larQ = [False for _ in range(len(runs))]
    # polg = [False for _ in range(len(runs))]
    # TITLE = 'results_vs_nag_20220209'
    # type_learning = ['q_learning'] * len(n_ags)
    #
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
    TITLE = 'q_learning_vs_nag_20230522'
    if save:
        if 'high_res' in prm['save'] and prm['save']['high_res']:
            fig.savefig(PATH0 + TITLE + '.pdf', bbox_inches='tight',
                        format='pdf', dpi=1200)
        else:
            fig.savefig(PATH0 + TITLE, bbox_inches='tight')
    print('end plots results vs n_ag')
