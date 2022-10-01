#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:14:20 2020.

@author: Flora Charbonnier
"""

# =============================================================================
# # packages
# =============================================================================
import time  # to record time it takes to run simulations
import traceback

import numpy as np  # array management / data handling
import torch as th

from config.initialise_objects import initialise_objects
from config.initialise_prm import get_settings_i, load_existing_prm
from config.input_data import input_paths
from post_analysis.plot_summary_no_agents import plot_results_vs_nag
from post_analysis.post_processing import post_processing
from simulations.local_elec import LocalElecEnv
from simulations.runner import Runner
from utils.userdeftools import play_sound

# %% =========================================================================
#  Inputs
# ============================================================================

settings = {
    'heat': {'file': 'heat2'},

    'RL': {
        'type_learning': 'q_learning',
        'type_env': 'continuous',
        # 'explo_reward_type': [['random', 'env_r_c']],
        'gamma': {'q_learning': 0.99, 'facmac': 0.85},
        'aggregate_actions': True,
        'mixer': 'qmix',

        # current experiment
        'rnn_hidden_dim': 1e2,
        # 'rnn_hidden_dim': [1e2, 1e3, 5e3, 1e4] + [5e4] * 3 + [1e2, 1e3, 5e3],
        # 'n_hidden_layers': [2] * 7 + [3] * 3,
        # 'state_space': [['grdC', 'bat_dem_agg', 'avail_EV_step']] * 10,
        'state_space': 'grdC',
        'n_epochs': 10,
        'n_repeats': 2,
        'lr': 1e-6,
        'facmac': {'critic_lr': 1e-6}
    },

    'ntw': {
        'n': 20
    },
}

# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
LEARN_PLOT = 1
# no_runs = [504]   # if plotting
no_runs = [610]   # if plotting

# %%===========================================================================
# # Learning
# =============================================================================
prm = input_paths()

if LEARN_PLOT == 1:
    try:

        # obtain the number of runs from the longest settings entry
        N_RUNS = 1

        for sub_dict in settings.values():
            for val in list(sub_dict.values()):
                if isinstance(val, dict):
                    for val_ in list(val.values()):
                        if isinstance(val_, list) and len(val_) > N_RUNS:
                            N_RUNS = len(val_)
                elif isinstance(val, list) and len(val) > N_RUNS:
                    N_RUNS = len(val)

        # loop through runs
        for i in range(N_RUNS):
            remove_old_prms = [e for e in prm if e != 'paths']
            for e in remove_old_prms:
                del prm[e]

            start_time = time.time()  # start recording time
            # obtain run-specific settings

            settings_i = get_settings_i(settings, i)
            # initialise learning parameters, system parameters and recording
            prm, record, profiles = initialise_objects(
                prm, settings=settings_i)

            DESCRIPTION_RUN = 'current code '
            for e in ['type_learning', 'n_repeats', 'n_epochs',
                      'server', 'state_space']:
                DESCRIPTION_RUN += f"prm['RL'][{e}] {prm['RL'][e]} "
            print(DESCRIPTION_RUN)
            prm['save']['description_run'] = DESCRIPTION_RUN

            if prm['RL']['type_learning'] == 'facmac':
                # Setting the random seed throughout the modules
                np.random.seed(prm['syst']["seed"])
                th.manual_seed(prm['syst']["seed"])

            env = LocalElecEnv(prm, profiles)
            # second part of initialisation specifying environment
            # with relevant parameters
            record.init_env(env)  # record progress as we train
            runner = Runner(env, prm, record)
            runner.run_experiment()
            record.save(end_of='end')  # save progress at end
            post_processing(
                record, env, prm, start_time=start_time, settings_i=settings_i
            )
            print(f"--- {time.time() - start_time} seconds ---")
            if prm["syst"]["play_sound"]:
                play_sound()
    except Exception as ex:
        print(f"ex {ex}")
        print(traceback.format_exc())
        print(f'ex.args {ex.args}')

# %===========================================================================
# # post learning analysis / plotting
# =============================================================================
elif LEARN_PLOT == 2:
    if isinstance(no_runs, int):
        no_runs = [no_runs]  # the runs need to be in an array

    for no_run in no_runs:
        lp, prm = load_existing_prm(prm, no_run)

        prm, record, profiles = initialise_objects(prm, no_run=no_run)
        # make user defined environment
        env = LocalElecEnv(prm, profiles)
        record.init_env(env)  # record progress as we train
        post_processing(record, env, prm, no_run=no_run)

elif LEARN_PLOT == 3:
    plot_results_vs_nag()
