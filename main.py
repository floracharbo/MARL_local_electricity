#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:14:20 2020.
@author: Flora Charbonnier
"""

# =============================================================================
# # packages
# =============================================================================

from src.environment.experiment_manager.runner import run

# Enter experiment-specific settings in the dictionary below if using different parameters
# to the default parameters in config_files/default_input_parameters, using the example syntax below.

# if 'trajectory' in settings, parameters will be updated in run()

settings = {
    'RL': {
       'state_space': 'grdC',
       'n_epochs': 20,
       'type_learning': 'q_learning',
        'act_noise': 0.01,
        'facmac': {
            'batch_size': 10,
        },
        'lr': [1e-3] * 4 + [1e-2, 1e-3, 1e-4] + [1e-3] * (3 + 1 + 2 + 2 + 2 + 2),
        'n_repeats': 5,
       # 'evaluation_methods': 'env_r_c',
        'n_discrete_actions': [3] * (4 + 3 + 3 + 1 + 2) + [2, 10] + [3] * (2 + 2),
        'n_grdC_level': [3] * (4 + 3 + 3 + 1 + 2 + 2) + [2, 10] + [3] * 2,
        'n_other_states': [3] * (4 + 3 + 3 + 1 + 2 + 2 + 2) + [2, 10],
        'aggregate_actions': [False] * (4 + 3 + 3 + 1 + 2 + 2 + 2 + 2) + [True],
        'q_learning': {
            'gamma': [0.5, 0.7, 0.9, 0.99] + [0.7] * (3 + 3 + 1 + 2 + 2 + 2 + 2),
            'eps': [0.5] * (4 + 3) + [0.2, 0.8, 0.9] + [0.5] * (1 + 2 + 2 + 2 + 2),
            'hysteretic': [True] * (4 + 3 + 3) + [False] + [True] * (2 + 2 + 2 + 2),
            'beta_to_alpha': [0.5] * (4 + 3 + 3 + 1) + [0.1, 0.9] + [0.5] * (2 + 2 + 2),
        }
    },
    'syst': {
       'n_homes': 30,
        'clus_dist_share': 0.999,
        'f0': {'gen': 8.012, 'loads': 9.459, 'car': 8.893}
    },
}

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)