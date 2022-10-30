#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:14:20 2020.

@author: Flora Charbonnier
"""

# =============================================================================
# # packages
# =============================================================================

from src.simulations.runner import run

# Inputs
settings = {
    'heat': {'file': 'heat2'},
    'RL': {
        'type_learning': 'q_learning',
        'aggregate_actions': False,
        'mixer': 'qmix',
        # current experiment
        'state_space': [['grdC']] * 1,

        # 'state_space': [['grdC', 'avail_car_step', 'bat_dem_agg']] * 1,
        'n_epochs': 6,
        'n_repeats': 2,
        # 'facmac': {
        #     'gamma': [0.7, 0.99],
        #     'rnn_hidden_dim': [100, 100, 10, 50, 1000],
        #     'n_hidden_layers': [1] * 5 + [2]
        # }
    },
    'ntw': {
        'n': 5
    },
}

# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [610]  # if plotting

run(RUN_MODE, settings)

# for type_learning in ['facmac', 'q_learning']:
#     settings['RL']['type_learning'] = type_learning
#     for aggregate_actions in [True, False]:
#         settings['RL']['aggregate_actions'] = aggregate_actions
#         print(f"test {type_learning} aggregate_actions {aggregate_actions}")
#         run(RUN_MODE, settings)
