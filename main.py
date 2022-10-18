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

#  Inputs
settings = {
    'heat': {'file': 'heat2'},

    'RL': {
        'type_learning': 'facmac',
        # 'evaluation_methods': 'env_r_c',
        'gamma': {'q_learning': 0.99, 'facmac': 0.99},
        'aggregate_actions': False,
        'mixer': 'qmix',

        # current experiment
        'batch_size': 2,
        'rnn_hidden_dim': 1e2,
        'state_space': [['grdC', 'bat_dem_agg', 'avail_EV_step']],
        'n_epochs': 5,
        'n_repeats': 2,
        'lr': 1e-4,
        'facmac': {'critic_lr': 1e-4}
    },

    'ntw': {
        'n': 3
    },
}

# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [610]  # if plotting

# run(RUN_MODE, settings)

for type_learning in ['q_learning', 'facmac']:
    settings['RL']['type_learning'] = type_learning
    for aggregate_actions in [True, False]:
        settings['RL']['aggregate_actions'] = aggregate_actions
        print(f"test {type_learning} aggregate_actions {aggregate_actions}")
        run(RUN_MODE, settings)
