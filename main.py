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
        # 'state_space': ['grdC_level', 'hour', 'store0', 'dT', 'dT_next', 'loads_cons_prev', 'day_type'],
        'n_epochs': 20,
        'n_repeats': 5,
        'rnn_hidden_dim': 5e3,
        # 'n_hidden_layers': 1,
        # 'lr': 1e-4,
        # 'facmac': {'lr': 1e-4}
    },
    'ntw': {
        'n': 10
    },
}

# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [899]  # if plotting

run(RUN_MODE, settings, no_runs)

# for type_learning in ['facmac', 'q_learning']:
#     settings['RL']['type_learning'] = type_learning
#     for aggregate_actions in [True, False]:
#         settings['RL']['aggregate_actions'] = aggregate_actions
#         print(f"test {type_learning} aggregate_actions {aggregate_actions}")
#         run(RUN_MODE, settings)
