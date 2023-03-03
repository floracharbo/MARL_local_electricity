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

# Enter experiment-specific settings in the dictionary below if using different parameters
# to the default parameters in config_files/default_input_parameters, using the example syntax below.
# q learning / facmac no traj / facmac traj / facmac supervised no traj / facmac supervised traj
settings = {
    'RL': {
        'state_space': [['flexibility', 'grdC_n2']] * 6 * 5,
        'n_epochs': 20,
        'n_repeats': 3,
        'type_learning': ['q_learning'] * 6 + ['facmac'] * 6 * 4,
        'trajectory': [False] * 6 * 2 + [True] * 6 + [False] * 6 + [True] * 6,
        'supervised_loss': [False] * 6 * 3 + [True] * 6 * 2,
        'act_noise': 0.01,
        'lr': 1e-2,
    },
    'syst': {
        'force_optimisation': True,
        'n_homes': [1, 3, 5, 10, 20, 50] * 5,
    }
}
trajectory = settings['RL']['trajectory']
settings['RL']['evaluation_methods'] = None if settings['RL']['type_learning'] == 'q_learning' else 'env_r_c'
settings['RL']['obs_agent_id'] = False if trajectory else True
settings['RL']['nn_type'] = 'cnn' if trajectory else 'linear'
settings['RL']['rnn_hidden_dim'] = 1e3 if trajectory else 5e2

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [282]  # if plotting

run(RUN_MODE, settings, no_runs)

