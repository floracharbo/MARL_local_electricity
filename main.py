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
        'type_learning': 'facmac',
        'evaluation_methods': [['env_r_c', 'opt_r_c']],
        'aggregate_actions': False,
        'cnn_kernel_size': 2,
        'normalise_states': True,
        'obs_agent_id': True,
        'trajectory': False,
        'rnn_hidden_dim': 1e2,
        'state_space': [['flexibility', 'grdC_n2']],
        'n_epochs': 5,
        'n_repeats': 2,
        'facmac': {
            'critic_lr': 1e-4,
        },
    },
    'grd': {
        'manage_agg_power': False,
        'max_grid_import': 13,
        'max_grid_export': 13,
        'penalty_import': 0.01,
        'penalty_export': 0.01,
        'manage_voltage': False,
        'penalty_overvoltage': 0.1,
        'penalty_undervoltage': 0.1,
        'v_mag_over': 1.001,
        'v_mag_under': 0.999,
        'weight_network_costs': 1,
        'subset_line_losses_modelled': 30
    },

    'car': {
        'c_max': 4
    },
    'syst': {'H': 24}
}

# obs_last_action: False # default was True - Include the agent's last action  (one_hot) in the observation
# obs_agent_id: True # Include the agent's one_hot id in the observation
# rnn_hidden_dim: 1.e+2 # for rnn agent (from 64)
# n_hidden_layers: 1
# exploration_mode: "gaussian"
# hyper_initialization_nonzeros: 0
# lr: 1.e-5
# buffer_size: 5000
# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)

# for type_learning in ['facmac', 'q_learning']:
#     settings['RL']['type_learning'] = type_learning
#     for aggregate_actions in [True, False]:
#         settings['RL']['aggregate_actions'] = aggregate_actions
#         print(f"test {type_learning} aggregate_actions {aggregate_actions}")
#         run(RUN_MODE, settings)
