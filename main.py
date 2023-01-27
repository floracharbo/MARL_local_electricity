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
        # current experiment
        'batch_size': 2,
        'state_space': [['grdC']],
        'n_epochs': 5,
        'n_repeats': 2,
    },
    'syst': {
        'test_on_run': True,
        'n_homes': 3,
        'n_homesP': 0
    },
    'grd': {
        'max_grid_in': 5,
        'max_grid_out': 5,
        'penalty_coefficient_in': 0.001,
        'penalty_coefficient_out': 0.001,
        'manage_agg_power': True,
        'max_grid_import': 13,
        'max_grid_export': 13,
        'penalty_import': 0.01,
        'penalty_export': 0.01,
        'manage_voltage': True,
        'penalty_overvoltage': 0.1,
        'penalty_undervoltage': 0.1,
        'max_voltage': 1.001,
        'min_voltage': 0.999,
        'weight_network_costs': 1,
        'subset_line_losses_modelled': 15,
        'compare_pandapower_optimisation': False,
        'computational_burden_analysis': True
    }
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
