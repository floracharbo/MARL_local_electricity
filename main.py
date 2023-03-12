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

settings = {
    'RL': {
       'state_space': ['grdC'] * 2,
       'n_epochs': 5,
       'n_repeats': 2,
       'type_learning': ['q_learning'] * 2,
       'evaluation_methods': [['opt', 'opt_d_d', 'env_r_c']] * 2,
    },
    'syst': {
       'n_homes': [5, 10],
       'force_optimisation': True

    },
    'grd': {
        'manage_agg_power': [True, True],
        'max_grid_import': 15,
        'max_grid_export': 15,
        'penalty_import': 0.01,
        'penalty_export': 0.01,
        'manage_voltage': [True, True],
        'penalty_overvoltage': 0.1,
        'penalty_undervoltage': 0.1,
        'max_voltage': 1.01,
        'min_voltage': 0.99,
        'weight_network_costs': 1,
        'subset_line_losses_modelled': [100, 100],
        'reactive_power_for_voltage_control': False,
        'compare_pandapower_optimisation': [False, True],
        'pf_flexible_homes': [0.95, 0.95],
        'line_losses_method': ['iteration', 'iteration']
    }
}

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)