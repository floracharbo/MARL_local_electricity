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
        'type_learning': 'q_learning',
        'n_epochs': 20,
        'n_repeats': 10,
        'trajectory': False,
        },
    'grd': {
        'manage_voltage': True,
        'reactive_power_for_voltage_control': True,
        'min_voltage': 0.99,
        'penalty_export': False,
        'penalty_overvoltage': [1e-6, 1e2],
        'penalty_undervoltage': [1e-6, 1e2],
    },
    'syst': {
        'n_homes': 10,
    },
}

# 1 to run simulation, 2 to plot runs in no_runs
RUN_MODE = 2
no_runs = [4]
run(RUN_MODE, settings, no_runs)
