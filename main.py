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
import numpy as np
# Enter experiment-specific settings in the dictionary below if using different parameters
# to the default parameters in config_files/default_input_parameters, using the example syntax below.

# if 'trajectory' in settings, parameters will be updated in run()
settings = {
    'RL': {
        'type_learning': 'q_learning',
        'n_epochs': 20,
        # 'n_end_test': 10,
        'n_repeats': 10,
        # 'competitive': True,
        'trajectory': False,
        # 'state_space': [['grdC','min_voltage']] * 5 * 2,
        },
    'grd': {
        'manage_voltage': True,
        'reactive_power_for_voltage_control': True,
        'min_voltage': 0.99,
        'penalty_export': False,
        'penalty_overvoltage': [1e-6, 1e2],
        'penalty_undervoltage': [1e-6, 1e2],
        # 'simulate_panda_power_only': True,
    },
    # 'penalty_overvoltage': [1e-6, 1e-4, 1e-2, 1e0, 1e2],

    #     'penalty_import': 0.0,
    #     'penalty_export': 0.00,
    #     'penalise_individual_exports': False,
    #     'manage_voltage': True,
    #     'reactive_power_for_voltage_control': [True] * 5 + [False] * 5,
    #     'quadratic_voltage_penalty': True,
    # },
    'syst': {
        'n_homes': 10,
    },
}

# 1 to run simulation, 2 to plot runs in no_runs
RUN_MODE = 1
no_runs = [1680, 1678]
run(RUN_MODE, settings, no_runs)
