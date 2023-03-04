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
        'state_space': ['grdC'] * 6 + [['flexibility', 'grdC_n2']] * 6 * 4,
        'n_epochs': 20,
        'n_repeats': 10,
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

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [282]  # if plotting

run(RUN_MODE, settings, no_runs)

