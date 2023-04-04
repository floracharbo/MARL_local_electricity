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
        'state_space': 'grdC',
        'n_epochs': 20,
        'n_repeats': 10,
        'type_learning': 'facmac',
        'evaluation_methods': 'env_r_c',
        'trajectory': True,
        # 'supervised_loss': True,
        'act_noise': 0.01,
        'lr': 1e-2,
        'obs_agent_id': True,
        'optimizer': 'rmsprop',
        'facmac': {'hysteretic': False},
    },
    'syst': {
        'force_optimisation': True,
        'n_homes': 30,
    }
}

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [1151]  # if plotting

run(RUN_MODE, settings, no_runs)
