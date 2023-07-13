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
        'type_learning': ['q_learning'] * 7 + ['facmac'] * (4 * 2),
        'supervised_loss': [False] * 7 + [True] * 4 + [False] * 4,
        # 'n_epochs': 20,
        # 'n_repeats': 10,
    },
    'syst': {
        'n_homes': [2, 3, 4, 15, 20, 25, 30] + [3, 4, 15, 25] * 2,
        # 'n_homesP': 45,
    },
    # 'grd': {
    #     'simulate_panda_power_only': True,
    # }
}
# n = 16
# settings['RL']['n_grdC_level'] = [3 for _ in range(n)]
# settings['RL']['n_grdC_level'][0: 4] = [1, 3, 5, 20]
# settings['RL']['eps'] = [0.1 for _ in range(n)]
# settings['RL']['eps'][4: 7] = [0.01, 0.5, 0.9]
# settings['RL']['lr'] = [1e-4 for _ in range(n)]
# settings['RL']['lr'][7: 11] = [1e-5, 1e-3, 1e-2, 1e-1]
# settings['RL']['initialise_q'] = ['zeros' for _ in range(n)]
# settings['RL']['initialise_q'][11] = 'random'
# settings['RL']['gamma'] = [0.7 for _ in range(n)]
# settings['RL']['gamma'][12: 14] = [0.1, 0.9]
# settings['RL']['beta_to_alpha'] = [0.5 for _ in range(n)]
# settings['RL']['beta_to_alpha'][14: 16] = [0.1, 0.9]

# 1 to run simulation, 2 to plot runs in no_runs
RUN_MODE = 1
no_runs = [1240]  # if plotting

run(RUN_MODE, settings, no_runs)
