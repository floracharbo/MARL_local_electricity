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
        'type_learning': 'facmac',
        # 'evaluation_methods': 'env_r_c',
        # 'q_learning': {'alpha': [1e-3, 1e-1]}
    },
    'syst': {
        'n_homes': 1000,
        # 'force_optimisation': True,
    },
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
no_runs = [1216]  # if plotting

run(RUN_MODE, settings, no_runs)
