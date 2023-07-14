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
    },
    'syst': {
        'n_homes': 20,
    },

}


# 1 to run simulation, 2 to plot runs in no_runs
RUN_MODE = 1
no_runs = [1350]  # if plotting

run(RUN_MODE, settings, no_runs)
