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
       'state_space': [['grdC_n2', 'flexibility'], ['grdC']],
       'n_epochs': 20,
       'n_repeats': 3,
       'type_learning': ['facmac'] * 2,
       'evaluation_methods': [['env_r_c', 'opt_r_c']] * 2,
       'facmac': {
           'hysteretic': True,
           'beta_to_alpha': 0.11,
        },
    },
    'syst': {
       'n_homes': 10,

    }
}

# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)
