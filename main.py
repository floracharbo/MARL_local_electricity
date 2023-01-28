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

# Enter experiment-specific settings in config_files/experiment_settings.yaml if using different parameters
# to the default parameters in config_files/default_input_parameters
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, no_runs)
