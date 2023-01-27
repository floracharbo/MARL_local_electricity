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

# Inputs
settings = {
    'heat': {'file': 'heat2'},

    'RL': {
        # current experiment
        'state_space': [['grdC_n2', 'flexibility']] * 5,
        'n_epochs': 20,
        'n_repeats': 3,
        'type_learning': ['facmac'],
        'evaluation_methods': ['env_r_c', 'opt_d_d'] * 5,
        'facmac': {'hysteretic': True, 'beta_to_alpha': 0.1}
    },
    'syst': {
        'n_homes': 10
    },
    'grd': {
        'manage_agg_power': False,
        'manage_voltage': False,
    }
}

# obs_last_action: False # default was True - Include the agent's last action  (one_hot) in the observation
# obs_agent_id: True # Include the agent's one_hot id in the observation
# rnn_hidden_dim: 1.e+2 # for rnn agent (from 64)
# n_hidden_layers: 1
# exploration_mode: "gaussian"
# hyper_initialization_nonzeros: 0
# lr: 1.e-5
# buffer_size: 5000
# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)
