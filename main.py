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
        'type_learning': 'facmac',
        'explo_reward_type': [['env_r_c', 'env_d_c', 'opt_r_c']] * 2,
        'gamma': {'q_learning': 0.99, 'facmac': 0.99},
        'aggregate_actions': False,
        'mixer': 'qmix',

        # current experiment
        'batch_size': [2] * 2,
        'rnn_hidden_dim': [1e2] * 2,
        'state_space': [['grdC', 'bat_dem_agg', 'avail_EV_step']] * 2,
        'n_epochs': [5] * 2,
        'n_repeats': 2,
        'lr': 1e-4,
        'facmac': {
            'critic_lr': 1e-4,
        },
    },

    'ntw': {
        'n': 3
    },

    'grd': {
        'manage_agg_power': [True, False] * 1,
        'max_grid_in': 15,
        'max_grid_out': 15,
        'penalty_coefficient_in': 0.01 ,
        'penalty_coefficient_out': 0.01 
        #'manage_agg_power' : True
    }
}

# on server check centralised opts false - next lr sensitivity
# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [610]  # if plotting

run(RUN_MODE, settings)

#for type_learning in ['facmac', 'q_learning']:
#    settings['RL']['type_learning'] = type_learning
#    for aggregate_actions in [True, False]:
#        settings['RL']['aggregate_actions'] = aggregate_actions
#        for manage_agg_power in [True, False]:
#            settings['grd']['manage_agg_power'] = manage_agg_power
#            print(f"test {type_learning} aggregate_actions {aggregate_actions} manage_agg_power {manage_agg_power}")
#            run(RUN_MODE, settings)
