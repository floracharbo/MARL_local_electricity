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
        'aggregate_actions': False,
        'cnn_kernel_size': 2,
        'n_epochs': 20,
        'normalise_states': True,
        'obs_agent_id': True,
        # current experiment
        # grdC_level, hour, car_tau, store0, grdC
        # # avail_car_step, loads_clus_step, loads_fact_step
        # # gen_fact_step, bat_fact_step, loads_cons_step, gen_prod_step
        # # bat_cons_step, dT, dT_next, bat_cons_prev
        # # bat_dem_agg, gen_fact_prev, bat_fact_prev, loads_cons_prev
        # # gen_prod_prev, bat_clus_step, bat_clus_prev, loads_clus_prev
        # # avail_car_prev, loads_fact_prev, day_type, car_cons_step, car_fact_step, bool_flex, store_bool_flex

        # # flexibility
        'state_space': [['flexibility', 'grdC_n2']],
        'type_learning': 'facmac',
        'evaluation_methods': [['env_r_c', 'opt']] * 3,
        'n_repeats': 3,
        'lr': 1e-3,
        'facmac': {'critic_lr': 5e-4, 'batch_size': 5},
        'ou_stop_episode': 1e3,  # for cqmix controller - training  oise goes to zero after this episode
        'start_steps': 100,  # Number of steps for uniform-random action selection, before running real policy. Helps exploration.
        'hyper_initialization_nonzeros': 0.1,
        'n_hidden_layers': 2,
        'n_hidden_layers_critic': 1,
        'trajectory': False,
        'n_cnn_layers': 1,
        'rnn_hidden_dim': 500,
        # 'supervised_loss': True,
        # 'n_start_opt_explo': 5,
        # 'nn_type': 'cnn'
    },
    'ntw': {
        'n': 10,
        # 'nP': 10,
    },
    'syst': {'H': 48}
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

# for type_learning in ['facmac', 'q_learning']:
#     settings['RL']['type_learning'] = type_learning
#     for aggregate_actions in [True, False]:
#         settings['RL']['aggregate_actions'] = aggregate_actions
#         print(f"test {type_learning} aggregate_actions {aggregate_actions}")
#         run(RUN_MODE, settings)
