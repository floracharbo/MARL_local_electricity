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
# settings = {
#     'RL': {
#        'state_space': 'grdC',
#        'n_epochs': 20,
#        'type_learning': 'facmac',
#         'trajectory': True,
#         'n_repeats': 10,
#         'evaluation_methods': 'env_r_c',
#         'aggregate_actions': False,
#     },
#     'syst': {
#        'n_homes': 10,
#         'clus_dist_share': 0.999,
#         'f0': {'gen': 8.012, 'loads': 9.459, 'car': 8.893}
#     },
#     'grd': {'simulate_panda_power_only': False}
# }

# n = 23
# settings['RL']['facmac'] = {}
# settings['RL']['facmac']['lr'] = 1e-1
#
# settings['RL']['facmac']['critic_lr'] = [5e-4 for _ in range(n)]
# settings['RL']['facmac']['critic_lr'][1: 4] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# # settings['RL']['facmac']['hysteretic'] = [True for _ in range(n)]
# # settings['RL']['facmac']['hysteretic'][5] = False
# settings['RL']['facmac']['gamma'] = [0.85 for _ in range(n)]
# settings['RL']['facmac']['gamma'][1: 3] = [0.5, 0.99]
# settings['RL']['act_noise'] = [0.01 for _ in range(n)]
# settings['RL']['act_noise'][3] = 1e-4
# settings['RL']['buffer_size'] = [5000 for _ in range(n)]
# settings['RL']['buffer_size'][4: 6] = [10, 5e4]
# settings['RL']['optimizer'] = ['rmsprop' for _ in range(n)]
# settings['RL']['optimizer'][6] = 'adam'
# settings['RL']['hyper_initialization_nonzeros'] = [0.1 for _ in range(n)]
# settings['RL']['hyper_initialization_nonzeros'][7: 9] = [0.01, 1]
# settings['RL']['init_weights_zero'] = [True for _ in range(n)]
# settings['RL']['init_weights_zero'][9: 11] = [False, False]
# settings['RL']['rnn_hidden_dim'] = [1e3 for _ in range(n)]
# settings['RL']['rnn_hidden_dim'][11: 13] = [1e2, 1e4]
# settings['RL']['n_cnn_layers'] = [1 for _ in range(n)]
# settings['RL']['n_cnn_layers'][13] = 2
# settings['RL']['n_cnn_layers_critic'] = [1 for _ in range(n)]
# settings['RL']['n_cnn_layers_critic'][14] = 2
# settings['RL']['rnn_hidden_dim'][15] = 64
# settings['RL']['n_hidden_layers'] = [2 for _ in range(n)]
# settings['RL']['n_hidden_layers'][16: 17] = [1, 3]
# settings['RL']['n_hidden_layers_critic'] = [1 for _ in range(n)]
# settings['RL']['n_hidden_layers_critic'][17] = 2
# settings['RL']['facmac']['batch_size'] = [10 for _ in range(n)]
# settings['RL']['facmac']['batch_size'][18] = [30]




# settings['RL']['facmac']['beta_to_alpha'] = [0.1 for _ in range(n)]
# settings['RL']['facmac']['beta_to_alpha'][8: 10] = [0.01, 0.8]

settings = {
    'RL': {
       'n_epochs': 20,
       'type_learning': 'q_learning',
        'act_noise': 0.01,
        'facmac': {
            'batch_size': 10,
        },
        'n_repeats': 10,
       # 'evaluation_methods': 'env_r_c',
        'n_other_states': 3,
        'aggregate_actions': False,
        'q_learning': {
            'hysteretic': True,
        }
    },
    'syst': {
       'n_homes': 10,
        'clus_dist_share': 0.999,
        'f0': {'gen': 8.012, 'loads': 9.459, 'car': 8.893}
    },
    'grd': {'simulate_panda_power_only': False},
}
n = 9
settings['RL']['lr'] = [1e-2 for _ in range(n)]
settings['RL']['lr'][0] = 1e-1
settings['RL']['q_learning']['gamma'] = [0.7 for _ in range(n)]
settings['RL']['q_learning']['gamma'][1] = 0.5
settings['RL']['q_learning']['eps'] = [0.5 for _ in range(n)]
settings['RL']['q_learning']['eps'][2:4] = [0.99, 1]
settings['RL']['n_discrete_actions'] = [3 for _ in range(n)]
settings['RL']['n_discrete_actions'][4] = 4
settings['RL']['n_grdC_level'] = [3 for _ in range(n)]
settings['RL']['n_grdC_level'][5] = 1
settings['RL']['q_learning']['beta_to_alpha'] = [0.5 for _ in range(n)]
settings['RL']['q_learning']['beta_to_alpha'][6] = 0.01
settings['RL']['state_space'] = ['grdC' for _ in range(n)]
settings['RL']['state_space'][7: 9] = [None, 'flexibility']


# 1 to run simulation, 2 to plot runs in no_runs, 3 plots results vs n_ag
RUN_MODE = 1
no_runs = [823]  # if plotting

run(RUN_MODE, settings, no_runs)
