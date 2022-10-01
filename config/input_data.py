#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:36:21 2020.

@author: floracharbonnier

"""
import datetime
import os.path

import numpy as np
import yaml


def input_paths():
    """Load paths parameters."""
    prm = {}
    # Get the defaults from default.yaml
    with open("input_parameters/paths.yaml", "r") as f:
        prm['paths'] = yaml.safe_load(f)

    return prm


def input_params(prm, settings=None):
    """Load input parameters."""
    prm_preset_entries = {}
    for key in prm.keys():
        prm_preset_entries[key] = {}
        for subkey in prm[key].keys():
            prm_preset_entries[key][subkey] = prm[key][subkey]

    # saving results
    if 'paths' in prm:
        for e in ['save', 'syst', 'grd', 'ntw', 'loads', 'heat',
                  'bat', 'gen', 'RL']:
            if e == 'heat' \
                    and settings is not None \
                    and 'heat' in settings \
                    and 'file' in settings['heat'] \
                    and 'file' != 'heat':
                e_file = settings['heat']['file']
            else:
                e_file = e
            path_exists = os.path.exists(f'input_parameters/{e}.yaml')
            if path_exists:
                with open(f"input_parameters/{e_file}.yaml", "r") as f:
                    prm[e] = yaml.safe_load(f)
            else:
                print(f"input_parameters/{e_file}.yaml does not exist")
    else:
        print("'paths' not in prm")
    prm['RL']['RL_to_save'] = list(prm['RL'].keys())

    # general system parameters
    for e in ['date0', 'max_dateend']:
        if e in prm['syst'] \
                and not isinstance(prm['syst'][e], datetime.datetime):
            prm['syst'][e] = datetime.datetime(*prm['syst'][e])

    # demand / generation factor initialisation for RL data generation
    # https://www.ukpower.co.uk/home_energy/average-household-gas-and-electricity-usage
    # https://www.choice.com.au/home-improvement/energy-saving/solar/articles/how-much-solar-do-i-need
    # https://www.statista.com/statistics/513456/annual-mileage-of-motorists-in-the-united-kingdom-uk/
    prm['syst']['f0'] = {'loads': 9, 'gen': 8, 'bat': 8}

    # demand / generation cluster initialisation
    # for RL data generation
    prm['syst']['clus0'] = {'loads': 0, 'bat': 0}

    if 'heat' in prm:
        prm['heat']['L'] = np.sqrt(prm['heat']['L2'])

    if settings is not None:
        for key in settings.keys():
            for sub_key in settings[key].keys():
                if sub_key == settings['RL']['type_learning']:
                    for sub_sub_key in settings[key][sub_key].keys():
                        prm[key][sub_key][sub_sub_key] = \
                            settings[key][sub_key][sub_sub_key]
                else:
                    prm[key][sub_key] = settings[key][sub_key]

    if type(prm['RL']['state_space']) is str:
        prm['RL']['state_space'] = [prm['RL']['state_space']]

    for e in ['instant_feedback', 'n_epochs']:
        # has to be 0 or 1 rather than True or False
        # as it will be converted to string later
        prm['RL'][e] = int(prm['RL'][e])

    # learning parameters
    largeQ_bool = False
    if prm['RL']['distr_learning'] == 'joint':
        prm['RL']['difference_bool'] = False

    if 'opt_bool' in prm['RL']:

        if prm['RL']['type_learning'] == 'facmac':
            if 'explo_reward_type' in prm['RL']:
                type_eval_list = prm['RL']['explo_reward_type'] \
                    if type(prm['RL']['explo_reward_type']) is list \
                    else [prm['RL']['explo_reward_type']]
            else:
                methods_combs = ['r_c', 'd_c']
                data_sources = ['env_', 'opt_']
                type_eval_list = []
                for d in data_sources:
                    type_eval_list += [d + mc for mc in methods_combs]
        else:
            if prm['RL']['type_learning'] == 'q_learning':
                data_sources = ['env_'] if not prm['RL']['opt_bool'] \
                    else ['env_', 'opt_']
                methods_combs = ['r_c', 'r_d', 'A_Cc', 'A_c', 'A_d'] \
                    if largeQ_bool else ['r_c', 'r_d', 'A_c', 'A_d']

                if prm['RL']['difference_bool']:
                    add_difference = ['d_Cc', 'd_c', 'd_d'] \
                        if largeQ_bool else ['d_c', 'd_d']
                    methods_combs += add_difference

            elif prm['RL']['type_learning'] in ['DDPG', 'DQN', 'DDQN']:
                data_sources = ['env_'] \
                    if not prm['RL']['opt_bool'] else ['opt_']
                if prm['RL']['distr_learning'] == 'decentralised':
                    methods_combs = ['d_d'] \
                        if prm['RL']['difference_bool'] else ['r_d']
                else:
                    methods_combs = ['d_c'] \
                        if prm['RL']['difference_bool'] else ['r_c']

            methods_combs_opt = ['n_c', 'n_d']
            type_eval_list = []
            for d in data_sources:
                type_eval_list += [d + mc for mc in methods_combs]
            if 'opt_' in data_sources \
                    and prm['RL']['type_learning'] == 'q_learning':
                type_eval_list += ['opt_' + m for m in methods_combs_opt]
        if prm['RL']['opt_bool']:
            type_eval_list += ['opt']
        type_eval_list += ['baseline']

        prm['RL']['type_eval'] = type_eval_list
        print(f"type_eval_list {type_eval_list}")
    # revert to pre set entries if there were pre set entries
    for key in prm_preset_entries.keys():
        for subkey in prm_preset_entries[key].keys():
            if subkey not in ['maindir', 'server']:
                prm[key][subkey] = prm_preset_entries[key][subkey]

    return prm
