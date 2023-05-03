#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:36:21 2020.

@author: floracharbonnier

"""
import datetime
import os.path
import sys
from pathlib import Path

import numpy as np
import yaml


def _command_line_parameters(settings_i):
    obs = []
    args = sys.argv[1:]
    # o is observation, l is learning type,
    # n is number of agents - applied to all repetitions
    for i in range(int(len(args) / 2)):
        key, val = args[i * 2], args[i * 2 + 1]
        if key == '-o':
            obs.append(val)
        elif key == '-l':
            settings_i['RL']['type_learning'] = val
        elif key == '-n':
            settings_i['syst']['n_homes'] = int(val)
        elif key[2:].split('_')[0] == 'facmac':
            key_ = key[2 + len('facmac') + 1:]
            settings_i['RL']['facmac'][key_] = val
        else:
            settings_i['RL'][key[2:]] = val
            print(f"RL['{key[2:]}'] = {val}")
    if len(obs) > 0:
        settings_i['RL']['state_space'] = obs

    return settings_i


def input_paths():
    """Load paths parameters."""
    prm = {}
    # Get the defaults from default.yaml
    with open("config_files/default_input_parameters/paths.yaml", "rb") as file:
        prm["paths"] = yaml.safe_load(file)

    return prm


def _store_initial_parameters(prm):
    prm_preset_entries = {}
    for key in prm.keys():
        prm_preset_entries[key] = {}
        for subkey in prm[key].keys():
            prm_preset_entries[key][subkey] = prm[key][subkey]

    return prm_preset_entries


def _load_parameters(prm, settings):
    if "paths" in prm:
        for info in ["save", "syst", "grd", "loads", "heat", "car", "gen", "RL"]:
            if info == "heat" \
                    and settings is not None \
                    and "heat" in settings \
                    and "file" in settings["heat"] \
                    and settings["heat"]["file"] != "heat":
                info_file = settings["heat"]["file"]
            else:
                info_file = info
            path_param = f"config_files/default_input_parameters/{info_file}.yaml"
            if os.path.exists(path_param):
                with open(path_param, "rb") as file:
                    prm[info] = yaml.safe_load(file)
            else:
                print(f"{path_param} does not exist")
    else:
        print("'paths' not in prm")

    return prm


def _add_settings_to_prm(settings, prm):
    if settings is not None:
        for key in settings.keys():
            for sub_key in settings[key].keys():
                type_learning = \
                    settings["RL"]["type_learning"] \
                    if "RL" in settings and "type_learning" in settings["RL"] \
                    else prm["RL"]["type_learning"]
                if sub_key == type_learning:
                    for sub_sub_key in settings[key][sub_key].keys():
                        prm[key][sub_key][sub_sub_key] = \
                            settings[key][sub_key][sub_sub_key]
                else:
                    prm[key][sub_key] = settings[key][sub_key]

    return prm


def get_settings_i(settings, i):
    """Get run-specific settings from general settings dictionary."""
    # first, get settings from the main.py file
    settings_i = {}
    for key, sub_dict in settings.items():
        settings_i[key] = {}
        for sub_key, val in sub_dict.items():
            if isinstance(val, list):
                settings_i[key][sub_key] = val[i]
            elif isinstance(val, dict):
                settings_i[key][sub_key] = {}
                for subsubkey in val.keys():
                    if isinstance(val[subsubkey], list):
                        settings_i[key][sub_key][subsubkey] = \
                            val[subsubkey][i]
                    else:
                        settings_i[key][sub_key][subsubkey] = val[subsubkey]

            else:
                settings_i[key][sub_key] = val

    # then, override with command line parameters
    settings_i = _command_line_parameters(settings_i)

    return settings_i


def input_params(prm, settings=None):
    """Load input parameters."""
    prm_preset_entries = _store_initial_parameters(prm)
    prm = _load_parameters(prm, settings)
    prm["RL"]["RL_to_save"] = list(prm["RL"].keys())

    # demand / generation factor initialisation for RL data generation
    # https://www.ukpower.co.uk/home_energy/average-household-gas-and-electricity-usage
    # https://www.choice.com.au/home-improvement/energy-saving/solar/articles/how-much-solar-do-i-need
    # https://www.statista.com/statistics/513456/annual-mileage-of-motorists-in-the-united-kingdom-uk/
    prm["syst"]["f0"] = {"loads": 9, "gen": 8, "car": 8}

    # demand / generation cluster initialisation
    # for RL data generation
    prm["syst"]["clus0"] = {"loads": 0, "car": 0}

    if "heat" in prm:
        prm["heat"]["L"] = np.sqrt(prm["heat"]["L2"])

    prm = _add_settings_to_prm(settings, prm)

    # learning parameters
    if prm["RL"]["distr_learning"] == "joint":
        prm["RL"]["difference_bool"] = False

    # revert to pre set entries if there were pre set entries
    for key, value in prm_preset_entries.items():
        for subkey, subvalue in value.items():
            if subkey not in ["maindir", "server"]:
                prm[key][subkey] = subvalue

    return prm


def load_existing_prm(prm, no_run):
    """Load input data for the previous run no_run."""
    prev_paths = prm['paths'].copy()
    input_folder = Path('outputs') / 'results' / f'run{no_run}' / 'inputData'

    # if input data was saved, load input data
    if input_folder.exists():
        if (input_folder / 'lp.npy').exists():
            rl_params = np.load(
                input_folder / 'lp.npy', allow_pickle=True
            ).item()
            if 'n_action' in rl_params and 'n_acitons' not in rl_params:
                rl_params['n_actions'] = rl_params['n_action']
            existing_paths = prm['paths'].copy()
            prm = np.load(
                input_folder / 'syst.npy', allow_pickle=True
            ).item()
            prm['RL'] = rl_params
            prm['paths'] = existing_paths
            prm['save'] = {}
        else:
            prm = np.load(
                input_folder / 'prm.npy', allow_pickle=True
            ).item()
            rl_params = None
        if 'repeats' in prm['RL']:
            prm['RL']['n_repeats'] = prm['RL']['repeats']
        for path in prev_paths:
            if path not in prm['paths']:
                prm['paths'][path] = prev_paths[path]
    else:  # else use current input data
        rl_params, prm = None, None
        print(f"not os.path.exists({input_folder})")

    return rl_params, prm
