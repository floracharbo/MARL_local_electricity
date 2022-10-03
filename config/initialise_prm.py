#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:08:33 2020.

@author: Flora Charbonnier
"""
import datetime
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
from gym import spaces

from config.generate_colors import generate_colors
from config.get_heat_coeffs import get_heat_coeffs
from learners.facmac.components.transforms import OneHot
from utils.userdeftools import (distr_learning, initialise_dict, reward_type,
                                str_to_int)

sys.path.append(
    '/Users/floracharbonnier/OneDrive - Nexus365/DPhil/Python/'
    'Phase2/GettingData')


def _facmac_initialise(prm):
    """
    Compute parameters relating to the FACMAC reinforcement learning.

    inputs:
    prm:
        here the RL and ntw entries are relevant

    outputs:
    rl:
        correpsonding to prm['RL']; with updated parameters
    """
    rl = prm['RL']
    rl['n_agents'] = prm['ntw']['n']

    rl['obs_shape'] = len(rl['state_space'])
    rl['state_shape'] = rl['obs_shape'] * rl['n_agents']

    if not rl['server']:
        rl['use_cuda'] = False
    rl['device'] = "cuda" if rl['use_cuda'] else "cpu"
    if rl['discretize_actions']:
        action_space = spaces.Discrete(rl['n_discrete_actions'])
    else:
        action_space = spaces.Box(
            low=np.array(rl['low_action'], dtype=np.float32),
            high=np.array(rl['high_action'], dtype=np.float32),
            shape=(rl['dim_actions'],), dtype=np.float32)
    rl['action_space'] = [action_space] * rl['n_agents']

    ttype = th.FloatTensor if not rl['use_cuda'] else th.cuda.FloatTensor
    mult_coef_tensor = ttype(rl['n_agents'], rl['dim_actions'])
    action_min_tensor = ttype(rl['n_agents'], rl['dim_actions'])
    if not rl['discretize_actions']:
        for _aid in range(rl['n_agents']):
            for _actid in range(rl['dim_actions']):
                _action_min = rl['action_space'][_aid].low[_actid]
                _action_max = rl['action_space'][_aid].high[_actid]
                mult_coef_tensor[_aid, _actid] = \
                    (_action_max - _action_min).item()
                action_min_tensor[_aid, _actid] = _action_min.item()

    else:
        print("need to set up rl['type_learning'] "
              "== 'facmac and rl['discretize_actions']")

    rl['actions2unit_coef'] = mult_coef_tensor
    rl['actions2unit_coef_cpu'] = mult_coef_tensor.cpu()
    rl['actions2unit_coef_numpy'] = mult_coef_tensor.cpu().numpy()
    rl['actions_min'] = action_min_tensor
    rl['actions_min_cpu'] = action_min_tensor.cpu()
    rl['actions_min_numpy'] = action_min_tensor.cpu().numpy()
    rl['avail_actions'] = np.ones((rl['n_agents'], rl['dim_actions']))

    def _actions_to_unit_box(actions, rl):
        if isinstance(actions, np.ndarray):
            return rl['actions2unit_coef_numpy'] * actions \
                + rl['actions_min_numpy']
        elif actions.is_cuda:
            return rl['actions2unit_coef'] * actions + rl['actions_min']
        else:
            return rl['actions2unit_coef_cpu'] * actions \
                + rl['actions_min_cpu']

    def _actions_from_unit_box(actions, rl):
        if isinstance(actions, np.ndarray):
            return th.div((actions - rl['actions_min_numpy']),
                          rl['actions2unit_coef_numpy'])
        elif actions.is_cuda:
            return th.div((actions - rl['actions_min']),
                          rl['actions2unit_coef'])
        else:
            return th.div((actions - rl['actions_min_cpu']),
                          rl['actions2unit_coef_cpu'])

    # make conversion functions globally available
    rl['actions2unit'] = _actions_to_unit_box
    rl['unit2actions'] = _actions_from_unit_box
    rl['actions_dtype'] = np.float32
    action_dtype = th.long if not rl['actions_dtype'] == np.float32 \
        else th.float
    if not rl['discretize_actions']:
        actions_vshape = rl['dim_actions']
    elif all([isinstance(act_space, spaces.Tuple)
              for act_space in rl['action_spaces']]):
        actions_vshape = 1 if not rl['actions_dtype'] == np.float32 else \
            max([i.spaces[0].shape[0] + i.spaces[1].shape[0]
                 for i in rl['action_spaces']])
    # Default/Base scheme
    rl['scheme'] = {
        "state": {"vshape": rl["state_shape"]},
        "obs": {"vshape": rl["obs_shape"], "group": "agents"},
        "actions":
            {"vshape": (actions_vshape,),
             "group": "agents",
             "dtype": action_dtype},
        "avail_actions":
            {"vshape": (rl["dim_actions"],),
             "group": "agents",
             "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    rl['groups'] = {
        "agents": rl['n_agents']
    }

    if not rl['actions_dtype'] == np.float32:
        rl['preprocess'] = {
            "actions": ("actions_onehot", [OneHot(out_dim=rl['dim_actions'])])
        }
    else:
        rl['preprocess'] = {}

    rl['obs_shape'] = len(rl['state_space'])
    rl['state_shape'] = rl['obs_shape'] * rl['n_agents']
    rl['env_info'] = {"state_shape": rl['state_shape'],
                      "obs_shape": rl['obs_shape'],
                      "n_actions": rl['dim_actions'],
                      "n_agents": rl['n_agents'],
                      "episode_limit": rl['episode_limit'],
                      "actions_dtype": np.float32
                      }

    return prm


def _update_paths(paths, prm, no_run):
    """
    Compute paths needed for later parameter/data loading.

    inputs:
    prm:
        here the paths, RL, save entries are relevant

    outputs:
    paths:
        correpsonding to prm['paths']; with updated parameters
    """
    paths['folder_run'] = Path('results') / f'run{no_run}'
    paths['record_folder'] = paths['folder_run'] / 'record'
    prm['paths']['fig_folder'] = paths['folder_run'] / 'figures'
    paths['res_path'] = Path(paths['res_folder'])
    paths['input_folder'] = Path(paths['input_folder'])
    paths['factors_path'] = paths['input_folder'] / paths['factors_folder']

    return paths


def _load_data_dictionaries(loads, gen, bat, paths, syst):
    # load data into dictionaries
    dicts = [loads, gen, bat]
    dict_label = ['dem', 'gen', 'EV']
    factors_path = paths['factors_path']
    for d in range(len(dicts)):
        dicts[d]['listfactors'] = np.load(
            factors_path / f'list_factors_{dict_label[d]}.npy',
            allow_pickle=True
        )
        if dict_label[d] == 'gen':  # no distinction for day types / no cluster
            dicts[d]['f_prms'] = np.load(factors_path / 'prm_gen.npy')[0]
            dicts[d]['f_mean'] = np.load(factors_path / 'meandistr_gen.npy')
        else:
            if dict_label[d] == 'dem':  # get factor parameters
                dicts[d]['f_prms'], dicts[d]['f_mean'] = {}, {}
                for dtt in syst['labels_day_trans']:
                    dicts[d]['f_prms'][dtt] = np.load(
                        factors_path / f'prm_{dict_label[d]}_{dtt}.npy')[0]
                    dicts[d]['f_mean'][dtt] = np.load(
                        factors_path / f'meandistr_{dict_label[d]}_{dtt}.npy')
            for p in ['pclus', 'ptrans']:
                dicts[d][p] = np.reshape(
                    np.load(paths['input_folder'] / paths['clusfolder']
                            / f'{p}_{dict_label[d]}.npy',
                            allow_pickle=True), 1)[0]
            dicts[d]['n_clus'] = len(dicts[d]['pclus']['wd'])

    return dicts


def _load_profiles(paths, bat, syst, loads, gen):
    # load EV profiles and availability
    # (mmap_mode = 'r' means not actually loaded, but elements accessible)

    profiles = {'bat': {}}
    for data in ['cons', 'avail']:
        profiles['bat'][data] = initialise_dict(syst['labels_day'])
        for day_type in syst['labels_day']:
            profiles['bat'][data][day_type] = \
                initialise_dict(range(bat['n_clus']))

    bat['n_prof'] = initialise_dict(syst['labels_day'])

    for data in ['cons', 'avail']:
        path = paths['input_folder'] / paths[f'ev_{data}_folder']
        files = os.listdir(path)
        for file in files:
            if file[0] != '.':
                dt = file[-9:-7]
                c = int(file[-5:-4])
                profiles['bat'][data][dt][c] = \
                    np.load(path / file, mmap_mode='r')
                if len(np.shape(profiles['bat'][data][dt][c])) == 1:
                    new_shape = (1, len(profiles['bat'][data][dt][c]))
                    profiles['bat'][data][dt][c] = np.reshape(
                        profiles['bat'][data][dt][c], new_shape)

    for day_type in syst['labels_day']:
        bat['n_prof'][day_type] = [len(profiles['bat']['cons'][day_type][clus])
                                   for clus in range(bat['n_clus'])]

    prof_path = paths['input_folder'] / paths['profiles_folder']
    profiles['loads'] = {}
    loads['n_prof'] = {}
    for day_type in syst['labels_day']:
        profiles['loads'][day_type] = \
            [np.load(prof_path / f'bank_normdem_c{clus}_{day_type}.npy',
                     mmap_mode='r') for clus in range(syst['n_loads_clus'])]
        loads['n_prof'][day_type] = [len(profiles['loads'][day_type][clus])
                                     for clus in range(loads['n_clus'])]
    loads['perc'] = np.load(paths['input_folder'] / paths['loads_cons_perc'])
    bat['perc'] = np.load(paths['input_folder'] / paths['EV_perc'])

    # PV generation bank and month
    gen_profs = np.load(prof_path / 'normbank_gen.npy', mmap_mode='r')
    gen_months = np.load(prof_path / 'months_gen.npy', mmap_mode='r')
    profiles['gen'] = {}
    for month in range(12):
        profiles['gen'][month] = []
        for i, [gen_month, gen_prof] in enumerate(zip(gen_months, gen_profs)):
            if gen_month == month + 1:
                profiles['gen'][month].append(
                    [gen_data if gen_data > 0 else 0 for gen_data in gen_prof])
    gen['n_prof'] = [len(profiles['gen'][m]) for m in range(12)]
    list_fact_gen = np.load(paths['input_folder'] / 'factorsstats'
                            / 'list_factors_gen.npy')
    month = syst['date0'].month
    non_0_norm_gen = [x for x in profiles['gen'][month] if x != 0]
    non_0_fact_gen = [x for x in list_fact_gen if x != 0]
    perc_nom_gen = [np.percentile(non_0_norm_gen, i) for i in range(0, 101)]
    perc_f_gen = [np.percentile(non_0_fact_gen, i) for i in range(0, 101)]
    gen['perc'] = [perc_nom_gen[i] * perc_f_gen[i] for i in range(0, 101)]

    return profiles, bat, loads, gen


def _update_bat_prm(prm):
    """
    Compute parameters relating to the EV battery for the experiments.

    inputs:
    prm:
        here the bat, ntw, paths and syst entries are relevant

    outputs:
    bat:
        correpsonding to prm['bat']; with updated parameters
    """
    bat, ntw, paths, syst = \
        [prm[key] for key in ['bat', 'ntw', 'paths', 'syst']]

    bat['C'] = bat['dep']  # GBP/kWh storage costs

    # have list of EV capacities based on capacity and ownership inputs
    if 'own_EV' in bat:
        bat['cap'] = bat['cap'] if isinstance(bat['cap'], list) \
            else [bat['cap'] for _ in range(ntw['n'])]
        bat['own_EV'] = [1 for _ in range(ntw['n'])] \
            if bat['own_EV'] == 1 else bat['own_EV']
        bat['cap'] = [c if o == 1 else 0
                      for c, o in zip(bat['cap'], bat['own_EV'])]

    bat['f_prob'], bat['mid_fs'], bat['bracket_fs'] = [{} for _ in range(3)]
    if isinstance(bat['cap'], (int, float)):
        bat['cap'] = [bat['cap'] for _ in range(ntw['n'])]
    if 'labels_day_trans' in syst:
        for dtt in syst['labels_day_trans']:
            bat['f_prob'][dtt] = np.load(
                paths['factors_path']
                / f"prob_f1perf0_{bat['intervals_fprob']}_{dtt}.npy")
            bat['mid_fs'][dtt] = np.load(
                paths['factors_path']
                / f"midxs_f0f1probs_n{bat['intervals_fprob']}_{dtt}.npy")
            bat['bracket_fs'][dtt] = np.load(
                paths['factors_path']
                / f"xs_f0f1probs_n{bat['intervals_fprob']}_{dtt}.npy")

    # battery characteristics
    bat['min_charge'] = [bat['cap'][a] * max(bat['SoCmin'], bat['baseld'])
                         for a in range(ntw['n'])]
    bat['store0'] = [bat['SoC0'] * bat['cap'][a] for a in range(ntw['n'])]
    if 'capP' not in bat:
        bat['capP'] = [bat['cap'][0] for _ in range(ntw['nP'])]
    bat['store0P'] = [bat['SoC0'] * bat['capP'][a] for a in range(ntw['nP'])]
    bat['min_chargeP'] = [bat['capP'][a] * max(bat['SoCmin'], bat['baseld'])
                          for a in range(ntw['nP'])]
    bat['phi0'] = np.arctan(bat['c_max'])

    return bat


def _update_rl_prm(prm, initialise_all):
    """
    Compute parameters relating to RL experiments.

    inputs:
    prm:
        here the RL and ntw entries are relevant
    initialise_all:
        whether we need to initialise all parameters
        or just the minimum

    outputs:
    rl:
        correpsonding to prm['RL']; with updated parameters
    """
    rl, syst, ntw, heat = [prm[key] for key in ['RL', 'syst', 'ntw', 'heat']]

    # learning parameter variables
    rl['ncpu'] = mp.cpu_count() if rl['server'] else 10
    rl['dim_actions'] = 1 if rl['aggregate_actions'] else 3

    if 'n_epochs' not in rl and 'n_learn_cycle' in rl:
        # for compatibility with old code versions
        rl['n_epochs'] = rl['n_learn_cycle']
    rl['T_decay_param'] = (rl['Tend'] / rl['T0']) ** (1 / rl['n_epochs'])
    rl['tot_learn_cycles'] = rl['n_epochs'] * rl['ncpu'] \
        if rl['parallel'] else rl['n_epochs']
    for key in ["n_epochs", "n_repeats"]:
        rl[key] = int(rl[key])
    rl['lr'] = float(rl['lr'])
    if isinstance(rl['default_action'], (int, float)):
        for p in ["P", ""]:
            rl['default_action' + p] = [
                [rl['default_action'] for _ in range(rl['dim_actions'])]
                for _ in range(ntw['n' + p])
            ]

    # methods lists
    if rl['competitive']:  # there can be no centralised learning
        rl['type_eval'] = [t for t in rl['type_eval']
                           if t in ['opt', 'baseline']
                           or (reward_type(t) != 'A'
                               and distr_learning(t) != 'c')]

    rl['type_explo'] = [t for t in rl['type_eval']
                        if not (t[0:3] == 'opt' and len(t) > 3)]

    if rl['type_learning'] == 'facmac':
        # limit available exploration / evaluation: no difference rewards;
        valid_explo = ['env_r_c', 'env_d_c', 'opt', 'baseline']
        valid_eval \
            = ['env_r_c', 'env_d_c', 'opt_r_c', 'opt', 'baseline', 'random']
        rl['type_explo'] = [t for t in rl['type_explo'] if t in valid_explo]
        rl['type_eval'] = [t for t in rl['type_eval'] if t in valid_eval]

    rl['eval_action_choice'] = [t for t in rl['type_eval']
                                if t not in ['baseline', 'opt']]

    rl['type_Qs'] = rl['eval_action_choice'] \
        + [ac + '0' for ac in rl['eval_action_choice']
           if len(ac.split('_')) >= 3
           and (reward_type(ac) == 'A' or distr_learning(ac)[0] == 'C')
           ]
    rl['start_end_eval'] = int(rl['share_epochs_start_end_eval']
                               * rl['n_epochs'])
    rl['n_all_epochs'] = rl['n_epochs'] + rl['n_end_test']

    # obtain decay parameter
    type_learning = rl['type_learning']
    if type_learning in ['DDQN', 'DQN', 'q_learning']:
        if rl[type_learning]['control_eps'] == 1 \
                and 'baseline' not in rl['type_eval']:
            rl['type_eval'].append('baseline')
        if rl[type_learning]['epsilon_end'] == 'best':
            # take result of sensitivity analysis
            if rl[type_learning]['control_eps'] < 2:
                rl[type_learning]['epsilon_end'] \
                    = rl[type_learning]['best_eps_end'][
                    rl[type_learning]['control_eps']]
            else:
                rl[type_learning]['epsilon_end'] = 1e-2  # not going to use it

        if isinstance(rl[type_learning]['epsilon_end'], float):
            rl[type_learning]['epsilon_decay_param'] = \
                (rl[type_learning]['epsilon_end']
                 / rl[type_learning]['epsilon0']) \
                ** (1 / (rl[type_learning]['end_decay']
                         - rl[type_learning]['start_decay']))
        else:
            rl[type_learning]['epsilon_decay_param'] = {}
            for type_explo in rl[type_learning]['epsilon_end'].keys():
                rl[type_learning]['epsilon_decay_param'][type_explo] \
                    = (rl[type_learning]['epsilon_end'][type_explo]
                       / rl[type_learning]['epsilon0']) \
                    ** (1 / rl['tot_learn_cycles'])

    for key in ['epsilon_end', 'T', 'tauMT', 'tauLT',
                'control_window_eps', 'epsilon_decay_param']:
        if key in rl[type_learning]:
            if isinstance(rl[type_learning][key], (float, int)):
                # if only one value of eps end is given,
                # give them all methods the same eps value
                var = rl[type_learning][key]
                rl[type_learning][key] = {}
                for type_eval in rl['eval_action_choice']:
                    rl[type_learning][key][type_eval] = var
            elif key == 'control_window_eps':
                window_per_method_specified = \
                    len(list(rl[type_learning][key].keys())) > 0
                if window_per_method_specified:
                    specified_per_reward_only = \
                        len(list(rl[type_learning][key].keys())[0].split('_')
                            ) == 1
                    if specified_per_reward_only:
                        for type_eval in rl['eval_action_choice']:
                            rl[type_learning][key][type_eval] = \
                                rl[type_learning][key][reward_type(type_eval)]
    rl['dim_states'] = len(rl['state_space'])

    if 'trajectory' not in rl:
        rl['trajectory'] = False
    if rl['distr_learning'] == 'joint':
        rl['dim_actions'] *= rl['n_agents']
        rl['trajectory'] = False
    if rl['trajectory']:
        for key in ['dim_states', 'dim_actions']:
            rl[key] *= syst['prm']['N']
    if rl['type_learning'] == 'DDPG':
        rl['instant_feedback'] = True
    for key in ['batch_size', 'buffer_capacity']:
        if key in rl[type_learning]:
            rl[type_learning][key] = int(rl[type_learning][key])
    rl['rnn_hidden_dim'] = int(rl['rnn_hidden_dim'])
    if rl['competitive'] and rl['distr_learning'] != 'decentralised':
        print("changing distr_learning to decentralised as rl['competitive']")
        rl['distr_learning'] = 'decentralised'
    if rl['competitive'] and rl['trajectory']:
        print("cannot use trajectory with competitive setting")
        rl['trajectory'] = False

    if initialise_all and heat is not None:
        rl['statecomb_str'] = ''
        if rl['state_space'] is None:
            rl['statecomb_str'] = 'None_'
        else:
            rl['statecomb_str'] = ''
            for state in rl['state_space']:
                str_state = 'None' if state is None else str(state)
                rl['statecomb_str'] += str_state + '_'
        rl['statecomb_str'] = rl['statecomb_str'][:-1]

    # %% do not save batches if too many of them!
    if rl['n_repeats'] * rl['n_epochs'] * (rl['n_explore'] + 1) > 200:
        rl['plotting_batch'] = False

    if not rl['aggregate_actions']:
        rl['low_action'] = rl['low_actions']
        rl['high_action'] = rl['high_actions']

    return rl


def _seed_save_paths(prm):
    """
    Get strings and seeds which will be used to identify runs.

    inputs:
    prm:
        here the rl, heat, syst, ntw, paths, bat entries are relevant

    output:
    rl, paths with updated entries

    """
    rl, heat, syst, ntw, paths = \
        [prm[key] for key in ['RL', 'heat', 'syst', 'ntw', 'paths']]

    paths['opt_res_file'] = \
        f"_D{syst['D']}_{syst['solver']}_Uval{heat['Uvalues']}" \
        f"_ntwn{ntw['n']}_nP{ntw['nP']}"
    if 'file' in heat and heat['file'] != 'heat.yaml':
        paths['opt_res_file'] += f"{heat['file']}"
    paths['seeds_file'] = 'seeds/seeds' + paths['opt_res_file']
    if rl['deterministic'] == 2:
        for file in ['opt_res_file', 'seeds_file']:
            paths[file] += '_noisy'
    for file in ['opt_res_file', 'seeds_file']:
        paths[file] += f"_r{rl['n_repeats']}_epochs{rl['n_epochs']}" \
            f"_explore{rl['n_explore']}_endtest{rl['n_end_test']}"
    if prm['syst']['change_start']:
        paths['opt_res_file'] += '_changestart'

    # eff does not matter for seeds, but only for res
    if prm['bat']['efftype'] == 1:
        paths['opt_res_file'] += '_eff1'
    for file in ['opt_res_file', 'seeds_file']:
        paths[file] += '.npy'

    if os.path.exists(paths['seeds_file']):
        rl['seeds'] = np.load(paths['seeds_file'], allow_pickle=True).item()
    else:
        rl['seeds'] = {'P': [], '': []}
    rl['init_len_seeds'] = {}
    for passive_str in ['', 'P']:
        rl['init_len_seeds'][passive_str] = len(rl['seeds'][passive_str])

    return rl, paths


def _update_grd_prm(prm):
    """
    Update the parameters relating to grid information for the run.

    namely the carbon intensity and pricing historical data used

    input:
    prm:
        here the paths, grd, and syst entries are relevant

    output:
    grd:
        with updated information
    """
    paths, grd, syst = [prm[key] for key in ['paths', 'grd', 'syst']]
    # wholesale
    wholesale_path = paths['input_folder'] / paths['wholesale']
    wholesale = [x * 1e-3 for x in np.load(wholesale_path)]  # p/kWh -> GBP/kWh
    grd['wholesale_all'] = wholesale
    carbon_intensity_path = paths['input_folder'] / paths['carbon_intensity']

    # gCO2/kWh to tCO2/kWh
    grd['cintensity_all'] = np.load(
        carbon_intensity_path, allow_pickle=True) * 1e-6

    #  carbon intensity
    grd['Call'] = [price + carbon * syst['co2tax']
                   for price, carbon in zip(wholesale, grd['cintensity_all'])]
    grd['perc'] = [np.percentile(grd['Call'], i) for i in range(0, 101)]

    return grd


def initialise(prm, no_run, initialise_all=True):
    """
    Compute useful variables from input data.

    inputs:
    prm:
        dictionary of run parameters;
        has attributes bat, grd, loads, ntw, prm, gen,
        rl (from inputs files)
    no_run:
        no of current run for folder naming/recording purposes
    initialise_all:
        boolean, false if we are only initialising
        minimum required data
        e.g. to plot the summary plot rewards vs agents
        where we don't need everything

    outputs:
    prm:
        updated parameters
    profiles:
        the battery, generation and loads profiles
        to input to the environment
    """
    prm_entries = ['paths', 'syst', 'grd', 'bat', 'ntw', 'loads',
                   'gen', 'save', 'heat']
    [paths, syst, grd, bat, ntw, loads, gen, save, heat] = \
        [prm[key] if key in prm else None for key in prm_entries]
    if 'H' not in syst:
        syst['H'] = 24
    syst['N'] = syst['D'] * syst['H']
    syst['duration'] = datetime.timedelta(days=syst['D'])
    ntw['n_all'] = ntw['n'] + ntw['nP']
    if syst['play_sound']:
        if syst["play_sound"] and os.file.exist(syst["sound_file"]):
            syst["play_sound"] = False

    for p in ["", "P"]:
        if 'own_PV' + p in gen:
            gen['own_PV' + p] = [1 for _ in range(ntw['n' + p])] \
                if gen['own_PV' + p] == 1 else gen['own_PV' + p]
        if 'own_heat' + p in heat:
            heat['own_heat' + p] = [1 for _ in range(ntw['n' + p])] \
                if heat['own_heat' + p] == 1 else heat['own_heat' + p]
        for e in ['own_loads' + p, 'own_flex' + p]:
            if e in loads:
                loads[e] = [1 for _ in range(ntw['n' + p])] \
                    if loads[e] == 1 else loads[e]

    # update paths and parameters from inputs
    if paths is not None:
        paths = _update_paths(paths, prm, no_run)

        if initialise_all:
            loads, gen, bat = _load_data_dictionaries(
                loads, gen, bat, paths, syst)
            syst['datetimes'] = \
                np.load(paths['input_folder'] / paths['datetimes'],
                        allow_pickle=True)

            # dates
            dates_path = paths['input_folder'] / paths['dates']
            dates = np.load(dates_path)
            dates = [str_to_int(date) for date in dates]
            syst['date0_dates'] = \
                datetime.datetime(day=dates[0][0],
                                  month=dates[0][1],
                                  year=dates[0][2])

            grd = _update_grd_prm(prm)
            profiles, bat, loads, gen = _load_profiles(
                paths, bat, syst, loads, gen)

            # loads and cluster prob and profile banks (mmap = 'r')
            syst['n_int_per_hr'] = int(syst['H'] / 24)
            # duration of time interval in hrs
            syst['dt'] = 1 / syst['n_int_per_hr']

            loads['share_flex'], loads['max_delay'] = loads['flex']
            loads['share_flexs'] = \
                [0 if not loads['own_flex'][a]
                 else loads['share_flex'] for a in range(ntw['n'])]
        else:
            profiles = None
    else:
        profiles = None

    # EV avail, type, factors
    bat = _update_bat_prm(prm)

    # grid loss
    grd['loss'] = grd['R'] / (grd['V'] ** 2)

    rl = _update_rl_prm(prm, initialise_all)
    rl, paths = _seed_save_paths(prm)

    if rl['type_learning'] == 'facmac':
        prm = _facmac_initialise(prm)
    if rl['type_learning'] in ["q_learning", "DQN"]:
        prm['RL']['type_env'] = 'discrete'

    # calculate heating coefficients for recursive expression
    # based on input data
    if initialise_all and heat is not None:
        prm['heat'] = get_heat_coeffs(heat, ntw, syst, loads, paths)

    # list all possible for consistent colors ordering
    all_type_eval = []
    reward_structure_combs = \
        ['r_c', 'r_d', 'A_Cc', 'A_c', 'A_d', 'd_Cc', 'd_c', 'd_d']
    for experience_source in ['env_', 'opt_']:
        all_type_eval += [experience_source + rs_comb
                          for rs_comb in reward_structure_combs]
    all_type_eval += ['opt', 'opt_n_c', 'opt_n_d']

    save, prm = generate_colors(save, prm, all_type_eval)

    return prm, profiles


def load_existing_prm(prm, no_run):
    """Load input data for the previous run no_run."""
    prev_paths = prm['paths'].copy()
    input_folder = Path('results') / f'run{no_run}' / 'inputData'

    # if input data was saved, load input data
    if os.path.exists(input_folder):
        if os.path.exists(input_folder / 'lp.npy'):
            lp = np.load(input_folder / 'lp.npy',
                         allow_pickle=True).item()
            if 'n_action' in lp and 'n_acitons' not in lp:
                lp['n_actions'] = lp['n_action']
            existing_paths = prm['paths'].copy()
            prm = np.load(input_folder / 'syst.npy',
                          allow_pickle=True).item()
            prm['RL'] = lp
            prm['paths'] = existing_paths
            prm['save'] = {}
        else:
            prm = np.load(input_folder / 'prm.npy',
                          allow_pickle=True).item()
            lp = None
        if 'repeats' in prm['RL']:
            prm['RL']['n_repeats'] = prm['RL']['repeats']
        for path in prev_paths:
            if path not in prm['paths']:
                prm['paths'][path] = prev_paths[path]
    else:  # else use current input data
        lp, prm = None, None
        print(f"not os.path.exists({input_folder})")

    return lp, prm


def get_settings_i(settings, i):
    """Get run-specific settings from general settings dictionary."""
    settings_i = {}
    for key, sub_dict in settings.items():
        settings_i[key] = {}
        for sub_key, val in sub_dict.items():
            if isinstance(val, list):
                settings_i[key][sub_key] = val[i]
            elif isinstance(val, dict):
                for subsubkey in val.keys():
                    settings_i[key][sub_key] = {}
                    if isinstance(val[subsubkey], list):
                        settings_i[key][sub_key][subsubkey] = \
                            val[subsubkey][i]
            else:
                settings_i[key][sub_key] = val

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
            settings_i['ntw']['n'] = int(val)
        else:
            settings_i['RL'][key[2:]] = val
            print(f"RL['{key[2:]}'] = {val}")
    if len(obs) > 0:
        settings_i['RL']['state_space'] = obs

    return settings_i
