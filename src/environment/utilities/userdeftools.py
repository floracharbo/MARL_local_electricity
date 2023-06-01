#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:56:31 2020.

@author: floracharbonnier
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import yaml


def _is_empty(data):
    """Check if data is empty."""
    if data == '' \
            or data == ' ' \
            or data == [] \
            or data is None \
            or not isinstance(data, str) and np.isnan(data):
        return True
    else:
        return False


def comb(dims):
    """Obtain all possible combinations between variables."""
    obj = []
    if len(dims) == 2:
        for i in range(dims[0]):
            for j in range(dims[1]):
                obj.append((i, j))
    elif len(dims) == 3:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    obj.append((i, j, k))

    return obj


def str_to_float(listarr):
    """Convert string to float."""
    if isinstance(listarr, str):
        obj = float(listarr) if not _is_empty(listarr) else []
    elif isinstance(listarr, pd.core.series.Series):
        obj = [float(s) if not _is_empty(s) else [] for s in listarr]
    elif isinstance(listarr, np.float64):
        obj = float(listarr)
    elif isinstance(listarr[0], list):
        obj = [float(s[0]) if not _is_empty(s) else [] for s in listarr]
    else:
        obj = [float(s) if not _is_empty(s) else [] for s in listarr]

    return obj


def str_to_int(listarr):
    """Convert strings to integers."""
    if isinstance(listarr, str):
        obj = int(listarr) if not _is_empty(listarr) else []
    elif isinstance(listarr, pd.core.series.Series):
        obj = [int(s) if not _is_empty(s) else [] for s in listarr]
    elif isinstance(listarr, np.float64):
        obj = int(listarr)
    elif isinstance(listarr[0], list):
        obj = [int(s[0]) if not _is_empty(s) else [] for s in listarr]
    else:
        obj = [int(s) if not _is_empty(s) else [] for s in listarr]

    return obj


def initialise_dict(
        entries, type_obj='empty_list', n=1,
        second_level_entries=[], second_type='empty_list'):
    """Initialise a dictionary with keys 'entries'."""
    obj_dict = {
        'empty_list': [],
        'empty_dict': {},
        'zeros': np.zeros(n),
        'zero': 0,
        'Nones': [None] * n,
        'empty_np_array': np.array([])
    }
    if len(second_level_entries) > 0:
        type_obj = 'empty_dict'
    obj = {}
    for e in entries:
        obj[e] = obj_dict[type_obj].copy()
        for e2 in second_level_entries:
            obj[e][e2] = obj_dict[second_type].copy()

    return obj


def play_sound():
    """Play a sound to alert user."""
    import simpleaudio as sa

    filename = 'beep-24.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    wave_obj.play()





def _naming_file_extension_network_parameters(grd):
    """ Adds the manage_voltage and manage_agg_power settings to optimization results in opt_res """
    upper_quantities = ['max_voltage', 'max_grid_import']
    lower_quantities = ['min_voltage', 'max_grid_export']
    penalties_upper = ['overvoltage', 'import']
    penalties_lower = ['undervoltage', 'export']
    managements = ['manage_voltage', 'manage_agg_power']
    file_extension = ''
    with open("config_files/default_input_parameters/grd.yaml", "rb") as file:
        default_grd = yaml.safe_load(file)
    for lower_quantity, upper_quantity, penalty_upper, penalty_lower, management in zip(
            lower_quantities, upper_quantities, penalties_upper, penalties_lower, managements
    ):
        if grd[management]:
            if default_grd[upper_quantity] != grd[upper_quantity]:
                file_extension += f"_{management}_limit" + str(grd[upper_quantity])
            if (
                default_grd[lower_quantity] != grd[lower_quantity]
                and grd[upper_quantity] != grd[lower_quantity]
            ):
                file_extension += f"_{grd[lower_quantity]}"
            if default_grd[f'penalty_{penalty_upper}'] != grd[f'penalty_{penalty_upper}']:
                file_extension += "_penalty_coeff" + str(grd[f'penalty_{penalty_upper}'])
            if (
                default_grd[f'penalty_{penalty_lower}'] != grd[f'penalty_{penalty_lower}']
                and grd[f'penalty_{penalty_upper}'] != grd[f'penalty_{penalty_lower}']
            ):
                file_extension += "_" + str(grd[f'penalty_{penalty_lower}'])

            if management == 'manage_voltage':
                if grd['subset_line_losses_modelled'] != default_grd['subset_line_losses_modelled']:
                    file_extension += f"subset_losses{grd['subset_line_losses_modelled']}"
                if grd['reactive_power_for_voltage_control']:
                    file_extension += 'q_action'

    return file_extension


def get_opt_res_file(prm, test=False):
    syst, heat, car, loads, paths, rl, grd = [
        prm[info] for info in ['syst', 'heat', 'car', 'loads', 'paths', 'RL', 'grd']
    ]
    ext = '_test' if test else ''

    if np.all(car['caps'] == car['cap']):
        cap_str = car['cap']
    else:
        caps = {}
        for home, cap in enumerate(car['caps']):
            if cap not in caps:
                caps[cap] = []
            caps[cap].append(home)
        cap_str = ''
        for cap, homes in caps.items():
            cap_str += f"{cap}"
            for home in homes:
                cap_str += f"_{home}"

    paths['opt_res_file'] = \
        f"_D{syst['D']}_H{syst['H']}_{syst['solver']}_Uval{heat['Uvalues']}" \
        f"_ntwn{syst['n_homes' + ext]}_cmax{car['c_max0']}_" \
        f"dmax{car['d_max']}_cap{cap_str}_SoC0{car['SoC0']}"
    if syst['n_homesP'] > 0:
        paths['opt_res_file'] += f"_nP{syst['n_homesP' + ext]}"
    # if syst['n_homes_test'] != syst['n_homes']:
    #     opt_res_file += f"_ntest{syst['n_homes_test']}"
    if syst['clus_dist_share'] < 1:
        paths['opt_res_file'] += f"_clus_share{int(syst['clus_dist_share'] * 100)}"
    if "file" in heat and heat["file"] != "heat.yaml":
        paths['opt_res_file'] += f"_{heat['file']}"

    for obj, label in zip([car, heat, loads], ['car', 'heat', 'loads']):
        ownership = obj[f'own_{label}']
        if sum(ownership) != len(ownership):
            paths['opt_res_file'] += f"_no_{label}"
            for home in np.where(ownership == 0)[0]:
                paths['opt_res_file'] += f"_{home}"

    paths["seeds_file"] = f"outputs/seeds/seeds{paths['opt_res_file']}"
    if rl["deterministic"] == 2:
        for file in ["opt_res_file", "seeds_file"]:
            paths[file] += "_noisy"

    for file in ["opt_res_file", "seeds_file"]:
        if rl["deterministic"] == 2:
            paths[file] += "_noisy"
        paths[file] += f"_r{rl['n_repeats']}_epochs{rl['n_epochs']}" \
                       f"_explore{rl['n_explore']}_endtest{rl['n_end_test']}"
        if file == "opt_res_file" and prm["syst"]["change_start"]:
            paths["opt_res_file"] += "_changestart"
        paths[file] += _naming_file_extension_network_parameters(grd)
        # eff does not matter for seeds, but only for res
        if file == "opt_res_file" and prm["car"]["efftype"] == 1:
            paths["opt_res_file"] += "_eff1"

        paths[file] += ".npy"

    return paths


def get_moving_average(array, n_window, Nones=True):
    """Get moving average of array over window n_window."""
    x = max(int(n_window / 2 - 0.5), 1)
    n = len(array)
    mova = np.full(n, np.nan)
    for j in range(x):
        if not Nones:
            if sum(a is None for a in array[0: j * 2 + 1]) == 0:
                mova[j] = np.mean(array[0: j * 2 + 1])

    for j in range(x, n - x):
        if sum(a is None for a in array[j - x: j + x]) == 0:
            mova[j] = np.mean(array[j - x: j + x])

    for j in range(n - x, n):
        if not Nones:
            n_to_end = len(array) - j
            if sum(a is None for a in array[j - n_to_end: -1]) == 0:
                mova[j] = np.mean(array[j - n_to_end: -1])

    return mova


def set_seeds_rdn(seed):
    """Seed the random generators."""
    if not isinstance(seed, int):
        seed = int(seed)
    np.random.seed(seed), random.seed(seed)
    th.manual_seed(seed)


def data_source(q, epoch=None):
    if len(q.split('_')) == 3:
        return q.split('_')[0]
    elif len(q.split('_')) == 1:
        return None
    else:
        n_opt = int(q.split('_')[3])
        if epoch is None or epoch >= n_opt:
            return 'env'
        else:
            return 'opt'


def reward_type(q):
    return q.split('_')[1]


def distr_learning(q):
    return q.split('_')[2]


def current_no_run(results_path):
    if not Path(results_path).exists():
        no_run = 1
    else:
        prev_runs = [
            r for r in os.listdir(results_path) if r[0:3] == "run"
        ]
        no_prev_runs = [int(r[3:]) for r in prev_runs]
        no_run = max(no_prev_runs + [0]) + 1

    return no_run


def methods_learning_from_exploration(t_explo, epoch, rl):
    methods_to_update = [] if t_explo == 'baseline' \
        else [t_explo] if t_explo[0:3] == 'env' \
        else [method for method in rl['type_Qs']
              if data_source(method, epoch) == 'opt'
              and method[-1] != '0'
              and method in rl["evaluation_methods"]
              ]

    return methods_to_update


def add_prm_save_list(val, dict_, key):
    """Save all int/float/bool items."""
    if isinstance(val, list):
        dict_[key] = []
        for item in val:
            if isinstance(item, (str, int, float, bool)):
                dict_[key].append(item)

    return dict_


def get_prm_save_RL(prm_save, prm):
    """Add all int/float/bool items in prm["RL"] to dict to save."""
    prm_save["RL"] = {}
    for e, val in prm["RL"].items():
        prm_save["RL"] = add_prm_save_list(val, prm_save["RL"], e)
        if isinstance(val, dict):
            prm_save["RL"][e] = {}
            for key in val.keys():
                prm_save["RL"][e] = add_prm_save_list(
                    val[key], prm_save["RL"][e], key
                )
                if isinstance(val[key], (str, int, float, bool)):
                    prm_save["RL"][e][key] = val[key]
        elif isinstance(val, (str, int, float, bool)):
            prm_save["RL"][e] = val

    prm_save["RL"]["nn_learned"] = prm["RL"]["nn_learned"]

    return prm_save


def get_prm_save(prm):
    """Save run parameters for record-keeping."""
    prm_save = {}  # save selected system parameters
    with open(prm["paths"]["open_inputs"] / "prm_to_save.yaml", "rb") as file:
        prm_save = yaml.safe_load(file)

    for key in prm_save:
        prm_save[key] = {}
        sub_keys = prm_save[key] \
            if len(prm_save[key]) > 0 \
            else prm[key].keys()
        for sub_key in sub_keys:
            if sub_key == 'n__clus':
                sub_key = 'n_clus'
            if sub_key in prm[key]:
                prm_save[key][sub_key] = prm[key][sub_key]
            elif sub_key not in prm[key] \
                    and prm["RL"]["type_learning"] in prm[key] \
                    and sub_key in prm[key][prm["RL"]["type_learning"]]:
                prm_save[key][sub_key] = \
                    prm[key][prm["RL"]["type_learning"]][sub_key]
            elif sub_key in ["fprms", "fmean", "n_clus"]:
                sub_key = f"{sub_key[0]}_{sub_key[1:]}"
                prm_save[key][sub_key] = prm[key][sub_key]
            else:
                print(f"{sub_key} not in prm[{key}]")
                # np.save(f"keys_prm_{key}", list(prm[key].keys()))

    prm_save = get_prm_save_RL(prm_save, prm)

    return prm_save


def should_optimise_for_supervised_loss(epoch, rl):
    return (
        rl['supervised_loss']
        and epoch < rl['n_epochs_supervised_loss']
    )


def compute_import_export_costs(grid, grd, n_int_per_hr):
    # grid_in in kWh in the time interval
    # grid_in * n_int_per_hr in kW
    if grd['manage_agg_power']:
        grid_in = np.where(np.array(grid) >= 0, grid, 0)
        grid_out = np.where(np.array(grid) < 0, - grid, 0)
        grid_in_power, grid_out_power = grid_in * n_int_per_hr, grid_out * n_int_per_hr
        import_costs = np.where(
            grid_in_power >= grd['max_grid_import'],
            grd['penalty_import'] * (grid_in_power - grd['max_grid_import']),
            0
        )
        export_costs = np.where(
            grid_out_power >= grd['max_grid_export'],
            grd['penalty_export'] * (grid_out_power - grd['max_grid_export']),
            0
        )
        import_export_costs = import_costs + export_costs
    else:
        import_export_costs, import_costs, export_costs = 0, 0, 0

    return import_export_costs, import_costs, export_costs


def compute_voltage_costs(voltage_squared, grd):
    over_voltage_costs = grd['penalty_overvoltage'] * np.where(
        voltage_squared > grd['max_voltage'] ** 2,
        voltage_squared - grd['max_voltage'] ** 2,
        0
    )
    under_voltage_costs = grd['penalty_undervoltage'] * np.where(
        voltage_squared < grd['min_voltage'] ** 2,
        grd['min_voltage'] ** 2 - voltage_squared,
        0
    )

    return np.sum(over_voltage_costs + under_voltage_costs)


def mean_max_hourly_voltage_deviations(voltage_squared, max_voltage, min_voltage):
    overvoltage_deviation = \
        (np.sqrt(voltage_squared) - max_voltage)[voltage_squared > max_voltage ** 2]
    undervoltage_deviation = \
        (min_voltage - np.sqrt(voltage_squared))[voltage_squared < min_voltage ** 2]
    voltage_deviation = np.concatenate([overvoltage_deviation, undervoltage_deviation])
    if len(voltage_deviation) > 0:
        mean = np.mean(voltage_deviation)
        max = np.max(voltage_deviation)
        n_deviations_bus = len(voltage_deviation)
        n_deviations_hour = 1
    else:
        mean, max, n_deviations_bus, n_deviations_hour = 0, 0, 0, 0

    return mean, max, n_deviations_bus, n_deviations_hour


def f_to_interval(f, fs_brackets):
    interval = np.where(f >= fs_brackets[:-1])[0][-1]

    return interval


def list_potential_paths(
        prm, data_types=['gen', 'loads', 'car'],
        root_path='data',
        data_folder='other_outputs',
        sub_data_folder='outs'
):
    if 'n_rows0' not in prm:
        prm['n_rows0'] = {data_type: 'all' for data_type in data_types}
    potential_paths = []
    for folder in os.listdir(root_path / data_folder):
        data_in_folder = all(
            f"{data_type}_{prm['n_rows0'][data_type]}" in folder
            for data_type in data_types
        )
        if f"n{prm['syst']['H']}" in folder and data_in_folder:
            potential_paths.append(Path("data") / data_folder / folder / sub_data_folder)
    if all(prm['n_rows0'][data_type] == 'all' for data_type in data_types):
        potential_paths.append(
            Path(root_path) / data_folder / f"n{prm['syst']['H']}" / sub_data_folder
        )

    return potential_paths
