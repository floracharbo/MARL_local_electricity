import os
import pickle
import shutil
from datetime import datetime
from itertools import chain
from pathlib import Path
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# plot timing vs performance for n layers / dim layers; runs 742-656
ANNOTATE_RUN_NOS = False
FILTER_N_HOMES = False
COLUMNS_OF_INTEREST = [
    'grd-voltage_penalty',
    # 'syst-year_test', 'syst-year'
]
for col in [
    # 'loads-own_flex',
    # 'car-own_car',
    # 'loads-own_loads',
    # 'heat-own_heat',
    # 'syst-own_der',
    # 'loads-own_flexible_loads'
]:
    # COLUMNS_OF_INTEREST.append(col)
    # COLUMNS_OF_INTEREST.append(f"{col}_no")
    pass

# METRIC_VOLTAGE = 'mean_deviation'
METRIC_VOLTAGE = 'n_violation'
NORMALISED_n_violation = True

y_label_voltage = {
    'mean_deviation': 'Mean voltage deviation [p.u.]',
    'n_violation': 'Number of voltage deviations\nover the day',
}

FILTER = {
	'trajectory': False,
	'n_repeats':10,
	'manage_voltage':True,
	'n_homes':55,
    # 'n_epochs': 20,
    'type_learning': 'facmac',
	'reactive_power_for_voltage_control': True,
	'min_voltage':0.96,
	'penalty_export':False,
#     'state_space': 'grdC_min_voltage',
    'quadratic_voltage_penalty': False,
}

if len(COLUMNS_OF_INTEREST) != len(set(COLUMNS_OF_INTEREST)):
    raise ValueError("COLUMNS_OF_INTEREST has duplicates.")

IGNORE_FORCE_OPTIMISATION = True
FILL_BETWEEN = True
PLOT_ENV_ONLY = False
PLOT_TIME = False
PLOT_VOLTAGE_DEVIATIONS = True
n_subplots = 1 + PLOT_ENV_ONLY + PLOT_TIME + PLOT_VOLTAGE_DEVIATIONS
i_subplot_time = 1 if not PLOT_ENV_ONLY else 2
font_size = 12
font = {'size': font_size}
matplotlib.rc('font', **font)

best_score_type = 'p50'
# p50 or ave

X_LABELS = {
    'facmac-batch_size': 'Batch size',
    'facmac-lr': 'Agent learning rate',
    'facmac-critic_lr': 'Critic learning rate',
    'hyper_initialization_nonzeros':
        'Variance of normal distribution\nfor neural network weight initialisation',
    'n_cnn_layers': 'Number of convolutional layers',
    'optimizer': 'Optimiser',
    'rnn_hidden_dim': 'Neural network hidden dimension',
    'aggregate_actions': 'Aggregated action space',
    'lr': 'Learning rate',
    'q_learning-eps': 'Random exploration rate $\epsilon$',
    'n_discrete_actions': 'Number of discrete actions intervals',
    'n_grdC_level': 'Number of discrete state intervals',
    'q_learning-alpha': 'Learning rate',
    'n_epochs': 'Number of epochs',
    'buffer_size': 'Buffer size',
    'nn_type': 'Neural network architecture',
    'cnn_out_channels': 'Number of output channels',
    'cnn_kernel_size': 'Kernel size',
    'state_space': 'State space',
    'n_homes': 'Number of homes',
    'voltage_penalty': 'Voltage constraint violation penalty [£/p.u.]',
    'year': 'Training year',
}


def rename_runs(results_path):
    folders = os.listdir(results_path)
    initial_numbers = sorted([int(folder[3:]) for folder in folders if folder[0: 3] == "run"])
    for i, initial_number in enumerate(initial_numbers):
        if initial_number != i + 1:
            os.rename(results_path / f"run{initial_number}", results_path / f"run{i + 1}")


def remove_nans_best_scores_sorted(values_of_interest_sorted, best_scores_sorted):
    new_values_of_interest_sorted = {}
    for k in ['all', 'env']:
        i_not_nans = [
            i for i, y in enumerate(best_scores_sorted[k][best_score_type]) if not np.isnan(y)
        ]
        for key in best_scores_sorted[k]:
            best_scores_sorted[k][key] = [best_scores_sorted[k][key][i] for i in i_not_nans]
        new_values_of_interest_sorted[k] = [values_of_interest_sorted[i] for i in i_not_nans]

    return new_values_of_interest_sorted, best_scores_sorted



def check_if_orders_of_magnitude_gap(values_of_interest_sorted, prev_boolean):
    order_of_magnitudes_gap = prev_boolean
    if isinstance(values_of_interest_sorted[0], (int, float)):
        deltas_x_axis = [
            next - prev
            for next, prev in zip(
                values_of_interest_sorted[1:],
                values_of_interest_sorted[:-1]
            )
        ]
        if (
                min(deltas_x_axis) <= 10 * max(deltas_x_axis)
        ):
            order_of_magnitudes_gap = True

    return order_of_magnitudes_gap

def fix_learning_specific_values(log):
    """If there are no optimisations, this is equivalent to if we had forced optimisations."""
    ave_opt_cols = [
        col for col in log.columns
        if col[0: len(f'{best_score_type}_opt')] == f'{best_score_type}_opt'
    ]
    for i in range(len(log)):
        if all(log[col].loc[i] is None for col in ave_opt_cols):
            if 'syst-force_optimisation' in log and not log['syst-force_optimisation'].loc[i]:
                log.loc[i, 'syst-force_optimisation'] = True
            if (
                'syst-error_with_opt_to_rl_discharge' in log
                and log['syst-error_with_opt_to_rl_discharge'].loc[i]
            ):
                log.loc[i, 'syst-error_with_opt_to_rl_discharge'] = False

    # gaussian_params = ['start_steps', 'act_noise']
    # for param in gaussian_params:
    #     log[f'RL-{param}'] = np.where(
    #     log['RL-exploration_mode'] != 'gaussian', 0, log[f"RL-{param}"]
    #     )
    # ou_params = ['ou_theta', 'ou_sigma', 'ou_noise_scale', 'ou_stop_episode']
    # for param in ou_params:
    #     log[f'RL-{param}'] = np.where(
    #     log['RL-exploration_mode'] != 'ornstein_uhlenbeck', 0, log[f'RL-{param}']
    #     )

    return log


def is_short_type(val):
    return isinstance(val, (int, float, bool, str))


def list_obs_to_str(list_obs):
    """From a list of observations, get one string in alphabetical order for this combination."""
    sorted_obs = sorted(list_obs)
    obs_str = ''
    for obs in sorted_obs:
        obs_str += f'{obs}_'
    obs_str = obs_str[0:-1]

    return obs_str


def add_subkey_to_list_columns(key, subkey, ignore, subval, columns0):
    ntw_to_grd = ['charge_type', 'manage_agg_power', 'Bcap', 'loads', 'flex', 'gen']
    replace_ntw = {
        'C': ['grd', 'export_C', None],
        'n': ['syst', 'n_homes', None],
        'nP': ['syst', 'n_homesP', None],
        'n_all': ['syst', 'n_all_homes', None],
    }
    if subkey == 'n_homes' and key != 'syst':
        return columns0

    if subkey not in ignore and f"{key}-{subkey}" not in ignore:
        if key == 'ntw':
            if subkey in ntw_to_grd:
                key = 'grd'
            elif subkey in replace_ntw:
                key, subkey, subsubkey = replace_ntw[subkey]
            else:
                print(f"{(key, subkey)} not in replace_ntw")
        discard_n_homes_test = key == 'RL' and subkey == 'n_homes_test'
        if (
            not discard_n_homes_test and f"{key}-{subkey}" not in columns0
            and (
                subkey[0: len("own_")] == 'own_'
                or is_short_type(subval)
                or subkey == "state_space"
            )
        ):
            columns0.append(f"{key}-{subkey}")
        elif isinstance(subval, dict):
            for subsubkey, subsubval in subval.items():
                new_col = f"{key}-{subkey}-{subsubkey}" not in columns0
                ignore_col = subsubkey in ignore or f"{subkey}-{subsubkey}" in ignore
                short_type = is_short_type(subsubval)
                if short_type and new_col and not ignore_col:
                    columns0.append(f"{key}-{subkey}-{subsubkey}")

    return columns0


def get_list_all_fields(results_path):
    ignore = [
        'use_cuda', 'dim_states', 'dim_actions', 'dim_actions_1', 'episode_limit',
        'tot_learn_cycles', 'start_end_eval', 'n_all_epochs', 'T_decay_param',
        'statecomb_str', 'init_len_seeds', 'opt_res_file', 'seeds_file', 'plot_type',
        'plot_profiles', 'plotting_batch', 'description_run', 'type_env',
        'obs_shape', 'results_file', 'n_actions', 'state_shape', 'state_shape_test', 'agents',
        'save', 'groups', 'paths', 'end_decay', 'f_max-loads', 'f_min-loads', 'dt',
        'env_info', 'clust_dist_share', 'f_std_share', 'phi0', 'run_mode',
        'no_flex_action_to_target', 'N', 'n_int_per_hr', 'possible_states', 'n_all',
        'n_opti_constraints', 'dim_states_1', 'facmac-lr_decay_param',
        'facmac-critic_lr_decay_param', 'RL-n_homes_test', 'car-cap', 'RL-default_action',
        'RL-lr', 'c_max0', 'net-version', 'test_different_to_train'
    ]
    if IGNORE_FORCE_OPTIMISATION:
        ignore += ['syst-force_optimisation', 'device', 'ncpu', 'server']
    result_files = os.listdir(results_path)
    result_nos = sorted([int(file.split('n')[1]) for file in result_files if file[0: 3] == "run"])
    columns0 = []
    remove_nos = []
    current_run_invalid = None
    for result_no in result_nos:
        path_prm = results_path / f"run{result_no}" / 'inputData' / 'prm.npy'
        there_are_figures = len(os.listdir(results_path / f"run{result_no}" / 'figures')) > 0
        if path_prm.is_file() and there_are_figures:
            prm = np.load(path_prm, allow_pickle=True).item()
            for key, val in prm.items():
                if key not in ignore:
                    for subkey, subval in val.items():
                        columns0 = add_subkey_to_list_columns(key, subkey, ignore, subval, columns0)
        else:
            current_run_no = np.load("outputs/current_run_no.npy")
            if result_no not in [max(result_nos), current_run_no]:  # this may be something currently running
                shutil.rmtree(results_path / f"run{result_no}")
                remove_nos.append(result_no)
            else:
                current_run_invalid = current_run_no
                print(f"run{result_no} is invalid")

    print(f"delete run(s) {remove_nos}")
    for result_no in remove_nos:
        result_nos.pop(result_nos.index(result_no))

    columns0 = ["run", "date"] + sorted(columns0)
    if 'RL-batch_size' in columns0:
        columns0.pop(columns0.index('RL-batch_size'))

    return columns0, result_nos, current_run_invalid


def get_names_evaluation_methods(results_path, result_nos, current_run_invalid):
    evaluation_methods_found = False
    it = 0
    keys_methods = []
    while not evaluation_methods_found and it < len(result_nos):
        it += 1
        if result_nos[-it] != current_run_invalid:
            path_metrics0 = results_path / f"run{result_nos[-it]}" / 'figures' / 'metrics.npy'
            if os.path.exists(path_metrics0):
                metrics0 = np.load(path_metrics0, allow_pickle=True).item()
                keys_methods_run = list(metrics0['end_test_bl'][best_score_type].keys())
                for method in keys_methods_run:
                    if method not in keys_methods:
                        keys_methods.append(method)
                if len(keys_methods) == 16:
                    evaluation_methods_found = True

    keys_methods.remove("baseline")

    return keys_methods


def replace_single_default_value(value, default_data, subkey, subsubkey):
    if value is None and default_data is not None:
        if subkey in default_data:
            if subsubkey is None:
                value = default_data[subkey]
            elif subsubkey in default_data[subkey]:
                value = default_data[subkey][subsubkey]

    return value


def fill_in_log_value_with_run_data(
    log, row, column, prm_default,
    params_prev_default, values_prev_default, timestamp_changes
):
    timestamp_run = log.loc[row, 'syst-timestamp']

    if column in params_prev_default:
        idx_changes = np.where(
            (np.array(params_prev_default) == column) & (timestamp_changes < timestamp_run)
        )[0]
        if len(idx_changes) > 0:
            idx_change = idx_changes[np.argmax([timestamp_changes[idx] for idx in idx_changes])]
            log.loc[row, column] = values_prev_default[idx_change]
    else:
        key, subkey, subsubkey = get_key_subkeys_column(column)
        if key in prm_default:
            if subkey in prm_default[key]:
                if subsubkey is None:
                    log.loc[row, column] = prm_default[key][subkey]
                elif subsubkey in prm_default[key][subkey]:
                    log.loc[row, column] = prm_default[key][subkey][subsubkey]

    return log


def save_default_values_to_run_data(log):
    for row in range(len(log)):
        run_no = log.loc[row, 'run']
        path_default = Path(f"outputs/results/run{run_no}/inputData/prm_with_defaults.pickle")
        prm_default = {}
        for column in log.columns:
            key, subkey, subsubkey = get_key_subkeys_column(column)
            if key is not None:
                if key not in prm_default:
                    prm_default[key] = {}
                if subsubkey is None:
                    prm_default[key][subkey] = log.loc[row, column]
                else:
                    if subkey not in prm_default[key]:
                        prm_default[key][subkey] = {}
                    prm_default[key][subkey][subsubkey] = log.loc[row, column]
        with open(path_default, "wb") as file:
            pickle.dump(prm_default, file)


def row_to_assets_str(row, columns0):
    assets = ''
    if row[columns0.index('car-own_car')] == 1:
        assets += 'car, '
    if row[columns0.index('loads-own_flex')] == 1:
        assets += 'flex, '
    if row[columns0.index('heat-own_heat')] == 1:
        assets += 'heat, '
    if len(assets) > 0:
        assets = assets[:-2]
    else:
        assets = 'none'

    return assets


def str_prev_default_to_param(str_prev_default):
    param = ''
    for i in range(len(str_prev_default.split('-')) - 1):
        param += str_prev_default.split('-')[i] + '-'
    param = param[:-1]

    return param


def add_default_values(log, new_columns):
    with open("config_files/default_input_parameters/previous_defaults.yaml", "rb") as file:
        previous_defaults = yaml.safe_load(file)
    timestamp_changes_all = previous_defaults['timestamp_changes']
    del previous_defaults['timestamp_changes']
    params_prev_default = [
        str_prev_default_to_param(str_prev_default)
        for str_prev_default in previous_defaults.keys()
    ]
    timestamp_changes = [
        timestamp_changes_all[int(column.split('-')[-1])]
        for column in previous_defaults.keys()
    ]
    values_prev_default = [
        previous_defaults[column] for column in previous_defaults.keys()
    ]
    file_name = ''
    # add any default value previously saved row by row
    for row in range(len(log)):
        run_no = log.loc[row, 'run']
        path_default = Path(f"outputs/results/run{run_no}/inputData/prm_with_defaults.pickle")
        if path_default.is_file():
            with open(path_default, "rb") as file:
                prm_default = pickle.load(file)
            for column in log.columns:
                if log.loc[row, column] is None and column != 'syst-time_end':
                    log = fill_in_log_value_with_run_data(
                        log, row, column, prm_default,
                        params_prev_default, values_prev_default, timestamp_changes
                    )


    share_active = log.apply(
        lambda x: x['syst-n_homes'] / x['syst-n_homes_all'], axis=1
    )
    if 'syst-share_active' in log.columns:
        log['syst-share_active'] = share_active
    else:
        log.insert(len(new_columns), 'syst-share_active', share_active)
        new_columns.insert(len(new_columns), 'syst-share_active')
    if all(
        f"{prm}-own_{der}" in log.columns for prm, der in zip(
            ['loads', 'loads'], ['loads', 'flex']
        )
    ):
        own_flexible_loads = log.apply(
            lambda x: x['loads-own_loads'] if x['loads-own_loads'] == x['loads-own_flex'] else np.nan,
            axis=1
        )
        if 'loads-own_flexible_loads' in log.columns:
            log['loads-own_flexible_loads'] = own_flexible_loads
        else:
            log.insert(len(new_columns), 'loads-own_flexible_loads', own_flexible_loads)
            new_columns.insert(len(new_columns), 'loads-own_flexible_loads')

    if all(
        f"{prm}-own_{der}" in log.columns for prm, der in zip(
            ['car', 'heat', 'loads'], ['car', 'heat', 'flex']
        )
    ):
        own_der = log.apply(
            lambda x: x['car-own_car'] if x['car-own_car'] == x['heat-own_heat'] and x['car-own_car'] == x['loads-own_flex'] else np.nan,
            axis=1
        )
        log.insert(len(new_columns), 'syst-own_der', own_der)
        new_columns.insert(len(new_columns), 'syst-own_der')

    for col in ['car-own_car', 'loads-own_loads', 'heat-own_heat', 'loads-own_flex', 'loads-own_flexible_loads', 'syst-own_der']:
        new_col_name = f"{col}_no"
        own_no_homes = log.apply(
            lambda x: x[col] * x['syst-n_homes'], axis=1
        )
        if new_col_name in log.columns:
            log[new_col_name] = own_no_homes
        else:
            log.insert(len(new_columns), new_col_name, own_no_homes)
            new_columns.insert(len(new_columns), new_col_name)

    log['grd-penalty_overvoltage'] = log.apply(
        lambda x: x['grd-penalty_overvoltage'] if x['grd-manage_voltage'] else 0, axis=1
    )
    log['grd-penalty_undervoltage'] = log.apply(
        lambda x: x['grd-penalty_undervoltage'] if x['grd-manage_voltage'] else 0, axis=1
    )

    voltage_penalty = log.apply(
        lambda x: x['grd-penalty_overvoltage']
        if x['grd-penalty_overvoltage'] == x['grd-penalty_undervoltage'] else np.nan,
        axis=1
    )
    if 'grd-voltage_penalty' in log.columns:
        log['grd-voltage_penalty'] = voltage_penalty
    else:
        log.insert(len(new_columns), 'voltage_penalty', voltage_penalty)
        new_columns.insert(len(new_columns), 'voltage_penalty')

    if 'RL-critic_optimizer' in log.columns:
        critic_optimizer_none = log['RL-critic_optimizer'].isnull()
        log.loc[
            critic_optimizer_none, 'RL-critic_optimizer'
        ] = log.loc[critic_optimizer_none].apply(
            lambda x: x['RL-optimizer'], axis=1
        )
    for ext in ['', 'P']:
        if 'syst-n_homes_test' + ext in log.columns:
            log['syst-n_homes_test' + ext] = log.apply(
                lambda x: np.nan if x['syst-n_homes' + ext] == x['syst-n_homes_test' + ext] else x['syst-n_homes_test' + ext], axis=1
        )

    if 'RL-offset_reward' in log.columns:
        log['RL-delta_reward'] = log.apply(
            lambda x: x['RL-delta_reward'] if x['RL-offset_reward'] else 0, axis=1
        )
        log['RL-offset_reward'] = True
    if 'RL-initialise_positive_weights_hyper_b_1_bias' in log.columns:
        weights_initialised = (log['run'] >= 326) & (log['run'] <= 348)
        log.loc[weights_initialised, 'RL-initialise_positive_weights_hyper_b_1_bias'] = True
    n_homes_test_none = log['syst-n_homes_test'].isnull()
    log.loc[n_homes_test_none, 'syst-n_homes_test'] = log.loc[n_homes_test_none].apply(
        lambda x: x['syst-n_homes'], axis=1
    )
    n_homes_all_test_none = log['syst-n_homes_all_test'].isnull()
    log.loc[n_homes_all_test_none, 'syst-n_homes_all_test'] = log.loc[n_homes_all_test_none].apply(
        lambda x: x['syst-n_homes_test'] + x['syst-n_homesP'], axis=1
    )
    if 'syst-share_active_test' in log.columns:
        share_active_test_none = log['syst-share_active_test'].isnull()
        log.loc[
            share_active_test_none, 'syst-share_active_test'
        ] = log.loc[share_active_test_none].apply(
            lambda x: x['syst-n_homes_test'] / x['syst-n_homes_all_test'], axis=1
        )

    # then replace column by column the missing data with current defaults
    for column in log.columns:
        key, subkey, subsubkey = get_key_subkeys_column(column)
        if key is not None:
            # get you have correct data loaded or load it
            path = Path(f"config_files/default_input_parameters/{key}.yaml")
            if key != file_name:
                if path.is_file():
                    file_path = f"config_files/default_input_parameters/{key}.yaml"
                    with open(file_path, "rb") as file:
                        default_data = yaml.safe_load(file)
                else:
                    default_data = None
            # replace default value
            log[column] = log[column].apply(
                lambda x: replace_single_default_value(x, default_data, subkey, subsubkey)
            )

    save_default_values_to_run_data(log)

    return log, new_columns


def get_key_subkeys_column(column):
    split = column.split('-')
    key, subkey, subsubkey = None, None, None
    if len(split) == 2:
        key, subkey = split
    elif len(split) == 3:
        key, subkey, subsubkey = split
    elif len(split) == 4:
        key, subkey = split[0: 2]
        subsubkey = split[2] + '-' + split[3]
    if key == 'bat':
        key = 'car'

    return key, subkey, subsubkey


def get_grdC_n(prm, key, subkey, str_state_space):
    if subkey not in prm[key]:
        prm[key][subkey] = 0
    elif 'grdC_t' in str_state_space:
        indices = [
            i for i in range(len(str_state_space) - 6)
            if str_state_space[i: i + 6] == 'grdC_t'
        ]
        if len(indices) > 0:
            max_t = max([int(str_state_space[i + 6:].split('_')[0]) for i in indices])
            prm[key][subkey] = max_t + 1
    elif 'grdC' in str_state_space:
        prm[key][subkey] = 1

    return prm


def get_n_epochs_supervised_loss(prm, key, subkey):
    if 'supervised_loss' not in prm['RL'] or not prm['RL']['supervised_loss']:
        prm[key][subkey] = 0
    elif 'supervised_loss' in prm['RL'] \
            and prm['RL']['supervised_loss'] \
            and 'n_epochs_supervised_loss' not in prm['RL']:
        prm[key][subkey] = prm["RL"]['n_epochs']

    return prm


def get_own_items_from_prm(prm, key, subkey):
    if key not in prm:
        return prm

    potential_exts = ['P', '_test']
    ext = ''
    for potential_ext in potential_exts:
        if subkey[- len(potential_ext):] == potential_ext:
            ext = potential_ext

    if subkey not in prm[key]:
        prm[key][subkey] = 1
    elif prm[key][subkey] is None:
        prm[key][subkey] = 1
    elif isinstance(prm[key][subkey], (np.ndarray, list)):
        prm[key][subkey] = \
            sum(prm[key][subkey]) / prm['syst']['n_homes' + ext] \
            if prm['syst']['n_homes' + ext] > 0 \
            else 1
    elif prm[key][subkey] != 1:
        print(f"prm[{key}][{subkey}] = {prm[key][subkey]}")

    return prm


def data_specific_modifications(prm, key, subkey, subsubkey):
    previous_ntw = {
        'export_C': 'C',
        'n_homes': 'n',
        'n_homesP': 'nP',
        'n_all_homes': 'n_all',
    }
    if subkey == 'gamma':
        subsubkey = 'gamma'
        subkey = prm['RL']['type_learning']
    if key in prm and subkey not in prm[key] and 'ntw' in prm:
        if subkey in prm['ntw']:
            key = 'ntw'
            subkey = subkey
        elif subkey in previous_ntw:
            key = 'ntw'
            subkey = previous_ntw[subkey]
    if key == "RL" and subkey == 'n_homes':
        key = None

    str_state_space = list_obs_to_str(prm['RL']['state_space'])

    if subkey == 'grdC_n':
        prm = get_grdC_n(prm, key, subkey, str_state_space)
    elif subkey[0: len('own_')] == 'own_':
        prm = get_own_items_from_prm(prm, key, subkey)
    elif (
        subkey == 'beta_to_alpha'
        and not prm['RL'][prm['RL']['type_learning']]['hysteretic']
    ):
        prm[key][subkey] = 1
    elif subkey == 'n_epochs_supervised_loss':
        prm = get_n_epochs_supervised_loss(prm, key, subkey)
    elif subkey == 'no_flex_action' and "no_flex_action_to_target" in prm["RL"]:
        prm[key][subkey] = "target" if prm["RL"]["no_flex_action_to_target"] else "one"
    elif subkey == 'supervised_loss_weight' and not prm['RL']['supervised_loss']:
        prm['RL']['supervised_loss_weight'] = 0

    return prm, key, subkey, subsubkey


def get_caps_str(caps):
    caps_str = ''
    for cap in set(caps):
        n_homes = caps.count(cap)
        caps_str += f'{cap}_{n_homes}_'
    if caps_str[-1] == '_':
        caps_str = caps_str[:-1]

    return caps_str


def get_prm_data_for_a_result_no(results_path, result_no, columns0):
    path_prm = results_path / f"run{result_no}" / 'inputData' / 'prm.npy'

    if path_prm.is_file():
        prm = np.load(path_prm, allow_pickle=True).item()
        if 'type_learning' not in prm['RL']:
            os.remove(path_prm)
            return None
        if 'cap' in prm['car'] and 'caps' not in prm['car']:
            prm['car']['caps'] = prm['car']['cap']
        date_str = datetime.fromtimestamp(os.stat(path_prm).st_birthtime).strftime("%d/%m/%Y")
        row = [result_no, date_str]
        for column in columns0[2:]:
            key, subkey, subsubkey = get_key_subkeys_column(column)
            prm, key, subkey, subsubkey = data_specific_modifications(prm, key, subkey, subsubkey)
            if column == 'syst-assets':
                row.append(row_to_assets_str(row, columns0))
            elif key is None or key not in prm:
                row.append(None)
                # if column != 'RL-n_homes':
                #     print(f"column {column} does not correspond to a prm key")
            elif subsubkey is None:
                if subkey == 'state_space' and subkey in prm[key]:
                    row.append(list_obs_to_str(prm[key][subkey]))
                elif subkey == 'caps':
                    x = prm[key][subkey]
                    row.append(int(x[0]) if isinstance(x, np.ndarray) and all(
                        x[i] == x[0] for i in range(len(x))) else get_caps_str(x))
                else:
                    row.append(prm[key][subkey] if subkey in prm[key] else None)
            else:
                if subsubkey == 'gamma' and subsubkey not in prm[key][subkey]:
                    if subkey != prm['RL']['type_learning']:
                        val = None
                    else:
                        val = prm[key][subsubkey]
                else:
                    val = prm[key][subkey][subsubkey] \
                        if subkey in prm[key] and subsubkey in prm[key][subkey] \
                        else None
                if subkey == 'start_steps':
                    assert isinstance(val, int), f"start_steps is not an int: {val}"
                row.append(val)
    else:
        row = None

    assert row is None or len(columns0) == len(row), \
        f"len(columns0) {len(columns0)} != len(row) {len(row)}"

    return row


def append_metrics_data_for_a_result_no(results_path, result_no, keys_methods, row):
    path_figures = results_path / f"run{result_no}" / 'figures'
    path_metrics = path_figures / 'metrics.npy'
    path_voltage_metrics = path_figures / 'voltage_deviation_metrics.npy'
    if row is not None and path_metrics.is_file():
        # now add results
        metrics = np.load(path_metrics, allow_pickle=True).item()
        if path_voltage_metrics.is_file():
            voltage_metrics = np.load(path_voltage_metrics, allow_pickle=True).item()
        else:
            voltage_metrics = None
        for method in keys_methods:
            method_ = method
            if method not in metrics['end_test_bl'][best_score_type]:
                potential_replacement = [
                    method_metrics for method_metrics in metrics['end_test_bl'][best_score_type]
                    if method_metrics[0: 3] == 'env' and method in method_metrics
                ]
                if len(potential_replacement) == 1:
                    method_ = potential_replacement[0]
            for value in [best_score_type, 'p25', 'p75']:
                row.append(
                    metrics['end_test_bl'][value][method_]
                    if method_ in metrics['end_test_bl'][value]
                    else None
                )
        row = add_voltage_metrics_to_row(row, voltage_metrics, keys_methods)

    else:
        row = None

    return row

def add_voltage_metrics_to_row(row, voltage_metrics, keys_methods):
    if voltage_metrics is not None:
        if 'mean_voltage_violation' not in voltage_metrics['baseline']:
            labels = [
                'mean_voltage_deviation', None, 'max_voltage_deviation',
                'n_voltage_deviation_bus', 'n_voltage_deviation_hour'
            ]
        else:
            labels = [
                'mean_voltage_violation', 'mean_voltage_deviation', 'max_voltage_deviation',
                'n_voltage_violation_bus', 'n_voltage_violation_hour'
            ]

        for method in keys_methods + ['baseline']:
            for label in labels:
                if method in voltage_metrics and label is not None and label in voltage_metrics[method]:
                    row.append(
                        voltage_metrics[method][label]
                    )
                else:
                    row.append(None)
    else:
        row += [None] * len(keys_methods + ['baseline']) * 5

    return row


def remove_columns_that_never_change_and_tidy(log, columns0, columns_results_methods):
    new_columns = []
    do_not_remove = [
        'syst-server', "RL-state_space", 'RL-trajectory', 'RL-type_learning',
        'syst-n_homes', 'syst-share_active_test', 'syst-force_optimisation', 'syst-gan_generation'
    ]
    drop_columns = ['grd-simulate_panda_power_only', 'RL-gamma', 'RL-offset_mixer_qs']

    for column in columns0:
        unique_value = len(log[column][log[column].notnull()].unique()) == 1
        if (column not in do_not_remove and unique_value) or column in drop_columns:
            log.drop([column], axis=1, inplace=True)
        else:
            new_columns.append(column)

    # check there are no duplicates
    if len(new_columns) != len(set(new_columns)):
        print(f"len(new_columns) {len(new_columns)} len(set(new_columns)) {len(set(new_columns))}")
        for new_column in new_columns:
            indices = [j for j, x in enumerate(new_columns) if x == new_column]
            if len(indices) > 1:
                print(f"{new_column} times {len(indices)}")

    log.columns = new_columns + columns_results_methods

    return new_columns, log


def compute_best_score_per_run(keys_methods, log):
    for prefix in [f'{best_score_type}_', 'p25_', 'p75_']:
        cols = [prefix + method for method in keys_methods]
        log[cols] = log.apply(lambda row: 0 if row.n_homes == 0 else row[cols], axis=1)
    keys_methods_not_opt = [method for method in keys_methods if method != 'opt']
    ave_cols_non_opt = [f"{best_score_type}_{method}" for method in keys_methods_not_opt]
    ave_cols_methods_env = [
        f"{best_score_type}_{method}" for method in keys_methods_not_opt if method[0: 3] == 'env'
    ]
    ave_cols_methods_opt = [
        f"{best_score_type}_{method}" for method in keys_methods_not_opt if method[0: 3] == 'opt'
    ]
    initial_columns = log.columns

    for score, cols_methods in zip(
        ['best_score_all', 'best_score_opt', 'best_score_env'],
        [ave_cols_non_opt, ave_cols_methods_opt, ave_cols_methods_env]
    ):
        log[score] = log[cols_methods].max(axis=1)
        cols_scores = [col for col in initial_columns if col[0: 4] == f'{best_score_type}_']
        log[f'method_{score}'] = log[cols_scores + [score]].apply(
            lambda row: row[row == row[score]].index, axis=1
        )
        log[f"p25_{score}"] = log.apply(
            lambda row: row[f"p25_{row[f'method_{score}'][0][4:]}"]
            if len(row[f'method_{score}']) > 0 else np.nan,
            axis=1
        )

        log[f"p75_{score}"] = log.apply(
            lambda row: row[f"p75_{row[f'method_{score}'][0][4:]}"]
            if len(row[f'method_{score}']) > 0 else np.nan,
            axis=1
        )
        log[f'{METRIC_VOLTAGE}_best_all'] = log.apply(
            lambda row:
            row[f"{METRIC_VOLTAGE}_{row[f'method_{score}'][0][4:]}"]
            if len(row[f'method_{score}']) > 0 else np.nan,
            axis=1
        )
        log[f'{METRIC_VOLTAGE}_best_all'] = log[f'{METRIC_VOLTAGE}_best_all'].apply(lambda x: np.nan if x is None else x)
        log = log.drop(columns=[f'method_{score}'])
    # methods_best_score = log.columns[log[ave_cols_non_opt].argmax(axis=1)]
    # col_best_score_env = log[ave_cols_methods_env].argmax(axis=1)
    # col_best_score_opt = log[ave_cols_methods_opt].argmax(axis=1)

    # log['best_score'] = log[col_best_score].max(axis=1)
    # log['best_score_opt'] = log[keys_methods_opt].max(axis=1)
    # log['best_score_env'] = log[keys_methods_env].max(axis=1)

    return log


def check_that_only_grdCn_changes_in_state_space(
        other_columns, current_setup, row_setup, initial_setup_row,
        row, indexes_columns_ignore_q_learning, indexes_columns_ignore_facmac
):
    indexes_ignore = [
        other_columns.index(col)
        for col in ['state_space', 'nn_learned']
    ]
    if current_setup[other_columns.index('type_learning')] == 'q_learning':
        indexes_ignore += indexes_columns_ignore_q_learning
    if current_setup[other_columns.index('type_learning')] == 'facmac':
        indexes_ignore += indexes_columns_ignore_facmac
    only_col_of_interest_changes_without_state_space = all(
        current_col == row_col or (
            not isinstance(current_col, str) and np.isnan(current_col)
            and not isinstance(row_col, str) and np.isnan(row_col))
        for i, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
        if i not in indexes_ignore
    )

    if only_col_of_interest_changes_without_state_space:
        grdC_n_current_setup, state_space_current_setup = [
            log[column].loc[initial_setup_row]
            for column in ['grdC_n', 'state_space']
        ]
        grdC_n_row_setup, state_space_row_setup = [
            log[column].loc[row]
            for column in ['grdC_n', 'state_space']
        ]
        if grdC_n_current_setup > 0 and 'grdC' not in state_space_current_setup:
            grdC_n_current_setup = 0
        if grdC_n_row_setup > 0 and 'grdC' not in state_space_row_setup:
            grdC_n_row_setup = 0
        if state_space_current_setup == state_space_row_setup:
            only_col_of_interest_changes = True
        elif grdC_n_current_setup == 0 and grdC_n_row_setup == 0:
            only_col_of_interest_changes = False
        else:
            start_n = min([grdC_n_current_setup, grdC_n_row_setup])
            end_d = max([grdC_n_current_setup, grdC_n_row_setup])
            expected_diff_str = ''
            if start_n % 1 > 0:
                print(f"log.loc[{initial_setup_row}] = {log.loc[initial_setup_row]}")
            for i in range(int(start_n), int(end_d)):
                expected_diff_str += f'grdC_t{i}_'
            actual_diff_str_len = abs(len(state_space_current_setup) - len(state_space_row_setup))
            only_col_of_interest_changes = actual_diff_str_len == len(expected_diff_str)
    else:
        only_col_of_interest_changes = False

    return only_col_of_interest_changes


def annotate_run_nos(
        axs, values_of_interest_sorted, best_score_sorted,
        best_env_score_sorted, runs_sorted, mean_deviations_sorted
):
    if ANNOTATE_RUN_NOS:
        for i, (x, best_score, best_env_score, run, mean_deviations) in enumerate(zip(
                values_of_interest_sorted, best_score_sorted,
                best_env_score_sorted, runs_sorted, mean_deviations_sorted
        )):
            ax0 = axs if n_subplots == 1 else axs[0]
            print(f"run = {run} x {x} best_score {best_score} {METRIC_VOLTAGE}s {mean_deviations}")
            ax0.annotate(
                run, (x, best_score), textcoords="offset points", xytext=(0, 10),
                ha='center'
            )
            if PLOT_ENV_ONLY:
                axs[1].annotate(
                    run, (x, best_env_score), textcoords="offset points", xytext=(0, 10),
                    ha='center'
                )
            if PLOT_VOLTAGE_DEVIATIONS:
                axs[-1].annotate(
                    run, (x, mean_deviations), textcoords="offset points", xytext=(0, 10),
                    ha='center'
                )
        for j in np.argsort(best_score_sorted):
            print(f"run {runs_sorted[j]} {values_of_interest_sorted[j]}: {best_score_sorted[j]}")

    return axs


def get_relevant_columns_for_type_learning(other_columns, log, i_row):
    if log['type_learning'].loc[i_row] == 'q_learning':
        columns_irrelevant_to_comparisons = [
            'obs_agent_id', 'DDPG-rdn_eps_greedy_indiv', 'act_noise', 'agent_facmac',
            'buffer_size', 'cnn_kernel_size', 'cnn_out_channels', 'facmac-batch_size',
            'facmac-beta_to_alpha', 'facmac-critic_lr', 'facmac-hysteretic', 'learner',
            'mixer', 'n_hidden_layers', 'n_hidden_layers_critic', 'nn_type',
            'nn_type_critic', 'ou_stop_episode', 'rnn_hidden_dim', 'target_update_mode'
        ]
    else:
        columns_irrelevant_to_comparisons = []

    other_relevant_columns = [
        col for col in other_columns if col not in columns_irrelevant_to_comparisons
    ]

    return other_relevant_columns


def get_indexes_to_ignore_in_setup_comparison(
    column_of_interest, other_columns, current_setup, row_setup, initial_setup_row,
    indexes_columns_ignore_q_learning, indexes_columns_ignore_facmac, row,
    columns_irrelevant_to_q_learning, columns_irrelevant_to_facmac,
):
    ignore_cols = {
        'supervised_loss_weight': ['supervised_loss'],
        'state_space': ['grdC_n'],
        'share_active': [
            'n_homes', 'n_homes_test', 'n_homesP', 'n_homes_testP',
            'share_active_test', 'own_car_no', 'own_loads_no', 'own_flex_no', 'own_heat_no',
            'own_flexible_loads_no', 'own_der_no'
        ],
        'type_learning': [
            'act_noise', 'agent_facmac', 'buffer_size', 'cnn_kernel_size',
            'cnn_out_channels', 'facmac-batch_size', 'facmac-critic_lr',
            'hyper_initialization_nonzeros', 'lr', 'mixer', 'n_hidden_layers',
            'n_hidden_layers_critic', 'nn_type', 'nn_type_critic', 'obs_agent_id',
            'ou_stop_episode', 'rnn_hidden_dim', 'start_steps', 'q_learning-alpha',
            'gamma', 'timestamp', 'instant_feedback'
        ],
        'voltage_penalty': ['penalty_overvoltage', 'penalty_undervoltage'],
    }
    for col in ['own_car', 'own_loads', 'own_heat', 'own_flex']:
        ignore_cols[col] = [f"{col}_no", 'own_der', 'assets']
        ignore_cols[f"{col}_no"] = [col, 'own_der', 'assets']
    ignore_cols['own_der'] = ['own_der_no', 'assets']
    ignore_cols['own_der_no'] = ['own_der', 'assets']
    ignore_cols['assets'] = ['own_der', 'own_der_no']
    for col in ['own_car', 'own_heat', 'own_flex']:
        ignore_cols['own_der'] += [col, f"{col}_no"]
        ignore_cols['own_der_no'] += [col, f"{col}_no"]
        ignore_cols['assets'] += [col, f"{col}_no"]

    ignore_cols['own_flexible_loads'] = ['own_flex', 'own_loads', 'own_flex_no', 'own_loads_no', 'own_der', 'own_flexible_loads_no', 'assets']
    ignore_cols['own_flexible_loads_no'] = ['own_flex', 'own_loads', 'own_flex_no', 'own_loads_no', 'own_der', 'own_flexible_loads', 'assets']

    indexes_ignore = []
    if column_of_interest in ['n_homesP', 'n_homes', 'n_homes_test']:
        indexes_ignore.append(other_columns.index('n_homes_all'))
        if column_of_interest in ['n_homesP', 'n_homes']:
            if 'share_active' in other_columns:
                indexes_ignore.append(other_columns.index('share_active'))
            i_n_homes_test = other_columns.index('n_homes_test')
            row_n_homes = log[column_of_interest].loc[row]
            initial_setup_row_n_hommes = log[column_of_interest].loc[initial_setup_row]
            # if n_homes_test = n_homes set as np.nan so as to ignore it changing with n_homes
            if current_setup[i_n_homes_test] == initial_setup_row_n_hommes:
                current_setup[i_n_homes_test] = np.nan
            # if row_setup[i_n_homes_test] == row_n_homes:
            #     row_setup[i_n_homes_test] = np.nan
        if column_of_interest in ['n_homesP', 'n_homes_test']:
            indexes_ignore.append(other_columns.index('share_active_test'))
        if column_of_interest == 'n_homesP':
            indexes_ignore.append(other_columns.index('n_homes_testP'))

    else:
        if column_of_interest in ignore_cols:
            for ignore_col in ignore_cols[column_of_interest]:
                if ignore_col in other_columns:
                    indexes_ignore.append(other_columns.index(ignore_col))
        if (
                'type_learning' in other_columns
                and current_setup[other_columns.index('type_learning')] == 'q_learning'
        ):
            indexes_ignore += indexes_columns_ignore_q_learning
        elif column_of_interest in columns_irrelevant_to_facmac:
            # this is about q_learning
            indexes_ignore += indexes_columns_ignore_q_learning
        if (
                'type_learning' in other_columns
                and current_setup[other_columns.index('type_learning')] == 'facmac'
        ):
            indexes_ignore += indexes_columns_ignore_facmac
        elif column_of_interest in columns_irrelevant_to_q_learning:
            # this is about facmac
            indexes_ignore += indexes_columns_ignore_facmac

    return indexes_ignore


def compare_all_runs_for_column_of_interest(
    column_of_interest, other_columns, axs, log
):
    columns_irrelevant_to_q_learning = [
        'obs_agent_id', 'DDPG-rdn_eps_greedy_indiv', 'act_noise', 'agent_facmac',
        'buffer_size', 'cnn_kernel_size', 'cnn_out_channels', 'facmac-batch_size',
        'facmac-beta_to_alpha', 'facmac-critic_lr', 'facmac-hysteretic', 'learner',
        'mixer', 'n_hidden_layers', 'n_hidden_layers_critic', 'nn_type',
        'nn_type_critic', 'ou_stop_episode', 'rnn_hidden_dim', 'target_update_mode',
        'instant_feedback'
    ] + [col for col in other_columns if col[0: len('facmac')] == 'facmac']
    columns_irrelevant_to_facmac = [
        'n_discrete_actions'
    ] + [col for col in other_columns if col[0: len('q_learning')] == 'q_learning']
    indexes_columns_ignore_q_learning = [
        other_columns.index(col) for col in other_columns
        if col in columns_irrelevant_to_q_learning
    ]
    indexes_columns_ignore_facmac = [
        other_columns.index(col) for col in other_columns
        if col in columns_irrelevant_to_facmac
    ]
    rows_considered = []
    setup_no = 0
    plotted_something = False
    setups = []
    if column_of_interest == 'state_space':
        x_labels = []
        best_values = []
        env_values = []
    time_values = []
    mean_deviation_values = []
    mean_deviation_baseline_values = []
    any_zero_values_x = False
    any_zero_values_voltage = False
    order_of_magnitudes_gap_x = False
    order_of_magnitudes_gap_voltage = False
    it = 0
    while len(rows_considered) < len(log) * 2 and it < len(log) * 2:
        it += 1
        if it == len(log) * 2 - 1:
            print(f"it = {it} for column_of_interest = {column_of_interest}")
        initial_setup_rows = [i for i in range(len(log)) if i not in rows_considered]
        if len(initial_setup_rows) == 0:
            break

        initial_setup_row = initial_setup_rows[0]
        rows_considered.append(initial_setup_row)

        if (
                not isinstance(log[column_of_interest].loc[initial_setup_row], str)

                and (
                    log[column_of_interest].loc[initial_setup_row] is None
                    or np.isnan(log[column_of_interest].loc[initial_setup_row])
            )
        ):
            continue

        current_setup = log[other_columns].loc[initial_setup_row].values
        values_of_interest = [log[column_of_interest].loc[initial_setup_row]]
        best_scores = {k: {} for k in ['all', 'env']}
        for k in ['all', 'env']:
            best_scores[k][best_score_type] = [log[f'best_score_{k}'].loc[initial_setup_row]]
            for p in [25, 75]:
                best_scores[k][f'p{p}'] = [log[f'p{p}_best_score_{k}'].loc[initial_setup_row]]
        time_best_score = [log['time_end'].loc[initial_setup_row]]
        if NORMALISED_n_violation and METRIC_VOLTAGE == 'n_violation':
            multiplier_voltage_metric = 1/(24 * 906) * 100
        else:
            multiplier_voltage_metric = 1
        mean_deviation = [log[f'{METRIC_VOLTAGE}_best_all'].loc[initial_setup_row] * multiplier_voltage_metric]
        mean_deviation_baseline = [log[f'{METRIC_VOLTAGE}_baseline'].loc[initial_setup_row] * multiplier_voltage_metric]
        for row in range(len(log)):
            row_setup = [log[col].loc[row] for col in other_columns]
            new_row = row not in rows_considered
            # new_row = True
            relevant_cnn = not (
                column_of_interest[0: 3] == 'cnn'
                and log['nn_type'].loc[row] != 'cnn'
            )
            relevant_facmac = not (
                column_of_interest[0: 6] == 'facmac'
                and log['type_learning'].loc[row] != 'facmac'
            )
            relevant_supervised_loss = not (
                column_of_interest[0: 16] == 'supervised_loss_weight'
                and not log['supervised_loss'].loc[row]
            )
            relevant_eps = not (
                column_of_interest == 'facmac-epsilon' and log['facmac-epsilon_decay'].loc[row]
            )
            if column_of_interest == 'grdC_n':
                only_col_of_interest_changes = check_that_only_grdCn_changes_in_state_space(
                    other_columns, current_setup, row_setup, initial_setup_row,
                    row, indexes_columns_ignore_q_learning, indexes_columns_ignore_facmac
                )
            else:
                indexes_ignore = get_indexes_to_ignore_in_setup_comparison(
                    column_of_interest, other_columns, current_setup, row_setup, initial_setup_row,
                    indexes_columns_ignore_q_learning, indexes_columns_ignore_facmac, row,
                    columns_irrelevant_to_q_learning, columns_irrelevant_to_facmac,
                )
                # try:
                only_col_of_interest_changes = all(
                    current_col == row_col or (
                        (not isinstance(current_col, str) and np.isnan(current_col))
                        and (not isinstance(row_col, str) and np.isnan(row_col))
                    )
                    for i_col, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
                    if i_col not in indexes_ignore
                )
                # except Exception as ex:
                #     print(ex)
                if (
                        column_of_interest == 'n_homesP'
                        and not (np.isnan(log.loc[initial_setup_row, 'n_homes_testP']) and np.isnan(log.loc[row, 'n_homes_testP']))
                        and log.loc[initial_setup_row, 'n_homes_testP'] != log.loc[row, 'n_homes_testP']
                ):
                    only_col_of_interest_changes = False

            if FILTER_N_HOMES:
                n_homes_facmac_traj_only = not (
                    column_of_interest == 'n_homes'
                    and current_setup[other_columns.index('type_learning')] == 'facmac'
                    and not current_setup[other_columns.index('trajectory')]
                )
                n_homes_on_laptop_only = not (
                    column_of_interest == 'n_homes' and current_setup[other_columns.index('server')]
                )
            else:
                n_homes_facmac_traj_only = True
                n_homes_on_laptop_only = True
            relevant_data = \
                relevant_cnn \
                and relevant_facmac \
                and relevant_supervised_loss \
                and n_homes_on_laptop_only \
                and n_homes_facmac_traj_only \
                and relevant_eps
            if new_row and only_col_of_interest_changes and relevant_data and not np.isnan(log[column_of_interest].loc[row]):
                rows_considered.append(row)
                values_of_interest.append(log[column_of_interest].loc[row])
                for k in ['all', 'env']:
                    best_scores[k][best_score_type].append(log[f'best_score_{k}'].loc[row])
                    for p in [25, 75]:
                        best_scores[k][f'p{p}'].append(log[f'p{p}_best_score_{k}'].loc[row])

                time_best_score.append(log['time_end'].loc[row])
                mean_deviation.append(log[f'{METRIC_VOLTAGE}_best_all'].loc[row] * multiplier_voltage_metric)
                mean_deviation_baseline.append(log[f'{METRIC_VOLTAGE}_baseline'].loc[row] * multiplier_voltage_metric)

        if len(values_of_interest) > 1:
            all_setups_same_as_0 = all(
                values_of_interest_ == values_of_interest[0]
                for values_of_interest_ in values_of_interest
            )
            runs = log.loc[rows_considered[- len(values_of_interest):], 'run'].values
            if all_setups_same_as_0:
                row0 = rows_considered[- len(values_of_interest)]
                for row in rows_considered[- len(values_of_interest) + 1:]:
                    if all(
                        log.loc[row, col] == log.loc[row0, col]
                        for col in other_columns + [column_of_interest]
                    ):
                        print(f"runs {runs} equal?")
            else:
                setups.append(current_setup)
                if column_of_interest == 'assets':
                    i_sorted = []
                    ordered_assets = ['car', 'heat', 'flex', 'none']
                    i_3assets = [
                        i for i in range(len(values_of_interest))
                        if len(values_of_interest[i].split(',')) == 3
                    ]
                    if len(i_3assets) > 0:
                        i_sorted.append(i_3assets[0])
                    for asset1 in ordered_assets:
                        for asset2 in ordered_assets:
                            asset_comb = f"{asset1}, {asset2}"
                            if asset_comb in values_of_interest:
                                i_sorted.append(values_of_interest.index(asset_comb))
                    for asset in ordered_assets:
                        if asset in values_of_interest:
                            i_sorted.append(values_of_interest.index(asset))
                else:
                    i_sorted = np.argsort(values_of_interest)
                values_of_interest_sorted = [values_of_interest[i] for i in i_sorted]
                best_scores_sorted = {k: {} for k in ['all', 'env']}
                for k in ['all', 'env']:
                    best_scores_sorted[k][best_score_type] = [
                        best_scores[k][best_score_type][i] for i in i_sorted
                    ]
                    for p in [25, 75]:
                        best_scores_sorted[k][f'p{p}'] = [
                            best_scores[k][f'p{p}'][i] for i in i_sorted
                        ]
                time_best_score_sorted = [time_best_score[i] for i in i_sorted]
                mean_deviation_sorted = [mean_deviation[i] for i in i_sorted]
                mean_deviation_baseline_sorted = [mean_deviation_baseline[i] for i in i_sorted]
                runs_sorted = [runs[i] for i in i_sorted]
                ls = '--' if 'server' in log and log.loc[rows_considered[-1], 'server'] else '-'
                values_of_interest_sorted_k, best_scores_sorted = remove_nans_best_scores_sorted(
                    values_of_interest_sorted, best_scores_sorted
                )
                for ax_i, k in enumerate(['all', 'env']):
                    if k == 'env' and not PLOT_ENV_ONLY:
                        continue
                    else:
                        ax = axs[ax_i] if n_subplots > 1 else axs
                    label = \
                        current_setup[other_columns.index('type_learning')] + f"({len(setups)})" \
                        if column_of_interest == 'n_homes' \
                        else len(setups)
                    p = ax.plot(
                        values_of_interest_sorted_k[k],
                        best_scores_sorted[k][best_score_type],
                        'o', label=label, linestyle=ls,
                        markerfacecolor='None'
                    )
                    colour = p[0].get_color()
                    if column_of_interest in ['share_active', 'n_homesP']:
                        n_homes, n_homes_all = [
                            [
                                log.loc[log['run'] == runs_sorted[i], n]
                                for i in range(len(values_of_interest_sorted_k[k]))
                            ]
                            for n in ['n_homes_test', 'n_homes_all_test']
                        ]
                        score_per_active_home = [
                            best_scores_sorted[k][best_score_type][i] * n_homes_all[i] / n_homes[i]
                            for i in range(len(values_of_interest_sorted_k[k]))
                        ]
                        ax.plot(
                            values_of_interest_sorted_k[k],
                            score_per_active_home,
                            'o', markerfacecolor='None', linestyle=':',
                            label=None, color=colour
                        )
                        ax.legend()

                    # i_best = np.argmax(best_scores_sorted[k][best_score_type])
                    i_where_best = np.where(
                        best_scores_sorted[k][best_score_type] == np.max(best_scores_sorted[k][best_score_type])
                    )[0]
                    i_best = np.max(i_where_best)
                    ax.plot(
                        values_of_interest_sorted_k[k][i_best],
                        best_scores_sorted[k][best_score_type][i_best],
                        'o', markerfacecolor=colour, markeredgecolor=colour
                    )
                    if FILL_BETWEEN:
                        ax.fill_between(
                            values_of_interest_sorted_k[k],
                            best_scores_sorted[k]['p25'],
                            best_scores_sorted[k]['p75'],
                            alpha=0.2
                        )
                    # if column_of_interest == 'n_epochs':
                    #     ax.set_yscale('log')

                if PLOT_TIME:
                    axs[i_subplot_time].plot(
                        values_of_interest_sorted, time_best_score_sorted, 'o',
                        label=label,
                        linestyle=ls,
                        markerfacecolor='None',
                        color=colour
                    )
                if PLOT_VOLTAGE_DEVIATIONS:
                    axs[-1].plot(
                        values_of_interest_sorted, mean_deviation_sorted, 'o',
                        label=label,
                        linestyle=ls,
                        markerfacecolor='None',
                        color=colour
                    )
                    axs[-1].plot(
                        values_of_interest_sorted, mean_deviation_baseline_sorted, 'x--',
                        label=f"{label} baseline",
                        linestyle=ls,
                        markerfacecolor='None',
                        color=colour
                    )
                    print(f"{column_of_interest} {values_of_interest_sorted} {mean_deviation_sorted}")
                for i in range(len(values_of_interest_sorted) - 1):
                    if values_of_interest_sorted[i + 1] == values_of_interest_sorted[i]:
                        print(
                            f"we have two values for {column_of_interest} = "
                            f"{values_of_interest_sorted[i]}, "
                            f"runs {runs_sorted[i]} and {runs_sorted[i+1]}"
                        )
                axs = annotate_run_nos(
                    axs, values_of_interest_sorted, best_scores_sorted['all'][best_score_type],
                    best_scores_sorted['env'][best_score_type], runs_sorted, mean_deviation_sorted
                )
                order_of_magnitudes_gap_x = check_if_orders_of_magnitude_gap(values_of_interest_sorted, order_of_magnitudes_gap_x)
                if PLOT_VOLTAGE_DEVIATIONS:
                    order_of_magnitudes_gap_voltage = check_if_orders_of_magnitude_gap(mean_deviation_sorted, order_of_magnitudes_gap_voltage)
                if 0 in values_of_interest:
                    any_zero_values_x = True
                if 0 in mean_deviation_sorted:
                    any_zero_values_voltage = True
                if column_of_interest == 'state_space':
                    x_labels.append(values_of_interest_sorted)
                    best_values.append(best_scores_sorted['all'][best_score_type])
                    env_values.append(best_scores_sorted['env'][best_score_type])
                time_values.append(time_best_score_sorted)
                mean_deviation_values.append(mean_deviation_sorted)
                mean_deviation_baseline_values.append(mean_deviation_baseline_sorted)
                plotted_something = True

        setup_no += 1
    if order_of_magnitudes_gap_x and not any_zero_values_x and column_of_interest[-2:] != 'no':
        axs_ = [axs] if n_subplots == 1 else axs
        for ax in axs_:
            ax.set_xscale('log')
    if PLOT_VOLTAGE_DEVIATIONS and order_of_magnitudes_gap_voltage and not any_zero_values_voltage:
    #     axs[-1].set_yscale('log')
        ylim_ = axs[-1].get_ylim()
        axs[-1].set_ylim(0, ylim_[1])
    if len(time_values) > 1 and PLOT_TIME:
        end_time_best_score_sorted = [time_values_[-1] for time_values_ in time_values]
        if max(end_time_best_score_sorted) / min(end_time_best_score_sorted) > 30:
            axs[i_subplot_time].set_yscale('log')
    state_space_vals = [x_labels, best_values, env_values, time_values, mean_deviation_values, mean_deviation_baseline_values] \
        if column_of_interest == 'state_space' else None
    if not plotted_something:
        plt.close('all')

    return plotted_something, axs, setups, state_space_vals


def adapt_figure_for_state_space(state_space_vals, axs):
    x_labels, best_values, env_values, time_values, mean_deviation_values, mean_deviation_baseline_values = state_space_vals
    all_x_labels = []
    x_labels_flattened = list(chain.from_iterable(x_labels))
    for label in x_labels_flattened:
        if label not in all_x_labels:
            all_x_labels.append(label)

    all_best_vals, all_env_vals, all_time_vals, all_mean_deviation_vals, all_mean_deviation_baseline_vals = [
        np.empty((len(x_labels), len(all_x_labels))) for _ in range(5)
    ]
    for i in range(len(x_labels)):
        for j in range(len(x_labels[i])):
            idx_value = all_x_labels.index(x_labels[i][j])
            all_best_vals[i, idx_value] = best_values[i][j]
            all_env_vals[i, idx_value] = env_values[i][j]
            all_time_vals[i, idx_value] = time_values[i][j]
            all_mean_deviation_vals[i, idx_value] = mean_deviation_values[i][j]
            all_mean_deviation_baseline_vals[i, idx_value] = mean_deviation_baseline_values[i][j]

    plt.close()

    i_sorted = np.argsort(all_x_labels)
    x_labels_sorted = [all_x_labels[i] for i in i_sorted]
    fig, axs = plt.subplots(n_subplots, 1, figsize=(6.4, 11 / 3 * n_subplots))

    for i, (best, env, time, mean_deviation, mean_deviation_baseline) in enumerate(
        zip(all_best_vals, all_env_vals, all_time_vals, all_mean_deviation_vals, all_mean_deviation_baseline_vals)
    ):
        best_sorted = [best[i] for i in i_sorted]
        env_sorted = [env[i] for i in i_sorted]
        time_sorted = [time[i] for i in i_sorted]
        mean_deviation_sorted = [mean_deviation[i] for i in i_sorted]
        mean_deviation_baseline_sorted = [mean_deviation_baseline[i] for i in i_sorted]
        ax0 = axs if n_subplots == 1 else axs[0]
        p = ax0.plot(x_labels_sorted, best_sorted, '-o', label=i + 1, markerfacecolor='None')

        i_best = np.argmax(best_sorted)
        ax0.plot(
            x_labels_sorted[i_best], best_sorted[i_best],
            'o', markerfacecolor=p[0].get_color(), markeredgecolor=p[0].get_color()
        )
        if PLOT_ENV_ONLY:
            p = axs[1].plot(x_labels_sorted, env_sorted, '-o', label=i + 1, markerfacecolor='None')
            i_best = np.argmax(env_sorted)
            axs[1].plot(
                x_labels_sorted[i_best], env_sorted[i_best],
                'o', markerfacecolor=p[0].get_color(), markeredgecolor=p[0].get_color()
            )
        if PLOT_TIME:
            axs[i_subplot_time].plot(x_labels_sorted, time_sorted, '-o', label=i + 1, markerfacecolor='None')
        if PLOT_VOLTAGE_DEVIATIONS:
            axs[-1].plot(x_labels_sorted, mean_deviation_sorted, '-o', label=i + 1, markerfacecolor='None')
            axs[-1].plot(
                x_labels_sorted, mean_deviation_baseline_sorted, 'x--',
                label=f"{i + 1}_baseline", markerfacecolor='None'
            )

    return fig, axs


def add_table_legend(setups, fig, varied_columns, column_of_interest, other_columns, axs):
    height_row0 = 0.1
    height_intra_row = 0.11
    if len(setups) > 1:
        col0 = ['\n'.join(wrap(col, 12)) for col in varied_columns if col != 'n']
        setups_nos = np.array(list(range(len(setups)))) + 1
        column_names = [''] + list(setups_nos)
        values = [
            [
                '\n'.join(wrap(str(setup[other_columns.index(column)]), 8))
                for setup in setups
            ]
            for column in varied_columns if column != 'n'
        ]
        table_body = np.concatenate(
            [np.reshape(col0, (len(col0), 1)), np.reshape(values, (np.shape(values)))],
            axis=1
        )
        df = pd.DataFrame(table_body, columns=column_names)

        sum_rows = 1
        for i in range(len(df)):
            cols = list(df.loc[i])
            sum_rows += max([len(col.split('\n')) for col in cols])
        width = (len(df.columns) + 1) * 0.3
        height = len(df) * height_row0 + sum_rows * height_intra_row
        x_low = 1 - height
        if column_of_interest == 'state_space':
            x_low += 0.2
        ax0 = axs if n_subplots == 1 else axs[0]
        table = ax0.table(
            cellText=df.values, colLabels=df.columns, bbox=[1.03, x_low, width, height]
        )
        right = 1 / (1 + width) + 0.04
        left = 0.05 if column_of_interest == 'state_space' else 0.1
        bottom = 1 / (1 + height / 3) + 0.05
        if x_low < - 1:
            plt.subplots_adjust(left=left, right=right, bottom=bottom)
        else:
            plt.subplots_adjust(left=left, right=right)

        cellDict = table.get_celld()
        col0_width = 0.1 if column_of_interest == 'state_space' else 0.25
        cellDict[(0, 0)].set_width(col0_width)
        for i in range(1, len(df) + 1):
            cols = list(df.loc[i - 1])
            max_n_rows = max([len(col.split('\n')) for col in cols])
            for j in range(len(df.columns)):
                cellDict[(i, j)].set_height(height_row0 + height_intra_row * max_n_rows)
            cellDict[(i, 0)].set_width(col0_width)

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        ax0.legend()
        set_width = 6 + len(df.columns) * 0.35
        if column_of_interest == 'state_space':
            set_width += 3
        fig.set_figwidth(set_width)

    return axs, fig


def list_columns_that_vary_between_setups(setups, other_columns):
    varied_columns = []
    for i, setup_i in enumerate(setups):
        for j, setup_j in enumerate(setups):
            if i != j:
                columns_diff_i_j = [
                    column
                    for column, setup_i_, setup_j_ in zip(other_columns, setup_i, setup_j)
                    if setup_i_ != setup_j_
                ]
                for column in columns_diff_i_j:
                    if column not in varied_columns:
                        varied_columns.append(column)

    return varied_columns


def plot_sensitivity_analyses(new_columns, log):
    # loop through each column
    # search for runs that are all the same except for that one columnx changing
    # and plot y axis best score vs x axis value (numerical or categorical)
    # with each line being another set of parameters being fixed with legend
    # each plot being a 2 row subplot with best score / best score env
    if COLUMNS_OF_INTEREST is None:
        columns_of_interest = [
            column for column in new_columns[2:]
            if column not in ['nn_learned', 'time_end', 'machine_id', 'timestamp']
        ]
    else:
        columns_of_interest = COLUMNS_OF_INTEREST

    for column_of_interest in tqdm(columns_of_interest, position=0, leave=True):
        fig, axs = plt.subplots(n_subplots, 1, figsize=(8, 10 / 3 * n_subplots))
        other_columns = [
            column for column in new_columns[2:]
            if column not in [
                column_of_interest, 'nn_learned', 'time_end', 'machine_id',
                'timestamp', 'n_homes_all_test'
            ]
        ]

        plotted_something, axs, setups, state_space_vals = compare_all_runs_for_column_of_interest(
            column_of_interest, other_columns, axs, log
        )
        if plotted_something:
            ax0 = axs if n_subplots == 1 else axs[0]
            if column_of_interest == 'state_space':
                fig, axs = adapt_figure_for_state_space(state_space_vals, axs)
            elif column_of_interest.startswith('year'):
                axs_ = [axs] if n_subplots == 1 else axs
                for ax in axs_:
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # see what varies between setups
            varied_columns = list_columns_that_vary_between_setups(setups, other_columns)
            if column_of_interest in ['state_space', 'type_learning']:
                for i in range(n_subplots - 1):
                    axs[i].axes.xaxis.set_ticklabels([])
                plt.xticks(rotation=90)
            elif column_of_interest in ['rnn_hidden_dim', 'lr']:
                ax0.set_xscale('log')
                for i in range(n_subplots):
                    axs[i].set_xscale('log')
            # remove columns that are irrelevant to the types learning in the current setups
            if column_of_interest != 'type_learning':
                types_learning = [setup[other_columns.index('type_learning')] for setup in setups]
                varied_columns = [
                    column for column in varied_columns
                    if len(column.split('-')) == 1 or column.split('-')[0] in types_learning
                ]
            if len(varied_columns) > 0:
                axs, fig = add_table_legend(
                    setups, fig, varied_columns, column_of_interest, other_columns, axs
                )
            ax0 = axs if n_subplots == 1 else axs[0]
            ylabel = "Savings [£/home/h]" if n_subplots == 1 else "Best score [£/home/h]"
            ax0.set_ylabel(ylabel)
            if PLOT_ENV_ONLY:
                axs[1].set_ylabel(
                    '\n'.join(
                        wrap(
                            "Best score without optimisation-based exploration [£/home/h]",
                            30
                        )
                    )
                )
            if PLOT_TIME:
                axs[i_subplot_time].set_ylabel("Time [s]")
            if PLOT_VOLTAGE_DEVIATIONS:
                if METRIC_VOLTAGE == 'n_violation' and NORMALISED_n_violation:
                    y_label_voltage_ = 'Percentage of Voltage Violations\namong all Bus-Time Steps'
                else:
                    y_label_voltage_ = y_label_voltage[METRIC_VOLTAGE]
                axs[-1].set_ylabel(y_label_voltage_)
            ax_xlabel = axs if n_subplots == 1 else axs[-1]
            x_label = X_LABELS[column_of_interest] if column_of_interest in X_LABELS \
                else column_of_interest
            ax_xlabel.set_xlabel('\n'.join(wrap(x_label, 50)))

            if column_of_interest == 'state_space':
                plt.tight_layout(rect=(0, 0.1, 1, 1))
            else:
                plt.tight_layout()
            fig.savefig(
                f"outputs/results_analysis/{column_of_interest}_sensitivity.pdf",
                bbox_inches='tight', format='pdf', dpi=1200
            )
            plt.close('all')
        elif column_of_interest == 'grdC_n':
            print("column_of_interest grdC_n and did not plot anything")


def remove_key_from_columns_names(new_columns):
    # remove key from column name
    for i in range(len(new_columns)):
        splits = new_columns[i].split('-')
        if len(splits) > 1:
            len_start_remove = len(splits[0]) + 1
            new_columns[i] = new_columns[i][len_start_remove:]

    return new_columns


def remove_duplicates(log, columns0):
    if 'RL-server' in columns0:
        log['syst-server'] = log.apply(
            lambda row: row['RL-server'] if row['syst-server'] is None else row['syst-server'],
            axis=1
        )
        log.drop(columns=['RL-server'], inplace=True)
        columns0.remove('RL-server')
    for col in ['RL-eps', 'RL-alpha']:
        if col in columns0:
            log.drop(columns=[col], inplace=True)
            columns0.remove(col)

    return log, columns0


def filter_current_analysis(log):
    for col, value in FILTER.items():
        if col in log.columns:
            log = log.drop(log[log[col] != value].index)
    log = log.reset_index()

    return log


if __name__ == "__main__":
    results_path = Path("outputs/results")
    results_analysis_path = Path("outputs/results_analysis")

    if not results_analysis_path.exists():
        os.mkdir(results_analysis_path)

    columns0, result_nos, current_run_invalid = get_list_all_fields(results_path)
    columns0.append('syst-assets')
    columns0 += [col for col in COLUMNS_OF_INTEREST if col not in columns0]

    # get the names of all the evaluations methods
    keys_methods = get_names_evaluation_methods(results_path, result_nos, current_run_invalid)
    columns_results_methods = []
    for method in keys_methods:
        for value in [best_score_type, 'p25', 'p75']:
            columns_results_methods.append(f"{value}_{method}")

    for method in keys_methods + ['baseline']:
        for label in ['mean_violation', 'mean_deviation', 'max_deviation', 'n_violation', 'n_hour_violation']:
            columns_results_methods.append(f"{label}_{method}")

    log_path = results_analysis_path / "log_runs.csv"
    log = pd.DataFrame(columns=columns0 + columns_results_methods)

    newly_added_runs = []
    for result_no in result_nos:
        if result_no not in list(log['run']):
            row = get_prm_data_for_a_result_no(results_path, result_no, columns0)
            row = append_metrics_data_for_a_result_no(results_path, result_no, keys_methods, row)
            if row is not None:
                log.loc[len(log.index)] = row
                newly_added_runs.append(row[0])

    log, columns0 = remove_duplicates(log, columns0)
    new_columns, log = remove_columns_that_never_change_and_tidy(
        log, columns0, columns_results_methods
    )
    log, new_columns = add_default_values(log, new_columns)
    log = fix_learning_specific_values(log)
    new_columns = remove_key_from_columns_names(new_columns)
    COLUMNS_OF_INTEREST = remove_key_from_columns_names(COLUMNS_OF_INTEREST)
    log.columns = new_columns + columns_results_methods
    log = compute_best_score_per_run(keys_methods, log)
    log.to_csv(log_path)
    log = filter_current_analysis(log)
    plot_sensitivity_analyses(new_columns, log)

    # plot with and without trajectory
    runs = {
        'Individual time steps': 398,
        'Trajectory': 525,
    }
    if all(run in list(log['run']) for run in runs.values()):
        labels = list(runs.keys())
        medians = np.array(
            [log.loc[log['run'] == run, 'best_score_all'].item() for run in runs.values()]
        )
        percentile_25, percentile_75 = [
            np.array(
                [log.loc[log['run'] == run, f'p{p}_best_score_all'].item() for run in runs.values()]
            ) for p in [25, 75]
        ]

        # Plotting
        matplotlib.rcParams.update({'font.size': font_size})

        x = range(len(labels))
        fig = plt.figure(figsize=(5, 5))
        plt.bar(x, medians, tick_label=labels, alpha=0.7)
        plt.errorbar(
            x, medians, yerr=[medians - percentile_25, percentile_75 - medians],
            fmt='none', color='black', capsize=4
        )
        plt.ylabel('Savings [£/home/month]')
        fig.savefig(
            "outputs/results_analysis/best_trajectory_sensitivity.pdf",
            bbox_inches='tight', format='pdf', dpi=1200
        )
