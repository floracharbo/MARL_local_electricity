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
from tqdm import tqdm

# plot timing vs performance for n layers / dim layers; runs 742-656
ANNOTATE_RUN_NOS = True
FILTER_N_HOMES = False
COLUMNS_OF_INTEREST = ['n_homes_test']
# COLUMNS_OF_INTEREST = None

FILTER = {
    # 'supervised_loss': False,
    # 'facmac-beta_to_alpha': 0.1,
    'SoC0': 1,
    # 'grdC_n': 2,
    'error_with_opt_to_rl_discharge': False,
    # 'server': True,
    # 'n_repeats': 10,
    # 'facmac-hysteretic': True,
}

best_score_type = 'p50'
# p50 or ave


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


def fix_learning_specific_values(log):
    """If there are no optimisations, this is equivalent to if we had forced optimisations."""
    ave_opt_cols = [
        col for col in log.columns
        if col[0: len(f'{best_score_type}_opt')] == f'{best_score_type}_opt'
    ]
    for i in range(len(log)):
        if all(log[col].loc[i] is None for col in ave_opt_cols):
            if not log['syst-force_optimisation'].loc[i]:
                log.loc[i, 'syst-force_optimisation'] = True
            if log['syst-error_with_opt_to_rl_discharge'].loc[i]:
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
        'plot_profiles', 'plotting_batch', 'description_run', 'type_env', 'n_all_homes',
        'obs_shape', 'results_file', 'n_actions', 'state_shape', 'state_shape_test', 'agents',
        'save', 'groups', 'paths', 'end_decay', 'f_max-loads', 'f_min-loads', 'dt',
        'env_info', 'clust_dist_share', 'f_std_share', 'phi0', 'run_mode',
        'no_flex_action_to_target', 'N', 'n_int_per_hr', 'possible_states', 'n_all',
        'n_opti_constraints', 'dim_states_1', 'facmac-lr_decay_param',
        'facmac-critic_lr_decay_param', 'RL-n_homes_test', 'car-cap',
    ]
    result_files = os.listdir(results_path)
    result_nos = sorted([int(file.split('n')[1]) for file in result_files if file[0: 3] == "run"])
    columns0 = []
    remove_nos = []
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
            shutil.rmtree(results_path / f"run{result_no}")
            remove_nos.append(result_no)

    print(f"delete run(s) {remove_nos}")
    for result_no in remove_nos:
        result_nos.pop(result_nos.index(result_no))

    columns0 = ["run", "date"] + sorted(columns0)
    if 'RL-batch_size' in columns0:
        columns0.pop(columns0.index('RL-batch_size'))

    return columns0, result_nos


def get_names_evaluation_methods(results_path, result_nos):
    evaluation_methods_found = False
    it = 0
    keys_methods = []
    while not evaluation_methods_found and it < len(result_nos):
        it += 1
        path_metrics0 = results_path / f"run{result_nos[-it]}" / 'figures' / 'metrics.npy'
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


def fill_in_log_value_with_run_data(log, row, column, prm_default):
    with open("config_files/default_input_parameters/previous_defaults.yaml", "rb") as file:
        previous_defaults = yaml.safe_load(file)

    timestamp_changes = previous_defaults['timestamp_changes']
    del previous_defaults['timestamp_changes']

    if column in previous_defaults:
        timestamp_change_idx = previous_defaults[column][0]
        timestamp_change = timestamp_changes[timestamp_change_idx]
        timestamp_run = log.loc[row, 'syst-timestamp']
        if timestamp_run < timestamp_change:
            log.loc[row, column] = previous_defaults[column][1]
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


def add_default_values(log):
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
                    log = fill_in_log_value_with_run_data(log, row, column, prm_default)

    share_active_none = log['syst-share_active'].isnull()
    log.loc[share_active_none, 'syst-share_active'] = log.loc[share_active_none].apply(
        lambda x: x['syst-n_homes'] / x['syst-n_homes_all'], axis=1
    )
    n_homes_test_none = log['syst-n_homes_test'].isnull()
    log.loc[n_homes_test_none, 'syst-n_homes_test'] = log.loc[n_homes_test_none].apply(
        lambda x: x['syst-n_homes'], axis=1
    )
    n_homes_all_test_none = log['syst-n_homes_all_test'].isnull()
    log.loc[n_homes_all_test_none, 'syst-n_homes_all_test'] = log.loc[n_homes_all_test_none].apply(
        lambda x: x['syst-n_homes_test'] + x['syst-n_homesP'], axis=1
    )
    share_active_test_none = log['syst-share_active_test'].isnull()
    log.loc[share_active_test_none, 'syst-share_active_test'] = log.loc[share_active_none].apply(
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

    return log


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


def get_own_items(prm, key, subkey):
    potential_exts = ['P', '_test']
    ext = ''
    for potential_ext in potential_exts:
        if subkey[- len(potential_ext):] == potential_ext:
            ext = potential_ext
    if subkey not in prm[key]:
        prm[key][subkey] = 0
    elif prm[key][subkey] is None:
        prm[key][subkey] = 1
    elif isinstance(prm[key][subkey], (np.ndarray, list)):
        prm[key][subkey] = \
            sum(prm[key][subkey]) / prm['syst']['n_homes' + ext] \
            if prm['syst']['n_homes' + ext] > 0 \
            else 0
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
    if subkey not in prm[key] and 'ntw' in prm:
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
        prm = get_own_items(prm, key, subkey)
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


def get_prm_data_for_a_result_no(results_path, result_no, columns0):
    path_prm = results_path / f"run{result_no}" / 'inputData' / 'prm.npy'

    if path_prm.is_file():
        prm = np.load(path_prm, allow_pickle=True).item()
        if 'type_learning' not in prm['RL']:
            os.remove(path_prm)
            return None
        if result_no == 802:
            prm['RL']['nn_learned'] = False
        if 'cap' in prm['car'] and 'caps' not in prm['car']:
            prm['car']['caps'] = prm['car']['cap']
        date_str = datetime.fromtimestamp(os.stat(path_prm).st_birthtime).strftime("%d/%m/%Y")
        row = [result_no, date_str]
        for column in columns0[2:]:
            key, subkey, subsubkey = get_key_subkeys_column(column)
            prm, key, subkey, subsubkey = data_specific_modifications(prm, key, subkey, subsubkey)

            if key is None:
                row.append(None)
                if column != 'RL-n_homes':
                    print(f"column {column} does not correspond to a prm key")
            elif subsubkey is None:
                if subkey == 'state_space' and subkey in prm[key]:
                    row.append(list_obs_to_str(prm[key][subkey]))
                elif subkey == 'caps':
                    x = prm[key][subkey]
                    row.append(int(x[0]) if isinstance(x, np.ndarray) and all(
                        x[i] == x[0] for i in range(len(x))) else str(x))
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
    path_metrics = results_path / f"run{result_no}" / 'figures' / 'metrics.npy'
    if row is not None and path_metrics.is_file():
        # now add results
        metrics = np.load(path_metrics, allow_pickle=True).item()
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
    else:
        row = None

    return row


def remove_columns_that_never_change_and_tidy(log, columns0, columns_results_methods):
    new_columns = []
    do_not_remove = [
        'syst-server', "RL-state_space", 'RL-trajectory', 'RL-type_learning',
        'syst-n_homes', 'syst-share_active_test', 'syst-force_optimisation'
    ]
    for column in columns0:
        unique_value = len(log[column][log[column].notnull()].unique()) == 1
        if column not in do_not_remove and unique_value:
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
            if len(row[f'method_{score}']) > 0 else None,
            axis=1
        )

        log[f"p75_{score}"] = log.apply(
            lambda row: row[f"p75_{row[f'method_{score}'][0][4:]}"]
            if len(row[f'method_{score}']) > 0 else None,
            axis=1
        )
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
        row, indexes_columns_ignore_q_learning
):
    index_state_space = other_columns.index('state_space')
    indexes_ignore = [index_state_space]
    if current_setup[other_columns.index('type_learning')] == 'q_learning':
        indexes_ignore += indexes_columns_ignore_q_learning
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


def only_columns_relevant_learning_type_comparison(
    other_columns, current_setup, row_setup
):
    columns_irrelevant_for_q_learning_facmac_comparison = [
        'act_noise', 'agent_facmac', 'buffer_size', 'cnn_kernel_size',
        'cnn_out_channels', 'facmac-batch_size', 'facmac-critic_lr',
        'hyper_initialization_nonzeros', 'lr', 'mixer', 'n_hidden_layers',
        'n_hidden_layers_critic', 'nn_type', 'nn_type_critic', 'obs_agent_id',
        'ou_stop_episode', 'rnn_hidden_dim', 'start_steps', 'q_learning-alpha',
        'gamma', 'timestamp', 'instant_feedback'
    ]
    only_col_of_interest_changes = all(
        current_col == row_col or (
            not isinstance(current_col, str) and np.isnan(current_col)
            and not isinstance(row_col, str) and np.isnan(row_col)
        )
        for i, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
        if other_columns[i] not in columns_irrelevant_for_q_learning_facmac_comparison
    )

    return only_col_of_interest_changes


def annotate_run_nos(
        axs, values_of_interest_sorted, best_score_sorted,
        best_env_score_sorted, runs_sorted
):
    if ANNOTATE_RUN_NOS:
        for i, (x, best_score, best_env_score, run) in enumerate(zip(
                values_of_interest_sorted, best_score_sorted,
                best_env_score_sorted, runs_sorted
        )):
            axs[0].annotate(
                run, (x, best_score), textcoords="offset points", xytext=(0, 10),
                ha='center'
            )
            axs[1].annotate(
                run, (x, best_env_score), textcoords="offset points", xytext=(0, 10),
                ha='center'
            )

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
    ]
    indexes_columns_ignore_q_learning = [
        other_columns.index(col) for col in columns_irrelevant_to_q_learning if col in other_columns
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
    while len(rows_considered) < len(log):
        initial_setup_row = [i for i in range(len(log)) if i not in rows_considered][0]
        rows_considered.append(initial_setup_row)
        current_setup = log[other_columns].loc[initial_setup_row].values
        values_of_interest = [log[column_of_interest].loc[initial_setup_row]]
        best_scores = {k: {} for k in ['all', 'env']}
        for k in ['all', 'env']:
            best_scores[k][best_score_type] = [log[f'best_score_{k}'].loc[initial_setup_row]]
            for p in [25, 75]:
                best_scores[k][f'p{p}'] = [log[f'p{p}_best_score_{k}'].loc[initial_setup_row]]
        time_best_score = [log['time_end'].loc[initial_setup_row]]
        for row in range(len(log)):
            row_setup = [log[col].loc[row] for col in other_columns]
            new_row = row not in rows_considered
            relevant_cnn = not (
                column_of_interest[0:3] == 'cnn'
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
                    row, indexes_columns_ignore_q_learning
                )

            elif column_of_interest == 'type_learning':
                only_col_of_interest_changes = only_columns_relevant_learning_type_comparison(
                    other_columns, current_setup, row_setup
                )
            else:
                if column_of_interest == 'supervised_loss_weight':
                    indexes_ignore = [other_columns.index('supervised_loss')]
                elif column_of_interest == 'state_space':
                    indexes_ignore = [other_columns.index('grdC_n')]
                elif current_setup[other_columns.index('type_learning')] == 'q_learning':
                    indexes_ignore = indexes_columns_ignore_q_learning
                else:
                    indexes_ignore = []
                only_col_of_interest_changes = all(
                    current_col == row_col or (
                        (not isinstance(current_col, str) and np.isnan(current_col))
                        and (not isinstance(row_col, str) and np.isnan(row_col))
                    )
                    for i_col, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
                    if i_col not in indexes_ignore
                )

            n_homes_on_laptop_only = True
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
            if new_row and only_col_of_interest_changes and relevant_data:
                rows_considered.append(row)

                values_of_interest.append(log[column_of_interest].loc[row])
                for k in ['all', 'env']:
                    best_scores[k][best_score_type].append(log[f'best_score_{k}'].loc[row])
                    for p in [25, 75]:
                        best_scores[k][f'p{p}'].append(log[f'p{p}_best_score_{k}'].loc[row])

                time_best_score.append(log['time_end'].loc[row])

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
                runs_sorted = [runs[i] for i in i_sorted]
                ls = '--' if log.loc[rows_considered[-1], 'server'] else '-'
                values_of_interest_sorted_k, best_scores_sorted = remove_nans_best_scores_sorted(
                    values_of_interest_sorted, best_scores_sorted
                )
                for ax_i, k in enumerate(['all', 'env']):
                    label = \
                        current_setup[other_columns.index('type_learning')] + f"({len(setups)})" \
                        if column_of_interest == 'n_homes' \
                        else len(setups)
                    p = axs[ax_i].plot(
                        values_of_interest_sorted_k[k],
                        best_scores_sorted[k][best_score_type],
                        'o', label=label, linestyle=ls,
                        markerfacecolor='None'
                    )
                    colour = p[0].get_color()
                    i_best = np.argmax(best_scores_sorted[k][best_score_type])
                    axs[ax_i].plot(
                        values_of_interest_sorted_k[k][i_best],
                        best_scores_sorted[k][best_score_type][i_best],
                        'o', markerfacecolor=colour, markeredgecolor=colour
                    )
                    axs[ax_i].fill_between(
                        values_of_interest_sorted_k[k],
                        best_scores_sorted[k]['p25'],
                        best_scores_sorted[k]['p75'],
                        alpha=0.2
                    )
                    if column_of_interest == 'n_epochs':
                        axs[ax_i].set_yscale('log')
                axs[2].plot(
                    values_of_interest_sorted, time_best_score_sorted, 'o',
                    label=label,
                    linestyle=ls,
                    markerfacecolor='None',
                    color=colour
                )
                for i in range(len(values_of_interest_sorted) - 1):
                    if values_of_interest_sorted[i + 1] == values_of_interest_sorted[i]:
                        print(
                            f"we have two values for {column_of_interest} = "
                            f"{values_of_interest_sorted[i]}, "
                            f"runs {runs_sorted[i]} and {runs_sorted[i+1]}"
                        )
                axs = annotate_run_nos(
                    axs, values_of_interest_sorted, best_scores_sorted['all'][best_score_type],
                    best_scores_sorted['env'][best_score_type], runs_sorted
                )
                if column_of_interest == 'state_space':
                    x_labels.append(values_of_interest_sorted)
                    best_values.append(best_scores_sorted['all'][best_score_type])
                    env_values.append(best_scores_sorted['env'][best_score_type])
                time_values.append(time_best_score_sorted)
                plotted_something = True

        setup_no += 1

    if len(time_values) > 1:
        end_time_best_score_sorted = [time_values_[-1] for time_values_ in time_values]
        if max(end_time_best_score_sorted) / min(end_time_best_score_sorted) > 30:
            axs[2].set_yscale('log')

    state_space_vals = [x_labels, best_values, env_values, time_values] \
        if column_of_interest == 'state_space' else None

    return plotted_something, axs, setups, state_space_vals


def adapt_figure_for_state_space(state_space_vals, axs):
    x_labels, best_values, env_values, time_values = state_space_vals
    all_x_labels = []
    x_labels_flattened = list(chain.from_iterable(x_labels))
    for label in x_labels_flattened:
        if label not in all_x_labels:
            all_x_labels.append(label)

    all_best_vals = np.empty((len(x_labels), len(all_x_labels)))
    all_env_vals = np.empty((len(x_labels), len(all_x_labels)))
    all_time_vals = np.empty((len(x_labels), len(all_x_labels)))

    for i in range(len(x_labels)):
        for j in range(len(x_labels[i])):
            idx_value = all_x_labels.index(x_labels[i][j])
            all_best_vals[i, idx_value] = best_values[i][j]
            all_env_vals[i, idx_value] = env_values[i][j]
            all_time_vals[i, idx_value] = time_values[i][j]

    plt.close()

    i_sorted = np.argsort(all_x_labels)
    x_labels_sorted = [all_x_labels[i] for i in i_sorted]

    fig, axs = plt.subplots(3, 1, figsize=(6.4, 11))
    for i, (best, env, time) in enumerate(zip(all_best_vals, all_env_vals, all_time_vals)):
        best_sorted = [best[i] for i in i_sorted]
        env_sorted = [env[i] for i in i_sorted]
        time_sorted = [time[i] for i in i_sorted]
        p = axs[0].plot(x_labels_sorted, best_sorted, 'o', label=i + 1, markerfacecolor='None')
        i_best = np.argmax(best_sorted)
        axs[0].plot(
            x_labels_sorted[i_best], best_sorted[i_best],
            'o', markerfacecolor=p[0].get_color(), markeredgecolor=p[0].get_color()
        )
        p = axs[1].plot(x_labels_sorted, env_sorted, 'o', label=i + 1, markerfacecolor='None')
        i_best = np.argmax(env_sorted)
        axs[0].plot(
            x_labels_sorted[i_best], env_sorted[i_best],
            'o', markerfacecolor=p[0].get_color(), markeredgecolor=p[0].get_color()
        )
        axs[2].plot(x_labels_sorted, time_sorted, 'o', label=i + 1, markerfacecolor='None')

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
        table = axs[0].table(
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
        axs[0].legend()
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
    font = {'size': 7}

    matplotlib.rc('font', **font)
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
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        other_columns = [
            column for column in new_columns[2:]
            if column not in [
                column_of_interest, 'nn_learned', 'time_end', 'machine_id',
                'timestamp', 'n_homes_all', 'n_homes_all_test'
            ]
        ]

        plotted_something, axs, setups, state_space_vals = compare_all_runs_for_column_of_interest(
            column_of_interest, other_columns, axs, log
        )

        if plotted_something:
            if column_of_interest == 'state_space':
                fig, axs = adapt_figure_for_state_space(state_space_vals, axs)

            # see what varies between setups
            varied_columns = list_columns_that_vary_between_setups(setups, other_columns)

            if column_of_interest in ['state_space', 'type_learning']:
                axs[0].axes.xaxis.set_ticklabels([])
                axs[1].axes.xaxis.set_ticklabels([])
                plt.xticks(rotation=90)
            elif column_of_interest in ['rnn_hidden_dim', 'lr']:
                axs[0].set_xscale('log')
                axs[1].set_xscale('log')
                axs[2].set_xscale('log')

            # remove columns that are irrelevant to the types learning in the current setups
            if column_of_interest != 'type_learning':
                types_learning = [setup[other_columns.index('type_learning')] for setup in setups]
                varied_columns = [
                    column for column in varied_columns
                    if len(column.split('-')) == 1 or column.split('-')[0] in types_learning
                ]

            axs, fig = add_table_legend(
                setups, fig, varied_columns, column_of_interest, other_columns, axs
            )

            axs[0].set_ylabel("best score [£/home/h]")
            axs[1].set_ylabel(
                '\n'.join(
                    wrap("best score with without optimisation-based exploration [£/home/h]", 30)
                )
            )
            axs[2].set_ylabel("time [s]")
            axs[2].set_xlabel('\n'.join(wrap(column_of_interest, 50)))
            if column_of_interest == 'state_space':
                plt.tight_layout(rect=(0, 0.1, 1, 1))

            fig.savefig(f"outputs/results_analysis/{column_of_interest}_sensitivity")
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
        log = log.drop(log[log[col] != value].index)
    log = log.reset_index()

    return log


if __name__ == "__main__":
    results_path = Path("outputs/results")
    results_analysis_path = Path("outputs/results_analysis")

    if not results_analysis_path.exists():
        os.mkdir(results_analysis_path)

    columns0, result_nos = get_list_all_fields(results_path)

    # get the names of all the evaluations methods
    keys_methods = get_names_evaluation_methods(results_path, result_nos)
    columns_results_methods = []
    for method in keys_methods:
        for value in [best_score_type, 'p25', 'p75']:
            columns_results_methods.append(f"{value}_{method}")

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
    log = add_default_values(log)
    log = fix_learning_specific_values(log)
    new_columns = remove_key_from_columns_names(new_columns)
    log.columns = new_columns + columns_results_methods
    log = compute_best_score_per_run(keys_methods, log)
    log.to_csv(log_path)
    log = filter_current_analysis(log)
    plot_sensitivity_analyses(new_columns, log)
