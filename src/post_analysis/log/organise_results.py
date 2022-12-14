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
    if subkey not in ignore:
        if subkey == 'n_start_opt':
            print(f"run {result_no} n_start_opt should be n_start_opt_explo")
        elif (
                (is_short_type(subval) or subkey == "state_space")
                and f"{key}-{subkey}" not in columns0
        ):
            columns0.append(f"{key}-{subkey}")
        elif isinstance(subval, dict):
            for subsubkey, subsubval in subval.items():
                new_col = f"{key}-{subkey}-{subsubkey}" not in columns0
                ignore_col = subsubkey in ignore
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
        'n_homes', 'obs_shape', 'results_file', 'n_actions', 'state_shape', 'agents',
        'save', 'groups', 'paths', 'end_decay', 'f_max-loads', 'f_min-loads', 'dt'
    ]
    result_files = os.listdir(results_path)
    result_nos = sorted([int(file.split('n')[1]) for file in result_files if file[0: 3] == "run"])
    columns0 = []
    for result_no in result_nos:
        path_prm = results_path / f"run{result_no}" / 'inputData' / 'prm.npy'
        if path_prm.is_file():
            prm = np.load(path_prm, allow_pickle=True).item()
            for key, val in prm.items():
                if key not in ignore:
                    for subkey, subval in val.items():
                        columns0 = add_subkey_to_list_columns(key, subkey, ignore, subval, columns0)

    columns0 = ["run", "date"] + sorted(columns0)

    return columns0, result_nos


def get_names_evaluation_methods(results_path, result_nos):
    evaluation_methods_found = False
    it = 0
    while not evaluation_methods_found and it < 100:
        it += 1
        path_metrics0 = results_path / f"run{result_nos[-it]}" / 'figures' / 'metrics.npy'
        if path_metrics0.is_file():
            metrics0 = np.load(path_metrics0, allow_pickle=True).item()
            keys_methods = list(metrics0['end_test_bl']['ave'].keys())
            if len(keys_methods) == 16:
                evaluation_methods_found = True
        else:
            shutil.rmtree(results_path / f"run{result_nos[-it]}")

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


def fill_in_log_value_with_run_data(log, row, column, run_no, prm_default, previous_defaults):
    if column in previous_defaults and run_no <= previous_defaults[column][0]:
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


def add_default_values(log, previous_defaults):
    file_name = ''
    # add any default value previously saved row by row
    for row in range(len(log)):
        run_no = log.loc[row, 'run']
        path_default = Path(f"outputs/results/run{run_no}/inputData/prm_with_defaults.pickle")
        if path_default.is_file():
            with open(path_default, "rb") as file:
                prm_default = pickle.load(file)
            for column in log.columns:
                if log.loc[row, column] is None:
                    log = fill_in_log_value_with_run_data(
                        log, row, column, run_no, prm_default, previous_defaults
                    )

    # then replace column by column the missing data with current defaults
    for column in log.columns:
        key, subkey, subsubkey = get_key_subkeys_column(column)
        if key is not None:
            # get you have correct data loaded or load it
            path = Path(f"config_files/input_parameters/{key}.yaml")
            if key != file_name:
                if path.is_file():
                    file_name = key
                    with open(f"config_files/input_parameters/{file_name}.yaml", "rb") as file:
                        default_data = yaml.safe_load(file)
                else:
                    default_data = None
            # replace default value
            log[column] = log[column].apply(
                lambda x: replace_single_default_value(x, default_data, subkey, subsubkey)
            )

    # save all defaults in prm_default row by row
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

    return key, subkey, subsubkey


def get_prm_data_for_a_result_no(results_path, result_no, columns0):
    path_prm = results_path / f"run{result_no}" / 'inputData' / 'prm.npy'
    if path_prm.is_file():
        prm = np.load(path_prm, allow_pickle=True).item()
        if result_no == 802:
            prm['RL']['nn_learned'] = False
        date_str = datetime.fromtimestamp(os.stat(path_prm).st_birthtime).strftime("%d/%m/%Y")
        row = [result_no, date_str]
        for column in columns0[2:]:
            key, subkey, subsubkey = get_key_subkeys_column(column)
            str_state_space = list_obs_to_str(prm['RL']['state_space'])
            if subkey == 'grdC_n' and 'grdC_t' in str_state_space:
                indices = [
                    i for i in range(len(str_state_space) - 6)
                    if str_state_space[i: i + 6] == 'grdC_t'
                ]
                if len(indices) > 0:
                    max_t = max([int(str_state_space[i + 6:].split('_')[0]) for i in indices])
                    prm[key][subkey] = max_t
            if key is None:
                row.append(None)
                print(f"column {column} does not correspond to a prm key")
            elif subsubkey is None:
                if subkey == 'state_space' and subkey in prm[key]:
                    row.append(list_obs_to_str(prm[key][subkey]))
                else:
                    row.append(prm[key][subkey] if subkey in prm[key] else None)

            else:
                row.append(
                    prm[key][subkey][subsubkey]
                    if subkey in prm[key] and subsubkey in prm[key][subkey]
                    else None
                )
    else:
        row = None

    return row


def append_metrics_data_for_a_result_no(results_path, result_no, keys_methods, row):
    path_metrics = results_path / f"run{result_no}" / 'figures' / 'metrics.npy'
    if row is not None and path_metrics.is_file():
        # now add results
        metrics = np.load(path_metrics, allow_pickle=True).item()
        for method in keys_methods:
            method_ = method
            if method not in metrics['end_test_bl']['ave']:
                potential_replacement = [
                    method_metrics for method_metrics in metrics['end_test_bl']['ave']
                    if method_metrics[0: 3] == 'env' and method in method_metrics
                ]
                if len(potential_replacement) == 1:
                    method_ = potential_replacement[0]
            row.append(
                metrics['end_test_bl']['ave'][method_]
                if method_ in metrics['end_test_bl']['ave']
                else None
            )
    else:
        row = None

    return row


def remove_columns_that_never_change_and_tidy(log, columns0):
    new_columns = []
    for column in columns0:
        try:
            unique_value = len(log[column][log[column].notnull()].unique()) == 1
            if not column == "RL-state_space" and unique_value:
                log.drop([column], axis=1, inplace=True)
            else:
                new_columns.append(column)
        except Exception as ex:
            print(f"column {column} ex {ex}")

    # check there are no duplicates
    if len(new_columns) != len(set(new_columns)):
        print(f"len(new_columns) {len(new_columns)} len(set(new_columns)) {len(set(new_columns))}")
        for i in range(len(new_columns)):
            indices = [j for j, x in enumerate(new_columns) if x == new_columns[i]]
            if len(indices) > 1:
                print(f"{new_columns[i]} times {len(indices)}")

    log.columns = new_columns + keys_methods

    return new_columns, log


def compute_best_score_per_run(keys_methods, log):
    keys_methods_not_opt = [method for method in keys_methods if method != 'opt']
    keys_methods_env = [method for method in keys_methods_not_opt if method[0: 3] == 'env']
    keys_methods_opt = [method for method in keys_methods_not_opt if method[0: 3] == 'opt']

    log['best_score'] = log[keys_methods_not_opt].max(axis=1)
    log['best_score_opt'] = log[keys_methods_opt].max(axis=1)
    log['best_score_env'] = log[keys_methods_env].max(axis=1)

    return log


def check_that_only_grdCn_changes_in_state_space(
        other_columns, current_setup, row_setup, initial_setup_row, row
):
    index_state_space = other_columns.index('state_space')
    only_col_of_interest_changes_without_state_space = all(
        current_col == row_col
        for i, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
        if i != index_state_space
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
        if state_space_current_setup == state_space_row_setup:
            only_col_of_interest_changes = True
        else:
            start_n, end_d = min([grdC_n_current_setup, grdC_n_row_setup]), max(
                [grdC_n_current_setup, grdC_n_row_setup])
            expected_diff_str = ''
            for i in range(start_n, end_d):
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
        'ou_stop_episode', 'rnn_hidden_dim', 'start_steps', 'q_learning-alpha'
    ]
    only_col_of_interest_changes = all(
        current_col == row_col
        for i, (current_col, row_col) in enumerate(zip(current_setup, row_setup))
        if other_columns[i] not in columns_irrelevant_for_q_learning_facmac_comparison
    )

    return only_col_of_interest_changes


def annotate_run_nos(
        axs, values_of_interest_sorted, best_score_sorted, best_env_score_sorted, runs_sorted
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


def compare_all_runs_for_column_of_interest(
    column_of_interest, other_columns, axs, log
):
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
        best_score = [log['best_score'].loc[initial_setup_row]]
        best_env_score = [log['best_score_env'].loc[initial_setup_row]]
        time_best_score = [log['time_end'].loc[initial_setup_row]]
        for row in range(len(log)):
            # if row not in rows_considered and all(col in log.loc[row])
            row_setup = log[other_columns].loc[row].values
            new_row = row not in rows_considered
            relevant_cnn = not (
                column_of_interest[0:3] == 'cnn'
                and log['nn_type'].loc[row] != 'cnn'
            )
            relevant_facmac = not (
                column_of_interest[0: 6] == 'facmac'
                and log['type_learning'].loc[row] != 'facmac'
            )

            if column_of_interest == 'grdC_n':
                only_col_of_interest_changes = check_that_only_grdCn_changes_in_state_space(
                    other_columns, current_setup, row_setup, initial_setup_row, row
                )
            elif column_of_interest == 'type_learning':
                only_col_of_interest_changes = only_columns_relevant_learning_type_comparison(
                    other_columns, current_setup, row_setup
                )
            else:
                only_col_of_interest_changes = all(
                    current_col == row_col
                    for current_col, row_col in zip(current_setup, row_setup)
                )

            if new_row and only_col_of_interest_changes and relevant_cnn and relevant_facmac:
                rows_considered.append(row)
                values_of_interest.append(log[column_of_interest].loc[row])
                best_score.append(log['best_score'].loc[row])
                best_env_score.append(log['best_score_env'].loc[row])
                time_best_score.append(log['time_end'].loc[row])

        if len(values_of_interest) > 1:
            all_setups_same_as_0 = all(
                values_of_interest_ == values_of_interest[0]
                for values_of_interest_ in values_of_interest
            )
            runs = log.loc[rows_considered[- len(values_of_interest):], 'run'].values
            if all_setups_same_as_0:
                print(f"runs {runs} equal?")
            else:
                setups.append(current_setup)
                i_sorted = np.argsort(values_of_interest)
                values_of_interest_sorted = [values_of_interest[i] for i in i_sorted]
                best_score_sorted = [best_score[i] for i in i_sorted]
                best_env_score_sorted = [best_env_score[i] for i in i_sorted]
                time_best_score_sorted = [time_best_score[i] for i in i_sorted]
                runs_sorted = [runs[i] for i in i_sorted]
                axs[0].plot(values_of_interest_sorted, best_score_sorted, label=len(setups))
                axs[1].plot(values_of_interest_sorted, best_env_score_sorted, label=len(setups))
                axs[2].plot(
                    values_of_interest_sorted, time_best_score_sorted,
                    label=len(setups)
                )
                axs = annotate_run_nos(
                    axs, values_of_interest_sorted, best_score_sorted,
                    best_env_score_sorted, runs_sorted
                )

                if column_of_interest == 'state_space':
                    x_labels.append(values_of_interest_sorted)
                    best_values.append(best_score_sorted)
                    env_values.append(best_env_score_sorted)
                    time_values.append(time_best_score_sorted)
                plotted_something = True

        setup_no += 1

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
    for i in range(len(all_best_vals)):
        for j in range(len(all_best_vals[i])):
            if all_best_vals[i][j] < 1e-5:
                all_best_vals[i][j] = None
            if all_env_vals[i][j] < 1e-5:
                all_env_vals[i][j] = None
    fig, axs = plt.subplots(3, 1, figsize=(6.4, 11))
    for i in range(len(x_labels)):
        axs[0].plot(all_x_labels, all_best_vals[i], label=i + 1)
        axs[1].plot(all_x_labels, all_env_vals[i], label=i + 1)
        axs[2].plot(all_x_labels, all_time_vals[i], label=i + 1)

    return axs


def add_table_legend(setups, fig, varied_columns, column_of_interest, other_columns, axs):
    height_row0 = 0.1
    height_intra_row = 0.11
    if len(setups) > 1:
        col0 = ['\n'.join(wrap(col, 12)) for col in varied_columns]
        setups_nos = np.array(list(range(len(setups)))) + 1
        column_names = [''] + list(setups_nos)
        values = [
            [
                '\n'.join(wrap(str(setup[other_columns.index(column)]), 8))
                for setup in setups
            ]
            for column in varied_columns
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
    columns_of_interest = [
        column for column in new_columns[2:]
        if column not in ['nn_learned', 'time_end']
    ]
    for column_of_interest in tqdm(columns_of_interest, position=0, leave=True):
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        other_columns = [
            column for column in new_columns[2:]
            if column not in [column_of_interest, 'nn_learned', 'time_end']
        ]

        plotted_something, axs, setups, state_space_vals = compare_all_runs_for_column_of_interest(
            column_of_interest, other_columns, axs, log
        )

        if plotted_something:
            if column_of_interest == 'state_space':
                axs = adapt_figure_for_state_space(state_space_vals, axs)

            # see what varies between setups
            varied_columns = list_columns_that_vary_between_setups(setups, other_columns)

            if column_of_interest == 'state_space':
                plt.xticks(rotation=90)
                axs[0].set_xticks([])
                axs[1].set_xticks([])
            elif column_of_interest == 'rnn_hidden_dim':
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

            axs[0].set_ylabel("best score [??/home/h]")
            axs[1].set_ylabel(
                '\n'.join(wrap("best score with env-based exploration [??/home/h]", 30))
            )
            axs[2].set_ylabel("time [s]")
            axs[2].set_xlabel('\n'.join(wrap(column_of_interest, 50)))

            fig.savefig(f"outputs/results_analysis/{column_of_interest}_sensitivity")
            plt.close('all')
        elif column_of_interest == 'grdC_n':
            print("column_of_interest grdC_n and did not plot anything")


if __name__ == "__main__":
    results_path = Path("outputs/results")
    results_analysis_path = Path("outputs/results_analysis")
    previous_defaults = {
        'n_hidden_layers': [813, 1],
        'aggregate_actions': [813, True],
        'supervised_loss': [813, True],
    }

    if not results_analysis_path.exists():
        os.mkdir(results_analysis_path)

    columns0, result_nos = get_list_all_fields(results_path)

    # get the names of all the evaluations methods
    keys_methods = get_names_evaluation_methods(results_path, result_nos)

    log_path = results_analysis_path / "log_runs.csv"
    log = pd.DataFrame(columns=columns0 + keys_methods)

    newly_added_runs = []
    for result_no in result_nos:
        if result_no not in list(log['run']):
            row = get_prm_data_for_a_result_no(results_path, result_no, columns0)
            row = append_metrics_data_for_a_result_no(results_path, result_no, keys_methods, row)
            if row is not None:
                log.loc[len(log.index)] = row
                newly_added_runs.append(row[0])
            else:
                shutil.rmtree(results_path / f"run{result_no}")

    new_columns, log = remove_columns_that_never_change_and_tidy(log, columns0)

    log = add_default_values(log, previous_defaults=previous_defaults)

    # remove key from column name
    for i in range(len(new_columns)):
        splits = new_columns[i].split('-')
        if len(splits) > 1:
            len_start_remove = len(splits[0]) + 1
            new_columns[i] = new_columns[i][len_start_remove:]
    log.columns = new_columns + keys_methods
    log['share_active'] = log.apply(lambda x: x.n_homes / (x.n_homes + x.n_homesP), axis=1)
    new_columns.append('share_active')
    log = compute_best_score_per_run(keys_methods, log)

    log.to_csv(log_path)

    plot_sensitivity_analyses(new_columns, log)
