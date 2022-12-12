import matplotlib.pyplot as plt
import numpy as np

from src.utilities.userdeftools import distr_learning, initialise_dict


def _save(prm):
    fig_folder = prm['paths']['fig_folder']
    rl = prm["RL"]
    np.save(fig_folder / 'state_space_0', rl["state_space_0"])
    np.save(fig_folder / 'action_state_space_0',
            rl["action_state_space_0"])
    np.save(fig_folder / 'metrics', rl["metrics"])


def _get_mean_rewards_from_record(prm, record):
    record.get_mean_rewards(
        prm,
        prm["RL"]["action_state_space_0"],
        prm["RL"]["state_space_0"],
        prm["save"]["eval_entries_plot"]
    )
    prm["RL"]["metrics"], prm["RL"]["metrics_entries"] \
        = record.get_metrics(
        prm, prm["save"]["eval_entries_plot"]
    )

    prm["RL"]["mean_eval_rewards_per_hh"] = record.mean_eval_rewards_per_hh

    return prm


def _eval_entries_plot_colours(prm):
    rl = prm["RL"]
    evaluation_methods = rl['evaluation_methods']
    if prm['syst']['n_homes'] == 1:
        evaluation_methods = [
            evaluation_method for evaluation_method in evaluation_methods
            if evaluation_method in ['baseline', 'opt']
            or distr_learning(evaluation_method) == 'd']
        if len([
            evaluation_method for evaluation_method in evaluation_methods
            if len(evaluation_method.split('_')) > 1
        ]) == 0:
            evaluation_methods = [
                evaluation_method for evaluation_method in rl['evaluation_methods']
                if evaluation_method in ['baseline', 'opt']
                or distr_learning(evaluation_method) == 'c'
            ]

    prm["save"]["eval_entries_plot"] = [
        evaluation_method
        for evaluation_method in evaluation_methods
        if evaluation_method != 'baseline'
    ] + ['baseline'] if 'baseline' in evaluation_methods \
        else evaluation_methods
    methods_to_plot = {}
    for evaluation_method in evaluation_methods:
        methods_to_plot[evaluation_method] = '-'
    eval_entries_distr = [
        evaluation_method for evaluation_method in evaluation_methods
        if len(evaluation_method.split('_')) > 1 and distr_learning(evaluation_method) == 'd'
    ]
    eval_entries_centr = [
        evaluation_method for evaluation_method in evaluation_methods
        if len(evaluation_method.split('_')) > 1 and distr_learning(evaluation_method) == 'c'
    ]
    other_eval_entries = [
        evaluation_method
        for evaluation_method in evaluation_methods
        if len(evaluation_method.split("_")) == 1
    ]

    prm["save"]["eval_entries_plot_indiv"] = [
        evaluation_method for evaluation_method in prm["save"]["eval_entries_plot"]
        if len(evaluation_method.split('_')) > 1
        and distr_learning(evaluation_method) not in ['Cc0', 'Cd0']
    ]
    prm["save"]["base_entries"] = eval_entries_distr if len(eval_entries_distr) > 0 \
        else eval_entries_centr
    prm["save"]["base_entries"] += other_eval_entries

    red = (234 / 255, 53 / 255, 37 / 255)
    for method in ['env_r_d', 'env_r_c']:
        prm['save']['colourse'][method] = red
    prm['save']['colourse']['opt'] = 'grey'
    env_d_d, opt_d_d = [
        [method for method in prm['save']['colourse'] if method[0: len(method0)] == method0][0]
        for method0 in ['env_d_d', 'opt_d_d']
    ]
    green = prm['save']['colourse'][env_d_d]
    prm['save']['colourse'][opt_d_d] = green

    return prm


def initialise_variables(prm, spaces, record):
    plt.rcParams['font.size'] = '11'
    rl = prm['RL']
    for space in ['action_state_space_0', 'state_space_0']:
        rl[space] = initialise_dict(range(rl['n_repeats']))

    rl['eval_rewards'] = record.eval_rewards  # [repeat][method][epoch]
    prm["save"]["n_window"] = int(
        max(min(100, rl['n_all_epochs'] / 10), 2)
    )
    prm["save"]["save_qtables"] = record.save_qtables
    spaces.new_state_space(rl['state_space'])
    rl["q_tables"], rl["counters"] = record.q_tables, record.counter
    if rl["type_env"] == "discrete":
        rl["possible_states"] = record.possible_states \
            if record.possible_states > 0 else 1
    else:
        rl["possible_states"] = None

    if rl['type_learning'] == 'q_learning':
        if isinstance(rl["q_tables"][0], list):
            if len(rl["counters"][0]) > 0:
                rl["q_entries"] = list(rl["counters"][0][0].keys())
                record.save_qtables = True
        else:
            rl["q_entries"] = list(rl["q_tables"][0][0].keys()) \
                if record.save_qtables else list(rl["q_tables"][0].keys())
    else:
        rl["q_entries"] = None
    if 'plot_profiles' not in prm['save']:
        prm['save']['plot_profiles'] = False

    prm = _eval_entries_plot_colours(prm)
    prm = _get_mean_rewards_from_record(prm, record)

    if prm['save']['save_run']:
        _save(prm)

    return prm
