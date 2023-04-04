
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from src.post_analysis.plotting.plotting_utils import title_and_save
from src.utilities.userdeftools import get_prm_save


def _plot_eval_action_type_repeat(actions_, prm, evaluation_method, labels, i_action, repeat):
    """Plot evaluation actions selected over epochs for one repeat"""
    n_intervals = 50
    density_matrix = np.zeros((prm["RL"]["n_epochs"], n_intervals))
    min_action = np.min(actions_[:, :, :, i_action])
    max_action = np.max(actions_[:, :, :, i_action])
    intervals = np.linspace(min_action, max_action, n_intervals)

    for epoch in range(prm["RL"]["n_epochs"]):
        if actions_[epoch] is None:
            if evaluation_method != 'opt':
                print(f"None in {evaluation_method}")
            continue
        for step in range(len(actions_[epoch])):
            for home in range(len(actions_[epoch][step])):
                action = actions_[epoch][step][home][i_action]
                if action is None:
                    if evaluation_method != 'opt':
                        print(f"None in {evaluation_method}")
                    continue
                i_interval = np.where(action >= intervals)[0][-1]
                density_matrix[epoch, i_interval] += 1
                # plt.plot(
                #     epoch,
                #     action,
                #     'o'
                # )
    fig = plt.figure()
    plt.imshow(np.transpose(density_matrix), interpolation='none')
    # min_label = min_action + (0.05 - min_action % 0.05)
    # max_label = max_action - (max_action % 0.01)
    y_labels = [f"{label:.2f}" for label in np.linspace(min_action, max_action, 5)]

    ytickslocs = plt.gca().get_yticks()
    plt.yticks(ytickslocs[1:-1], y_labels)

    plt.ylabel(labels[i_action])
    plt.xlabel("Epoch")
    title = f"actions {evaluation_method} {labels[i_action]} {repeat}"
    title_and_save(title, fig, prm)


def plot_eval_action(record, prm):
    """Plot actions selected during evaluation phase over the epochs for all repeats."""
    actions = record.eval_actions
    if len(list(actions.keys())) == 0:
        return

    if prm["RL"]["aggregate_actions"]:
        labels = prm["RL"]["action_labels_aggregate"]
    else:
        labels = prm["RL"]["action_labels_disaggregate"]
    for evaluation_method in prm["RL"]["evaluation_methods"]:
        if evaluation_method == "baseline" \
                or any(len(actions[repeat]) == 0 for repeat in range(prm["RL"]["n_repeats"])):
            continue
        for repeat in range(prm["RL"]["n_repeats"]):
            actions_ = actions[repeat][evaluation_method]
            for i_action in range(prm['RL']['dim_actions_1']):
                _plot_eval_action_type_repeat(
                    actions_, prm, evaluation_method, labels, i_action, repeat
                )


def check_model_changes_q_learning(prm):
    for repeat in range(prm['RL']['n_repeats']):
        q_tables = np.load(
            prm["paths"]["record_folder"] / f"q_tables_repeat{repeat}.npy", allow_pickle=True
        ).item()
        for evaluation_method in q_tables[0].keys():
            assert not np.all(
                np.array(q_tables[prm['RL']['n_epochs'] - 1][evaluation_method][0]) == 0
            ), f"q_table for {evaluation_method} is all zeros repeat {repeat}"


def check_model_changes_facmac(prm):
    networks = [
        method for method in prm["RL"]["evaluation_methods"]
        if method not in ["baseline", "opt", "random"]
    ]
    agents_learned = {}
    mixer_learned = {}
    prm["RL"]["nn_learned"] = True
    for method in networks:
        folders = [
            folder for folder in os.listdir(prm["paths"]["record_folder"])
            if folder[0:len(f"models_{method}")] == f"models_{method}"
        ]
        if len(folders) > 0:
            nos = [int(folder.split("_")[-1]) for folder in folders]
            nos.sort()
            agents, mixers = [], []
            for no in nos:
                path = prm["paths"]["record_folder"] / f"models_{method}_{no}"
                try:
                    agent = th.load(path / "agent.th")
                    mixer = th.load(path / "mixer.th")
                except Exception:
                    agent = th.load(path / "agent.th", map_location=th.device('cpu'))
                    mixer = th.load(path / "mixer.th", map_location=th.device('cpu'))
                agents.append(agent)
                if prm['syst']['n_homes'] > 1:
                    mixers.append(mixer)

            for weight in agents[0].keys():
                agents_learned[method] = not th.all(agents[0][weight] == agents[-1][weight])
                if not agents_learned[method]:
                    prm["RL"]["nn_learned"] = False
                    print(f"agents_learned {agents_learned} {weight}")

            check_mixer = prm['syst']['n_homes'] > 1 and prm["RL"]["mixer"] == "qmix"
            if check_mixer:
                for weight in mixers[0].keys():
                    mixer_learned[method] = not th.all(mixers[0][weight] == mixers[-1][weight])
                    if not mixer_learned[method]:
                        prm["RL"]["nn_learned"] = False
                        print(f"mixer_learned {mixer_learned} {weight}")

    prm_save = get_prm_save(prm)
    np.save(prm['paths']["folder_run"] / "inputData" / "prm", prm_save)

    assert all(agents_learned.values()), f"agent network has not changed {agents_learned}"
    assert all(mixer_learned.values()), f"mixers network has not changed {mixer_learned}"


def check_model_changes(prm):
    if prm["RL"]["type_learning"] == "q_learning" \
            and (prm["RL"]["initialise_q"] == "zeros") \
            and prm['save']['save_qtables']:
        check_model_changes_q_learning(prm)

    else:
        check_model_changes_facmac(prm)
