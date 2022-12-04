
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from src.post_analysis.plotting.plotting_utils import title_and_save
from src.utilities.userdeftools import get_prm_save


def _plot_eval_action_type_repeat(actions_, prm, evaluation_method, labels, i_action, repeat):
    """Plot evaluation actions selected over epochs for one repeat"""
    fig = plt.figure()
    for epoch in range(prm["RL"]["n_epochs"]):
        if actions_[epoch] is None:
            if evaluation_method != 'opt':
                print(f"None in {evaluation_method}")
            continue
        for step in range(len(actions_[epoch])):
            for home in range(len(actions_[epoch][step])):
                if actions_[epoch][step][home][i_action] is None:
                    if evaluation_method != 'opt':
                        print(f"None in {evaluation_method}")
                    continue
                plt.plot(
                    epoch,
                    actions_[epoch][step][home][i_action],
                    'o'
                )
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


def check_model_changes(prm):
    if prm["RL"]["type_learning"] == "q_learning" \
            and (prm["RL"]["initialise_q"] == "zeros") \
            and prm['save']['save_qtables']:
        for repeat in range(prm['RL']['n_repeats']):
            q_tables = np.load(
                prm["paths"]["record_folder"] / f"q_tables_repeat{repeat}.npy", allow_pickle=True
            ).item()
            for evaluation_method in q_tables[0].keys():
                assert not np.all(
                    np.array(q_tables[prm['RL']['n_repeats'] - 1][evaluation_method][0]) == 0
                ), f"q_table for {evaluation_method} is all zeros repeat {repeat}"
    else:
        networks = [
            method for method in prm["RL"]["evaluation_methods"]
            if method not in ["baseline", "opt", "random"]
        ]
        print(f"check_model_changes networks {networks}")
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
                    agents.append(th.load(path / "agent.th"))
                    if prm['ntw']['n'] > 1:
                        mixers.append(th.load(path / "mixer.th"))

                # fc1_bias = "fc1.module.bias" if prm['RL']['data_parallel'] else "fc1.bias"
                # hyper_b_1_bias = "hyper_b_1.module.bias" if prm['RL']['data_parallel'] \
                #     else "hyper_b_1.bias"
                for weight in agents[0].keys():
                    agents_learned[method] = not th.all(agents[0][weight] == agents[-1][weight])
                    if not agents_learned[method]:
                        prm["RL"]["nn_learned"] = False
                        print(f"agents_learned {agents_learned} {weight}")

                check_mixer = prm['ntw']['n'] > 1 and prm["RL"]["mixer"] == "qmix"
                if check_mixer:
                    for weight in mixers[0].keys():
                        mixer_learned[method] = not th.all(mixers[0][weight] == mixers[-1][weight])
                        if not mixer_learned[method]:
                            prm["RL"]["nn_learned"] = False
                            print(f"mixer_learned {mixer_learned} {weight}")

        prm_save = get_prm_save(prm)
        np.save(prm["paths"]["save_inputs"] / "prm", prm_save)

        assert all(agents_learned.values()), f"agent network has not changed {agents_learned}"
        assert all(mixer_learned.values()), f"mixers network has not changed {mixer_learned}"
