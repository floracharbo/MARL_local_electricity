
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from src.post_analysis.plotting.plotting_utils import title_and_save
from src.utilities.userdeftools import get_prm_save


def _plot_eval_action_type_repeat(actions_, prm, evaluation_method, labels, i_action, repeat):
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
    if prm["RL"]["type_learning"] == "q_learning":
        return
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

            agents_learned[method] = not all(a0 == a1 for a0, a1 in zip(agents[0]["fc1.bias"], agents[-1]["fc1.bias"]))
            mixer_learned[method] = prm['ntw']['n'] == 1 or not prm["RL"]["mixer"] == "qmix" or not all(
                mixer0 == mixer1 for mixer0, mixer1 in zip(mixers[0]["hyper_b_1.bias"], mixers[-1]["hyper_b_1.bias"])
                )
            if not (agents_learned and mixer_learned):
                prm["RL"]["nn_learned"] = False
                print(f"agents_learned {agents_learned} mixer_learned {mixer_learned}")

    prm_save = get_prm_save(prm)
    np.save(prm["paths"]["save_inputs"] / "prm", prm_save)

    assert all(agents_learned.values()), f"agent network has not changed {agents_learned}"
    assert all(mixer_learned.values()), f"mixers network has not changed {mixer_learned}"
