
import os

import matplotlib.pyplot as plt
import torch as th

from src.post_analysis.plotting.plotting_utils import title_and_save


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
            for i_action in range(prm['RL']['dim_actions']):
                _plot_eval_action_type_repeat(
                    actions_, prm, evaluation_method, labels, i_action, repeat
                )


def check_model_changes(prm):
    change = True
    if prm["RL"]["type_learning"] == "q_learning":
        return
    networks = [
        method for method in prm["RL"]["evaluation_methods"]
        if method not in ["baseline", "opt", "random"]
    ]
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

            assert not all(agents[0]["fc1.bias"] == agents[-1]["fc1.bias"]), \
                "agent network has not changed"
            if prm['ntw']['n'] > 1:
                assert not all(
                    mixers[0]["hyper_b_1.bias"] == mixers[-1]["hyper_b_1.bias"]
                ),  "mixers network has not changed"
