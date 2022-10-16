
import os

import matplotlib.pyplot as plt
import torch as th
from post_analysis.plotting.plotting_utils import title_and_save


def _plot_eval_action_type_repeat(actions_, prm, type_eval, labels, i_action, repeat):
    fig = plt.figure()
    for epoch in range(prm["RL"]["n_epochs"]):
        if actions_[epoch] is None:
            if type_eval != 'opt':
                print(f"None in {type_eval}")
            continue
        for step in range(len(actions_[epoch])):
            for home in range(len(actions_[epoch][step])):
                if actions_[epoch][step][home][i_action] is None:
                    if type_eval != 'opt':
                        print(f"None in {type_eval}")
                    continue
                plt.plot(
                    epoch,
                    actions_[epoch][step][home][i_action],
                    'o'
                )
    plt.ylabel(labels[i_action])
    plt.xlabel("Epoch")
    title = f"actions {type_eval} labels[i_action] {repeat}"
    title_and_save(title, fig, prm)


def plot_eval_action(record, prm):
    actions = record.eval_actions
    if len(list(actions.keys())) == 0:
        return

    if prm["RL"]["aggregate_actions"]:
        labels = prm["RL"]["action_labels_aggregate"]
    else:
        labels = prm["RL"]["action_labels_disaggregate"]

    for type_eval in prm["RL"]["type_eval"]:
        if type_eval == "baseline" \
                or any(len(actions[repeat]) == 0
                       for repeat in range(prm["RL"]["n_repeats"])):
            continue
        for repeat in range(prm["RL"]["n_repeats"]):
            actions_ = actions[repeat][type_eval]
            for i_action in range(prm['RL']['dim_actions']):
                _plot_eval_action_type_repeat(
                    actions_, prm, type_eval, labels, i_action, repeat
                )


def check_model_changes(prm):
    if prm["RL"]["type_learning"] == "q_learning":
        return
    networks = [
        t for t in prm["RL"]["type_eval"]
        if t not in ["baseline", "opt", "random"]
    ]
    for t in networks:
        folders = [
            folder for folder in os.listdir(prm["paths"]["record_folder"])
            if folder[0:len(f"models_{t}")] == f"models_{t}"
        ]
        if len(folders) > 0:
            nos = [int(folder.split("_")[-1]) for folder in folders]
            nos.sort()
            agents, mixers = [], []
            for no in nos:
                path = prm["paths"]["record_folder"] / f"models_{t}_{no}"
                agents.append(th.load(path / "agent.th"))
                mixers.append(th.load(path / "mixer.th"))

            assert not all(agents[0]["fc1.bias"] == agents[-1]["fc1.bias"]), \
                "agent network has not changed"

            assert not all(
                mixers[0]["hyper_b_1.bias"] == mixers[-1]["hyper_b_1.bias"]
            ), "mixers network has not changed"
