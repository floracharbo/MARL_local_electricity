import os
import torch as th
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 764 - 10 batch
# 765 - 64 batch
for run in [764, 765]:
    path = Path(f'/Users/floracharbonnier/OneDrive - Nexus365/DPhil/Python/Phase2/results/run{run}/record')
    folders = [folder for folder in os.listdir(path) if folder[0:len("models_env_r_c")] == "models_env_r_c"]
    nos = [int(folder.split("_")[-1]) for folder in folders]
    nos.sort()
    agents, mixers, opts = [], [], []


    for no in nos:
        agents.append(th.load(path / f"models_env_r_c_{no}/agent.th"))
        mixers.append(th.load(path / f"models_env_r_c_{no}/mixer.th"))
        opts.append(th.load(path / f"models_env_r_c_{no}/opt.th"))


    for i in range(len(agents) - 1):
        if not all(agents[0]["fc1.bias"] == agents[i + 1]["fc1.bias"]):
            print(f"agent changes at {i}")
            break


    for i in range(len(agents) - 1):
        if not all(mixers[0]["hyper_b_1.bias"] == mixers[i + 1]["hyper_b_1.bias"]):
            print(f"mixer change at {i}")
            break

    files_actions = [file for file in os.listdir(path) if file[0:len("train_actions")] == "train_actions"]
    nos = [int((file.split("_")[-1][4:]).split(".")[0]) for file in files_actions]
    nos.sort()
    actions = []
    for i in nos:
        actions.append(np.load(path / f"train_actions_ridx{i}.npy", allow_pickle=True).item())

    for type_train in ["random", "env_r_c"]:
        for ridx in nos:
            for mu in range(3):
                fig = plt.figure()
                for epoch in range(len(actions[ridx][type_train])):
                    for t in range(len(actions[ridx][type_train][epoch])):
                        for home in range(len(actions[ridx][type_train][epoch][t])):
                            plt.plot(epoch, actions[ridx][type_train][epoch][t][home][mu], 'o')
                title = f"actions {type_train} mu {mu} {ridx}"
                plt.title(title)
                fig.savefig(path / title)
