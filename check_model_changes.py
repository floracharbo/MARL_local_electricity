import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

# 764 - 10 batch
# 765 - 64 batch

# 770 64
# 771 10

# 608 10 - random: -0.0125053596241545, env_r_c: 0.0915180588975196
# 609 64 - random: -0.0125053596241545, env_r_c: 0.09150014818818443
# 610 lr / critic lr 1e-1 from 1e-6 and batch size 2; random -0.0125053596241545, env_r_c 0.06258718891171977
# 611 lr / critic lr 1e-2 from 1e-6 and batch size 2; random -0.0125053596241545, env_r_c -0.005191469272441462

for run in range(608, 611):
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
            print(f"run {run} agent changes at {i}")
            break


    for i in range(len(agents) - 1):
        if not all(mixers[0]["hyper_b_1.bias"] == mixers[i + 1]["hyper_b_1.bias"]):
            print(f"run {run} mixer change at {i}")
            break

    files_actions = [file for file in os.listdir(path) if file[0:len("eval_actions")] == "eval_actions"]
    nos = [int((file.split("_")[-1][4:]).split(".")[0]) for file in files_actions]
    nos.sort()
    actions = []
    for i in nos:
        actions.append(np.load(path / f"eval_actions_ridx{i}.npy", allow_pickle=True).item())

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
