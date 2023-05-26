import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandapower.plotting as plot
import seaborn as sns

from src.post_analysis.plotting.plotting_utils import (formatting_figure,
                                                       title_and_save)
from src.utilities.userdeftools import (data_source, initialise_dict,
                                        reward_type)


def _plot_last_epochs_actions(
        list_repeat, means, e, method, prm, all_vals, ax, xs, lw_mean, linestyles
):
    means[e][method] = []
    for action in range(prm["RL"]["dim_actions_1"]):
        all_vals_e_t_step_mean = np.zeros(prm["syst"]["N"])
        for step in range(prm["syst"]["N"]):
            all_vals_e_t_step = np.array(
                [[all_vals[e][method][repeat][home][step][action]
                  for repeat in list_repeat]
                 for home in range(prm["syst"]["n_homes"])]
            )
            if all(
                    [[all_vals[e][method][repeat][home][step][action] is None
                      for repeat in list_repeat]
                     for home in range(prm["syst"]["n_homes"])]
            ):
                all_vals_e_t_step = None

            all_vals_e_t_step_mean[step] = None \
                if all_vals_e_t_step is None \
                else np.nanmean(
                np.nanmean(all_vals_e_t_step)
            )

        ax.step(
            xs, all_vals_e_t_step_mean,
            where="post", label=method,
            color=prm["save"]["colourse"][method],
            lw=lw_mean, alpha=1, linestyle=linestyles[action]
        )
        means[e][method].append(all_vals_e_t_step)

    return ax


def _plot_all_agents_mean_res(
        entries, all_methods_to_plot, axs, all_T_air,
        prm, lw_mean, all_cum_rewards, labels,
        rows, columns, all_vals, list_repeat, linestyles, sum_agents=False
):
    means = {entry: {} for entry in ["T_air", "cum_rewards"] + entries}
    for method in all_methods_to_plot:
        if not sum_agents:
            axs[1, 1].step(range(prm['syst']['N']), np.mean(all_T_air[method], axis=0),
                           where="post", label=method,
                           color=prm["save"]["colourse"][method],
                           lw=lw_mean, alpha=1)
            means["T_air"][method] = np.mean(all_T_air[method], axis=0)
        if sum_agents:
            ax = axs[2]
        else:
            ax = axs[3, 0]
        ax.plot(
            [- 0.01] + list(range(prm['syst']['N'])),
            [0] + list(np.mean(all_cum_rewards[method], axis=0)),
            label=labels[method],
            color=prm["save"]["colourse"][method], lw=lw_mean, alpha=1
        )
        means["cum_rewards"][method] = np.mean(all_T_air[method], axis=0)
        for r, c, e in zip(rows, columns, entries):
            xs = list(range(prm['syst']['N']))
            if e == "store":
                xs = [-0.01] + xs
            if e == "action":
                axs[r, c] = _plot_last_epochs_actions(
                    list_repeat, means, e, method, prm, all_vals,
                    axs[r, c], xs, lw_mean, linestyles
                )
            else:
                n = len(all_vals[e][method][list_repeat[0]][0])
                all_vals_e_t_step_mean = np.zeros(n)
                for step in range(n):
                    all_vals_e_t_step = np.array(
                        [[all_vals[e][method][repeat][home][step]
                          for repeat in list_repeat]
                         for home in range(prm["syst"]["n_homes"])]
                    )
                    all_vals_e_t_step_mean[step] = np.mean(
                        np.mean(all_vals_e_t_step)
                    )
                ys = all_vals_e_t_step_mean * prm['syst']['n_homes'] \
                    if sum_agents else all_vals_e_t_step_mean
                ax = axs[r] if sum_agents else axs[r, c]
                ax.step(
                    xs, ys,
                    where="post", label=method,
                    color=prm["save"]["colourse"][method],
                    lw=lw_mean, alpha=1
                )
                means[e][method] = all_vals_e_t_step_mean

    return axs


def _plot_ev_loads_and_availability(
        axs, xs, loads_car, home, bands_car_availability, N, row=2, col=1, reduced_version=False
):
    ax = axs[row, col] if len(np.shape(axs)) > 1 else axs[row]

    for band in bands_car_availability:
        ax.axvspan(band[0], band[1], alpha=0.3, color="grey")
    if not reduced_version:
        ax.step(xs[0: N], loads_car[home][0: N], color="k", where="post")
        ax.set_ylabel("EV load [kWh]")
    grey_patch = matplotlib.patches.Patch(
        alpha=0.3, color="grey", label="EV on a trip")
    ax.legend(handles=[grey_patch], fancybox=True)

    return axs


def _plot_indoor_air_temp(
        axs, methods_to_plot, last,
        title_ylabel_dict, prm, home, row=0,
        col=0, alpha=1, display_labels=True, lw=2
):
    T_air_a = {}
    ax = axs[row, col]
    for method in methods_to_plot:
        T_air_a[method] = [
            last["T_air"][method][step][home]
            for step in range(len(last["T_air"][method]))
        ]
        label = method if display_labels else None
        ax.step(range(prm['syst']['N']), T_air_a[method], where="post", label=label,
                color=prm["save"]["colourse"][method], lw=lw, alpha=alpha)
    ax.set_ylabel(
        f"{title_ylabel_dict['T_air'][0]} {title_ylabel_dict['T_air'][1]}")
    ax.step(range(prm['syst']['N']), prm["heat"]["T_LB"][0][0: prm['syst']['N']], "--",
            where="post", color="k", label="T_LB")
    ax.step(range(prm['syst']['N']), prm["heat"]["T_UB"][0][0: prm['syst']['N']], "--",
            where="post", color="k", label="T_UB")

    return T_air_a


def _plot_cum_rewards(
        axs, last, methods_to_plot, labels, prm, row=0,
        col=0, alpha=1, lw=2, display_labels=True, sum_agents=False, reduced_version=False
):
    cumrewards = {}
    if sum_agents or reduced_version:
        ax = axs[row]
    else:
        ax = axs[row, col]

    for method in methods_to_plot:
        cumrewards[method] = [
            sum(last["reward"][method][0: i + 1])
            for i in range(len(last["reward"][method]))
        ]
        label = labels[method] if display_labels else None
        ax.plot([- 0.01] + list(range(prm['syst']['N'])), [0] + cumrewards[method], label=label,
                color=prm["save"]["colourse"][method], lw=lw, alpha=alpha)
    ax.legend(fancybox=True, loc="best", ncol=2)
    ax.set_ylabel("Cumulative rewards [£]")
    ax.set_xlabel("Time [h]")
    ax.set_ylim([min(cumrewards["baseline"]) * 1.3, 5])

    return cumrewards


def _plot_grid_price(
        title_ylabel_dict, N, axs=None, cintensity_kg=None,
        row=0, col=0, last=None, colours_non_methods=None,
        lw=None, display_legend=True, sum_agents=False, reduced_version=False):
    if sum_agents or reduced_version:
        ax = axs[row]
    else:
        ax = axs[row, col]
    ax.step(range(N), last["wholesale"]["baseline"],
            where="post", label="Wholesale",
            color=colours_non_methods[2], lw=lw)
    ax.step(range(N), last["grdC"]["baseline"], where="post",
            label="$C_g$", color=colours_non_methods[0], lw=lw)
    if display_legend:
        ax.set_ylabel("Grid price [£/kWh]")
        ax.legend(fancybox=True, loc="best")
    ax2 = ax.twinx()
    ax2.step(range(N), cintensity_kg, where="post",
             label=title_ylabel_dict["cintensity"][0],
             color=colours_non_methods[1], lw=lw)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    if display_legend:
        ax2.set_ylabel("Carbon intensity \n[kgCO$_2$/kWh]",
                       color=colours_non_methods[1])
    ylim = ax2.get_ylim()
    ax2.set_ylim([ylim[0], ylim[1] * 1.15])


def _plot_all_agents_all_repeats_res(
        list_repeat, all_methods_to_plot, title_ylabel_dict,
        axs, colours_non_methods, lw_indiv, labels,
        alpha_not_indiv, prm, lw_all, all_cum_rewards, all_T_air,
        rows, columns, entries, all_vals, sum_agents=False
):
    for repeat in list_repeat:
        last, cintensity_kg, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, prm["paths"]["folder_run"])
        _plot_grid_price(
            title_ylabel_dict, prm['syst']['N'], axs=axs, cintensity_kg=cintensity_kg,
            row=0, col=0, last=last,
            colours_non_methods=colours_non_methods, lw=lw_indiv,
            sum_agents=sum_agents
        )

        row = 2 if sum_agents else 3
        cum_rewards_repeat = _plot_cum_rewards(
            axs, last, methods_to_plot, labels, prm, row=row,
            col=0, alpha=alpha_not_indiv, lw=lw_all,
            display_labels=False, sum_agents=sum_agents
        )
        for method in all_methods_to_plot:
            all_cum_rewards[method].append(cum_rewards_repeat[method])
        if not sum_agents:
            for home in range(prm["syst"]["n_homes"]):
                T_air_a = _plot_indoor_air_temp(
                    axs, methods_to_plot, last, title_ylabel_dict,
                    prm, home, row=1, col=1, alpha=alpha_not_indiv,
                    display_labels=False, lw=lw_all)
                # returned is home dictionary per method of
                # 24 h profie for that last epoch
                for method in methods_to_plot:
                    all_T_air[method].append(T_air_a[method])
        for r, c, e in zip(rows, columns, entries):
            for home in range(prm["syst"]["n_homes"]):
                for method in methods_to_plot:
                    xs, ys = list(range(prm['syst']['N'])), last[e][method]
                    ys = [ys[step][home] for step in range(len(ys))]
                    if e == "store":
                        xs = [-0.01] + xs
                        ys = [prm["car"]["store0"][home]] + ys
                    if not sum_agents:
                        axs[r, c].step(xs, ys, where="post",
                                       color=prm["save"]["colourse"][method],
                                       lw=lw_all, alpha=alpha_not_indiv)
                    all_vals[e][method][repeat].append(ys)
                if not sum_agents:
                    axs[r, c].set_ylabel(
                        f"{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}")
                    if r == 2:
                        axs[r, c].set_xlabel("Time [h]")

    return axs, all_T_air, all_vals, all_cum_rewards


def _get_bands_car_availability(availabilities_car, home, N):
    bands_car_availability = []
    non_avail = [i for i in range(N) if availabilities_car[home][i] == 0]
    if len(non_avail) > 0:
        current_band = [non_avail[0]]
        if len(non_avail) > 1:
            for i in range(1, len(non_avail)):
                if non_avail[i] != non_avail[i - 1] + 1:
                    current_band.append(non_avail[i - 1] + 0.99)
                    bands_car_availability.append(current_band)
                    current_band = [non_avail[i]]
        current_band.append(non_avail[-1] + 0.999)
        bands_car_availability.append(current_band)

    return bands_car_availability


def _plot_all_agents_res(
        list_repeat, lw_all, prm, lw_all_list_repeat,
        all_methods_to_plot, title_ylabel_dict, colours_non_methods, labels,
        lw_indiv, alpha_not_indiv, lw_mean, linestyles, sum_agents=False
):
    # do one figure with all agents and repeats
    title_repeat = "all_repeats" if list_repeat is None \
        else f"repeat_{list_repeat}"
    lw_all = lw_all if list_repeat is None else lw_all_list_repeat
    list_repeat = range(prm["RL"]["n_repeats"]) \
        if list_repeat is None else list_repeat
    # Action variable
    # Heating E
    # Total consumption
    # Indoor temperature
    # Cumulative rewards
    # Battery level
    if not sum_agents:
        n_rows, n_cols = 4, 2
        rows = [1, 2, 0, 2]
        columns = [0, 0, 1, 1]
        entries = ["action", "totcons", "tot_E_heat", "store"]
    else:
        n_rows, n_cols = 3, 1
        rows = [1]
        columns = [0]
        entries = ["netp"]

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 13))
    all_cum_rewards = {method: [] for method in all_methods_to_plot}
    all_T_air = {method: [] for method in all_methods_to_plot}
    all_vals = initialise_dict(
        entries, second_level_entries=all_methods_to_plot
    )
    for e in entries:
        for method in all_methods_to_plot:
            all_vals[e][method] = {repeat: [] for repeat in range(prm["RL"]["n_repeats"])}

    axs, all_T_air, all_vals, all_cum_rewards = _plot_all_agents_all_repeats_res(
        list_repeat, all_methods_to_plot, title_ylabel_dict,
        axs, colours_non_methods, lw_indiv, labels,
        alpha_not_indiv, prm, lw_all, all_cum_rewards, all_T_air,
        rows, columns, entries, all_vals, sum_agents=sum_agents
    )
    axs = _plot_all_agents_mean_res(
        entries, all_methods_to_plot, axs, all_T_air,
        prm, lw_mean, all_cum_rewards, labels,
        rows, columns, all_vals, list_repeat, linestyles, sum_agents=sum_agents
    )

    fig.tight_layout()
    title = f"subplots example day all agents {title_repeat}"
    if sum_agents:
        title += " sum_agents"
    title_display = "subplots example day"
    subtitles = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for c in range(n_cols):
        for r in range(n_rows):
            ax = axs[r] if n_cols == 1 else axs[r, c]
            ax.set_title(subtitles[c + r * n_cols])
    formatting_figure(
        prm, fig=fig, title=title,
        legend=False,
        display_title=False,
        title_display=title_display
    )


def _get_repeat_data(repeat, all_methods_to_plot, folder_run):
    # last = last epoch of each repeat
    last = np.load(folder_run / "record" / f"last_repeat{repeat}.npy",
                   allow_pickle=True).item()
    cintensity_kg = [c * 1e3 for c in last["cintensity"]["baseline"]]
    methods_to_plot = [m for m in all_methods_to_plot if m in last["reward"]]

    return last, cintensity_kg, methods_to_plot


def _plot_indiv_agent_res_action(prm, ys, xs, lw_indiv, linestyles, method, ax):
    for action in range(prm["RL"]["dim_actions_1"]):
        ys_ = [0] + [
            ys[step][action]
            for step in range(prm["syst"]["N"])
        ]
        ax.step(xs, ys_, where="post",
                label=f"t_action{action}",
                color=prm["save"]["colourse"][method],
                lw=lw_indiv, linestyle=linestyles[action])

    return ax


def _plot_indiv_agent_res(
        prm, all_methods_to_plot, title_ylabel_dict,
        colours_non_methods, lw_indiv, labels, linestyles
):
    reduced_version = True
    # Grid price / intensity
    # Heating E
    # Action variable
    # Indoor temperature
    # Total consumption
    # EV load / availability
    # Cumulative rewards
    # Battery level

    for repeat in range(prm["RL"]["n_repeats"]):
        last, cintensity_kg, methods_to_plot = \
            _get_repeat_data(repeat, all_methods_to_plot, prm["paths"]["folder_run"])
        methods_to_plot = [
            method for method in ['baseline', 'opt', 'opt_d_d']
            if method in last['reward'].keys()
        ]

        # plot EV availability + EV cons on same plot
        loads_car, availabilities_car = [
            last["batch"][e] for e in ["loads_car", "avail_car"]
        ]

        if reduced_version:
            n_rows, n_cols = 3, 1
            figsize = (7, 13)
            row_method_legend = 1
        else:
            n_rows, n_cols = 4, 2
            figsize = (13, 13)
            row_method_legend = n_rows - 1

        for home in range(prm["syst"]["n_homes"]):
            xs = range(len(loads_car[home]))
            bands_car_availability = _get_bands_car_availability(
                availabilities_car, home, prm['syst']['N']
            )

            fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
            # carbon intensity, wholesale price and grid cost coefficient
            _plot_grid_price(
                title_ylabel_dict, prm['syst']['N'], axs=axs, cintensity_kg=cintensity_kg,
                row=0, col=0, last=last,
                colours_non_methods=colours_non_methods,
                lw=lw_indiv, reduced_version=reduced_version
            )
            if reduced_version:
                row, col = 1, 0
            else:
                row, col = 2, 1
            axs = _plot_ev_loads_and_availability(
                axs, xs, loads_car, home, bands_car_availability, prm['syst']['N'],
                row=row, col=col, reduced_version=reduced_version
            )
            row = 2 if reduced_version else 3
            _plot_cum_rewards(
                axs, last, methods_to_plot, labels, prm,
                row=row, col=0, lw=lw_indiv, reduced_version=reduced_version
            )
            # cum rewards
            if not reduced_version:
                # indoor air temp
                _plot_indoor_air_temp(
                    axs, methods_to_plot, last,
                    title_ylabel_dict, prm, home,
                    row=1, col=1, lw=lw_indiv
                )

            if reduced_version:
                # rows = [1, 2]
                # columns = [0, 0]
                # entries = ["totcons", "store"]
                rows = [1]
                columns = [0]
                entries = ["store"]
            else:
                rows = [1, 2, 0, 3]
                columns = [0, 0, 1, 1]
                entries = ["action", "totcons", "tot_E_heat", "store"]
            for r, c, e in zip(rows, columns, entries):
                ax = axs[r, c] if len(np.shape(axs)) > 1 else axs[r]
                for method in methods_to_plot:
                    xs = [-0.01] + list(range(prm['syst']['N']))
                    ys = last[e][method]
                    ys = [ys[step][home] for step in range(len(ys))]
                    if e == "action":
                        ax = _plot_indiv_agent_res_action(
                            prm, ys, xs, lw_indiv, linestyles, method, ax
                        )
                    else:
                        if e == "store" and method == "opt":
                            ys = ys + [prm["car"]["store0"][home]]
                        elif e == "store":
                            ys = [prm["car"]["store0"][home]] + ys
                        else:
                            ys = [0] + ys
                        # if reduced_version and method not in ['baseline', 'opt']:
                        #     colour = (192 / 255, 0, 0)
                        # else:
                        colour = prm["save"]["colourse"][method]
                        # if reduced_version and method not in ['baseline', 'opt']:
                        #     label = 'MARL policy'
                        # else:
                        label = method
                        ax.step(xs, ys, where="post", label=label,
                                color=colour,
                                lw=lw_indiv)

                ax.set_ylabel(
                    f"{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}")
                if r == n_rows - 1:
                    ax.set_xlabel("Time [h]")
                if r == row_method_legend:
                    if c == n_cols - 1:
                        ax.legend()

            fig.tight_layout()
            title = f"subplots example day repeat {repeat} home {home}"
            title_display = "subplots example day"
            if reduced_version:
                title += '_reduced_version'
            subtitles = ["a", "b", "c", "d", "e", "f", "g", "h"]
            for c in range(n_cols):
                for r in range(n_rows):
                    ax = axs[r, c] if len(np.shape(axs)) > 1 else axs[r]
                    ax.set_title(subtitles[c + r * n_cols])
            formatting_figure(
                prm, fig=fig, title=title,
                legend=False,
                display_title=False,
                title_display=title_display
            )


def plot_res(prm, indiv=True, list_repeat=None, sum_agents=False):
    # indiv = plot figure for one agent at a time
    # if false, do all the lines on the same plot in light
    # with one thick line average
    # Do big figure with subplots
    all_methods_to_plot = prm["RL"]["evaluation_methods"]

    title_ylabel_dict = {
        "T_air": ["Indoor air temperature", "[$^o$C]"],
        "T": ["Building temperature", "[$^o$C]"],
        "grdC": ["Grid cost coefficient", "[£/kWh]"],
        "cumulative rewards": ["Cumulative rewards", "[£]"],
        "reward": ["Rewards", "£"],
        "wholesale": ["Wholesale electricity price", "[£/kW]h"],
        "cintensity": ["Grid carbon intensity", "[kgCO$_2$/kWh]"],
        "tot_E_heat": ["Heating", "[kWh]"],
        "tot_cons_loads": ["Household consumption", "[kWh]"],
        "totcons": ["Home consumption", "[kWh]"],
        "ldfixed": ["Consumption of non-flexible household loads", "[kWh]"],
        "ldflex": ["Consumption of flexible household loads", "[kWh]"],
        "store": ["EV battery level", "[kWh]"],
        "store_outs": ["Discharge", "[kWh]"],
        "netp": ["Total household imports", "[kWh]"],
        "action": ["Action variable", r"$\psi$ [-]"]
    }
    linestyles = {
        0: "-",
        1: "--",
        2: ":"
    }

    lw_indiv = 2
    lw_all = 0.4
    lw_all_list_repeat = 1
    lw_mean = 2.5
    alpha_not_indiv = 0.15
    plt.rcParams["font.size"] = "16"
    colours_methods = [prm["save"]["colourse"][method] for method in all_methods_to_plot]
    colours_non_methods = [
        c for c in prm["save"]["colours"] if c not in colours_methods
    ]
    prm["save"]["colourse"]["opt_n_c"] = prm["save"]["colourse"]["opt_n_d"]
    labels = {}
    reward_labels = {
        "d": "M",
        "r": "T",
        "n": "N",
        "A": "A"
    }
    experience_labels = {"opt": "O", "env": "E"}
    for method in prm["RL"]["evaluation_methods"]:
        if method == "opt":
            labels[method] = "optimal"
        elif method == "baseline":
            labels[method] = method
        else:
            label = reward_labels[reward_type(method)]
            label += experience_labels[data_source(method)]
            labels[method] = label

    if indiv:  # do one figure per agent and per repeat
        _plot_indiv_agent_res(
            prm, all_methods_to_plot, title_ylabel_dict,
            colours_non_methods, lw_indiv, labels, linestyles
        )
    elif not indiv:
        _plot_all_agents_res(
            list_repeat, lw_all, prm, lw_all_list_repeat,
            all_methods_to_plot, title_ylabel_dict, colours_non_methods, labels,
            lw_indiv, alpha_not_indiv, lw_mean, linestyles, sum_agents=sum_agents
        )


def _plot_noisy_deterministic_inputs(prm, batch_entries, record, repeat):
    seeds = np.load(prm["paths"]["seeds_path"])
    heatavail = {}
    for e in batch_entries:
        fig, axs = plt.subplots(prm["syst"]["n_homes"], 1, squeeze=0)
        axs = axs.ravel()
        for home in range(prm["syst"]["n_homes"]):
            n = len(np.load(f"batch_{int(seeds[0] + 1)}_a0_lds.npy",
                            mmap_mode="c"))
            heatavail[home] = np.zeros((n,))
            for seed in record.seed[repeat]:
                str_seed = str(int(seed)) if seed > seeds[0] + 1 else \
                    f"deterministic_{record.ind_seed_deterministic[repeat]}"
                str_batch = f"batch_{str_seed}_a{home}_{e}.npy"
                if os.path.exists(str_batch):
                    batch_a_e = np.load(str_batch, mmap_mode="c")
                    if e == "avail_car":
                        heatavail[home] += batch_a_e
                    else:
                        axs[home].plot(
                            batch_a_e, alpha=1 / prm["RL"]["n_epochs"])
            if e == "avail_car":
                heatavail_plot = np.reshape(heatavail[home], (1, n))
                sns.heatmap(heatavail_plot, cmap="YlGn",
                            cbar=True, ax=axs[home])
        title = f"noisy repeat {repeat} {e}"
        title_and_save(title, fig, prm)
        fig.show()
        fig.savefig(e)


def plot_env_input(repeat, prm, record):
    if prm["RL"]["deterministic"] is None \
            or prm["RL"]["deterministic"] == 0 \
            or not prm["save"]["plotting_batch"]:
        return

    batch_entries = ["loads", "gen", "loads_car", "avail_car"]
    if prm["RL"]["deterministic"] == 2:
        # 0 is indeterministic, 1 is deterministic, 2 is deterministic noisy
        _plot_noisy_deterministic_inputs(
            prm, batch_entries, record, repeat
        )

    else:
        if prm["RL"]["deterministic"] == 1 \
                and os.path.exists(f"deterministic_parameters_repeat{repeat}.npy"):
            batch = np.load(f"deterministic_parameters_repeat{repeat}.npy",
                            allow_pickle=True)[-1]
        elif prm["RL"]["deterministic"] == 0 and "batch" in record.last[repeat]:
            # indeterministic, just plot the last epoch, evaluation step
            batch = record.last[repeat]["batch"]["eval"]
        n_homes_plot = max(prm["syst"]["n_homes"], 10)
        for e in batch_entries:
            fig, axs = plt.subplots(n_homes_plot, 1, squeeze=0)
            axs = axs.ravel()
            for home in range(n_homes_plot):
                axs[home].plot(batch[e][home])
                axs[home].set_title("{home}")
            title = f"deterministic repeat {repeat} {e}"
            title_and_save(title, fig, prm)
        else:
            print(f"no new deterministic batch for repeat = {repeat}")


def plot_imp_exp_violations(
        prm, all_methods_to_plot, folder_run):
    """ Plots grid [kWh] and import/export penalties for last day """
    plt.rcParams['font.size'] = '16'
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            print(method)
            fig, ax1 = plt.subplots(figsize=(8, 6))
            ax2 = ax1.twinx()
            netp = last['netp'][method]  # [step][a]
            netp0 = last['netp0'][method]  # [step][a]
            grid_flex = np.sum(netp, axis=1)
            grid_passive = np.sum(netp0, axis=1)
            break_down_rewards = last['break_down_rewards'][method]
            i_import_export_costs = prm['syst']['break_down_rewards_entries'].index(
                'import_export_costs'
            )
            import_export_costs = [
                break_down_rewards[step][i_import_export_costs]
                for step in range(prm['syst']['N'])
            ]
            ax1.plot(grid_flex, label='Flex Import/Export', color='coral')
            ax1.plot(grid_passive, label='Passive Import/Export', color='coral', linestyle='dashed')
            ax2.bar(
                range(prm['syst']['N']),
                import_export_costs,
                label='Penalty import export',
                color='olive'
            )
            ax1.axhline(y=prm['grd']['max_grid_import'], color='k', linestyle='dotted')
            ax1.axhline(y=-prm['grd']['max_grid_export'], color='k', linestyle='dotted')
            ax1.set_ylabel('Grid import/export [kWh]')
            ax2.set_ylabel('System penalty [£]')
            ax2.set_ylim([0, 1.1 * max(import_export_costs)])
            ax1.spines['right'].set_color('coral')
            ax1.spines['left'].set_color('coral')
            ax1.spines['right'].set_color('olive')
            ax1.spines['left'].set_color('olive')
            ax1.yaxis.label.set_color('coral')
            ax2.yaxis.label.set_color('olive')
            ax1.legend(loc='center', bbox_to_anchor=(0.3, 0.91))
            ax2.legend(loc='center', bbox_to_anchor=(0.3, 0.83))
            plt.tight_layout()
            title = f'Import and export and corresponding penalties, repeat{repeat}, {method}'
            title_and_save(title, fig, prm)


def plot_reactive_power(
        prm, all_methods_to_plot, folder_run):
    """ Plots total_reactive_power [kWh] and voltage penalties for last day """
    plt.rcParams['font.size'] = '16'
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            fig, ax1 = plt.subplots(figsize=(18, 12))
            ax2 = ax1.twinx()
            total_reactive_power = last['q_ext_grid'][method]
            q_house = np.sum(last['q_house'][method], axis=1)
            q_car = np.sum(last['q_car'][method], axis=1)
            break_down_rewards = last['break_down_rewards'][method]
            i_voltage_costs = prm['syst']['break_down_rewards_entries'].index('voltage_costs')
            voltage_costs = [
                break_down_rewards[step][i_voltage_costs]
                for step in range(prm['syst']['N'])
            ]
            ax1.plot(total_reactive_power, label='Reactive power', color='salmon')
            ax1.plot(q_house, label='Reactive power house')
            ax1.plot(q_car, label='Reactive power car')
            ax1.plot(q_car, label='Reactive power passive')
            ax2.bar(
                range(prm['syst']['N']),
                voltage_costs,
                label='Voltage costs',
                color='forestgreen'
            )
            ax1.set_ylabel('Sum reactive power [kWh]')
            ax2.set_ylabel('System voltage costs [£]')
            ax2.set_ylim([0, 1.1 * max(voltage_costs)])
            ax1.spines['right'].set_color('salmon')
            ax1.spines['left'].set_color('salmon')
            ax1.spines['right'].set_color('forestgreen')
            ax1.spines['left'].set_color('forestgreen')
            ax1.yaxis.label.set_color('coral')
            ax2.yaxis.label.set_color('forestgreen')
            ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.90))
            ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.77))
            plt.tight_layout()
            title = f'Reactive power and total costs repeat {repeat} {method}'
            title_and_save(title, fig, prm)


def plot_indiv_reactive_power(
        prm, all_methods_to_plot, folder_run):
    """ Plots flex_reactive_power [kWh] and import/export penalties for last day """
    plt.rcParams['font.size'] = '16'
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            fig, ax1 = plt.subplots(figsize=(18, 12))
            ax2 = ax1.twinx()
            q_car = last['q_car'][method]
            break_down_rewards = last['break_down_rewards'][method]
            i_voltage_costs = prm['syst']['break_down_rewards_entries'].index('voltage_costs')
            voltage_costs = [
                break_down_rewards[step][i_voltage_costs]
                for step in range(prm['syst']['N'])
            ]
            for home in range(prm["syst"]["n_homes"]):
                ax1.plot(np.array(q_car)[:, home], label=f'Reactive power car, home {home}')
            ax2.bar(
                range(prm['syst']['N']),
                voltage_costs,
                label='Voltage costs',
                color='forestgreen'
            )
            ax1.set_ylabel('Sum reactive power [kWh]')
            ax2.set_ylabel('System voltage costs [£]')
            ax2.set_ylim([0, 1.1 * max(voltage_costs)])
            ax1.spines['right'].set_color('salmon')
            ax1.spines['left'].set_color('salmon')
            ax1.spines['right'].set_color('forestgreen')
            ax1.spines['left'].set_color('forestgreen')
            ax1.yaxis.label.set_color('coral')
            ax2.yaxis.label.set_color('forestgreen')
            ax1.legend(loc='center', bbox_to_anchor=(0.5, 0.90))
            ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.77))
            plt.tight_layout()
            title = f'Indiv reactive power and total costs, repeat{repeat}, {method}'
            title_and_save(title, fig, prm)


def plot_voltage_violations(
        prm, all_methods_to_plot, folder_run):
    """ Plots grid [kWh] and voltage penalties for last day """
    plt.rcParams['font.size'] = '16'
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            fig, ax1 = plt.subplots(figsize=(8, 6))
            ax2 = ax1.twinx()
            netp = last['netp'][method]  # [step][a]
            netp0 = last['netp0'][method]  # [step][a]
            grid_flex = [sum(netp[step]) for step in range(prm['syst']['N'])]
            grid_passive = [sum(netp0[step]) for step in range(prm['syst']['N'])]
            break_down_rewards = last['break_down_rewards'][method]
            i_voltage_costs = prm['syst']['break_down_rewards_entries'].index('voltage_costs')
            voltage_costs = [
                break_down_rewards[step][i_voltage_costs]
                for step in range(prm['syst']['N'])
            ]
            ax1.plot(grid_flex, label='Flex Import/Export', color='coral')
            ax1.plot(grid_passive, label='Passive Import/Export', color='coral', linestyle='dashed')
            ax2.bar(
                range(prm['syst']['N']),
                voltage_costs,
                label='Penalty voltage',
                color='cadetblue'
            )
            ax1.set_ylabel('Grid import/export [kWh]')
            ax2.set_ylabel('System penalty [£]')
            ax2.set_ylim([0, 1.1 * max(voltage_costs)])
            ax1.spines['right'].set_color('coral')
            ax1.spines['left'].set_color('coral')
            ax1.spines['right'].set_color('cadetblue')
            ax1.spines['left'].set_color('cadetblue')
            ax1.yaxis.label.set_color('coral')
            ax2.yaxis.label.set_color('cadetblue')
            ax1.legend(loc='center', bbox_to_anchor=(0.3, 0.91))
            ax2.legend(loc='center', bbox_to_anchor=(0.3, 0.83))
            plt.tight_layout()
            title = f'Import and export and voltage penalties, repeat{repeat}, {method}'
            title_and_save(title, fig, prm)


def plot_imp_exp_check(
        prm, all_methods_to_plot, folder_run):
    """ Plots grid [kWh], grid_import and grid_export for last day """
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            fig = plt.figure()
            netp = last['netp'][method]  # [step][a]
            grid = [sum(netp[step]) for step in range(prm['syst']['N'])]
            grid_in = np.where(np.array(grid) >= 0, grid, 0)
            grid_out = np.where(np.array(grid) < 0, grid, 0)
            plt.plot(grid_in, label='grid imp', color='b')
            plt.plot(grid_out, label='grid exp', color='g')
            plt.plot(grid, label='grid tot', color='r', linestyle='--')
            plt.legend()
            plt.tight_layout()
            title = f'grid_imports_exports_check_repeat{repeat}_{method}'
            title_and_save(title, fig, prm)


def voltage_penalty_per_bus(prm, all_methods_to_plot, folder_run):
    """ Plots voltages of first 150 buses that exceed the limits """
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in [method for method in methods_to_plot if method not in ['opt']]:
            overvoltage_bus_index, undervoltage_bus_index = \
                get_index_over_under_voltage_last_time_step(last, method, prm)
            overvoltage_value = \
                last['voltage_squared'][method][prm['syst']['N'] - 1][overvoltage_bus_index]
            undervoltage_value = \
                last['voltage_squared'][method][prm['syst']['N'] - 1][undervoltage_bus_index]
            n_voltage_violations = len(overvoltage_bus_index) + len(undervoltage_bus_index)
            if n_voltage_violations > 150:
                fig_length = 22
            else:
                fig_length = 10

            fig, ax1 = plt.subplots(figsize=(fig_length, 6))
            # ax2 = ax1.twinx()

            if len(undervoltage_value) > 0:
                ax1.bar(undervoltage_bus_index - 0.2, undervoltage_value, width=0.4,
                        label='undervoltage', color='navy')
                # ax2.bar(under_index+0.2, prm['grd']['penalty_undervoltage'] \
                #    * (prm['grd']['min_voltage']**2 - np.square(under_value)),
                #    width=0.4, color ='dodgerblue', label = 'under penalty')
                first_bus_under = undervoltage_bus_index[0]
                lower_lim = min(undervoltage_value) - 0.002
            else:
                first_bus_under = 1000
                lower_lim = 0.998
            if len(overvoltage_bus_index) > 0:
                ax1.bar(overvoltage_bus_index - 0.2, overvoltage_value, width=0.4,
                        label='overvoltage', color='maroon')
                # ax2.bar(over_index + 0.2, np.array(prm['grd']['penalty_overvoltage'] \
                #    * (np.square(over_value) - prm['grd']['max_voltage']**2)),
                #    width=0.4, label = 'over penalty', color ='coral')
                first_bus_over = overvoltage_bus_index[0]
                upper_lim = max(overvoltage_value) + 0.002
            else:
                first_bus_over = 1000
                upper_lim = 1.002

            if n_voltage_violations > 150:
                ax1.set_xlim(min(first_bus_under, first_bus_over),
                             150 + min(first_bus_under, first_bus_over))
                title = f'Over-, undervoltage and corresponding penalty for last time step,' \
                    f'first 150 buses, repeat{repeat}_{method}'
            else:
                title = f'Over-, undervoltage and corresponding penalty for last time step,' \
                    f'repeat{repeat}_{method}'
            ax1.axhline(y=prm['grd']['max_voltage'], color='k')
            ax1.axhline(y=prm['grd']['min_voltage'], color='k')
            ax1.set_ylabel('Voltage magnitude [p.u.]')
            ax1.set_xlabel('Bus number')
            # ax2.set_ylabel('penalty')
            ax1.set_ylim(lower_lim, upper_lim)
            ax1.legend(loc='center left', bbox_to_anchor=(0.6, 0.91))
            # ax2.legend(loc='center left', bbox_to_anchor=(0.6, 0.83))
            plt.tight_layout()
            title_and_save(title, fig, prm)


def get_index_over_under_voltage_last_time_step(last, method, prm):
    overvoltage_bus_index = np.where(
        last['voltage_squared'][method][prm["syst"]["N"] - 1] > prm['grd']['max_voltage'] ** 2
    )[0]
    undervoltage_bus_index = np.where(
        last['voltage_squared'][method][prm["syst"]["N"] - 1] < prm['grd']['min_voltage'] ** 2
    )[0]

    return overvoltage_bus_index, undervoltage_bus_index


def map_over_undervoltage(
        prm, all_methods_to_plot, folder_run, net):
    """ Map of the network with over- and undervoltages marked """
    for repeat in range(prm['RL']['n_repeats']):
        last, _, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, folder_run)
        for method in methods_to_plot:
            if method != 'opt':
                # Plot all the buses
                bc = plot.create_bus_collection(net, net.bus.index, size=.2,
                                                color="black", zorder=10)

                # Plot Transformers
                tlc, tpc = plot.create_trafo_collection(net, net.trafo.index,
                                                        color="dimgrey", size=1.5)

                # Plot all the lines
                lcd = plot.create_line_collection(net, net.line.index, color="grey",
                                                  linewidths=0.5, use_bus_geodata=True)

                # Plot the external grid
                sc = plot.create_bus_collection(net, net.ext_grid.bus.values, patch_type="poly3",
                                                size=.7, color="grey", zorder=11)

                # Plot all the loads and generations
                ldA = plot.create_bus_collection(
                    net, last['loaded_buses'][method][prm["syst"]["N"] - 1],
                    patch_type="poly3", size=1.4, color="r", zorder=11
                )
                ldB = plot.create_bus_collection(
                    net, last['sgen_buses'][method][prm["syst"]["N"] - 1],
                    patch_type="poly3", size=1.4, color="g", zorder=11
                )

                # Plot over and under voltages
                overvoltage_bus_index, undervoltage_bus_index = \
                    get_index_over_under_voltage_last_time_step(last, method, prm)

                over = plot.create_bus_collection(
                    net,
                    overvoltage_bus_index,
                    size=0.6, color="coral", zorder=10
                )
                under = plot.create_bus_collection(
                    net,
                    undervoltage_bus_index,
                    size=0.6, color="dodgerblue", zorder=10
                )
                # Draw all the collected plots
                ax = plot.draw_collections([lcd, bc, tlc, tpc, sc, ldA, ldB, over, under],
                                           figsize=(20, 20))
                # Add legend to homes
                for bus, i in zip(
                    last['sgen_buses'][method][prm["syst"]["N"] - 1], range(prm["syst"]["n_homes"])
                ):
                    q_house = last['q_house'][method][prm["syst"]["N"] - 1][i]
                    q_car = last['q_car'][method][prm["syst"]["N"] - 1][i]
                    voltage = np.sqrt(last['voltage_squared'][method][prm["syst"]["N"] - 1][bus])
                    x, y = net.bus_geodata.loc[bus, ["x", "y"]]
                    plot.plt.annotate(
                        f"Generation bus, \n Voltage is {round(voltage, 3)} p.u."
                        f"\n Q_house is {round(q_house, 3)} kVAR"
                        f"\n Q_car is {round(q_car, 3)} kVAR",
                        xy=(x, y), xytext=(x + 5, y + 5), fontsize=10)
                for bus, i in zip(
                    last['loaded_buses'][method][prm["syst"]["N"] - 1],
                    range(prm["syst"]["n_homes"])
                ):
                    q_house = last['q_house'][method][prm["syst"]["N"] - 1][i]
                    q_car = last['q_car'][method][prm["syst"]["N"] - 1][i]
                    voltage = np.sqrt(last['voltage_squared'][method][prm["syst"]["N"] - 1][bus])
                    x, y = net.bus_geodata.loc[bus, ["x", "y"]]
                    plot.plt.annotate(
                        f"Load bus, \n Voltage is {round(voltage, 3)} p.u."
                        f"\n Q_house is {round(q_house, 3)} kVAR"
                        f"\n Q_car is {round(q_car, 3)} kVAR",
                        xy=(x, y), xytext=(x + 5, y + 5), fontsize=10)

                # Save
                title = f'map_over_under_voltage{repeat}_{method}'
                title_and_save(title, ax.figure, prm)
