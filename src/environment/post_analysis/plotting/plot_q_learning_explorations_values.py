import os

import matplotlib.pyplot as plt
import numpy as np

from src.environment.post_analysis.plotting.plotting_utils import (
    formatting_ticks, title_and_save)
from src.environment.utilities.env_spaces import granularity_to_multipliers
from src.environment.utilities.userdeftools import (distr_learning,
                                                    initialise_dict,
                                                    reward_type)


def _plot_1d_state_space_best_psi(
        prm, best_theta, repeat
):
    rl = prm["RL"]
    for home in range(prm['syst']['n_homes']):
        fig, ax = plt.subplots()
        theta_M = []
        for method in prm["save"]["eval_entries_plot_indiv"]:
            index_a = home if distr_learning(method) in ['d', 'Cd'] else 0
            theta_M.append(
                [best_theta[method][index_a][int(s)] for s in range(rl["possible_states"])]
            )
        im = ax.imshow(theta_M, vmin=0, vmax=1)
        title = f"best theta per method and per state state space " \
                f"{rl['statecomb_str']} repeat {repeat} home {home}"
        formatting_ticks(
            ax, fig, prm,
            xs=prm["save"]["eval_entries_plot_indiv"],
            ys=range(rl["possible_states"]),
            title=title, im=im,
            grid=False,
            display_title=False
        )


def _plot_2d_state_space_best_psi(
    prm, record, spaces, best_theta, repeat
):
    rl = prm["RL"]
    for home in range(prm['syst']['n_homes']):
        for i_t in range(len(prm["save"]["eval_entries_plot_indiv"])):
            method = prm["save"]["eval_entries_plot_indiv"][i_t]
            M = np.zeros((record.granularity_state0,
                          record.granularity_state1))
            index_a = home if distr_learning(method) in ['d', 'Cd'] else 0
            for s in range(rl["possible_states"]):
                s1, s2 = spaces.global_to_indiv_index(
                    'state', s, multipliers=granularity_to_multipliers(
                        [record.granularity_state0,
                         record.granularity_state1]))
                M[s1, s2] = best_theta[method][index_a][s]

            fig, ax = plt.subplots()
            plt.xlabel(rl['state_space'][1])
            plt.ylabel(rl['state_space'][0])
            im = ax.imshow(M, vmin=0, vmax=1)
            title = f"best theta per state combination state space " \
                    f"{rl['state_space']} repeat {repeat} home {home}"
            formatting_ticks(
                ax, fig, prm,
                ys=record.granularity_state1,
                xs=record.granularity_state0,
                title=title, im=im, grid=False,
                display_title=False
            )


def _plot_unique_state_best_psi(
        prm, best_theta, index_to_val, q, repeat
):
    rl = prm["RL"]
    possible_states = rl["possible_states"]
    eval_entries_plot_indiv = prm["save"]["eval_entries_plot_indiv"]

    for method in eval_entries_plot_indiv:
        n_homes = prm['syst']['n_homes'] if distr_learning(method) in ['d', 'Cd'] else 1
        best_theta[method] = initialise_dict(range(n_homes), type_obj='empty_dict')
        for home in range(n_homes):
            best_theta[method][home] = np.zeros((possible_states,))
            for s in range(possible_states):
                indmax = np.argmax(q[method][home][s])
                best_theta[method][home][s] = \
                    index_to_val([indmax], typev='action')[0]

            # plot historgram of best theta per method
            fig, ax = plt.subplots()
            y_pos = np.arange(len(eval_entries_plot_indiv))
            i_tables = [
                home if distr_learning(method) == 'd' else 0
                for method in eval_entries_plot_indiv
            ]
            best_thetas = \
                [best_theta[eval_entries_plot_indiv[it]][i_tables[it]][0]
                 for it in range(len(eval_entries_plot_indiv))]
            colours_bars = [
                prm['save']['colourse'][method]
                for method in eval_entries_plot_indiv
                if method[-1] != '0'
            ]
            ax.bar(y_pos, best_thetas, align='center',
                   alpha=0.5, color=colours_bars)
            plt.ylim([0, 1])
            title = f'best theta per method state None ' \
                    f'repeat {repeat} home {home}'

            ax.set_ylabel(r'best $\theta$ [-]')
            formatting_ticks(
                ax, fig, ys=eval_entries_plot_indiv,
                title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'],
                display_title=False
            )


def plot_best_actions(
    repeat, prm, record, spaces
):
    rl = prm["RL"]
    if not (
            rl['type_learning'] == 'q_learning'
            and prm['syst']['n_homes'] < 4
    ):
        return

    best_theta = initialise_dict(prm["save"]["eval_entries_plot_indiv"])
    q = rl["q_tables"][repeat][rl['n_epochs'] - 1] \
        if record.save_qtables else rl["q_tables"][repeat]

    for method in prm["save"]["eval_entries_plot_indiv"]:
        n_homes = prm['syst']['n_homes'] if distr_learning(method) in ['d', 'Cd'] else 1
        best_theta[method] = initialise_dict(range(n_homes), type_obj='empty_dict')
        for home in range(n_homes):
            best_theta[method][home] = np.zeros((rl['possible_states'],))
            for s in range(rl['possible_states']):
                indmax = np.argmax(q[method][home][s])
                action_indexes = spaces.global_to_indiv_index("action", indmax)
                best_theta[method][home][s] = spaces.index_to_val(action_indexes, typev='action')[0]
    if rl['state_space'] == [None]:
        _plot_unique_state_best_psi(
            prm, best_theta, spaces.index_to_val, q, repeat
        )

    # if one dimensional state space - best theta[method, state]
    elif rl['dim_states'] == 1:
        _plot_1d_state_space_best_psi(prm, best_theta, repeat)

    # if two dimensional state space heatmap for each method
    # besttheta[state1, state2]
    elif rl['dim_states'] == 2:
        _plot_2d_state_space_best_psi(
            prm, record, spaces, best_theta, repeat
        )


def plot_q_values(repeat, index_to_val, prm):
    rl = prm["RL"]
    eval_entries_plot_indiv = prm["save"]["eval_entries_plot_indiv"]
    if rl['type_learning'] != 'q_learning':
        return
    # plot all values in one figure if there is only one state
    if rl['state_space'] == [None] and prm['syst']['n_homes'] < 4:
        for home in range(prm['syst']['n_homes']):
            # plot heat map of value of each action for different methods
            # 2D array of best theta values
            M = np.zeros((len(eval_entries_plot_indiv), rl['n_action']))

            for i_t in range(len(eval_entries_plot_indiv)):
                method = eval_entries_plot_indiv[i_t]
                i_table = home if distr_learning(method) == 'd' else 0
                qvals = rl["q_tables"][repeat][rl['n_epochs'] - 1][method][i_table][
                    0]  # in final epoch, in 0-th (unique) state
                M[i_t, :] = qvals

            xs = rl["q_entries"]
            b = index_to_val([0], typev='action')[0]
            m = index_to_val([1], typev='action')[0] - b
            ys = [m * i + b for i in range(rl['n_action'])]
            fig, ax = plt.subplots()
            im = ax.imshow(M)
            title = f"qval per action per state None repeat {repeat} home {home}"
            formatting_ticks(
                ax, fig, xs=xs, ys=ys, title=title, im=im, grid=False,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'], display_title=False)


def video_visit_states(
        repeat, record, spaces, prm
):
    rl = prm["RL"]
    if rl['type_learning'] != 'q_learning' \
            or not prm['save']['make_video'] \
            or len(rl["counters"][0]) == 0 \
            or rl['server'] \
            or rl['state_space'] == [None]:
        return

    import cv2
    counters = rl["counters"][repeat]
    for method in rl["q_entries"]:
        maxval = np.max(
            [np.sum(counters[rl['n_epochs'] - 1][method][s])
             for s in range(rl["possible_states"])])

        for epoch in range(rl['n_epochs']):
            fig = plt.figure()
            if rl['dim_states'] == 2:
                counters_per_state = \
                    np.zeros((record.granularity_state0,
                              record.granularity_state1))
                plt.ylabel(rl['state_space'][0])
                plt.xlabel(rl['state_space'][1])
                for s in range(rl["possible_states"]):
                    s1, s2 = spaces.global_to_indiv_index(
                        'state', s, multipliers=granularity_to_multipliers(
                            [record.granularity_state0,
                             record.granularity_state1]))
                    counters_per_state[s1, s2] = sum(counters[epoch][method][s])
                plt.ylabel(rl['state_space'][1])

            elif rl['dim_states'] == 1:
                counters_per_state = \
                    np.zeros((record.granularity_state0, rl['n_action']))
                plt.ylabel(rl['state_space'][0])
                plt.xlabel(r'$\theta$ [-]')
                for s in range(rl["possible_states"]):
                    for home in range(rl['n_action']):
                        counters_per_state[s, home] = counters[epoch][method][s][home]
            else:
                counters_per_state = [
                    sum(counters[epoch][method][s]) for s in range(rl["possible_states"])
                ]
                plt.xlabel(rl['state_space'][0])
            plt.imshow(counters_per_state, vmin=0, vmax=maxval)

            plt.colorbar()
            plt.close('all')
            if epoch == 0:  # initialise video
                fig.savefig('fig')
                img0 = cv2.imread('fig.png')
                height, width, layers = img0.shape
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                title = \
                    prm['paths']['fig_folder'] / f'qlearn_repeat{repeat}_{method}.avi' \
                    if prm['paths']['fig_folder'] is not None \
                    else f'qlearn_repeat{repeat}_{method}.avi'
                out = cv2.VideoWriter(title, fourcc, 40.0, (width, height))
                os.remove('fig.png')
            else:
                figname = f'fig{epoch}'
                fig.savefig(figname)
                plt.close()
                frame = cv2.imread(figname + '.png', 1)
                os.remove(figname + '.png')
                out.write(frame)
        out.release()


def plot_final_explorations(repeat, record, prm):
    rl = prm["RL"]
    if rl['type_learning'] != 'q_learning' \
            or len(rl["counters"][0]) == 0:
        return

    for method in rl["q_entries"]:
        n_homes = prm['syst']['n_homes'] if reward_type(method) == '1' else 1
        for home in range(n_homes):
            rl["action_state_space_0"][repeat][home], rl["state_space_0"][repeat][home] =\
                [initialise_dict(rl["q_entries"]) for _ in range(2)]
            fig, ax = plt.subplots()
            counters_plot = rl["counters"][repeat][rl['n_epochs'] - 1][method][home] \
                if record.save_qtables else rl["counters"][repeat][method][home]
            im = ax.imshow(counters_plot, aspect='auto')
            fig.colorbar(im, ax=ax)
            title = f"Explorations per state action pair state space " \
                    f"{rl['statecomb_str']} repeat {repeat} " \
                    f"method {method} home {home}"
            title_and_save(title, fig, prm)

            sum_action_0, sum_state_0 = 0, 0
            for s in range(rl['n_total_discrete_states']):
                if sum(counters_plot[s]) == 0:
                    sum_state_0 += 1
                for ac in range(rl['n_discrete_actions']):
                    if counters_plot[s][ac] == 0:
                        sum_action_0 += 1
            rl["state_space_0"][repeat][home][method] = \
                sum_state_0 / rl['n_total_discrete_states']
            rl["action_state_space_0"][repeat][home][method] = \
                sum_action_0 / (rl['n_total_discrete_states'] * rl['n_discrete_actions'])
