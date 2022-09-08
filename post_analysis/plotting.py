#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:03:35 2020.

@author: floracharbonnier.

Plotting results RL
"""

import os

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.userdeftools import (get_moving_average, granularity_to_multipliers,
                                initialise_dict)


def _formatting_figure(
        fig=None, title=None, pos_leg='right', anchor_pos=None,
        ncol_leg=None, fig_folder=None, save_run=False,
        high_res=False, legend=True, display_title=True,
        title_display=None
):

    if anchor_pos is None:
        if pos_leg == 'right':
            anchor_pos = (1.25, 0.5)
        elif pos_leg == 'upper center':
            anchor_pos = (0.5, 1.2)

    if ncol_leg is None:
        if pos_leg == 'right':
            ncol_leg = 1
        elif pos_leg == 'upper center':
            ncol_leg = 3
    if legend:
        fig.legend(loc=pos_leg, bbox_to_anchor=anchor_pos,
                   ncol=ncol_leg, fancybox=True)
    fig.tight_layout()
    _title_and_save(
        title, fig, fig_folder, save_run, high_res=high_res,
        display_title=display_title, title_display=title_display
    )


def _formatting_ticks(
        ax, fig, xs=None, ys=None, title=None, im=None,
        grid=True, fig_folder=None, save_run=False,
        high_res=False, display_title=True
):
    if ys is not None:
        ax.set_xticks(np.arange(len(ys)))
        ax.set_xticklabels(ys)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    if xs is not None:
        ax.set_yticks(np.arange(len(xs)))
        ax.set_yticklabels(xs)
    if im is not None:
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if grid is True:
        plt.grid()
    _title_and_save(
        title, fig, fig_folder, save_run, ax=ax,
        high_res=high_res, display_title=display_title
    )


def _title_and_save(
        title, fig, fig_folder, save_run, ax=None,
        high_res=False, display_title=True, title_display=None
):
    if title_display is None:
        if 'theta' in title:
            ind = title.index('theta')
            title_display = title[0:ind] + r'$\theta$' + title[ind + 5:]
        elif '_air' in title:
            ind = title.index('_air')
            title_display = title[0:ind] + r'$_air$' + title[ind + 4:]
        else:
            title_display = title
    if display_title:
        if ax is not None:
            ax.set_title(title_display)
        else:
            plt.title(title_display)
    if save_run:
        if high_res:
            fig.savefig(fig_folder / f"{title.replace(' ', '_')}.pdf",
                        bbox_inches='tight', format='pdf', dpi=1200)
        else:
            fig.savefig(fig_folder / title.replace(' ', '_'),
                        bbox_inches='tight')
    plt.close('all')

    return fig


def _barplot(
        bars, eval_types, prm, baseline=None, opt=None, text_labels=True,
        colors=None, error=None, title=None, display_title=True,
        lower_bound=None, upper_bound=None, display_ticks=True,
        ax0=None, display_legend=False
):

    n_evaltype = len(eval_types)
    barWidth = 1 / (n_evaltype + 1)
    rs = [x * 1 / len(bars) + (1 / len(bars) / 2) for x in range(len(bars))]

    if ax0 is None:
        fig, ax = plt.subplots(figsize=(3.25, 7 * 0.75))
    else:
        ax = ax0
    for ind_e in range(n_evaltype):
        rsir = rs[ind_e]
        err = None if error is None else error[ind_e]
        barplot = bars[ind_e]
        barplot = barplot - baseline if title[0:7] == 'Average' else barplot
        ax.bar(rsir, barplot, yerr=err, capsize=10, width=barWidth,
               edgecolor='white', label=eval_types[ind_e], color=colors[ind_e])
    if baseline is not None:
        drawbsl = 0 if title[0:7] == 'Average' else baseline
        ax.axhline(drawbsl, linestyle='--', color="k")
    if title[0:7] == 'Average' \
            and lower_bound is not None \
            and upper_bound is not None:
        plt.ylim([lower_bound, upper_bound])
    if opt is not None:
        drawopt = opt - baseline if title[0:7] == 'Average' else opt
        ax.axhline(drawopt, linestyle='--',
                   color=prm['save']['colorse']['opt'])

    if text_labels:
        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            label_height = height
            ax.text(rect.get_x() + rect.get_width() / 2, height,
                    '%.3f' % float(label_height),
                    ha='center', va='bottom', fontsize=9)
    if not display_ticks:
        plt.gca().set_yticks([])
        plt.gca().set_xticks([])

    if display_legend:
        plt.legend(
            loc='center right', bbox_to_anchor=(1.2, 0.5), fancybox=True
        )

    if title is not None and display_title:
        ax.set_title(title)

    if ax0 is None:
        return fig
    else:
        return ax


def _eval_entries_plot_colors(prm):
    eval_entries = prm['RL']['type_eval']
    if prm['ntw']['n'] == 1:
        eval_entries = [e for e in eval_entries
                        if e.split('_')[-1] == 'd'
                        or e in ['baseline', 'opt']]
        if len([e for e in eval_entries if len(e.split('_')) > 1]) == 0:
            eval_entries = [e for e in prm['RL']['type_eval']
                            if e.split('_')[-1] == 'c'
                            or e in ['baseline', 'opt']]

    eval_entries_plot = \
        [e for e in eval_entries if e != 'baseline'] \
        + ['baseline'] if 'baseline' in eval_entries else eval_entries
    type_plot = {}
    for e in eval_entries:
        type_plot[e] = '-'
    eval_entries_bars = [e for e in eval_entries_plot
                         if e not in ['baseline', 'opt']]
    eval_entries_distr = [e for e in eval_entries if len(e.split('_')) == 1 or e.split('_')[-1] == 'd']
    eval_entries_centr = [e for e in eval_entries if len(e.split('_')) == 1 or e.split('_')[-1] == 'c']
    eval_entries_notCd = [e for e in eval_entries_plot
                          if not(len(e.split('_')) > 1
                          and e.split('_')[-1] == 'Cd')]
    eval_entries_plot_indiv = [e for e in eval_entries_plot
                               if len(e.split('_')) > 1
                               and e.split('_')[2] not in ['Cc0', 'Cd0']]
    colors_barplot = [prm['save']['colorse'][e] for e in eval_entries_bars]
    base_entries = eval_entries_distr if len(eval_entries_distr) > 0 \
        else eval_entries_centr
    colors_barplot_baseentries = [prm['save']['colorse'][e]
                                  for e in base_entries]

    return [eval_entries_plot, eval_entries_bars, eval_entries,
            eval_entries_notCd, eval_entries_plot_indiv,
            colors_barplot_baseentries, colors_barplot, base_entries]


def _plot_cum_rewards(
        axs, last, methods_to_plot, labels, prm, row=0,
        col=0, alpha=1, lw=2, display_labels=True
):
    cumrewards = {}
    ax = axs[row, col]

    for t in methods_to_plot:
        cumrewards[t] = [sum(last['reward'][t][0: i + 1])
                         for i in range(len(last['reward'][t]))]
        label = labels[t] if display_labels else None
        ax.plot([- 0.01] + list(range(24)), [0] + cumrewards[t], label=label,
                color=prm['save']['colorse'][t], lw=lw, alpha=alpha)
    ax.legend(fancybox=True, loc='best', ncol=2)
    ax.set_ylabel('Cumulative rewards [£]')
    ax.set_xlabel('Time [h]')
    ax.set_ylim([min(cumrewards['baseline']) * 1.3, 5])

    return cumrewards


def _plot_indoor_air_temp(
        axs, methods_to_plot, last,
        title_ylabel_dict, prm, a, row=0,
        col=0, alpha=1, display_labels=True, lw=2
):
    T_air_a = {}
    ax = axs[row, col]
    for t in methods_to_plot:
        T_air_a[t] = [last['T_air'][t][step][a]
                      for step in range(len(last['T_air'][t]))]
        label = t if display_labels else None
        ax.step(range(24), T_air_a[t], where='post', label=label,
                color=prm['save']['colorse'][t], lw=lw, alpha=alpha)
    ax.set_ylabel(
        f"{title_ylabel_dict['T_air'][0]} {title_ylabel_dict['T_air'][1]}")
    ax.step(range(24), prm['heat']['T_LB'][0][0:24], '--',
            where='post', color='k', label='T_LB')
    ax.step(range(24), prm['heat']['T_UB'][0][0:24], '--',
            where='post', color='k', label='T_UB')

    return T_air_a


def _get_ridx_data(ridx, all_methods_to_plot, root):
    last = np.load(root / 'record' / f'last_ridx{ridx}.npy',
                   allow_pickle=True).item()
    cintensity_kg = [c * 1e3 for c in last['cintensity']['baseline']]
    methods_to_plot = [m for m in all_methods_to_plot if m in last['reward']]

    return last, cintensity_kg, methods_to_plot


def _plot_grid_price(
        title_ylabel_dict, axs=None, cintensity_kg=None,
        row=0, col=0, last=None, colors_non_methods=None,
        lw=None, display_legend=True):
    ax = axs[row, col]
    ax.step(range(24), last['wholesale']['baseline'],
            where='post', label='Wholesale',
            color=colors_non_methods[2], lw=lw)
    ax.step(range(24), last['grdC']['baseline'], where='post',
            label='$C_g$', color=colors_non_methods[0], lw=lw)
    if display_legend:
        ax.set_ylabel("Grid price [£/kWh]")
        ax.legend(fancybox=True, loc='best')
    ax2 = axs[row, col].twinx()
    ax2.step(range(24), cintensity_kg, where='post',
             label=title_ylabel_dict['cintensity'][0],
             color=colors_non_methods[1], lw=lw)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    if display_legend:
        ax2.set_ylabel('Carbon intensity [kgCO$_2$/kWh]',
                       color=colors_non_methods[1])
    ylim = ax2.get_ylim()
    ax2.set_ylim([ylim[0], ylim[1] * 1.15])


def _plotting_res(root, prm, indiv=True, list_ridx=None):
    # indiv = plot figure for one agent at a time
    # if false, do all the lines on the same plot in light
    # with one thick line average
    # Do big figure with subplots
    all_methods_to_plot = prm['RL']['type_eval']

    title_ylabel_dict = {
        'T_air': ['Indoor air temperature', '[$^o$C]'],
        'T': ['Building temperature', '[$^o$C]'],
        'grdC': ['Grid cost coefficient', '[£/kWh]'],
        'cumulative rewards': ['Cumulative rewards', '[£]'],
        'reward': ['Rewards', '£'],
        'wholesale': ['Wholesale electricity price', '[£/kW]h'],
        'cintensity': ['Grid carbon intensity', '[kgCO$_2$/kWh]'],
        'tot_E_heat': ['Heating', '[kWh]'],
        'tot_cons_loads': ['Household consumption', '[kWh]'],
        'totcons': ['Total consumption', '[kWh]'],
        'ldfixed': ['Consumption of non-flexible household loads', '[kWh]'],
        'ldflex': ['Consumption of flexible household loads', '[kWh]'],
        'store': ['Battery level', '[kWh]'],
        'store_outs': ['Discharge', '[kWh]'],
        'netp': ['Total household imports', '[kWh]'],
        'action': ['Action variable', '$\psi$ [-]']
    }

    lw_indiv = 2
    lw_all = 0.4
    lw_all_list_ridx = 1
    lw_mean = 2.5
    alpha_not_indiv = 0.15
    plt.rcParams['font.size'] = '16'
    colors_methods = [prm['save']['colorse'][t] for t in all_methods_to_plot]
    colors_non_methods = [c for c in prm['save']['colors']
                          if c not in colors_methods]
    prm['save']['colorse']['opt_n_c'] = prm['save']['colorse']['opt_n_d']
    labels = {}
    reward_labels = {'c': 'C', 'd': 'M', 'r': 'T'}
    experience_labels = {'opt': 'O', 'env': 'E'}
    for t in prm['RL']['type_eval']:
        if t == 'opt':
            labels[t] = 'optimal'
        elif t == 'baseline':
            labels[t] = t
        else:
            label = reward_labels[t.split('_')[1]]
            label += experience_labels[t.split('_')[0]]
            labels[t] = label

    if indiv:  # do one figure per agent and per repeat
        # Grid price / intensity
        # Heating E
        # Action variable
        # Indoor temperature
        # Total consumption
        # EV load / availability
        # Cumulative rewards
        # Battery level
        for ridx in range(prm['RL']['n_repeats']):
            last, cintensity_kg, methods_to_plot = \
                _get_ridx_data(ridx, all_methods_to_plot, root)

            # plot EV availability + EV cons on same plot
            lEV, bEV = [[last['batch'][a][e]
                         for a in range(prm['ntw']['n'])]
                        for e in ['loads_EV', 'avail_EV']]
            if ridx == 14:
                print(f"lEV {lEV}")
            for a in range(min(prm['ntw']['n'],
                               prm['save']['max_n_profiles_plot'])):
                lEVa = [load for load in lEV[a]]
                xs = [i for i in range(len(lEV[a]))]
                bands_bEV = []
                non_avail = [i for i in range(24) if bEV[a][i] == 0]
                if len(non_avail) > 0:
                    current_band = [non_avail[0]]
                    if len(non_avail) > 1:
                        for i in range(1, len(non_avail)):
                            if non_avail[i] != non_avail[i - 1] + 1:
                                current_band.append(non_avail[i - 1] + 0.99)
                                bands_bEV.append(current_band)
                                current_band = [non_avail[i]]
                    current_band.append(non_avail[-1] + 0.999)
                    bands_bEV.append(current_band)

                fig, axs = plt.subplots(4, 2, figsize=(13, 13))

                # carbon intensity, wholesale price and grid cost coefficient
                _plot_grid_price(
                    title_ylabel_dict, axs=axs, cintensity_kg=cintensity_kg,
                    row=0, col=0, last=last,
                    colors_non_methods=colors_non_methods, lw=lw_indiv)

                # EV loads and availability
                ax = axs[2, 1]
                ax.step(xs[0:24], lEVa[0:24], color='k', where='post')
                for band in bands_bEV:
                    ax.axvspan(band[0], band[1], alpha=0.3, color='grey')
                ax.set_ylabel('EV load [kWh]')
                grey_patch = matplotlib.patches.Patch(
                    alpha=0.3, color='grey', label='EV unavailable')
                ax.legend(handles=[grey_patch], fancybox=True)

                # cum rewards
                cumrewards = _plot_cum_rewards(
                    axs, last, methods_to_plot, labels, prm,
                    row=3, col=0, lw=lw_indiv)
                if ridx == 14 and a == 0:
                    print(f"ridx {ridx} a {a} ")
                    for t in cumrewards.keys():
                        print(f"final cumrewards[{t}] = {cumrewards[t][-1]}")
                # indoor air temp
                _plot_indoor_air_temp(
                    axs, methods_to_plot, last,
                    title_ylabel_dict, prm, a,
                    row=1, col=1, lw=lw_indiv
                )

                rows = [1, 2, 0, 3]
                columns = [0, 0, 1, 1]
                entries = ['action', 'totcons', 'tot_E_heat', 'store']
                for r, c, e in zip(rows, columns, entries):
                    ax = axs[r, c]
                    for t in methods_to_plot:
                        xs = [-0.01] + list(range(24))
                        ys = last[e][t]
                        ys = [ys[step][a] for step in range(len(ys))]
                        if e == 'store' and t == 'opt':
                            ys = ys + [prm['bat']['store0'][a]]
                        elif e == 'store':
                            ys = [prm['bat']['store0'][a]] + ys

                        if e == 'totcons' \
                                and ridx == 14 \
                                and a == 0 \
                                and t in ['opt_d_d', 'baseline']:
                            print(f"totcons ridx {ridx} a {a} "
                                  f"sum(ys) {sum(ys)} t {t}")
                        ax.step(xs, ys, where='post', label=t,
                                color=prm['save']['colorse'][t],
                                lw=lw_indiv)
                    axs[r, c].set_ylabel(
                        f'{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}')
                    if r == 3:
                        axs[r, c].set_xlabel('Time [h]')

                fig.tight_layout()
                title = f'subplots example day repeat {ridx} a {a}'
                title_display = 'subplots example day'
                subtitles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                for r in range(4):
                    for c in range(2):
                        axs[r, c].set_title(subtitles[r + c * 4])
                _formatting_figure(
                    fig=fig, title=title,
                    fig_folder=prm['paths']['fig_folder'],
                    save_run=prm['save']['save_run'], legend=False,
                    high_res=prm['save']['high_res'], display_title=False,
                    title_display=title_display)

    elif not indiv:
        # do one figure with all agents and repeats
        title_ridx = 'all_ridx' if list_ridx is None \
            else f'ridx_{list_ridx}'
        lw_all = lw_all if list_ridx is None else lw_all_list_ridx
        list_ridx = range(prm['RL']['n_repeats']) \
            if list_ridx is None else list_ridx

        # Action variable
        # Heating E
        # Total consumption
        # Indoor temperature
        # Cumulative rewards
        # Battery level
        fig, axs = plt.subplots(4, 2, figsize=(13, 13))
        all_cum_rewards = initialise_dict(all_methods_to_plot)
        all_T_air = initialise_dict(all_methods_to_plot)
        rows = [1, 2, 0, 2]
        columns = [0, 0, 1, 1]
        entries = ['action', 'totcons', 'tot_E_heat', 'store']
        all_vals = initialise_dict(entries)
        for e in entries:
            all_vals[e] = initialise_dict(all_methods_to_plot)

        for ridx in list_ridx:
            last, cintensity_kg, methods_to_plot = _get_ridx_data(
                ridx, all_methods_to_plot, root)
            _plot_grid_price(
                title_ylabel_dict, axs=axs, cintensity_kg=cintensity_kg,
                row=0, col=0, last=last,
                colors_non_methods=colors_non_methods, lw=lw_indiv)

            cum_rewards_ridx = _plot_cum_rewards(
                axs, last, methods_to_plot, labels, prm, row=3,
                col=0, alpha=alpha_not_indiv, lw=lw_all,
                display_labels=False)
            for t in all_methods_to_plot:
                all_cum_rewards[t].append(cum_rewards_ridx[t])
            for a in range(prm['ntw']['n']):
                T_air_a = _plot_indoor_air_temp(
                    axs, methods_to_plot, last, title_ylabel_dict,
                    prm, a, row=1, col=1, alpha=alpha_not_indiv,
                    display_labels=False, lw=lw_all)
                # returned is a dictionary per method of
                # 24 h profie for that last epoch
                for t in methods_to_plot:
                    all_T_air[t].append(T_air_a[t])

            for r, c, e in zip(rows, columns, entries):
                for a in range(prm['ntw']['n']):
                    for t in methods_to_plot:
                        xs, ys = list(range(24)), last[e][t]
                        ys = [ys[step][a] for step in range(len(ys))]
                        if e == 'store':
                            xs = [-0.01] + xs
                            ys = [prm['bat']['store0'][a]] + ys
                        axs[r, c].step(xs, ys, where='post',
                                       color=prm['save']['colorse'][t],
                                       lw=lw_all, alpha=alpha_not_indiv)
                        all_vals[e][t].append(ys)
                    axs[r, c].set_ylabel(
                        f"{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}")
                    if r == 2:
                        axs[r, c].set_xlabel('Time [h]')

        means = initialise_dict(['T_air', 'cum_rewards'] + entries,
                                'empty_dict')
        for t in all_methods_to_plot:
            axs[1, 1].step(range(24), np.mean(all_T_air[t], axis=0),
                           where='post', label=t,
                           color=prm['save']['colorse'][t],
                           lw=lw_mean, alpha=1)
            means['T_air'][t] = np.mean(all_T_air[t], axis=0)

            axs[3, 0].plot([- 0.01] + list(range(24)),
                           [0] + list(np.mean(all_cum_rewards[t], axis=0)),
                           label=labels[t],
                           color=prm['save']['colorse'][t],
                           lw=lw_mean, alpha=1)
            means['cum_rewards'][t] = np.mean(all_T_air[t], axis=0)
            for r, c, e in zip(rows, columns, entries):
                xs = list(range(24))
                if e == 'store':
                    xs = [-0.01] + xs
                axs[r, c].step(xs, np.mean(all_vals[e][t], axis=0),
                               where='post', label=t,
                               color=prm['save']['colorse'][t],
                               lw=lw_mean, alpha=1)
                means[e][t] = np.mean(all_vals[e][t], axis=0)

        fig.tight_layout()
        title = f'subplots example day all agents {title_ridx}'
        title_display = 'subplots example day'
        subtitles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        for r in range(4):
            for c in range(2):
                axs[r, c].set_title(subtitles[r + c * 4])
        _formatting_figure(
            fig=fig, title=title,
            fig_folder=prm['paths']['fig_folder'],
            save_run=prm['save']['save_run'], legend=False,
            high_res=prm['save']['high_res'],
            display_title=False,
            title_display=title_display
        )


def _initialise_variables(prm, spaces, record, f):
    plt.rcParams['font.size'] = '11'
    action_state_space_0, state_space_0 = \
        [initialise_dict(range(prm['RL']['n_repeats']))
         for _ in range(2)]

    eval_rewards = record.eval_rewards      # [ridx][t][epoch]
    n_window = int(max(min(100, prm['RL']['n_all_epochs'] / 10), 2))

    spaces.new_state_space(prm['RL']['state_space'])
    q_tables, counters = record.q_tables, record.counter
    mean_eval_rewards = record.mean_eval_rewards  # [ridx][t][epoch]
    if prm["RL"]["type_env"] == "discrete":
        possible_states = record.possible_states \
            if record.possible_states > 0 else 1
    else:
        possible_states = None

    if prm['RL']['type_learning'] == 'q_learning':
        if type(q_tables[0]) is list:
            if len(counters[0]) > 0:
                q_entries = list(counters[0][0].keys())
                record.save_qtables = True
        else:
            q_entries = list(q_tables[0][0].keys()) \
                if record.save_qtables else list(q_tables[0].keys())
    else:
        q_entries = None
    if 'plot_profiles' not in prm['save']:
        prm['save']['plot_profiles'] = False

    return [prm, action_state_space_0, state_space_0, f, q_tables,
            counters, mean_eval_rewards, possible_states, q_entries,
            n_window, eval_rewards]


def _plot_mova_eval_per_ridx(
        ridx, eval_rewards, prm, n_window, eval_entries_plot
):
    fig = plt.figure()
    mova_baseline = get_moving_average(
        [eval_rewards[ridx]['baseline']
         for ridx in prm['RL']['n_repeats']], n_window)
    # 2 - moving average of all rewards evaluation
    for e in [e for e in eval_entries_plot if e != 'baseline']:
        mova_e = get_moving_average(
            [eval_rewards[ridx][e] for ridx in prm['RL']['n_repeats']],
            n_window)
        diff = [m - mb if m is not None else None
                for m, mb in zip(mova_e, mova_baseline)]
        plt.plot(diff, label=e, color=prm['save']['colorse'][e])
    plt.xlabel('episodes')
    plt.ylabel('reward difference rel. to baseline')
    title = f"Moving average all rewards minus baseline " \
            f"state comb {prm['RL']['statecomb_str']} repeat {ridx} " \
            f"n_window = {n_window}"
    _formatting_figure(
        fig=fig, title=title,
        fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run']
    )
    plt.close('all')


def _plot_epsilon(ridx, prm, record):
    if record.eps != {}:
        if type(record.eps[ridx][0]) is dict:
            fig = plt.figure()
            for t in prm['RL']['eval_action_choice']:
                plt.plot([record.eps[ridx][e][t]
                          for e in range(prm['RL']['n_epochs'])],
                         label=t, color=prm['save']['colorse'][t])
            plt.xlabel('epoch')
            plt.ylabel('eps')
            title = "epsilon over epochs, state comb " \
                    "{prm['RL']['state_space']}, repeat {ridx}"
            _formatting_figure(
                fig=fig, title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'], legend=True,
                high_res=prm['save']['high_res']
            )
            plt.close('all')
        else:
            fig = plt.figure()
            plt.plot([record.eps[ridx][e]
                      for e in range(prm['RL']['n_epochs'])])
            plt.xlabel('epoch')
            plt.ylabel('eps')
            title = f"epsilon over epochs, state comb " \
                    f"{prm['RL']['state_space']}, repeat {ridx}"
            _formatting_figure(
                fig=fig, title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run']
            )
            plt.close('all')


def _plot_best_psi(
        ridx, q_tables, prm, record, possible_states,
        index_to_val, eval_entries_plot_indiv, spaces
):
    best_theta = initialise_dict(eval_entries_plot_indiv)
    q = q_tables[ridx][prm['RL']['n_epochs'] - 1] \
        if record.save_qtables else q_tables[ridx]
    if prm['RL']['state_space'] == [None]:
        for t in eval_entries_plot_indiv:
            na = prm['ntw']['n'] if t.split('_')[2] in ['d', 'Cd'] else 1
            best_theta[t] = initialise_dict(range(na), type_obj='empty_dict')
            for a in range(na):
                best_theta[t][a] = np.zeros((possible_states,))
                for s in range(possible_states):
                    indmax = np.argmax(q[t][a][s])
                    best_theta[t][a][s] = \
                        index_to_val([indmax], typev='action')[0]

                # plot historgram of best theta per method
                fig, ax = plt.subplots()
                y_pos = np.arange(len(eval_entries_plot_indiv))
                i_tables = [a if t.split('_')[2] == 'd' else 0
                            for t in eval_entries_plot_indiv]
                best_thetas = \
                    [best_theta[eval_entries_plot_indiv[it]][i_tables[it]][0]
                        for it in range(len(eval_entries_plot_indiv))]
                colors_bars = [prm['save']['colorse'][t]
                               for t in eval_entries_plot_indiv
                               if t[-1] != '0']
                ax.bar(y_pos, best_thetas, align='center',
                       alpha=0.5, color=colors_bars)
                plt.ylim([0, 1])
                title = f'best theta per method state None ' \
                        f'repeat {ridx} a {a}'

                ax.set_ylabel(r'best $\theta$ [-]')
                _formatting_ticks(
                    ax, fig, ys=eval_entries_plot_indiv,
                    title=title,
                    fig_folder=prm['paths']['fig_folder'],
                    save_run=prm['save']['save_run'],
                    high_res=prm['save']['high_res'],
                    display_title=False
                )

    # if one dimensional state space - best theta[method, state]
    elif prm['RL']['dim_states'] == 1:
        for a in range(prm['ntw']['n']):
            fig, ax = plt.subplots()
            theta_M = []
            for t in eval_entries_plot_indiv:
                index_a = a if t.split('_')[2] in ['d', 'Cd'] else 0
                theta_M.append([best_theta[t][index_a][int(s)]
                                for s in range(possible_states)])
            im = ax.imshow(theta_M, vmin=0, vmax=1)
            title = f"best theta per method and per state state space " \
                    f"{prm['RL']['statecomb_str']} repeat {ridx} a {a}"
            _formatting_ticks(
                ax, fig, xs=eval_entries_plot_indiv,
                ys=range(possible_states),
                title=title, im=im,
                grid=False,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'],
                display_title=False
            )

    # if two dimensional state space heatmap for each method
    # besttheta[state1, state2]
    elif prm['RL']['dim_states'] == 2:
        for a in range(prm['ntw']['n']):
            for i_t in range(len(eval_entries_plot_indiv)):
                t = eval_entries_plot_indiv[i_t]
                M = np.zeros((record.granularity_state0,
                              record.granularity_state1))
                index_a = a if t.split('_')[2] in ['d', 'Cd'] else 0
                for s in range(possible_states):
                    s1, s2 = spaces.global_to_indiv_index(
                        'state', s, multipliers=granularity_to_multipliers(
                            [record.granularity_state0,
                             record.granularity_state1]))
                    M[s1, s2] = best_theta[t][index_a][s]

                fig, ax = plt.subplots()
                plt.xlabel(prm['RL']['state_space'][1])
                plt.ylabel(prm['RL']['state_space'][0])
                im = ax.imshow(M, vmin=0, vmax=1)
                title = f"best theta per state combination state space " \
                        f"{prm['RL']['state_space']} repeat {ridx} a {a}"
                _formatting_ticks(
                    ax, fig, ys=record.granularity_state1,
                    xs=record.granularity_state0,
                    title=title, im=im, grid=False,
                    fig_folder=prm['paths']['fig_folder'],
                    save_run=prm['save']['save_run'],
                    high_res=prm['save']['high_res'],
                    display_title=False
                )


def _plot_q_values(
        q_tables, ridx, prm, eval_entries_plot_indiv,
        index_to_val, q_entries
):
    # plot all values in one figure if there is only one state
    if prm['RL']['state_space'] == [None] and prm['ntw']['n'] < 4:
        for a in range(prm['ntw']['n']):
            # plot heat map of value of each action for different methods
            # 2D array of best theta values
            M = np.zeros((len(eval_entries_plot_indiv), prm['RL']['n_action']))

            for i_t in range(len(eval_entries_plot_indiv)):
                t = eval_entries_plot_indiv[i_t]
                distr_learning = t.split('_')[2]
                i_table = a if distr_learning == 'd' else 0
                qvals = q_tables[ridx][prm['RL']['n_epochs'] - 1][t][i_table][
                    0]  # in final epoch, in 0-th (unique) state
                M[i_t, :] = qvals

            xs = q_entries
            b = index_to_val([0], typev='action')[0]
            m = index_to_val([1], typev='action')[0] - b
            ys = [m * i + b for i in range(prm['RL']['n_action'])]
            fig, ax = plt.subplots()
            im = ax.imshow(M)
            title = "qval per action per t state None repeat {ridx} a {a}"
            _formatting_ticks(
                ax, fig, xs=xs, ys=ys, title=title, im=im, grid=False,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'], display_title=False)


def _plot_env_input(ridx, prm, record):
    batch_entries = ['loads', 'gen', 'lds_EV', 'avail_EV']
    if prm['RL']['deterministic'] == 2:
        seeds = np.load(prm['paths']['seeds_path'])
        heatavail = {}
        for e in batch_entries:
            fig, axs = plt.subplots(prm['ntw']['n'], 1, squeeze=0)
            axs = axs.ravel()
            for a in range(prm['ntw']['n']):
                n = len(np.load(f'batch_{int(seeds[0] + 1)}_a0_lds.npy',
                                mmap_mode='c'))
                heatavail[a] = np.zeros((n,))
                for seed in record.seed[ridx]:
                    str_seed = str(int(seed)) if seed > seeds[0] + 1 else \
                        f'deterministic_{record.ind_seed_deterministic[ridx]}'
                    str_batch = f'batch_{str_seed}_a{a}_{e}.npy'
                    if os.path.exists(str_batch):
                        batch_a_e = np.load(str_batch, mmap_mode='c')
                        if e == 'avail_EV':
                            heatavail[a] += batch_a_e
                        else:
                            axs[a].plot(
                                batch_a_e, alpha=1 / prm['RL']['n_epochs'])
                if e == 'avail_EV':
                    heatavail_plot = np.reshape(heatavail[a], (1, n))
                    sns.heatmap(heatavail_plot, cmap="YlGn",
                                cbar=True, ax=axs[a])
            title = f"noisy repeat {ridx} {e}"
            _title_and_save(
                title, fig, prm['paths']['fig_folder'],
                prm['save']['save_run']
            )
            fig.show()
            fig.savefig(e)

    else:
        if prm['RL']['deterministic'] == 1 \
                and os.path.exists(f'deterministic_parameters_ridx{ridx}.npy'):
            batch = np.load(f'deterministic_parameters_ridx{ridx}.npy',
                            allow_pickle=True)[-1]
        elif prm['RL']['deterministic'] == 0 and 'batch' in record.last[ridx]:
            # indeterministic, just plot the last epoch, evaluation step
            batch = record.last[ridx]['batch']['eval']
        for e in batch_entries:
            fig, axs = plt.subplots(prm['ntw']['n'], 1, squeeze=0)
            axs = axs.ravel()
            for a in range(prm['ntw']['n']):
                axs[a].plot(batch[a][e])
                axs[a].set_title('{a}')
            title = f"deterministic repeat {ridx} {e}"
            _title_and_save(
                title, fig, prm['paths']['fig_folder'],
                prm['save']['save_run']
            )
        else:
            print(f'no new deterministic batch for ridx = {ridx}')


def _video_visit_states(
        ridx, q_entries, counters, possible_states,
        prm, record, spaces
):
    import cv2
    counters = counters[ridx]
    rl = prm['RL']
    for t in q_entries:
        maxval = np.max(
            [np.sum(counters[rl['n_epochs'] - 1][t][s])
             for s in range(possible_states)])

        for epoch in range(rl['n_epochs']):
            fig = plt.figure()
            if rl['dim_states'] == 2:
                counters_per_state = \
                    np.zeros((record.granularity_state0,
                              record.granularity_state1))
                plt.ylabel(rl['state_space'][0])
                plt.xlabel(rl['state_space'][1])
                for s in range(possible_states):
                    s1, s2 = spaces.global_to_indiv_index(
                        'state', s, multipliers=granularity_to_multipliers(
                            [record.granularity_state0,
                             record.granularity_state1]))
                    counters_per_state[s1, s2] = sum(counters[epoch][t][s])
            elif rl['dim_states'] == 1:
                counters_per_state = \
                    np.zeros((record.granularity_state0, rl['n_action']))
                plt.ylabel(rl['state_space'][0])
                plt.xlabel(r'$\theta$ [-]')
                for s in range(possible_states):
                    for a in range(rl['n_action']):
                        counters_per_state[s, a] = counters[epoch][t][s][a]
            else:
                counters_per_state = np.zeros((possible_states, 1))
                for s in range(possible_states):
                    counters_per_state[s] = sum(counters[epoch][t][s])
                plt.xlabel(rl['state_space'][0])
            plt.imshow(counters_per_state, vmin=0, vmax=maxval)

            if rl['dim_states'] == 2:
                plt.ylabel(rl['state_space'][1])
            plt.colorbar()
            plt.close('all')
            if epoch == 0:  # initialise video
                fig.savefig('fig')
                img0 = cv2.imread('fig.png')
                height, width, layers = img0.shape
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                title = \
                    prm['paths']['fig_folder'] / f'qlearn_ridx{ridx}_{t}.avi' \
                    if prm['paths']['fig_folder'] is not None \
                    else f'qlearn_ridx{ridx}_{t}.avi'
                out = cv2.VideoWriter(title, fourcc, 40.0, (width, height))
                os.remove('fig.png')
            else:
                figname = 'fig{}'.format(epoch)
                fig.savefig(figname)
                plt.close()
                frame = cv2.imread(figname + '.png', 1)
                os.remove(figname + '.png')
                out.write(frame)
        out.release()


def _plot_final_explorations(
        ridx, prm, q_entries, action_state_space_0,
        state_space_0, record, counters
):
    rl = prm['RL']
    for t in q_entries:
        na = prm['ntw']['n'] if t.split('_')[1] == '1' else 1
        for a in range(na):
            action_state_space_0[ridx][a], state_space_0[ridx][a] =\
                [initialise_dict(q_entries) for _ in range(2)]
            fig, ax = plt.subplots()
            counters_plot = counters[ridx][rl['n_epochs'] - 1][t][a] \
                if record.save_qtables else counters[ridx][t][a]
            im = ax.imshow(counters_plot, aspect='auto')
            fig.colorbar(im, ax=ax)
            title = f"Explorations per state action pair state space " \
                    f"{rl['statecomb_str']} repeat {ridx} " \
                    f"method {t} a {a}"
            _title_and_save(
                title, fig, prm['paths']['fig_folder'], prm['save']['save_run']
            )

            sum_action_0, sum_state_0 = 0, 0
            for s in range(rl['n_total_discrete_states']):
                if sum(counters_plot[s]) == 0:
                    sum_state_0 += 1
                for ac in range(rl['n_discrete_actions']):
                    if counters_plot[s][ac] == 0:
                        sum_action_0 += 1
            state_space_0[ridx][a][t] = \
                sum_state_0 / rl['n_total_discrete_states']
            action_state_space_0[ridx][a][t] = \
                sum_action_0 / (rl['n_total_discrete_states']
                                * rl['n_discrete_actions'])


def _plot_unfeasible_attempts(ridx, record, prm):
    fig = plt.figure()
    plt.plot(record.n_not_feas[ridx])
    plt.xlabel('epoch')
    plt.ylabel('no of non feasible attempts before feasibility')
    title = f"no of non feasible attempts before feasibility " \
        f"ridx {ridx} state space {prm['RL']['statecomb_str']}"
    _title_and_save(
        title, fig, prm['paths']['fig_folder'], prm['save']['save_run']
    )
    plt.close('all')


def _plot_results_all_ridx(
        n_window, prm, eval_entries_plot, record,
        moving_average=True):
    fig = plt.figure(figsize=(4.600606418100001, 4.2))
    min_val, max_val = 100, - 100
    plt.rcParams['font.size'] = '10'

    for e in [e for e in eval_entries_plot if e != 'baseline']:
        print(f"len(mean_eval_rewards_per_hh) = "
              f"{len(record.mean_eval_rewards_per_hh)} "
              f"vs prm['RL']['n_repeats'] {prm['RL']['n_repeats']}")
        print(f"len(mean_eval_rewards_per_hh[0][e]) = "
              f"{len(record.mean_eval_rewards_per_hh[0][e])} "
              f"vs prm['RL']['n_all_epochs'] {prm['RL']['n_all_epochs']}")

        results = np.array(
            [[None if record.mean_eval_rewards_per_hh[ridx][e][epoch] is None
              else record.mean_eval_rewards_per_hh[ridx][e][epoch]
              - record.mean_eval_rewards_per_hh[ridx]['baseline'][epoch]
              for epoch in range(prm['RL']['n_all_epochs'])]
             for ridx in range(prm['RL']['n_repeats'])])
        if moving_average:
            results = np.array(
                [get_moving_average(results[ridx], n_window, Nones=False)
                 for ridx in range(prm['RL']['n_repeats'])])
        results = np.array(results, dtype=np.float)

        all_nans = True if sum(not np.isnan(r) for r in results[0]) == 0 \
            else False
        try:
            mean_results = np.nanmean(results, axis=0) \
                if not all_nans else None
        except Exception as ex:
            print(f"ex {ex} results {results} l 400 plotting")
        p25, p50, p75, p25_not_None, p75_not_None, epoch_not_None = \
            record.results_to_percentiles(
                e, prm,
                mov_average=moving_average,
                n_window=n_window
            )

        min_val = np.min(p25_not_None) if np.min(p25_not_None) < min_val \
            else min_val
        max_val = np.max(p75_not_None) if np.max(p75_not_None) > max_val \
            else max_val

        ls = 'dotted' if e == 'opt' else '-'
        plt.plot(p50, label=e, color=prm['save']['colorse'][e], ls=ls)
        plt.fill_between(epoch_not_None, p25_not_None, p75_not_None,
                         color=prm['save']['colorse'][e], alpha=0.3)
    try:
        plt.hlines(y=0, xmin=0, xmax=len(mean_results), colors='k',
                   linestyle='dotted')
    except Exception as ex:
        print(f"ex {ex} mean_results {mean_results} l 821 plotting")

    if moving_average and prm['ntw']['n'] == 1:
        spread = max_val - min_val
        lower_bound = min_val - 0.02 * spread
        upper_bound = max_val + 0.02 * spread
        np.save('lower_bound0', lower_bound)
        np.save('upper_bound0', upper_bound)
        print(f'new up lb [{lower_bound}, {upper_bound}]')

    lower_bound, upper_bound = [np.load(prm['paths']['main_dir'] / f"{e}0.npy")
                                for e in ['lower_bound', 'upper_bound']]
    plt.ylim([lower_bound, upper_bound])
    plt.gca().set_yticks(np.arange(-0.25, 0.2, 0.05))
    plt.ylim([lower_bound, upper_bound])

    plt.gca().set_yticks(np.arange(-0.25, 0.2, 0.05))
    plt.xlabel('Episode')
    ylabel = 'Moving average ' if moving_average else ''
    ylabel += 'Difference between reward and baseline reward [£/hr]'
    plt.ylabel(ylabel)
    title = f"moving average n_window = {n_window} " if moving_average else ""
    title += f"med, 25-75th percentile over repeats state comb " \
             f"{prm['RL']['statecomb_str']}"
    print(f"prm['paths']['fig_folder'] {prm['paths']['fig_folder']}")
    _formatting_figure(
        fig=fig, title=title,
        fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run'],
        high_res=prm['save']['high_res'],
        legend=False, display_title=False
    )

    return lower_bound, upper_bound


def _barplot_metrics(
        prm, metrics, metric_entries, base_entries, colors_barplot_baseentries,
        eval_entries_notCd, eval_entries_bars, colors_barplot, lower_bound,
        upper_bound, eval_entries_plot, f):
    titles = {'end': 'Average reward over the last 20% evaluations',
              'end_test':
                  f"Average reward over {prm['RL']['n_end_test']} "
                  f"test days after the end of learning",
              'mean': 'Average reward over all evaluations',
              'DT':
                  'Dispersion across Time (DT): '
                  'Interquartile range (IQR) of diffence between '
                  'subsequent rewards over all evaluations',
              'SRT':
                  'Short-term risk across Time (SRT): '
                  'Conditional value at risk (CVaR) of the difference'
                  ' in evaluation returns among successive evaluations',
              'LRT':
                  'Long-term Risk across Time (LRT): worst drop in '
                  'returns relative to the so-far best evaluation reward',
              'DR':
                  'Dispersion across Runs (DR) - IQR of final'
                  ' evaluation returns across different random seeds',
              'RR':
                  'Risk across Runs (RR): CVaR across Runs = measure '
                  'of the expected performance of the worst runs',
              'end_bl':
                  'Average reward over the last 20% evaluations '
                  'relative to the baseline',
              'end_test_bl':
                  f"Average reward over {prm['RL']['n_end_test']} "
                  f"test days after the end of learning relative to "
                  f"the baseline",
              }
    subplots_i_j = {'end_test_bl': [0, 0],
                    'LRT': [0, 1],
                    'DR': [0, 2],
                    'RR': [1, 0],
                    'SRT': [1, 1],
                    'DT': [1, 2]
                    }
    for plot_type in ['indivs', 'subplots']:
        if plot_type == 'subplots':
            fig, axs = plt.subplots(2, 3)
        for m in metric_entries + [m + '_p50' for m in metric_entries[0:4]]:
            eval_entries_bars_ = eval_entries_notCd
            print(f"eval_entries_bars_ {eval_entries_bars_}")
            print("eval_entries_bars_")
            colors_barplot_ = [prm['save']['colorse'][e]
                               for e in eval_entries_notCd]
            ave = 'ave'
            m_ = m
            if m[-3:] == 'p50':
                ave = 'p50'
                m_ = m[0:-4]
                eval_entries_bars_ = eval_entries_bars
                colors_barplot_ = colors_barplot
            bars, err = [[metrics[m_][s][e]
                          for e in eval_entries_bars_]
                         for s in [ave, 'std']]
            baseline, opt = [metrics[m_][ave][e]
                             if e in eval_entries_plot else None
                             for e in ['baseline', 'opt']]
            print(f"np.shape(err) {np.shape(err)}")
            if err[0] is None:
                err = None
            display_legend = False if m_ == 'end' else True
            display_title = False if m_ == 'end' else True
            lb = lower_bound if m_ == 'end' else None
            ub = upper_bound if m_ == 'end' else None
            display_ticks = False if m_ == 'end' else True
            if plot_type == 'indivs':
                fig = _barplot(
                    bars, eval_entries_bars_, prm, baseline=baseline,
                    opt=opt, colors=colors_barplot_, error=err,
                    title=titles[m_], display_legend=display_legend,
                    display_title=display_title, lower_bound=lb,
                    upper_bound=ub, display_ticks=display_ticks)
                _formatting_figure(
                    fig=fig, title=m, fig_folder=prm['paths']['fig_folder'],
                    save_run=prm['save']['save_run'], legend=False,
                    high_res=prm['save']['high_res'], display_title=False,
                    title_display=titles[m_])

            elif plot_type == 'subplots':
                if m in subplots_i_j:
                    i, j = subplots_i_j[m]
                    axs[i, j] = _barplot(
                        bars, eval_entries_bars_, prm,
                        baseline=baseline, opt=opt,
                        colors=colors_barplot_, error=err,
                        title=m, display_legend=display_legend,
                        display_title=display_title, lower_bound=lb,
                        upper_bound=ub, display_ticks=display_ticks,
                        ax0=axs[i, j])
                    if j == 0:
                        axs[i, j].set_ylabel('£/home/h')
            # do end for with signs for each
            fig2 = plt.figure(figsize=(3.25, 7 * 0.75))
            ax = plt.gca()

            xs, colors_plot_end = {}, {}
            print(f"base_entries {base_entries}")
            for i in range(len(base_entries)):
                splits = base_entries[i].split('_')
                label = splits[0] + '_' + splits[1] if len(splits) > 1 else base_entries[i]
                xs[label] = i
                colors_plot_end[label] = colors_barplot_baseentries[i]
            baseline, opt = [metrics[m_][ave][e]
                             if e in metrics[m_][ave] else None
                             for e in ['baseline', 'opt']]
            for e in eval_entries_notCd:
                splits = e.split('_')
                label = splits[0] + '_' + splits[1] if len(splits) > 1 else e
                if len(splits) < 2:
                    ls = 'o'
                elif splits[2] == 'd':
                    ls = 'o'
                elif splits[2] == 'c':
                    ls = 'x'
                elif splits[2] == 'Cc':
                    ls = '^'
                legend = splits[2] if len(splits) > 1 else e
                to_plot = metrics[m_][ave][e] - baseline \
                    if m_ == 'end' else metrics[m_][ave][e]
                ax.plot(xs[label], to_plot, ls,
                        color=colors_plot_end[label],
                        markeredgewidth=2, markerfacecolor="None",
                        label=legend, linewidth=2)
            maxval = {}
            for label in ['env_r', 'env_d', 'env_A', 'opt_r',
                          'opt_d', 'opt_A', 'opt_n']:
                if sum([label_[0:len(label)] == label
                        for label_ in metrics[m_]['ave'].keys()]) > 0:
                    maxval[label] = \
                        max([metrics[m_][ave][e]
                             for e in metrics[m_][ave].keys()
                             if len(e.split('_')) > 1
                             and e.split('_')[0] + '_' + e.split('_')[1]
                             == label])
                    plottext = maxval[label] - baseline \
                        if m_ == 'end' else maxval[label]
                    ax.text(xs[label], plottext + 0.01,
                            s='%.3f' % float(plottext),
                            ha='center', va='bottom', fontsize=9)
            plotbaseline = 0 if m_ == 'end' else baseline
            ax.axhline(plotbaseline, linestyle='--', color="k")
            if opt is not None:
                plotopt = opt - baseline if m_ == 'end' else opt
                ax.axhline(plotopt, linestyle='--',
                           color=prm['save']['colorse']['opt'])
            title = 'compare all median 25-75th percentile' + m
            if m_ == 'end':
                plt.ylim([lower_bound, upper_bound])

            ax = plt.gca()
            ax.set_yticks([])
            ax.set_xticks([])
            _title_and_save(
                title, fig2, prm['paths']['fig_folder'],
                prm['save']['save_run'], high_res=prm['save']['high_res'],
                display_title=False)

    _formatting_figure(
        fig=fig, title='subplots_all_metrics',
        fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run'], legend=False,
        high_res=prm['save']['high_res'], display_title=False
    )

    return f


def _heatmap_savings_per_method(metrics, prm):
    # do heat maps of reduction rel to baseline
    M = {}
    rewards = ['r', 'd', 'A', 'n']
    distrs = ['c', 'd', 'Cc', 'Cd']
    M['opt'] = [[np.nan for r in rewards] for d in distrs]
    M['env'] = [[np.nan for r in rewards] for d in distrs]
    all_vals = []
    for e in metrics['end']['ave'].keys():
        if e not in ['opt', 'baseline', 'random']:
            source, rewardtype, distr = [e.split('_')[i] for i in range(3)]
            i_reward, i_distr = \
                [[i for i in range(len(arr)) if arr[i] == x][0]
                 for arr, x in zip([rewards, distrs], [rewardtype, distr])]
            val = metrics['end']['ave'][e] - metrics['end']['ave']['baseline']
            M[source][i_distr][i_reward] = val
            all_vals.append(val)
    if len(all_vals) > 0:
        minval, maxval = np.min(all_vals), np.max(all_vals)
        for s in ['opt', 'env']:
            fig, ax = plt.subplots()
            current_cmap = matplotlib.cm.get_cmap()
            current_cmap.set_bad(color='grey')
            im = ax.imshow(M[s], aspect='auto', vmin=minval, vmax=maxval)
            fig.colorbar(im, ax=ax)
            # ax.clim(minval, maxval)
            plt.ylabel('distribution of learning')
            plt.xlabel('reward definition')
            ax.set_xticks(range(len(rewards)))
            ax.set_xticklabels(rewards)
            ax.set_yticks(range(len(distrs)))
            ax.set_yticklabels(distrs)
            title = f"reduction in costs relative to baseline, " \
                    f"data source = {s}"
            _title_and_save(
                title, fig, prm['paths']['fig_folder'], prm['save']['save_run']
            )
        np.save(prm['paths']['fig_folder'] / 'M_reductions', M)


def _barplot_breakdown_savings(record, prm):
    # all break down rewards except for the last three ones
    # which are individual values
    labels = record.break_down_rewards_entries[:-3]
    bars = [[] for _ in range(len(labels))]
    shares_reduc = {}
    tots = {}
    for t in prm['RL']['type_eval']:
        shares_reduc[t] = []
        for i, label in enumerate(labels):
            record_obj = record.__dict__[label]
            mult = prm['syst']['co2tax'] if label == 'emissions' else 1
            bars[i].append(
                np.mean([[(record_obj[ridx]['baseline'][epoch]
                           - record_obj[ridx][t][epoch])
                          * mult / (prm['ntw']['n'] + prm['ntw']['nP'])
                          for epoch in range(prm['RL']['start_end_eval'],
                                             len(record_obj[ridx][t]))]
                         for ridx in range(prm['RL']['n_repeats'])]))
        tots[t] = sum(bars[i][-1] for i, label in enumerate(labels)
                      if label in ['dc', 'sc', 'gc'])
        shares_reduc[t].append(
            [bars[i][-1] / tots[t] if tots[t] > 0 else None
             for i in range(len(labels))]
        )

    barWidth = 1 / (len(labels) + 1)
    rs = []
    rs.append(np.arange(len(prm['RL']['type_eval'])))
    for ir in range(len(labels) - 1):
        rs.append([x + barWidth for x in rs[ir]])
    plt.figure()
    for ir in range(len(labels)):
        plt.bar(rs[ir], bars[ir], width=barWidth, label=labels[ir])
    plt.xlabel('evaluation')
    plt.xticks([r + barWidth
                for r in range(len(bars[0]))], prm['RL']['type_eval'],
               rotation='vertical')
    plt.legend()
    plt.title('savings relative to baseline costs / emissions')
    plt.close('all')


def _barplot_indiv_savings(record, prm):
    # plot the invidivual savings
    if len(np.shape(record.indiv_sc[0]['baseline'][0])) == 0:
        indiv_savings = True \
            if not np.isnan(record.indiv_sc[0]['baseline'][0]) \
            else False
    else:
        indiv_savings = True \
            if not np.isnan(record.indiv_sc[0]['baseline'][0][0]) \
            else False
    if indiv_savings:
        eval_not_baseline = [t for t in prm['RL']['type_eval']
                             if t != 'baseline']
        savings_a, share_sc, std_savings = [
            [[] for _ in range(prm['ntw']['n'])]
            for _ in range(3)
        ]
        for a in range(prm['ntw']['n']):
            for t in eval_not_baseline:
                savings_sc_a, savings_gc_a = [
                    np.mean([[(reward[ridx]['baseline'][epoch][a]
                               - reward[ridx][t][epoch])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 prm['RL']['n_epochs'])]
                             for ridx in range(prm['RL']['n_repeats'])])
                    for reward in [record.__dict__['indiv_sc'],
                                   record.__dict__['indiv_gc']]]
                share_sc[a].append(
                    savings_sc_a / (savings_sc_a + savings_gc_a))
                savings_a_all = \
                    [[(record.__dict__['indiv_c'][ridx]['baseline'][epoch][a]
                       - record.__dict__['indiv_c'][ridx][t][epoch])
                      for epoch in range(prm['RL']['start_end_eval'],
                                         prm['RL']['n_epochs'])]
                     for ridx in range(prm['RL']['n_repeats'])]
                savings_a[a].append(np.mean(savings_a_all))
                std_savings[a].append(np.std(savings_a_all))
        for it in range(len(eval_not_baseline)):
            if eval_not_baseline[it] == 'opt_d_d':
                savings_opt_d_d = [savings_a[a][it]
                                   for a in range(prm['ntw']['n'])]
                print(f"savings per agent opt_d_d: {savings_opt_d_d}")
                print(f"mean {np.mean(savings_opt_d_d)}, "
                      f"std {np.std(savings_opt_d_d)}, "
                      f"min {min(savings_opt_d_d)}, "
                      f"max {max(savings_opt_d_d)}")

        # plot total individual savings
        labels = range(prm['ntw']['n'])
        barWidth = 1 / (len(labels) + 1)
        rs = []
        rs.append(np.arange(len(prm['RL']['type_eval']) - 1))
        for a in range(len(labels) - 1):
            rs.append([x + barWidth for x in rs[a]])

        fig = plt.figure()
        for a in range(len(labels)):
            plt.bar(rs[a], savings_a[a], width=barWidth,
                    label=labels[a], yerr=std_savings[a])
        plt.xlabel('savings per agent')
        plt.xticks([r + barWidth
                    for r in range(len(prm['RL']['type_eval']))],
                   prm['RL']['type_eval'], rotation='vertical')
        plt.legend()
        title = "savings per agent relative to " \
                "individual baseline costs"
        _title_and_save(
            title, fig, prm['paths']['fig_folder'],
            prm['save']['save_run']
        )
        plt.close('all')

        # plot share of energy vs battery savings individually
        fig = plt.figure()
        for a in range(len(labels)):
            plt.bar(rs[a], share_sc[a], width=barWidth, label=labels[a])
        plt.xlabel('share of individual savings from battery costs savings')
        plt.xticks([r + barWidth
                    for r in range(len(prm['RL']['type_eval']))],
                   prm['RL']['type_eval'], rotation='vertical')
        plt.legend()
        title = "share of individual savings from battery costs " \
            "savings relative to individual baseline costs"
        _title_and_save(
            title, fig, prm['paths']['fig_folder'],
            prm['save']['save_run']
        )
        plt.close('all')


def _distribution_savings(mean_eval_rewards_per_hh, prm, aggregate='daily'):
    fig = plt.figure()
    test_savings = {}
    for t in [t for t in mean_eval_rewards_per_hh[0] if t != 'baseline']:
        test_savings[t] = []
        for ridx in range(prm['RL']['n_repeats']):
            rewards_t, rewards_bsl = \
                [mean_eval_rewards_per_hh[ridx][type_eval][
                 prm['RL']['n_epochs']:]
                 for type_eval in [t, 'baseline']]
            savings_rel_baseline = \
                [reward - baseline for reward, baseline
                 in zip(rewards_t, rewards_bsl)]
            if aggregate == 'daily':
                test_savings[t] += savings_rel_baseline
            elif aggregate == 'test_period':
                test_savings[t] += [np.mean(savings_rel_baseline)]

        plt.hist(test_savings[t], alpha=0.5, label=t,
                 color=prm['save']['colorse'][t])

    plt.legend()
    plt.xlabel("Average test savings [£/h/home]")
    plt.ylabel("Count")
    title = 'distribution of daily savings per household and learning type'
    if aggregate == 'daily':
        title += ' for each day of testing'
    elif aggregate == 'test_period':
        title += f" over each testing period of {prm['RL']['n_end_test']} days"
    _formatting_figure(
        fig=fig, title=title, fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run'], legend=True,
        high_res=prm['save']['high_res']
    )


def plotting(record, spaces, prm, f):
    """Plot and save results."""
    [prm, action_state_space_0, state_space_0, f,
     q_tables, counters, mean_eval_rewards, possible_states,
     q_entries, n_window, eval_rewards] = _initialise_variables(
        prm, spaces, record, f)
    [eval_entries_plot, eval_entries_bars, eval_entries,
     eval_entries_notCd, eval_entries_plot_indiv,
     colors_barplot_baseentries, colors_barplot, base_entries] = \
        _eval_entries_plot_colors(prm)
    # red = prm['save']['colorse']['opt']
    # purple = prm['save']['colorse']['opt_d_d']
    red = (234 / 255, 53 / 255, 37 / 255)
    prm['save']['colorse']['env_r_d'] = red
    prm['save']['colorse']['env_r_d'] = red
    prm['save']['colorse']['env_r_c'] = red
    # blue = prm['save']['colorse']['opt_r_d']
    prm['save']['colorse']['opt'] = 'grey'
    green = prm['save']['colorse']['env_d_d']
    prm['save']['colorse']['opt_d_d'] = green
    # mean_end_rewards, mean_end_test_rewards, mean_eval_rewards_per_hh = \
    #     _get_mean_rewards(
    #         prm, mean_eval_rewards, action_state_space_0,
    #         state_space_0, eval_entries_plot)
    # metrics, metrics_entries = _get_metrics(
    #     mean_end_rewards, mean_end_test_rewards, prm,
    #     mean_eval_rewards_per_hh, eval_entries_plot)

    record.get_mean_rewards(
        prm, action_state_space_0,
        state_space_0, eval_entries_plot
    )
    metrics, metrics_entries = record.get_metrics(
        prm, eval_entries_plot)

    mean_eval_rewards_per_hh = record.mean_eval_rewards_per_hh

    # 1 - plot non moving  average results 25th, 50th,
    # 75th percentile for all ridx
    _plot_results_all_ridx(
        n_window, prm, eval_entries_plot, record,
        moving_average=False)

    # 2 - plot moving average results 25th, 50th, 75th percentile for all ridx
    lower_bound, upper_bound = _plot_results_all_ridx(
        n_window, prm, eval_entries_plot, record,
        moving_average=True)

    # 3 - bar plot metrics
    f = _barplot_metrics(
        prm, metrics, metrics_entries, base_entries,
        colors_barplot_baseentries, eval_entries_notCd,
        eval_entries_bars, colors_barplot, lower_bound,
        upper_bound, eval_entries_plot, f)

    # 4 - plot distribution of daily savings
    _distribution_savings(mean_eval_rewards_per_hh, prm, aggregate='daily')
    _distribution_savings(
        mean_eval_rewards_per_hh, prm, aggregate='test_period'
    )

    # 4 - heat map of reductions rel to baseline per data source,
    # reward ref and MARL structure
    _heatmap_savings_per_method(metrics, prm)

    if prm['save']['plot_type'] > 0:
        # 5 - do bar plot of all costs reduction rel to baseline,
        _barplot_breakdown_savings(record, prm)

        # 6 - plot individual savings as well as share battery
        # vs energy costs in individual savings
        _barplot_indiv_savings(record, prm)

    # 7 - plotting results example day household variables
    if prm['save']['plot_profiles']:
        root = prm['paths']['folder_run']
        _plotting_res(root, prm, indiv=False)
        _plotting_res(root, prm, indiv=True)
        for ridx in range(prm['RL']['n_repeats']):
            _plotting_res(root, prm, indiv=False, list_ridx=[ridx])

    # other repeat-specific plots:
    for ridx in range(prm['RL']['n_repeats']):
        if prm['save']['plot_type'] > 0:
            # 10 - plot moving average of all evaluation rewards
            # for each repeat
            if prm['save']['plot_type'] > 0 \
                    and prm['save']['plot_indiv_repeats_rewards']:
                _plot_mova_eval_per_ridx(
                    ridx, eval_rewards, prm, n_window, eval_entries_plot)
        if prm['save']['plot_type'] > 1:
            # 11 - plot epsilon over time for each repeat
            if prm['save']['plot_type'] > 1 \
                    and 'eps' in record.__dict__.keys():
                _plot_epsilon(ridx, prm, record)

            # 12 - plot best psi action value per state
            if prm['RL']['type_learning'] == 'q_learning' \
                    and prm['ntw']['n'] < 4:
                _plot_best_psi(
                    ridx, q_tables, prm, record, possible_states,
                    spaces.index_to_val, eval_entries_plot_indiv, spaces)

            # 13 - plot q values
            if prm['RL']['type_learning'] == 'q_learning':
                _plot_q_values(q_tables, ridx)

            # 14 - plot environment input
            if prm['RL']['deterministic'] is not None \
                    and prm['RL']['deterministic'] > 0 \
                    and prm['save']['plotting_batch']:
                _plot_env_input(ridx, prm, record)

            # 15 - make video of visits to each state
            if prm['RL']['type_learning'] == 'q_learning' \
                    and prm['save']['make_video'] \
                    and len(counters[0]) > 0 \
                    and not prm['RL']['server'] \
                    and prm['RL']['state_space'] != [None]:
                _video_visit_states(
                    ridx, q_entries, counters, possible_states,
                    prm, record, spaces)

            # 16 -  final number of exploration of actions in each state
            if prm['RL']['type_learning'] == 'q_learning' \
                    and len(counters[0]) > 0:
                _plot_final_explorations(
                    ridx, prm, q_entries, action_state_space_0,
                    state_space_0, record, counters)

            # 17 - n not feas vs variables vs time step
            _plot_unfeasible_attempts(ridx, record, prm)

    if prm['save']['save_run']:
        np.save(prm['paths']['fig_folder'] / 'eval_entries', eval_entries)
        np.save(prm['paths']['fig_folder'] / 'state_space_0', state_space_0)
        np.save(prm['paths']['fig_folder'] / 'action_state_space_0',
                action_state_space_0)
        np.save(prm['paths']['fig_folder'] / 'metrics', metrics)
    np.save('metrics', metrics)
    plt.close('all')

    return f, metrics
