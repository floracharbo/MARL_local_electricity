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
import torch as th
from config.generate_colors import generate_colors
from utilities.userdeftools import (data_source, distr_learning,
                                    get_moving_average,
                                    granularity_to_multipliers,
                                    initialise_dict, reward_type)


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


def _barplot_text_labels(ax, text_labels):
    if text_labels:
        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            label_height = height
            ax.text(rect.get_x() + rect.get_width() / 2, height,
                    '%.3f' % float(label_height),
                    ha='center', va='bottom', fontsize=9)

    return ax


def _barplot_baseline_opt_benchmarks(baseline, title, opt, ax, prm):
    if baseline is not None:
        drawbsl = 0 if title[0:7] == 'Average' else baseline
        ax.axhline(drawbsl, linestyle='--', color="k")
    if opt is not None:
        drawopt = opt - baseline if title[0:7] == 'Average' else opt
        ax.axhline(drawopt, linestyle='--',
                   color=prm['save']['colorse']['opt'])

    return ax


def _barplot(
        bars, eval_types, prm, baseline=None, opt=None, text_labels=True,
        colors=None, error=None, title=None, display_title=True,
        lower_bound=None, upper_bound=None, display_ticks=True,
        ax0=None, display_legend=False, ylabel=None, xlabel=None
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

    ax = _barplot_baseline_opt_benchmarks(baseline, title, opt, ax, prm)

    if title[0:7] == 'Average' \
            and lower_bound is not None \
            and upper_bound is not None:
        plt.ylim([lower_bound, upper_bound])

    ax = _barplot_text_labels(ax, text_labels)

    if not display_ticks:
        plt.gca().set_yticks([])

    plt.gca().set_xticks([])

    if title is not None and display_title:
        ax.set_title(title)

    if display_legend:
        if ax0 is None:
            fig.legend(
                loc='lower center', bbox_to_anchor=(0.57, -0.35),
                fancybox=True, ncol=2
            )
        else:
            plt.legend(
                loc='center right', bbox_to_anchor=(1.2, 0.5), fancybox=True
            )

    if ax0 is not None:
        return ax
    else:
        return fig


def _eval_entries_plot_colors(prm):
    eval_entries = prm['RL']['type_eval']
    if prm['ntw']['n'] == 1:
        eval_entries = [e for e in eval_entries
                        if distr_learning(e) == 'd'
                        or e in ['baseline', 'opt']]
        if len([e for e in eval_entries if len(e.split('_')) > 1]) == 0:
            eval_entries = [e for e in prm['RL']['type_eval']
                            if distr_learning(e) == 'c'
                            or e in ['baseline', 'opt']]

    eval_entries_plot = \
        [e for e in eval_entries if e != 'baseline'] \
        + ['baseline'] if 'baseline' in eval_entries else eval_entries
    type_plot = {}
    for e in eval_entries:
        type_plot[e] = '-'
    eval_entries_bars = [e for e in eval_entries_plot
                         if e not in ['baseline', 'opt']]
    eval_entries_distr = [
        e for e in eval_entries
        if len(e.split('_')) > 1 and distr_learning(e) == 'd'
    ]
    eval_entries_centr = [
        e for e in eval_entries
        if len(e.split('_')) > 1 and distr_learning(e) == 'c'
    ]
    other_eval_entries = [
        e for e in eval_entries if len(e.split("_")) == 1
    ]
    eval_entries_notCd = [
        e for e in eval_entries_plot
        if not (len(e.split('_')) > 1 and distr_learning(e) == 'Cd')
    ]
    eval_entries_plot_indiv = [e for e in eval_entries_plot
                               if len(e.split('_')) > 1
                               and distr_learning(e) not in ['Cc0', 'Cd0']]
    colors_barplot = [prm['save']['colorse'][e] for e in eval_entries_bars]
    base_entries = eval_entries_distr if len(eval_entries_distr) > 0 \
        else eval_entries_centr
    base_entries += other_eval_entries
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


def _get_repeat_data(repeat, all_methods_to_plot, root):
    # last = last epoch of each repeat
    last = np.load(root / 'record' / f'last_repeat{repeat}.npy',
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


def _get_bands_EV_availability(bEV, a):
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

    return bands_bEV


def _plot_ev_loads_and_availability(axs, xs, lEV, a, bands_bEV):
    ax = axs[2, 1]
    ax.step(xs[0:24], lEV[a][0:24], color='k', where='post')
    for band in bands_bEV:
        ax.axvspan(band[0], band[1], alpha=0.3, color='grey')
    ax.set_ylabel('EV load [kWh]')
    grey_patch = matplotlib.patches.Patch(
        alpha=0.3, color='grey', label='EV unavailable')
    ax.legend(handles=[grey_patch], fancybox=True)

    return axs


def _plot_indiv_agent_res_action(prm, ys, xs, lw_indiv, linestyles, t, ax):
    for action in range(prm['RL']['dim_actions']):
        ys_ = [0] + [
            ys[step][action]
            for step in range(prm['syst']['N'])
        ]
        ax.step(xs, ys_, where='post',
                label=f"t_action{action}",
                color=prm['save']['colorse'][t],
                lw=lw_indiv, linestyle=linestyles[action])

    return ax


def _plot_indiv_agent_res(
        prm, all_methods_to_plot, root, title_ylabel_dict,
        colors_non_methods, lw_indiv, labels, linestyles
):
    # Grid price / intensity
    # Heating E
    # Action variable
    # Indoor temperature
    # Total consumption
    # EV load / availability
    # Cumulative rewards
    # Battery level
    for repeat in range(prm['RL']['n_repeats']):
        last, cintensity_kg, methods_to_plot = \
            _get_repeat_data(repeat, all_methods_to_plot, root)

        # plot EV availability + EV cons on same plot
        lEV, bEV = [[last['batch'][a][e]
                     for a in range(prm['ntw']['n'])]
                    for e in ['loads_EV', 'avail_EV']]

        for a in range(min(prm['ntw']['n'],
                           prm['save']['max_n_profiles_plot'])):
            xs = range(len(lEV[a]))
            bands_bEV = _get_bands_EV_availability(bEV, a)

            fig, axs = plt.subplots(4, 2, figsize=(13, 13))

            # carbon intensity, wholesale price and grid cost coefficient
            _plot_grid_price(
                title_ylabel_dict, axs=axs, cintensity_kg=cintensity_kg,
                row=0, col=0, last=last,
                colors_non_methods=colors_non_methods, lw=lw_indiv)

            axs = _plot_ev_loads_and_availability(
                axs, xs, lEV, a, bands_bEV
            )

            # cum rewards
            _plot_cum_rewards(
                axs, last, methods_to_plot, labels, prm,
                row=3, col=0, lw=lw_indiv)

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
                    if e == 'action':
                        ax = _plot_indiv_agent_res_action(
                            prm, ys, xs, lw_indiv, linestyles, t, ax
                        )

                    else:
                        if e == 'store' and t == 'opt':
                            ys = ys + [prm['bat']['store0'][a]]
                        elif e == 'store':
                            ys = [prm['bat']['store0'][a]] + ys
                        else:
                            ys = [0] + ys
                        ax.step(xs, ys, where='post', label=t,
                                color=prm['save']['colorse'][t],
                                lw=lw_indiv)
                axs[r, c].set_ylabel(
                    f'{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}')
                axs[3, c].set_xlabel('Time [h]')

            fig.tight_layout()
            title = f'subplots example day repeat {repeat} a {a}'
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


def _plot_all_agents_all_repeats_res(
        list_repeat, all_methods_to_plot, root, title_ylabel_dict,
        axs, colors_non_methods, lw_indiv, labels,
        alpha_not_indiv, prm, lw_all, all_cum_rewards, all_T_air,
        rows, columns, entries, all_vals
):
    for repeat in list_repeat:
        last, cintensity_kg, methods_to_plot = _get_repeat_data(
            repeat, all_methods_to_plot, root)
        _plot_grid_price(
            title_ylabel_dict, axs=axs, cintensity_kg=cintensity_kg,
            row=0, col=0, last=last,
            colors_non_methods=colors_non_methods, lw=lw_indiv)

        cum_rewards_repeat = _plot_cum_rewards(
            axs, last, methods_to_plot, labels, prm, row=3,
            col=0, alpha=alpha_not_indiv, lw=lw_all,
            display_labels=False)
        for t in all_methods_to_plot:
            all_cum_rewards[t].append(cum_rewards_repeat[t])
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
                    all_vals[e][t][repeat].append(ys)
                axs[r, c].set_ylabel(
                    f"{title_ylabel_dict[e][0]} {title_ylabel_dict[e][1]}")
                if r == 2:
                    axs[r, c].set_xlabel('Time [h]')

    return axs, all_T_air


def _plot_last_epochs_actions(
        list_repeat, means, e, t, prm, all_vals, ax, xs, lw_mean, linestyles
):
    means[e][t] = []

    for action in range(prm['RL']['dim_actions']):
        all_vals_e_t_step_mean = np.zeros(prm['syst']['N'])
        for step in range(prm['syst']['N']):
            all_vals_e_t_step = np.array(
                [[all_vals[e][t][repeat][home][step][action]
                  for repeat in list_repeat]
                 for home in range(prm['ntw']['n'])]
            )
            if all(
                    [[all_vals[e][t][repeat][home][step][action] is None
                      for repeat in list_repeat]
                     for home in range(prm['ntw']['n'])]
            ):
                all_vals_e_t_step = None

            all_vals_e_t_step_mean[step] = None \
                if all_vals_e_t_step is None \
                else np.nanmean(
                np.nanmean(all_vals_e_t_step)
            )

        ax.step(
            xs, all_vals_e_t_step_mean,
            where='post', label=t,
            color=prm['save']['colorse'][t],
            lw=lw_mean, alpha=1, linestyle=linestyles[action]
        )
        means[e][t].append(all_vals_e_t_step)

    return ax


def _plot_all_agents_mean_res(
        entries, all_methods_to_plot, axs, all_T_air,
        prm, lw_mean, all_cum_rewards, labels,
        rows, columns, all_vals, list_repeat, linestyles
):
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
            if e == 'action':
                axs[r, c] = _plot_last_epochs_actions(
                    list_repeat, means, e, t, prm, all_vals,
                    axs[r, c], xs, lw_mean, linestyles
                )

            else:
                n = len(all_vals[e][t][list_repeat[0]][0])
                all_vals_e_t_step_mean = np.zeros(n)
                for step in range(n):
                    all_vals_e_t_step = np.array(
                        [[all_vals[e][t][repeat][home][step]
                          for repeat in list_repeat]
                         for home in range(prm['ntw']['n'])]
                    )
                    all_vals_e_t_step_mean[step] = np.mean(
                        np.mean(all_vals_e_t_step)
                    )
                axs[r, c].step(xs, all_vals_e_t_step_mean,
                               where='post', label=t,
                               color=prm['save']['colorse'][t],
                               lw=lw_mean, alpha=1)
                means[e][t] = all_vals_e_t_step_mean
    return axs


def _plot_all_agents_res(
        list_repeat, lw_all, prm, lw_all_list_repeat, all_methods_to_plot,
        root, title_ylabel_dict, colors_non_methods, labels,
        lw_indiv, alpha_not_indiv, lw_mean, linestyles
):
    # do one figure with all agents and repeats
    title_repeat = 'all_repeats' if list_repeat is None \
        else f'repeat_{list_repeat}'
    lw_all = lw_all if list_repeat is None else lw_all_list_repeat
    list_repeat = range(prm['RL']['n_repeats']) \
        if list_repeat is None else list_repeat
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
    all_vals = initialise_dict(
        entries, second_level_entries=all_methods_to_plot
    )
    for e in entries:
        for t in all_methods_to_plot:
            all_vals[e][t] = initialise_dict(range(prm['RL']['n_repeats']))

    axs, all_T_air = _plot_all_agents_all_repeats_res(
        list_repeat, all_methods_to_plot, root, title_ylabel_dict,
        axs, colors_non_methods, lw_indiv, labels,
        alpha_not_indiv, prm, lw_all, all_cum_rewards, all_T_air,
        rows, columns, entries, all_vals
    )
    axs = _plot_all_agents_mean_res(
        entries, all_methods_to_plot, axs, all_T_air,
        prm, lw_mean, all_cum_rewards, labels,
        rows, columns, all_vals, list_repeat, linestyles
    )

    fig.tight_layout()
    title = f'subplots example day all agents {title_repeat}'
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


def _plot_res(root, prm, indiv=True, list_repeat=None):
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
        'action': ['Action variable', r"$\psi$ [-]"]
    }
    linestyles = {
        0: '-',
        1: '--',
        2: ':'
    }

    lw_indiv = 2
    lw_all = 0.4
    lw_all_list_repeat = 1
    lw_mean = 2.5
    alpha_not_indiv = 0.15
    plt.rcParams['font.size'] = '16'
    colors_methods = [prm['save']['colorse'][t] for t in all_methods_to_plot]
    colors_non_methods = [c for c in prm['save']['colors']
                          if c not in colors_methods]
    prm['save']['colorse']['opt_n_c'] = prm['save']['colorse']['opt_n_d']
    labels = {}
    reward_labels = {
        'd': 'M',
        'r': 'T',
        'n': 'N',
        'A': 'A'
    }
    experience_labels = {'opt': 'O', 'env': 'E'}
    for t in prm['RL']['type_eval']:
        if t == 'opt':
            labels[t] = 'optimal'
        elif t == 'baseline':
            labels[t] = t
        else:
            label = reward_labels[reward_type(t)]
            label += experience_labels[data_source(t)]
            labels[t] = label

    if indiv:  # do one figure per agent and per repeat
        _plot_indiv_agent_res(
            prm, all_methods_to_plot, root, title_ylabel_dict,
            colors_non_methods, lw_indiv, labels, linestyles
        )
    elif not indiv:
        _plot_all_agents_res(
            list_repeat, lw_all, prm, lw_all_list_repeat, all_methods_to_plot,
            root, title_ylabel_dict, colors_non_methods, labels,
            lw_indiv, alpha_not_indiv, lw_mean, linestyles
        )


def _initialise_variables(prm, spaces, record, f):
    plt.rcParams['font.size'] = '11'
    action_state_space_0, state_space_0 = \
        [initialise_dict(range(prm['RL']['n_repeats']))
         for _ in range(2)]

    eval_rewards = record.eval_rewards      # [repeat][t][epoch]
    n_window = int(max(min(100, prm['RL']['n_all_epochs'] / 10), 2))

    spaces.new_state_space(prm['RL']['state_space'])
    q_tables, counters = record.q_tables, record.counter
    mean_eval_rewards = record.mean_eval_rewards  # [repeat][t][epoch]
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


def _plot_mova_eval_per_repeat(
        repeat, eval_rewards, prm, n_window, eval_entries_plot
):
    if not prm['save']['plot_indiv_repeats_rewards']:
        return
    fig = plt.figure()
    mova_baseline = get_moving_average(
        [eval_rewards[repeat]['baseline']
         for repeat in prm['RL']['n_repeats']], n_window)
    # 2 - moving average of all rewards evaluation
    for e in [e for e in eval_entries_plot if e != 'baseline']:
        mova_e = get_moving_average(
            [eval_rewards[repeat][e] for repeat in prm['RL']['n_repeats']],
            n_window)
        diff = [m - mb if m is not None else None
                for m, mb in zip(mova_e, mova_baseline)]
        plt.plot(diff, label=e, color=prm['save']['colorse'][e])
    plt.xlabel('episodes')
    plt.ylabel('reward difference rel. to baseline')
    title = f"Moving average all rewards minus baseline " \
            f"state comb {prm['RL']['statecomb_str']} repeat {repeat} " \
            f"n_window = {n_window}"
    _formatting_figure(
        fig=fig, title=title,
        fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run']
    )
    plt.close('all')


def _plot_epsilon(repeat, prm, record):
    if 'eps' in record.__dict__.keys() and record.eps != {}:
        if type(record.eps[repeat][0]) is dict:
            fig = plt.figure()
            for t in prm['RL']['eval_action_choice']:
                plt.plot([record.eps[repeat][e][t]
                          for e in range(prm['RL']['n_epochs'])],
                         label=t, color=prm['save']['colorse'][t])
            plt.xlabel('epoch')
            plt.ylabel('eps')
            title = "epsilon over epochs, state comb " \
                    "{prm['RL']['state_space']}, repeat {repeat}"
            _formatting_figure(
                fig=fig, title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'], legend=True,
                high_res=prm['save']['high_res']
            )
            plt.close('all')
        else:
            fig = plt.figure()
            plt.plot([record.eps[repeat][e]
                      for e in range(prm['RL']['n_epochs'])])
            plt.xlabel('epoch')
            plt.ylabel('eps')
            title = f"epsilon over epochs, state comb " \
                    f"{prm['RL']['state_space']}, repeat {repeat}"
            _formatting_figure(
                fig=fig, title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run']
            )
            plt.close('all')


def _plot_unique_state_best_psi(
        prm, eval_entries_plot_indiv, best_theta, index_to_val,
        possible_states, q, repeat
):
    for t in eval_entries_plot_indiv:
        na = prm['ntw']['n'] if distr_learning(t) in ['d', 'Cd'] else 1
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
            i_tables = [a if distr_learning(t) == 'd' else 0
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
                    f'repeat {repeat} a {a}'

            ax.set_ylabel(r'best $\theta$ [-]')
            _formatting_ticks(
                ax, fig, ys=eval_entries_plot_indiv,
                title=title,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'],
                display_title=False
            )


def _plot_1d_state_space_best_psi(
        prm, eval_entries_plot_indiv, best_theta, possible_states, repeat
):
    for a in range(prm['ntw']['n']):
        fig, ax = plt.subplots()
        theta_M = []
        for t in eval_entries_plot_indiv:
            index_a = a if distr_learning(t) in ['d', 'Cd'] else 0
            theta_M.append([best_theta[t][index_a][int(s)]
                            for s in range(possible_states)])
        im = ax.imshow(theta_M, vmin=0, vmax=1)
        title = f"best theta per method and per state state space " \
                f"{prm['RL']['statecomb_str']} repeat {repeat} a {a}"
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


def _plot_2d_state_space_best_psi(
        prm, eval_entries_plot_indiv, record, possible_states,
        spaces, best_theta, repeat
):
    for a in range(prm['ntw']['n']):
        for i_t in range(len(eval_entries_plot_indiv)):
            t = eval_entries_plot_indiv[i_t]
            M = np.zeros((record.granularity_state0,
                          record.granularity_state1))
            index_a = a if distr_learning(t) in ['d', 'Cd'] else 0
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
                    f"{prm['RL']['state_space']} repeat {repeat} a {a}"
            _formatting_ticks(
                ax, fig, ys=record.granularity_state1,
                xs=record.granularity_state0,
                title=title, im=im, grid=False,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'],
                display_title=False
            )


def _plot_best_psi(
        repeat, q_tables, prm, record, possible_states,
        index_to_val, eval_entries_plot_indiv, spaces
):
    if not (
            prm['RL']['type_learning'] == 'q_learning'
            and prm['ntw']['n'] < 4
    ):
        return

    best_theta = initialise_dict(eval_entries_plot_indiv)
    q = q_tables[repeat][prm['RL']['n_epochs'] - 1] \
        if record.save_qtables else q_tables[repeat]
    if prm['RL']['state_space'] == [None]:
        _plot_unique_state_best_psi(
            prm, eval_entries_plot_indiv, best_theta, index_to_val,
            possible_states, q, repeat
        )

    # if one dimensional state space - best theta[method, state]
    elif prm['RL']['dim_states'] == 1:
        _plot_1d_state_space_best_psi(
            prm, eval_entries_plot_indiv, best_theta, possible_states, repeat
        )

    # if two dimensional state space heatmap for each method
    # besttheta[state1, state2]
    elif prm['RL']['dim_states'] == 2:
        _plot_2d_state_space_best_psi(
            prm, eval_entries_plot_indiv, record, possible_states,
            spaces, best_theta, repeat
        )


def _plot_q_values(
        q_tables, repeat, prm, eval_entries_plot_indiv,
        index_to_val, q_entries
):
    if prm['RL']['type_learning'] != 'q_learning':
        return
    # plot all values in one figure if there is only one state
    if prm['RL']['state_space'] == [None] and prm['ntw']['n'] < 4:
        for a in range(prm['ntw']['n']):
            # plot heat map of value of each action for different methods
            # 2D array of best theta values
            M = np.zeros((len(eval_entries_plot_indiv), prm['RL']['n_action']))

            for i_t in range(len(eval_entries_plot_indiv)):
                t = eval_entries_plot_indiv[i_t]
                i_table = a if distr_learning(t) == 'd' else 0
                qvals = q_tables[repeat][prm['RL']['n_epochs'] - 1][t][i_table][
                    0]  # in final epoch, in 0-th (unique) state
                M[i_t, :] = qvals

            xs = q_entries
            b = index_to_val([0], typev='action')[0]
            m = index_to_val([1], typev='action')[0] - b
            ys = [m * i + b for i in range(prm['RL']['n_action'])]
            fig, ax = plt.subplots()
            im = ax.imshow(M)
            title = "qval per action per t state None repeat {repeat} a {a}"
            _formatting_ticks(
                ax, fig, xs=xs, ys=ys, title=title, im=im, grid=False,
                fig_folder=prm['paths']['fig_folder'],
                save_run=prm['save']['save_run'],
                high_res=prm['save']['high_res'], display_title=False)


def _plot_noisy_deterministic_inputs(prm, batch_entries, record, repeat):
    seeds = np.load(prm['paths']['seeds_path'])
    heatavail = {}
    for e in batch_entries:
        fig, axs = plt.subplots(prm['ntw']['n'], 1, squeeze=0)
        axs = axs.ravel()
        for a in range(prm['ntw']['n']):
            n = len(np.load(f'batch_{int(seeds[0] + 1)}_a0_lds.npy',
                            mmap_mode='c'))
            heatavail[a] = np.zeros((n,))
            for seed in record.seed[repeat]:
                str_seed = str(int(seed)) if seed > seeds[0] + 1 else \
                    f'deterministic_{record.ind_seed_deterministic[repeat]}'
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
        title = f"noisy repeat {repeat} {e}"
        _title_and_save(
            title, fig, prm['paths']['fig_folder'],
            prm['save']['save_run']
        )
        fig.show()
        fig.savefig(e)


def _plot_env_input(repeat, prm, record):
    if prm['RL']['deterministic'] is None \
            or prm['RL']['deterministic'] == 0 \
            or not prm['save']['plotting_batch']:
        return

    batch_entries = ['loads', 'gen', 'lds_EV', 'avail_EV']
    if prm['RL']['deterministic'] == 2:
        # 0 is indeterministic, 1 is deterministic, 2 is deterministic noisy
        _plot_noisy_deterministic_inputs(
            prm, batch_entries, record, repeat
        )

    else:
        if prm['RL']['deterministic'] == 1 \
                and os.path.exists(f'deterministic_parameters_repeat{repeat}.npy'):
            batch = np.load(f'deterministic_parameters_repeat{repeat}.npy',
                            allow_pickle=True)[-1]
        elif prm['RL']['deterministic'] == 0 and 'batch' in record.last[repeat]:
            # indeterministic, just plot the last epoch, evaluation step
            batch = record.last[repeat]['batch']['eval']
        for e in batch_entries:
            fig, axs = plt.subplots(prm['ntw']['n'], 1, squeeze=0)
            axs = axs.ravel()
            for a in range(prm['ntw']['n']):
                axs[a].plot(batch[a][e])
                axs[a].set_title('{a}')
            title = f"deterministic repeat {repeat} {e}"
            _title_and_save(
                title, fig, prm['paths']['fig_folder'],
                prm['save']['save_run']
            )
        else:
            print(f'no new deterministic batch for repeat = {repeat}')


def _video_visit_states(
        repeat, q_entries, counters, possible_states,
        prm, record, spaces
):
    if prm['RL']['type_learning'] != 'q_learning' \
            or not prm['save']['make_video'] \
            or len(counters[0]) == 0 \
            or prm['RL']['server'] \
            or prm['RL']['state_space'] == [None]:
        return

    import cv2
    counters = counters[repeat]
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
                plt.ylabel(rl['state_space'][1])

            elif rl['dim_states'] == 1:
                counters_per_state = \
                    np.zeros((record.granularity_state0, rl['n_action']))
                plt.ylabel(rl['state_space'][0])
                plt.xlabel(r'$\theta$ [-]')
                for s in range(possible_states):
                    for a in range(rl['n_action']):
                        counters_per_state[s, a] = counters[epoch][t][s][a]
            else:
                counters_per_state = [
                    sum(counters[epoch][t][s]) for s in range(possible_states)
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
                    prm['paths']['fig_folder'] / f'qlearn_repeat{repeat}_{t}.avi' \
                    if prm['paths']['fig_folder'] is not None \
                    else f'qlearn_repeat{repeat}_{t}.avi'
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


def _plot_final_explorations(
        repeat, prm, q_entries, action_state_space_0,
        state_space_0, record, counters
):
    if prm['RL']['type_learning'] != 'q_learning' \
            or len(counters[0]) == 0:
        return

    rl = prm['RL']
    for t in q_entries:
        na = prm['ntw']['n'] if reward_type(t) == '1' else 1
        for a in range(na):
            action_state_space_0[repeat][a], state_space_0[repeat][a] =\
                [initialise_dict(q_entries) for _ in range(2)]
            fig, ax = plt.subplots()
            counters_plot = counters[repeat][rl['n_epochs'] - 1][t][a] \
                if record.save_qtables else counters[repeat][t][a]
            im = ax.imshow(counters_plot, aspect='auto')
            fig.colorbar(im, ax=ax)
            title = f"Explorations per state action pair state space " \
                    f"{rl['statecomb_str']} repeat {repeat} " \
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
            state_space_0[repeat][a][t] = \
                sum_state_0 / rl['n_total_discrete_states']
            action_state_space_0[repeat][a][t] = \
                sum_action_0 / (rl['n_total_discrete_states']
                                * rl['n_discrete_actions'])


def _plot_unfeasible_attempts(repeat, record, prm):
    fig = plt.figure()
    plt.plot(record.n_not_feas[repeat])
    plt.xlabel('epoch')
    plt.ylabel('no of non feasible attempts before feasibility')
    title = f"no of non feasible attempts before feasibility " \
        f"repeat {repeat} state space {prm['RL']['statecomb_str']}"
    _title_and_save(
        title, fig, prm['paths']['fig_folder'], prm['save']['save_run']
    )
    plt.close('all')


def _plot_results_all_repeats(
        n_window, prm, eval_entries_plot, record,
        moving_average=True):
    fig = plt.figure(figsize=(4.600606418100001, 4.2))
    min_val, max_val = 100, - 100
    plt.rcParams['font.size'] = '10'

    for e in [e for e in eval_entries_plot if e != 'baseline']:
        results = np.array(
            [[None if record.mean_eval_rewards_per_hh[repeat][e][epoch] is None
              else record.mean_eval_rewards_per_hh[repeat][e][epoch]
              - record.mean_eval_rewards_per_hh[repeat]['baseline'][epoch]
              for epoch in range(prm['RL']['n_all_epochs'])]
             for repeat in range(prm['RL']['n_repeats'])])
        if moving_average:
            results = np.array(
                [get_moving_average(results[repeat], n_window, Nones=False)
                 for repeat in range(prm['RL']['n_repeats'])])
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

    lower_bound, upper_bound = [
        np.load(prm['paths']['open_inputs'] / f"{e}0.npy")
        for e in ['lower_bound', 'upper_bound']
    ]
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
    _formatting_figure(
        fig=fig, title=title,
        fig_folder=prm['paths']['fig_folder'],
        save_run=prm['save']['save_run'],
        high_res=prm['save']['high_res'],
        legend=False, display_title=False
    )

    return lower_bound, upper_bound


def _plot_compare_all_signs(
        base_entries, colors_barplot_baseentries, metrics, m_, ave,
        eval_entries_notCd, prm, lower_bound, upper_bound, m
):
    fig2 = plt.figure(figsize=(3.25, 7 * 0.75))
    ax = plt.gca()
    xs, colors_plot_end = {}, {}
    for i in range(len(base_entries)):
        splits = base_entries[i].split('_')
        label = f"{splits[0]}_{splits[1]}" if len(splits) > 1 \
            else base_entries[i]
        xs[label] = i
        colors_plot_end[label] = colors_barplot_baseentries[i]
    baseline, opt = [metrics[m_][ave][e]
                     if e in metrics[m_][ave] else None
                     for e in ['baseline', 'opt']]
    for e in eval_entries_notCd:
        label = data_source(e) + '_' + reward_type(e) \
            if len(e.split('_')) > 1 else e
        distr_learning_to_ls = {
            'd': 'o',
            'c': 'x',
            'Cc': '^'
        }
        if len(e.split('_')) < 2:
            ls = 'o'
        else:
            ls = distr_learning_to_ls[distr_learning(e)]
        legend = distr_learning(e) if len(e.split('_')) > 1 else e
        to_plot = metrics[m_][ave][e] - baseline \
            if m_ == 'end' else metrics[m_][ave][e]
        ax.plot(xs[label], to_plot, ls,
                color=colors_plot_end[label],
                markeredgewidth=2, markerfacecolor="None",
                label=legend, linewidth=2)
    maxval = {}
    for label in ['env_r', 'env_d', 'env_A', 'opt_r',
                  'opt_d', 'opt_A', 'opt_n']:
        if sum([label_[0: len(label)] == label
                for label_ in metrics[m_]['ave'].keys()]) > 0:
            maxval[label] = \
                max([metrics[m_][ave][e]
                     for e in metrics[m_][ave].keys()
                     if len(e.split('_')) > 1
                     and f"{data_source(e)}_{reward_type(e)}"
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
            eval_entries_bars_ = eval_entries_bars
            ave = 'ave'
            m_ = m
            if m[-3:] == 'p50':
                ave = 'p50'
                m_ = m[0:-4]

            colors_barplot_ = generate_colors(
                prm["save"], prm, colours_only=True, entries=eval_entries_bars_
            )

            bars, err = [
                [metrics[m_][s][e] for e in eval_entries_bars_]
                for s in [ave, 'std']
            ]

            baseline, opt = [
                metrics[m_][ave][e] if e in eval_entries_plot else None
                for e in ['baseline', 'opt']
            ]

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
            _plot_compare_all_signs(
                base_entries, colors_barplot_baseentries, metrics, m_, ave,
                eval_entries_notCd, prm, lower_bound, upper_bound, m
            )

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
            i_reward, i_distr = \
                [
                    [i for i in range(len(arr)) if arr[i] == x][0]
                    for arr, x in zip(
                        [rewards, distrs], [reward_type(e), distr_learning(e)]
                    )
                ]
            val = metrics['end']['ave'][e] - metrics['end']['ave']['baseline']
            M[data_source(e)][i_distr][i_reward] = val
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
                np.mean([[(record_obj[repeat]['baseline'][epoch]
                           - record_obj[repeat][t][epoch])
                          * mult / (prm['ntw']['n'] + prm['ntw']['nP'])
                          for epoch in range(prm['RL']['start_end_eval'],
                                             len(record_obj[repeat][t]))]
                         for repeat in range(prm['RL']['n_repeats'])]))
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
                    np.mean([[(reward[repeat]['baseline'][epoch][a]
                               - reward[repeat][t][epoch])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 prm['RL']['n_epochs'])]
                             for repeat in range(prm['RL']['n_repeats'])])
                    for reward in [record.__dict__['indiv_sc'],
                                   record.__dict__['indiv_gc']]]
                share_sc[a].append(
                    savings_sc_a / (savings_sc_a + savings_gc_a))
                savings_a_all = \
                    [[(record.__dict__['indiv_c'][repeat]['baseline'][epoch][a]
                       - record.__dict__['indiv_c'][repeat][t][epoch])
                      for epoch in range(prm['RL']['start_end_eval'],
                                         prm['RL']['n_epochs'])]
                     for repeat in range(prm['RL']['n_repeats'])]
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
        for repeat in range(prm['RL']['n_repeats']):
            rewards_t, rewards_bsl = \
                [mean_eval_rewards_per_hh[repeat][type_eval][
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
    _title_and_save(
        title, fig, prm["paths"]["fig_folder"],
        prm["save"]["save_run"]
    )


def _plot_eval_action(record, prm):
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


def _check_model_changes(prm):
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
    red = (234 / 255, 53 / 255, 37 / 255)
    prm['save']['colorse']['env_r_d'] = red
    prm['save']['colorse']['env_r_d'] = red
    prm['save']['colorse']['env_r_c'] = red
    prm['save']['colorse']['opt'] = 'grey'
    green = prm['save']['colorse']['env_d_d']
    prm['save']['colorse']['opt_d_d'] = green

    record.get_mean_rewards(
        prm, action_state_space_0,
        state_space_0, eval_entries_plot
    )
    metrics, metrics_entries = record.get_metrics(
        prm, eval_entries_plot)

    mean_eval_rewards_per_hh = record.mean_eval_rewards_per_hh

    # 1 - plot non moving  average results 25th, 50th,
    # 75th percentile for all repeat
    _plot_results_all_repeats(
        n_window, prm, eval_entries_plot, record,
        moving_average=False)

    # 2 - plot moving average results 25th, 50th, 75th percentile for all repeat
    lower_bound, upper_bound = _plot_results_all_repeats(
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
        _plot_res(root, prm, indiv=False)
        _plot_res(root, prm, indiv=True)
        for repeat in range(prm['RL']['n_repeats']):
            _plot_res(root, prm, indiv=False, list_repeat=[repeat])

    # other repeat-specific plots:
    for repeat in range(prm['RL']['n_repeats']):
        if prm['save']['plot_type'] > 0:
            # 10 - plot moving average of all evaluation rewards
            # for each repeat
            _plot_mova_eval_per_repeat(
                repeat, eval_rewards, prm, n_window, eval_entries_plot)
        if prm['save']['plot_type'] > 1:
            # 11 - plot epsilon over time for each repeat
            _plot_epsilon(repeat, prm, record)

            # 12 - plot best psi action value per state
            _plot_best_psi(
                repeat, q_tables, prm, record, possible_states,
                spaces.index_to_val, eval_entries_plot_indiv, spaces)

            # 13 - plot q values
            _plot_q_values(
                q_tables, repeat, prm, eval_entries_plot_indiv,
                spaces.index_to_val, q_entries
            )
            # 14 - plot environment input
            _plot_env_input(repeat, prm, record)

            # 15 - make video of visits to each state
            _video_visit_states(
                repeat, q_entries, counters, possible_states,
                prm, record, spaces)

            # 16 -  final number of exploration of actions in each state
            _plot_final_explorations(
                repeat, prm, q_entries, action_state_space_0,
                state_space_0, record, counters)

            # 17 - n not feas vs variables vs time step
            _plot_unfeasible_attempts(repeat, record, prm)

    # 18 - plot eval_actions over time
    _plot_eval_action(record, prm)

    # 19 - check that some learning has occurred
    _check_model_changes(prm)

    if prm['save']['save_run']:
        np.save(prm['paths']['fig_folder'] / 'eval_entries', eval_entries)
        np.save(prm['paths']['fig_folder'] / 'state_space_0', state_space_0)
        np.save(prm['paths']['fig_folder'] / 'action_state_space_0',
                action_state_space_0)
        np.save(prm['paths']['fig_folder'] / 'metrics', metrics)
    plt.close('all')

    return f, metrics
