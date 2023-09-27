import matplotlib.pyplot as plt
import numpy as np

import src.environment.utilities.userdeftools as utils
from src.environment.post_analysis.plotting.plotting_utils import (
    formatting_figure, title_and_save)


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
                   color=prm['save']['colourse']['opt'])

    return ax


def _barplot(
        bars, evaluation_methods, prm, baseline=None, opt=None, text_labels=True,
        colours=None, error=None, title=None, display_title=True,
        lower_bound=None, upper_bound=None, display_ticks=True,
        ax0=None, display_legend=False, ylabel=None, xlabel=None, annotate=False
):

    n_evaluation_methods = len(evaluation_methods)
    barWidth = 1 / (n_evaluation_methods + 1)
    rs = [x * 1 / len(bars) + (1 / len(bars) / 2) for x in range(len(bars))]

    if ax0 is None:
        fig, ax = plt.subplots(figsize=(3.25, 7 * 0.75))
    else:
        ax = ax0

    for ind_e in range(n_evaluation_methods):
        rsir = rs[ind_e]
        err = None if error is None or not isinstance(error[ind_e], float) else error[ind_e]
        asym_err = None if error is None or isinstance(error[ind_e], float) else error[ind_e]
        barplot = bars[ind_e]
        barplot = barplot - baseline if title[0:7] == 'Average' else barplot

        if asym_err is not None:
            ax.errorbar(
                rsir, barplot, yerr=np.reshape(asym_err, (2, 1)),
                fmt='.', ecolor=colours[ind_e]
            )
        else:
            ax.bar(
                rsir, barplot, yerr=err, capsize=10, width=barWidth,
                edgecolor='white',
                label=evaluation_methods[ind_e],
                color=colours[ind_e],
                hatch='//' if evaluation_methods[ind_e][0:3] == 'opt' else None
            )

    if prm is not None:
        ax = _barplot_baseline_opt_benchmarks(baseline, title, opt, ax, prm)

    if title[0:7] == 'Average' \
            and lower_bound is not None \
            and upper_bound is not None:
        plt.ylim([lower_bound, upper_bound])
    if annotate:
        ax = _barplot_text_labels(ax, text_labels)

    if not display_ticks:
        ax.set_yticks([])

    ax.set_xticks([])

    if title is not None and display_title:
        ax.set_title(title)

    if display_legend:
        if ax0 is None:
            fig.legend(
                loc='lower center', bbox_to_anchor=(0.57, -0.35),
                fancybox=True, ncol=2
            )
        else:
            ax0.legend(
                loc='center right', bbox_to_anchor=(1.2, 0.5), fancybox=True
            )

    if ax0 is not None:
        return ax
    else:
        return fig


def compute_lists_barplot_colours(prm):
    eval_entries_bars = [
        e for e in prm["save"]["eval_entries_plot"]
        if e not in ['baseline', 'opt']
    ]
    eval_entries_notCd = [
        e for e in prm["save"]["eval_entries_plot"]
        if not (len(e.split('_')) > 1 and utils.distr_learning(e) == 'Cd')
    ]
    colours_barplot = [
        prm['save']['colourse'][e] for e in eval_entries_bars
    ]
    colours_barplot_baseentries = [
        prm['save']['colourse'][e] for e in prm["save"]["base_entries"]
    ]
    return eval_entries_bars, eval_entries_notCd, colours_barplot, colours_barplot_baseentries


def barplot_metrics(prm, lower_bound, upper_bound):
    metrics = prm["RL"]["metrics"]
    eval_entries_bars, eval_entries_notCd, colours_barplot, colours_barplot_baseentries \
        = compute_lists_barplot_colours(prm)

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
    ylims = {'end_test_bl': [-80, 80],
                    'LRT': [0, 300],
                    'DR': [0, 45],
                    'RR': [-320, 0],
                    'SRT': [-200, 0],
                    'DT': [0, 160]
                    }
    for plot_type in ['indivs', 'subplots']:
        if plot_type == 'subplots':
            fig, axs = plt.subplots(2, 3)
        all_metrics_entries \
            = prm["RL"]["metrics_entries"] \
            + [m + '_p50' for m in prm["RL"]["metrics_entries"][0:4]]
        for m in all_metrics_entries:
            eval_entries_bars_ = eval_entries_bars
            ext = 'd' if prm['RL']['type_learning'] == 'q_learning' else 'c'
            if len(eval_entries_bars_) > 3:
                eval_entries_bars_ = [
                    # entry for entry in ['env_r_c', 'opt_d_d']
                    # if entry in eval_entries_bars_
                    entry for entry in eval_entries_bars_
                    if entry[-1] == ext and entry.split('_')[1] != 'n'
                ]
            ave = 'ave'
            m_ = m
            if m[-3:] == 'p50':
                ave = 'p50'
                m_ = m[0:-4]

            # colours_barplot_ = generate_colours(
            #     prm["save"], prm, colours_only=True, entries=eval_entries_bars_
            # )
            colours_barplot_ = [prm['save']['colourse'][e] for e in eval_entries_bars_]

            bars, err = [
                [metrics[m_][s][e] for e in eval_entries_bars_]
                for s in [ave, 'std']
            ]

            baseline, opt = [
                metrics[m_][ave][e] if e in prm["save"]["eval_entries_plot"] else None
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
                    opt=opt,
                    colours=colours_barplot_,
                    error=err,
                    title=titles[m_], display_legend=display_legend,
                    display_title=display_title, lower_bound=lb,
                    upper_bound=ub, display_ticks=display_ticks)
                formatting_figure(
                    prm, fig=fig, title=m,
                    legend=False,
                    display_title=False,
                    title_display=titles[m_]
                )

            elif plot_type == 'subplots':
                if m in subplots_i_j:
                    i, j = subplots_i_j[m]
                    # if i > 0 or j > 0:
                    display_legend = False
                    annotate = False
                    title = 'Savings' if m == 'end_test_bl' else m
                    axs[i, j] = _barplot(
                        bars, eval_entries_bars_, prm,
                        baseline=baseline, opt=opt,
                        colours=colours_barplot_, error=err,
                        title=title, display_legend=display_legend,
                        display_title=display_title, lower_bound=lb,
                        upper_bound=ub, display_ticks=display_ticks,
                        ax0=axs[i, j], annotate=annotate
                    )
                    axs[i, j].set_ylim(ylims[m])
                    if j == 0:
                        axs[i, j].set_ylabel('£/home/h')

            # do end for with signs for each
            _plot_compare_all_signs(
                prm, eval_entries_notCd, m_, ave, lower_bound, upper_bound, m
            )

    formatting_figure(
        prm, fig=fig, title='subplots_all_metrics',
        legend=False,
        display_title=False
    )


def _plot_compare_all_signs(
        prm, eval_entries_notCd, m_, ave, lower_bound, upper_bound, m
):
    metrics = prm["RL"]["metrics"]
    fig2 = plt.figure(figsize=(3.25, 7 * 0.75))
    ax = plt.gca()
    xs, colours_plot_end = {}, {}

    for i, entry in enumerate(eval_entries_notCd):
        label = utils.data_source(entry) + '_' + utils.reward_type(entry) \
            if len(entry.split('_')) > 1 else entry
        xs[label] = i
        colours_plot_end[label] = prm['save']['colourse'][entry]

    baseline, opt = [
        metrics[m_][ave][e]
        if e in metrics[m_][ave] else None
        for e in ['baseline', 'opt']
    ]
    for e in eval_entries_notCd:
        label = utils.data_source(e) + '_' + utils.reward_type(e) \
            if len(e.split('_')) > 1 else e
        utils.distr_learning_to_ls = {
            'd': 'o',
            'c': 'x',
            'Cc': '^'
        }
        if len(e.split('_')) < 2:
            ls = 'o'
        else:
            ls = utils.distr_learning_to_ls[utils.distr_learning(e)]
        legend = utils.distr_learning(e) if len(e.split('_')) > 1 else e
        to_plot = metrics[m_][ave][e] - baseline \
            if m_ == 'end' else metrics[m_][ave][e]
        ax.plot(xs[label], to_plot, ls,
                color=colours_plot_end[label],
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
                     and f"{utils.data_source(e)}_{utils.reward_type(e)}"
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
                   color=prm['save']['colourse']['opt'])
    title = 'compare all median 25-75th percentile' + m
    if m_ == 'end':
        plt.ylim([lower_bound, upper_bound])

    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    title_and_save(title, fig2, prm, display_title=False)
