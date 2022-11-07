#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:03:35 2020.

@author: floracharbonnier.

Plotting results RL
"""
import matplotlib.pyplot as plt

from src.post_analysis.plotting.check_learning_over_time import \
    plot_eval_action
from src.post_analysis.plotting.initialise_plotting_variables import \
    initialise_variables
from src.post_analysis.plotting.plot_break_down_savings import (
    barplot_breakdown_savings, barplot_indiv_savings, distribution_savings,
    heatmap_savings_per_method)
from src.post_analysis.plotting.plot_episode_results import (plot_env_input,
                                                             plot_res)
from src.post_analysis.plotting.plot_moving_average_rewards import (
    plot_mova_eval_per_repeat, plot_results_all_repeats)
from src.post_analysis.plotting.plot_q_learning_explorations_values import (
    plot_best_actions, plot_final_explorations, plot_q_values,
    video_visit_states)
from src.post_analysis.plotting.plot_rl_performance_metrics import \
    barplot_metrics
from src.post_analysis.plotting.plotting_utils import (formatting_figure,
                                                       title_and_save)


def _plot_epsilon(repeat, prm, record):
    if 'eps' in record.__dict__.keys() and record.eps != {}:
        if isinstance(record.eps[repeat][0], dict):
            fig = plt.figure()
            for method in prm['RL']['eval_action_choice']:
                plt.plot(
                    [
                        record.eps[repeat][e][method]
                        for e in range(prm['RL']['n_epochs'])
                    ],
                    label=method, color=prm['save']['colourse'][method]
                )
            plt.xlabel('epoch')
            plt.ylabel('eps')
            title = "epsilon over epochs, state comb " \
                    "{prm['RL']['state_space']}, repeat {repeat}"
            formatting_figure(
                prm, fig=fig, title=title, legend=True
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
            formatting_figure(prm, fig=fig, title=title)
            plt.close('all')


def _plot_unfeasible_attempts(repeat, record, prm):
    fig = plt.figure()
    plt.plot(record.n_not_feas[repeat])
    plt.xlabel('epoch')
    plt.ylabel('no of non feasible attempts before feasibility')
    title = f"no of non feasible attempts before feasibility " \
        f"repeat {repeat} state space {prm['RL']['statecomb_str']}"
    title_and_save(title, fig, prm)
    plt.close('all')


def plotting(record, spaces, prm, f):
    """Plot and save results."""
    prm = initialise_variables(prm, spaces, record)

    # 1 - plot non-moving  average results 25th, 50th,
    # 75th percentile for all repeat
    for moving_average in [False, True]:
        for diff_to_opt in [False, True]:
            lower_bound, upper_bound = plot_results_all_repeats(
                prm, record,
                moving_average=moving_average,
                diff_to_opt=diff_to_opt
            )

    # 3 - bar plot metrics
    f = barplot_metrics(prm, lower_bound, upper_bound, f)

    # 4 - plot distribution of daily savings
    distribution_savings(prm, aggregate='daily')
    distribution_savings(prm, aggregate='test_period')

    if prm['save']['plot_type'] > 0:
        # 4 - heat map of reductions rel to baseline per data source,
        # reward ref and MARL structure
        heatmap_savings_per_method(prm)

        # 5 - do bar plot of all costs reduction rel to baseline,
        barplot_breakdown_savings(record, prm)

        # 6 - plot individual savings as well as share battery
        # vs energy costs in individual savings
        barplot_indiv_savings(record, prm)

    # 7 - plotting results example day household variables
    if prm['save']['plot_profiles']:
        plot_res(prm, indiv=False)
        plot_res(prm, indiv=True)
        for repeat in range(prm['RL']['n_repeats']):
            plot_res(prm, indiv=False, list_repeat=[repeat])

    # other repeat-specific plots:
    for repeat in range(prm['RL']['n_repeats']):
        if prm['save']['plot_type'] > 0:
            # 10 - plot moving average of all evaluation rewards for each repeat
            plot_mova_eval_per_repeat(repeat, prm)
        if prm['save']['plot_type'] > 1:
            # 11 - plot epsilon over time for each repeat
            _plot_epsilon(repeat, prm, record)
            # 12 - plot best action value per state (q learning)
            plot_best_actions(
                repeat, prm, record, spaces
            )
            # 13 - plot q values
            plot_q_values(repeat, spaces.index_to_val, prm)

            # 14 - make video of visits to each state (q learning)
            video_visit_states(repeat, record, spaces, prm)

            # 15 -  final number of exploration of actions in each state
            # (q learning)
            plot_final_explorations(repeat, record, prm)

            # 16 - plot environment input
            plot_env_input(repeat, prm, record)

            # 17 - n not feas vs variables vs time step
            _plot_unfeasible_attempts(repeat, record, prm)

    # 18 - plot eval_actions over time
    if prm['save']['plot_type'] > 0:
        plot_eval_action(record, prm)

    plt.close('all')

    return f, prm["RL"]["metrics"]
