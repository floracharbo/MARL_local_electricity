#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:03:35 2020.

@author: floracharbonnier.

Plotting results RL
"""
import matplotlib.pyplot as plt

from src.post_analysis.plotting.check_learning_over_time import (
    check_model_changes, plot_eval_action)
from src.post_analysis.plotting.initialise_plotting_variables import \
    initialise_variables
from src.post_analysis.plotting.plot_break_down_savings import (
    barplot_breakdown_savings, barplot_grid_energy_costs,
    barplot_indiv_savings, distribution_savings, heatmap_savings_per_method,
    plot_voltage_statistics)
from src.post_analysis.plotting.plot_episode_results import (
    map_over_undervoltage, plot_env_input, plot_imp_exp_check,
    plot_imp_exp_violations, plot_indiv_reactive_power, plot_reactive_power,
    plot_res, plot_voltage_violations, voltage_penalty_per_bus)
from src.post_analysis.plotting.plot_moving_average_rewards import (
    plot_mova_eval_per_repeat, plot_results_all_repeats)
from src.post_analysis.plotting.plot_q_learning_explorations_values import (
    plot_best_actions, plot_final_explorations, plot_q_values,
    video_visit_states)
from src.post_analysis.plotting.plot_rl_performance_metrics import \
    barplot_metrics
from src.post_analysis.plotting.plotting_utils import (formatting_figure,
                                                       title_and_save)

HIGH_RES = True


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
    if HIGH_RES:
        prm['save']['high_res'] = True

    # 1 - plot non-moving  average results 25th, 50th,
    # 75th percentile for all repeat
    diff_to_opt = False
    if sum(1 for method in prm['RL']['evaluation_methods'] if method != 'baseline') > 0:
        for moving_average in [False, True]:
            # for diff_to_opt in [False, True]:
            lower_bound, upper_bound = plot_results_all_repeats(
                prm, record,
                moving_average=moving_average,
                diff_to_opt=diff_to_opt
            )

        # 2 - bar plot metrics
        barplot_metrics(prm, lower_bound, upper_bound)

        if prm['save']['plot_type'] > 0:
            # 3 - plot distribution of daily savings
            distribution_savings(prm, aggregate='daily')
            distribution_savings(prm, aggregate='test_period')

            # 4 - heat map of reductions rel to baseline per data source,
            # reward ref and MARL structure
            heatmap_savings_per_method(prm)

            # 5 - do bar plot of all costs reduction rel to baseline,
            barplot_breakdown_savings(record, prm, plot_type='savings')

            # 6 - do bar plot of all costs reduction rel to baseline,
            barplot_breakdown_savings(record, prm, plot_type='costs')

            # 7 - plot individual savings as well as share battery
            # vs energy costs in individual savings
            barplot_indiv_savings(record, prm)

    # 8 - plotting results example day household variables
    if prm['save']['plot_profiles']:
        plot_res(prm, indiv=False)
        plot_res(prm, indiv=True)
        for repeat in range(prm['RL']['n_repeats']):
            plot_res(prm, indiv=False, list_repeat=[repeat])
            plot_res(prm, indiv=False, list_repeat=[repeat], sum_agents=True)

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

    # 19 - grid import and export and corresponding limit violations
    all_methods_to_plot = prm['RL']['evaluation_methods']
    folder_run = prm["paths"]["folder_run"]
    # 20 - plot the aggregated hourly import and export and the limits
    if prm['grd']['manage_agg_power']:
        plot_imp_exp_violations(
            prm, all_methods_to_plot, folder_run)
        if not prm['grd']['manage_voltage']:
            barplot_breakdown_savings(record, prm, plot_type='costs')
    # 21 - (Sanity Check) plot grid = grid_in - grid_out
    if prm['save']['plot_imp_exp_check']:
        plot_imp_exp_check(
            prm, all_methods_to_plot, folder_run)

    # 21 - over- and undervoltage
    if prm['grd']['manage_voltage']:
        map_over_undervoltage(
            prm, all_methods_to_plot, folder_run, net=prm["grd"]["net"]
        )
        plot_voltage_violations(
            prm, all_methods_to_plot, folder_run)
        barplot_breakdown_savings(record, prm, plot_type='costs')
        barplot_grid_energy_costs(record, prm, plot_type='costs')
        plot_voltage_statistics(record, prm)
        voltage_penalty_per_bus(prm, all_methods_to_plot, folder_run)
        plot_reactive_power(prm, all_methods_to_plot, folder_run)
        plot_indiv_reactive_power(prm, all_methods_to_plot, folder_run)

    barplot_breakdown_savings(record, prm, plot_type='costs')

    # 22 - check that some learning has occurred
    check_model_changes(prm)

    plt.close('all')

    return f, prm["RL"]["metrics"]
