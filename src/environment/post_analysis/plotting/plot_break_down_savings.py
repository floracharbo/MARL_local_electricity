import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

import src.environment.utilities.userdeftools as utils
from src.environment.post_analysis.plotting.plotting_utils import (
    formatting_figure, title_and_save)


def distribution_savings(prm, aggregate='daily'):
    rl = prm["RL"]
    fig = plt.figure()
    test_savings = {}
    for method in [method for method in rl["evaluation_methods"] if method != 'baseline']:
        if np.nansum(rl["monthly_mean_eval_rewards_per_home"][method]) != 0:
            test_savings[method] = []
            for repeat in range(prm['RL']['n_repeats']):
                rewards_t, rewards_bsl = [
                    rl["monthly_mean_eval_rewards_per_home"][evaluation_method][repeat][
                     prm['RL']['n_epochs']:
                    ] for evaluation_method in [method, 'baseline']
                ]
                savings_rel_baseline = \
                    [reward - baseline for reward, baseline
                     in zip(rewards_t, rewards_bsl)]
                if aggregate == 'daily':
                    test_savings[method] += savings_rel_baseline
                elif aggregate == 'test_period':
                    test_savings[method] += [np.mean(savings_rel_baseline)]
            plt.hist(
                test_savings[method], alpha=0.5, label=method,
                color=prm['save']['colourse'][method]
            )

    plt.legend()
    plt.xlabel("Average test savings [£/h/home]")
    plt.ylabel("Count")
    title = 'distribution of daily savings per household and learning type'
    if aggregate == 'daily':
        title += ' for each day of testing'
    elif aggregate == 'test_period':
        title += f" over each testing period of {prm['RL']['n_end_test']} days"
    formatting_figure(
        prm, fig=fig, title=title, legend=True
    )


def heatmap_savings_per_method(prm):
    metrics = prm["RL"]["metrics"]

    # do heat maps of reduction rel to baseline
    M = {}
    rewards = ['r', 'd', 'A', 'n']
    distrs = ['c', 'd', 'Cc', 'Cd']
    M['opt'] = np.full((len(distrs), len(rewards)), np.nan)
    M['env'] = np.full((len(distrs), len(rewards)), np.nan)
    all_vals = []
    for e in metrics['end']['ave'].keys():
        if e not in ['opt', 'baseline', 'random']:
            i_reward, i_distr = \
                [
                    [i for i in range(len(arr)) if arr[i] == x][0]
                    for arr, x in zip(
                        [rewards, distrs], [utils.reward_type(e), utils.distr_learning(e)]
                    )
                ]
            val = metrics['end']['ave'][e] - metrics['end']['ave']['baseline']
            M[utils.data_source(e)][i_distr][i_reward] = val
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
            title_and_save(title, fig, prm)
        np.save(prm['paths']['fig_folder'] / 'M_reductions', M)


def barplot_breakdown_savings(record, prm, plot_type='savings'):
    """ Creates barplot of monthly average cost or savings distribution """
    labels = [
        record.break_down_rewards_entries[prm['syst']['break_down_rewards_entries'].index(label)]
        for label in [
            'grid_energy_costs', 'emissions', 'battery_degradation_costs',
            'distribution_network_export_costs',
            'import_export_costs', 'voltage_costs', 'total_costs'
        ]
    ]
    bars = [[] for _ in range(len(labels))]
    shares_reduc = {}
    tots = {}
    for method in prm['RL']['evaluation_methods']:
        if method != 'baseline':
            shares_reduc[method] = []
            for i, label in enumerate(labels):
                record_obj = getattr(record, label)
                mult = prm['syst']['co2tax'] if label == 'emissions' else 1
                if plot_type == 'savings':
                    bars[i].append(
                        np.mean([[(record_obj[repeat]['baseline'][epoch]
                                - record_obj[repeat][method][epoch])
                                * mult / prm['syst']['n_homes_all_test']
                                * prm['syst']['interval_to_month']
                                for epoch in range(prm['RL']['start_end_eval'],
                                                   len(record_obj[repeat][method]))]
                                for repeat in range(prm['RL']['n_repeats'])]))
                else:
                    bars[i].append(
                        np.mean([[(record_obj[repeat][method][epoch])
                                * prm['syst']['interval_to_month']
                                * mult / prm['syst']['n_homes_all_test']
                                for epoch in range(prm['RL']['start_end_eval'],
                                                   len(record_obj[repeat][method]))]
                                for repeat in range(prm['RL']['n_repeats'])])
                    )
            tots[method] = bars[-1][-1]
            shares_reduc[method].append(
                [bars[i][-1] / tots[method] for i in range(len(labels))]
            )
    if 'env_r_c' in shares_reduc:
        print(f"shares_reduc['env_r_c'] {shares_reduc['env_r_c']}")
    if plot_type == 'costs':
        print(f"labels {labels} bars {bars}")
    new_labels = [
        'Grid costs', 'Emissions', 'Storage costs', 'Distribution Costs', 'Import/Export Penalty',
        'Voltage Penalty', 'Total Costs'
    ]
    bars.append(list(tots.values()))
    barWidth = 1 / (len(new_labels) + 1)
    rs = []
    rs.append(np.arange(len(prm['RL']['evaluation_methods']) - 1))
    print(f"prm['RL']['evaluation_methods'] {prm['RL']['evaluation_methods']}")
    for ir in range(len(new_labels)):
        rs.append([x + barWidth for x in rs[ir]])
    plt.figure(figsize=(16, 8))
    for ir in range(len(new_labels)):
        print(f"{new_labels[ir]}: {bars[ir]}")
        plt.bar(rs[ir], bars[ir], width=barWidth, label=new_labels[ir])
    plt.xlabel('Evaluation Method')
    plt.ylabel('Costs [£/month/home]')
    method_list = []
    for method in prm['RL']['evaluation_methods']:
        if method != 'baseline':
            method_list.append(method)
    plt.xticks(
        [r + barWidth for r in range(len(bars[0]))],
        method_list,
        rotation=45
    )
    plt.legend()
    plt.tight_layout()
    if plot_type == 'savings':
        plt.title('savings relative to baseline costs / emissions')
    else:
        plt.title('Monthly average of costs distribution per household')
    plt.savefig(
        f"{prm['paths']['fig_folder']}/mean_{plot_type}_bar_plots.png",
        bbox_inches='tight'
    )
    plt.close('all')


def barplot_grid_energy_costs(record, prm, plot_type='savings'):
    """ Creates a barplot of monthly average energy costs distribution """
    # all break down rewards except for the last three ones
    # which are individual values
    plt.rcParams['font.size'] = '16'
    labels = [
        'grid_energy_costs',
        'cost_distribution_network_losses',
        'costs_wholesale',
        'costs_upstream_losses',
        'voltage_costs'
    ]
    bars = [[] for _ in range(len(labels))]
    shares_reduc = {}
    tots = {}
    for method in prm['RL']['evaluation_methods']:
        shares_reduc[method] = []
        for i, label in enumerate(labels):
            record_obj = getattr(record, label)
            mult = prm['syst']['co2tax'] if label == 'emissions' else 1
            mult *= prm['syst']['interval_to_month']
            if plot_type == 'savings':
                bars[i].append(
                    np.mean([[(record_obj[repeat]['baseline'][epoch]
                               - record_obj[repeat][method][epoch])
                              * mult / prm['syst']['n_homes_all_test']
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 len(record_obj[repeat][method]))]
                             for repeat in range(prm['RL']['n_repeats'])]))
            else:
                bars[i].append(
                    np.mean([[(record_obj[repeat][method][epoch])
                              * mult / prm['syst']['n_homes_all_test']
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 len(record_obj[repeat][method]))]
                             for repeat in range(prm['RL']['n_repeats'])]))
        tots[method] = sum(
            bars[i][-1] for i, label in enumerate(labels)
            if label in [
                'distribution_network_export_costs',
                'battery_degradation_costs',
                'grid_energy_costs',
                'voltage_costs'
            ]
        )
        shares_reduc[method].append(
            [bars[i][-1] / tots[method] if tots[method] > 0 else None
             for i in range(len(labels))]
        )

    barWidth = 1 / (len(labels) + 1)
    new_labels = ['Grid costs', 'Costs of distribution network losses',
                  'Wholesale costs', 'Costs of upstream losses', 'Voltage costs']
    rs = []
    rs.append(np.arange(len(prm['RL']['evaluation_methods'])))
    for ir in range(len(new_labels) - 1):
        rs.append([x + barWidth for x in rs[ir]])
    plt.figure(figsize=(10, 8))
    for ir in range(len(new_labels)):
        plt.bar(rs[ir], bars[ir], width=barWidth, label=new_labels[ir])
    plt.xlabel('Evaluation Method')
    plt.ylabel('Costs [£/month/home]')
    plt.xticks(
        [r + barWidth for r in range(len(bars[0]))],
        prm['RL']['evaluation_methods'],
        rotation=45
    )
    plt.legend()
    plt.tight_layout()
    if plot_type == 'savings':
        plt.title('savings relative to baseline costs / emissions')
    else:
        plt.title('Monthly average of grid energy costs distribution per household')
    plt.savefig(
        f"{prm['paths']['fig_folder']}/mean_grid_{plot_type}_bar_plots.png", bbox_inches='tight')
    plt.close('all')


def savings_are_individual(indiv_grid_battery_costs):
    if len(np.shape(indiv_grid_battery_costs[0]['baseline'][0])) == 0:
        indiv_savings = True \
            if not np.isnan(indiv_grid_battery_costs[0]['baseline'][0]) \
            else False
    else:
        indiv_savings = True \
            if not np.isnan(indiv_grid_battery_costs[0]['baseline'][0][0]) \
            else False
    return indiv_savings


def plot_histogram_all_private_savings(saving_per_month_repeat_all, prm, method):
    fig = plt.figure(figsize=(10, 8))
    # plt.hist(saving_per_month_repeat_all_2041, histtype='stepfilled', alpha=0.3, density=True, bins=40,
    #          ec="k", label='Monthly private savings', color='blue')
    # plt.axvline(saving_per_month_repeat_all_2041.mean(), ls='dashed',
    #             label='Mean monthly private saving', color='blue')
    # plt.hist(saving_per_month_repeat_all_2220, histtype='stepfilled', alpha=0.3, density=True, bins=40,
    #          ec="k", label='Monthly private savings\nwith voltage management', color='orange')
    # plt.axvline(saving_per_month_repeat_all_2220.mean(), ls='dashed', color='orange',
    #             label='Mean monthly private saving\nwith voltage management')
    plt.hist(saving_per_month_repeat_all, histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k",
             label='Monthly private savings', color='green')
    plt.axvline(saving_per_month_repeat_all.mean(), ls='dashed', color='green',
                label='Mean monthly private saving')
    plt.legend(loc='upper left', fancybox=True)
    plt.xlabel('Monthly private savings [£]')
    plt.ylabel('Density')
    np.save(f"{prm['paths']['fig_folder']}/saving_per_month_repeat_all_{method}.npy", saving_per_month_repeat_all)
    fig.savefig(
        f"{prm['paths']['fig_folder']}/hist_saving_per_month_repeat_all_{method}.pdf",
        bbox_inches='tight', format='pdf', dpi=1200
    )
    plt.close('all')


def print_opt_d_d_savings(eval_not_baseline, savings_a, prm):
    for it in range(len(eval_not_baseline)):
        if eval_not_baseline[it] == 'opt_d_d':
            savings_opt_d_d = [
                savings_a[home][it] for home in range(prm['syst']['n_homes_test'])
            ]
            print(f"savings per agent opt_d_d: {savings_opt_d_d}")
            print(f"mean {np.mean(savings_opt_d_d)}, "
                  f"std {np.std(savings_opt_d_d)}, "
                  f"min {min(savings_opt_d_d)}, "
                  f"max {max(savings_opt_d_d)}")


def plot_bar_plot_individual_savings_per_home(prm, savings_a, min_savings, max_savings, eval_not_baseline):
    labels = range(prm['syst']['n_homes_test'])
    barWidth = 1 / (len(labels) + 1)

    for i_method, method in enumerate(eval_not_baseline):
        rs = []
        rs.append(np.arange(1))
        for home, _ in enumerate(labels[:- 1]):
            rs.append([x + barWidth for x in rs[home]])

        fig = plt.figure()
        for home, label in enumerate(labels):
            plt.bar(
                rs[home], savings_a[home, i_method], width=barWidth,
                label=label, color='gray', edgecolor="white"
            )
            plt.vlines(rs[home], min_savings[home, i_method], max_savings[home, i_method], color='k')
            # plt.scatter(rs[home], p1_savings[home][0], 'x', color='k')
            # plt.scatter(rs[home], p99_savings[home], 'x', color='k')

        plt.xlabel("Home")
        plt.ylabel("Private monthly saving [£]")
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False
        )
        # plt.xticks([r + barWidth
        #             for r in range(len(eval_not_baseline))],
        #            eval_not_baseline, rotation='vertical')
        # plt.legend()
        title = "Savings per agent relative to " \
                f"individual baseline costs over {prm['RL']['n_end_test']} days {method}"
        title_and_save(title, fig, prm, display_title=False)
        plt.close('all')


def plot_share_energy_battery_individual_savings(prm, share_sc, eval_not_baseline):
    labels = range(prm['syst']['n_homes_test'])
    barWidth = 1 / (len(labels) + 1)

    for i_method, method in enumerate(eval_not_baseline):
        rs = []
        rs.append(np.arange(1))
        for home, _ in enumerate(labels[:- 1]):
            rs.append([x + barWidth for x in rs[home]])

        fig = plt.figure()
        for home in range(len(labels)):
            plt.bar(rs[home], share_sc[home], width=barWidth, label=labels[home])
        plt.xlabel('share of individual savings from battery costs savings')
        plt.xticks(
            [r + barWidth for r in range(len(prm['RL']['evaluation_methods']))],
            prm['RL']['evaluation_methods'], rotation='vertical'
        )
        plt.legend()
        title = "share of individual savings from battery costs " \
            f"savings relative to individual baseline costs {method}"
        title_and_save(title, fig, prm)
        plt.close('all')


def barplot_indiv_savings(record, prm):
    # plot the invidivual savings
    if not savings_are_individual(record.indiv_grid_battery_costs):
        return

    eval_not_baseline = [
        method for method in prm['RL']['evaluation_methods']
        # if method != 'baseline'
    ]
    [
        savings_a, share_sc, std_savings, p10_savings,
        p90_savings, p1_savings, p99_savings, min_savings, max_savings
    ] = [
        np.zeros((prm['syst']['n_homes_test'], len(eval_not_baseline)))
        for _ in range(9)
    ]
    savings_per_repeat = np.zeros((prm['RL']['n_repeats'], len(eval_not_baseline)))
    for i_method, method in enumerate(eval_not_baseline):
        all_saving_per_month_repeat = np.zeros((prm['RL']['n_repeats'], prm['RL']['n_homes_test']))
        for home in range(prm['syst']['n_homes_test']):
            savings_battery_degradation_costs_a, savings_grid_energy_costs_a = [
                np.mean([[(reward[repeat]['baseline'][epoch][home]
                           - reward[repeat][method][epoch])
                          for epoch in range(prm['RL']['n_epochs'],
                                             prm['RL']['n_all_epochs'])]
                         for repeat in range(prm['RL']['n_repeats'])])
                for reward in [record.indiv_grid_battery_costs, record.indiv_grid_energy_costs]
            ]

            share_sc[home, i_method] = savings_battery_degradation_costs_a \
                / (savings_battery_degradation_costs_a + savings_grid_energy_costs_a)
            indiv_grid_battery_costs = record.indiv_grid_battery_costs
            savings_a_all = [
                [
                    (
                        indiv_grid_battery_costs[repeat]['baseline'][epoch][home]
                        - indiv_grid_battery_costs[repeat][method][epoch][home]
                    ) * prm['syst']['interval_to_month']
                    for epoch in range(prm['RL']['n_epochs'], prm['RL']['n_all_epochs'])
                ]
                for repeat in range(prm['RL']['n_repeats'])
            ]
            saving_per_month_repeat = np.mean(savings_a_all, axis=1)
            all_saving_per_month_repeat[:, home] = saving_per_month_repeat
            savings_a[home, i_method] = np.mean(saving_per_month_repeat)
            min_savings[home, i_method] = np.min(saving_per_month_repeat)
            max_savings[home, i_method] = np.max(saving_per_month_repeat)
            p1_savings[home, i_method] = np.percentile(saving_per_month_repeat, 1)
            p99_savings[home, i_method] = np.percentile(saving_per_month_repeat, 99)
            p10_savings[home, i_method] = np.percentile(saving_per_month_repeat, 10)
            p90_savings[home, i_method] = np.percentile(saving_per_month_repeat, 90)
            std_savings[home, i_method] = np.std(saving_per_month_repeat)
            savings_per_repeat[:, i_method] = saving_per_month_repeat
        saving_per_month_repeat_all = np.array(all_saving_per_month_repeat).flatten()

        plot_histogram_all_private_savings(saving_per_month_repeat_all, prm, method)

    print_opt_d_d_savings(eval_not_baseline, savings_a, prm)
    share_sc_neg_pos, share_sc_pos_pos, share_sc_neg_neg, share_sc_pos_neg = [
        np.full((len(eval_not_baseline), prm['syst']['n_homes_test'], prm['RL']['n_repeats'], prm['RL']['n_epochs']), np.nan) for _ in
        range(4)]
    for i_method, method in enumerate(eval_not_baseline):
        for home in range(prm['syst']['n_homes_test']):
            for repeat in range(prm['RL']['n_repeats']):
                for epoch in range(prm['RL']['n_epochs']):
                    savings_battery_degradation_costs_a, savings_grid_energy_costs_a = [
                        reward[repeat]['baseline'][epoch][home] - reward[repeat][method][epoch][home]
                        for reward in [record.indiv_grid_battery_costs, record.indiv_grid_energy_costs]
                    ]
                    share_sc_ = savings_battery_degradation_costs_a \
                        / (savings_battery_degradation_costs_a + savings_grid_energy_costs_a)
                    if savings_battery_degradation_costs_a < 0 and savings_grid_energy_costs_a < 0:
                        share_sc_neg_neg[i_method, home, repeat, epoch] = share_sc_
                    elif savings_battery_degradation_costs_a < 0 and savings_grid_energy_costs_a > 0:
                        share_sc_neg_pos[i_method, home, repeat, epoch] = share_sc_
                    elif savings_battery_degradation_costs_a > 0 and savings_grid_energy_costs_a < 0:
                        share_sc_pos_neg[i_method, home, repeat, epoch] = share_sc_
                    elif savings_battery_degradation_costs_a > 0 and savings_grid_energy_costs_a > 0:
                        share_sc_pos_pos[i_method, home, repeat, epoch] = share_sc_

    n_tot = np.prod(np.shape(share_sc_neg_neg))
    print(f'share_sc_neg battery costs _neg energy costs {np.nanmean(share_sc_neg_neg)} (n={np.count_nonzero(~np.isnan(share_sc_neg_neg))/n_tot})')
    print(f'share_sc_neg_pos {np.nanmean(share_sc_neg_pos)} (n={np.count_nonzero(~np.isnan(share_sc_neg_pos))/n_tot})')
    print(f'share_sc_pos_neg {np.nanmean(share_sc_pos_neg)} (n={np.count_nonzero(~np.isnan(share_sc_pos_neg))/n_tot})')
    print(f'share_sc_pos_pos {np.nanmean(share_sc_pos_pos)} (n={np.count_nonzero(~np.isnan(share_sc_pos_pos))/n_tot})')

    # plot variation per home and epoch within repeat
    for repeat in [1, 4]:
        savings_a_all_repeat1 = [
            [
                (
                        indiv_grid_battery_costs[repeat]['baseline'][epoch][home]
                        - indiv_grid_battery_costs[repeat][method][epoch][home]
                ) * prm['syst']['interval_to_month']
                for epoch in range(prm['RL']['n_epochs'], prm['RL']['n_all_epochs'])
            ]
            for home in range(prm['RL']['n_homes'])
        ]
        print(f"repeat {repeat} np.mean across homes and epochs: {np.mean(savings_a_all_repeat1)}")
        print(f"repeat {repeat} np.std across homes and epochs: {np.std(savings_a_all_repeat1)}")
        print(f"repeat {repeat} np.std across homes: {np.std(np.mean(savings_a_all_repeat1, axis=1))}")
        print(f"repeat {repeat} np.std across epochs: {np.std(np.mean(savings_a_all_repeat1, axis=0))}")

        # Turn off interactive mode
        plt.ioff()

        fig = plt.figure()
        plt.imshow(savings_a_all_repeat1, cmap='hot', interpolation='nearest')
        title = f'savings_a_all_repeat{repeat}_per_home_and_epoch'
        title_and_save(title, fig, prm)
        plt.close('all')

    # plot total individual savings
    plot_bar_plot_individual_savings_per_home(prm, savings_a, min_savings, max_savings, eval_not_baseline)

    # plot share of energy vs battery savings individually
    plot_share_energy_battery_individual_savings(prm, share_sc, eval_not_baseline)


def plot_voltage_statistics(record, prm):
    """ Creates plot of mean, max and number of voltage deviations per method """
    if 'mean_voltage_violation' not in record.__dict__.keys() and 'max_voltage_violation' not in record.__dict__.keys():
        labels = [
            'mean_voltage_deviation',
            # None,
            'max_voltage_deviation',
            'n_voltage_deviation_bus',
            # 'n_voltage_deviation_hour'
        ]
    elif 'max_voltage_violation' not in record.__dict__.keys():
        labels = [
            'mean_voltage_violation',
            # None,
            'max_voltage_deviation',
            'n_voltage_violation_bus',
            # 'n_voltage_deviation_hour'
        ]
    else:
        labels = [
            'mean_voltage_violation',
            # 'mean_voltage_deviation',
            'max_voltage_violation',
            'n_voltage_violation_bus',
            # 'n_voltage_violation_hour'
        ]

    for label in labels:
        if label is None:
            continue
        record_obj = getattr(record, label)
        fig, ax = plt.subplots(figsize=(16, 8))
        for method in prm['RL']['evaluation_methods']:
            n = len(record_obj[0][method])
            x_axis = range(n)
            values = np.zeros((prm['RL']['n_repeats'], n))
            for repeat in range(prm['RL']['n_repeats']):
                values[repeat] = record_obj[repeat][method]
            ax.plot(x_axis, np.mean(values, axis=0), label=method)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f"Mean {label} over repeats")
        ax.legend()
        title = f"{label} vs epochs"
        title_and_save(title, fig, prm)
        plt.close('all')

    hist_values = {
        method: {} for method in prm['RL']['evaluation_methods']
    }
    for method in prm['RL']['evaluation_methods']:
        for label in labels:
            if label is None:
                continue
            hist_values[method][label] = np.mean(
                [
                    np.mean(getattr(record, label)[repeat][method][- prm['RL']['n_end_test']:])
                    for repeat in range(prm['RL']['n_repeats'])
                ]
            )
    np.save(
        prm['paths']['fig_folder'] / "voltage_deviation_metrics",
        hist_values
    )
    y_axis_labels = [
        'Mean \nvoltage constraint violation \n[p.u.]',
        # 'Mean \nvoltage constraint deviation \n[p.u.]',
        'Max \nvoltage constraint violation \n[p.u.]',
        "Percentage of voltage violations\nover all buses and time steps"
        # "Percentage of voltage violations\nover all buses and time steps"
        # 'Number of hours \nwith voltage constraint violation'
    ]
    methods_hist = [
        method for method in prm['RL']['evaluation_methods']
        if method in ['opt_d_d', 'env_r_c', 'baseline', 'opt']
    ]
    methods_labels = {
        method: method for method in methods_hist
    }
    methods_labels['baseline'] = 'Baseline'
    methods_labels['env_r_c'] = 'MARL'
    methods_labels['opt'] = 'Optimal'
    fig, axs = plt.subplots(1, 3, figsize=(8*1.5, 4*1.5))
    # rows = [0, 0, 1, 1]
    # cols = [0, 1, 0, 1]
    min_val, max_val = 1e10, -1e10
    ylims = [2.3e-3, 0.023, 18]
    print(f"hist_values {hist_values}")
    for i, (label, ylim) in enumerate(zip(labels, ylims)):
        Y = [hist_values[method][label] for method in methods_hist]
        if i == 2:
            Y = [y/(24 * 906) * 100 for y in Y]
        # ax = axs[row, col]
        ax = axs[i]
        min_val = min(min_val, min(Y))
        max_val = max(max_val, max(Y))
        methods_labels_i = [methods_labels[method] for method in methods_hist]
        ax.bar(methods_labels_i, Y, color='gray')
        print(f"{label} {methods_labels_i} {Y}")
        # axs[row, col].set_xlabel("Learning method")
        ax.set_ylabel(y_axis_labels[i])
        ax.set_ylim([-0.02 * ylim, ylim])
        # Set the tick label formatter to ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
        # Apply the formatter to the tick labels on the y-axis
        ax.yaxis.set_major_formatter(formatter)
        # axs[row, col].set_ticklabel_format(axis='both', style='sci', scilimits=(4, 4))
        # Remove x-axis ticks and labels
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xticklabels([])  # Remove x-axis tick labels
    plt.tight_layout()
    # plt.ticklabel_format(axis='both', style='sci', scilimits=(4, 4))
    title = "histogram voltage deviations"
    title_and_save(title, fig, prm, display_title=False)
    plt.close('all')
