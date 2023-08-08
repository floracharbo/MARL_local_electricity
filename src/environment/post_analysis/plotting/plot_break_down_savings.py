import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from src.environment.post_analysis.plotting.plotting_utils import (
    formatting_figure, title_and_save)
from src.environment.utilities.userdeftools import (data_source,
                                                    distr_learning,
                                                    reward_type)


def distribution_savings(prm, aggregate='daily'):
    rl = prm["RL"]
    fig = plt.figure()
    test_savings = {}
    for method in [method for method in rl["evaluation_methods"] if method != 'baseline']:
        test_savings[method] = []
        for repeat in range(prm['RL']['n_repeats']):
            rewards_t, rewards_bsl = \
                [rl["monthly_mean_eval_rewards_per_home"][evaluation_method][repeat][
                 prm['RL']['n_epochs']:]
                 for evaluation_method in [method, 'baseline']]
            savings_rel_baseline = \
                [reward - baseline for reward, baseline
                 in zip(rewards_t, rewards_bsl)]
            if aggregate == 'daily':
                test_savings[method] += savings_rel_baseline
            elif aggregate == 'test_period':
                test_savings[method] += [np.mean(savings_rel_baseline)]

        plt.hist(test_savings[method], alpha=0.5, label=method,
                 color=prm['save']['colourse'][method])

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
                [bars[i][-1] / tots[method] if tots[method] > 0 else None
                    for i in range(len(labels))]
            )
    print(f"shares_reduc['env_r_c'] {shares_reduc['env_r_c']}")

    new_labels = [
        'Grid costs', 'Storage costs', 'Distribution Costs', 'Import/Export Penalty',
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


def barplot_indiv_savings(record, prm):
    # plot the invidivual savings
    if len(np.shape(record.indiv_grid_battery_costs[0]['baseline'][0])) == 0:
        indiv_savings = True \
            if not np.isnan(record.indiv_grid_battery_costs[0]['baseline'][0]) \
            else False
    else:
        indiv_savings = True \
            if not np.isnan(record.indiv_grid_battery_costs[0]['baseline'][0][0]) \
            else False
    if indiv_savings:
        eval_not_baseline = [
            method for method in prm['RL']['evaluation_methods']
            if method != 'baseline'
        ]
        savings_a, share_sc, std_savings = [
            [[] for _ in range(prm['syst']['n_homes_test'])]
            for _ in range(3)
        ]
        for home in range(prm['syst']['n_homes_test']):
            for method in eval_not_baseline:
                savings_battery_degradation_costs_a, savings_grid_energy_costs_a = [
                    np.mean([[(reward[repeat]['baseline'][epoch][home]
                               - reward[repeat][method][epoch])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 prm['RL']['n_epochs'])]
                             for repeat in range(prm['RL']['n_repeats'])])
                    for reward in [record.indiv_grid_battery_costs, record.indiv_grid_energy_costs]
                ]
                share_sc[home].append(
                    savings_battery_degradation_costs_a
                    / (savings_battery_degradation_costs_a + savings_grid_energy_costs_a)
                )
                indiv_grid_battery_costs = record.indiv_grid_battery_costs
                savings_a_all = [
                    [
                        (
                            indiv_grid_battery_costs[repeat]['baseline'][epoch][home]
                            - indiv_grid_battery_costs[repeat][method][epoch]
                        )
                        for epoch in range(prm['RL']['start_end_eval'], prm['RL']['n_epochs'])
                    ]
                    for repeat in range(prm['RL']['n_repeats'])
                ]
                savings_a[home].append(np.mean(savings_a_all))
                std_savings[home].append(np.std(savings_a_all))
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

        # plot total individual savings
        labels = range(prm['syst']['n_homes_test'])
        barWidth = 1 / (len(labels) + 1)
        rs = []
        rs.append(np.arange(len(prm['RL']['evaluation_methods']) - 1))
        for home, _ in enumerate(labels[:- 1]):
            rs.append([x + barWidth for x in rs[home]])

        fig = plt.figure()
        for home, label in enumerate(labels):
            plt.bar(rs[home], savings_a[home], width=barWidth,
                    label=label, yerr=std_savings[home])
        plt.xlabel('savings per agent')
        plt.xticks([r + barWidth
                    for r in range(len(eval_not_baseline))],
                   eval_not_baseline, rotation='vertical')
        plt.legend()
        title = "savings per agent relative to " \
                "individual baseline costs"
        title_and_save(title, fig, prm)
        plt.close('all')

        # plot share of energy vs battery savings individually
        fig = plt.figure()
        for home in range(len(labels)):
            plt.bar(rs[home], share_sc[home], width=barWidth, label=labels[home])
        plt.xlabel('share of individual savings from battery costs savings')
        plt.xticks([r + barWidth
                    for r in range(len(prm['RL']['evaluation_methods']))],
                   prm['RL']['evaluation_methods'], rotation='vertical')
        plt.legend()
        title = "share of individual savings from battery costs " \
            "savings relative to individual baseline costs"
        title_and_save(title, fig, prm)
        plt.close('all')


def plot_voltage_statistics(record, prm):
    """ Creates plot of mean, max and number of voltage deviations per method """
    labels = [
        record.break_down_rewards_entries[prm['syst']['break_down_rewards_entries'].index(label)]
        for label in [
            'mean_voltage_deviation', 'max_voltage_deviation',
            'n_voltage_deviation_bus', 'n_voltage_deviation_hour'
        ]
    ]
    for label in labels:
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
        'Mean \nvoltage constraint violation \n[p.u.]', 'Max \nvoltage constraint violation \n[p.u.]',
        'Number of bus-hours \nwith voltage constraint violation',
        'Number of hours \nwith voltage constraint violation'
    ]
    methods_hist = [
        method for method in prm['RL']['evaluation_methods']
        if method in ['opt_d_d', 'env_r_c', 'baseline']
    ]
    methods_labels = {
        method: method for method in methods_hist
    }
    methods_labels['baseline'] = 'Baseline'
    methods_labels['env_r_c'] = 'MARL'
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
    min_val, max_val = 1e10, -1e10
    for i, (label, row, col) in enumerate(zip(labels, rows, cols)):
        Y = [hist_values[method][label] for method in methods_hist]
        min_val = min(min_val, min(Y))
        max_val = max(max_val, max(Y))
        methods_labels_i = [methods_labels[method] for method in methods_hist]
        axs[row, col].bar(methods_labels_i, Y, color='gray')
        print(f"{label} {methods_labels_i} {Y}")
        # axs[row, col].set_xlabel("Learning method")
        axs[row, col].set_ylabel(y_axis_labels[i])
        # Set the tick label formatter to ScalarFormatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))  # Adjust the power limits as needed
        # Apply the formatter to the tick labels on the y-axis
        axs[row, col].yaxis.set_major_formatter(formatter)
        # axs[row, col].set_ticklabel_format(axis='both', style='sci', scilimits=(4, 4))

    plt.tight_layout()
    # plt.ticklabel_format(axis='both', style='sci', scilimits=(4, 4))
    title = "histogram voltage deviations"
    title_and_save(title, fig, prm, display_title=False)
    plt.close('all')
