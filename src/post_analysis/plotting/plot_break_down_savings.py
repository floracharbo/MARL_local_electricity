import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.post_analysis.plotting.plotting_utils import (formatting_figure,
                                                       title_and_save)
from src.utilities.userdeftools import data_source, distr_learning, reward_type


def distribution_savings(prm, aggregate='daily'):
    rl = prm["RL"]
    fig = plt.figure()
    test_savings = {}
    for method in [method for method in rl["mean_eval_rewards_per_hh"][0] if method != 'baseline']:
        test_savings[method] = []
        for repeat in range(prm['RL']['n_repeats']):
            rewards_t, rewards_bsl = \
                [rl["mean_eval_rewards_per_hh"][repeat][evaluation_method][
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
            title_and_save(title, fig, prm)
        np.save(prm['paths']['fig_folder'] / 'M_reductions', M)


def barplot_breakdown_savings(record, prm, plot_type='savings'):
    # all break down rewards except for the last three ones
    # which are individual values
    labels = record.break_down_rewards_entries[:-3]
    bars = [[] for _ in range(len(labels))]
    shares_reduc = {}
    tots = {}
    for method in prm['RL']['evaluation_methods']:
        shares_reduc[method] = []
        for i, label in enumerate(labels):
            record_obj = record.__dict__[label]
            mult = prm['syst']['co2tax'] if label == 'emissions' else 1
            if plot_type == 'savings':
                bars[i].append(
                    np.mean([[(record_obj[repeat]['baseline'][epoch]
                               - record_obj[repeat][method][epoch])
                              * mult / (prm['syst']['n_homes'] + prm['syst']['n_homesP'])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 len(record_obj[repeat][method]))]
                             for repeat in range(prm['RL']['n_repeats'])]))
            else:
                bars[i].append(
                    np.mean([[(record_obj[repeat][method][epoch])
                              * mult / (prm['syst']['n_homes'] + prm['syst']['n_homesP'])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 len(record_obj[repeat][method]))]
                             for repeat in range(prm['RL']['n_repeats'])]))
        tots[method] = sum(
            bars[i][-1] for i, label in enumerate(labels)
            if label in ['dc', 'sc', 'gc', 'pc']
        )
        shares_reduc[method].append(
            [bars[i][-1] / tots[method] if tots[method] > 0 else None
             for i in range(len(labels))]
        )

    barWidth = 1 / (len(labels) + 1)
    rs = []
    rs.append(np.arange(len(prm['RL']['evaluation_methods'])))
    for ir in range(len(labels) - 1):
        rs.append([x + barWidth for x in rs[ir]])
    plt.figure()
    for ir in range(len(labels)):
        plt.bar(rs[ir], bars[ir], width=barWidth, label=labels[ir])
    plt.xlabel('evaluation')
    plt.xticks([r + barWidth
                for r in range(len(bars[0]))], prm['RL']['evaluation_methods'],
               rotation='vertical')
    plt.legend()
    plt.tight_layout()
    if plot_type == 'savings':
        plt.title('savings relative to baseline costs / emissions')
    else:
        plt.title('mean costs distribution')
    plt.savefig(
        f"{prm['paths']['fig_folder']}/mean_{plot_type}_bar_plots.png", bbox_inches='tight')
    plt.close('all')


def barplot_indiv_savings(record, prm):
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
        eval_not_baseline = [
            method for method in prm['RL']['evaluation_methods']
            if method != 'baseline'
        ]
        savings_a, share_sc, std_savings = [
            [[] for _ in range(prm['syst']['n_homes'])]
            for _ in range(3)
        ]
        for home in range(prm['syst']['n_homes']):
            for method in eval_not_baseline:
                savings_sc_a, savings_gc_a = [
                    np.mean([[(reward[repeat]['baseline'][epoch][home]
                               - reward[repeat][method][epoch])
                              for epoch in range(prm['RL']['start_end_eval'],
                                                 prm['RL']['n_epochs'])]
                             for repeat in range(prm['RL']['n_repeats'])])
                    for reward in [record.__dict__['indiv_sc'],
                                   record.__dict__['indiv_gc']]]
                share_sc[home].append(
                    savings_sc_a / (savings_sc_a + savings_gc_a))
                savings_a_all = \
                    [[(record.__dict__['indiv_c'][repeat]['baseline'][epoch][home]
                       - record.__dict__['indiv_c'][repeat][method][epoch])
                      for epoch in range(prm['RL']['start_end_eval'],
                                         prm['RL']['n_epochs'])]
                     for repeat in range(prm['RL']['n_repeats'])]
                savings_a[home].append(np.mean(savings_a_all))
                std_savings[home].append(np.std(savings_a_all))
        for it in range(len(eval_not_baseline)):
            if eval_not_baseline[it] == 'opt_d_d':
                savings_opt_d_d = [savings_a[home][it]
                                   for home in range(prm['syst']['n_homes'])]
                print(f"savings per agent opt_d_d: {savings_opt_d_d}")
                print(f"mean {np.mean(savings_opt_d_d)}, "
                      f"std {np.std(savings_opt_d_d)}, "
                      f"min {min(savings_opt_d_d)}, "
                      f"max {max(savings_opt_d_d)}")

        # plot total individual savings
        labels = range(prm['syst']['n_homes'])
        barWidth = 1 / (len(labels) + 1)
        rs = []
        rs.append(np.arange(len(prm['RL']['evaluation_methods']) - 1))
        for home in range(len(labels) - 1):
            rs.append([x + barWidth for x in rs[home]])

        fig = plt.figure()
        for home in range(len(labels)):
            plt.bar(rs[home], savings_a[home], width=barWidth,
                    label=labels[home], yerr=std_savings[home])
        plt.xlabel('savings per agent')
        plt.xticks([r + barWidth
                    for r in range(len(prm['RL']['evaluation_methods']))],
                   prm['RL']['evaluation_methods'], rotation='vertical')
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
