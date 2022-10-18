import matplotlib.pyplot as plt
import numpy as np

from src.post_analysis.plotting.plotting_utils import formatting_figure
from src.utilities.userdeftools import get_moving_average


def plot_results_all_repeats(
        prm, record, moving_average=True):
    fig = plt.figure(figsize=(4.600606418100001, 4.2))
    min_val, max_val = 100, - 100
    plt.rcParams['font.size'] = '10'

    for e in [e for e in prm["save"]["eval_entries_plot"] if e != 'baseline']:
        results = np.array(
            [[None if record.mean_eval_rewards_per_hh[repeat][e][epoch] is None
              else record.mean_eval_rewards_per_hh[repeat][e][epoch]
              - record.mean_eval_rewards_per_hh[repeat]['baseline'][epoch]
              for epoch in range(prm['RL']['n_all_epochs'])]
             for repeat in range(prm['RL']['n_repeats'])])
        if moving_average:
            results = np.array(
                [get_moving_average(results[repeat], prm["save"]["n_window"], Nones=False)
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
                n_window=prm["save"]["n_window"]
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
    title = f"moving average n_window = {prm['save']['n_window']} " if moving_average else ""
    title += f"med, 25-75th percentile over repeats state comb " \
             f"{prm['RL']['statecomb_str']}"
    formatting_figure(
        prm, fig=fig, title=title,
        legend=False, display_title=False
    )

    return lower_bound, upper_bound


def plot_mova_eval_per_repeat(repeat, prm):
    rl = prm["RL"]
    if not prm['save']['plot_indiv_repeats_rewards']:
        return
    fig = plt.figure()
    mova_baseline = get_moving_average(
        [rl["eval_rewards"][repeat]['baseline']
         for repeat in prm['RL']['n_repeats']], prm["save"]["n_window"])
    # 2 - moving average of all rewards evaluation
    for e in [e for e in prm["save"]["eval_entries_plot"] if e != 'baseline']:
        mova_e = get_moving_average(
            [rl["eval_rewards"][repeat][e] for repeat in prm['RL']['n_repeats']],
            prm["save"]["n_window"])
        diff = [m - mb if m is not None else None
                for m, mb in zip(mova_e, mova_baseline)]
        plt.plot(diff, label=e, color=prm['save']['colorse'][e])
    plt.xlabel('episodes')
    plt.ylabel('reward difference rel. to baseline')
    title = f"Moving average all rewards minus baseline " \
            f"state comb {prm['RL']['statecomb_str']} repeat {repeat} " \
            f"n_window = {prm['save']['n_window']}"
    formatting_figure(prm, fig=fig, title=title)
    plt.close('all')
