import matplotlib.pyplot as plt
import numpy as np

from src.environment.post_analysis.plotting.plotting_utils import \
    formatting_figure
from src.environment.utilities.userdeftools import (data_source,
                                                    get_moving_average)


def _update_lower_upper_bounds(min_val, max_val, lower_bound, upper_bound, paths):
    spread = max_val - min_val
    current_lower_bound = min_val - 0.02 * spread
    current_upper_bound = max_val + 0.02 * spread
    if current_lower_bound < lower_bound:
        np.save(paths['open_inputs'] / 'lower_bound0', current_lower_bound)
        lower_bound = current_lower_bound
    if current_upper_bound > upper_bound:
        np.save(paths['open_inputs'] / 'upper_bound0', current_upper_bound)
        upper_bound = current_upper_bound

    return lower_bound, upper_bound


def str_to_float(label):
    if label[0] == '−':
        return -float(label[1:])
    else:
        return float(label)


def plot_results_all_repeats(prm, record, moving_average=True, diff_to_opt=False):
    plt.rcParams['font.size'] = '16'
    fig = plt.figure(figsize=(8, 8))

    min_val, max_val = 100, - 100
    plt.rcParams['font.size'] = '14'
    lower_bound, upper_bound = [
        np.load(prm['paths']['open_inputs'] / f"{e}0.npy")
        for e in ['lower_bound', 'upper_bound']
    ]
    baseline = 'opt' if diff_to_opt else 'baseline'
    for e in [e for e in prm["save"]["eval_entries_plot"] if e not in [baseline, 'opt_n_d']]:
        p25, p50, p75, p25_not_nan, p75_not_nan, epoch_not_nan = record.results_to_percentiles(
            e, prm,
            mov_average=moving_average,
            n_window=prm["save"]["n_window"],
            baseline=baseline
        )

        min_val = np.min(p25_not_nan) if np.min(p25_not_nan) < min_val \
            else min_val
        max_val = np.max(p75_not_nan) if np.max(p75_not_nan) > max_val \
            else max_val

        lower_bound, upper_bound = _update_lower_upper_bounds(
            min_val, max_val, lower_bound, upper_bound, prm['paths']
        )
        if e == 'opt':
            ls = 'dotted'
        elif data_source(e) == 'opt':
            ls = 'dashed'
        else:
            ls = 'solid'
        n_epochs = prm['RL']['n_epochs']
        plt.plot(p50[0: n_epochs], label=e, color=prm['save']['colourse'][e], ls=ls)
        plt.fill_between(
            epoch_not_nan[0: n_epochs],
            p25_not_nan[0: n_epochs],
            p75_not_nan[0: n_epochs],
            color=prm['save']['colourse'][e], alpha=0.1
        )
    plt.hlines(
        y=0, xmin=0, xmax=n_epochs - 1, colors='k',
        linestyle='dotted'
    )
    plt.legend()
    # plt.ylim([lower_bound, upper_bound])
    plt.ylim(- 30, 135)

    plt.tight_layout()
    if moving_average:
        plt.title('Moving average of difference between baseline and reward')

    plt.xlabel('Episode')
    title_display = "Savings relative to baseline"
    if moving_average:
        title_display += " (moving average)"
    plt.title(title_display)
    ax = fig.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = [int(str_to_float(label)) for label in labels]
    ax.set_xticklabels(new_labels)
    ylabel = '[£/hr/home]'
    plt.ylabel(ylabel)
    title = f"moving average n_window = {prm['save']['n_window']} " if moving_average else ""
    title += f"med, 25-75th percentile over repeats state comb " \
             f"{prm['RL']['statecomb_str']}"
    if diff_to_opt:
        title += "_diff_to_opt"
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
        plt.plot(diff, label=e, color=prm['save']['colourse'][e])
    plt.xlabel('episodes')

    plt.ylabel('reward difference rel. to baseline')
    title = f"Moving average all rewards minus baseline " \
            f"state comb {prm['RL']['statecomb_str']} repeat {repeat} " \
            f"n_window = {prm['save']['n_window']}"
    formatting_figure(prm, fig=fig, title=title)
    plt.close('all')
