
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# plot timing vs performance for n layers / dim layers; runs 742-656
font = {'size': 14}
matplotlib.rc('font', **font)

labels = {
    'IQ': 'Independent Q-learning',
    'OIQ': 'Optimisation-informed independent Q-learning',
    'MQ': 'Marginal reward Q-learning',
    'OMQ': 'Optimisation-informed, marginal reward Q-learning',
    # 'FD': 'FACMAC (day-ahead)',
    'FH': 'FACMAC',  # (hourly)
    'FHO': 'Optimisation-informed FACMAC (hourly)'
}
type_learning = {
    'IQ': 'env_r_d',
    'OIQ': 'opt_r_d',
    'MQ': 'env_d_d',
    'OMQ': 'opt_d_d',
    'FD': 'env_r_c',
    'FH': 'env_r_c',
    'FHO': 'env_r_c',

}
opt_informed = {
    'IQ': False,
    'OIQ': True,
    'MQ': False,
    'OMQ': True,
    'FD': False,
    'FH': False,
    'FHO': True,
}
runs = {
    'OMQ': [170, 94, 96, 97, 98, 99],
    'IQ': [170, 94, 96, 97, 98, 99],
    'OIQ': [170, 94, 96, 97, 98, 99],
    'MQ': [170, 94, 96, 97, 98, 99],
    'FD': [160, 105, 106, 123, 124, 125],
    'FH': [126, 127, 128, 129, 130, 131],
    'FHO': [161, 162, 163, 164, 168, 169]
}


green = (117/255, 189/255, 167/255)
blue = (80/255, 146/255, 182/255)
red = (192/255, 0, 0)
colours = {
    'IQ': blue,
    'OIQ': blue,
    'MQ': green,
    'OMQ': green,
    'FD': red,
    'FH': red,
    'FHO': red,
}


n_homes = [1, 3, 5, 10, 20, 50]

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs.csv"
log = pd.read_csv(log_path)
metrics = ['ave', 'p25', 'p75']
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

for line_label in labels.keys():
    values = {metric: np.full(len(n_homes), np.nan) for metric in metrics}
    time_end = np.full(len(n_homes), np.nan)
    colour = colours[line_label]
    for i, run in enumerate(runs[line_label]):
        if run is not None:
            for metric in metrics:
                try:
                    values[metric][i] = log.loc[log['run'] == run, f'{metric}_{type_learning[line_label]}'].values[0]
                except Exception as ex:
                    print()
                time_end[i] = log.loc[log['run'] == run, 'time_end'].values[0]
    ls = '--' if opt_informed[line_label] else '-'
    label = None if opt_informed[line_label] else labels[line_label]

    axs[0].plot(
        n_homes,
        values['ave'],
        label=label, linestyle=ls,
        color=colour
    )
    axs[0].fill_between(
        n_homes,
        values['p25'],
        values['p75'],
        alpha=0.2,
        color=colour
    )
    if line_label in ['OMQ', 'FH']:
        axs[1].plot(
            n_homes, time_end,
            linestyle=ls,
            color=colour
        )
        print(f"{line_label}: time_end {time_end}")

axs[0].set_ylabel("Savings [£/home/month]")
axs[0].axes.xaxis.set_ticklabels([])

axs[1].set_yscale('log')
axs[1].set_ylabel("Time [s]")
axs[1].set_xlabel("Number of homes")

axs[0].legend(fancybox=True)
fig.savefig(
    f"outputs/results_analysis/facmac_results.pdf",
    bbox_inches='tight', format='pdf', dpi=1200
)
plt.close('all')


def exponential(x, a):
    return a ** x

def second_order(x, a):
    return a * x ** 2

def second_order_b(x, a, b):
    return a * x ** 2 + b

def first_order(x, a, b):
    """First-order function."""
    return a * x + b

x_max = 50
xs_facmac = [1, 3, 5, 10, 20, 50]
ys_facmac = [ 26.50374389 , 38.85938716 , 49.69802213 , 73.43057513, 123.11931181, 281.53323126]
xs_iql = [3, 5, 10, 20, 50]
ys_iql = [302.6728189 ,  328.64307976 , 710.36226296, 1860.68218756, 8911.68623304]
for xs, ys, label in zip(
    [xs_facmac, xs_iql],
    [ys_facmac, ys_iql],
    ['facmac', 'iql']
):
    popt, pcov = optimize.curve_fit(first_order, xs, ys)
    a, b = popt
    print(f"{label} a, b first errors = {np.sqrt(np.diag(pcov))}")
    first_order_fitted = [first_order(x, a, b) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y)**2 for y_fitted, y in zip(first_order_fitted, ys)):.2e}")

    popt, pcov = optimize.curve_fit(second_order, xs, ys)
    c = popt[0]
    print(f"{label} c,d second errors b = {np.sqrt(np.diag(pcov))}")
    second_order_fitted = [second_order(x, c) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y)**2 for y_fitted, y in zip(second_order_fitted, ys)):.2e}")

    popt, pcov = optimize.curve_fit(second_order_b, xs, ys)
    f, g = popt
    print(f"{label} c,d second errors b = {np.sqrt(np.diag(pcov))}")
    second_order_b_fitted = [second_order_b(x, f, g) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y)**2 for y_fitted, y in zip(second_order_b_fitted, ys)):.2e}")


    popt, pcov = optimize.curve_fit(exponential, xs, ys)
    e = popt[0]
    print(f"{label} e exponential errors = {np.sqrt(np.diag(pcov))}")
    exp_fitted = [exponential(x, e) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y)**2 for y_fitted, y in zip(exp_fitted, ys)):.2e}")

    fig = plt.figure()
    plt.plot(xs, ys, 'o', label='data')
    plt.plot(list(range(x_max)), first_order_fitted,
             label='first order')
    plt.plot(list(range(x_max)), second_order_fitted,
             label='second order')
    plt.plot(list(range(x_max)), second_order_b_fitted,
             label='second order + b')
    plt.plot(list(range(x_max)), exp_fitted,
             label='exponential')
    plt.legend()
    plt.gca().set_yscale('log')
    plt.show()
    plt.title(label)
    fig.savefig(f"outputs/results_analysis/{label}_growth")
    plt.close('all')
