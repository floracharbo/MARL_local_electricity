
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# plot timing vs performance for n layers / dim layers; runs 742-656
font = {'size': 14}
matplotlib.rc('font', **font)

compare_times = ['OMQ', 'FH']
labels = {
    'IQ': 'Independent Q-learning',
    'OIQ': 'Optimisation-informed independent Q-learning',
    'MQ': 'Marginal reward Q-learning',
    'OMQ': 'Optimisation-informed, marginal reward Q-learning',
    'FD': 'FACMAC (day-ahead)',
    'FDO': 'Optimisation-informed  FACMAC (day-ahead)',
    'FH': 'FACMAC',  # (hourly)
    'FHO': 'Optimisation-informed FACMAC (hourly)'
}
type_learning = {
    'IQ': 'env_r_d',
    'OIQ': 'opt_r_d',
    'MQ': 'env_d_d',
    'OMQ': 'opt_d_d',
    'FD': 'env_r_c',
    'FDO': 'env_r_c',
    'FH': 'env_r_c',
    'FHO': 'env_r_c',

}
opt_informed = {
    'IQ': False,
    'OIQ': True,
    'MQ': False,
    'OMQ': True,
    'FD': False,
    'FDO': True,
    'FH': False,
    'FHO': True,
}
runQ = list(range(534, 539)) + [677],

runs = {
    # 'OMQ': [170, 94, 96, 97, 98, 99],
    # 'OMQ': [236, 237, 238, 234,  240, 244],
    # 'OMQ': [420, 421, 422, 423, 424, 509],
    'OMQ': runQ,

    # 'IQ': [170, 94, 96, 97, 98, 99],
    # 'IQ': [236, 237, 238, 234,  240, 244],
    'IQ': runQ,
    # 'OIQ': [170, 94, 96, 97, 98, 99],
    # 'MQ': [170, 94, 96, 97, 98, 99],
    # 'OIQ': [236, 237, 238, 234,  240, 244],
    'OIQ': runQ,
    # 'MQ': [236, 237, 238, 234,  240, 244],
    'MQ': runQ,

    # 'FD': [160, 105, 106, 123, 124, 125],
    # 'FD': list(range(265, 271)),
    # 'FD': list(range(516, 522)),
    'FD': list(range(684, 690)),

    # 'FDO': list(range(257, 261)) + list(range(263, 265)),
    # 'FDO': list(range(528, 534)),
    'FDO': list(range(582, 587)),
    # missing 30

    # 'FH': [126, 127, 128, 129, 130, 131],
    # 'FH': list(range(245, 251)),
    # 'FH': list(range(510, 516)),
    'FH': list(range(678, 684)),

    # 'FHO': [161, 162, 163, 164, 168, 169]
    # 'FHO': list(range(251, 257)),
    # 'FHO': list(range(522, 528)),
    'FHO': list(range(522, 527)),

}


green = (117/255, 189/255, 167/255)
# '#ff7f00' colorblind orange
# green = '#4daf4a'
# colorblind green
# '#f781bf'colorblind pink
#  '#a65628' colorblind brown
# green = '#984ea3' colorblind purple i dont think so visible w red
#  '#999999' colorblind grey
grey = '#999999'
#  '#e41a1c' colorblind red
#  '#dede00' oclorbing yellow
# blue = (80/255, 146/255, 182/255)
blue = '#377eb8' #  colorblind blue

red = (192/255, 0, 0)
# [, , ,
#                   , , ,
#                  , , ]

colours = {
    'IQ': blue,
    'OIQ': blue,
    'MQ': green,
    'OMQ': green,
    'FD': grey,
    'FDO': grey,
    'FH': red,
    'FHO': red,
}

# n_homes = [1, 3, 5, 10, 20, 50]
n_homes = [1, 3, 5, 10, 20, 30]

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs.csv"
log = pd.read_csv(log_path)
best_score_type = 'p50'
# p50 or ave

metrics = [best_score_type, 'p25', 'p75']

def exponential(x, a):
    return a ** x

def second_order(x, a):
    return a * x ** 2

def second_order_b(x, a, b):
    return a * x ** 2 + b

def first_order(x, a, b):
    """First-order function."""
    return a * x + b

def polynomial2(x, a, b, c):
    return a * x ** 2 + b * x + c

def unknown_order(x, a, b, c):
    return a * x ** b + c

def polynomial3(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def xlogx(x, a, b):
    return a * x * np.log(x) + b

def logx(x, a, b):
    return a * np.log(x) + b

x_max = n_homes[-1]
ys_time = {entry: [] for entry in compare_times}

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot([0, n_homes[-1]], [0, 0], color="gray", label=None, alpha=0.5)

for line_label in labels.keys():
    values = {metric: np.full(len(n_homes), np.nan) for metric in metrics}
    time_end = np.full(len(n_homes), np.nan)
    colour = colours[line_label]
    if isinstance(runs[line_label], tuple):
        runs[line_label] = runs[line_label][0]
    for i, run in enumerate(runs[line_label]):
        if run is not None:
            for metric in metrics:
                try:
                    values[metric][i] = log.loc[log['run'] == run, f'{metric}_{type_learning[line_label]}'].values[0]
                except Exception as ex:
                    print(ex)
                time_end[i] = log.loc[log['run'] == run, 'time_end'].values[0]
    ls = '--' if opt_informed[line_label] else '-'
    label = None if opt_informed[line_label] else labels[line_label]

    axs[0].plot(
        n_homes,
        values[best_score_type],
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
    if line_label in compare_times:
        axs[1].plot(
            n_homes, time_end, 'o',
            color=colour
        )
        ys_time[line_label] = time_end
        function = polynomial2
        popt, pcov = optimize.curve_fit(function, n_homes, time_end)
        coeffs = popt
        xs = list(range(1, x_max))
        f_fitted = [function(x, *coeffs) for x in xs]
        label = r'$y = $' + f'{coeffs[0]:.2f}' + f'$x^2 $' + f' + {coeffs[1]:.2f}' + r'$x $' + f' + {coeffs[2]:.2f}'
        plt.plot(xs, f_fitted, label=label, color=colour, linestyle=ls)
        print(f"{line_label}: ")
        print(f"times : {time_end}")
        sum_square_error = sum((y_fitted - y) ** 2 for y_fitted, y in zip(f_fitted, time_end))
        print(f"sum square error {sum_square_error:.2e}")
        print(f"{label} a, b first errors = {np.sqrt(np.diag(pcov))}")

axs[0].set_ylabel("Savings [£/home/month]")
axs[0].axes.xaxis.set_ticklabels([])

axs[1].set_yscale('log')
axs[1].set_ylabel("Time [s]")
axs[1].set_xlabel("Number of homes")

axs[0].legend(fancybox=True)
axs[1].legend(fancybox=True)
fig.savefig(
    f"outputs/results_analysis/facmac_results3.pdf",
    bbox_inches='tight', format='pdf', dpi=1200
)
plt.close('all')

xs_facmac = [1, 3, 5, 10, 20, 50]
ys_facmac = [26.50374389, 38.85938716, 49.69802213, 73.43057513, 123.11931181, 281.53323126]
xs_iql = [1, 3, 5, 10, 20, 50]
# ys_iql = [133.11673403, 302.6728189, 328.64307976, 710.36226296, 1860.68218756, 8911.68623304]
ys_iql = [132.52754283, 297.46064401, 508.72118497, 1051.24059987, 2497.18453503, 9999.79186893]
for xs, ys, label in zip(
    [xs_facmac, xs_iql],
    [ys_facmac, ys_iql],
    ['facmac', 'iql']
):
    fig = plt.figure()
    plt.plot(xs, ys, 'o', label='data')

    functions = [
        first_order, second_order, second_order_b, exponential, polynomial2, polynomial3,
        unknown_order, xlogx
    ]
    labels_functions = [
        'first_order', 'second_order', 'second_order_b', 'exponential', 'polynomial2',
        'polynomial3', 'unknown_order', 'xlogx'
    ]
    for function, label_func in zip(functions, labels_functions):
        params, pcov = optimize.curve_fit(function, xs, ys)
        print(f"{label} params {label_func} = {np.sqrt(np.diag(pcov))}")
        fitted = [function(x, *params) for x in list(range(1, x_max))]
        sum_square_error = sum((y_fitted - y) ** 2 for y_fitted, y in zip(fitted, ys))
        print(f"sum square error {sum_square_error:.2e}")
        print(f"params {label} {label_func}: {params}")
        plt.plot(list(range(1, x_max)), fitted,
                 label=label_func)

    plt.legend()
    plt.gca().set_yscale('log')
    plt.title(label)
    fig.savefig(f"outputs/results_analysis/{label}_growth3")
    plt.close('all')
