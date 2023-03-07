
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
    # 'FD': 'FACMAC (day-ahead)',
    'FDO': 'Optimisation-informed  FACMAC (day-ahead)',
    'FH': 'FACMAC',  # (hourly)
    # 'FHO': 'Optimisation-informed FACMAC (hourly)'
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
runs = {
    # 'OMQ': [170, 94, 96, 97, 98, 99],
    # 'OMQ': [236, 237, 238, 234,  240, 244],
    'OMQ': [420, 421, 422, 423, 424, 509],

    # 'IQ': [170, 94, 96, 97, 98, 99],
    # 'IQ': [236, 237, 238, 234,  240, 244],
    'IQ': [420, 421, 422, 423, 424, 509],
    # 'OIQ': [170, 94, 96, 97, 98, 99],
    # 'MQ': [170, 94, 96, 97, 98, 99],
    # 'OIQ': [236, 237, 238, 234,  240, 244],
    'OIQ': [420, 421, 422, 423, 424, 509],
    # 'MQ': [236, 237, 238, 234,  240, 244],
    'MQ': [420, 421, 422, 423, 424, 509],

    # 'FD': [160, 105, 106, 123, 124, 125],
    # 'FD': list(range(265, 271)),
    'FD': list(range(516, 522)),

    # 'FDO': list(range(257, 261)) + list(range(263, 265)),
    'FDO': list(range(528, 534)),

    # 'FH': [126, 127, 128, 129, 130, 131],
    # 'FH': list(range(245, 251)),
    'FH': list(range(510, 516)),

    # 'FHO': [161, 162, 163, 164, 168, 169]
    # 'FHO': list(range(251, 257)),
    'FHO': list(range(522, 528)),
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
    'FDO': red,
    'FH': red,
    'FHO': red,
}

n_homes = [1, 3, 5, 10, 20, 50]

results_analysis_path = Path("outputs/results_analysis")
log_path = results_analysis_path / "log_runs.csv"
log = pd.read_csv(log_path)
metrics = ['ave', 'p25', 'p75']

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
    for i, run in enumerate(runs[line_label]):
        if run is not None:
            for metric in metrics:
                values[metric][i] = log.loc[log['run'] == run, f'{metric}_{type_learning[line_label]}'].values[0]
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
    if line_label in compare_times:
        axs[1].plot(
            n_homes, time_end, 'o',
            color=colour
        )
        ys_time[line_label] = time_end
        function = polynomial2
        # if line_label == 'FH':
        #     label = 'first order: ' + r'$y = ax + b$'
        # else:
        #     label = 'second order: ' + r'$y = ax^2 + bx + c$'
        popt, pcov = optimize.curve_fit(function, n_homes, time_end)
        coeffs = popt
        xs = list(range(1, x_max))
        f_fitted = [function(x, *coeffs) for x in xs]
        label = r'$y = $' + f'{coeffs[0]:.2f}' + f'$x^2 $' + f' + {coeffs[1]:.2f}' + r'$x $' + f' + {coeffs[2]:.2f}'
        plt.plot(xs, f_fitted, label=label, color=colour, linestyle=ls)
        print(f"{line_label}: ")
        print(f"times : {time_end}")
        print(f"sum square error {sum((y_fitted - y) ** 2 for y_fitted, y in zip(f_fitted, time_end)):.2e}")
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
ys_iql = [ 132.52754283, 297.46064401, 508.72118497, 1051.24059987, 2497.18453503, 9999.79186893]
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

    popt, pcov = optimize.curve_fit(polynomial2, xs, ys)
    f, g, h = popt
    print(f"{label} e polynomial2 errors = {np.sqrt(np.diag(pcov))}")
    pol_fitted = [polynomial2(x, f, g, h) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y) ** 2 for y_fitted, y in zip(pol_fitted, ys)):.2e}")
    print(f"fitted polynomial2: {f} * x ** 2 + {g} * x + {h}")

    popt, pcov = optimize.curve_fit(unknown_order, xs, ys)
    i, j, k = popt
    print(f"{label} e unknown_order errors = {np.sqrt(np.diag(pcov))}")
    unknown_order_fitted = [unknown_order(x, i, j, k) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y) ** 2 for y_fitted, y in zip(unknown_order_fitted, ys)):.2e}")
    print(f"fitted unknown_order: {i} * x ** {j} + {k}")

    popt, pcov = optimize.curve_fit(polynomial3, xs, ys)
    l, m, n, o = popt
    print(f"{label} e polynomial3 errors = {np.sqrt(np.diag(pcov))}")
    polynomial3_fitted = [polynomial3(x, l, m, n, o) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y) ** 2 for y_fitted, y in zip(polynomial3_fitted, ys)):.2e}")
    print(f"fitted polynomial3: {l} * x ** 3 + {m} * x ** 2 + {n} * x + {o}")

    popt, pcov = optimize.curve_fit(xlogx, xs, ys)
    p, q = popt
    print(f"{label} e xlogx errors = {np.sqrt(np.diag(pcov))}")
    xlogx_fitted = [xlogx(x, p, q) for x in list(range(x_max))]
    print(f"sum square error {sum((y_fitted - y) ** 2 for y_fitted, y in zip(xlogx_fitted, ys)):.2e}")
    print(f"fitted xlogx: {p} * xlog(x) + {q}")

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
    plt.plot(list(range(x_max)), pol_fitted,
             label='polynomial2')
    plt.plot(list(range(x_max)), unknown_order_fitted,
             label='unknown_order')
    plt.plot(list(range(x_max)), polynomial3_fitted,
             label='polynomial3_fitted')
    plt.plot(list(range(x_max)), xlogx_fitted,
             label='xlogx_fitted')
    plt.legend()
    plt.gca().set_yscale('log')
    plt.title(label)
    fig.savefig(f"outputs/results_analysis/{label}_growth3")
    plt.close('all')
