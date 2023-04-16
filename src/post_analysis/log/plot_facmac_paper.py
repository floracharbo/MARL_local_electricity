
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from src.post_analysis.plotting.plot_rl_performance_metrics import _barplot

# plot timing vs performance for n layers / dim layers; runs 742-656
font = {'size': 14}
matplotlib.rc('font', **font)

compare_times = ['OMQ', 'FD']
labels = {
    'IQ': 'Independent Q-learning',
    'OIQ': 'Optimisation-informed independent Q-learning',
    'MQ': 'Marginal reward Q-learning',
    'OMQ': 'Optimisation-informed, marginal reward Q-learning',
    'FD': 'Centralised but factored critic',
    'FDO': 'Optimisation-informed  FACMAC (day-ahead)',
    # 'FH': 'FACMAC',  # (hourly)
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
# runQ = list(range(534, 539)) + [677]
# runQ = [534, 535, 536, 537, 538, 677]
runQ = list(range(1132, 1136)) + list(range(1137, 1138)) + list(range(1166, 1174)) + [1176]


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
    # 'FD': list(range(684, 690)),
    # 'FD': list(range(1089, 1091)) + list(range(1092, 1096)),
    'FD': list(range(1146, 1152)) + list(range(1158, 1166)),

    # 'FDO': list(range(257, 261)) + list(range(263, 265)),
    # 'FDO': list(range(528, 534)),
    # 'FDO': list(range(582, 587)) + [697],
    # 'FDO': [1115, 1116, 1119, 1121, 1122, 1123],
    'FDO': list(range(1152, 1158)) + [1174, 1175],
    # missing 30

    # 'FH': [126, 127, 128, 129, 130, 131],
    # 'FH': list(range(245, 251)),
    # 'FH': list(range(510, 516)),
    # 'FH': list(range(678, 684)),
    'FH': list(range(1103, 1109)),

    # 'FHO': [161, 162, 163, 164, 168, 169]
    # 'FHO': list(range(251, 257)),
    # 'FHO': list(range(522, 528)),
    # 'FHO': list(range(522, 527)),
    # 'FHO': list(range(771, 775)) + list(range(776, 778)),
    'FHO': list(range(1109, 1115))

}


green = (117 / 255, 189 / 255, 167 / 255)
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
blue = '#377eb8'  # colorblind blue
red = (192 / 255, 0, 0)

colours = {
    'IQ': grey,
    'OIQ': grey,
    'MQ': blue,
    'OMQ': blue,
    'FD': green,
    'FDO': green,
    'FH': red,
    'FHO': red,
}

# n_homes = [1, 3, 5, 10, 20, 50]
# n_homes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
n_homes = [1, 5, 10, 15, 20, 25, 30]

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

reliability_metric_labels = ['DT', 'SRT', 'LRT', 'DR', 'RR']
reliability_metrics = {line_label: {} for line_label in labels.keys()}

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot([0, n_homes[-1]], [0, 0], color="gray", label=None, alpha=0.5)

for line_label in labels.keys():
    reliability_metrics[line_label] = {
        reliability_metric_label: {}
        for reliability_metric_label in reliability_metric_labels
    }
    values = {metric: np.full(len(n_homes), np.nan) for metric in metrics}
    time_end = np.full(len(n_homes), np.nan)
    colour = colours[line_label]
    if isinstance(runs[line_label], tuple):
        runs[line_label] = runs[line_label][0]
    for i, run in enumerate(runs[line_label]):
        if run is not None:
            n_homes_i = log.loc[
                log['run'] == run, 'n_homes'
            ].values[0]
            if n_homes_i in n_homes:
                values_i = n_homes.index(n_homes_i)
                for metric in metrics:
                    values[metric][values_i] = log.loc[
                        log['run'] == run, f'{metric}_{type_learning[line_label]}'
                    ].values[0]
                    if n_homes_i == 10:
                        print(f"{line_label} {metric} {values[metric][values_i]}")
                    time_end[values_i] = log.loc[log['run'] == run, 'time_end'].values[0]
                print(
                    f"{line_label} n_homes {n_homes_i} run {run} "
                    f"{best_score_type} {values[best_score_type][values_i]} "
                    f"time_end {time_end[values_i]}"
                )

    last_run = runs[line_label][-1]
    reliability_metrics_run = np.load(
        f"outputs/results/run{last_run}/figures/metrics.npy", allow_pickle=True
    ).item()
    for reliability_metric_label in reliability_metric_labels:
        reliability_metrics[line_label][reliability_metric_label]['ave'] = \
            reliability_metrics_run[reliability_metric_label]['ave'][type_learning[line_label]]
        reliability_metrics[line_label][reliability_metric_label]['std'] = \
            reliability_metrics_run[reliability_metric_label]['std'][type_learning[line_label]]

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
        function = polynomial2 \
            if line_label == 'OMQ' else first_order
        popt, pcov = optimize.curve_fit(function, n_homes, time_end)
        coeffs = popt
        xs = list(range(1, x_max + 1))
        f_fitted = [function(x, *coeffs) for x in xs]
        if line_label == 'OMQ':
            label = r'$y = $'\
                + f'{coeffs[0]:.2f}' + r'$x^2 $'\
                + f' + {coeffs[1]:.2f}' + r'$x $'\
                + f' + {coeffs[2]:.2f}'
        else:
            label = r'$y = $' \
                    + f'{coeffs[0]:.1f}' + r'$x$' \
                    + f' + {coeffs[1]:.1f}'
        axs[1].plot(xs, f_fitted, label=label, color=colour, linestyle=ls)
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

axs[0].legend(fancybox=True, loc='lower right')
axs[1].legend(fancybox=True)
fig.savefig(
    "outputs/results_analysis/facmac_results3.pdf",
    bbox_inches='tight', format='pdf', dpi=1200
)
plt.close('all')


# figure metrics
subplots_i_j = {
    'LRT': [0, 2],
    'DR': [1, 0],
    'RR': [1, 1],
    'SRT': [0, 1],
    'DT': [0, 0]
}
fig, axs = plt.subplots(2, 3, figsize=(8, 10))
axs = [
    plt.subplot2grid((2, 6), (0, 0), 1, 2),
    plt.subplot2grid((2, 6), (0, 2), 1, 2),
    plt.subplot2grid((2, 6), (0, 4), 1, 2),

    plt.subplot2grid((2, 6), (1, 0), 1, 2),
    plt.subplot2grid((2, 6), (1, 3), 1, 2),
]
for i_subplot, reliability_metric_label in enumerate(reliability_metric_labels):
    bars, err = [
        [
            reliability_metrics[line_label][reliability_metric_label][s]
            for line_label in compare_times
        ]
        for s in ['ave', 'std']
    ]
    axs[i_subplot] = _barplot(
        bars, compare_times, None,
        # error=err,
        title=reliability_metric_label,
        ax0=axs[i_subplot],
        colours=[colours[line_label] for line_label in compare_times],
        text_labels=False
    )
    if i_subplot in [0, 3]:
        axs[i_subplot].set_ylabel('£/home/h')
# for i, method in enumerate(compare_times):
#     axs[1, 2].bar(i + 0.5, [0], label=labels[method], color=colours[method])
# axs[1, 2].legend(fancybox=True)
# axs[1, 2].set_yticks([])
# axs[1, 2].set_xticks([])

plt.tight_layout()

fig.savefig("outputs/results_analysis/metrics.pdf", bbox_inches='tight', format='pdf', dpi=1200)

# time facmac

# xs_facmac = [1, 3, 5, 10, 20, 50]
# ys_facmac = [26.50374389, 38.85938716, 49.69802213, 73.43057513, 123.11931181, 281.53323126]
# ys_facmac = [
# 67.54928350448610,
# 114.3641710281370,
# 160.202467918396,
# 274.6239285469060,
# 490.543824672699,
# 693.8394634723660
# ]

# xs_iql = [1, 3, 5, 10, 20, 50]
# ys_iql = [133.11673403, 302.6728189, 328.64307976, 710.36226296, 1860.68218756, 8911.68623304]
# ys_iql = [132.52754283, 297.46064401, 508.72118497, 1051.24059987, 2497.18453503, 9999.79186893]
# ys_iql = [
# 810.4848906993870,
# 1819.6897401809700,
# 2892.1081290245100,
# 6172.315644741060,
# 15143.180866479900,
# 27013.816419124600
# ]
for line_label in compare_times:
    xs = n_homes
    ys = ys_time[line_label]
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
        fitted = [function(x, *params) for x in list(range(1, x_max + 1))]
        sum_square_error = sum((y_fitted - y) ** 2 for y_fitted, y in zip(fitted, ys))
        print(f"sum square error {sum_square_error:.2e}")
        print(f"params {line_label} {label_func}: {params}")
        print(f"variability params {line_label}  {label_func} = {np.sqrt(np.diag(pcov))}")
        plt.plot(list(range(1, x_max + 1)), fitted,
                 label=label_func)

    plt.legend()
    plt.gca().set_yscale('log')
    plt.title(line_label)
    fig.savefig(f"outputs/results_analysis/{line_label}_growth3")
    plt.close('all')


# plot impact CNN / hysteretic
# runs 1186 1187 1200 1205
aves = [-13.07588752335410, 18.433919317843800,
        # 32.657462349269700,
        28.172149800077700]
p25s = [-23.30582077699420, -1.3533136375158700,
        # 45.20975524154160,
        45.20975524154160]
p75s = [47.42991041275560, 47.42991041275560,
        # 49.22962895821640,
        49.22962895821640]
runs = [1205, 1186,
        # 1187,
        1200]
all_data = np.zeros((10, 3))
for i, run in enumerate(runs):
    data = np.load(f"outputs/results_analysis/end_test_above_bl_env_r_c_run{run}.npy")
    all_data[:, i] = data
    print(f"run {run} ave {aves[i]:.2f} p25 {p25s[i]:.2f} p75 {p75s[i]:.2f}")
    print(
        f"from data ave {np.mean(data)}, "
        f"p25 {np.percentile(data, 25)}, "
        f"p75 {np.percentile(data, 75)}"
    )

labels = [
    'linear neural network, no hysteretic learning',
    'linear neural network, hysteretic learning',
    # 'convolutional neural network, no hysteretic learning',
    'convolutional neural network, hysteretic learning',
]
title = 'Ablations'

positions = np.arange(3) + 1

fig = plt.figure(figsize=(3, 4))

# matplotlib > 1.4
bp = plt.boxplot(all_data, positions=positions, showmeans=True, showfliers=False)

# plt.gca().set_xticklabels(labels, rotation=45, ha='right')
plt.tight_layout()

fig.savefig("outputs/results_analysis/ablations.pdf", bbox_inches='tight', format='pdf', dpi=1200)
