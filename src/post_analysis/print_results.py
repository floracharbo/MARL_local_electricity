"""Print key results statistics to screen / to txt file."""


def _print_str(str_, save_run, f=None):
    """Print string to file / to screen."""
    if save_run == 'both':
        f.write(str_ + ' \n')
        print(str_)
    elif save_run:
        f.write(str_ + ' \n')
    else:
        print(str_)

    return f


def _print_savings(metrics, prm):
    # different ways of expressing savings
    # A - savings per hour per agent relative to the baseline
    # B - savings per month per agent relative to the baseline
    # C - percentage saving relative to the baseline
    # D - percentage of optimal savings achieved
    for e in ['opt_d_d', 'opt_n_c']:
        e = e if prm['ntw']['n'] > 1 else e[:-1] + 'd'
        if e in metrics['end']['ave'].keys():
            for p in ['ave', 'p50']:
                A = metrics['end'][p][e] - metrics['end'][p]['baseline']
                B = A * 24 * 365 / 12
                C = A / (- metrics['end'][p]['baseline']) * 100
                D = A / (metrics['end'][p]['opt']
                         - metrics['end'][p]['baseline']) * 100
                print(f'{e}, {p}: A = {A}, B = {B}, C = {C}, D = {D}')


def _print_stats(f, prm, record, metrics):
    # 8 - print stats
    sum_duration = 0
    sum_duration_test = 0
    sum_n_not_feas = 0
    for repeat in range(prm['RL']['n_repeats']):
        for epoch in range(prm['RL']['n_all_epochs']):
            if epoch < prm['RL']['n_epochs']:
                sum_duration += record.duration_epoch[repeat][epoch]
            else:
                sum_duration_test += record.duration_epoch[repeat][epoch]

    average_duration = sum_duration / (prm['RL']['n_repeats']
                                       * prm['RL']['n_epochs'])
    average_n_not_feas = sum_n_not_feas / (prm['RL']['n_repeats']
                                           * prm['RL']['n_epochs'])
    f = _print_str(
        f"\nTested {prm['RL']['evaluation_methods']}",
        prm['save']['save_run'],
        f
    )
    f = _print_str(
        f"average duration of epoch is {average_duration} "
        f"for {prm['RL']['n_explore']} exploration(s),"
        f"parallel = {prm['RL']['parallel']}",
        prm['save']['save_run'],
        f
    )
    if prm['RL']['n_end_test'] > 0:
        average_duration_test = \
            sum_duration_test \
            / (prm['RL']['n_repeats'] * prm['RL']['n_end_test'])
        f = _print_str(
            f"average duration of epoch during testing"
            f" is {average_duration_test} "
            f"({prm['RL']['n_end_test']} days)",
            prm['save']['save_run'],
            f
        )
    f = _print_str(
        f"average number runs not feasible per run "
        f"feasible after feasibility checked with opt "
        f"is {average_n_not_feas}",
        prm['save']['save_run'],
        f
    )
    for e in metrics['end_test']['ave'].keys():
        baseline = metrics['end_test']['ave'][e] \
            - metrics['end_test']['ave']['baseline']
        f = _print_str(
            f"metrics['end_test']['ave'][{e}] - baseline {baseline}",
            "both",
            f
        )

    return f


def print_results(prm, f, record, metrics):
    """Print key statistics (both on screen and .txt file)."""
    f = _print_str(
        f"state_space = {prm['RL']['state_space']}",
        prm['save']['save_run'],
        f
    )
    # 8 - print stats
    f = _print_stats(f, prm, record, metrics)

    # 9 - print savings
    _print_savings(metrics, prm)

    return f
