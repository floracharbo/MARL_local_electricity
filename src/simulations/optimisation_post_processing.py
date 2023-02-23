import numpy as np

def add_val_to_res(res, var, val, size, arr):
    """Add value to result dict."""
    if size[0] < 2 and size[1] < 2:
        res[var] = val
    else:
        for i in range(size[0]):
            for j in range(size[1]):
                arr[i, j] = val[i, j]
        res[var] = arr

    return res

def add_home_time_step_pairs_to_list(
    total_list_homes, total_list_time_steps, new_list_homes, new_list_time_steps
):
    for home, time_step in zip(new_list_homes, new_list_time_steps):
        i_already_updated = np.where(
            (total_list_homes == home) & (total_list_time_steps == time_step)
        )[0]
        if len(i_already_updated) == 0:
            total_list_homes = np.append(total_list_homes, home)
            total_list_time_steps = np.append(total_list_time_steps, time_step)

    return total_list_homes, total_list_time_steps

def efficiencies(res, prm, bat_cap):
    """Compute efficiencies"""
    store = res['store']
    P_ch = res['charge']
    P_dis = res['discharge_tot']
    n_homes = len(store)

    P = (P_ch - P_dis) * 1e3
    SoC = np.zeros((n_homes, prm['N']))
    for home in range(n_homes):
        if bat_cap[home] == 0:
            SoC[home] = np.zeros(prm['N'])
        else:
            # in battery(times, bus)/cap(bus)
            SoC[home] = np.divide(store[home], bat_cap[home])
    a0 = - 0.852
    a1 = 63.867
    a2 = 3.6297
    a3 = 0.559
    a4 = 0.51
    a5 = 0.508

    b0 = 0.1463
    b1 = 30.27
    b2 = 0.1037
    b3 = 0.0584
    b4 = 0.1747
    b5 = 0.1288

    c0 = 0.1063
    c1 = 62.49
    c2 = 0.0437

    e0 = 0.0712
    e1 = 61.4
    e2 = 0.0288

    kappa = (130 * 215)

    # as a function of SoC
    Voc = a0 * np.exp(-a1 * SoC) \
        + a2 + a3 * SoC - a4 * SoC ** 2 + a5 * SoC ** 3
    Rs = b0 * np.exp(-b1 * SoC) \
        + b2 \
        + b3 * SoC \
        - b4 * SoC ** 2 \
        + b5 * SoC ** 3
    Rts = c0 * np.exp(-c1 * SoC) + c2
    Rtl = e0 * np.exp(-e1 * SoC) + e2
    Rt = Rs + Rts + Rtl

    # solve for current
    from sympy import Symbol
    from sympy.solvers import solve

    x = Symbol('x')
    i_cell = np.zeros(np.shape(P))
    eta = np.zeros(np.shape(P))
    for home in range(n_homes):
        for time_step in range(prm['N']):
            s = solve(
                P[home, time_step]
                + (x ** 2 - x * (Voc[home, time_step] / Rt[home, time_step]))
                * kappa * Rt[home, time_step],
                x)
            A = Rt[home, time_step] * kappa
            B = - Voc[home, time_step] * kappa
            C = P[home, time_step]
            s2_pos = (- B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A) \
                if A > 0 else 0
            s2_neg = (- B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A) \
                if A > 0 else 0
            s2 = [s2_pos, s2_neg]
            etas, etas2 = [], []
            for sign in range(2):
                if s[sign] == 0:
                    etas.append(0)
                    etas2.append(0)
                else:
                    etas.append(np.divide(
                        s[sign] * Voc[home, time_step],
                        s[sign] * (Voc[home, time_step] - s[sign] * Rt[home, time_step]))
                    )
                    etas2.append(np.divide(
                        s2[sign] * Voc[home, time_step],
                        s2[sign] * (Voc[home, time_step] - s2[sign] * Rt[home, time_step]))
                    )
            print(f'etas = {etas}, etas2={etas2}')
            eta[home, time_step] = etas[np.argmin(abs(etas - 1))]
            i_cell[home, time_step] = s[np.argmin(abs(etas - 1))]

    return eta, s, s2
