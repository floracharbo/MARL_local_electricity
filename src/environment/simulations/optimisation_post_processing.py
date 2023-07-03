import numpy as np

from src.environment.utilities.userdeftools import (
    compute_import_export_costs, compute_voltage_costs,
    mean_max_hourly_voltage_deviations)


def _check_loads_are_met(constl_loads_constraints, prm):
    loads = prm['loads']
    N, n_homes, tol_constraints = [
        prm['syst'][info] for info in ['N', 'n_homes', 'tol_constraints']
    ]

    homes_to_update, time_steps_to_update = [np.array([], dtype=np.int) for _ in range(2)]

    slacks_constl_loads = np.array([
        [
            [
                constl_loads_constraints[load_type][home][time_step].slack
                for time_step in range(N)
            ]
            for home in range(n_homes)
        ]
        for load_type in range(loads['n_types'])
    ])
    load_types_slack_loads, homes_slack_loads, time_steps_slack_loads = np.where(
        slacks_constl_loads < - tol_constraints
    )
    if len(load_types_slack_loads) > 0:
        # loads are not met for homes_slack_loads.
        pp_simulation_required = True
        homes_to_update = np.append(homes_to_update, homes_slack_loads)
        time_steps_to_update = np.append(time_steps_to_update, time_steps_slack_loads)

    else:
        pp_simulation_required = False

    return pp_simulation_required, homes_to_update, time_steps_to_update


def _check_power_flow_equations(res, grd, N, input_hourly_lij):
    # power flow equations
    if grd['active_to_reactive_flex'] == 0:
        assert np.all(abs(res['q_car_flex']) < 1e-3)
        assert np.all(abs(res['qi']) < 1e-3)
    if grd['line_losses_method'] == 'iteration':
        res['lij'] = input_hourly_lij
    for time_step in range(N):
        abs_diffs = abs(
            res['pi'][:, time_step]
            - np.matmul(grd['flex_buses'], res['netp'][:, time_step])
            * grd['kW_to_per_unit_conversion']
        )
        assert np.all(abs_diffs < 1e-3)
        abs_pi_lij_constraint = abs(
            res['pi'][1:, time_step]
            - (
                - np.matmul(grd['incidence_matrix'][1:, :], res['pij'][:, time_step])
                + np.matmul(
                    np.matmul(
                        grd['in_incidence_matrix'][1:, :],
                        np.diag(grd['line_resistance'], k=0)
                    ),
                    res['lij'][:, time_step]
                )
            )
        )
        pi_lij_constraint_holds = np.all(abs_pi_lij_constraint < 1e-3)
        abs_pij0_constraint = abs(res['pij'][0] - res['grid'] * grd['kW_to_per_unit_conversion'])
        pij0_constraint_holds = np.all(abs_pij0_constraint < 1e-3)

        if grd['line_losses_method'] == 'iteration':
            if not pi_lij_constraint_holds or not pij0_constraint_holds:
                print("with iterations, not pi_lij_constraint_holds, not pij0_constraint_holds")
                print(f"max pij0_constraint gap: {np.max(abs_pij0_constraint)}")
                print(f"max abs_pi_lij_constraint gap: {np.max(abs_pi_lij_constraint)}")
        else:
            assert pi_lij_constraint_holds
            assert pij0_constraint_holds

        if grd['line_losses_method'] == 'subset_of_lines':
            assert np.all(
                abs(
                    res['v_line'][:, time_step]
                    - np.matmul(grd['out_incidence_matrix'].T, res['voltage_squared'][:, time_step])
                ) < 1e-3
            )
            for line in range(grd['subset_line_losses_modelled']):
                assert (
                    res['v_line'][line, time_step] * res['lij'][line, time_step] + 1e-3
                    >= res['pij'][line, time_step] * res['pij'][line, time_step]
                    + res['qij'][line, time_step] * res['qij'][line, time_step]
                )
        assert np.all(
            abs(
                res['voltage_squared'][1:, time_step]
                - (
                    np.matmul(
                        grd['bus_connection_matrix'][1:, :],
                        res['voltage_squared'][:, time_step]
                    )
                    + 2 * (
                        np.matmul(
                            np.matmul(
                                grd['in_incidence_matrix'][1:, :],
                                np.diag(grd['line_resistance'], k=0)
                            ),
                            res['pij'][:, time_step]
                        )
                        + np.matmul(
                            grd['in_incidence_matrix'][1:, :],
                            np.diag(grd['line_reactance'], k=0)
                        ) * res['qij'][:, time_step]
                    )
                    - np.matmul(
                        grd['in_incidence_matrix'][1:, :],
                        np.diag(np.square(grd['line_resistance']))
                        + np.diag(np.square(grd['line_reactance']))
                    ) * res['lij'][:, time_step]
                )
            ) < 1e-3
        )
        assert np.all(
            abs(
                res['line_losses_pu'][:, time_step]
                - np.diag(grd['line_resistance']) * res['lij'][:, time_step]
            ) < 1e-3
            for time_step in range(N)
        )


def _check_storage_equations(res, N, car, grd, syst):
    # storage constraints
    assert np.all(
        abs(
            res['discharge_tot']
            - (res['discharge_other'] / car['eta_dis'] + car['batch_loads_car'][:, 0: N])
        ) < 1e-3
    )
    store_end = car['SoC0'] * grd['Bcap'][:, N - 1]
    for time_step in range(N - 1):
        assert np.all(
            abs(
                res['charge'][:, time_step] - res['discharge_tot'][:, time_step]
                - (res['store'][:, time_step + 1] - res['store'][:, time_step])
            ) < 1e-3
        )
        assert np.all(
            res['store'][:, time_step + 1] + 1e-3 >= car['SoCmin']
            * grd['Bcap'][:, time_step] * car['batch_avail_car'][:, time_step]
        )
        assert np.all(
            res['store'][:, time_step + 1] + 1e-3
            >= car['baseld'] * car['batch_avail_car'][:, time_step]
        )

    assert np.all(
        res['charge'] <= car['batch_avail_car'][:, 0: N] * syst['M']
    )
    assert np.all(
        res['discharge_other']
        <= car['batch_avail_car'][:, 0: N] * syst['M']
    )
    assert np.all(
        res['store'] <= grd['Bcap'] + 1e-3
    )
    assert np.all(
        car['c_max'] + 1e-3 >= res['charge']
    )
    assert np.all(
        car['d_max'] + 1e-3 >= res['discharge_tot'])
    assert np.all(
        res['store'] + 1e-3 >= 0)
    assert np.all(
        res['discharge_other'] + 1e-3 >= 0)
    assert np.all(
        res['discharge_tot'] + 1e-3 >= 0)
    assert np.all(
        res['charge'] + 1e-3 >= 0)
    assert np.all(
        res['store'][:, N - 1]
        + res['charge'][:, N - 1]
        - res['discharge_tot'][:, N - 1]
        + 1e-3
        >= store_end
    )
    assert np.all(
        abs(res['store'][:, 0] - car['SoC0'] * grd['Bcap'][:, 0]) < 1e-3
    )
    assert np.all(
        res['p_car_flex'] ** 2 + res['q_car_flex'] ** 2 <= car['max_apparent_power_car']**2 + 1e-3
    )


def _check_cons_equations(res, N, loads, syst, grd):
    n_homes, tol_constraints = [syst[info] for info in ['n_homes', 'tol_constraints']]

    # positivity constraints
    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)
    for load_type in range(loads['n_types']):
        assert np.all(res[f'consa({load_type})'] >= - tol_constraints)
    assert np.all(res['totcons'] >= - 1e-3)

    for load_type in range(loads['n_types']):
        for home in range(n_homes):
            for time_step in range(N):
                # loads are met by constl
                assert (
                    abs(
                        np.sum(
                            [
                                res[f'constl({time_step}, {load_type})'][home, tC]
                                * grd['flex'][time_step, load_type, home, tC]
                                for tC in range(N)
                            ]
                        )
                        - grd['loads'][load_type, home, time_step]
                    ) < 1e-3
                ), \
                    f"constl not adding up to loads " \
                    f"home {home} time_step {time_step} load_type {load_type}"

                # constl adds up to consa
                assert (
                    abs(
                        np.sum(
                            [res[f'constl({tD}, {load_type})'][home, time_step] for tD in range(N)]
                        )
                        - res[f'consa({load_type})'][home, time_step]
                    ) < tol_constraints
                )

    # tot cons adds up
    assert np.all(
        abs(np.sum(res['totcons'][home, :] - res['E_heat'][home, :]) - np.sum(
            grd['loads'][:, home, :]) < syst['tol_constraints'])
        for home in range(n_homes)
    ), "still totcons minus E_geat not adding up to loads"


def _check_temp_equations(res, syst, heat):
    n_homes, N = [syst[info] for info in ['n_homes', 'N']]

    for home in range(n_homes):
        if heat['own_heat'][home]:
            assert res['T'][home, 0] == heat['T0']
            for time_step in range(N - 1):
                assert (
                    abs(
                        heat['T_coeff'][home][0]
                        + heat['T_coeff'][home][1] * res['T'][home, time_step]
                        + heat['T_coeff'][home][2] * heat['T_out'][time_step]
                        # heat['T_coeff'][home][3] * heat['phi_sol'][time_step]
                        + heat['T_coeff'][home][4] * res['E_heat'][home, time_step]
                        * 1e3 * syst['n_int_per_hr']
                        - res['T'][home, time_step + 1]
                    ) < 1e-3
                )
            assert np.all(
                abs(
                    heat['T_air_coeff'][home][0]
                    + heat['T_air_coeff'][home][1] * res['T'][home, :]
                    + heat['T_air_coeff'][home][2] * heat['T_out'][0: N]
                    # heat['T_air_coeff'][home][3] * heat['phi_sol'][time_step] +
                    + heat['T_air_coeff'][home][4] * res['E_heat'][home, :]
                    * 1e3 * syst['n_int_per_hr']
                    - res['T_air'][home, :]
                ) < 1e-3
            )
            assert np.all(res['T_air'][home, :] + 1e-3 >= heat['T_LB'][home, 0: N])
            assert np.all(res['T_air'][home, :] <= heat['T_UB'][home, 0: N] + 1e-3)
        else:
            assert np.all(abs(res['E_heat'][home]) < 1e-3)
            assert np.all(
                abs(
                    res['T_air'][home, :]
                    - (heat['T_LB'][home, 0: N] + heat['T_UB'][home, 0: N]) / 2
                ) < 1e-3
            )
            assert np.all(
                abs(
                    res['T'][home, :] - (heat['T_LB'][home, 0: N] + heat['T_UB'][home, 0: N]) / 2
                ) < 1e-3
            )

        assert np.all(res['E_heat'] + 1e-3 >= 0)


def check_constraints_hold(res, prm, input_hourly_lij=None):
    N, n_homes, tol_constraints = [
        prm['syst'][info] for info in ['N', 'n_homes', 'tol_constraints']
    ]
    grd, loads, car, syst, heat = [
        prm[info] for info in ['grd', 'loads', 'car', 'syst', 'heat']
    ]

    assert np.all(
        abs(
            res['grid']
            - np.sum(res['netp'], axis=0)
            - loads['hourly_tot_netp0']
            - res['hourly_line_losses_pu'] * prm['grd']['per_unit_to_kW_conversion']
        ) < 1e-3
    )
    assert np.all(
        abs(res['grid2'] - np.square(res['grid'])) < 1e-2
    )
    # _check_power_flow_equations(res, grd, N, input_hourly_lij)
    _check_storage_equations(res, N, car, grd, syst)
    _check_cons_equations(res, N, loads, syst, grd)
    _check_temp_equations(res, syst, heat)


def _add_val_to_res(res, var, val, size, arr):
    """Add value to result dict."""
    if size[0] < 2 and size[1] < 2:
        res[var] = val
    else:
        for i in range(size[0]):
            for j in range(size[1]):
                arr[i, j] = val[i, j]
        res[var] = arr

    return res


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

    return eta


def _add_home_time_step_pairs_to_list(
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


def _check_constl_to_consa(
        constl_consa_constraints, res, pp_simulation_required,
        homes_to_update, time_steps_to_update, prm
):
    loads = prm['loads']
    N, n_homes, tol_constraints = [
        prm['syst'][info] for info in ['N', 'n_homes', 'tol_constraints']
    ]

    if pp_simulation_required:
        # "as we have already had to make changes in constl, we are not checking the slack of the "
        # "original optimisation changes, but rather checking whether the equalities with updated "
        # "variables hold for the translation of constl to consa"
        load_types_slack, homes_slack, time_steps_slack = [
            np.array([], dtype=np.int) for _ in range(3)
        ]
        max_violation = 0
        for load_type in range(loads['n_types']):
            for home in range(n_homes):
                for time_step in range(N):
                    delta = abs(
                        np.sum(
                            [res[f'constl({tD}, {load_type})'][home, time_step] for tD in range(N)]
                        )
                        - res[f'consa({load_type})'][home, time_step]
                    )
                    if delta > tol_constraints:
                        load_types_slack = np.append(load_types_slack, load_type)
                        homes_slack = np.append(homes_slack, home)
                        time_steps_slack = np.append(time_steps_slack, time_step)
                        max_violation = max(max_violation, delta)

    else:
        # checking the slack of optimisation constraints for translating constl to consa
        slacks_constl_consa = np.array([
            [
                [
                    constl_consa_constraints[load_type][home][time_step].slack
                    for time_step in range(N)
                ]
                for home in range(n_homes)
            ]
            for load_type in range(loads['n_types'])
        ])

        load_types_slack, homes_slack, time_steps_slack = np.where(
            slacks_constl_consa < - tol_constraints
        )
        max_violation = max(abs(np.min(slacks_constl_consa)), np.max(slacks_constl_consa))
    if len(load_types_slack) > 0:
        # these conslt do not add up to consa: load_types_slack, homes_slack, time_steps_slack
        homes_to_update, time_steps_to_update = _add_home_time_step_pairs_to_list(
            homes_to_update, time_steps_to_update, homes_slack, time_steps_slack
        )
        res['max_cons_slack'] = max_violation
        pp_simulation_required = True
    else:
        res['max_cons_slack'] = 0

    for load_type, home, time_step in zip(
        load_types_slack, homes_slack, time_steps_slack
    ):
        constl_tD_lt = np.array(
            [
                res[f'constl({tD}, {int(load_type)})'][home, time_step]
                for tD in range(N)
            ]
        )
        res[f'consa({load_type})'][home, time_step] = sum(constl_tD_lt)

    return res, pp_simulation_required, homes_to_update, time_steps_to_update


def _check_constl_non_negative(
    res, pp_simulation_required, homes_to_update, time_steps_to_update, prm
):
    N, n_homes, tol_constraints = [
        prm['syst'][info] for info in ['N', 'n_homes', 'tol_constraints']
    ]
    grd, loads = [prm[info] for info in ['grd', 'loads']]

    for tD in range(N):
        homes_neg_constl, tC_neg_constl = np.where(
            res[f'constl({tD}, 0)'] < - 1e-3
        )
        for home, tC in zip(homes_neg_constl, tC_neg_constl):
            if grd['flex'][tD, 0, home, tC]:
                print(
                    "ERROR: there are negative consumptions at relevant times for fixed cons"
                    f"tD {tD} home {home} tC {tC}"
                )
            else:
                # fixed constl negative for tD {tD} homes_neg_constl, tC_neg_constl {home, tC}
                # this number was multiplied by a zero flex coefficient
                # so it should not matter anyway.
                # Setting it to zero and no further action taken.
                res[f'constl({tD}, 0)'][home, tC] = 0

        homes_neg_constl, time_step_neg_constl = np.where(
            res[f'constl({tD}, 1)'] < - 1e-3
        )
        if len(homes_neg_constl) > 0:
            pp_simulation_required = True
            # constl negative for tD, homes_neg_constl, time_step_neg_constl
            for home, time_cons in zip(homes_neg_constl, time_step_neg_constl):
                if not grd['flex'][tD, 1, home, time_cons]:
                    # this number was multiplied by a zero flex coefficient
                    # so it should not matter anyway
                    # Setting it to zero and no further action taken
                    res[f'constl({tD}, 1)'][home, time_cons] = 0
                else:
                    # we are adding {- res[f'constl({tD}, 1)'][home, time_cons]}
                    # to res[f'constl({tD}, 1)'][home={home}, time_step={time_cons}]
                    # to make it 0
                    # We will reduce the consumption at the other consumption steps
                    # matching this demand evenly.\n"
                    # The new consa given updated constl should be computed at the next step.
                    window_cons_time_steps = []
                    for potential_time_cons in range(N):
                        if (
                            grd['flex'][tD, 1, home, potential_time_cons]
                            and potential_time_cons != time_cons
                        ):
                            window_cons_time_steps.append(potential_time_cons)
                    window_cons_time_steps = np.array(window_cons_time_steps)
                    constl_other_time_steps = res[f'constl({tD}, 1)'][home, window_cons_time_steps]
                    i_sorted = np.argsort(constl_other_time_steps)
                    window_other_cons_time_steps_ordered = window_cons_time_steps[i_sorted]
                    n_other_time_cons = len(window_other_cons_time_steps_ordered)
                    total_to_remove = abs(res[f'constl({tD}, 1)'][home, time_cons])
                    total_left_to_remove = abs(res[f'constl({tD}, 1)'][home, time_cons])
                    even_split_for_remaining_time_cons = total_left_to_remove / n_other_time_cons
                    to_remove_each_time_cons = np.zeros(n_other_time_cons)

                    for i, time_cons_other in enumerate(window_other_cons_time_steps_ordered):
                        if (
                            res[f'constl({tD}, 1)'][home, time_cons_other]
                            < even_split_for_remaining_time_cons
                        ):
                            to_remove_each_time_cons[i] \
                                = res[f'constl({tD}, 1)'][home, time_cons_other]
                            total_left_to_remove -= to_remove_each_time_cons[i]
                            n_other_time_cons -= 1
                            even_split_for_remaining_time_cons \
                                = total_left_to_remove / n_other_time_cons
                        else:
                            to_remove_each_time_cons[i] = even_split_for_remaining_time_cons
                        # "remove {to_remove_each_time_cons[i]}
                        # from res[f'constl({tD}, 1)'][home={home}, time_cons={time_cons_other}]")
                        res[f'constl({tD}, 1)'][home, time_cons_other] \
                            -= to_remove_each_time_cons[i]
                    res[f'constl({tD}, 1)'][home, time_cons] = 0
                    assert sum(to_remove_each_time_cons) == total_to_remove

            homes_to_update, time_steps_to_update = _add_home_time_step_pairs_to_list(
                homes_to_update, time_steps_to_update, homes_neg_constl, time_step_neg_constl
            )

    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)
    return res, pp_simulation_required, homes_to_update, time_steps_to_update


def _update_res_variables(
        res, homes_to_update, time_steps_to_update, time_steps_grid,
        pp_simulation_required, prm, input_hourly_lij, test
):
    grd, loads, car = [prm[info] for info in ['grd', 'loads', 'car']]
    ext = '_test' if test else ''
    N = prm['syst']['N']
    for home, time_step in zip(
        homes_to_update, time_steps_to_update
    ):
        res['totcons'][home, time_step] = sum(
            [
                res[f'consa({load_type})'][home, time_step]
                for load_type in range(loads['n_types'])
            ]
        ) + res['E_heat'][home, time_step]
        res['netp'][home, time_step] = \
            res['charge'][home, time_step] / car['eta_ch'] \
            - res['discharge_other'][home, time_step] \
            - grd['gen'][home, time_step] \
            + res['totcons'][home, time_step]
        res['netq_flex'][home, time_step] = \
            res['q_car_flex'][home, time_step] \
            + res['totcons'][home, time_step] * grd['active_to_reactive_flex'] \
            - grd['gen'][home, time_step] * grd['active_to_reactive_flex']
        if grd['penalise_individual_exports']:
            res['netp_export'][home, time_step] = np.where(
                res['netp'][home, time_step] < 0, abs(res['netp'][home, time_step]), 0
            )
    new_grid = \
        np.sum(res['netp'], axis=0) + loads['hourly_tot_netp0'] \
        + res['hourly_line_losses_pu'] * grd['per_unit_to_kW_conversion']
    if np.any(abs(res['grid'] - new_grid) > 1e-3) or len(time_steps_grid) > 0:
        # update grid and grid2 and grid_energy_costs and pi
        res['grid'] = new_grid
        res['grid2'] = np.square(res['grid'])
        new_grid_energy_costs = np.sum(
            np.multiply(grd['C'][0: N], (res['grid'] + grd['loss'] * res['grid2']))
        )
        delta = new_grid_energy_costs - res['grid_energy_costs']
        res['grid_energy_costs'] = new_grid_energy_costs
        res['total_costs'] += delta
        if grd['line_losses_method'] == 'iteration':
            res['lij'] = input_hourly_lij
        if grd['manage_voltage']:
            for time_step in range(N):
                res['q_ext_grid'][time_step] = \
                    sum(res['netq_flex'][:, time_step]) \
                    + sum(
                        np.matmul(np.diag(grd['line_reactance'], k=0), res['lij'][:, time_step])
                    ) * grd['per_unit_to_kW_conversion'] \
                    + sum(loads['q_heat_home_car_passive'][:, time_step])
                res['pi'][:, time_step] = \
                    np.matmul(grd['flex_buses' + ext], res['netp'][:, time_step]) \
                    * grd['kW_to_per_unit_conversion']
                res['qi'][:, time_step] = \
                    np.matmul(grd['flex_buses' + ext], res['netq_flex'][:, time_step]) \
                    * grd['kW_to_per_unit_conversion']

        pp_simulation_required = True

    cons_difference = abs(res['consa(0)'] + res['consa(1)'] + res['E_heat'] - res['totcons'])
    assert np.all(cons_difference < 1e-2), \
        f"Consumption does not add up: {cons_difference[cons_difference > 1e-2]}"

    return res, pp_simulation_required


def _check_consa_to_totcons_netp_grid(
        res, pp_simulation_required, homes_to_update, time_steps_to_update, prm
):
    tol_constraints, N = [prm['syst'][info] for info in ['tol_constraints', 'N']]
    car, grd, loads = [prm[info] for info in ['car', 'grd', 'loads']]

    homes, time_steps = np.where(
        abs(res['consa(0)'] + res['consa(1)'] + res['E_heat'] - res['totcons']) > tol_constraints
    )

    if len(homes) > 0:
        pp_simulation_required = True
    homes_to_update, time_steps_to_update = _add_home_time_step_pairs_to_list(
        homes_to_update, time_steps_to_update, homes, time_steps
    )
    homes, time_steps = np.where(
        abs(
            res['charge'] / car['eta_ch']
            - res['discharge_other']
            - grd['gen'][:, 0: N]
            + res['totcons']
            - res['netp']
        ) > tol_constraints
    )
    if len(homes) > 0:
        pp_simulation_required = True
    homes_to_update, time_steps_to_update = _add_home_time_step_pairs_to_list(
        homes_to_update, time_steps_to_update, homes, time_steps
    )
    time_steps_grid = np.where(
        abs(
            res['grid']
            - loads['hourly_tot_netp0']
            - np.sum(res['netp'], axis=0)
            - res['hourly_line_losses_pu'] * grd['per_unit_to_kW_conversion']
        ) > tol_constraints
    )[0]
    if len(time_steps_grid) > 0:
        pp_simulation_required = True
    time_steps_grid2 = np.where(
        abs(np.square(res['grid']) - res['grid2']) > tol_constraints
    )[0]
    time_steps_grid = \
        list(time_steps_grid) + list(set(time_steps_grid2) - set(time_steps_grid))

    return pp_simulation_required, homes_to_update, time_steps_to_update, time_steps_grid


def check_and_correct_constraints(
    res, constl_consa_constraints, constl_loads_constraints, prm, input_hourly_lij=None, evaluation=False
):
    N = prm['syst']['N']
    loads = prm['loads']
    # 1 - check that loads are met
    pp_simulation_required, homes_to_update, time_steps_to_update = _check_loads_are_met(
        constl_loads_constraints, prm
    )

    # 2 - check that constl are non-negative
    res, pp_simulation_required, homes_to_update, time_steps_to_update = _check_constl_non_negative(
        res, pp_simulation_required, homes_to_update, time_steps_to_update, prm
    )
    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)
    # 3 - check that const translates into consa
    res, pp_simulation_required, homes_to_update, time_steps_to_update \
        = _check_constl_to_consa(
            constl_consa_constraints, res, pp_simulation_required,
            homes_to_update, time_steps_to_update, prm
        )
    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)
    pp_simulation_required, homes_to_update, time_steps_to_update, time_steps_grid \
        = _check_consa_to_totcons_netp_grid(
            res, pp_simulation_required, homes_to_update, time_steps_to_update, prm
        )
    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)

    # 4 - update tot_cons and grid etc
    res, pp_simulation_required = _update_res_variables(
        res, homes_to_update, time_steps_to_update, time_steps_grid,
        pp_simulation_required, prm, input_hourly_lij, evaluation
    )
    for time_step in range(N):
        for load_type in range(loads['n_types']):
            assert np.all(res[f'constl({time_step}, {load_type})'] >= - 1e-3)
    assert np.all(
        abs(res['grid2'] - np.square(res['grid'])) < 1e-2
    )
    # 5 - check constraints hold
    check_constraints_hold(res, prm, input_hourly_lij)

    return res, pp_simulation_required


def res_post_processing(res, prm, input_hourly_lij, perform_checks):
    N, n_homes, tol_constraints = [
        prm['syst'][info] for info in ['N', 'n_homes', 'tol_constraints']
    ]
    grd, syst, loads, car = [prm[info] for info in ['grd', 'syst', 'loads', 'car']]

    res['house_cons'] = res['totcons'] - res['E_heat']
    if grd['manage_agg_power']:
        res['hourly_import_export_costs'] = \
            res['hourly_import_costs'] + res['hourly_export_costs']
    else:
        res['hourly_import_costs'] = np.zeros(N)
        res['hourly_export_costs'] = np.zeros(N)
        res['hourly_import_export_costs'] = np.zeros(N)

    if syst['n_homesP'] > 0:
        res['netp0'] = loads['netp0']
        res['netq0'] = loads['netp0'] * grd['active_to_reactive_passive']
    else:
        res['netp0'] = np.zeros([1, N])
        res['netq0'] = np.zeros([1, N])

    if grd['manage_voltage']:
        res['mean_voltage_deviation'] = []
        res['max_voltage_deviation'] = []
        res['n_voltage_deviation_bus'] = []
        res['n_voltage_deviation_hour'] = []
        res['voltage_squared'][abs(res['voltage_squared']) < 1e-6] = 0
        res['voltage'] = np.sqrt(res['voltage_squared'])
        res['hourly_voltage_costs'] = np.sum(
            res['overvoltage_costs'] + res['undervoltage_costs'], axis=0
        )
        res['hourly_line_losses'] = \
            res['hourly_line_losses_pu'] * grd['per_unit_to_kW_conversion']
        if grd['line_losses_method'] == 'iteration':
            res['lij'] = input_hourly_lij
            res['v_line'] = np.matmul(
                grd['out_incidence_matrix'].T, res['voltage_squared'])
            res['grid'] = np.sum(res['netp'], axis=0) \
                + res['hourly_line_losses'] \
                + np.sum(res['netp0'], axis=0)
            res['hourly_reactive_line_losses'] = \
                np.sum(np.matmul(np.diag(grd['line_reactance'], k=0), res['lij']), axis=0) \
                * grd['per_unit_to_kW_conversion']
            res['q_ext_grid'] = np.sum(res['netq_flex'], axis=0) \
                + res['hourly_reactive_line_losses'] \
                + np.sum(res['netq0'], axis=0)
        res['p_solar_flex'] = grd['gen'][:, 0: N]
        res['q_solar_flex'] = grd['gen'][:, 0: N] * grd['active_to_reactive_flex']
        res["hourly_reactive_losses"] = \
            np.sum(np.matmul(np.diag(grd['line_reactance'], k=0), res['lij'][:, 0: N])
                   * grd['per_unit_to_kW_conversion'], axis=0)
        for time_step in range(N):
            res['hourly_import_export_costs'][time_step], _, _ = compute_import_export_costs(
                res['grid'][time_step], prm['grd'], prm['syst']['n_int_per_hr']
            )
            res['hourly_voltage_costs'][time_step] = \
                compute_voltage_costs(res['voltage_squared'][:, time_step], prm['grd'])
            (mean, max, n_bus, n_hour) = \
                mean_max_hourly_voltage_deviations(
                res['voltage_squared'][:, time_step],
                prm['grd']['max_voltage'],
                prm['grd']['min_voltage'],
            )
            res['mean_voltage_deviation'].append(mean)
            res['max_voltage_deviation'].append(max)
            res['n_voltage_deviation_bus'].append(n_bus)
            res['n_voltage_deviation_hour'].append(n_hour)

    else:
        res['voltage_squared'] = np.zeros((1, N))
        res['voltage_costs'] = 0
        res['hourly_voltage_costs'] = np.zeros(N)
        res['hourly_line_losses'] = np.zeros(N)
        res['q_ext_grid'] = np.zeros(N)

    res['hourly_grid_energy_costs'] = grd['C'][0: N] * (
        res["grid"] + grd["loss"] * res["grid2"]
    )
    res['hourly_battery_degradation_costs'] = car["C"] * (
        np.sum(res["discharge_tot"] + res["charge"], axis=0)
        + np.sum(loads['discharge_tot0'], axis=0)
        + np.sum(loads['charge0'], axis=0)
    )
    if grd['penalise_individual_exports']:
        res['hourly_distribution_network_export_costs'] = grd["export_C"] * (
            np.sum(res["netp_export"], axis=0)
            + np.sum(loads['netp0_export'], axis=0)
        )
    else:
        res['hourly_distribution_network_export_costs'] = np.zeros(N)

    res['hourly_total_costs'] = \
        (res['hourly_import_export_costs'] + res['hourly_voltage_costs']) \
        * grd["weight_network_costs"] \
        + res['hourly_grid_energy_costs'] \
        + res['hourly_battery_degradation_costs'] \
        + res['hourly_distribution_network_export_costs']

    res['grid_energy_costs'] = sum(res['hourly_grid_energy_costs'])
    res['total_costs'] = sum(res['hourly_total_costs'])

    if perform_checks:
        for key, val in res.items():
            if key[0: len('hourly')] == 'hourly':
                assert len(val) == N, f"np.shape(res[{key}]) = {np.shape(val)}"

        assert np.all(res['totcons'] > - 5e-3), f"min(res['totcons']) = {np.min(res['totcons'])}"

        simultaneous_dis_charging = \
            np.logical_and(res['charge'] > 1e-3, res['discharge_other'] > 1e-3)
        assert not simultaneous_dis_charging.any(), \
            "Simultaneous charging and discharging is happening" \
            f"For charging of {res['charge'][simultaneous_dis_charging]}" \
            f"and discharging of {res['discharge_other'][simultaneous_dis_charging]}"

        assert np.all(res['consa(1)'] > - syst['tol_constraints']), \
            f"negative flexible consumptions in the optimisation! " \
            f"np.min(res['consa(1)']) = {np.min(res['consa(1)'])}"
        max_losses_condition = np.logical_and(
            res['hourly_line_losses'] > 1,
            res['hourly_line_losses'] > 0.15 * abs(res['grid'] - res['hourly_line_losses'])
        )
        assert np.all(~max_losses_condition), \
            f"Hourly line losses are larger than 15% of the total import. " \
            f"Losses: {res['hourly_line_losses'][~(max_losses_condition)]} kWh " \
            f"Grid imp/exp: " \
            f"{abs(res['grid'] - res['hourly_line_losses'])[~(max_losses_condition)]} kWh."

    return res


def save_results(pvars, prm):
    """Save optimisation results to file."""
    res = {}
    constls1, constls0 = [], []
    for var in pvars:
        if var[0:6] == 'constl':
            if var[-2] == '1':
                constls1.append(var)
            elif var[-2] == '0':
                constls0.append(var)
        size = pvars[var].size
        val = pvars[var].value
        arr = np.zeros(size)
        res = _add_val_to_res(res, var, val, size, arr)

    for key, val in res.items():
        if len(np.shape(val)) == 2 and np.shape(val)[1] == 1:
            res[key] = res[key][:, 0]

    if prm['save']['saveresdata']:
        np.save(prm['paths']['record_folder'] / 'res', res)

    return res
