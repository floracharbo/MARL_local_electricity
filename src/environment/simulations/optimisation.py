#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  7 17:10:28 2020.

@author: floracharbonnier

"""

import copy

import numpy as np
import picos as pic

from src.environment.simulations.optimisation_post_processing import (
    check_and_correct_constraints, efficiencies, res_post_processing,
    save_results)
from src.environment.utilities.userdeftools import comb


class Optimiser:
    """The Optimiser object manages convex optimisations."""

    def __init__(self, prm, compare_optimiser_pandapower):
        """Initialise solver object for doing optimisations."""
        for attribute in ['N', 'n_homes', 'tol_constraints', 'n_homesP']:
            setattr(self, attribute, prm['syst'][attribute])
        for info in [
            'manage_agg_power', 'penalise_individual_exports', 'reactive_power_for_voltage_control',
            'per_unit_to_kW_conversion', 'kW_to_per_unit_conversion'
        ]:
            setattr(self, info, prm['grd'][info])
        self.save = prm["save"]
        self.paths = prm["paths"]
        self.compare_optimiser_pandapower = compare_optimiser_pandapower
        self.prm = prm
        self.input_hourly_lij = None

    def solve(self, prm, test=False):
        """Solve optimisation problem given prm input data."""
        self._update_prm(prm)
        self.n_homes = prm['syst']['n_homes_test'] if test else prm['syst']['n_homes']
        self.ext = '_test' if test else ''
        if self.grd['manage_voltage'] and self.grd['line_losses_method'] == 'iteration':
            res, pp_simulation_required = self._solve_line_losses_iteration(test)
        else:
            res, pp_simulation_required, _, _ = self._problem(test)
            perform_checks = True
            res = res_post_processing(res, prm, self.input_hourly_lij, perform_checks)

        if prm['car']['efftype'] == 1:
            res = self._car_efficiency_iterations(prm, res, test)
            res = res_post_processing(res, prm, self.input_hourly_lij, perform_checks)

        return res, pp_simulation_required

    def _solve_line_losses_iteration(self, evaluation):
        it = 0
        self.input_hourly_lij = np.zeros((self.grd['n_lines'], self.N))
        res, _, constl_consa_constraints, constl_loads_constraints = self._problem(evaluation)
        perform_checks = False
        res = res_post_processing(res, self.prm, self.input_hourly_lij, perform_checks)
        opti_voltages = copy.deepcopy(res['voltage'])
        opti_losses = copy.deepcopy(res['hourly_line_losses'])
        for time_step in range(self.N):
            if self.n_homesP > 0:
                netp0 = self.loads['netp0'][:, time_step]
            else:
                netp0 = np.zeros([1, self.N])
            grdCt = self.grd['C'][time_step]
            res = self.compare_optimiser_pandapower(
                res, time_step, netp0, grdCt)
        corr_voltages = copy.deepcopy(res['voltage'])
        corr_losses = copy.deepcopy(res['hourly_line_losses'])
        corr_lij = copy.deepcopy(res['lij'])
        delta_losses = opti_losses - corr_losses
        delta_voltages = opti_voltages - corr_voltages
        print(f"max hourly delta voltages initialization: {abs(delta_voltages).max()}")
        print(f"max hourly delta losses initialization: {abs(delta_losses).max()}")
        while abs(delta_voltages).max() > self.grd['tol_voltage_iteration'] and it < 10:
            it += 1
            self.input_hourly_lij = corr_lij
            res, pp_simulation_required, constl_consa_constraints, constl_loads_constraints = \
                self._problem(evaluation)
            res = res_post_processing(res, self.prm, self.input_hourly_lij, perform_checks)
            opti_voltages = copy.deepcopy(res['voltage'])
            opti_losses = copy.deepcopy(res['hourly_line_losses'])
            for time_step in range(self.N):
                if self.n_homesP > 0:
                    netp0 = self.loads['netp0'][:, time_step]
                else:
                    netp0 = np.zeros(1)
                grdCt = self.grd['C'][time_step]
                res = self.compare_optimiser_pandapower(
                    res, time_step, netp0, grdCt)
            corr_voltages = copy.deepcopy(res['voltage'])
            corr_losses = copy.deepcopy(res['hourly_line_losses'])
            corr_lij = copy.deepcopy(res['lij'])
            delta_losses = opti_losses - corr_losses
            delta_voltages = opti_voltages - corr_voltages
            print(f"max hourly delta voltages iteration {it}: {abs(delta_voltages).max()}")
            print(f"max hourly delta losses iteration {it}: {abs(delta_losses).max()}")

        res, pp_simulation_required = check_and_correct_constraints(
            res, constl_consa_constraints, constl_loads_constraints,
            self.prm, corr_lij, evaluation=evaluation
        )
        perform_checks = True
        res = res_post_processing(res, self.prm, res['lij'], perform_checks)
        return res, pp_simulation_required

    def _car_efficiency_iterations(self, prm, res, evaluation):
        init_eta = prm['car']['etach']
        prm['car']['etach'] = efficiencies(
            res, prm, prm['car']['caps']
        )
        deltamax, its = 0.5, 0
        prm['car']['eff'] = 2
        while deltamax > 0.01 and its < 10:
            its += 1
            eta_old = copy.deepcopy(prm['car']['etach'])
            print(f"prm['grd']['loads'][0][0][0] = "
                  f"{prm['grd']['loads'][0][0][0]}")
            res, _, _, _ = self._problem(evaluation)
            print(f"res['constl(0, 0)'][0][0] "
                  f"= {res['constl(0, 0)'][0][0]}")
            if prm['grd']['loads'][0][0][0] < res['constl(0, 0)'][0][0]:
                print('fixed loads smaller than fixed consumption home=0 time=0')
            if abs(np.sum(res['totcons']) - np.sum(res['E_heat'])
                   - np.sum(prm['grd']['loads'])) > 1e-3:
                print(f"tot load cons "
                      f"{np.sum(res['totcons']) - np.sum(res['E_heat'])} "
                      f"not equal to loads {np.sum(prm['loads'])}")
            prm['car']['etach'] = efficiencies(
                res, prm, prm['car']['caps'])
            deltamax = np.amax(abs(prm['car']['etach'] - eta_old))
        prm['car']['etach'] = init_eta
        prm['car']['eff'] = 1

        return res

    def _power_flow_equations(
            self, p, netp, grid, hourly_line_losses_pu,
            charge, discharge_other, totcons
    ):
        # power flows variables
        voltage_costs = p.add_variable('voltage_costs', 1)  # daily voltage violation costs
        pi = p.add_variable('pi', (self.grd['n_buses'] - 1, self.N), vtype='continuous')
        netq_flex = p.add_variable('netq_flex', (self.n_homes, self.N), vtype='continuous')
        q_car_flex = p.add_variable('q_car_flex', (self.n_homes, self.N), vtype='continuous')
        p_car_flex = p.add_variable('p_car_flex', (self.n_homes, self.N), vtype='continuous')
        if self.reactive_power_for_voltage_control:
            q_car_flex2 = p.add_variable('q_car_flex2', (self.n_homes, self.N), vtype='continuous')
            p_car_flex2 = p.add_variable('p_car_flex2', (self.n_homes, self.N), vtype='continuous')
        qi = p.add_variable('qi', (self.grd['n_buses'] - 1, self.N), vtype='continuous')
        pij = p.add_variable('pij', (self.grd['n_lines'], self.N), vtype='continuous')
        qij = p.add_variable('qij', (self.grd['n_lines'], self.N), vtype='continuous')
        voltage_squared = p.add_variable(
            'voltage_squared', (self.grd['n_buses'] - 1, self.N), vtype='continuous'
        )
        q_ext_grid = p.add_variable('q_ext_grid', self.N, vtype='continuous')
        line_losses_pu = p.add_variable(
            'line_losses_pu', (self.grd['n_lines'], self.N), vtype='continuous'
        )
        if self.grd['line_losses_method'] == 'subset_of_lines':
            # lij: square of the complex current
            lij = p.add_variable('lij', (self.grd['n_lines'], self.N), vtype='continuous')
            v_line = p.add_variable('v_line', (self.grd['n_lines'], self.N), vtype='continuous')
        else:
            lij = self.input_hourly_lij

        # decision variables: hourly voltage penalties for the whole network
        overvoltage_costs = p.add_variable(
            'overvoltage_costs', (self.grd['n_buses'] - 1, self.N), vtype='continuous'
        )
        undervoltage_costs = p.add_variable(
            'undervoltage_costs', (self.grd['n_buses'] - 1, self.N), vtype='continuous'
        )

        p.add_constraint(p_car_flex == charge / self.car['eta_ch'] - discharge_other)

        # if we don't allow the use of the battery reactive power for control
        # then we restain it by using the power factor
        if self.reactive_power_for_voltage_control:
            for time_step in range(self.N):

                p.add_list_of_constraints([
                    p_car_flex2[home, time_step] >= p_car_flex[home, time_step]
                    * p_car_flex[home, time_step] for home in range(self.n_homes)
                ])
                p.add_list_of_constraints([
                    q_car_flex2[home, time_step] >= q_car_flex[home, time_step]
                    * q_car_flex[home, time_step] for home in range(self.n_homes)
                ])
                p.add_list_of_constraints([
                    p_car_flex2[home, time_step] + q_car_flex2[home, time_step]
                    <= self.car['max_apparent_power_car']**2 for home in range(self.n_homes)
                ])
            # can only use reactive power of battery if car is available
            p.add_constraint(
                q_car_flex2 <= self.car['batch_avail_car'][:, 0: self.N] * self.syst['M']
            )

        else:
            p.add_constraint(
                q_car_flex == p_car_flex * self.grd['active_to_reactive_flex']
            )
            p.add_constraint(
                p_car_flex <= self.car['max_apparent_power_car'] * self.car['eta_ch']
            )
        p.add_list_of_constraints(
            [
                pi[:, time_step]
                == (
                    self.grd['flex_buses'] * netp[:, time_step]
                    + self.loads['active_power_passive_homes'][time_step]
                ) * self.kW_to_per_unit_conversion
                for time_step in range(self.N)
            ]
        )
        p.add_constraint(
            netq_flex
            == q_car_flex
            + (totcons - self.grd['gen'][:, 0: self.N]) * self.grd['active_to_reactive_flex']
        )

        p.add_list_of_constraints(
            [
                qi[:, time_step]
                == (
                    self.grd['flex_buses'] * netq_flex[:, time_step]
                    + self.loads['reactive_power_passive_homes'][time_step]
                ) * self.kW_to_per_unit_conversion
                for time_step in range(self.N)
            ]
        )

        p.add_list_of_constraints(
            [
                pij[0, time_step] == grid[time_step] * self.kW_to_per_unit_conversion
                for time_step in range(self.N)
            ]
        )
        p.add_list_of_constraints(
            [
                qij[0, time_step] == q_ext_grid[time_step] * self.kW_to_per_unit_conversion
                for time_step in range(self.N)
            ]
        )

        # bus voltage
        p.add_list_of_constraints([
            voltage_squared[0, time_step] == 1.0
            for time_step in range(self.N)
        ])

        if self.grd['line_losses_method'] == 'subset_of_lines':
            # external grid between bus 1 and 2
            p.add_list_of_constraints(
                [q_ext_grid[time_step]
                    == pic.sum(netq_flex[:, time_step])
                    + pic.sum(np.diag(self.grd['line_reactance'], k=0) * lij[:, time_step])
                    * self.per_unit_to_kW_conversion
                    + sum(self.prm['loads']['q_heat_home_car_passive'][:, time_step])
                    for time_step in range(self.N)])
            # active power flow
            p.add_list_of_constraints(
                [
                    pi[1:, time_step]
                    == - self.grd['incidence_matrix'][1:, :] * pij[:, time_step]
                    + np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(self.grd['line_resistance'], k=0)
                    ) * lij[:, time_step]
                    for time_step in range(self.N)
                ]
            )

            # reactive power flow
            p.add_list_of_constraints(
                [
                    qi[1:, time_step] == - self.grd['incidence_matrix'][1:, :] * qij[:, time_step]
                    + np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(self.grd['line_reactance'], k=0)
                    ) * lij[:, time_step]
                    for time_step in range(self.N)
                ]
            )
            # auxiliary constraint
            p.add_list_of_constraints([
                v_line[:, time_step]
                == self.grd['out_incidence_matrix'].T * voltage_squared[:, time_step]
                for time_step in range(self.N)
            ])
            # voltages
            p.add_list_of_constraints(
                [
                    voltage_squared[1:, time_step] == self.grd['bus_connection_matrix'][1:, :]
                    * voltage_squared[:, time_step]
                    + 2 * (
                        np.matmul(
                            self.grd['in_incidence_matrix'][1:, :],
                            np.diag(self.grd['line_resistance'], k=0)
                        )
                        * pij[:, time_step]
                        + np.matmul(
                            self.grd['in_incidence_matrix'][1:, :],
                            np.diag(self.grd['line_reactance'], k=0)
                        ) * qij[:, time_step]
                    ) - np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(np.square(self.grd['line_resistance']))
                        + np.diag(np.square(self.grd['line_reactance']))
                    ) * lij[:, time_step]
                    for time_step in range(self.N)
                ]
            )

            # relaxed constraint
            for time_step in range(self.N):
                p.add_list_of_constraints(
                    [
                        v_line[line, time_step] * lij[line, time_step] >= pij[line, time_step]
                        * pij[line, time_step] + qij[line, time_step] * qij[line, time_step]
                        for line in range(self.grd['subset_line_losses_modelled'])
                    ]
                )
            # lij == 0 for remaining lines
            p.add_list_of_constraints(
                [
                    lij[self.grd['subset_line_losses_modelled']:self.grd['n_lines'], time_step] == 0
                    for time_step in range(self.N)
                ]
            )
            # hourly line losses
            p.add_list_of_constraints(
                [
                    line_losses_pu[:, time_step]
                    == np.diag(self.grd['line_resistance']) * lij[:, time_step]
                    for time_step in range(self.N)
                ]
            )
        else:
            # external grid between bus 1 and 2
            p.add_list_of_constraints(
                [q_ext_grid[time_step]
                    == pic.sum(netq_flex[:, time_step])
                    + sum(np.matmul(np.diag(self.grd['line_reactance'], k=0), lij[:, time_step]))
                    * self.per_unit_to_kW_conversion
                    + sum(self.prm['loads']['q_heat_home_car_passive'][:, time_step])
                    for time_step in range(self.N)])

            # active power flow
            p.add_list_of_constraints(
                [
                    pi[1:, time_step]
                    == - self.grd['incidence_matrix'][1:, :] * pij[:, time_step]
                    + np.matmul(np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(self.grd['line_resistance'], k=0)
                    ), lij[:, time_step])
                    for time_step in range(self.N)
                ]
            )

            # reactive power flow
            p.add_list_of_constraints(
                [
                    qi[1:, time_step] == - self.grd['incidence_matrix'][1:, :] * qij[:, time_step]
                    + np.matmul(np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(self.grd['line_reactance'], k=0)
                    ), lij[:, time_step])
                    for time_step in range(self.N)
                ]
            )
            # voltages
            p.add_list_of_constraints(
                [
                    voltage_squared[1:, time_step] == self.grd['bus_connection_matrix'][1:, :]
                    * voltage_squared[:, time_step]
                    + 2 * (
                        np.matmul(
                            self.grd['in_incidence_matrix'][1:, :],
                            np.diag(self.grd['line_resistance'], k=0)
                        )
                        * pij[:, time_step]
                        + np.matmul(
                            self.grd['in_incidence_matrix'][1:, :],
                            np.diag(self.grd['line_reactance'], k=0)
                        ) * qij[:, time_step]
                    ) - np.matmul(np.matmul(
                        self.grd['in_incidence_matrix'][1:, :],
                        np.diag(np.square(self.grd['line_resistance']))
                        + np.diag(np.square(self.grd['line_reactance']))
                    ), lij[:, time_step])
                    for time_step in range(self.N)
                ]
            )
            # hourly line losses
            p.add_list_of_constraints([
                line_losses_pu[:, time_step]
                == np.matmul(np.diag(self.grd['line_resistance']), lij[:, time_step])
                for time_step in range(self.N)
            ])

        p.add_list_of_constraints(
            [
                hourly_line_losses_pu[time_step] == pic.sum(line_losses_pu[:, time_step])
                for time_step in range(self.N)
            ]
        )

        # Voltage limitation penalty
        # for each bus
        p.add_constraint(overvoltage_costs >= 0)
        p.add_constraint(
            overvoltage_costs
            >= self.grd['penalty_overvoltage'] * (voltage_squared - self.grd['max_voltage'] ** 2)
        )
        p.add_constraint(undervoltage_costs >= 0)
        p.add_constraint(
            undervoltage_costs
            >= self.grd['penalty_undervoltage'] * (self.grd['min_voltage'] ** 2 - voltage_squared)
        )

        # sum over all buses
        p.add_constraint(
            voltage_costs == pic.sum(overvoltage_costs + undervoltage_costs)
        )

        return p, voltage_costs, q_ext_grid

    def _grid_constraints(self, p, charge, discharge_other, totcons):
        # variables
        grid = p.add_variable('grid', self.N, vtype='continuous')
        grid2 = p.add_variable('grid2', self.N, vtype='continuous')
        netp = p.add_variable('netp', (self.n_homes, self.N), vtype='continuous')
        hourly_line_losses_pu = p.add_variable('hourly_line_losses_pu', self.N, vtype='continuous')
        grid_energy_costs = p.add_variable('grid_energy_costs', 1)  # grid costs

        # constraints
        # substation energy balance
        self.loads['hourly_tot_netp0'] = \
            np.sum(self.loads['netp0'], axis=0) if len(self.loads['netp0']) > 0 \
            else np.zeros(self.N)
        p.add_list_of_constraints(
            [
                grid[time_step]
                - self.loads['hourly_tot_netp0'][time_step]
                - pic.sum([netp[home, time_step] for home in range(self.n_homes)])
                - hourly_line_losses_pu[time_step] * self.per_unit_to_kW_conversion
                == 0
                for time_step in range(self.N)
            ]
        )

        # costs constraints
        p.add_list_of_constraints(
            [grid2[time_step] >= grid[time_step] * grid[time_step] for time_step in range(self.N)]
        )

        if self.grd['manage_voltage']:
            p, voltage_costs, _ = self._power_flow_equations(
                p, netp, grid, hourly_line_losses_pu,
                charge, discharge_other, totcons)
        else:
            p.add_constraint(hourly_line_losses_pu == 0)
            voltage_costs = 0

        # grid costs
        p.add_constraint(
            grid_energy_costs
            == (self.grd['C'][0: self.N] | (grid + self.grd['loss'] * grid2))
        )

        return p, netp, grid, grid_energy_costs, voltage_costs

    def _storage_constraints(self, p):
        """Storage constraints."""
        charge = p.add_variable('charge', (self.n_homes, self.N), vtype='continuous')
        discharge_tot = p.add_variable('discharge_tot', (self.n_homes, self.N), vtype='continuous')
        discharge_other = p.add_variable(
            'discharge_other', (self.n_homes, self.N), vtype='continuous'
        )
        store = p.add_variable('store', (self.n_homes, self.N), vtype='continuous')
        battery_degradation_costs = p.add_variable('battery_degradation_costs', 1)  # storage costs

        store_end = self.car['SoC0'] * self.grd['Bcap'][:, self.N - 1]
        car = self.car

        # battery energy balance
        p.add_constraint(
            discharge_tot
            == discharge_other / self.car['eta_dis']
            + self.car['batch_loads_car'][:, 0: self.N]
        )

        if car['eff'] == 1:
            p.add_list_of_constraints(
                [
                    charge[:, time_step] - discharge_tot[:, time_step]
                    == store[:, time_step + 1] - store[:, time_step]
                    for time_step in range(self.N - 1)
                ]
            )
            p.add_constraint(
                store[:, self.N - 1]
                + charge[:, self.N - 1]
                - discharge_tot[:, self.N - 1]
                >= store_end
            )

        elif car['eff'] == 2:
            for home in range(self.n_homes):
                p.add_constraint(
                    car['eta_ch'][home, self.N - 1] * charge[home, self.N - 1]
                    - car['eta_ch'][home, self.N - 1]
                    * discharge_tot[home, self.N - 1]
                    == store_end[home] - store[home, self.N - 1]
                )
                for time_step in range(self.N - 1):
                    p.add_constraint(
                        car['eta_ch'][home, time_step] * charge[home, time_step]
                        - car['eta_dis'][home, time_step] * discharge_tot[home, time_step]
                        == store[home, time_step + 1] - store[home, time_step]
                    )

        # initialise storage
        p.add_list_of_constraints(
            [store[home, 0] == car['SoC0'] * self.grd['Bcap'][home, 0]
             for home in range(self.n_homes)])

        p.add_list_of_constraints(
            [store[:, time_step + 1] >= car['SoCmin']
             * self.grd['Bcap'][:, time_step] * car['batch_avail_car'][:, time_step]
             for time_step in range(self.N - 1)]
        )

        # if EV not avail at a given time step,
        # it is ok to start the following time step with less than minimum
        p.add_list_of_constraints(
            [store[:, time_step + 1] >= car['baseld'] * car['batch_avail_car'][:, time_step]
             for time_step in range(self.N - 1)]
        )

        # can charge only when EV is available
        p.add_constraint(
            charge <= car['batch_avail_car'][:, 0: self.N] * self.syst['M']
        )

        # can discharge only when EV is available (Except EV cons is ok)
        p.add_constraint(
            discharge_other
            <= car['batch_avail_car'][:, 0: self.N] * self.syst['M']
        )

        p.add_constraint(
            battery_degradation_costs == self.car['C']
            * (pic.sum(discharge_tot) + pic.sum(charge)
               + np.sum(self.loads['discharge_tot0'])
               + np.sum(self.loads['charge0']))
        )
        p.add_constraint(store <= self.grd['Bcap'])
        p.add_constraint(car['c_max'] >= charge)
        p.add_constraint(car['d_max'] >= discharge_tot)
        p.add_constraint(store >= 0)
        p.add_constraint(discharge_other >= 0)
        p.add_constraint(discharge_tot >= 0)
        p.add_constraint(charge >= 0)

        return p, charge, discharge_other, battery_degradation_costs

    def _distribution_costs(self, p, netp):
        distribution_network_export_costs = p.add_variable('distribution_network_export_costs', 1)

        grd = self.grd
        if grd['charge_type'] == 0:
            netp_export = p.add_variable(
                'netp_export',
                (self.n_homes, self.N),
                vtype='continuous'
            )
            self.loads['netp0_export'] = np.where(
                self.loads['netp0'] < 0,
                abs(self.loads['netp0']),
                0
            )
            p.add_constraint(netp_export >= - netp)
            p.add_constraint(netp_export >= 0)

            # distribution costs
            p.add_constraint(
                distribution_network_export_costs
                == grd['export_C'] * (pic.sum(netp_export) + np.sum(self.loads['netp0_export']))
            )

        else:

            netp2 = p.add_variable(
                'netp2', (self.n_homes, self.N), vtype='continuous'
            )
            sum_netp0_squared = np.sum(np.square(self.loads['netp0']))
            for home in range(self.n_homes):
                p.add_list_of_constraints(
                    [netp2[home, time_step] >= netp[home, time_step] * netp[home, time_step]
                     for time_step in range(self.N)])
            p.add_constraint(
                distribution_network_export_costs == grd['export_C'] * (
                    pic.sum(netp2) + sum_netp0_squared
                )
            )

        return p, distribution_network_export_costs

    def _update_prm(self, prm):
        """Update parameters with new values"""
        if isinstance(prm, (list, tuple)):
            self.syst, self.loads, self.car, self.grd, self.heat \
                = prm
        else:
            self.syst, self.loads, self.car, self.grd, self.heat \
                = [prm[e]
                   for e in ['syst', 'loads', 'car', 'grd', 'heat']]

    def _cons_constraints(self, p, E_heat):
        """Add constraints for consumption."""
        totcons = p.add_variable('totcons', (self.n_homes, self.N), vtype='continuous')

        consa = []
        for load_type in range(self.loads['n_types']):
            consa.append(p.add_variable('consa({0})'.format(load_type), (self.n_homes, self.N)))
        constl = {}
        tlpairs = comb(np.array([self.N, self.loads['n_types']]))
        for tl in tlpairs:
            constl[tl] = p.add_variable('constl{0}'.format(tl), (self.n_homes, self.N))

        constl_consa_constraints = []
        constl_loads_constraints = []
        for load_type in range(self.loads['n_types']):
            constl_consa_constraints_lt = []
            constl_loads_constraints_lt = []
            for home in range(self.n_homes):
                constl_consa_constraints_lt_home = []
                constl_loads_constraints_lt_home = []
                for time_step in range(self.N):
                    # time_step = tD
                    constl_loads_constraints_lt_home_t = p.add_constraint(
                        pic.sum(
                            [constl[time_step, load_type][home, tC]
                             * self.grd['flex'][time_step, load_type, home, tC]
                             for tC in range(self.N)]
                        )
                        == self.grd['loads'][load_type, home, time_step])
                    # time_step = tC
                    constl_consa_constraints_lt_home_t = p.add_constraint(
                        pic.sum(
                            [constl[tD, load_type][home, time_step] for tD in range(self.N)]
                        ) == consa[load_type][home, time_step]
                    )
                    constl_consa_constraints_lt_home.append(constl_consa_constraints_lt_home_t)
                    constl_loads_constraints_lt_home.append(constl_loads_constraints_lt_home_t)

                constl_consa_constraints_lt.append(constl_consa_constraints_lt_home)
                constl_loads_constraints_lt.append(constl_loads_constraints_lt_home)

            constl_consa_constraints.append(constl_consa_constraints_lt)
            constl_loads_constraints.append(constl_loads_constraints_lt)

        p.add_constraint(
            pic.sum(
                [consa[load_type]
                 for load_type in range(self.loads['n_types'])]
            ) + E_heat
            == totcons)
        for load_type in range(self.loads['n_types']):
            p.add_constraint(consa[load_type] >= 0)
        p.add_list_of_constraints([constl[tl] >= 0 for tl in tlpairs])
        p.add_constraint(totcons >= 0)

        return p, totcons, constl_consa_constraints, constl_loads_constraints

    def _temperature_constraints(self, p):
        """Add temperature constraints to the problem."""
        T = p.add_variable('T', (self.n_homes, self.N), vtype='continuous')
        T_air = p.add_variable('T_air', (self.n_homes, self.N), vtype='continuous')
        # in kWh use P in W  Eheat * 1e3 * self.syst['prm']['H']/24 =  Pheat
        E_heat = p.add_variable('E_heat', (self.n_homes, self.N), vtype='continuous')
        heat = self.heat
        for home in range(self.n_homes):
            if heat['own_heat' + self.ext][home]:
                p.add_constraint(T[home, 0] == heat['T0'])
                p.add_list_of_constraints(
                    [T[home, time_step + 1] == heat['T_coeff' + self.ext][home][0]
                        + heat['T_coeff' + self.ext][home][1] * T[home, time_step]
                        + heat['T_coeff' + self.ext][home][2] * heat['T_out'][time_step]
                        # heat['T_coeff'][home][3] * heat['phi_sol'][time_step]
                        + heat['T_coeff' + self.ext][home][4] * E_heat[home, time_step]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time_step in range(self.N - 1)]
                )

                p.add_list_of_constraints(
                    [T_air[home, time_step] == heat['T_air_coeff' + self.ext][home][0]
                        + heat['T_air_coeff' + self.ext][home][1] * T[home, time_step]
                        + heat['T_air_coeff' + self.ext][home][2] * heat['T_out'][time_step]
                        # heat['T_air_coeff'][home][3] * heat['phi_sol'][time_step] +
                        + heat['T_air_coeff' + self.ext][home][4] * E_heat[home, time_step]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time_step in range(self.N)]
                )

                p.add_list_of_constraints(
                    [
                        T_air[home, time_step] <= heat['T_UB' + self.ext][home][time_step]
                        for time_step in range(self.N)
                    ]
                )
                p.add_list_of_constraints(
                    [
                        T_air[home, time_step] >= heat['T_LB' + self.ext][home][time_step]
                        for time_step in range(self.N)
                    ]
                )
            else:
                p.add_constraint(
                    E_heat[home] == 0
                )
                p.add_constraint(
                    T_air[home, :]
                    == (heat['T_LB' + self.ext][home, 0: self.N] + heat['T_UB' + self.ext][home, 0: self.N]) / 2
                )
                p.add_constraint(
                    T[home, :]
                    == (heat['T_LB' + self.ext][home, 0: self.N] + heat['T_UB' + self.ext][home, 0: self.N]) / 2
                )

        p.add_constraint(E_heat >= 0)

        return p, E_heat

    def _set_objective(
            self, p, netp, grid, grid_energy_costs, battery_degradation_costs, voltage_costs
    ):
        network_costs = p.add_variable('network_costs', 1)
        total_costs = p.add_variable('total_costs', 1, vtype='continuous')

        p, import_export_costs = self._import_export_costs(p, grid)

        p.add_constraint(
            network_costs
            == self.grd['weight_network_costs'] * (import_export_costs + voltage_costs)
        )
        if self.penalise_individual_exports:
            p, distribution_network_export_costs = self._distribution_costs(p, netp)
        else:
            distribution_network_export_costs = 0

        p.add_constraint(
            total_costs
            == grid_energy_costs
            + battery_degradation_costs
            + distribution_network_export_costs
            + network_costs
        )

        p.set_objective('min', total_costs)

        return p

    def _import_export_costs(self, p, grid):
        # penalty for import and export violations
        import_export_costs = p.add_variable('import_export_costs', 1)  # total import export costs
        if self.grd['manage_agg_power']:
            hourly_import_costs = p.add_variable('hourly_import_costs', self.N, vtype='continuous')
            hourly_export_costs = p.add_variable('hourly_export_costs', self.N, vtype='continuous')
            grid_in = p.add_variable('grid_in', self.N, vtype='continuous')  # hourly grid import
            grid_out = p.add_variable('grid_out', self.N, vtype='continuous')  # hourly grid export
            # import and export definition
            p.add_constraint(grid == grid_in - grid_out)
            p.add_constraint(grid_in >= 0)
            p.add_constraint(grid_out >= 0)
            p.add_constraint(hourly_import_costs >= 0)
            p.add_constraint(
                hourly_import_costs
                >= self.grd['penalty_import'] * (
                    grid_in * self.syst['n_int_per_hr'] - self.grd['max_grid_import']
                )
            )
            p.add_condstraint(hourly_export_costs >= 0)
            p.add_constraint(
                hourly_export_costs
                >= self.grd['penalty_export'] * (
                    grid_out * self.syst['n_int_per_hr'] - self.grd['max_grid_export']
                )
            )
            p.add_constraint(
                import_export_costs == pic.sum(hourly_import_costs) + pic.sum(hourly_export_costs)
            )
        else:
            p.add_constraint(import_export_costs == 0)

        return p, import_export_costs

    def _problem(self, evaluation=False):
        """Solve optimisation problem."""
        # initialise problem
        p = pic.Problem()

        p, charge, discharge_other, battery_degradation_costs = self._storage_constraints(p)
        p, E_heat = self._temperature_constraints(p)
        p, totcons, constl_consa_constraints, constl_loads_constraints \
            = self._cons_constraints(p, E_heat)
        p, netp, grid, grid_energy_costs, voltage_costs = self._grid_constraints(
            p, charge, discharge_other, totcons)
        # prosumer energy balance, active power
        p.add_constraint(
            netp - charge / self.car['eta_ch']
            + discharge_other
            + self.grd['gen'][:, 0: self.N]
            - totcons == 0
        )
        # costs constraints
        p = self._set_objective(
            p, netp, grid, grid_energy_costs, battery_degradation_costs, voltage_costs
        )
        # solve
        p.solve(verbose=0, solver=self.syst['solver'])

        # save results
        res = save_results(p.variables, self.prm)
        number_opti_constraints = len(p.constraints)
        if 'n_opti_constraints' not in self.syst:
            self.syst['n_opti_constraints'] = number_opti_constraints

        if self.grd['manage_voltage'] and self.grd['line_losses_method'] != 'iteration':
            res, pp_simulation_required = check_and_correct_constraints(
                res, constl_consa_constraints, constl_loads_constraints,
                self.prm, self.input_hourly_lij, evaluation
            )
        else:
            pp_simulation_required = False
            res['max_cons_slack'] = -1

        res['corrected_cons'] = pp_simulation_required

        return res, pp_simulation_required, constl_consa_constraints, constl_loads_constraints
