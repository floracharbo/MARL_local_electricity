#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  7 17:10:28 2020.

@author: floracharbonnier

"""

import copy
import math

import numpy as np
import picos as pic

from src.utilities.userdeftools import calculate_reactive_power, comb


class Optimiser:
    """The Optimiser object manages convex optimisations."""

    def __init__(self, prm, compute_import_export_costs, prepare_and_compare_optimiser_pandapower):
        """Initialise solver object for doing optimisations."""
        for attribute in ['N', 'n_homes', 'tol_cons_constraints', 'n_homesP']:
            setattr(self, attribute, prm['syst'][attribute])
        self.save = prm["save"]
        self.paths = prm["paths"]
        self.manage_agg_power = prm["grd"]["manage_agg_power"]
        self.penalise_individual_exports = prm["grd"]["penalise_individual_exports"]
        self.kW_to_per_unit_conversion = 1000 / prm['grd']['base_power']
        self.per_unit_to_kW_conversion = prm['grd']['base_power'] / 1000
        self.reactive_power_for_voltage_control = \
            prm['grd']['reactive_power_for_voltage_control']
        self.compute_import_export_costs = compute_import_export_costs
        self.prepare_and_compare_optimiser_pandapower = prepare_and_compare_optimiser_pandapower

    def res_post_processing(self, res):
        res['house_cons'] = res['totcons'] - res['E_heat']
        if self.grd['manage_agg_power']:
            res['hourly_import_export_costs'] = \
                res['hourly_import_costs'] + res['hourly_export_costs']
        else:
            res['hourly_import_costs'] = np.zeros(self.N)
            res['hourly_export_costs'] = np.zeros(self.N)
            res['hourly_import_export_costs'] = np.zeros(self.N)

        if self.grd['manage_voltage']:
            res['voltage'] = np.sqrt(res['voltage_squared'])
            res['hourly_voltage_costs'] = np.sum(
                res['overvoltage_costs'] + res['undervoltage_costs'], axis=0
            )
            res['hourly_line_losses'] = \
                res['hourly_line_losses_pu'] * self.per_unit_to_kW_conversion
            if self.grd['line_losses_method'] == 'iteration':
                res['lij'] = self.input_hourly_lij
                res['v_line'] = np.matmul(
                    self.grd['out_incidence_matrix'].T, res['voltage_squared'])
            res['p_solar_flex'] = self.grd['gen'][:, 0: self.N]
            res['q_solar_flex'] = calculate_reactive_power(
                self.grd['gen'][:, 0: self.N], self.grd['pf_flexible_homes'])
        else:
            res['voltage_squared'] = np.empty((1, self.N))
            res['voltage_costs'] = 0
            res['hourly_voltage_costs'] = np.zeros(self.N)
            res['hourly_line_losses'] = np.zeros(self.N)
            res['q_ext_grid'] = np.zeros(self.N)

        if self.n_homesP > 0:
            res['netp0'] = self.loads['netp0']
        else:
            res['netp0'] = np.zeros([1, self.N])

        res['hourly_grid_energy_costs'] = self.grd['C'][0: self.N] * (
            res["grid"] + self.grd["loss"] * res["grid2"]
        )
        res['hourly_battery_degradation_costs'] = self.car["C"] * (
            np.sum(res["discharge_tot"] + res["charge"], axis=0)
            + np.sum(self.loads['discharge_tot0'], axis=0)
            + np.sum(self.loads['charge0'], axis=0)
        )
        if self.penalise_individual_exports:
            res['hourly_distribution_network_export_costs'] = self.grd["export_C"] * (
                np.sum(res["netp_export"], axis=0)
                + np.sum(self.netp0_export, axis=0)
            )
        else:
            res['hourly_distribution_network_export_costs'] = np.zeros(self.N)

        res['hourly_total_costs'] = \
            (res['hourly_import_export_costs'] + res['hourly_voltage_costs']) \
            * self.grd["weight_network_costs"] \
            + res['hourly_grid_energy_costs'] \
            + res['hourly_battery_degradation_costs'] \
            + res['hourly_distribution_network_export_costs']

        for key, val in res.items():
            if key[0: len('hourly')] == 'hourly':
                assert len(val) == self.N, f"np.shape(res[{key}]) = {np.shape(val)}"
        assert np.all(res['consa(1)'] > - self.tol_cons_constraints), \
            f"negative flexible consumptions in the optimisation! " \
            f"np.min(res['consa(1)']) = {np.min(res['consa(1)'])}"

        return res

    def solve(self, prm):
        """Solve optimisation problem given prm input data."""
        self._update_prm(prm)
        if self.grd['line_losses_method'] == 'iteration':
            res = self._solve_line_losses_iteration()
            pp_simulation_required = False

        elif self.grd['line_losses_method'] == 'subset_of_lines':
            res, pp_simulation_required = self._problem()
            res = self.res_post_processing(res)

        if prm['car']['efftype'] == 1:
            res = self._car_efficiency_iterations(prm, res)
            res = self.res_post_processing(res)

        return res, pp_simulation_required

    def _solve_line_losses_iteration(self):
        it = 0
        self.input_hourly_lij = np.zeros((self.grd['n_lines'], self.N))
        res, _ = self._problem()
        res = self.res_post_processing(res)
        # print pi and qi
        opti_voltages = copy.deepcopy(res['voltage'])
        opti_losses = copy.deepcopy(res['hourly_line_losses'])
        for time_step in range(self.N):
            # net0 = self.loads['netp0'][time_step]
            netp0 = np.zeros([1, self.N])
            grdCt = self.grd['C'][time_step]
            res = self.prepare_and_compare_optimiser_pandapower(
                res, time_step, netp0, grdCt, self.grd['line_losses_method'])
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
            res, _ = self._problem()
            res = self.res_post_processing(res)
            opti_voltages = copy.deepcopy(res['voltage'])
            opti_losses = copy.deepcopy(res['hourly_line_losses'])
            for time_step in range(self.N):
                # net0 = self.loads['netp0'][time_step]
                netp0 = np.zeros([1, self.N])
                grdCt = self.grd['C'][time_step]
                res = self.prepare_and_compare_optimiser_pandapower(
                    res, time_step, netp0, grdCt, self.grd['line_losses_method'])
            corr_voltages = copy.deepcopy(res['voltage'])
            corr_losses = copy.deepcopy(res['hourly_line_losses'])
            corr_lij = copy.deepcopy(res['lij'])
            delta_losses = opti_losses - corr_losses
            delta_voltages = opti_voltages - corr_voltages
            print(f"max hourly delta voltages iteration {it}: {abs(delta_voltages).max()}")
            print(f"max hourly delta losses iteration {it}: {abs(delta_losses).max()}")
        return res

    def _car_efficiency_iterations(self, prm, res):
        init_eta = prm['car']['etach']
        prm['car']['etach'] = self._efficiencies(
            res, prm, prm['car']['cap'])
        deltamax, its = 0.5, 0
        prm['car']['eff'] = 2
        while deltamax > 0.01 and its < 10:
            its += 1
            eta_old = copy.deepcopy(prm['car']['etach'])
            print(f"prm['grd']['loads'][0][0][0] = "
                  f"{prm['grd']['loads'][0][0][0]}")
            res = self._problem()
            print(f"res['constl(0, 0)'][0][0] "
                  f"= {res['constl(0, 0)'][0][0]}")
            if prm['grd']['loads'][0][0][0] < res['constl(0, 0)'][0][0]:
                print('fixed loads smaller than fixed consumption home=0 time=0')
            if abs(np.sum(res['totcons']) - np.sum(res['E_heat'])
                   - np.sum(prm['grd']['loads'])) > 1e-3:
                print(f"tot load cons "
                      f"{np.sum(res['totcons']) - np.sum(res['E_heat'])} "
                      f"not equal to loads {np.sum(prm['loads'])}")
            prm['car']['etach'] = self._efficiencies(
                res, prm, prm['car']['cap'])
            deltamax = np.amax(abs(prm['car']['etach'] - eta_old))
        prm['car']['etach'] = init_eta
        prm['car']['eff'] = 1

        return res

    def _power_flow_equations(
            self, p, netp, grid, hourly_line_losses_pu,
            charge, discharge_other, totcons):
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

        # active and reactive loads
        # flex houses: car
        p.add_constraint(p_car_flex == charge / self.car['eta_ch'] - discharge_other)
        p.add_constraint(p_car_flex <= self.car['max_active_power_car'])

        # if we don't allow the use of the battery reactive power for control
        # then we restain it by using the power factor
        if self.reactive_power_for_voltage_control:
            for time in range(self.N):
                p.add_list_of_constraints([
                    p_car_flex2[home, time] >= p_car_flex[home, time]
                    * p_car_flex[home, time] for home in range(self.n_homes)
                ])
                p.add_list_of_constraints([
                    q_car_flex2[home, time] >= q_car_flex[home, time]
                    * q_car_flex[home, time] for home in range(self.n_homes)
                ])
                p.add_list_of_constraints([
                    p_car_flex2[home, time] + p_car_flex2[home, time]
                    <= self.car['max_apparent_power_car']**2 for home in range(self.n_homes)
                ])
        else:
            p.add_constraint(q_car_flex == calculate_reactive_power(
                p_car_flex, self.grd['pf_flexible_homes']))

        p.add_list_of_constraints(
            [pi[:, time_step]
                == self.grd['flex_buses'] * netp[:, time_step] * self.kW_to_per_unit_conversion
                # + self.grd['passive_buses']
                # * self.loads['active_power_passive_homes'][t]
                # * self.kW_to_per_unit_conversion
                for time_step in range(self.N)])

        p.add_list_of_constraints(
            [
                netq_flex[:, time_step] == q_car_flex[:, time_step]
                + totcons[:, time_step] * math.tan(math.acos(self.grd['pf_flexible_homes']))
                - self.grd['gen'][:, time_step] * math.tan(math.acos(self.grd['pf_flexible_homes']))
                for time_step in range(self.N)
            ]
        )
        # p.add_list_of_constraints(
        #    [
        #        netq_passive[time_step] ==
        #        * self.loads['reactive_power_passive_homes'][t]
        #        * self.kW_to_per_unit_conversion
        #    ]
        # )
        p.add_list_of_constraints(
            [qi[:, time_step]
                == self.grd['flex_buses'] * netq_flex[:, time_step] * self.kW_to_per_unit_conversion
                # + self.grd['passive_buses']
                # * netq_passive[:, time_step]
                # * self.kW_to_per_unit_conversion
                for time_step in range(self.N)])

        # external grid between bus 1 and 2
        p.add_list_of_constraints(
            [q_ext_grid[time_step]
                == pic.sum(netq_flex[:, time_step])
                + sum(np.matmul(np.diag(self.grd['line_reactance'], k=0), lij[:, time_step]))
                * self.per_unit_to_kW_conversion
                # + pic.sum(netq_passive[:, time_step])
                for time_step in range(self.N)])

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

        return p, voltage_costs

    def _grid_constraints(self, p, charge, discharge_other, totcons):
        # variables
        grid = p.add_variable('grid', self.N, vtype='continuous')
        grid2 = p.add_variable('grid2', self.N, vtype='continuous')
        netp = p.add_variable('netp', (self.n_homes, self.N), vtype='continuous')
        hourly_line_losses_pu = p.add_variable('hourly_line_losses_pu', self.N, vtype='continuous')
        grid_energy_costs = p.add_variable('grid_energy_costs', 1)  # grid costs

        # constraints
        # substation energy balance
        self.hourly_tot_netp0 = \
            np.sum(self.loads['netp0'], axis=0) if len(self.loads['netp0']) > 0 \
            else np.zeros(self.N)
        p.add_list_of_constraints(
            [
                grid[time_step]
                - self.hourly_tot_netp0[time_step]
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

        # grid costs
        p.add_constraint(
            grid_energy_costs
            == (self.grd['C'][0: self.N] | (grid + self.grd['loss'] * grid2))
        )

        if self.grd['manage_voltage']:
            p, voltage_costs = self._power_flow_equations(
                p, netp, grid, hourly_line_losses_pu,
                charge, discharge_other, totcons)
        else:
            p.add_constraint(hourly_line_losses_pu == 0)
            voltage_costs = 0

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
            self.netp0_export = np.where(
                self.loads['netp0'] < 0,
                abs(self.loads['netp0']),
                0
            )
            p.add_constraint(netp_export >= - netp)
            p.add_constraint(netp_export >= 0)

            # distribution costs
            p.add_constraint(
                distribution_network_export_costs
                == grd['export_C'] * (pic.sum(netp_export) + np.sum(self.netp0_export))
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

        constl_constraints = []
        for load_type in range(self.loads['n_types']):
            constl_constraints_lt = []
            for home in range(self.n_homes):
                constl_constraints_lt_home = []
                for time_step in range(self.N):
                    # time_step = tD
                    p.add_constraint(
                        pic.sum(
                            [constl[time_step, load_type][home, tC]
                             * self.grd['flex'][time_step, load_type, home, tC]
                             for tC in range(self.N)]
                        )
                        == self.grd['loads'][load_type, home, time_step])
                    # time_step = tC
                    constl_constraints_lt_home_t = p.add_constraint(
                        pic.sum(
                            [constl[tD, load_type][home, time_step] for tD in range(self.N)]
                        ) == consa[load_type][home, time_step]
                    )
                    constl_constraints_lt_home.append(constl_constraints_lt_home_t)
                constl_constraints_lt.append(constl_constraints_lt_home)
            constl_constraints.append(constl_constraints_lt)
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

        return p, totcons, constl_constraints

    def _temperature_constraints(self, p):
        """Add temperature constraints to the problem."""
        T = p.add_variable('T', (self.n_homes, self.N), vtype='continuous')
        T_air = p.add_variable('T_air', (self.n_homes, self.N), vtype='continuous')
        # in kWh use P in W  Eheat * 1e3 * self.syst['prm']['H']/24 =  Pheat
        E_heat = p.add_variable('E_heat', (self.n_homes, self.N), vtype='continuous')
        heat = self.heat
        for home in range(self.n_homes):
            if heat['own_heat'][home]:
                p.add_constraint(T[home, 0] == heat['T0'])
                p.add_list_of_constraints(
                    [T[home, time_step + 1] == heat['T_coeff'][home][0]
                        + heat['T_coeff'][home][1] * T[home, time_step]
                        + heat['T_coeff'][home][2] * heat['T_out'][time_step]
                        # heat['T_coeff'][home][3] * heat['phi_sol'][time_step]
                        + heat['T_coeff'][home][4] * E_heat[home, time_step]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time_step in range(self.N - 1)]
                )

                p.add_list_of_constraints(
                    [T_air[home, time_step] == heat['T_air_coeff'][home][0]
                        + heat['T_air_coeff'][home][1] * T[home, time_step]
                        + heat['T_air_coeff'][home][2] * heat['T_out'][time_step]
                        # heat['T_air_coeff'][home][3] * heat['phi_sol'][time_step] +
                        + heat['T_air_coeff'][home][4] * E_heat[home, time_step]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time_step in range(self.N)]
                )

                p.add_list_of_constraints(
                    [
                        T_air[home, time_step] <= heat['T_UB'][home][time_step]
                        for time_step in range(self.N)
                    ]
                )
                p.add_list_of_constraints(
                    [
                        T_air[home, time_step] >= heat['T_LB'][home][time_step]
                        for time_step in range(self.N)
                    ]
                )
            else:
                p.add_list_of_constraints(
                    [E_heat[home, time_step] == 0 for time_step in range(self.N)]
                )
                p.add_list_of_constraints(
                    [
                        T_air[home, time_step]
                        == (heat['T_LB'][home][time_step] + heat['T_UB'][home][time_step]) / 2
                        for time_step in range(self.N)
                    ]
                )
                p.add_list_of_constraints(
                    [
                        T[home, time_step]
                        == (heat['T_LB'][home][time_step] + heat['T_UB'][home][time_step]) / 2
                        for time_step in range(self.N)
                    ]
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
                >= self.grd['penalty_import'] * (grid_in - self.grd['max_grid_import'])
            )
            p.add_constraint(hourly_export_costs >= 0)
            p.add_constraint(
                hourly_export_costs
                >= self.grd['penalty_export'] * (grid_out - self.grd['max_grid_export'])
            )
            p.add_constraint(
                import_export_costs == pic.sum(hourly_import_costs) + pic.sum(hourly_export_costs)
            )
        else:
            p.add_constraint(import_export_costs == 0)

        return p, import_export_costs

    def _problem(self):
        """Solve optimisation problem."""
        # initialise problem
        p = pic.Problem()

        p, charge, discharge_other, battery_degradation_costs = self._storage_constraints(p)
        p, E_heat = self._temperature_constraints(p)
        p, totcons, constl_constraints = self._cons_constraints(p, E_heat)
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
        res = self._save_results(p.variables)
        number_opti_constraints = len(p.constraints)
        if 'n_opti_constraints' not in self.syst:
            self.syst['n_opti_constraints'] = number_opti_constraints

        if self.grd['manage_voltage']:
            res, pp_simulation_required = self._check_and_correct_cons_constraints(
                res, constl_constraints
            )
        else:
            pp_simulation_required = False
            res['max_cons_slack'] = -1

        res['corrected_cons'] = pp_simulation_required

        return res, pp_simulation_required

    def _check_and_correct_cons_constraints(
            self, res, constl_constraints
    ):
        slacks_constl = np.array([
            [
                [
                    constl_constraints[load_type][home][time_step].slack
                    for time_step in range(self.N)
                ]
                for home in range(self.n_homes)
            ]
            for load_type in range(self.loads['n_types'])
        ])
        load_types_slack, homes_slack, time_steps_slack = np.where(
            slacks_constl < - self.tol_cons_constraints
        )

        pp_simulation_required = len(time_steps_slack) > 0
        res['max_cons_slack'] = abs(np.min(slacks_constl))

        for load_type, home, time_step in zip(
                load_types_slack, homes_slack, time_steps_slack
        ):
            res[f'consa({load_type})'][home, time_step] = sum(
                [
                    res[f'constl({tD}, {load_type})'][home, time_step]
                    for tD in range(self.N)
                ]
            )
            res['totcons'][home, time_step] = sum(
                [
                    res[f'consa({load_type})'][home, time_step]
                    for load_type in range(self.loads['n_types'])
                ]
            ) + res['E_heat'][home, time_step]
            res['netp'][home, time_step] = \
                res['charge'][home, time_step] / self.car['eta_ch'] \
                - res['discharge_other'][home, time_step] \
                - self.grd['gen'][home, time_step] \
                + res['totcons'][home, time_step]

            res['grid'][time_step] = \
                sum(res['netp'][:, time_step]) \
                + self.hourly_tot_netp0[time_step] \
                + res['hourly_line_losses_pu'][time_step] * self.grd['base_power'] / 1000
            res['grid2'][time_step] = res['grid'][time_step] * res['grid'][time_step]

            if self.penalise_individual_exports:
                res['netp_export'][home, time_step] = np.where(
                    res['netp'][home, time_step] < 0, abs(res['netp'][home, time_step]), 0
                )
            if self.grd['manage_agg_power']:
                [
                    _,
                    res['hourly_import_costs'][time_step],
                    res['hourly_export_costs'][time_step]
                ] = self.compute_import_export_costs(res['grid'][time_step])

        if pp_simulation_required:
            res['grid_energy_costs'] = np.sum(
                np.multiply(self.grd['C'][0: self.N], res['grid'] + self.grd['loss'] * res['grid2'])
            )
            if self.penalise_individual_exports:
                res['distribution_network_export_costs'] = self.grd['export_C'] * (
                    np.sum(res['netp_export']) + np.sum(self.netp0_export)
                )
            else:
                res['distribution_network_export_costs'] = np.zeros(self.N)

            if self.grd['manage_agg_power']:
                res['import_export_costs'] = np.sum(
                    res['hourly_import_costs'] + res['hourly_export_costs']
                )

            res['network_costs'] = self.grd['weight_network_costs'] * res['import_export_costs']

            res['total_costs'] = \
                res['grid_energy_costs'] \
                + res['battery_degradation_costs'] \
                + res['distribution_network_export_costs'] \
                + res['network_costs']

        return res, pp_simulation_required

    def _efficiencies(self, res, prm, bat_cap):
        """Compute efficiencies"""
        store = res['store']
        P_ch = res['charge']
        P_dis = res['discharge_tot']

        P = (P_ch - P_dis) * 1e3
        SoC = np.zeros((self.n_homes, prm['N']))
        for home in range(self.n_homes):
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
        for home in range(self.n_homes):
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

    def _add_val_to_res(self, res, var, val, size, arr):
        """Add value to result dict."""
        if size[0] < 2 and size[1] < 2:
            res[var] = val
        else:
            for i in range(size[0]):
                for j in range(size[1]):
                    arr[i, j] = val[i, j]
            res[var] = arr

        return res

    def _save_results(self, pvars):
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
            res = self._add_val_to_res(res, var, val, size, arr)

        for key, val in res.items():
            if len(np.shape(val)) == 2 and np.shape(val)[1] == 1:
                res[key] = res[key][:, 0]

        if self.save['saveresdata']:
            np.save(self.paths['record_folder'] / 'res', res)

        return res
