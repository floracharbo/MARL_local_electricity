#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 2022.

@author: julie-vienne
"""

import random
import time

import numpy as np
import pandapower as pp
import pandapower.networks

from src.environment.utilities.userdeftools import (
    compute_import_export_costs, compute_voltage_costs)


class Network:
    """
    IEEE Low Voltage European Distrubtion Network.

    Public methods:
    network_line_data:
        Create a table with data on the network lines.
    network_incidence_matrix:
        Create the network incidence matrix.
    loads_single_phase:
        Transform a three phase loaded network to a single phase one
    pf_simulation:
        Run the hourly pandapower simulation on the network
    network_line_data:
        Extract the resistance and reactance from the network
    compare_optimiser_pandapower:
        Pre and post processing of comparison between load flow and opti

    Private methods:
    _matrix_flexible_buses and _matrix_passive_buses:
        Create matrices on agents location for the optimization of the network
    _network_incidence_matrix and _network_bus_connection:
        Create matrices describing the lines and buses of the network
    _assign_power_to_load_or_sgen:
        Distribute the correct active and reative loads to the buses
    _power_flow_res_with_pandapower:
        Pre and post processing of pf_simulation
    _check_voltage_differences and _check_losses_differences:
        Perform a comparison between pandapower load flow and optimization
    _replace_res_values_with_pp_simulation:
        If required, replaces the opti results with accurate load flow results
    """

    def __init__(self, prm):
        """
        Initialise Network object.

        inputs:
        prm:
            input parameters
        """
        for info in [
            'n_homes', 'n_homesP', 'n_homes_test', 'n_homes_testP',
            'n_int_per_hr', 'M', 'N'
        ]:
            setattr(self, info, prm['syst'][info])
        self.homes = range(self.n_homes)
        self.homesP = range(self.n_homesP)

        # upper and lower voltage limits
        self.grd = {}
        for info in [
            'base_power', 'loss', 'weight_network_costs', 'active_to_reactive_passive',
            'active_to_reactive_flex', 'tol_rel_voltage_diff', 'tol_rel_voltage_costs',
            'tol_abs_line_losses', 'penalty_overvoltage', 'penalty_undervoltage',
            'max_voltage', 'min_voltage', 'manage_agg_power', 'penalty_import',
            'penalty_export', 'max_grid_import', 'max_grid_export'
        ]:
            self.grd[info] = prm['grd'][info]

        if prm['grd']['manage_voltage'] or prm['grd']['simulate_panda_power_only']:
            self.folder_run = prm['paths']['folder_run']

            # ieee network and corresponding incidence matrix
            self.net = pandapower.networks.ieee_european_lv_asymmetric('on_peak_566')
            self.n_passive_homes = self.n_homesP

            # replacing asymmetric loads with single-phase
            self.existing_homes_network = list(self.net.asymmetric_load['bus'])
            random.shuffle(self.existing_homes_network)
            self.loads_single_phase()

            self.in_incidence_matrix = np.where(
                self.incidence_matrix == -1, self.incidence_matrix, 0
            )
            self.out_incidence_matrix = np.where(
                self.incidence_matrix == 1, self.incidence_matrix, 0
            )

            # line data matrix
            self.line_resistance, self.line_reactance = self.network_line_data()

            # external grid: define grid voltage at 1.0 and slack bus as bus 1
            self.net.ext_grid['vm_pu'] = 1.0
            self.net.ext_grid['bus'] = 1

            self.max_losses_error = - 1
            self.max_voltage_rel_error = - 1

            for attribute in ['n_voltage_error', 'n_losses_error']:
                setattr(self, attribute, 0)
            for attribute in [
                'timer_pp', 'timer_comparison', 'max_rel_diff_voltage',
                'mean_rel_diff_voltage', 'std_rel_diff_voltage'
            ]:
                setattr(self, attribute, [])

        self.homes = range(self.n_homes)

    def _matrix_flexible_buses(self, test=False):
        """ Creates a matrix indicating at which bus there is a flexible agents """
        n_homes = self.n_homes_test + self.n_homes_testP if test else self.n_homes + self.n_homesP
        flex_buses = np.zeros((len(self.net.bus), n_homes))
        free_buses = [i for i in range(len(self.net.bus)) if i not in self.existing_homes_network]
        free_buses.remove(0)
        self.home_to_bus = np.zeros(n_homes)
        for i in range(n_homes):
            if i < len(self.existing_homes_network):
                flex_buses[self.existing_homes_network[i], i] = 1
                self.home_to_bus[i] = self.existing_homes_network[i]
            else:
                bus = random.choice(free_buses)
                flex_buses[bus, i] = 1
                free_buses.remove(bus)
                self.home_to_bus[i] = bus

        return flex_buses

    def _matrix_passive_buses(self):
        """ Creates a matrix indicating at which bus there is a non-flexible home """
        if self.n_passive_homes > 0:
            passive_buses = np.zeros((len(self.net.bus), self.n_passive_homes))
            for i in range(self.n_passive_homes):
                passive_buses[self.existing_homes_network[i + self.n_homes], i] = 1
        else:
            passive_buses = np.zeros((len(self.net.bus), 0))

        return passive_buses

    def network_line_data(self):
        """ Returns line resistance and reactance arrays from pandapower network """
        line_data = self.net.line[['from_bus', 'to_bus',
                                  'r_ohm_per_km', 'x_ohm_per_km', 'length_km']]
        line_resistance = np.asarray(line_data['r_ohm_per_km'] * line_data['length_km'])
        line_reactance = np.asarray(line_data['x_ohm_per_km'] * line_data['length_km'])
        return line_resistance, line_reactance

    def _network_incidence_matrix(self):
        """ Returns incidence matrix connecting the buses and lines of the network """
        incidence_matrix = np.zeros((len(self.net.bus), len(self.net.line)))
        for i in range(len(self.net.line)):
            incidence_matrix[self.net.line['from_bus'].iloc[i], i] = 1
            incidence_matrix[self.net.line['to_bus'].iloc[i], i] = -1
        return incidence_matrix

    def _network_bus_connection(self):
        """ Returns a matrix connecting each bus to its neighbour """
        bus_connection_matrix = np.zeros((len(self.net.bus), len(self.net.bus)))
        for i in range(len(self.net.line)):
            bus_connection_matrix[self.net.line['to_bus'].iloc[i],
                                  self.net.line['from_bus'].iloc[i]] = 1
        return bus_connection_matrix

    def loads_single_phase(self):
        """ Replaces asymetric loads with single phase and removes bus zero for optimization """
        # Generate incidence matrix
        self.incidence_matrix = self._network_incidence_matrix()
        # Generate bus connection matrix
        self.bus_connection_matrix = self._network_bus_connection()
        # Generate matice of (non) flexible buses/loads
        self.flex_buses = self._matrix_flexible_buses()
        self.flex_buses_test = self._matrix_flexible_buses(test=True)

        self.passive_buses = self._matrix_passive_buses()

        if len(self.net.asymmetric_load) > 0:
            # Remove asymmetric loads on the three phases
            self.net.asymmetric_load['in_service'] = False
            # Add single phase loads and generations instead
            for home in self.homes:
                pp.create_load(
                    self.net, bus=self.home_to_bus[home],
                    p_mw=0, q_mvar=0, name=f'flex{home}'
                )
                pp.create_sgen(
                    self.net, bus=self.home_to_bus[home],
                    p_mw=0, q_mvar=0, name=f'flex{home}')
            if self.n_homesP > 0:
                for homeP in self.homesP:
                    pp.create_load(
                        self.net, bus=self.home_to_bus[self.n_homes + homeP],
                        p_mw=0, q_mvar=0, name=f'passive{homeP}'
                    )
                    pp.create_sgen(
                        self.net, bus=self.home_to_bus[self.n_homes + homeP],
                        p_mw=0, q_mvar=0, name=f'passive{homeP}'
                    )

            # Remove zero sequence line resistance and reactance
            self.net.line['r0_ohm_per_km'] = None
            self.net.line['x0_ohm_per_km'] = None

        # Remove bus zero/source bus in matrices used in optimisation
        self.incidence_matrix = np.delete(self.incidence_matrix, (0), axis=0)
        self.bus_connection_matrix = np.delete(self.bus_connection_matrix, (0), axis=0)
        self.bus_connection_matrix = np.delete(self.bus_connection_matrix, (0), axis=1)
        self.flex_buses = np.delete(self.flex_buses, (0), axis=0)
        self.flex_buses_test = np.delete(self.flex_buses_test, (0), axis=0)
        self.passive_buses = np.delete(self.passive_buses, (0), axis=0)

    def pf_simulation(
            self,
            netp: list,
            netp0: list = None,
            netq_flex: list = None,
            netq_passive: list = None,
            passive=False
    ):
        start = time.time()
        """ Given selected action, obtain voltage on buses and lines using pandapower """
        # removing old loads
        for power in ['p_mw', 'q_mvar']:
            self.net.load[power] = 0
            self.net.sgen[power] = 0

        # assign flexible homes
        if self.n_homes > 0 and not passive:
            for home in self.homes:
                self._assign_power_to_load_or_sgen(netp[home], home, type='p_mw')
                self._assign_power_to_load_or_sgen(netq_flex[home], home, type='q_mvar')

        # assign passive homes
        if self.n_homesP > 0:
            for homeP in self.homesP:
                self._assign_power_to_load_or_sgen(
                    netp0[homeP], self.n_homes + homeP, type='p_mw'
                )
                self._assign_power_to_load_or_sgen(
                    netq_passive[homeP], self.n_homes + homeP, type='q_mvar'
                )
        pp.runpp(self.net)
        self.loaded_buses = np.array(self.net.load.bus[self.net.load.p_mw > 0])
        self.sgen_buses = np.array(self.net.sgen.bus[self.net.sgen.p_mw > 0])
        hourly_line_losses = sum(self.net.res_line['pl_mw']) * 1e3
        hourly_reactive_power_losses = sum(self.net.res_line['ql_mvar']) * 1e3
        voltage = np.array(self.net.res_bus['vm_pu'])
        pij = self.net.res_line.p_from_mw * 1e3
        qij = self.net.res_line.q_from_mvar * 1e3
        end = time.time()
        duration_pp = end - start
        self.timer_pp.append(duration_pp)

        return hourly_line_losses, voltage, pij, qij, hourly_reactive_power_losses

    def _assign_power_to_load_or_sgen(self, power, house_index, type):
        # pandapower uses MW while the simulations uses kW
        # add a load if power < 0 or a generation if power > 0
        if power >= 0:
            self.net.load[type].iloc[house_index] = power / 1000
        else:
            self.net.sgen[type].iloc[house_index] = abs(power) / 1000

    def _power_flow_res_with_pandapower(self, home_vars, netp0, q_car_flex, passive=False):
        """Using active power, calculates reactive power and solves power flow
        with pandapower """

        if self.n_homesP > 0:
            q_heat_home_car_passive = netp0 * self.grd['active_to_reactive_passive']
        else:
            q_heat_home_car_passive = []
        q_heat_home_flex = home_vars['tot_cons'] * self.grd['active_to_reactive_flex']
        q_solar_flex = home_vars['gen'] * self.grd['active_to_reactive_flex']

        netq_flex = q_car_flex + q_heat_home_flex - q_solar_flex
        netq_passive = q_heat_home_car_passive

        hourly_line_losses, voltage, _, _, reactive_power_losses = self.pf_simulation(
            home_vars['netp'], netp0, netq_flex, netq_passive, passive=passive
        )
        #  import/export external grid
        q_ext_grid = sum(q_heat_home_car_passive) + sum(q_car_flex) \
            + sum(q_heat_home_flex) + reactive_power_losses
        voltage_squared = np.square(voltage)

        return voltage_squared, hourly_line_losses, q_ext_grid, netq_flex

    def _check_voltage_differences(self, res, time_step, netp0, netq_flex, netq_passive):
        replace_with_pp_simulation = False
        # Results from pandapower
        hourly_line_losses_pp, voltage_pp, pij_pp_kW, \
            qij_pp_kW, reactive_power_losses = self.pf_simulation(
                res["netp"][:, time_step],
                netp0,
                netq_flex[:, time_step],
                netq_passive
            )

        # Voltage test
        all_abs_diff_voltage = abs(res['voltage'][:, time_step] - voltage_pp[1:])
        all_rel_diff_voltage = all_abs_diff_voltage / res['voltage'][:, time_step]
        max_rel_diff_voltage = max(all_rel_diff_voltage)
        self.max_rel_diff_voltage.append(max_rel_diff_voltage)
        self.mean_rel_diff_voltage.append(np.mean(all_rel_diff_voltage))
        self.std_rel_diff_voltage.append(np.std(all_rel_diff_voltage))

        if max_rel_diff_voltage > self.grd['tol_rel_voltage_diff']:
            replace_with_pp_simulation = True

        # Impact of voltage costs on total costs
        hourly_voltage_costs_pp = compute_voltage_costs(
            np.square(voltage_pp), self.grd
        )
        abs_rel_voltage_error = abs(
            (res['hourly_voltage_costs'][time_step] - hourly_voltage_costs_pp)
            / res['total_costs'])
        if np.any(abs_rel_voltage_error > self.grd['tol_rel_voltage_costs']):
            if abs_rel_voltage_error > self.max_voltage_rel_error:
                self.max_voltage_rel_error = abs_rel_voltage_error
            replace_with_pp_simulation = True

        return [
            replace_with_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp,
            voltage_pp, pij_pp_kW, qij_pp_kW, reactive_power_losses
        ]

    def _check_losses_differences(
            self, res, hourly_line_losses_pp, time_step, replace_with_pp_simulation
    ):
        # Line losses test
        abs_loss_error = abs(res['hourly_line_losses'][time_step] - hourly_line_losses_pp)
        if abs_loss_error > self.grd['tol_abs_line_losses']:
            self.n_losses_error += 1
            if abs_loss_error > self.max_losses_error:
                self.max_losses_error = abs_loss_error
            replace_with_pp_simulation = True

        return replace_with_pp_simulation

    def compare_optimiser_pandapower(
            self, res, time_step, netp0, grdCt):
        """Prepares the reactive power injected and compares optimization with pandapower"""
        if self.n_homesP > 0:
            netq_passive = netp0 * self.grd['active_to_reactive_passive']
        else:
            netq_passive = 0
            netp0 = 0
        # Compare hourly results from network modelling in optimizer and pandapower
        start = time.time()
        [
            replace_with_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp,
            voltage_pp, pij_pp_kW, qij_pp_kW, reactive_power_losses
        ] = self._check_voltage_differences(
            res, time_step, netp0, res['netq_flex'], netq_passive
        )
        replace_with_pp_simulation = self._check_losses_differences(
            res, hourly_line_losses_pp, time_step, replace_with_pp_simulation
        )
        if replace_with_pp_simulation:
            res = self._replace_res_values_with_pp_simulation(
                res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt,
                voltage_pp, pij_pp_kW, qij_pp_kW, reactive_power_losses
            )
        end = time.time()
        duration_comparison = end - start
        self.timer_comparison.append(duration_comparison)

        return res

    def _replace_res_values_with_pp_simulation(
            self, res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt, voltage_pp,
            pij_pp_kW, qij_pp_kW, reactive_power_losses
    ):
        # corrected hourly_line_losses and grid values
        self.n_voltage_error += 1
        delta_voltage_costs = hourly_voltage_costs_pp - res['hourly_voltage_costs'][time_step]
        delta_hourly_active_line_losses = \
            hourly_line_losses_pp - res["hourly_line_losses"][time_step]
        grid_pp = res["grid"][time_step] + delta_hourly_active_line_losses
        delta_hourly_reactive_line_losses = \
            reactive_power_losses - res["hourly_reactive_losses"][time_step]
        q_ext_grid_pp = res["q_ext_grid"][time_step] + delta_hourly_reactive_line_losses

        hourly_grid_energy_costs_pp = grdCt * (
            grid_pp + self.grd['loss'] * grid_pp ** 2
        )
        delta_grid_energy_costs = \
            hourly_grid_energy_costs_pp - res['hourly_grid_energy_costs'][time_step]

        import_export_costs_pp, _, _ = compute_import_export_costs(
            grid_pp, self.grd, self.n_int_per_hr
        )
        delta_import_export_costs = \
            import_export_costs_pp - res['hourly_import_export_costs'][time_step]

        delta_total_costs = \
            delta_voltage_costs + delta_grid_energy_costs + delta_import_export_costs

        # update variable values given updated losses and voltages
        res["grid"][time_step] = grid_pp
        res["grid2"][time_step] = grid_pp ** 2
        res["q_ext_grid"][time_step] = q_ext_grid_pp
        res['voltage'][:, time_step] = voltage_pp[1:]
        res['voltage_squared'][:, time_step] = np.square(voltage_pp[1:])
        res["hourly_reactive_losses"][time_step] += delta_hourly_reactive_line_losses
        res["hourly_line_losses"][time_step] += delta_hourly_active_line_losses
        res["v_line"][:, time_step] = np.matmul(
            self.out_incidence_matrix.T,
            res["voltage_squared"][:, time_step])
        res["pij"][:, time_step] = pij_pp_kW * 1000 / self.grd['base_power']
        res["qij"][:, time_step] = qij_pp_kW * 1000 / self.grd['base_power']
        res["lij"][:, time_step] = np.divide(
            (np.square(res["pij"][:, time_step]) + np.square(res["qij"][:, time_step])),
            res["v_line"][:, time_step])

        # update individual cost components
        res['hourly_import_export_costs'][time_step] = import_export_costs_pp
        res['import_export_costs'] += delta_import_export_costs

        res['hourly_voltage_costs'][time_step] = hourly_voltage_costs_pp
        res["voltage_costs"] += delta_voltage_costs

        res["hourly_grid_energy_costs"][time_step] = hourly_grid_energy_costs_pp
        res["grid_energy_costs"] = np.sum(res["hourly_grid_energy_costs"])

        # update total costs
        res["network_costs"] += (delta_import_export_costs + delta_voltage_costs) \
            * self.grd['weight_network_costs']
        res["hourly_total_costs"][time_step] += delta_total_costs
        res["total_costs"] += delta_total_costs

        sum_indiv_components = \
            (
                res['hourly_import_export_costs'][time_step]
                + res['hourly_voltage_costs'][time_step]
            ) * self.grd['weight_network_costs'] \
            + res['hourly_grid_energy_costs'][time_step] \
            + res['hourly_battery_degradation_costs'][time_step] \
            + res['hourly_distribution_network_export_costs'][time_step]
        assert abs(sum_indiv_components - res['hourly_total_costs'][time_step]) < 1e-4, \
            "total hourly costs do not add up"
        assert abs(res["total_costs"] - sum(res['hourly_total_costs'])) < 1e-3, \
            "total costs do not match sum of hourly costs"
        abs_diff = abs(
            res['grid'][time_step]
            - sum(res['netp'][:, time_step])
            - res['hourly_line_losses'][time_step]
            - sum(res['netp0'][:, time_step])
        )
        assert abs_diff < 1e-3

        return res
