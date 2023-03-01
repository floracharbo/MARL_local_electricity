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

from src.utilities.userdeftools import calculate_reactive_power


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

    Private methods
    _identify_duplicates_buses_lines
    _remove_duplicates_buses_lines

    """

    def __init__(self, prm):
        """
        Initialise Network object.

        inputs:
        prm:
            input parameters
        """
        for info in ['n_homes', 'M', 'N', 'n_homesP']:
            setattr(self, info, prm['syst'][info])
        self.homes = range(self.n_homes)
        self.homesP = range(self.n_homesP)

        # upper and lower voltage limits
        for info in [
            'max_voltage', 'min_voltage', 'penalty_undervoltage', 'penalty_overvoltage',
            'base_power', 'subset_line_losses_modelled', 'loss', 'weight_network_costs',
            'manage_agg_power', 'max_grid_import', 'penalty_import', 'max_grid_export',
            'penalty_export', 'reactive_power_for_voltage_control',
            'pf_passive_homes', 'pf_flexible_homes', 'tol_rel_voltage_diff',
            'tol_rel_voltage_costs', 'tol_abs_line_losses',
        ]:
            setattr(self, info, prm['grd'][info])

        if prm['grd']['manage_voltage']:
            self.network_data_path = prm['paths']['network_data']
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

    def _matrix_flexible_buses(self):
        """ Creates a matrix indicating at which bus there is a flexible agents """
        flex_buses = np.zeros((len(self.net.bus), self.n_homes))
        for i in range(self.n_homes):
            flex_buses[self.existing_homes_network[i], i] = 1
        return flex_buses

    def _matrix_passive_buses(self):
        """ Creates a matrix indicating at which bus there is a non-flexible home """
        if self.n_passive_homes > 0:
            passive_buses = np.zeros((len(self.net.bus), self.n_passive_homes))
            for i in range(self.n_passive_homes):
                passive_buses[self.existing_homes_network[i + self.n_homes], i] = 1
        else:
            passive_buses = np.zeros((len(self.net.bus), 1))
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
        self.passive_buses = self._matrix_passive_buses()

        if len(self.net.asymmetric_load) > 0:
            # Remove asymmetric loads on the three phases
            self.net.asymmetric_load['in_service'] = False
            # Add single phase loads and generations instead
            for home in self.homes:
                pp.create_load(self.net, bus=self.existing_homes_network[home],
                               p_mw=0, q_mvar=0, name=f'flex{home}')
                pp.create_sgen(self.net, bus=self.existing_homes_network[home],
                               p_mw=0, q_mvar=0, name=f'flex{home}')
            if self.n_homesP > 0:
                for homeP in self.homesP:
                    pp.create_load(self.net, bus=self.existing_homes_network[self.n_homes + homeP],
                                   p_mw=0, q_mvar=0, name=f'passive{homeP}')
                    pp.create_sgen(self.net, bus=self.existing_homes_network[self.n_homes + homeP],
                                   p_mw=0, q_mvar=0, name=f'passive{homeP}')

            # Remove bus duplicates
            # buscoords = pd.read_csv(self.network_data_path / 'Buscoords.csv', skiprows=1)
            # self._remove_duplicates_buses_lines(buscoords)

            # Remove zero sequence line resistance and reactance
            self.net.line['r0_ohm_per_km'] = None
            self.net.line['x0_ohm_per_km'] = None

        # Remove bus zero/source bus in matrices used in optimisation
        self.incidence_matrix = np.delete(self.incidence_matrix, (0), axis=0)
        self.bus_connection_matrix = np.delete(self.bus_connection_matrix, (0), axis=0)
        self.bus_connection_matrix = np.delete(self.bus_connection_matrix, (0), axis=1)
        self.flex_buses = np.delete(self.flex_buses, (0), axis=0)
        self.passive_buses = np.delete(self.passive_buses, (0), axis=0)

    def _identify_duplicates_buses_lines(self, buscoords):
        bus_duplicates = buscoords.duplicated(
            subset=[buscoords.columns[1], buscoords.columns[2]], keep=False
        )
        duplicates = bus_duplicates[bus_duplicates is True]
        duplicates.index += 1
        return list(duplicates.keys())

    def _remove_duplicates_buses_lines(self, buscoords):
        duplicates = self._identify_duplicates_buses_lines(buscoords)
        if len(duplicates) > 0:
            duplicated_buses = []
            duplicated_lines = []
            for i in range(len(duplicates)):
                for j in range(i + 1, len(duplicates)):
                    if (buscoords[' x'].iloc[duplicates[i] - 1]
                        == buscoords[' x'].iloc[duplicates[j] - 1]) & \
                            (buscoords[' y'].iloc[duplicates[i] - 1]
                             == buscoords[' y'].iloc[duplicates[j] - 1]):
                        if duplicates[j] not in self.net.load['bus'].values:
                            duplicated_lines.append(self.net.line.loc[self.net.line['to_bus']
                                                    == duplicates[j]].index[0])
                            duplicated_buses.append(duplicates[j])
                            break

            # remove duplicates from incidence matrix and bus connection matrix
            self.incidence_matrix = np.delete(self.incidence_matrix,
                                              duplicated_buses, axis=0)
            self.incidence_matrix = np.delete(self.incidence_matrix,
                                              duplicated_lines, axis=1)
            self.bus_connection_matrix = np.delete(self.bus_connection_matrix,
                                                   duplicated_buses, axis=0)
            self.bus_connection_matrix = np.delete(self.bus_connection_matrix,
                                                   duplicated_buses, axis=1)
            self.flex_buses = np.delete(self.flex_buses, duplicated_buses, axis=0)

        # remove duplicates from network
        self.net.line.drop(duplicated_lines, inplace=True)
        self.net.bus.drop(duplicated_buses, inplace=True)

    def pf_simulation(
            self,
            netp: list,
            netp0: list = None,
            netq_flex: list = None,
            netq_passive: list = None):
        start = time.time()
        """ Given selected action, obtain voltage on buses and lines using pandapower """
        # removing old loads
        for power in ['p_mw', 'q_mvar']:
            self.net.load[power] = 0
            self.net.sgen[power] = 0
        # assign flexible homes
        if self.n_homes > 0:
            for home in self.homes:
                self._assign_power_to_load_or_sgen(
                    netp[home], home, type='p_mw')
                self._assign_power_to_load_or_sgen(
                    netq_flex[home], home, type='q_mvar')
        # assign passive homes
        if self.n_homesP > 0:
            for homeP in self.homesP:
                self._assign_power_to_load_or_sgen(
                    netp0[homeP], self.n_homes + homeP, type='p_mw')
                self._assign_power_to_load_or_sgen(
                    netq_passive[homeP], self.n_homes + homeP, type='q_mvar')
        pp.runpp(self.net)
        self.loaded_buses = np.array(self.net.load.bus[self.net.load.p_mw >= 0])
        self.sgen_buses = np.array(self.net.sgen.bus[self.net.sgen.p_mw > 0])
        hourly_line_losses = sum(self.net.res_line['pl_mw']) * 1e3
        voltage = np.array(self.net.res_bus['vm_pu'])
        pij = self.net.res_line.p_from_mw * 1e3
        qij = self.net.res_line.q_from_mvar * 1e3
        end = time.time()
        duration_pp = end - start
        self.timer_pp.append(duration_pp)

        return hourly_line_losses, voltage, pij, qij

    def _assign_power_to_load_or_sgen(self, power, house_index, type):
        # pandapower uses MW while the simulations uses kW
        # add a load if power < 0 or a generation if power > 0
        if power >= 0:
            self.net.load[type].iloc[house_index] = power / 1000
        else:
            self.net.sgen[type].iloc[house_index] = abs(power) / 1000

    def _power_flow_res_with_pandapower(self, home_vars, netp0, q_car_flex):
        """Using active power, calculates reactive power and solves power flow
        with pandapower """

        if self.n_homesP > 0:
            q_heat_home_car_passive = calculate_reactive_power(
                netp0, self.pf_passive_homes)
        else:
            q_heat_home_car_passive = []
        q_heat_home_flex = calculate_reactive_power(
            home_vars['tot_cons'], self.pf_flexible_homes)
        q_solar_flex = calculate_reactive_power(
            home_vars['gen'], self.pf_flexible_homes)

        netq_flex = q_car_flex + q_heat_home_flex - q_solar_flex
        netq_passive = q_heat_home_car_passive

        #  import/export external grid
        q_ext_grid = sum(q_heat_home_car_passive) + sum(q_car_flex) \
            + sum(q_heat_home_flex)

        hourly_line_losses, voltage, _, _ = self.pf_simulation(
            home_vars['netp'], netp0, netq_flex, netq_passive
        )
        voltage_squared = np.square(voltage)

        return voltage_squared, hourly_line_losses, q_ext_grid

    def _check_voltage_differences(self, res, time_step, netp0, netq_flex, netq_passive):
        replace_with_pp_simulation = False
        # Results from pandapower
        hourly_line_losses_pp, voltage_pp, pij_pp_kW, qij_pp_kW = self.pf_simulation(
            res["netp"][:, time_step],
            netp0[:, time_step],
            netq_flex[:, time_step],
            netq_passive[:, time_step])

        # Voltage test
        all_abs_diff_voltage = abs(res['voltage'][:, time_step] - voltage_pp[1:])
        all_rel_diff_voltage = all_abs_diff_voltage / res['voltage'][:, time_step]
        max_rel_diff_voltage = max(all_rel_diff_voltage)
        self.max_rel_diff_voltage.append(max_rel_diff_voltage)
        self.mean_rel_diff_voltage.append(np.mean(all_rel_diff_voltage))
        self.std_rel_diff_voltage.append(np.std(all_rel_diff_voltage))

        if max_rel_diff_voltage > self.tol_rel_voltage_diff:
            replace_with_pp_simulation = True

        # Impact of voltage costs on total costs
        hourly_voltage_costs_pp = self.compute_voltage_costs(np.square(voltage_pp))
        abs_rel_voltage_error = abs(
            (res['hourly_voltage_costs'][time_step] - hourly_voltage_costs_pp)
            / res['total_costs']
        )
        if np.any(abs_rel_voltage_error > self.tol_rel_voltage_costs):
            if abs_rel_voltage_error > self.max_voltage_rel_error:
                self.max_voltage_rel_error = abs_rel_voltage_error
            replace_with_pp_simulation = True

        if replace_with_pp_simulation:
            with open(f"{self.folder_run}/voltage_comparison.txt", "a") as file:
                file.write(
                    f"The max diff of voltage between the optimiser and pandapower for time step "
                    f"{time_step} is {max_rel_diff_voltage * 100}% ({max(all_abs_diff_voltage)}V) "
                    f"at bus {np.argmax(all_rel_diff_voltage)}. "
                    f"The absolute difference of hourly voltage costs is "
                    f"{abs_rel_voltage_error * 100}% of the daily optimisation costs. "
                    f"The network will be simulated with pandapower to correct the voltages"
                )

        return [
            replace_with_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp,
            voltage_pp, pij_pp_kW, qij_pp_kW
        ]

    def _check_losses_differences(self, res, hourly_line_losses_pp, time_step):
        # Line losses test
        abs_loss_error = abs(res['hourly_line_losses'][time_step] - hourly_line_losses_pp)
        if abs_loss_error > self.tol_abs_line_losses:
            self.n_losses_error += 1
            if abs_loss_error > self.max_losses_error:
                self.max_losses_error = abs_loss_error
            replace_with_pp_simulation = True
            with open(f"{self.folder_run}/line_losses.txt", "a") as file:
                file.write(
                    f"The difference in hourly line losses "
                    f"between pandapower and the optimiser for time step {time_step} "
                    f"is {abs(res['hourly_line_losses'][time_step] - hourly_line_losses_pp)} kW. "
                    f"To increase accuracy, the user could increase the subset_line_losses_modelled"
                    f" (currently: {self.subset_line_losses_modelled} lines)\n"
                )
        else:
            replace_with_pp_simulation = False
        return replace_with_pp_simulation

    def compare_optimiser_pandapower(
            self, res, time_step, netp0, grdCt, line_losses_method):
        """Prepares the reactive power injected and compares optimization with pandapower"""
        if self.n_homesP > 0:
            netq_passive = calculate_reactive_power(
                netp0, self.grd['pf_passive_homes']
            )
        else:
            netq_passive = np.zeros([1, self.N])
            netp0 = np.zeros([1, self.N])

        # q_car_flex will be a decision variable
        q_heat_home_flex = calculate_reactive_power(res['totcons'], self.pf_flexible_homes)
        netq_flex = res['q_car_flex'] + q_heat_home_flex - res['q_solar_flex']

        # Compare hourly results from network modelling in optimizer and pandapower
        start = time.time()
        replace_with_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp, voltage_pp, \
            pij_pp_kW, qij_pp_kW = self._check_voltage_differences(
                res, time_step, netp0, netq_flex, netq_passive)
        replace_with_pp_simulation = self._check_losses_differences(
            res, hourly_line_losses_pp, time_step)
        if replace_with_pp_simulation or line_losses_method == 'iteration':
            res = self._replace_res_values_with_pp_simulation(
                res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt,
                voltage_pp, pij_pp_kW, qij_pp_kW
            )
        end = time.time()
        duration_comparison = end - start
        self.timer_comparison.append(duration_comparison)

        return res

    def _replace_res_values_with_pp_simulation(
            self, res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt, voltage_pp,
            pij_pp_kW, qij_pp_kW
    ):
        # corrected hourly_line_losses and grid values
        self.n_voltage_error += 1
        delta_voltage_costs = hourly_voltage_costs_pp - res['hourly_voltage_costs'][time_step]
        delta_hourly_line_losses = hourly_line_losses_pp - res["hourly_line_losses"][time_step]
        grid_pp = res["grid"][time_step] + delta_hourly_line_losses

        hourly_grid_energy_costs_pp = grdCt * (
            grid_pp + self.loss * grid_pp ** 2
        )
        delta_grid_energy_costs = \
            hourly_grid_energy_costs_pp - res['hourly_grid_energy_costs'][time_step]

        import_export_costs_pp, _, _ = self.compute_import_export_costs(grid_pp)
        delta_import_export_costs = \
            import_export_costs_pp - res['hourly_import_export_costs'][time_step]

        delta_total_costs = \
            delta_voltage_costs + delta_grid_energy_costs + delta_import_export_costs

        # update variable values given updated losses and voltages
        res["grid"][time_step] = grid_pp
        res["grid2"][time_step] = grid_pp ** 2
        res['voltage'][:, time_step] = voltage_pp[1:]
        res['voltage_squared'][:, time_step] = np.square(voltage_pp[1:])
        res["hourly_line_losses"][time_step] += delta_hourly_line_losses
        res["v_line"][:, time_step] = np.matmul(
            self.out_incidence_matrix.T,
            res["voltage_squared"][:, time_step])
        res["pij"][:, time_step] = pij_pp_kW * 1000 / self.base_power
        res["qij"][:, time_step] = qij_pp_kW * 1000 / self.base_power
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
        res["network_costs"] += delta_voltage_costs * self.weight_network_costs
        res["hourly_total_costs"][time_step] += delta_total_costs
        res["total_costs"] += delta_total_costs

        sum_indiv_components = \
            (
                res['hourly_import_export_costs'][time_step]
                + res['hourly_voltage_costs'][time_step]
            ) * self.weight_network_costs \
            + res['hourly_grid_energy_costs'][time_step] \
            + res['hourly_battery_degradation_costs'][time_step] \
            + res['hourly_distribution_network_export_costs'][time_step]
        assert abs(sum_indiv_components - res['hourly_total_costs'][time_step]) < 1e-4, \
            "total hourly costs do not add up"
        control_sum_grid = sum(res['netp'][:, time_step]) + res['hourly_line_losses'][time_step]
        assert abs(res['grid'][time_step] - control_sum_grid) < 1e-3

        return res

    def compute_import_export_costs(self, grid):
        if self.manage_agg_power:
            grid_in = np.where(np.array(grid) >= 0, grid, 0)
            grid_out = np.where(np.array(grid) < 0, - grid, 0)
            import_costs = np.where(
                grid_in >= self.max_grid_import,
                self.penalty_import * (grid_in - self.max_grid_import),
                0
            )
            export_costs = np.where(
                grid_out >= self.max_grid_export,
                self.penalty_export * (grid_out - self.max_grid_export),
                0
            )
            import_export_costs = import_costs + export_costs
        else:
            import_export_costs, import_costs, export_costs = 0, 0, 0

        return import_export_costs, import_costs, export_costs

    def compute_voltage_costs(self, voltage_squared):
        over_voltage_costs = self.penalty_overvoltage * np.where(
            voltage_squared > self.max_voltage ** 2,
            voltage_squared - self.max_voltage ** 2,
            0
        )
        under_voltage_costs = self.penalty_undervoltage * np.where(
            voltage_squared < self.min_voltage ** 2,
            self.min_voltage ** 2 - voltage_squared,
            0
        )

        return np.sum(over_voltage_costs + under_voltage_costs)
