#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 2022.

@author: julie-vienne
"""

import numpy as np
import pandapower as pp
import pandapower.networks


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
        for info in ['n_homes', 'M', 'N']:
            setattr(self, info, prm['syst'][info])
        self.homes = range(self.n_homes)

        # upper and lower voltage limits
        for info in [
            'max_voltage', 'min_voltage', 'penalty_undervoltage', 'penalty_overvoltage',
            'base_power', 'subset_line_losses_modelled', 'loss', 'weight_network_costs',
            'manage_agg_power', 'max_grid_import', 'penalty_import',
            'max_grid_export', 'penalty_export'
        ]:
            setattr(self, info, prm['grd'][info])

        self.network_data_path = prm['paths']['network_data']

        # ieee network and corresponding incidence matrix
        self.net = pandapower.networks.ieee_european_lv_asymmetric('on_peak_566')
        self.n_non_flex_homes = len(self.net.asymmetric_load) - self.n_homes
        self.loads_single_phase()
        self.in_incidence_matrix = np.where(self.incidence_matrix == -1, self.incidence_matrix, 0)
        self.out_incidence_matrix = np.where(self.incidence_matrix == 1, self.incidence_matrix, 0)

        # line data matrix
        self.line_resistance, self.line_reactance = self.network_line_data()

        # external grid: define grid voltage at 1.0 and slack bus as bus 1
        self.net.ext_grid['vm_pu'] = 1.0
        self.net.ext_grid['bus'] = 1

        self.n_losses_error = 0
        self.max_losses_error = - 1
        self.n_voltage_error = 0
        self.max_voltage_rel_error = - 1

    def _matrix_flexible_buses(self):
        """ Creates a matrix indicating at which bus there is a flexible agents """
        flex_buses = np.zeros((len(self.net.bus), self.n_homes))
        for i in range(self.n_homes):
            flex_buses[self.net.asymmetric_load['bus'][i], i] = 1
        return flex_buses

    def _matrix_non_flexible_buses(self):
        """ Creates a matrix indicating at which bus there is a non-flexible home """
        non_flex_buses = np.zeros((len(self.net.bus), self.n_non_flex_homes))
        for i in range(self.n_non_flex_homes):
            non_flex_buses[self.net.asymmetric_load['bus'][i + self.n_homes], i] = 1
        return non_flex_buses

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
        self.non_flex_buses = self._matrix_non_flexible_buses()

        if len(self.net.asymmetric_load) > 0:
            # Add single phase loads and sgen to originally loaded buses
            houses = self.net.asymmetric_load
            for i in range(len(houses)):
                pp.create_load(self.net, houses['bus'][i], p_mw=0, q_mvar=0, name=f"load{i+1}")
                pp.create_sgen(self.net, houses['bus'][i], p_mw=0, q_mvar=0, name=f"sgen{i+1}")
            # Remove asymmetric loads on the three phases
            for i in range(len(self.net.asymmetric_load)):
                self.net.asymmetric_load['in_service'] = False

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
        self.non_flex_buses = np.delete(self.non_flex_buses, (0), axis=0)

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

    def pf_simulation(self, netp: list):
        """ Given selected action, obtain voltage on buses and lines using pandapower """
        self.net.load.p_mw = 0
        self.net.sgen.p_mw = 0
        for home in self.homes:
            # Pandapower uses MW while the simulations uses kW
            # Add a load if netp < 0 or a generation if netp > 0
            if netp[home] >= 0:
                self.net.load['p_mw'].iloc[home] = netp[home] / 1000
            else:
                self.net.sgen['p_mw'].iloc[home] = abs(netp[home]) / 1000

        pp.runpp(self.net)
        self.loaded_buses = np.array(self.net.load.bus[self.net.load.p_mw > 0])
        self.sgen_buses = np.array(self.net.sgen.bus[self.net.sgen.p_mw > 0])
        hourly_line_losses = sum(self.net.res_line['pl_mw']) * 1e3
        voltage = np.array(self.net.res_bus['vm_pu'])

        return hourly_line_losses, voltage

    def _check_voltage_differences(self, res, time_step):
        do_pp_simulation = False
        # Results from pandapower
        hourly_line_losses_pp, voltage_pp = self.pf_simulation(res["netp"][:, time_step])

        # Voltage test
        abs_diff_voltage = abs(res['voltage'][:, time_step] - voltage_pp[1:])
        rel_diff_voltage = abs_diff_voltage / res['voltage'][:, time_step]
        max_rel_diff_voltage = max(rel_diff_voltage)
        if max_rel_diff_voltage > 0.1:
            print(
                f"The max diff of voltage between the optimizer and pandapower for hour {time_step}"
                f" is {max_rel_diff_voltage * 100}% ({max(abs_diff_voltage)} kWh) "
                f"at bus {np.argmax(rel_diff_voltage)}"
                f"The network will be simulated with pandapower to correct the voltages"
            )
            do_pp_simulation = True

        # Impact of voltage costs on total costs
        hourly_voltage_costs_pp = self.compute_voltage_costs(np.square(voltage_pp))
        abs_rel_voltage_error = abs(
            (res['hourly_voltage_costs'][time_step] - hourly_voltage_costs_pp)
            / res['total_costs']
        )
        if abs_rel_voltage_error > 1e-4:
            self.n_voltage_error += 1
            if abs_rel_voltage_error > self.max_voltage_rel_error:
                self.max_voltage_rel_error = abs_rel_voltage_error
            # print(
            #     f"Warning: The difference in voltage costs between "
            #     f"the optimisation and pandapower for hour {time_step} "
            #     f"is {abs_rel_voltage_error} of the total daily costs. "
            #     f"The network will be simulated with pandapower to correct the voltages."
            # )
            do_pp_simulation = True

        return do_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp, voltage_pp

    def _check_losses_differences(self, res, hourly_line_losses_pp, time_step):
        # Line losses test
        abs_loss_error = abs(res['hourly_line_losses'][time_step] - hourly_line_losses_pp)
        if abs_loss_error > 1e-2:
            self.n_losses_error += 1
            if abs_loss_error > self.max_losses_error:
                self.max_losses_error = abs_loss_error
            # print(
            #     f"Warning: The difference in hourly line losses "
            #     f"between pandapower and optimizer for hour {time_step} "
            #     f"is {abs(res['hourly_line_losses'][time_step] - hourly_line_losses_pp)}. "
            #     f"To increase accuracy, the user could increase the subset_line_losses_modelled "
            #     f"(currently: {self.subset_line_losses_modelled} lines)"
            # )

    def test_network_comparison_optimiser_pandapower(self, res, time_step, grdCt):
        """Compares hourly results from network modelling in optimizer and pandapower"""
        # Results from optimization
        do_pp_simulation, hourly_line_losses_pp, hourly_voltage_costs_pp, voltage_pp \
            = self._check_voltage_differences(res, time_step)
        self._check_losses_differences(res, hourly_line_losses_pp, time_step)
        if do_pp_simulation:
            res = self._replace_res_values_with_pp_simulation(
                res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt, voltage_pp
            )

        return res, hourly_line_losses_pp, hourly_voltage_costs_pp

    def _replace_res_values_with_pp_simulation(
            self, res, time_step, hourly_line_losses_pp, hourly_voltage_costs_pp, grdCt, voltage_pp
    ):
        # corrected hourly_line_losses and grid values
        delta_voltage_costs = hourly_voltage_costs_pp - res['hourly_voltage_costs'][time_step]

        grid_pp = \
            res["grid"][time_step] \
            - res["hourly_line_losses"][time_step] \
            + hourly_line_losses_pp
        hourly_grid_energy_costs_pp = grdCt * (
            grid_pp + self.loss * grid_pp ** 2
        )
        delta_grid_energy_costs = \
            hourly_grid_energy_costs_pp - res['hourly_grid_energy_costs'][time_step]

        import_export_costs_pp = self.compute_import_export_costs(grid_pp)
        delta_import_export_costs = \
            import_export_costs_pp - res['hourly_import_export_costs'][time_step]

        delta_total_costs = \
            delta_voltage_costs + delta_grid_energy_costs + delta_import_export_costs

        # update variable values given updated losses and voltages
        res["grid"][time_step] = grid_pp
        res["grid2"][time_step] = grid_pp ** 2
        res['voltage'][:, time_step] = voltage_pp[1:]
        res['voltage_squared'][:, time_step] = np.square(voltage_pp[1:])
        res["hourly_line_losses"][time_step] = hourly_line_losses_pp

        # update individual cost components
        res['hourly_import_export_costs'][time_step] = import_export_costs_pp
        res['import_export_costs'] += delta_import_export_costs

        res['hourly_voltage_costs'][time_step] = hourly_voltage_costs_pp
        res["voltage_costs"] += delta_voltage_costs

        res["hourly_grid_energy_costs"][time_step] = hourly_grid_energy_costs_pp
        res["grid_energy_costs"] += delta_grid_energy_costs

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
            import_export_costs = 0

        return import_export_costs

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
