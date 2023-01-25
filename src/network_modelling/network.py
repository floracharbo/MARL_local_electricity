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
        for info in ['v_mag_over', 'v_mag_under', 'penalty_undervoltage', 'penalty_overvoltage']:
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

    def compute_over_under_voltage_bus(self, voltage_mag):
        overvoltage_bus = voltage_mag[np.square(voltage_mag) > self.v_mag_over ** 2]
        undervoltage_bus = voltage_mag[np.square(voltage_mag) < self.v_mag_under ** 2]

        return overvoltage_bus, undervoltage_bus

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
        overvoltage_bus, undervoltage_bus = self.compute_over_under_voltage_bus(
            self.net.res_bus['vm_pu']
        )
        loaded_buses = np.array(self.net.load.bus[self.net.load.p_mw > 0])
        sgen_buses = np.array(self.net.sgen.bus[self.net.sgen.p_mw > 0])
        hourly_line_losses = sum(self.net.res_line['pl_mw']) * 1e3
        v_mag = np.array(self.net.res_bus['vm_pu'])

        return [
            overvoltage_bus, undervoltage_bus, loaded_buses,
            sgen_buses, hourly_line_losses, v_mag
        ]
