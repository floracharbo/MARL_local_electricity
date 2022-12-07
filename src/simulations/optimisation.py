#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  7 17:10:28 2020.

@author: floracharbonnier

"""

import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import picos as pic

from src.utilities.userdeftools import comb


class Optimiser():
    """The Optimiser object manages convex optimisations."""

    def __init__(self, prm):
        """Initialise solver object for doing optimisations."""
        self.N = prm["syst"]["N"]
        self.n_homes = prm["syst"]["n_homes"]
        self.save = prm["save"]
        self.paths = prm["paths"]
        self.manage_agg_power = prm["grd"]["manage_agg_power"]

    def solve(self, prm):
        """Solve optimisation problem given prm input data."""
        self._update_prm(prm)
        res = self._problem()
        if prm['car']['efftype'] == 1:
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
                res = self._problem(prm)
                print(f"res['constl(0, 0)'][0][0] "
                      f"= {res['constl(0, 0)'][0][0]}")
                if prm['grd']['loads'][0][0][0] < res['constl(0, 0)'][0][0]:
                    print('fixed loads smaller than fixed onsumption home=0 time=0')
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

    def plot_results(self, res, prm, folder=None):
        """Plot the optimisation results for homes."""
        store = res['store']
        grid = res['grid']
        totcons = res['totcons']
        netp = res['netp']

        time = np.arange(0, prm['syst']['N'])
        font = {'size': 22}
        matplotlib.rc('font', **font)

        if self.save['plot_inputs']:
            lin, fin, cin = self._plot_inputs(prm, time)

        if self.save['plot_results']:
            if prm['syst']['pu'] == 0:
                y_label = '[kWh]'
            elif prm['syst']['pu'] == 1:
                y_label = 'p.u.'

            # results
            figs = {}
            labels = {}
            count = 0

            # storage level over time
            figs[count] = plt.figure()
            labels[count] = 'storage'
            self._plot_y(prm, store, time)
            plt.title('Storage levels')
            plt.xlabel('Time [h]')
            plt.ylabel(y_label)
            if abs(np.max(store) - np.min(store)) < 1e-2:
                plt.ylim(np.max(store) - 0.5, np.max(store) + 0.5)
            plt.tight_layout()

            # grid import
            count += 1
            figs[count] = plt.figure()
            labels[count] = 'gridimport'
            plt.plot(time, grid)
            plt.title('Total import from grid')
            plt.xlabel('Time [h]')
            plt.ylabel(y_label)
            plt.tight_layout()

            # consumption
            count += 1
            figs[count] = plt.figure()
            labels[count] = 'cons'
            self._plot_y(prm, totcons, time)
            plt.title('Consumption')
            plt.xlabel('Time [h]')
            plt.ylabel(y_label)
            plt.tight_layout()

            # Net import
            count += 1
            figs[count] = plt.figure()
            labels[count] = 'prosumerimport'
            self._plot_y(prm, netp, time)

            plt.title('Net import')
            plt.xlabel('Time [h]')
            plt.ylabel(y_label)
            plt.tight_layout()

            # curtailment
            count += 1
            figs[count] = plt.figure()
            labels[count] = 'curtailment'
            self._plot_y(prm, res['curt'], time)
            plt.title('Curtailment')
            plt.xlabel('Time [h]')
            plt.ylabel(y_label)
            plt.tight_layout()

        save_res_path = os.path.join(folder, 'saveres')
        for i in range(count + 1):
            figs[i].savefig(os.path.join(save_res_path, labels[i]))

        save_inputs_path = os.path.join(folder, 'saveinputs')
        if os.path.exists(save_inputs_path) == 0:
            os.mkdir(save_inputs_path)
        for i in range(cin + 1):
            fin[i].savefig(os.path.join(save_inputs_path, lin[i]))

    def _grid_constraints(self, p):
        # variables
        grid = p.add_variable('grid', (self.N), vtype='continuous')
        netp = p.add_variable('netp', (self.n_homes, self.N), vtype='continuous')
        grid2 = p.add_variable('grid2', (self.N), vtype='continuous')

        # constraints
        # substation energy balance
        p.add_list_of_constraints(
            [grid[time]
             - np.sum(self.loads['netp0'][b0][time]
                      for b0 in range(self.syst['n_homesP']))
             - pic.sum([netp[home, time] for home in range(self.n_homes)])
             == 0 for time in range(self.N)])
        p.add_list_of_constraints(
            [grid2[time] >= grid[time] * grid[time] for time in range(self.N)])
        p, pc = self._import_export_constraints(p, grid)

        return p, netp, grid, grid2, pc

    def _problem(self):
        """Solve optimisation problem."""
        # initialise problem
        p = pic.Problem()

        p, charge, discharge_tot, discharge_other = self._storage_constraints(p)
        p, E_heat = self._temperature_constraints(p)
        p, totcons = self._cons_constraints(p, E_heat)
        p, netp, grid, grid2, pc = self._grid_constraints(p)

        # prosumer energy balance
        p.add_constraint(netp - charge / self.car['eta_ch']
                         + discharge_other
                         + self.grd['gen'][:, 0: self.N]
                         - totcons == 0)

        # costs constraints
        p = self._set_objective(p, netp, grid, discharge_tot, charge, grid2, pc)

        # solve
        p.solve(verbose=0, solver=self.syst['solver'])

        # save results
        res = self._save_results(p.variables)

        return res

    def _storage_constraints(self, p):
        """Storage constraints."""
        charge = p.add_variable('charge', (self.n_homes, self.N),
                                vtype='continuous')
        discharge_tot = p.add_variable('discharge_tot', (self.n_homes, self.N),
                                       vtype='continuous')
        discharge_other = p.add_variable('discharge_other', (self.n_homes, self.N),
                                         vtype='continuous')
        store = p.add_variable('store', (self.n_homes, self.N), vtype='continuous')

        store_end = self.car['SoC0'] * self.grd['Bcap'][:, self.N - 1]
        car = self.car

        # battery energy balance
        p.add_constraint(discharge_tot
                         == discharge_other / self.car['eta_dis']
                         + self.car['batch_loads_car'][:, 0: self.N])

        if car['eff'] == 1:
            p.add_list_of_constraints(
                [charge[:, time] - discharge_tot[:, time]
                 == store[:, time + 1] - store[:, time]
                 for time in range(self.N - 1)])
            p.add_constraint(store[:, self.N - 1]
                             + charge[:, self.N - 1]
                             - discharge_tot[:, self.N - 1]
                             >= store_end)

        elif car['eff'] == 2:
            for home in range(self.n_homes):
                p.add_constraint(
                    car['eta_ch'][home, self.N - 1] * charge[home, self.N - 1]
                    - car['eta_ch'][home, self.N - 1]
                    * discharge_tot[home, self.N - 1]
                    == store_end[home] - store[home, self.N - 1]
                )
                for time in range(self.N - 1):
                    p.add_constraint(
                        car['eta_ch'][home, time] * charge[home, time]
                        - car['eta_dis'][home, time] * discharge_tot[home, time]
                        == store[home, time + 1] - store[home, time]
                    )

        # initialise storage
        p.add_list_of_constraints(
            [store[home, 0] == car['SoC0'] * self.grd['Bcap'][home, 0]
             for home in range(self.n_homes)])

        p.add_list_of_constraints(
            [store[:, time + 1] >= car['SoCmin']
             * self.grd['Bcap'][:, time] * car['batch_avail_car'][:, time]
             for time in range(self.N - 1)])

        # if EV not avail at a given time step,
        # it is ok to start the following time step with less than minimum
        p.add_list_of_constraints(
            [store[:, time + 1] >= car['baseld'] * car['batch_avail_car'][:, time]
             for time in range(self.N - 1)])

        # can charge only when EV is available
        p.add_constraint(
            charge <= car['batch_avail_car'][:, 0: self.N] * self.syst['M'])

        # can discharge only when EV is available (Except EV cons is ok)
        p.add_constraint(
            discharge_other
            <= car['batch_avail_car'][:, 0: self.N] * self.syst['M']
        )
        p.add_constraint(store <= self.grd['Bcap'])
        p.add_constraint(car['c_max'] >= charge)
        p.add_constraint(car['d_max'] >= discharge_tot)
        p.add_constraint(store >= 0)
        p.add_constraint(charge >= 0)
        p.add_constraint(discharge_tot >= 0)
        p.add_constraint(discharge_other >= 0)

        return p, charge, discharge_tot, discharge_other

    def _distribution_costs(self, p, dc, netp):
        """Distribution costs."""
        grd, syst = self.grd, self.syst
        if grd['charge_type'] == 0:
            netp_abs = p.add_variable('netp_abs', (self.n_homes, self.N),
                                      vtype='continuous')
            netp0_abs = np.sum(
                [[abs(self.loads['netp0'][b0][time])
                  if self.loads['netp0'][b0][time] < 0
                  else 0
                  for b0 in range(syst['n_homesP'])] for time in range(self.N)])
            p.add_constraint(netp_abs >= - netp)
            p.add_constraint(netp_abs >= 0)
            # distribution costs
            p.add_constraint(dc == grd['export_C'] * (pic.sum(netp_abs) + netp0_abs))
        else:
            netp2 = p.add_variable('netp2', (self.n_homes, self.N),
                                   vtype='continuous')
            sum_netp2 = np.sum([[self.loads['netp0'][b0][time] ** 2
                                 for b0 in range(syst['n_homesP'])]
                                for time in range(self.N)])
            for home in range(self.n_homes):
                p.add_list_of_constraints(
                    [netp2[home, time] >= netp[home, time] * netp[home, time]
                     for time in range(self.N)])
            p.add_constraint(dc == grd['export_C'] * (pic.sum(netp2) + sum_netp2))

        return p

    def _update_prm(self, prm):
        """Update parameters with new values"""
        if isinstance(prm, (list, tuple)):
            self.syst, self.loads, self.car, self.grd, self.heat \
                = prm
        else:
            self.syst, self.loads, self.car, self.grd, self.heat \
                = [prm[e]
                   for e in ['syst', 'loads', 'car', 'grd', 'heat']]

    def _plot_y(self, prm, y, time):
        for home in range(prm['syst']['n_homes']):
            yb = np.reshape(y[home, :], (prm['syst']['N']))
            plt.plot(time, yb, label=f'agent {home}')

    def _plot_inputs(self, prm, time):
        """Plot inputs"""
        n_homes = prm['syst']['n_homes']
        # inputs
        fin = {}
        lin = {}
        cin = 0

        # PV
        fin[cin] = plt.figure()
        lin[cin] = 'PVgen'
        self._plot_y(prm, prm['grd']['gen'], time)
        plt.title('PV generation')
        plt.xlabel('Time [h]')
        plt.ylabel('[kWh]')
        plt.legend()
        plt.tight_layout()

        # grid import price
        cin += 1
        fin[cin] = plt.figure()
        lin[cin] = 'importprice'
        plt.plot(time, prm['grd']['C'])
        plt.xlabel('Time [h]')
        plt.ylabel('grid price [GBP/kWh]')
        plt.tight_layout()

        # demand
        cin += 1
        fin[cin] = plt.figure()
        lin[cin] = 'demand'
        for home in range(n_homes):
            labelb = 'agent ' + str(home)
            yb = np.sum(prm['grd']['loads'], axis=0)[home]
            plt.plot(time, yb, label=labelb)
        plt.title('Demand')
        plt.xlabel('Time [h]')
        plt.ylabel('[kWh]')
        plt.legend()
        plt.tight_layout()

        # EV availability
        cin += 1
        fin[cin], axs = plt.subplots(n_homes, 1)
        lin[cin] = 'batch_avail_car'
        for home in range(n_homes):
            labelb = 'agent ' + str(home)
            axb = axs[home] if n_homes > 1 else axs

            for time in range(prm['syst']['N']):
                if prm['car']['batch_avail_car'][home, time] == 0:
                    axb.axvspan(time - 0.5, time + 0.5, alpha=0.1, color='red')
            axb.set_ylabel(labelb)
        plt.xlabel('Time [h]')
        plt.title('EV unavailable')
        plt.tight_layout()

        # EV cons
        cin += 1
        fin[cin] = plt.figure()
        lin[cin] = 'batch_loads_car'
        self._plot_y(prm, prm['car']['batch_loads_car'], time)
        plt.xlabel('Time [h]')
        plt.ylabel('EV consumption [kWh]')
        if n_homes > 1:
            plt.legend()
        plt.tight_layout()

        return lin, fin, cin

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
            for time in range(prm['N']):
                s = solve(
                    P[home, time]
                    + (x ** 2 - x * (Voc[home, time] / Rt[home, time])) * kappa * Rt[home, time],
                    x)
                A = Rt[home, time] * kappa
                B = - Voc[home, time] * kappa
                C = P[home, time]
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
                            s[sign] * Voc[home, time],
                            s[sign] * (Voc[home, time] - s[sign] * Rt[home, time]))
                        )
                        etas2.append(np.divide(
                            s2[sign] * Voc[home, time],
                            s2[sign] * (Voc[home, time] - s2[sign] * Rt[home, time]))
                        )
                print(f'etas = {etas}, etas2={etas2}')
                eta[home, time] = etas[np.argmin(abs(etas - 1))]
                i_cell[home, time] = s[np.argmin(abs(etas - 1))]

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
        """Save results to file."""
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

            if self.save['saveresdata']:
                np.save(self.paths['record_folder'] / 'res', res)

        return res

    def _cons_constraints(self, p, E_heat):
        """Add constraints for consumption."""
        totcons = p.add_variable('totcons', (self.n_homes, self.N), vtype='continuous')

        consa = []
        for load_type in range(self.loads['n_types']):
            consa.append(p.add_variable('consa({0})'.format(load_type),
                                        (self.n_homes, self.N)))
        constl = {}
        tlpairs = comb(np.array([self.N, self.loads['n_types']]))
        for tl in tlpairs:
            constl[tl] = p.add_variable('constl{0}'.format(tl),
                                        (self.n_homes, self.N))

        for load_type in range(self.loads['n_types']):
            for home in range(self.n_homes):
                for time in range(self.N):
                    # time = tD
                    p.add_constraint(
                        pic.sum([constl[time, load_type][home, tC]
                                 * self.grd['flex'][time, load_type, home, tC]
                                 for tC in range(self.N)])
                        == self.grd['loads'][load_type, home, time])
                    # time = tC
                    p.add_constraint(
                        pic.sum([constl[tD, load_type][home, time]
                                 for tD in range(self.N)])
                        == consa[load_type][home, time])
        p.add_constraint(
            pic.sum([consa[load_type]
                     for load_type in range(self.loads['n_types'])])
            + E_heat
            == totcons)

        p.add_list_of_constraints(
            [consa[load_type] >= 0
             for load_type in range(self.loads['n_types'])])
        p.add_list_of_constraints(
            [constl[tl] >= 0 for tl in tlpairs])
        p.add_constraint(totcons >= 0)

        return p, totcons

    def _temperature_constraints(self, p):
        """Add temperature constraints to the problem."""
        T = p.add_variable('T', (self.n_homes, self.N), vtype='continuous')
        T_air = p.add_variable('T_air', (self.n_homes, self.N), vtype='continuous')
        # in kWh use P in W  Eheat * 1e3 * self.syst['prm']['H']/2 =  Pheat
        E_heat = p.add_variable('E_heat', (self.n_homes, self.N),
                                vtype='continuous')
        heat = self.heat
        for home in range(self.n_homes):
            if heat['own_heat'][home]:
                p.add_constraint(T[home, 0] == heat['T0'])
                p.add_list_of_constraints(
                    [T[home, time + 1] == heat['T_coeff'][home][0]
                        + heat['T_coeff'][home][1] * T[home, time]
                        + heat['T_coeff'][home][2] * heat['T_out'][time]
                        # heat['T_coeff'][home][3] * heat['phi_sol'][time]
                        + heat['T_coeff'][home][4] * E_heat[home, time]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time in range(self.N - 1)])

                p.add_list_of_constraints(
                    [T_air[home, time] == heat['T_air_coeff'][home][0]
                        + heat['T_air_coeff'][home][1] * T[home, time]
                        + heat['T_air_coeff'][home][2] * heat['T_out'][time]
                        # heat['T_air_coeff'][home][3] * heat['phi_sol'][time] +
                        + heat['T_air_coeff'][home][4] * E_heat[home, time]
                        * 1e3 * self.syst['n_int_per_hr']
                        for time in range(self.N)])

                p.add_list_of_constraints(
                    [T_air[home, time] <= heat['T_UB'][home][time]
                        for time in range(self.N)])
                p.add_list_of_constraints(
                    [T_air[home, time] >= heat['T_LB'][home][time]
                        for time in range(self.N)])
            else:
                p.add_list_of_constraints(
                    [E_heat[home, time] == 0 for time in
                     range(self.N)])
                p.add_list_of_constraints(
                    [T_air[home, time]
                     == (heat['T_LB'][home][time] + heat['T_UB'][home][time]) / 2
                     for time in range(self.N)])
                p.add_list_of_constraints(
                    [T[home, time] == (heat['T_LB'][home][time] + heat['T_UB'][home][time]) / 2
                     for time in range(self.N)])

        p.add_constraint(E_heat >= 0)

        return p, E_heat

    def _import_export_constraints(self, p, grid):
        pc = p.add_variable('pc', 1)  # total penalty costs
        pci = p.add_variable('pci', self.N, vtype='continuous')  # hourly import penalty
        pco = p.add_variable('pco', self.N, vtype='continuous')  # hourly export penalty
        if self.manage_agg_power:
            # hourly grid import
            grid_in = p.add_variable('grid_in', (self.N), vtype='continuous')
            # hourly grid export
            grid_out = p.add_variable('grid_out', (self.N), vtype='continuous')

            p.add_list_of_constraints(pci[t] >= 0 for t in range(self.N))
            p.add_list_of_constraints(
                pci[t]
                >= self.grd['penalty_coefficient_in']
                * (grid_in[t] - self.grd['max_grid_in']) for t in range(self.N)
            )
            p.add_list_of_constraints(pco[t] >= 0 for t in range(self.N))
            p.add_list_of_constraints(
                pco[t]
                >= self.grd['penalty_coefficient_out']
                * (grid_out[t] - self.grd['max_grid_out'])
                for t in range(self.N)
            )

            # import and export definition
            p.add_list_of_constraints(
                [grid[t] == grid_in[t] - grid_out[t] for t in range(self.N)])
            p.add_list_of_constraints(grid_in[t] >= 0 for t in range(self.N))
            p.add_list_of_constraints(grid_out[t] >= 0 for t in range(self.N))
            p.add_constraint(pc == pic.sum(pci) + pic.sum(pco))
        else:
            p.add_constraint(pc == 0)
            p.add_constraint(pci == 0)
            p.add_constraint(pco == 0)

        return p, pc

    def _set_objective(self, p, netp, grid, discharge_tot, charge, grid2, pc):
        gc = p.add_variable('gc', 1)   # grid costs
        sc = p.add_variable('sc', 1)   # storage costs
        dc = p.add_variable('dc', 1)   # distribution costs

        # grid costs
        p.add_constraint(
            gc
            == pc
            + (self.grd['C'][0: self.N] | (grid + self.grd['R'] / (self.grd['V'] ** 2) * grid2))
        )

        p.add_constraint(sc == self.car['C']
                         * (pic.sum(discharge_tot) + pic.sum(charge)
                            + np.sum(self.loads['discharge_tot0'])
                            + np.sum(self.loads['charge0'])
                            )
                         )
        p = self._distribution_costs(p, dc, netp)

        p.set_objective('min', gc + sc + dc)

        return p
