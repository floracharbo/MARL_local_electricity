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

from utilities.userdeftools import comb


class Solver():
    """The Solver manages convex optimisations."""

    def __init__(self, prm):
        """Initialise solver object for doing optimisations."""
        self.N = prm["syst"]["N"]
        self.n_homes = prm["ntw"]["n"]
        self.save = prm["save"]
        self.paths = prm["paths"]

    def solve(self, prm):
        """Solve optimisation problem given prm input data."""
        self._update_prm(prm)
        res = self._problem()
        if prm['bat']['efftype'] == 1:
            init_eta = prm['bat']['etach']
            prm['bat']['etach'] = self._efficiencies(
                res, prm['syst'], prm['ntw'], prm['bat']['cap'])
            deltamax, its = 0.5, 0
            prm['bat']['eff'] = 2
            while deltamax > 0.01 and its < 10:
                its += 1
                eta_old = copy.deepcopy(prm['bat']['etach'])
                print(f"prm['ntw']['dem'][0][0][0] = "
                      f"{prm['ntw']['dem'][0][0][0]}")
                res = self._problem(prm)
                print(f"res['constl(0, 0)'][0][0] "
                      f"= {res['constl(0, 0)'][0][0]}")
                if prm['ntw']['dem'][0][0][0] < res['constl(0, 0)'][0][0]:
                    print('fixed dem smaller than fixed onsumption a=0 t=0')
                if abs(np.sum(res['totcons']) - np.sum(res['E_heat'])
                       - np.sum(prm['ntw']['dem'])) > 1e-3:
                    print(f"tot load cons "
                          f"{np.sum(res['totcons']) - np.sum(res['E_heat'])} "
                          f"not equal to dem {np.sum(prm['dem'])}")
                prm['bat']['etach'] = self._efficiencies(
                    res, prm['syst'], prm['ntw'], prm['bat']['cap'])
                deltamax = np.amax(abs(prm['bat']['etach'] - eta_old))
            prm['bat']['etach'] = init_eta
            prm['bat']['eff'] = 1

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

        # if prm['savefigsres']:
        save_res_path = os.path.join(folder, 'saveres')
        for i in range(count + 1):
            figs[i].savefig(os.path.join(save_res_path, labels[i]))

        save_inputs_path = os.path.join(folder, 'saveinputs')
        if os.path.exists(save_inputs_path) == 0:
            os.mkdir(save_inputs_path)
        for i in range(cin + 1):
            fin[i].savefig(os.path.join(save_inputs_path, lin[i]))

    def _problem(self):
        N = self.N
        n_homes = self.n_homes
        # initialise problem
        p = pic.Problem()

        # variables
        grid = p.add_variable('grid', (N), vtype='continuous')
        grid2 = p.add_variable('grid2', (N), vtype='continuous')
        netp = p.add_variable('netp', (n_homes, N), vtype='continuous')
        totcons = p.add_variable('totcons', (n_homes, N),
                                 vtype='continuous')
        consa = []
        for load_type in range(self.loads['n_types']):
            consa.append(p.add_variable('consa({0})'.format(load_type),
                                        (n_homes, N)))
        constl = {}
        tlpairs = comb(np.array([N, self.loads['n_types']]))
        for tl in tlpairs:
            constl[tl] = p.add_variable('constl{0}'.format(tl),
                                        (n_homes, N))

        charge = p.add_variable('charge', (n_homes, N),
                                vtype='continuous')
        discharge_tot = p.add_variable('discharge_tot', (n_homes, N),
                                       vtype='continuous')
        discharge_other = p.add_variable('discharge_other', (n_homes, N),
                                         vtype='continuous')
        store = p.add_variable('store', (n_homes, N), vtype='continuous')
        T = p.add_variable('T', (n_homes, N), vtype='continuous')
        T_air = p.add_variable('T_air', (n_homes, N), vtype='continuous')
        # in kWh use P in W  Eheat * 1e3 * self.syst['prm']['H']/2 =  Pheat
        E_heat = p.add_variable('E_heat', (n_homes, N),
                                vtype='continuous')
        gc = p.add_variable('gc', 1)   # grid costs
        sc = p.add_variable('sc', 1)   # storage costs
        dc = p.add_variable('dc', 1)   # distribution costs

        # constraints
        # substation energy balance
        p.add_list_of_constraints(
            [grid[t]
             - np.sum(self.loads['netp0'][b0][t]
                      for b0 in range(self.ntw['nP']))
             - pic.sum([netp[b, t] for b in range(n_homes)])
             == 0 for t in range(N)])

        # prosumer energy balance
        p.add_constraint(netp - charge / self.bat['eta_ch']
                         + discharge_other
                         + self.ntw['gen'][:, 0: N]
                         - totcons == 0)

        # battery energy balance
        p.add_constraint(discharge_tot
                         == discharge_other / self.bat['eta_dis']
                         + self.bat['batch_loads_EV'][:, 0: N])

        p = self._storage_constraints(
            p, charge, discharge_tot, discharge_other, store
        )

        # consumption
        p = self._cons_constraints(p, constl, consa, E_heat, totcons)

        # temperature
        p = self._temperature_constraints(p, T, E_heat, T_air)

        # positivity constraints
        p.add_constraint(totcons >= 0)
        p.add_constraint(store >= 0)
        p.add_constraint(charge >= 0)
        p.add_constraint(discharge_tot >= 0)
        p.add_constraint(discharge_other >= 0)
        p.add_constraint(E_heat >= 0)
        p.add_list_of_constraints(
            [consa[load_type] >= 0
             for load_type in range(self.loads['n_types'])])
        p.add_list_of_constraints(
            [constl[tl] >= 0 for tl in tlpairs])

        # costs constraints
        p.add_list_of_constraints(
            [grid2[t] >= grid[t] * grid[t] for t in range(N)])

        p.add_constraint(
            gc == (self.grd['C']
                   | (grid + self.grd['R'] / (self.grd['V'] ** 2) * grid2))
        )
        p.add_constraint(sc == self.bat['C']
                         * (pic.sum(discharge_tot) + pic.sum(charge)
                            + np.sum(self.loads['discharge_tot0'])
                            + np.sum(self.loads['charge0'])
                            )
                         )

        p = self._distribution_costs(p, dc, netp)

        # solve
        p.set_objective('min', gc + sc + dc)
        p.solve(verbose=0, solver=self.syst['solver'])

        # %% save results
        res = self._save_results(p.variables)

        return res

    def _storage_constraints(
            self, p, charge, discharge_tot, discharge_other, store
    ):
        store_end = self.bat['SoC0'] * self.ntw['Bcap'][:, self.N - 1]
        bat = self.bat

        if bat['eff'] == 1:
            p.add_list_of_constraints(
                [charge[:, t] - discharge_tot[:, t]
                 == store[:, t + 1] - store[:, t]
                 for t in range(self.N - 1)])
            p.add_constraint(store[:, self.N - 1]
                             + charge[:, self.N - 1]
                             - discharge_tot[:, self.N - 1]
                             >= store_end)

        elif bat['eff'] == 2:
            for b in range(self.n_homes):
                p.add_constraint(
                    bat['eta_ch'][b, self.N - 1] * charge[b, self.N - 1]
                    - bat['eta_ch'][b, self.N - 1]
                    * discharge_tot[b, self.N - 1]
                    == store_end[b] - store[b, self.N - 1]
                )
                for t in range(self.N - 1):
                    p.add_constraint(
                        bat['eta_ch'][b, t] * charge[b, t]
                        - bat['eta_dis'][b, t] * discharge_tot[b, t]
                        == store[b, t + 1] - store[b, t]
                    )

        # initialise storage
        p.add_list_of_constraints(
            [store[b, 0] == bat['SoC0'] * self.ntw['Bcap'][b, 0]
             for b in range(self.n_homes)])

        p.add_list_of_constraints(
            [store[:, t + 1] >= bat['SoCmin']
             * self.ntw['Bcap'][:, t] * bat['batch_avail_EV'][:, t]
             for t in range(self.N - 1)])

        # if EV not avail at a given time step,
        # it is ok to start the following time step with less than minimum
        p.add_list_of_constraints(
            [store[:, t + 1] >= bat['baseld'] * bat['batch_avail_EV'][:, t]
             for t in range(self.N - 1)])

        # can charge only when EV is available
        p.add_constraint(
            charge <= bat['batch_avail_EV'][:, 0: self.N] * self.syst['M'])

        # can discharge only when EV is available (Except EV cons is ok)
        p.add_constraint(
            discharge_other
            <= bat['batch_avail_EV'][:, 0: self.N] * self.syst['M']
        )
        p.add_constraint(store <= self.ntw['Bcap'])
        p.add_constraint(bat['c_max'] >= charge)
        p.add_constraint(bat['d_max'] >= discharge_tot)

        return p

    def _distribution_costs(self, p, dc, netp):
        ntw = self.ntw
        if ntw['charge_type'] == 0:
            netp_abs = p.add_variable('netp_abs', (self.n_homes, self.N),
                                      vtype='continuous')
            netp0_abs = np.sum(
                [[abs(self.loads['netp0'][b0][t])
                  if self.loads['netp0'][b0][t] < 0
                  else 0
                  for b0 in range(ntw['nP'])] for t in range(self.N)])
            p.add_constraint(netp_abs >= - netp)
            p.add_constraint(netp_abs >= 0)
            # distribution costs
            p.add_constraint(dc == ntw['C'] * (pic.sum(netp_abs) + netp0_abs))
        else:
            netp2 = p.add_variable('netp2', (self.n_homes, self.N),
                                   vtype='continuous')
            sum_netp2 = np.sum([[self.loads['netp0'][b0][t] ** 2
                                 for b0 in range(ntw['nP'])]
                                for t in range(self.N)])
            for b in range(self.n_homes):
                p.add_list_of_constraints(
                    [netp2[b, t] >= netp[b, t] * netp[b, t]
                     for t in range(self.N)])
            p.add_constraint(dc == ntw['C'] * (pic.sum(netp2) + sum_netp2))

        return p

    def _update_prm(self, prm):
        if type(prm) is list or type(prm) is tuple:
            self.syst, self.loads, self.ntw, self.bat, self.grd, self.heat \
                = prm
        else:
            self.syst, self.loads, self.ntw, self.bat, self.grd, self.heat \
                = [prm[e]
                   for e in ['syst', 'loads', 'ntw', 'bat', 'grd', 'heat']]

    def _plot_y(self, prm, y, time):
        for b in range(prm['ntw']['n']):
            yb = np.reshape(y[b, :], (prm['syst']['N']))
            plt.plot(time, yb, label=f'agent {b}')

    def _plot_inputs(self, prm, time):
        # inputs
        fin = {}
        lin = {}
        cin = 0

        # PV
        fin[cin] = plt.figure()
        lin[cin] = 'PVgen'
        self._plot_y(prm, prm['ntw']['gen'], time)
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
        for b in range(prm['ntw']['n']):
            labelb = 'agent ' + str(b)
            yb = np.sum(prm['ntw']['dem'], axis=0)[b]
            plt.plot(time, yb, label=labelb)
        plt.title('Demand')
        plt.xlabel('Time [h]')
        plt.ylabel('[kWh]')
        plt.legend()
        plt.tight_layout()

        # EV availability
        cin += 1
        fin[cin], axs = plt.subplots(prm['ntw']['n'], 1)
        lin[cin] = 'batch_avail_EV'
        for b in range(prm['ntw']['n']):
            labelb = 'agent ' + str(b)
            axb = axs[b] if prm['ntw']['n'] > 1 else axs

            for t in range(prm['syst']['N']):
                if prm['bat']['batch_avail_EV'][b, t] == 0:
                    axb.axvspan(t - 0.5, t + 0.5, alpha=0.1, color='red')
            axb.set_ylabel(labelb)
        plt.xlabel('Time [h]')
        plt.title('EV unavailable')
        plt.tight_layout()

        # EV cons
        cin += 1
        fin[cin] = plt.figure()
        lin[cin] = 'batch_loads_EV'
        self._plot_y(prm, prm['bat']['batch_loads_EV'], time)
        plt.xlabel('Time [h]')
        plt.ylabel('EV consumption [kWh]')
        if prm['ntw']['n'] > 1:
            plt.legend()
        plt.tight_layout()

        return lin, fin, cin

    def _efficiencies(self, res, prm, ntw, bat_cap):
        store = res['store']
        P_ch = res['charge']
        P_dis = res['discharge_tot']

        P = (P_ch - P_dis) * 1e3
        SoC = np.zeros((self.n_homes, prm['N']))
        for b in range(self.n_homes):
            if bat_cap[b] == 0:
                SoC[b] = np.zeros(prm['N'])
            else:
                # in battery(times, bus)/cap(bus)
                SoC[b] = np.divide(store[b], bat_cap[b])
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
        for a in range(self.n_homes):
            for t in range(prm['N']):
                s = solve(
                    P[a, t]
                    + (x ** 2 - x * (Voc[a, t] / Rt[a, t])) * kappa * Rt[a, t],
                    x)
                A = Rt[a, t] * kappa
                B = - Voc[a, t] * kappa
                C = P[a, t]
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
                            s[sign] * Voc[a, t],
                            s[sign] * (Voc[a, t] - s[sign] * Rt[a, t]))
                        )
                        etas2.append(np.divide(
                            s2[sign] * Voc[a, t],
                            s2[sign] * (Voc[a, t] - s2[sign] * Rt[a, t]))
                        )
                print(f'etas = {etas}, etas2={etas2}')
                eta[a, t] = etas[np.argmin(abs(etas - 1))]
                i_cell[a, t] = s[np.argmin(abs(etas - 1))]

        return eta, s, s2

    def _add_val_to_res(self, res, var, val, size, arr):
        if size[0] < 2 and size[1] < 2:
            res[var] = val
        else:
            for i in range(size[0]):
                for j in range(size[1]):
                    arr[i, j] = val[i, j]
            res[var] = arr

        return res

    def _save_results(self, pvars):
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

    def _cons_constraints(self, p, constl, consa, E_heat, totcons):
        loads = self.loads
        for load_type in range(loads['n_types']):
            for b in range(self.n_homes):
                for t in range(self.N):
                    # t = tD
                    p.add_constraint(
                        pic.sum([constl[t, load_type][b, tC]
                                 * self.ntw['flex'][t, load_type, b, tC]
                                 for tC in range(self.N)])
                        == self.ntw['dem'][load_type, b, t])
                    # t = tC
                    p.add_constraint(
                        pic.sum([constl[tD, load_type][b, t]
                                 for tD in range(self.N)])
                        == consa[load_type][b, t])
        p.add_constraint(
            pic.sum([consa[load_type]
                     for load_type in range(loads['n_types'])])
            + E_heat
            == totcons)

        return p

    def _temperature_constraints(self, p, T, E_heat, T_air):
        heat = self.heat
        for a in range(self.n_homes):
            if heat['own_heat'][a]:
                p.add_constraint(T[a, 0] == heat['T0'])
                p.add_list_of_constraints(
                    [T[a, t + 1] == heat['T_coeff'][a][0]
                        + heat['T_coeff'][a][1] * T[a, t]
                        + heat['T_coeff'][a][2] * heat['T_out'][t]
                        # heat['T_coeff'][a][3] * heat['phi_sol'][t]
                        + heat['T_coeff'][a][4] * E_heat[a, t]
                        * 1e3 * self.syst['H'] / 24
                        for t in range(self.N - 1)])

                p.add_list_of_constraints(
                    [T_air[a, t] == heat['T_air_coeff'][a][0]
                        + heat['T_air_coeff'][a][1] * T[a, t]
                        + heat['T_air_coeff'][a][2] * heat['T_out'][t]
                        # heat['T_air_coeff'][a][3] * heat['phi_sol'][t] +
                        + heat['T_air_coeff'][a][4] * E_heat[a, t]
                        * 1e3 * self.syst['H'] / 24
                        for t in range(self.N)])

                p.add_list_of_constraints(
                    [T_air[a, t] <= heat['T_UB'][a][t]
                        for t in range(self.N)])
                p.add_list_of_constraints(
                    [T_air[a, t] >= heat['T_LB'][a][t]
                        for t in range(self.N)])
            else:
                p.add_list_of_constraints(
                    [E_heat[a, t] == 0 for t in
                     range(self.N)])
                p.add_list_of_constraints(
                    [T_air[a, t]
                     == (heat['T_LB'][a][t] + heat['T_UB'][a][t]) / 2
                     for t in range(self.N)])
                p.add_list_of_constraints(
                    [T[a, t] == (heat['T_LB'][a][t] + heat['T_UB'][a][t]) / 2
                     for t in range(self.N)])

        return p
