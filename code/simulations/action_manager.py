#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues 14 Dec 15:40:20 2021.

@author: floracharbonnier
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from utilities.userdeftools import initialise_dict


class Action_manager:
    """
    Manage computations of the RL flexibilitiy action variable.

    k  - line coefficients per interval
    xs - intervals
    dp - positive import, negative export (after fixed consumption)
    ds - change in storage
    l  - losses
    fl - flexible load consumed
    """

    def __init__(self, prm, env):
        """Initialise action_manager object and add relevant properties."""
        self.name = 'action manager'
        self.entries = ['dp', 'ds', 'l_ch', 'l_dis', 'c']
        self.plotting = prm['RL']['plotting_action']
        self.server = prm['RL']['server']
        self.colors = [(0, 0, 0)] + prm['save']['colors']
        self.n_agents = env.n_agents
        self.labels = [r'$\Delta$p', r'$\Delta$s', 'Losses', 'Consumption']
        self.z_orders = [1, 3, 2, 0, 4]
        self.H = prm['syst']['H']
        for e in ['aggregate_actions', 'dim_actions', 'low_action',
                  'high_action', 'type_env']:
            self.__dict__[e] = prm['RL'][e]
        self.bat_dep = prm['bat']['dep']
        self.ntw_C = prm['ntw']['C']

    def optimisation_to_rl_env_action(self, h, date, netp, loads, home, res):
        """
        From home energy values, get equivalent RL flexibility actions.

        Given home consumption and import variables,
        compute equiavalent RL flexibility action values.
        """
        # loads: l_flex, l_fixed

        as_ = range(self.n_agents)

        self.bat.min_max_charge_t(h, date)
        self.initial_processing(loads, home)

        error = [False for _ in as_]
        bool_flex, actions = [], []
        for a in as_:
            if self.aggregate_actions:
                actions, bool_flex = self._get_aggregate_mu(
                    actions, bool_flex, netp, a
                )
            else:
                actions, bool_flex = self._get_disaggregated_mus(
                    actions, bool_flex, res, loads, a, h
                )

            error = self._check_action_errors(
                actions, error, res, loads, a, h, bool_flex
            )

        return bool_flex, actions, error

    def initial_processing(self, loads, home):
        """Compute current flexibility variables."""
        # inputs
        # loads: l_flex, l_fixed
        # home: gen0
        eta_dis, eta_ch = self.bat.eta_dis, self.bat.eta_ch
        as_ = range(self.n_agents)

        s_avail_dis, s_add_0, s_remove_0, C_avail = \
            self.bat.initial_processing()

        self._check_input_types(
            loads, home, s_add_0, s_avail_dis, C_avail, s_remove_0
        )

        # translate inputs into relevant quantities
        self.tot_l_fixed = loads['l_fixed'] + self.heat.E_heat_min
        tot_l_flex = loads['l_flex'] + self.heat.potential_E_flex()

        # gen to min charge
        g_to_add0 = np.minimum(home['gen'], s_add_0 / eta_dis)

        # gen left after contributing to reaching min charge
        g_net_add0 = home['gen'] - g_to_add0

        # required addition to storage for min charge left
        # after contribution from gen
        s_add0_net = s_add_0 - g_to_add0 * eta_ch

        # gen to fixed consumption
        g_to_fixed = np.minimum(g_net_add0, self.tot_l_fixed)

        # gen left after contributing to fixed consumption
        gnet_fixed = g_net_add0 - g_to_fixed

        # fixed load left after contribution from gen
        lnet_fixed = self.tot_l_fixed - g_to_fixed

        # prof to flex consumption
        g_to_flex = np.minimum(gnet_fixed, tot_l_flex)

        # gen left after contributing to flex consumption
        gnet_flex = gnet_fixed - g_to_flex

        # flex load left after contribution from gen
        lnet_flex = tot_l_flex - g_to_flex

        # gen that can be put in store
        g_to_store = np.minimum(gnet_flex, C_avail / eta_ch)

        # How much generation left after storing as much as possible
        gnet_store = gnet_flex - g_to_store
        self.k = initialise_dict(as_, 'empty_dict')

        # get relevant points in graph
        d = initialise_dict(self.entries, 'empty_dict')
        action_points, xs = [{} for _ in range(2)]

        d['ds']['A'] = - s_avail_dis + s_add_0
        dsB = np.maximum(- s_avail_dis + s_add_0, - lnet_fixed / eta_dis)
        d['ds']['B'] = np.where(
            s_remove_0 > 1e-2, np.minimum(dsB, - s_remove_0), dsB
        )

        d['ds']['C'] = s_add_0 - s_remove_0
        d['ds']['D'] = s_add_0 - s_remove_0
        dsE = np.maximum(np.minimum(gnet_flex * eta_ch, C_avail), s_add_0)
        d['ds']['E'] = [min(dsE[a], - s_remove_0[a])
                        if s_remove_0[a] > 1e-2 else dsE[a] for a in as_]
        d['ds']['F'] = np.where(s_remove_0 > 1e-2, - s_remove_0, C_avail)
        d['dp']['A'] = - s_avail_dis * eta_dis - g_net_add0 \
            + s_add0_net / eta_ch + self.tot_l_fixed

        dspB = np.where(
            d['ds']['B'] > 0, s_add0_net / eta_ch, d['ds']['B'] * eta_dis
        )

        d['dp']['B'] = - g_net_add0 + dspB + self.tot_l_fixed
        d['dp']['C'] = - gnet_fixed + lnet_fixed \
            + s_add0_net / eta_ch - eta_dis * s_remove_0
        d['dp']['D'] = - gnet_flex + lnet_fixed + lnet_flex \
            + s_add0_net / eta_ch - eta_dis * s_remove_0
        dpE_dspos = (dsE - eta_ch * (g_to_store + g_to_add0)) / eta_ch \
            + lnet_fixed + lnet_flex - gnet_store
        dpE_dsneg = lnet_fixed + lnet_flex - gnet_flex - eta_dis * s_remove_0
        d['dp']['E'] = np.where(dsE > 0, dpE_dspos, dpE_dsneg)

        dpF_dspos = (C_avail - eta_ch * (g_to_add0 + g_to_store)) / eta_ch \
            + lnet_fixed + lnet_flex - gnet_store
        dpF_dsneg = - s_remove_0 * eta_dis + lnet_fixed + lnet_flex - gnet_flex
        d['dp']['F'] = np.where(d['ds']['F'] >= 0, dpF_dspos, dpF_dsneg)

        for i in ['A', 'B', 'C']:
            d['c'][i] = self.tot_l_fixed
        for i in ['D', 'E', 'F']:
            d['c'][i] = self.tot_l_fixed + tot_l_flex
        a_dp = d['dp']['F'] - d['dp']['A']
        b_dp = d['dp']['A']

        action_points['A'], action_points['F'] = np.zeros(self.n_agents), np.ones(self.n_agents)
        for i in ['B', 'C', 'D', 'E']:
            action_points[i] = np.zeros(self.n_agents)
            mask = a_dp > 1e-3
            action_points[i][mask] = (d['dp'][i][mask] - b_dp[mask]) / a_dp[mask]
            for a in as_:
                assert action_points[i][a] > - 1e-4, \
                    f"action_points[{i}][{a}] {action_points[i][a]} < 0"
                assert action_points[i][a] < 1 + 1e-4, \
                    f"action_points[{i}][{a}] {action_points[i][a]} > 1"
                if - 1e-4 < action_points[i][a] < 0:
                    action_points[i][a] = 0
                if 1 < action_points[i][a] < 1 + 1e-4:
                    action_points[i][a] = 1
        self.d = d
        self.k, self.action_intervals = [[] for _ in range(2)]

        for a in as_:
            self._compute_k(a, a_dp, b_dp, action_points, d)
            assert self.heat.E_heat_min[a] + loads['l_fixed'][a] \
                   <= self.k[a]['c'][0][1] + 1e-3,\
                   "min c smaller than min required"

        # these variables are useful in optimisation_to_rl_env_action and actions_to_env_vars
        # in the case where action variables are not aggregated
        self.max_discharge = np.array(
            [(self.k[a]['ds'][0][0] * 0 + self.k[a]['ds'][0][1]) for a in as_]
        )
        self.max_charge = np.array(
            [self.k[a]['ds'][-1][0] * 1 + self.k[a]['ds'][-1][1] for a in as_]
        )
        self.min_charge = np.where(
            self.max_discharge > 0, self.max_discharge, 0
        )
        self.min_discharge = np.where(
            self.max_charge < 0, self.max_charge, 0
        )
        self.max_discharge = np.where(
            self.max_discharge > 0, 0, self.max_discharge
        )

    def actions_to_env_vars(self, loads, home, action, date, h):
        """Update variables after non flexible consumption is met."""
        # other variables
        self.error = False
        as_ = range(self.n_agents)

        # problem variables
        bool_penalty = self.bat.min_max_charge_t(h, date)
        for e in ['netp', 'tot_cons']:
            home[e] = np.zeros(self.n_agents)

        self.initial_processing(loads, home)

        # check initial errors
        self.res = {}
        [home['bool_flex'], loads['flex_cons'],
         loads['tot_cons_loads'], self.heat.tot_E] \
            = [[] for _ in range(4)]
        self.l_flex = loads['l_flex']
        flex_heat = []
        for a in as_:
            # boolean for whether or not we have flexibility
            home['bool_flex'].append(abs(self.k[a]['dp'][0][0]) > 1e-2)
            if self.aggregate_actions:
                flex_heat = None
                # update variables for given action
                # obtain the interval in which action_points lies
                ik = [i for i in range(len(self.action_intervals[a]) - 1)
                      if action[a][0] >= self.action_intervals[a][i]][-1]
                res = {}  # resulting values (for dp, ds, fl, l)
                for e in self.entries:
                    ik_ = 0 if e == 'dp' else ik
                    # use coefficients to obtain value
                    res[e] = self.k[a][e][ik_][0] * action[a][0] \
                        + self.k[a][e][ik_][1]

                home['tot_cons'][a] = res['c']
                home['netp'][a] = res['dp']
                if res['c'] > loads['l_flex'][a] + self.tot_l_fixed[a]:
                    loads['flex_cons'].append(loads['l_flex'][a])
                else:
                    loads['flex_cons'].append(res['c'] - self.tot_l_fixed[a])
                    assert loads['flex_cons'][-1] > - 1e-2, \
                        f"loads['flex_cons'][-1] {loads['flex_cons'][-1]} < 0"
                    if - 1e-2 < loads['flex_cons'][-1] < 0:
                        loads['flex_cons'][-1] = 0
            else:
                flexible_cons_action, flexible_heat_action, battery_action = action[a]
                # flex cons between 0 and 1
                # flex heat between 0 and 1
                # charge between -1 and 1 where
                # -1 max discharge
                # 0 nothing (or just minimum)
                # 1 max charge
                res = {}
                flexible_cons_action_ = 0 if flexible_cons_action is None else flexible_cons_action
                loads['flex_cons'].append(flexible_cons_action_ * loads['l_flex'][a])
                flex_heat.append(flexible_heat_action
                                 * self.heat.potential_E_flex()[a])
                home['tot_cons'][a] = self.tot_l_fixed[a] \
                    + loads['flex_cons'][a] \
                    + flex_heat[a]
                res['c'] = home['tot_cons'][a]
                res = self._battery_action_to_ds(a, battery_action, res)

                discharge = - res['ds'] * self.bat.eta_dis \
                    if res['ds'] < 0 else 0
                charge = res['ds'] if res['ds'] > 0 else 0
                home['netp'][a] = loads['flex_cons'][a] \
                    + loads['l_fixed'][a] \
                    + self.heat.E_heat_min[a] \
                    + flex_heat[a] \
                    + charge - discharge + res['l_ch'] \
                    - home['gen'][a]

                res['dp'] = home['netp'][a]

            loads['tot_cons_loads'].append(
                loads['flex_cons'][a] + loads['l_fixed'][a])
            self.res[a] = copy.copy(res)

        self.bat.actions_to_env_vars(self.res)
        self.heat.actions_to_env_vars(
            self.res, loads['l_flex'], self.tot_l_fixed, E_flex=flex_heat
        )

        # check for errors
        for a in as_:
            # energy balance
            e_balance = abs((self.res[a]['dp'] + home['gen'][a]
                             + self.bat.discharge[a] - self.bat.charge[a]
                             - self.bat.loss_ch[a] - home['tot_cons'][a]))
            assert e_balance <= 1e-3, f"energy balance {e_balance}"
            assert abs(loads['tot_cons_loads'][a] + self.heat.tot_E[a]
                   - home['tot_cons'][a]) <= 1e-3, \
                f"tot_cons_loads {loads['tot_cons_loads'][a]}, "\
                f"self.heat.tot_E[a] {self.heat.tot_E[a]}, " \
                f"home['tot_cons'][a] {home['tot_cons'][a]}"

        bool_penalty = self.bat.check_errors_apply_step(
            as_, bool_penalty, action, self.res)
        if sum(bool_penalty) > 0:
            self.error = True
        if not self.error and self.plotting:
            self._plot_graph_actions()
        # outputs
        # loads: flex_cons, tot_cons_loads
        # home: netp, bool_flex, tot_cons
        # bool_penalty

        return loads, home, bool_penalty

    def _battery_action_to_ds(self, a, battery_action, res):
        if battery_action is None:
            assert abs(self.min_charge[a] - self.max_charge[a]) <= 1e-4, \
                "battery_action is None but " \
                "self.min_charge[a] != self.max_charge[a]"
            res['ds'] = self.min_charge[a]
        elif battery_action < 0:
            if self.min_charge[a] > 0:
                res['ds'] = self.min_charge[a]
            elif self.min_discharge[a] > 0:
                res['ds'] = self.min_discharge[a]
            elif self.min_discharge[a] <= 0:
                res['ds'] = self.min_discharge[a] + abs(battery_action) \
                    * (self.max_discharge[a] - self.min_discharge[a])
        elif battery_action >= 0:
            if self.min_discharge[a] < 0:
                res['ds'] = self.min_discharge[a]
            elif self.max_charge[a] < 0:
                res['ds'] = self.max_charge[a]
            elif self.max_charge[a] >= 0:
                res['ds'] = self.min_charge[a] + battery_action \
                    * (self.max_charge[a] - self.min_charge[a])
        res['l_ch'] = 0 if res['ds'] < 0 \
            else (1 - self.bat.eta_ch) / self.bat.eta_ch * res['ds']
        res['l_dis'] = - res['ds'] * (1 - self.bat.eta_dis) \
            if res['ds'] < 0 else 0

        return res

    def _plot_graph_actions(self, save_fig=False, name_fig='action_points', legend=False):
        line_width = 2.5
        font_size = 15
        sns.set_palette("bright")
        ymin = 0
        ymax = 0
        fig = plt.figure(figsize=(2, 3))
        ax1 = fig.add_subplot(111)
        gs = gridspec.GridSpec(4, 1)
        ax1.set_position(gs[0:3].get_position(fig))
        ax1.set_subplotspec(gs[0:3])  # only necessary if using tight_layout()

        entries_plot = ['Losses' if e == 'l_ch' else e
                        for e in self.entries if e != 'l_dis']
        ys = {}
        for ie in range(len(entries_plot)):
            e = entries_plot[ie]
            e0 = 'l_ch' if e == 'Losses' else e
            wd = line_width * 1.5 if e == 'dp' else line_width
            col, zo, label = \
                [self.colors[ie], self.z_orders[ie], self.labels[ie]]
            n = len(self.k[0][e0])
            xs = [0, 1] if e == 'dp' else self.action_intervals[0]
            if e == 'Losses':
                ys[e] = [
                    sum(self.k[0][e_][i][0] * xs[i] + self.k[0][e_][i][1]
                        for e_ in ['l_ch', 'l_dis']) for i in range(n)]
                ys[e].append(
                    sum(self.k[0][e_][-1][0] * xs[-1] + self.k[0][e_][-1][1]
                        for e_ in ['l_ch', 'l_dis']))
            else:
                ys[e] = [self.k[0][e][i][0] * xs[i] + self.k[0][e][i][1]
                         for i in range(n)]
                ys[e].append(
                    self.k[0][e][-1][0] * xs[-1] + self.k[0][e][-1][1])
            ax1.plot(xs, ys[e], label=label, linewidth=wd,
                     color=col, zorder=zo)
            if min(ys[e]) < ymin:
                ymin = min(ys[e])
            if max(ys[e]) > ymax:
                ymax = max(ys[e])
        print(f'name_fig {name_fig}, ymin {ymin}, ymax {ymax}')
        y_bottom = - 56
        y_top = 60
        x_left = - 0.02
        x_right = 1.02
        for e in self.action_intervals.keys():
            ax1.vlines(
                x=self.action_intervals[e][0],
                ymin=y_bottom,
                ymax=y_top,
                color='gray',
                linestyle='--',
                linewidth=0.5
            )
        ax1.hlines(
            y=0,
            xmin=x_left,
            xmax=x_right,
            linestyle='--',
            linewidth=0.5
        )
        ax1.set_ylim([y_bottom, y_top])
        ax1.set_xlim([x_left, x_right])
        ax1.set_xlim([x_left, x_right])
        ax1.set_yticks([0])
        ax1.set_yticklabels([0], fontsize=font_size)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        if legend:
            ax1.legend(loc='right', bbox_to_anchor=(2, 0), fancybox=True)
        ax2 = fig.add_subplot(gs[3])
        discharge = [abs(ds) if ds < 0 else 0 for ds in ys['ds']]
        all_dps = [x * self.k[0]['dp'][0][0] + self.k[0]['dp'][0][1]
                   for x in self.action_intervals[0]]
        export = [abs(dp) if dp < 0 else 0 for dp in all_dps]
        costs = [loss * 0.1 + self.bat_dep * d + e * self.ntw_C
                 for loss, d, e in zip(ys['Losses'], discharge, export)]
        ax2.plot(self.action_intervals[0], costs,
                 color=self.colors[len(entries_plot)],
                 linewidth=line_width,
                 label='Battery, loss & export costs')
        ax2.set_yticks([0])
        ax2.set_yticklabels([0], fontsize=font_size)
        ax2.set_xticks([0, 1])
        if legend:
            ax2.legend(loc='right', bbox_to_anchor=(2, -2), fancybox=True)
        ax2.set_ylim([-0.05 * max(costs), max(costs)])
        ax2.set_xticklabels([0, 1], fontsize=font_size)
        # ax2.set_ylabel('Cost', fontsize=font_size)
        # ax2.set_xlabel('Action variable '+ r'$\mu$ [-]',
        # fontsize=font_size)

        fig.tight_layout()

        plt.show()
        if save_fig:
            fig.save_fig(name_fig + '.svg',
                         bbox_inches='tight', format='svg', dpi=1200)

    def _compute_k(self, a, a_dp, b_dp, action_points, d):
        letters = ['A', 'B', 'C', 'D', 'E', 'F']

        self.k.append(initialise_dict(self.entries))
        # reference line - dp
        self.k[a]['dp'] = [[a_dp[a], b_dp[a]]]
        for i in range(2):
            if abs(self.k[a]['dp'][0][i]) < 0:
                if abs(self.k[a]['dp'][0][i]) > - 1e-3:
                    self.k[a]['dp'][0][i] = 0
                else:
                    print(f"self.k[{a}]['dp'][0][{i}] = "
                          f"{self.k[a]['dp'][0][i]}")

        self.action_intervals.append([0])

        for e in ['ds', 'c', 'l_ch', 'l_dis']:
            self.k[a][e] = []
        for z in range(5):
            l1, l2 = letters[z: z + 2]
            if action_points[l2][a] > action_points[l1][a]:
                self.action_intervals[a].append(action_points[l2][a])
                for e in ['ds', 'c', 'losses']:
                    if e == 'losses':
                        self.k = self.bat.k_losses(
                            a,
                            self.k,
                            action_points[l1][a],
                            action_points[l2][a]
                        )
                    else:
                        ad = (d[e][l2][a] - d[e][l1][a]) / \
                             (action_points[l2][a] - action_points[l1][a])
                        bd = d[e][l2][a] - action_points[l2][a] * ad
                        self.k[a][e].append([ad, bd])

    def _check_input_types(
            self, loads, home, s_add_0, s_avail_dis, C_avail, s_remove_0
    ):
        assert isinstance(loads['l_fixed'], np.ndarray), \
            f"type(loads['l_fixed']) {type(loads['l_fixed'])}"
        assert isinstance(loads['l_flex'], np.ndarray), \
            f"type(loads['l_flex']) {type(loads['l_flex'])}"
        assert isinstance(self.heat.E_heat_min, np.ndarray), \
            f"type(self.heat.E_heat_min) {type(self.heat.E_heat_min)}"
        assert isinstance(home['gen'], np.ndarray), \
            f"type(home['gen']) = {type(home['gen'])}"
        assert isinstance(s_add_0, np.ndarray), \
            f"type(s_add_0) = {type(s_add_0)}"
        assert isinstance(s_avail_dis, np.ndarray), \
            f"type(s_avail_dis) = {type(s_avail_dis)}"
        assert isinstance(C_avail, np.ndarray), \
            f"type(C_avail) = {type(C_avail)}"
        assert isinstance(s_remove_0, np.ndarray), \
            f"type(s_remove_0) = {type(s_remove_0)}"

    def _get_aggregate_mu(self, actions, bool_flex, netp, a):
        if abs(self.k[a]['dp'][0][0]) < 1e-2:
            self.k[a]['dp'][0][0] = 0
        if self.k[a]['dp'][0][0] == 0:  # there is not flexibility
            # boolean for whether or not we have flexibility
            bool_flex.append(0)
            actions.append([None])
        else:
            bool_flex.append(1)
            # action none if no flexibility
            assert netp[a] - self.k[a]['dp'][0][1] > - 1e-2, \
                "netp smaller than k['dp'][0]"
            delta = 0 if abs(netp[a] - self.k[a]['dp'][0][1]) < 1e-2 \
                else netp[a] - self.k[a]['dp'][0][1]
            actions.append(
                [
                    None if self.k[a]['dp'][0][0] == 0
                    else delta / self.k[a]['dp'][0][0]
                ]
            )
        if actions[a] is not None:
            assert - 1e-2 < actions[a][0] < 1 + 1e-2, \
                "action should be between 0 and 1"

        return actions, bool_flex

    def _get_disaggregated_mus(self, actions, bool_flex, res, loads, a, h):
        cons = res['totcons'][a, h] - res['E_heat'][a, h]
        if cons < 1e-3:
            cons = 0
        flex_cons = cons - loads['l_fixed'][a]

        if flex_cons < 1e-3:
            flex_cons = 0
        elif loads['l_flex'][a] < flex_cons < loads['l_flex'][a] + 1e-3:
            flex_cons = loads['l_flex'][a]

        if loads['l_flex'][a] > 0:
            flexible_cons_action = flex_cons / loads['l_flex'][a]
        else:
            flexible_cons_action = None

        E_heat \
            = 0 if res['E_heat'][a][h] < 1e-3 else res['E_heat'][a][h]
        if self.heat.potential_E_flex()[a] > 0:
            flexible_heat_action = \
                (E_heat - self.heat.E_heat_min[a]) / \
                (self.heat.E_heat_max[a] - self.heat.E_heat_min[a])
        else:
            flexible_heat_action = None
        max_charge_a, min_charge_a = [
            self.max_charge[a], self.min_charge[a]
        ]
        max_discharge_a, min_discharge_a = [
            self.max_discharge[a], self.min_discharge[a]
        ]
        assert min_charge_a - 1e-3 <= res['charge'][a, h] \
               <= max_charge_a + 1e-3, \
               f"res charge {res['charge'][a, h]} " \
               f"min_charge_a {min_charge_a} max_charge_a {max_charge_a}"
        assert max_discharge_a - 1e-3 <= - res['discharge_other'][a, h] \
               <= min_discharge_a + 1e-3, \
               f"res discharge_other {res['discharge_other'][a, h]} " \
               f"min_discharge_a {min_discharge_a} " \
               f"max_discharge_a {max_discharge_a}"

        if (
            (
                abs(max_charge_a - min_charge_a) < 1e-3  # or
                and abs(min_discharge_a - max_discharge_a) < 1e-3
            )
            or (
                abs(min_discharge_a - max_discharge_a) < 1e-3
                and res['discharge_other'][a, h] > 1e-3
            )
            or (
                abs(max_charge_a - min_charge_a) < 1e-3
                and res['charge'][a, h] > 1e-3
            )
        ):
            # abs(max_charge_a - max_discharge_a) < 1e-3 or
            # no flexibility in charging
            battery_action = 0 if self.type_env == 'discrete' else None
            assert abs(self.min_charge[a] - self.max_charge[a]) <= 1e-4, \
                "battery_action is None but " \
                "self.min_charge[a] != self.max_charge[a]"
        elif abs(res['discharge_other'][a, h] < 1e-3
                 and abs(res['charge'][a, h]) < 1e-3):
            battery_action = 0
        elif res['discharge_other'][a, h] > 1e-3:
            battery_action = \
                (min_discharge_a - res['charge'][a, h]) \
                / (min_discharge_a - max_discharge_a)
        else:
            battery_action = (res['charge'][a, h] - min_charge_a) \
                / (max_charge_a - min_charge_a)
        actions.append(
            [flexible_cons_action, flexible_heat_action, battery_action]
        )
        bool_flex.append(
            False if sum(action is None for action in actions[a]) == 3
            else True)

        return actions, bool_flex

    def _check_action_errors(
            self, actions, error, res, loads, a, h, bool_flex
    ):
        for i in range(self.dim_actions):
            if actions[a][i] is not None \
                    and actions[a][i] < self.low_action[i]:
                if actions[a][i] < self.low_action[i] - 1e-2:
                    error[a] = True
                else:
                    actions[a][i] = 0

            if actions[a][i] is not None \
                    and actions[a][i] > self.high_action[i]:
                if actions[a][i] > self.high_action[i] + 1e-2:
                    error[a] = True
                else:
                    actions[a][i] = self.high_action[i]

            if error[a]:
                print(f"h {h} action[{a}] = {actions[a]}")
                np.save('res_error', res)
                np.save('loads', loads)
                np.save('E_heat_min', self.heat.E_heat_min)
                np.save('E_heat_max', self.heat.E_heat_max)
                np.save('action_error', actions)

        actions_none = (self.aggregate_actions and actions[a] is None) \
            or (not self.aggregate_actions
                and all(action is None for action in actions[a]))
        assert not (bool_flex[a] is True and actions_none), \
            f"actions[{a}] are none whereas there is flexibility"

        return error

# figs1
# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.initial_processing(
# [45,45], [10, 10], [10,10], [1,1], [5,5],[20, 20],[5, 5],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_smallgen')

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[30,30]
# env.action_manager.store0=[37.5,37.5]
# env.action_manager.initial_processing(
# [20,20], [10, 10], [15,15], [1,1], [0,0],[10, 10],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_midgen')

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.store0=[37.5,37.5]
# env.action_manager.initial_processing(
# [20,20], [10, 10], [30,30], [1,1], [0,0],[10, 10],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_largegen')

# figs2
# def initial_processing(self, store0, l_flex, gen0,
# avail_EV, E_flex, l_fixed, E_heat_min, as_):

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.initial_processing(
# [35,35], [20, 20], [5,5], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_smallgen')

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.initial_processing(
# [35,35], [20, 20], [5,5], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_legend', legend=True)

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.initial_processing(
# [35,35], [20, 20], [30,30], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_midgen')

# env.action_manager.mincharge=[7.5,7.5]
# env.action_manager.maxcharge=[75,75]
# env.action_manager.initial_processing(
# [35,35], [20, 20], [50,50], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_manager._plot_graph_actions(True, 'mu_largegen')
