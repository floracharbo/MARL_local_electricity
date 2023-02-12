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


class Action_translator:
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
        """Initialise action_translator object and add relevant properties."""
        self.name = 'action translator'
        self.entries = ['dp', 'ds', 'l_ch', 'l_dis', 'c']
        self.plotting = prm['save']['plotting_action']
        self.colours = [(0, 0, 0)] + prm['save']['colours']
        self.n_homes = env.n_homes
        self.N = env.N
        self.labels = [r'$\Delta$p', r'$\Delta$s', 'Losses', 'Consumption']
        self.z_orders = [1, 3, 2, 0, 4]
        self.H = prm['syst']['H']

        type_action = env.spaces.space_info['name'].map(
            lambda x: x[- min(len(x), len('action')):] == 'action'
        )
        self.action_info = env.spaces.space_info[type_action]
        for info in [
            'aggregate_actions', 'dim_actions_1', 'low_action',
            'high_action', 'type_env', 'no_flex_action'
        ]:
            setattr(self, info, prm['RL'][info])
        self.bat_dep = prm['car']['dep']
        self.export_C = prm['grd']['export_C']

    def _get_disaggregated_actions_all(self, actions, bool_flex, res, loads, time_step):
        flexible_cons_action, loads_bool_flex = self._flex_loads_action_all(loads, res, time_step)
        flexible_heat_action, heat_bool_flex = self._flex_heat_action_all(res, time_step)
        flexible_store_action, store_bool_flex = self._flex_store_action_all(res, time_step)

        actions = [flexible_cons_action, flexible_heat_action, flexible_store_action]
        bool_flex.append(
            not sum(action is None for action in actions) == 3
        )

        return actions, bool_flex

    def optimisation_to_rl_env_action(self, time_step, date, netp, loads, home_vars, res):
        """
        From home energy values, get equivalent RL flexibility actions.

        Given home consumption and import variables,
        compute equiavalent RL flexibility action values.
        """
        # loads: l_flex, l_fixed

        homes = range(self.n_homes)

        self.car.min_max_charge_t(time_step, date)
        self.initial_processing(loads, home_vars)
        error = np.zeros(self.n_homes, dtype=bool)
        bool_flex, actions = [], []
        for home in homes:
            if self.aggregate_actions:
                actions, bool_flex = self.get_aggregate_action(
                    actions, bool_flex, netp, home
                )
            else:
                actions, bool_flex = self._get_disaggregated_actions(
                    actions, bool_flex, res, loads, home, time_step
                )
            error = self._check_action_errors(
                actions, error, res, loads, home, time_step, bool_flex
            )

        return bool_flex, actions, error

    def initial_processing(self, loads, home_vars):
        """Compute current flexibility variables."""
        # inputs
        # loads: l_flex, l_fixed
        # home_vars: gen0
        eta_dis, eta_ch = self.car.eta_dis, self.car.eta_ch
        homes = range(self.n_homes)

        s_avail_dis, s_add_0, s_remove_0, potential_charge = self.car.initial_processing()

        self._check_input_types(
            loads, home_vars, s_add_0, s_avail_dis, potential_charge, s_remove_0
        )

        # translate inputs into relevant quantities
        self.tot_l_fixed = loads['l_fixed'] + self.heat.E_heat_min
        tot_l_flex = loads['l_flex'] + self.heat.potential_E_flex()

        # gen to min charge
        g_to_add0 = np.minimum(home_vars['gen'], s_add_0 / eta_dis)

        # gen left after contributing to reaching min charge
        g_net_add0 = home_vars['gen'] - g_to_add0

        # required addition to storage for min charge left
        # after contribution from gen
        s_add0_net = s_add_0 - g_to_add0 * eta_ch
        assert np.all(s_add0_net <= self.car.c_max + 1e-3), \
            f"s_add0_net {s_add0_net} > self.car.c_max {self.car.c_max}"
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
        g_to_store = np.minimum(gnet_flex, potential_charge / eta_ch)

        # How much generation left after storing as much as possible
        gnet_store = gnet_flex - g_to_store
        self.k = {home: {} for home in homes}

        # get relevant points in graph
        d = {entry: {} for entry in self.entries}
        action_points, xs = [{} for _ in range(2)]

        d['ds']['A'] = - s_avail_dis + s_add_0
        dsB = np.maximum(- s_avail_dis + s_add_0, - lnet_fixed / eta_dis)
        d['ds']['B'] = np.where(
            s_remove_0 > 1e-2, np.minimum(dsB, - s_remove_0), dsB
        )

        d['ds']['C'] = s_add_0 - s_remove_0
        d['ds']['D'] = s_add_0 - s_remove_0
        dsE = np.maximum(np.minimum(gnet_flex * eta_ch, potential_charge), s_add_0)
        d['ds']['E'] = [min(dsE[home], - s_remove_0[home])
                        if s_remove_0[home] > 1e-2 else dsE[home] for home in homes]
        d['ds']['F'] = np.where(s_remove_0 > 1e-2, - s_remove_0, potential_charge)
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

        dpF_dspos = (potential_charge - eta_ch * (g_to_add0 + g_to_store)) / eta_ch \
            + lnet_fixed + lnet_flex - gnet_store
        dpF_dsneg = - s_remove_0 * eta_dis + lnet_fixed + lnet_flex - gnet_flex
        d['dp']['F'] = np.where(d['ds']['F'] >= 0, dpF_dspos, dpF_dsneg)

        for i in ['A', 'B', 'C']:
            d['c'][i] = self.tot_l_fixed
        for i in ['D', 'E', 'F']:
            d['c'][i] = self.tot_l_fixed + tot_l_flex
        a_dp = d['dp']['F'] - d['dp']['A']
        a_dp = np.where((- 1e-3 < a_dp) & (a_dp < 0), 0, a_dp)
        assert len(np.where(a_dp < 0)[0]) == 0, f"a_dp {a_dp}"

        b_dp = d['dp']['A']

        action_points['A'], action_points['F'] = np.zeros(self.n_homes), np.ones(self.n_homes)
        for i in ['B', 'C', 'D', 'E']:
            action_points[i] = np.zeros(self.n_homes)
            mask = a_dp > 1e-3
            action_points[i][mask] = (d['dp'][i][mask] - b_dp[mask]) / a_dp[mask]
            for home in homes:
                assert action_points[i][home] > - 5e-4, \
                    f"action_points[{i}][{home}] {action_points[i][home]} < 0"
                if action_points[i][home] > 1 + 5e-4 and self.car.time_step == self.N:
                    action_points[i][home] = 1
                assert action_points[i][home] < 1 + 5e-4, \
                    f"action_points[{i}][{home}] {action_points[i][home]} > 1 " \
                    f"mask {mask} d['dp'][i][mask] {d['dp'][i][mask]} a_dp {a_dp} b_dp {b_dp}"
                if - 1e-4 < action_points[i][home] < 0:
                    action_points[i][home] = 0
                if 1 < action_points[i][home] < 1 + 5e-4:
                    action_points[i][home] = 1
        self.d = d
        self.k, self.action_intervals = [[] for _ in range(2)]

        for home in homes:
            self._compute_k(home, a_dp, b_dp, action_points, d)
            assert self.heat.E_heat_min[home] + loads['l_fixed'][home] \
                   <= self.k[home]['c'][0][1] + 1e-3,\
                   "min c smaller than min required"

        # these variables are useful in optimisation_to_rl_env_action and actions_to_env_vars
        # in the case where action variables are not aggregated
        min_val_ds = np.array(
            [(self.k[home]['ds'][0][0] * 0 + self.k[home]['ds'][0][1]) for home in homes]
        )
        max_val_ds = np.array(
            [self.k[home]['ds'][-1][0] * 1 + self.k[home]['ds'][-1][1] for home in homes]
        )

        self.min_charge = np.where(min_val_ds > 0, min_val_ds, 0)
        self.max_discharge = np.where(min_val_ds < 0, min_val_ds, 0)
        self.max_charge = np.where(max_val_ds > 0, max_val_ds, 0)
        self.min_discharge = np.where(max_val_ds < 0, max_val_ds, 0)

    def actions_to_env_vars(self, loads, home_vars, action, date, time_step):
        """Update variables after non flexible consumption is met."""
        # other variables
        self.error = False
        homes = range(self.n_homes)

        # problem variables
        bool_penalty = self.car.min_max_charge_t(time_step, date)
        for e in ['netp', 'tot_cons']:
            home_vars[e] = np.zeros(self.n_homes)

        self.initial_processing(loads, home_vars)

        # check initial errors
        self.res = {}
        [home_vars['bool_flex'], loads['flex_cons'],
         loads['tot_cons_loads'], self.heat.tot_E] \
            = [[] for _ in range(4)]
        self.l_flex = loads['l_flex']
        flex_heat = np.zeros(self.n_homes)
        for home in homes:
            # boolean for whether we have flexibility
            home_vars['bool_flex'].append(abs(self.k[home]['dp'][0][0]) > 1e-2)
            if self.aggregate_actions:
                flex_heat = None
                # update variables for given action
                # obtain the interval in which action_points lies
                ik = [i for i in range(len(self.action_intervals[home]) - 1)
                      if action[home][0] >= self.action_intervals[home][i]][-1]
                res = {}  # resulting values (for dp, ds, fl, l)
                for e in self.entries:
                    ik_ = 0 if e == 'dp' else ik
                    # use coefficients to obtain value
                    res[e] = self.k[home][e][ik_][0] * action[home][0] \
                        + self.k[home][e][ik_][1]

                home_vars['tot_cons'][home] = res['c']
                home_vars['netp'][home] = res['dp']
                if res['c'] > loads['l_flex'][home] + self.tot_l_fixed[home]:
                    loads['flex_cons'].append(loads['l_flex'][home])
                else:
                    loads['flex_cons'].append(res['c'] - self.tot_l_fixed[home])
                    assert loads['flex_cons'][-1] > - 1e-2, \
                        f"loads['flex_cons'][-1] {loads['flex_cons'][-1]} < 0"
                    if - 1e-2 < loads['flex_cons'][-1] < 0:
                        loads['flex_cons'][-1] = 0
            else:
                flexible_cons_action, flexible_heat_action, flexible_store_action = action[home]
                # flex cons between 0 and 1
                # flex heat between 0 and 1
                # charge between -1 and 1 where
                # -1 max discharge
                # 0 nothing (or just minimum)
                # 1 max charge
                res = {}
                flexible_cons_action_ = 0 if flexible_cons_action is None else flexible_cons_action
                loads['flex_cons'].append(flexible_cons_action_ * loads['l_flex'][home])
                flex_heat[home] = flexible_heat_action * self.heat.potential_E_flex()[home]
                home_vars['tot_cons'][home] = self.tot_l_fixed[home] \
                    + loads['flex_cons'][home] \
                    + flex_heat[home]
                res['c'] = home_vars['tot_cons'][home]
                res = self._flexible_store_action_to_ds(home, flexible_store_action, res)

                discharge = - res['ds'] * self.car.eta_dis \
                    if res['ds'] < 0 else 0
                charge = res['ds'] if res['ds'] > 0 else 0
                home_vars['netp'][home] = loads['flex_cons'][home] \
                    + loads['l_fixed'][home] \
                    + self.heat.E_heat_min[home] \
                    + flex_heat[home] \
                    + charge - discharge + res['l_ch'] \
                    - home_vars['gen'][home]

                res['dp'] = home_vars['netp'][home]

            loads['tot_cons_loads'].append(
                loads['flex_cons'][home] + loads['l_fixed'][home])
            self.res[home] = copy.copy(res)

        self.car.actions_to_env_vars(self.res)
        self.heat.actions_to_env_vars(
            self.res, loads['l_flex'], self.tot_l_fixed, E_flex=flex_heat
        )

        # check for errors
        for home in homes:
            # energy balance
            e_balance = abs((self.res[home]['dp'] + home_vars['gen'][home]
                             + self.car.discharge[home] - self.car.charge[home]
                             - self.car.loss_ch[home] - home_vars['tot_cons'][home]))
            assert e_balance <= 1e-2, f"energy balance {e_balance}"
            assert abs(loads['tot_cons_loads'][home] + self.heat.tot_E[home]
                   - home_vars['tot_cons'][home]) <= 1e-3, \
                f"tot_cons_loads {loads['tot_cons_loads'][home]}, "\
                f"self.heat.tot_E[home] {self.heat.tot_E[home]}, " \
                f"home_vars['tot_cons'][home] {home_vars['tot_cons'][home]}"

        bool_penalty = self.car.check_errors_apply_step(
            homes, bool_penalty, action, self.res)
        if sum(bool_penalty) > 0:
            self.error = True
        if not self.error and self.plotting:
            self._plot_graph_actions()
        # outputs
        # loads: flex_cons, tot_cons_loads
        # home_vars: netp, bool_flex, tot_cons
        # bool_penalty

        return loads, home_vars, bool_penalty

    def _flexible_store_action_to_ds(self, home, flexible_store_action, res):
        """Convert flexible store action to change in storage level ds."""
        if flexible_store_action is None:
            assert abs(self.min_charge[home] - self.max_charge[home]) <= 1e-3, \
                "flexible_store_action is None but " \
                "self.min_charge[home] != self.max_charge[home]"
            if self.min_charge[home] > 1e-3:
                res['ds'] = self.min_charge[home]
            elif self.min_discharge[home] < - 1e-3:
                res['ds'] = self.min_discharge[home]
            else:
                res['ds'] = 0
        elif flexible_store_action < 0:
            if self.min_charge[home] > 0:
                res['ds'] = self.min_charge[home]
            elif self.min_discharge[home] <= 0:
                res['ds'] = self.min_discharge[home] + abs(flexible_store_action) \
                    * (self.max_discharge[home] - self.min_discharge[home])
        elif flexible_store_action >= 0:
            if self.min_discharge[home] < 0:
                res['ds'] = self.min_discharge[home]
            elif self.max_charge[home] >= 0:
                res['ds'] = self.min_charge[home] + flexible_store_action \
                    * (self.max_charge[home] - self.min_charge[home])
        res['l_ch'] = 0 if res['ds'] < 0 else (1 - self.car.eta_ch) / self.car.eta_ch * res['ds']
        res['l_dis'] = - res['ds'] * (1 - self.car.eta_dis) if res['ds'] < 0 else 0

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
                [self.colours[ie], self.z_orders[ie], self.labels[ie]]
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
        costs = [loss * 0.1 + self.bat_dep * d + e * self.export_C
                 for loss, d, e in zip(ys['Losses'], discharge, export)]
        ax2.plot(self.action_intervals[0], costs,
                 color=self.colours[len(entries_plot)],
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

    def _compute_k(self, home, a_dp, b_dp, action_points, d):
        """Get the coefficients of the linear function for the action variable."""
        letters = ['A', 'B', 'C', 'D', 'E', 'F']

        self.k.append({entry: [] for entry in self.entries})
        # reference line - dp

        self.k[home]['dp'] = [[a_dp[home], b_dp[home]]]
        self.action_intervals.append([0])

        for e in ['ds', 'c', 'l_ch', 'l_dis']:
            self.k[home][e] = []
        for z in range(5):
            l1, l2 = letters[z: z + 2]
            if action_points[l2][home] > action_points[l1][home]:
                self.action_intervals[home].append(action_points[l2][home])
                for e in ['ds', 'c', 'losses']:
                    if e == 'losses':
                        self.k = self.car.k_losses(
                            home,
                            self.k,
                            action_points[l1][home],
                            action_points[l2][home]
                        )
                    else:
                        ad = (d[e][l2][home] - d[e][l1][home]) / \
                             (action_points[l2][home] - action_points[l1][home])
                        bd = d[e][l2][home] - action_points[l2][home] * ad
                        self.k[home][e].append([ad, bd])

    def _check_input_types(
            self, loads, home_vars, s_add_0, s_avail_dis, potential_charge, s_remove_0
    ):
        """Check the input types."""
        assert isinstance(loads['l_fixed'], np.ndarray), \
            f"type(loads['l_fixed']) {type(loads['l_fixed'])}"
        assert isinstance(loads['l_flex'], np.ndarray), \
            f"type(loads['l_flex']) {type(loads['l_flex'])}"
        assert isinstance(self.heat.E_heat_min, np.ndarray), \
            f"type(self.heat.E_heat_min) {type(self.heat.E_heat_min)}"
        assert isinstance(home_vars['gen'], np.ndarray), \
            f"type(home_vars['gen']) = {type(home_vars['gen'])}"
        assert isinstance(s_add_0, np.ndarray), \
            f"type(s_add_0) = {type(s_add_0)}"
        assert isinstance(s_avail_dis, np.ndarray), \
            f"type(s_avail_dis) = {type(s_avail_dis)}"
        assert isinstance(potential_charge, np.ndarray), \
            f"type(potential_charge) = {type(potential_charge)}"
        assert isinstance(s_remove_0, np.ndarray), \
            f"type(s_remove_0) = {type(s_remove_0)}"

    def aggregate_action_bool_flex(self, home):
        """Check that there is flexibility over all three sub-actions."""
        return self.k[home]['dp'][0][0] > 0

    def get_flexibility(self, home):
        """Compute total flexibility between minimum and maximum possible imports/exports."""
        return self.k[home]['dp'][0][0]

    def store_bool_flex(self, home):
        """Check that there is flexibility over the storage sub-action."""
        return abs(self.k[home]['ds'][0][1] - sum(self.k[home]['ds'][-1])) > 1e-3

    def get_aggregate_action(self, actions, bool_flex, netp, home):
        """Compute the aggregate action variable from the current import/export."""
        if abs(self.k[home]['dp'][0][0]) < 1e-2:
            self.k[home]['dp'][0][0] = 0
        bool_flex_ = self.aggregate_action_bool_flex(home)
        bool_flex.append(bool_flex_)

        if not bool_flex_:  # there is no flexibility
            # boolean for whether we have flexibility
            actions.append([self._get_no_flex_action('action')])
        else:
            # action none if no flexibility
            assert netp[home] - self.k[home]['dp'][0][1] > - 1e-2, \
                "netp smaller than k['dp'][0]"
            delta = 0 if abs(netp[home] - self.k[home]['dp'][0][1]) < 1e-2 \
                else netp[home] - self.k[home]['dp'][0][1]
            actions.append(
                [
                    None if self.k[home]['dp'][0][0] == 0
                    else delta / self.k[home]['dp'][0][0]
                ]
            )
        if actions[home] is not None:
            try:
                assert - 1e-2 < actions[home][0] < 1 + 1e-2, \
                    "action should be between 0 and 1"
            except Exception as ex:
                print(ex)

        return actions, bool_flex

    def _get_no_flex_action(self, action_type):
        if self.no_flex_action == 'one':
            action = 1
        elif self.no_flex_action == 'random' or self.type_env == 'discrete':
            action = np.random.rand()
        elif self.no_flex_action == 'None':
            action = None

        min_action, max_action = [
            self.action_info.loc[self.action_info["name"] == action_type, col].values[0]
            for col in ['min', 'max']
        ]

        action = action * (max_action - min_action) + min_action

        return action

    def _flex_loads_action(self, loads, home, res, time_step):
        """Compute the flexible household loads consumption action from the optimisation result."""
        loads_bool_flex = True
        flexible_cons_action = None
        if loads['l_flex'][home] < 1e-3:
            loads_bool_flex = False
            flexible_cons_action = self._get_no_flex_action('flexible_cons_action')

        if loads_bool_flex:
            cons = res['house_cons'][home, time_step]
            if cons < 1e-3:
                cons = 0

            flex_cons = cons - loads['l_fixed'][home]
            if flex_cons < 1e-3:
                flex_cons = 0
            elif loads['l_flex'][home] < flex_cons < loads['l_flex'][home] + 1e-3:
                flex_cons = loads['l_flex'][home]
            flexible_cons_action = flex_cons / loads['l_flex'][home]

        if flexible_cons_action > 1:
            if abs(flex_cons - loads['l_flex'][home]) < 5e-3:
                flexible_cons_action = 1
            else:
                print(f"flex_cons {flex_cons} loads['l_flex'][home] {loads['l_flex'][home]}")

        return flexible_cons_action, loads_bool_flex

    def _flex_heat_action(self, home, res, time_step):
        """Compute the flexible heat energy consumption action from the optimisation result."""
        heat_bool_flex = True
        flexible_heat_action = None
        if self.heat.potential_E_flex()[home] < 1e-3:
            heat_bool_flex = False
            flexible_heat_action = self._get_no_flex_action('flexible_heat_action')
        if heat_bool_flex:
            E_heat = 0 if res['E_heat'][home][time_step] < 1e-3 else res['E_heat'][home][time_step]
            if abs(res['E_heat'][home][time_step] - self.heat.E_heat_min[home]) < 1e-3:
                flexible_heat_action = 0
            else:
                flexible_heat_action = \
                    (E_heat - self.heat.E_heat_min[home]) / \
                    (self.heat.E_heat_max[home] - self.heat.E_heat_min[home])

        return flexible_heat_action, heat_bool_flex

    def _flex_store_action(self, res, home, time_step):
        """Compute the flexible storage action from the optimisation result."""
        store_bool_flex = True
        flexible_store_action = None
        no_flex_charge = abs(self.max_charge[home] - self.min_charge[home]) < 1e-3
        no_flex_discharge = abs(self.min_discharge[home] - self.max_discharge[home]) < 1e-3
        if (
            (no_flex_charge and no_flex_discharge)
            or (no_flex_discharge and res['discharge_other'][home, time_step] > 1e-3)
            or (no_flex_charge and res['charge'][home, time_step] > 1e-3)
        ):
            store_bool_flex = False
            flexible_store_action = self._get_no_flex_action('battery_action')

            assert abs(self.min_charge[home] - self.max_charge[home]) <= 1e-3, \
                "flexible_store_action is None but " \
                "self.min_charge[home] != self.max_charge[home]"
        if store_bool_flex:
            assert self.min_charge[home] - 1e-3 <= res['charge'][home, time_step] \
                   <= self.max_charge[home] + 1e-3, \
                   f"res charge {res['charge'][home, time_step]} " \
                   f"self.min_charge[home] {self.min_charge[home]} " \
                   f"self.max_charge[home] {self.max_charge[home]}"
            assert self.max_discharge[home] - 1e-3 \
                   <= - res['discharge_other'][home, time_step] / self.car.eta_dis \
                   <= self.min_discharge[home] + 1e-3, \
                   f"res discharge_other {res['discharge_other'][home, time_step]} " \
                   f"self.min_discharge[home] {- self.min_discharge[home]} " \
                   f"self.max_discharge[home] {- self.max_discharge[home]}"
            if abs(res['discharge_other'][home, time_step]) < 1e-3 \
                    and abs(res['charge'][home, time_step]) < 1e-3:
                flexible_store_action = 0
            elif res['discharge_other'][home, time_step] > 1e-3:
                flexible_store_action = \
                    (self.min_discharge[home] - res['charge'][home, time_step]) \
                    / (self.min_discharge[home] - self.max_discharge[home])
            else:
                if abs(res['charge'][home, time_step] - self.max_charge[home]) < 1e-3:
                    flexible_store_action = 1
                else:
                    flexible_store_action = (
                        res['charge'][home, time_step] - self.min_charge[home]
                    ) / (self.max_charge[home] - self.min_charge[home])

        return flexible_store_action, store_bool_flex

    def _get_disaggregated_actions(self, actions, bool_flex, res, loads, home, time_step):
        """Get all three sub-action values from the optimisation result."""
        flexible_cons_action, loads_bool_flex = self._flex_loads_action(loads, home, res, time_step)
        flexible_heat_action, heat_bool_flex = self._flex_heat_action(home, res, time_step)
        flexible_store_action, store_bool_flex = self._flex_store_action(res, home, time_step)

        actions.append(
            [flexible_cons_action, flexible_heat_action, flexible_store_action]
        )
        bool_flex.append(
            not sum(action is None for action in actions[home]) == 3
        )

        return actions, bool_flex

    def _check_action_errors(
            self, actions, error, res, loads, home, time_step, bool_flex
    ):
        """Check assertion errors for the translation of optimisation results into rl actions."""
        for i in range(self.dim_actions_1):
            if actions[home][i] is not None \
                    and actions[home][i] < self.low_action[i]:
                if actions[home][i] < self.low_action[i] - 1e-2:
                    error[home] = True
                else:
                    actions[home][i] = 0

            if actions[home][i] is not None \
                    and actions[home][i] > self.high_action[i]:
                if actions[home][i] > self.high_action[i] + 1e-2:
                    error[home] = True
                else:
                    actions[home][i] = self.high_action[i]

            if error[home]:
                print(f"time_step {time_step} action[{home}] = {actions[home]}")
                np.save('res_error', res)
                np.save('loads', loads)
                np.save('E_heat_min', self.heat.E_heat_min)
                np.save('E_heat_max', self.heat.E_heat_max)
                np.save('action_error', actions)

        actions_none = (self.aggregate_actions and actions[home] is None) \
            or (not self.aggregate_actions
                and all(action is None for action in actions[home]))
        assert not (bool_flex[home] is True and actions_none), \
            f"actions[{home}] are none whereas there is flexibility"

        return error

# figs1
# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.initial_processing(
# [45,45], [10, 10], [10,10], [1,1], [5,5],[20, 20],[5, 5],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_smallgen')

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[30,30]
# env.action_translator.store0=[37.5,37.5]
# env.action_translator.initial_processing(
# [20,20], [10, 10], [15,15], [1,1], [0,0],[10, 10],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_midgen')

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.store0=[37.5,37.5]
# env.action_translator.initial_processing(
# [20,20], [10, 10], [30,30], [1,1], [0,0],[10, 10],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_largegen')

# figs2
# def initial_processing(self, store0, l_flex, gen0,
# avail_EV, E_flex, l_fixed, E_heat_min, homes):

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.initial_processing(
# [35,35], [20, 20], [5,5], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_smallgen')

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.initial_processing(
# [35,35], [20, 20], [5,5], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_legend', legend=True)

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.initial_processing(
# [35,35], [20, 20], [30,30], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_midgen')

# env.action_translator.mincharge=[7.5,7.5]
# env.action_translator.maxcharge=[75,75]
# env.action_translator.initial_processing(
# [35,35], [20, 20], [50,50], [1,1], [0,0],[20, 20],[0, 0],[0, 1])
# env.action_translator._plot_graph_actions(True, 'mu_largegen')
