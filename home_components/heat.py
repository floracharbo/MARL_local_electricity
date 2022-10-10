#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:39:04 2021.

@author: Flora Charbonnier
"""
import numpy as np


class Heat:
    """
    Heating components of homes.

    Public methods:
    __init__:
        Initialise Heat object.
    reset:
        Reset object for new episode.
    next_T:
        Obtain the next temperature given heating energy.
    current_temperature_bounds:
        Lower and upper temperature bounds for current time step.
    E_heat_min_max:
        Update the max and min heating requirement for current time step.
    potential_E_flex:
        Obtain the amount of energy that can be flexibly consumed.
    update_step:
        At the end of the step, update the current temperature.
    check_constraints:
        From env.policy_to_rewardvar, check heat constraints.
    apply_step:
        Get fixed/flexible heat consumption from current actions.

    Private method:
    _next_E_heat:
        Obtain heating energy required to reach next T_air_target.
    """

    def __init__(self, prm, i0_costs, passive_ext, E_req_only):
        """
        Initialise Heat object.

        inputs:
        prm:
            input parameters
        i0_costs:
            index start date current day
        passive_ext:
            are the current agents passive? '' is not, 'P' if yes
        E_req_only:
            boolean - if True, there is no flexibility around T_req
        """
        self.p = ''
        self._update_passive_active_vars(prm)

        # flexibility around T_req
        self.dT = prm["heat"]['dT']

        # number of hours (time intervals) in a day
        self.H = prm["syst"]['H']

        # number of hours (time intervals) in episode
        self.N = prm["syst"]['N']

        # amount of energy consumed at time step that was optional
        # / could have been delayed
        self.E_flex = None

        self.reset(prm, i0_costs, passive_ext, E_req_only)

    def reset(self, prm, i0_costs=None, passive_ext=None, E_req_only=None):
        """
        Reset object for new episode.

        inputs:
        prm:
            input parameters
        i0_costs:
            index start date current day
        passive_ext:
            are the current agents passive? '' is not, 'P' if yes
        E_req_only:
            if True, there is no flexibility around T_req
        """
        # current time step
        self.i_step = 0

        # extension to add to variable names if agents are passive
        if passive_ext is not None:
            self.p = passive_ext
            self._update_passive_active_vars(prm)

        if i0_costs is not None:
            self.i0_costs = i0_costs

        if E_req_only is None:
            E_req_only = False

        # current indoor air temperatures
        self.T_air = [prm["heat"]["T_req" + self.p][a][0]
                      for a in range(self.n_agents)]

        # required temperature profile
        self.T_req = prm["heat"]["T_req" + self.p]

        # temperature bounds based on whether there is flexibility or not
        if not E_req_only:
            self.T_UB = prm["heat"]["T_UB" + self.p]
            self.T_LB = prm["heat"]["T_LB" + self.p]
        else:
            self.T_UB, self.T_LB = [self.T_req for _ in range(2)]

        # external temperature
        self.T_out = prm["heat"]['T_out_all'][
            self.i0_costs: self.i0_costs + prm['syst']['N']]

    def next_T(self, T_start=None, E_heat=None, T_out_t=None,
               update=False):
        """Obtain the next temperature given heating energy.

        Inputs:
        T_start:
            the current building air mass temperature [deg C]
            (start of time step)
        E_heat:
            the heating energy [kWh]
        T_out_t:
            the current outdoor temperature [deg C]
        update:
            boolean: if True, update current temperature values

        Output:
        T_end:
            the resulting building mass temperature at the end of the
            time step / start of the next one [deg C]
        T_air
            the resulting air temperature over the current time step [deg C]
        """
        # use inputs or current object variables
        T_start = self.T if T_start is None else T_start
        E_heat = [e_min + e_flex for e_min, e_flex
                  in zip(self.E_heat_min, self.E_flex)] \
            if E_heat is None else E_heat
        T_out_t = self.T_out[self.i_step] if T_out_t is None else T_out_t

        # number of agents considered in the functio
        na = len(T_start)

        Pheat = [E_heat[a] * 1e3 * self.H / 24 for a in range(na)]
        T_end = [self.T_coeff[a][0]
                 + self.T_coeff[a][1] * T_start[a]
                 + self.T_coeff[a][2] * T_out_t
                 + self.T_coeff[a][4] * Pheat[a] for a in range(na)]
        # + self.T_coeff[a][3] * phi_solt
        T_air = [self.T_air_coeff[a][0]
                 + self.T_air_coeff[a][1] * T_start[a]
                 + self.T_air_coeff[a][2] * T_out_t
                 + self.T_air_coeff[a][4] * Pheat[a] for a in range(na)]
        # + self.T_coeffair[a][3] * phi_solt

        if update:
            self.T_next = T_end
            self.T_air = T_air

        return T_end, T_air

    def current_temperature_bounds(self, i_step):
        """
        Lower and upper temperature bounds for current time step.

        input:
        i_step:
            index current time step
        """
        for e in ["T_LB", "T_UB"]:
            self.__dict__[e + "_t"] = [self.__dict__[e][a][i_step]
                                       for a in range(self.n_agents)]

    def E_heat_min_max(self, i_step):
        """
        Update the max and min heating requirement for current time step.

        input:
        i_step:
            index current time step
        """
        self.current_temperature_bounds(i_step)

        # minimum and maximum heating for reaching minimum and
        # maximum temperature at the next time step
        E_heat_min0, E_heat_max0 = [self._next_E_heat(
            T_req, self.T, self.T_out[i_step])
            for T_req in [self.T_LB_t, self.T_UB_t]]

        # if there is a drop in temperature in one or two steps,
        # check we are not allowing too much heating
        for a in range(self.n_agents):
            diff_E_heat_min_max = E_heat_min0[a] != E_heat_max0[a]
            lower_nextTUB = self.T_UB[a][i_step + 1] < self.T_LB_t[a]
            lower_afternextTUB = False if i_step > len(self.T_UB[a]) - 3 \
                else (self.T_UB[a][i_step + 2] < self.T_UB_t[0])
            if diff_E_heat_min_max and (lower_nextTUB or lower_afternextTUB):
                # temperatures at time 1 if heating to the max at time 0
                T1_max, T_air0_max = self.next_T(
                    [self.T[a]], [E_heat_max0[a]], self.T_out[i_step])
                # check all works well, T_air0_max should be T_UB
                if abs(T_air0_max[0] - self.T_UB_t[a]) > 1e-2:
                    print(f'T_air0_max[0] {T_air0_max[0]}, '
                          f'self.T_UB_t[a] {self.T_UB_t[a]}')
                # check at time step 1 if we would be over limit
                # if heating was 0 starting from max temp
                T2_noheatt1, T_air1_noheatt1 = self.next_T(
                    T1_max, [0], self.T_out[i_step + 1])
                if T_air1_noheatt1[0] > self.T_UB[a][i_step + 1]:
                    # find T_max next such that if you do not heat
                    # at the next step you land on T_UB
                    T1_corrected = \
                        (self.T_UB[a][i_step + 1] - self.T_air_coeff[a][0]
                         - self.T_air_coeff[a][2] * self.T_out[i_step + 1]) \
                        / self.T_air_coeff[a][1]
                    # find how much to heat to reach that T1_corrected
                    e_max0_corrected = self._next_E_heat(
                        [T1_corrected], [self.T[a]],
                        self.T_out[i_step])[0]
                    # obtain corrected next temperatures
                    T2_noheatt1_corrected, T_air1_noheatt1_corrected = \
                        self.next_T([T1_corrected], [e_max0_corrected],
                                    self.T_out[i_step + 1])
                    T2_noheatt1 = T2_noheatt1_corrected
                    E_heat_max0[a] = e_max0_corrected

                # check in two time steps' time if you do not heat
                if i_step < len(self.T_out) - 1:
                    T3_noheatt2, T_air2_noheatt2 = \
                        self.next_T(T2_noheatt1, [0] * self.n_agents,
                                    self.T_out[i_step + 2])
                    if T_air2_noheatt2[0] > self.T_UB[a][i_step + 2]:
                        # find T_max next such that if you do not heat
                        # at the next step you land on T_UB
                        T2_corrected = \
                            (self.T_UB[a][i_step + 2]
                             - self.T_air_coeff[a][0]
                             - self.T_air_coeff[a][2]
                             * self.T_out[i_step + 2]) \
                            / self.T_air_coeff[a][1]
                        T1_corrected = \
                            (T2_corrected - self.T_air_coeff[a][0]
                             - self.T_air_coeff[a][2]
                             * self.T_out[i_step + 1]) / \
                            self.T_air_coeff[a][1]

                        # find how much to heat to reach that T2_corrected
                        e_max0_corrected = self._next_E_heat(
                            [T1_corrected], [self.T[a]],
                            self.T_out[i_step])[0]

                        # obtain corrected next temperatures
                        E_heat_max0[a] = e_max0_corrected

            # check E_heat_min makes sense
            assert E_heat_min0[a] < 100, f"E_heat_min0 = {E_heat_min0}"
            assert E_heat_min0[a] <= E_heat_max0[a], \
                f"E_heat_min0[{a}] {E_heat_min0[a]} " \
                f"> E_heat_max0[{a}] {E_heat_max0[a]}"

        self.E_heat_min, self.E_heat_max = [
            np.where(self.own_heat, heat0, 0)
            for heat0 in [E_heat_min0, E_heat_max0]
        ]

    def potential_E_flex(self):
        """Obtain the amount of energy that can be flexibly consumed."""
        return self.E_heat_max - self.E_heat_min

    def update_step(self, res=None):
        """
        At the end of the step, update the current temperature.

        Input:
        res:
            results from optimisation, to update the
            temperature if learning from an optimisation rather than
            applying decisions
        """
        self.i_step += 1
        if res is None:
            self.T = self.T_next
        elif self.i_step < self.N:
            self.T = [res["T"][a][self.i_step] for a in range(self.n_agents)]

    def check_constraints(self, a, h, E_req_only):
        """
        From env.policy_to_rewardvar, check heat constraints.

        Inputs:
        a:
            index relative to agent
        h:
            time step
        bool_penalty:
            the array of booleans keeping track of whether
            each agent breaks constraints
        E_req_only:
            if True, there is no flexibility around T_req

        output:
        updated bool_penalty
        """
        # whether there is already an error at the start of the method

        # check E_heat_min calculation was correct; resulting
        # in a temperature larger than the minimum temperature
        if self.own_heat[a]:
            assert self.next_T(
                self.T, self.E_heat_min, self.T_out[h]
            )[1][a] >= self.T_LB_t[a] - 1e-2, "next_T < T_LB"

        # check if target temperature is met
        if not E_req_only:
            assert self.T_air[a] <= self.T_UB[a][h] + 1e-1, "T_air > T_UB"

            assert self.T_air[a] >= self.T_LB[a][h] - 1e-1, "T_air < T_LB"

            # check E_heat_max makes sense
            assert self.next_T(
                [self.T[a]], [self.E_heat_max[a]], self.T_out[h]
            )[1][0] <= self.T_UB[a][h] + 0.05, "next_T > T_UB"

        else:
            # E_req_only is for baseline where we do not use the
            # flexibility and only target the middle temperature requirement
            if self.own_heat[a]:
                assert not (
                    self.T_air[a] > self.T_UB[a][h] + self.dT + 1e-1
                    or self.T_air[a] < self.T_LB[a][h] - 1e-1 - self.dT), \
                    "T_air > T_UB + dT or T_air < T_LB - dT"

        # positivity
        assert self.potential_E_flex()[a] >= 0, "potential_E_flex < 0"

    def apply_step(self, res, l_flex, tot_l_fixed, E_flex=None):
        """Get fixed/flexible heat consumption from current actions."""
        self.E_flex, self.tot_E = [], []
        if E_flex is None:
            for a in range(self.n_agents):
                if self.own_heat[a] \
                        and res[a]['c'] > l_flex[a] + tot_l_fixed[a]:
                    self.E_flex.append(
                        res[a]['c'] - tot_l_fixed[a] - l_flex[a])
                else:
                    self.E_flex.append(0)
                self.tot_E.append(self.E_flex[a] + self.E_heat_min[a])
        else:
            # if flexible energy consumption was defined through
            # mu_heat in apply_step mu_manager
            self.E_flex = E_flex
            self.tot_E = [flex + min for flex, min in
                          zip(self.E_flex, self.E_heat_min)]

    def _next_E_heat(self, T_air_target, T_start, T_out_t):
        """
        Obtain heating energy required to reach next T_air_target.

        Inputs:
        T_air_target:
            target air temperature (middle of time step) [deg C]
        T_start:
            the current building air mass temperature
            (start of time step) [deg C]
        T_out_t:
            the current outdoor temperature [deg C]

        Output:
        E_heat:
            the heating energy required [kWh]
        """
        # number of agents considered
        na = len(T_air_target)

        # heating power
        p_heat = [(T_air_target[a]
                   - (self.T_air_coeff[a][0]
                      + self.T_air_coeff[a][1] * T_start[a]
                      + self.T_air_coeff[a][2] * T_out_t))
                  / self.T_air_coeff[a][4] for a in range(na)]
        # + self.T_air_coeff[a][3] * phi_solt

        # conversion of power to energy based on duration of time interval
        E_heat = [p_heat[a] * 1e-3 * 24 / self.H for a in range(na)]

        # check
        E_heat = np.array([0 if e < 0 else e for e in E_heat])

        return E_heat

    def _update_passive_active_vars(self, prm):
        # number of agents / households
        self.n_agents = prm["ntw"]["n" + self.p]

        # current building mass temperatures
        self.T = [prm["heat"]["T0"] for _ in range(self.n_agents)]

        # heating coefficients for recursive description
        self.T_air_coeff = prm["heat"]["T_air_coeff" + self.p]
        self.T_coeff = prm["heat"]["T_coeff" + self.p]

        self.own_heat = prm["heat"]["own_heat" + self.p]
