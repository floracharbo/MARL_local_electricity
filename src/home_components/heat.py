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
    actions_to_env_vars:
        Get fixed/flexible heat consumption from current actions.

    Private method:
    _next_E_heat:
        Obtain heating energy required to reach next T_air_target.
    """

    def __init__(self, prm, i0_costs, ext, E_req_only):
        """
        Initialise Heat object.

        inputs:
        prm:
            input parameters
        i0_costs:
            index start date current day
        ext:
            are the current agents passive? '' is not, 'P' if yes
            are we looking at a different number of tested homes? '_test' if yes else ''
        E_req_only:
            boolean - if True, there is no flexibility around T_req
        """
        self.ext = ''
        self._update_passive_active_vars(prm)

        # flexibility around T_req
        self.dT = prm["heat"]['dT']

        # number of hours (time intervals) in a day
        self.H = prm["syst"]['H']

        # number of hours (time intervals) in episode
        self.N = prm["syst"]['N']

        # number of intervals per hour
        self.n_int_per_hr = prm["syst"]['n_int_per_hr']

        # amount of energy consumed at time step that was optional
        # / could have been delayed
        self.E_flex = None

        self.reset(prm, i0_costs, ext, E_req_only)
        for attribute in ['T_air', 'tot_E', 'T_next']:
            setattr(self, attribute, np.full(self.n_homes, np.nan))

    def reset(self, prm, i0_costs=None, ext=None, E_req_only=None):
        """
        Reset object for new episode.

        inputs:
        prm:
            input parameters
        i0_costs:
            index start date current day
        ext:
            are the current agents passive? '' is not, 'P' if yes
            are we looking at a different number of tested homes? '_test' if yes else ''
        E_req_only:
            if True, there is no flexibility around T_req
        """
        # current time step
        self.time_step = 0

        # extension to add to variable names if agents are passive
        if ext is not None:
            self.ext = ext
            self._update_passive_active_vars(prm)

        if i0_costs is not None:
            self.update_i0_costs(prm, i0_costs)

        if E_req_only is None:
            E_req_only = False

        # current indoor air temperatures
        self.T = prm["heat"]["T_req" + self.ext][:, 0]

        # required temperature profile
        self.T_req = prm["heat"]["T_req" + self.ext]

        # temperature bounds based on whether there is flexibility or not
        if not E_req_only:
            self.T_UB = prm["heat"]["T_UB" + self.ext]
            self.T_LB = prm["heat"]["T_LB" + self.ext]
        else:
            self.T_UB, self.T_LB = [self.T_req for _ in range(2)]

    def update_i0_costs(self, prm, i0_costs):
        self.i0_costs = i0_costs
        # external temperature
        self.T_out = prm["heat"]['T_out_all'][
            self.i0_costs: self.i0_costs + prm['syst']['N'] + 1]

    def next_T(self, T_start=None, E_heat=None, T_out_t=None,
               update=False, home=None):
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
        homes = list(range(self.n_homes)) if home is None else [home]
        n_homes = len(homes)
        if n_homes > 0:
            if T_start is None:
                T_start = self.T[homes]
            if E_heat is None:
                E_heat = self.E_heat_min[homes] + self.E_flex[homes]
            if T_out_t is None:
                T_out_t = self.T_out[self.time_step]
            P_heat = E_heat * 1e3 * self.n_int_per_hr
            M = np.ones((5, n_homes))
            M[1, :] = T_start
            M[2, :] *= T_out_t
            M[3, :] *= 0
            M[4, :] = P_heat
            K = self.T_coeff[homes]
            T_end = np.sum(np.multiply(K, M.T), axis=1)

            K_air = self.T_air_coeff[homes]
            T_air = np.sum(np.multiply(K_air, M.T), axis=1)
            T_air = np.where(self.own_heat[homes], T_air, self.T_req[homes, self.time_step])
            if update:
                self.T_next = T_end
                self.T_air = T_air
        else:
            T_end = np.array([])
            T_air = np.array([])

        return T_end, T_air

    def current_temperature_bounds(self, time_step):
        """
        Lower and upper temperature bounds for current time step.

        input:
        time_step:
            index current time step
        """
        for T in ["T_LB", "T_UB"]:
            setattr(
                self,
                f"{T}_t",
                self.__dict__[T][:, time_step]
            )

    def E_heat_min_max(self, time_step):
        """
        Update the max and min heating requirement for current time step.

        input:
        time_step:
            index current time step
        """
        self.current_temperature_bounds(time_step)

        # minimum and maximum heating for reaching minimum and
        # maximum temperature at the next time step
        E_heat_min0, E_heat_max0 = [
            self._next_E_heat(T_req, self.T, self.T_out[time_step])
            for T_req in [self.T_LB_t, self.T_UB_t]
        ]

        # if there is a drop in temperature in one or two steps,
        # check we are not allowing too much heating
        for home in range(self.n_homes):
            if self.own_heat[home]:
                diff_E_heat_min_max = E_heat_min0[home] != E_heat_max0[home]
                lower_next_T_UB = self.T_UB[home][time_step + 1] < self.T_UB_t[home]
                lower_after_next_T_UB = False if time_step > len(self.T_UB[home]) - 3 \
                    else (self.T_UB[home][time_step + 2] < self.T_UB_t[0])
                if diff_E_heat_min_max and (lower_next_T_UB or lower_after_next_T_UB):
                    # temperatures at time 1 if heating to the max at time 0
                    T1_max, T_air0_max = self.next_T(
                        self.T[home], E_heat_max0[home], self.T_out[time_step], home=home
                    )
                    # check all works well, T_air0_max should be T_UB
                    assert abs(T_air0_max[0] - self.T_UB_t[home]) < 1e-2, \
                        f'T_air0_max[0] {T_air0_max[0]}, ' \
                        f'self.T_UB_t[home] {self.T_UB_t[home]}'
                    # check at time step 1 if we would be over limit
                    # if heating was 0 starting from max temp
                    T2_noheatt1, T_air1_noheatt1 = self.next_T(
                        T1_max, 0, self.T_out[time_step + 1], home=home
                    )
                    if T_air1_noheatt1[0] > self.T_UB[home][time_step + 1]:
                        # find T_max next such that if you do not heat
                        # at the next step you land on T_UB
                        # T1_corrected = (
                        #     self.T_UB[home][time_step + 1] - self.T_air_coeff[home][0]
                        #     - self.T_air_coeff[home][2] * self.T_out[time_step + 1]
                        # ) / self.T_air_coeff[home][1]
                        # find how much to heat to reach that T1_corrected
                        e_max0_corrected = self._next_E_heat(
                            [self.T_UB[home][time_step + 1]],
                            [self.T[home]],
                            self.T_out[time_step],
                            home=home
                        )[0]

                        # check current time step
                        T1_max_corrected, T_air0_max_corrected = self.next_T(
                            self.T[home], e_max0_corrected, self.T_out[time_step], home=home
                        )
                        #  check time step after
                        T2_noheatt1_corrected, T_air1_noheatt1_corrected = self.next_T(
                            T1_max_corrected, 0, self.T_out[time_step + 1], home=home
                        )
                        T2_noheatt1 = T2_noheatt1_corrected
                        E_heat_max0[home] = e_max0_corrected

                    # check in two time steps' time if you do not heat
                    if time_step < len(self.T_out) - 1:
                        T3_noheatt2, T_air2_noheatt2 = self.next_T(
                            T2_noheatt1, 0, self.T_out[time_step + 2], home=home
                        )
                        if T_air2_noheatt2[0] > self.T_UB[home][time_step + 2]:
                            # find T_max next such that if you do not heat
                            # at the next step you land on T_UB
                            T2_corrected = (
                                self.T_UB[home][time_step + 2]
                                - self.T_air_coeff[home][0]
                                - self.T_air_coeff[home][2] * self.T_out[time_step + 2]
                            ) / self.T_air_coeff[home][1]
                            T1_corrected = (
                                T2_corrected
                                - self.T_coeff[home][0]
                                - self.T_coeff[home][2] * self.T_out[time_step + 1]
                            ) / self.T_coeff[home][1]

                            # find how much to heat to reach that T2_corrected
                            e_max0_corrected = self._next_E_heat(
                                [T1_corrected], [self.T[home]],
                                self.T_out[time_step], home=home, target='building_mass_temp'
                            )[0]

                            # check current time step
                            T1_max_corrected, T_air0_max_corrected = self.next_T(
                                self.T[home], e_max0_corrected, self.T_out[time_step], home=home
                            )
                            #  check time step after
                            T2_noheatt1_corrected, T_air1_noheatt1_corrected = self.next_T(
                                T1_max_corrected, 0, self.T_out[time_step + 1], home=home
                            )

                            #  check time step after (t+2)
                            # T3_noheatt2_corrected, T_air2_noheatt2_corrected = self.next_T(
                            #     T2_noheatt1_corrected, 0, self.T_out[time_step + 2], home=home
                            # )
                            # obtain corrected next temperatures
                            E_heat_max0[home] = e_max0_corrected

            # check E_heat_min makes sense
            assert E_heat_min0[home] < 100, f"E_heat_min0 = {E_heat_min0}"
            assert E_heat_min0[home] <= E_heat_max0[home], \
                f"E_heat_min0[{home}] {E_heat_min0[home]} " \
                f"> E_heat_max0[{home}] {E_heat_max0[home]}"

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
        self.time_step += 1
        if res is None:
            self.T = self.T_next
        elif self.time_step < self.N:
            self.T = res["T"][:, self.time_step]

    def check_constraints(self, home, h, E_req_only):
        """
        From env.policy_to_rewardvar, check heat constraints.

        Inputs:
        home:
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
        # in home temperature larger than the minimum temperature
        if self.own_heat[home]:
            assert self.next_T(
                self.T, self.E_heat_min, self.T_out[h]
            )[1][home] >= self.T_LB_t[home] - 1e-2, "next_T < T_LB"

        # check if target temperature is met
        if not E_req_only:
            assert self.T_air[home] >= self.T_LB[home][h] - 1e-1, \
                f"T_air {self.T_air[home]} < T_LB {self.T_LB[home][h]}"

        else:
            # E_req_only is for baseline where we do not use the
            # flexibility and only target the middle temperature requirement
            if self.own_heat[home]:
                assert not (
                    self.T_air[home] > self.T_UB[home][h] + self.dT + 1e-1
                    or self.T_air[home] < self.T_LB[home][h] - 1e-1 - self.dT), \
                    "T_air > T_UB + dT or T_air < T_LB - dT"

        # positivity
        assert self.potential_E_flex()[home] >= 0, "potential_E_flex < 0"

    def actions_to_env_vars(
            self, res, l_flex, tot_l_fixed, E_flex=None
    ):
        """Get fixed/flexible heat consumption from current actions."""
        if E_flex is None:
            res_c = np.array([res[home]['c'] for home in range(self.n_homes)])
            self.E_flex = np.multiply(
                np.where(res_c > l_flex + tot_l_fixed, res_c - tot_l_fixed - l_flex, 0),
                self.own_heat
            )
        else:
            self.E_flex = E_flex
        self.tot_E = self.E_flex + self.E_heat_min

    def _next_E_heat(self, T_air_target, T_start, T_out_t, home=None, target='air_temp'):
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
        homes = list(range(na)) if home is None else [home]
        M = np.transpose(np.array([np.ones(na), T_start, np.ones(na) * T_out_t]))
        if target == 'air_temp':
            K = self.T_air_coeff[homes, 0: 3]
            p_heat = np.divide(
                T_air_target - np.sum(np.multiply(K, M), axis=1),
                self.T_air_coeff[homes, 4]
            )
        else:
            K = self.T_coeff[homes, 0: 3]
            p_heat = np.divide(
                T_air_target - np.sum(np.multiply(K, M), axis=1),
                self.T_coeff[homes, 4]
            )
        E_heat = np.multiply(
            np.where(p_heat > 0, p_heat * 1e-3 * 24 / self.H, 0),
            self.own_heat[homes]
        )

        return E_heat

    def _update_passive_active_vars(self, prm):
        # number of agents / households
        self.n_homes = prm["syst"]["n_homes" + self.ext]

        # current building mass temperatures
        self.T = np.ones(self.n_homes) * prm["heat"]["T0"]

        # heating coefficients for recursive description
        self.T_air_coeff = prm["heat"]["T_air_coeff" + self.ext]
        self.T_coeff = prm["heat"]["T_coeff" + self.ext]

        self.own_heat = prm["heat"]["own_heat" + self.ext]
