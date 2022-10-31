#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 16:40:04 2021.

@author: Flora Charbonnier
"""
import copy
import datetime

import numpy as np


class Battery:
    """
    EV batteries in homes.

    Public methods:
    reset:
        Reset object for new episode.
    update_step:
        Update current object variables for new step.
    compute_bat_dem_agg:
        Compute bat_dem_agg, i.e. having all demand at start of trip.
    add_batch:
        Once batch data computed, update in battery data batch.
    next_trip_details:
        Get next trip's load requirements, time until trip, and end.
    EV_tau:
        Compute EV_tau, i.e. how much charge is needed over in how long.
    min_max_charge_t:
        For current time step, define minimum/maximum charge.
    check_constraints:
        From env.policy_to_rewardvar, check battery constraints.
    actions_to_env_vars:
        Update battery state for current actions.
    initial_processing:
        Get current available battery flexibility.
    k_losses:
        Get charge/discharge loss slope parameters between 2 action points.
    check_errors_rl_actions_to_env_vars:
        From action_translator.actions_to_env_vars, check battery constraints.
    """

    def __init__(self, prm, passive_ext=None):
        """
        Initialise Battery object.

        inputs:
        prm:
            input parameters
        passive_ext:
            are the current agents passive? '' is not, 'P' if yes
        """
        # number of agents / households
        self.n_homes = prm['ntw']['n']

        # very large number for enforcing constraints
        self.M = prm['syst']['M']

        # import input parameters
        for e in ['own_EV', 'dep', 'c_max', 'd_max', 'eta_ch',
                  'eta_ch', 'eta_dis', 'SoCmin']:
            self.__dict__[e] = prm['bat'][e]

        # total number of time steps
        self.N = prm['syst']['N']

        # date of end of episode - updated from update_date() is env
        self.date_end = None
        # start date episode -- updated from update_date in env
        self.date0 = None

        # batch data entries
        self.batch_entries = ['loads_EV', 'avail_EV']

        # episode specific parameters
        self.reset(prm, passive_ext)

    def reset(self, prm, passive_ext=None):
        """Reset object for new episode."""
        # time step
        self.i_step = 0

        if passive_ext is not None:
            # passive extension
            self.passive_ext = passive_ext
            self.n_homes = prm["ntw"]["n" + self.passive_ext]

            # initial state
            self.store = {}
            for home in range(self.n_homes):
                self.store[home] = prm['bat']['store0' + passive_ext][home] \
                    if prm['bat']['own_EV'][home] else 0

            # storage at the start of current time step
            self.start_store = self.store.copy()

            # store0 is initial/final charge at the start and end
            # of the episode
            # cap is the maximum capacity of tbe battery
            # min_charge is the minimum state of charge / base load
            # that needs to always be available
            for e in ['store0', 'cap', 'min_charge']:
                self.__dict__[e] = prm['bat'][e + self.passive_ext]

    def update_step(self, res=None):
        """Update current object variables for new step."""
        self.i_step += 1
        self._current_batch_step()
        if res is not None and self.i_step < self.N:
            self.store = [res['store'][home][self.i_step]
                          for home in range(self.n_homes)]
        self.start_store = self.store.copy()

    def compute_bat_dem_agg(
            self,
            batch: dict
    ) -> dict:
        """Compute bat_dem_agg, i.e. having all demand at start of trip."""
        for home in range(self.n_homes):
            batch[home]['bat_dem_agg'] = \
                [0 for _ in range(len(batch[home]['avail_EV']))]
            if self.own_EV[home]:
                start_trip, end_trip = [], []
                if batch[home]['avail_EV'][0] == 0:
                    start_trip.append(0)
                for time in range(len(batch[home]['avail_EV']) - 1):
                    if batch[home]['avail_EV'][time] == 0 \
                            and batch[home]['avail_EV'][time + 1] == 1:
                        end_trip.append(time)
                    if time > 0 and batch[home]['avail_EV'][time] == 0 \
                            and batch[home]['avail_EV'][time - 1] == 1:
                        start_trip.append(time)
                if len(start_trip) > len(end_trip) \
                        and batch[home]['avail_EV'][time] == 0:
                    end_trip.append(self.N - 1)
                for start, end in zip(start_trip, end_trip):
                    batch[home]['bat_dem_agg'][start] = \
                        sum(batch[home]['loads_EV'][start: end + 1])

        return batch

    def add_batch(self, batch):
        """Once batch data computed, update in battery data batch."""
        for e in self.batch_entries:
            e_batch = e if e in batch[0] else 'lds_EV'
            self.batch[e] = [batch[home][e_batch] for home in range(self.n_homes)]
        self._current_batch_step()

    def next_trip_details(self, start_h, date, home):
        """Get next trip's load requirements, time until trip, and end."""
        # next time the EV is on a trip
        iT = np.asarray(~np.array(self.batch['avail_EV'][home][start_h: self.N + 1], dtype=bool)).nonzero()[0]
        d_to_end = self.date_end - date
        h_end = start_h + d_to_end.days * 24 + (d_to_end.seconds) / 3600
        if len(iT) > 0 and start_h + iT[0] < h_end:
            # future trip that starts before end
            iT = int(start_h + iT[0])

            # next time the EV is back from the trip to the garage
            iG = iT + np.asarray(self.batch['avail_EV'][home][iT: self.N + 1]).nonzero()[0]
            iG = int(iG[0]) if len(iG) > 0 else len(self.batch['avail_EV'][home])
            deltaT = iT - start_h  # time until trip
            i_endtrip = int(min(iG, h_end))
            # EV load while on trip
            loads_T = np.sum(self.batch['loads_EV'][home][iT: i_endtrip])

            return loads_T, deltaT, i_endtrip
        else:
            return None, None, None

    def EV_tau(self, hour, date, home, store_a):
        """Compute EV_tau, i.e. how much charge is needed over in how long."""
        loads_T, deltaT, i_endtrip = self.next_trip_details(hour, date, home)
        if loads_T is not None and deltaT > 0:
            val = (loads_T - store_a) / deltaT
        else:
            val = - 1

        return val

    def _check_trip_feasible(
            self, loads_T, deltaT, bool_penalty, print_error, home, h
    ):
        if loads_T > self.cap[home] + 1e-2:
            # load during trip larger than whole
            bool_penalty[home] = True
            if print_error:
                print(f"home = {home}, h = {h} deltaT {deltaT} "
                      f"load EV trip {loads_T} larger than"
                      f" cap, self.batch['loads_EV'][{home}] "
                      f"= {self.batch['loads_EV'][home]}")
        elif deltaT > 0 \
                and sum(self.batch['avail_EV'][home][0: h]) == 0 \
                and loads_T / deltaT \
                > self.store0[home] + self.c_max:
            bool_penalty[home] = True
            if print_error:
                print(f'home = {home}, loads_T {loads_T} '
                      f'deltaT {deltaT} not enough '
                      f'steps to fill what is needed')

        return bool_penalty

    def _get_list_future_trips(self, h, date_a, home, bool_penalty, print_error):
        trips = []
        end = False
        h_trip = h
        while not end:
            loads_T, deltaT, i_end_trip = self.next_trip_details(h_trip, date_a, home)
            if loads_T is None or h_trip + deltaT > self.N:
                end = True
            else:
                self._check_trip_feasible(
                    loads_T, deltaT, bool_penalty, print_error, home, h
                )
                trips.append([loads_T, deltaT, i_end_trip])
                date_a += datetime.timedelta(
                    hours=int(i_end_trip - h_trip))
                h_trip = i_end_trip

        return trips

    def min_max_charge_t(self, h=None, date=None, print_error=True,
                         simulation=True):
        """
        For current time step, define minimum/maximum charge.

        simulation:
            yes if the self.store value if updated in
            current simulation. false if we are just testing the
            feasibility of a generic batch of data without updating
            the storage levels
        """
        if h is None:
            h = self.i_step
        if date is None:
            date = self.date0 + datetime.timedelta(hours=h)
        avail_EV = [self.batch['avail_EV'][home][h] for home in range(self.n_homes)]
        bool_penalty = np.zeros(self.n_homes, dtype=bool)
        last_step = self._last_step(date)
        # regular initial minimum charge
        min_charge_t_0 = np.where(last_step, self.store0, self.min_charge) * avail_EV

        # min_charge if need to charge up ahead of last step
        Creq = []
        for home in range(self.n_homes):
            if not self.avail_EV[home]:
                # if EV not currently in garage
                Creq.append(0)
                continue

            # obtain all future trips
            trips = self._get_list_future_trips(
                h, date, home, bool_penalty, print_error
            )
            # trips[i] = [loads_T, deltaT, i_endtrip]

            # obtain required charge before each trip, starting with end
            final_i_endtrip = trips[-1][2] if len(trips) > 0 else h
            n_avail_until_end = sum(self.batch['avail_EV'][home][final_i_endtrip: self.N])

            if len(trips) == 0:
                n_avail_until_end -= 1

            Creq.append(
                max(0, self.store0[home] - self.c_max * n_avail_until_end)
            )

            for it in range(len(trips)):
                loads_T, deltaT = trips[- (it + 1)][0:2]
                if it == len(trips) - 1:
                    deltaT -= 1
                # this is the required charge at the current step
                # if this is the most recent trip, or right after
                # the previous trip
                Creq[home] = max(0, Creq[home] + loads_T - deltaT * self.c_max)

        min_charge_t = np.maximum(min_charge_t_0, Creq)

        self._check_min_charge_t_feasible(
            min_charge_t, h, date, bool_penalty, print_error, simulation
        )
        for home in range(self.n_homes):
            if h == 22 and avail_EV[home]:
                assert min_charge_t[home] >= self.store0[home] - self.c_max, \
                    f"h == 22 and min_charge_t {min_charge_t} " \
                    f"< {self.store0[home]} - {self.c_max}"

        self.min_charge_t = min_charge_t

        self.max_charge_t = np.where(last_step and self.avail_EV, self.store0, self.cap)

        return bool_penalty

    def check_constraints(self, home, date, h):
        """
        From env.policy_to_rewardvar, check battery constraints.

        Inputs:
        home:
            index relative to agent
        h:
            time step
        date:
            current date
        bool_penalty:
            the array of booleans keeping track of whether
            each agent breaks constraints
        """
        last_step = self._last_step(date)

        # battery energy balance
        bat_e_balance = self.charge[home] - self.discharge[home] \
            - self.loads_EV[home] - self.loss_dis[home] \
            - (self.store[home] - self.start_store[home])
        assert abs(bat_e_balance) < 1e-2, \
            f"home = {home}, battery energy balance sum = {bat_e_balance}"

        # initial and final storage level
        if date == self.date0:
            assert self.start_store[home] == self.store0[home], \
                f'start_store[{home}] {self.start_store[home]} not store0'

        if date == self.date_end - datetime.timedelta(hours=1) \
                and self.avail_EV[home]:
            assert self.store[home] >= self.store0[home] - 1e-2, \
                f"home = {home}, store end {self.store[home]} " \
                f"smaller than store0 {self.store0[home]}"

        # max storage level
        assert self.store[home] <= self.cap[home] + 1e-2, \
            f'store[{home}] {self.store[home]} larger than cap'

        if self.max_charge_t[home] is not None \
                and self.avail_EV[home]:
            assert self.store[home] <= self.max_charge_t[home] + 1e-2, \
                f'store[{home}] {self.store[home]} > max_charge_t[home] ' \
                f'{self.max_charge_t[home]} after flex, ' \
                f'last_step = {last_step}'

        # minimum storage level
        assert self.store[home] \
               >= self.SoCmin * self.cap[home] * self.avail_EV[home] - 1e-2, \
               f"store[{home}] {self.store[home]} " \
               f"smaller than SoCmin and no bool_penalty, " \
               f"availEV[home] = {self.avail_EV[home]}, " \
               f"charge[home] = {self.charge[home]}, " \
               f"c_max = {self.c_max}"

        assert \
            self.store[home] >= self.min_charge_t[home] * self.avail_EV[home] - 1e-2, \
            f'store[{home}] {self.store[home]} ' \
            f'smaller than min_charge_t[home] {self.min_charge_t[home]} ' \
            f'avail_EV[home] {self.avail_EV[home]}'

        # charge and discharge losses
        abs_loss_charge = \
            self.loss_ch[home] - (
                (self.charge[home] + self.loss_ch[home])
                * (1 - self.eta_ch)
            )
        assert abs(abs_loss_charge) <= 1e-2, \
            f"self.cap = {self.cap}, h = {h}, home = {home} " \
            f"sum loss charge = {abs_loss_charge}"

        abs_loss_charge = \
            self.loss_dis[home] - (
                (self.discharge[home] + self.loss_dis[home])
                * (1 - self.eta_dis)
            )
        assert abs(abs_loss_charge) <= 1e-2, \
            f"h = {h}, home = {home} sum loss charge = {abs_loss_charge}"

        # only charge and discharge if EV is available
        assert self.charge[home] <= self.avail_EV[home] * self.M, \
            'charge but EV not available and no bool_penalty'
        assert self.discharge[home] <= self.avail_EV[home] * self.M, \
            f'home = {home} discharge (else than EV cons) but ' \
            f'EV not available'

        # max charge rate
        assert self.charge[home] <= self.c_max + 1e-2, \
            f"charge {self.charge} > c_max {self.c_max}"

        # max discharge rate
        assert self.discharge[home] + self.loss_dis[home] + self.loads_EV[home] \
               <= self.d_max + 1e-2, \
               f'home = {home}, discharge[home] {self.discharge[home]} > self.d_max'

        # positivity
        for e in ['store', 'charge', 'discharge', 'loss_ch', 'loss_dis']:
            assert self.__dict__[e][home] >= - 1e-2, \
                f'home = {home} negative {e}[home] {self.__dic__[e][home]}'

    def actions_to_env_vars(self, res):
        """Update battery state for current actions."""
        for e in ['store', 'charge', 'discharge', 'loss_ch',
                  'loss_dis', 'store_out_tot', 'discharge_tot']:
            self.__dict__[e] = [None for home in range(self.n_homes)]
        for home in range(self.n_homes):
            self.store[home] = self.start_store[home] \
                + res[home]['ds'] - self.loads_EV[home]
            if self.store[home] < self.min_charge_t[home] - 1e-3:
                print(f"home {home} store[{home}] {self.store[home]} "
                      f"self.start_store[home] {self.start_store[home]} "
                      f"res[{home}['ds'] {res[home]['ds']} "
                      f"self.loads_EV[home] {self.loads_EV[home]} "
                      f"self.min_charge_t[home] {self.min_charge_t[home]}")
            self.charge[home] = res[home]['ds'] if res[home]['ds'] > 0 else 0
            self.discharge[home] = - res[home]['ds'] * self.eta_dis \
                if res[home]['ds'] < 0 else 0
            self.loss_ch[home] = res[home]['l_ch']
            self.loss_dis[home] = res[home]['l_dis']
            self.store_out_tot[home] = self.discharge[home] \
                + self.loss_dis[home] + self.loads_EV[home]
            self.discharge_tot[home] = self.discharge[home] / self.eta_dis \
                + self.loads_EV[home]

    def initial_processing(self):
        """Get current available battery flexibility."""
        # storage available above minimum charge that can be discharged
        s_avail_dis = np.array(
            [min(
                max(self.start_store[home] - self.min_charge_t[home], 0),
                self.d_max
            ) * self.avail_EV[home] for home in range(self.n_homes)]
        )

        # how much i need to add to store
        s_add_0 = np.array(
            [max(self.min_charge_t[home] - self.start_store[home], 0)
             * self.avail_EV[home] for home in range(self.n_homes)]
        )

        # how much i need to remove from store
        s_remove_0 = np.array(
            [max(self.start_store[home] - self.max_charge_t[home], 0)
             * self.avail_EV[home] for home in range(self.n_homes)]
        )

        # how much can I charge it by rel. to current level
        potential_charge = np.array(
            [min(self.c_max, self.max_charge_t[home] - self.start_store[home])
             * self.avail_EV[home] for home in range(self.n_homes)]
        )

        return s_avail_dis, s_add_0, s_remove_0, potential_charge

    def k_losses(self, home, k, action_prev, action_next):
        """Get charge/discharge loss slope parameters between 2 action points."""
        ds_start = k[home]['ds'][-1][0] * action_prev + k[home]['ds'][-1][1]
        ds_end = k[home]['ds'][-1][0] * action_next + k[home]['ds'][-1][1]
        loss, a_loss, b_loss = [{} for _ in range(3)]
        loss['ch'] = [0 if ds < 0
                      else ds / self.eta_ch * (1 - self.eta_ch)
                      for ds in [ds_start, ds_end]]
        loss['dis'] = [- ds * (1 - self.eta_dis) if ds < 0
                       else 0 for ds in [ds_start, ds_end]]
        for e in ['ch', 'dis']:
            a_loss[e] = (loss[e][1] - loss[e][0]) / (action_next - action_prev)
            b_loss[e] = loss[e][1] - a_loss[e] * action_next
            k[home]['l_' + e].append([a_loss[e], b_loss[e]])

        return k

    def check_errors_apply_step(self, homes, bool_penalty, action, res):
        """From action_translator.actions_to_env_vars, check battery constraints."""
        for home in homes:
            # bat
            if self.min_charge_t[home] - self.start_store[home] > self.c_max + 1e-2:
                print(f"self.min_charge_t[{home}] = {self.min_charge_t[home]},"
                      f"start_store[home] = {self.start_store[home]}")
                bool_penalty[home] = True

            if abs(self.loss_ch[home]
                   - ((self.charge[home] + self.loss_ch[home])
                      * (1 - self.eta_ch))) > 1e-2:
                print(f'in actions_to_env_vars loss_ch[home] = '
                      f'{self.loss_ch[home]} charge[home] = {self.charge[home]}')
                bool_penalty[home] = True

            # with discharge / loss_dis ratio
            if abs(self.loss_dis[home]
                   - ((self.discharge[home] + self.loss_dis[home])
                      * (1 - self.eta_dis))) > 1e-2:
                print(f'in actions_to_env_vars loss_dis[{home}] = '
                      f'{self.loss_dis[home]} '
                      f'discharge[home] = {self.discharge[home]}')
                bool_penalty[home] = True

            # discharge rate
            if self.discharge[home] + self.loads_EV[home] > self.d_max + 1e-2:
                print(f"discharge[{home}] {self.discharge[home]} > "
                      f"self.d_max {self.d_max}")
                bool_penalty[home] = True

            if not self.avail_EV[home] and self.loads_EV[home] > 1e-2 \
                    and self.loads_EV[home] > self.start_store[home] + 1e-2:
                print(f"self.loads_EV[{home}] = {self.loads_EV[home]}, "
                      f"self.start_store[home] = {self.start_store[home]}")
                bool_penalty[home] = True

            if self.avail_EV[home] == 0 and res[home]['ds'] > 0:
                print('in update action dch > 0, EVavail = 0')
                bool_penalty[home] = True

            if self.max_charge_t[home] is not None \
                    and self.store[home] > self.max_charge_t[home] + 1e-2:
                print(f'self.max_charge_t[{home}] = {self.max_charge_t[home]} '
                      f'store[home] = {self.store[home]}')
                print(f'action[home] = {action[home]}')
                bool_penalty[home] = True

        if not type(self.store[0]) in [float, np.float64]:
            print('not type(store[0]) in [float, np.float64]')
            bool_penalty[home] = True

        return bool_penalty

    def _current_batch_step(self):
        for e in self.batch_entries:
            self.__dict__[e] = [self.batch[e][home][self.i_step]
                                for home in range(self.n_homes)]

    def _last_step(self, date):
        return date == self.date_end - datetime.timedelta(hours=1)

    def _check_first_time_step_feasibility(
            self, h, date, bool_penalty, print_error
    ):
        for home in range(self.n_homes):
            # check if opportunity to charge before trip > 37.5
            if h == 0 and not self.batch['avail_EV'][home][0]:
                loads_T, deltaT, _ = self.next_trip_details(h, date, home)
                if loads_T > self.store0[home]:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[home] = True
                    error_message = \
                        f'home = {home}, trip {loads_T} larger than ' \
                        f'initial charge - straight away not available'
                    self._print_error(error_message, print_error)
                if sum(self.batch['avail_EV'][home][0:23]) == 0 \
                        and sum(self.batch['loads_EV'][home][0:23]) \
                        > self.c_max + 1e-2:
                    bool_penalty[home] = True
                    error_message = \
                        f'home = {home}, only last step available, ' \
                        'not enough to go back to initial charge ' \
                        'at the last time step'
                    self._print_error(error_message, print_error)
                loads_T_next, deltaT_next, _ = \
                    self.next_trip_details(
                        deltaT, date + datetime.timedelta(hours=deltaT), home)
                if deltaT_next > 0 \
                        and loads_T_next - (self.store0[home] - loads_T) \
                        < self.c_max / deltaT_next:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[home] = True
                    error_message = f'home = {home}, directly initial trip ' \
                                    f'{loads_T} then {loads_T_next} ' \
                                    f'with only {deltaT_next} to charge'
                    self._print_error(error_message, print_error)

        return bool_penalty

    def _check_min_charge_t_feasible(
            self, min_charge_t, h, date, bool_penalty, print_error, simulation
    ):
        bool_penalty = self._check_first_time_step_feasibility(
            h, date, bool_penalty, print_error
        )

        for home in range(self.n_homes):
            # check if any hourly load is larger than d_max
            if any(self.batch['loads_EV'][home][h] > self.d_max + 1e-2
                   for h in range(self.N)):
                # you would have to break constraints to meet demand
                bool_penalty[home] = True
                self._print_error(
                    f'home = {home}, load EV larger than d_max',
                    print_error
                )

            if min_charge_t[home] > self.cap[home] + 1e-2:
                bool_penalty[home] = True  # min_charge_t larger than total cap
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger than cap'
                self._print_error(error_message, print_error)
            if min_charge_t[home] > self.store0[home] \
                    - sum(self.batch['loads_EV'][home][0:h]) + (
                    sum(self.batch['loads_EV'][home][0:h]) + 1) \
                    * self.c_max + 1e-3:
                bool_penalty[home] = True
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger what car can be charged to'
                self._print_error(error_message, print_error)

            if simulation \
                    and min_charge_t[home] > self.store[home] + self.c_max + 1e-3:
                bool_penalty[home] = True
                error_message = \
                    f"date {date} h {h} " \
                    f"min_charge_t[{home}] {min_charge_t[home]} " \
                    f"> self.store[{home}] {self.store[home]} " \
                    f"+ self.c_max {self.c_max}"
                self._print_error(error_message, print_error)

            elif not simulation \
                    and h > 0 \
                    and sum(self.batch['avail_EV'][home][0: h]) == 0:
                # the EV has not been available at home to recharge until now
                store_t_a = self.store0[home] \
                    - sum(self.batch['loads_EV'][home][0: h])
                if min_charge_t[home] > store_t_a + self.c_max + 1e-3:
                    bool_penalty[home] = True

    def _print_error(self, error_message, print_error):
        if print_error:
            print(error_message)

    def check_feasible_bat(self, prm, ntw, passive_ext, bat, syst):
        """Check charging constraints for proposed data batch."""
        feasible = np.ones(ntw['n' + passive_ext], dtype=bool)
        for home in range(ntw['n' + passive_ext]):
            if bat['d_max'] < np.max(bat['batch_loads_EV'][home]):
                feasible[home] = False
                print("bat['d_max'] < np.max(bat['batch_loads_EV'][home])")
                for time in range(len(bat['batch_loads_EV'][home])):
                    if bat['batch_loads_EV'][home, time] > bat['d_max']:
                        bat['batch_loads_EV'][home, time] = bat['d_max']

        time = 0

        self.reset(prm)
        while all(feasible) and time < syst['N']:
            date = self.date0 + datetime.timedelta(hours=time)
            bool_penalty = self.min_max_charge_t(
                time, date, print_error=False,
                simulation=False)
            feasible[bool_penalty] = False
            self.update_step()
            time += 1

        return feasible
