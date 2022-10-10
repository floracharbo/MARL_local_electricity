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
    apply_step:
        Update battery state for current actions.
    initial_processing:
        Get current available battery flexibility.
    k_losses:
        Get charge/discharge loss slope parameters between 2 mu points.
    check_errors_apply_step:
        From mu_manager.apply_step, check battery constraints.
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
        self.n_agents = prm['ntw']['n']

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
            self.p = passive_ext
            self.n_agents = prm["ntw"]["n" + self.p]

            # initial state
            self.store = {}
            for a in range(self.n_agents):
                self.store[a] = prm['bat']['store0' + passive_ext][a] \
                    if prm['bat']['own_EV'][a] else 0

            # storage at the start of current time step
            self.start_store = self.store.copy()

            # store0 is initial/final charge at the start and end
            # of the episode
            # cap is the maximum capacity of tbe battery
            # min_charge is the minimum state of charge / base load
            # that needs to always be available
            for e in ['store0', 'cap', 'min_charge']:
                self.__dict__[e] = prm['bat'][e + self.p]

    def update_step(self, res=None):
        """Update current object variables for new step."""
        self.i_step += 1
        self._current_batch_step()
        if res is not None and self.i_step < self.N:
            self.store = [res['store'][a][self.i_step]
                          for a in range(self.n_agents)]
        self.start_store = self.store.copy()

    def compute_bat_dem_agg(self, batch):
        """Compute bat_dem_agg, i.e. having all demand at start of trip."""
        for a in range(self.n_agents):
            batch[a]['bat_dem_agg'] = \
                [0 for _ in range(len(batch[a]['avail_EV']))]
            if self.own_EV[a]:
                start_trip, end_trip = [], []
                if batch[a]['avail_EV'][0] == 0:
                    start_trip.append(0)
                for t in range(len(batch[a]['avail_EV']) - 1):
                    if batch[a]['avail_EV'][t] == 0 \
                            and batch[a]['avail_EV'][t + 1] == 1:
                        end_trip.append(t)
                    if t > 0 and batch[a]['avail_EV'][t] == 0 \
                            and batch[a]['avail_EV'][t - 1] == 1:
                        start_trip.append(t)
                if len(start_trip) > len(end_trip) \
                        and batch[a]['avail_EV'][t] == 0:
                    end_trip.append(self.N - 1)
                for start, end in zip(start_trip, end_trip):
                    batch[a]['bat_dem_agg'][start] = \
                        sum(batch[a]['loads_EV'][start: end + 1])

        return batch

    def add_batch(self, batch):
        """Once batch data computed, update in battery data batch."""
        for e in self.batch_entries:
            e_batch = e if e in batch[0] else 'lds_EV'
            self.batch[e] = [batch[a][e_batch] for a in range(self.n_agents)]
        self._current_batch_step()

    def next_trip_details(self, start_h, date, a):
        """Get next trip's load requirements, time until trip, and end."""
        # next time the EV is on a trip
        iT = [i for i in range(len(self.batch['avail_EV'][a][start_h:])) if
              self.batch['avail_EV'][a][start_h + i] == 0]
        d_to_end = self.date_end - date
        h_end = start_h + d_to_end.days * 24 + (d_to_end.seconds) / 3600
        if len(iT) > 0 and start_h + iT[0] < h_end:
            # future trip that starts before end
            iT = int(start_h + iT[0])

            # next time the EV is back from the trip to the garage
            iG = [iT + i for i in range(len(self.batch['avail_EV'][a][iT:])) if
                  self.batch['avail_EV'][a][iT + i] == 1]
            iG = int(iG[0]) if len(iG) > 0 else len(self.batch['avail_EV'][a])
            deltaT = iT - start_h  # time until trip
            i_endtrip = int(min(iG, h_end))
            # EV load while on trip
            loads_T = np.sum(self.batch['loads_EV'][a][iT: i_endtrip])

            return loads_T, deltaT, i_endtrip
        else:
            return None, None, None

    def EV_tau(self, hour, date, a, store_a):
        """Compute EV_tau, i.e. how much charge is needed over in how long."""
        loads_T, deltaT, i_endtrip = self.next_trip_details(hour, date, a)
        if loads_T is not None and deltaT > 0:
            val = (loads_T - store_a) / deltaT
        else:
            val = - 1

        return val

    def _check_trip_feasible(
            self, loads_T, deltaT, bool_penalty, print_error, a, h
    ):
        if loads_T > self.cap[a] + 1e-2:
            # load during trip larger than whole
            bool_penalty[a] = True
            if print_error:
                print(f"a = {a}, "
                      f"load EV trip {loads_T} larger than"
                      f" cap, self.batch['loads_EV'][{a}] "
                      f"= {self.batch['loads_EV'][a]}")
        elif deltaT > 0 \
                and sum(self.batch['avail_EV'][a][0: h]) == 0 \
                and loads_T / deltaT \
                > self.store0[a] + self.c_max:
            bool_penalty[a] = True
            if print_error:
                print(f'a = {a}, loads_T {loads_T} '
                      f'deltaT {deltaT} not enough '
                      f'steps to fill what is needed')

        return bool_penalty

    def _get_list_future_trips(self, h, date_a, a, bool_penalty, print_error):
        trips = []
        end = False
        h_trip = h
        while not end:
            loads_T, deltaT, i_endtrip = self.next_trip_details(
                h_trip, date_a, a)
            if loads_T is None:
                end = True
            else:
                self._check_trip_feasible(
                    loads_T, deltaT, bool_penalty, print_error, a, h
                )
                trips.append([loads_T, deltaT, i_endtrip])
                date_a += datetime.timedelta(
                    hours=int(i_endtrip - h_trip))
                h_trip = i_endtrip

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
        avail_EV = [self.batch['avail_EV'][a][h] for a in range(self.n_agents)]
        bool_penalty = np.zeros(self.n_agents, dtype=bool)
        last_step = self._last_step(date)
        # regular initial minimum charge
        min_charge_t_0 = [
            self.store0[a] * avail_EV[a] if last_step else
            self.min_charge[a] * avail_EV[a] for a in range(self.n_agents)]

        # min_charge if need to charge up ahead of last step
        Creq = []
        for a in range(self.n_agents):
            date_a = copy.deepcopy(date)
            if not self.avail_EV[a]:
                # if EV not currently in garage
                Creq.append(0)
                continue

            # obtain all future trips
            trips = self._get_list_future_trips(
                h, date_a, a, bool_penalty, print_error
            )
            # trips[i] = [loads_T, deltaT, i_endtrip]

            # obtain required charge before each trip, starting with end
            final_i_endtrip = trips[-1][2] if len(trips) > 0 else h
            n_avail_until_end = sum(
                self.batch['avail_EV'][a][h_]
                for h_ in range(final_i_endtrip, self.N)
            )

            if len(trips) == 0:
                n_avail_until_end -= 1

            Creq.append(
                max(0, self.store0[a] - self.c_max * n_avail_until_end)
            )

            for it in range(len(trips)):
                loads_T, deltaT = trips[- (it + 1)][0:2]
                if it == len(trips) - 1:
                    deltaT -= 1
                # this is the required charge at the current step
                # if this is the most recent trip, or right after
                # the previous trip
                Creq[a] = max(0, Creq[a] + loads_T - deltaT * self.c_max)

        min_charge_t = [max(min_charge_t_0[a], Creq[a])
                        for a in range(self.n_agents)]

        self._check_min_charge_t_feasible(
            min_charge_t, h, date, bool_penalty, print_error, simulation
        )
        for a in range(self.n_agents):
            if h == 22 and avail_EV[a]:
                assert min_charge_t[a] >= self.store0[a] - self.c_max, \
                    f"h == 22 and min_charge_t {min_charge_t} " \
                    f"< {self.store0[a]} - {self.c_max}"

        self.min_charge_t = min_charge_t

        self.max_charge_t = \
            [self.store0[a] if last_step and self.avail_EV[a]
             else self.cap[a] for a in range(self.n_agents)]

        return bool_penalty

    def check_constraints(self, a, date, h):
        """
        From env.policy_to_rewardvar, check battery constraints.

        Inputs:
        a:
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
        bat_e_balance = self.charge[a] - self.discharge[a] \
            - self.loads_EV[a] - self.loss_dis[a] \
            - (self.store[a] - self.start_store[a])
        assert abs(bat_e_balance) < 1e-2, \
            f"a = {a}, battery energy balance sum = {bat_e_balance}"

        # initial and final storage level
        if date == self.date0:
            assert self.start_store[a] == self.store0[a], \
                f'start_store[{a}] {self.start_store[a]} not store0'

        if date == self.date_end - datetime.timedelta(hours=1) \
                and self.avail_EV[a]:
            assert self.store[a] >= self.store0[a] - 1e-2, \
                f"a = {a}, store end {self.store[a]} " \
                f"smaller than store0 {self.store0[a]}"

        # max storage level
        assert self.store[a] <= self.cap[a] + 1e-2, \
            f'store[{a}] {self.store[a]} larger than cap'

        if self.max_charge_t[a] is not None \
                and self.avail_EV[a]:
            assert self.store[a] <= self.max_charge_t[a] + 1e-2, \
                f'store[{a}] {self.store[a]} > max_charge_t[a] ' \
                f'{self.max_charge_t[a]} after flex, ' \
                f'last_step = {last_step}'

        # minimum storage level
        assert self.store[a] \
               >= self.SoCmin * self.cap[a] * self.avail_EV[a] - 1e-2, \
               f"store[{a}] {self.store[a]} " \
               f"smaller than SoCmin and no bool_penalty, " \
               f"availEV[a] = {self.avail_EV[a]}, " \
               f"charge[a] = {self.charge[a]}, " \
               f"c_max = {self.c_max}"

        assert \
            self.store[a] >= self.min_charge_t[a] * self.avail_EV[a] - 1e-2, \
            f'store[{a}] {self.store[a]} ' \
            f'smaller than min_charge_t[a] {self.min_charge_t[a]} ' \
            f'avail_EV[a] {self.avail_EV[a]}'

        # charge and discharge losses
        abs_loss_charge = \
            self.loss_ch[a] - ((self.charge[a] + self.loss_ch[a])
                               * (1 - self.eta_ch))
        assert abs(abs_loss_charge) <= 1e-2, \
            f"self.cap = {self.cap}, h = {h}, a = {a} " \
            f"sum loss charge = {abs_loss_charge}"

        abs_loss_charge = \
            self.loss_dis[a] - ((self.discharge[a] + self.loss_dis[a])
                                * (1 - self.eta_dis))
        assert abs(abs_loss_charge) <= 1e-2, \
            f"h = {h}, a = {a} sum loss charge = {abs_loss_charge}"

        # only charge and discharge if EV is available
        assert self.charge[a] <= self.avail_EV[a] * self.M, \
            'charge but EV not available and no bool_penalty'
        assert self.discharge[a] <= self.avail_EV[a] * self.M, \
            f'a = {a} discharge (else than EV cons) but ' \
            f'EV not available'

        # max charge rate
        assert self.charge[a] <= self.c_max + 1e-2, \
            f"charge {self.charge} > c_max {self.c_max}"

        # max discharge rate
        assert self.discharge[a] + self.loss_dis[a] + self.loads_EV[a] \
               <= self.d_max + 1e-2, \
               f'a = {a}, discharge[a] {self.discharge[a]} > self.d_max'

        # positivity
        for e in ['store', 'charge', 'discharge', 'loss_ch', 'loss_dis']:
            assert self.__dict__[e][a] >= - 1e-2, \
                f'a = {a} negative {e}[a] {self.__dic__[e][a]}'

    def apply_step(self, res):
        """Update battery state for current actions."""
        for e in ['store', 'charge', 'discharge', 'loss_ch',
                  'loss_dis', 'store_out_tot', 'discharge_tot']:
            self.__dict__[e] = [None for a in range(self.n_agents)]
        for a in range(self.n_agents):
            self.store[a] = self.start_store[a] \
                + res[a]['ds'] - self.loads_EV[a]
            if self.store[a] < self.min_charge_t[a] - 1e-3:
                print(f"a {a} store[{a}] {self.store[a]} "
                      f"self.start_store[a] {self.start_store[a]} "
                      f"res[{a}['ds'] {res[a]['ds']} "
                      f"self.loads_EV[a] {self.loads_EV[a]} "
                      f"self.min_charge_t[a] {self.min_charge_t[a]}")
            self.charge[a] = res[a]['ds'] if res[a]['ds'] > 0 else 0
            self.discharge[a] = - res[a]['ds'] * self.eta_dis \
                if res[a]['ds'] < 0 else 0
            self.loss_ch[a] = res[a]['l_ch']
            self.loss_dis[a] = res[a]['l_dis']
            self.store_out_tot[a] = self.discharge[a] \
                + self.loss_dis[a] + self.loads_EV[a]
            self.discharge_tot[a] = self.discharge[a] / self.eta_dis \
                + self.loads_EV[a]

    def initial_processing(self):
        """Get current available battery flexibility."""
        # storage available above minimum charge that can be discharged
        s_avail_dis = np.array(
            [min(
                max(self.start_store[a] - self.min_charge_t[a], 0),
                self.d_max
            ) * self.avail_EV[a] for a in range(self.n_agents)]
        )

        # how much i need to add to store
        s_add_0 = np.array(
            [max(self.min_charge_t[a] - self.start_store[a], 0)
             * self.avail_EV[a] for a in range(self.n_agents)]
        )

        # how much i need to remove from store
        s_remove_0 = np.array(
            [max(self.start_store[a] - self.max_charge_t[a], 0)
             * self.avail_EV[a] for a in range(self.n_agents)]
        )

        # how much can I charge it by rel. to current level
        C_avail = np.array(
            [min(self.c_max, self.max_charge_t[a] - self.start_store[a])
             * self.avail_EV[a] for a in range(self.n_agents)]
        )

        return s_avail_dis, s_add_0, s_remove_0, C_avail

    def k_losses(self, a, k, mu_prev, mu_next):
        """Get charge/discharge loss slope parameters between 2 mu points."""
        ds_start = k[a]['ds'][-1][0] * mu_prev + k[a]['ds'][-1][1]
        ds_end = k[a]['ds'][-1][0] * mu_next + k[a]['ds'][-1][1]
        loss, a_loss, b_loss = [{} for _ in range(3)]
        loss['ch'] = [0 if ds < 0
                      else ds / self.eta_ch * (1 - self.eta_ch)
                      for ds in [ds_start, ds_end]]
        loss['dis'] = [- ds * (1 - self.eta_dis) if ds < 0
                       else 0 for ds in [ds_start, ds_end]]
        for e in ['ch', 'dis']:
            a_loss[e] = (loss[e][1] - loss[e][0]) / (mu_next - mu_prev)
            b_loss[e] = loss[e][1] - a_loss[e] * mu_next
            k[a]['l_' + e].append([a_loss[e], b_loss[e]])

        return k

    def check_errors_apply_step(self, as_, bool_penalty, mu_action, res):
        """From mu_manager.apply_step, check battery constraints."""
        for a in as_:
            # bat
            if self.min_charge_t[a] - self.start_store[a] > self.c_max + 1e-2:
                print(f"self.min_charge_t[{a}] = {self.min_charge_t[a]},"
                      f"start_store[a] = {self.start_store[a]}")
                bool_penalty[a] = True

            if abs(self.loss_ch[a]
                   - ((self.charge[a] + self.loss_ch[a])
                      * (1 - self.eta_ch))) > 1e-2:
                print(f'in update_vars_from_mu loss_ch[a] = '
                      f'{self.loss_ch[a]} charge[a] = {self.charge[a]}')
                bool_penalty[a] = True

            # with discharge / loss_dis ratio
            if abs(self.loss_dis[a]
                   - ((self.discharge[a] + self.loss_dis[a])
                      * (1 - self.eta_dis))) > 1e-2:
                print(f'in update_vars_from_mu loss_dis[{a}] = '
                      f'{self.loss_dis[a]} '
                      f'discharge[a] = {self.discharge[a]}')
                bool_penalty[a] = True

            # discharge rate
            if self.discharge[a] + self.loads_EV[a] > self.d_max + 1e-2:
                print(f"discharge[{a}] {self.discharge[a]} > "
                      f"self.d_max {self.d_max} all from mu flex")
                bool_penalty[a] = True

            if not self.avail_EV[a] and self.loads_EV[a] > 1e-2 \
                    and self.loads_EV[a] > self.start_store[a] + 1e-2:
                print(f"self.loads_EV[{a}] = {self.loads_EV[a]}, "
                      f"self.start_store[a] = {self.start_store[a]}")
                bool_penalty[a] = True

            if self.avail_EV[a] == 0 and res[a]['ds'] > 0:
                print('in update mu action dch > 0, EVavail = 0')
                bool_penalty[a] = True

            if self.max_charge_t[a] is not None \
                    and self.store[a] > self.max_charge_t[a] + 1e-2:
                print(f'self.max_charge_t[{a}] = {self.max_charge_t[a]} '
                      f'store[a] = {self.store[a]}')
                print(f'mu_action[a] = {mu_action[a]}')
                bool_penalty[a] = True

        if not type(self.store[0]) in [float, np.float64]:
            print('not type(store[0]) in [float, np.float64]')
            bool_penalty[a] = True

        return bool_penalty

    def _current_batch_step(self):
        for e in self.batch_entries:
            self.__dict__[e] = [self.batch[e][a][self.i_step]
                                for a in range(self.n_agents)]

    def _last_step(self, date):
        return date == self.date_end - datetime.timedelta(hours=1)

    def _check_first_time_step_feasibility(
            self, h, date, bool_penalty, print_error
    ):
        for a in range(self.n_agents):
            # check if opportunity to charge before trip > 37.5
            if h == 0 and not self.batch['avail_EV'][a][0]:
                loads_T, deltaT, i_endtrip = self.next_trip_details(h, date, a)
                if loads_T > self.store0[a]:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[a] = True
                    error_message = \
                        f'a = {a}, trip {loads_T} larger than ' \
                        f'initial charge - straight away not available'
                    self._print_error(error_message, print_error)
                if sum(self.batch['avail_EV'][a][0:23]) == 0 \
                        and sum(self.batch['loads_EV'][a][0:23]) \
                        > self.c_max + 1e-2:
                    bool_penalty[a] = True
                    error_message = \
                        f'a = {a}, only last step available, ' \
                        'not enough to go back to initial charge ' \
                        'at the last time step'
                    self._print_error(error_message, print_error)
                loads_T_next, deltaT_next, i_endtrip_next = \
                    self.next_trip_details(
                        deltaT, date + datetime.timedelta(hours=deltaT), a)
                if deltaT_next > 0 \
                        and loads_T_next - (self.store0[a] - loads_T) \
                        < self.c_max / deltaT_next:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[a] = True
                    error_message = f'a = {a}, directly initial trip ' \
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

        for a in range(self.n_agents):
            # check if any hourly load is larger than d_max
            if any(self.batch['loads_EV'][a][h] > self.d_max + 1e-2
                   for h in range(self.N)):
                # you would have to break constraints to meet demand
                bool_penalty[a] = True
                self._print_error(
                    f'a = {a}, load EV larger than d_max',
                    print_error
                )

            if min_charge_t[a] > self.cap[a] + 1e-2:
                bool_penalty[a] = True  # min_charge_t larger than total cap
                error_message = f'a = {a}, min_charge_t {min_charge_t[a]} ' \
                    'larger than cap'
                self._print_error(error_message, print_error)
            if min_charge_t[a] > self.store0[a] \
                    - sum(self.batch['loads_EV'][a][0:h]) + (
                    sum(self.batch['loads_EV'][a][0:h]) + 1) \
                    * self.c_max + 1e-3:
                bool_penalty[a] = True
                error_message = f'a = {a}, min_charge_t {min_charge_t[a]} ' \
                    'larger what car can be charged to'
                self._print_error(error_message, print_error)

            if simulation \
                    and min_charge_t[a] > self.store[a] + self.c_max + 1e-3:
                bool_penalty[a] = True
                error_message = \
                    f"date {date} h {h} " \
                    f"min_charge_t[{a}] {min_charge_t[a]} " \
                    f"> self.store[{a}] {self.store[a]} " \
                    f"+ self.c_max {self.c_max}"
                self._print_error(error_message, print_error)

            elif not simulation \
                    and h > 0 \
                    and sum(self.batch['avail_EV'][a][0: h]) == 0:
                # the EV has not been available at home to recharge until now
                store_t_a = self.store0[a] \
                    - sum(self.batch['loads_EV'][a][0: h])
                if min_charge_t[a] > store_t_a + self.c_max + 1e-3:
                    bool_penalty[a] = True

    def _print_error(self, error_message, print_error):
        if print_error:
            print(error_message)

    def check_feasible_bat(self, prm, ntw, p, bat, syst):
        """Check charging constraints for proposed data batch."""
        feasible = np.ones(ntw['n' + p], dtype=bool)
        for a in range(ntw['n' + p]):
            if bat['d_max'] < np.max(bat['batch_loads_EV'][a]):
                feasible[a] = False
                print("bat['d_max'] < np.max(bat['batch_loads_EV'][a])")
                for t in range(len(bat['batch_loads_EV'][a])):
                    if bat['batch_loads_EV'][a, t] > bat['d_max']:
                        bat['batch_loads_EV'][a, t] = bat['d_max']

        t = 0

        self.reset(prm)
        while all(feasible) and t < syst['N']:
            date = self.date0 + datetime.timedelta(hours=t)
            bool_penalty = self.min_max_charge_t(
                t, date, print_error=False,
                simulation=False)
            feasible[bool_penalty] = False
            self.update_step()
            t += 1

        return feasible
