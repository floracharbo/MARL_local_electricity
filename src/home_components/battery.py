#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 16:40:04 2021.

@author: Flora Charbonnier
"""
import datetime

import numpy as np

from src.utilities.userdeftools import _calculate_reactive_power


class Battery:
    """
    EV batteries in homes.

    Public methods:
    reset:
        Reset object for new episode.
    update_step:
        Update current object variables for new step.
    compute_battery_demand_aggregated_at_start_of_trip:
        Compute bat_dem_agg, i.e. having all demand at start of trip.
    add_batch:
        Once batch data computed, update in battery data batch.
    next_trip_details:
        Get next trip's load requirements, time until trip, and end.
    car_tau:
        Compute car_tau, i.e. how much charge is needed over in how long.
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

    def __init__(self, prm, passive=False):
        """
        Initialise Battery object.

        inputs:
        prm:
            input parameters
        passive_ext:
            are the current agents passive? '' is not, 'P' if yes
        """
        # very large number for enforcing constraints
        self.set_passive_active(passive, prm)

        self.pf_flexible_homes = prm['grd']['pf_flexible_homes']

        for info in ['M', 'N', 'dt']:
            setattr(self, info, prm['syst'][info])

        for info in [
            'dep', 'c_max', 'd_max', 'eta_ch', 'eta_ch', 'eta_dis', 'SoCmin',
        ]:
            setattr(self, info, prm['car'][info])

        # date of end of episode - updated from update_date() is env
        self.date_end = None
        # start date episode -- updated from update_date in env
        self.date0 = None

        # batch data entries
        self.batch_entries = ['loads_car', 'avail_car']

        # episode specific parameters
        self.reset(prm)

    def reset(self, prm):
        """Reset object for new episode."""
        # time step
        self.time_step = 0

        # initial state
        self.store = {}
        for home in range(self.n_homes):
            self.store[home] = prm['car']['store0' + self.passive_ext][home] \
                if prm['car']['own_car' + self.passive_ext][home] else 0

        # storage at the start of current time step
        self.start_store = self.store.copy()

    def update_step(self, res=None, time_step=None, implement=True):
        """Update current object variables for new step."""
        self.prev_time_step = self.time_step
        self.prev_start_store = self.start_store.copy()

        if time_step is not None:
            self.time_step = time_step
        else:
            self.time_step += 1
        self._current_batch_step()
        if res is not None and self.time_step < self.N:
            self.store = [
                res['store'][home][self.time_step] for home in range(self.n_homes)
            ]
        if implement:
            self.start_store = self.store.copy()

    def revert_last_update_step(self):
        self.start_store = self.prev_start_store
        self.update_step(time_step=self.prev_time_step, implement=False)

    def set_passive_active(self, passive: bool, prm: dict):
        self.passive_ext = 'P' if passive else ''
        # number of agents / households
        self.n_homes = prm['syst']['n_homes' + self.passive_ext]
        for info in ['own_car', 'store0', 'cap', 'min_charge']:
            setattr(self, info, prm['car'][info + self.passive_ext])

    def compute_battery_demand_aggregated_at_start_of_trip(
            self,
            batch: dict
    ) -> dict:
        for home in range(self.n_homes):
            batch[home]['bat_dem_agg'] = np.zeros(len(batch[home]['avail_car']))
            if self.own_car[home]:
                start_trip, end_trip = [], []
                if batch[home]['avail_car'][0] == 0:
                    start_trip.append(0)
                for time in range(len(batch[home]['avail_car']) - 1):
                    if batch[home]['avail_car'][time] == 0 \
                            and batch[home]['avail_car'][time + 1] == 1:
                        end_trip.append(time)
                    if time > 0 and batch[home]['avail_car'][time] == 0 \
                            and batch[home]['avail_car'][time - 1] == 1:
                        start_trip.append(time)
                if len(start_trip) > len(end_trip) \
                        and batch[home]['avail_car'][time] == 0:
                    end_trip.append(self.N - 1)
                for start, end in zip(start_trip, end_trip):
                    batch[home]['bat_dem_agg'][start] = \
                        sum(batch[home]['loads_car'][start: end + 1])

        return batch

    def add_batch(self, batch):
        """Once batch data computed, update in battery data batch."""
        for info in self.batch_entries:
            self.batch[info] = [batch[home][info] for home in range(self.n_homes)]
        self._current_batch_step()

    def next_trip_details(self, start_h, date, home):
        """Get next trip's load requirements, time until trip, and end."""
        # next time the EV is on a trip
        iT = np.asarray(
            ~np.array(self.batch['avail_car'][home][start_h: self.N + 1], dtype=bool)
        ).nonzero()[0]
        d_to_end = self.date_end - date
        h_end = \
            start_h \
            + (d_to_end.days * 24 + (d_to_end.seconds) / 3600) * 1 / self.dt
        if len(iT) > 0 and start_h + iT[0] < h_end:
            # future trip that starts before end
            iT = int(start_h + iT[0])

            # next time the EV is back from the trip to the garage
            iG = iT + np.asarray(self.batch['avail_car'][home][iT: self.N + 1]).nonzero()[0]
            iG = int(iG[0]) if len(iG) > 0 else len(self.batch['avail_car'][home])
            deltaT = iT - start_h  # time until trip
            i_end_trip = int(min(iG, h_end))
            # car load while on trip
            loads_T = np.sum(self.batch['loads_car'][home][iT: i_end_trip])

            return loads_T, deltaT, i_end_trip
        else:
            return None, None, None

    def car_tau(self, hour, date, home, store_a):
        """Compute car_tau, i.e. how much charge is needed over in how long."""
        loads_T, deltaT, i_end_trip = self.next_trip_details(hour, date, home)
        if loads_T is not None and deltaT > 0:
            val = (loads_T - store_a) / deltaT
        else:
            val = - 1

        return val

    def _check_trip_feasible(
            self, loads_T, deltaT, bool_penalty, print_error, home, time
    ):
        if loads_T > self.cap[home] + 1e-2:
            # load during trip larger than whole
            bool_penalty[home] = True
            if print_error:
                print(f"home = {home}, time = {time} deltaT {deltaT} "
                      f"load EV trip {loads_T} larger than"
                      f" cap, self.batch['loads_car'][{home}] "
                      f"= {self.batch['loads_car'][home]}")
        elif deltaT > 0 \
                and sum(self.batch['avail_car'][home][0: time]) == 0 \
                and loads_T / deltaT \
                > self.store0[home] + self.c_max:
            bool_penalty[home] = True
            if print_error:
                print(f'home = {home}, loads_T {loads_T} '
                      f'deltaT {deltaT} not enough '
                      f'steps to fill what is needed')

        return bool_penalty

    def _get_list_future_trips(self, time, date_a, home, bool_penalty, print_error):
        trips = []
        end = False
        h_trip = time
        while not end:
            loads_T, deltaT, i_end_trip = self.next_trip_details(h_trip, date_a, home)
            # if i_end_trip is not None:
            #     i_end_trip = np.min([self.N, i_end_trip])
            if loads_T is None or h_trip + deltaT > self.N:
                end = True
            else:
                self._check_trip_feasible(
                    loads_T, deltaT, bool_penalty, print_error, home, time
                )
                if time + deltaT < self.N:
                    trips.append([loads_T, deltaT, i_end_trip])
                date_a += datetime.timedelta(
                    hours=int(i_end_trip - h_trip) * self.dt)
                h_trip = i_end_trip

        return trips

    def min_max_charge_t(self, time=None, date=None, print_error=True,
                         simulation=True):
        """
        For current time step, define minimum/maximum charge.

        simulation:
            yes if the self.store value if updated in
            current simulation. false if we are just testing the
            feasibility of a generic batch of data without updating
            the storage levels
        """
        if time is None:
            time = self.time_step
        if date is None:
            date = self.date0 + datetime.timedelta(hours=time * self.dt)
        avail_car = [self.batch['avail_car'][home][time] for home in range(self.n_homes)]
        bool_penalty = np.zeros(self.n_homes, dtype=bool)
        last_step = self._last_step(date)
        # regular initial minimum charge
        min_charge_t_0 = np.where(last_step, self.store0, self.min_charge) * avail_car

        # min_charge if need to charge up ahead of last step
        charge_required = []
        for home in range(self.n_homes):
            if not avail_car[home]:
                # if EV not currently in garage
                charge_required.append(0)
                continue

            # obtain all future trips
            trips = self._get_list_future_trips(
                time, date, home, bool_penalty, print_error
            )
            # obtain required charge before each trip, starting with end
            final_i_endtrip = trips[-1][2] if len(trips) > 0 else time
            n_avail_until_end = sum(self.batch['avail_car'][home][final_i_endtrip: self.N])

            if len(trips) == 0:
                n_avail_until_end -= 1

            charge_for_final_step = self.store0[home] - self.c_max * n_avail_until_end
            charge_for_reaching_min_charge_next = \
                self.min_charge[home] * self.batch['avail_car'][home][final_i_endtrip + 1] \
                - self.c_max
            charge_required.append(
                max(charge_for_final_step, charge_for_reaching_min_charge_next, 0)
            )

            for it in range(len(trips)):
                loads_T, deltaT = trips[- (it + 1)][0:2]
                if it == len(trips) - 1:
                    deltaT -= 1
                    charge_required[home] += max(0, self.min_charge[home] - self.c_max)
                # this is the required charge at the current step
                # if this is the most recent trip, or right after
                # the previous trip
                charge_required[home] = max(
                    0,
                    charge_required[home] + loads_T - deltaT * self.c_max
                )

        min_charge_t = np.maximum(min_charge_t_0, charge_required)
        self._check_min_charge_t_feasible(
            min_charge_t, time, date, bool_penalty, print_error, simulation
        )
        for home in range(self.n_homes):
            if time == self.N - 2 and avail_car[home]:
                assert min_charge_t[home] >= self.store0[home] - self.c_max, \
                    f"time == {self.N - 2} and min_charge_t {min_charge_t[home]} " \
                    f"< {self.store0[home]} - {self.c_max}"

        self.min_charge_t = min_charge_t
        self.max_charge_t = np.where(last_step and self.avail_car, self.store0, self.cap)

        return bool_penalty

    def check_constraints(self, home, date, time):
        """
        From env.policy_to_rewardvar, check battery constraints.

        Inputs:
        home:
            index relative to agent
        time:
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
            - self.loads_car[home] - self.loss_dis[home] \
            - (self.store[home] - self.start_store[home])
        assert abs(bat_e_balance) < 1e-2, \
            f"home = {home}, battery energy balance sum = {bat_e_balance}"

        # initial and final storage level
        if date == self.date0:
            assert self.start_store[home] == self.store0[home], \
                f'start_store[{home}] {self.start_store[home]} not store0'

        if date == self.date_end - datetime.timedelta(hours=self.dt) \
                and self.avail_car[home]:
            assert self.store[home] >= self.store0[home] - 1e-2, \
                f"home = {home}, store end {self.store[home]} " \
                f"smaller than store0 {self.store0[home]}"

        # max storage level
        assert self.store[home] <= self.cap[home] + 1e-2, \
            f'store[{home}] {self.store[home]} larger than cap'

        if self.max_charge_t[home] is not None \
                and self.avail_car[home]:
            assert self.charge[home] <= self.max_charge_t[home] + 1e-2, \
                f'charge[{home}] {self.charge[home]} > max_charge_t[home] ' \
                f'{self.max_charge_t[home]} after flex, ' \
                f'last_step = {last_step}'

        # minimum storage level
        assert self.store[home] \
               >= self.SoCmin * self.cap[home] * self.avail_car[home] - 1e-2, \
               f"store[{home}] {self.store[home]} " \
               f"smaller than SoCmin and no bool_penalty, " \
               f"availcar[home] = {self.avail_car[home]}, " \
               f"charge[home] = {self.charge[home]}, " \
               f"c_max = {self.c_max}"

        assert \
            self.store[home] >= self.min_charge_t[home] * self.avail_car[home] - 1e-2, \
            f'store[{home}] {self.store[home]} ' \
            f'smaller than min_charge_t[home] {self.min_charge_t[home]} ' \
            f'avail_car[home] {self.avail_car[home]}'

        # charge and discharge losses
        abs_loss_charge = \
            self.loss_ch[home] - (
                (self.charge[home] + self.loss_ch[home])
                * (1 - self.eta_ch)
            )
        assert abs(abs_loss_charge) <= 1e-2, \
            f"self.cap = {self.cap}, time = {time}, home = {home} " \
            f"sum loss charge = {abs_loss_charge}"

        abs_loss_charge = \
            self.loss_dis[home] - (
                (self.discharge[home] + self.loss_dis[home])
                * (1 - self.eta_dis)
            )
        assert abs(abs_loss_charge) <= 1e-2, \
            f"time = {time}, home = {home} sum loss charge = {abs_loss_charge}"

        # only charge and discharge if EV is available
        assert self.charge[home] <= self.avail_car[home] * self.M, \
            'charge but EV not available and no bool_penalty'
        assert self.discharge[home] <= self.avail_car[home] * self.M, \
            f'home = {home} discharge (else than EV cons) but ' \
            f'EV not available'

        # max charge rate
        assert self.charge[home] <= self.c_max + 1e-2, \
            f"charge {self.charge} > c_max {self.c_max}"

        # max discharge rate
        assert self.discharge[home] + self.loss_dis[home] + self.loads_car[home] \
               <= self.d_max + 1e-2, \
               f'home = {home}, discharge[home] {self.discharge[home]} > self.d_max'

        # positivity
        for info in ['store', 'charge', 'discharge', 'loss_ch', 'loss_dis']:
            assert self.__dict__[info][home] >= - 1e-2, \
                f'home = {home} negative {info}[home] {self.__dic__[info][home]}'

    def actions_to_env_vars(self, res):
        """Update battery state for current actions."""
        for info in [
            'store', 'charge', 'discharge', 'loss_ch', 'loss_dis', 'store_out_tot', 'discharge_tot'
        ]:
            setattr(self, info, [None for _ in range(self.n_homes)])
        for home in range(self.n_homes):
            self.store[home] = self.start_store[home] \
                + res[home]['ds'] - self.loads_car[home]
            if self.store[home] < self.min_charge_t[home] - 5e-3:
                print(f"home {home} store[{home}] {self.store[home]} "
                      f"self.start_store[home] {self.start_store[home]} "
                      f"res[{home}['ds'] {res[home]['ds']} "
                      f"self.loads_car[home] {self.loads_car[home]} "
                      f"self.min_charge_t[home] {self.min_charge_t[home]}")
            self.charge[home] = res[home]['ds'] if res[home]['ds'] > 0 else 0
            self.discharge[home] = - res[home]['ds'] * self.eta_dis \
                if res[home]['ds'] < 0 else 0
            self.loss_ch[home] = res[home]['l_ch']
            self.loss_dis[home] = res[home]['l_dis']
            self.store_out_tot[home] = self.discharge[home] \
                + self.loss_dis[home] + self.loads_car[home]
            self.discharge_tot[home] = self.discharge[home] / self.eta_dis \
                + self.loads_car[home]

    def initial_processing(self):
        """Get current available battery flexibility."""
        # storage available above minimum charge that can be discharged
        s_avail_dis = np.array(
            [min(
                max(self.start_store[home] - self.min_charge_t[home], 0),
                self.d_max
            ) * self.avail_car[home] for home in range(self.n_homes)]
        )

        # how much is to be added to store
        s_add_0 = np.array(
            [
                max(self.min_charge_t[home] - self.start_store[home], 0) * self.avail_car[home]
                for home in range(self.n_homes)
            ]
        )
        home_too_large = np.where(s_add_0 > self.c_max + 1e-3)[0]
        if len(home_too_large) > 0:
            if self.time_step == self.N:
                for home in home_too_large:
                    s_add_0[home] = self.c_max
        assert self.time_step == self.N or len(home_too_large) == 0, \
            f"s_add_0: {s_add_0[home_too_large[0]]} > self.c_max {self.c_max} " \
            f"self.min_charge_t[i_too_large[0]] {self.min_charge_t[home_too_large[0]]} " \
            f"self.start_store[i_too_large[0]] {self.start_store[home_too_large[0]]} " \
            f"self.time_step {self.time_step} " \
            f"self.store[i_too_large[0]] {self.store[home_too_large[0]]} "

        # how much i need to remove from store
        s_remove_0 = np.array(
            [max(self.start_store[home] - self.max_charge_t[home], 0)
             * self.avail_car[home] for home in range(self.n_homes)]
        )

        # how much can I charge it by rel. to current level
        potential_charge = np.array(
            [min(self.c_max, self.max_charge_t[home] - self.start_store[home])
             * self.avail_car[home] for home in range(self.n_homes)]
        )

        assert all(add <= potential + 1e-3 for add, potential in zip(s_add_0, potential_charge)
                   if potential > 0), f"s_add_0 {s_add_0} > potential_charge {potential_charge}"
        assert all(remove <= avail + 1e-3 for remove, avail in zip(s_remove_0, s_avail_dis)
                   if avail > 0), f"s_remove_0 {s_remove_0} > s_avail_dis {s_avail_dis}"

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
            # car
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
            if self.discharge[home] + self.loads_car[home] > self.d_max + 1e-2:
                print(f"discharge[{home}] {self.discharge[home]} > "
                      f"self.d_max {self.d_max}")
                bool_penalty[home] = True

            if not self.avail_car[home] and self.loads_car[home] > 1e-2 \
                    and self.loads_car[home] > self.start_store[home] + 1e-2:
                print(f"self.loads_car[{home}] = {self.loads_car[home]}, "
                      f"self.start_store[home] = {self.start_store[home]}")
                bool_penalty[home] = True

            if self.avail_car[home] == 0 and res[home]['ds'] > 0:
                print('in update action dch > 0, caravail = 0')
                bool_penalty[home] = True

            if self.max_charge_t[home] is not None \
                    and res[home]['ds'] > self.max_charge_t[home] + 1e-2:
                print(f"self.max_charge_t[{home}] = {self.max_charge_t[home]} "
                      f"store[home] = {res[home]['ds']}")
                print(f'action[home] = {action[home]}')
                bool_penalty[home] = True

        if not type(self.store[0]) in [float, np.float64]:
            print('not type(store[0]) in [float, np.float64]')
            bool_penalty[home] = True

        return bool_penalty

    def _current_batch_step(self):
        for info in self.batch_entries:
            setattr(
                self,
                info,
                [self.batch[info][home][self.time_step] for home in range(self.n_homes)]
            )

    def _last_step(self, date):
        return date == self.date_end - datetime.timedelta(hours=self.dt)

    def _check_first_time_step_feasibility(
            self, time, date, bool_penalty, print_error
    ):
        for home in range(self.n_homes):
            # check if opportunity to charge before trip > 37.5
            if time == 0 and not self.batch['avail_car'][home][0]:
                loads_T, deltaT, _ = self.next_trip_details(time, date, home)
                if loads_T > self.store0[home]:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[home] = True
                    error_message = \
                        f'home = {home}, trip {loads_T} larger than ' \
                        f'initial charge - straight away not available'
                    self._print_error(error_message, print_error)
                if sum(self.batch['avail_car'][home][0:23]) == 0 \
                        and sum(self.batch['loads_car'][home][0:23]) \
                        > self.c_max + 1e-2:
                    bool_penalty[home] = True
                    error_message = \
                        f'home = {home}, only last step available, ' \
                        'not enough to go back to initial charge ' \
                        'at the last time step'
                    self._print_error(error_message, print_error)
                loads_T_next, deltaT_next, _ = \
                    self.next_trip_details(
                        deltaT,
                        date + datetime.timedelta(hours=deltaT * self.dt),
                        home
                    )
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
            self, min_charge_t, time, date, bool_penalty, print_error, simulation
    ):
        bool_penalty = self._check_first_time_step_feasibility(
            time, date, bool_penalty, print_error
        )

        for home in range(self.n_homes):
            # check if any hourly load is larger than d_max
            if any(self.batch['loads_car'][home][time] > self.d_max + 1e-2
                   for time in range(self.N)):
                # you would have to break constraints to meet demand
                bool_penalty[home] = True
                self._print_error(
                    f'home = {home}, load car larger than d_max',
                    print_error
                )

            if min_charge_t[home] > self.cap[home] + 1e-2:
                bool_penalty[home] = True  # min_charge_t larger than total cap
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger than cap'
                self._print_error(error_message, print_error)
            if min_charge_t[home] > self.store0[home] \
                    - sum(self.batch['loads_car'][home][0: time]) + (
                    sum(self.batch['loads_car'][home][0: time]) + 1) \
                    * self.c_max + 1e-3:
                bool_penalty[home] = True
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger what car can be charged to'
                self._print_error(error_message, print_error)

            if self.time_step < self.N and simulation \
                    and min_charge_t[home] > self.store[home] + self.c_max + 1e-3:
                bool_penalty[home] = True
                error_message = \
                    f"date {date} time {time} " \
                    f"min_charge_t[{home}] {min_charge_t[home]} " \
                    f"> self.store[{home}] {self.store[home]} "

                self._print_error(error_message, print_error)

            elif not simulation \
                    and time > 0 \
                    and sum(self.batch['avail_car'][home][0: time]) == 0:
                # the car has not been available at home to recharge until now
                store_t_a = self.store0[home] \
                    - sum(self.batch['loads_car'][home][0: time])
                if min_charge_t[home] > store_t_a + self.c_max + 1e-3:
                    bool_penalty[home] = True

    def _print_error(self, error_message, print_error):
        if print_error:
            print(error_message)

    def check_feasible_bat(self, prm, passive_ext):
        """Check charging constraints for proposed data batch."""
        feasible = np.ones(prm['syst']['n_homes' + passive_ext], dtype=bool)
        for home in range(prm['syst']['n_homes' + passive_ext]):
            if prm['car']['d_max'] < np.max(prm['car']['batch_loads_car'][home]):
                feasible[home] = False
                print("car['d_max'] < np.max(car['batch_loads_car'][home])")
                for time in range(len(prm['car']['batch_loads_car'][home])):
                    if prm['car']['batch_loads_car'][home, time] > prm['car']['d_max']:
                        prm['car']['batch_loads_car'][home, time] = prm['car']['d_max']

        time = 0

        self.reset(prm)
        while all(feasible) and time < self.N:
            date = self.date0 + datetime.timedelta(hours=time * self.dt)
            bool_penalty = self.min_max_charge_t(
                time, date, print_error=False,
                simulation=False)
            feasible[bool_penalty] = False
            self.update_step()
            time += 1

        return feasible

    def _active_reactive_power_car(self):
        self.p_car_flex = np.array(self.loss_ch) + np.array(self.charge) \
            - np.array(self.discharge)
        self.q_car_flex = _calculate_reactive_power(
            self.p_car_flex, self.pf_flexible_homes)
