#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 8 16:40:04 2021.

@author: Flora Charbonnier
"""
import copy
import datetime

import numpy as np

from src.utilities.userdeftools import calculate_reactive_power


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

    def __init__(self, prm, ext: str = ''):
        """
        Initialise Battery object.

        inputs:
        prm:
            input parameters
        ext:
            are the current agents passive? '' is not, 'P' if yes
            are we looking at a different number of tested homes? '_test' if yes else ''        """
        # very large number for enforcing constraints
        self.set_passive_active(ext, prm)

        self.pf_flexible_homes = prm['grd']['pf_flexible_homes']
        self.reactive_power_for_voltage_control = \
            prm['grd']['reactive_power_for_voltage_control']

        for info in ['M', 'N', 'dt']:
            setattr(self, info, prm['syst'][info])

        for info in [
            'dep', 'c_max', 'd_max', 'eta_ch', 'eta_ch', 'eta_dis', 'SoCmin',
            'max_apparent_power_car',
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
        self.store = np.where(
            prm['car']['own_car' + self.ext],
            prm['car']['store0' + self.ext],
            0
        )
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
        if res is not None:
            if self.time_step < self.N:
                self.store = res['store'][:, self.time_step]
            else:
                self.store = \
                    res['store'][:, self.time_step - 1] \
                    + res['charge'][:, self.time_step - 1] \
                    - res['discharge_tot'][:, self.time_step - 1]
        if implement:
            self.start_store = self.store.copy()

    def revert_last_update_step(self):
        self.start_store = self.prev_start_store
        self.update_step(time_step=self.prev_time_step, implement=False)

    def set_passive_active(self, ext, prm: dict):
        self.ext = ext
        # number of agents / households
        self.n_homes = prm['syst']['n_homes' + self.ext]
        for info in ['own_car', 'store0', 'caps', 'min_charge']:
            setattr(self, info, prm['car'][info + self.ext])

    def compute_battery_demand_aggregated_at_start_of_trip(
            self,
            batch: dict
    ) -> dict:
        batch['bat_dem_agg'] = np.zeros((self.n_homes, len(batch['avail_car'][0])))
        for home in range(self.n_homes):
            if self.own_car[home]:
                start_trip, end_trip = [], []
                if batch['avail_car'][home, 0] == 0:
                    start_trip.append(0)
                for time_step in range(len(batch['avail_car'][home]) - 1):
                    if batch['avail_car'][home, time_step] == 0 \
                            and batch['avail_car'][home, time_step + 1] == 1:
                        end_trip.append(time_step)
                    if time_step > 0 and batch['avail_car'][home, time_step] == 0 \
                            and batch['avail_car'][home, time_step - 1] == 1:
                        start_trip.append(time_step)
                if len(start_trip) > len(end_trip) \
                        and batch['avail_car'][home, time_step] == 0:
                    end_trip.append(self.N - 1)
                for start, end in zip(start_trip, end_trip):
                    batch['bat_dem_agg'][home, start] = \
                        sum(batch['loads_car'][home, start: end + 1])

        return batch

    def add_batch(self, batch):
        """Once batch data computed, update in battery data batch."""
        for info in self.batch_entries:
            dtype = bool if info == 'avail_car' else float
            self.batch[info] = np.array(batch[info], dtype=dtype)
        self._current_batch_step()

    def next_trip_details(self, start_time_step, date, home):
        """Get next trip's load requirements, time until trip, and end."""
        # next time the EV is on a trip
        iT = np.asarray(
            ~np.array(self.batch['avail_car'][home, start_time_step: self.N + 1], dtype=bool)
        ).nonzero()[0]
        d_to_end = self.date_end - date
        h_end = \
            start_time_step \
            + (d_to_end.days * 24 + (d_to_end.seconds) / 3600) * 1 / self.dt
        if len(iT) > 0 and start_time_step + iT[0] < h_end:
            # future trip that starts before end
            iT = int(start_time_step + iT[0])

            # next time the EV is back from the trip to the garage
            iG = iT + np.asarray(self.batch['avail_car'][home, iT: self.N + 1]).nonzero()[0]
            iG = int(iG[0]) if len(iG) > 0 else len(self.batch['avail_car'][home])
            deltaT = iT - start_time_step  # time until trip
            i_end_trip = int(min(iG, h_end))
            # car load while on trip
            loads_T = np.sum(self.batch['loads_car'][home, iT: i_end_trip])

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
            self, loads_T, deltaT, bool_penalty, print_error, home, time_step
    ):
        if loads_T > self.caps[home] + 1e-2:
            # load during trip larger than whole
            bool_penalty[home] = True
            if print_error:
                print(f"home = {home}, time_step = {time_step} deltaT {deltaT} "
                      f"load EV trip {loads_T} larger than"
                      f" cap, self.batch['loads_car'][{home}] "
                      f"= {self.batch['loads_car'][home]}")
        elif deltaT > 0 \
                and sum(self.batch['avail_car'][home, 0: time_step]) == 0 \
                and loads_T / deltaT \
                > self.store0[home] + self.c_max:
            bool_penalty[home] = True
            if print_error:
                print(f'home = {home}, loads_T {loads_T} '
                      f'deltaT {deltaT} not enough '
                      f'steps to fill what is needed')

        return bool_penalty

    def _get_list_future_trips(self, time_step, date_a, home, bool_penalty, print_error):
        trips = []
        end = False
        h_trip = time_step
        while not end:
            loads_T, deltaT, i_end_trip = self.next_trip_details(h_trip, date_a, home)
            # if i_end_trip is not None:
            #     i_end_trip = np.min([self.N, i_end_trip])
            if loads_T is None or h_trip + deltaT > self.N:
                end = True
            else:
                self._check_trip_feasible(
                    loads_T, deltaT, bool_penalty, print_error, home, time_step
                )
                if time_step + deltaT < self.N:
                    trips.append([loads_T, deltaT, i_end_trip])
                date_a += datetime.timedelta(
                    hours=int(i_end_trip - h_trip) * self.dt)
                h_trip = i_end_trip

        return trips

    def min_max_charge_t(self, time_step=None, date=None, print_error=True,
                         simulation=True):
        """
        For current time step, define minimum/maximum charge.

        simulation:
            yes if the self.store value if updated in
            current simulation. false if we are just testing the
            feasibility of a generic batch of data without updating
            the storage levels
        """
        if time_step is None:
            time_step = self.time_step
        if date is None:
            date = self.date0 + datetime.timedelta(hours=time_step * self.dt)
        avail_car = self.batch['avail_car'][:, time_step]
        bool_penalty = np.zeros(self.n_homes, dtype=bool)
        last_step = self._last_step(date)
        # regular initial minimum charge
        min_charge_t_0 = np.where(last_step, self.store0, self.min_charge) * avail_car

        # min_charge if need to charge up ahead of last step
        min_charge_required = np.zeros(self.n_homes)
        max_charge_for_final_step = copy.deepcopy(self.caps)
        for home in range(self.n_homes):
            if not avail_car[home]:
                # if EV not currently in garage
                continue

            # obtain all future trips
            trips = self._get_list_future_trips(
                time_step, date, home, bool_penalty, print_error
            )
            if self.time_step <= self.N - 1:
                max_charge_for_final_step[home] = \
                    self.store0[home] \
                    + sum(trip[0] for trip in trips) \
                    + self.d_max * sum(self.batch['avail_car'][home, time_step: self.N - 1])

            # obtain required charge before each trip, starting with end
            final_i_endtrip = trips[-1][2] if len(trips) > 0 else time_step + 1
            min_charge_after_final_trip = max(
                self.store0[home]
                - self.c_max * sum(self.batch['avail_car'][home, final_i_endtrip: self.N]),
                self.min_charge[home] - self.c_max,
                # this is because we can take one step to recover the minimum charge after a trip
                0
            )
            min_charge_after_next_trip = min_charge_after_final_trip
            for it in range(len(trips)):
                loads_T, deltaT = trips[- (it + 1)][0: 2]
                if it == len(trips) - 1:
                    # do not count current time step
                    deltaT -= 1
                min_charge_ahead_of_trip = max(
                    self.min_charge[home] - self.c_max,
                    min_charge_after_next_trip + loads_T - deltaT * self.c_max,
                    0
                )
                min_charge_after_next_trip = min_charge_ahead_of_trip

            min_charge_required[home] = max(min_charge_after_next_trip, self.min_charge[home])

        min_charge_t = np.maximum(min_charge_t_0, min_charge_required)
        self._check_min_charge_t_feasible(
            min_charge_t, time_step, date, bool_penalty, print_error, simulation
        )
        for home in range(self.n_homes):
            if time_step == self.N - 1 and avail_car[home]:
                assert min_charge_t[home] >= self.store0[home] - self.c_max, \
                    f"time_step == {self.N - 1} and min_charge_t {min_charge_t[home]} " \
                    f"< {self.store0[home]} - {self.c_max}"

        absolute_max_charge_t = np.where(last_step and self.avail_car, self.store0, self.caps)
        self.max_charge_t = np.minimum(max_charge_for_final_step, absolute_max_charge_t)
        self.min_charge_t = np.where(
            abs(min_charge_t - self.max_charge_t) < 1e-2, self.max_charge_t, min_charge_t
        )

        return bool_penalty

    def check_constraints(self, home, date, time_step):
        """
        From env.policy_to_rewardvar, check battery constraints.

        Inputs:
        home:
            index relative to agent
        time_step:
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
        assert self.store[home] <= self.caps[home] + 1e-2, \
            f'store[{home}] {self.store[home]} larger than cap'

        if self.max_charge_t[home] is not None \
                and self.avail_car[home]:
            assert self.charge[home] <= self.max_charge_t[home] + 1e-2, \
                f'charge[{home}] {self.charge[home]} > max_charge_t[home] ' \
                f'{self.max_charge_t[home]} after flex, ' \
                f'last_step = {last_step}'

        # minimum storage level
        assert self.store[home] \
               >= self.SoCmin * self.caps[home] * self.avail_car[home] - 1e-2, \
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
            f"self.caps = {self.caps}, time_step = {time_step}, home = {home} " \
            f"sum loss charge = {abs_loss_charge}"

        abs_loss_charge = \
            self.loss_dis[home] - (
                (self.discharge[home] + self.loss_dis[home])
                * (1 - self.eta_dis)
            )
        assert abs(abs_loss_charge) <= 1e-2, \
            f"time_step = {time_step}, home = {home} sum loss charge = {abs_loss_charge}"

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
        # calculate active and reactive power for all homes
        if not self.reactive_power_for_voltage_control:
            self.active_reactive_power_car()
        apparent_power_car = np.square(self.p_car_flex) + np.square(self.q_car_flex)
        assert all(apparent_power_car <= self.max_apparent_power_car**2), \
            f"The sum of squares of p_car_flex and q_car_flex exceeds the" \
            f" maximum apparent power of the car: {self.max_apparent_power_car**2} < " \
            f"{apparent_power_car.max()}"

    def initial_processing(self):
        """Get current available battery flexibility."""
        # storage available above minimum charge that can be discharged
        s_avail_dis = np.where(
            self.avail_car,
            np.minimum(np.maximum(self.start_store - self.min_charge_t, 0), self.d_max),
            0
        )
        # how much is to be added to store
        s_add_0 = np.multiply(self.avail_car, np.maximum(self.min_charge_t - self.start_store, 0))
        if self.time_step == self.N:
            s_add_0 = np.where(s_add_0 > self.c_max + 1e-3, self.c_max, s_add_0)
        assert all(s_add_0 <= self.c_max + 1e-3), \
            f"s_add_0: {s_add_0} > self.c_max {self.c_max} " \
            f"self.min_charge_t[i_too_large[0]] {self.min_charge_t} " \
            f"self.start_store[i_too_large[0]] {self.start_store} " \
            f"self.time_step {self.time_step} " \
            f"self.store[i_too_large[0]] {self.store} "

        # how much I need to remove from store
        s_remove_0 = np.multiply(
            np.maximum(self.start_store - self.max_charge_t, 0),
            self.avail_car
        )
        same_as_avail_dis = abs(s_avail_dis - s_remove_0) < 1e-2
        s_remove_0[same_as_avail_dis] = s_avail_dis[same_as_avail_dis]

        # how much can I charge it by rel. to current level
        potential_charge = np.multiply(
            self.avail_car,
            np.minimum(self.c_max, np.maximum(self.max_charge_t - self.start_store, 0))
        )
        for home in range(len(s_add_0)):
            if s_add_0[home] > potential_charge[home] + 1e-3:
                print(f"home {home} "
                      f"self.avail_car[home] {self.avail_car[home]} "
                      f"self.min_charge_t[home] {self.min_charge_t[home]} "
                      f"self.max_charge_t[home] {self.max_charge_t[home]} "
                      f"self.start_store[home] {self.start_store[home]} "
                      f"self.time_step {self.time_step} "
                      f"self.store[home] {self.store[home]} "
                      f"self.c_max {self.c_max} "
                      f"potential_charge[home] {potential_charge[home]} "
                      f"s_add_0[home] {s_add_0[home]}"
                      )

        assert all(add <= potential + 1e-3 for add, potential in zip(s_add_0, potential_charge)
                   if potential > 0), f"s_add_0 {s_add_0} > potential_charge {potential_charge}"
        assert all(remove <= avail + 5e-2 for remove, avail in zip(s_remove_0, s_avail_dis)
                   if avail > 0), f"s_remove_0 {s_remove_0} > s_avail_dis {s_avail_dis}"
        if self.time_step == self.N - 1:
            assert all((~self.avail_car) | (self.min_charge_t == self.store0)), \
                "end time but min_charge_t != store0"

        assert all(s_avail_dis > s_remove_0 - 1e-3)

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

            if not type(self.store[home]) in [float, np.float64]:
                print('not type(store[home]]) in [float, np.float64]')
                bool_penalty[home] = True

        return bool_penalty

    def _current_batch_step(self):
        for info in self.batch_entries:
            setattr(
                self,
                info,
                self.batch[info][:, self.time_step]
            )

    def _last_step(self, date):
        return date == self.date_end - datetime.timedelta(hours=self.dt)

    def _check_first_time_step_feasibility(
            self, time_step, date, bool_penalty, print_error
    ):
        for home in range(self.n_homes):
            # check if opportunity to charge before trip > 37.5
            if time_step == 0 and not self.batch['avail_car'][home, 0]:
                loads_T, deltaT, _ = self.next_trip_details(time_step, date, home)
                if loads_T > self.store0[home]:
                    # trip larger than initial charge and straight
                    # away not available
                    bool_penalty[home] = True
                    error_message = \
                        f'home = {home}, trip {loads_T} larger than ' \
                        f'initial charge - straight away not available'
                    self._print_error(error_message, print_error)
                if sum(self.batch['avail_car'][home, 0: self.N - 1]) == 0 \
                        and sum(self.batch['loads_car'][home, 0: self.N - 1]) \
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
            self, min_charge_t, time_step, date, bool_penalty, print_error, simulation
    ):
        bool_penalty = self._check_first_time_step_feasibility(
            time_step, date, bool_penalty, print_error
        )

        for home in range(self.n_homes):
            # check if any hourly load is larger than d_max
            if any(self.batch['loads_car'][home, :] > self.d_max + 1e-2):
                # you would have to break constraints to meet demand
                bool_penalty[home] = True
                self._print_error(
                    f'home = {home}, load car larger than d_max',
                    print_error
                )

            if min_charge_t[home] > self.caps[home] + 1e-2:
                bool_penalty[home] = True  # min_charge_t larger than total cap
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger than cap'
                self._print_error(error_message, print_error)
            if min_charge_t[home] > self.store0[home] \
                    - sum(self.batch['loads_car'][home, 0: time_step]) + (
                    sum(self.batch['loads_car'][home, 0: time_step]) + 1) \
                    * self.c_max + 1e-3:
                bool_penalty[home] = True
                error_message = f'home = {home}, min_charge_t {min_charge_t[home]} ' \
                    'larger what car can be charged to'
                self._print_error(error_message, print_error)

            if (
                self.time_step < self.N
                and simulation
                and min_charge_t[home]
                > (self.store[home] + self.c_max) * self.avail_car[home] + 1e-3
            ):
                bool_penalty[home] = True
                error_message = \
                    f"date {date} time_step {time_step} " \
                    f"min_charge_t[{home}] {min_charge_t[home]} " \
                    f"> self.store[{home}] {self.store[home]} + self.c_max {self.c_max} "

                self._print_error(error_message, print_error)

            elif not simulation \
                    and time_step > 0 \
                    and sum(self.batch['avail_car'][home, 0: time_step]) == 0:
                # the car has not been available at home to recharge until now
                store_t_a = self.store0[home] \
                    - sum(self.batch['loads_car'][home, 0: time_step])
                if min_charge_t[home] > store_t_a + self.c_max + 1e-3:
                    bool_penalty[home] = True

    def _print_error(self, error_message, print_error):
        if print_error:
            print(error_message)

    def check_feasible_bat(self, prm, ext):
        """Check charging constraints for proposed data batch."""
        feasible = np.ones(prm['syst']['n_homes' + ext], dtype=bool)
        for home in range(prm['syst']['n_homes' + ext]):
            if prm['car']['d_max'] < np.max(prm['car']['batch_loads_car'][home]):
                feasible[home] = False
                for time_step in range(len(prm['car']['batch_loads_car'][home])):
                    if prm['car']['batch_loads_car'][home, time_step] > prm['car']['d_max']:
                        prm['car']['batch_loads_car'][home, time_step] = prm['car']['d_max']

        time_step = 0

        self.reset(prm)
        while all(feasible) and time_step < self.N:
            date = self.date0 + datetime.timedelta(hours=time_step * self.dt)
            bool_penalty = self.min_max_charge_t(
                time_step, date, print_error=False,
                simulation=False
            )
            feasible[bool_penalty] = False

            self.update_step()
            time_step += 1

        return feasible

    def active_reactive_power_car(self):
        self.p_car_flex = np.array(self.charge) - np.array(self.discharge)
        self.q_car_flex = calculate_reactive_power(
            self.p_car_flex, self.pf_flexible_homes)
