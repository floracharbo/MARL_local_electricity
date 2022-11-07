#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:57 2020.

@author: floracharbonnier

"""

import copy
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.stats import norm
from six import integer_types

from src.home_components.battery import Battery
from src.home_components.heat import Heat
from src.simulations.action_translator import Action_translator
from src.simulations.hedge import HEDGE
from src.utilities.env_spaces import EnvSpaces
from src.utilities.userdeftools import initialise_dict


class LocalElecEnv():
    """
    Local electricity environment.

    Includes home-level modelling of the flexible assets,
    computes step actions, updating states and rewards.
    """

    # =============================================================================
    # # initialisation / data interface
    # =============================================================================
    def __init__(self, prm, profiles):
        """Initialise Local Elec environment, add properties."""
        self.batchfile0 = 'batch'
        self.envseed = self._seed()
        self.random_seeds_use = {}
        self.batch_entries = ['loads', 'gen', 'loads_car', 'avail_car', 'flex']
        self.clus, self.f = {}, {}
        self.labels = ['loads', 'car', 'gen']
        self.labels_clus = ['loads', 'car']
        self.test = prm['syst']['test_on_run']
        self.prm = prm
        self.rl = prm['RL']
        self.labels_day_trans = prm['syst']['labels_day_trans']

        self.prof = profiles
        self.n_homes = prm['ntw']['n']
        self.homes = range(self.n_homes)

        # initialise parameters
        self._init_factors_clusters_profiles_parameters(prm)

        self.server = self.rl['server']
        self.action_translator = Action_translator(prm, self)
        self.i0_costs = 0
        self.car = Battery(prm)
        self.spaces = EnvSpaces(self)
        self.spaces.new_state_space(self.rl['state_space'])
        self.add_noise = 1 if self.rl['deterministic'] == 2 else 0
        for data in [
            "competitive", "n_grdC_level", "offset_reward", "delta_reward"
        ]:
            self.__dict__[data] = self.rl[data]
        self.share_flexs = prm['loads']['share_flexs']
        self.opt_res_file = self.prm['paths']['opt_res_file']
        self.res_path = prm['paths']['opt_res']
        self.slid_day = False

        if self.rl['type_learning'] == 'facmac':
            self.action_space = self.rl['action_space']
            self.observation_space = []
            for _ in self.homes:
                self.observation_space.append(spaces.Box(
                    low=-np.inf, high=+np.inf,
                    shape=(self.rl['obs_shape'],),
                    dtype=np.float32))

        self.max_delay = int(
            prm["loads"]["max_delay"] * self.n_int_per_hr
        )

        self.hedge = HEDGE(
            n_homes=prm['ntw']['n'],
            factors0=prm['syst']['f0'],
            clusters0=prm['syst']['clus0'],
            prm=prm
        )
        if prm['ntw']['nP'] > 0:
            self.passive_hedge = HEDGE(
                n_homes=prm['ntw']['nP'],
                factors0=prm['syst']['f0'],
                clusters0=prm['syst']['clus0'],
                prm=prm
            )

    def reset(
            self,
            seed: int = 0,
            load_data: bool = False,
            passive: bool = False,
            E_req_only: bool = False
    ) -> Tuple[str, dict]:
        """Reset environment for new day with new data."""
        if seed is not None:
            self.envseed = self._seed(seed)
            self.random_seeds_use[seed] = 0

        # different agent caracteristics for passive and active homes
        self.set_passive_active(passive)

        # initialise environment time
        self.date = self.date0
        # date0 is not the total date0 for all runs,
        # but the specific date0 for this run
        # as defined in learning.py at the same time as i0_costs
        self.time = 0  # hrs since start
        self.steps_beyond_done = None
        self.done = False
        self.idt = 0 if self.date0.weekday() < 5 else 1
        self.idt0 = 0 if (self.date0 - timedelta(days=1)).weekday() < 5 else 1

        # data management
        self.load_data = load_data
        self.dloaded = 0
        self.add_noise = False

        # update grid costs
        self._update_grid_costs()

        self.batch_file = self.batchfile0
        self.save_file = self.batch_file
        self.no_name_file = str(int(seed))
        self.factors, self.clusters = [initialise_dict(self.homes) for _ in range(2)]
        if self.load_data:
            self.factors = np.load(
                self.res_path / f"factors{self._file_id()}",
                allow_pickle=True).item()
            self.clusters = np.load(
                self.res_path / f"clusters{self._file_id()}",
                allow_pickle=True).item()
        else:
            for home in self.homes:
                self.factors[home] = initialise_dict(self.data_types)
                self.clusters[home] = initialise_dict(self.behaviour_types)

        # initialise heating and battery objects
        self.heat = Heat(self.prm, self.i0_costs, self.passive_ext, E_req_only)
        self.spaces.E_heat_min_max = self.heat.E_heat_min_max

        self.car.reset(self.prm, self.passive_ext)
        self.action_translator.heat = self.heat
        self.action_translator.car = self.car

        # initialise demand ahead (2 days)
        self.batch = {}
        self.car.batch = {}
        self._initialise_batch_entries()

        if not load_data or (load_data and self.add_noise):
            self._initialise_new_data()

        for _ in range(2):
            self._load_next_day()

        self.car.add_batch(self.batch)
        self.batch = self.car.compute_bat_dem_agg(self.batch)
        self._loads_test()

        return self.save_file, self.batch

    def reinitialise_envfactors(
            self, date0, epoch, i_explore, evaluation_add1=False
    ):
        """Reinitialise factors and clusters lists."""
        if evaluation_add1:
            i_explore += 1

        random_clus, random_f = [[[
            self.np_random.rand()
            for _ in range(self.prm['ntw']['n_all'])]
            for _ in labels]
            for labels in [self.behaviour_types, self.data_types]
        ]

        i_day_type = 0 if date0.weekday() < 5 else 1
        next_dt = 0 if (date0 + timedelta(days=1)).weekday() < 5 else 1
        transition_type = self.labels_day_trans[i_day_type * 2 + next_dt * 1]

        for passive_ext in ['', 'P']:
            for i_data_type, data_type in enumerate(self.behaviour_types):
                day_type = self.prm["syst"]["labels_day"][i_day_type]
                clusas = []
                da = self.prm['ntw']['n'] if passive_ext == 'P' else 0
                transition_type_ = self._p_trans_label(transition_type, data_type)
                for home in range(self.prm['ntw']['n' + passive_ext]):
                    if epoch == 0 and i_explore == 0:
                        psclus = self.p_clus[data_type][day_type]
                    else:
                        clus = self.clus[f"{data_type}{passive_ext}"][home]
                        psclus = self.p_trans[data_type][transition_type_][clus]
                    choice = self._ps_rand_to_choice(
                        psclus, random_clus[i_data_type][home + da])
                    clusas.append(choice)
                self.clus[data_type + passive_ext] = clusas

            if epoch == 0 and i_explore == 0:
                for data_type in self.data_types:
                    self.f[data_type + passive_ext] = [
                        self.prm["syst"]["f0"][data_type]
                        for _ in range(self.prm["ntw"]["n" + passive_ext])
                    ]
            else:
                ia0 = self.prm["ntw"]["n"] if passive_ext == "P" else 0
                iaend = ia0 + self.prm["ntw"]["nP"] if passive_ext == "P" \
                    else ia0 + self.prm["ntw"]["n"]
                self._next_factors(
                    passive_ext=passive_ext, transition_type=transition_type,
                    rands=[
                        random_f[i_data_type][ia0: iaend]
                        for i_data_type in range(len(self.data_types))
                    ]
                )

    def set_passive_active(self, passive: bool = False):
        """Update environment properties for passive or active case."""
        self.passive_ext = 'P' if passive else ''
        for e in ['cap', 'T_LB', 'T_UB', 'T_req', 'store0',
                  'mincharge', 'car', 'loads', 'gen']:
            # set variables for passive or active case
            self.__dict__[e + '_p'] = e + self.passive_ext
        self.coeff_T = self.prm['heat']['T_coeff' + self.passive_ext]
        self.coeff_Tair = self.prm['heat']['T_air_coeff' + self.passive_ext]
        self.n_homes = self.prm['ntw']['n' + self.passive_ext]
        self.homes = range(self.n_homes)
        for e in ['cap_p', 'store0_p', 'mincharge_p', 'n_homes']:
            self.action_translator.__dict__[e] = self.__dict__[e]
            self.spaces.__dict__[e] = self.__dict__[e]
        self.T_air = [self.prm['heat']['T_req' + self.passive_ext][home][0]
                      for home in self.homes]

    def update_date(self, i0_costs: int, date0: datetime = None):
        """Update new date for new day."""
        self.i0_costs = i0_costs
        if date0 is not None:
            self.date0 = date0
            self.spaces.current_date0 = self.date0
            self.action_translator.date0 = self.date0
            self.date_end = date0 + timedelta(hours=self.N * self.dt)
            self.car.date0 = self.date0
            self.car.date_end = self.date_end

    def fix_data_a(self, homes, file_id, its=0):
        """Recompute data for home a that is infeasible."""
        self._seed(self.envseed[0] + its)
        for home in homes:
            self.factors[home] = initialise_dict(
                [self.loads_p, self.gen_p, self.car_p]
            )
            self.clusters[home] = initialise_dict([self.loads_p, self.car_p])
        self._initialise_batch_entries(homes)
        date_load = self.date0
        self.dloaded = 0
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day(homes=homes)
            date_load += timedelta(days=1)
        self.car.add_batch(self.batch)
        self.batch = self.car.compute_bat_dem_agg(self.batch)
        np.save(self.res_path / f"batch{file_id}", self.batch)
        np.save(self.res_path / f"clusters{file_id}", self.clusters)
        np.save(self.res_path / f"factors{file_id}", self.factors)

    def update_flex(
            self,
            cons_flex: list,
            opts: list = None
    ) -> np.ndarray:
        """Given step flexible consumption, update remaining flexibility."""
        if opts is None:
            h = self._get_time_step()
            n_homes = self.n_homes
            batch_flex = [self.batch[home]['flex'] for home in range(n_homes)]
        else:
            h, batch_flex, max_delay, n_homes = opts

        assert np.shape(batch_flex)[0] == self.n_homes, \
            f"np.shape(batch_flex) {np.shape(batch_flex)} " \
            f"self.n_homes {self.n_homes}"
        assert np.shape(batch_flex)[2] == self.max_delay + 1, \
            f"np.shape(batch_flex) {np.shape(batch_flex)} " \
            f"self.max_delay {self.max_delay}"

        new_batch_flex = np.array(
            [
                [copy.deepcopy(batch_flex[home][ih]) for ih in range(h, h + 2)]
                for home in range(n_homes)
            ]
        )

        for home in range(n_homes):
            remaining_cons = max(cons_flex[home], 0)
            assert cons_flex[home] <= np.sum(batch_flex[home][h][1:]) + 1e-3, \
                f"cons_flex[home] {cons_flex[home]} " \
                f"> np.sum(batch_flex[home][h][1:]) {np.sum(batch_flex[home][h][1:])} + 1e-3"

            # remove what has been consumed
            for i_flex in range(1, self.max_delay + 1):
                delta_cons = min(new_batch_flex[home, 0, i_flex], remaining_cons)
                remaining_cons -= delta_cons
                new_batch_flex[home, 0, i_flex] -= delta_cons
            assert remaining_cons <= 1e-2, \
                f"remaining_cons = {remaining_cons} too large"

            # move what has not been consumed to one step more urgent
            self._new_flex_tests(batch_flex, new_batch_flex, h, home)

        return new_batch_flex

    def step(
            self, action: list, implement: bool = True,
            record: bool = False, evaluation: bool = False,
            netp_storeout: bool = False, E_req_only: bool = False
    ) -> list:
        """Compute environment updates and reward from selected action."""
        h = self._get_time_step()
        homes = self.homes
        batch_flex = [self.batch[home]['flex'] for home in homes]
        self._batch_tests(batch_flex, h)

        # update batch if needed
        daynumber = (self.date - self.date0).days
        if h == 1 and self.time > 1 \
                and self.dloaded < daynumber + 2 == 0 \
                and not self.slid_day:
            self._load_next_day()
            self.slid_day = True
        self.car.add_batch(self.batch)

        if h == 2:
            self.slid_day = False
        home_vars, loads, constraint_ok = self.policy_to_rewardvar(
            action, E_req_only=E_req_only)
        if not constraint_ok:
            print('constraint false not returning to original values')
            return [None, None, None, None, None, constraint_ok, None]

        else:
            reward, break_down_rewards = self.get_reward(
                home_vars['netp'], evaluation=evaluation)
            self.netps = home_vars['netp']

            # ----- update environment variables and state
            new_batch_flex = self.update_flex(loads['flex_cons'])
            next_date = self.date + timedelta(seconds=60 * 60 * self.dt)
            next_done = next_date == self.date_end
            inputs_next_state = [
                self.time + 1, next_date, next_done, new_batch_flex, self.car.store
            ]
            next_state = self.get_state_vals(inputs=inputs_next_state) \
                if not self.done \
                else [None for _ in homes]
            if implement:
                for home in homes:
                    batch_flex[home][h: h + 2] = new_batch_flex[home]
                self.time += 1
                self.date = next_date
                self.idt = 0 if self.date.weekday() < 5 else 1
                self.done = next_done
                self.heat.update_step()
                self.car.update_step(time_step=self.time)

            for ih in range(h + 1, h + self.N):
                assert all(
                    self.batch[home]['loads'][ih]
                    <= batch_flex[home][ih][0] + batch_flex[home][ih][-1] + 1e-3
                    for home in self.homes
                ), f"h {h} ih {ih} self.batch[home]['loads'][ih] {self.batch[home]['loads'][ih]} " \
                   f"batch_flex[home][ih] {batch_flex[home][ih]} " \
                   f"len(batch_flex[home]) {len(batch_flex[home])}"

            if record or evaluation:
                loads_fixed = [sum(batch_flex[home][h][:]) for home in homes] \
                    if self.date == self.date_end - timedelta(hours=2 * self.dt) \
                    else [batch_flex[home][h][0] for home in homes]

            if record:
                loads_flex = np.zeros(self.n_homes) if self.date == self.date_end \
                    else [sum(batch_flex[home][h][1:]) for home in homes]
                record_output = \
                    [home_vars['netp'], self.car.discharge, action, reward,
                     break_down_rewards, self.car.store, loads_flex, loads_fixed,
                     home_vars['tot_cons'].copy(), loads['tot_cons_loads'].copy(),
                     self.heat.tot_E.copy(), self.heat.T.copy(),
                     self.heat.T_air.copy(), self.grdC[self.time].copy(),
                     self.wholesale[self.time].copy(),
                     self.cintensity[self.time].copy()]
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok, record_output]
            elif netp_storeout:
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok,
                        [home_vars['netp'], self.car.discharge_tot,
                         self.car.charge]]
            else:
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok, None]

    def get_reward(
            self,
            netp: list,
            discharge_tot: list = None,
            charge: list = None,
            passive_vars: list = None,
            time_step: int = None,
            evaluation: bool = False
    ) -> Tuple[list, list]:
        """Compute reward from netp and battery charge at time step."""
        if passive_vars is not None:
            netp0, discharge_tot0, charge0 = passive_vars
        elif self.cap_p == 'capP':
            netp0, discharge_tot0, charge0 = [
                [0 for _ in range(self.prm['ntw']['nP'])] for _ in range(3)
            ]
        else:
            seconds_per_interval = 3600 * 24 / self.prm['syst']['H']
            hour = int((self.date - self.date0).seconds / seconds_per_interval)
            netp0, discharge_tot0, charge0 = [
                [self.prm['loads'][e][home][hour]
                 for home in range(self.prm['ntw']['nP'])]
                for e in ['netp0', 'discharge_tot0', 'charge0']]
        time_step = self.time if time_step is None else time_step
        if discharge_tot is None:
            discharge_tot = self.car.discharge_tot
        charge = self.car.charge if charge is None else charge
        grdCt, wholesalet, cintensityt = [
            self.grdC[time_step], self.wholesale[time_step], self.cintensity[time_step]
        ]

        # negative netp is selling, positive buying
        grid = sum(netp) + sum(netp0)
        if self.prm['ntw']['charge_type'] == 0:
            sum_netp = sum([abs(netp[home]) if netp[home] < 0
                           else 0 for home in self.homes])
            sum_netp0 = sum([abs(netp0[home]) if netp0[home] < 0
                             else 0 for home in range(len(netp0))])
            netpvar = sum_netp + sum_netp0
            dc = self.prm['ntw']['C'] * netpvar
        else:
            netpvar = sum([netp[home] ** 2 for home in self.homes]) \
                + sum([netp0[home] ** 2 for home in range(len(netp0))])
            dc = self.prm['ntw']['C'] * netpvar
        gc = grdCt * (grid + self.prm['grd']['loss'] * grid ** 2)
        gc_a = [wholesalet * netp[home] for home in self.homes]
        sc = self.prm['car']['C'] \
            * (sum(discharge_tot[home] + charge[home]
                   for home in self.homes)
                + sum(discharge_tot0[home] + charge0[home]
                      for home in range(len(discharge_tot0))))
        sc_a = [self.prm['car']['C'] * (discharge_tot[home] + charge[home])
                for home in self.homes]
        c_a = [g + s for g, s in zip(gc_a, sc_a)]
        reward = - (gc + sc + dc)
        costs_wholesale = wholesalet * grid
        costs_losses = wholesalet * self.prm['grd']['loss'] * grid ** 2
        emissions = cintensityt * (grid + self.prm['grd']['loss'] * grid ** 2)
        emissions_from_grid = cintensityt * grid
        emissions_from_loss = cintensityt * self.prm['grd']['loss'] * grid ** 2
        break_down_rewards = [gc, sc, dc, costs_wholesale, costs_losses,
                              emissions, emissions_from_grid,
                              emissions_from_loss, gc_a, sc_a, c_a]

        if self.offset_reward:
            reward -= self.delta_reward

        return reward, break_down_rewards

    def get_loads_fixed_flex_gen(self, date, time_step):
        batch_flex = [self.batch[home]['flex'][time_step] for home in self.homes]
        loads, home_vars = {}, {}
        if date == self.date_end - timedelta(hours=self.dt):
            loads['l_flex'] = np.zeros(self.n_homes)
            loads['l_fixed'] = np.array(
                [sum(batch_flex[home]) for home in self.homes]
            )
        else:
            loads['l_flex'] = np.array(
                [sum(batch_flex[home][1:]) for home in self.homes]
            )
            loads['l_fixed'] = np.array(
                [batch_flex[home][0] for home in self.homes]
            )

        home_vars['gen'] = np.array([self.batch[home]['gen'][time_step] for home in self.homes])

        return loads, home_vars

    def policy_to_rewardvar(
            self,
            action: list,
            other_input: list = None,
            E_req_only: bool = False
    ):
        """Given selected action, obtain results of the step."""
        home_vars, loads = [{} for _ in range(2)]
        if other_input is None:
            date = self.date
            h = self._get_time_step()
            loads, home_vars = self.get_loads_fixed_flex_gen(date, h)

            if action is None:
                return None
            try:
                for home in self.homes:
                    if action[home] is None:
                        print(f'action[{home}] is None, action = {action[home]}')
            except Exception as ex:
                print(f"ex {ex}")
            self.heat.current_temperature_bounds(h)

        else:
            date, action, gens, loads = other_input
            gens = np.array(gens)
            self.date = date
            h = self._get_time_step()
            home_vars = {'gen': gens}
        self.heat.E_heat_min_max(h)
        last_step = True \
            if date == self.date_end - timedelta(hours=self.dt) \
            else False
        bool_penalty = self.car.min_max_charge_t(h, date)
        self.heat.potential_E_flex()

        #  ----------- meet consumption + check constraints ---------------
        constraint_ok = True
        loads, home_vars, bool_penalty = \
            self.action_translator.actions_to_env_vars(loads, home_vars, action, date, h)
        for home in self.homes:
            assert home_vars['tot_cons'][home] + 1e-3 >= loads['l_fixed'][home], \
                f"home = {home}, no flex cons at last time step"
            assert loads['l_fixed'][home] \
                   >= self.batch[home]['loads'][h] * (1 - self.share_flexs[home]), \
                   f"home {home} l_fixed and batch do not match"

        self.heat.next_T(update=True)
        self._check_constraints(
            bool_penalty, date, loads, E_req_only, h, last_step, home_vars)

        if sum(bool_penalty) > 0:
            constraint_ok = False

        return home_vars, loads, constraint_ok

    def get_state_vals(
            self,
            descriptors: list = None,
            inputs: list = None
    ) -> np.ndarray:
        """
        Get values corresponding to array of descriptors inputted.

        (before translation into index)
        """
        if inputs is None:
            t, date, done, store = [
                self.time, self.date, self.done, self.car.store
            ]
            batch_flex_h = [
                self.batch[home]['flex'][self.step] for home in self.homes
            ]
        else:
            date = inputs[1]

        inputs_ = [t, date, done, batch_flex_h, store] if inputs is None \
            else inputs

        idt = 0 if date.weekday() < 5 else 1
        descriptors = descriptors if descriptors is not None \
            else self.spaces.descriptors['state']
        vals = np.zeros((self.n_homes, len(descriptors)))
        time_step = self._get_time_step(date)
        if any(descriptor in ['bool_flex', 'store_bool_flex'] for descriptor in descriptors):
            self.car.update_step(time_step=time_step)
            self.car.min_max_charge_t(time_step, date)
            assert self.car.time_step == time_step, \
                f"self.car.time_step {self.car.time_step} time_step {time_step}"
            self.heat.E_heat_min_max(time_step)
            loads, home_vars = self.get_loads_fixed_flex_gen(date, time_step)
            self.action_translator.initial_processing(loads, home_vars)
        for home in self.homes:
            for i, descriptor in enumerate(descriptors):
                vals[home, i] = self._descriptor_to_val(
                    descriptor, inputs_, time_step, idt, home
                )
        if any(descriptor in ['bool_flex', 'store_bool_flex'] for descriptor in descriptors):
            self.car.revert_last_update_step()

        return vals

    def _seed(self, seed=None):
        if seed is not None and not isinstance(seed, integer_types):
            seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _p_trans_label(self, transition_type, data_type):
        if transition_type == "wd2wd" and "wd2wd" not in self.p_trans[data_type]:
            transition_type_ = "wd"
        elif transition_type == "we2we" and "we2we" not in self.p_trans[data_type]:
            transition_type_ = "we"
        else:
            transition_type_ = transition_type

        return transition_type_

    def _file_id(self):
        return f"{self.no_name_file}{self.passive_ext}{self.opt_res_file}"

    def _ps_rand_to_choice(self, ps: list, rand: float) -> int:
        """Given probability of each choice and a random number, select."""
        p_intervals = [sum(ps[0:i]) for i in range(len(ps))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]
        return choice

    def _next_factors(self, passive_ext=None, transition_type=None, rands=None, homes=[]):
        """Compute self-correlated random scaling factors for data profiles."""
        homes = range(self.prm['ntw']['n' + passive_ext]) if len(homes) == 0 else homes
        if passive_ext is None:
            passive_ext = '' if self.car_p == 'car' else 'P'
        transition_type = transition_type if transition_type is not None \
            else self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
        transition_type_ = transition_type[0:2] \
            if transition_type not in self.residual_distribution_prms["loads"] \
            else transition_type
        factor_ev_new_interval = np.zeros((len(homes),))
        for i_home in range(len(homes)):
            home = homes[i_home]
            # factor for demand - differentiate between day types
            df = norm.ppf(
                rands[0][i_home],
                *list(self.residual_distribution_prms["loads"][transition_type_])
            )
            self.f['loads' + passive_ext][home] = \
                self.f["loads" + passive_ext][home] \
                + df \
                - self.mean_residual["loads"][transition_type_]
            # factor for generation - without differentiation between day types
            df = norm.ppf(
                rands[2][i_home], *self.residual_distribution_prms["gen"]
            )
            self.f["gen" + passive_ext][home] = \
                self.f["gen" + passive_ext][home] + df - self.mean_residual["gen"]

            # factor for EV consumption
            fs_brackets = self.prm['car']['fs_brackets'][transition_type_]
            current_interval = \
                [i for i in range(self.prm['car']['intervals_fprob'] - 1)
                 if fs_brackets[i] <= self.f['car' + passive_ext][home]][-1]

            if self.cluss[home]["car"][-1] == self.n_clus["car"]:
                # no trip day the day before
                probs = self.prm["car"]["f_prob_zero2pos"][transition_type_]
            else:
                probs = self.prm["car"]["f_prob_pos"][transition_type_][current_interval]
            choice = self._ps_rand_to_choice(
                probs, rands[1][i_home]
            )
            factor_ev_new_interval[i_home] = choice
            self.f["car" + passive_ext][home] \
                = self.prm["car"]["mid_fs_brackets"][transition_type_][int(choice)]
            for data_type in ["loads", "car"]:
                self.f[f"{data_type}{passive_ext}"][home] = min(
                    max(self.f_min[data_type], self.f[f"{data_type}{passive_ext}"][home]),
                    self.f_max[data_type]
                )
            i_month = self.date.month - 1
            self.f[f"gen{passive_ext}"][home] = min(
                max(self.f_min["gen"][i_month], self.f[f"gen{passive_ext}"][home]),
                self.f_max["gen"][i_month]
            )

        return factor_ev_new_interval

    def _get_next_clusters(self, transition_type, homes):
        for home in homes:
            for data_type in self.behaviour_types:
                # get next cluster
                transition_type_ = \
                    transition_type[0:2] \
                    if transition_type not in self.p_trans[data_type] \
                    else transition_type
                clus_a = self.clus[data_type + self.passive_ext][home]
                ps = self.p_trans[data_type][transition_type_][clus_a]
                cump = [sum(ps[0:i]) for i in range(1, len(ps))] + [1]
                rdn = self.np_random.rand()
                self.clus[data_type + self.passive_ext][home] = \
                    [c > rdn for c in cump].index(True)
                self.cluss[home][data_type].append(self.clus[data_type + self.passive_ext][home])

    def _adjust_car_cons(self, homes, dt, transition_type, day, i_car, factor_ev_new_interval):
        transition_type_ = transition_type[0:2] \
            if transition_type not in self.prm['car']['mid_fs_brackets'] \
            else transition_type
        for i_home in range(len(homes)):
            home = homes[i_home]
            clus = self.clus[self.car_p][home]
            it = 0
            while (
                    np.max(day['loads_car'][i_home])
                    > self.prm['car'][self.cap_p][home]
                    and it < 100
            ):
                if factor_ev_new_interval[i_home] > 0:
                    factor_ev_new_interval[i_home] -= 1
                    interval = int(factor_ev_new_interval[i_home])
                    self.f[self.car_p][home] = \
                        self.prm["car"]["mid_fs_brackets"][transition_type_][interval]
                    prof = self.prof['car']['cons'][dt][clus][i_car[i_home]]
                    day['loads_car'][i_home] = [
                        x * self.f[self.car_p][home] for x in prof
                    ]
                else:
                    i_car[i_home] = self.np_random.choice(
                        np.arange(self.n_prof['car'][dt][clus]))
                it += 1

        return day, i_car

    # def _generate_new_day(self, homes: list):
    #     """If new day of data was not presaved, load here."""
    #     # intialise variables
    #     homes = self.homes if len(homes) == 0 else homes
    #     day = {}
    #     dt = self.prm['syst']['labels_day'][self.idt]
    #     transition_type = self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
    #     loads_p = self.loads_p if self.loads_p in self.factors[0] \
    #         else 'lds' + self.loads_p[5:]
    #
    #     # save factors and clusters at the start of the episode
    #     for home in homes:
    #         for e in [loads_p, self.gen_p, self.car_p]:
    #             self.factors[home][e].append(self.f[e][home])
    #         for e in [self.loads_p, self.car_p]:
    #             self.clusters[home][e].append(self.clus[e][home])
    #
    #     # get next clusters (for load and car)
    #     self._get_next_clusters(transition_type, homes)
    #
    #     # get load profile indexes, normalised profile, and scaled profile
    #     i_prof_load = self._compute_i_profs('loads', dt, homes=homes)
    #     cluss = [self.clus[self.loads_p][home] for home in homes]
    #     load_prof = [
    #         self.prof["loads"][dt][clus][i_prof]
    #         for clus, i_prof in zip(cluss, i_prof_load)
    #     ]
    #     day["loads"] = \
    #         [load_prof[i_home] * self.f[self.loads_p][home]
    #          if self.prm["loads"]["own_loads"][home]
    #          else [0 for _ in range(self.N)]
    #          for i_home, home in enumerate(homes)]
    #
    #     # get PV profile index, and day profile
    #     month = self.date.month
    #     while not self.n_prof['gen'][month - 1] > 0:
    #         month += 1
    #         month = 1 if month == 12 else month
    #     i_prof_gen = self._compute_i_profs('gen', idx_month=month - 1, homes=homes)
    #     day['gen'] = [[g * self.f[self.gen_p][home]
    #                    for g in self.prof['gen'][month - 1][i_prof_gen[i_home]]]
    #                   if self.prm['gen']['own_PV'][home]
    #                   else [0 for _ in range(self.N)]
    #                   for i_home, home in zip(range(len(homes)), homes)]
    #
    #     # get car cons factor, profile index, normalised profile, scaled profile
    #     factor_ev_new_interval = self._next_factors(
    #         transition_type=transition_type,
    #         rands=[[self.np_random.rand() for _ in range(len(homes))]
    #                for _ in range(len(self.data_types))],
    #         homes=homes)
    #     i_car = self._compute_i_profs('car', dt=dt, homes=homes)
    #     prof = [
    #         self.prof['car']['cons'][dt][self.clus[self.car_p][home]][i_car[i_home]]
    #         for i_home, home in enumerate(homes)
    #     ]
    #     day['loads_car'] = \
    #         [[x * self.f[self.car_p][home] if self.prm['car']['own_car'][home]
    #           else 0 for x in prof[i_home]]
    #          for i_home, home in enumerate(homes)]
    #
    #     # check car consumption is not larger than capacity - if so, correct
    #     day, i_car = self._adjust_car_cons(
    #         homes, dt, transition_type, day, i_car, factor_ev_new_interval
    #     )
    #
    #     # get car availability profile
    #     day['avail_car'] = \
    #         [self.prof['car']['avail'][dt][self.clus[self.car_p][home]][i_car[i_home]]
    #          for i_home, home in zip(range(len(homes)), homes)]
    #     for i_home in range(len(homes)):
    #         if sum(day['loads_car'][i_home]) == 0 and sum(day["avail_car"][i_home]) == 0:
    #             day["avail_car"][i_home] = np.ones(self.prm["syst"]["N"])
    #     for i_home, home in enumerate(homes):
    #         for e in day.keys():
    #             self.batch[home][e] = self.batch[home][e] + list(day[e][i_home])
    #     self._loads_to_flex(homes)
    #     self.dloaded += 1
    #
    #     assert len(self.batch[0]['avail_car']) > 0, "empty avail_car batch"

    def _load_next_day(self, homes: list = []):
        """
        Load next day of data.

        Either it is not presaved and needs to be generated,
        or it can just be loaded
        """
        if not self.load_data or len(homes) > 0:
            homes = homes if len(homes) > 0 else self.homes
            if self.passive_ext == '':
                day = self.hedge.make_next_day(homes)
            else:
                day = self.passive_hedge.make_next_day(homes)

            for i_home, home in enumerate(homes):
                for e in day.keys():
                    self.batch[home][e] = self.batch[home][e] + list(day[e][i_home])
            self._loads_to_flex(homes)
            # if len(homes) == 0:
            self.dloaded += 1

            assert len(self.batch[0]['avail_car']) > 0, "empty avail_car batch"

        else:
            for e in ['batch', 'clusters', 'factors']:
                self.__dict__[e] = np.load(
                    self.res_path / f"{e}{self._file_id()}",
                    allow_pickle=True).item()

            loads_p = self.loads_p \
                if self.loads_p in self.factors[0] \
                else 'lds' + self.loads_p[5:]
            self.dloaded += len(self.factors[0][loads_p])

        assert len(self.batch) > 0, "empty batch"
        assert len(self.batch[0]['avail_car']) > 0, "empty avail_car batch"

    def _compute_i_profs(
            self,
            data_type: str,
            dt: str = None,
            idx_month: int = None,
            homes: list = None
    ) -> list:
        """Get random indexes for profile selection."""
        homes = self.homes if len(homes) == 0 else homes
        i_profs = []
        n_profs = self.n_prof[data_type][dt] \
            if dt is not None \
            else [self.n_prof[data_type][idx_month]]
        n_profs = [int(self.prm['syst']['share_centroid'] * n_prof)
                   for n_prof in n_profs]
        available_profiles = \
            [[i for i in range(n_prof)] for n_prof in n_profs]
        for home in homes:
            if data_type in self.labels_clus:
                clus = self.clus[self.__dict__[data_type + '_p']][home]
                avail_prof = available_profiles[clus]
            else:
                avail_prof = available_profiles[0]
            i_prof = self.np_random.choice(avail_prof)

            if len(avail_prof) > 1:
                avail_prof.remove(i_prof)
            i_profs.append(i_prof)

        return i_profs

    def _loads_to_flex(self, homes: list = None):
        """Apply share of flexible loads to new day loads data."""
        homes = self.homes if len(homes) == 0 else homes
        for home in homes:
            dayflex_a = np.zeros((self.N, self.max_delay + 1))
            for t in range(self.N):
                try:
                    tot_t = self.dloaded * self.prm["syst"]["N"] + t
                    loads_t = self.batch[home]["loads"][tot_t]
                except Exception as ex:
                    print(f"ex {ex} t {t} self.dloaded {self.dloaded} home {home}")
                    print(f"np.shape(self.batch[home]['loads']) {np.shape(self.batch[home]['loads'])}")
                    print(f"self.prm['syst']['N'] {self.prm['syst']['N']} self.N {self.N}")
                dayflex_a[t, 0] = (1 - self.share_flexs[home]) * loads_t
                dayflex_a[t, self.max_delay] = self.share_flexs[home] * loads_t
            self.batch[home]['flex'] = np.concatenate(
                (self.batch[home]['flex'], dayflex_a)
            )

            assert np.shape(self.batch[home]["flex"])[1] == self.max_delay + 1, \
                f"shape batch[{home}]['flex'] {np.shape(self.batch[home]['flex'])} " \
                f"self.max_delay {self.max_delay}"

    def _get_time_step(self, date: datetime = None) -> int:
        """Given date, obtain hour."""
        date = self.date if date is None else date
        time_elapsed = date - self.date0
        h = int(
            time_elapsed.days * self.prm["syst"]["H"]
            + time_elapsed.seconds / (60 * 60) * self.n_int_per_hr
        )

        return h

    def _check_loads(
        self,
        home: int,
        date: datetime,
        h: int,
        loads: dict,
        bool_penalty: List[bool]
    ) -> List[bool]:
        """Check load-related constraints for given home after step."""
        flex_cons, l_fixed = [loads[e] for e in ['flex_cons', 'l_fixed']]

        if date == self.date_end - timedelta(hours=self.dt) \
                and flex_cons[home] > 1e-2:
            print(f"home = {home}, flex_cons[home] = {flex_cons[home]}")
            bool_penalty[home] = True

        if loads['l_flex'][home] > 1e2:
            print(f"h = {h}, home = {home}, l_flex[home] = {loads['l_flex'][home]}")
            bool_penalty[home] = True

        return bool_penalty

    def _check_constraints(
            self,
            bool_penalty: List[bool],
            date: datetime,
            loads: dict,
            E_req_only: bool,
            h: int,
            last_step: bool,
            home_vars: dict
    ) -> List[bool]:
        """Given result of the step action, check environment constraints."""
        for home in [home for home, bool in enumerate(bool_penalty) if not bool]:
            self.car.check_constraints(home, date, h)
            self.heat.check_constraints(home, h, E_req_only)
            bool_penalty = self._check_loads(home, date, h, loads, bool_penalty)
            # prosumer balance
            prosumer_balance_sum = \
                abs(home_vars['netp'][home]
                    - (self.car.loss_ch[home] + self.car.charge[home])
                    + self.car.discharge[home]
                    + home_vars['gen'][home]
                    - home_vars['tot_cons'][home])
            if prosumer_balance_sum > 1e-2:
                print(f"home {home} prosumer_balance_sum = {prosumer_balance_sum}")
                print(f"self.car.loss_ch[{home}] = {self.car.loss_ch[home]}")
                print(f"self.car.charge[{home}] = {self.car.charge[home]}")
                print(f"self.car.discharge[{home}] = {self.car.discharge[home]}")
                print(f"home = {home}, loads = {loads}")
                np.save('action_translator_d', self.action_translator.d)
                np.save('action_translator_mu', self.action_translator.action_intervals)
                np.save('action_translator_k', self.action_translator.k)
                bool_penalty[home] = True

            # check tot cons
            if home_vars['tot_cons'][home] < - 1e-2:
                print(f"negative tot_cons {home_vars['tot_cons'][home]} home = {home}")
                bool_penalty[home] = True
            if last_step \
                    and home_vars['tot_cons'][home] < \
                    self.batch[home]['loads'][h] * (1 - self.share_flexs[home]):
                print(f"home = {home}, no flex cons at last time step")
                bool_penalty[home] = True
        # self.car.revert_last_update_step()

        return bool_penalty

    def _descriptor_to_val(
            self,
            descriptor: str,
            inputs: list,
            hour: int,
            idt: int,
            home: int
    ):
        """Given state of action space descriptor, get value."""
        t, date, done, batch_flex_h, store = inputs

        dict_vals = {
            "None": 0,
            "hour": hour % self.prm["syst"]["H"],
            "bat_dem_agg": self.batch[home]["bat_dem_agg"][hour],
            "store0": store[home],
            "grdC": self.grdC[t],
            "day_type": idt,
            "dT": self.prm["heat"]["T_req" + self.passive_ext][home][hour] - self.T_air[home],
        }
        if descriptor in dict_vals:
            val = dict_vals[descriptor]
        elif descriptor == "bool_flex":
            val = self.action_translator.aggregate_action_bool_flex(home)
        elif descriptor == "store_bool_flex":
            val = self.action_translator.store_bool_flex(home)
        elif len(descriptor) >= 4 and descriptor[0:4] == 'grdC':
            val = self.normalised_grdC[t]
        elif descriptor == 'dT_next':
            T_req = self.prm['heat']['T_req' + self.passive_ext][home]
            t_change_T_req = [t for t in range(hour + 1, self.N) if T_req[t] != T_req[hour]]
            if len(t_change_T_req) > 0:
                current_T_req = T_req[hour]
                next_T_req = T_req[t_change_T_req[0]]
                val = (next_T_req - current_T_req) / (t_change_T_req[0] - hour)
            else:
                val = 0
        elif descriptor == 'car_tau':
            val = self.car.car_tau(hour, date, home, store[home])
        elif len(descriptor) > 9 \
                and (descriptor[-9:-5] == 'fact'
                     or descriptor[-9:-5] == 'clus'):
            # scaling factors / profile clusters for the whole day
            module = descriptor.split('_')[0]  # car, loads or gen
            if descriptor.split('_')[-1] == 'prev':
                prev_data = self.factors if descriptor[-9:-5] == 'fact' \
                    else self.clusters
                val = prev_data[home][module][-1]
            else:  # step
                step_data = self.f if descriptor[-9:-5] == 'fact' \
                    else self.clus
                val = step_data[module][home]
        else:  # select current or previous hour - step or prev
            h = self._get_time_step() if descriptor[-4:] == 'step' \
                else self._get_time_step() - 1
            if len(descriptor) > 8 and descriptor[0: len('avail_car')] == 'avail_car':
                val = self.batch[home]['avail_car'][h]
            elif descriptor[0:5] == 'loads':
                val = np.sum(batch_flex_h[home][1])
            else:
                # gen_prod_step / prev and car_cons_step / prev
                batch_type = 'gen' if descriptor[0:3] == 'gen' else 'loads_car'
                val = self.batch[home][batch_type][h]

        if self.rl['normalise_states']:
            val = val / self.spaces.space_info[descriptor]['max']

        return val

    def _batch_tests(self, batch_flex, h):
        if self.test:
            for home in self.homes:
                assert sum(batch_flex[home][h][1: 5]) <= \
                       sum(batch_flex[home][ih][0] / (1 - self.share_flexs[home])
                           * self.share_flexs[home]
                           for ih in range(0, h + 1)), "batch_flex too large h"

                assert sum(batch_flex[home][h + 1][1: 5]) <= sum(
                    batch_flex[home][ih][0]
                    / (1 - self.share_flexs[home]) * self.share_flexs[home]
                    for ih in range(0, h + 2)), "batch_flex too large h + 1"

                for ih in range(h, h + 30):
                    assert self.batch[home]['loads'][ih] \
                           <= batch_flex[home][ih][0] + batch_flex[home][ih][-1] + 1e-3, \
                           "loads larger than with flex"

    def _new_flex_tests(self, batch_flex, new_batch_flex, h, home):
        if self.test:
            assert np.sum(new_batch_flex[home][0][1:5]) <= sum(
                batch_flex[home][ih][0] / (1 - self.share_flexs[home])
                * self.share_flexs[home] for ih in range(0, h)
            ) + 1e-3, "flex too large"
        for i_flex in range(self.max_delay):
            loads_next_flex = new_batch_flex[home][0][i_flex + 1]
            if self.test:
                assert not (
                    0 < i_flex < 4
                    and new_batch_flex[home][1][i_flex] + loads_next_flex
                    > np.sum([batch_flex[home][ih][0] for ih in range(0, h + 1)])
                    / (1 - self.share_flexs[home]) * self.share_flexs[home] + 1e-3
                ), "loads_next_flex error"
            new_batch_flex[home][1][i_flex] += loads_next_flex
            if self.test:
                assert not (
                    loads_next_flex > np.sum([batch_flex[home][ih][0] for ih in range(0, h + 1)])
                ), "loads_next_flex too large"

    def _loads_test(self):
        for home in self.homes:
            for ih in range(self.N):
                assert (self.batch[home]['loads'][ih]
                        <= self.batch[home]['flex'][ih][0]
                        + self.batch[home]['flex'][ih][-1] + 1e-3
                        ), "loads too large"

    def _update_grid_costs(self):
        grd = self.prm['grd']
        self.grdC = \
            grd['Call'][self.i0_costs: self.i0_costs + self.N + 1]
        self.wholesale = \
            grd['wholesale_all'][self.i0_costs: self.i0_costs + self.N + 1]
        self.cintensity = \
            grd['cintensity_all'][self.i0_costs: self.i0_costs + self.N + 1]
        i_grdC_level = [
            i for i in range(len(self.spaces.descriptors['state']))
            if self.spaces.descriptors['state'][i] == 'grdC_level'
        ]
        if len(i_grdC_level) > 0:
            self.normalised_grdC = \
                [(gc - min(self.grdC[0: self.N]))
                 / (max(self.grdC[0: self.N]) - min(self.grdC[0: self.N]))
                 for gc in self.grdC[0: self.N + 1]]
            if not (self.spaces.type_env == "continuous"):
                self.spaces.brackets['state'][i_grdC_level[0]] = [
                    [np.percentile(self.normalised_grdC, i * 100 / self.n_grdC_level)
                     for i in range(self.n_grdC_level)] + [1]
                    for _ in self.homes
                ]

    def _initialise_new_data(self):
        # we have not loaded data from file -> save new data
        date_load = self.date0

        # date_end is not max date end but date end based on
        # current date0 and duration as specified in learning.py
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day()
            date_load += timedelta(days=1)
        self.batch = self.car.compute_bat_dem_agg(self.batch)

        for e in ['batch', 'clusters', 'factors']:
            file_id = f"{e}{self._file_id()}"
            np.save(self.res_path / file_id, self.__dict__[e])

        self._initialise_batch_entries()

        self.batch_file = self.save_file
        self.load_data = True
        self.dloaded = 0
        self.add_noise = False

    def _initialise_batch_entries(self, homes=[]):
        if len(homes) == 0:
            homes = self.homes
        for home in homes:
            self.batch[home] = initialise_dict(self.batch_entries)
            self.car.batch[home] = initialise_dict(self.car.batch_entries)
            self.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))
            self.car.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))
            self.car.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))

    def _init_factors_clusters_profiles_parameters(self, prm):
        for data in ["f_min", "f_max", "residual_distribution_prms", "mean_residual",
                     "n_clus", "p_clus", "p_trans", "n_prof",
                     "N", "n_int_per_hr", "dt", "behaviour_types",
                     "data_types", "labels_day_trans"
                     ]:
            if data in prm["syst"]:
                self.__dict__[data] = prm["syst"][data]
        self.cluss = initialise_dict(self.homes, "empty_dict")
        for home in self.homes:
            for data_type in self.behaviour_types:
                clus0 = prm["syst"]["clus0"][data_type][home] \
                    if isinstance(prm["syst"]["clus0"][data_type], list) \
                    else prm["syst"]["clus0"][data_type]
                self.cluss[home][data_type] = [clus0]
