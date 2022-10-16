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
from home_components.battery import Battery
from home_components.heat import Heat
from scipy.stats import gamma
from simulations.action_translator import Action_translator
from six import integer_types
from utilities.env_spaces import EnvSpaces
from utilities.userdeftools import initialise_dict


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
        self.batch_entries = ['loads', 'gen', 'loads_EV', 'avail_EV', 'flex']
        self.clus, self.f = {}, {}
        self.labels = ['loads', 'bat', 'gen']
        self.labels_clus = ['loads', 'bat']

        self.prm = prm
        self.rl = prm['RL']
        self.labels_day_trans = prm['syst']['labels_day_trans']

        self.prof = profiles
        self.n_homes = prm['ntw']['n']
        self.homes = range(self.n_homes)
        self.N = prm['syst']['N']

        # initialise parameters
        self._init_factors_clusters_profiles_parameters(prm)

        self.server = self.rl['server']
        self.action_translator = Action_translator(prm, self)
        self.i0_costs = 0
        self.spaces = EnvSpaces(self)
        self.spaces.new_state_space(self.rl['state_space'])
        self.add_noise = 1 if self.rl['deterministic'] == 2 else 0
        for e in ['competitive', 'n_grdC_level',
                  'offset_reward', 'delta_reward']:
            self.__dict__[e] = self.rl[e]
        self.opt_res_file = self.prm['paths']['opt_res_file']
        self.res_path = prm['paths']['res_path']
        self.bat = Battery(prm)
        self.slid_day = False

        if self.rl['type_learning'] == 'facmac':
            self.action_space = self.rl['action_space']
            self.observation_space = []
            for _ in self.homes:
                self.observation_space.append(spaces.Box(
                    low=-np.inf, high=+np.inf,
                    shape=(self.rl['obs_shape'],),
                    dtype=np.float32))

        self.max_delay = self.prm['loads']['max_delay']

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

        # different agent caracteristics for passive and active agents
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
        self.nonamefile = str(int(seed))
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
                self.factors[home] = initialise_dict(
                    [self.loads_p, self.gen_p, self.bat_p])
                self.clusters[home] = initialise_dict([self.loads_p, self.bat_p])

        # initialise heating and battery objects
        self.heat = Heat(self.prm, self.i0_costs, self.passive_ext, E_req_only)
        self.bat.reset(self.prm, self.passive_ext)
        self.action_translator.heat = self.heat
        self.action_translator.bat = self.bat

        # initialise demand ahead (2 days)
        self.batch = {}
        self.bat.batch = {}
        self._initialise_batch_entries()

        if not load_data or (load_data and self.add_noise):
            self._initialise_new_data()

        for _ in range(2):
            self._load_next_day()
        self.bat.add_batch(self.batch)

        self.batch = self.bat.compute_bat_dem_agg(self.batch)

        self._loads_test()

        return self.save_file, self.batch

    def reinitialise_envfactors(self, date0, epoch, i_explore,
                                evaluation_add1=False):
        """Reinitialise factors and clusters lists."""
        if evaluation_add1:
            i_explore += 1

        random_clus, random_f = [[[
            self.np_random.rand()
            for _ in range(self.prm['ntw']['n_all'])]
            for _ in labels]
            for labels in [self.labels_clus, self.labels]
        ]

        i_dt = 0 if date0.weekday() < 5 else 1
        next_dt = 0 if (date0 + timedelta(days=1)).weekday() < 5 else 1
        dtt = self.labels_day_trans[i_dt * 2 + next_dt * 1]

        for passive_ext in ['', 'P']:
            for i, label in enumerate(self.labels_clus):
                dt = self.prm['syst']['labels_day'][i_dt]
                clusas = []
                da = self.prm['ntw']['n'] if passive_ext == 'P' else 0
                dtt_ = self._p_trans_label(dtt, label)
                for home in range(self.prm['ntw']['n' + passive_ext]):
                    if epoch == 0 and i_explore == 0:
                        psclus = self.prm[label]['pclus'][dt]
                    else:
                        clus = self.clus[f"{label}{passive_ext}"][home]
                        psclus = self.prm[label]['ptrans'][dtt_][clus]
                    choice = self._ps_rand_to_choice(
                        psclus, random_clus[i][home + da])
                    clusas.append(choice)
                self.clus[label + passive_ext] = clusas

            if epoch == 0 and i_explore == 0:
                for e in self.labels:
                    self.f[e + passive_ext] = [
                        self.prm['syst']['f0'][e]
                        for _ in range(self.prm['ntw']['n' + passive_ext])
                    ]
            else:
                ia0 = self.prm['ntw']['n'] if passive_ext == 'P' else 0
                iaend = ia0 + self.prm['ntw']['nP'] if passive_ext == 'P' \
                    else ia0 + self.prm['ntw']['n']
                self._next_factors(
                    passive_ext=passive_ext, dtt=dtt,
                    rands=[random_f[ie][ia0: iaend]
                           for ie in range(len(self.labels))])

    def set_passive_active(self, passive: bool = False):
        """Update environment properties for passive or active case."""
        self.passive_ext = 'P' if passive else ''
        for e in ['cap', 'T_LB', 'T_UB', 'T_req', 'store0',
                  'mincharge', 'bat', 'loads', 'gen']:
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
            self.action_translator.date0 = date0
            self.date_end = date0 + timedelta(hours=self.N)
            self.bat.date0 = date0
            self.bat.date_end = self.date_end

    def fix_data_a(self, homes, file_id, its=0):
        """Recompute data for home a that is infeasible."""
        self._seed(self.envseed[0] + its)
        for home in homes:
            self.factors[home] = initialise_dict(
                [self.loads_p, self.gen_p, self.bat_p]
            )
            self.clusters[home] = initialise_dict([self.loads_p, self.bat_p])
        self._initialise_batch_entries(homes)
        date_load = self.date0
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day(homes=homes)
            date_load += timedelta(days=1)
        self.bat.add_batch(self.batch)
        self.batch = self.bat.compute_bat_dem_agg(self.batch)
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
            h = self._get_h()
            n_homes = self.n_homes
            batch_flex = [self.batch[home]['flex'] for home in range(n_homes)]
        else:
            h, batch_flex, max_delay, n_homes = opts
        share_flexs = self.prm['loads']['share_flexs']
        new_batch_flex = np.array(
            [
                [
                    copy.deepcopy(batch_flex[home][ih])
                    for ih in range(h, h + 2)
                ]
                for home in range(n_homes)
            ]
        )

        for home in range(n_homes):
            remaining_cons = max(cons_flex[home], 0)
            assert cons_flex[home] <= np.sum(batch_flex[home][h][1:]) + 1e-3, \
                "cons_flex[home] > np.sum(batch_flex[home][h][1:]) + 1e-3"

            # remove what has been consumed
            for i_flex in range(1, self.max_delay + 1):
                delta_cons = min(new_batch_flex[home, 0, i_flex], remaining_cons)
                remaining_cons -= delta_cons
                new_batch_flex[home, 0, i_flex] -= delta_cons
            assert remaining_cons <= 1e-2, \
                f"remaining_cons = {remaining_cons} too large"

            # move what has not been consumed to one step more urgent
            self._new_flex_tests(batch_flex, new_batch_flex, share_flexs, h, home)

        return new_batch_flex

    def step(
            self, action: list, implement: bool = True,
            record: bool = False, evaluation: bool = False,
            netp_storeout: bool = False, E_req_only: bool = False
    ) -> list:
        """Compute environment updates and reward from selected action."""
        h = self._get_h()
        agents = self.homes
        batch_flex = [self.batch[home]['flex'] for home in agents]
        self._batch_tests(batch_flex, h)

        # update batch if needed
        daynumber = (self.date - self.date0).days
        if h == 1 and self.time > 1 \
                and self.dloaded < daynumber + 2 == 0 \
                and not self.slid_day:
            self._load_next_day()
            self.slid_day = True
        self.bat.add_batch(self.batch)

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
            next_date = self.date + timedelta(hours=1)
            next_done = next_date == self.date_end
            inputs_next_state = [self.time + 1, next_date, next_done,
                                 new_batch_flex, self.bat.store]
            next_state = self.get_state_vals(inputs=inputs_next_state) \
                if not self.done \
                else [None for home in agents]
            if implement:
                for home in agents:
                    batch_flex[home][h: h + 2] = new_batch_flex[home]
                self.time += 1
                self.date = next_date
                self.idt = 0 if self.date.weekday() < 5 else 1
                self.done = next_done
                self.heat.update_step()
                self.bat.update_step()

            for ih in range(h + 1, h + 30):
                assert all(
                    self.batch[home]['loads'][ih]
                    <= batch_flex[home][ih][0] + batch_flex[home][ih][-1] + 1e-3
                    for home in self.homes
                ), f"h {h} ih {ih}"
            if record or evaluation:
                ld_fixed = [sum(batch_flex[home][h][:]) for home in agents] \
                    if self.date == self.date_end - timedelta(hours=2) \
                    else [batch_flex[home][h][0] for home in agents]

            if record:
                ldflex = [0 for home in agents] if self.date == self.date_end \
                    else [sum(batch_flex[home][h][1:]) for home in agents]
                record_output = \
                    [home_vars['netp'], self.bat.discharge, action, reward,
                     break_down_rewards, self.bat.store, ldflex, ld_fixed,
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
                        [home_vars['netp'], self.bat.discharge_tot,
                         self.bat.charge]]
            else:
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok, None]

    def get_reward(
            self,
            netp: list,
            discharge_tot: list = None,
            charge: list = None,
            passive_vars: list = None,
            i_step: int = None,
            evaluation: bool = False
    ) -> Tuple[list, list]:
        """Compute reward from netp and battery charge at time step."""
        if passive_vars is not None:
            netp0, discharge_tot0, charge0 = passive_vars
        elif self.cap_p == 'capP':
            netp0, discharge_tot0, charge0 = \
                [[0 for _ in range(self.prm['ntw']['nP'])] for _ in range(3)]
        else:
            seconds_per_interval = 3600 * 24 / self.prm['syst']['H']
            hour = int((self.date - self.date0).seconds / seconds_per_interval)
            netp0, discharge_tot0, charge0 = [
                [self.prm['loads'][e][home][hour]
                 for home in range(self.prm['ntw']['nP'])]
                for e in ['netp0', 'discharge_tot0', 'charge0']]
        i_step = self.time if i_step is None else i_step
        if discharge_tot is None:
            discharge_tot = self.bat.discharge_tot
        charge = self.bat.charge if charge is None else charge
        grdCt, wholesalet, cintensityt = [
            self.grdC[i_step], self.wholesale[i_step], self.cintensity[i_step]]

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
        sc = self.prm['bat']['C'] \
            * (sum(discharge_tot[home] + charge[home]
                   for home in self.homes)
                + sum(discharge_tot0[home] + charge0[home]
                      for home in range(len(discharge_tot0))))
        sc_a = [self.prm['bat']['C'] * (discharge_tot[home] + charge[home])
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
            h = self._get_h()
            batch_flex = [self.batch[home]['flex'][h] for home in self.homes]
            if date == self.date_end - timedelta(hours=1):
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
            home_vars['gen'] = np.array(
                [self.batch[home]['gen'][h] for home in self.homes]
            )

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
            h = self._get_h()
            home_vars = {'gen': gens}
        self.heat.E_heat_min_max(h)
        last_step = True \
            if date == self.date_end - timedelta(hours=1) \
            else False
        bool_penalty = self.bat.min_max_charge_t(h, date)
        self.heat.potential_E_flex()

        #  ----------- meet consumption + check constraints ---------------
        constraint_ok = True
        loads, home_vars, bool_penalty = \
            self.action_translator.actions_to_env_vars(loads, home_vars, action, date, h)

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
                self.time, self.date, self.done, self.bat.store
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
        hour = self._get_h(date)
        for home in self.homes:
            for i, descriptor in enumerate(descriptors):
                vals[home, i] = self._descriptor_to_val(
                    descriptor, inputs_, hour, idt, home
                )

        return vals

    def _seed(self, seed=None):
        if seed is not None and not isinstance(seed, integer_types):
            seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _p_trans_label(self, dtt, e):
        if dtt == 'wd2wd' and 'wd2wd' not in self.prm[e]['ptrans']:
            dtt_ = 'wd'
        elif dtt == 'we2we' and 'we2we' not in self.prm[e]['ptrans']:
            dtt_ = 'we'
        else:
            dtt_ = dtt

        return dtt_

    def _file_id(self):
        return f"{self.nonamefile}{self.passive_ext}{self.opt_res_file}"

    def _ps_rand_to_choice(self, ps: list, rand: float) -> int:
        """Given probability of each choice and a random number, select."""
        p_intervals = [sum(ps[0:i]) for i in range(len(ps))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]
        return choice

    def _next_factors(self, passive_ext=None, dtt=None, rands=None, homes=[]):
        """Compute self-correlated random scaling factors for data profiles."""
        homes = range(self.prm['ntw']['n' + passive_ext]) if len(homes) == 0 else homes
        if passive_ext is None:
            passive_ext = '' if self.bat_p == 'bat' else 'P'
        dtt = dtt if dtt is not None \
            else self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
        dtt_ = dtt[0:2] if dtt not in self.f_prm['loads'] else dtt
        fEV_new_interval = np.zeros((len(homes),))
        for i_home in range(len(homes)):
            home = homes[i_home]
            # factor for demand - differentiate between day types
            df = gamma.ppf(rands[0][i_home], *list(self.f_prm['loads'][dtt_]))
            self.f['loads' + passive_ext][home] = \
                self.f['loads' + passive_ext][home] + df - self.f_mean['loads'][dtt_]
            # factor for generation - without differentiation between day types
            df = gamma.ppf(rands[2][i_home], * self.f_prm['gen'])
            self.f['gen' + passive_ext][home] = \
                self.f['gen' + passive_ext][home] + df - self.f_mean['gen']

            # factor for EV consumption
            bracket_fs = self.prm['bat']['bracket_fs'][dtt_]
            current_interval = \
                [i for i in range(self.prm['bat']['intervals_fprob'] - 1)
                 if bracket_fs[i] <= self.f['bat' + passive_ext][home]][-1]
            choice = self._ps_rand_to_choice(
                self.prm['bat']['f_prob'][dtt_][current_interval],
                rands[1][i_home])
            fEV_new_interval[i_home] = choice
            self.f['bat' + passive_ext][home] = self.prm['bat']['mid_fs'][dtt_][int(choice)]
            for e in ['loads', 'gen', 'bat']:
                self.f[e + passive_ext][home] = min(
                    max(self.min_f[e], self.f[e + passive_ext][home]), self.max_f[e])

        return fEV_new_interval

    def _get_next_clusters(self, dtt, homes):
        for home in homes:
            for e in self.labels_clus:
                # get next cluster
                dtt_ = dtt[0:2] if dtt not in self.ptrans[e] else dtt
                clus_a = self.clus[self.__dict__[e + '_p']][home]
                ps = self.ptrans[e][dtt_][clus_a]
                cump = [sum(ps[0:i]) for i in range(1, len(ps))] + [1]
                rdn = self.np_random.rand()
                self.clus[self.__dict__[e + '_p']][home] = \
                    [c > rdn for c in cump].index(True)

    def _adjust_EV_cons(self, homes, dt, dtt, day, i_EV, fEV_new_interval):
        dtt_ = dtt[0:2] if dtt not in self.prm['bat']['mid_fs'] else dtt
        for i_home in range(len(homes)):
            home = homes[i_home]
            clus = self.clus[self.bat_p][home]
            it = 0
            while (
                    np.max(day['loads_EV'][i_home])
                    > self.prm['bat'][self.cap_p][home]
                    and it < 100
            ):
                if fEV_new_interval[i_home] > 0:
                    fEV_new_interval[i_home] -= 1
                    interval = int(fEV_new_interval[i_home])
                    self.f[self.bat_p][home] = \
                        self.prm['bat']['mid_fs'][dtt_][interval]
                    prof = self.prof['bat']['cons'][dt][clus][i_EV[i_home]]
                    day['loads_EV'][i_home] = [
                        x * self.f[self.bat_p][home] for x in prof
                    ]
                else:
                    i_EV[i_home] = self.np_random.choice(
                        np.arange(self.n_prof['bat'][dt][clus]))
                it += 1

        return day, i_EV

    def _generate_new_day(self, homes: list):
        """If new day of data was not presaved, load here."""
        # intialise variables
        homes = self.homes if len(homes) == 0 else homes
        day = {}
        dt = self.prm['syst']['labels_day'][self.idt]
        dtt = self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
        loads_p = self.loads_p if self.loads_p in self.factors[0] \
            else 'lds' + self.loads_p[5:]

        # save factors and clusters at the start of the episode
        for home in homes:
            for e in [loads_p, self.gen_p, self.bat_p]:
                self.factors[home][e].append(self.f[e][home])
            for e in [self.loads_p, self.bat_p]:
                self.clusters[home][e].append(self.clus[e][home])

        # get next clusters (for load and EV)
        self._get_next_clusters(dtt, homes)

        # get load profile indexes, normalised profile, and scaled profile
        i_prof_load = self._compute_i_profs('loads', dt, homes=homes)
        load_prof = [
            self.prof['loads'][dt][self.clus[self.loads_p][home]][i_prof_load[i_home]]
            for i_home, home in enumerate(homes)
        ]
        day['loads'] = \
            [load_prof[i_home] * self.f[self.loads_p][home]
             if self.prm['loads']['own_loads'][home]
             else [0 for _ in range(self.N)]
             for i_home, home in enumerate(homes)]

        # get PV profile index, and day profile
        month = self.date.month
        while not self.n_prof['gen'][month - 1] > 0:
            month += 1
            month = 1 if month == 12 else month
        i_prof_gen = self._compute_i_profs('gen', idx_month=month - 1, homes=homes)
        day['gen'] = [[g * self.f[self.gen_p][home]
                       for g in self.prof['gen'][month - 1][i_prof_gen[i_home]]]
                      if self.prm['gen']['own_PV'][home]
                      else [0 for _ in range(self.N)]
                      for i_home, home in zip(range(len(homes)), homes)]

        # get EV cons factor, profile index, normalised profile, scaled profile
        fEV_new_interval = self._next_factors(
            dtt=dtt,
            rands=[[self.np_random.rand() for _ in range(len(homes))]
                   for _ in range(len(self.labels))],
            homes=homes)
        i_EV = self._compute_i_profs('bat', dt=dt, homes=homes)
        prof = [
            self.prof['bat']['cons'][dt][self.clus[self.bat_p][home]][i_EV[i_home]]
            for i_home, home in enumerate(homes)
        ]
        day['loads_EV'] = \
            [[x * self.f[self.bat_p][home] if self.prm['bat']['own_EV'][home]
              else 0 for x in prof[i_home]]
             for i_home, home in enumerate(homes)]

        # check EV consumption is not larger than capacity - if so, correct
        day, i_EV = self._adjust_EV_cons(
            homes, dt, dtt, day, i_EV, fEV_new_interval
        )

        # get EV availability profile
        day['avail_EV'] = \
            [self.prof['bat']['avail'][dt][self.clus[self.bat_p][home]][i_EV[i_home]]
             for i_home, home in zip(range(len(homes)), homes)]
        for i_home in range(len(homes)):
            if sum(day['loads_EV'][i_home]) == 0 and sum(day["avail_EV"][i_home]) == 0:
                day["avail_EV"][i_home] = np.ones(self.prm["syst"]["N"])
        for i_home, home in enumerate(homes):
            for e in day.keys():
                self.batch[home][e] = self.batch[home][e] + list(day[e][i_home])
        self._loads_to_flex(homes)
        self.dloaded += 1

        assert len(self.batch[0]['avail_EV']) > 0, "empty avail_EV batch"

    def _load_next_day(self, homes: list = []):
        """
        Load next day of data.

        Either it is not presaved and needs to be generated,
        or it can just be loaded
        """
        if not self.load_data or len(homes) > 0:
            self._generate_new_day(homes)
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
        assert len(self.batch[0]['avail_EV']) > 0, "empty avail_EV batch"

    def _compute_i_profs(self,
                         type_clus: str,
                         dt: str = None,
                         idx_month: int = None,
                         homes: list = None
                         ) -> list:
        """Get random indexes for profile selection."""
        homes = self.homes if len(homes) == 0 else homes
        iprofs = []
        n_profs = self.n_prof[type_clus][dt] \
            if dt is not None \
            else [self.n_prof[type_clus][idx_month]]
        n_profs = [int(self.prm['syst']['share_centroid'] * n_prof)
                   for n_prof in n_profs]
        available_profiles = \
            [[i for i in range(n_prof)] for n_prof in n_profs]
        for home in homes:
            if type_clus in self.labels_clus:
                clus = self.clus[self.__dict__[type_clus + '_p']][home]
                avail_prof = available_profiles[clus]
            else:
                avail_prof = available_profiles[0]
            i_prof = self.np_random.choice(avail_prof)

            if len(avail_prof) > 1:
                avail_prof.remove(i_prof)
            iprofs.append(i_prof)

        return iprofs

    def _loads_to_flex(self, homes: list = None):
        """Apply share of flexible loads to new day loads data."""
        homes = self.homes if len(homes) == 0 else homes
        for home in homes:
            dayflex_a = np.zeros((self.N, self.max_delay + 1))
            for t in range(24):
                dayflex_a[t, 0] = \
                    (1 - self.prm['loads']['share_flexs'][home]) \
                    * self.batch[home]['loads'][self.dloaded * 24 + t]
                dayflex_a[t, self.max_delay] = \
                    self.prm['loads']['share_flexs'][home] \
                    * self.batch[home]['loads'][self.dloaded * 24 + t]
            self.batch[home]['flex'] = np.concatenate(
                (self.batch[home]['flex'], dayflex_a)
            )

    def _get_h(self, date: datetime = None) -> int:
        """Given date, obtain hour."""
        date = self.date if date is None else date
        time_elapsed = date - self.date0
        h = int(time_elapsed.days * 24 + time_elapsed.seconds / 3600)

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

        if date == self.date_end - timedelta(hours=1) \
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
            self.bat.check_constraints(home, date, h)
            self.heat.check_constraints(home, h, E_req_only)
            bool_penalty = self._check_loads(home, date, h, loads, bool_penalty)

            # prosumer balance
            prosumer_balance_sum = \
                abs(home_vars['netp'][home]
                    - (self.bat.loss_ch[home] + self.bat.charge[home])
                    + self.bat.discharge[home]
                    + home_vars['gen'][home]
                    - home_vars['tot_cons'][home])
            if prosumer_balance_sum > 1e-2:
                print(f"home {home} prosumer_balance_sum = {prosumer_balance_sum}")
                print(f"self.bat.loss_ch[{home}] = {self.bat.loss_ch[home]}")
                print(f"self.bat.charge[{home}] = {self.bat.charge[home]}")
                print(f"self.bat.discharge[{home}] = {self.bat.discharge[home]}")
                print(f"home = {home}, loads = {loads}")
                np.save('action_translator_d', self.action_translator.d)
                np.save('action_translator_mu', self.action_translator.action_intervals)
                np.save('action_translator_k', self.action_translator.k)
                bool_penalty[home] = True

            # check tot cons
            if home_vars['tot_cons'][home] < - 1e-2:
                print(f"negative tot_cons {home_vars['tot_cons'][home]} home = {home}")
                bool_penalty[home] = True
            share_fixed = (1 - self.prm['loads']['share_flexs'][home])
            if last_step \
                    and home_vars['tot_cons'][home] < \
                    self.batch[home]['loads'][h] * share_fixed:
                print(f"home = {home}, no flex cons at last time step")
                bool_penalty[home] = True

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
            None: None,
            'hour': hour % 24,
            'bat_dem_agg': self.batch[home]['bat_dem_agg'][hour],
            'store0': store[home],
            'grdC': self.grdC[t],
            'day_type': idt,
            'dT': self.prm['heat']['T_req' + self.passive_ext][home][hour] - self.T_air[home]

        }
        if descriptor in dict_vals:
            val = dict_vals[descriptor]
        elif len(descriptor) >= 4 and descriptor[0:4] == 'grdC':
            val = self.normalised_grdC[t]
        elif descriptor == 'dT_next':
            T_req = self.prm['heat']['T_req' + self.passive_ext][home]
            t_change_T_req = [t for t in range(hour + 1, self.N)
                              if T_req[t] != T_req[hour]]
            next_T_req = T_req[t_change_T_req[0]]
            current_T_req = T_req[hour]
            val = 0 if len(t_change_T_req) == 0 \
                else (next_T_req - current_T_req) / (t_change_T_req[0] - hour)
        elif descriptor == 'EV_tau':
            val = self.bat.EV_tau(hour, date, home, store[home])
        elif len(descriptor) > 9 \
                and (descriptor[-9:-5] == 'fact'
                     or descriptor[-9:-5] == 'clus'):
            # scaling factors / profile clusters for the whole day
            module = descriptor.split('_')[0]  # EV, loads or gen
            if descriptor.split('_')[-1] == 'prev':
                prev_data = self.factors if descriptor[-9:-5] == 'fact' \
                    else self.clusters
                val = prev_data[home][module][-1]
            else:  # step
                step_data = self.f if descriptor[-9:-5] == 'fact' \
                    else self.clus
                val = step_data[module][home]
        else:  # select current or previous hour - step or prev
            h = self._get_h() if descriptor[-4:] == 'step' \
                else self._get_h() - 1
            if len(descriptor) > 8 and descriptor[0:8] == 'avail_EV':
                val = self.batch[home]['avail_EV'][h]
            elif descriptor[0:5] == 'loads':
                val = np.sum(batch_flex_h[home][1])
            else:
                # gen_prod_step / prev and EV_cons_step / prev
                batch_type = 'gen' if descriptor[0:3] == 'gen' else 'loads_EV'
                val = self.batch[home][batch_type][h]

        return val

    def _batch_tests(self, batch_flex, h):
        share_flexs = self.prm['loads']['share_flexs']
        for home in self.homes:
            assert sum(batch_flex[home][h][1: 5]) <= \
                sum(batch_flex[home][ih][0] / (1 - share_flexs[home])
                    * share_flexs[home]
                    for ih in range(0, h + 1)), "batch_flex too large h"

            assert sum(batch_flex[home][h + 1][1: 5]) <= sum(
                batch_flex[home][ih][0]
                / (1 - share_flexs[home]) * share_flexs[home]
                for ih in range(0, h + 2)), "batch_flex too large h + 1"

            for ih in range(h, h + 30):
                assert self.batch[home]['loads'][ih] <= \
                    batch_flex[home][ih][0] + batch_flex[home][ih][-1] + 1e-3, \
                    "loads larger than with flex"

    def _new_flex_tests(self, batch_flex, new_batch_flex, share_flexs, h, home):
        assert np.sum(new_batch_flex[home][0][1:5]) <= \
               sum(batch_flex[home][ih][0] / (1 - share_flexs[home])
                   * share_flexs[home]
                   for ih in range(0, h)) + 1e-3, \
               "flex too large"
        for i_flex in range(self.max_delay):
            loads_next_flex = new_batch_flex[home][0][i_flex + 1]
            assert not (
                i_flex > 0
                and i_flex < 4
                and new_batch_flex[home][1][i_flex] + loads_next_flex
                > np.sum([batch_flex[home][ih][0] for ih in range(0, h + 1)])
                / (1 - share_flexs[home]) * share_flexs[home] + 1e-3
            ), "loads_next_flex error"
            new_batch_flex[home][1][i_flex] += loads_next_flex
            assert not (
                loads_next_flex
                > np.sum([batch_flex[home][ih][0] for ih in range(0, h + 1)])
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
        i_grdC_level = [i for i in range(len(self.spaces.descriptors['state']))
                        if self.spaces.descriptors['state'][i] == 'grdC_level']
        if len(i_grdC_level) > 0:
            self.normalised_grdC = \
                [(gc - min(self.grdC[0: self.N]))
                 / (max(self.grdC[0: self.N])
                    - min(self.grdC[0: self.N]))
                 for gc in self.grdC[0: self.N + 1]]
            self.spaces.brackets['state'][i_grdC_level[0]] = \
                [[np.percentile(self.normalised_grdC,
                                i * 100 / self.n_grdC_level)
                  for i in range(self.n_grdC_level)] + [1]
                 for _ in self.homes]

    def _initialise_new_data(self):
        # we have not loaded data from file -> save new data
        date_load = self.date0

        # date_end is not max date end but date end based on
        # current date0 and duration as specified in learning.py
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day()
            date_load += timedelta(days=1)
        self.batch = self.bat.compute_bat_dem_agg(self.batch)

        for e in ['batch', 'clusters', 'factors']:
            file_id = \
                f"{e}{self.nonamefile}{self.passive_ext}{self.opt_res_file}"
            np.save(self.res_path / file_id, self.__dict__[e])

        self._initialise_batch_entries()

        self.batch_file = self.save_file
        self.load_data = True
        self.dloaded = 0
        self.add_noise = False

    def _initialise_batch_entries(self, agents=[]):
        if len(agents) == 0:
            agents = self.homes
        for home in agents:
            self.batch[home] = initialise_dict(self.batch_entries)
            self.bat.batch[home] = initialise_dict(self.bat.batch_entries)
            self.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))
            self.bat.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))
            self.bat.batch[home]['flex'] = np.zeros((0, self.max_delay + 1))

    def _init_factors_clusters_profiles_parameters(self, prm):
        for e in ['min_f', 'max_f', 'f_prm', 'f_mean',
                  'n_clus', 'pclus', 'ptrans', 'n_prof']:
            self.__dict__[e] = {}
        for obj, labobj in zip([prm['loads'], prm['gen']], ['loads', 'gen']):
            self.min_f[labobj] = np.min(obj['listfactors'])
            self.max_f[labobj] = np.max(obj['listfactors'])
            self.f_prm[labobj] = obj['f_prms']
            self.f_mean[labobj] = obj['f_mean']
        self.min_f['bat'] = min(
            min(prm['bat']['bracket_fs'][dtt]) for dtt in self.labels_day_trans
        )
        self.max_f['bat'] = max(
            max(prm['bat']['bracket_fs'][dtt]) for dtt in self.labels_day_trans
        )

        for obj, labobj in zip([prm['loads'], prm['bat']], self.labels_clus):
            for e in ['n_clus', 'pclus', 'ptrans']:
                self.__dict__[e][labobj] = obj[e]

        prms = [prm[e] for e in self.labels]
        for obj, label in zip(prms, self.labels):
            self.__dict__['n_prof'][label] = obj['n_prof']
