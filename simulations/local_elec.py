#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:57 2020.

@author: floracharbonnier

"""

import copy
from datetime import datetime, timedelta

import numpy as np
from gym import spaces
from gym.utils import seeding
from scipy.stats import gamma
from six import integer_types

from home_components.battery import Battery
from home_components.heat import Heat
from simulations.mu_manager import Mu_manager
from utils.env_spaces import EnvSpaces
from utils.userdeftools import initialise_dict


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

    # def my_init(self, ):
        self.prm = prm
        self.rl = prm['RL']
        self.labels_day_trans = prm['syst']['labels_day_trans']

        self.prof = profiles
        self.n_agents = prm['ntw']['n']
        self.agents = range(self.n_agents)
        self.N = prm['syst']['N']

        # initialise parameters
        for e in ['minf', 'maxf', 'f_prm', 'f_mean',
                  'n_clus', 'pclus', 'ptrans', 'n_prof']:
            self.__dict__[e] = {}
        for obj, labobj in zip([prm['loads'], prm['gen']], ['loads', 'gen']):
            self.minf[labobj] = np.min(obj['listfactors'])
            self.maxf[labobj] = np.max(obj['listfactors'])
            self.f_prm[labobj] = obj['f_prms']
            self.f_mean[labobj] = obj['f_mean']
        self.minf['bat'] = min(min(prm['bat']['bracket_fs'][dtt])
                               for dtt in self.labels_day_trans)
        self.maxf['bat'] = max(max(prm['bat']['bracket_fs'][dtt])
                               for dtt in self.labels_day_trans)

        for obj, labobj in zip([prm['loads'], prm['bat']], self.labels_clus):
            for e in ['n_clus', 'pclus', 'ptrans']:
                self.__dict__[e][labobj] = obj[e]

        prms = [prm[e] for e in self.labels]
        for obj, label in zip(prms, self.labels):
            self.__dict__['n_prof'][label] = obj['n_prof']

        self.server = self.rl['server']
        self.mu_manager = Mu_manager(prm, self)
        self.spaces = EnvSpaces(self)
        self.spaces.new_state_space(self.rl['state_space'])
        self.add_noise = 1 if self.rl['deterministic'] == 2 else 0
        for e in ['competitive', 'n_grdC_level',
                  'offset_reward', 'delta_reward']:
            self.__dict__[e] = self.rl[e]
        self.opt_res_file = self.prm['paths']['opt_res_file']
        self.res_path = prm['paths']['res_path']
        self.i0_costs = 0
        self.bat = Battery(prm)

        if self.rl['type_learning'] == 'facmac':
            self.action_space = self.rl['action_space']
            self.observation_space = []
            for a in self.agents:
                self.observation_space.append(spaces.Box(
                    low=-np.inf, high=+np.inf,
                    shape=(self.rl['obs_shape'],),
                    dtype=np.float32))

        self.max_delay = self.prm['loads']['max_delay']

    def reset(self, seed=None, load_data=False,
              passive=False, E_req_only=False):
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
        self.t = 0  # hrs since start
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
        self.fs, self.cluss = [initialise_dict(self.agents) for _ in range(2)]
        if self.load_data:
            self.fs = np.load(
                self.res_path / f"fs{self._file_id()}",
                allow_pickle=True).item()
            self.cluss = np.load(
                self.res_path / f"cluss{self._file_id()}",
                allow_pickle=True).item()
        else:
            for a in self.agents:
                self.fs[a] = initialise_dict(
                    [self.loadse, self.gene, self.bate])
                self.cluss[a] = initialise_dict([self.loadse, self.bate])

        # initialise heating and battery objects
        self.heat = Heat(self.prm, self.i0_costs, self.p, E_req_only)
        self.bat.reset(self.prm, self.p)
        self.mu_manager.heat = self.heat
        self.mu_manager.bat = self.bat

        # initialise demand ahead (2 days)
        self.batch = {}
        self.bat.batch = {}
        for a in self.agents:
            self.batch[a] = initialise_dict(self.batch_entries)
            self.bat.batch[a] = initialise_dict(self.bat.batch_entries)

        if not load_data or (load_data and self.add_noise):
            self._initialise_new_data()

        for _ in range(2):
            self._load_next_day()
        self.bat.add_batch(self.batch)

        if 'bat_dem_agg' in self.spaces.descriptors['state'] \
                and 'bat_dem_agg' not in self.batch[0]:
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

        for p in ['', 'P']:
            for i, label in enumerate(self.labels_clus):
                dt = self.prm['syst']['labels_day'][i_dt]
                clusas = []
                da = self.prm['ntw']['n'] if p == 'P' else 0
                dtt_ = self._p_trans_label(dtt, label)
                for a in range(self.prm['ntw']['n' + p]):
                    if epoch == 0 and i_explore == 0:
                        psclus = self.prm[label]['pclus'][dt]
                    else:
                        clus = self.clus[f"{label}{p}"][a]
                        psclus = self.prm[label]['ptrans'][dtt_][clus]
                    choice = self._ps_rand_to_choice(
                        psclus, random_clus[i][a + da])
                    clusas.append(choice)
                self.clus[label + p] = clusas

            if epoch == 0 and i_explore == 0:
                for e in self.labels:
                    self.f[e + p] = [self.prm['syst']['f0'][e]
                                     for _ in range(self.prm['ntw']['n' + p])]
            else:
                ia0 = self.prm['ntw']['n'] if p == 'P' else 0
                iaend = ia0 + self.prm['ntw']['nP'] if p == 'P' \
                    else ia0 + self.prm['ntw']['n']
                self._next_factors(
                    p=p, dtt=dtt,
                    rands=[random_f[ie][ia0: iaend]
                           for ie in range(len(self.labels))])

    def set_passive_active(self, passive: bool = False):
        """Update environment properties for passive or active case."""
        self.p = 'P' if passive else ''
        for e in ['cap', 'T_LB', 'T_UB', 'T_req', 'store0',
                  'mincharge', 'bat', 'loads', 'gen']:
            self.__dict__[e + 'e'] = e + self.p
            self.coeff_T = self.prm['heat']['T_coeff' + self.p]
        self.coeff_Tair = self.prm['heat']['T_air_coeff' + self.p]
        self.n_agents = self.prm['ntw']['n' + self.p]
        self.agents = range(self.n_agents)
        for e in ['cape', 'store0e', 'minchargee', 'n_agents']:
            self.mu_manager.__dict__[e] = self.__dict__[e]
            self.spaces.__dict__[e] = self.__dict__[e]
        self.T_air = [self.prm['heat']['T_req' + self.p][a][0]
                      for a in self.agents]

    def update_date(self, i0_costs: int, date0: datetime = None):
        """Update new date for new day."""
        self.i0_costs = i0_costs
        if date0 is not None:
            self.date0 = date0
            self.mu_manager.date0 = date0
            self.date_end = date0 + timedelta(hours=self.N)
            self.bat.date0 = date0
            self.bat.date_end = self.date_end

    def fix_data_a(self, as_, file_id, its=0):
        """Recompute data for home a that is infeasible."""
        self._seed(self.envseed[0] + its)
        for a in as_:
            self.fs[a] = initialise_dict([self.loadse, self.gene, self.bate])
            self.cluss[a] = initialise_dict([self.loadse, self.bate])
            self.batch[a] = initialise_dict(self.batch_entries)
        date_load = self.date0
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day(as_=as_)
            date_load += timedelta(days=1)
        self.bat.add_batch(self.batch)
        if 'bat_dem_agg' in self.spaces.descriptors['state']:
            self.batch = self.bat.compute_bat_dem_agg(self.batch)
        np.save(self.res_path / f"batch{file_id}", self.batch)
        np.save(self.res_path / f"cluss{file_id}", self.cluss)
        np.save(self.res_path / f"fs{file_id}", self.fs)

    def update_flex(self, cons_flex: list, opts: list = None) -> dict:
        """Given step flexible consumption, update remaining flexibility."""
        if opts is None:
            h = self._get_h()
            n_agents = self.n_agents
            batch_flex = [self.batch[a]['flex'] for a in range(n_agents)]
        else:
            h, batch_flex, max_delay, n_agents = opts
        share_flexs = self.prm['loads']['share_flexs']
        new_batch_flex = [[copy.deepcopy(batch_flex[a][ih])
                           for ih in range(h, h + 2)]
                          for a in range(n_agents)]

        for a in range(n_agents):
            remaining_cons = max(cons_flex[a], 0)
            assert cons_flex[a] <= np.sum(batch_flex[a][h][1:]) + 1e-3, \
                "cons_flex[a] > np.sum(batch_flex[a][h][1:]) + 1e-3"

            # remove what has been consumed
            for i_flex in range(1, self.max_delay + 1):
                delta_cons = min(new_batch_flex[a][0][i_flex], remaining_cons)
                remaining_cons -= delta_cons
                new_batch_flex[a][0][i_flex] -= delta_cons
            assert remaining_cons <= 1e-2, \
                f"remaining_cons = {remaining_cons} too large"

            # move what has not be consumed to one step more urgent
            self._new_flex_tests(batch_flex, new_batch_flex, share_flexs, h, a)

        return new_batch_flex

    def step(
            self, action: list, implement: bool = True,
            record: bool = False, evaluation: bool = False,
            netp_storeout: bool = False, E_req_only: bool = False
    ) -> list:
        """Compute environment updates and reward from selected action."""
        h = self._get_h()
        agents = self.agents
        batch_flex = [self.batch[a]['flex'] for a in agents]
        self._batch_tests(batch_flex, h)

        # update batch if needed
        daynumber = (self.date - self.date0).days
        if h == 1 and self.t > 1 \
                and self.dloaded < daynumber + 2 == 0 \
                and not self.slid_day:
            self._load_next_day()
            self.slid_day = True
        self.bat.add_batch(self.batch)

        if h == 2:
            self.slid_day = False
        home, loads, constraint_ok = self.policy_to_rewardvar(
            action, E_req_only=E_req_only)
        if not constraint_ok:
            print('constraint false not returning to original values')
            return None, None, None, None, None, constraint_ok, None

        else:
            reward, break_down_rewards = self.get_reward(
                home['netp'], evaluation=evaluation)
            self.netps = home['netp']

            # ----- update environment variables and state
            new_batch_flex = self.update_flex(loads['flex_cons'])
            next_date = self.date + timedelta(hours=1)
            next_done = next_date == self.date_end
            inputs_next_state = [self.t + 1, next_date, next_done,
                                 new_batch_flex, self.bat.store]
            next_state = self.get_state_vals(inputs=inputs_next_state) \
                if not self.done \
                else [None for a in agents]
            if implement:
                for a in agents:
                    batch_flex[a][h: h + 2] = new_batch_flex[a]
                self.t += 1
                self.date = next_date
                self.idt = 0 if self.date.weekday() < 5 else 1
                self.done = next_done
                self.heat.update_step()
                self.bat.update_step()

            for ih in range(h + 1, h + 30):
                loads_str = 'loads' if 'loads' in self.batch[a] else 'lds'
                assert self.batch[a][loads_str][ih] <= \
                    batch_flex[a][ih][0] + batch_flex[a][ih][-1] + 1e-3,\
                    f"h {h} ih {ih}"
            if record or evaluation:
                ld_fixed = [sum(batch_flex[a][h][:]) for a in agents] \
                    if self.date == self.date_end - timedelta(hours=2) \
                    else [batch_flex[a][h][0] for a in agents]

            if record:
                ldflex = [0 for a in agents] if self.date == self.date_end \
                    else [sum(batch_flex[a][h][1:]) for a in agents]
                record_output = \
                    [home['netp'], self.bat.discharge, action, reward,
                     break_down_rewards, self.bat.store, ldflex, ld_fixed,
                     home['tot_cons'].copy(), loads['tot_cons_loads'].copy(),
                     self.heat.tot_E.copy(), self.heat.T.copy(),
                     self.heat.T_air.copy(), self.grdC[self.t].copy(),
                     self.wholesale[self.t].copy(),
                     self.cintensity[self.t].copy()]
                return [next_state, self.done, reward, break_down_rewards,
                        home['bool_flex'], constraint_ok, record_output]
            elif netp_storeout:
                return [next_state, self.done, reward, break_down_rewards,
                        home['bool_flex'], constraint_ok,
                        [home['netp'], self.bat.discharge_tot,
                         self.bat.charge]]
            else:
                return [next_state, self.done, reward, break_down_rewards,
                        home['bool_flex'], constraint_ok, None]

    def get_reward(self, netp: list, discharge_tot: list = None,
                   charge: list = None, passive_vars: list = None,
                   i_step: int = None, evaluation: bool = False) \
            -> [list, list]:
        """Compute reward from netp and battery charge at time step."""
        if passive_vars is not None:
            netp0, discharge_tot0, charge0 = passive_vars
        elif self.cape == 'capP':
            netp0, discharge_tot0, charge0 = \
                [[0 for _ in range(self.prm['ntw']['nP'])] for _ in range(3)]
        else:
            seconds_per_interval = 3600 * 24 / self.prm['syst']['H']
            hour = int((self.date - self.date0).seconds / seconds_per_interval)
            netp0, discharge_tot0, charge0 = [
                [self.prm['loads'][e][a][hour]
                 for a in range(self.prm['ntw']['nP'])]
                for e in ['netp0', 'discharge_tot0', 'charge0']]
        i_step = self.t if i_step is None else i_step
        if discharge_tot is None:
            discharge_tot = self.bat.discharge_tot
        charge = self.bat.charge if charge is None else charge
        grdCt, wholesalet, cintensityt = [
            self.grdC[i_step], self.wholesale[i_step], self.cintensity[i_step]]

        # negative netp is selling, positive buying
        grid = sum(netp) + sum(netp0)
        if self.prm['ntw']['charge_type'] == 0:
            sum_netp = sum([abs(netp[a]) if netp[a] < 0
                           else 0 for a in self.agents])
            sum_netp0 = sum([abs(netp0[a]) if netp0[a] < 0
                             else 0 for a in range(len(netp0))])
            netpvar = sum_netp + sum_netp0
            dc = self.prm['ntw']['C'] * netpvar
        else:
            netpvar = sum([netp[a] ** 2 for a in self.agents]) \
                + sum([netp0[a] ** 2 for a in range(len(netp0))])
            dc = self.prm['ntw']['C'] * netpvar
        gc = grdCt * (grid + self.prm['grd']['loss'] * grid ** 2)
        gc_a = [wholesalet * netp[a] for a in self.agents]
        sc = self.prm['bat']['C'] \
            * (sum(discharge_tot[a] + charge[a]
                   for a in self.agents)
                + sum(discharge_tot0[a] + charge0[a]
                      for a in range(len(discharge_tot0))))
        sc_a = [self.prm['bat']['C'] * (discharge_tot[a] + charge[a])
                for a in self.agents]
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

        if not evaluation and self.competitive:
            # cost per agent (bill + battery degradation)
            reward = c_a

        return reward, break_down_rewards

    def policy_to_rewardvar(self, action: list, other_input: list = None,
                            mu: list = None, E_req_only: bool = False):
        """Given selected action, obtain results of the step."""
        home, loads = [{} for _ in range(2)]
        if other_input is None:
            date = self.date
            h = self._get_h()
            batch_flex = [self.batch[a]['flex'][h] for a in self.agents]
            if date == self.date_end - timedelta(hours=1):
                loads['l_flex'] = np.zeros(self.n_agents)
                loads['l_fixed'] = np.array(
                    [sum(batch_flex[a]) for a in self.agents]
                )
            else:
                loads['l_flex'] = np.array(
                    [sum(batch_flex[a][1:]) for a in self.agents]
                )
                loads['l_fixed'] = np.array(
                    [batch_flex[a][0] for a in self.agents]
                )
            home['gen'] = np.array(
                [self.batch[a]['gen'][h] for a in self.agents]
            )

            if action is None and mu is None:
                return None
            try:
                for a in self.agents:
                    if action[a] is None:
                        print(f'action[{a}] is None, action = {action[a]}')
            except Exception as ex:
                print(f"ex {ex}")
            self.heat.current_temperature_bounds(h)

        else:
            date, action, gens, loads = other_input
            self.date = date
            h = self._get_h()
            home = {'gen': gens}
        self.heat.E_heat_min_max(h)
        last_step = True \
            if date == self.date_end - timedelta(hours=1) \
            else False
        bool_penalty = self.bat.min_max_charge_t(h, date)
        self.heat.potential_E_flex()

        #  ----------- meet consumption + check constraints ---------------
        constraint_ok = True
        loads, home, bool_penalty = \
            self.mu_manager.apply_step(loads, home, action, date, h)

        self.heat.next_T(update=True)
        self._check_constraints(
            bool_penalty, date, loads, E_req_only, h, last_step, home)

        if sum(bool_penalty) > 0:
            constraint_ok = False

        return home, loads, constraint_ok

    def get_state_vals(self, descriptors: list = None, inputs: list = None)\
            -> float:
        """
        Get values corresponding to array of descriptors inputted.

        (before translation into index)
        """
        t, date, done, batch_flex_h, store = inputs
        idt = 0 if date.weekday() < 5 else 1
        descriptors = descriptors if descriptors is not None \
            else self.spaces.descriptors['state']
        vals = []
        hour = self._get_h(date)
        for a in self.agents:
            vals_a = []
            for descriptor in descriptors:
                vals_a.append(self._descriptor_to_val(
                    descriptor, inputs, hour, idt, a))
            vals.append(vals_a)
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
        return f"{self.nonamefile}{self.p}{self.opt_res_file}"

    def _ps_rand_to_choice(self, ps: list, rand: float) -> int:
        """Given probability of each choice and a random number, select."""
        p_intervals = [sum(ps[0:i]) for i in range(len(ps))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]
        return choice

    def _next_factors(self, p=None, dtt=None, rands=None, as_=None):
        """Compute self-correlated random scaling factors for data profiles."""
        as_ = range(self.prm['ntw']['n' + p]) if as_ is None else as_
        p = p if p is not None else ['' if self.bate == 'bat' else 'P'][0]
        dtt = dtt if dtt is not None \
            else self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
        dtt_ = dtt[0:2] if dtt not in self.f_prm['loads'] else dtt
        fEV_new_interval = np.zeros((len(as_),))
        for ia in range(len(as_)):
            a = as_[ia]
            # factor for demand - differentiate between day types
            df = gamma.ppf(rands[0][ia], *list(self.f_prm['loads'][dtt_]))
            self.f['loads' + p][a] = \
                self.f['loads' + p][a] + df - self.f_mean['loads'][dtt_]
            # factor for generation - without differentiation between day types
            df = gamma.ppf(rands[2][ia], * self.f_prm['gen'])
            self.f['gen' + p][a] = \
                self.f['gen' + p][a] + df - self.f_mean['gen']

            # factor for EV consumption
            bracket_fs = self.prm['bat']['bracket_fs'][dtt_]
            current_interval = \
                [i for i in range(self.prm['bat']['intervals_fprob'] - 1)
                 if bracket_fs[i] <= self.f['bat' + p][a]][-1]
            choice = self._ps_rand_to_choice(
                self.prm['bat']['f_prob'][dtt_][current_interval],
                rands[1][ia])
            fEV_new_interval[ia] = choice
            self.f['bat' + p][a] = self.prm['bat']['mid_fs'][dtt_][int(choice)]
            for e in ['loads', 'gen', 'bat']:
                self.f[e + p][a] = min(
                    max(self.minf[e], self.f[e + p][a]), self.maxf[e])

        return fEV_new_interval

    def _get_next_clusters(self, dtt, as_):
        for a in as_:
            for e in self.labels_clus:
                # get next cluster
                dtt_ = dtt[0:2] if dtt not in self.ptrans[e] else dtt
                ps = self.ptrans[e][dtt_][self.clus[self.__dict__[e + 'e']][a]]
                cump = [sum(ps[0:i]) for i in range(1, len(ps))] + [1]
                rdn = self.np_random.rand()
                self.clus[self.__dict__[e + 'e']][a] = \
                    [c > rdn for c in cump].index(True)

    def _adjust_EV_cons(self, as_, dt, dtt, day, i_EV, fEV_new_interval):
        dtt_ = dtt[0:2] if dtt not in self.prm['bat']['mid_fs'] else dtt
        for ia in range(len(as_)):
            a = as_[ia]
            clus = self.clus[self.bate][a]
            it = 0
            while np.max(day['loads_EV'][ia]) > self.prm['bat'][self.cape][a] \
                    and it < 100:
                if fEV_new_interval[ia] > 0:
                    fEV_new_interval[ia] -= 1
                    interval = int(fEV_new_interval[ia])
                    self.f[self.bate][a] = \
                        self.prm['bat']['mid_fs'][dtt_][interval]
                    prof = self.prof['bat']['cons'][dt][clus][i_EV[ia]]
                    day['loads_EV'][ia] = [x * self.f[self.bate][a]
                                           for x in prof]
                else:
                    i_EV[ia] = self.np_random.choice(
                        np.arange(self.n_prof['bat'][dt][clus]))
                it += 1

        return day, i_EV

    def _generate_new_day(self, as_: list):
        """If new day of data was not presaved, load here."""
        # intialise variables
        as_ = self.agents if as_ is None else as_
        day = {}
        dt = self.prm['syst']['labels_day'][self.idt]
        dtt = self.labels_day_trans[self.idt0 * 2 + self.idt * 1]
        loadse = self.loadse if self.loadse in self.fs[0] \
            else 'lds' + self.loadse[5:]

        # save fs and cluss at the start of the episode
        for a in as_:
            for e in [loadse, self.gene, self.bate]:
                self.fs[a][e].append(self.f[e][a])
            for e in [self.loadse, self.bate]:
                self.cluss[a][e].append(self.clus[e][a])

        # get next clusters (for load and EV)
        self._get_next_clusters(dtt, as_)

        # get load profile indexes, normalised profile, and scaled profile
        i_prof_load = self._compute_i_profs('loads', dt, as_=as_)
        load_prof = \
            [self.prof['loads'][dt][self.clus[self.loadse][a]][i_prof_load[ia]]
             for ia, a in enumerate(as_)]
        day['loads'] = \
            [load_prof[ia] * self.f[self.loadse][a]
             if self.prm['loads']['own_loads'][a]
             else [0 for _ in range(self.N)]
             for ia, a in enumerate(as_)]

        # get PV profile index, and day profile
        month = self.date.month
        while not self.n_prof['gen'][month - 1] > 0:
            month += 1
            month = 1 if month == 12 else month
        i_prof_gen = self._compute_i_profs('gen', idx_month=month - 1, as_=as_)
        day['gen'] = [[g * self.f[self.gene][a]
                       for g in self.prof['gen'][month - 1][i_prof_gen[ia]]]
                      if self.prm['gen']['own_PV'][a]
                      else [0 for _ in range(self.N)]
                      for ia, a in zip(range(len(as_)), as_)]

        # get EV cons factor, profile index, normalised profile, scaled profile
        fEV_new_interval = self._next_factors(
            dtt=dtt,
            rands=[[self.np_random.rand() for _ in range(len(as_))]
                   for _ in range(len(self.labels))],
            as_=as_)
        i_EV = self._compute_i_profs('bat', dt=dt, as_=as_)
        prof = [self.prof['bat']['cons'][dt][self.clus[self.bate][a]][i_EV[ia]]
                for ia, a in enumerate(as_)]
        day['loads_EV'] = \
            [[x * self.f[self.bate][a] if self.prm['bat']['own_EV'][a]
              else 0 for x in prof[ia]]
             for ia, a in enumerate(as_)]

        # check EV consumption is not larger than capacity - if so, correct
        day, i_EV = self._adjust_EV_cons(
            as_, dt, dtt, day, i_EV, fEV_new_interval
        )

        # get EV availability profile
        day['avail_EV'] = \
            [self.prof['bat']['avail'][dt][self.clus[self.bate][a]][i_EV[ia]]
             for ia, a in zip(range(len(as_)), as_)]
        for ia in range(len(as_)):
            if sum(day['loads_EV'][ia]) == 0 and sum(day["avail_EV"][ia]) == 0:
                day["avail_EV"][ia] = np.ones(self.prm["syst"]["N"])
        for ia, a in enumerate(as_):
            for e in day.keys():
                self.batch[a][e] = self.batch[a][e] + list(day[e][ia])
        self._loads_to_flex(as_)
        self.dloaded += 1

    def _load_next_day(self, as_: list = None):
        """
        Load next day of data.

        Either it is not presaved and needs to be generated,
        or it can just be loaded
        """
        if not self.load_data or as_ is not None:
            self._generate_new_day(as_)
        else:
            for e in ['batch', 'cluss', 'fs']:
                self.__dict__[e] = np.load(
                    self.res_path / f"{e}{self._file_id()}",
                    allow_pickle=True).item()

            loadse = self.loadse \
                if self.loadse in self.fs[0] \
                else 'lds' + self.loadse[5:]
            self.dloaded += len(self.fs[0][loadse])

    def _compute_i_profs(self,
                         type_clus: str,
                         dt: str = None,
                         idx_month: int = None,
                         as_: list = None
                         ) -> list:
        """Get random indexes for profile selection."""
        as_ = self.agents if as_ is None else as_
        iprofs = []
        n_profs = self.n_prof[type_clus][dt] \
            if dt is not None \
            else [self.n_prof[type_clus][idx_month]]
        n_profs = [int(self.prm['syst']['share_centroid'] * n_prof)
                   for n_prof in n_profs]
        available_profiles = \
            [[i for i in range(n_prof)] for n_prof in n_profs]
        for a in as_:
            if type_clus in self.labels_clus:
                clus = self.clus[self.__dict__[type_clus + 'e']][a]
                avail_prof = available_profiles[clus]
            else:
                avail_prof = available_profiles[0]
            i_prof = self.np_random.choice(avail_prof)

            if len(avail_prof) > 1:
                avail_prof.remove(i_prof)
            iprofs.append(i_prof)

        return iprofs

    def _loads_to_flex(self, as_: list = None):
        """Apply share of flexible loads to new day loads data."""
        as_ = self.agents if as_ is None else as_
        for a in as_:
            dayflex_a = [None for _ in range(24)]
            for t in range(24):
                dayflex_a[t] = [0 for _ in range(self.max_delay + 1)]
                dayflex_a[t][0] = \
                    (1 - self.prm['loads']['share_flexs'][a]) \
                    * self.batch[a]['loads'][self.dloaded * 24 + t]
                dayflex_a[t][self.max_delay] = \
                    self.prm['loads']['share_flexs'][a] \
                    * self.batch[a]['loads'][self.dloaded * 24 + t]
            self.batch[a]['flex'] = self.batch[a]['flex'] + dayflex_a

    def _get_h(self, date: datetime = None) -> int:
        """Given date, obtain hour."""
        date = self.date if date is None else date
        time_elapsed = date - self.date0
        h = int(time_elapsed.days * 24 + time_elapsed.seconds / 3600)

        return h

    def _check_loads(self,
                     a: int,
                     date: datetime,
                     h: int,
                     loads: dict,
                     bool_penalty: list
                     ) -> list:
        """Check load-related constraints for given home after step."""
        flex_cons, l_fixed = [loads[e] for e in ['flex_cons', 'l_fixed']]

        if date == self.date_end - timedelta(hours=1) \
                and flex_cons[a] > 1e-2:
            print(f"a = {a}, flex_cons[a] = {flex_cons[a]}")
            bool_penalty[a] = True

        if loads['l_flex'][a] > 1e2:
            print(f"h = {h}, a = {a}, l_flex[a] = {loads['l_flex'][a]}")
            bool_penalty[a] = True

        return bool_penalty

    def _check_constraints(self,
                           bool_penalty: bool,
                           date: datetime,
                           loads: dict,
                           E_req_only: bool,
                           h: int,
                           last_step: bool,
                           home: dict
                           ) -> list:
        """Given result of the step action, check environment constraints."""
        for a in [a for a, bool in enumerate(bool_penalty) if not bool]:
            bool_penalty = self.bat.check_constraints(a, date, h, bool_penalty)
            bool_penalty = self.heat.check_constraints(
                a, h, bool_penalty, E_req_only)
            bool_penalty = self._check_loads(a, date, h, loads, bool_penalty)

            # prosumer balance
            prosumer_balance_sum = \
                abs(home['netp'][a]
                    - (self.bat.loss_ch[a] + self.bat.charge[a])
                    + self.bat.discharge[a]
                    + home['gen'][a]
                    - home['tot_cons'][a])
            if prosumer_balance_sum > 1e-2:
                print(f"a {a} prosumer_balance_sum = {prosumer_balance_sum}")
                print(f"self.bat.loss_ch[{a}] = {self.bat.loss_ch[a]}")
                print(f"self.bat.charge[{a}] = {self.bat.charge[a]}")
                print(f"self.bat.discharge[{a}] = {self.bat.discharge[a]}")
                print(f"home = {home}, loads = {loads}")
                np.save('mu_manager_d', self.mu_manager.d)
                np.save('mu_manager_mu', self.mu_manager.mu)
                np.save('mu_manager_k', self.mu_manager.k)
                bool_penalty[a] = True

            # check tot cons
            if home['tot_cons'][a] < - 1e-2:
                print(f"negative tot_cons {home['tot_cons'][a]} a = {a}")
                bool_penalty[a] = True
            loads_str = 'loads' if 'loads' in self.batch[a] else 'lds'
            share_fixed = (1 - self.prm['loads']['share_flexs'][a])
            if last_step \
                    and home['tot_cons'][a] < \
                    self.batch[a][loads_str][h] * share_fixed:
                print(f"a = {a}, no flex cons at last time step")
                bool_penalty[a] = True

        return bool_penalty

    def _descriptor_to_val(self,
                           descriptor: str,
                           inputs: list,
                           hour: int,
                           idt: int,
                           a: int):
        """Given state of action space descriptor, get value."""
        t, date, done, batch_flex_h, store = inputs

        dict_vals = {
            None: None,
            'hour': hour % 24,
            'bat_dem_agg': self.batch[a]['bat_dem_agg'][hour],
            'store0': store[a],
            'grdC': self.grdC[t],
            'day_type': idt,
            'dT': self.prm['heat']['T_req' + self.p][a][hour] - self.T_air[a]

        }
        if descriptor in dict_vals:
            val = dict_vals[descriptor]
        elif len(descriptor) >= 4 and descriptor[0:4] == 'grdC':
            val = self.normalised_grdC[t]
        elif descriptor == 'dT_next':
            T_req = self.prm['heat']['T_req' + self.p][a]
            t_change_T_req = [t for t in range(hour + 1, self.N)
                              if T_req[t] != T_req[hour]]
            next_T_req = T_req[t_change_T_req[0]]
            current_T_req = T_req[hour]
            val = 0 if len(t_change_T_req) == 0 \
                else (next_T_req - current_T_req) / (t_change_T_req[0] - hour)
        elif descriptor == 'EV_tau':
            val = self.bat.EV_tau(hour, date, a, store[a])
        elif len(descriptor) > 9 \
                and (descriptor[-9:-5] == 'fact'
                     or descriptor[-9:-5] == 'clus'):
            # scaling factors / profile clusters for the whole day
            module = descriptor.split('_')[0]  # EV, loads or gen
            if descriptor.split('_')[-1] == 'prev':
                prev_data = self.fs if descriptor[-9:-5] == 'fact' \
                    else self.cluss
                val = prev_data[a][module][-1]
            else:  # step
                step_data = self.f if descriptor[-9:-5] == 'fact' \
                    else self.clus
                val = step_data[module][a]
        else:  # select current or previous hour - step or prev
            h = self._get_h() if descriptor[-4:] == 'step' \
                else self._get_h() - 1
            if len(descriptor) > 8 and descriptor[0:8] == 'avail_EV':
                val = self.batch[a]['avail_EV'][h]
            elif descriptor[0:5] == 'loads':
                val = np.sum(batch_flex_h[a][1])
            else:
                # gen_prod_step / prev and EV_cons_step / prev
                batch_type = 'gen' if descriptor[0:3] == 'gen' else 'loads_EV'
                val = self.batch[a][batch_type][h]

        return val

    def _batch_tests(self, batch_flex, h):
        share_flexs = self.prm['loads']['share_flexs']
        for a in self.agents:
            assert sum(batch_flex[a][h][1: 5]) <= \
                sum(batch_flex[a][ih][0] / (1 - share_flexs[a])
                    * share_flexs[a]
                    for ih in range(0, h + 1)), "batch_flex too large h"

            assert sum(batch_flex[a][h + 1][1: 5]) <= sum(
                batch_flex[a][ih][0]
                / (1 - share_flexs[a]) * share_flexs[a]
                for ih in range(0, h + 2)), "batch_flex too large h + 1"

            for ih in range(h, h + 30):
                loads_str = 'loads' if 'loads' in self.batch[a] else 'lds'
                assert self.batch[a][loads_str][ih] <= \
                    batch_flex[a][ih][0] + batch_flex[a][ih][-1] + 1e-3, \
                    "loads larger than with flex"

    def _new_flex_tests(self, batch_flex, new_batch_flex, share_flexs, h, a):
        assert np.sum(new_batch_flex[a][0][1:5]) <= \
               sum(batch_flex[a][ih][0] / (1 - share_flexs[a])
                   * share_flexs[a]
                   for ih in range(0, h)) + 1e-3, \
               "flex too large"
        for i_flex in range(self.max_delay):
            loads_next_flex = new_batch_flex[a][0][i_flex + 1]
            assert not (
                i_flex > 0
                and i_flex < 4
                and new_batch_flex[a][1][i_flex] + loads_next_flex
                > np.sum(batch_flex[a][ih][0] for ih in range(0, h + 1))
                / (1 - share_flexs[a]) * share_flexs[a] + 1e-3
            ), "loads_next_flex error"
            new_batch_flex[a][1][i_flex] += loads_next_flex
            assert not (
                loads_next_flex
                > np.sum(batch_flex[a][ih][0] for ih in range(0, h + 1))
            ), "loads_next_flex too large"

    def _loads_test(self):
        for a in self.agents:
            for ih in range(self.N):
                loads_str = 'loads' if 'loads' in self.batch[a] else 'lds'
                assert (self.batch[a][loads_str][ih]
                        <= self.batch[a]['flex'][ih][0]
                        + self.batch[a]['flex'][ih][-1] + 1e-3
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
                 for _ in self.agents]

    def _initialise_new_data(self):
        # we have not loaded data from file -> save new data
        date_load = self.date0

        # date_end is not max date end but date end based on
        # current date0 and duration as specified in learning.py
        while date_load < self.date_end + timedelta(days=2):
            self._load_next_day()
            date_load += timedelta(days=1)
        if 'bat_dem_agg' in self.spaces.descriptors['state']:
            self.batch = self.bat.compute_bat_dem_agg(self.batch)

        for e in ['batch', 'cluss', 'fs']:
            file_id = \
                f"{e}{self.nonamefile}{self.p}{self.opt_res_file}"
            np.save(self.res_path / file_id, self.__dict__[e])

        for a in self.agents:
            self.batch[a] = initialise_dict(self.batch_entries)

        self.batch_file = self.save_file
        self.load_data = True
        self.dloaded = 0
        self.add_noise = False
