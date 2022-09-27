#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:49:57 2020.

@author: floracharbonnier
"""
import datetime
import os
import random
import time
from datetime import date, timedelta
from functools import partial
from typing import Tuple

import numpy as np
import torch as th
from tqdm import tqdm

# from learners.DDPG import Learner_DDPG
# from learners.DDQN import Agent_DDQN
# from learners.DQN import Agent_DQN
from learners.facmac.components.episode_buffer import (EpisodeBatch,
                                                       ReplayBuffer)
from learners.facmac.controllers import REGISTRY as mac_REGISTRY
from learners.facmac.learners import REGISTRY as le_REGISTRY
from learners.Qlearning import TabularQLearner
from simulations.explorer import Explorer
from utils.userdeftools import initialise_dict, set_seeds_rdn


class Runner():
    """Run experiments for all repeats and epochs."""

    def __init__(self, env, prm, record):
        """Initialise Runner input data and objects."""
        self.prm = prm
        self.n = prm['ntw']['n']  # number of households / agents
        self.rl = prm['RL']  # learning parameters
        self.env = env
        self.record = record
        self._initialise_buffer_learner_mac()

        # create an instance of the explorer object
        # which will interact with the environment
        self.explorer = Explorer(env, prm, self.learner, record, self.mac)

    def run_experiment(self):
        """For a given state space, explore and learn from the environment."""
        ridx = 0  # initialise repetition number
        new_env = True  # boolean for creating a new environment
        # initialise the seed for creating environments
        self.explorer.ind_seed_deterministic = - 1
        date0, delta, i0_costs = None, None, None
        # multiple repetition to sample different learning trajectories
        while ridx < self.rl['n_repeats']:
            print(f"ridx {ridx}")
            episode, converged = self._new_repeat(ridx, new_env)

            # looping through epochs
            # have progress bar while running through epochs

            model_save_time = 0
            for epoch in tqdm(range(self.rl['n_epochs']),
                              position=0, leave=True):
                t_start = time.time()  # start recording time
                # explorations in series
                # (could maybe parallelise but may cause issues)
                train_steps_vals = []  # initialise

                # loop through number of explorations for each epoch
                for i_explore in range(self.rl['n_explore']):
                    print(f"i_explore {i_explore}")
                    episode += 1

                    steps_vals, date0, delta, i0_costs, type_explo \
                        = self._exploration_episode(
                            ridx, epoch, i_explore, date0, delta, i0_costs,
                            new_env, evaluation=False, evaluation_add1=False
                        )

                    train_steps_vals.append(steps_vals)

                    if self.rl['type_learning'] == 'facmac':
                        # insert episode batch in buffer, sample, train
                        self._facmac_episode_batch_insert_and_sample(episode)

                    # append record
                    for e in ['seed', 'n_not_feas', 'not_feas_vars']:
                        self.record.__dict__[e][ridx].append(
                            train_steps_vals[-1][e])

                    print(f"self.explorer.t_env {self.explorer.t_env}")
                    if self.rl['save_model'] and (
                            self.explorer.t_env - model_save_time
                            >= self.rl['save_model_interval']
                            or model_save_time == 0):
                        model_save_time = self.explorer.t_env

                        if self.prm["save"]["save_nns"]:
                            for t_explo in self.rl['type_explo']:
                                if t_explo not in self.learner:
                                    continue
                                save_path \
                                    = self.prm["paths"]["record_folder"] \
                                    / f"models_{t_explo}_{self.explorer.t_env}"
                                os.makedirs(save_path, exist_ok=True)
                                self.learner[t_explo].save_models(save_path)

                # learning step at the end of the exploration
                # if it was not done instantly after each step
                if not self.rl['instant_feedback'] \
                        and self.rl['type_learning'] == 'q_learning' \
                        and epoch > 0:
                    # if we did not learn instantly after each step,
                    # learn here after exploration
                    self.learner.learn_from_explorations(train_steps_vals)

                elif self.rl['type_learning'] == 'DQN':
                    for t in self.rl['type_Qs']:
                        if self.rl['distr_learning'] == 'decentralised':
                            for a in range(self.n):
                                self.learner[t][a].target_update()
                        else:
                            self.learner[t].target_update()

                # evaluation step
                # REPLACE HERE
                print("evaluation step")
                type_eval = self._check_if_opt_needed(epoch, evaluation=True)
                assert i_explore + 1 == self.rl['n_explore']
                self.env.reinitialise_envfactors(
                    date0, epoch, self.rl['n_explore'])
                eval_steps, _ = self.explorer.get_steps(
                    type_eval, ridx, epoch, self.rl['n_explore'],
                    evaluation=True, new_episode_batch=self.new_episode_batch)

                # record
                for e in ['seed', 'n_not_feas', 'not_feas_vars']:
                    self.record.__dict__[e][ridx].append(eval_steps[e])
                duration_epoch = time.time() - t_start

                # make a list, one exploration after the other
                # rather than a list of 'explorations' in 2D
                list_train_stepvals = self._train_vals_to_list(
                    train_steps_vals, type_explo)
                self.record.end_epoch(epoch, eval_steps, list_train_stepvals,
                                      self.rl, self.learner, duration_epoch)

                if self.rl['deterministic']:
                    converged = self._check_convergence(ridx, epoch, converged)

                self._end_of_epoch_parameter_updates(ridx, epoch)

            # then do evaluation only for one month, no learning
            print("eval month")
            for epoch_test in \
                    tqdm(range(self.rl['n_epochs'], self.rl['n_all_epochs']),
                         position=0, leave=True):
                t_start = time.time()  # start recording time
                i_explore = 0
                episode += 1
                date0, delta, i0_costs = \
                    self._set_date(
                        ridx, epoch_test, i_explore, date0,
                        delta, i0_costs, new_env
                    )
                self.env.reinitialise_envfactors(
                    date0, epoch_test, i_explore, evaluation_add1=True)
                eval_steps, _ = self.explorer.get_steps(
                    type_eval, ridx, epoch_test, self.rl['n_explore'],
                    evaluation=True, new_episode_batch=self.new_episode_batch)
                duration_epoch = time.time() - t_start

                self.record.end_epoch(
                    epoch_test, eval_steps, list_train_stepvals,
                    self.rl, self.learner, duration_epoch, end_test=True
                )

                episode += 1

                date0, delta, i0_costs = self._set_date(
                    ridx, epoch, i_explore, date0, delta,
                    i0_costs, new_env)

                # initialise environment cluster, scaling factors, etc.
                self.env.reinitialise_envfactors(
                    date0, epoch, i_explore, evaluation_add1=False)

                # exploration - obtain experience
                type_explo = self._check_if_opt_needed(
                    epoch, evaluation=False)

                steps_vals, self.episode_batch = self.explorer.get_steps(
                    type_explo, ridx, epoch, i_explore,
                    new_episode_batch=self.new_episode_batch)

                train_steps_vals.append(steps_vals)

            new_env = True \
                if ((ridx + 1) % self.rl['n_init_same_env'] == 0
                    or self.rl['deterministic'] == 2) \
                else False
            self.record.save(end_of='repeat')
            ridx += 1

        for p in ['P', '']:
            if len(self.explorer.data.seeds[p]) > len(self.rl['seeds'][p]):
                self.rl['seeds'][p] = self.explorer.seeds[p].copy()

    def _initialise_buffer_learner_mac(self, ridx=0):
        if self.rl['type_learning'] in ['DDPG', 'DQN', 'DDQN', 'facmac']:
            if 'learner' not in self.__dict__.keys():
                self.learner = {}
            for t in self.rl['type_Qs']:
                if self.rl['type_learning'] == 'facmac':
                    if 'buffer' not in self.__dict__.keys():
                        self.buffer = {}
                        self.mac = {}

                    self.buffer[t] = ReplayBuffer(
                        self.rl['scheme'], self.rl['groups'],
                        self.rl['buffer_size'],
                        self.rl['env_info']["episode_limit"] + 1
                        if self.rl['runner_scope'] == "episodic" else 2,
                        preprocess=self.rl['preprocess'],
                        device="cpu" if self.rl['buffer_cpu_only']
                        else self.rl['device'])
                    # Setup multiagent controller here
                    self.mac[t] = mac_REGISTRY[self.rl['mac']](
                        self.buffer[t].scheme, self.rl['groups'],
                        self.rl)
                    self.new_episode_batch = \
                        partial(EpisodeBatch, self.rl['scheme'],
                                self.rl['groups'],
                                self.rl['batch_size_run'],
                                self.rl['episode_limit'] + 1,
                                preprocess=self.rl['preprocess'],
                                device=self.rl['device'])
                    self.learner[t] = le_REGISTRY[self.rl['learner']](
                        self.mac[t], self.buffer[t].scheme, self.rl)
                    if self.rl['use_cuda']:
                        self.learner[t].cuda()

                elif self.rl['distr_learning'] == 'decentralised':
                    if t in self.learner.keys():
                        # the learner as already been intialised; simply reset
                        for a in range(self.n):
                            self.learner[t][a].reset()
                    else:
                        if self.rl['type_learning'] == 'DDPG':
                            self.learner[t] = [Learner_DDPG(
                                self.rl, t + f'_{a}') for a in range(self.n)]
                        elif self.rl['type_learning'] == 'DDQN':
                            self.learner[t] = [Agent_DDQN(
                                self.env, self.rl, t) for _ in range(self.n)]
                        else:
                            self.learner[t] = [Agent_DQN(
                                self.rl, t + f'_{a}', t, self.prm['syst']['N'])
                                for a in range(self.n)]
                else:
                    if t in self.learner.keys():
                        # the learner as already been intialised;
                        # simply reset
                        self.learner[t].reset()  # reset learner
                    else:  # initialise objects
                        if self.rl['type_learning'] == 'DDPG':
                            self.learner[t] = Learner_DDPG(self.rl, t)
                        elif self.rl['type_learning'] == 'DQN':
                            self.learner[t] = Agent_DQN(
                                self.rl, t, t, self.prm['syst']['N'])
                        elif self.rl['type_learning'] == 'DDQN':
                            self.learner[t] = Agent_DDQN(self.env, self.rl, t)

        elif self.rl['type_learning'] == 'q_learning':
            if 'learner' in self.__dict__.keys():
                # the Q learner needs to initialise epsilon
                # and temperature values, and set counters
                # and Q tables to 0
                self.learner._new_repeat(ridx)
            else:  # create one instance for all types
                self.learner = TabularQLearner(self.env, self.rl)

        if self.rl['type_learning'] != 'facmac':
            self.new_episode_batch = None
            self.mac = None

    def _new_repeat(self, ridx, new_env):
        # track whether the learning has converged
        # (useful for deterministic case)
        converged = False
        self.explorer.t_env = 0
        self._initialise_buffer_learner_mac()
        # initialise dictionaries for storing relevant values during repeat
        self.record.new_repeat(ridx, self.rl)
        #  at each repeat reinitialise agent clusters
        #  according to clus0 probabilities
        if new_env:  # need to create a new environment
            # data for this environment's deterministic runs not yet generated
            self.explorer.deterministic_created = False
            self.explorer.ind_seed_deterministic += 1
        # record current environment creation seed
        self.record.ind_seed_deterministic[ridx] = \
            self.explorer.ind_seed_deterministic

        # Set seeds (for reproduceability)
        np.random.seed(ridx), random.seed(ridx)
        th.manual_seed(ridx)
        if self.rl['type_learning'] == 'q_learning' \
                and self.rl['q_learning']['control_eps'] == 2:
            self.learner.rMT, self.learner.rLT = 0, 0
            # exploration based on XU et al. 2018 Reward-Based Exploration
        episode = 0

        return episode, converged

    def _set_date(self,
                  ridx,
                  epoch,
                  i_explore,
                  date0,
                  delta,
                  i0_costs,
                  new_env
                  ) -> Tuple[date, timedelta, int]:
        if self.rl['deterministic'] > 0:
            new_date = True if epoch == 0 and i_explore == 0 and new_env == 1 \
                else False
        else:
            new_date = True if self.prm['syst']['change_start'] else False
        if new_date:
            seed = self.explorer.data.get_seed_ind(ridx, epoch, i_explore)
            set_seeds_rdn(seed)
            delta_days = int(np.random.choice(range(
                (self.prm['syst']['max_dateend']
                    - self.prm['syst']['date0']).days
                - self.prm['syst']['D'])))
            date0 = self.prm['syst']['date0'] \
                + datetime.timedelta(days=delta_days)
            self.prm['syst']['current_date0'] = date0
            delta = date0 - self.prm['syst']['date0_dates']
            i0_costs = int(delta.days * 24 + delta.seconds / 3600)
            self.prm['grd']['C'] = \
                self.prm['grd']['Call'][
                i0_costs: i0_costs + self.prm['syst']['N']]
            self.explorer.i0_costs = i0_costs
            self.env.update_date(i0_costs, date0)

        return date0, delta, i0_costs

    def _facmac_episode_batch_insert_and_sample(self, episode):
        print("_facmac_episode_batch_insert_and_sample")
        for t_explo in self.rl['type_explo']:
            print(f"t_explo {t_explo}")
            t_to_update = [] if t_explo == 'baseline' \
                else [t_explo] if t_explo[0:3] == 'env' \
                else [t for t in self.rl['type_Qs']
                      if t.split('_')[0] == 'opt' and t[-1] != '0']
            print(f"t_to_update {t_to_update}")
            for t in t_to_update:
                diff = True if t.split('_')[1] == 'd' else False
                opt = True if t.split('_')[0] == 'opt' else False
                self.buffer[t].insert_episode_batch(
                    self.episode_batch[t_explo], difference=diff,
                    optimisation=opt)
                can_sample = self.buffer[t].can_sample(
                    self.rl['facmac']['batch_size']
                )
                print(f"self.buffer[t].can_sample("
                      f"self.rl['facmac']['batch_size']) "
                      f"{can_sample}")
                buffer_warm_up_bool = (
                    self.buffer[t].episodes_in_buffer
                    > self.rl['buffer_warmup']
                )
                print(f"self.buffer[t].episodes_in_buffer "
                      f"> self.rl['buffer_warmup'] = "
                      f"{buffer_warm_up_bool}")
                if self.buffer[t].can_sample(self.rl['facmac']['batch_size']) \
                        and (self.buffer[t].episodes_in_buffer
                             > self.rl['buffer_warmup']):
                    episode_sample = self.buffer[t].sample(
                        self.rl['facmac']['batch_size'])

                    # Truncate batch to only filled time steps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.rl['device']:
                        episode_sample.to(self.rl['device'])
                    print(f"train {t}")
                    self.learner[t].train(episode_sample,
                                          self.explorer.t_env, episode)

    def _train_vals_to_list(self, train_steps_vals, type_explo):

        list_train_stepvals = initialise_dict(
            self.rl['type_explo'], type_obj='empty_dict')

        for t in self.rl['type_explo']:
            for e in train_steps_vals[0][self.rl['type_explo'][0]].keys():
                if e not in ['seeds', 'n_not_feas', 'not_feas_vars'] \
                        and e in train_steps_vals[0][type_explo[0]].keys():
                    list_train_stepvals[t][e] = []
                    for i_explore in range(self.rl['n_explore']):
                        if t in type_explo:
                            for x in train_steps_vals[i_explore][t][e]:
                                list_train_stepvals[t][e].append(x)
                        else:
                            vals = \
                                train_steps_vals[i_explore][type_explo[0]][e]
                            for _ in enumerate(vals):
                                list_train_stepvals[t][e].append(None)

        return list_train_stepvals

    def _DDQN_epsilon_update(self):

        for t in self.rl['type_Qs']:
            if self.rl['distr_learning'] == 'centralised':
                self.learner[t].epsilon_update()
            elif self.rl['distr_learning'] == 'decentralised':
                for a in range(self.n_agents):
                    self.learner[t][a].epsilon_update()

    def _check_convergence(self, ridx, epoch, converged):
        if not converged and \
                sum(1 for t in self.rl['type_eval']
                    if self.record.stability[ridx][t] != [None]) \
                == len(self.rl['type_eval']) * 5:
            converged = True
            print(f'repeat {ridx} converged at epoch = {epoch}')

        return converged

    def _DQN_T_decay(self):
        for t in self.rl['type_Qs']:
            if self.rl['distr_learning'] == 'centralised':
                self.learner[t].ActionStateModel.T = \
                    self.learner[t].ActionStateModel.T * \
                    self.rl['T_decay_param']
            elif self.rl['distr_learning'] == 'decentralised':
                for a in range(self.n):
                    self.learner[t][a].ActionStateModel.T = \
                        self.learner[t][a].ActionStateModel.T * \
                        self.rl['T_decay_param']

    def _check_if_opt_needed(self, epoch, evaluation=False):
        opts_in_eval = sum(t != 'opt' and t[0:3] == 'opt'
                           for t in self.rl['type_eval']) > 0
        eval_stage = True \
            if evaluation and epoch >= self.rl['start_end_eval'] \
            else False
        candidate_types = self.rl['type_eval'] if evaluation \
            else self.rl['type_explo']

        # if no evaluation types rely on optimisation
        # and we are not checking feasibility with optimisations
        # and we are not evaluating using optimisations in the latest stages
        # then do not use optimisations
        if not opts_in_eval \
                and not self.rl['check_feasibility_with_opt'] \
                and not eval_stage:
            types_needed = [t for t in candidate_types if t[0:3] != 'opt']
        else:
            types_needed = candidate_types

        return types_needed

    def _end_of_epoch_parameter_updates(self, ridx, epoch):

        if self.rl['type_learning'] == 'DDQN':
            self._DDQN_epsilon_update()

        if self.rl['type_learning'] == 'q_learning':
            if self.rl['q_learning']['epsilon_decay']:
                self.learner.epsilon_decay(
                    ridx, epoch, self.record.mean_eval_rewards)
            if self.rl['q_learning']['T_decay']:
                self.learner.T = self.learner.T * self.rl['T_decay_param']

        elif self.rl['type_learning'] == 'DQN' and self.rl['DQN']['T_decay']:
            self._DQN_T_decay()

    def _exploration_episode(
            self, ridx, epoch, i_explore, date0, delta,
            i0_costs, new_env, evaluation=False, evaluation_add1=False,
            set_date=True
    ):
        if set_date:
            date0, delta, i0_costs = self._set_date(
                ridx, epoch, i_explore, date0, delta,
                i0_costs, new_env)

        # initialise environment cluster, scaling factors, etc.
        self.env.reinitialise_envfactors(
            date0, epoch, i_explore, evaluation_add1=evaluation_add1)

        # exploration - obtain experience
        type_explo = self._check_if_opt_needed(
            epoch, evaluation=evaluation)

        steps_vals, self.episode_batch = self.explorer.get_steps(
            type_explo, ridx, epoch, i_explore,
            new_episode_batch=self.new_episode_batch, evaluation=evaluation)

        return steps_vals, date0, delta, i0_costs, type_explo
