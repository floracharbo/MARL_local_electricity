#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:49:57 2020.

@author: floracharbonnier
"""
import datetime
import os
import random
import time  # to record time it takes to run simulations
from datetime import date, timedelta
from functools import partial
from typing import Tuple

import numpy as np
import torch as th
from tqdm import tqdm

from src.initialisation.initialise_objects import initialise_objects
from src.initialisation.input_data import (get_settings_i, input_paths,
                                           load_existing_prm)
from src.learners.DDPG import Learner_DDPG
from src.learners.DDQN import Agent_DDQN
from src.learners.DQN import Agent_DQN
from src.learners.facmac.components.episode_buffer import (EpisodeBatch,
                                                           ReplayBuffer)
from src.learners.facmac.controllers import REGISTRY as mac_REGISTRY
from src.learners.facmac.learners import REGISTRY as le_REGISTRY
from src.learners.Qlearning import TabularQLearner
from src.post_analysis.plotting.plot_summary_no_agents import \
    plot_results_vs_nag
from src.post_analysis.post_processing import post_processing
from src.simulations.explorer import Explorer
from src.simulations.local_elec import LocalElecEnv
from src.utilities.userdeftools import (data_source, initialise_dict,
                                        reward_type, set_seeds_rdn)


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

    def _save_nn_model(self, model_save_time):
        if self.rl['save_model'] and (
                self.explorer.t_env - model_save_time
                >= self.rl['save_model_interval']
                or model_save_time == 0):
            model_save_time = self.explorer.t_env

            if self.prm['RL']['type_learning'] == 'facmac' \
                    and self.prm["save"]["save_nns"]:
                for t_explo in self.rl["exploration_methods"]:
                    if t_explo not in self.learner:
                        continue
                    save_path \
                        = self.prm["paths"]["record_folder"] \
                        / f"models_{t_explo}_{self.explorer.t_env}"
                    os.makedirs(save_path, exist_ok=True)
                    self.learner[t_explo].save_models(save_path)
        return model_save_time

    def _end_evaluation(
            self, repeat, new_env, evaluations_methods, i0_costs, delta, date0
    ):
        i_explore = 0
        for epoch_test in \
                tqdm(range(self.rl['n_epochs'], self.rl['n_all_epochs']),
                     position=0, leave=True):
            t_start = time.time()  # start recording time
            date0, delta, i0_costs = \
                self._set_date(
                    repeat, epoch_test, i_explore, date0,
                    delta, i0_costs, new_env
                )
            self.env.reinitialise_envfactors(
                date0, epoch_test, i_explore, evaluation_add1=True)
            eval_steps, _ = self.explorer.get_steps(
                evaluations_methods, repeat, epoch_test, self.rl['n_explore'],
                evaluation=True, new_episode_batch=self.new_episode_batch)
            duration_epoch = time.time() - t_start

            self.record.end_epoch(
                epoch_test, eval_steps, None,
                self.rl, self.learner, duration_epoch, end_test=True
            )

    def _post_exploration_learning(self, epoch, train_steps_vals):
        if not self.rl['instant_feedback'] \
                and self.rl['type_learning'] == 'q_learning' \
                and epoch > 0:
            # if we did not learn instantly after each step,
            # learn here after exploration
            self.learner.learn_from_explorations(train_steps_vals)

        elif self.rl['type_learning'] == 'DQN':
            for method in self.rl['type_Qs']:
                if self.rl['distr_learning'] == 'decentralised':
                    for home in range(self.n):
                        self.learner[method][home].target_update()
                else:
                    self.learner[method].target_update()

    def run_experiment(self):
        """For a given state space, explore and learn from the environment."""
        repeat = 0  # initialise repetition number
        new_env = True  # boolean for creating a new environment
        # initialise the seed for creating environments
        self.explorer.ind_seed_deterministic = - 1
        date0, delta, i0_costs = None, None, None
        # multiple repetition to sample different learning trajectories
        while repeat < self.rl['n_repeats']:
            print(f"repeat {repeat}")
            episode, converged = self._new_repeat(repeat, new_env)

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
                    episode += 1

                    steps_vals, date0, delta, i0_costs, exploration_methods \
                        = self._exploration_episode(
                            repeat, epoch, i_explore, date0, delta, i0_costs,
                            new_env, evaluation=False, evaluation_add1=False
                        )
                    train_steps_vals.append(steps_vals)

                    if self.rl['type_learning'] == 'facmac':
                        # insert episode batch in buffer, sample, train
                        self._facmac_episode_batch_insert_and_sample(episode)

                    # append record
                    for e in ['seed', 'n_not_feas', 'not_feas_vars']:
                        self.record.__dict__[e][repeat].append(
                            train_steps_vals[-1][e])

                    model_save_time = self._save_nn_model(model_save_time)

                # learning step at the end of the exploration
                # if it was not done instantly after each step
                self._post_exploration_learning(epoch, train_steps_vals)

                # evaluation step
                evaluations_methods = self._check_if_opt_needed(epoch, evaluation=True)
                assert i_explore + 1 == self.rl['n_explore']

                self.env.reinitialise_envfactors(
                    date0, epoch, self.rl['n_explore'])
                eval_steps, _ = self.explorer.get_steps(
                    evaluations_methods, repeat, epoch, self.rl['n_explore'],
                    evaluation=True, new_episode_batch=self.new_episode_batch)

                # record
                for e in ['seed', 'n_not_feas', 'not_feas_vars']:
                    self.record.__dict__[e][repeat].append(eval_steps[e])
                duration_epoch = time.time() - t_start

                # make a list, one exploration after the other
                # rather than a list of 'explorations' in 2D
                list_train_stepvals = self._train_vals_to_list(
                    train_steps_vals, exploration_methods)
                self.record.end_epoch(epoch, eval_steps, list_train_stepvals,
                                      self.rl, self.learner, duration_epoch)

                if self.rl['deterministic']:
                    converged = self._check_convergence(repeat, epoch, converged)
                self._end_of_epoch_parameter_updates(repeat, epoch)

            # then do evaluation only for one month, no learning
            self._end_evaluation(
                repeat, new_env, evaluations_methods, i0_costs, delta, date0
            )

            new_env = True \
                if ((repeat + 1) % self.rl['n_init_same_env'] == 0
                    or self.rl['deterministic'] == 2) \
                else False
            self.record.save(end_of='repeat')
            repeat += 1

        for passive_ext in ['P', '']:
            if len(self.explorer.data.seeds[passive_ext]) > len(self.rl['seeds'][passive_ext]):
                self.rl['seeds'][passive_ext] = self.explorer.seeds[passive_ext].copy()

    def _initialise_buffer_learner_mac_facmac(self, method):
        if 'buffer' not in self.__dict__.keys():
            self.buffer = {}
            self.mac = {}

        self.buffer[method] = ReplayBuffer(
            self.rl['scheme'], self.rl['groups'],
            self.rl['buffer_size'],
            self.rl['env_info']["episode_limit"] + 1
            if self.rl['runner_scope'] == "episodic" else 2,
            preprocess=self.rl['preprocess'],
            device="cpu" if self.rl['buffer_cpu_only']
            else self.rl['device'])
        # Setup multiagent controller here
        self.mac[method] = mac_REGISTRY[self.rl['mac']](
            self.buffer[method].scheme, self.rl['groups'],
            self.rl)
        self.new_episode_batch = \
            partial(EpisodeBatch, self.rl['scheme'],
                    self.rl['groups'],
                    self.rl['batch_size_run'],
                    self.rl['episode_limit'] + 1,
                    preprocess=self.rl['preprocess'],
                    device=self.rl['device'])
        self.learner[method] = le_REGISTRY[self.rl['learner']](
            self.mac[method],
            self.buffer[method].scheme,
            self.rl,
        )
        if self.rl['use_cuda']:
            self.learner[method].cuda()

    def _initialise_buffer_learner_mac_deep_learning(self, method):
        if self.rl['distr_learning'] == 'decentralised':
            if method in self.learner.keys():
                # the learner as already been intialised; simply reset
                for home in range(self.n):
                    self.learner[method][home].reset()
            else:
                if self.rl['type_learning'] == 'DDPG':
                    self.learner[method] = [Learner_DDPG(
                        self.rl, method + f'_{home}') for home in range(self.n)]
                elif self.rl['type_learning'] == 'DDQN':
                    self.learner[method] = [Agent_DDQN(
                        self.env, self.rl, method) for _ in range(self.n)]
                else:
                    self.learner[method] = [Agent_DQN(
                        self.rl, method + f'_{home}', method, self.prm['syst']['N'])
                        for home in range(self.n)]
        else:
            if method in self.learner.keys():
                # the learner as already been intialised;
                # simply reset
                self.learner[method].reset()  # reset learner
            else:  # initialise objects
                if self.rl['type_learning'] == 'DDPG':
                    self.learner[method] = Learner_DDPG(self.rl, method)
                elif self.rl['type_learning'] == 'DQN':
                    self.learner[method] = Agent_DQN(
                        self.rl, method, method, self.prm['syst']['N'])
                elif self.rl['type_learning'] == 'DDQN':
                    self.learner[method] = Agent_DDQN(self.env, self.rl, method)

    def _initialise_buffer_learner_mac(self, repeat=0):
        if self.rl['type_learning'] in ['DDPG', 'DQN', 'DDQN', 'facmac']:
            if 'learner' not in self.__dict__.keys():
                self.learner = {}
            for method in self.rl['type_Qs']:
                if self.rl['type_learning'] == 'facmac':
                    self._initialise_buffer_learner_mac_facmac(method)
                else:
                    self._initialise_buffer_learner_mac_deep_learning(method)

        elif self.rl['type_learning'] == 'q_learning':
            if 'learner' in self.__dict__.keys():
                # the Q learner needs to initialise epsilon
                # and temperature values, and set counters
                # and Q tables to 0
                self.learner.new_repeat(repeat)
            else:  # create one instance for all types
                self.learner = TabularQLearner(self.env, self.rl)
                self.learner = TabularQLearner(self.env, self.rl)

        if self.rl['type_learning'] != 'facmac':
            self.new_episode_batch = None
            self.mac = None

    def _new_repeat(self, repeat, new_env):
        # track whether the learning has converged
        # (useful for deterministic case)
        converged = False
        self.explorer.t_env = 0
        self._initialise_buffer_learner_mac()
        # initialise dictionaries for storing relevant values during repeat
        self.record.new_repeat(repeat, self.rl)
        #  at each repeat reinitialise agent clusters
        #  according to clus0 probabilities
        if new_env:  # need to create a new environment
            # data for this environment's deterministic runs not yet generated
            self.explorer.data.deterministic_created = False
            self.explorer.ind_seed_deterministic += 1
        # record current environment creation seed
        self.record.ind_seed_deterministic[repeat] = \
            self.explorer.ind_seed_deterministic

        # Set seeds (for reproduceability)
        np.random.seed(repeat), random.seed(repeat)
        th.manual_seed(repeat)
        if self.rl['type_learning'] == 'q_learning' \
                and self.rl['q_learning']['control_eps'] == 2:
            self.learner.rMT, self.learner.rLT = 0, 0
            # exploration based on XU et al. 2018 Reward-Based Exploration
        episode = 0

        return episode, converged

    def _set_date(self,
                  repeat,
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
            seed = self.explorer.data.get_seed_ind(repeat, epoch, i_explore)
            set_seeds_rdn(seed)
            delta_days = int(np.random.choice(range(
                (self.prm['syst']['max_date_end_dtm']
                    - self.prm['syst']['date0_dtm']).days
                - self.prm['syst']['D'])))
            date0 = self.prm['syst']['date0_dtm'] \
                + datetime.timedelta(days=delta_days)
            # self.prm['syst']['current_date0_dtm'] = date0
            delta = date0 - self.prm['syst']['date0_dtm']
            i0_costs = int(delta.days * 24 + delta.seconds / 3600)
            self.prm['grd']['C'] = \
                self.prm['grd']['Call'][
                i0_costs: i0_costs + self.prm['syst']['N']]
            self.explorer.i0_costs = i0_costs
            self.env.update_date(i0_costs, date0)

        return date0, delta, i0_costs

    def _facmac_episode_batch_insert_and_sample(self, episode):

        for t_explo in self.rl["exploration_methods"]:
            methods_to_update = [] if t_explo == 'baseline' \
                else [t_explo] if t_explo[0:3] == 'env' \
                else [method for method in self.rl['type_Qs']
                      if data_source(method) == 'opt' and method[-1] != '0']
            for method in methods_to_update:
                diff = True if reward_type(method) == 'd' else False
                opt = True if data_source(method) == 'opt' else False

                self.buffer[method].insert_episode_batch(
                    self.episode_batch[method], difference=diff,
                    optimisation=opt)

                if self.buffer[method].can_sample(self.rl['facmac']['batch_size']) \
                        and (self.buffer[method].episodes_in_buffer
                             > self.rl['buffer_warmup']):
                    episode_sample = self.buffer[method].sample(
                        self.rl['facmac']['batch_size'])

                    # Truncate batch to only filled time steps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.rl['device']:
                        episode_sample.to(self.rl['device'])
                    self.learner[method].train(
                        episode_sample,
                        self.explorer.t_env,
                        episode
                    )

    def _train_vals_to_list(self, train_steps_vals, exploration_methods):

        list_train_stepvals = initialise_dict(
            self.rl["exploration_methods"], type_obj='empty_dict')

        for method in self.rl["exploration_methods"]:
            for e in train_steps_vals[0][self.rl["exploration_methods"][0]].keys():
                if e not in ['seeds', 'n_not_feas', 'not_feas_vars'] \
                        and e in train_steps_vals[0][exploration_methods[0]].keys():
                    list_train_stepvals[method][e] = []
                    for i_explore in range(self.rl['n_explore']):
                        if method in exploration_methods:
                            for x in train_steps_vals[i_explore][method][e]:
                                list_train_stepvals[method][e].append(x)
                        else:
                            vals = \
                                train_steps_vals[i_explore][exploration_methods[0]][e]
                            for _ in enumerate(vals):
                                list_train_stepvals[method][e].append(None)

        return list_train_stepvals

    def _DDQN_epsilon_update(self):

        for method in self.rl['type_Qs']:
            if self.rl['distr_learning'] == 'centralised':
                self.learner[method].epsilon_update()
            elif self.rl['distr_learning'] == 'decentralised':
                for home in range(self.n_homes):
                    self.learner[method][home].epsilon_update()

    def _check_convergence(self, repeat, epoch, converged):
        if not converged and \
                sum(1 for method in self.rl["evaluation_methods"]
                    if self.record.stability[repeat][method] != [None]) \
                == len(self.rl["evaluation_methods"]) * 5:
            converged = True
            print(f'repeat {repeat} converged at epoch = {epoch}')

        return converged

    def _DQN_T_decay(self):
        for method in self.rl['type_Qs']:
            if self.rl['distr_learning'] == 'centralised':
                self.learner[method].ActionStateModel.T = \
                    self.learner[method].ActionStateModel.T * \
                    self.rl['T_decay_param']
            elif self.rl['distr_learning'] == 'decentralised':
                for home in range(self.n):
                    self.learner[method][home].ActionStateModel.T = \
                        self.learner[method][home].ActionStateModel.T * \
                        self.rl['T_decay_param']

    def _check_if_opt_needed(self, epoch, evaluation=False):
        opts_in_eval = sum(method != 'opt' and method[0:3] == 'opt'
                           for method in self.rl["evaluation_methods"]) > 0
        eval_stage = True \
            if evaluation and epoch >= self.rl['start_end_eval'] \
            else False
        candidate_types = self.rl["evaluation_methods"] if evaluation \
            else self.rl["exploration_methods"]

        # if no evaluation types rely on optimisation
        # and we are not checking feasibility with optimisations
        # and we are not evaluating using optimisations in the latest stages
        # then do not use optimisations
        if not opts_in_eval \
                and not self.rl['check_feasibility_with_opt'] \
                and not eval_stage:
            types_needed = [method for method in candidate_types if method[0:3] != 'opt']
        else:
            types_needed = candidate_types

        return types_needed

    def _end_of_epoch_parameter_updates(self, repeat, epoch):

        if self.rl['type_learning'] == 'DDQN':
            self._DDQN_epsilon_update()

        if self.rl['type_learning'] == 'q_learning':
            if self.rl['q_learning']['epsilon_decay']:
                self.learner.epsilon_decay(
                    repeat, epoch, self.record.mean_eval_rewards)
            if self.rl['q_learning']['T_decay']:
                self.learner.T = self.learner.T * self.rl['T_decay_param']

        elif self.rl['type_learning'] == 'DQN' and self.rl['DQN']['T_decay']:
            self._DQN_T_decay()

    def _exploration_episode(
            self, repeat, epoch, i_explore, date0, delta,
            i0_costs, new_env, evaluation=False, evaluation_add1=False,
            set_date=True
    ):
        if set_date:
            date0, delta, i0_costs = self._set_date(
                repeat, epoch, i_explore, date0, delta,
                i0_costs, new_env)

        # initialise environment cluster, scaling factors, etc.
        self.env.reinitialise_envfactors(
            date0, epoch, i_explore, evaluation_add1=evaluation_add1)

        # exploration - obtain experience
        exploration_methods = self._check_if_opt_needed(
            epoch, evaluation=evaluation)

        steps_vals, self.episode_batch = self.explorer.get_steps(
            exploration_methods, repeat, epoch, i_explore,
            new_episode_batch=self.new_episode_batch, evaluation=evaluation)

        return steps_vals, date0, delta, i0_costs, exploration_methods


def get_number_runs(settings):
    n_runs = 1

    for sub_dict in settings.values():
        for val in list(sub_dict.values()):
            if isinstance(val, dict):
                for val_ in list(val.values()):
                    if isinstance(val_, list) and len(val_) > n_runs:
                        n_runs = len(val_)
            elif isinstance(val, list) and len(val) > n_runs:
                n_runs = len(val)

    return n_runs


def run(run_mode, settings, no_runs=None):
    prm = input_paths()

    if run_mode == 1:
        # obtain the number of runs from the longest settings entry
        n_runs = get_number_runs(settings)

        # loop through runs
        for i in range(n_runs):
            remove_old_prms = [e for e in prm if e != 'paths']
            for e in remove_old_prms:
                del prm[e]

            start_time = time.time()  # start recording time
            # obtain run-specific settings

            settings_i = get_settings_i(settings, i)
            # initialise learning parameters, system parameters and recording
            prm, record, profiles = initialise_objects(
                prm, settings=settings_i)

            DESCRIPTION_RUN = 'current code '
            for e in ['type_learning', 'n_repeats', 'n_epochs',
                      'server', 'state_space']:
                DESCRIPTION_RUN += f"prm['RL'][{e}] {prm['RL'][e]} "
            print(DESCRIPTION_RUN)
            prm['save']['description_run'] = DESCRIPTION_RUN

            if prm['RL']['type_learning'] == 'facmac':
                # Setting the random seed throughout the modules
                np.random.seed(prm['syst']["seed"])
                th.manual_seed(prm['syst']["seed"])

            env = LocalElecEnv(prm, profiles)
            # second part of initialisation specifying environment
            # with relevant parameters
            record.init_env(env)  # record progress as we train
            runner = Runner(env, prm, record)
            runner.run_experiment()
            record.save(end_of='end')  # save progress at end
            post_processing(
                record, env, prm, start_time=start_time, settings_i=settings_i
            )
            print(f"--- {time.time() - start_time} seconds ---")

    # post learning analysis / plotting
    elif run_mode == 2:
        if isinstance(no_runs, int):
            no_runs = [no_runs]  # the runs need to be in an array

        for no_run in no_runs:
            lp, prm = load_existing_prm(prm, no_run)

            prm, record, profiles = initialise_objects(prm, no_run=no_run)
            # make user defined environment
            env = LocalElecEnv(prm, profiles)
            record.init_env(env)  # record progress as we train
            post_processing(record, env, prm, no_run=no_run)

    elif run_mode == 3:
        plot_results_vs_nag()
