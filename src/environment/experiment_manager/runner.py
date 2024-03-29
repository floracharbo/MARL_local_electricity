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

import src.environment.initialisation.input_data as input_data
import src.environment.utilities.userdeftools as utils
from src.environment.experiment_manager.explorer import Explorer
from src.environment.initialisation.initialise_objects import \
    initialise_objects
from src.environment.post_analysis.plotting.plot_summary_no_agents import \
    plot_results_vs_nag
from src.environment.post_analysis.post_processing import post_processing
from src.environment.simulations.local_elec import LocalElecEnv
from src.environment.utilities.userdeftools import test_str
from src.learners.DDPG import Learner_DDPG
from src.learners.DDQN import Agent_DDQN
from src.learners.DQN import Agent_DQN
from src.learners.facmac.components.episode_buffer import (EpisodeBatch,
                                                           ReplayBuffer)
from src.learners.facmac.controllers import REGISTRY as mac_REGISTRY
from src.learners.facmac.learners import REGISTRY as le_REGISTRY
from src.learners.q_learning import TabularQLearner


class Runner:
    """Run experiments for all repeats and epochs."""

    def __init__(self, env, prm, record):
        """Initialise Runner input data and objects."""
        self.prm = prm
        self.n_homes = prm['syst']['n_homes']  # number of households / agents
        self.rl = prm['RL']  # learning parameters
        self.env = env
        self.record = record
        self.N = prm['syst']['N']
        self._initialise_buffer_learner_mac()
        # create an instance of the explorer object
        # which will interact with the environment
        self.explorer = Explorer(env, prm, self.learner, record, self.mac)

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
            for epoch in tqdm(range(self.rl['n_epochs']), position=0, leave=True):
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
                            new_env, evaluation_add1=False
                        )
                    train_steps_vals.append(steps_vals)

                    if self.rl['type_learning'] == 'facmac' and self.n_homes > 0:
                        # insert episode batch in buffer, sample, train
                        self._facmac_episode_batch_insert_and_sample(epoch)

                    # append record
                    for info in ['seed', 'n_not_feas']:
                        self.record.__dict__[info][repeat][epoch] = train_steps_vals[-1][info]

                    model_save_time = self._save_nn_model(model_save_time)

                # learning step at the end of the exploration
                # if it was not done instantly after each step
                t_start_learn = time.time()
                self._post_exploration_learning(epoch, train_steps_vals)
                self.explorer.duration_learning += time.time() - t_start_learn

                # evaluation step
                evaluations_methods = self._check_if_opt_env_needed(epoch, evaluation=True)
                assert i_explore + 1 == self.rl['n_explore']

                time_start_test = time.time()
                eval_steps, _ = self.explorer.get_steps(
                    evaluations_methods, repeat, epoch, self.rl['n_explore'],
                    evaluation=True, new_episode_batch=self.new_episode_batch
                )
                duration_test = time.time() - time_start_test

                # record
                for info in ['seed', 'n_not_feas']:
                    self.record.__dict__[info][repeat][epoch] = eval_steps[info]
                duration_epoch = time.time() - t_start

                # make a list, one exploration after the other
                # rather than a list of 'explorations' in 2D
                list_train_stepvals = self._train_vals_to_list(
                    train_steps_vals, exploration_methods, i_explore
                )
                self.record.end_epoch(
                    epoch, eval_steps, list_train_stepvals,
                    self.rl, self.learner, duration_epoch, duration_test
                )

                if self.rl['deterministic']:
                    converged = self._check_convergence(repeat, epoch, converged)
                self._end_of_epoch_parameter_updates(repeat, epoch)

            # then do evaluation only for one month, no learning
            evaluations_methods = self._check_if_opt_env_needed(epoch + 1, evaluation=True)
            self._end_evaluation(
                repeat, new_env, evaluations_methods, i0_costs, delta, date0
            )

            new_env = True \
                if ((repeat + 1) % self.rl['n_init_same_env'] == 0
                    or self.rl['deterministic'] == 2) \
                else False
            self.record.save(end_of='repeat')
            repeat += 1

        for ext in self.prm['syst']['n_homes_extensions_all']:
            if len(self.explorer.data.seeds[ext]) > len(self.rl['seeds'][ext]):
                self.rl['seeds'][ext] = self.explorer.seeds[ext].copy()
        self.record.duration_learning = self.explorer.duration_learning

    def _initialise_buffer_learner_mac_facmac(self, method):
        rl = self.rl
        if 'buffer' not in self.__dict__:
            self.buffer = {}
            self.mac = {}
        self.buffer[method] = ReplayBuffer(
            rl['scheme'], rl['groups'],
            rl['buffer_size'],
            rl['env_info']["episode_limit"] + 1 if rl['runner_scope'] == "episodic" else 2,
            preprocess=rl['preprocess'],
            device="cpu" if rl['buffer_cpu_only']
            else rl['device'],
        )

        # Setup multiagent controller here
        self.mac[method] = mac_REGISTRY[rl['mac']](
            self.buffer[method].scheme, rl['groups'],
            rl, self.N
        )
        target_mac = mac_REGISTRY[rl['mac']](
            self.buffer[method].scheme, rl['groups'],
            rl, self.N
        )
        self.new_episode_batch = \
            partial(EpisodeBatch, rl['scheme'],
                    rl['groups'],
                    rl['batch_size_run'],
                    rl['episode_limit'] + 1,
                    preprocess=rl['preprocess'],
                    device=rl['device'])
        self.learner[method] = le_REGISTRY[rl['learner']](
            self.mac[method],
            self.buffer[method].scheme,
            rl,
            self.N,
            target_mac,
        )
        if rl['use_cuda']:
            self.learner[method].cuda()

    def _initialise_buffer_learner_mac_deep_learning(self, method):
        if self.rl['distr_learning'] == 'decentralised':
            if method in self.learner:
                # the learner as already been intialised; simply reset
                for home in range(self.n_homes):
                    self.learner[method][home].reset()
            else:
                if self.rl['type_learning'] == 'DDPG':
                    self.learner[method] = [Learner_DDPG(
                        self.rl, method + f'_{home}') for home in range(self.n_homes)]
                elif self.rl['type_learning'] == 'DDQN':
                    self.learner[method] = [Agent_DDQN(
                        self.env, self.rl, method) for _ in range(self.n_homes)]
                else:
                    self.learner[method] = [Agent_DQN(
                        self.rl, method + f'_{home}', method, self.N)
                        for home in range(self.n_homes)]
        else:
            if method in self.learner:
                # the learner as already been intialised;
                # simply reset
                self.learner[method].reset()  # reset learner
            else:  # initialise objects
                if self.rl['type_learning'] == 'DDPG':
                    self.learner[method] = Learner_DDPG(self.rl, method)
                elif self.rl['type_learning'] == 'DQN':
                    self.learner[method] = Agent_DQN(
                        self.rl, method, method, self.N)
                elif self.rl['type_learning'] == 'DDQN':
                    self.learner[method] = Agent_DDQN(self.env, self.rl, method)

    def _initialise_buffer_learner_mac(self, repeat=0):
        if self.rl['type_learning'] in ['DDPG', 'DQN', 'DDQN', 'facmac']:
            if 'learner' not in self.__dict__:
                self.learner = {}
            for method in self.rl['type_Qs']:
                if self.rl['type_learning'] == 'facmac':
                    self._initialise_buffer_learner_mac_facmac(method)
                else:
                    self._initialise_buffer_learner_mac_deep_learning(method)

        elif self.rl['type_learning'] == 'q_learning':
            if 'learner' in self.__dict__:
                # the Q learner needs to initialise epsilon
                # and temperature values, and set counters
                # and Q tables to 0
                self.learner.new_repeat(repeat)
            else:  # create one instance for all types
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

    def _set_date(
            self,
            repeat,
            epoch,
            i_explore,
            date0,
            delta,
            i0_costs,
            new_env,
            evaluation=False
    ) -> Tuple[date, timedelta, int]:
        if self.rl['deterministic'] > 0:
            new_date = True if epoch == 0 and i_explore == 0 and new_env == 1 \
                else False
        else:
            new_date = True if self.prm['syst']['change_start'] else False
        if new_date:
            test_str_ = test_str(evaluation)
            seed = self.explorer.data.get_seed_ind(repeat, epoch, i_explore)
            utils.set_seeds_rdn(seed)
            delta_days = int(np.random.choice(range(
                (self.prm['syst'][f'max_date_end{test_str_}_dtm']
                    - self.prm['syst'][f'date0{test_str_}_dtm']).days
                - self.prm['syst']['D'])))
            date0 = self.prm['syst'][f'date0{test_str_}_dtm'] \
                + datetime.timedelta(days=delta_days)
            delta = date0 - self.prm['syst'][f'date0{test_str_}_dtm']
            i0_costs = int(delta.days * 24 + delta.seconds / 3600)
            self.env.update_date(i0_costs, date0)

        return date0, delta, i0_costs

    def _facmac_episode_batch_insert_and_sample(self, epoch):
        for t_explo in self.rl["exploration_methods"]:
            methods_to_update = utils.methods_learning_from_exploration(t_explo, epoch, self.rl)
            for method in methods_to_update:
                diff = True if utils.reward_type(method) == 'd' else False
                opt = True if utils.data_source(method) == 'opt' else False

                self.buffer[method].insert_episode_batch(
                    self.episode_batch[method], difference=diff, optimisation=opt
                )

                if self.buffer[method].can_sample(self.rl['facmac']['batch_size']) \
                        and (self.buffer[method].episodes_in_buffer
                             > self.rl['buffer_warmup']):
                    episode_sample = self.buffer[method].sample(
                        self.rl['facmac']['batch_size']
                    )

                    # Truncate batch to only filled time steps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != self.rl['device']:
                        episode_sample.to(self.rl['device'])
                    self.learner[method].train(
                        episode_sample, self.explorer.t_env
                    )

    def _train_vals_to_list(self, train_steps_vals, exploration_methods, i_explore):
        list_train_stepvals = {method: {} for method in self.rl["exploration_methods"]}
        train_vals = [
            info for info in train_steps_vals[0][self.rl["exploration_methods"][0]].keys()
            if info not in ['seeds', 'n_not_feas']
            and info in train_steps_vals[0][exploration_methods[0]].keys()
        ]
        for method in self.rl["exploration_methods"]:
            for info in train_vals:
                shape0 = np.shape(train_steps_vals[0][exploration_methods[0]][info])
                if (
                    info[0: len('indiv')] == 'indiv'
                    or info in self.prm['syst']['indiv_step_vals_entries']
                    or (info == 'reward' and self.prm['RL']['competitive'])

                ):
                    shape0 = list(shape0)
                    shape0[1] = self.n_homes
                    shape0 = tuple(shape0)
                    train_steps_vals[i_explore][method][info] = \
                        train_steps_vals[i_explore][method][info][:, :self.n_homes]

                new_shape = (self.rl['n_explore'] * shape0[0], ) + shape0[1:]
                list_train_stepvals[method][info] = np.full(new_shape, np.nan)
                for i_explore in range(self.rl['n_explore']):
                    if method in exploration_methods:
                        try:
                            if (
                                info[0: len('indiv')] == 'indiv'
                                or info in self.prm['syst']['indiv_step_vals_entries']
                                or (info == 'reward' and self.prm['RL']['competitive'])
                            ):
                                list_train_stepvals[method][info][
                                    i_explore * self.N: (i_explore + 1) * self.N
                                ] = train_steps_vals[i_explore][method][info][:, 0: self.n_homes]
                            else:
                                list_train_stepvals[method][info][
                                    i_explore * self.N: (i_explore + 1) * self.N
                                ] = train_steps_vals[i_explore][method][info]
                        except Exception:
                            # these may be recorded differently for optimisation,
                            # e.g. no grid_energy_costs, etc.
                            pass

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
                for home in range(self.n_homes):
                    self.learner[method][home].ActionStateModel.T = \
                        self.learner[method][home].ActionStateModel.T * \
                        self.rl['T_decay_param']

    def _check_if_opt_env_needed(self, epoch, evaluation=False):
        opts_in_eval = sum(
            method != "opt" and method.startswith("opt") for method in self.rl["evaluation_methods"]
        ) > 0
        opt_stage = False
        for method in self.rl["evaluation_methods"]:
            if not evaluation and len(method.split('_')) == 5 and int(method.split('_')[3]) > epoch:
                opt_stage = True
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
                and not eval_stage \
                and not opt_stage:
            types_needed = [method for method in candidate_types if method[0:3] != 'opt']
        else:
            types_needed = candidate_types
        if opt_stage:
            types_needed = [method for method in types_needed if len(method.split("_")) < 5]
        if utils.should_optimise_for_supervised_loss(epoch, self.rl) and 'opt' not in types_needed:
            types_needed.append('opt')

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

        if self.rl['type_learning'] == 'facmac':
            if self.rl['facmac']['epsilon_decay']:
                for method in self.mac:
                    self.mac[method].epsilon *= self.rl['facmac']['epsilon_decay_param'][method]
                    self.learner[method].target_mac.epsilon \
                        *= self.rl['facmac']['epsilon_decay_param'][method]
            if self.rl['facmac']['lr_decay']:
                for method in self.learner:
                    self.learner[method].lr *= self.rl['facmac']['lr_decay_param']
                    self.learner[method].critic_lr *= self.rl['facmac']['critic_lr_decay_param']

        elif self.rl['type_learning'] == 'DQN' and self.rl['DQN']['T_decay']:
            self._DQN_T_decay()

    def _exploration_episode(
            self, repeat, epoch, i_explore, date0, delta,
            i0_costs, new_env, evaluation_add1=False,
            set_date=True
    ):
        if set_date:
            date0, delta, i0_costs = self._set_date(
                repeat, epoch, i_explore, date0, delta, i0_costs, new_env, evaluation=False
            )

        # exploration - obtain experience
        exploration_methods = self._check_if_opt_env_needed(epoch, evaluation=False)
        steps_vals, self.episode_batch = self.explorer.get_steps(
            exploration_methods, repeat, epoch, i_explore,
            new_episode_batch=self.new_episode_batch, evaluation=False
        )

        return steps_vals, date0, delta, i0_costs, exploration_methods

    def _save_nn_model(self, model_save_time):
        if self.prm["save"]["save_nns"] and (
                self.explorer.t_env - model_save_time
                >= self.rl['save_model_interval']
                or model_save_time == 0):
            model_save_time = self.explorer.t_env

            if self.prm['RL']['type_learning'] == 'facmac':
                for evaluation_method in self.rl["evaluation_methods"]:
                    if evaluation_method not in self.learner:
                        continue
                    save_path \
                        = self.prm["paths"]["record_folder"] \
                        / f"models_{evaluation_method}_{self.explorer.t_env}"
                    os.makedirs(save_path, exist_ok=True)
                    self.learner[evaluation_method].save_models(save_path)
        return model_save_time

    def _end_evaluation(
            self, repeat, new_env, evaluations_methods, i0_costs, delta, date0
    ):
        i_explore = 0
        for epoch_test in \
                tqdm(range(self.rl['n_epochs'], self.rl['n_all_epochs']),
                     position=0, leave=True):
            t_start = time.time()  # start recording time
            date0, delta, i0_costs = self._set_date(
                repeat, epoch_test, i_explore, date0,
                delta, i0_costs, new_env, evaluation=True
            )
            eval_steps, _ = self.explorer.get_steps(
                evaluations_methods, repeat, epoch_test, self.rl['n_explore'],
                evaluation=True, new_episode_batch=self.new_episode_batch
            )
            duration_epoch = time.time() - t_start
            duration_test = duration_epoch
            self.record.end_epoch(
                epoch_test, eval_steps, None,
                self.rl, self.learner, duration_epoch, duration_test, end_test=True
            )

    def _post_exploration_learning(self, epoch, train_steps_vals):
        if not self.rl['instant_feedback'] \
                and self.rl['type_learning'] == 'q_learning' \
                and epoch > 0:
            # if we did not learn instantly after each step,
            # learn here after exploration
            self.learner.learn_from_explorations(train_steps_vals, epoch)

        elif self.rl['type_learning'] == 'DQN':
            for method in self.rl['type_Qs']:
                if self.rl['distr_learning'] == 'decentralised':
                    for home in range(self.n_homes):
                        self.learner[method][home].target_update()
                else:
                    self.learner[method].target_update()

    def save_computation_statistics(self):
        for info in self.prm["save"]["pandapower_voltage_entries"]:
            value = getattr(self.explorer.env.network, info) \
                if self.prm["grd"]["compare_pandapower_optimisation"] \
                else None
            setattr(self.record, info, value)

        timer_pp = self.explorer.env.network.timer_pp if self.prm['grd']['manage_voltage'] else []
        timer_comparison = self.explorer.env.network.timer_comparison \
            if self.prm["grd"]['compare_pandapower_optimisation'] else []

        self.record.timer_stats(
            timer_pp, timer_comparison,
            self.explorer.data.timer_optimisation,
            self.explorer.data.timer_feasible_data,
        )


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


def print_description_run(prm, settings):
    description_run = 'current code '
    base_RL_settings = ['type_learning', 'n_repeats', 'n_epochs', 'state_space']
    for setting in base_RL_settings:
        description_run += f"prm['RL'][{setting}] {prm['RL'][setting]} "
    for key, setting in settings.items():
        for subkey, subsetting in setting.items():
            if subkey not in base_RL_settings:
                if subkey[0:3] == 'own':
                    description_run += f"prm[{key}][{subkey}] {sum(subsetting) / len(subsetting)}"
                else:
                    description_run += f"settings[{key}][{subkey}] {subsetting}"

    print(description_run)
    prm['save']['description_run'] = description_run

    return prm


def run(run_mode, settings, no_runs=None):
    prm = input_data.input_paths()

    if run_mode == 1:
        # obtain the number of runs from the longest settings entry
        n_runs = get_number_runs(settings)
        # loop through runs
        for i in range(n_runs):
            remove_old_prms = [info for info in prm if info != 'paths']
            for info in remove_old_prms:
                del prm[info]

            start_time = time.time()  # start recording time
            # obtain run-specific settings

            settings_i = input_data.get_settings_i(settings, i)
            fewer_than_10_homes = settings_i['syst']['n_homes'] < 10
            if 'type_learning' not in settings_i['RL']:
                settings_i['RL']['type_learning'] = 'q_learning' if fewer_than_10_homes \
                    else 'facmac'
                settings_i['RL']['evaluation_methods'] = None if fewer_than_10_homes \
                    else 'env_r_c'

                if 'trajectory' not in settings_i['RL']:
                    settings_i['RL']['trajectory'] = False if fewer_than_10_homes \
                        else True

            else:
                if 'trajectory' not in settings_i['RL']:
                    settings_i['RL']['trajectory'] = True \
                        if settings_i['RL']['type_learning'] == 'facmac' \
                        else False
                if 'evaluation_methods' not in settings['RL']:
                    settings_i['RL']['evaluation_methods'] = 'env_r_c' \
                        if settings_i['RL']['type_learning'] == 'facmac' \
                        else None

            trajectory = settings_i['RL']['trajectory']
            values_traj = {
                'obs_agent_id': {True: False, False: True},
                'nn_type': {True: 'cnn', False: 'linear'},
                'rnn_hidden_dim': {True: 1e3, False: 5e2},
                'optimizer': {True: 'rmsprop', False: 'adam'},
            }
            for info in values_traj.keys():
                if info not in settings_i['RL']:
                    print(f"replace {info}")
                    settings_i['RL'][info] = values_traj[info][trajectory]

            # initialise learning parameters, system parameters and recording
            prm, record = initialise_objects(prm, settings=settings_i)

            prm = print_description_run(prm, settings_i)

            if prm['RL']['type_learning'] == 'facmac':
                # Setting the random seed throughout the modules
                utils.set_seeds_rdn(prm["syst"]["seed"])

            env = LocalElecEnv(prm)
            # second part of initialisation specifying environment
            # with relevant parameters
            record.init_env(env)  # record progress as we train
            runner = Runner(env, prm, record)
            runner.run_experiment()
            runner.save_computation_statistics()
            record.save(end_of='end')  # save progress at end
            post_processing(
                record, env, runner.explorer.data, prm,
                start_time=start_time, settings_i=settings_i, run_mode=run_mode
            )
            print(f"--- {time.time() - start_time} seconds ---")

    # post learning analysis / plotting
    elif run_mode == 2:
        list_runs = [
            int(file_name[3:]) for file_name in os.listdir('outputs/results') if file_name[0] != '.'
        ]
        if isinstance(no_runs, int):
            no_runs = [no_runs]  # the runs need to be in an array

        for no_run in no_runs:
            if no_run not in list_runs:
                print(f"run {no_run} not found")
                continue
            rl, prm = input_data.load_existing_prm(prm, no_run)

            prm, record = initialise_objects(prm, no_run=no_run, run_mode=run_mode)
            # make user defined environment
            env = LocalElecEnv(prm)
            record.init_env(env)  # record progress as we train
            post_processing(record, env, None, prm, no_run=no_run, run_mode=run_mode)

    elif run_mode == 3:
        plot_results_vs_nag()
