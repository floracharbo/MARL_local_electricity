#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:21:17 2021

@author: floracharbonnier
tabular Q learner
"""
import math

import numpy as np

from src.utilities.userdeftools import (data_source, distr_learning,
                                        methods_learning_from_exploration,
                                        reward_type)


# %% TabularQLearner
class TabularQLearner:
    """ Tabular Q-learning learner """
    def __init__(self, env, rl):
        self.name = 'TabularQLearner'
        self.rl = rl
        self.n_agents = env.n_homes
        self.n_states, self.n_actions = {}, {}
        for str_frame, multiplier in zip(['1', 'all'], [1, self.n_agents]):
            self.n_states[str_frame], self.n_actions[str_frame] = \
                [n ** multiplier for n in
                 [env.spaces.n['state'], env.spaces.n['action']]]
        self.global_to_indiv_index = env.spaces.global_to_indiv_index
        self.indiv_to_global_index = env.spaces.indiv_to_global_index
        self.get_space_indexes = env.spaces.get_space_indexes
        self.rand_init_seed = 0
        self.q_tables, self.counter = [
            {type_Q: {} for type_Q in rl['type_Qs']} for _ in range(2)
        ]
        self.repeat = 0
        for info in ['hysteretic', 'alpha']:
            setattr(self, info, self.rl['q_learning'][info])

    def set0(self):
        """ for each repeat, reinitialise q_tables and counters """
        for method in self.rl['type_Qs']:
            str_frame = 'all' if distr_learning(method) in ['Cc0', 'Cd0'] else '1'
            n_homes = self.n_agents if (
                distr_learning(method) in ['d', 'Cd', 'd0']
                or self.rl['competitive']
            ) else 1
            shape = [n_homes, self.n_states[str_frame], self.n_actions[str_frame]]
            if self.rl['initialise_q'] == 'random':
                # minq, maxq = np.load('minq.npy'), np.load('maxq.npy')
                minq, maxq = 0, self.rl['q_noise_0']
                np.random.seed(self.rand_init_seed)
            self.rand_init_seed += 1

            if method[-1] == 'n' or self.rl['initialise_q'] == 'zeros':
                # this is a count = copy of corresponding counter;
                self.q_tables[method] = np.zeros(shape)
            elif self.rl['initialise_q'] == 'bias_towards_0':
                self.q_tables[method] = np.zeros(shape)
                self.q_tables[method][:, :, 0] = self.rl['q_noise_0']

            else:  # this is a normal q table - make random initialisation
                self.q_tables[method] = np.random.uniform(low=minq, high=maxq, size=shape)
            self.counter[method] = np.zeros(shape)

    def new_repeat(self, repeat):
        """ method called at the beginning of a new repeat
        to update epsilon and T values """
        self.repeat = repeat
        self.set0()
        # compute epsilon value(s) based on if they are allocated
        # per method or not and if they are decayed or not
        self.eps = {}
        if self.rl['q_learning']['epsilon_decay']:
            if isinstance(self.rl['q_learning']['epsilon_end'], float):
                self.eps = self.rl['q_learning']['epsilon0']
            else:
                for method in self.rl['evaluation_methods']:
                    self.eps[method] = self.rl['q_learning']['epsilon0']
        else:
            if isinstance(self.rl['q_learning']['eps'], (float, int)):
                self.eps = self.rl['q_learning']['eps']
            else:
                for method in self.rl['eval_action_choice']:
                    self.eps[method] = self.rl['q_learning']['eps'][method]
        self.T = self.rl['T0'] if self.rl['q_learning']['T_decay'] \
            else self.rl['q_learning']['T']

    def learn_from_explorations(self, train_steps_vals, epoch):
        for i_explore in range(self.rl['n_explore']):
            current_exploration_methods = [
                method for method in train_steps_vals[i_explore]
                if method in self.rl['exploration_methods']
            ]
            for method in current_exploration_methods:
                # learn for each experience bundle individually
                # for the tabular Q learner
                self.learn(method, train_steps_vals[i_explore][method], epoch)

    def sample_action(self, q, ind_state, home, eps_greedy=True):
        ind_action = []
        i_table = home if distr_learning(q) == 'd' else 0
        q_table = np.array(self.q_tables[q][i_table])
        for s in ind_state:
            n_action = len(q_table[s])
            rdn_eps = np.random.uniform(0, 1)
            rdn_action = np.random.uniform(0, 1)
            rdn_max = np.random.uniform(0, 1)
            if self.rl['q_learning']['policy'] in ['eps-greedy', 'mixed']:
                ps = np.ones(n_action) * (1 / n_action)
                cumps = np.cumsum(ps)
                random_ind_action = np.asarray(cumps > rdn_action).nonzero()[0][0]
                # random_ind_action = [i for i in range(n_action) if rdn_action < cumps[i]][0]
                actions_maxval = np.asarray(q_table[s] == np.max(q_table[s])).nonzero()[0]
                cump_maxac = np.cumsum(np.ones(len(actions_maxval)) * 1 / len(actions_maxval))
                greedy_ind_action = actions_maxval[np.asarray(cump_maxac > rdn_max).nonzero()[0][0]]
                if isinstance(self.eps, (float, int)):
                    eps = self.eps
                else:
                    eps = self.eps[q] \
                        if q in self.rl['eval_action_choice'] else None

            if self.rl['q_learning']['policy'] in ['boltzmann', 'mixed']:
                actions = range(len(q_table[s]))
                values_positive = \
                    [v + max(- np.min(q_table[s]), 0) for v in q_table[s]]
                values = [v / sum(values_positive)
                          for v in values_positive] \
                    if sum(values_positive) != 0 \
                    else [1 / len(actions) for _ in actions]
                sum_exp = sum(math.exp(values[home] / self.T[q]) for home in actions)

                pAs = [math.exp(values[action] / self.T[q]) / sum_exp
                       for action in actions]
                cumps = [sum(pAs[0:i]) for i in range(n_action)]
                ind_action_bolztmann = \
                    [i for i in range(n_action)
                     if rdn_action > cumps[i]][-1]
            if self.rl['q_learning']['policy'] == 'eps-greedy':
                ind_action_epsgreedy = random_ind_action \
                    if eps_greedy and rdn_eps < eps \
                    else greedy_ind_action
                ind_action.append(ind_action_epsgreedy)
            elif self.rl['q_learning']['policy'] == 'boltzmann':
                ind_action.append(ind_action_bolztmann)
            elif self.rl['q_learning']['policy'] == 'mixed':
                ind_action_mixed = ind_action_bolztmann \
                    if eps_greedy and rdn_eps < eps \
                    else greedy_ind_action
                ind_action.append(ind_action_mixed)
        if distr_learning(q) == 'C':
            ind_action = self.global_to_indiv_index(
                'global_action', ind_action[0])

        return ind_action

    def update_q(
            self, reward, done, ind_state, ind_action, ind_next_state, epoch,
            i_table=0, q_table_name=None
    ):
        val_q = 0 if ind_next_state is None or np.isnan(ind_next_state) \
            else max(self.q_tables[q_table_name][i_table][ind_next_state])
        qs_state = self.q_tables[q_table_name][i_table][ind_state]
        if type(val_q) in [list, np.ndarray]:
            print(f'val_q {val_q}')
        value = reward + (not done) * self.rl['q_learning']['gamma'] * val_q
        if type(value) in [list, np.ndarray]:
            print(f'value {value}')
        if ind_action is not None:
            td_error = value - qs_state[ind_action]
            if type(td_error) in [list, np.ndarray]:
                print(f'td_error {td_error}')
            add_supervised_loss = True \
                if data_source(q_table_name, epoch) == 'opt' \
                and self.rl['supervised_loss'] \
                and epoch < self.rl['n_epochs_supervised_loss'] \
                else False
            if add_supervised_loss:
                n_possible_actions = len(qs_state)
                lE = np.ones(n_possible_actions) * self.rl['expert_margin']
                lE[ind_action] = 0
                Q_plus_lE = np.array(qs_state) + lE
                supervised_loss = np.max(Q_plus_lE) - qs_state[ind_action]
                td_error += self.rl['supervised_loss_weight'] * supervised_loss
            lr = self.get_lr(td_error, q_table_name)
            self.q_tables[q_table_name][i_table][ind_state][ind_action] += lr * td_error
            self.counter[q_table_name][i_table][ind_state][ind_action] += 1

    def get_lr(self, td_error, q):
        if isinstance(self.alpha, (int, float)):
            lr = self.alpha if (not self.hysteretic or td_error > 0) \
                else self.alpha * self.rl['q_learning']['beta_to_alpha']
        else:
            beta_to_alpha = self.rl['beta_to_alpha'] \
                if isinstance(self.rl['beta_to_alpha'], float) \
                else self.rl['q_learning']['beta_to_alpha'][q]
            lr = self.alpha[q] \
                if (not self.hysteretic or td_error > 0) \
                else self.alpha[q] * beta_to_alpha

        return lr

    def learn(self, t_explo, step_vals, epoch, step=None):
        q_to_update = methods_learning_from_exploration(t_explo, epoch, self.rl)
        rangestep = range(len(step_vals['reward'])) if step is None else [step]
        for step in rangestep:
            for q in q_to_update:
                self.update_q_step(q, step, step_vals, epoch)

    def _control_decrease_eps(self, method, epoch, mean_eval_rewards, decrease_eps):
        n_window \
            = self.rl['control_window_eps'][method] * self.rl['n_explore']
        if epoch >= self.rl['control_window_eps'][method]:
            baseline_last = sum(mean_eval_rewards['baseline'][
                                - n_window:])
            baseline_prev = \
                sum(mean_eval_rewards['baseline']
                    [- 2 * self.rl['control_window_eps'][method]
                     * self.rl['n_explore']: - n_window])
            t_eval = method
            reward_last = sum(mean_eval_rewards[t_eval][- n_window:])
            reward_prev = sum(mean_eval_rewards[t_eval]
                              [- 2 * n_window: - n_window])
            decrease_eps[method] = (
                True
                if (reward_last - baseline_last)
                >= (reward_prev - baseline_prev)
                else False
            )
        else:
            decrease_eps[method] = True

        return decrease_eps

    def _reward_based_eps_control(self, mean_eval_rewards):
        # XU et al. 2018 Reward-Based Exploration
        k = {}
        for method in self.rl['eval_action_choice']:
            self.rMT = self.rMT / self.rl['tauMT'][method] + np.mean(
                mean_eval_rewards[method][- self.rl['n_explore']:])
            self.rLT = self.rLT / self.rl['tauLT'][method] + self.rMT
            sum_exp = np.exp(self.rMT / self.rl['q_learning']['T'][method]) \
                + np.exp(self.rLT / self.rl['q_learning']['T'][method])
            k[method] = (
                np.exp(self.rMT / self.rl['q_learning']['T'][method])
                - np.exp(self.rLT / self.rl['q_learning']['T'][method])
            ) / sum_exp

        assert not (isinstance(self.eps, (float, int))), \
            "have eps per method"
        for method in self.rl['eval_action_choice']:
            eps = self.rl['lambda'] * k[method] \
                + (1 - self.rl['lambda']) * self.eps[method]
            self.eps[method] = min(1, max(0, eps))

    def epsilon_decay(self, repeat, epoch, mean_eval_rewards):
        mean_eval_rewards = mean_eval_rewards[repeat]
        decrease_eps = {}
        if self.rl['q_learning']['control_eps'] == 2:
            self._reward_based_eps_control(mean_eval_rewards)

        else:
            for method in self.rl['eval_action_choice']:
                if self.rl['q_learning']['control_eps'] == 1:
                    decrease_eps = self._control_decrease_eps(
                        method, epoch, mean_eval_rewards, decrease_eps
                    )
                else:
                    decrease_eps[method] = True

            if isinstance(self.eps, (float, int)):
                decrease_eps = True \
                    if sum(1 for method in self.rl['eval_action_choice']
                           if decrease_eps[method]) > 0 \
                    else False
                factor = self.rl['epsilon_decay_param'] if decrease_eps \
                    else (1 / self.rl['epsilon_decay_param'])
                self.eps = self.eps * factor if self.eps * factor <= 1 else 1
                if epoch < self.rl['q_learning']['start_decay']:
                    self.eps = 1
                if epoch >= self.rl['q_learning']['end_decay']:
                    self.eps = 0
            else:
                for method in self.rl['eval_action_choice']:
                    factor = self.rl['epsilon_decay_param'][method] \
                        if decrease_eps[method] \
                        else (1 / self.rl['epsilon_decay_param'][method])
                    self.eps[method] = min(
                        max(0, self.eps[method] * factor), 1)
                    if epoch < self.rl['q_learning']['start_decay']:
                        self.eps[method] = 1
                    if epoch >= self.rl['q_learning']['end_decay']:
                        self.eps[method] = 0

    def _get_reward_home(self, diff_rewards, indiv_grid_battery_costs, reward, q, home):
        if reward_type(q) == 'd':
            reward_a = diff_rewards[home]
        elif self.rl['competitive']:
            reward_a = indiv_grid_battery_costs[home]
        else:
            reward_a = reward

        return reward_a

    def update_q_step(self, q, step, step_vals, epoch):
        reward, diff_rewards, indiv_grid_battery_costs = [
            step_vals[key][step]
            for key in ["reward", "diff_rewards", "indiv_grid_battery_costs"]
        ]
        if len(np.shape(diff_rewards)) == 2 and np.shape(diff_rewards)[1] == 1:
            diff_rewards = np.reshape(diff_rewards, (len(diff_rewards),))
        [ind_global_s, ind_global_ac, indiv_s, indiv_ac, ind_next_global_s, next_indiv_s, done] = [
            step_vals[e][step] for e in [
                'ind_global_state', 'ind_global_action', 'state', 'action',
                'ind_next_global_state', 'next_state', 'done'
            ]
        ]

        ind_indiv_ac = self.get_space_indexes(
            done=done, all_vals=indiv_ac, value_type='action')
        ind_indiv_s, ind_next_indiv_s = [
            self.get_space_indexes(done=done, all_vals=vals, value_type=types)
            for vals, types in zip([indiv_s, next_indiv_s], ['state', 'next_state'])
        ]

        if reward_type(q) == 'n':
            for home in range(self.n_agents):
                if indiv_ac[home] is not None:
                    i_table = home if distr_learning(q) == 'd' else 0
                    self.q_tables[q][i_table][ind_indiv_s[home]][
                        ind_indiv_ac[home]] += 1
                    self.counter[q][i_table][ind_indiv_s[home]][ind_indiv_ac[home]] += 1
        else:
            if reward_type(q) == 'A':
                self.advantage_update_q_step(
                    q, indiv_ac, reward, done, ind_indiv_ac, ind_indiv_s, ind_next_indiv_s,
                    ind_global_ac, ind_global_s, ind_next_global_s, epoch
                )

            elif distr_learning(q) in ['Cc', 'Cd']:
                # this is env_d_C or opt_d_C
                # difference to global baseline
                if ind_global_ac[0] is not None:
                    self.update_q(
                        reward[-1], done, ind_next_global_s[0],
                        ind_global_ac[0], ind_next_global_s[0], epoch,
                        i_table=0, q_table_name=q + '0'
                    )
                    for home in range(self.n_agents):
                        i_table = 0 if distr_learning == 'Cc' else home
                        local_q_val = self.q_tables[q][i_table][
                            ind_indiv_s[home]][ind_indiv_ac[home]]
                        global_q_val = self.q_tables[q + '0'][0][
                            ind_global_s[0]][ind_global_ac[0]]
                        error = global_q_val - local_q_val
                        lr = self.get_lr(error, q)
                        self.q_tables[q][i_table][ind_indiv_s[home]][
                            ind_indiv_ac[home]] += lr * error
                        self.counter[q][i_table][ind_indiv_s[home]][
                            ind_indiv_ac[home]] += 1
            else:
                for home in range(self.n_agents):
                    if indiv_ac[home] is not None:
                        i_table = 0 if distr_learning(q) == 'c' else home
                        reward_home = self._get_reward_home(
                            diff_rewards, indiv_grid_battery_costs, reward, q, home
                        )
                        self.update_q(
                            reward_home, done, ind_indiv_s[home], ind_indiv_ac[home],
                            ind_next_indiv_s[home], epoch,
                            i_table=i_table, q_table_name=q
                        )

    def advantage_update_q_step(
            self, q, indiv_ac, reward, done,
            ind_indiv_ac, ind_indiv_s, ind_next_indiv_s,
            ind_global_ac, ind_global_s, ind_next_global_s, epoch
    ):
        if distr_learning(q) in ['Cc', 'Cd']:
            self._advantage_global_table(
                reward, done, ind_global_s, ind_global_ac, ind_next_global_s,
                indiv_ac, q, ind_indiv_ac, ind_indiv_s, epoch
            )

        elif distr_learning(q) == 'c':
            self._advantage_centralised_table(
                reward, done, ind_indiv_s, ind_indiv_ac,
                ind_next_indiv_s, indiv_ac, q, epoch
            )
        elif distr_learning(q) == 'd':
            self._advantage_decentralised_table(
                reward, done, ind_indiv_s, ind_indiv_ac,
                ind_next_indiv_s, indiv_ac, q, epoch
            )

    def _advantage_global_table(
            self, reward, done, ind_global_s, ind_global_ac, ind_next_global_s,
            indiv_ac, q, ind_indiv_ac, ind_indiv_s, epoch
    ):
        if ind_global_ac[0] is not None:
            self.update_q(
                reward, done, ind_global_s[0], ind_global_ac[0], ind_next_global_s[0], epoch,
                i_table=0, q_table_name=q + '0'
            )
        indiv_ind_actions_baselinea = \
            [[self.rl['dim_actions'] - 1 if i_home == home else ind_indiv_ac[home]
              for i_home in range(self.n_agents)]
             for home in range(self.n_agents)]
        ind_a_global_abaseline = \
            [self.indiv_to_global_index('action', indexes=iab)
             for iab in indiv_ind_actions_baselinea]
        for home in range(self.n_agents):
            if indiv_ac[home] is not None:
                i_table = 0 if distr_learning(q) == 'Cc' else home
                q0 = self.q_tables[q + '0'][0]
                q0_a = q0[ind_global_s[0]][ind_global_ac[0]]
                q0_baseline_a = q0[ind_global_s[0]][ind_a_global_abaseline[home]]
                if type(self.q_tables[q][i_table][ind_indiv_s[home]][
                    ind_indiv_ac[home]]) in [float, np.float64] \
                        and type(q0_a) in [float, np.float64] \
                        and type(q0_baseline_a) in [float, np.float64]:
                    reward_a = q0_a - q0_baseline_a
                    error = reward_a - self.q_tables[q][i_table][
                        ind_indiv_s[home]][ind_indiv_ac[home]]
                    if isinstance(error, list):
                        print(f'type(error) if list,'
                              f' q = {q}  reward_a = {reward_a}')
                    lr = self.get_lr(error, q)
                    self.q_tables[q][i_table][ind_indiv_s[home]][
                        ind_indiv_ac[home]] += lr * error
                    self.counter[q][i_table][ind_indiv_s[home]][
                        ind_indiv_ac[home]] += 1

    def _advantage_centralised_table(
            self, reward, done, ind_indiv_s, ind_indiv_ac,
            ind_next_indiv_s, indiv_ac, q, epoch
    ):
        for home in range(self.n_agents):
            if ind_indiv_ac[home] is not None:
                self.update_q(
                    reward, done, ind_indiv_s[home], ind_indiv_ac[home],
                    ind_next_indiv_s[home], epoch,
                    i_table=0, q_table_name=q + '0'
                )
                q0 = self.q_tables[q + '0'][0]
                reward_a \
                    = q0[ind_indiv_s[home]][ind_indiv_ac[home]] \
                    - q0[ind_indiv_s[home]][-1]
                error = reward_a - \
                    self.q_tables[q][0][ind_indiv_s[home]][ind_indiv_ac[home]]
                lr = self.get_lr(error, q)
                self.q_tables[q][0][ind_indiv_s[home]][
                    ind_indiv_ac[home]] += lr * error
                self.counter[q][0][ind_indiv_s[home]][
                    ind_indiv_ac[home]] += 1

    def _advantage_decentralised_table(
            self, reward, done, ind_indiv_s, ind_indiv_ac,
            ind_next_indiv_s, indiv_ac, q, epoch
    ):
        for home in range(self.n_agents):
            if ind_indiv_ac[home] is not None:
                self.update_q(
                    reward, done, ind_indiv_s[home], ind_indiv_ac[home],
                    ind_next_indiv_s[home], epoch,
                    i_table=home, q_table_name=q + '0'
                )
                q0 = self.q_tables[q + '0'][home][ind_indiv_s[home]][
                    ind_indiv_ac[home]]
                q0_baseline_a = self.q_tables[q + '0'][home][
                    ind_indiv_s[home]][
                    self.rl['dim_actions'] - 1]
                reward_a = q0 - q0_baseline_a
                error = reward_a \
                    - self.q_tables[q][home][ind_indiv_s[home]][ind_indiv_ac[home]]
                lr = self.get_lr(error, q)
                self.q_tables[q][home][ind_indiv_s[home]][
                    ind_indiv_ac[home]] += lr * error
                self.counter[q][home][ind_indiv_s[home]][ind_indiv_ac[home]] += 1
