#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:21:17 2021

@author: floracharbonnier
tabular Q learner
"""
import math

import numpy as np
from utilities.userdeftools import (data_source, distr_learning,
                                    initialise_dict, reward_type)


# %% TabularQLearner
class TabularQLearner:
    """ Tabular Q-learning learner """
    def __init__(self, env, rl):
        self.name = 'TabularQLearner'
        self.rl = rl
        self.n_agents = env.n_agents
        self.n_states, self.n_actions = {}, {}
        for str_frame, multiplier in zip(['1', 'all'], [1, env.n_agents]):
            self.n_states[str_frame], self.n_actions[str_frame] = \
                [n ** multiplier for n in
                 [env.spaces.n['state'], env.spaces.n['action']]]
        self.global_to_indiv_index = env.spaces.global_to_indiv_index
        self.indiv_to_global_index = env.spaces.indiv_to_global_index
        self.get_space_indexes = env.spaces.get_space_indexes
        self.rand_init_seed = 0
        self.q_tables, self.counter = \
            [initialise_dict(rl['type_Qs'], 'empty_dict')
             for _ in range(2)]
        self.repeat = 0
        for e in ['hysteretic', 'alpha']:
            self.__dict__[e] = self.rl['q_learning'][e]

    def set0(self):
        """ for each repeat, reinitialise q_tables and counters """
        for t in self.rl['type_Qs']:
            str_frame = 'all' if distr_learning(t) in ['Cc0', 'Cd0'] else '1'
            shape = [self.n_states[str_frame], self.n_actions[str_frame]]
            if self.rl['initialise_q'] == 'random':
                minq, maxq = np.load('minq.npy'), np.load('maxq.npy')
                np.random.seed(self.rand_init_seed)
            self.rand_init_seed += 1
            na = self.n_agents if (distr_learning(t) in ['d', 'Cd', 'd0']
                                   or self.rl['competitive']) else 1
            if t[-1] == 'n' or self.rl['initialise_q'] == 'zeros':
                # this is a count = copy of corresponding counter;
                self.q_tables[t] = [np.zeros(shape) for _ in range(na)]
            else:  # this is a normal q table - make random initialisation
                self.q_tables[t] = \
                    [np.random.uniform(low=minq, high=maxq, size=shape)
                     for _ in range(na)]
            self.counter[t] = [np.zeros(shape) for a in range(na)]

    def new_repeat(self, repeat):
        """ method called at the beginning of a new repeat
        to update epsilon and T values """
        self.repeat = repeat
        self.set0()
        # compute epsilon value(s) based on if they are allocated
        # per method or not and if they are decayed or not
        self.eps = {}
        if self.rl['q_learning']['epsilon_decay']:
            if type(self.rl['q_learning']['epsilon_end']) is float:
                self.eps = self.rl['q_learning']['epsilon0']
            else:
                for t in self.rl['type_eval']:
                    self.eps[t] = self.rl['q_learning']['epsilon0']
        else:
            if type(self.rl['q_learning']['eps']) is float:
                self.eps = self.rl['q_learning']['eps']
            else:
                for t in self.rl['eval_action_choice']:
                    self.eps[t] = self.rl['q_learning']['eps'][t]
        self.T = self.rl['T0'] if self.rl['q_learning']['T_decay'] \
            else self.rl['q_learning']['T']

    def learn_from_explorations(self, train_steps_vals):
        for i_explore in range(self.rl['n_explore']):
            for t in self.rl['type_explo']:
                # learn for each experience bundle individually
                # for the tabular Q learner
                self.learn(t, train_steps_vals[i_explore][t])

    def sample_action(self, q, ind_state, a, eps_greedy=True):
        ind_action = []
        i_table = a if distr_learning(q) == 'd' else 0
        q_table = self.q_tables[q][i_table]
        for s in ind_state:
            n_action = len(q_table[s])
            rdn_eps = np.random.uniform(0, 1)
            rdn_action = np.random.uniform(0, 1)
            rdn_max = np.random.uniform(0, 1)
            if self.rl['q_learning']['policy'] in ['eps-greedy', 'mixed']:
                ps = [1 / len(q_table[s]) for _ in range(n_action)]
                cumps = [sum(ps[0:i]) for i in range(n_action)]
                random_ind_action = [i for i in range(n_action)
                                     if rdn_action > cumps[i]][-1]
                max_value = max(q_table[s])
                actions_maxval = [ac for ac in range(len(q_table[s]))
                                  if q_table[s][ac] == max_value]
                cump_maxac = [1 / len(actions_maxval) * i
                              for i in range(len(actions_maxval))]
                greedy_ind_action = actions_maxval[
                    [i for i in range(len(cump_maxac))
                     if rdn_max > cump_maxac[i]][-1]]
                if type(self.eps) is float or type(self.eps) is int:
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
                sum_exp = sum(math.exp(values[a] / self.T[q]) for a in actions)

                pAs = [math.exp(values[a] / self.T[q]) / sum_exp
                       for a in actions]
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

    def update_q(self, reward, done, ind_state, ind_action,
                 ind_next_state, i_table=0, q_table_name=None):
        val_q = 0 if ind_next_state is None \
            else max(self.q_tables[q_table_name][i_table][ind_next_state])
        if type(val_q) in [list, np.ndarray]:
            print(f'val_q {val_q}')
        value = reward + (not done) * self.rl['q_learning']['gamma'] * val_q
        if type(value) in [list, np.ndarray]:
            print(f'value {value}')
        if ind_action is not None:
            td_error = value \
                - self.q_tables[q_table_name][i_table][ind_state][ind_action]
            if type(td_error) in [list, np.ndarray]:
                print(f'td_error {td_error}')
            try:
                lr = self.get_lr(td_error, q_table_name)
            except Exception as ex:
                print(f'ex = {ex}')
                print(f'td_error = {td_error}')
                print(f'type(td_error) = {type(td_error)}')
                print(f'value = {value}')
                print(f"ind_action = {ind_action}")
            self.q_tables[q_table_name][i_table][ind_state][ind_action] \
                += lr * td_error
            self.counter[q_table_name][i_table][ind_state][ind_action] += 1

    def get_lr(self, td_error, q):
        if type(self.alpha) is float:
            try:
                lr = self.alpha if (not self.hysteretic or td_error > 0) \
                    else self.alpha * self.rl['q_learning']['beta_to_alpha']
            except Exception as ex:
                print(f"ex = {ex}")
        else:
            beta_to_alpha = self.rl['beta_to_alpha'] \
                if type(self.rl['beta_to_alpha']) is float \
                else self.rl['q_learning']['beta_to_alpha'][q]
            lr = self.alpha[q] \
                if (not self.hysteretic or td_error > 0) \
                else self.alpha[q] * beta_to_alpha
        return lr

    def learn(self, t_explo, step_vals, step=None):
        q_to_update = [] if t_explo == 'baseline' \
            else [t_explo] if t_explo[0:3] == 'env' \
            else [q for q in self.rl['type_Qs']
                  if data_source(q) == 'opt' and q[-1] != '0']
        rangestep = range(len(step_vals['reward'])) if step is None else [step]
        for step in rangestep:
            for q in q_to_update:
                self.update_q_step(q, step, step_vals)

    def _control_decrease_eps(self, t, epoch, mean_eval_rewards, decrease_eps):
        n_window \
            = self.rl['control_window_eps'][t] * self.rl['n_explore']
        if epoch >= self.rl['control_window_eps'][t]:
            baseline_last = sum(mean_eval_rewards['baseline'][
                                - n_window:])
            baseline_prev = \
                sum(mean_eval_rewards['baseline']
                    [- 2 * self.rl['control_window_eps'][t]
                     * self.rl['n_explore']: - n_window])
            t_eval = t
            reward_last = sum(mean_eval_rewards[t_eval][- n_window:])
            reward_prev = sum(mean_eval_rewards[t_eval]
                              [- 2 * n_window: - n_window])
            decrease_eps[t] = (
                True
                if (reward_last - baseline_last)
                >= (reward_prev - baseline_prev)
                else False
            )
        else:
            decrease_eps[t] = True

        return decrease_eps

    def _reward_based_eps_control(self, mean_eval_rewards):
        # XU et al. 2018 Reward-Based Exploration
        k = {}
        for t in self.rl['eval_action_choice']:
            self.rMT = self.rMT / self.rl['tauMT'][t] + np.mean(
                mean_eval_rewards[t][- self.rl['n_explore']:])
            self.rLT = self.rLT / self.rl['tauLT'][t] + self.rMT
            sum_exp = np.exp(self.rMT / self.rl['q_learning']['T'][t]) \
                + np.exp(self.rLT / self.rl['q_learning']['T'][t])
            k[t] = (np.exp(self.rMT / self.rl['q_learning']['T'][t])
                    - np.exp(self.rLT / self.rl['q_learning']['T'][t])) \
                / sum_exp

        assert not (type(self.eps) is float or type(self.eps) is int), \
            "have eps per method"
        for t in self.rl['eval_action_choice']:
            eps = self.rl['lambda'] * k[t] \
                + (1 - self.rl['lambda']) * self.eps[t]
            self.eps[t] = min(1, max(0, eps))

    def epsilon_decay(self, repeat, epoch, mean_eval_rewards):
        mean_eval_rewards = mean_eval_rewards[repeat]
        decrease_eps = {}
        if self.rl['control_eps'] == 2:
            self._reward_based_eps_control(mean_eval_rewards)

        else:
            for t in self.rl['eval_action_choice']:
                if self.rl['control_eps'] == 1:
                    decrease_eps = self._control_decrease_eps(
                        t, epoch, mean_eval_rewards, decrease_eps
                    )
                else:
                    decrease_eps[t] = True

            if type(self.eps) is float or type(self.eps) is int:
                decrease_eps = True \
                    if sum(1 for t in self.rl['eval_action_choice']
                           if decrease_eps[t]) > 0 \
                    else False
                factor = self.rl['epsilon_decay_param'] if decrease_eps \
                    else (1 / self.rl['epsilon_decay_param'])
                self.eps = self.eps * factor if self.eps * factor <= 1 else 1
                if epoch < self.rl['q_learning']['start_decay']:
                    self.eps = 1
                if epoch >= self.rl['q_learning']['end_decay']:
                    self.eps = 0
            else:
                for t in self.rl['eval_action_choice']:
                    factor = self.rl['epsilon_decay_param'][t] \
                        if decrease_eps[t] \
                        else (1 / self.rl['epsilon_decay_param'][t])
                    self.eps[t] = min(
                        max(0, self.eps[t] * factor), 1)
                    if epoch < self.rl['q_learning']['start_decay']:
                        self.eps[t] = 1
                    if epoch >= self.rl['q_learning']['end_decay']:
                        self.eps[t] = 0

    def _get_reward_a(self, diff_rewards, indiv_rewards, reward, q, a):
        if reward_type(q) == 'd':
            reward_a = diff_rewards[a]
        elif self.rl['competitive']:
            reward_a = indiv_rewards[a]
        else:
            reward_a = reward

        return reward_a

    def update_q_step(self, q, step, step_vals):
        reward, diff_rewards, indiv_rewards = [
            step_vals[key][step]
            for key in ["reward", "diff_rewards", "indiv_rewards"]
        ]

        [ind_global_s, ind_global_ac, indiv_s, indiv_ac,
         ind_next_global_s, next_indiv_s, done] = \
            [step_vals[e][step]
             for e in ['ind_global_state', 'ind_global_action',
                       'state', 'action', 'ind_next_global_state',
                       'next_state', 'done']]

        ind_indiv_ac = self.get_space_indexes(
            done=done, all_vals=indiv_ac, type_='action')
        ind_indiv_s, ind_next_indiv_s = \
            [self.get_space_indexes(
                done=done, all_vals=vals, type_=types)
                for vals, types in zip([indiv_s, next_indiv_s],
                                       ['state', 'next_state'])]

        if reward_type(q) == 'n':
            for a in range(self.n_agents):
                if indiv_ac[a] is not None:
                    i_table = a if distr_learning(q) == 'd' else 0
                    self.q_tables[q][i_table][ind_indiv_s[a]][
                        ind_indiv_ac[a]] += 1
                    self.counter[q][i_table][ind_indiv_s[a]][
                        ind_indiv_ac[a]] += 1
        else:
            if reward_type(q) == 'A':
                self.advantage_update_q_step(
                    q, indiv_ac, reward, done,
                    ind_indiv_ac, ind_indiv_s, ind_next_indiv_s,
                    ind_global_ac, ind_global_s, ind_next_global_s
                )

            elif distr_learning(q) in ['Cc', 'Cd']:
                # this is env_d_C or opt_d_C
                # difference to global baseline
                if ind_global_ac[0] is not None:
                    self.update_q(reward[-1], done,
                                  ind_next_global_s[0],
                                  ind_global_ac[0],
                                  ind_next_global_s[0],
                                  i_table=0, q_table_name=q + '0')

                    for a in range(self.n_agents):
                        i_table = 0 if distr_learning == 'Cc' else a
                        local_q_val = self.q_tables[q][i_table][
                            ind_indiv_s[a]][ind_indiv_ac[a]]
                        global_q_val = self.q_tables[q + '0'][0][
                            ind_global_s[0]][ind_global_ac[0]]
                        error = global_q_val - local_q_val
                        lr = self.get_lr(error, q)
                        self.q_tables[q][i_table][ind_indiv_s[a]][
                            ind_indiv_ac[a]] += lr * error
                        self.counter[q][i_table][ind_indiv_s[a]][
                            ind_indiv_ac[a]] += 1
            else:
                for a in range(self.n_agents):
                    if indiv_ac[a] is not None:
                        i_table = 0 if distr_learning(q) == 'c' else a
                        reward_a = self._get_reward_a(
                            diff_rewards, indiv_rewards, reward, q, a
                        )
                        self.update_q(
                            reward_a, done, ind_indiv_s[a],
                            ind_indiv_ac[a], ind_next_indiv_s[a],
                            i_table=i_table, q_table_name=q)

    def advantage_update_q_step(
            self, q, indiv_ac, reward, done,
            ind_indiv_ac, ind_indiv_s, ind_next_indiv_s,
            ind_global_ac, ind_global_s, ind_next_global_s):
        if distr_learning(q) in ['Cc', 'Cd']:
            self._advantage_global_table(
                reward, done, ind_global_s, ind_global_ac,
                ind_next_global_s, indiv_ac, q, ind_indiv_ac, ind_indiv_s
            )

        elif distr_learning(q) == 'c':
            self._advantage_centralised_table(
                reward, done, ind_indiv_s, ind_indiv_ac,
                ind_next_indiv_s, indiv_ac, q
            )
        elif distr_learning(q) == 'd':
            self._advantage_decentralised_table(
                reward, done, ind_indiv_s, ind_indiv_ac,
                ind_next_indiv_s, indiv_ac, q
            )

    def _advantage_global_table(
            self, reward, done, ind_global_s, ind_global_ac,
            ind_next_global_s, indiv_ac, q, ind_indiv_ac, ind_indiv_s
    ):
        if ind_global_ac[0] is not None:
            self.update_q(
                reward, done, ind_global_s[0],
                ind_global_ac[0], ind_next_global_s[0],
                i_table=0, q_table_name=q + '0')
        indiv_ind_actions_baselinea = \
            [[self.rl['dim_actions'] - 1 if i_a == a else ind_indiv_ac[a]
              for i_a in range(self.n_agents)]
             for a in range(self.n_agents)]
        ind_a_global_abaseline = \
            [self.indiv_to_global_index('action', indexes=iab)
             for iab in indiv_ind_actions_baselinea]
        for a in range(self.n_agents):
            if indiv_ac[a] is not None:
                i_table = 0 if distr_learning(q) == 'Cc' else a
                q0 = self.q_tables[q + '0'][0]
                q0_a = q0[ind_global_s[0]][ind_global_ac[0]]
                q0_baseline_a = q0[ind_global_s[0]][ind_a_global_abaseline[a]]
                if type(self.q_tables[q][i_table][ind_indiv_s[a]][
                    ind_indiv_ac[a]]) in [float, np.float64] \
                        and type(q0_a) in [float, np.float64] \
                        and type(q0_baseline_a) in [float, np.float64]:
                    reward_a = q0_a - q0_baseline_a
                    error = reward_a - self.q_tables[q][i_table][
                        ind_indiv_s[a]][ind_indiv_ac[a]]
                    if type(error) is list:
                        print(f'type(error) if list,'
                              f' q = {q}  reward_a = {reward_a}')
                    lr = self.get_lr(error, q)
                    self.q_tables[q][i_table][ind_indiv_s[a]][
                        ind_indiv_ac[a]] += lr * error
                    self.counter[q][i_table][ind_indiv_s[a]][
                        ind_indiv_ac[a]] += 1

    def _advantage_centralised_table(
            self, reward, done, ind_indiv_s, ind_indiv_ac,
            ind_next_indiv_s, indiv_ac, q
    ):
        for a in range(self.n_agents):
            if ind_indiv_ac[a] is not None:
                self.update_q(reward, done, ind_indiv_s[a],
                              ind_indiv_ac[a], ind_next_indiv_s[a],
                              i_table=0, q_table_name=q + '0')
                q0 = self.q_tables[q + '0'][0]
                reward_a \
                    = q0[ind_indiv_s[a]][ind_indiv_ac[a]] \
                    - q0[ind_indiv_s[a]][-1]
                error = reward_a - \
                    self.q_tables[q][0][ind_indiv_s[a]][ind_indiv_ac[a]]
                lr = self.get_lr(error, q)
                self.q_tables[q][0][ind_indiv_s[a]][
                    ind_indiv_ac[a]] += lr * error
                self.counter[q][0][ind_indiv_s[a]][
                    ind_indiv_ac[a]] += 1

    def _advantage_decentralised_table(
            self, reward, done, ind_indiv_s, ind_indiv_ac,
            ind_next_indiv_s, indiv_ac, q
    ):
        for a in range(self.n_agents):
            if ind_indiv_ac[a] is not None:
                self.update_q(
                    reward, done, ind_indiv_s[a], ind_indiv_ac[a],
                    ind_next_indiv_s[a], i_table=a, q_table_name=q + '0')
                q0 = self.q_tables[q + '0'][a][ind_indiv_s[a]][
                    ind_indiv_ac[a]]
                q0_baseline_a = self.q_tables[q + '0'][a][
                    ind_indiv_s[a]][
                    self.rl['dim_actions'] - 1]
                reward_a = q0 - q0_baseline_a
                error = reward_a \
                    - self.q_tables[q][a][ind_indiv_s[a]][ind_indiv_ac[a]]
                lr = self.get_lr(error, q)
                self.q_tables[q][a][ind_indiv_s[a]][
                    ind_indiv_ac[a]] += lr * error
                self.counter[q][a][ind_indiv_s[a]][ind_indiv_ac[a]] += 1
