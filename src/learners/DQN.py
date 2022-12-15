#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:48:51 2021

@author: floracharbonnier

doing DQN super simple discrete to see how that works
adapted from:
https://github.com/marload/DeepRL-TensorFlow2/blob/master/DQN/DQN_Discrete.py

"""
import math
import random
from collections import deque

import numpy as np
# import wandb
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx('float64')


class ReplayBuffer:
    def __init__(self, rl):
        self.buffer = deque(maxlen=rl['DQN']['buffer_capacity'])
        self.rl = rl

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.rl['DQN']['batch_size'])
        states, actions, rewards, next_states, done = \
            map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.rl['DQN']['batch_size'], -1)
        next_states = np.array(next_states).reshape(
            self.rl['DQN']['batch_size'], -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, rl, t, n_steps):
        self.state_dim = len(rl['state_space'])
        if rl['trajectory']:
            self.state_dim *= n_steps
        # n_actions is discrete number of possible actions
        # per agent per time step
        # dim_actions is the number of decisions to make
        # (e.g. for all agents, for all time steps...)
        # action_dim is how many different possible actions to consider
        # in the Q table overall
        self.action_dim = rl['n_discrete_actions'] ** rl['dim_actions']
        self.rl = rl
        self.model = self._create_model()
        self.eps = rl['DQN']['epsilon0'] if rl['DQN']['epsilon_decay'] \
            else rl['DQN']['eps']
        self.eps_decay = rl['DQN']['epsilon_decay_param'][t]
        self.T = self.rl['T0'][t] if self.rl['DQN']['T_decay'] \
            else self.rl['DQN']['T'][t]
        self.t = 0

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state, eps_greedy=True,
                   rdn_eps_greedy_indiv=False, eps_min=0.01):
        state = np.reshape(state, [1, self.state_dim])
        if self.rl['DQN']['epsilon_decay']:
            self.eps *= self.eps_decay
            self.eps = max(self.eps, eps_min)
        q_value = self.predict(state)[0]
        rdn = np.random.random()

        if eps_greedy:
            if not rdn_eps_greedy_indiv:
                if rdn < self.eps:
                    return random.randint(0, self.action_dim - 1)

        i_max_vals = [i for i in range(len(q_value))
                      if q_value[i] == np.max(q_value)]
        action_epsgreedy = np.random.choice(i_max_vals)

        # if boltzmanns
        values_positive = [v + max(- np.min(q_value), 0) for v in q_value]
        values = [v / sum(values_positive)
                  for v in values_positive] if sum(values_positive) != 0 \
            else [1 / len(q_value) for _ in range(len(q_value))]
        sumexp = sum(math.exp(values[ac] / self.T)
                     for ac in range(len(q_value)))
        pAs = [math.exp(values[ac] / self.T) / sumexp
               for ac in range(len(q_value))]
        cumps = [sum(pAs[0:i]) for i in range(len(q_value))]
        rdn_action = np.random.uniform(0, 1)
        ind_action_bolztmann = [i for i in range(len(q_value))
                                if rdn_action > cumps[i]][-1]

        if self.rl['DQN']['policy'] == 'eps-greedy':
            return action_epsgreedy
        elif self.rl['DQN']['policy'] == 'boltzmann':
            return ind_action_bolztmann
        elif self.rl['DQN']['policy'] == 'mixed':
            return ind_action_bolztmann \
                if eps_greedy and rdn < self.eps \
                else action_epsgreedy

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
        lr = max(self.rl['DQN']['min_alpha'],
                 self.rl['DQN']['alpha']
                 * self.rl['DQN']['decay_alpha'] ** self.t)
        self.t += 1
        K.set_value(self.model.optimizer.learning_rate, lr)

    def _create_model(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(self.rl['DQN']['alpha']))

        return model



class Agent_DQN:
    def __init__(self, rl, name, t, n_steps):
        self.rl = rl
        self.name = name
        self.t = t
        self.n_steps = n_steps
        self.reset()

    def reset(self):
        self.model = ActionStateModel(self.rl, self.t, self.n_steps)
        self.eps = self.model.eps
        self.target_model = ActionStateModel(self.rl, self.t, self.n_steps)
        self.target_update()
        self.buffer = ReplayBuffer(self.rl)

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(1):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.rl['DQN']['batch_size']), actions] = \
                rewards \
                + (1 - done) * next_q_values * self.rl['DQN']['gamma']
            self.model.train(states, targets)

    def sample_action(self, ind_state, eps_greedy=True,
                      rdn_eps_greedy_indiv=False):
        ind_action = self.model.get_action(
            ind_state, eps_greedy=eps_greedy,
            rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)

        return ind_action

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= self.rl['DQN']['batch_size']:
                self.replay()
            self.target_update()

    def learn(self, current_state, action, reward, state, done=False):
        self.buffer.put(current_state, action, reward, state, done)
        if self.buffer.size() >= self.rl['DQN']['batch_size']:
            self.replay()

