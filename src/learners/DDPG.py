#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/09/21
Description: Implementing DDPG algorithm on the Inverted Pendulum Problem.
## Introduction
**Deep Deterministic Policy Gradient (DDPG)**
is a model-free off-policy algorithm for learning continous actions.
It combines ideas from DPG (Deterministic Policy Gradient)
and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN,
and it is based on DPG,
which can operate over continuous action spaces.
This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning]
(https://arxiv.org/pdf/1509.02971.pdf)
## Problem
We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.
What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.
## Quick theory
Just like the Actor-Critic method, we have two networks:
1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value)
or bad (negative value)
given a state and an action.
DDPG uses two more techniques not present in the original DQN:
**First, it uses two Target networks.**
**Why?** Because it add stability to training.
In short, we are learning from estimated
targets and Target networks are updated slowly,
hence keeping our estimated targets stable.
Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this
entire game after every move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).
**Second, it uses Experience Replay.**
We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience,
we learn from sampling all of our experience accumulated so far.
Now, let's see how is it implemented.
"""

import numpy as np
# import gym
import tensorflow as tf
from tensorflow.keras import layers

"""
We use [OpenAIGym](http://gym.openai.com/docs) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.
"""

"""
To implement better exploration by the Actor network,
we use noisy perturbations, specifically an **Ornstein-Uhlenbeck
process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2,
                 x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    @tf.function
    def __call__(self):
        # Formula taken from
        # https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt)
            * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network.
`y` is a moving target that the critic model tries to achieve;
we make this target stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value
given by the Critic network for the actions taken by the Actor network.
We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, DDPG):
        self.rl = DDPG.rl
        self.sample_action = DDPG.sample_action
        for info in ['buffer_capacity', 'critic_lr', 'decay_alpha', 'min_alpha']:
            setattr(self, info, self.rl['DDPG'][info])
        self.reset()
        self.t = 0

    def reset(self):
        self.buffer_counter = 0
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros(
            (self.buffer_capacity, self.rl['dim_states']))
        self.action_buffer = np.zeros(
            (self.buffer_capacity, self.rl['dim_actions']))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros(
            (self.buffer_capacity, self.rl['dim_states']))
        self.t = 0

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2.
    # Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and
    # computations in our function.
    # This provides a large speed up for blocks of code that contain
    # many small TensorFlow operations such as this one.
    # @tf.function
    def update(self, target_actor, target_critic, actor_model, critic_model,
               actor_optimizer, critic_optimizer, state_batch, action_batch,
               reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        state_batch = tf.reshape(state_batch, (-1, 1, self.rl['dim_states']))
        next_state_batch = tf.reshape(
            next_state_batch, (-1, 1, self.rl['dim_states'])
        )
        action_batch = tf.reshape(action_batch, (-1, 1, self.rl['dim_actions']))

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            # target_actions = tf.reshape(target_actions, (-1, 1, self.rl['dim_actions']))
            target_val = target_critic(
                [next_state_batch, target_actions], training=True
            )
            y = reward_batch + self.rl['DDPG']['gamma'] * target_val
            critic_value = critic_model(
                [state_batch, action_batch], training=True
            )
            if self.rl['DDPG']['hysteretic']:
                lr = tf.cond(
                    tf.math.reduce_mean(y - critic_value) < 0,
                    lambda: self.critic_lr * self.rl['DDPG']['beta_to_alpha'],
                    lambda: self.critic_lr)
                critic_optimizer.lr.assign(lr)
                if self.rl['hysteretic_actor']:
                    lr_actor = tf.cond(
                        tf.math.reduce_mean(y - critic_value) < 0,
                        lambda: self.rl['actor_lr']
                        * self.rl['DDPG']['beta_to_alpha'],
                        lambda: self.rl['actor_lr'])
                    actor_optimizer.lr.assign(lr_actor)

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(
            critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )
        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = - tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    # @tf.function
    def learn(self, target_actor, target_critic, actor_model,
              critic_model, actor_optimizer, critic_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(
            record_range, self.rl['DDPG']['batch_size'])

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        # state_batch = tf.reshape(state_batch, (-1, 1, self.rl['dim_states']))
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])
        tf.config.set_soft_device_placement(True)
        self.update(
            target_actor, target_critic, actor_model,
            critic_model, actor_optimizer, critic_optimizer,
            state_batch, action_batch, reward_batch, next_state_batch
        )
        if self.decay_alpha is not None:
            lr = max(self.min_alpha,
                     self.critic_lr * self.decay_alpha ** self.t)
            critic_optimizer.lr.assign(lr)
            lr_actor = max(self.min_alpha,
                           self.rl['actor_lr'] * self.decay_alpha ** self.t)
            actor_optimizer.lr.assign(lr_actor)
            self.t += 1


class Learner_DDPG:
    def __init__(self, rl, name=None):
        self.rl = rl
        self.name = name
        self.reset()

    def reset(self):
        self.actor_model = self.get_actor('actor_model')
        self.critic_model = self.get_critic('critic_model')
        self.target_actor = self.get_actor('target_actor')
        self.target_critic = self.get_critic('target_critic')
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(
            self.rl['DDPG']['critic_lr'])
        self.actor_optimizer = tf.keras.optimizers.Adam(self.rl['actor_lr'])
        self.buffer = Buffer(self)
        self.noise_object = OUActionNoise(
            mean=np.zeros((self.rl['dim_actions'],)),
            std_deviation=float(self.rl['std_dev'])
            * np.ones((self.rl['dim_actions'],)))

    @tf.function
    def update_target(self, target_weights, weights):
        """# This update target parameters slowly
        # Based on rate `tau`, which is much less than one.
        """
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.rl['tau'] + a * (1 - self.rl['tau']))

    def get_actor(self, name_model=None):
        """
        Here we define the Actor and Critic networks.
        These are basic Dense models with `ReLU` activation.
        Note: We need the initialization for last layer of the
        Actor to be between `-0.003` and `0.003` as this prevents us
        from getting `1` or `-1` output values in the initial stages,
        which would squash our gradients to zero,
        as we use the `tanh` activation.
        """
        last_init = tf.random_uniform_initializer(
            minval=-0.003 * self.rl['init_weight_mult'],
            maxval=+0.003 * self.rl['init_weight_mult'])

        model = tf.keras.Sequential()
        if self.rl['LSTM']:
            model.add(layers.LSTM(self.rl['dim_out_layer12'],
                                  input_shape=(1, self.rl['dim_states']),
                                  activation=self.rl['activation'],
                                  name='actor_LSTM1'))
        else:
            model.add(layers.Dense(self.rl['dim_out_layer12'],
                                   input_shape=(1, self.rl['dim_states']),
                                   activation=self.rl['activation'],
                                   name='actor_Dense1'))
        model.add(layers.Dense(self.rl['dim_out_layer12'],
                               activation=self.rl['activation'],
                               name='actor_Dense2'))
        model.add(layers.Dense(self.rl['dim_actions'],
                               activation="tanh",
                               kernel_initializer=last_init,
                               name='actor_Dense3'))
        if self.name is not None:
            model._name = self.name + name_model

        return model

    def get_critic(self, name_model=None):
        # State as input
        if self.rl['LSTM']:
            state_input = layers.Input(
                shape=(1, self.rl['dim_states']), name='critic_stateInput')
        else:
            state_input = layers.Input(
                shape=(1, self.rl['dim_states']), name='critic_stateInput')

        state_out = layers.Dense(
            16, activation=self.rl['activation'],
            name='critic_stateDense1')(state_input)
        state_out = layers.Dense(
            32, activation=self.rl['activation'],
            name='critic_stateDense2')(state_out)

        # Action as input
        if self.rl['LSTM']:
            action_input = layers.Input(
                shape=(1, self.rl['dim_actions']), name='critic_actionInput')
        else:
            action_input = layers.Input(
                shape=(1, self.rl['dim_actions']), name='critic_actionInput')

        action_out = layers.Dense(
            32, activation=self.rl['activation'],
            name='critic_actionDense1')(action_input)
        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate(name='critic_Concatenate')(
            [state_out, action_out])

        if self.rl['LSTM']:
            out = layers.LSTM(self.rl['dim_out_layer12'],
                              input_shape=(1, self.rl['DDPG']['batch_size']),
                              activation=self.rl['activation'],
                              name='critic_LSTM1')(concat)
        else:
            out = layers.Dense(self.rl['dim_out_layer12'],
                               activation=self.rl['activation'],
                               name='critic_Dense1')(concat)
        out = layers.Dense(self.rl['dim_out_layer12'],
                           activation=self.rl['activation'],
                           name='critic_Dense2')(out)
        outputs = layers.Dense(1, name='critic_Dense3')(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        if self.name is not None:
            model._name = self.name + name_model

        return model

    def sample_action(self, state, eps_greedy=True, rdn_eps_greedy=False,
                      rdn_eps_greedy_indiv=False):
        """`sample_action()` returns an action sampled from
        our Actor network plus some noise for exploration."""
        state = tf.reshape(state, (-1, 1, self.rl['dim_states']))
        x = self.actor_model(state)
        sampled_actions = tf.squeeze(x)
        noise = self.noise_object() if eps_greedy \
            else [0 for _ in range(self.rl['dim_actions'])]
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(
            sampled_actions, self.rl['low_action'], self.rl['high_action'])
        if rdn_eps_greedy:
            rdn = np.random.rand()
            rdn_action = [np.random.rand()
                          for _ in range(self.rl['dim_actions'])]
            legal_action = rdn_action if rdn > self.rl['DDPG']['eps'] \
                else legal_action
        elif rdn_eps_greedy_indiv:
            rdn = [np.random.rand() for _ in range(self.rl['dim_actions'])]
            legal_action = [
                np.random.rand()
                if rdn[i] > self.rl['DDPG']['eps']
                else legal_action[i]
                for i in range(self.rl['dim_actions'])
            ]

        return [np.squeeze(legal_action)]

    def learn(self, prev_state, action, reward, state):
        self.buffer.record((prev_state, action, reward, state))
        self.buffer.learn(self.target_actor, self.target_critic,
                          self.actor_model, self.critic_model,
                          self.actor_optimizer, self.critic_optimizer)
        self.update_target(self.target_actor.variables,
                           self.actor_model.variables)
        self.update_target(self.target_critic.variables,
                           self.critic_model.variables)

    def end(self):
        self.actor_model.save_weights("pendulum_actor.h5")
        self.critic_model.save_weights("pendulum_critic.h5")
        self.target_actor.save_weights("pendulum_target_actor.h5")
        self.target_critic.save_weights("pendulum_target_critic.h5")
