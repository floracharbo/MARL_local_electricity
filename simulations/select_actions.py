"""
This file contains the ActionSelector class.

Author: Flora Charbonnier
"""

from datetime import timedelta
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch as th

from utils.userdeftools import granularity_to_multipliers


class ActionSelector:
    """Select actions for exploration."""

    def __init__(self, prm, learner, episode_batch, env):
        """Initialise ActionSelector instance."""
        self.prm = prm
        self.learner = learner
        self.n_agents = prm["ntw"]["n"]
        self.rl = prm["RL"]
        self.env = env
        self.agents = range(prm["ntw"]["n"])
        self.episode_batch = episode_batch

    def select_action(self,
                      t: str,
                      step: int,
                      actions: list,
                      mus_opt: list,
                      evaluation: bool,
                      current_state: list,
                      eps_greedy: bool,
                      rdn_eps_greedy: bool,
                      rdn_eps_greedy_indiv: bool,
                      t_env
                      ) -> Tuple[list, list, list]:
        """Select exploration action."""
        rl = self.rl
        # if rl['type_learning'] in ['DDPG', 'facmac']:
        if rl['LSTM']:
            tf_prev_state = [tf.expand_dims(
                tf.convert_to_tensor(np.reshape(
                    current_state[a], (1, 1))), 0)
                for a in self.agents]
        else:
            tf_prev_state = [tf.expand_dims(tf.convert_to_tensor(
                current_state[a]), 0) for a in self.agents]

        # action choice for current time step
        if t == 'baseline':
            action = self.rl['default_action']
        elif t == 'random':
            action = np.random.random(np.shape(self.rl['default_action']))

        elif t == 'tryopt':
            action = mus_opt[step]

        elif rl['type_learning'] in ['DDPG', 'DQN'] and rl['trajectory']:
            action = [actions[a][step]
                      for a in self.agents]

        elif rl['type_learning'] == 'DDPG' and not rl['trajectory']:
            action = self._select_action_DDPG(
                tf_prev_state, eps_greedy, rdn_eps_greedy,
                rdn_eps_greedy_indiv, t
            )

        elif rl['type_learning'] == 'facmac':
            action = self._select_action_facmac(
                current_state, tf_prev_state, step, evaluation, t, t_env
            )

        else:
            ind_current_state = self.env.spaces.get_space_indexes(
                all_vals=current_state, indiv_indexes=True)

            if rl['type_learning'] == 'q_learning':
                ind_action = [
                    self.learner.sample_action(
                        t, ind_current_state[a], a, eps_greedy=eps_greedy
                    )[0]
                    for a in self.agents
                ]

            elif rl['type_learning'] == 'DDQN':
                ind_action = self._select_action_DDQN(
                    ind_current_state, eps_greedy
                )

            elif rl['type_learning'] == 'DQN':
                ind_action = self._select_action_DQN(
                    ind_current_state, eps_greedy, rdn_eps_greedy_indiv
                )

            action_indexes = [self.env.spaces.global_to_indiv_index(
                "action", ind_action[a_]) for a_ in self.agents]
            action = [self.env.spaces.index_to_val(
                action_indexes[a_], typev="action")
                for a_ in self.agents]

        return action, tf_prev_state

    def trajectory_actions(self, t, rdn_eps_greedy_indiv,
                           eps_greedy, rdn_eps_greedy):
        """Select actions for all episode time steps."""
        env, rl = self.env, self.rl
        states = np.zeros(
            (self.N + 1, self.n_agents, len(self.rl['state_space'])))

        for i_step in range(self.N + 1):
            inputs_state_val = \
                [i_step, env.date + timedelta(hours=i_step), False,
                 [[env.batch[a]['flex'][ih] for ih in range(0, 2)]
                  for a in self.agents]]
            states[i_step] = env.get_state_vals(inputs=inputs_state_val)

        if t == 'baseline':
            actions = [[self.rl['default_action'][a] for _ in range(
                self.N)] for a in self.agents]
            ind_actions = \
                np.ones(self.n_agents) * (env.spaces.n["actions"] - 1)

        # with DDPG we input an array of states for each agent and time
        elif rl['type_learning'] == 'DDPG':
            actions, ind_actions = self._trajectory_actions_DDPG(
                states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
            )

        # with DQN we convert the list of states to a global state descriptor
        elif rl['type_learning'] == 'DQN':
            actions, ind_actions = self._trajectory_actions_DQN(
                states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
            )

        elif rl['type_learning'] == 'DDQN':
            actions, ind_actions = self._trajectory_actions_DDQN(eps_greedy)

        return actions, ind_actions, states

    def _select_action_DDPG(
            self, tf_prev_state, eps_greedy, rdn_eps_greedy,
            rdn_eps_greedy_indiv, t
    ):
        if self.rl["distr_learning"] == "decentralised":
            action = [
                self.learner[t][a].sample_action(
                    tf_prev_state[a], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0]
                for a in self.agents
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_state = tf_prev_state
            action = [
                self.learner[t].sample_action(
                    tf_prev_state[a], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)[
                    0] for a in self.agents]
        elif self.rl["distr_learning"] == 'joint':
            action = self.learner[t].sample_action(
                tf_prev_state[0], eps_greedy=eps_greedy,
                rdn_eps_greedy=rdn_eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)[0]
            if np.shape(action) == ():
                action = [float(action)]
            else:
                action = list(action)

        return action

    def _select_action_facmac(
            self, current_state, tf_prev_state, step, evaluation, t, t_env
    ):
        pre_transition_data = {
            "state": [current_state[a] for a in self.agents],
            "avail_actions": [self.rl['avail_actions']],
            "obs": [np.reshape(tf_prev_state,
                               (self.n_agents, self.rl['obs_shape']))]
        }
        self.episode_batch[t].update(pre_transition_data, ts=step)
        if self.rl['action_selector'] == "gumbel":
            actions = self.mac[t].select_actions(
                self.episode_batch[t], t_ep=step, t_env=t_env,
                test_mode=evaluation, explore=(not evaluation))
            action = th.argmax(actions, dim=-1).long()
        else:
            action = self.mac[t].select_actions(
                self.episode_batch[t], t_ep=step,
                t_env=t_env, test_mode=evaluation)

        action = [[float(action[0][a][i])
                   for i in range(self.rl['dim_actions'])]
                  for a in self.agents]

        return action

    def _select_action_DDQN(self, ind_current_state, eps_greedy):
        t = "DDQN"
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[t][a].sample_action(
                ind_current_state[a], eps_greedy=eps_greedy)
                for a in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[t].sample_action(
                ind_current_state[a], eps_greedy=eps_greedy)
                for a in self.agents]

        return ind_action

    def _select_action_DQN(
            self, ind_current_state, eps_greedy, rdn_eps_greedy_indiv
    ):
        t = "DQN"
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[t][a].sample_action(
                ind_current_state[a], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for a in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[t].sample_action(
                ind_current_state[a], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for a in self.agents]

        return ind_action

    def _trajectory_actions_DDPG(
            self, states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
    ):
        t = "DDPG"
        if self.rl['LSTM']:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, a],
                           (1, self.rl['dim_states']))), 0)
                for a in self.agents]
        else:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, a], self.rl['dim_states'])
            ), 0) for a in self.agents]

        if self.rl["distr_learning"] == "decentralised":
            actions = [
                self.learner[t][a].sample_action(
                    tf_prev_states[a],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for a in self.agents
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_states = tf_prev_states
            actions = [
                self.learner[t].sample_action(
                    tf_prev_states[a],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for a in self.agents
            ]

        ind_actions = None

        return actions, ind_actions

    def _trajectory_actions_DQN(
            self, states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
    ):
        t = "DQN"
        ind_states = [self.env.spaces.get_space_indexes(
            all_vals=current_state, indiv_indexes=True)
            for current_state in states]
        ind_states_a_step = [[ind_states[i_step][a] for i_step in range(
            self.N)] for a in self.agents]
        granularity = [self.rl["n_other_states"] for _ in range(24)]
        multipliers_traj = granularity_to_multipliers(
            granularity)
        traj_ind_state = [self.env.indiv_to_global_index(
            "state", indexes=ind_states_a_step[a],
            multipliers=multipliers_traj)
            for a in self.agents]

        if self.rl["distr_learning"] == "decentralised":
            ind_actions = [self.learner[t][a].sample_action(
                traj_ind_state[a], eps_greedy=eps_greedy,
                rdn_eps_greedy=rdn_eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for a in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = [self.learner[t].sample_action(
                ind_states_a_step[a], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for a in self.agents]
        actions = [
            [self.env.spaces.index_to_val(
                [ind_actions[a_][i_step]], typev="action")[0]
             for i_step in range(self.N)]
            for a_ in self.agents
        ]

        return actions, ind_actions

    def _trajectory_actions_DDQN(self, eps_greedy):
        t = "DDQN"
        if self.rl["distr_learning"] == "decentralised":
            ind_actions = [self.learner[t][a].sample_action(
                eps_greedy=eps_greedy)
                for a in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = self.learner[t].sample_action(
                eps_greedy=eps_greedy)
        actions = None

        return actions, ind_actions
