"""
This file contains the ActionSelector class.

Author: Flora Charbonnier
"""

from datetime import timedelta
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch as th
from utilities.env_spaces import granularity_to_multipliers


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

    def _format_tf_prev_state(
            self,
            current_state: list
    ) -> list:
        if self.rl['LSTM']:
            tf_prev_state = [tf.expand_dims(
                tf.convert_to_tensor(np.reshape(
                    current_state[home], (1, 1))), 0)
                for home in self.agents]
        else:
            tf_prev_state = [tf.expand_dims(tf.convert_to_tensor(
                current_state[home]), 0) for home in self.agents]

        return tf_prev_state

    def select_action(
            self,
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
    ) -> Tuple[list, list]:
        """Select exploration action."""
        rl = self.rl
        tf_prev_state = self._format_tf_prev_state(current_state)

        # action choice for current time step
        if t == 'baseline':
            action = self.rl['default_action']
        elif t == 'random':
            action = np.random.random(np.shape(self.rl['default_action']))
        elif t == 'tryopt':
            action = mus_opt[step]
        elif rl['type_learning'] in ['DDPG', 'DQN'] and rl['trajectory']:
            action = [actions[home][step]
                      for home in self.agents]

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
                        t, ind_current_state[home], home, eps_greedy=eps_greedy
                    )[0]
                    for home in self.agents
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
                 [[env.batch[home]['flex'][ih] for ih in range(0, 2)]
                  for home in self.agents]]
            states[i_step] = env.get_state_vals(inputs=inputs_state_val)

        if t == 'baseline':
            actions = [[self.rl['default_action'][home] for _ in range(
                self.N)] for home in self.agents]
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
                self.learner[t][home].sample_action(
                    tf_prev_state[home], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0]
                for home in self.agents
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_state = tf_prev_state
            action = [
                self.learner[t].sample_action(
                    tf_prev_state[home], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)[
                    0] for home in self.agents]
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
            "state": [current_state[home] for home in self.agents],
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

        action = [[float(action[0][home][i])
                   for i in range(self.rl['dim_actions'])]
                  for home in self.agents]

        return action

    def _select_action_DDQN(self, ind_current_state, eps_greedy):
        t = "DDQN"
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[t][home].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy)
                for home in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[t].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy)
                for home in self.agents]

        return ind_action

    def _select_action_DQN(
            self, ind_current_state, eps_greedy, rdn_eps_greedy_indiv
    ):
        t = "DQN"
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[t][home].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[t].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in self.agents]

        return ind_action

    def _trajectory_actions_DDPG(
            self, states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
    ):
        t = "DDPG"
        if self.rl['LSTM']:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, home],
                           (1, self.rl['dim_states']))), 0)
                for home in self.agents]
        else:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, home], self.rl['dim_states'])
            ), 0) for home in self.agents]

        if self.rl["distr_learning"] == "decentralised":
            actions = [
                self.learner[t][home].sample_action(
                    tf_prev_states[home],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for home in self.agents
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_states = tf_prev_states
            actions = [
                self.learner[t].sample_action(
                    tf_prev_states[home],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for home in self.agents
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
        ind_states_a_step = [[ind_states[i_step][home] for i_step in range(
            self.N)] for home in self.agents]
        granularity = [self.rl["n_other_states"] for _ in range(24)]
        multipliers_traj = granularity_to_multipliers(
            granularity)
        traj_ind_state = [self.env.indiv_to_global_index(
            "state", indexes=ind_states_a_step[home],
            multipliers=multipliers_traj)
            for home in self.agents]

        if self.rl["distr_learning"] == "decentralised":
            ind_actions = [self.learner[t][home].sample_action(
                traj_ind_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy=rdn_eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = [self.learner[t].sample_action(
                ind_states_a_step[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in self.agents]
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
            ind_actions = [self.learner[t][home].sample_action(
                eps_greedy=eps_greedy)
                for home in self.agents]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = self.learner[t].sample_action(
                eps_greedy=eps_greedy)
        actions = None

        return actions, ind_actions


    def set_eps_greedy_vars(self, rl, epoch, evaluation):
        # if eps_greedy is true we are adding random action selection
        eps_greedy = False if (
            evaluation and rl["eval_deterministic"] and epoch > 0) else True
        if eps_greedy and rl["type_learning"] in ["DDPG", "DQN", "DDQN"] \
                and rl[rl["type_learning"]]["rdn_eps_greedy"]:
            # DDPG with random action when exploring,
            # not just the best with added noise
            rdn_eps_greedy = True
            eps_greedy = False
            rdn_eps_greedy_indiv = False
        elif eps_greedy and rl["type_learning"] in ["DDPG", "DQN", "DDQN"] \
                and self.rl[rl["type_learning"]]["rdn_eps_greedy_indiv"]:
            rdn_eps_greedy = False
            rdn_eps_greedy_indiv = True
            eps_greedy = False
        else:
            rdn_eps_greedy = False
            rdn_eps_greedy_indiv = False

        return eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv
