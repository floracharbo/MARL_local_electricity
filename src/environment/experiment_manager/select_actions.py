"""
This file contains the ActionSelector class.

Author: Flora Charbonnier
"""

from datetime import timedelta
from typing import Tuple

import numpy as np
import tensorflow as tf
import torch as th

from src.environment.utilities.env_spaces import granularity_to_multipliers


class ActionSelector:
    """Select actions for exploration."""

    def __init__(self, prm, learner, episode_batch, env):
        """Initialise ActionSelector instance."""
        self.prm = prm
        self.learner = learner
        for attribute in ['n_homes', 'n_homes_test', 'n_homesP', 'N']:
            setattr(self, attribute, prm['syst'][attribute])
        self.rl = prm["RL"]
        self.env = env
        self.n_homes = prm["syst"]["n_homes"]
        self.episode_batch = episode_batch

    def _format_tf_prev_state(
            self,
            current_state: list
    ) -> list:
        if self.rl['LSTM']:
            tf_prev_state = [
                tf.expand_dims(
                    tf.convert_to_tensor(
                        np.reshape(current_state[home], (1, 1))
                    ), 0
                )
                for home in range(self.n_homes)
            ]
        elif self.rl['type_learning'] == 'facmac':
            tf_prev_state = th.Tensor(current_state)
        else:
            tf_prev_state = tf.convert_to_tensor(current_state)

        return tf_prev_state

    def select_action_q_learning(self, ext, ind_current_state, method, eps_greedy):
        rl = self.rl
        if ext == '_test':
            ind_action = np.zeros(self.n_homes_test, dtype=int)
            for it in range(rl['action_selection_its']):
                ind_current_state_it = np.zeros((self.n_homes, rl['dim_states']), dtype=int)
                for i in range(rl['dim_states']):
                    ind_current_state_it[:, i] = np.matmul(
                        ind_current_state[:, i], rl['state_exec_to_train'][it]
                    )
                ind_action_it = [
                    self.learner.sample_action(
                        method, ind_current_state_it[home], home, eps_greedy=eps_greedy
                    )[0]
                    for home in range(len(ind_current_state_it))
                ]
                for home_train in range(self.n_homes):
                    home_execs = np.where(rl['action_train_to_exec'][it][home_train])[0]
                    assert len(home_execs) <= 1
                    if len(home_execs) > 0:
                        ind_action[home_execs[0]] = ind_action_it[home_train]

        else:
            ind_action = [
                self.learner.sample_action(
                    method, ind_current_state[home], home, eps_greedy=eps_greedy
                )[0]
                for home in range(self.n_homes)
            ]

        return ind_action

    def select_action(
            self,
            method: str,
            step: int,
            actions: list,
            evaluation: bool,
            current_state: list,
            eps_greedy: bool,
            rdn_eps_greedy: bool,
            rdn_eps_greedy_indiv: bool,
            t_env: int,
            ext: str = ""
    ) -> Tuple[list, list]:
        """Select exploration action."""
        if self.n_homes == 0:
            return [], []

        spaces = self.env.get_current_spaces(evaluation)
        rl = self.rl
        if rl['type_learning'] in ['facmac', 'DDPG']:
            tf_prev_state = self._format_tf_prev_state(current_state)
        else:
            tf_prev_state = None
        # action choice for current time step
        action_dict = {
            'baseline': self.rl['default_action' + ext],
            'random': np.random.random(np.shape(self.rl['default_action' + ext])),
        }
        if rl['type_learning'] in ['DDPG', 'DQN', 'facmac'] and rl['trajectory']:
            action = actions[:, step]

        elif method in action_dict:
            action = action_dict[method]
        elif self.n_homes > 0:
            if rl['type_learning'] == 'DDPG' and not rl['trajectory']:
                action = self._select_action_DDPG(
                    tf_prev_state, eps_greedy, rdn_eps_greedy,
                    rdn_eps_greedy_indiv, method
                )
            elif rl['type_learning'] == 'facmac':
                if ext == '_test':
                    action = np.zeros((self.n_homes_test, rl['dim_actions']))
                    for it in range(rl['action_selection_its']):
                        current_state_it = np.zeros((self.n_homes, rl['dim_states']))
                        for i in range(rl['dim_states']):
                            current_state_it[:, i] = np.matmul(
                                current_state[:, i], rl['state_exec_to_train'][it]
                            )
                        tf_prev_state_it = self._format_tf_prev_state(current_state_it)

                        action_it = self._select_action_facmac(
                            tf_prev_state_it, step, evaluation, method, t_env
                        )[0]
                        for home_train in range(self.n_homes):
                            home_execs = np.where(rl['action_train_to_exec'][it][home_train])[0]
                            assert len(home_execs) <= 1
                            if len(home_execs) > 0:
                                action[home_execs[0]] = action_it[home_train]

                else:
                    action = self._select_action_facmac(
                        tf_prev_state, step, evaluation, method, t_env
                    )
            else:
                ind_current_state = np.array(
                    spaces.get_space_indexes(
                        all_vals=current_state, indiv_indexes=True
                    )
                )
                if rl['type_learning'] == 'q_learning':
                    ind_action = self.select_action_q_learning(
                        ext, ind_current_state, method, eps_greedy
                    )
                elif rl['type_learning'] == 'DDQN':
                    ind_action = self._select_action_DDQN(
                        ind_current_state, eps_greedy, method
                    )
                elif rl['type_learning'] == 'DQN':
                    ind_action = self._select_action_DQN(
                        ind_current_state, eps_greedy, rdn_eps_greedy_indiv, method
                    )

                action_indexes = [
                    self.env.spaces.global_to_indiv_index("action", ind_action[home])
                    for home in range(self.__dict__['n_homes' + ext])
                ]
                action = [
                    self.env.spaces.index_to_val(action_indexes[home], typev="action")
                    for home in range(self.__dict__['n_homes' + ext])
                ]
        else:
            action = None

        return action, tf_prev_state

    def trajectory_actions(
        self, method, rdn_eps_greedy_indiv=None, eps_greedy=None,
        rdn_eps_greedy=None, evaluation=None, t_env=None, ext=None
    ):
        """Select actions for all episode time steps."""
        env, rl = self.env, self.rl
        n_homes = self.n_homes if ext is None else getattr(self, 'n_homes' + ext)
        if n_homes == 0:
            return [], [], []
        states = np.zeros(
            (self.N + 1, n_homes, len(self.rl['state_space']))
        )
        for time_step in range(self.N + 1):
            inputs_state_val = [
                time_step,
                env.date0 + timedelta(hours=time_step * self.prm['syst']['dt']),
                False,
                env.batch['flex'][:, 0: 2],
                env.car.store
            ]
            states[time_step] = env.get_state_vals(inputs=inputs_state_val, evaluation=evaluation)
        ind_actions = None
        n_actions = 1 if self.rl['aggregate_actions'] else rl['dim_actions_1']

        if method == 'baseline':
            actions = self.rl['default_action' + ext]
            if self.rl['type_env'] == "discrete":
                ind_actions = np.ones(n_homes) * (env.spaces.n["actions"] - 1)

        elif rl['type_learning'] == 'DDPG':
            actions, ind_actions = self._trajectory_actions_ddpg(
                states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv, method
            )
        # with DQN we convert the list of states to a global state descriptor
        elif rl['type_learning'] == 'DQN':
            actions, ind_actions = self._trajectory_actions_dqn(
                states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv, method
            )

        elif rl['type_learning'] == 'DDQN':
            actions, ind_actions = self._trajectory_actions_ddqn(eps_greedy, method)

        elif rl['type_learning'] == 'facmac':
            step = 0

            if ext == '_test':
                actions = np.zeros((self.n_homes_test, self.N, n_actions))
                for it in range(rl['action_selection_its']):
                    state_it = np.zeros((self.N + 1, self.n_homes, rl['dim_states_1']))
                    for i in range(rl['dim_states_1']):
                        state_it[:, :, i] = np.matmul(
                            states[:, :, i], rl['state_exec_to_train'][it]
                        )
                    tf_prev_state_it = self._format_tf_prev_state(state_it)

                    action_it = self._select_action_facmac(
                        tf_prev_state_it, step, evaluation, method, t_env
                    )
                    action_it = action_it.reshape((self.n_homes, self.N, n_actions))
                    for home_train in range(self.n_homes):
                        home_execs = np.where(rl['action_train_to_exec'][it][home_train])[0]
                        assert len(home_execs) <= 1
                        if len(home_execs) > 0:
                            actions[home_execs[0]] = action_it[home_train]
            else:
                tf_prev_state = self._format_tf_prev_state(states)
                actions = self._select_action_facmac(
                    tf_prev_state, step, evaluation, method, t_env
                )

        if isinstance(actions, np.ndarray):
            actions = np.reshape(actions, (n_homes, self.N, n_actions))
        elif th.is_tensor(actions):
            actions = actions.reshape((n_homes, self.N, n_actions))

        return actions, ind_actions, states

    def _select_action_DDPG(
            self, tf_prev_state, eps_greedy, rdn_eps_greedy,
            rdn_eps_greedy_indiv, method
    ):
        if self.rl["distr_learning"] == "decentralised":
            action = [
                self.learner[method][home].sample_action(
                    tf_prev_state[home], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0]
                for home in range(self.n_homes)
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_state = tf_prev_state
            action = [
                self.learner[method].sample_action(
                    tf_prev_state[home], eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)[
                    0] for home in range(self.n_homes)]
        elif self.rl["distr_learning"] == 'joint':
            action = self.learner[method].sample_action(
                tf_prev_state[0], eps_greedy=eps_greedy,
                rdn_eps_greedy=rdn_eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)[0]
            if np.shape(action) == ():
                action = [float(action)]
            else:
                action = list(action)

        return action

    def _select_action_facmac(
        self, tf_prev_state, step, evaluation, method, t_env
    ):
        pre_transition_data = {"avail_actions": th.Tensor(self.rl['avail_actions']), }
        if self.rl['trajectory']:
            pre_transition_data["state"] = tf_prev_state[0: self.N]
            pre_transition_data["obs"] = tf_prev_state[0: self.N]
        else:
            pre_transition_data["state"] = tf_prev_state
            pre_transition_data["obs"] = tf_prev_state

        self.episode_batch[method].update(pre_transition_data, ts=step)
        if self.rl['action_selector'] == "gumbel":
            actions = self.mac[method].select_actions(
                self.episode_batch[method], t_ep=step, t_env=t_env,
                test_mode=evaluation
            )
            action = th.argmax(actions, dim=-1).long()
        else:
            action = self.mac[method].select_actions(
                self.episode_batch[method], t_ep=step,
                t_env=t_env, test_mode=evaluation
            )

        return action

    def _select_action_DDQN(self, ind_current_state, eps_greedy, method):
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[method][home].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy)
                for home in range(self.n_homes)]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[method].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy)
                for home in range(self.n_homes)]

        return ind_action

    def _select_action_DQN(
            self, ind_current_state, eps_greedy, rdn_eps_greedy_indiv, method
    ):
        if self.rl["distr_learning"] == "decentralised":
            ind_action = [self.learner[method][home].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in range(self.n_homes)]
        elif self.rl["distr_learning"] == "centralised":
            ind_action = [self.learner[method].sample_action(
                ind_current_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in range(self.n_homes)]

        return ind_action

    def _trajectory_actions_ddpg(
            self, states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv, method
    ):
        if self.rl['LSTM']:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, home],
                           (1, self.rl['dim_states']))), 0)
                for home in range(self.n_homes)]
        else:
            tf_prev_states = [tf.expand_dims(tf.convert_to_tensor(
                np.reshape(states[0: self.N, home], self.rl['dim_states'])
            ), 0) for home in range(self.n_homes)]

        if self.rl["distr_learning"] == "decentralised":
            actions = [
                self.learner[method][home].sample_action(
                    tf_prev_states[home],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for home in range(self.n_homes)
            ]
        elif self.rl["distr_learning"] == "centralised":
            self.tf_prev_states = tf_prev_states
            actions = [
                self.learner[method].sample_action(
                    tf_prev_states[home],
                    eps_greedy=eps_greedy,
                    rdn_eps_greedy=rdn_eps_greedy,
                    rdn_eps_greedy_indiv=rdn_eps_greedy_indiv
                )[0] for home in range(self.n_homes)
            ]

        ind_actions = None

        return actions, ind_actions

    def _trajectory_actions_dqn(
            self, states, eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv, method
    ):
        if self.prm['syst']['test_different_to_train']:
            print("implement spaces_test for DQN")
        ind_states = [self.env.spaces.get_space_indexes(
            all_vals=current_state, indiv_indexes=True)
            for current_state in states]
        ind_states_a_step = [[ind_states[time_step][home] for time_step in range(
            self.N)] for home in range(self.n_homes)]
        granularity = [
            self.prm["RL"]["n_other_states"]
            for _ in range(self.prm['syst']['N'])
        ]
        multipliers_traj = granularity_to_multipliers(
            granularity)
        traj_ind_state = [self.env.indiv_to_global_index(
            "state", indexes=ind_states_a_step[home],
            multipliers=multipliers_traj)
            for home in range(self.n_homes)]

        if self.rl["distr_learning"] == "decentralised":
            ind_actions = [self.learner[method][home].sample_action(
                traj_ind_state[home], eps_greedy=eps_greedy,
                rdn_eps_greedy=rdn_eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in range(self.n_homes)]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = [self.learner[method].sample_action(
                ind_states_a_step[home], eps_greedy=eps_greedy,
                rdn_eps_greedy_indiv=rdn_eps_greedy_indiv)
                for home in range(self.n_homes)]
        actions = [
            [self.env.spaces.index_to_val(
                [ind_actions[a_][time_step]], typev="action")[0]
             for time_step in range(self.N)]
            for a_ in range(self.n_homes)
        ]

        return actions, ind_actions

    def _trajectory_actions_ddqn(self, eps_greedy, method):
        if self.rl["distr_learning"] == "decentralised":
            ind_actions = [self.learner[method][home].sample_action(
                eps_greedy=eps_greedy)
                for home in range(self.n_homes)]
        elif self.rl["distr_learning"] == "centralised":
            ind_actions = self.learner[method].sample_action(
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
