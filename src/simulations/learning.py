"""
This file contains the Learner class.

author: Flora Charbonnier
"""

from typing import List

import numpy as np

from src.utilities.env_spaces import granularity_to_multipliers
from src.utilities.userdeftools import data_source, reward_type


class LearningManager():
    """Learn from collected experience."""

    def __init__(self, env, prm, learner, episode_batch):
        """Initialise Learner object."""
        self.env = env
        self.rl = prm["RL"]
        self.n_homes = prm["ntw"]["n"]
        self.homes = range(prm["ntw"]["n"])
        self.learner = learner
        self.N = prm["syst"]["N"]
        self.episode_batch = episode_batch
        self.methods_opt = [
            method for method in self.rl["evaluation_methods"]
            if method[0:3] == "opt" and method != "opt"
        ]

    def learning(self,
                 current_state: list,
                 state: list,
                 action: list,
                 reward: float,
                 done: bool,
                 method: str,
                 step: int,
                 evaluation: bool,
                 traj_reward: list
                 ) -> list:
        """Learn from experience tuple."""
        if self.rl['type_learning'] == 'facmac':
            post_transition_data = {
                "actions": action,
                "reward": [(reward,)],
                "terminated": [(done,)],
            }
            self.episode_batch[method].update(post_transition_data, ts=step)

        if self.rl['type_learning'] in ['DDPG', 'DQN', 'DDQN'] \
                and not evaluation \
                and method != 'baseline' \
                and not done:
            if type(reward) in [float, int, np.float64]:
                traj_reward = self._learning_total_rewards(
                    traj_reward, reward, current_state, action, state, method
                )
            else:
                traj_reward = self._learning_difference_rewards(
                    traj_reward, reward, current_state, action, state, method
                )

        return traj_reward

    def learn_trajectory_opt(self, step_vals, epoch):
        """Learn from optimisation episode using DDPG or DQN."""
        rl = self.prm["RL"]
        for home in self.homes:
            states_a, next_states_a = \
                [np.reshape([step_vals["opt"][e][time_step][home]
                             for time_step in range(self.N)],
                            rl["dim_states"])
                 for e in ["state", "next_state"]]
            for method in self.methods_opt:
                if reward_type(method) == "d":
                    reward_diff_e = "reward_diff" if data_source(method, epoch) == "opt" else "reward"
                    traj_reward_a = sum(
                        [step_vals["opt"][reward_diff_e][time_step][home]
                         for time_step in range(self.N)]
                    )

                elif reward_type(method) == "r":
                    traj_reward_a = \
                        sum([step_vals["opt"]["reward"][time_step]
                             for time_step in range(self.N)])
                actions_a = [step_vals["opt"]["action"][time_step][home]
                             for time_step in range(self.N)]

                if rl["type_learning"] == "DQN":
                    self._learn_DQN(
                        home, method, states_a, next_states_a,
                        actions_a, traj_reward_a)

                elif rl["type_learning"] == "DDPG":
                    if rl["distr_learning"] == "decentralised":
                        self.learner[method][home].learn(
                            states_a, actions_a, traj_reward_a,
                            next_states_a)
                    else:
                        self.learner[method].learn(
                            states_a, actions_a, traj_reward_a,
                            next_states_a)

    def independent_deep_learning(self,
                                  current_state: list,
                                  actions: list,
                                  reward: float,
                                  indiv_rewards: List[float],
                                  state: list,
                                  reward_diffs: list
                                  ):
        """Learn using DDPG, DQN, or DDQN."""
        for method in self.methods_opt:
            # this assumes the states are the same for all
            # and that no trajectory
            if self.rl["distr_learning"] == 'joint':
                self.learner[method].learn(
                    current_state[0], actions, reward, state[0])
            else:
                for home in self.homes:
                    if reward_type(method) == 'r' and self.rl['competitive']:
                        reward = indiv_rewards[home]
                    elif reward_type(method) == 'd':
                        reward = reward_diffs[home]
                    if self.rl['type_learning'] in ['DQN', 'DDQN']:
                        i_current_state, i_action, i_state = [
                            self.env.spaces.get_space_indexes(
                                all_vals=val, type_=type_, indiv_indexes=True)
                            for val, type_ in zip(
                                [current_state, actions, state],
                                ["state", "action", "state"])]
                        if self.rl["distr_learning"] == "decentralised":
                            self.learner[method][home].learn(
                                i_current_state[home], i_action[home],
                                reward, i_state[home])
                        else:
                            self.learner[method].learn(
                                i_current_state[home], i_action[home],
                                reward, i_state[home])
                    else:
                        if self.rl["distr_learning"] == "decentralised":
                            self.learner[method][home].learn(
                                current_state[home], actions[home], reward, state[home])
                        else:
                            self.learner[method].learn(
                                current_state[home], actions[home], reward, state[home])

    def trajectory_deep_learn(self,
                              states: list,
                              actions: list,
                              traj_reward: list,
                              method: str,
                              evaluation: bool
                              ):
        """Learn from trajectory."""
        for home in self.homes:
            states_a = [states[time_step][home]
                        for time_step in range(self.N)]
            next_states_a = [states[time_step][home]
                             for time_step in range(1, self.N + 1)]
            traj_reward_a = traj_reward[home] \
                if len(method.split('_')) > 1 and reward_type(method) == 'd' \
                and not evaluation \
                else traj_reward
            if self.rl['type_learning'] == 'facmac':
                self.learning(states[0: self.N], np.zeros((self.N, self.n_homes, self.rl['dim_actions'])), actions, traj_reward, True, method, 0, evaluation, traj_reward)

            elif self.rl["distr_learning"] == "decentralised":
                self.learner[method][home].learn(
                    states_a, actions[home], traj_reward_a, next_states_a)
            else:
                self.learner[method].learn(
                    states_a, actions[home], traj_reward_a, next_states_a)

    def q_learning_instant_feedback(self, evaluation, method, step_vals, step):
        """At the end of each step, learn from experience using Q-learning."""
        if evaluation is False and self.rl["type_learning"] == "q_learning":
            if self.rl["instant_feedback"] \
                    and not evaluation \
                    and method in self.rl["exploration_methods"]:
                # learner agent learns from this step
                self.learner.learn(method, step_vals[method], step)

    def _learning_difference_rewards(
            self, traj_reward, reward, current_state, action, state, method
    ):
        # this is difference rewards and/or competitive -
        # no joint action as we would not use diff rewards
        # with joint action
        for home in self.homes:
            if self.rl['trajectory']:
                traj_reward[home] += reward[home]
            else:
                if self.rl["distr_learning"] == "decentralised":
                    self.learner[method][home].learn(
                        current_state[home], action[home],
                        reward[home], state[home])
                else:
                    self.learner[method].learn(
                        current_state[home], action[home],
                        reward[home], state[home])

        return traj_reward

    def _learning_total_rewards(
            self, traj_reward, reward, current_state, action, state, method
    ):
        if self.rl['trajectory']:
            traj_reward += reward
            return traj_reward

        if self.rl['type_learning'] in ['DQN', 'DDQN']:
            ind_current_state, ind_action, ind_state = \
                [self.env.spaces.get_space_indexes(
                    all_vals=val, type_=type_)
                    for val, type_ in zip(
                    [current_state, action, state],
                    ["state", "action", "state"])]
            current_state_, action_, state_ \
                = ind_current_state, ind_action, ind_state
        elif self.rl['type_learning'] == 'DDPG':
            current_state_, action_, state_ \
                = current_state, action, state

        for home in self.homes:
            if self.rl["distr_learning"] == "decentralised":
                self.learner[method][home].learn(
                    current_state_[home], action_[home],
                    reward, state_[home])

            if self.rl["distr_learning"] == "centralised":
                self.learner[method].learn(
                    current_state_[home], action_[home],
                    reward, state_[home])

            if self.rl["distr_learning"] == 'joint':
                self.learner[method].learn(
                    current_state_[0], action_,
                    reward, state_[0])

        return traj_reward

    def _learn_DQN(self,
                   home: int,
                   method: str,
                   states_a: int,
                   next_states_a: int,
                   actions_a: int,
                   traj_reward_a: int
                   ):
        """Learn from experience using deep Q-learning (DQN)."""
        env = self.env
        ind_states, ind_next_states, ind_actions = \
            [[env.spaces.get_space_indexes(
                all_vals=current_state) for current_state in states]
                for states in [states_a, next_states_a, actions_a]]
        granularity = [self.rl["n_other_states"] for _ in range(24)]
        multipliers_traj = granularity_to_multipliers(
            granularity)
        traj_ind_state, traj_ind_next_state, traj_ind_actions = \
            [
                [
                    env.indiv_to_global_index(
                        type_, indexes=ind, multipliers=multipliers_traj
                    )
                    for home in self.homes
                ]
                for type_, ind
                in zip(["state", "state", "action"],
                       [ind_states, ind_next_states, ind_actions]
                       )
            ]
        learn_inputs = [traj_ind_state, traj_ind_actions,
                        traj_reward_a, traj_ind_next_state]
        if self.rl["distr_learning"] == "decentralised":
            self.learner[method][home].learn(learn_inputs)
        else:
            self.learner[method].learn(learn_inputs)
