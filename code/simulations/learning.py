"""
This file contains the Learner class.

author: Flora Charbonnier
"""

from typing import List

import numpy as np
from utilities.userdeftools import (data_source, granularity_to_multipliers,
                                    reward_type)


class LearningManager():
    """Learn from collected experience."""

    def __init__(self, env, prm, learner, episode_batch):
        """Initialise Learner object."""
        self.env = env
        self.rl = prm["RL"]
        self.agents = range(prm["ntw"]["n"])
        self.learner = learner
        self.N = prm["syst"]["N"]
        self.episode_batch = episode_batch

    def learning(self,
                 current_state: list,
                 state: list,
                 action: list,
                 reward: float,
                 done: bool,
                 t: str,
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
            self.episode_batch[t].update(post_transition_data, ts=step)

        if self.rl['type_learning'] in ['DDPG', 'DQN', 'DDQN'] \
                and not evaluation \
                and t != 'baseline' \
                and not done:
            if type(reward) in [float, int, np.float64]:
                traj_reward = self._learning_total_rewards(
                    traj_reward, reward, current_state, action, state, t
                )
            else:
                traj_reward = self._learning_difference_rewards(
                    traj_reward, reward, current_state, action, state, t
                )

        return traj_reward

    def learn_trajectory_opt(self, step_vals):
        """Learn from optimisation episode using DDPG or DQN."""
        rl = self.prm["RL"]
        for a in self.agents:
            states_a, next_states_a = \
                [np.reshape([step_vals["opt"][e][i_step][a]
                             for i_step in range(self.N)],
                            rl["dim_states"])
                 for e in ["state", "next_state"]]
            t_opts = [t for t in rl["type_eval"]
                      if t[0:3] == "opt" and t != "opt"]
            for t in t_opts:
                if reward_type(t) == "d":
                    reward_diff_e = "reward_diff" \
                        if data_source(t) == "opt" \
                        else "reward"
                    traj_reward_a = sum(
                        [step_vals["opt"][reward_diff_e][i_step][a]
                         for i_step in range(self.N)])

                elif reward_type(t) == "r":
                    traj_reward_a = \
                        sum([step_vals["opt"]["reward"][i_step]
                             for i_step in range(self.N)])
                actions_a = [step_vals["opt"]["action"][i_step][a]
                             for i_step in range(self.N)]

                if rl["type_learning"] == "DQN":
                    self._learn_DQN(
                        a, t, states_a, next_states_a,
                        actions_a, traj_reward_a)

                elif rl["type_learning"] == "DDPG":
                    if rl["distr_learning"] == "decentralised":
                        self.learner[t][a].learn(
                            states_a, actions_a, traj_reward_a,
                            next_states_a)
                    else:
                        self.learner[t].learn(
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
        t_opts = [t_ for t_ in self.rl['type_eval']
                  if t_[0:3] == "opt" and t_ != "opt"]
        for t_ in t_opts:
            # this assumes the states are the same for all
            # and that no trajectory
            if self.rl["distr_learning"] == 'joint':
                self.learner[t_].learn(
                    current_state[0], actions, reward, state[0])
            else:
                for a in self.agents:
                    if reward_type(t_) == 'r' and self.rl['competitive']:
                        reward = indiv_rewards[a]
                    elif reward_type(t_) == 'd':
                        reward = reward_diffs[a]
                    if self.rl['type_learning'] in ['DQN', 'DDQN']:
                        i_current_state, i_action, i_state = [
                            self.env.spaces.get_space_indexes(
                                all_vals=val, type_=type_, indiv_indexes=True)
                            for val, type_ in zip(
                                [current_state, actions, state],
                                ["state", "action", "state"])]
                        if self.rl["distr_learning"] == "decentralised":
                            self.learner[t_][a].learn(
                                i_current_state[a], i_action[a],
                                reward, i_state[a])
                        else:
                            self.learner[t_].learn(
                                i_current_state[a], i_action[a],
                                reward, i_state[a])
                    else:
                        if self.rl["distr_learning"] == "decentralised":
                            self.learner[t_][a].learn(
                                current_state[a], actions[a], reward, state[a])
                        else:
                            self.learner[t_].learn(
                                current_state[a], actions[a], reward, state[a])

    def trajectory_deep_learn(self,
                              states: list,
                              actions: list,
                              traj_reward: list,
                              t: str,
                              evaluation: bool
                              ):
        """Learn from trajectory."""
        for a in self.agents:
            states_a = [states[i_step][a]
                        for i_step in range(self.N)]
            next_states_a = [states[i_step][a]
                             for i_step in range(1, self.N + 1)]
            traj_reward_a = traj_reward[a] \
                if len(t.split('_')) > 1 and reward_type(t) == 'd' \
                and not evaluation \
                else traj_reward
            if self.rl["distr_learning"] == "decentralised":
                self.learner[t][a].learn(
                    states_a, actions[a], traj_reward_a, next_states_a)
            else:
                self.learner[t].learn(
                    states_a, actions[a], traj_reward_a, next_states_a)

    def q_learning_instant_feedback(self, evaluation, t, step_vals, step):
        """At the end of each step, learn from experience using Q-learning."""
        if evaluation is False \
                and self.rl["type_learning"] == "q_learning":
            if self.rl["instant_feedback"] \
                    and not evaluation \
                    and t in self.rl["type_explo"]:
                # learner agent learns from this step
                self.learner.learn(t, step_vals[t], step)

    def _learning_difference_rewards(
            self, traj_reward, reward, current_state, action, state, t
    ):
        # this is difference rewards and/or competitive -
        # no joint action as we would not use diff rewards
        # with joint action
        for a in self.agents:
            if self.rl['trajectory']:
                traj_reward[a] += reward[a]
            else:
                if self.rl["distr_learning"] == "decentralised":
                    self.learner[t][a].learn(
                        current_state[a], action[a],
                        reward[a], state[a])
                else:
                    self.learner[t].learn(
                        current_state[a], action[a],
                        reward[a], state[a])

        return traj_reward

    def _learning_total_rewards(
            self, traj_reward, reward, current_state, action, state, t
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

        for a in self.agents:
            if self.rl["distr_learning"] == "decentralised":
                self.learner[t][a].learn(
                    current_state_[a], action_[a],
                    reward, state_[a])

            if self.rl["distr_learning"] == "centralised":
                self.learner[t].learn(
                    current_state_[a], action_[a],
                    reward, state_[a])

            if self.rl["distr_learning"] == 'joint':
                self.learner[t].learn(
                    current_state_[0], action_,
                    reward, state_[0])

        return traj_reward

    def _learn_DQN(self,
                   a: int,
                   t: str,
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
                    for a in self.agents
                ]
                for type_, ind
                in zip(["state", "state", "action"],
                       [ind_states, ind_next_states, ind_actions]
                       )
            ]
        learn_inputs = [traj_ind_state, traj_ind_actions,
                        traj_reward_a, traj_ind_next_state]
        if self.rl["distr_learning"] == "decentralised":
            self.learner[t][a].learn(learn_inputs)
        else:
            self.learner[t].learn(learn_inputs)
