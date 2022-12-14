#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:47:57 2022.

@author: floracharbonnier

"""

import copy
import datetime
import glob
import os
from datetime import timedelta
from typing import Tuple

import numpy as np

from src.simulations.data_manager import DataManager
from src.simulations.learning import LearningManager
from src.simulations.select_actions import ActionSelector
from src.utilities.userdeftools import (initialise_dict,
                                        methods_learning_from_exploration,
                                        reward_type, set_seeds_rdn)


# %% Environment exploration
class Explorer():
    """Explore environment to get data."""

    def __init__(self, env, prm, learner, record, mac):
        """
        Initialise Explorer.

        Inputting environment, learner,
        input steps and parameters to be used throughout.
        """
        # create link to data/methods in the Explorer's methods:
        self.env, self.prm = env, prm
        self.rl = self.prm["RL"]
        if self.rl["type_env"] == "discrete":
            self.rl["n_total_discrete_states"] = env.spaces.n["state"]
        for e in ["n_homes", "discrete", "descriptors", "multipliers",
                  "global_multipliers", "granularity", "brackets"]:
            self.__dict__[e] = env.spaces.__dict__[e]
        self.last_epoch = record.last_epoch
        self.res_path = prm["paths"]["opt_res"]
        for e in ["D", "solver", "N"]:
            self.__dict__[e] = prm["syst"][e]
        self.episode_batch = {}

        self.data = DataManager(env, prm, self)
        self.action_selector = ActionSelector(
            prm, learner, self.episode_batch, env
        )
        self.action_selector.mac = mac

        self.learning_manager = LearningManager(
            env, prm, learner, self.episode_batch
        )

        self.break_down_rewards_entries = \
            prm["syst"]["break_down_rewards_entries"]
        self.step_vals_entries = [
            "state", "ind_global_state", "action", "ind_global_action",
            "reward", "diff_rewards", "indiv_rewards", "next_state",
            "ind_next_global_state", "done", "bool_flex", "constraint_ok"
        ] + self.break_down_rewards_entries
        self.method_vals_entries = ["seeds", "n_not_feas", "not_feas_vars"]

        self.env.update_date(0)

        self.paths = prm["paths"]

    def _initialise_passive_vars(self, env, repeat, epoch, i_explore):
        self.n_homes = self.prm['syst']['n_homesP']
        self.homes = range(self.n_homes)
        # get environment seed
        seed_ind = self.ind_seed_deterministic \
            if self.rl["deterministic"] == 1 \
            else self.data.get_seed_ind(repeat, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.passive_ext]
        env.set_passive_active(passive=True)
        method = "baseline"
        done = 0
        sequence_feasible = True
        record, evaluation = False, False

        return seed_ind, method, done, sequence_feasible, record, evaluation

    def _init_passive_data(self):
        for e in ["netp0", "discharge_tot0", "charge0"]:
            self.prm["loads"][e] = []

    def _passive_get_steps(
            self, env, repeat, epoch, i_explore, methods, step_vals
    ):
        self.data.passive_ext = "P"
        self._init_passive_data()
        if self.prm['syst']['n_homesP'] == 0:
            return step_vals

        # initialise variables for passive case
        seed_ind, _, done, sequence_feasible, record, evaluation \
            = self._initialise_passive_vars(env, repeat, epoch, i_explore)

        # find feasible data
        _, step_vals = self.data.find_feasible_data(
            seed_ind, methods, step_vals, evaluation,
            epoch, passive=True
        )

        # reset environment
        env.reset(seed=self.data.seed[self.data.passive_ext],
                  load_data=True, passive=True)

        # interact with environment in a passive way for each step
        while sequence_feasible and not done:
            action = np.ones((self.n_homes, self.prm['RL']['dim_actions_1']))
            _, done, _, _, _, sequence_feasible, [
                netp, discharge_tot, charge] = env.step(
                action, record=record,
                evaluation=evaluation, netp_storeout=True)
            for info, val in zip(
                    ["netp0", "discharge_tot0", "charge0"],
                    [netp, discharge_tot, charge]
            ):
                self.prm["loads"][info].append(val)
            if not sequence_feasible:
                # if data is not feasible, make new data
                if seed_ind < len(self.data.seeds[self.data.passive_ext]):
                    self.data.d_ind_seed[self.data.passive_ext] += 1
                    seed_ind += 1
                else:
                    for info in ["factors", "cluss", "batch"]:
                        files = glob.glob(
                            self.paths["opt_res"]
                            / f"{info}{self.data.file_id()}"
                        )
                        for filename in files:
                            os.remove(filename)
                    self.data.d_seed[self.data.passive_ext] += 1

                print("infeasible in loop passive")

                self.data.seeds[self.data.passive_ext] = np.delete(
                    self.data.seeds[self.data.passive_ext],
                    len(self.data.seeds[self.data.passive_ext]) - 1)
                self.data.d_ind_seed[self.data.passive_ext] += 1
                seed_ind += 1
                self.data.deterministic_created = False

                _, step_vals = self.data.find_feasible_data(
                    seed_ind, methods, step_vals,
                    evaluation, epoch, passive=True
                )

                self._init_passive_data()

                env.reset(seed=self.data.seed[self.data.passive_ext],
                          load_data=True, passive=True)

                inputs_state_val = \
                    [0, env.date, False,
                     [[env.batch[home]["flex"][ih] for ih in range(0, 2)]
                      for home in self.homes],
                     env.car.store]
                env.get_state_vals(inputs=inputs_state_val)
                sequence_feasible = True
        for info in ["netp0", "discharge_tot0", "charge0"]:
            self.prm["loads"][info] = [
                [self.prm["loads"][info][time][home] for time in range(self.N)]
                for home in self.homes
            ]

        return step_vals

    def _baseline_rewards(self, method, evaluation, action, env):
        sequence_feasible = True
        # substract baseline rewards to reward -
        # for training, not evaluating
        if evaluation or len(method.split("_")) == 1 or reward_type(method) != "d":
            return None, sequence_feasible

        # for each agent, get rewards
        # if they acted in the default way
        # and the others acted the chosen way
        # without implementing in the environment -
        # for training, not evaluating
        rewards_baseline = []
        combs_actions = []
        for home in self.homes:
            # array of actions = the same as chosen
            # except for agent a which has the default action
            actions_baseline_a = action.copy()
            actions_baseline_a[home] = self.rl["default_action"][home]
            combs_actions.append(actions_baseline_a)
        combs_actions.append(self.rl["default_action"])
        for comb_actions in combs_actions:
            # get outp d
            [_, _, reward_a, _, _, constraint_ok, _] = env.step(
                comb_actions,
                implement=False,
                record=False,
                E_req_only=method == "baseline"
            )

            # add penalty if the constraints are violated
            if not constraint_ok:
                sequence_feasible = False
                reward_a = self._apply_reward_penalty(
                    evaluation, reward_a)
            rewards_baseline.append(reward_a)
            if reward_a is None:
                print(f"reward_a {reward_a}")

        return rewards_baseline, sequence_feasible

    def _init_traj_reward(self, method, evaluation):
        if (
            len(method.split("_")) > 1
            and reward_type(method) == "d"
            and not evaluation
        ):
            traj_reward = [0 for _ in self.homes]
        else:
            traj_reward = 0

        return traj_reward

    def _compute_diff_rewards(
            self, method, evaluation, reward, rewards_baseline, break_down_rewards
    ):
        if len(method.split("_")) > 1 \
                and reward_type(method) == "d" \
                and not evaluation:

            if self.rl["competitive"]:
                indiv_rewards = - break_down_rewards[-1]
                diff_rewards = [
                    indiv_rewards[home] - rewards_baseline[home][home]
                    for home in self.homes
                ]
            else:
                if rewards_baseline is None:
                    print("rewards_baseline is None")
                diff_rewards = [
                    reward - baseline for baseline in rewards_baseline
                ]
        else:
            diff_rewards = None

        return diff_rewards

    def _compute_global_ind_state_action(self, current_state, state, action, done, method):
        if self.rl["type_env"] == "discrete" and method[-2] == 'C':
            global_ind = self.env.spaces.get_global_ind(
                current_state, state, action, done, method
            )
        else:
            global_ind = {
                "state": None,
                "action": None,
                "next_state": None
            }

        return global_ind

    def _get_one_episode(
            self, method, epoch, actions, state,
            evaluation, env, batch, step_vals
    ):
        step, done = 0, 0
        sequence_feasible = True
        traj_reward = self._init_traj_reward(method, evaluation)
        # loop through steps until either end of sequence
        # or one step if infeasible
        eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv \
            = self.action_selector.set_eps_greedy_vars(self.rl, epoch, evaluation)

        while not done and sequence_feasible:
            current_state = state

            action, _ = self.action_selector.select_action(
                method, step, actions, evaluation,
                current_state, eps_greedy, rdn_eps_greedy,
                rdn_eps_greedy_indiv, self.t_env
            )

            # interact with environment to get rewards
            # record last epoch for analysis of results
            record = epoch == self.rl["n_epochs"] - 1

            rewards_baseline, sequence_feasible = self._baseline_rewards(
                method, evaluation, action, env
            )
            [state, done, reward, break_down_rewards, bool_flex,
             constraint_ok, record_output] = env.step(
                action, record=record,
                evaluation=evaluation, E_req_only=method == "baseline")

            if record:
                self.last_epoch(
                    evaluation, method, record_output, batch, done)
            if not constraint_ok:
                sequence_feasible = False
                reward = self._apply_reward_penalty(evaluation, reward)
            else:
                if not self.rl['trajectory']:
                    for eval_method in methods_learning_from_exploration(method, epoch, self.rl):
                        traj_reward = self.learning_manager.learning(
                            current_state, state, action, reward, done,
                            eval_method, step, evaluation, traj_reward, step_vals
                        )
                else:
                    traj_reward += reward

                diff_rewards = self._compute_diff_rewards(
                    method, evaluation, reward, rewards_baseline, break_down_rewards
                )
                global_ind = self._compute_global_ind_state_action(
                    current_state, state, action, done, method
                )
                indiv_rewards = - np.array(break_down_rewards[-1])
                step_vals_ = [
                    current_state, global_ind["state"], action, global_ind["action"], reward,
                    diff_rewards, indiv_rewards, state, global_ind["next_state"], done,
                    bool_flex, constraint_ok, *break_down_rewards
                ]
                for info, var in zip(self.step_vals_entries, step_vals_):
                    step_vals[method][info].append(var)

                # if instant feedback,
                # learn right away at the end of the step
                self.learning_manager.q_learning_instant_feedback(
                    evaluation, method, step_vals, step
                )

                step += 1

        return step_vals, traj_reward, sequence_feasible

    def _active_get_steps(
            self, env, repeat, epoch, i_explore, methods,
            step_vals, evaluation
    ):
        rl = self.rl
        self.data.passive_ext = ""
        self.n_homes = self.prm["syst"]["n_homes"]
        self.homes = range(self.n_homes)
        # initialise data
        methods_nonopt = [method for method in methods if method != "opt"]
        method0 = methods_nonopt[0]
        initt0 = 0
        eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv \
            = self.action_selector.set_eps_greedy_vars(rl, epoch, evaluation)
        # make data for optimisation
        # seed_mult = 1 # for initial passive consumers
        seed_ind = self.ind_seed_deterministic if rl["deterministic"] == 1 \
            else self.data.get_seed_ind(repeat, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.passive_ext]

        [_, batch], step_vals = self.data.find_feasible_data(
            seed_ind, methods, step_vals, evaluation, epoch
        )

        n_not_feas, not_feas_vars = 0, []

        # loop through types of actions specified to interact with environment
        # start assuming data is infeasible until proven otherwise
        sequence_feasible = False
        iteration = 0
        while not sequence_feasible and iteration < 1e3:
            # sequence_feasible will be False
            # if it turns out the data for the sequence is not feasible
            iteration += 1
            sequence_feasible = True
            vars_env = {}
            i_t = 0
            # loop through other non optimisation types
            # -> start over steps with new data
            while i_t < len(methods_nonopt) and sequence_feasible:
                method = methods_nonopt[i_t]
                i_t += 1
                self.data.get_seed(seed_ind)
                set_seeds_rdn(self.data.seed[self.data.passive_ext])

                # reset environment with adequate data
                env.reset(
                    seed=self.data.seed[self.data.passive_ext],
                    load_data=True, E_req_only=method == "baseline"
                )
                # get data from environment
                inputs_state_val = [
                    0,
                    env.date,
                    False,
                    [[env.batch[home]["flex"][ih] for ih in range(0, 2)] for home in self.homes],
                    env.car.store
                ]

                # initialise data for current method
                if method == method0:
                    initt0 += 1
                step_vals[method] = initialise_dict(
                    self.step_vals_entries + self.method_vals_entries
                )
                vars_env[method] = initialise_dict(self.prm["save"]["last_entries"])

                actions = None
                if rl["type_learning"] in ["DDPG", "DQN", "facmac", "DDQN"] and rl["trajectory"]:
                    actions, _, states = self.action_selector.trajectory_actions(
                        method, rdn_eps_greedy_indiv, eps_greedy,
                        rdn_eps_greedy, evaluation, self.t_env
                    )
                state = env.get_state_vals(inputs=inputs_state_val)
                step_vals, traj_reward, sequence_feasible = self._get_one_episode(
                    method, epoch, actions, state, evaluation, env, batch, step_vals
                )

                if rl["type_learning"] in ["DDPG", "DQN", "DDQN", "facmac"] \
                        and rl["trajectory"] \
                        and not evaluation \
                        and method != "baseline":
                    for eval_method in methods_learning_from_exploration(method, epoch, self.rl):
                        self.learning_manager.trajectory_deep_learn(
                            states, actions, traj_reward, eval_method, evaluation, step_vals
                        )

            if not sequence_feasible:  # if data is not feasible, make new data
                n_not_feas += 1
                not_feas_vars.append([env.car.store0, method])
                seed_ind = self.data.infeasible_tidy_files_seeds(seed_ind)

                print("infeasible in loop active")

                self.data.deterministic_created = False
                print("find feas opt data again!")
                [_, _, _, batch], step_vals = self.data.find_feasible_data(
                    seed_ind, methods, step_vals,
                    evaluation, epoch
                )

        step_vals["seed"] = self.data.seed[self.data.passive_ext]
        step_vals["not_feas_vars"] = not_feas_vars
        step_vals["n_not_feas"] = n_not_feas
        if not evaluation:
            self.t_env += self.prm['syst']['N']

        return step_vals

    def get_steps(self, methods, repeat, epoch, i_explore,
                  evaluation=False, new_episode_batch=None, parallel=False):
        """Get episode steps interacting with environment.

        For all inputted types of explorations.
        """
        eval0 = evaluation
        self.data.seed_ind = {}
        self.data.seed = {"P": 0, "": 0}
        # create link to objects/data needed in method
        env = copy.deepcopy(self.env) if parallel else self.env

        # initialise output
        step_vals = initialise_dict(methods)
        self._init_facmac_mac(methods, new_episode_batch, epoch)

        # passive consumers
        step_vals = self._passive_get_steps(
            env, repeat, epoch, i_explore, methods, step_vals
        )
        evaluation = eval0

        step_vals = self._active_get_steps(
            env, repeat, epoch, i_explore, methods, step_vals, evaluation
        )

        self._check_rewards_match(methods, evaluation, step_vals)

        if self.rl["type_learning"] != "facmac":
            self.episode_batch = None

        return step_vals, self.episode_batch

    def _check_rewards_match(self, methods, evaluation, step_vals):
        if "opt" in methods and evaluation:
            for method in [method for method in methods if method != "opt"]:
                if step_vals[method]["reward"][-1] is not None:
                    # rewards should not be better than optimal rewards
                    assert np.mean(step_vals[method]["reward"]) \
                           < np.mean(step_vals["opt"]["reward"]) + 1e-3, \
                           f"reward {method} {np.mean(step_vals[method]['reward'])} " \
                           f"better than opt {np.mean(step_vals['opt']['reward'])}"

    def _opt_step_init(
            self, time_step, batchflex_opt, batch_avail_ev, res
    ):
        step_vals_i = {}
        # update time at each time step
        date = self.env.date0 + timedelta(
            hours=time_step * self.prm["syst"]["dt"]
        )

        # update consumption etc. at the beginning of the time step
        loads = {}
        loads["l_flex"], loads["l_fixed"], loads_step = self._fixed_flex_loads(
            time_step, batchflex_opt
        )
        assert all(
            res['totcons'][:, time_step] - res['E_heat'][:, time_step]
            <= loads["l_flex"] + loads["l_fixed"] + 1e-3
        ), f"res loads cons {res['totcons'][:, time_step] - res['E_heat'][:, time_step]}, " \
           f"available loads {loads['l_flex'] + loads['l_fixed']}"
        _, _, loads_prev = self._fixed_flex_loads(
            max(0, time_step - 1), batchflex_opt)
        home_vars = {
            "gen": np.array(
                [self.prm["grd"]["gen"][home][time_step] for home in self.homes]
            )
        }

        step_vals_i["state"] = self.env.spaces.opt_step_to_state(
            self.prm, res, time_step, loads_prev,
            loads_step, batch_avail_ev, loads, home_vars
        )

        self.env.heat.E_heat_min_max(time_step)
        self.env.heat.potential_E_flex()

        return step_vals_i, date, loads, loads_step, loads_prev, home_vars

    def _get_passive_vars(self, time_step):
        passive_vars = \
            [[self.prm["loads"][e][home][time_step]
              for home in range(self.prm['syst']['n_homesP'])]
             for e in ["netp0", "discharge_tot0", "charge0"]]

        return passive_vars

    def _get_diff_rewards(
            self, evaluation, time_step, action, date,
            loads, res, feasible, reward, indiv_rewards
    ):
        obtain_diff_reward = any(
            len(q.split("_")) >= 2
            and reward_type(q) == "d"
            for q in self.prm["RL"]["type_Qs"]
        )
        if obtain_diff_reward and not evaluation:
            rewards_baseline, feasible_getting_baseline = \
                self._get_artificial_baseline_reward_opt(
                    time_step, action, date, loads, res, evaluation
                )
            if not feasible_getting_baseline:
                feasible = False
            if self.prm["RL"]["competitive"]:
                diff_rewards = [
                    indiv_rewards[home] - rewards_baseline[home][home]
                    for home in self.homes
                ]
            else:
                diff_rewards = [
                    reward - reward_baseline
                    for reward_baseline in rewards_baseline
                ]
        else:
            diff_rewards = None

        return diff_rewards, feasible

    def _append_step_vals(
            self, method, step_vals_i, res, time_step,
            loads_prev, loads_step, batch_avail_ev, step_vals,
            break_down_rewards, feasible, loads, home_vars
    ):
        keys = self.break_down_rewards_entries + ["constraint_ok"]
        vars = break_down_rewards + [feasible]
        for key_, var in zip(keys, vars):
            step_vals_i[key_] = var

        keys = [
            "state", "action", "reward", "indiv_rewards", "diff_rewards",
            "bool_flex", "constraint_ok",
            "ind_global_action", "ind_global_state"
        ]
        for key_ in keys:
            step_vals[method][key_].append(step_vals_i[key_])

        if time_step > 0:
            step_vals[method]["next_state"].append(step_vals_i["state"])
            if self.prm["RL"]["type_env"] == "discrete" and method[-2] == 'C':
                step_vals[method]["ind_next_global_state"].append(
                    step_vals_i["ind_global_state"])
            else:
                step_vals[method]["ind_next_global_state"].append(None)
        if time_step == len(res["grid"]) - 1:
            step_vals[method]["next_state"].append(
                self.env.spaces.opt_step_to_state(
                    self.prm, res, time_step + 1, loads_prev,
                    loads_step, batch_avail_ev, loads, home_vars
                )
            )
            if self.prm["RL"]["type_env"] == "discrete" and method[-2] == 'C':
                ind_next_state = self.env.spaces.get_space_indexes(
                    all_vals=step_vals[method]["next_state"][-1])
                step_vals[method]["ind_next_global_state"].append(
                    self.env.spaces.indiv_to_global_index(
                        "state", indexes=ind_next_state,
                        multipliers=self.global_multipliers["state"]))
            else:
                step_vals[method]["ind_next_global_state"].append(None)

        step_vals[method]["done"].append(
            False if time_step <= len(res["grid"]) - 2 else True)

        return step_vals

    def _tests_individual_step_rl_matches_res(
            self, res, time_step, batch, reward
    ):
        prm = self.prm
        assert isinstance(batch[0], dict), f"type(batch[0]) {type(batch)}"
        flex, loads = [np.array(
            [batch[home][e] for home in range(len(batch))])
            for e in ["flex", "loads"]
        ]

        # check tot cons
        for home in self.homes:
            assert res["totcons"][home][time_step] <= \
                   sum(flex[home][time_step]) \
                   + self.env.heat.E_heat_min[home] \
                   + self.env.heat.potential_E_flex()[home] + 1e-3, \
                   f"cons more than sum fixed + flex!, " \
                   f"home = {home}, time_step = {time_step}"

        # check loads and consumption match
        sum_consa = 0
        for load_type in range(2):
            sum_consa += np.sum(res[f'consa({load_type})'])

        assert len(np.shape(loads)) == 2, f"np.shape(loads) == {np.shape(loads)}"
        assert abs((np.sum(loads[:, 0: prm['syst']['N']]) - sum_consa) / sum_consa) < 1e-2, \
            f"res cons {sum_consa} does not match input demand " \
            f"{np.sum(loads[:, 0: prm['syst']['N']])}"

        gc_i = res['pci'][time_step] + res['pco'][time_step]  \
            + prm["grd"]["C"][time_step] * (
                res['grid'][time_step][0] + prm["grd"]['loss'] * res['grid2'][time_step][0]
        )
        gc_per_start_i = [
            prm["grd"]["Call"][i + time_step] * (
                res['grid'][time_step][0]
                + prm["grd"]['loss'] * res['grid2'][time_step][0]
            ) + res['pci'][time_step] + res['pco'][time_step]
            for i in range(len(prm['grd']['Call']) - self.N)
        ]
        potential_i0s = [
            i for i, gc_start_i in enumerate(gc_per_start_i)
            if abs(gc_start_i - gc_i) < 1e-3
        ]
        assert self.env.i0_costs in potential_i0s

        # check reward from environment and res variables match
        gc_t = res['pci'][time_step][0] + res['pco'][time_step][0] \
            + prm["grd"]["C"][time_step] * (
                res["grid"][time_step][0]
                + prm["grd"]["R"] / (prm["grd"]["V"] ** 2) * res["grid2"][time_step][0]
        )
        sc_t = prm["car"]["C"] * (
            sum(
                res["discharge_tot"][home][time_step]
                + res["charge"][home][time_step]
                for home in range(prm['syst']['n_homes'])
            ) + sum(
                self.prm["loads"]['discharge_tot0'][home][time_step]
                + self.prm["loads"]['charge0'][home][time_step]
                for home in range(prm['syst']['n_homesP'])
            )
        )
        dc_t = prm["grd"]["export_C"] * (
            sum(
                res["netp_abs"][home][time_step]
                for home in range(self.prm['syst']['n_homes'])
            )
            + sum(
                self.env.netp_to_exports(self.prm['loads']['netp0'])[home][time_step]
                for home in range(self.prm['syst']['n_homesP'])
            )
        )
        res_reward_t = - (gc_t + sc_t + dc_t)

        if not prm["RL"]["competitive"]:
            assert abs(reward - res_reward_t) < 5e-3, \
                f"reward {reward} != res_reward_t " \
                f"from res variables {res_reward_t}"

    def _instant_feedback_steps_opt(
            self, evaluation, exploration_method, time_step, step_vals, epoch
    ):
        rl = self.prm["RL"]
        if (rl["type_learning"] in ["DQN", "DDQN", "DDPG", "facmac"]
            or rl["instant_feedback"]) \
                and not evaluation \
                and exploration_method in rl["exploration_methods"] \
                and time_step > 0 \
                and not rl["trajectory"]:

            [
                current_state, actions, reward, state,
                reward_diffs, indiv_rewards
            ] = [
                step_vals["opt"][e][-1]
                for e in [
                    "state", "action", "reward", "next_state",
                    "diff_rewards", "indiv_rewards"
                ]
            ]
            if rl["type_learning"] == "q_learning":
                # learner agent learns from this step
                self.learning_manager.learner.learn(
                    "opt", step_vals[exploration_method], time_step - 1
                )
            elif rl["type_learning"] == "facmac":
                pre_transition_data = {
                    "state": np.reshape(
                        current_state, (self.n_homes, rl["obs_shape"])),
                    "avail_actions": [rl["avail_actions"]],
                    "obs": [np.reshape(
                        current_state, (self.n_homes, rl["obs_shape"]))]
                }

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(time_step == self.prm["syst"]["N"] - 1,)],
                }

                evaluation_methods = methods_learning_from_exploration(
                    exploration_method, epoch, rl
                )
                for evaluation_method in evaluation_methods:
                    self.episode_batch[evaluation_method].update(
                        pre_transition_data, ts=time_step)
                    self.episode_batch[evaluation_method].update(
                        post_transition_data, ts=time_step)

            elif rl["type_learning"] in ["DDPG", "DQN", "DDQN"]:
                self.learning_manager.independent_deep_learning(
                    current_state, actions, reward, indiv_rewards,
                    state, reward_diffs
                )

    def _test_total_rewards_match(self, evaluation, res, sum_rl_rewards):
        sum_res_rewards = (- (res["gc"] + res["sc"] + res["dc"]))
        if not (self.prm["RL"]["competitive"] and not evaluation):
            if abs(sum_rl_rewards - sum_res_rewards) > 5e-2:
                print("abs(sum_rl_rewards - sum_res_rewards) > 5e-2")
                np.save('res_error', res)
                np.save('sum_rl_rewards', sum_rl_rewards)

            assert abs(sum_rl_rewards - sum_res_rewards) < 5e-2, \
                "tot rewards don't match: "\
                f"sum_rl_rewards = {sum_rl_rewards}, sum_res_rewards {sum_res_rewards} "
            f"sum costs opt = {- (res['gc'] + res['sc'] + res['dc'])}"

    def sum_gc_for_start_Call_index(self, res, i):
        C = self.prm["grd"]["Call"][i: i + self.N]
        loss = self.prm['grd']['loss']
        sum_gc_i = res['pc'] + np.sum(
            [
                C[time_step_]
                * (res['grid'][time_step_][0] + loss * res['grid2'][time_step_][0])
                for time_step_ in range(self.N)
            ]
        )

        return sum_gc_i

    def _check_i0_costs_res(self, res):
        # check the correct i0_costs is used
        sum_gc_0 = np.sum(
            [self.prm["grd"]["C"][time_step_] * (
                res['grid'][time_step_][0]
                + self.prm["grd"]['loss'] * res['grid2'][time_step_][0]
            ) for time_step_ in range(self.N)]
        )
        if not (abs(sum_gc_0 - res['gc']) < 1e-3):
            i_start_res = [
                i for i in range(len(self.prm['grd']['Call']) - self.N)
                if abs(self.sum_gc_for_start_Call_index(res, i) - res['gc']) < 1e-3
            ]
            if self.env.i0_costs != i_start_res[0]:
                print("update res i0_costs")
                self.env.update_i0_costs(i_start_res[0])
                np.save(self.env.res_path / f"i0_costs{self.env._file_id()}", i_start_res[0])

    def get_steps_opt(
            self, res, step_vals, evaluation, batch, epoch
    ):
        """Translate optimisation results to states, actions, rewards."""
        env, rl = self.env, self.prm["RL"]
        last_epoch = epoch == rl['n_epochs'] - 1
        feasible = True
        method = "opt"
        sum_rl_rewards = 0
        step_vals[method] = initialise_dict(self.step_vals_entries)
        batchflex_opt, batch_avail_ev = [
            [batch[home][e] for home in range(len(batch))] for e in ["flex", "avail_car"]
        ]
        self._check_i0_costs_res(res)

        # copy the initial flexible and non-flexible demand -
        # table will be updated according to optimiser's decisions
        self.env.car.reset(self.prm)
        self.env.car.add_batch(batch)
        self.env.heat.reset(self.prm)
        for time_step in range(len(res["grid"])):
            # initialise step variables
            [step_vals_i, date, loads, loads_step, loads_prev, home_vars] = self._opt_step_init(
                time_step, batchflex_opt, batch_avail_ev, res
            )

            # translate dp into action value
            step_vals_i["bool_flex"], step_vals_i["action"], error = \
                env.action_translator.optimisation_to_rl_env_action(
                    time_step, date, res["netp"][:, time_step],
                    loads, home_vars, res)

            step_vals_i = self.env.spaces.get_ind_global_state_action(step_vals_i)
            feasible = not any(error)

            # determine rewards
            step_vals_i["reward"], break_down_rewards = env.get_reward(
                res["netp"][:, time_step],
                res["discharge_tot"][:, time_step],
                res["charge"][:, time_step],
                time_step=time_step,
                passive_vars=self._get_passive_vars(time_step),
                evaluation=evaluation
            )

            step_vals_i["indiv_rewards"] = - np.array(break_down_rewards[-1])
            self._tests_individual_step_rl_matches_res(
                res, time_step, batch, step_vals_i["reward"]
            )

            # substract baseline rewards to reward -
            # for training, not evaluating
            step_vals_i["diff_rewards"], feasible = self._get_diff_rewards(
                evaluation, time_step, step_vals_i["action"], date, loads, res,
                feasible, step_vals_i["reward"], step_vals_i["indiv_rewards"]
            )
            if not feasible:
                step_vals_i["reward"], step_vals_i["diff_rewards"] = self._apply_reward_penalty(
                    evaluation, step_vals_i["reward"],
                    step_vals_i["diff_rewards"]
                )

            if not (rl["competitive"] and not evaluation):
                sum_rl_rewards += step_vals_i["reward"]

            # append experience dictionaries
            step_vals = self._append_step_vals(
                method, step_vals_i, res, time_step,
                loads_prev, loads_step, batch_avail_ev, step_vals,
                break_down_rewards, feasible, loads, home_vars
            )

            # update flexibility table
            batchflex_opt = self.data.update_flexibility_opt(
                batchflex_opt, res, time_step
            )

            # instant learning feedback
            self._instant_feedback_steps_opt(
                evaluation, method, time_step, step_vals, epoch
            )

            # update battery and heat objects
            self.env.car.update_step(res, time_step=time_step + 1)
            self.env.heat.update_step(res)

            # record if last epoch
            self._record_last_epoch_opt(
                res, time_step, break_down_rewards, batchflex_opt,
                last_epoch, step_vals_i, batch, evaluation
            )

        self._test_total_rewards_match(evaluation, res, sum_rl_rewards)
        if not evaluation \
                and rl["type_learning"] in ["DDPG", "DQN", "facmac"] \
                and rl["trajectory"]:
            self.learning_manager.learn_trajectory_opt(step_vals, epoch)

        return step_vals, feasible

    def _record_last_epoch_opt(
            self, res, time_step, break_down_rewards, batchflex_opt,
            last_epoch, step_vals_i, batch, evaluation
    ):
        if not last_epoch:
            return

        done = time_step == self.prm["syst"]["N"] - 1
        ldflex = [0 for _ in self.homes] \
            if done \
            else [sum(batchflex_opt[home][time_step][1:])
                  for home in self.homes]
        if done:
            ldfixed = [sum(batchflex_opt[home][time_step][:])
                       for home in self.homes]
        else:
            ldfixed = [batchflex_opt[home][time_step][0]
                       for home in self.homes]
        tot_cons_loads = \
            [res["totcons"][home][time_step] - res["E_heat"][home][time_step]
             for home in self.homes]
        wholesalet, cintensityt = \
            [self.prm["grd"][e][self.env.i0_costs + time_step]
             for e in ["wholesale_all", "cintensity_all"]]

        record_output = \
            [res["netp"][:, time_step], res["discharge_other"][:, time_step],
             step_vals_i["action"], step_vals_i["reward"], break_down_rewards,
             res["store"][:, time_step], ldflex, ldfixed,
             res["totcons"][:, time_step], tot_cons_loads,
             res["E_heat"][:, time_step], res["T"][:, time_step],
             res["T_air"][:, time_step], self.prm["grd"]["C"][time_step],
             wholesalet, cintensityt]

        self.last_epoch(evaluation, "opt", record_output, batch, done)

    def _apply_reward_penalty(self, evaluation, reward, diff_rewards=None):
        if self.rl["apply_penalty"] and not evaluation:
            if self.rl["competitive"]:
                assert diff_rewards is not None
            if diff_rewards is not None:
                for home in self.homes:
                    diff_rewards[home] -= self.rl["penalty"]
            else:
                reward -= self.rl["penalty"]

        return reward, diff_rewards

    def _fixed_flex_loads(self, time_step, batchflex_opt):
        """
        Get fixed and flexible consumption equivalent to optimisation results.

        Obtain total fixed and flexible loads for each agent
        for a given time step based on current optimisation results
        """
        # note that we could also obtain the fixed cons / flexible
        # load as below,
        # however we want to count it consistently with our
        # batchflex_opt updates:
        # l_fixed = [ntw['loads'][0, home, time_step] for home in range(n_homes)]
        # flex_load = [ntw['loads'][1, home, time_step] for home in range(n_homes)]

        if time_step == self.prm["syst"]["N"] - 1:
            flex_load = np.zeros(self.n_homes)
            l_fixed = np.array(
                [sum(batchflex_opt[home][time_step][:]) for home in self.homes]
            )
        else:
            flex_load = np.array(
                [sum(batchflex_opt[home][time_step][1:]) for home in self.homes]
            )
            l_fixed = np.array(
                [batchflex_opt[home][time_step][0] for home in self.homes]
            )

        loads_step = l_fixed + flex_load

        return flex_load, l_fixed, loads_step

    def _get_artificial_baseline_reward_opt(
            self,
            time_step: int,
            actions: np.ndarray,
            date: datetime.datetime,
            loads: dict,
            res: dict,
            evaluation: bool
    ) -> Tuple[list, bool]:
        """
        Get instantaneous rewards if agent took baseline actions.

        Get instantaneous rewards if each agent took baseline
        action instead of current action.
        """
        prm, env = self.prm, self.env
        rewards_baseline = []
        gens = prm["grd"]["gen"][:, time_step]
        self.env.heat.T = res["T"][:, time_step]
        self.env.car.store = res["store"][:, time_step]
        combs_actions = np.ones((self.n_homes + 1, self.n_homes, self.prm['RL']['dim_actions_1']))
        for home in self.homes:
            actions_baseline_a = np.array(actions)
            actions_baseline_a[home] = 1
            combs_actions[home] = actions_baseline_a
        combs_actions[-1] = 1
        feasible = True
        for home in self.homes:
            T_air = res["T_air"][home][time_step]
            if T_air < self.env.heat.T_LB[home][time_step] - 1e-1 \
                    or T_air > self.env.heat.T_UB[home][time_step] + 1e-1:
                print(f"home {home} time_step {time_step} "
                      f"res['T_air'][home][time_step] {T_air} "
                      f"T_LB[home] {self.env.heat.T_LB[home][time_step]} "
                      f"T_UB[home] {self.env.heat.T_UB[home][time_step]}")
        for comb_actions in combs_actions:
            bat_store = self.env.car.store.copy()
            input_take_action = date, comb_actions, gens, loads
            home_vars, loads, constraint_ok = env.policy_to_rewardvar(
                None, other_input=input_take_action)
            self.env.car.store = bat_store
            passive_vars = self._get_passive_vars(time_step)

            reward_baseline_a, _ = env.get_reward(
                home_vars["netp"], self.env.car.discharge_tot, self.env.car.charge,
                time_step=time_step, passive_vars=passive_vars,
                evaluation=evaluation)

            if not constraint_ok:
                feasible = False
                print(f"self.data.seed = {self.data.seed} "
                      f"constraint_ok False, time_step {time_step}")
                self._apply_reward_penalty(evaluation, reward_baseline_a)

            rewards_baseline.append(reward_baseline_a)

            # revert back store
            self.env.car.store = [res["store"][home][time_step]
                                  for home in self.homes]

        return rewards_baseline, feasible

    def _init_facmac_mac(self, methods, new_episode_batch, epoch):
        if self.rl["type_learning"] == "facmac":
            for exploration_method in methods:
                evaluation_methods = methods_learning_from_exploration(
                    exploration_method, epoch, self.rl
                )
                for evaluation_method in evaluation_methods:
                    self.episode_batch[evaluation_method] = new_episode_batch()
                    if evaluation_method not in ["baseline", "opt"]:
                        self.action_selector.mac[evaluation_method].init_hidden(
                            batch_size=self.rl["batch_size_run"]
                        )
