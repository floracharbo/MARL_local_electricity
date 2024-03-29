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
import time
from datetime import timedelta
from typing import Tuple

import numpy as np
import torch as th

import src.environment.utilities.userdeftools as utils
from src.environment.experiment_manager.data_manager import DataManager
from src.environment.experiment_manager.select_actions import ActionSelector
from src.learners.learning import LearningManager
from src.tests.explorer_tests import ExplorerTests


# %% Environment exploration
class Explorer:
    """Explore environment to get data."""

    def __init__(self, env, prm, learner, record, mac):
        """
        Initialise Explorer.

        Inputting environment, learner,
        input steps and parameters to be used throughout.
        """
        # create link to data/methods in the Explorer's methods:
        self.env, self.prm = env, prm
        self.rl, self.grd = [self.prm[key] for key in ['RL', 'grd']]
        if self.rl["type_env"] == "discrete":
            self.rl["n_total_discrete_states"] = env.spaces.n["state"]
        for info in [
            "n_homes", "discrete", "descriptors", "multipliers",
            "global_multipliers", "granularity", "brackets"
        ]:
            setattr(self, info, getattr(env.spaces, info))

        self.last_epoch = record.last_epoch
        self.res_path = prm["paths"]["opt_res"]
        for info in ["D", "solver", "N"]:
            setattr(self, info, prm["syst"][info])
        self.data = DataManager(env, prm, self)
        self.episode_batch = {}
        self.action_selector = ActionSelector(
            prm, learner, self.episode_batch, env
        )

        self.action_selector.mac = mac

        self.learning_manager = LearningManager(
            env, prm, learner, self.episode_batch
        )
        self._initialise_step_vals_entries(prm)
        self.method_vals_entries = ["seeds", "n_not_feas"]

        self.env.update_date(0)

        self.paths = prm["paths"]
        self.duration_learning = 0
        self.tests = ExplorerTests(self)

    def _initialise_step_vals_entries(self, prm):
        self.dim_step_vals = {
            "state": prm['RL']['dim_states_1'],
            "next_state": prm['RL']['dim_states_1'],
            "action": prm['RL']['dim_actions_1'],
            "diff_rewards": 1,
            "bool_flex": 1,
        }

        self.global_step_vals_entries = [
            "ind_global_state", "ind_global_action", "reward",
            "ind_next_global_state", "done", "constraint_ok"
        ]
        self.step_vals_entries = \
            prm['syst']['indiv_step_vals_entries'] \
            + self.global_step_vals_entries \
            + prm['syst']['break_down_rewards_entries']

    def _initialise_passive_vars(self, env, repeat, epoch, i_explore):
        self.n_homes = self.prm['syst']['n_homesP']
        self.homes = range(self.n_homes)
        # get environment seed
        seed_ind = self.ind_seed_deterministic \
            if self.rl["deterministic"] == 1 \
            else self.data.get_seed_ind(repeat, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.ext]
        env.set_passive_active(passive=True)
        method = "baseline"
        done = 0
        sequence_feasible = True
        record, evaluation = False, False

        return seed_ind, method, done, sequence_feasible, record, evaluation

    def _init_passive_data(self):
        for e in ["netp0", "discharge_tot0", "charge0"]:
            self.prm["loads"][e] = np.zeros((self.prm['syst']['n_homesP'], self.N))

    def _passive_get_steps(
            self, env, repeat, epoch, i_explore, methods, step_vals
    ):
        self.data.ext = "P"
        self.env.spaces.ext = "P"

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
        env.reset(
            seed=self.data.seed[self.data.ext], load_data=True, passive=True
        )

        # interact with environment in a passive way for each step
        while sequence_feasible and not done:
            if self.rl['type_learning'] in ['DDPG', 'DQN', 'facmac'] and self.rl['trajectory']:
                actions, _, _ = self.action_selector.trajectory_actions('baseline', ext='P')
                action = actions[:, env.time_step]
            else:
                action = self.rl['default_action' + self.env.ext]

            _, done, _, _, _, sequence_feasible, [
                netp, discharge_tot, charge] = env.step(
                    action, record=record, netp_storeout=True, evaluation=evaluation,
                )
            if not done:
                for info, val in zip(
                        ["netp0", "discharge_tot0", "charge0"],
                        [netp, discharge_tot, charge]
                ):
                    self.prm["loads"][info][:, env.time_step] = val
            if not sequence_feasible:
                # if data is not feasible, make new data
                if seed_ind < len(self.data.seeds[self.data.ext]):
                    self.data.d_ind_seed[self.data.ext] += 1
                    seed_ind += 1
                else:
                    for info in ["factors", "cluss", "batch"]:
                        files = glob.glob(
                            self.paths["opt_res"]
                            / f"{info}{self.data.file_id()}"
                        )
                        for filename in files:
                            os.remove(filename)

                    self.data.d_seed[self.data.ext] += 1

                print("infeasible in loop passive")

                self.data.seeds[self.data.ext] = np.delete(
                    self.data.seeds[self.data.ext],
                    len(self.data.seeds[self.data.ext]) - 1)
                self.data.d_ind_seed[self.data.ext] += 1
                seed_ind += 1
                self.data.deterministic_created = False

                _, step_vals = self.data.find_feasible_data(
                    seed_ind, methods, step_vals,
                    evaluation, epoch, passive=True
                )

                self._init_passive_data()

                env.reset(seed=self.data.seed[self.data.ext],
                          load_data=True, passive=True)

                inputs_state_val = [0, env.date, False, env.batch["flex"][:, 0: 2], env.car.store]
                env.get_state_vals(inputs=inputs_state_val, evaluation=evaluation)
                sequence_feasible = True

        return step_vals

    def _baseline_rewards(self, method, evaluation, action, env):
        sequence_feasible = True
        # substract baseline rewards to reward -
        # for training, not evaluating
        if evaluation or len(method.split("_")) == 1 or utils.reward_type(method) != "d":
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
        for home, comb_actions in enumerate(combs_actions):
            # get outp d
            [_, _, reward_a, _, _, constraint_ok, _] = env.step(
                comb_actions,
                implement=False,
                record=False,
                E_req_only=method == "baseline",
                evaluation=evaluation,
            )
            if self.rl['competitive'] and home < self.n_homes:
                reward_a = reward_a[home]
            # add penalty if the constraints are violated
            if not constraint_ok:
                sequence_feasible = False
                reward_a, _ = self._apply_reward_penalty(evaluation, reward_a)
            rewards_baseline.append(reward_a)

        return rewards_baseline, sequence_feasible

    def _init_traj_reward(self, method, evaluation):
        if (
            len(method.split("_")) > 1
            and utils.reward_type(method) == "d"
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
                and utils.reward_type(method) == "d" \
                and not evaluation:

            if self.rl["competitive"]:
                indiv_grid_battery_costs = - np.array(self._get_break_down_reward(
                    break_down_rewards, 'indiv_grid_battery_costs'
                ))
                diff_rewards = [
                    indiv_grid_battery_costs[home] - rewards_baseline[home]
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

    def _get_break_down_reward(self, break_down_rewards, label):
        return break_down_rewards[self.prm['syst']['break_down_rewards_entries'].index(label)]

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
        # record last epoch for analysis of results
        record = epoch == self.rl["n_epochs"]
        while not done and sequence_feasible:
            current_state = state

            action, _ = self.action_selector.select_action(
                method, step, actions, evaluation,
                current_state, eps_greedy, rdn_eps_greedy,
                rdn_eps_greedy_indiv, self.t_env, ext=self.env.ext,
            )

            # interact with environment to get rewards
            rewards_baseline, sequence_feasible = self._baseline_rewards(
                method, evaluation, action, env
            )
            action_ = np.array(action.cpu()) if th.is_tensor(action) else action
            [
                state, done, reward, break_down_rewards, bool_flex, constraint_ok, record_output
            ] = env.step(
                action_, record=record, E_req_only=method == "baseline", evaluation=evaluation
            )
            if record:
                self.last_epoch(evaluation, method, record_output, batch, done)
            if not constraint_ok:
                sequence_feasible = False
                reward, _ = self._apply_reward_penalty(evaluation, reward)
            else:
                step_vals = self._append_step_vals_from_explo(
                    method, evaluation, reward, rewards_baseline, break_down_rewards,
                    current_state, state, action, done, bool_flex, constraint_ok, step_vals
                )
                # if instant feedback,
                # learn right away at the end of the step
                if self.n_homes > 0:
                    t_start_learn = time.time()
                    if not evaluation and not self.rl['trajectory']:
                        for eval_method in utils.methods_learning_from_exploration(
                                method, epoch, self.rl
                        ):
                            self.learning_manager.learning(
                                current_state, state, action, reward, done,
                                eval_method, step, step_vals, epoch
                            )
                    self.learning_manager.q_learning_instant_feedback(
                        evaluation, method, step_vals, step
                    )
                    self.duration_learning += time.time() - t_start_learn

                if self.rl['trajectory']:
                    if type(reward) in [float, int, np.float64]:
                        traj_reward += reward
                    else:
                        for home in self.homes:
                            traj_reward[home] += reward[home]
                step += 1

        sequence_feasible = self._check_rewards_match(
            method, evaluation, step_vals, sequence_feasible
        )

        return step_vals, traj_reward, sequence_feasible

    def _active_get_steps(
            self, env, repeat, epoch, i_explore, methods,
            step_vals, evaluation
    ):
        self.n_homes = self.prm["syst"]["n_homes"]
        self.homes = range(self.n_homes)
        env.set_passive_active(passive=False, evaluation=evaluation)
        rl = self.rl
        if evaluation and self.prm['syst']['test_different_to_train']:
            self.data.ext = "_test"
            self.env.spaces.ext = "_test"
        else:
            self.data.ext = ""
            self.env.spaces.ext = ""

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
        seed_ind += self.data.d_ind_seed[self.data.ext]

        [_, batch], step_vals = self.data.find_feasible_data(
            seed_ind, methods, step_vals, evaluation, epoch
        )

        n_not_feas = 0

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
                utils.set_seeds_rdn(self.data.seed[self.data.ext])

                # reset environment with adequate data
                env.reset(
                    seed=self.data.seed[self.data.ext],
                    load_data=True, E_req_only=method == "baseline",
                    evaluation=evaluation
                )
                # get data from environment
                inputs_state_val = [0, env.date, False, env.batch["flex"][:, 0: 2], env.car.store]

                # initialise data for current method
                if method == method0:
                    initt0 += 1
                vars_env[method] = {entry: [] for entry in self.prm["save"]["last_entries"]}

                actions = None
                if rl["type_learning"] in ["DDPG", "DQN", "facmac", "DDQN"] and rl["trajectory"]:
                    actions, _, states = self.action_selector.trajectory_actions(
                        method, rdn_eps_greedy_indiv, eps_greedy,
                        rdn_eps_greedy, evaluation, self.t_env, ext=self.data.ext
                    )
                state = env.get_state_vals(inputs=inputs_state_val, evaluation=evaluation)
                # check batch corresponds to optimisation

                step_vals, traj_reward, sequence_feasible = self._get_one_episode(
                    method, epoch, actions, state, evaluation, env, batch, step_vals
                )

                if rl["type_learning"] in ["DDPG", "DQN", "DDQN", "facmac"] \
                        and rl["trajectory"] \
                        and not evaluation \
                        and method != "baseline":
                    t_start_learn = time.time()
                    eval_methods = utils.methods_learning_from_exploration(method, epoch, self.rl)
                    for eval_method in eval_methods:
                        self.learning_manager.trajectory_deep_learn(
                            states, actions, traj_reward, eval_method, evaluation, step_vals, epoch
                        )
                    self.duration_learning += time.time() - t_start_learn

            if not sequence_feasible:  # if data is not feasible, make new data
                n_not_feas += 1
                seed_ind = self.data.infeasible_tidy_files_seeds(seed_ind, evaluation)

                print("infeasible in loop active")

                self.data.deterministic_created = False
                print("find feas opt data again!")
                [_, batch], step_vals = self.data.find_feasible_data(
                    seed_ind, methods, step_vals, evaluation, epoch
                )

        step_vals["seed"] = self.data.seed[self.data.ext]
        step_vals["n_not_feas"] = n_not_feas
        if not evaluation:
            self.t_env += self.N

        return step_vals

    def _get_shape_step_vals(self, info, evaluation):
        n_homes = self.prm['syst']['n_homes_test'] if evaluation else self.n_homes
        if info in self.prm['syst']['break_down_rewards_entries']:
            if utils.var_len_is_n_homes(info, self.rl['competitive']):
                shape = (self.N, n_homes)
            else:
                shape = (self.N, 1)
        elif utils.var_len_is_n_homes(info, self.rl['competitive']):
            shape = (self.N, n_homes)
        elif info in self.prm['syst']['indiv_step_vals_entries']:
            shape = (self.N, n_homes, self.dim_step_vals[info])
        elif info in self.global_step_vals_entries:
            shape = (self.N,)
        else:
            print(f"info {info} not recognised")

        return shape

    def get_steps(self, methods, repeat, epoch, i_explore,
                  evaluation=False, new_episode_batch=None, parallel=False):
        """Get episode steps interacting with environment.

        For all inputted types of explorations.
        """
        eval0 = evaluation
        self.data.seed_ind = {}
        self.data.seed = {ext: 0 for ext in self.prm['syst']['n_homes_extensions_all']}
        # create link to objects/data needed in method
        env = copy.deepcopy(self.env) if parallel else self.env

        # initialise output
        step_vals = {}
        for method in methods:
            step_vals[method] = {
                entry: []
                for entry in
                self.prm['syst']['break_down_rewards_entries'] + self.method_vals_entries
            }
            entries = self.prm['syst']['indiv_step_vals_entries'] \
                + self.global_step_vals_entries \
                + self.prm['syst']['break_down_rewards_entries']
            for info in entries:
                shape = self._get_shape_step_vals(info, evaluation)
                step_vals[method][info] = np.full(shape, np.nan)

        self._init_facmac_mac(methods, new_episode_batch, epoch)

        # passive consumers
        step_vals = self._passive_get_steps(
            env, repeat, epoch, i_explore, methods, step_vals
        )
        evaluation = eval0

        step_vals = self._active_get_steps(
            env, repeat, epoch, i_explore, methods, step_vals, evaluation
        )

        if self.rl["type_learning"] != "facmac":
            self.episode_batch = None

        return step_vals, self.episode_batch

    def _check_rewards_match(self, method, evaluation, step_vals, sequence_feasible):
        if self.n_homes > 0 and "opt" in step_vals and step_vals[method]["reward"][-1] is not None:
            if not (
                np.mean(step_vals[method]["reward"]) < np.mean(step_vals["opt"]["reward"]) + 1e-3
            ):
                print(
                    f"reward {method} {np.mean(step_vals[method]['reward'])} "
                    f"better than opt {np.mean(step_vals['opt']['reward'])} "
                    f"self.data.seed[{self.data.ext}] {self.data.seed[self.data.ext]}"
                )
                sequence_feasible = False
                self.tests.investigate_opt_env_rewards_unequal(step_vals, evaluation)

        return sequence_feasible

    def _opt_step_init(
        self, time_step, batchflex_opt, batch_avail_car, res, evaluation
    ):
        step_vals_i = {}
        # update time at each time step
        date = self.env.date0 + timedelta(
            hours=time_step * self.prm["syst"]["dt"]
        )

        # update consumption etc. at the beginning of the time step
        loads = {}
        loads["l_flex"], loads["l_fixed"], loads_step = self._fixed_flex_loads(
            time_step, batchflex_opt, evaluation
        )
        self.tests.check_cons_less_than_or_equal_to_available_loads(
            loads, res, time_step, batchflex_opt, evaluation
        )
        _, _, loads_prev = self._fixed_flex_loads(
            max(0, time_step - 1), batchflex_opt, evaluation
        )
        home_vars = {
            "gen": self.grd["gen"][:, time_step]
        }
        spaces = self.env.get_current_spaces(evaluation)
        step_vals_i["state"] = spaces.opt_step_to_state(
            self.prm, res, time_step, loads_prev,
            loads_step, batch_avail_car, loads, home_vars
        )
        self.env.heat.E_heat_min_max(time_step)
        self.env.heat.potential_E_flex()

        return step_vals_i, date, loads, loads_step, loads_prev, home_vars

    def _check_res_T_air(self, res, time_step):
        for home in self.homes:
            T_air = res["T_air"][home][time_step]
            if T_air < self.env.heat.T_LB[home][time_step] - 1e-1 \
                    or T_air > self.env.heat.T_UB[home][time_step] + 1e-1:
                print(f"home {home} time_step {time_step} "
                      f"res['T_air'][home][time_step] {T_air} "
                      f"T_LB[home] {self.env.heat.T_LB[home][time_step]} "
                      f"T_UB[home] {self.env.heat.T_UB[home][time_step]}")

    def _get_diff_rewards(
            self, evaluation, time_step, action, date,
            loads, res, feasible, reward, indiv_grid_battery_costs
    ):
        obtain_diff_reward = any(
            len(q.split("_")) >= 2
            and utils.reward_type(q) == "d"
            for q in self.rl["type_Qs"]
        )
        if obtain_diff_reward and not evaluation:
            self._check_res_T_air(res, time_step)
            rewards_baseline, feasible_getting_baseline = \
                self._get_artificial_baseline_reward_opt(
                    time_step, action, date, loads, res, evaluation
                )
            if not feasible_getting_baseline:
                feasible = False
            if self.rl["competitive"]:
                diff_rewards = [
                    indiv_grid_battery_costs[home] - rewards_baseline[home]
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

    def _append_step_vals_from_explo(
            self, method, evaluation, reward, rewards_baseline, break_down_rewards,
            current_state, state, action, done, bool_flex, constraint_ok, step_vals
    ):
        diff_rewards = self._compute_diff_rewards(
            method, evaluation, reward, rewards_baseline, break_down_rewards
        )
        global_ind = self._compute_global_ind_state_action(
            current_state, state, action, done, method
        )
        indiv_step_vals = [
            current_state, action, diff_rewards, state, bool_flex
        ]
        global_step_vals = [
            global_ind["state"], global_ind["action"], reward,
            global_ind["next_state"], done, constraint_ok
        ]
        time_step = self.env.time_step - 1
        n_homes = self.prm['syst']['n_homes_test'] if evaluation else self.n_homes
        for info, var in zip(self.prm['syst']['break_down_rewards_entries'], break_down_rewards):
            n = n_homes if utils.var_len_is_n_homes(info, self.rl['competitive']) else 1
            step_vals[method][info][time_step][:n] = var
        for info, var in zip(self.prm['syst']['indiv_step_vals_entries'], indiv_step_vals):
            if var is not None:
                if (
                    utils.var_len_is_n_homes(info, self.rl['competitive'])
                    and len(var) == self.n_homes + 1
                ):
                    var = var[:-1]
                var_ = np.array(var.cpu()) if th.is_tensor(var) else var
                shape = self._get_shape_step_vals(info, evaluation)[1:]
                # try:
                step_vals[method][info][time_step, 0: n_homes] = np.reshape(
                    var_, shape
                )

        for info, var in zip(self.global_step_vals_entries, global_step_vals):
            step_vals[method][info][time_step] = var

        return step_vals

    def _append_step_vals_from_opt(
            self, method, step_vals_i, res, time_step,
            loads_prev, loads_step, batch_avail_car, step_vals,
            break_down_rewards, feasible, loads, home_vars, evaluation
    ):
        keys = self.prm["syst"]["break_down_rewards_entries"] + ["constraint_ok"]
        vars = break_down_rewards + [feasible]
        n_homes = self.prm['syst']['n_homes_test'] if evaluation else self.n_homes
        for key_, var in zip(keys, vars):
            step_vals_i[key_] = var
        for key_ in step_vals_i.keys():
            if step_vals_i[key_] is None:
                continue
            target_shape = self._get_shape_step_vals(key_, evaluation)
            if not isinstance(target_shape, int):
                target_shape = target_shape[1:]
            if key_ == 'diff_rewards' and len(step_vals_i[key_]) == n_homes + 1:
                step_vals_i[key_] = step_vals_i[key_][:-1]
            if len(target_shape) > 0 and target_shape != np.shape(step_vals_i[key_]):
                step_vals_i[key_] = np.reshape(step_vals_i[key_], target_shape)
            if (
                    key_[0: len('indiv')] == 'indiv'
                    or key_ in self.prm['syst']['indiv_step_vals_entries']
                    or (key_ == 'reward' and self.rl['competitive'])
            ):
                step_vals[method][key_][time_step][0: n_homes] = step_vals_i[key_]
            else:
                step_vals[method][key_][time_step] = step_vals_i[key_]

        if time_step > 0:
            step_vals[method]["next_state"][time_step][:n_homes] = step_vals_i["state"]
            if self.rl["type_env"] == "discrete" and method[-2] == 'C':
                step_vals[method]["ind_next_global_state"][time_step] = \
                    step_vals_i["ind_global_state"]
            else:
                step_vals[method]["ind_next_global_state"][time_step] = np.nan
        if time_step == len(res["grid"]) - 1:
            spaces = self.env.get_current_spaces(evaluation)

            step_vals[method]["next_state"][time_step][:n_homes] = \
                spaces.opt_step_to_state(
                    self.prm, res, time_step + 1, loads_prev,
                    loads_step, batch_avail_car, loads, home_vars
                )
            if self.rl["type_env"] == "discrete" and method[-2] == 'C':
                spaces = self.env.get_current_spaces(evaluation)
                ind_next_state = spaces.get_space_indexes(
                    all_vals=step_vals[method]["next_state"][-1])
                step_vals[method]["ind_next_global_state"].append(
                    spaces.indiv_to_global_index(
                        "state", indexes=ind_next_state,
                        multipliers=self.global_multipliers["state"]))
            else:
                step_vals[method]["ind_next_global_state"][time_step] = None

        step_vals[method]["done"][time_step] = \
            False if time_step <= len(res["grid"]) - 2 else True

        return step_vals

    def _instant_feedback_steps_opt(
            self, evaluation, exploration_method, time_step, step_vals, epoch, ext
    ):
        rl = self.rl
        if (rl["type_learning"] in ["DQN", "DDQN", "DDPG", "facmac"]
            or rl["instant_feedback"]) \
                and not evaluation \
                and exploration_method in rl["exploration_methods"] \
                and time_step > 0 \
                and not rl["trajectory"]:

            [
                current_state, actions, reward, state,
                reward_diffs, indiv_grid_battery_costs
            ] = [
                step_vals["opt"][e][time_step]
                for e in [
                    "state", "action", "reward", "next_state",
                    "diff_rewards", "indiv_grid_battery_costs"
                ]
            ]
            if rl["type_learning"] == "q_learning":
                # learner agent learns from this step
                self.learning_manager.learner.learn(
                    "opt", step_vals[exploration_method], time_step - 1
                )
            elif rl["type_learning"] == "facmac":
                pre_transition_data = {
                    "state": th.from_numpy(
                        np.reshape(current_state, (self.n_homes, rl["obs_shape"]))
                    ),
                    "avail_actions": th.Tensor(rl["avail_actions"]),
                    "obs": th.from_numpy(
                        np.reshape(current_state, (self.n_homes, rl["obs_shape"]))
                    )
                }

                post_transition_data = {
                    "actions": th.from_numpy(actions),
                    "reward": th.from_numpy(np.array(reward)),
                    "terminated": th.from_numpy(np.array(time_step == self.N - 1)),
                }

                evaluation_methods = utils.methods_learning_from_exploration(
                    exploration_method, epoch, rl
                )
                for evaluation_method in evaluation_methods:
                    self.episode_batch[evaluation_method].update(
                        pre_transition_data, ts=time_step)
                    self.episode_batch[evaluation_method].update(
                        post_transition_data, ts=time_step)

            elif rl["type_learning"] in ["DDPG", "DQN", "DDQN"]:
                self.learning_manager.independent_deep_learning(
                    current_state, actions, reward, indiv_grid_battery_costs,
                    state, reward_diffs
                )

    def sum_gc_for_start_Call_index(self, res, i, evaluation):
        C = self.grd[f"Call{utils.test_str(evaluation)}"][i: i + self.N]
        loss = self.grd['loss']
        sum_gc_i = np.sum(
            [
                C[time_step_]
                * (res['grid'][time_step_] + loss * res['grid2'][time_step_])
                for time_step_ in range(self.N)
            ]
        )

        return sum_gc_i

    def _check_i0_costs_res(self, res, evaluation):
        # check the correct i0_costs is used
        sum_gc_0 = np.sum(
            [self.grd[f"C{utils.test_str(evaluation)}"][time_step_] * (
                res['grid'][time_step_] + self.grd['loss'] * res['grid2'][time_step_]
            ) for time_step_ in range(self.N)]
        )
        if not (abs(sum_gc_0 - res['grid_energy_costs']) < 1e-3):
            i_start_res = [
                i for i in range(len(self.grd[f'Call{utils.test_str(evaluation)}']) - self.N)
                if abs(
                    self.sum_gc_for_start_Call_index(res, i, evaluation) - res['grid_energy_costs']
                ) < 1e-3
            ]
            if self.env.i0_costs != i_start_res[0]:
                print("update res i0_costs")
                self.env.update_i0_costs(i0_costs=i_start_res[0])
                np.save(self.env.res_path / f"i0_costs{self.env._file_id()}", i_start_res[0])

    def get_steps_opt(
            self, res, pp_simulation_required, step_vals, evaluation, batch, epoch
    ):
        """Translate optimisation results to states, actions, rewards."""
        env, rl = self.env, self.rl
        last_epoch = epoch == rl['n_epochs'] - 1
        feasible = True
        method = "opt"
        sum_rl_rewards = 0
        batchflex_opt, batch_avail_car = [copy.deepcopy(batch[e]) for e in ["flex", "avail_car"]]
        self._check_i0_costs_res(res, evaluation)

        # copy the initial flexible and non-flexible demand -
        # table will be updated according to optimiser's decisions
        self.env.car.reset(self.prm)
        self.env.car.add_batch(batch)
        self.env.heat.reset(self.prm, evaluation=evaluation)
        for time_step in range(len(res["grid"])):
            # initialise step variables
            [step_vals_i, date, loads, loads_step, loads_prev, home_vars] = self._opt_step_init(
                time_step, batchflex_opt, batch_avail_car, res, evaluation
            )

            # translate individual imports/exports into action value
            step_vals_i["bool_flex"], step_vals_i["action"], error = \
                env.action_translator.optimisation_to_rl_env_action(
                    time_step, date, res["netp"][:, time_step],
                    loads, home_vars, res
            )

            step_vals_i = self.env.spaces.get_ind_global_state_action(step_vals_i)
            feasible = not any(error)
            if not feasible:
                break

            if self.grd['compare_pandapower_optimisation'] or pp_simulation_required:
                netp0, _, _ = self.env.get_passive_vars(time_step)
                grdCt = self.grd[f'C{utils.test_str(evaluation)}'][time_step]
                res = self.env.network.compare_optimiser_pandapower(
                    res, time_step, netp0, grdCt
                )
            if not self.grd['manage_voltage'] and self.grd['simulate_panda_power_only']:
                gens = self.grd["gen"][:, time_step]
                input_take_action = date, step_vals_i["action"], gens, loads
                (_, _, _, voltage_squared, _, _, _, _) = env.policy_to_rewardvar(
                    None, other_input=input_take_action
                )
            elif self.grd['manage_voltage']:
                voltage_squared = res['voltage_squared'][:, time_step]
            else:
                voltage_squared = None
            step_vals_i["reward"], break_down_rewards = env.get_reward(
                netp=res["netp"][:, time_step],
                discharge_tot=res["discharge_tot"][:, time_step],
                charge=res["charge"][:, time_step],
                time_step=time_step,
                passive_vars=self.env.get_passive_vars(time_step),
                hourly_line_losses=res['hourly_line_losses'][time_step],
                voltage_squared=voltage_squared,
                evaluation=evaluation
            )
            step_vals_i["indiv_grid_battery_costs"] = - np.array(
                self._get_break_down_reward(break_down_rewards, "indiv_grid_battery_costs")
            )
            self.tests.tests_individual_step_rl_matches_res(
                res, time_step, batch, step_vals_i["reward"], break_down_rewards,
                batchflex_opt, evaluation
            )

            # substract baseline rewards to reward -
            # for training, not evaluating
            step_vals_i["diff_rewards"], feasible = self._get_diff_rewards(
                evaluation, time_step, step_vals_i["action"], date, loads, res,
                feasible, step_vals_i["reward"], step_vals_i["indiv_grid_battery_costs"]
            )
            if not feasible:
                step_vals_i["reward"], step_vals_i["diff_rewards"] = self._apply_reward_penalty(
                    evaluation, step_vals_i["reward"],
                    step_vals_i["diff_rewards"]
                )

            if not (rl["competitive"] and not evaluation):
                sum_rl_rewards += step_vals_i["reward"]

            # update battery and heat objects
            self.env.car.update_step(res, time_step=time_step + 1)
            self.env.heat.update_step(res)

            # append experience dictionaries
            step_vals = self._append_step_vals_from_opt(
                method, step_vals_i, res, time_step,
                loads_prev, loads_step, batch_avail_car, step_vals,
                break_down_rewards, feasible, loads, home_vars, evaluation
            )

            # update flexibility table
            batchflex_opt = self.data.update_flexibility_opt(
                batchflex_opt, res, time_step
            )

            # instant learning feedback
            t_start_learn = time.time()
            self._instant_feedback_steps_opt(
                evaluation, method, time_step, step_vals, epoch, self.env.ext
            )
            self.duration_learning += time.time() - t_start_learn

            # record if last epoch
            self._record_last_epoch_opt(
                res, time_step, break_down_rewards, batchflex_opt,
                last_epoch, step_vals_i, batch, evaluation, voltage_squared
            )

        if not self.rl['competitive'] and feasible:
            self.tests.test_total_rewards_match(evaluation, res, sum_rl_rewards)
        if not evaluation \
                and rl["type_learning"] in ["DDPG", "DQN", "facmac"] \
                and rl["trajectory"] and feasible:
            self.learning_manager.learn_trajectory_opt(step_vals, epoch)

        return step_vals, feasible

    def _record_last_epoch_opt(
            self, res, time_step, break_down_rewards, batchflex_opt,
            last_epoch, step_vals_i, batch, evaluation, voltage_squared
    ):
        if not last_epoch:
            return
        done = time_step == self.N - 1
        ldflex = np.zeros(self.n_homes) \
            if done \
            else np.sum(batchflex_opt[:, time_step, 1:])
        if done:
            ldfixed = np.sum(batchflex_opt[:, time_step])
        else:
            ldfixed = batchflex_opt[:, time_step, 0]
        tot_cons_loads = res["totcons"][:, time_step] - res["E_heat"][:, time_step]
        flex_cons = tot_cons_loads - ldfixed
        wholesalet, cintensityt = [
            self.grd[f"{e}{utils.test_str(evaluation)}"][self.env.i0_costs + time_step]
            for e in ["wholesale_all", "cintensity_all"]
        ]
        if self.grd['manage_voltage']:
            q_car = res["q_car_flex"][:, time_step]
            q_house = res["netq_flex"][:, time_step] - q_car
        elif self.grd['simulate_panda_power_only']:
            p_car_flex = res['charge'][:, time_step] / self.prm['car']['eta_ch'] \
                - res['discharge_other'][:, time_step]
            q_car = p_car_flex * self.grd['active_to_reactive_flex']
            q_house = (
                res['totcons'][:, time_step] - self.grd['gen'][:, time_step]
            ) * self.grd['active_to_reactive_flex']
        else:
            q_car, q_house = None, None

        if (
                self.grd['compare_pandapower_optimisation']
                or self.grd['simulate_panda_power_only']
        ):
            loaded_buses, sgen_buses = self.env.network.loaded_buses, self.env.network.sgen_buses
        else:
            loaded_buses, sgen_buses, = None, None

        record_output = []
        for entry in [
            'netp', 'netp0', 'discharge_other', 'store', 'totcons', 'E_heat',
            'T', 'T_air'
        ]:
            record_output.append(res[entry][:, time_step])
        record_output.append(voltage_squared)
        record_output.append(res['hourly_line_losses'][time_step])
        record_output += [
            step_vals_i["action"], step_vals_i["reward"], flex_cons,
            ldflex, ldfixed, tot_cons_loads,
            self.grd[f"C{utils.test_str(evaluation)}"][time_step], wholesalet, cintensityt,
            break_down_rewards,
            loaded_buses, sgen_buses,
            res['q_ext_grid'][time_step],
            q_car, q_house
        ]

        self.last_epoch(evaluation, "opt", record_output, batch, done)

    def _apply_reward_penalty(self, evaluation, reward, diff_rewards=None):
        if self.rl["apply_penalty"] and not evaluation:
            self.tests.check_competitive_has_diff_rewards(diff_rewards)
            if diff_rewards is not None:
                for home in self.homes:
                    diff_rewards[home] -= self.rl["penalty"]
            else:
                reward -= self.rl["penalty"]

        return reward, diff_rewards

    def _fixed_flex_loads(self, time_step, batchflex_opt, evaluation):
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
        n_homes = self.prm['syst']['n_homes' + self.data.ext]
        if time_step == self.N - 1:
            flex_load = np.zeros(n_homes)
            l_fixed = np.array(
                [sum(batchflex_opt[home][time_step][:]) for home in range(n_homes)]
            )
        else:
            flex_load = np.array(
                [sum(batchflex_opt[home][time_step][1:]) for home in range(n_homes)]
            )
            l_fixed = np.array(
                [batchflex_opt[home][time_step][0] for home in range(n_homes)]
            )

        loads_step = l_fixed + flex_load

        return flex_load, l_fixed, loads_step

    def _get_combs_actions(self, actions):
        combs_actions = np.ones((self.n_homes + 1, self.n_homes, self.rl['dim_actions_1']))
        for home in self.homes:
            actions_baseline_a = np.array(actions)
            actions_baseline_a[home] = 1
            combs_actions[home] = actions_baseline_a
        combs_actions[-1] = 1

        return combs_actions

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
        combs_actions = self._get_combs_actions(actions)
        feasible = True

        for home, comb_actions in enumerate(combs_actions):
            bat_store = self.env.car.store.copy()
            input_take_action = date, comb_actions, gens, loads
            home_vars, loads, hourly_line_losses, voltage_squared, \
                q_ext_grid, constraint_ok, _, _ = \
                env.policy_to_rewardvar(None, other_input=input_take_action)
            self.env.car.store = bat_store
            passive_vars = self.env.get_passive_vars(time_step)
            reward_baseline_a, _ = env.get_reward(
                netp=home_vars["netp"],
                discharge_tot=self.env.car.discharge_tot,
                charge=self.env.car.charge,
                time_step=time_step,
                passive_vars=passive_vars,
                hourly_line_losses=hourly_line_losses,
                voltage_squared=voltage_squared,
                evaluation=evaluation,
            )
            if self.rl['competitive'] and home < self.n_homes:
                reward_baseline_a = reward_baseline_a[home]
            if not constraint_ok:
                feasible = False
                print(f"self.data.seed = {self.data.seed} "
                      f"constraint_ok False, time_step {time_step}")
                self._apply_reward_penalty(evaluation, reward_baseline_a)

            rewards_baseline.append(reward_baseline_a)

            # revert back store
            self.env.car.store = [res["store"][home][time_step] for home in self.homes]

        return rewards_baseline, feasible

    def _init_facmac_mac(self, methods, new_episode_batch, epoch):
        if self.rl["type_learning"] == "facmac":
            for exploration_method in methods:
                evaluation_methods = utils.methods_learning_from_exploration(
                    exploration_method, epoch, self.rl
                )
                for evaluation_method in evaluation_methods:
                    self.episode_batch[evaluation_method] = new_episode_batch()
                    if evaluation_method not in ["baseline", "opt"]:
                        self.action_selector.mac[evaluation_method].init_hidden(
                            batch_size=self.rl["batch_size_run"]
                        )
