#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:47:57 2022.

@author: floracharbonnier

"""

import copy
import glob
import os
from datetime import timedelta
from typing import Tuple

import numpy as np

from src.simulations.data_manager import DataManager
from src.simulations.learning import LearningManager
from src.simulations.select_actions import ActionSelector
from src.utilities.userdeftools import (data_source, initialise_dict,
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
        self.res_path = prm["paths"]["res_path"]
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

        self._init_i0_costs()

        self.paths = prm["paths"]

    def _init_i0_costs(self):
        self.i0_costs = 0
        self.prm["grd"]["C"] = \
            self.prm["grd"]["Call"][
            self.i0_costs: self.i0_costs + self.prm["syst"]["N"]
        ]
        self.env.update_date(self.i0_costs)

    def _initialise_passive_vars(self, env, repeat, epoch, i_explore):
        self.n_homes = self.prm["ntw"]["nP"]
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
        if self.prm["ntw"]["nP"] == 0:
            return step_vals

        # initialise variables for passive case
        seed_ind, method, done, sequence_feasible, record, evaluation \
            = self._initialise_passive_vars(env, repeat, epoch, i_explore)

        # find feasible data
        _, step_vals, mus_opt = \
            self.data.find_feasible_data(
                seed_ind, methods, step_vals, evaluation,
                epoch, passive=True)

        # reset environment
        env.reset(seed=self.data.seed[self.data.passive_ext],
                  load_data=True, passive=True)

        # interact with environment in a passive way for each step
        while sequence_feasible and not done:
            action = self.rl["default_actionP"]
            state, done, reward, _, _, sequence_feasible, [
                netp, discharge_tot, charge] = env.step(
                action, record=record,
                evaluation=evaluation, netp_storeout=True)
            for e, val in zip(["netp0", "discharge_tot0", "charge0"],
                              [netp, discharge_tot, charge]):
                self.prm["loads"][e].append(val)
            if not sequence_feasible:
                # if data is not feasible, make new data
                if seed_ind < len(self.data.seeds[self.data.passive_ext]):
                    self.data.d_ind_seed[self.data.passive_ext] += 1
                    seed_ind += 1
                else:
                    for e in ["factors", "cluss", "batch"]:
                        files = glob.glob(self.paths["res_path"]
                                          / f"{e}{self.data.file_id()}")
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

                _, step_vals, mus_opt = \
                    self.data.find_feasible_data(
                        seed_ind, methods, step_vals,
                        evaluation, epoch, passive=True)

                self._init_passive_data()

                env.reset(seed=self.data.seed[self.data.passive_ext],
                          load_data=True, passive=True)

                inputs_state_val = \
                    [0, env.date, False,
                     [[env.batch[home]["flex"][ih] for ih in range(0, 2)]
                      for home in self.homes],
                     env.bat.store]
                env.get_state_vals(inputs=inputs_state_val)
                sequence_feasible = True
        for e in ["netp0", "discharge_tot0", "charge0"]:
            self.prm["loads"][e] = \
                [[self.prm["loads"][e][time][home] for time in range(self.N)]
                 for home in self.homes]

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
        for ca in combs_actions:
            # get outp d
            [_, _, reward_a, _, _, constraint_ok, _] = \
                env.step(ca, implement=False,
                         record=False,
                         E_req_only=method == "baseline")

            # add penalty if the constraints are violated
            if not constraint_ok:
                sequence_feasible = False
                reward_a = self._apply_reward_penalty(
                    evaluation, reward_a)
            rewards_baseline.append(reward_a)
            if reward_a is None:
                print(f"reward_a {reward_a}")

        return rewards_baseline, sequence_feasible

    def _get_one_episode(
            self, method, epoch, actions, state,
            mus_opt, evaluation, env, batch, step_vals
    ):
        step, done = 0, 0
        sequence_feasible = True
        if (
            len(method.split("_")) > 1
            and reward_type(method) == "d"
            and not evaluation
        ):
            traj_reward = [0 for _ in self.homes]
        else:
            traj_reward = 0
        # loop through steps until either end of sequence
        # or one step if infeasible
        eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv \
            = self.action_selector.set_eps_greedy_vars(self.rl, epoch, evaluation)

        while not done and sequence_feasible:
            current_state = state

            action, tf_prev_state \
                = self.action_selector.select_action(
                    method, step, actions, mus_opt, evaluation,
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
                traj_reward = self.learning_manager.learning(
                    current_state, state, action, reward,
                    done, method, step, evaluation, traj_reward)

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
                indiv_rewards = - np.array(break_down_rewards[-1])
                step_vals_ = \
                    [current_state, global_ind["state"], action,
                     global_ind["action"], reward, diff_rewards, indiv_rewards,
                     state, global_ind["next_state"], done, bool_flex,
                     constraint_ok, *break_down_rewards]

                for e, var in zip(self.step_vals_entries, step_vals_):
                    step_vals[method][e].append(var)

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
        self.n_homes = self.prm["ntw"]["n"]
        self.homes = range(self.n_homes)
        # initialise data
        type_actions_nonopt = [method for method in methods if method != "opt"]
        t0 = type_actions_nonopt[0]
        initt0 = 0
        eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv \
            = self.action_selector.set_eps_greedy_vars(rl, epoch, evaluation)
        # make data for optimisation
        # seed_mult = 1 # for initial passive consumers
        seed_ind = self.ind_seed_deterministic \
            if rl["deterministic"] == 1 \
            else self.data.get_seed_ind(repeat, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.passive_ext]

        [res, _, _, batch], step_vals, mus_opt = \
            self.data.find_feasible_data(
                seed_ind, methods, step_vals, evaluation, epoch)

        n_not_feas, not_feas_vars = 0, []

        # loop through types of actions specified to interact with environment
        # start assuming data is infeasible until proven otherwise
        sequence_feasible = False
        it = 0
        while not sequence_feasible:
            # sequence_feasible will be False
            # if it turns out the data for the sequence is not feasible
            it += 1
            sequence_feasible = True
            vars_env = {}
            i_t = 0
            # loop through other non optimisation types
            # -> start over steps with new data
            while i_t < len(type_actions_nonopt) and sequence_feasible:
                method = type_actions_nonopt[i_t]
                i_t += 1
                self.data.get_seed(seed_ind)
                set_seeds_rdn(self.data.seed[self.data.passive_ext])

                # reset environment with adequate data
                env.reset(seed=self.data.seed[self.data.passive_ext],
                          load_data=True, E_req_only=method == "baseline")
                # get data from environment
                inputs_state_val = \
                    [0, env.date, False,
                     [[env.batch[home]["flex"][ih] for ih in range(0, 2)]
                      for home in self.homes], env.bat.store]

                # initialise data for current method
                if method == t0:
                    initt0 += 1
                step_vals[method] = initialise_dict(
                    self.step_vals_entries + self.method_vals_entries)
                vars_env[method] = initialise_dict(self.prm["save"]["last_entries"])

                actions = None
                if rl["type_learning"] in ["DDPG", "DQN"] \
                        and rl["trajectory"]:
                    actions, ind_actions, states = \
                        self.action_selector.trajectory_actions(
                            method, rdn_eps_greedy_indiv,
                            eps_greedy, rdn_eps_greedy)
                state = env.get_state_vals(inputs=inputs_state_val)
                step_vals, traj_reward, sequence_feasible \
                    = self._get_one_episode(
                        method, epoch, actions, state,
                        mus_opt, evaluation, env, batch, step_vals
                    )

                if rl["type_learning"] in ["DDPG", "DQN"] \
                        and rl["trajectory"] \
                        and not evaluation \
                        and method != "baseline":
                    self.learning_manager.trajectory_deep_learn(
                        states, actions, traj_reward, method, evaluation)

            if not sequence_feasible:  # if data is not feasible, make new data
                n_not_feas += 1
                not_feas_vars.append([env.bat.store0, method])
                seed_ind = self.data.infeasible_tidy_files_seeds(seed_ind)

                print("infeasible in loop active")

                self.data.deterministic_created = False
                print("find feas opt data again!")
                [res, _, _, batch], step_vals, mus_opt = \
                    self.data.find_feasible_data(
                        seed_ind, methods, step_vals,
                        evaluation, epoch)

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
        self._init_facmac_mac(methods, new_episode_batch)

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
                           f"reward {method} better than opt"

    def _opt_step_init(
            self, i_step, batchflex_opt, cluss, factors, batch_avail_EV, res
    ):
        step_vals_i = {}
        # update time at each time step
        date = self.prm["syst"]["current_date0"] + timedelta(hours=i_step)

        # update consumption etc at the beginning of the time step
        loads = {}
        loads["l_flex"], loads["l_fixed"], loads_step = self._fixed_flex_loads(
            i_step, batchflex_opt)
        _, _, loads_prev = self._fixed_flex_loads(
            max(0, i_step - 1), batchflex_opt)
        home_vars = {
            "gen": np.array(
                [self.prm["ntw"]["gen"][home][i_step] for home in self.homes]
            )
        }

        step_vals_i["state"] = self.env.spaces.opt_step_to_state(
            self.prm, res, i_step, cluss, factors, loads_prev, loads_step, batch_avail_EV)

        self.env.heat.E_heat_min_max(i_step)
        self.env.heat.potential_E_flex()

        return step_vals_i, date, loads, loads_step, loads_prev, home_vars

    def _get_passive_vars(self, i_step):
        passive_vars = \
            [[self.prm["loads"][e][home][i_step]
              for home in range(self.prm["ntw"]["nP"])]
             for e in ["netp0", "discharge_tot0", "charge0"]]

        return passive_vars

    def _get_diff_rewards(
            self, evaluation, i_step, action, date,
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
                    i_step, action, date, loads, res, evaluation
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
            self, method, step_vals_i, res, i_step, cluss, factors,
            loads_prev, loads_step, batch_avail_EV, step_vals,
            break_down_rewards, feasible
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

        if i_step > 0:
            step_vals[method]["next_state"].append(step_vals_i["state"])
            if self.prm["RL"]["type_env"] == "discrete" and method[-2] == 'C':
                step_vals[method]["ind_next_global_state"].append(
                    step_vals_i["ind_global_state"])
            else:
                step_vals[method]["ind_next_global_state"].append(None)
        if i_step == len(res["grid"]) - 1:
            step_vals[method]["next_state"].append(self.env.spaces.opt_step_to_state(
                self.prm, res, i_step + 1, cluss, factors, loads_prev,
                loads_step, batch_avail_EV))
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
            False if i_step <= len(res["grid"]) - 2 else True)

        return step_vals

    def _tests_individual_step_rl_matches_res(
            self, res, i_step, batch, reward
    ):
        prm = self.prm
        flex, dem, gen = [np.array(
            [batch[home][e] for home in range(len(batch))])
            for e in ["flex", "loads", "gen"]
        ]
        # check tot cons
        for home in self.homes:
            assert res["totcons"][home][i_step] <= \
                   sum(flex[home][i_step]) \
                   + self.env.heat.E_heat_min[home] \
                   + self.env.heat.potential_E_flex()[home] + 1e-3, \
                   f"cons more than sum fixed + flex!, " \
                   f"home = {home}, i_step = {i_step}"

        # check loads and consumption match
        sum_consa = 0
        for load_type in range(2):
            sum_consa += np.sum(res[f'consa({load_type})'])
        assert abs(np.sum(dem[:, 0: prm['syst']['N']]) - sum_consa) < 1e-3, \
            f"res cons {sum_consa} does not match input demand {np.sum(dem)}"

        # check environment uses the same grid coefficients
        assert self.env.grdC[i_step] == prm["grd"]["C"][i_step], \
            f"env grdC {self.env.grdC[i_step]} " \
            f"!= explorer {prm['grd']['C'][i_step]}"

        # check we can replicate res['gc']
        sumgc = np.sum(
            [prm["grd"]["C"][i_step_] * (
                res['grid'][i_step_][0]
                + prm["grd"]['loss'] * res['grid2'][i_step_][0]
            ) for i_step_ in range(24)]
        )
        assert abs(res['gc'] - sumgc) < 1e-3, \
            f"we cannot replicate res['gc'] {res['gc']} vs {sumgc}"

        # check reward from environment and res variables match
        res_reward_t = \
            - (prm["grd"]["C"][i_step]
               * (res["grid"][i_step][0]
                  + prm["grd"]["R"] / (prm["grd"]["V"] ** 2)
                  * res["grid2"][i_step][0])
               + prm["bat"]["C"]
               * sum(res["discharge_tot"][home][i_step]
                     + res["charge"][home][i_step]
                     for home in self.homes)
               + prm["ntw"]["C"]
               * sum(res["netp_abs"][home][i_step]
                     for home in self.homes))
        if not prm["RL"]["competitive"]:
            assert abs(reward - res_reward_t) < 1e-3, \
                f"reward {reward} != res_reward_t " \
                f"from res variables {res_reward_t}"

    def _instant_feedback_steps_opt(
            self, evaluation, method, i_step, step_vals
    ):
        rl = self.prm["RL"]
        if (rl["type_learning"] in ["DQN", "DDQN", "DDPG", "facmac"]
            or rl["instant_feedback"]) \
                and not evaluation \
                and method in rl["exploration_methods"] \
                and i_step > 0 \
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
                    "opt", step_vals[method], i_step - 1
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
                    "terminated": [(i_step == self.prm["syst"]["N"] - 1,)],
                }
                methods = self.types_that_learn_from_t(method)
                for t_ in methods:
                    self.episode_batch[t_].update(
                        pre_transition_data, ts=i_step)
                    self.episode_batch[t_].update(
                        post_transition_data, ts=i_step)

            elif rl["type_learning"] in ["DDPG", "DQN", "DDQN"]:
                self.learning_manager.independent_deep_learning(
                    current_state, actions, reward, indiv_rewards,
                    state, reward_diffs
                )

    def _test_total_rewards_match(self, evaluation, res, sum_RL_rewards):
        sum_res_rewards = (- (res["gc"] + res["sc"] + res["dc"]))
        if not (self.prm["RL"]["competitive"] and not evaluation):
            assert abs(sum_RL_rewards - sum_res_rewards) < 5e-3, \
                "tot rewards don't match: "\
                f"sum_RL_rewards = {sum_RL_rewards}, "
            f"sum costs opt = {- (res['gc'] + res['sc'] + res['dc'])}"

    def get_steps_opt(self, res, step_vals, evaluation, cluss,
                      factors, batch, seed, last_epoch=False):
        """Translate optimisation results to states, actions, rewards."""
        env, rl = self.env, self.prm["RL"]
        feasible = True
        method = "opt"
        sum_RL_rewards = 0
        all_actions = []
        step_vals[method] = initialise_dict(
            self.step_vals_entries)
        batchflex_opt, batch_avail_EV = \
            [[batch[home][e] for home in range(len(batch))]
             for e in ["flex", "avail_EV"]]
        # copy the initial flexible and non flexible demand -
        # table will be updated according to optimiser's decisions
        self.env.bat.reset(self.prm)
        self.env.bat.add_batch(batch)
        self.env.heat.reset(self.prm)

        for i_step in range(len(res["grid"])):
            # initialise step variables
            [step_vals_i, date, loads, loads_step, loads_prev, home] \
                = self._opt_step_init(
                i_step, batchflex_opt, cluss, factors, batch_avail_EV, res
            )

            # translate dp into action value
            step_vals_i["bool_flex"], step_vals_i["action"], error = \
                env.action_translator.optimisation_to_rl_env_action(
                    i_step, date, res["netp"][:, i_step],
                    loads, home, res)

            step_vals_i = self.env.spaces.get_ind_global_state_action(step_vals_i)
            feasible = not any(error)

            # determine rewards
            step_vals_i["reward"], break_down_rewards = env.get_reward(
                res["netp"][:, i_step],
                res["discharge_tot"][:, i_step],
                res["charge"][:, i_step],
                i_step=i_step,
                passive_vars=self._get_passive_vars(i_step),
                evaluation=evaluation
            )
            step_vals_i["indiv_rewards"] = - np.array(break_down_rewards[-1])
            self._tests_individual_step_rl_matches_res(
                res, i_step, batch, step_vals_i["reward"]
            )

            # substract baseline rewards to reward -
            # for training, not evaluating
            step_vals_i["diff_rewards"], feasible = self._get_diff_rewards(
                evaluation, i_step, step_vals_i["action"], date, loads, res,
                feasible, step_vals_i["reward"], step_vals_i["indiv_rewards"]
            )
            if not feasible:
                step_vals_i["reward"] = self._apply_reward_penalty(
                    evaluation, step_vals_i["reward"],
                    step_vals_i["diff_rewards"]
                )
            if not (rl["competitive"] and not evaluation):
                sum_RL_rewards += step_vals_i["reward"]

            # append experience dictionaries
            step_vals = self._append_step_vals(
                method, step_vals_i, res, i_step, cluss, factors,
                loads_prev, loads_step, batch_avail_EV, step_vals,
                break_down_rewards, feasible
            )

            # update flexibility table
            batchflex_opt = self.data.update_flexibility_opt(
                batchflex_opt, res, i_step
            )

            # instant learning feedback
            self._instant_feedback_steps_opt(
                evaluation, method, i_step, step_vals
            )

            # update battery and heat objects
            self.env.bat.update_step(res)
            self.env.heat.update_step(res)

            # record if last epoch
            self._record_last_epoch_opt(
                res, i_step, break_down_rewards, batchflex_opt,
                last_epoch, step_vals_i, batch, evaluation
            )

        self._test_total_rewards_match(evaluation, res, sum_RL_rewards)

        if not evaluation \
                and rl["type_learning"] in ["DDPG", "DQN"] \
                and rl["trajectory"]:
            self.learning_manager.learn_trajectory_opt()

        return step_vals, all_actions, feasible

    def _record_last_epoch_opt(
            self, res, i_step, break_down_rewards, batchflex_opt,
            last_epoch, step_vals_i, batch, evaluation
    ):
        if not last_epoch:
            return

        done = i_step == self.prm["syst"]["N"] - 1
        ldflex = [0 for _ in self.homes] \
            if done \
            else [sum(batchflex_opt[home][i_step][1:])
                  for home in self.homes]
        if done:
            ldfixed = [sum(batchflex_opt[home][i_step][:])
                       for home in self.homes]
        else:
            ldfixed = [batchflex_opt[home][i_step][0]
                       for home in self.homes]
        tot_cons_loads = \
            [res["totcons"][home][i_step] - res["E_heat"][home][i_step]
             for home in self.homes]
        wholesalet, cintensityt = \
            [self.prm["grd"][e][self.i0_costs + i_step]
             for e in ["wholesale_all", "cintensity_all"]]

        record_output = \
            [res["netp"][:, i_step], res["discharge_other"][:, i_step],
             step_vals_i["action"], step_vals_i["reward"], break_down_rewards,
             res["store"][:, i_step], ldflex, ldfixed,
             res["totcons"][:, i_step], tot_cons_loads,
             res["E_heat"][:, i_step], res["T"][:, i_step],
             res["T_air"][:, i_step], self.prm["grd"]["C"][i_step],
             wholesalet, cintensityt]

        self.last_epoch(evaluation, "opt", record_output, batch, done)

    def _apply_reward_penalty(self, evaluation, reward, diff_rewards=None):
        if self.rl["apply_penalty"] and not evaluation:
            if self.rl["competitive"]:
                assert diff_rewards is not None
            for home in self.homes:
                diff_rewards[home] -= self.rl["penalty"]
            else:
                reward -= self.rl["penalty"]

        return reward, diff_rewards

    def _fixed_flex_loads(self, i_step, batchflex_opt):
        """
        Get fixed and flexible consumption equivalent to optimisation results.

        Obtain total fixed and flexible loads for each agent
        for a given time step based on current optimisation results
        """
        # note that we could also obtain the fixed cons / flexible
        # load as below,
        # however we want to count it consistently with our
        # batchflex_opt updates:
        # l_fixed = [ntw['dem'][0, home, i_step] for home in range(n_homes)]
        # flex_load = [ntw['dem'][1, home, i_step] for home in range(n_homes)]

        if i_step == self.prm["syst"]["N"] - 1:
            flex_load = np.zeros(self.n_homes)
            l_fixed = np.array(
                [sum(batchflex_opt[home][i_step][:]) for home in self.homes]
            )
        else:
            flex_load = np.array(
                [sum(batchflex_opt[home][i_step][1:]) for home in self.homes]
            )
            l_fixed = np.array(
                [batchflex_opt[home][i_step][0] for home in self.homes]
            )

        loads_step = l_fixed + flex_load

        return flex_load, l_fixed, loads_step

    def _get_artificial_baseline_reward_opt(self,
                                            i_step,
                                            actions,
                                            date,
                                            loads,
                                            res,
                                            evaluation
                                            ) -> Tuple[list, bool]:
        """
        Get instantaneous rewards if agent took baseline actions.

        Get instantaneous rewards if each agent took baseline
        action instead of current action.
        """
        prm, env = self.prm, self.env
        rewards_baseline = []
        gens = [prm["ntw"]["gen"][home][i_step] for home in self.homes]
        self.env.heat.T = res["T"][:, i_step]
        self.env.bat.store = \
            [res["store"][home][i_step] for home in self.homes]
        combs_actions = []
        for home in self.homes:
            actions_baseline_a = actions.copy()
            actions_baseline_a[home] = \
                [1 for _ in range(self.prm["RL"]["dim_actions"])]
            combs_actions.append(actions_baseline_a)
        combs_actions.append([[1 for _ in range(self.prm["RL"]["dim_actions"])]
                              for _ in self.homes])
        feasible = True
        for home in self.homes:
            T_air = res["T_air"][home][i_step]
            if T_air < self.env.heat.T_LB[home][i_step] - 1e-1 \
                    or T_air > self.env.heat.T_UB[home][i_step] + 1e-1:
                print(f"home {home} i_step {i_step} "
                      f"res['T_air'][home][i_step] {T_air} "
                      f"T_LB[home] {self.env.heat.T_LB[home][i_step]} "
                      f"T_UB[home] {self.env.heat.T_UB[home][i_step]}")
        for ca in combs_actions:
            bat_store = self.env.bat.store.copy()
            input_take_action = date, ca, gens, loads
            home_vars, loads, constraint_ok = env.policy_to_rewardvar(
                None, other_input=input_take_action)
            self.env.bat.store = bat_store
            passive_vars = self._get_passive_vars(i_step)

            reward_baseline_a, _ = env.get_reward(
                home_vars["netp"], self.env.bat.discharge_tot, self.env.bat.charge,
                i_step=i_step, passive_vars=passive_vars,
                evaluation=evaluation)

            if not constraint_ok:
                feasible = False
                print(f"self.data.seed = {self.data.seed} "
                      f"constraint_ok False, i_step {i_step}")
                self._apply_reward_penalty(evaluation, reward_baseline_a)

            rewards_baseline.append(reward_baseline_a)

            # revert back store
            self.env.bat.store = [res["store"][home][i_step]
                                  for home in self.homes]

        return rewards_baseline, feasible

    def types_that_learn_from_t(self, method):
        """Compute list of methods that learn from method."""
        if method == 'opt':
            methods = [
                method for method in self.rl['type_Qs']
                if data_source(method) == "opt" and method in self.rl["evaluation_methods"]
            ]
        else:
            methods = [method]

        return methods

    def _init_facmac_mac(self, exploration_methods, new_episode_batch):
        if self.rl["type_learning"] == "facmac":
            for exploration_method in exploration_methods:
                evaluation_methods = self.types_that_learn_from_t(exploration_method)
                for evaluation_method in evaluation_methods:
                    self.episode_batch[evaluation_method] = new_episode_batch()
                    if evaluation_method not in ["baseline", "opt"]:
                        self.action_selector.mac[evaluation_method].init_hidden(
                            batch_size=self.rl["batch_size_run"]
                        )
