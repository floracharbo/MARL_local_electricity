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

from simulations.data_manager import Data_manager
from simulations.learning import LearningManager
from simulations.select_actions import ActionSelector
from utils.userdeftools import initialise_dict, set_seeds_rdn


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
        for e in ["n_agents", "discrete", "descriptors", "multipliers",
                  "global_multipliers", "granularity", "brackets"]:
            self.__dict__[e] = env.spaces.__dict__[e]
        self.last_epoch = record.last_epoch
        self.res_path = prm["paths"]["res_path"]
        for e in ["D", "solver", "N"]:
            self.__dict__[e] = prm["syst"][e]
        self.episode_batch = {}

        self.data = Data_manager(env, prm, self)
        self.action_selector = ActionSelector(prm, learner, self.episode_batch)
        self.action_selector.mac = mac

        self.learning_manager = LearningManager(
            env, prm, learner, self.episode_batch
        )

        self.break_down_rewards_entries = \
            prm["syst"]["break_down_rewards_entries"]
        self.step_vals_entries = \
            ["state", "ind_global_state", "action", "ind_global_action",
             "reward", "next_state", "ind_next_global_state",
             "done", "bool_flex", "constraint_ok"] \
            + self.break_down_rewards_entries
        self.method_vals_entries = ["seeds", "n_not_feas", "not_feas_vars"]

        self.i0_costs = 0
        self.prm["grd"]["C"] = \
            prm["grd"]["Call"][self.i0_costs: self.i0_costs + prm["syst"]["N"]]
        env.update_date(self.i0_costs)
        self.paths = prm["paths"]
        self.state_funcs = {
            "store0": self._get_store,
            "grdC_level": self._get_grdC_level,
            "dT_next": self._get_dT_next,
            "EV_tau": self._get_EV_tau,
            "bat_dem_agg": self._get_bat_dem_agg
        }

    def _initialise_passive_vars(self, env, ridx, epoch, i_explore):
        self.n_agents = self.prm["ntw"]["nP"]
        self.agents = range(self.n_agents)
        # get environment seed
        seed_ind = self.ind_seed_deterministic \
            if self.rl["deterministic"] == 1 \
            else self.data.get_seed_ind(ridx, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.p]
        env.set_passive_active(passive=True)
        t = "baseline"
        done = 0
        sequence_feasible = True
        record, evaluation = False, False

        return seed_ind, t, done, sequence_feasible, record, evaluation

    def get_steps(self, type_actions, ridx, epoch, i_explore,
                  evaluation=False, new_episode_batch=None, parallel=False):
        """Get episode steps interacting with environment.

        For all inputted types of explorations.
        """
        eval0 = evaluation
        self.data.seed_ind = {}
        self.data.seed = {"P": 0, "": 0}
        # create link to objects/data needed in method
        rl = self.rl
        env = copy.deepcopy(self.env) if parallel else self.env

        # initialise data
        type_actions_nonopt = [t for t in type_actions if t != "opt"]
        t0 = type_actions_nonopt[0]
        initt0 = 0

        eps_greedy, rdn_eps_greedy, rdn_eps_greedy_indiv \
            = self._set_eps_greedy_vars(rl, epoch, evaluation)

        # initialise output
        step_vals = initialise_dict(type_actions)
        self._init_facmac_mac(type_actions, new_episode_batch)

        # passive consumers
        self.data.p = "P"
        for e in ["netp0", "discharge_tot0", "charge0"]:
            self.prm["loads"][e] = []
        if self.prm["ntw"]["nP"] > 0:
            # initialise variables for passive case
            seed_ind, t, done, sequence_feasible, record, evaluation \
                = self._initialise_passive_vars(env, ridx, epoch, i_explore)

            # find feasible data
            _, step_vals, mus_opt = \
                self.data.find_feasible_data(
                    seed_ind, type_actions, step_vals, evaluation,
                    epoch, passive=True)

            # reset environment
            env.reset(seed=self.data.seed[self.data.p],
                      load_data=True, passive=True)

            # interact with environment in a passive way for each step
            while not done:
                action = rl["default_actionP"]
                state, done, reward, _, _, constraint_ok, [
                    netp, discfharge_tot, charge] = env.step(
                    action, record=record,
                    evaluation=evaluation, netp_storeout=True)
                for e, val in zip(["netp0", "discharge_tot0", "charge0"],
                                  [netp, discharge_tot, charge]):
                    self.prm["loads"][e].append(val)
                if not constraint_ok:
                    sequence_feasible = False
                if not sequence_feasible:
                    # if data is not feasible, make new data
                    if seed_ind < len(self.data.seeds[self.data.p]):
                        self.data.d_ind_seed[self.data.p] += 1
                        seed_ind += 1
                    else:
                        for e in ["fs", "cluss", "batch"]:
                            files = glob.glob(self.paths["res_path"]
                                              / f"{e}{self.data.file_id()}")
                            for filename in files:
                                os.remove(filename)
                        self.data.d_seed[self.data.p] += 1

                    print("infeasible in loop passive")
                    self.data.seeds[self.data.p] = np.delete(
                        self.data.seeds[self.data.p],
                        len(self.data.seeds[self.data.p]) - 1)
                    self.data.d_ind_seed[self.data.p] += 1
                    seed_ind += 1
                    self.deterministic_created = False

                    _, step_vals, mus_opt = \
                        self.data.find_feasible_data(
                            seed_ind, type_actions, step_vals,
                            evaluation, epoch, passive=True)
                    for e in ["netp0", "discharge_tot0", "charge0"]:
                        self.prm["loads"][e] = []
                    env.reset(seed=self.data.seed[self.data.p],
                              load_data=True, passive=True)
                    inputs_state_val = \
                        [0, env.date, False,
                         [[env.batch[a]["flex"][ih] for ih in range(0, 2)]
                          for a in self.agents],
                         env.bat.store]
                    env.get_state_vals(inputs=inputs_state_val)
                    sequence_feasible = True
            for e in ["netp0", "discharge_tot0", "charge0"]:
                self.prm["loads"][e] = \
                    [[self.prm["loads"][e][t][a] for t in range(self.N)]
                     for a in self.agents]

        evaluation = eval0

        self.data.p = ""
        self.n_agents = self.prm["ntw"]["n"]
        self.agents = range(self.n_agents)
        # make data for optimisation
        # seed_mult = 1 # for initial passive consumers
        seed_ind = self.ind_seed_deterministic \
            if rl["deterministic"] == 1 \
            else self.data.get_seed_ind(ridx, epoch, i_explore)
        seed_ind += self.data.d_ind_seed[self.data.p]

        [res, _, _, batch], step_vals, mus_opt = \
            self.data.find_feasible_data(
                seed_ind, type_actions, step_vals, evaluation, epoch)

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
                t = type_actions_nonopt[i_t]
                i_t += 1
                self.data.get_seed(seed_ind)
                set_seeds_rdn(self.data.seed[self.data.p])

                # reset environment with adequate data
                E_req_only = True if t == "baseline" else False
                env.reset(seed=self.data.seed[self.data.p],
                          load_data=True, E_req_only=E_req_only)
                # get data from environment
                inputs_state_val = \
                    [0, env.date, False,
                     [[env.batch[a]["flex"][ih] for ih in range(0, 2)]
                      for a in self.agents], env.bat.store]
                state = env.get_state_vals(inputs=inputs_state_val)

                # initialise data for current method
                if t == t0:
                    initt0 += 1
                step_vals[t] = initialise_dict(
                    self.step_vals_entries + self.method_vals_entries)
                vars_env[t] = initialise_dict(self.prm["save"]["last_entries"])
                step, done = 0, 0
                actions = None

                if rl["type_learning"] in ["DDPG", "DQN"] \
                        and rl["trajectory"]:
                    actions, ind_actions, states = \
                        self.action_selector.trajectory_actions(
                            t, rdn_eps_greedy_indiv,
                            eps_greedy, rdn_eps_greedy)
                traj_reward = [0 for _ in self.agents] \
                    if len(t.split("_")) > 1 \
                    and t.split("_")[1] == "d" \
                    and not evaluation else 0

                # loop through steps until either end of sequence
                # or one step if infeasible
                while not done and sequence_feasible:
                    current_state = state
                    action, mu, tf_prev_state \
                        = self.action_selector.select_action(
                            t, step, actions, mus_opt, evaluation,
                            current_state, eps_greedy, rdn_eps_greedy,
                            rdn_eps_greedy_indiv, self.t_env
                        )
                    # interact with environment to get rewards
                    # record last epoch for analysis of results
                    record = epoch == rl["n_epochs"] - 1
                    if action is None \
                            and mu is None \
                            and step == len(res["grid"]) - 1:
                        # no flexiblity so mu value does not matter
                        if rl["type_env"] == "discrete":
                            mu = 0.5 if self.rl["aggregate_actions"] \
                                else [0.5] * self.rl["dim_actions"]
                        else:
                            mu = None

                    # substract baseline rewards to reward -
                    # for training, not evaluating
                    if len(t.split("_")) > 1 \
                            and t.split("_")[1] == "d" \
                            and not evaluation:
                        # for each agent, get rewards
                        # if they acted in the default way
                        # and the others acted the chosen way
                        # without implementing in the environment -
                        # for training, not evaluating
                        rewards_baseline = []
                        combs_actions = []
                        for a in self.agents:
                            # array of actions = the same as chosen
                            # except for agent a the default action
                            mu_actions_baseline_a = action.copy()
                            mu_actions_baseline_a[a] = rl["default_action"][a]
                            combs_actions.append(mu_actions_baseline_a)
                        combs_actions.append(rl["default_action"])
                        for ca in combs_actions:
                            # get outp d
                            [_, _, reward_a, _, _, constraint_ok, _] = \
                                env.step(ca, implement=False,
                                         record=False,
                                         E_req_only=E_req_only)

                            # add penalty if the constraints are violated
                            if not constraint_ok:
                                sequence_feasible = False
                                reward = self._apply_reward_penalty(
                                    evaluation, reward)
                            rewards_baseline.append(reward_a)
                            if reward_a is None:
                                print(f"reward_a {reward_a}")

                    [state, done, reward, break_down_rewards, bool_flex,
                     constraint_ok, record_output] = env.step(
                        action, record=record, mu=mu,
                        evaluation=evaluation, E_req_only=E_req_only)
                    if constraint_ok:
                        traj_reward = self.learning_manager.learning(
                            current_state, state, action, reward,
                            done, t, step, evaluation, traj_reward)
                    if record:
                        self.last_epoch(
                            evaluation, t, record_output, batch, done)
                    if not constraint_ok:
                        sequence_feasible = False
                        reward = self._apply_reward_penalty(evaluation, reward)
                    else:
                        if len(t.split("_")) > 1 \
                                and t.split("_")[1] == "d" \
                                and not evaluation:
                            if rl["competitive"]:
                                reward = [reward[a] - rewards_baseline[a][a]
                                          for a in self.agents]
                            else:
                                if rewards_baseline is None:
                                    print("rewards_baseline is None")
                                reward = [reward - baseline
                                          for baseline in rewards_baseline]
                        if rl["type_env"] == "discrete":
                            global_ind = self.env.spaces.get_global_ind(
                                current_state, state, action, done, t
                            )
                        else:
                            global_ind = {
                                "state": None,
                                "action": None,
                                "next_state": None
                            }

                        step_vals_ = \
                            [current_state, global_ind["state"], action,
                             global_ind["action"], reward, state,
                             global_ind["next_state"], done, bool_flex,
                             constraint_ok, *break_down_rewards]
                        for e, var in zip(self.step_vals_entries, step_vals_):
                            step_vals[t][e].append(var)

                        # if instant feedback,
                        # learn right away at the end of the step
                        self.learning_manager.q_learning_instant_feedback(
                            evaluation, t, step_vals, step
                        )

                        step += 1

                if rl["type_learning"] in ["DDPG", "DQN"] \
                        and rl["trajectory"] \
                        and not evaluation \
                        and t != "baseline":
                    self.learning_manager.trajectory_deep_learn(
                        states, actions, traj_reward, t, evaluation)

            if not sequence_feasible:  # if data is not feasible, make new data
                n_not_feas += 1
                not_feas_vars.append([env.bat.store0, t])
                seed_ind = self.data.infeasible_tidy_files_seeds(seed_ind)

                print("infeasible in loop active")

                self.deterministic_created = False
                print("find feas opt data again!")
                [res, _, _, batch], step_vals, mus_opt = \
                    self.data.find_feasible_data(
                        seed_ind, type_actions, step_vals,
                        evaluation, epoch)

        step_vals["seed"] = self.data.seed[self.data.p]
        step_vals["not_feas_vars"] = not_feas_vars
        step_vals["n_not_feas"] = n_not_feas
        if not evaluation:
            self.t_env += step

        if "opt" in type_actions and evaluation:
            for t in [t for t in type_actions if t != "opt"]:
                if step_vals[t]["reward"][-1] is not None:
                    # rewards should not be better than optimal rewards
                    assert np.mean(step_vals[t]["reward"]) \
                           < np.mean(step_vals["opt"]["reward"]) + 1e-3, \
                           f"reward {t} better than opt"
        if rl["type_learning"] != "facmac":
            self.episode_batch = None
        return step_vals, self.episode_batch

    def _opt_step_init(
            self, i_step, batchflex_opt, cluss, fs, batch_avail_EV, res
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
        home = {
            "gen": np.array(
                [self.prm["ntw"]["gen"][a][i_step] for a in self.agents]
            )
        }

        step_vals_i["state"] = self._opt_step_to_state(
            res, i_step, cluss, fs, loads_prev, loads_step, batch_avail_EV)

        self.env.heat.E_heat_min_max(i_step)
        self.env.heat.potential_E_flex()

        return step_vals_i, date, loads, loads_step, loads_prev, home

    def _get_ind_global_state_action(self, step_vals_i):
        mu_action = step_vals_i["action"]
        if self.prm["RL"]["type_env"] == "discrete":
            ind_state = self.env.spaces.get_space_indexes(
                all_vals=step_vals_i["state"])
            step_vals_i["ind_global_state"] = \
                [self.env.spaces.indiv_to_global_index(
                    "state", indexes=ind_state,
                    multipliers=self.global_multipliers["state"])]
            ind_action = self.env.spaces.get_space_indexes(
                all_vals=mu_action, type_="action")
            for a in self.agents:
                assert not (ind_action is None and mu_action[a] is not None), \
                    f"action[{a}] {step_vals_i['action'][a]} " \
                    f"is none whereas mu {mu_action[a]} is not"
            step_vals_i["ind_global_action"] = \
                [self.env.spaces.indiv_to_global_index(
                    "action", indexes=ind_action,
                    multipliers=self.global_multipliers["action"])]
        else:
            step_vals_i["ind_global_state"] = None
            step_vals_i["ind_global_action"] = None

        return step_vals_i

    def _get_passive_vars(self, i_step):
        passive_vars = \
            [[self.prm["loads"][e][a][i_step]
              for a in range(self.prm["ntw"]["nP"])]
             for e in ["netp0", "discharge_tot0", "charge0"]]

        return passive_vars

    def _get_diff_rewards(
            self, evaluation, i_step, mu_action, date,
            loads, res, feasible, reward
    ):
        obtain_diff_reward = any(
            len(q.split("_")) >= 2
            and q.split("_")[1] == "d"
            for q in self.prm["RL"]["type_Qs"]
        )
        if obtain_diff_reward and not evaluation:
            rewards_baseline, feasible_getting_baseline = \
                self._get_artificial_baseline_reward_opt(
                    i_step, mu_action, date, loads, res, evaluation
                )
            if not feasible_getting_baseline:
                feasible = False
            if self.prm["RL"]["competitive"]:
                reward_diff = \
                    [reward[a] - rewards_baseline[a][a]
                     for a in self.agents]
            else:
                reward_diff = \
                    [reward - reward_baseline
                     for reward_baseline in rewards_baseline]
        else:
            reward_diff = None

        return reward_diff, feasible

    def _append_step_vals(
            self, t, step_vals_i, res, i_step, cluss, fs,
            loads_prev, loads_step, batch_avail_EV, step_vals,
            break_down_rewards, feasible
    ):
        keys = self.break_down_rewards_entries + ["constraint_ok"]
        vars = break_down_rewards + [feasible]
        for key_, var in zip(keys, vars):
            step_vals_i[key_] = var

        keys = ["state", "action", "reward", "reward_diff", "bool_flex",
                "constraint_ok", "ind_global_action",
                "ind_global_state"]
        for key_ in keys:
            step_vals[t][key_].append(step_vals_i[key_])

        if i_step > 0:
            step_vals[t]["next_state"].append(step_vals_i["state"])
            if self.prm["RL"]["type_env"] == "discrete":
                step_vals[t]["ind_next_global_state"].append(
                    step_vals_i["ind_global_state"])
        if i_step == len(res["grid"]) - 1:
            step_vals[t]["next_state"].append(self._opt_step_to_state(
                res, i_step + 1, cluss, fs, loads_prev,
                loads_step, batch_avail_EV))
            if self.prm["RL"]["type_env"] == "discrete":
                ind_next_state = self.env.spaces.get_space_indexes(
                    all_vals=step_vals[t]["next_state"][-1])
                step_vals[t]["ind_next_global_state"].append(
                    self.env.spaces.indiv_to_global_index(
                        "state", indexes=ind_next_state,
                        multipliers=self.global_multipliers["state"]))

        step_vals[t]["done"].append(
            False if i_step <= len(res["grid"]) - 2 else True)

        return step_vals

    def _tests_res(self, res, i_step, batchflex_opt, reward):
        # check tot cons
        for a in self.agents:
            assert res["totcons"][a][i_step] <= \
                   sum(batchflex_opt[a][i_step]) \
                   + self.env.heat.E_heat_min[a] \
                   + self.env.heat.potential_E_flex()[a] + 1e-3, \
                   f"cons more than sum fixed + flex!, " \
                   f"a = {a}, i_step = {i_step}"

        # check reward from environment and res variables match
        prm = self.prm
        res_reward_t = \
            - (prm["grd"]["C"][i_step]
               * (res["grid"][i_step][0]
                  + prm["grd"]["R"] / (prm["grd"]["V"] ** 2)
                  * res["grid2"][i_step][0])
               + prm["bat"]["C"]
               * sum(res["discharge_tot"][a][i_step]
                     + res["charge"][a][i_step]
                     for a in self.agents)
               + prm["ntw"]["C"]
               * sum(res["netp_abs"][a][i_step]
                     for a in self.agents))
        if not prm["RL"]["competitive"]:
            assert abs(reward - res_reward_t) < 1e-3, \
                f"reward {reward} != res_reward_t " \
                f"from res variables {res_reward_t}"

    def _instant_feedback_steps_opt(
            self, evaluation, t, i_step, step_vals
    ):
        rl = self.prm["RL"]
        if (rl["type_learning"] in ["DQN", "DDQN", "DDPG", "facmac"]
            or rl["instant_feedback"]) \
                and not evaluation \
                and t in rl["type_explo"] \
                and i_step > 0 \
                and not rl["trajectory"]:
            current_state, actions, reward, state, reward_diffs = [
                step_vals["opt"][e][-1]
                for e in ["state", "action", "reward",
                          "next_state", "reward_diff"]]
            if rl["type_learning"] == "q_learning":
                # learner agent learns from this step
                self.learning_manager.learner.learn(
                    "opt", step_vals[t], i_step - 1
                )
            elif rl["type_learning"] == "facmac":
                pre_transition_data = {
                    "state": [current_state[a][0]
                              for a in self.agents],
                    "avail_actions": [rl["avail_actions"]],
                    "obs": [np.reshape(
                        current_state, (self.n_agents, rl["obs_shape"]))]
                }
                self.episode_batch[t].update(
                    pre_transition_data, ts=i_step)
                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(i_step == self.prm["syst"]["N"] - 1,)],
                }
                self.episode_batch[t].update(
                    post_transition_data, ts=i_step)

            elif rl["type_learning"] in ["DDPG", "DQN", "DDQN"]:
                self.learning_manager.independent_deep_learning(
                    current_state, actions, reward, state, reward_diffs)

    def _update_flexibility_opt(self, batchflex_opt, res, i_step):
        cons_flex_opt = \
            [res["totcons"][a][i_step] - batchflex_opt[a][i_step][0]
             - res["E_heat"][a][i_step] for a in self.agents]
        inputs_update_flex = \
            [i_step, batchflex_opt, self.prm["loads"]["max_delay"],
             self.n_agents]
        new_batch_flex = self.env.update_flex(
            cons_flex_opt, opts=inputs_update_flex)
        for a in self.agents:
            batchflex_opt[a][i_step: i_step + 2] = new_batch_flex[a]

        assert batchflex_opt is not None, "batchflex_opt is None"

        return batchflex_opt

    def _test_total_rewards_match(self, evaluation, res, sum_rewards):
        sum_res_rewards = (- (res["gc"] + res["sc"] + res["dc"]))
        if not (self.prm["RL"]["competitive"] and not evaluation):
            assert abs(sum_rewards - sum_res_rewards) < 5e-3, \
                "tot rewards dont match: "\
                f"sum_rewards = {sum_rewards}, "
            f"sum costs opt = {- (res['gc'] + res['sc'] + res['dc'])}"

    def get_steps_opt(self, res, step_vals, evaluation, cluss,
                      fs, batch, seed, last_epoch=False):
        """Translate optimisation results to states, actions, rewards."""
        env, rl = self.env, self.prm["RL"]
        feasible = True
        t = "opt"
        sum_rewards = 0
        all_mus = []
        step_vals[t] = initialise_dict(
            self.step_vals_entries + ["reward_diff"])
        batchflex_opt, batch_avail_EV = \
            [[batch[a][e] for a in range(len(batch))]
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
                i_step, batchflex_opt, cluss, fs, batch_avail_EV, res
            )

            # translate dp into mu value
            step_vals_i["bool_flex"], step_vals_i["action"], error = \
                env.mu_manager.netp_to_mu(
                    i_step, date, res["netp"][:, i_step],
                    loads, home, res)
            self._get_ind_global_state_action(step_vals_i)
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
            self._tests_res(res, i_step, batchflex_opt, step_vals_i["reward"])

            # substract baseline rewards to reward -
            # for training, not evaluating
            step_vals_i["reward_diff"], feasible = self._get_diff_rewards(
                evaluation, i_step, step_vals_i["action"], date,
                loads, res, feasible, step_vals_i["reward"])
            if not feasible:
                step_vals_i["reward"] = self._apply_reward_penalty(
                    evaluation, step_vals_i["reward"])
            if not (rl["competitive"] and not evaluation):
                sum_rewards += step_vals_i["reward"]

            # append experience dictionaries
            step_vals = self._append_step_vals(
                t, step_vals_i, res, i_step, cluss, fs,
                loads_prev, loads_step, batch_avail_EV, step_vals,
                break_down_rewards, feasible
            )

            # update flexibility table
            batchflex_opt = self._update_flexibility_opt(
                batchflex_opt, res, i_step
            )

            # instant learning feedback
            self._instant_feedback_steps_opt(
                evaluation, t, i_step, step_vals
            )

            # update battery and heat objects
            self.env.bat.update_step(res)
            self.env.heat.update_step(res)

            # record if last epoch
            self._record_last_epoch_opt(
                res, i_step, break_down_rewards, batchflex_opt,
                last_epoch, step_vals_i, batch, evaluation
            )

        self._test_total_rewards_match(evaluation, res, sum_rewards)

        if not evaluation \
                and rl["type_learning"] in ["DDPG", "DQN"] \
                and rl["trajectory"]:
            self.learning_manager.learn_trajectory_opt()

        return step_vals, all_mus, feasible

    def _record_last_epoch_opt(
            self, res, i_step, break_down_rewards, batchflex_opt,
            last_epoch, step_vals_i, batch, evaluation
    ):
        if not last_epoch:
            return

        done = i_step == self.prm["syst"]["N"] - 1
        ldflex = [0 for _ in self.agents] \
            if done \
            else [sum(batchflex_opt[a][i_step][1:])
                  for a in self.agents]
        if done:
            ldfixed = [sum(batchflex_opt[a][i_step][:])
                       for a in self.agents]
        else:
            ldfixed = [batchflex_opt[a][i_step][0]
                       for a in self.agents]
        tot_cons_loads = \
            [res["totcons"][a][i_step] - res["E_heat"][a][i_step]
             for a in self.agents]
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

    def _apply_reward_penalty(self, evaluation, reward):
        if self.rl["apply_penalty"] and not evaluation:
            if self.rl["competitive"]:
                for a in self.agents:
                    reward[a] -= self.rl["penalty"]
            else:
                reward -= self.rl["penalty"]
        return reward

    def _fixed_flex_loads(self, i_step, batchflex_opt):
        """
        Get fixed and flexible consumption equivalent to optimtisation results.

        Obtain total fixed and flexible loads for each agent
        for a given time step based on current optimisation results
        """
        # note that we could also obtain the fixed cons / flexible
        # load as below,
        # however we want to count it consistently with our
        # batchflex_opt updates:
        # l_fixed = [ntw['dem'][0, a, i_step] for a in range(n_agents)]
        # flex_load = [ntw['dem'][1, a, i_step] for a in range(n_agents)]

        if i_step == self.prm["syst"]["N"] - 1:
            flex_load = np.zeros(self.n_agents)
            l_fixed = np.array(
                [sum(batchflex_opt[a][i_step][:]) for a in self.agents]
            )
        else:
            flex_load = np.array(
                [sum(batchflex_opt[a][i_step][1:]) for a in self.agents]
            )
            l_fixed = np.array(
                [batchflex_opt[a][i_step][0] for a in self.agents]
            )

        loads_step = l_fixed + flex_load

        return flex_load, l_fixed, loads_step

    def _get_artificial_baseline_reward_opt(self,
                                            i_step,
                                            mu_actions,
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
        gens = [prm["ntw"]["gen"][a][i_step] for a in self.agents]
        self.env.heat.T = [res["T"][a][i_step] for a in self.agents]
        self.env.bat.store = \
            [res["store"][a][i_step] for a in self.agents]
        combs_actions = []
        for a in self.agents:
            mu_actions_baseline_a = mu_actions.copy()
            mu_actions_baseline_a[a] = \
                [1 for _ in range(self.prm["RL"]["dim_actions"])]
            combs_actions.append(mu_actions_baseline_a)
        combs_actions.append([[1 for _ in range(self.prm["RL"]["dim_actions"])]
                              for _ in self.agents])
        feasible = True
        for a in self.agents:
            T_air = res["T_air"][a][i_step]
            if T_air < self.env.heat.T_LB[a][i_step] - 1e-1 \
                    or T_air > self.env.heat.T_UB[a][i_step] + 1e-1:
                print(f"a {a} i_step {i_step} "
                      f"res['T_air'][a][i_step] {T_air} "
                      f"T_LB[a] {self.env.heat.T_LB[a][i_step]} "
                      f"T_UB[a] {self.env.heat.T_UB[a][i_step]}")
        for ca in combs_actions:
            bat_store = self.env.bat.store.copy()
            input_take_action = date, ca, gens, loads
            home, loads, constraint_ok = env.policy_to_rewardvar(
                None, other_input=input_take_action)
            self.env.bat.store = bat_store
            passive_vars = self._get_passive_vars(i_step)

            reward_baseline_a, _ = env.get_reward(
                home["netp"], self.env.bat.discharge_tot, self.env.bat.charge,
                i_step=i_step, passive_vars=passive_vars,
                evaluation=evaluation)

            if not constraint_ok:
                feasible = False
                print(f"self.data.seed = {self.data.seed} "
                      f"constraint_ok False, i_step {i_step}")
                self._apply_reward_penalty(evaluation, reward_baseline_a)

            rewards_baseline.append(reward_baseline_a)

            # revert back store
            self.env.bat.store = [res["store"][a][i_step]
                                  for a in self.agents]

        return rewards_baseline, feasible

    def _opt_step_to_state(self,
                           res: dict,
                           i_step: int,
                           cluss: list,
                           fs: list,
                           loads_prev: list,
                           loads_step: list,
                           batch_avail_EV: np.ndarray
                           ) -> list:
        """
        Get state descriptor values.

        Get values corresponding to state descriptors specified,
        based on optimisation results.
        """
        vals = []
        date = self.prm["syst"]["current_date0"] + \
            timedelta(hours=i_step)
        for a in self.agents:
            vals_a = []
            state_vals = {
                None: None,
                "hour": i_step % 24,
                "grdC": self.prm["grd"]["Call"][self.i0_costs + i_step],
                "day_type": 0 if date.weekday() < 5 else 1,
                "loads_cons_step": loads_step,
                "loads_cons_prev": loads_prev,
                "dT": self.prm["heat"]["T_req"][a][i_step]
                - res["T_air"][a][min(i_step, len(res["T_air"][a]) - 1)]
            }

            for descriptor in self.descriptors["state"]:
                if descriptor in state_vals:
                    val = state_vals[descriptor][a] \
                        if type(state_vals[descriptor]) is list \
                        else state_vals[descriptor]
                elif descriptor in self.state_funcs:
                    inputs = i_step, res, a, date
                    val = self.state_funcs[descriptor](inputs)

                elif len(descriptor) > 9 \
                        and (descriptor[-9:-5] == "fact"
                             or descriptor[-9:-5] == "clus"):
                    # scaling factors / profile clusters for the whole day
                    day = (date - self.prm["syst"]["current_date0"]).days
                    module = descriptor.split("_")[0]  # EV, loads or gen
                    index_day = day - \
                        1 if descriptor.split("_")[-1] == "prev" else day
                    index_day = max(index_day, 0)
                    data = fs if descriptor[-9:-5] == "fact" else cluss
                    val = data[a][module][index_day]
                else:  # select current or previous hour - step or prev
                    i_step_val = i_step if descriptor[-4:] == "step" \
                        else i_step - 1
                    if i_step_val < 0:
                        i_step_val = 0
                    if len(descriptor) > 8 and descriptor[0:8] == "avail_EV":
                        if i_step_val < len(batch_avail_EV[0]):
                            val = batch_avail_EV[a][i_step_val]
                        else:
                            val = 1
                    elif descriptor[0:3] == "gen":
                        val = self.prm["ntw"]["gen"][a][i_step_val]
                    else:  # remaining are EV_cons_step / prev
                        val = self.prm["bat"]["batch_loads_EV"][a][i_step]
                vals_a.append(val)
            vals.append(vals_a)

        assert np.shape(vals) \
               == (self.n_agents, len(self.descriptors["state"])), \
               f"np.shape(vals) {np.shape(vals)} " \
               f"self.n_agents {self.n_agents} " \
               f"len descriptors['state'] {len(self.descriptors['state'])}"

        return vals

    def _set_eps_greedy_vars(self, rl, epoch, evaluation):
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

    def _get_dT_next(self, inputs):
        i_step, _, a, _ = inputs
        T_req = self.prm["heat"]["T_req"][a]
        t_next = [t for t in range(i_step + 1, self.N)
                  if T_req[t] != T_req[i_step]]
        if not t_next:
            val = 0
        else:
            val = (T_req[t_next[0]] - T_req[i_step]) \
                / (t_next[0] - i_step)

        return val

    def _get_EV_tau(self, inputs):
        i_step, res, a, date = inputs

        loads_T, deltaT, _ = \
            self.env.bat.next_trip_details(i_step, date, a)

        if loads_T is not None and deltaT > 0:
            val = ((loads_T - res["store"][a][i_step]) / deltaT)
        else:
            val = - 1

        return val

    def _get_store(self, inputs):
        i_step, res, a, _ = inputs
        if i_step < len(res["store"][a]):
            val = res["store"][a][i_step]
        else:
            val = self.prm["bat"]["store0"][a]

        return val

    def _get_grdC_level(self, inputs):
        i_step = inputs[0]
        costs = self.prm["grd"]["Call"][self.i0_costs:
                                        self.i0_costs + self.N + 1]
        val = (costs[i_step] - min(costs)) \
            / (max(costs) - min(costs))

        return val

    def _get_bat_dem_agg(self, inputs):
        i_step, _, a, _ = inputs
        val = self.prm["bat"]["bat_dem_agg"][a][i_step]

        return val

    def _init_facmac_mac(self, type_actions, new_episode_batch):
        if self.rl["type_learning"] == "facmac":
            for t in type_actions:
                self.episode_batch[t] = new_episode_batch()
                if t not in ["baseline", "opt"]:
                    self.action_selector.mac[t].init_hidden(
                        batch_size=self.rl["batch_size_run"]
                    )
