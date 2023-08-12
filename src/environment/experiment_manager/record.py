#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 16:53:03 2021.

@author: floracharbonnier

record - keeping record of learning trajectories
"""
import copy
# packages
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy as sp

from src.environment.utilities.userdeftools import (get_moving_average,
                                                    initialise_dict)


class Record:
    """Record - keeping record of learning trajectories."""

    def __init__(self, prm: dict, no_run: int = None):
        """Add relevant properties from prm to the object."""
        self.no_run = no_run
        for attribute in ['n_homes', 'n_homes_test', 'H', 'N', 'interval_to_month']:
            setattr(self, attribute, prm['syst'][attribute])

        self._add_rl_info_to_object(prm["RL"])

        for attribute in ['manage_voltage', 'manage_agg_power', 'compare_pandapower_optimisation']:
            setattr(self, attribute, prm['grd'][attribute])

        self._intialise_dictionaries_entries_to_record(prm)

        prm["paths"]["save_days"] = prm["paths"]["folder_run"] / "save_days"
        self.record_folder = prm["paths"]["record_folder"]
        for folder in ["folder_run", "record_folder", "save_days", "fig_folder"]:
            if not os.path.exists(prm["paths"][folder]):
                os.mkdir(prm["paths"][folder])
        self.repeat = 0  # current repeat

    def _intialise_dictionaries_entries_to_record(self, prm):
        # initialise entries
        # entries that change for each repeat
        self.break_down_rewards_entries = prm["syst"]["break_down_rewards_entries"]
        self.repeat_entries = prm["save"]["repeat_entries"] + self.break_down_rewards_entries
        # entries that change for run but are the same across repeats
        self.run_entries = \
            prm["save"]["run_entries0"] \
            + prm["save"]["pandapower_voltage_entries"] \
            + prm['save']['discrete_states_info_entries']
        self.discrete_states_info_entries = prm['save']['discrete_states_info_entries']
        self.last_entries = prm["save"]["last_entries"]
        for entry in self.repeat_entries:
            setattr(self, entry, initialise_dict(range(self.n_repeats)))

    def _add_rl_info_to_object(self, rl):
        # all exploration / evaluation methods
        for info in [
            "n_epochs", "instant_feedback", "type_env", "n_repeats", "state_space",
            "n_explore", "n_all_epochs", "evaluation_methods",
            'dim_actions_1', 'dim_states_1'
        ]:
            setattr(self, info, rl[info])
        self.all_methods = rl["evaluation_methods"] \
            + list(set(rl["exploration_methods"]) - set(rl["evaluation_methods"]))
        if rl['supervised_loss']:
            self.all_methods.append('opt')

        # depending on the dimension of the q tables
        self.save_qtables = True \
            if (self.state_space is None or len(self.state_space) <= 2) and self.n_epochs <= 1e3 \
            else False

        for info in ["gamma", "epsilon_decay"]:
            setattr(
                self,
                info,
                rl[rl["type_learning"]][info] if info in rl[rl['type_learning']] else None
            )

    def init_env(self, env: object):
        """
        Add relevant properties computed in the environment to record.

        Do this after the environment is initialised.
        """
        if self.type_env == "discrete":
            self.possible_states = int(env.spaces.possible["state"][-1]) + 1\
                if int(env.spaces.possible["state"][-1]) > 0\
                else 1
            self.granularity_state0 = None if self.state_space == [
                None] else env.spaces.granularity["state"][0]
            self.granularity_state1 = None if len(
                self.state_space) == 1 else env.spaces.granularity["state"][1]
            self.multipliers_state = env.spaces.multipliers["state"]
        else:
            for entry in self.discrete_states_info_entries:
                setattr(self, entry, None)

    def new_repeat(self,
                   repeat: int,
                   rl: dict,
                   ):
        """Reinitialise object properties at the start of new repeat."""
        self.repeat = repeat  # current repetition index

        for field in ["q_tables", "counter"]:
            if self.save_qtables:
                self.__dict__[field][repeat] = initialise_dict(range(rl["n_epochs"]))
            else:
                self.__dict__[field][repeat] = {}
        for info in ["duration_epoch", "eps", "seed", "n_not_feas"]:
            self.__dict__[info][repeat] = np.zeros(self.n_all_epochs)
        self.train_rewards[repeat] = {
            method: np.zeros((self.n_epochs, self.n_explore * self.N))
            for method in rl["exploration_methods"]
        }
        if rl["type_learning"] == "DQN" and rl["distr_learning"] == "decentralised":
            for method in rl["evaluation_methods"]:
                self.eps[repeat][method] = np.zeros((self.n_all_epochs, self.n_homes))

        all_evaluation_methods = \
            rl["evaluation_methods"] + ['opt'] if rl["supervised_loss"] \
            else rl["evaluation_methods"]
        self.eval_rewards[repeat] = {
            method: np.zeros((self.n_all_epochs, self.N)) for method in all_evaluation_methods
        }
        for reward in ["mean_eval_rewards"] + self.break_down_rewards_entries:
            shape = (self.n_all_epochs, self.n_homes_test) if reward[0: len('indiv')] == 'indiv' \
                else (self.n_all_epochs)
            self.__dict__[reward][repeat] = {
                method: np.zeros(shape) for method in all_evaluation_methods
            }
        self.eval_actions[repeat] = {
            method: np.zeros((self.n_all_epochs, self.N, self.n_homes_test, self.dim_actions_1))
            for method in all_evaluation_methods
        }
        self.train_actions[repeat] = {
            method:
                np.zeros((self.n_epochs, self.n_explore * self.N, self.n_homes, self.dim_actions_1))
            for method in rl["exploration_methods"]
        }
        self.train_states[repeat] = {
            method:
                np.zeros((self.n_epochs, self.n_explore * self.N, self.n_homes, self.dim_states_1))
            for method in rl["exploration_methods"]
        }
        self.stability[repeat] = {method: None for method in rl["evaluation_methods"]}

        self.last[repeat] = initialise_dict(self.last_entries + ["batch"], "empty_dict")
        for e in self.last_entries:
            self.last[repeat][e] = initialise_dict(self.all_methods)

    def end_epoch(self,
                  epoch: int,
                  eval_steps: dict,
                  list_train_stepvals: dict,
                  rl: dict,
                  learner: object,
                  duration_epoch: float,
                  end_test: bool = False
                  ):
        """At the end of each epoch, append training or evaluation record."""
        if list_train_stepvals is not None:
            for method in rl["exploration_methods"]:
                self.train_rewards[self.repeat][method][epoch] \
                    = list_train_stepvals[method]["reward"]
                self.train_actions[self.repeat][method][epoch] \
                    = list_train_stepvals[method]["action"]
                self.train_states[self.repeat][method][epoch] = list_train_stepvals[method]["state"]

        for method in rl["evaluation_methods"]:
            self._append_eval(eval_steps, method, epoch, end_test)

        if rl["type_learning"] == "q_learning" and not end_test:
            if self.save_qtables:
                for table in ["q_tables", "counter"]:
                    self.__dict__[table][self.repeat][epoch] = copy.deepcopy(
                        getattr(learner, table)
                    )
            elif epoch == self.n_epochs - 1:
                for table in ["q_tables", "counter"]:
                    self.__dict__[table][self.repeat] = copy.deepcopy(learner.__dict__[table])

        self._update_eps(rl, learner, epoch, end_test)

        self.duration_epoch[self.repeat][epoch] = duration_epoch

    def last_epoch(self, evaluation, method, record_output, batch, done):
        """Record more information for the final epoch in self.last."""
        type_step = "eval" if evaluation else "train"
        if type_step == "eval":
            for output, label in zip(record_output, self.last_entries):
                self.last[self.repeat][label][method].append(output)
            if done and method == "baseline":
                self.last[self.repeat]["batch"] = batch

    def timer_stats(
        self, timer_pp, timer_comparison, timer_optimisation, timer_feasible_data
    ):
        """
        Calculates the mean, standard deviation and count of
        the timer used to evaluate the computational burden
        of some functions and methods.
        """
        list_timer_attributes = [
            (self.manage_voltage or self.manage_agg_power, timer_pp, 'timer_pp'),
            (self.manage_voltage and self.compare_pandapower_optimisation,
                timer_comparison, 'timer_comparison'),
            (True, timer_optimisation, 'timer_optimisation'),
            (True, timer_feasible_data, 'timer_feasible_data'),
        ]

        for condition, timer, timer_name in list_timer_attributes:
            if condition and len(timer) != 0:
                timer_mean = np.mean(timer)
                timer_std = np.std(timer)
                timer_count = len(timer)
            else:
                timer_mean = 0
                timer_std = 0
                timer_count = 0

            setattr(self, f"{timer_name}_mean", timer_mean)
            setattr(self, f"{timer_name}_std", timer_std)
            setattr(self, f"{timer_name}_count", timer_count)

    def save(self, end_of: str = "repeat"):
        """
        Save the relevant record object information to files.

        Save the relevant information collected during
        the explorations and evaluations
        """
        labels = self.repeat_entries if end_of == "repeat" \
            else self.run_entries
        if not self.save_qtables:
            labels = [label for label in labels if label not in ["q_tables", "counter"]] \
                if end_of == "repeat" \
                else labels + ["q_tables", "counter"]
        repeat = self.repeat if end_of == "repeat" else None
        for label in labels:
            save_path = Path(f"{label}" if repeat is None else f"{label}_repeat{repeat}")
            if self.record_folder is not None:
                save_path = Path(self.record_folder) / save_path
            to_save = getattr(self, label) if end_of == "end" \
                else getattr(self, label)[self.repeat]
            np.save(save_path, to_save)

            del to_save

    def load(self, prm: dict):
        """List files to load for record object & call loading_file method."""
        repeat_labels = \
            [info for info in self.repeat_entries if info not in ["q_tables", "counter"]] \
            if not self.save_qtables \
            else self.repeat_entries
        run_labels = self.run_entries
        if prm['RL']['type_learning'] == 'q_learning' and not self.save_qtables:
            run_labels += ["q_tables", "counter"]

        for label in repeat_labels:
            for repeat in range(prm["RL"]["n_repeats"]):
                self._loading_file(label, repeat)
        for label in run_labels:
            self._loading_file(label)

    def _loading_file(self, label: str, repeat: int = None):
        """
        Load file and add to record.

        Given instruction for specific file to load by load method,
        """
        str_ = f"{label}" if repeat is None else f"{label}_repeat{repeat}"
        str_ = os.path.join(self.record_folder, str_ + ".npy")
        obj = np.load(str_, allow_pickle=True)
        if len(np.shape(obj)) == 0:
            obj = obj.item()
        if repeat is not None:
            self.__dict__[label][repeat] = obj
        else:
            setattr(self, label, obj)

    def results_to_percentiles(
        self,
        method,
        prm,
        mov_average=False,
        n_window=None,
        baseline='baseline'
    ):
        """For each epoch, percentiles of evaluation across repeats."""
        p_vals = [25, 50, 75]
        percentiles = {p: np.zeros(self.n_all_epochs) for p in p_vals}
        n_repeats = prm["RL"]["n_repeats"]
        for epoch in range(self.n_all_epochs):
            epoch_rewards = self.monthly_mean_eval_rewards_per_home[method][:, epoch]
            epoch_baseline_rewards = self.monthly_mean_eval_rewards_per_home[baseline][:, epoch]
            diff_repeats = [
                epoch_rewards[repeat] - epoch_baseline_rewards[repeat]
                for repeat in range(n_repeats)
                if epoch_rewards[repeat] is not None and epoch_baseline_rewards[repeat] is not None
            ]
            for p in [25, 50, 75]:
                percentiles[p][epoch] = \
                    np.nan if len(diff_repeats) == 0 else np.percentile(diff_repeats, p)
        if mov_average:
            for p in [25, 50, 75]:
                percentiles[p] = get_moving_average(percentiles[p], n_window, Nones=False)

        p25, p50, p75 = [percentiles[p] for p in p_vals]
        not_nan = ~np.isnan(p25)
        epoch_not_nan = np.arange(self.n_all_epochs)[not_nan]
        p25_not_nan, p75_not_nan = p25[not_nan], p75[not_nan]

        return p25, p50, p75, p25_not_nan, p75_not_nan, epoch_not_nan

    def get_mean_rewards(
        self,
        prm,
        action_state_space_0,
        state_space_0,
        evaluation_methods_plot
    ):
        """
        Get mean reward per home over final learning / over the testing period.

        monthly_mean_eval_rewards_per_home[repeat][e][epoch]:
            for each method, repeat and epoch, the average eval rewards per month
            over the whole evaluation episode
            (mean_eval_rewards[repeat][e][epoch]),
            divided by the number of homes

        monthly_mean_end_eval_rewards_per_home[repeat][e]:
            the average monthly_mean_eval_rewards_per_home over the end of the training,
            from start_end_eval -> n_epochs

        monthly_mean_test_rewards_per_home:
            the average monthly_mean_eval_rewards_per_home after the end of the training,
            from n_epochs onwards during the fixed policy, test only period.
        """
        self.monthly_mean_eval_rewards_per_home = {
            method: np.zeros((self.n_repeats, self.n_all_epochs))
            for method in self.evaluation_methods
        }
        for reward in [
            'monthly_mean_end_eval_rewards_per_home', 'monthly_mean_test_rewards_per_home'
        ]:
            setattr(
                self, reward, {
                    method: np.zeros(self.n_repeats) for method in self.evaluation_methods
                }
            )

        for repeat in range(prm["RL"]["n_repeats"]):  # loop through repetitions
            action_state_space_0[repeat], state_space_0[repeat] = \
                [initialise_dict(range(prm["syst"]["n_homes"])) for _ in range(2)]
            # 1 - mean rewards
            if "end_decay" not in prm["RL"] or "DQN" not in prm["RL"]:
                for type_learning in ["DQN", "DDQN", "q_learning"]:
                    prm["RL"][type_learning]["end_decay"] \
                        = self.n_epochs

            keys_ = self.mean_eval_rewards[repeat].keys()
            for method in evaluation_methods_plot.copy():
                if method not in keys_:
                    if (
                            prm["RL"]["type_learning"] == "facmac"
                            and any(
                                key[0: len(method[:-2])] == method[:-2]
                                for key in keys_
                            )
                    ):
                        new = [
                            key for key in keys_
                            if key[0: len(method[:-2])] == method[:-2]
                        ][0]
                        evaluation_methods_plot.remove(method)
                        evaluation_methods_plot.append(new)
                    else:
                        evaluation_methods_plot.remove(method)

            for method in evaluation_methods_plot:
                self.monthly_mean_eval_rewards_per_home[method][repeat] = np.where(
                    self.mean_eval_rewards[repeat][method] is None,
                    None,
                    self.mean_eval_rewards[repeat][method] / prm['syst']['n_homes_all_test']
                    * self.interval_to_month
                )
                self.monthly_mean_end_eval_rewards_per_home[method][repeat] = np.mean(
                    self.monthly_mean_eval_rewards_per_home[method][repeat][
                        prm["RL"]["start_end_eval"]: self.n_epochs
                    ]
                )
                self.monthly_mean_test_rewards_per_home[method][repeat] = np.mean(
                    self.monthly_mean_eval_rewards_per_home[method][repeat][self.n_epochs:]
                )

    def compute_IQR_and_CVaR_repeat(self, monthly_mean_eval_rewards_per_home, all_nans):
        detrended_rewards = [
            monthly_mean_eval_rewards_per_home[epoch]
            - monthly_mean_eval_rewards_per_home[epoch - 1]
            if sum(
                monthly_mean_eval_rewards_per_home[epoch_] is None
                for epoch_ in [epoch, epoch - 1]
            ) == 0
            else None
            for epoch in range(1, self.n_epochs)
        ]
        detrended_rewards_notNone = [d for d in detrended_rewards if d is not None]
        IQR_repeat = sp.stats.iqr(detrended_rewards_notNone) if not all_nans else None
        CVaR_repeat = np.mean(
            [
                dr for dr in detrended_rewards_notNone
                if dr <= np.percentile(detrended_rewards_notNone, 5)
            ]
        ) if not all_nans else None

        return IQR_repeat, CVaR_repeat

    def compute_largest_drawdown_repeat(self, monthly_mean_eval_rewards_per_home, best_eval):
        largest_drawdown = - 1e6
        epochs = [
            epoch for epoch in range(1, self.n_epochs)
            if monthly_mean_eval_rewards_per_home[epoch] is not None
        ]
        for epoch in epochs:
            drawdown \
                = best_eval - monthly_mean_eval_rewards_per_home[epoch]
            if drawdown > largest_drawdown:
                largest_drawdown = drawdown
            if monthly_mean_eval_rewards_per_home[epoch] > best_eval:
                best_eval = monthly_mean_eval_rewards_per_home[epoch]
            assert largest_drawdown is not None, \
                "largest_drawdown is None"

        return largest_drawdown

    def get_metrics(
            self,
            prm: dict,
            evaluation_methods_plot: list
    ) -> Tuple[dict, list]:
        """
        Get results metrics: measures of performance and risk.

        end:
            final evaluation rewards between start_end_eval -> n_epochs
        end_bl:
            final differences between evaluation rewards and baseline
            evaluation rewards between start_end_eval -> n_epochs
        end_test:
            evaluation rewards from n_epochs onwards (once we stop learning)
        end_test_bl:
            differences between evaluation rewards nd baseline evaluation
            rewards from n_epochs onwards (once we stop learning)
        mean: all rewards

        And from: Stephanie CY Chan, Sam Fishman, John Canny,
        Anoop Korattikara, and Sergio Guadarrama.
        Measuring the reliability of reinforcement learning algorithms.
        International Conference on Learning Representations, 2020.
        arXiv:1912.05663v2
        DT:
            Dispersion across Time (DT): IQR across Time of the detrended
            rewards (differences between subsequent rewards)
            shows higher-frequency variability
        SRT:
            Short-term Risk across Time (SRT): CVaR on Differences
            the most extreme short-term drop over time.
            CVaR to the changes in performance from one evaluation point to the
            next = The expected value of the distribution below the α-quantile.
        LRT:
            Long-term Risk across Time (LRT): CVaR on Drawdown
            whether an algorithm has the potential to lose a lot of performance
            relative to its peak, even if on a longer timescale,
            e.g. over an accumulation of small drops.
            Apply CVaR to the Drawdown = the drop in performance relative
            to the highest peak so far.
        DR:
            Dispersion across Runs (DR): IQR across Runs
            IQR across training runs at a set of evaluation points. We
        RR:
            Risk across Runs (RR): CVaR across Runs
            Apply CVaR to the final performance of all the training runs.

        For each of these we compute the average,
        standard deviation, and 25th, 50th and 75th percentiles.
        """
        n_repeats = prm["RL"]["n_repeats"]
        metric_entries = [
            "end", "end_test", "end_bl", "end_test_bl",
            "mean", "DT", "SRT", "LRT", "DR", "RR"
        ]
        subentries = ["ave", "std", "p25", "p75", "p50"]
        metrics = initialise_dict(
            metric_entries, "empty_dict",
            second_level_entries=subentries, second_type="empty_dict"
        )
        end_bl_rewards = [
            self.monthly_mean_end_eval_rewards_per_home["baseline"][repeat]
            for repeat in range(n_repeats)
        ]
        end_test_bl_rewards = [
            self.monthly_mean_test_rewards_per_home["baseline"][repeat]
            for repeat in range(n_repeats)
        ]

        for eval_entry in evaluation_methods_plot:
            mean_end_rewards_month = self.monthly_mean_end_eval_rewards_per_home[eval_entry]
            end_test_rewards_e = [
                self.monthly_mean_test_rewards_per_home[eval_entry][repeat]
                for repeat in range(n_repeats)
            ]
            all_nans = True \
                if sum(r is not None
                       for r in self.monthly_mean_eval_rewards_per_home[eval_entry][0]) == 0 \
                else False

            ave_rewards = [
                np.nanmean(self.monthly_mean_eval_rewards_per_home[eval_entry][repeat])
                if not all_nans else None
                for repeat in range(n_repeats)
            ]
            IQR, CVaR, LRT = [np.zeros(n_repeats) for _ in range(3)]
            best_eval = [
                m for m in self.monthly_mean_eval_rewards_per_home[eval_entry][0]
                if m is not None
            ][0] if not all_nans else None
            end_above_bl = [
                r - b for r, b in zip(mean_end_rewards_month, end_bl_rewards)
            ]
            end_test_above_bl = [
                r - b for r, b in zip(end_test_rewards_e, end_test_bl_rewards)
            ]
            if eval_entry == 'env_r_c':
                np.save(
                    Path(self.record_folder) / "end_test_above_bl_env_r_c.npy",
                    end_test_above_bl
                )
            for repeat in range(n_repeats):
                largest_drawdown = self.compute_largest_drawdown_repeat(
                    self.monthly_mean_eval_rewards_per_home[eval_entry][repeat], best_eval
                )
                IQR_repeat, CVaR_repeat = self.compute_IQR_and_CVaR_repeat(
                    self.monthly_mean_eval_rewards_per_home[eval_entry][repeat], all_nans
                )

                IQR[repeat] = IQR_repeat
                CVaR[repeat] = CVaR_repeat
                LRT[repeat] = largest_drawdown

            for metric, m in zip(
                    [mean_end_rewards_month, end_test_rewards_e, end_above_bl,
                     end_test_above_bl, ave_rewards, IQR, CVaR, LRT],
                    metric_entries[0:8]):
                metrics[m]["ave"][eval_entry] = np.mean(metric)
                metrics[m]["std"][eval_entry] = np.std(metric)
                for p in [25, 50, 75]:
                    metrics[m]["p" + str(p)][eval_entry] = np.percentile(metric, p)
            metrics["DR"]["ave"][eval_entry] = sp.stats.iqr(mean_end_rewards_month)
            metrics["DR"]["std"][eval_entry] = None
            metrics["RR"]["ave"][eval_entry] = np.mean(
                mean_end_rewards_month[
                    mean_end_rewards_month <= np.percentile(mean_end_rewards_month, 5)
                ]
            )
            metrics["RR"]["std"][eval_entry] = None

        return metrics, metric_entries

    def _append_eval(self, eval_steps, method, epoch, end_test):
        """Add evaluation results to the appropriate lists."""
        if method in eval_steps:
            if eval_steps[method]["reward"][-1] is not None:
                epoch_mean_eval_t = np.mean(eval_steps[method]["reward"])
            else:
                epoch_mean_eval_t = None
            if method in eval_steps:
                for info in ["reward", "action"]:
                    self.__dict__[f"eval_{info}s"][self.repeat][method][epoch] \
                        = eval_steps[method][info]
        else:
            for info in ["eval_rewards", "eval_actions"]:
                self.__dict__[info][self.repeat][method][epoch] = None
            epoch_mean_eval_t = None

        self.mean_eval_rewards[self.repeat][method][epoch] = epoch_mean_eval_t
        all_mean_eval_t = self.mean_eval_rewards[self.repeat][method]
        for info in self.break_down_rewards_entries:
            eval_step_t_e = \
                None if method not in eval_steps \
                else eval_steps[method][info]
            if info == 'max_voltage_deviation':
                self.__dict__[info][self.repeat][method][epoch] = \
                    max(eval_step_t_e[:-1]) if eval_step_t_e is not None else None
            # during one epoch, how many buses in total had a voltage deviation
            # during one epoch, how many hours had at least one voltage deviation
            elif info in ['n_voltage_violation_bus', 'n_voltage_violation_hour']:
                self.__dict__[info][self.repeat][method][epoch] = \
                    np.sum(eval_step_t_e[:-1]) if eval_step_t_e is not None else None
            elif info in ['mean_voltage_violation', 'mean_voltage_deviation']:
                self.__dict__[info][self.repeat][method][epoch] = \
                    np.mean(eval_step_t_e[:-1], axis=0) if eval_step_t_e is not None else None
            else:
                self.__dict__[info][self.repeat][method][epoch] = \
                    np.mean(eval_step_t_e, axis=0) if eval_step_t_e is not None else None

        # we have done at least 6 steps
        if not end_test and len(all_mean_eval_t) > 5 and method != "opt":
            equal_consecutive = \
                [abs(all_mean_eval_t[- i]
                     - all_mean_eval_t[- (i + 1)]) < 1e-5
                 for i in range(4)]
            if sum(equal_consecutive) == 4:
                self.stability[self.repeat][method] = epoch

    def _update_eps(self, rl, learner, epoch, end_test):
        """Add epsilon value for epsilon-greedy exploration."""
        if rl["type_learning"] == "DQN" and not end_test:
            for method in rl["type_Qs"]:
                if rl["distr_learning"] == "centralised":
                    self.eps[self.repeat][method][epoch] = copy.copy(learner[method].eps)
                else:
                    for home in range(self.n_homes):
                        self.eps[self.repeat][method][home][epoch] \
                            = copy.copy(learner[method][home].eps)

        elif rl["type_learning"] == "q_learning" and not end_test:
            for method in rl["type_Qs"]:
                if rl["q_learning"]["epsilon_decay"]:
                    self.eps[self.repeat][method][epoch] = copy.copy(learner.eps[method])
                else:
                    self.eps[self.repeat][epoch] = copy.copy(learner.eps)
