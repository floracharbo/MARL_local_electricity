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

from utils.userdeftools import get_moving_average, initialise_dict


class Record():
    """Record - keeping record of learning trajectories."""

    def __init__(self, prm: dict, no_run: int = None):
        """Add relevant properties from prm to the object."""
        rl = prm["RL"]
        self.no_run = no_run
        # initialise entries
        # entries that change for each repeat
        self.break_down_rewards_entries =\
            prm["syst"]["break_down_rewards_entries"]
        self.repeat_entries = prm["save"]["repeat_entries"] \
            + prm["syst"]["break_down_rewards_entries"]

        # entries that change for state ind but are the same across repeats
        self.stateind_entries = prm["save"]["stateind_entries"]
        self.entries = self.repeat_entries + self.stateind_entries
        # all entries
        for e in self.entries:
            self.__dict__.update({e: {}})
        # all exploration / evaluation methods
        self.all_methods = rl["type_eval"] + \
            list(set(rl["type_explo"]) - set(rl["type_eval"]))
        # parameters needed for generating paths string
        for e in ["n_epochs", "instant_feedback", "opt_res_file", "type_env"]:
            self.__dict__[e] = rl[e]
        for e in ["gamma", "epsilon_decay"]:
            self.__dict__[e] = rl[rl["type_learning"]]
        self.n_agents = prm["ntw"]["n"]
        # save q tables at each step in record
        # depending on the dimension of the q tables
        self.save_qtables = True \
            if (rl["state_space"] is None
                or len(rl["state_space"]) <= 2)\
            and rl["n_epochs"] <= 1e3 \
            else False
        self.n_epoch = rl["n_epochs"]
        self.last_entries = prm["save"]["last_entries"]
        # create folder
        self.paths = prm["paths"]

        prm["paths"]["save_days"] = prm["paths"]["folder_run"] / "save_days"
        for e in ["folder_run", "record_folder", "save_days", "fig_folder"]:
            if not os.path.exists(prm["paths"][e]):
                os.mkdir(prm["paths"][e])

        for entry in self.stateind_entries:
            self.__dict__[entry] = {}
        for entry in self.repeat_entries:
            self.__dict__[entry] = initialise_dict(range(rl["n_repeats"]))
        self.state_space = rl["state_space"]

        self.ridx = 0  # current repeat

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

    def new_repeat(self,
                   ridx: int,
                   rl: dict
                   ):
        """Reinitialise object properties at the start of new repeat."""
        self.ridx = ridx  # current repetition index

        for e in ["q_tables", "counter"]:
            if self.save_qtables:
                self.__dict__[e][ridx] = initialise_dict(
                    range(rl["n_epochs"]))
            else:
                self.__dict__[e][ridx] = {}

        self.duration_epoch[ridx], self.eps[ridx] \
            = [initialise_dict(range(rl["n_epochs"])) for _ in range(2)]

        self.train_rewards[ridx] = initialise_dict(rl["type_explo"])

        if rl["type_learning"] == "DQN"\
                or (rl["type_learning"] == "q_learning"
                    and rl["q_learning"]["epsilon_decay"]):
            self.eps[ridx] = initialise_dict(rl["type_eval"])

        if rl["type_learning"] == "DQN"\
                and rl["distr_learning"] == "decentralised":
            for t in rl["type_eval"]:
                self.eps[ridx][t] = initialise_dict(range(self.n_agents))

        for e in ["eval_rewards", "mean_eval_rewards", "eval_actions"] \
                + self.break_down_rewards_entries:
            self.__dict__[e][ridx] = initialise_dict(rl["type_eval"])

        for e in ["train_actions", "train_states"]:
            self.__dict__[e][ridx] = initialise_dict(rl["type_explo"])

        self.stability[ridx] = initialise_dict(
            rl["type_eval"], type_obj="Nones")

        for e in ["seed", "n_not_feas", "not_feas_vars"]:
            self.__dict__[e][ridx] = []

        self.last[ridx] = initialise_dict(
            self.last_entries + ["batch"], "empty_dict")

        for e in self.last_entries:
            self.last[ridx][e] = initialise_dict(self.all_methods)

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
        for t in rl["type_explo"]:
            self.train_rewards[self.ridx][t].append(
                list_train_stepvals[t]["reward"])
            self.train_actions[self.ridx][t].append(
                list_train_stepvals[t]["action"])
            self.train_states[self.ridx][t].append(
                list_train_stepvals[t]["state"])
        for t in rl["type_eval"]:
            self._append_eval(eval_steps, t, epoch, end_test)

        if rl["type_learning"] == "q_learning" and not end_test:
            if self.save_qtables:
                for e in ["q_tables", "counter"]:
                    self.__dict__[e][self.ridx][epoch] = copy.deepcopy(
                        learner.__dict__[e])
            elif epoch == self.n_epoch - 1:
                for e in ["q_tables", "counter"]:
                    self.__dict__[e][self.ridx] = copy.deepcopy(
                        learner.__dict__[e])

        self._update_eps(rl, learner, epoch, end_test)

        self.duration_epoch[self.ridx][epoch] = duration_epoch

    def last_epoch(self, evaluation, t, record_output, batch, done):
        """Record more information for the final epoch in self.last."""
        type_step = "eval" if evaluation else "train"
        if type_step == "eval":
            for output, label in zip(record_output, self.last_entries):
                self.last[self.ridx][label][t].append(output)
            if done and t == "baseline":
                self.last[self.ridx]["batch"] = batch

    def save(self, end_of: str = "repeat"):
        """
        Save the relevant record object information to files.

        Save the relevant information collected during
        the explorations and evaluations
        """
        labels = self.repeat_entries if end_of == "repeat" \
            else self.stateind_entries
        if not self.save_qtables:
            labels = [label for label in labels
                      if label not in ["q_tables", "counter"]] \
                if end_of == "repeat" \
                else labels + ["q_tables", "counter"]
        ridx = self.ridx if end_of == "repeat" else None
        for label in labels:
            str_ = f"{label}" if ridx is None else f"{label}_ridx{ridx}"
            save_path = str_ if self.paths["record_folder"] is None \
                else Path(self.paths["record_folder"]) / str_
            to_save = self.__dict__[label] if end_of == "end" \
                else self.__dict__[label][self.ridx]
            np.save(save_path, to_save)

            del to_save

    def load(self, prm: dict):
        """List files to load for record object & call loading_file method."""
        repeat_labels = [e for e in self.repeat_entries
                         if e not in ["q_tables", "counters"]]\
            if not self.save_qtables \
            else self.repeat_entries
        stateind_labels = self.stateind_entries + \
            ["q_tables", "counters"] if not self.save_qtables \
            else self.stateind_entries
        for label in repeat_labels:
            for ridx in range(prm["RL"]["n_repeats"]):
                self._loading_file(label, ridx)
        for label in stateind_labels:
            self._loading_file(label)

    def _loading_file(self, label: str, ridx: int = None):
        """
        Load file and add to record.

        Given instruction for specific file to load by load method,
        """
        str_ = f"{label}" if ridx is None else f"{label}_ridx{ridx}"
        str_ = os.path.join(self.paths["record_folder"], str_ + ".npy")
        try:
            obj = np.load(str_, allow_pickle=True)
            if len(np.shape(obj)) == 0:
                obj = obj.item()
            if ridx is not None:
                self.__dict__[label][ridx] = obj
            else:
                self.__dict__[label] = obj
        except Exception as ex:
            print(f"l 1433 record ex {ex}")

    def results_to_percentiles(
        self,
        e,
        prm,
        mov_average=False,
        n_window=None
    ):
        """For each epoch, percentiles of evaluation across repeats."""
        p_vals = [25, 50, 75]
        percentiles = initialise_dict(p_vals)
        for epoch in range(prm["RL"]["n_all_epochs"]):
            diff_repeats = [
                self.mean_eval_rewards_per_hh[ridx][e][epoch]
                - self.mean_eval_rewards_per_hh[ridx]["baseline"][epoch]
                for ridx in range(prm["RL"]["n_repeats"])
                if self.mean_eval_rewards_per_hh[ridx][e][epoch] is not None
            ]
            for p in [25, 50, 75]:
                percentiles[p].append(None if len(diff_repeats) == 0
                                      else np.percentile(diff_repeats, p))
        if mov_average:
            for p in [25, 50, 75]:
                percentiles[p] = get_moving_average(
                    percentiles[p], n_window, Nones=False)

        p25, p50, p75 = [percentiles[p] for p in p_vals]
        p25_not_None, p75_not_None = \
            [[m for m in mov_p if m is not None] for mov_p in [p25, p75]]
        epoch_not_None = \
            [epoch for epoch in range(len(p25)) if p25[epoch] is not None]

        return p25, p50, p75, p25_not_None, p75_not_None, epoch_not_None

    def get_mean_rewards(
        self,
        prm,
        action_state_space_0,
        state_space_0,
        eval_entries_plot
    ):
        """
        Get mean reward per home over final learning / over the testing period.

        mean_eval_rewards_per_hh[ridx][e][epoch]:
            for each method, repeat and epoch, the average eval rewards
            over the whole evaluation episode
            (mean_eval_rewards[ridx][e][epoch]),
            divided by the number of homes

        mean_end_rewards[ridx][e]:
            the average mean_eval_rewards_per_hh over the end of the training,
            from start_end_eval -> n_epochs

        mean_end_test_rewards:
            the average mean_eval_rewards_per_hh after the end of the training,
            from n_epochs onwards during the fixed policy, test only period.
        """
        self.mean_end_rewards, self.mean_end_test_rewards = [], []
        self.mean_eval_rewards_per_hh = {}
        for ridx in range(prm["RL"]["n_repeats"]):  # loop through repetitions
            mean_eval_rewards_per_hh = {}
            action_state_space_0[ridx], state_space_0[ridx] = \
                [initialise_dict(range(prm["ntw"]["n"])) for _ in range(2)]
            # 1 - mean rewards
            end_rewards_ridx, end_test_rewards_ridx = {}, {}
            if "end_decay" not in prm["RL"] or "DQN" not in prm["RL"]:
                for type_learning in ["DQN", "DDQN", "q_learning"]:
                    prm["RL"][type_learning]["end_decay"] \
                        = self.n_epochs

            keys_ = self.mean_eval_rewards[ridx].keys()
            for e in eval_entries_plot.copy():
                if e not in keys_:
                    if prm["RL"]["type_learning"] == "facmac" \
                        and any(key[0: len(e[:-2])] == e[:-2] for key in keys_):
                        new = [key for key in keys_ if key[0: len(e[:-2])] == e[:-2]][0]
                        eval_entries_plot.remove(e)
                        eval_entries_plot.append(new)
                    else:
                        eval_entries_plot.remove(e)

            for e in eval_entries_plot:
                mean_eval_rewards_per_hh[e] = \
                    [r / (prm["ntw"]["n"] + prm["ntw"]["nP"])
                     if r is not None else None
                     for r in self.mean_eval_rewards[ridx][e]]
                end_rewards_ridx[e] = np.mean(
                    mean_eval_rewards_per_hh[e][
                        prm["RL"]["start_end_eval"]: self.n_epochs
                    ]
                )
                end_test_rewards_ridx[e] = np.mean(
                    mean_eval_rewards_per_hh[e][self.n_epochs:])
            self.mean_eval_rewards_per_hh[ridx] = mean_eval_rewards_per_hh
            self.mean_end_rewards.append(end_rewards_ridx)
            self.mean_end_test_rewards.append(end_test_rewards_ridx)

    def get_metrics(
            self,
            prm: dict,
            eval_entries_plot: list
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
        metric_entries = ["end", "end_test", "end_bl", "end_test_bl",
                          "mean", "DT", "SRT", "LRT", "DR", "RR"]
        metrics = initialise_dict(metric_entries, "empty_dict")
        for m in metric_entries:
            subentries = ["ave", "std", "p25", "p75", "p50"]
            for s in subentries:
                metrics[m][s] = {}

        end_bl_rewards = [
            self.mean_end_rewards[ridx]["baseline"]
            for ridx in range(n_repeats)
        ]
        end_test_bl_rewards = [
            self.mean_end_test_rewards[ridx]["baseline"]
            for ridx in range(n_repeats)
        ]

        for e in eval_entries_plot:
            end_rewards_e = \
                [self.mean_end_rewards[ridx][e] for ridx in range(n_repeats)]
            end_test_rewards_e = [
                self.mean_end_test_rewards[ridx][e]
                for ridx in range(n_repeats)
            ]
            all_nans = True \
                if sum(r is not None
                       for r in self.mean_eval_rewards_per_hh[0][e]) == 0 \
                else False

            ave_rewards = \
                [np.nanmean(np.array(self.mean_eval_rewards_per_hh[ridx][e],
                                     dtype=np.float))
                 if not all_nans else None
                 for ridx in range(n_repeats)]
            IQR, CVaR, LRT = [], [], []
            best_eval = [m for m in self.mean_eval_rewards_per_hh[0][e]
                         if m is not None][0] if not all_nans else None
            end_above_bl = [
                r - b for r, b in zip(end_rewards_e, end_bl_rewards)
            ]
            end_test_above_bl = [
                r - b for r, b in zip(end_test_rewards_e, end_test_bl_rewards)
            ]
            for ridx in range(n_repeats):
                mean_eval_rewards_per_hh = self.mean_eval_rewards_per_hh[ridx]
                detrended_rewards = \
                    [mean_eval_rewards_per_hh[e][epoch]
                     - mean_eval_rewards_per_hh[e][epoch - 1]
                     if sum(mean_eval_rewards_per_hh[e][epoch_] is None
                            for epoch_ in [epoch, epoch - 1]) == 0 else None
                     for epoch in range(1, self.n_epochs)]
                largest_drawdown = - 1e6
                epochs = [epoch for epoch in range(1, self.n_epochs)
                          if mean_eval_rewards_per_hh[e][epoch] is not None]
                for epoch in epochs:
                    if best_eval - mean_eval_rewards_per_hh[e][epoch] \
                            > largest_drawdown:
                        largest_drawdown = \
                            best_eval \
                            - mean_eval_rewards_per_hh[e][epoch]
                    if mean_eval_rewards_per_hh[e][epoch] > best_eval:
                        best_eval = mean_eval_rewards_per_hh[e][epoch]
                    assert largest_drawdown is not None, \
                        "largest_drawdown is None"

                detrended_rewards_notNone = \
                    [d for d in detrended_rewards if d is not None]
                IQR.append(sp.stats.iqr(detrended_rewards_notNone)
                           if not all_nans else None)
                CVaR.append(np.mean(
                    [dr for dr in detrended_rewards_notNone
                     if dr <= np.percentile(detrended_rewards_notNone, 5)])
                    if not all_nans else None)
                LRT.append(largest_drawdown)

            for metric, m in zip(
                    [end_rewards_e, end_test_rewards_e, end_above_bl,
                     end_test_above_bl, ave_rewards, IQR, CVaR, LRT],
                    metric_entries[0:8]):
                metrics[m]["ave"][e] = np.mean(metric)
                metrics[m]["std"][e] = np.std(metric)
                for p in [25, 50, 75]:
                    metrics[m]["p" + str(p)][e] = np.percentile(metric, p)
            metrics["DR"]["ave"][e] = sp.stats.iqr(end_rewards_e)
            metrics["DR"]["std"][e] = None
            metrics["RR"]["ave"][e] = np.mean(
                [r for r in end_rewards_e
                 if r <= np.percentile(end_rewards_e, 5)])
            metrics["RR"]["std"][e] = None

        return metrics, metric_entries

    def _append_eval(self, eval_steps, t, epoch, end_test):
        if t in eval_steps:
            if eval_steps[t]["reward"][-1] is not None:
                epoch_mean_eval_t = np.mean(eval_steps[t]["reward"])
            else:
                epoch_mean_eval_t = None
            for e in ["reward", "action"]:
                self.__dict__[f"eval_{e}s"][self.ridx][t].append(
                    eval_steps[t][e]
                )
        else:
            for e in ["eval_rewards", "eval_actions"]:
                self.__dict__[e][self.ridx][t].append(None)
            epoch_mean_eval_t = None
        self.mean_eval_rewards[self.ridx][t].append(epoch_mean_eval_t)
        all_mean_eval_t = self.mean_eval_rewards[self.ridx][t]
        for e in self.break_down_rewards_entries:
            eval_step_t_e = \
                None if t not in eval_steps \
                else eval_steps[t][e]
            self.__dict__[e][self.ridx][t].append(
                np.mean(eval_step_t_e, axis=0) if eval_step_t_e is not None
                else None)
        # we have done at least 6 steps
        if not end_test and len(all_mean_eval_t) > 5 and t != "opt":
            equal_consecutive = \
                [abs(all_mean_eval_t[- i]
                     - all_mean_eval_t[- (i + 1)]) < 1e-5
                 for i in range(4)]
            if sum(equal_consecutive) == 4:
                self.stability[self.ridx][t] = epoch

    def _update_eps(self, rl, learner, epoch, end_test):
        if rl["type_learning"] == "DQN" and not end_test:
            for t in rl["type_Qs"]:
                if rl["distr_learning"] == "centralised":
                    self.eps[self.ridx][t].append(copy.copy(learner[t].eps))
                else:
                    for a in range(self.n_agents):
                        self.eps[self.ridx][t][a].append(
                            copy.copy(learner[t][a].eps))

        elif rl["type_learning"] == "q_learning" and not end_test:
            if rl["q_learning"]["epsilon_decay"]:
                self.eps[self.ridx][t].append(copy.copy(learner.eps[t]))
            else:
                self.eps[self.ridx][epoch] = copy.copy(learner.eps)
