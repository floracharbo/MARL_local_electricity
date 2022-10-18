#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:23:20 2021.

@author: floracharbonnier

after learning, save usable information and figures
"""
# packages
import os  # path management
import shutil  # to copy/remove files
import sys
import time  # to record time it takes to run simulations
from src.post_analysis.plotting.plotting import plotting
from src.post_analysis.print_results import print_results
from src.utilities.userdeftools import distr_learning
from pathlib import Path

import numpy as np


def _max_min_q(q_table, n_states, minq, maxq, prm):
    """Compare min/max q values with saved ones. Update if necessary."""
    for type_q in q_table.keys():
        n_homes = prm["ntw"]["n"] if distr_learning(type_q) == "d" else 1
        for agent in range(n_homes):
            for state in range(n_states):
                if min(q_table[type_q][agent][state]) < minq:
                    minq = min(q_table[type_q][agent][state])
                if max(q_table[type_q][agent][state]) > maxq:
                    maxq = max(q_table[type_q][agent][state])

    return minq, maxq


def _post_run_update(prm, record):
    """
    Update variables e.g. min/max q values, seeds, save inputs.

    if this post_processing is straight after running the
    learning.
    """
    min_max_q_saved = sum(
        [
            (prm["paths"]["open_inputs"] / f"{str_}.npy").is_file()
            for str_ in ["minq", "maxq"]]
    ) == 2

    # min_max_q_saved=False
    if prm["RL"]["type_learning"] == "q_learning" \
            and min_max_q_saved and record.save_qtables:
        minq, maxq = [np.load(os.path.join(
            prm["paths"]["open_inputs"], str_ + ".npy"))
            for str_ in ["minq", "maxq"]]
        if len(record.q_tables[0][0].keys()) > 0:
            repeat, epoch = 0, 0
            # [state_ind][repeat][epoch][q_table_name][state][action]
            n_states, _ = np.shape(
                record.q_tables[repeat][epoch][prm["RL"]["type_Qs"][0]][0])
            for repeat in range(prm["RL"]["n_repeats"]):
                for epoch in range(prm["RL"]["n_epochs"]):
                    minq, maxq = _max_min_q(
                        record.q_tables[repeat][epoch],
                        n_states, minq, maxq, prm
                    )

        elif len(record.q_tables[0].keys()) > 0:
            # [state_ind][r][epoch][q_table_name][state][action]
            n_states, _ = np.shape(record.q_tables[0][prm["RL"]["type_Qs"][0]])
            for repeat in range(prm["RL"]["n_repeats"]):
                minq, maxq = _max_min_q(
                    record.q_tables[repeat], n_states, minq, maxq, prm)

        np.save(prm["paths"]["open_inputs"] / "minq.npy", minq)
        np.save(prm["paths"]["open_inputs"] / "maxq.npy", maxq)

    # update seeds, if they have been appended compared to the initilally
    # loaded seeds
    if prm["RL"]["init_len_seeds"][""] < len(
            prm["RL"]["seeds"][""]) or prm["RL"]["init_len_seeds"]["P"] < len(
            prm["RL"]["seeds"]["P"]):
        np.save(prm["paths"]["seeds_file"], prm["RL"]["seeds"])

    # copy inputs to results folder
    prm["paths"]["save_inputs"] = prm["paths"]["folder_run"] / "inputData"
    if not os.path.exists(prm["paths"]["save_inputs"]):
        # make a folder for saving results if it does not yet exist
        os.mkdir(prm["paths"]["save_inputs"])

    prm_save = get_prm_save(prm)

    np.save(Path(prm["paths"]["save_inputs"]) / "prm", prm_save)


def _save_reliability(prm, record):
    """Save results in format for inputting into reliability metrics module."""
    paths = prm["paths"]
    paths["save_data_folder"] = paths["folder_run"] / "data"
    if not os.path.exists(paths["save_data_folder"]):
        os.mkdir(paths["save_data_folder"])
    for evaluation_method in prm["RL"]["evaluation_methods"]:
        algo_dir = paths["save_data_folder"] / f"algo{evaluation_method}"
        if not os.path.exists(algo_dir):
            os.mkdir(algo_dir)
        task_dir = algo_dir / "task"
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        for repeat in range(prm["RL"]["n_repeats"]):
            r_dir = task_dir / str(repeat)
            if not os.path.exists(r_dir):
                os.mkdir(r_dir)
            train_dir = r_dir / "train"
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            save_dir = train_dir / "mean_eval_rewards"
            np.save(save_dir, record.mean_eval_rewards[repeat][evaluation_method])


def _clean_up(prm, no_run):
    """Remove opt_res, batch left in main folder, unnecessary files."""
    # clean up folder
    opt_res_file = "res.npy"
    # remove opt_res and batch that were left in main folder
    if os.path.exists(opt_res_file):
        os.remove(opt_res_file)
    files = os.listdir(os.getcwd())
    for file in files:
        if file[0:5] == "batch":
            os.remove(file)

    # remove no_run folder if not save_run
    if no_run is None and not prm["save"]["save_run"]:
        shutil.rmtree(prm["paths"]["folder_run"])


def add_prm_save_list(val, dict_, key):
    """Save all int/float/bool items."""
    if isinstance(val, list):
        dict_[key] = []
        for item in val:
            if isinstance(item, (int, float, bool)):
                dict_[key].append(item)

    return dict_


def get_prm_save_RL(prm_save, prm):
    """Add all int/float/bool items in prm["RL"] to dict to save."""
    prm_save["RL"] = {}
    for e, val in prm["RL"].items():
        prm_save["RL"] = add_prm_save_list(val, prm_save["RL"], e)
        if isinstance(val, dict):
            prm_save["RL"][e] = {}
            for key in val.keys():
                prm_save["RL"][e] = add_prm_save_list(
                    val[key], prm_save["RL"][e], key
                )
                if isinstance(val[key], (int, float, bool)):
                    prm_save["RL"][e][key] = val[key]
        elif isinstance(val, (int, float, bool)):
            prm_save["RL"][e] = val

    return prm_save


def get_prm_save(prm):
    """Save run parameters for record-keeping."""
    prm_save = {}  # save selected system parameters
    to_save_entries_syst = \
        np.load(
            Path(prm["paths"]["open_inputs"]) / "to_save_entries_syst.npy",
            allow_pickle=True
        ).item()
    for key in to_save_entries_syst:
        prm_save[key] = {}
        sub_keys = to_save_entries_syst[key] \
            if len(to_save_entries_syst[key]) > 0 \
            else prm[key].keys()
        for sub_key in sub_keys:
            if sub_key in prm[key]:
                prm_save[key][sub_key] = prm[key][sub_key]
            elif sub_key not in prm[key] \
                    and prm["RL"]["type_learning"] in prm[key] \
                    and sub_key in prm[key][prm["RL"]["type_learning"]]:
                prm_save[key][sub_key] = \
                    prm[key][prm["RL"]["type_learning"]][sub_key]
            elif sub_key in ["fprms", "fmean", "nclus"]:
                sub_key = f"{sub_key[0]}_{sub_key[1:]}"
                prm_save[key][sub_key] = prm[key][sub_key]
            else:
                print(f"{sub_key} not in prm[{key}]")
                # np.save(f"keys_prm_{key}", list(prm[key].keys()))

    prm_save = get_prm_save_RL(prm_save, prm)

    return prm_save


def post_processing(
        record: object,
        env: object,
        prm: dict,
        no_run: int = None,
        start_time: float = None,
        settings_i: dict = None
):
    """Save results to files, plot, uptate seeds, etc."""
    paths = prm["paths"]

    if no_run is None:  # if this is straight after learning
        # update min and max q for random initialisation
        _post_run_update(prm, record)
    else:
        record.no_run = no_run  # current run number
        # load results from file and add to record object
        # record.load(record_number=record.no_run)
        record.load(prm)

    # plotting
    paths["results_file"] = os.path.join(
        paths["folder_run"], f"results{record.no_run}.txt")

    if no_run is None:
        file = open(paths["results_file"], "w+", encoding="utf-8")
    else:
        for entry in os.listdir(paths["folder_run"]):
            if entry[0:7] == "results" and entry[8:] != f"{record.no_run}.txt":
                new_file = entry[:-(len(f"{record.no_run}.txt") - 1)
                                 ] + f"{record.no_run}.txt"
                os.rename(
                    os.path.join(
                        paths["folder_run"], entry), os.path.join(
                        paths["folder_run"], new_file))
        file = open(paths["results_file"], "a", encoding="utf-8")  # append
    if "description_run" in prm["save"]:
        file.write("run description: " + prm["save"]["description_run"] + "\n")
    if start_time is not None:
        file.write(f"time {time.time() - start_time}" + "\n")

    if settings_i is not None:
        for key in settings_i:
            file.write(f"{key} {settings_i[key]}" + "\n")

    # plotting
    sys.path.append("//")
    file, metrics = plotting(record, env.spaces, prm, file)
    file = print_results(prm, file, record, metrics)
    file.close()

    # save for reliability measures
    _save_reliability(prm, record)

    # clean up folder
    _clean_up(prm, no_run)
