#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:48:27 2021.

@author: Flora Charbonnier
"""

# import python packages
import datetime
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch as th
from gym import spaces

from src.initialisation.generate_colours import generate_colours
from src.initialisation.get_heat_coeffs import get_heat_coeffs
from src.initialisation.input_data import input_params
from src.learners.facmac.components.transforms import OneHot
from src.simulations.record import Record
from src.utilities.env_spaces import (_actions_from_unit_box,
                                      _actions_to_unit_box)
from src.utilities.userdeftools import (current_no_run, distr_learning,
                                        initialise_dict, reward_type)


def initialise_objects(
        prm: dict,
        settings: dict = None,
        no_run: int = None,
        initialise_record: bool = True
) -> Tuple[dict, Optional[object], dict]:
    """
    Set up parameters dictionary, load data, initialise recording object.

    inputs:
    prm:
        dictionary of run parameters;
        has attributes car, grd, loads, ntw, prm, gen (from inputs files)
    settings:
        in main_rl.py, defaults settings may be overrun
    no_run:
        no of current run for folder naming/recording purposes
    initialise_record:
        boolean for whether we need to
        initialise the record object
        e.g. not needed from plot_summary_no_agents where
        we only need high-level data

    output:
    prm:
        dictionary of parameters; with settings updated
        and intermediate variables computed
    record:
        the object to keep a record of the data and experiments
    profiles:
        the battery, generation and loads profiles
        to input to the environment
        (not in prm so as not to carry those large datasets around)
    """
    # general input paths and system parameters are in inputs
    # where
    prm = input_params(prm, settings)
    if not Path("outputs").exists():
        os.mkdir("outputs")
    prm['paths']["opt_res"] = Path("outputs") / f"opt_res_v{prm['syst']['data_version']}"
    for folder in ["results", "seeds"]:
        prm['paths'][folder] = Path("outputs") / folder
    for folder in ["opt_res", "results", "seeds"]:
        if not (prm['paths'][folder]).exists():
            os.mkdir(prm['paths'][folder])

    if no_run is None:
        no_run = current_no_run(prm['paths']["results"])

    # turn into a usable format
    prm, profiles = initialise_prm(
        prm, no_run, initialise_all=initialise_record
    )

    if initialise_record:
        # initialise recording of progress - rewards, counters, etc.
        record = Record(prm, no_run=no_run)
    else:
        record = None

    return prm, record, profiles


def _make_action_space(rl):
    if rl["discretize_actions"]:
        action_space = spaces.Discrete(rl["n_discrete_actions"])
    else:
        action_space = spaces.Box(
            low=np.array(rl["low_action"], dtype=np.float32),
            high=np.array(rl["high_action"], dtype=np.float32),
            shape=(rl["dim_actions"],), dtype=np.float32)
    rl["action_space"] = [action_space] * rl["n_homes"]

    ttype = th.FloatTensor if not rl["use_cuda"] else th.cuda.FloatTensor
    mult_coef_tensor = ttype(rl["n_homes"], rl["dim_actions"])
    action_min_tensor = ttype(rl["n_homes"], rl["dim_actions"])
    if not rl["discretize_actions"]:
        for _aid in range(rl["n_homes"]):
            for _actid in range(rl["dim_actions"]):
                _action_min = rl["action_space"][_aid].low[_actid]
                _action_max = rl["action_space"][_aid].high[_actid]
                mult_coef_tensor[_aid, _actid] = \
                    (_action_max - _action_min).item()
                action_min_tensor[_aid, _actid] = _action_min.item()

    rl["actions2unit_coef"] = mult_coef_tensor
    rl["actions2unit_coef_cpu"] = mult_coef_tensor.cpu()
    rl["actions2unit_coef_numpy"] = mult_coef_tensor.cpu().numpy()
    rl["actions_min"] = action_min_tensor
    rl["actions_min_cpu"] = action_min_tensor.cpu()
    rl["actions_min_numpy"] = action_min_tensor.cpu().numpy()
    rl["avail_actions"] = np.ones((rl["n_homes"], rl["dim_actions"]))

    # make conversion functions globally available
    rl["actions2unit"] = _actions_to_unit_box
    rl["unit2actions"] = _actions_from_unit_box
    rl["actions_dtype"] = np.float32


def _make_scheme(rl):
    action_dtype = th.long if not rl["actions_dtype"] == np.float32 \
        else th.float
    if not rl["discretize_actions"]:
        actions_vshape = rl["dim_actions"]
    elif all(
            isinstance(act_space, spaces.Tuple)
            for act_space in rl["action_spaces"]
    ):
        actions_vshape = 1 if not rl["actions_dtype"] == np.float32 else \
            max([i.spaces[0].shape[0] + i.spaces[1].shape[0]
                 for i in rl["action_spaces"]])

    # Default/Base scheme
    rl["scheme"] = {
        "state": {"vshape": rl["state_shape"]},
        "obs": {"vshape": rl["obs_shape"], "group": "agents"},
        "actions":
            {"vshape": (actions_vshape,),
             "group": "agents",
             "dtype": action_dtype},
        "avail_actions":
            {"vshape": (rl["dim_actions"],),
             "group": "agents",
             "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    rl["groups"] = {
        "agents": rl["n_homes"]
    }

    if not rl["actions_dtype"] == np.float32:
        rl["preprocess"] = {
            "actions": ("actions_onehot", [OneHot(out_dim=rl["dim_actions"])])
        }
    else:
        rl["preprocess"] = {}

    rl["env_info"] = {"state_shape": rl["state_shape"],
                      "obs_shape": rl["obs_shape"],
                      "n_actions": rl["dim_actions"],
                      "n_homes": rl["n_homes"],
                      "episode_limit": rl["episode_limit"],
                      "actions_dtype": np.float32
                      }


def _facmac_initialise(prm):
    """
    Compute parameters relating to the FACMAC reinforcement learning.

    inputs:
    prm:
        here the RL and ntw entries are relevant

    outputs:
    rl:
        corresponding to prm["RL"]; with updated parameters
    """
    rl = prm["RL"]
    rl["n_homes"] = prm["ntw"]["n"]

    rl["obs_shape"] = len(rl["state_space"])
    rl["state_shape"] = rl["obs_shape"] * rl["n_homes"]

    if not rl["server"]:
        rl["use_cuda"] = False
    rl["device"] = "cuda" if rl["use_cuda"] else "cpu"

    _make_action_space(rl)
    _make_scheme(rl)

    return prm


def _update_paths(paths, prm, no_run):
    """
    Compute paths needed for later parameter/data loading.

    inputs:
    prm:
        here the paths, RL, save entries are relevant

    outputs:
    paths:
        correpsonding to prm["paths"]; with updated parameters
    """
    for data in ['carbon_intensity', 'temp']:
        paths[data] = f"{paths[data]}_{prm['syst']['H']}.npy"
    paths["wholesale_file"] \
        = f"{paths['wholesale']}_n{prm['syst']['H']}_{prm['syst']['prices_year']}.npy"

    paths["folder_run"] = Path("outputs") / "results" / f"run{no_run}"
    paths["record_folder"] = paths["folder_run"] / "record"
    prm["paths"]["fig_folder"] = paths["folder_run"] / "figures"
    paths["input_folder"] = Path(paths["input_folder"])
    paths["open_inputs"] = paths["input_folder"] / paths["open_inputs_folder"]
    paths['hedge_inputs'] \
        = paths["input_folder"] / paths["hedge_inputs_folder"]
    paths["factors_path"] = paths["hedge_inputs"] / paths["factors_folder"]
    paths['clus_path'] = paths['hedge_inputs'] / paths['clus_folder']

    return paths


def _load_data_dictionaries(paths, syst):
    # load data into dictionaries
    for info in ["gamma_prms", "mean_residual", "f_min", "f_max"]:
        with open(paths['factors_path'] / f"{info}.pickle", "rb") as file:
            syst[info] = pickle.load(file)

    for info in ["p_clus", "p_trans", "n_clus"]:
        with open(paths['clus_path'] / f"{info}.pickle", "rb") as file:
            syst[info] = pickle.load(file)

    # add one to syst['n_clus']['car'] for no trip days
    syst['n_clus']['car'] += 1

    return syst


def _load_profiles(paths, car, syst, loads, gen):
    # load car profiles and availability
    # (mmap_mode = "r" means not actually loaded, but elements accessible)

    profiles = {"car": {}}
    for data in ["cons", "avail"]:
        profiles["car"][data] = initialise_dict(syst["labels_day"])
        for day_type in syst["labels_day"]:
            profiles["car"][data][day_type] = \
                initialise_dict(range(syst['n_clus']["car"]))

    syst['n_prof'] = initialise_dict(syst['data_types'], "empty_dict")

    prof_path = paths['hedge_inputs'] / paths['profiles_folder']
    folders = {'cons': 'norm_EV', 'avail': 'EV_avail'}

    for data in ["cons", "avail"]:
        path = prof_path / folders[data]
        files = os.listdir(path)
        for file in files:
            if file[0] != ".":
                day_type = file[3:5]
                cluster = int(file[1])
                profiles["car"][data][day_type][cluster] = \
                    np.load(path / file, mmap_mode="r")
                if len(np.shape(profiles["car"][data][day_type][cluster])) == 1:
                    new_shape = (1, len(profiles["car"][data][day_type][cluster]))
                    profiles["car"][data][day_type][cluster] = np.reshape(
                        profiles["car"][data][day_type][cluster], new_shape)

    for day_type in syst["labels_day"]:
        syst['n_prof']['car'][day_type] = [
            len(profiles["car"]['cons'][day_type][clus])
            for clus in range(syst['n_clus']['car'])
        ]

    profiles["loads"] = {}
    for day_type in syst["labels_day"]:
        profiles["loads"][day_type] = [
            np.load(
                prof_path / 'norm_loads' / f'c{clus}_{day_type}.npy', mmap_mode="r"
            ) for clus in range(syst["n_loads_clus"])
        ]
        syst['n_prof']['loads'][day_type] = [
            len(profiles['loads'][day_type][clus])
            for clus in range(syst['n_clus']['loads'])
        ]

    with open(paths['hedge_inputs'] / "percentiles.pickle", "rb") as file:
        percentiles = pickle.load(file)
    loads['perc'] = percentiles["loads"]
    car['perc'] = percentiles["car"]
    gen['perc'] = percentiles["gen"]

    # PV generation bank and month
    profiles, gen = _load_gen_profiles(gen, profiles, prof_path, syst)

    return profiles, car, loads, gen


def _load_gen_profiles(gen, profiles, prof_path, syst):
    profiles["gen"] = {}
    for month in range(12):
        profiles['gen'][month] = np.load(
            prof_path / 'norm_gen' / f'i_month{month}.npy',
            mmap_mode='r'
        )
    syst['n_prof']['gen'] = [len(profiles['gen'][m]) for m in range(12)]

    return profiles, gen


def _load_bat_factors_parameters(paths, car):
    path = paths["factors_path"] / "EV_p_pos.pickle"
    with open(path, "rb") as file:
        car['f_prob_pos'] = pickle.load(file)
    path = paths["factors_path"] / "EV_p_zero2pos.pickle"
    with open(path, "rb") as file:
        car['f_prob_zero2pos'] = pickle.load(file)

    path = paths["factors_path"] / "EV_mid_fs_brackets.pickle"
    with open(path, "rb") as file:
        car['mid_fs_brackets'] = pickle.load(file)

    path = paths["factors_path"] / "EV_fs_brackets.pickle"
    with open(path, "rb") as file:
        car['fs_brackets'] = pickle.load(file)

    return car


def _update_bat_prm(prm):
    """
    Compute parameters relating to the car battery for the experiments.

    inputs:
    prm:
        here the car, ntw, paths and syst entries are relevant

    outputs:
    car:
        correpsonding to prm["car"]; with updated parameters
    """
    car, ntw, paths = [prm[key] for key in ["car", "ntw", "paths"]]

    car["C"] = car["dep"]  # GBP/kWh storage costs

    # have list of car capacities based on capacity and ownership inputs
    if "own_EV" in car:
        car["cap"] = car["cap"] if isinstance(car["cap"], list) \
            else [car["cap"] for _ in range(ntw["n"])]
        car["own_EV"] = [1 for _ in range(ntw["n"])] \
            if car["own_EV"] == 1 else car["own_EV"]
        car["cap"] = [c if o == 1 else 0
                      for c, o in zip(car["cap"], car["own_EV"])]

    if isinstance(car["cap"], (int, float)):
        car["cap"] = [car["cap"] for _ in range(ntw["n"])]

    car = _load_bat_factors_parameters(paths, car)

    # battery characteristics
    car["min_charge"] = [car["cap"][home] * max(car["SoCmin"], car["baseld"])
                         for home in range(ntw["n"])]
    car["store0"] = [car["SoC0"] * car["cap"][home] for home in range(ntw["n"])]
    if "capP" not in car:
        car["capP"] = [car["cap"][0] for _ in range(ntw["nP"])]
    car["store0P"] = [car["SoC0"] * car["capP"][home] for home in range(ntw["nP"])]
    car["min_chargeP"] = [
        car["capP"][home] * max(car["SoCmin"], car["baseld"])
        for home in range(ntw["nP"])
    ]
    car["phi0"] = np.arctan(car["c_max"])

    return car


def _format_rl_parameters(rl):
    for key in [
        "n_epochs", "n_repeats", "instant_feedback", "rnn_hidden_dim"
    ]:
        rl[key] = int(rl[key])
    rl["lr"] = float(rl["lr"])
    if isinstance(rl["state_space"], str):
        rl["state_space"] = [rl["state_space"]]
    for key in ["batch_size", "buffer_capacity"]:
        if key in rl[rl["type_learning"]]:
            rl[rl["type_learning"]][key] = int(rl[rl["type_learning"]][key])

    type_learning = rl['type_learning']
    for key in [
        "epsilon_end", "T", "tauMT", "tauLT",
        "control_window_eps", "epsilon_decay_param"
    ]:
        if key in rl[type_learning]:
            if isinstance(rl[type_learning][key], (float, int)):
                # if only one value of eps end is given,
                # give them all methods the same eps value
                var = rl[type_learning][key]
                rl[type_learning][key] = {}
                for evaluation_method in rl["eval_action_choice"]:
                    rl[type_learning][key][evaluation_method] = var

    return rl


def _exploration_parameters(rl):
    # obtain decay parameter
    rl["T_decay_param"] = (rl["Tend"] / rl["T0"]) ** (1 / rl["n_epochs"])

    type_learning = rl["type_learning"]
    if type_learning in ["DDQN", "DQN", "q_learning"]:
        if rl[type_learning]["control_eps"] == 1 \
                and "baseline" not in rl["evaluation_methods"]:
            rl["evaluation_methods"].append("baseline")
        if rl[type_learning]["epsilon_end"] == "best":
            # take result of sensitivity analysis
            if rl[type_learning]["control_eps"] < 2:
                rl[type_learning]["epsilon_end"] \
                    = rl[type_learning]["best_eps_end"][
                    rl[type_learning]["control_eps"]]
            else:
                rl[type_learning]["epsilon_end"] = 1e-2  # not going to use it

        if isinstance(rl[type_learning]["epsilon_end"], float):
            rl[type_learning]["epsilon_decay_param"] = \
                (rl[type_learning]["epsilon_end"]
                 / rl[type_learning]["epsilon0"]) \
                ** (1 / (rl[type_learning]["end_decay"]
                         - rl[type_learning]["start_decay"]))
        else:
            rl[type_learning]["epsilon_decay_param"] = {}
            epsilon_end = rl[type_learning]["epsilon_end"]
            epsilon0 = rl[type_learning]["epsilon0"]
            for exploration_method in epsilon_end:
                rl[type_learning]["epsilon_decay_param"][exploration_method] \
                    = (epsilon_end[exploration_method] / epsilon0) \
                    ** (1 / rl["tot_learn_cycles"])

    # for key in ["epsilon_end", "T", "tauMT", "tauLT",
    #             "control_window_eps", "epsilon_decay_param"]:
    #     if key in rl[type_learning]:
    #         if isinstance(rl[type_learning][key], (float, int)):
    #             # if only one value of eps end is given,
    #             # give them all methods the same eps value
    #             var = rl[type_learning][key]
    #             rl[type_learning][key] = {}
    #             for evaluation_method in rl["eval_action_choice"]:
    #                 rl[type_learning][key][evaluation_method] = var
    #         elif key == "control_window_eps":
        control_window_eps = rl["control_window_eps"]
        window_per_method_specified \
            = isinstance(control_window_eps, dict) \
            and len(list(control_window_eps.keys())) > 0
        if window_per_method_specified:
            specified_per_reward_only = \
                len(list(control_window_eps.keys())[0].split("_")
                    ) == 1
            if specified_per_reward_only:
                for evaluation_method in rl["eval_action_choice"]:
                    rl["control_window_eps"][evaluation_method] = \
                        control_window_eps[reward_type(evaluation_method)]


def _dims_states_actions(rl, syst):
    rl["dim_states"] = len(rl["state_space"])
    rl["dim_actions"] = 1 if rl["aggregate_actions"] else 3

    if "trajectory" not in rl:
        rl["trajectory"] = False
    if rl["distr_learning"] == "joint":
        rl["dim_actions"] *= rl["n_homes"]
        rl["trajectory"] = False
    if rl["trajectory"]:
        for key in ["dim_states", "dim_actions"]:
            rl[key] *= syst["prm"]["N"]


def _update_rl_prm(prm, initialise_all):
    """
    Compute parameters relating to RL experiments.

    inputs:
    prm:
        here the RL and ntw entries are relevant
    initialise_all:
        whether we need to initialise all parameters
        or just the minimum

    outputs:
    rl:
        correpsonding to prm["RL"]; with updated parameters
    """
    rl, syst, ntw, heat = [prm[key] for key in ["RL", "syst", "ntw", "heat"]]
    rl = _format_rl_parameters(rl)
    _dims_states_actions(rl, syst)

    # learning parameter variables
    rl["ncpu"] = mp.cpu_count() if rl["server"] else 10
    rl['episode_limit'] = syst['N']
    rl["tot_learn_cycles"] = rl["n_epochs"] * rl["ncpu"] \
        if rl["parallel"] else rl["n_epochs"]
    prm["RL"]["type_env"] = rl["type_learn_to_space"][rl["type_learning"]]
    rl["start_end_eval"] = int(rl["share_epochs_start_end_eval"]
                               * rl["n_epochs"])
    rl["n_all_epochs"] = rl["n_epochs"] + rl["n_end_test"]
    if rl["type_learning"] == "DDPG":
        rl["instant_feedback"] = True
    if not rl["aggregate_actions"]:
        rl["low_action"] = rl["low_actions"]
        rl["high_action"] = rl["high_actions"]
    for passive_ext in ["P", ""]:
        rl["default_action" + passive_ext] = [
            [rl["default_action"] for _ in range(rl["dim_actions"])]
            for _ in range(ntw["n" + passive_ext])
        ]

    _exploration_parameters(rl)

    if rl["competitive"] and rl["distr_learning"] != "decentralised":
        print("changing distr_learning to decentralised as rl['competitive']")
        rl["distr_learning"] = "decentralised"
    if rl["competitive"] and rl["trajectory"]:
        print("cannot use trajectory with competitive setting")
        rl["trajectory"] = False

    if initialise_all and heat is not None:
        rl["statecomb_str"] = ""
        if rl["state_space"] is None:
            rl["statecomb_str"] = "None_"
        else:
            rl["statecomb_str"] = ""
            for state in rl["state_space"]:
                str_state = "None" if state is None else str(state)
                rl["statecomb_str"] += str_state + "_"
        rl["statecomb_str"] = rl["statecomb_str"][:-1]

    return rl


def _seed_save_paths(prm):
    """
    Get strings and seeds which will be used to identify runs.

    inputs:
    prm:
        here the rl, heat, syst, ntw, paths, car entries are relevant

    output:
    rl, paths with updated entries

    """
    rl, heat, syst, ntw, paths = \
        [prm[key] for key in ["RL", "heat", "syst", "ntw", "paths"]]

    paths["opt_res_file"] = \
        f"_D{syst['D']}_H{syst['H']}_{syst['solver']}_Uval{heat['Uvalues']}" \
        f"_ntwn{ntw['n']}_nP{ntw['nP']}"
    if "file" in heat and heat["file"] != "heat.yaml":
        paths["opt_res_file"] += f"{heat['file']}"
    paths["seeds_file"] = "outputs/seeds/seeds" + paths["opt_res_file"]
    if rl["deterministic"] == 2:
        for file in ["opt_res_file", "seeds_file"]:
            paths[file] += "_noisy"
    for file in ["opt_res_file", "seeds_file"]:
        paths[file] += f"_r{rl['n_repeats']}_epochs{rl['n_epochs']}" \
                       f"_explore{rl['n_explore']}_endtest{rl['n_end_test']}"
    if prm["syst"]["change_start"]:
        paths["opt_res_file"] += "_changestart"

    # eff does not matter for seeds, but only for res
    if prm["car"]["efftype"] == 1:
        paths["opt_res_file"] += "_eff1"
    for file in ["opt_res_file", "seeds_file"]:
        paths[file] += ".npy"

    if os.path.exists(paths["seeds_file"]):
        rl["seeds"] = np.load(paths["seeds_file"], allow_pickle=True).item()
    else:
        rl["seeds"] = {"P": [], "": []}
    rl["init_len_seeds"] = {}
    for passive_str in ["", "P"]:
        rl["init_len_seeds"][passive_str] = len(rl["seeds"][passive_str])

    return rl, paths


def _update_grd_prm(prm):
    """
    Update the parameters relating to grid information for the run.

    namely the carbon intensity and pricing historical data used

    input:
    prm:
        here the paths, grd, and syst entries are relevant

    output:
    grd:
        with updated information
    """
    paths, grd, syst = [prm[key] for key in ["paths", "grd", "syst"]]

    # grid loss
    grd["loss"] = grd["R"] / (grd["V"] ** 2)

    # wholesale
    wholesale_path = paths["open_inputs"] / paths["wholesale_file"]
    wholesale = [x * 1e-3 for x in np.load(wholesale_path)]  # p/kWh -> £/kWh (nordpool was EUR/MWh so was * 1e-3)
    grd["wholesale_all"] = wholesale
    carbon_intensity_path = paths["open_inputs"] / paths["carbon_intensity"]

    # gCO2/kWh to tCO2/kWh
    grd["cintensity_all"] = np.load(
        carbon_intensity_path, allow_pickle=True) * 1e-6

    # carbon intensity
    grd["Call"] = [price + carbon * syst["co2tax"]
                   for price, carbon in zip(wholesale, grd["cintensity_all"])]
    grd["perc"] = [np.percentile(grd["Call"], i) for i in range(0, 101)]


def _time_info(prm):
    syst = prm["syst"]
    if "H" not in syst:
        syst["H"] = 24
    syst["N"] = syst["D"] * syst["H"]
    syst["duration"] = datetime.timedelta(days=syst["D"])
    # loads and cluster prob and profile banks (mmap = "r")
    syst["n_int_per_hr"] = int(syst["H"] / 24)
    # duration of time interval in hrs
    syst["dt"] = 1 / syst["n_int_per_hr"]
    syst['current_date0'] = syst['date0']


def _homes_info(loads, ntw, gen, heat):
    ntw["n_all"] = ntw["n"] + ntw["nP"]
    for passive_ext in ["", "P"]:
        gen["own_PV" + passive_ext] = [1 for _ in range(ntw["n" + passive_ext])] \
            if gen["own_PV" + passive_ext] == 1 else gen["own_PV" + passive_ext]
        heat["own_heat" + passive_ext] = [1 for _ in range(ntw["n" + passive_ext])] \
            if heat["own_heat" + passive_ext] == 1 else heat["own_heat" + passive_ext]
        for ownership in ["own_loads" + passive_ext, "own_flex" + passive_ext]:
            if ownership in loads:
                loads[ownership] = np.ones(ntw["n" + passive_ext]) \
                    if loads[ownership] == 1 else loads[ownership]


def initialise_prm(prm, no_run, initialise_all=True):
    """
    Compute useful variables from input data.

    inputs:
    prm:
        dictionary of run parameters;
        has attributes car, grd, loads, ntw, prm, gen,
        rl (from inputs files)
    no_run:
        no of current run for folder naming/recording purposes
    initialise_all:
        boolean, false if we are only initialising
        minimum required data
        e.g. to plot the summary plot rewards vs agents
        where we don't need everything

    outputs:
    prm:
        updated parameters
    profiles:
        the battery, generation and loads profiles
        to input to the environment
    """
    prm_entries = [
        "paths", "syst", "car", "ntw", "loads", "gen", "save", "heat"
    ]
    [paths, syst, car, ntw, loads, gen, save, heat] = \
        [prm[key] if key in prm else None for key in prm_entries]

    _make_type_eval_list(prm["RL"])
    if paths is not None:
        paths = _update_paths(paths, prm, no_run)
    _time_info(prm)
    _homes_info(loads, ntw, gen, heat)

    # update paths and parameters from inputs
    if paths is not None:
        if initialise_all:
            syst = _load_data_dictionaries(paths, syst)
            _update_grd_prm(prm)
            profiles, car, loads, gen = _load_profiles(
                paths, car, syst, loads, gen)
            loads["share_flex"], loads["max_delay"] = loads["flex"]
            loads["share_flexs"] = \
                [0 if not loads["own_flex"][home]
                 else loads["share_flex"] for home in range(ntw["n"])]
        else:
            profiles = None
    else:
        profiles = None

    # car avail, type, factors
    car = _update_bat_prm(prm)

    rl = _update_rl_prm(prm, initialise_all)
    rl, paths = _seed_save_paths(prm)

    if rl["type_learning"] == "facmac":
        prm = _facmac_initialise(prm)

    # calculate heating coefficients for recursive expression
    # based on input data
    if initialise_all and heat is not None:
        prm["heat"] = get_heat_coeffs(heat, ntw, syst, loads, paths)

    # %% do not save batches if too many of them!
    if rl["n_repeats"] * rl["n_epochs"] * (rl["n_explore"] + 1) > 200:
        save["plotting_batch"] = False

    save, prm = generate_colours(save, prm)

    return prm, profiles


def _filter_type_learning_facmac(rl):
    if rl["type_learning"] != "facmac":
        return

    valid_types = {
        "exploration": ["env_r_c", "env_d_c", "opt", "baseline"],
        "evaluation": [
            "env_r_c", "env_d_c", "opt_r_c", "opt_d_c",
            "opt", "baseline", "random"
        ]
    }
    for stage in ["evaluation", "exploration"]:
        new_methods = []
        for method in rl[f"{stage}_methods"]:
            new_method = method
            if len(method.split("_")) == 3:
                if distr_learning(method) == "d":
                    new_method = f"{method.split('_')[0]}_{method.split('_')[1]}_c"
            new_methods.append(new_method)
        rl[f"{stage}_methods"] \
            = [method for method in new_methods if method in valid_types[stage]]


def _filter_type_learning_competitive(rl):
    if rl["competitive"]:  # there can be no centralised learning
        rl["evaluation_methods"] = [
            t for t in rl["evaluation_methods"]
            if t in ["opt", "baseline"]
            or (reward_type(t) != "A" and distr_learning(t) != "c")
        ]


def _make_type_eval_list(rl, large_q_bool=False):
    evaluation_methods_list = ["baseline"]
    if rl["evaluation_methods"] is not None:
        input_evaluation_methods = rl["evaluation_methods"] \
            if isinstance(rl["evaluation_methods"], list) \
            else [rl["evaluation_methods"]]
        evaluation_methods_list += input_evaluation_methods
    else:
        data_sources = ["env_"]
        if rl["opt_bool"]:
            data_sources += ["opt_"]
            evaluation_methods_list += ["opt"]

        if rl["type_learning"] == "facmac":
            methods_combs = ["r_c", "d_c"]

        elif rl["type_learning"] == "q_learning":
            methods_combs = ["r_c", "r_d", "A_Cc", "A_c", "A_d"] \
                if large_q_bool else ["r_c", "r_d", "A_c", "A_d"]
            if rl["difference_bool"]:
                add_difference = ["d_Cc", "d_c", "d_d"] \
                    if large_q_bool else ["d_c", "d_d"]
                methods_combs += add_difference
            methods_combs_opt = ["n_c", "n_d"]
            if "opt_" in data_sources:
                evaluation_methods_list += ["opt_" + m for m in methods_combs_opt]

        elif rl["type_learning"] in ["DDPG", "DQN", "DDQN"]:
            if rl["distr_learning"] == "decentralised":
                methods_combs = ["d_d"] \
                    if rl["difference_bool"] else ["r_d"]
            else:
                methods_combs = ["d_c"] \
                    if rl["difference_bool"] else ["r_c"]

        for data_source in data_sources:
            evaluation_methods_list += [data_source + mc for mc in methods_combs]

    rl["evaluation_methods"] = evaluation_methods_list
    _filter_type_learning_competitive(rl)

    rl["exploration_methods"] = [
        t for t in rl["evaluation_methods"] if not (t[0:3] == "opt" and len(t) > 3)
    ]
    rl["eval_action_choice"] = [
        t for t in rl["evaluation_methods"] if t not in ["baseline", "opt"]
    ]
    assert len(rl["eval_action_choice"]) > 0, \
        "not valid eval_type with action_choice"

    _filter_type_learning_facmac(rl)

    rl["type_Qs"] \
        = rl["eval_action_choice"] + [
        ac + "0" for ac in rl["eval_action_choice"]
        if len(ac.split("_")) >= 3 and (
            reward_type(ac) == "A" or distr_learning(ac)[0] == "C"
        )
    ]

    print(f"evaluation_methods {evaluation_methods_list}")
