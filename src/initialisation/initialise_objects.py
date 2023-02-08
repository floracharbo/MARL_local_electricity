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
import uuid
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
                                        reward_type)


def initialise_objects(
        prm: dict,
        settings: dict = None,
        no_run: int = None,
        initialise_record: bool = True,
        run_mode: int = 1
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
    """
    # general input paths and system parameters are in inputs
    # where
    prm = input_params(prm, settings)
    prm['syst']['run_mode'] = run_mode
    if not Path("outputs").exists():
        os.mkdir("outputs")
    prm['paths']["opt_res"] = Path("outputs") / "opt_res"
    for folder in ["results", "seeds"]:
        prm['paths'][folder] = Path("outputs") / folder
    for folder in ["opt_res", "results", "seeds"]:
        if not (prm['paths'][folder]).exists():
            os.mkdir(prm['paths'][folder])

    if no_run is None:
        no_run = current_no_run(prm['paths']["results"])

    # turn into a usable format
    prm = initialise_prm(
        prm, no_run, initialise_all=initialise_record
    )

    if initialise_record:
        # initialise recording of progress - rewards, counters, etc.
        record = Record(prm, no_run=no_run)
    else:
        record = None

    if prm['RL']['supervised_loss'] and prm['RL']['supervised_loss_weight'] == 0:
        prm['RL']['supervised_loss_weight'] = 1
    if prm['RL']['supervised_loss'] and prm['RL']['n_epochs_supervised_loss'] == 0:
        prm['RL']['n_epochs_supervised_loss'] = prm['RL']['n_epochs']
    if not prm['RL']['supervised_loss'] and prm['RL']['n_epochs_supervised_loss'] > 0:
        prm['RL']['n_epochs_supervised_loss'] = 0

    if prm['grd']["subset_line_losses_modelled"] > 50:
        print(
            'Warning: More than 50 lines will be modelled with losses,'
            'the optimization process might take a lot of time '
            'and is best solved using a powerful computer.'
        )

    return prm, record


def _make_action_space(rl):
    """
    Make action space.

    inputs: rl dictionary
    """
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
    """Make scheme."""
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
        "optimal_actions":
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
    rl["n_homes"] = prm["syst"]["n_homes"]

    rl["obs_shape"] = len(rl["state_space"])
    if rl['trajectory']:
        rl['obs_shape'] *= prm['syst']['N']
    rl["state_shape"] = rl["obs_shape"] * rl["n_homes"]

    if not prm['syst']["server"]:
        rl["use_cuda"] = False
    if rl['use_cuda'] and not th.cuda.is_available():
        print(
            f"rl['use_cuda'] was True, and server is {prm['syst']['server']}, "
            "but not th.cuda.is_available(). Set use_cuda <- False"
        )
        rl['use_cuda'] = False

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
    for data in ["wholesale", "carbon_intensity", "temp"]:
        paths[f"{data}_file"] \
            = f"{paths[data]}_n{prm['syst']['H']}_{prm['syst']['year']}.npy"

    paths["folder_run"] = Path("outputs") / "results" / f"run{no_run}"
    paths["record_folder"] = paths["folder_run"] / "record"
    prm["paths"]["fig_folder"] = paths["folder_run"] / "figures"
    paths["input_folder"] = Path(paths["input_folder"])
    paths["open_inputs"] = paths["input_folder"] / paths['open_inputs_folder']
    paths['hedge_inputs'] \
        = paths["input_folder"] / paths['hedge_inputs_folder'] / f"n{prm['syst']['H']}"
    paths["factors_path"] = paths["hedge_inputs"] / paths["factors_folder"]
    paths["network_data"] = paths['open_inputs'] / paths['ieee_network_data']
    paths['clus_path'] = paths['hedge_inputs'] / paths['clus_folder']
    paths['test_data'] = paths['open_inputs'] / 'testing_data'

    return paths


def _load_data_dictionaries(paths, syst):
    # load data into dictionaries
    for info in ["residual_distribution_prms", "mean_residual", "f_min", "f_max"]:
        with open(paths['factors_path'] / f"{info}.pickle", "rb") as file:
            syst[info] = pickle.load(file)

    for info in ["p_clus", "p_trans", "n_clus"]:
        with open(paths['clus_path'] / f"{info}.pickle", "rb") as file:
            syst[info] = pickle.load(file)

    # add one to syst['n_clus']['car'] for no trip days
    syst['n_clus']['car'] += 1

    return syst


def _load_bat_factors_parameters(paths, car):
    path = paths["factors_path"] / "car_p_pos.pickle"
    with open(path, "rb") as file:
        car['f_prob_pos'] = pickle.load(file)
    path = paths["factors_path"] / "car_p_zero2pos.pickle"
    with open(path, "rb") as file:
        car['f_prob_zero2pos'] = pickle.load(file)

    path = paths["factors_path"] / "car_mid_fs_brackets.pickle"
    with open(path, "rb") as file:
        car['mid_fs_brackets'] = pickle.load(file)

    path = paths["factors_path"] / "car_fs_brackets.pickle"
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
    car, syst, paths = [prm[key] for key in ["car", "syst", "paths"]]

    car["C"] = car["dep"]  # GBP/kWh storage costs

    # have list of car capacities based on capacity and ownership inputs
    car["cap"] = np.array(car["cap"]) if isinstance(car["cap"], list) else np.full(syst["n_homes"], car["cap"], dtype=np.float32)
    if "own_car" in car:
        for passive_ext in ["", "P"]:
            car["own_car" + passive_ext] = np.ones(syst["n_homes" + passive_ext]) \
                if car["own_car" + passive_ext] == 1 else np.array(car["own_car" + passive_ext])
        car["cap"] = np.where(car["own_car"], car["cap"], 0)

    car = _load_bat_factors_parameters(paths, car)

    # battery characteristics
    car["min_charge"] = [car["cap"][home] * max(car["SoCmin"], car["baseld"])
                         for home in range(syst["n_homes"])]
    car["store0"] = [car["SoC0"] * car["cap"][home] for home in range(syst["n_homes"])]
    if "capP" not in car:
        car["capP"] = [car["cap"][0] for _ in range(syst["n_homesP"])]
    car["store0P"] = [car["SoC0"] * car["capP"][home] for home in range(syst["n_homesP"])]
    car["min_chargeP"] = [
        car["capP"][home] * max(car["SoCmin"], car["baseld"])
        for home in range(syst["n_homesP"])
    ]
    car["phi0"] = np.arctan(car["c_max"])

    return car


def _format_rl_parameters(rl):
    for key in [
        "n_epochs", "n_repeats", "instant_feedback", "rnn_hidden_dim", "buffer_size"
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
            specified_per_reward_only = len(list(control_window_eps.keys())[0].split("_")) == 1
            if specified_per_reward_only:
                for evaluation_method in rl["eval_action_choice"]:
                    rl["control_window_eps"][evaluation_method] = \
                        control_window_eps[reward_type(evaluation_method)]


def _dims_states_actions(rl, syst):
    rl["dim_states"] = len(rl["state_space"])
    rl["dim_actions"] = 1 if rl["aggregate_actions"] else 3
    rl["dim_actions_1"] = rl["dim_actions"]
    if not rl["aggregate_actions"]:
        rl["low_action"] = rl["low_actions"]
        rl["high_action"] = rl["high_actions"]
    if "trajectory" not in rl:
        rl["trajectory"] = False
    if rl["distr_learning"] == "joint":
        rl["dim_actions"] *= rl["n_homes"]
        rl["trajectory"] = False
    if rl["trajectory"]:
        for key in ["dim_states", "dim_actions"]:
            rl[key] *= syst["N"]
        if syst['run_mode'] == 1:
            for key in ["low_action", "high_action"]:
                rl[key] *= syst["N"]


def _remove_states_incompatible_with_trajectory(rl):
    if rl['type_learning'] == 'q_learning' and rl['trajectory']:
        print("q learning not implemented with trajectory -> set trajectory = False")
        rl['trajectory'] = False
    if rl['trajectory']:
        problematic_states = [
            state for state in rl['state_space']
            if state in ['store_bool_flex', 'store0', 'bool_flex', 'flexibility']
        ]
        for problematic_state in problematic_states:
            if problematic_state in rl['state_space']:
                print(
                    f"Trajectory learning is not compatible with {problematic_state} state. "
                    f"Removing it from state space."
                )
                idx = rl['state_space'].index(problematic_state)
                rl['state_space'].pop(idx)

    return rl


def _expand_grdC_states(rl):
    n_steps = None
    state_grdC = None
    for state in rl['state_space']:
        if state[0: len('grdC_n')] == 'grdC_n':
            n_steps = int(state[len('grdC_n'):])
            state_grdC = state
            rl['grdC_n'] = n_steps

    if n_steps is not None:
        rl['state_space'].pop(rl['state_space'].index(state_grdC))
        for step in range(n_steps):
            rl['state_space'].append(f'grdC_t{step}')

    return rl


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
    rl, syst, heat = [prm[key] for key in ["RL", "syst", "heat"]]
    rl = _format_rl_parameters(rl)
    rl = _expand_grdC_states(rl)
    rl = _remove_states_incompatible_with_trajectory(rl)

    _dims_states_actions(rl, syst)

    # learning parameter variables
    rl["ncpu"] = mp.cpu_count() if syst["server"] else 10
    rl['episode_limit'] = 0 if rl['trajectory'] else syst['N']
    rl["tot_learn_cycles"] = rl["n_epochs"] * rl["ncpu"] \
        if rl["parallel"] else rl["n_epochs"]
    prm["RL"]["type_env"] = rl["type_learn_to_space"][rl["type_learning"]]
    rl["start_end_eval"] = int(rl["share_epochs_start_end_eval"] * rl["n_epochs"])
    rl["n_all_epochs"] = rl["n_epochs"] + rl["n_end_test"]
    if rl["type_learning"] == "DDPG":
        rl["instant_feedback"] = True

    for passive_ext in ["P", ""]:
        rl["default_action" + passive_ext] = [
            [rl["default_action"] for _ in range(rl["dim_actions"])]
            for _ in range(syst["n_homes" + passive_ext])
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

    if rl['trajectory']:
        rl['gamma'] = 0

    return rl


def _naming_file_extension_network_parameters(grd):
    """ Adds the manage_voltage and manage_agg_power settings to optimization results in opt_res """
    upper_quantities = ['max_voltage', 'max_grid_import']
    lower_quantities = ['min_voltage', 'max_grid_export']
    penalties_upper = ['overvoltage', 'import']
    penalties_lower = ['undervoltage', 'export']
    managements = ['manage_voltage', 'manage_agg_power']
    file_extension = ''
    for lower_quantity, upper_quantity, penalty_upper, penalty_lower, management in zip(
            lower_quantities, upper_quantities, penalties_upper, penalties_lower, managements
    ):
        if grd[management]:
            file_extension += f"_{management}_limit" + str(grd[upper_quantity])
            if grd[upper_quantity] != grd[lower_quantity]:
                file_extension += f"_{grd[lower_quantity]}"
            file_extension += "_penalty_coeff" + str(grd[f'penalty_{penalty_upper}'])
            if grd[f'penalty_{penalty_upper}'] != grd[f'penalty_{penalty_lower}']:
                file_extension += "_" + str(grd[f'penalty_{penalty_lower}'])

            if management == 'manage_voltage':
                file_extension += f"subset_losses{grd['subset_line_losses_modelled']}"

    return file_extension


def opt_res_seed_save_paths(prm):
    """
    Get strings and seeds which will be used to identify runs.

    inputs:
    prm:
        here the rl, heat, syst, ntw, paths, car entries are relevant

    output:
    rl, paths with updated entries

    """
    rl, heat, syst, grd, paths, car = \
        [prm[key] for key in ["RL", "heat", "syst", "grd", "paths", "car"]]

    paths["opt_res_file"] = \
        f"_D{syst['D']}_H{syst['H']}_{syst['solver']}_Uval{heat['Uvalues']}" \
        f"_ntwn{syst['n_homes']}_nP{syst['n_homesP']}_cmax{car['c_max']}"
    if "file" in heat and heat["file"] != "heat.yaml":
        paths["opt_res_file"] += f"_{heat['file']}"
    if sum(car['own_car']) != len(car['own_car']):
        paths["opt_res_file"] += "_no_car"
        for i_car in np.where(car['own_car'] == 0)[0]:
            paths["opt_res_file"] += f"{i_car}_"
    paths["seeds_file"] = f"outputs/seeds/seeds{paths['opt_res_file']}"
    if rl["deterministic"] == 2:
        for file in ["opt_res_file", "seeds_file"]:
            paths[file] += "_noisy"

    for file in ["opt_res_file", "seeds_file"]:
        if rl["deterministic"] == 2:
            paths[file] += "_noisy"
        paths[file] += f"_r{rl['n_repeats']}_epochs{rl['n_epochs']}" \
                       f"_explore{rl['n_explore']}_endtest{rl['n_end_test']}"
        if file == "opt_res_file" and prm["syst"]["change_start"]:
            paths["opt_res_file"] += "_changestart"
        paths[file] += _naming_file_extension_network_parameters(grd)
        # eff does not matter for seeds, but only for res
        if file == "opt_res_file" and prm["car"]["efftype"] == 1:
            paths["opt_res_file"] += "_eff1"

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
    # p/kWh -> £/kWh (nordpool was EUR/MWh so was * 1e-3)
    wholesale = [x * 1e-2 for x in np.load(wholesale_path)]
    grd["wholesale_all"] = wholesale
    carbon_intensity_path = paths["open_inputs"] / paths["carbon_intensity_file"]

    # gCO2/kWh to tCO2/kWh
    grd["cintensity_all"] = np.load(
        carbon_intensity_path, allow_pickle=True) * 1e-6
    # carbon intensity
    grd["Call"] = [
        price + carbon * syst["co2tax"]
        for price, carbon in zip(wholesale, grd["cintensity_all"])
    ]
    grd["perc"] = [np.percentile(grd["Call"], i) for i in range(0, 101)]

    if grd['compare_pandapower_optimisation'] and not grd['manage_voltage']:
        # comparison between optimisation and pandapower is only relevant if simulating voltage.
        grd['compare_pandapower_optimisation'] = False


def _syst_info(prm):
    syst, paths = prm["syst"], prm['paths']
    if "H" not in syst:
        syst["H"] = 24
    syst["N"] = syst["D"] * syst["H"]
    syst["duration"] = datetime.timedelta(days=syst["D"])
    syst["n_int_per_hr"] = int(syst["H"] / 24)
    # duration of time interval in hrs
    syst["dt"] = 1 / syst["n_int_per_hr"]
    syst['server'] = os.getcwd()[0: len(paths['user_root_path'])] != paths['user_root_path']
    syst['machine_id'] = str(uuid.UUID(int=uuid.getnode()))
    syst['timestampe'] = datetime.datetime.now().timestamp()


def _homes_info(loads, syst, gen, heat):
    for passive_ext in ["", "P"]:
        gen["own_PV" + passive_ext] = [1 for _ in range(syst["n_homes" + passive_ext])] \
            if gen["own_PV" + passive_ext] == 1 else gen["own_PV" + passive_ext]
        heat["own_heat" + passive_ext] = [1 for _ in range(syst["n_homes" + passive_ext])] \
            if heat["own_heat" + passive_ext] == 1 else heat["own_heat" + passive_ext]
        for ownership in ["own_loads" + passive_ext, "own_flex" + passive_ext]:
            if ownership in loads:
                loads[ownership] = np.ones(syst["n_homes" + passive_ext]) * loads[ownership] \
                    if isinstance(loads[ownership], int) else loads[ownership]


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
    """
    [paths, syst, loads, gen, save, heat] = [
        prm[key] if key in prm else None
        for key in ["paths", "syst", "loads", "gen", "save", "heat"]
    ]

    _make_type_eval_list(prm["RL"])

    if paths is not None:
        paths = _update_paths(paths, prm, no_run)
    _syst_info(prm)
    _homes_info(loads, syst, gen, heat)

    # update paths and parameters from inputs
    if paths is not None and initialise_all:
        syst = _load_data_dictionaries(paths, syst)
        _update_grd_prm(prm)
        loads["share_flex"], loads["max_delay"] = loads["flex"]
        for passive_ext in ['', 'P']:
            loads["share_flexs" + passive_ext] = [
                0 if not loads["own_flex" + passive_ext][home] else loads["share_flex"]
                for home in range(syst["n_homes" + passive_ext])
            ]

    # car avail, type, factors
    prm['car'] = _update_bat_prm(prm)
    prm['RL'] = _update_rl_prm(prm, initialise_all)
    prm['RL'], prm['paths'] = opt_res_seed_save_paths(prm)

    if prm['RL']["type_learning"] == "facmac":
        prm = _facmac_initialise(prm)

    # calculate heating coefficients for recursive expression
    # based on input data
    if initialise_all and heat is not None:
        prm["heat"] = get_heat_coeffs(heat, syst, paths)

    save, prm = generate_colours(save, prm)

    return prm


def _filter_type_learning_facmac(rl):
    if rl["type_learning"] != "facmac":
        return

    valid_types = {
        "exploration": ["env_r_c", "opt", "baseline"],
        "evaluation": ["env_r_c", "opt_r_c", "opt", "baseline", "random"]
    }

    for stage in ["evaluation", "exploration"]:
        new_methods = []
        for method in rl[f"{stage}_methods"]:
            new_method = method
            if len(method.split("_")) == 3:
                if distr_learning(method) == "d":
                    new_method = f"{method.split('_')[0]}_{method.split('_')[1]}_c"
            new_methods.append(new_method)
        rl[f"{stage}_methods"] = []
        for method in new_methods:
            if method in valid_types[stage] or "env_r_c" in method:
                rl[f"{stage}_methods"].append(method)
            else:
                print(
                    f"Warning: {method} is not a valid method for {stage} stage "
                    f"with facmac and has been removed"
                )


def _filter_type_learning_competitive(rl):
    if rl["competitive"]:  # there can be no centralised learning
        rl["evaluation_methods"] = [
            t for t in rl["evaluation_methods"]
            if t in ["opt", "baseline"]
            or (reward_type(t) != "A" and distr_learning(t) != "c")
        ]


def _add_n_start_opt_explo(rl, evaluation_methods_list):
    if rl['n_start_opt_explo'] is not None and rl['n_start_opt_explo'] > 0:
        for i, initial_evaluation_method in enumerate(evaluation_methods_list):
            if initial_evaluation_method[0: 3] == 'env' and rl['n_start_opt_explo'] > 0:
                evaluation_methods_list[i] += f"_{rl['n_start_opt_explo']}_opt"

    return evaluation_methods_list


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

    evaluation_methods_list = _add_n_start_opt_explo(rl, evaluation_methods_list)

    rl["evaluation_methods"] = list(dict.fromkeys(evaluation_methods_list))
    _filter_type_learning_competitive(rl)

    rl["exploration_methods"] = [
        method for method in rl["evaluation_methods"]
        if not (method[0:3] == "opt" and len(method) > 3)
    ]

    if sum(method[0: 3] == 'opt' and len(method) > 3 for method in rl["evaluation_methods"]) > 0:
        rl["exploration_methods"] += ['opt']

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
