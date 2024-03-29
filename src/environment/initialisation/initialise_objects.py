#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:48:27 2021.

@author: Flora Charbonnier
"""

# import python packages
import datetime
import math
import multiprocessing as mp
import os
import pickle
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch as th
from gym import spaces

import src.environment.utilities.env_spaces as env_spaces
import src.environment.utilities.userdeftools as utils
from src.environment.experiment_manager.record import Record
from src.environment.initialisation.generate_colours import generate_colours
from src.environment.initialisation.get_heat_coeffs import get_heat_coeffs
from src.environment.initialisation.input_data import input_params
from src.learners.facmac.components.transforms import OneHot


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
        no_run = utils.current_no_run(prm['paths']["results"])

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


def _make_action_space(rl, reactive_power_for_voltage_control):
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
    rl["actions2unit"] = env_spaces._actions_to_unit_box
    rl["unit2actions"] = env_spaces._actions_from_unit_box
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
        "state": {"vshape": rl["state_shape"], },
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
        # "episode_const": True,
    }
    rl["groups"] = {
        "agents": rl["n_homes"],
        "agents_test": rl["n_homes_test"],
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
    for n_homes in ['n_homes', 'n_homes_test']:
        rl[n_homes] = prm["syst"][n_homes]
    rl["obs_shape"] = len(rl["state_space"])
    if rl['trajectory']:
        rl['obs_shape'] *= prm['syst']['N']
    rl["state_shape"] = rl["obs_shape"] * rl["n_homes"]
    rl["state_shape_test"] = rl["obs_shape"] * rl["n_homes_test"]
    if not rl['offset_reward']:
        rl['delta_reward'] = 0
    if not prm['syst']["server"]:
        rl["use_cuda"] = False
    if rl['use_cuda'] and not th.cuda.is_available():
        print(
            f"rl['use_cuda'] was True, and server is {prm['syst']['server']}, "
            "but not th.cuda.is_available(). Set use_cuda <- False"
        )
        rl['use_cuda'] = False

    rl["device"] = "cuda" if rl["use_cuda"] else "cpu"

    if prm['syst']['run_mode'] == 1:
        _make_action_space(rl, prm["grd"]["reactive_power_for_voltage_control"])
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
        for test_str in ['', '_test']:
            paths[f"{data}{test_str}_file"] \
                = f"{paths[data]}_n{prm['syst']['H']}_{prm['syst'][f'year{test_str}']}.npy"
    paths["folder_run"] = Path("outputs") / "results" / f"run{no_run}"
    np.save("outputs/current_run_no.npy", no_run)
    paths["record_folder"] = paths["folder_run"] / "record"
    prm["paths"]["fig_folder"] = paths["folder_run"] / "figures"
    paths["input_folder"] = Path(paths["input_folder"])
    paths["open_inputs"] = paths["input_folder"] / paths['open_inputs_folder']
    paths["hedge_inputs_root"] = paths["input_folder"] / paths['hedge_inputs_folder']
    paths['hedge_inputs'] = paths["hedge_inputs_root"] / f"n{prm['syst']['H']}"
    paths["factors_path"] = paths["hedge_inputs"] / paths["factors_folder"]
    paths['clus_path'] = paths['hedge_inputs'] / paths['clus_folder']
    paths['test_data'] = paths['open_inputs'] / 'testing_data'
    if os.path.isfile("outputs/opt_res/files_list.npy"):
        paths['files_list'] = list(np.load("outputs/opt_res/files_list.npy"))
    else:
        paths['files_list'] = []

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
    car, heat, syst, paths = [prm[key] for key in ["car", "heat", "syst", "paths"]]

    car["C"] = car["dep"]  # GBP/kWh storage costs
    car['c_max0'] = car['c_max']
    # if prm['grd']['reactive_power_for_voltage_control'] or not prm['grd']['manage_voltage']:
    #     c_max_reactive_power = car['max_apparent_power_car'] * car['eta_ch']
    # else:
    #     c_max_reactive_power = np.sqrt(
    #         car['max_apparent_power_car'] ** 2 / (1 + car['pf_passive_homes'] ** 2)
    #     ) * car['eta_ch']
    c_max_reactive_power = np.minimum(
        np.sqrt(
            car['max_apparent_power_car'] ** 2 / (1 + car['pf_passive_homes'] ** 2)
        ) * car['eta_ch'],
        car['max_apparent_power_car'] * car['eta_ch']
    )
    if c_max_reactive_power < car['c_max']:
        print(
            f"updated c_max {car['c_max']} to be consistent with max_apparent_power_car "
            f"<- {c_max_reactive_power}")
        car['c_max'] = c_max_reactive_power

    # have list of car capacities based on capacity and ownership inputs
    car["caps"] = np.array(
        car["cap"]) if isinstance(car["cap"], list) \
        else np.full(syst["n_homes"], car["cap"], dtype=np.float32)

    for der, obj in zip(["car", "heat"], [car, heat]):
        if f"own_{der}" in obj:
            for ext in syst['n_homes_extensions_all']:
                obj[f"own_{der}" + ext] \
                    = np.ones(syst["n_homes" + ext]) * obj[f"own_{der}" + ext] \
                    if isinstance(obj[f"own_{der}" + ext], (int, float)) \
                    else np.array(obj[f"own_{der}" + ext])

    # battery characteristics
    car["min_charge"] = car["caps"] * max(car["SoCmin"], car["baseld"])
    car["store0"] = car["caps"] * car["SoC0"]
    for ext in syst['n_homes_extensions']:
        if "cap" + ext in car and isinstance(car["cap"], list):
            car["caps" + ext] = car['cap' + ext]
        elif "cap" + ext not in car or car["cap" + ext] is None:
            car["caps" + ext] = np.full(syst["n_homes" + ext], car["cap"])
        else:
            car["caps" + ext] = np.full(syst["n_homes" + ext], car["cap" + ext])
        car["store0" + ext] = car["SoC0"] * car["caps" + ext]
        car["min_charge" + ext] = car["caps" + ext] * max(car["SoCmin"], car["baseld"])
    for ext in syst['n_homes_extensions_all']:
        for info in ['caps', 'store0', 'min_charge']:
            car[info + ext] = np.where(car["own_car" + ext], car[info + ext], 0)
    car["phi0"] = np.arctan(car["c_max"])

    return car


def _format_rl_parameters(rl):
    for key in [
        "n_epochs", "n_repeats", "instant_feedback", "rnn_hidden_dim", "buffer_size"
    ]:
        rl[key] = int(rl[key])

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
    if type_learning in ["DDQN", "DQN", "q_learning", "facmac"]:
        if 'control_eps' in rl[type_learning] \
                and rl[type_learning]["control_eps"] == 1 \
                and "baseline" not in rl["evaluation_methods"]:
            rl["evaluation_methods"].append("baseline")
        if "epsilon_end" in rl[type_learning]:
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
    if type_learning == 'facmac' and "lr_decay" in rl['facmac'] and rl['facmac']['lr_decay']:
        rl['facmac']['lr_decay_param'] = (
            rl['facmac']['lr_end'] / rl['facmac']['lr0']
        ) ** (1 / rl['n_epochs'])
        rl['facmac']['critic_lr_decay_param'] = (
            rl['facmac']['critic_lr_end'] / rl['facmac']['critic_lr0']
        ) ** (1 / rl['n_epochs'])

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
                        control_window_eps[utils.reward_type(evaluation_method)]


def _dims_states_actions(rl, syst, reactive_power_for_voltage_control):
    rl["dim_states"] = 0 if rl["state_space"] is None else len(rl["state_space"])
    rl["dim_states_1"] = rl["dim_states"]
    if rl["aggregate_actions"]:
        rl["dim_actions"] = 1
    elif reactive_power_for_voltage_control:
        rl["dim_actions"] = 4
    else:
        rl["dim_actions"] = 3

    rl["dim_states_1"] = rl["dim_states"]
    rl["dim_actions_1"] = rl["dim_actions"]
    if syst['run_mode'] == 1:
        rl['low_actions'] = np.array(rl['all_low_actions'][0: rl["dim_actions_1"]])
        rl['high_actions'] = np.array(rl['high_actions'][0: rl["dim_actions_1"]])
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
                rl[key] = np.tile(rl[key], syst["N"])


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
    if rl['state_space'] is None:
        return rl

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


def rl_apply_n_homes_test(syst, rl):
    if syst['test_different_to_train']:
        if rl['homes_exec_per_home_train'] is None:
            rl['homes_exec_per_home_train'] = [[] for home_train in range(syst['n_homes'])]
            home_train = 0
            for home_exec in range(syst['n_homes_test']):
                rl['homes_exec_per_home_train'][home_train].append(home_exec)
                home_train = home_train + 1 if home_train + 1 < syst['n_homes'] else 0
        rl['action_selection_its'] = np.max(
            [
                len(homes_exec_per_home_train)
                for homes_exec_per_home_train in rl['homes_exec_per_home_train']
            ]
        )
        rl['action_train_to_exec'] = np.zeros(
            (rl['action_selection_its'], syst['n_homes'], syst['n_homes_test'])
        )
        rl['state_exec_to_train'] = np.zeros(
            (rl['action_selection_its'], syst['n_homes_test'], syst['n_homes'])
        )
        # actions[n_homes_test] = actions[n_homes_train] x [n_homes_train x n_homes_test]
        # states[n_homes_train] = actions[n_homes_test] x [n_homes_test x n_homes_train]
        for it in range(rl['action_selection_its']):
            for home_train in range(syst['n_homes']):
                if len(rl['homes_exec_per_home_train'][home_train]) > it:
                    rl['action_train_to_exec'][
                        it, home_train, rl['homes_exec_per_home_train'][home_train][it]
                    ] = 1
            rl['state_exec_to_train'][it] = np.transpose(rl['action_train_to_exec'][it])
    else:
        rl['action_selection_its'] = 1
        rl['action_train_to_exec'] = np.ones(
            (rl['action_selection_its'], syst['n_homes'], syst['n_homes_test'])
        )
        rl['state_exec_to_train'] = np.ones(
            (rl['action_selection_its'], syst['n_homes_test'], syst['n_homes'])
        )

    if rl['type_learning'] == 'facmac':
        for info in ['action_train_to_exec', 'state_exec_to_train']:
            rl[info] = th.Tensor(rl[info])

    if 'n_homes_testP' not in syst:
        syst['n_homes_testP'] = syst['n_homesP']

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
    for n_homes in ['n_homes', 'n_homes_test']:
        rl[n_homes] = syst[n_homes]
    rl = _make_type_eval_list(rl)
    rl = _format_rl_parameters(rl)
    rl = _expand_grdC_states(rl)
    rl = _remove_states_incompatible_with_trajectory(rl)
    if 'min_voltage' in rl['state_space']:
        prm['grd']['simulate_panda_power_only'] = True

    reactive_power_for_voltage_control = prm["grd"]["reactive_power_for_voltage_control"]

    _dims_states_actions(rl, syst, reactive_power_for_voltage_control)

    # learning parameter variables
    rl["ncpu"] = mp.cpu_count() if syst["server"] else 10
    rl['episode_limit'] = 0 if rl['trajectory'] else syst['N']
    rl["tot_learn_cycles"] = rl["n_epochs"] * rl["ncpu"] \
        if rl["parallel"] else rl["n_epochs"]
    prm["RL"]["type_env"] = rl["type_learn_to_space"][rl["type_learning"]]
    rl["start_end_eval"] = min(
        int(rl["share_epochs_start_end_eval"] * rl["n_epochs"]),
        rl['n_epochs'] - 1
    )
    rl["n_all_epochs"] = rl["n_epochs"] + rl["n_end_test"]
    if rl["type_learning"] == "DDPG":
        rl["instant_feedback"] = True

    # if syst['run_mode'] == 1:
    for ext in syst['n_homes_extensions_all']:
        rl["default_action" + ext] = np.full(
            (syst["n_homes" + ext], rl["dim_actions"]), rl["default_action"]
        )

    if prm["grd"]["reactive_power_for_voltage_control"]:
        reactive_power_default = rl["default_action"][0][2] * prm['grd']['active_to_reactive_flex']
        for default_action in rl["default_action"]:
            default_action[3] = reactive_power_default

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

    rl_apply_n_homes_test(syst, rl)

    return rl


def opt_res_seed_save_paths(prm):
    """
    Get strings and seeds which will be used to identify runs.

    inputs:
    prm:
        here the rl, heat, syst, ntw, paths, car entries are relevant

    output:
    rl, paths with updated entries

    """
    rl, heat, syst, grd, paths, car, loads = \
        [prm[key] for key in ["RL", "heat", "syst", "grd", "paths", "car", "loads"]]

    paths = utils.get_opt_res_file(prm)

    if os.path.exists(paths["seeds_file"]):
        rl["seeds"] = np.load(paths["seeds_file"], allow_pickle=True).item()
        if "_test" not in rl['seeds']:
            rl['seeds']['_test'] = []
    else:
        rl["seeds"] = {ext: [] for ext in syst['n_homes_extensions_all']}
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
    paths, grd, syst, car = [prm[key] for key in ["paths", "grd", "syst", "car"]]

    # grid loss
    grd["loss"] = grd["R"] / (grd["V"] ** 2)
    grd['per_unit_to_kW_conversion'] = grd['base_power'] / 1000
    grd['kW_to_per_unit_conversion'] = 1000 / grd['base_power']
    grd['active_to_reactive_flex'] = math.tan(math.acos(car['pf_flexible_homes']))
    grd['active_to_reactive_passive'] = math.tan(math.acos(car['pf_passive_homes']))

    # wholesale
    for test_str in ["", "_test"]:
        # p/kWh -> £/kWh (nordpool was EUR/MWh so was * 1e-3)
        grd[f"wholesale_all{test_str}"] = [
            x * 1e-2 for x in np.load(paths["open_inputs"] / paths[f"wholesale{test_str}_file"])
        ]

        # gCO2/kWh to tCO2/kWh
        grd[f"cintensity_all{test_str}"] = np.load(
            paths["open_inputs"] / paths[f"carbon_intensity{test_str}_file"], allow_pickle=True
        ) * 1e-6
        # carbon intensity
        grd[f"Call{test_str}"] = [
            price + carbon * syst["co2tax"]
            for price, carbon in zip(
                grd[f"wholesale_all{test_str}"], grd[f"cintensity_all{test_str}"]
            )
        ]
        grd[f"perc{test_str}"] = [np.percentile(grd[f"Call{test_str}"], i) for i in range(0, 101)]

    if grd['compare_pandapower_optimisation'] and not grd['manage_voltage']:
        # comparison between optimisation and pandapower is only relevant if simulating voltage.
        grd['compare_pandapower_optimisation'] = False

    if grd['manage_voltage']:
        grd['penalise_individual_exports'] = False
    else:
        grd['reactive_power_for_voltage_control'] = False


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
    syst['n_homes_all'] = syst['n_homes'] + syst['n_homesP']
    if syst['n_homes_test'] is None:
        syst['n_homes_test'] = syst['n_homes']
    syst['n_homes_all_test'] = syst['n_homes_test'] + syst['n_homesP']
    assert syst['n_homes_all'] > 0, "No homes in the system"

    syst['n_homes_extensions'] = ["P"]
    syst['test_different_to_train'] = (
        syst['n_homes_test'] != syst['n_homes']
        or syst['year_test'] != syst['year']
    )
    if syst['test_different_to_train']:
        syst['n_homes_extensions'].append("_test")
    syst["n_homes_extensions_all"] = syst['n_homes_extensions'] + [""]
    syst['timestamp'] = datetime.datetime.now().timestamp()
    syst['share_active_test'] = syst['n_homes_test'] / syst['n_homes_all_test']
    syst['interval_to_month'] = prm['syst']['H'] * 365 / 12

    syst['day_trans'] = []
    for prev_day in syst["weekday_types"]:
        for next_day in syst["weekday_types"]:
            syst['day_trans'].append(f"{prev_day}2{next_day}")

    for test_str in ['', '_test']:
        syst[f'date0{test_str}'] = [syst[f'year{test_str}'], syst['month0'], 1, 0]
        syst[f'max_date_end{test_str}'] = [syst[f'year{test_str}'], syst['month_end'], 1, 0]
        # general system parameters
        for info in ["date0", "max_date_end"]:
            prm["syst"][f"{info}{test_str}_dtm"] = datetime.datetime(
                *prm["syst"][f"{info}{test_str}"]
            )


def _homes_info(loads, syst, gen, heat, car):
    for ext in syst['n_homes_extensions_all']:
        gen["own_PV" + ext] = np.ones(syst["n_homes" + ext]) \
            if isinstance(gen["own_PV" + ext], (int, float)) and gen["own_PV" + ext] == 1 \
            else gen["own_PV" + ext]
        heat["own_heat" + ext] = np.ones(syst["n_homes" + ext]) * heat["own_heat" + ext] \
            if isinstance(heat["own_heat" + ext], int) \
            else np.array(heat["own_heat" + ext])
        car["own_car" + ext] = np.ones(syst["n_homes" + ext]) * car["own_car" + ext] \
            if isinstance(car["own_car" + ext], int) \
            else np.array(car["own_car" + ext])
        for ownership in ["own_loads" + ext, "own_flex" + ext]:
            if ownership in loads:
                loads[ownership] = np.ones(syst["n_homes" + ext]) * loads[ownership] \
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
    [paths, syst, loads, gen, save, heat, car] = [
        prm[key] if key in prm else None
        for key in ["paths", "syst", "loads", "gen", "save", "heat", "car"]
    ]

    if paths is not None:
        paths = _update_paths(paths, prm, no_run)
    _syst_info(prm)
    _homes_info(loads, syst, gen, heat, car)

    # update paths and parameters from inputs
    if paths is not None and initialise_all:
        syst = _load_data_dictionaries(paths, syst)
        _update_grd_prm(prm)
        loads["share_flex"], loads["max_delay"] = loads["flex"]
        for ext in syst['n_homes_extensions_all']:
            loads["share_flexs" + ext] = [
                0 if not loads["own_flex" + ext][home] else loads["share_flex"]
                for home in range(syst["n_homes" + ext])
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
        return rl

    valid_types = {
        "exploration": ["env_r_c", "opt", "baseline"],
        "evaluation": ["env_r_c", "opt_r_c", "opt", "baseline", "random"]
    }

    for stage in ["evaluation", "exploration"]:
        new_methods = []
        for method in rl[f"{stage}_methods"]:
            new_method = method
            if len(method.split("_")) == 3:
                if utils.distr_learning(method) == "d":
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

    return rl


def _filter_type_learning_competitive(rl):
    if rl["competitive"]:  # there can be no centralised learning
        rl["evaluation_methods"] = [
            method for method in rl["evaluation_methods"]
            if method in ["opt", "baseline"]
            or (utils.reward_type(method) != "A" and utils.distr_learning(method) != "c")
        ]

    return rl


def _add_n_start_opt_explo(rl, evaluation_methods_list):
    if rl['n_start_opt_explo'] is not None and rl['n_start_opt_explo'] > 0:
        for i, initial_evaluation_method in enumerate(evaluation_methods_list):
            if initial_evaluation_method[0: 3] == 'env' and rl['n_start_opt_explo'] > 0:
                evaluation_methods_list[i] += f"_{rl['n_start_opt_explo']}_opt"

    return evaluation_methods_list


def _filter_type_evals_no_active_homes(rl):
    if rl['n_homes'] == 0:
        rl["evaluation_methods"] = [
            method for method in rl["evaluation_methods"]
            if method in ["random", "baseline"]
        ]

    return rl


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
    rl = _filter_type_learning_competitive(rl)

    rl["exploration_methods"] = [
        method for method in rl["evaluation_methods"]
        if not method.startswith("opt")
    ]

    if sum(method.startswith("opt") and len(method) > 3 for method in rl["evaluation_methods"]) > 0:
        rl["exploration_methods"] += ['opt']

    rl["eval_action_choice"] = [
        method for method in rl["evaluation_methods"] if method not in ["baseline", "opt"]
    ]
    # assert len(rl["eval_action_choice"]) > 0, \
    #     "not valid eval_type with action_choice"

    rl = _filter_type_learning_facmac(rl)
    rl = _filter_type_evals_no_active_homes(rl)

    rl["type_Qs"] \
        = rl["eval_action_choice"] + [
        ac + "0" for ac in rl["eval_action_choice"]
        if len(ac.split("_")) >= 3 and (
            utils.reward_type(ac) == "A" or utils.distr_learning(ac)[0] == "C"
        )
    ]

    print(f"rl['evaluation_methods'] {rl['evaluation_methods']}")

    return rl
