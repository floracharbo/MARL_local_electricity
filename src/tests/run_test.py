import pickle
import shutil
from datetime import timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.simulations.runner import run
from src.utilities.userdeftools import current_no_run, set_seeds_rdn

I0_COSTS = 288

def random_True_False(colours, colour, min_diffs=None):
    """Return True or False with equal probability."""
    return np.random.random() > 0.5


def patch_find_feasible_data(
        self,
        seed_ind: int,
        type_actions: list,
        step_vals: dict,
        evaluation: bool,
        epoch: int,
        passive: bool = False
):
    self.prm['syst']['jax_random_key'] = set_seeds_rdn(0)
    names_files = {}
    files = ['res', 'batch']
    for file in files:
        names_files[file] = f"{file}_test{self.prm['paths']['opt_res_file']}"
    res, batch = [
        np.load(self.paths['test_data'] / names_files[file], allow_pickle=True).item() for file in files
    ]
    if 'house_cons' not in res:
        res['house_cons'] = res['totcons'] - res['E_heat']
    for file in files:
        shutil.copyfile(
            self.paths['test_data'] / names_files[file],
            self.prm['paths']['opt_res'] / names_files[file]
        )

    self.res_name = names_files['res']
    self.batch_file, batch = self.env.reset(
        seed=0, load_data=True, passive=False, evaluation=evaluation
    )
    data_feasibles = self._format_data_optimiser(
        batch, passive=passive, test=evaluation
    )
    data_feasible = True
    pp_simulation_required = False
    if data_feasible and 'opt' in type_actions:  # start with opt
        # exploration through optimisation
        step_vals, data_feasible = self.get_steps_opt(
            res, pp_simulation_required, step_vals, evaluation, batch, epoch
        )

    seed_data = res, batch

    return seed_data, step_vals


def patch_update_date(self, i0_costs, date0=None):
    self.i0_costs = I0_COSTS
    n_days = I0_COSTS/24
    self.date0 = self.prm['syst']['date0_dtm'] + timedelta(days=n_days)
    self.action_translator.date0 = self.date0
    self.date_end = self.date0 + timedelta(hours=self.N)
    self.car.date0 = self.date0
    self.car.date_end = self.date_end


def patch_file_id(self):
    file_extension = f"_test{self.prm['paths']['opt_res_file']}"

    return file_extension

def patch_set_i0_costs(self, i0_costs):
    self.i0_costs = I0_COSTS

def patch_set_date(
        self, repeat, epoch, i_explore,
        date0, delta, i0_costs, new_env
):
    self.prm['syst']['jax_random_key'] = set_seeds_rdn(0)
    delta_days = I0_COSTS/24
    date0 = self.prm['syst']['date0_dtm'] \
        + timedelta(days=delta_days)
    # self.prm['syst']['current_date0_dtm'] = date0
    delta = date0 - self.prm['syst']['date0_dtm']
    i0_costs = int(delta.days * 24 + delta.seconds / 3600)
    self.prm['grd']['C'] = \
        self.prm['grd']['Call'][
        i0_costs: i0_costs + self.prm['syst']['N'] + 1]
    self.explorer.i0_costs = i0_costs
    self.env.update_date(i0_costs, date0)

    return date0, delta, i0_costs


def patch_load_data_dictionaries(paths, syst):
    syst['n_clus'] = {'loads': 4, 'car': 4}
    return syst


def patch_init_factors_profiles_parameters(self, prm):
    self.perc = {
        'grd': prm['grd']['perc']
    }


def patch_load_profiles(prm):
    profiles = {}
    return profiles


def patch_load_bat_factors_parameters(paths, car):
    return car


def patch_compute_max_car_cons_gen_values(env, state_space):
    labels = [
        "max_car_cons", "max_normcons", "max_normgen", "max_car_dem_agg"
    ]
    return [
        np.load(env.prm['paths']['test_data'] / f"{label}_test.npy")
        for label in labels
    ]


def patch_load_input_data(self, prm, other_prm, factors0, clusters0):
    test_data_path = prm['paths']['test_data']

    for info in ['f_min', 'f_max', 'n_clus']:
        file_path = test_data_path / f"{info}.pickle"
        with open(file_path, "rb") as file:
            setattr(self, info, pickle.load(file))
    prm['n_clus'] = self.n_clus
    self.clusters = {
        'loads': np.zeros(self.n_homes),
        'car': np.zeros(self.n_homes),
    }
    self.factors = {
        'loads': np.ones(self.n_homes) * 9,
        'car': np.ones(self.n_homes) * 8,
        'gen': np.ones(self.n_homes) * 8,
    }
    self.list_clusters = {
        'loads': np.zeros((self.n_homes, 2)),
        'car': np.zeros((self.n_homes, 2)),
    }
    self.list_factors = {
        'loads': np.ones((self.n_homes, 2)) * 9,
        'car': np.ones((self.n_homes, 2)) * 8,
        'gen': np.ones((self.n_homes, 2)) * 8,
    }

    return prm


def patch_plot_compare_all_signs(
    prm, colours_barplot_baseentries, eval_entries_notCd,
        m_, ave, lower_bound, upper_bound, m
):
    pass


def test_all(mocker):
    settings = {
        'RL': {
            # current experiment
            'batch_size': 2,
            'state_space': [['grdC']],
            'n_epochs': 5,
            'n_repeats': 2,
        },
        'syst': {
            'test_on_run': True,
            'n_homes': 4
        },
        'grd': {
            'max_grid_in': 5,
            'max_grid_out': 5,
            'penalty_coefficient_in': 0.001,
            'penalty_coefficient_out': 0.001,
            'manage_agg_power': True,
            'max_grid_import': 13,
            'max_grid_export': 13,
            'penalty_import': 0.01,
            'penalty_export': 0.01,
            'manage_voltage': True,
            'penalty_overvoltage': 0.1, 
            'penalty_undervoltage': 0.1,
            'max_voltage': 1.001,
            'min_voltage': 0.999,
            'weight_network_costs': 1,
            'subset_line_losses_modelled': 30
        },
        'heat': {
            'own_heat': [[1, 0, 1, 1]],
        },
        'loads': {
            'own_loads': [[1, 1, 0, 1]],
        },
        'car': {
            'own_car': [[1, 1, 1, 0]],
        },
    }
    run_mode = 1

    mocker.patch(
        "src.simulations.data_manager.DataManager.find_feasible_data",
        side_effect=patch_find_feasible_data,
        autospec=True
    )
    mocker.patch(
        "src.simulations.local_elec.LocalElecEnv.update_date",
        side_effect=patch_update_date,
        autospec=True
    )
    mocker.patch(
        "src.simulations.local_elec.LocalElecEnv._file_id",
        side_effect=patch_file_id,
        autospec=True
    )
    mocker.patch(
        "src.simulations.hedge.HEDGE._load_input_data",
        side_effect=patch_load_input_data,
        autospec=True
    )
    mocker.patch(
        "src.simulations.local_elec.LocalElecEnv.set_i0_costs",
        side_effect=patch_set_i0_costs,
        autospec=True
    )
    mocker.patch(
        "src.simulations.runner.Runner._set_date",
        side_effect=patch_set_date,
        autospec=True
    )
    mocker.patch(
        "src.initialisation.initialise_objects._load_data_dictionaries",
        side_effect=patch_load_data_dictionaries
    )
    mocker.patch(
        "src.simulations.hedge.HEDGE._load_profiles",
        side_effect=patch_load_profiles
    )
    mocker.patch(
        "src.initialisation.initialise_objects._load_bat_factors_parameters",
        side_effect=patch_load_bat_factors_parameters
    )
    mocker.patch(
        "src.utilities.env_spaces.EnvSpaces._init_factors_profiles_parameters",
        side_effect=patch_init_factors_profiles_parameters,
        autospec=True
    )
    mocker.patch(
        "src.utilities.env_spaces.compute_max_car_cons_gen_values",
        side_effect=patch_compute_max_car_cons_gen_values
    )
    mocker.patch(
        "src.post_analysis.plotting.plot_rl_performance_metrics._plot_compare_all_signs",
        patch_plot_compare_all_signs
    )

    paths_results = Path("outputs") / "results"
    prev_no_run = None
    for type_learning in ['facmac', 'q_learning']:
        settings['RL']['type_learning'] = type_learning
        for aggregate_actions in [True,  False]:
            settings['RL']['aggregate_actions'] = aggregate_actions
            for trajectory in [True, False]:
                if type_learning == 'q_learning' and trajectory:
                    break
                settings['RL']['trajectory'] = trajectory
                print(f"test {type_learning} aggregate_actions {aggregate_actions} trajectory {trajectory}")
                no_run = current_no_run(paths_results)

                if prev_no_run is not None:
                    assert no_run == prev_no_run + 1, "results not saving"
                run(run_mode, settings)
                prev_no_run = no_run

