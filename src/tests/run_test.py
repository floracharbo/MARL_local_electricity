import pickle
from datetime import timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from src.initialisation.generate_colours import (_check_colour_diffs,
                                                 generate_colours)
from src.simulations.runner import run
from src.utilities.userdeftools import current_no_run, set_seeds_rdn


def random_True_False(colours, colour, min_diffs=None):
    """Return True or False with equal probability."""
    return np.random.random() > 0.5


def test_generate_colours(mocker):
    colours_only = True
    all_type_eval = [
        'env_r_c', 'env_r_d', 'env_A_c', 'env_A_d', 'env_d_c', 'env_d_d',
        'opt_r_c', 'opt_r_d', 'opt_A_c', 'opt_A_d', 'opt_d_c', 'opt_d_d',
        'opt_n_c', 'opt_n_d', 'facmac'
    ]
    save, prm = None, None

    patched_check_colour_diffs = mocker.patch(
        "src.initialisation.generate_colours._check_colour_diffs",
        side_effect=random_True_False
    )

    colours = generate_colours(save, prm, entries=all_type_eval, colours_only=colours_only)

    assert len(colours) == len(all_type_eval), \
        "generate colours lengths do not match " \
        f"{len(colours)} != {len(all_type_eval)}"


def patch_find_feasible_data(
        self,
        seed_ind: int,
        type_actions: list,
        step_vals: dict,
        evaluation: bool,
        epoch: int,
        passive: bool = False
):
    set_seeds_rdn(0)
    self.res_name = "res_test.npy"
    res = np.load(self.paths['opt_res']
                  / self.res_name,
                  allow_pickle=True).item()
    cluss = np.load(self.paths['opt_res']
                  / "clusters_test.npy",
                  allow_pickle=True).item()
    factors = np.load(self.paths['opt_res']
                  / "factors_test.npy",
                  allow_pickle=True).item()
    self.batch_file, batch = self.env.reset(
            seed=0,
            load_data=True, passive=False)
    data_feasibles = self._format_data_optimiser(
        batch, passive=passive)
    data_feasible = True
    if data_feasible and 'opt' in type_actions:  # start with opt
        # exploration through optimisation
        step_vals, mus_opt, data_feasible = self.get_steps_opt(
            res, step_vals, evaluation, cluss, factors, batch,
            last_epoch=epoch == self.prm['RL']['n_epochs'] - 1)

    seed_data = res, factors, cluss, batch

    return seed_data, step_vals, mus_opt


def patch_update_date(self, i0_costs, date0=None):
    self.i0_costs = 12 * 24
    self.date0 = self.prm['syst']['date0_dtm'] + timedelta(days=12)
    self.action_translator.date0 = self.date0
    self.date_end = self.date0 + timedelta(hours=self.N)
    self.car.date0 = self.date0
    self.car.date_end = self.date_end


def patch_file_id(self):
    return "_test.npy"


def patch_init_i0_costs(self):
    self.i0_costs = 12 * 24
    self.prm["grd"]["C"] = \
        self.prm["grd"]["Call"][self.i0_costs: self.i0_costs + self.prm["syst"]["N"]]
    self.env.update_date(self.i0_costs)


def patch_set_date(
        self, repeat, epoch, i_explore,
        date0, delta, i0_costs, new_env
):
    set_seeds_rdn(0)
    delta_days = 12
    date0 = self.prm['syst']['date0_dtm'] \
        + timedelta(days=delta_days)
    # self.prm['syst']['current_date0_dtm'] = date0
    delta = date0 - self.prm['syst']['date0_dtm']
    i0_costs = int(delta.days * 24 + delta.seconds / 3600)
    self.prm['grd']['C'] = \
        self.prm['grd']['Call'][
        i0_costs: i0_costs + self.prm['syst']['N']]
    self.explorer.i0_costs = i0_costs
    self.env.update_date(i0_costs, date0)

    return date0, delta, i0_costs


def patch_load_data_dictionaries(loads, gen, car, paths, syst):
    return [loads, gen, car]


def patch_init_factors_clusters_profiles_parameters(self, prm):
    self.n_clus = {
        'loads': prm['loads']['n_clus'],
        'car': prm['car']['n_clus']
    }

    for e in ['f_min', 'f_max']:
        with open(f'input_data/open_data_v2/{e}.pickle', 'rb') as file:
            self.__dict__[e] = pickle.load(file)


def patch_init_factors_profiles_parameters(self, env, prm):
    self.perc = {
        'grd': prm['grd']['perc']
    }


def patch_reinitialise_envfactors(
        self, date0, epoch, i_explore, evaluation_add1=False
):
    return


def patch_load_profiles(paths, car, syst, loads, gen):
    profiles = {}
    return profiles, car, loads, gen


def _load_bat_factors_parameters(syst, paths, car):
    return car


def patch_compute_max_EV_cons_gen_values(env):
    labels = [
        "maxEVcons", "max_normcons_hour", "max_normgen_hour"
    ]
    return [
        np.load(f"input_data/open_data_v1/{label}_test.npy")
        for label in labels
    ]


def test_all(mocker):
    settings = {
        'heat': {'file': 'heat2'},

        'RL': {
            # current experiment
            'batch_size': 2,
            'state_space': [['grdC', 'bat_dem_agg', 'avail_EV_step']],
            'n_epochs': 5,
            'n_repeats': 2,
        },

        'ntw': {
            'n': 3
        },
        'syst': {
            'test_on_run': True
        }
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
        "src.simulations.explorer.Explorer._init_i0_costs",
        side_effect=patch_init_i0_costs,
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
        "src.initialisation.initialise_objects._load_profiles",
        side_effect=patch_load_profiles
    )
    mocker.patch(
        "src.initialisation.initialise_objects._load_bat_factors_parameters",
        side_effect=_load_bat_factors_parameters
    )
    mocker.patch(
        "src.simulations.local_elec.LocalElecEnv._init_factors_clusters_profiles_parameters",
        side_effect=patch_init_factors_clusters_profiles_parameters,
        autospec=True
    )
    mocker.patch(
        "src.simulations.local_elec.LocalElecEnv.reinitialise_envfactors",
        side_effect=patch_reinitialise_envfactors,
        autospec=True
    )
    mocker.patch(
        "src.utilities.env_spaces.EnvSpaces._init_factors_profiles_parameters",
        side_effect=patch_init_factors_profiles_parameters,
        autospec=True
    )
    mocker.patch(
        "src.utilities.env_spaces.compute_max_EV_cons_gen_values",
        side_effect=patch_compute_max_EV_cons_gen_values
    )

    paths_results = Path("outputs") / "results"
    prev_no_run = current_no_run(paths_results)
    for type_learning in ['facmac', 'q_learning']:
        settings['RL']['type_learning'] = type_learning
        for aggregate_actions in [True,  False]:
            settings['RL']['aggregate_actions'] = aggregate_actions
            print(f"test {type_learning} aggregate_actions {aggregate_actions}")
            run(run_mode, settings)
            no_run = current_no_run(paths_results)
            assert no_run == prev_no_run + 1, "results not saving"
            prev_no_run = no_run
