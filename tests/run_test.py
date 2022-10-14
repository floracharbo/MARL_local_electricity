from datetime import timedelta
from unittest import mock

import numpy as np
import pytest

from config.generate_colors import _check_color_diffs, generate_colors
from simulations.runner import run
from utilities.userdeftools import set_seeds_rdn


def random_True_False(colors, color, min_diffs=None):
    return np.random.random() > 0.5


def test_generate_colors(mocker):
    colours_only = True
    all_type_eval = [
        'env_r_c', 'env_r_d', 'env_A_c', 'env_A_d', 'env_d_c', 'env_d_d',
        'opt_r_c', 'opt_r_d', 'opt_A_c', 'opt_A_d', 'opt_d_c', 'opt_d_d',
        'opt_n_c', 'opt_n_d', 'facmac'
    ]
    save, prm = None, None

    patched_check_color_diffs = mocker.patch(
        "config.generate_colors._check_color_diffs",
        side_effect=random_True_False
    )

    colors = generate_colors(save, prm, entries=all_type_eval, colours_only=colours_only)

    assert len(colors) == len(all_type_eval), \
        "generate colours lengths do not match " \
        f"{len(colors)} != {len(all_type_eval)}"


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
    res = np.load(self.paths['res_path']
                  / self.res_name,
                  allow_pickle=True).item()
    cluss = np.load(self.paths['res_path']
                  / "cluss_test.npy",
                  allow_pickle=True).item()
    fs = np.load(self.paths['res_path']
                  / "fs_test.npy",
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
            res, step_vals, evaluation, cluss,
            fs, batch, self.seed[self.p],
            last_epoch=epoch == self.prm['RL']['n_epochs'] - 1)

    seed_data = res, fs, cluss, batch

    return seed_data, step_vals, mus_opt

def patch_update_date(self, i0_costs, date0=None):
    self.i0_costs = 12 * 24
    self.date0 = self.prm['syst']['date0'] + timedelta(days=12)
    self.mu_manager.date0 = self.date0
    self.date_end = self.date0 + timedelta(hours=self.N)
    self.bat.date0 = self.date0
    self.bat.date_end = self.date_end

def patch_self_id(self):
    return "_test.npy"

def patch_init_i0_costs(self):
    self.i0_costs = 12 * 24
    self.prm["grd"]["C"] = \
        self.prm["grd"]["Call"][self.i0_costs: self.i0_costs + self.prm["syst"]["N"]]
    self.env.update_date(self.i0_costs)

def patch_set_date(
        self, ridx, epoch, i_explore,
        date0, delta, i0_costs, new_env
):
    set_seeds_rdn(0)
    delta_days = 12
    date0 = self.prm['syst']['date0'] \
            + timedelta(days=delta_days)
    self.prm['syst']['current_date0'] = date0
    delta = date0 - self.prm['syst']['date0_dates']
    i0_costs = int(delta.days * 24 + delta.seconds / 3600)
    self.prm['grd']['C'] = \
        self.prm['grd']['Call'][
        i0_costs: i0_costs + self.prm['syst']['N']]
    self.explorer.i0_costs = i0_costs
    self.env.update_date(i0_costs, date0)

    return date0, delta, i0_costs


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
    }
    run_mode = 1

    mocker.patch(
        "simulations.data_manager.Data_manager.find_feasible_data",
        side_effect=patch_find_feasible_data,
        autospec=True
    )
    mocker.patch(
        "simulations.local_elec.LocalElecEnv.update_date",
        side_effect=patch_update_date,
        autospec=True
    )
    mocker.patch(
        "simulations.local_elec.LocalElecEnv._file_id",
        side_effect=patch_self_id,
        autospec=True
    )
    mocker.patch(
        "simulations.explorer.Explorer._init_i0_costs",
        side_effect=patch_init_i0_costs,
        autospec=True
    )
    mocker.patch(
        "simulations.runner.Runner._set_date",
        side_effect=patch_set_date,
        autospec=True
    )
    for type_learning in ['facmac', 'q_learning']:
        settings['RL']['type_learning'] = type_learning
        for aggregate_actions in [True,  False]:
            settings['RL']['aggregate_actions'] = aggregate_actions
            print(f"test {type_learning} aggregate_actions {aggregate_actions}")
            run(run_mode, settings)
