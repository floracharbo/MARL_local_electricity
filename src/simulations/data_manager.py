"""
Created on Thu Aug 11 2023.

@author: floracharbonnier

Object DataManager generating, checking, formatting data for explorations.

Methods are:
- __init__: Add relevant information to the properties of the object.
- loop_replace_data: Replace the data for infeasible homes.
- load_res: Load pre-saved day data.
- find_feasible_data: For a new day of exploration or evaluation,
generate feasible data.
- make_data: Make and save or load pre-saved factors, clusters and batch data
- format_data_optimiser: Turn input data into usable format
for optimisation problem.
- get_seed: Given the seed_ind, compute the random seed.
- get_seed_ind: Obtain the rank of the current seed.
- file_id: Generate string to identify the run in saved files.
"""

import copy
import glob
import os
import traceback
from typing import List, Optional, Tuple

import numpy as np

from src.simulations.optimisation import Optimiser
from src.utilities.userdeftools import set_seeds_rdn


class DataManager():
    """Generating, checking, formatting data for explorations."""

    def __init__(self, env: object, prm: dict, explorer: object):
        """Add relevant information to the properties of the object."""
        self.env = env
        self.prm = prm
        self.optimiser = Optimiser(prm)
        self.get_steps_opt = explorer.get_steps_opt

        self.paths = prm['paths']
        self.rl = prm['RL']
        self.check_feasibility_with_opt = self.rl['check_feasibility_with_opt']
        self.deterministic_created = False

        self.seeds = prm['RL']['seeds']
        # d_seed is the difference between the rank of the n-th seed (n)
        # and the n-th seed value (e.g. the 2nd seed might be = 3, d_seed = 1)
        self.d_seed = {}
        for passive_ext in ['P', '']:
            self.d_seed[passive_ext] = self.seeds[passive_ext][-1] - \
                (len(self.seeds[passive_ext]) - 1) if len(self.seeds[passive_ext]) > 0 else 0

        # d_ind_seed is the difference between the current
        # ind_seed from the seed multiplier value and epoch, repeat and
        # how many i have thrown out because it was infeasible during training
        # instead of looking at the n-th seed, look at the(n+d_ind_seed)th seed
        self.d_ind_seed = {'P': 0, '': 0}

    def format_grd(self, batch, passive_ext):
        """Format network parameters in preparation for optimisation."""
        grd, loads, syst, car = [
            self.prm[data_file] for data_file in ['grd', 'loads', 'syst', 'car']
        ]
        potential_delay = np.zeros((loads['n_types'], syst['N']), dtype=int)
        if loads['flextype'] == 1:
            potential_delay[0] = np.zeros(syst['N'])
            for time in range(syst['N']):
                potential_delay[1, time] = max(min(loads['flex'][1], syst['N'] - 1 - time), 0)
        else:
            for load_type in range(loads['n_types']):
                for time in range(syst['N']):
                    potential_delay[load_type][time] = max(
                        min(loads['flex'][load_type], syst['N'] - 1 - time), 0)

        # make grd matrices
        grd['Bcap'] = np.zeros((syst['n_homes' + passive_ext], syst['N']))
        grd['loads'] = np.zeros((loads['n_types'], syst['n_homes' + passive_ext], syst['N']))
        grd['flex'] = np.zeros(
            (syst['N'], loads['n_types'], syst['n_homes' + passive_ext], syst['N']))

        grd['gen'] = np.zeros((syst['n_homes' + passive_ext], syst['N'] + 1))
        share_flexs = loads['share_flexs' + passive_ext]
        for home in range(syst['n_homes' + passive_ext]):
            grd['gen'][home] = batch[home]['gen'][0: len(grd['gen'][home])]
            for time in range(syst['N']):
                grd['Bcap'][home, time] = car['cap' + passive_ext][home]
                for load_type in range(loads['n_types']):
                    grd['loads'][0][home][time] \
                        = batch[home]['loads'][time] * (1 - share_flexs[home])
                    grd['loads'][1][home][time] \
                        = batch[home]['loads'][time] * share_flexs[home]
                    for time_cons in range(syst['N']):
                        if time <= time_cons <= time + int(potential_delay[load_type][time]):
                            grd['flex'][time, load_type, home, time_cons] = 1

        return grd

    def _passive_find_feasible_data(self):
        passive = True

        file_id_path = self.paths['opt_res'] / f"batch{self.file_id()}"
        if file_id_path.is_file():
            # [factors, clusters] = self._load_res()
            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.passive_ext], load_data=True,
                passive=passive)
        else:
            # presave data to be used for multiple methods
            batch = self._env_make_data(
                int(self.seed[self.passive_ext]), passive=passive)

        # turn input data into usable format for optimisation problem
        data_feasibles = self._format_data_optimiser(batch, passive=passive)
        if not all(data_feasibles):
            batch, data_feasibles = self._loop_replace_data(data_feasibles, passive)

        data_feasible = all(data_feasibles)
        res = None
        new_res = False
        seed_data = res, batch

        return seed_data, new_res, data_feasible

    def _check_data_computations_required(self, type_actions, feasibility_checked):
        opt_needed = 'opt' in type_actions \
                     or (not feasibility_checked and self.check_feasibility_with_opt)

        if opt_needed:
            file_exists = (self.paths['opt_res'] / self.res_name).is_file()
        else:
            file_exists = (
                self.paths['opt_res'] / f"batch{self.file_id()}"
            ).is_file()

        new_data_needed = self.rl['deterministic'] == 2 or not file_exists

        return opt_needed, new_data_needed

    def _active_find_feasible_data(
            self,
            type_actions: List[str],
            feasibility_checked: bool,
            step_vals: dict,
            evaluation: bool,
            epoch: int,
    ) -> Tuple[list, bool, bool, dict, Optional[list], bool]:
        passive = False
        data_feasible = True
        new_res = False
        res = None

        opt_needed, new_data_needed = self._check_data_computations_required(
            type_actions, feasibility_checked
        )

        # check if data is feasible by solving optimisation problem
        if new_data_needed:
            # pre-save data to be used for multiple methods
            batch = self._env_make_data(
                int(self.seed[self.passive_ext]),
                passive=passive
            )
            assert batch[0]['loads'][0] == self.env.batch[0]['loads'][0]

        else:
            if opt_needed:
                res = np.load(self.paths['opt_res']
                              / self.res_name,
                              allow_pickle=True).item()
            # [factors, clusters] = self._load_res()
            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.passive_ext],
                load_data=True, passive=passive)
            new_res = False
        # turn input data into optimisation problem format
        data_feasibles = self._format_data_optimiser(batch, passive=passive)

        if not all(data_feasibles):
            batch, data_feasibles = self._loop_replace_data(
                data_feasibles, passive
            )
            feasibility_checked = False

        if all(data_feasibles) and opt_needed and new_data_needed:
            try:
                res = self.optimiser.solve(self.prm)
            except Exception as ex:  # if infeasible, make new data
                if str(ex)[0:6] != 'Code 3':
                    print(traceback.format_exc())
                    print(f'ex.args {ex.args}')
                data_feasible = False
            new_res = True

        if data_feasible and 'opt' in type_actions:  # start with opt
            # exploration through optimisation
            step_vals, data_feasible = self.get_steps_opt(
                res, step_vals, evaluation, batch, epoch
            )

        seed_data = [res, batch]

        return (seed_data, new_res, data_feasible, step_vals, feasibility_checked)

    def find_feasible_data(
            self,
            seed_ind: int,
            type_actions: list,
            step_vals: dict,
            evaluation: bool,
            epoch: int,
            passive: bool = False
    ) -> Tuple[list, dict, Optional[list]]:
        """
        For a new day of exploration or evaluation, generate feasible data.

        Check whether data has already been generated, saved, and checked
        and perform these actions accordingly.
        Depending on inputs, check feasibility with optimisation or heuristic
        checks only.

        Returns:
        res: optimisation results
        batch: load, EV and PV batch of data
        factors: scaling factors for the load, PV and EV load profiles
        clusters: behaviour clusters for the load and EV profiles
        step_vals: RL exploration/evaluation data (state, action, rewards, etc)
        """
        # boolean: whether optimisation problem is feasible;
        # start by assuming it is not
        data_feasible = 0
        iteration = -1
        while not data_feasible and iteration < 100:
            # try solving problem else making new data until problem solved
            iteration += 1
            feasibility_checked = self.get_seed(seed_ind)
            set_seeds_rdn(self.seed[self.passive_ext])
            self.res_name = \
                f"res_P{int(self.seed['P'])}_" \
                f"{int(self.seed[''])}{self.prm['paths']['opt_res_file']}"

            if passive:
                seed_data, new_res, data_feasible \
                    = self._passive_find_feasible_data()
            else:
                [seed_data, new_res, data_feasible, step_vals,
                 feasibility_checked] = self._active_find_feasible_data(
                    type_actions, feasibility_checked, step_vals,
                    evaluation, epoch
                )

            if not data_feasible:
                seed_ind = self.infeasible_tidy_files_seeds(seed_ind)

        if not feasibility_checked:
            self.seeds[self.passive_ext] = np.append(
                self.seeds[self.passive_ext], self.seed[self.passive_ext])

        if new_res:
            np.save(self.paths['opt_res'] / self.res_name, seed_data[0])

        return seed_data, step_vals

    def get_seed(self, seed_ind: int) -> bool:
        """Given the seed_ind, compute the random seed."""
        if seed_ind < len(self.seeds[self.passive_ext]):
            feasibility_checked = True
            self.seed[self.passive_ext] = self.seeds[self.passive_ext][seed_ind]
        else:
            self.seed[self.passive_ext] = seed_ind + self.d_seed[self.passive_ext]
            feasibility_checked = False

        return feasibility_checked

    def infeasible_tidy_files_seeds(self, seed_ind):
        """If data is infeasible, update seeds and remove saved files."""
        if seed_ind < len(self.seeds[self.passive_ext]):
            self.d_ind_seed[self.passive_ext] += 1
            seed_ind += 1
        else:
            for seeded_data in ['factors', 'clusters', 'batch']:
                file_names = glob.glob(
                    str(self.paths['opt_res']
                        / f"{seeded_data}{self.file_id()}"))
                for file_name in file_names:
                    os.remove(file_name)
            file_names = glob.glob(
                str(self.paths['opt_res'] / self.res_name))
            for file_name in file_names:
                os.remove(file_name)
            self.d_seed[self.passive_ext] += 1

        return seed_ind

    def get_seed_ind(self, repeat: int, epoch: int, i_explore: int) -> int:
        """
        Obtain the rank of the current seed.

        returns seed_ind;
        the random seed can then be selected
        as the seed_ind-th of the list of valid random seeds
        """
        if epoch < self.rl['n_epochs']:
            seed_ind = \
                repeat * (
                    self.rl['n_epochs'] * (self.rl['n_explore'] + 1)
                    + self.rl['n_end_test']
                ) + epoch * (self.rl['n_explore'] + 1) + i_explore
        if epoch >= self.rl['n_epochs']:
            seed_ind = \
                repeat * (self.rl['n_epochs'] * (self.rl['n_explore'] + 1)
                          + self.rl['n_end_test']) \
                + (self.rl['n_epochs'] * (self.rl['n_explore'] + 1)) \
                + (epoch - self.rl['n_epochs'])

        return seed_ind

    def file_id(self):
        """Generate string to identify the run in saved files."""
        return f"{int(self.seed[self.passive_ext])}{self.passive_ext}" \
               f"{self.prm['paths']['opt_res_file']}"

    def _loop_replace_data(
            self,
            data_feasibles: np.ndarray,
            passive: bool,
            # factors: dict,
            # clusters: dict
    ) -> Tuple[dict, dict, dict, np.ndarray]:
        """Replace the data for infeasible homes."""
        its = 0
        homes_0 = [i for i, ok in enumerate(data_feasibles) if not ok]

        homes = copy.deepcopy(homes_0)
        while not all(data_feasibles) and its < 100:
            self.env.dloaded = 0
            self.env.fix_data_a(homes, self.file_id(), its=its)
            # [factors, clusters] = self._load_res()

            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.passive_ext], load_data=True, passive=passive
            )

            assert all(
                len(batch[home]['loads']) == len(batch[0]['loads']) for home in homes
            ), f"len loads= {[len(batch[home]['loads']) for home in homes]}"

            # turn input data into usable format for optimisation problem
            data_feasibles = self._format_data_optimiser(
                batch, passive=passive
            )
            homes = [i for i, ok in enumerate(data_feasibles) if not ok]
            if its > 50:
                print(f'its {its}, sum(data_feasibles) {sum(data_feasibles)}')
                print(f'infeasibles homes = {homes}')
                for home in homes:
                    print(f'home = {home} in infeasibles')
            its += 1

        return batch, data_feasibles

    def _load_res(self, labels: list = ['factors', 'clusters']) -> List[dict]:
        """Load pre-saved day data."""
        files = [
            np.load(
                self.paths['opt_res'] / f"{label}{self.file_id()}",
                allow_pickle=True
            ).item() for label in labels
        ]

        return files

    def _env_make_data(
            self,
            seed: int,
            passive: bool = False
    ) -> List[dict]:
        """
        Instruct env to load/ make data given deterministic. Save and clean up.

        Make and save or load pre-saved factors, clusters and batch data.
        If we have already saved the deterministic environment set-up,
        load it and reset environment accordingly.

        Else, generate random data.
        """
        if self.rl['deterministic'] == 1 and self.deterministic_created:
            # deterministic
            load_data = True
            batch_file, batch = self.env.reset(
                seed=seed, load_data=load_data, passive=passive
            )

        elif self.rl['deterministic'] > 0 and not self.deterministic_created:
            # rl['deterministic'] 1: deterministic / 2: deterministic noisy
            load_data = False
            batch_file, batch = self.env.reset(
                seed=seed, load_data=load_data, passive=passive)
            np.save(f'deterministic_prms_seedind{self.ind_seed_deterministic}',
                    [batch_file, batch])
            self.deterministic_created = True

        elif self.rl['deterministic'] == 2 and self.deterministic_created:
            # deterministic noisy
            load_data = True
            batch_file, batch = self.env.reset(
                seed=seed, load_data=load_data,
                add_noise=True, passive=passive)

        else:
            # inderterministic
            # re-initialise and save data for future deterministic runs
            # do not load data for env reset - instead make new data
            load_data = False
            batch_file, batch = self.env.reset(
                seed=seed, load_data=load_data, passive=passive
            )

        # for subsequent methods, on reinitialisation, it will use this data
        # obtain data needed saved in batchfile
        [batch] = self._load_res(labels=['batch'])

        new_batch_file = 'batch_file' in self.__dict__ \
                         and batch_file != self.batch_file
        if not self.prm['save']['plotting_batch'] \
                and self.rl['deterministic'] == 0 \
                and new_batch_file:
            files = os.listdir(os.getcwd())
            for file in files:
                if file[0:len(self.batch_file)] == self.batch_file:
                    os.remove(file)
        self.batch_file = batch_file

        return batch

    def _format_data_optimiser(
            self,
            batch: dict,
            passive: bool = False
    ) -> np.ndarray:
        """Turn input data into usable format for optimisation problem."""
        # initialise dicts
        grd, loads, syst, car, heat = [
            self.prm[data_file] for data_file in ['grd', 'loads', 'syst', 'car', 'heat']
        ]
        passive_ext = 'P' if passive else ''

        # format battery info
        bat_entries = ['avail_car', 'loads_car']
        for bat_entry in bat_entries:
            car['batch_' + bat_entry] = np.zeros((syst['n_homes' + passive_ext], syst['N'] + 1))

        bat_entries += ['bat_dem_agg']
        car['bat_dem_agg'] = np.zeros((syst['n_homes' + passive_ext], syst['N'] + 1))
        for home in range(syst["n_homes" + passive_ext]):
            for info in self.env.car.batch_entries:
                car['batch_' + info][home] = \
                    batch[home][info][0: len(car['batch_' + info][home])]

        loads['n_types'] = 2

        self.format_grd(batch, passive_ext)

        feasible = self.env.car.check_feasible_bat(self.prm, passive_ext)

        heat['T_out'] = self.env.heat.T_out

        return feasible

    def update_flexibility_opt(self, batchflex_opt, res, time_step):
        """Update available flexibility based on optimisation results."""
        n_homes = len(res["E_heat"])
        cons_flex_opt = \
            [res["totcons"][home][time_step] - batchflex_opt[home][time_step][0]
             - res["E_heat"][home][time_step] for home in range(n_homes)]
        inputs_update_flex = \
            [time_step, batchflex_opt, self.prm["loads"]["max_delay"], n_homes]
        new_batch_flex = self.env.update_flex(
            cons_flex_opt, opts=inputs_update_flex)
        for home in range(n_homes):
            batchflex_opt[home][time_step: time_step + 2] = new_batch_flex[home]

        assert batchflex_opt is not None, "batchflex_opt is None"

        return batchflex_opt
