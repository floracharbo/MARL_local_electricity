"""
Created on Thu Aug 11 2023.

@author: floracharbonnier

Object Data_manager generating, checking, formatting data for explorations.

Methods are:
- __init__: Add relevant information to the properties of the object.
- loop_replace_data: Replace the data for infeasible homes.
- load_res: Load pre-saved day data.
- find_feasible_data: For a new day of exploration or evaluation,
generate feasible data.
- make_data: Make and save or load pre-saved fs, cluss and batch data
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
from typing import List, Tuple

import numpy as np

from simulations.problem import Solver
from utils.userdeftools import play_sound, set_seeds_rdn

list_bool = List[bool]


class Data_manager():
    """Generating, checking, formatting data for explorations."""

    def __init__(self, env: object, prm: dict, explorer: object):
        """Add relevant information to the properties of the object."""
        self.env = env
        self.prm = prm
        self.solver = Solver(prm)
        self.get_steps_opt = explorer.get_steps_opt

        self.paths = prm['paths']
        self.rl = prm['RL']
        self.check_feasibility_with_opt = self.rl['check_feasibility_with_opt']

        self.seeds = prm['RL']['seeds']
        # d_seed is the difference between the rank of the n-th seed (n)
        # and the n-th seed value (e.g. the 2nd seed might be = 3, d_seed = 1)
        self.d_seed = {}
        for p in ['P', '']:
            self.d_seed[p] = self.seeds[p][-1] - \
                (len(self.seeds[p]) - 1) if len(self.seeds[p]) > 0 else 0

        # d_ind_seed is the difference between the current
        # ind_seed from the seed multiplier value and epoch, repeat and
        # how many i have thrown out because it was infeasible during training
        # instead of looking at the n-th seed, look at the(n+d_ind_seed)th seed
        self.d_ind_seed = {'P': 0, '': 0}

    def _passive_find_feasible_data(self):
        passive = True
        data_feasible = True

        file_id_path = self.paths['res_path'] / f"batch{self.file_id()}"
        if file_id_path.is_file():
            fs, cluss = self._load_res()
            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.p], load_data=True,
                passive=passive)
        else:
            # presave data to be used for multiple methods
            [batch, fs, cluss] = self._env_make_data(
                int(self.seed[self.p]), passive=passive)

        # turn input data into usable format for optimisation problem
        data_feasibles = self._format_data_optimiser(
            batch, passive=passive)
        if not all(data_feasibles):
            fs, cluss, batch, data_feasibles = self._loop_replace_data(
                data_feasibles, passive, self.file_id())
        if not all(data_feasibles):
            data_feasible = False
        res = None
        new_res = False

        seed_data = res, fs, cluss, batch

        return seed_data, new_res, data_feasible

    def _active_find_feasible_data(
            self,
            type_actions: List[str],
            feasibility_checked: bool,
            step_vals: dict,
            evaluation: bool,
            epoch: int,
            mus_opt: list
    ) -> Tuple[list, bool, bool, dict, list]:
        passive = False
        data_feasible = True
        new_res = False
        res = None
        opt_needed = 'opt' in type_actions \
            or (not feasibility_checked and self.check_feasibility_with_opt)
        if opt_needed:
            file_exists = (self.paths['res_path'] / self.res_name).is_file()
        else:
            file_exists = (self.paths['res_path'] / f"batch{self.file_id()}"
                           ).is_file()
        just_load_data = not self.rl['deterministic'] == 2 and file_exists

        # check if data is feasible by solving optimisation problem
        if just_load_data:
            if opt_needed:
                res = np.load(self.paths['res_path']
                              / self.res_name,
                              allow_pickle=True).item()
            fs, cluss = self._load_res()
            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.p],
                load_data=True, passive=passive)
            new_res = False
        else:
            # pre-save data to be used for multiple methods
            [batch, fs, cluss] = self._env_make_data(
                int(self.seed[self.p]),
                passive=passive
            )

        # turn input data into optimisation problem format
        data_feasibles = self._format_data_optimiser(
            batch, passive=passive)
        if not all(data_feasibles):
            fs, cluss, batch, data_feasibles = \
                self._loop_replace_data(
                    data_feasibles, passive,
                    self.file_id())
            feasibility_checked = False

        if all(data_feasibles) and opt_needed:
            try:
                res = self.solver.solve(self.prm)
            except Exception as ex:  # if infeasible, make new data
                if str(ex)[0:6] != 'Code 3':
                    print(traceback.format_exc())
                    print(f'ex.args {ex.args}')
                    if self.prm["syst"]["play_sound"]:
                        play_sound()
                data_feasible = False
            new_res = True

        if data_feasible and 'opt' in type_actions:  # start with opt
            # exploration through optimisation
            step_vals, mus_opt, data_feasible = self.get_steps_opt(
                res, step_vals, evaluation, cluss,
                fs, batch, self.seed[self.p],
                last_epoch=epoch == self.prm['RL']['n_epochs'] - 1)

        seed_data = res, fs, cluss, batch

        return [seed_data, new_res, data_feasible, step_vals,
                mus_opt, feasibility_checked]

    def find_feasible_data(self,
                           seed_ind: int,
                           type_actions: list,
                           step_vals: dict,
                           evaluation: bool,
                           epoch: int,
                           passive: bool = False
                           ) -> [dict, dict, dict, dict, dict, list]:
        """
        For a new day of exploration or evaluation, generate feasible data.

        Check whether data has already been generated, saved, and checked
        and perform these actions accordingly.
        Depending on inputs, check feasibility with optimisation or heuristic
        checks only.

        Returns:
        res: optimisation results
        batch: load, EV and PV batch of data
        fs: scaling factors for the load, PV and EV load profiles
        cluss: behaviour clusters for the load and EV profiles
        step_vals: RL exploration/evaluation data (state, action, rewards, etc)
        mus_opt: actions selected by the optimiser
        """
        # boolean: whether optimisation problem is feasible;
        # start by assuming it is not
        data_feasible = 0
        it = -1
        while not data_feasible and it < 100:
            # try solving problem else making new data until problem solved
            it += 1
            feasibility_checked = self.get_seed(seed_ind)
            set_seeds_rdn(self.seed[self.p])
            mus_opt = None
            self.res_name = \
                f"res_P{int(self.seed['P'])}_" \
                f"{int(self.seed[''])}{self.prm['paths']['opt_res_file']}"

            if passive:
                seed_data, new_res, data_feasible \
                    = self._passive_find_feasible_data()
            else:
                [seed_data, new_res, data_feasible, step_vals,
                 mus_opt, feasibility_checked] \
                    = self._active_find_feasible_data(
                        type_actions, feasibility_checked, step_vals,
                        evaluation, epoch, mus_opt
                )

            if not data_feasible:
                seed_ind = self.infeasible_tidy_files_seeds(seed_ind)

        if not feasibility_checked:
            self.seeds[self.p] = np.append(
                self.seeds[self.p], self.seed[self.p])

        if new_res:
            np.save(self.paths['res_path'] / self.res_name, seed_data[0])

        return seed_data, step_vals, mus_opt

    def get_seed(self, seed_ind: int) -> bool:
        """Given the seed_ind, compute the random seed."""
        if seed_ind < len(self.seeds[self.p]):
            feasibility_checked = True
            self.seed[self.p] = self.seeds[self.p][seed_ind]
        else:
            self.seed[self.p] = seed_ind + self.d_seed[self.p]
            feasibility_checked = False

        return feasibility_checked

    def infeasible_tidy_files_seeds(self, seed_ind):
        """If data is infeasible, update seeds and remove saved files."""
        if seed_ind < len(self.seeds[self.p]):
            self.d_ind_seed[self.p] += 1
            seed_ind += 1
        else:
            for e in ['fs', 'cluss', 'batch']:
                file_names = glob.glob(
                    str(self.paths['res_path']
                        / f"{e}{self.file_id()}"))
                for file_name in file_names:
                    os.remove(file_name)
            file_names = glob.glob(
                str(self.paths['res_path'] / self.res_name))
            for file_name in file_names:
                os.remove(file_name)
            self.d_seed[self.p] += 1

        return seed_ind

    def get_seed_ind(self, ridx: int, epoch: int, i_explore: int) -> int:
        """
        Obtain the rank of the current seed.

        returns seed_ind;
        the random seed can then be selected
        as the seed_ind-th of the list of valid random seeds
        """
        if epoch < self.rl['n_epochs']:
            seed_ind = \
                ridx * (self.rl['n_epochs'] * (self.rl['n_explore'] + 1)
                        + self.rl['n_end_test']) \
                + epoch * (self.rl['n_explore'] + 1) + i_explore
        if epoch >= self.rl['n_epochs']:
            seed_ind = \
                ridx * (self.rl['n_epochs'] * (self.rl['n_explore'] + 1)
                        + self.rl['n_end_test']) \
                + (self.rl['n_epochs'] * (self.rl['n_explore'] + 1)) \
                + (epoch - self.rl['n_epochs'])

        return seed_ind

    def file_id(self):
        """Generate string to identify the run in saved files."""
        return f"{int(self.seed[self.p])}{self.p}{self.prm['paths']['opt_res_file']}"

    def _loop_replace_data(self,
                           data_feasibles: list_bool,
                           passive: bool,
                           file_id: str
                           ) -> [dict, dict, dict, list_bool]:
        """Replace the data for infeasible homes."""
        its = 0
        as_0 = [i for i, ok in enumerate(data_feasibles) if not ok]

        as_ = copy.deepcopy(as_0)
        while not all(data_feasibles) and its < 24:
            self.env.dloaded = 0
            self.env.fix_data_a(as_, file_id, its=its)
            fs, cluss = self._load_res()

            self.batch_file, batch = self.env.reset(
                seed=self.seed[self.p], load_data=True, passive=passive)
            # turn input data into usable format for optimisation problem
            data_feasibles = self._format_data_optimiser(
                batch, passive=passive
            )
            if its > 5:
                print(f'its {its}, sum(data_feasibles) {sum(data_feasibles)}')
            as_ = [i for i, ok in enumerate(data_feasibles) if not ok]
            if its > 5:
                print(f'its = {its} infeasibles as_ = {as_}')
                for a in as_:
                    print(f'a = {a} in infeasibles')
            its += 1

        return fs, cluss, batch, data_feasibles

    def _load_res(self, labels: list = ['fs', 'cluss']) -> dict:
        """Load pre-saved day data."""
        x = [np.load(self.paths['res_path']
                     / f"{label}{self.file_id()}",
                     allow_pickle=True).item()
             for label in labels]

        return x

    def _env_make_data(self, seed: int, passive: bool = False) -> [dict, dict]:
        """
        Instruct env to load/ make data given deterministic. Save and clean up.

        Make and save or load pre-saved fs, cluss and batch data.
        If we have already saved the deterministic environment set-up,
        load it and reset environment accordingly.

        Else, generate random data.
        """
        if self.rl['deterministic'] == 1 and self.deterministic_created:
            # deterministic
            load_data = True
            batch_file, batch = self.env.reset(
                seed=seed, load_data=load_data, passive=passive)

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
                seed=seed, load_data=load_data, passive=passive)

        # for subsequent methods, on reinitialisation, it will use this data
        # obtain data needed saved in batchfile
        batch, fs, cluss = self._load_res(labels=['batch', 'fs', 'cluss'])

        new_batch_file = 'batch_file' in self.__dict__.keys() \
                         and batch_file != self.batch_file
        if not self.rl['plotting_batch'] \
                and self.rl['deterministic'] == 0 \
                and new_batch_file:
            files = os.listdir(os.getcwd())
            for fi in files:
                if fi[0:len(self.batch_file)] == self.batch_file:
                    os.remove(fi)
        self.batch_file = batch_file

        return [batch, fs, cluss]

    def _format_data_optimiser(
            self,
            batch: dict,
            passive: bool = False
    ) -> bool:
        """Turn input data into usable format for optimisation problem."""
        # initialise dicts
        ntw, loads, grd, syst, bat, heat = \
            [self.prm[e] for e in
             ['ntw', 'loads', 'grd', 'syst', 'bat', 'heat']]
        p = 'P' if passive else ''

        # format battery info
        entries_bat = ['avail_EV', 'loads_EV']
        for e in entries_bat:
            bat['batch_' + e] = np.zeros((ntw['n' + p], syst['N'] + 1))

        if 'bat_dem_agg' in self.rl['state_space']:
            entries_bat += ['bat_dem_agg']
            bat['bat_dem_agg'] = np.zeros((ntw['n' + p], syst['N'] + 1))
        for a in range(ntw["n" + p]):
            for e in self.env.bat.batch_entries:
                e_batch = e if e in batch[0] else 'lds_EV'
                bat['batch_' + e][a] = \
                    batch[a][e_batch][0: len(bat['batch_' + e][a])]

        loads['n_types'] = 2

        ntw = self._format_ntw(ntw, loads, syst, bat, batch, p)

        feasible = self.env.bat.check_feasible_bat(self.prm, ntw, p, bat, syst)

        heat['T_out'] = self.env.heat.T_out

        return feasible

    def _format_ntw(self, ntw, loads, syst, bat, batch, p):
        # Willingness to delay (WTD)
        WTD = np.zeros((loads['n_types'], syst['N']), dtype=int)
        if loads['flextype'] == 1:
            WTD[0] = np.zeros(syst['N'])
            for t in range(syst['N']):
                WTD[1, t] = max(min(loads['flex'][1], syst['N'] - 1 - t), 0)
        else:
            for load_type in range(loads['n_types']):
                for t in range(syst['N']):
                    WTD[load_type][t] = max(
                        min(loads['flex'][load_type], syst['N'] - 1 - t), 0)

        # make ntw matrices
        ntw['Bcap'] = np.zeros((ntw['n' + p], syst['N']))
        ntw['dem'] = np.zeros((loads['n_types'], ntw['n' + p], syst['N']))
        ntw['flex'] = np.zeros(
            (syst['N'], loads['n_types'], ntw['n' + p], syst['N']))

        ntw['gen'] = np.zeros((ntw['n' + p], syst['N'] + 1))
        for a in range(ntw['n' + p]):
            ntw['gen'][a] = batch[a]['gen'][0: len(ntw['gen'][a])]
            for t in range(syst['N']):
                ntw['Bcap'][a, t] = bat['cap' + p][a]
                for load_type in range(loads['n_types']):
                    loads_str = 'loads' if 'loads' in batch[a] else 'lds'
                    ntw['dem'][0][a][t] = batch[a][loads_str][t] \
                        * (1 - loads['share_flexs'][a])
                    ntw['dem'][1][a][t] = batch[a][loads_str][t] \
                        * loads['share_flexs'][a]
                    for tC in range(syst['N']):
                        if tC >= t and tC <= t + int(WTD[load_type][t]):
                            ntw['flex'][t, load_type, a, tC] = 1

        return ntw
