#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:47:57 2020.

@author: floracharbonnier

"""

import copy
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from gym import spaces
from gym.utils import seeding
from six import integer_types

import src.environment.utilities.userdeftools as utils
from src.environment.experiment_manager.action_translator import \
    Action_translator
from src.environment.experiment_manager.hedge import HEDGE
from src.environment.simulations.battery import Battery
from src.environment.simulations.heat import Heat
from src.environment.simulations.network import Network
from src.environment.utilities.env_spaces import EnvSpaces
from src.environment.utilities.userdeftools import test_str
from src.tests.local_elec_tests import LocalElecTests


class LocalElecEnv:
    """
    Local electricity environment.

    Includes home-level modelling of the flexible assets,
    computes step actions, updating states and rewards.
    """

    # =============================================================================
    # # initialisation / data interface
    # =============================================================================
    def __init__(self, prm):
        """Initialise Local Elec environment, add properties."""
        self.batchfile0 = 'batch'
        self.envseed = self._seed()
        self.random_seeds_use = {}
        self.batch_entries = ['loads', 'gen', 'loads_car', 'avail_car', 'flex']
        self.clus, self.f = {}, {}
        self.labels = ['loads', 'car', 'gen']
        self.labels_clus = ['loads', 'car']
        self.test = prm['syst']['test_on_run']
        self.prm = prm
        self.rl = prm['RL']
        self.labels_day_trans = prm['syst']['labels_day_trans']
        self.n_homes = prm['syst']['n_homes']
        self.homes = range(self.n_homes)
        self.ext = ''

        if (
            self.prm['grd']['manage_voltage']
            or self.prm['grd']['manage_agg_power']
            or self.prm['grd']['simulate_panda_power_only']
        ):
            self.network = Network(prm)

        # initialise parameters
        for info in ['N', 'n_int_per_hr', 'dt', 'test_different_to_train']:
            setattr(self, info, prm['syst'][info])

        self.server = prm['syst']['server']
        self.i0_costs = 0
        self.car = Battery(prm)
        self.hedge = HEDGE(
            n_homes=prm['syst']['n_homes'],
            factors0=prm['syst']['f0'],
            clusters0=prm['syst']['clus0'],
            prm=prm
        )
        self.spaces = EnvSpaces(self)
        if self.prm['syst']['test_different_to_train']:
            self.spaces_test = EnvSpaces(self, evaluation=True)
        self.spaces.new_state_space(self.rl['state_space'])
        self.action_translator = Action_translator(prm, self)
        self.spaces.action_translator = self.action_translator

        self.add_noise = 1 if self.rl['deterministic'] == 2 else 0
        for data in [
            "competitive", "n_grdC_level", "offset_reward", "delta_reward"
        ]:
            setattr(self, data, self.rl[data])
        self.res_path = prm['paths']['opt_res']
        self.slid_day = False

        if self.rl['type_learning'] == 'facmac':
            self.action_space = self.rl['action_space']
            self.observation_space = []
            for _ in self.homes:
                self.observation_space.append(spaces.Box(
                    low=-np.inf, high=+np.inf,
                    shape=(self.rl['obs_shape'],),
                    dtype=np.float32))

        self.max_delay = int(
            prm["loads"]["max_delay"] * self.n_int_per_hr
        )

        if prm['syst']['n_homesP'] > 0:
            self.passive_hedge = HEDGE(
                n_homes=prm['syst']['n_homesP'],
                factors0=prm['syst']['f0'],
                clusters0=prm['syst']['clus0'],
                prm=prm,
                other_prm={
                    'car': {
                        'cap': prm['car']['capsP'],
                        'own_car': prm['car']['own_carP']
                    }
                },
                ext='P'
            )
        if self.prm['syst']['test_different_to_train']:
            self.test_hedge = HEDGE(
                n_homes=prm['syst']['n_homes_test'],
                factors0=prm['syst']['f0'],
                clusters0=prm['syst']['clus0'],
                prm=prm,
                other_prm={
                    'car': {
                        'cap': prm['car']['caps_test'],
                        'own_car': prm['car']['own_car_test'],
                    }
                },
                ext='_test'
            )
        self.tests = LocalElecTests(self)

    def reset(
            self,
            seed: int = 0,
            load_data: bool = False,
            passive: bool = False,
            E_req_only: bool = False,
            evaluation: bool = False,
    ) -> Tuple[str, dict]:
        """Reset environment for new day with new data."""
        self.tot_cons_loads = []
        if seed is not None:
            self.envseed = self._seed(seed)
            self.random_seeds_use[seed] = 0

        self.n_homes = \
            self.prm['syst']['n_homes_test'] if evaluation \
            else self.prm['syst']['n_homes']

        # different agent caracteristics for passive and active homes
        self.set_passive_active(passive, evaluation)

        # initialise environment time
        self.date = self.date0
        # date0 is not the total date0 for all runs,
        # but the specific date0 for this run
        # as defined in learning.py at the same time as i0_costs
        self.time_step = 0  # hrs since start
        self.steps_beyond_done = None
        self.done = False
        self.idt = 0 if self.date0.weekday() < 5 else 1
        self.idt0 = 0 if (self.date0 - timedelta(days=1)).weekday() < 5 else 1

        # data management
        self.load_data = load_data
        self.dloaded = 0
        self.add_noise = False

        # update grid costs
        self.update_i0_costs(evaluation)

        self.batch_file = self.batchfile0
        self.save_file = self.batch_file
        self.no_name_file = str(int(seed))
        self.heat = Heat(self.prm, self.i0_costs, self.ext, E_req_only, evaluation)
        self.spaces.E_heat_min_max = self.heat.E_heat_min_max
        self.car.reset(self.prm)
        self.action_translator.heat = self.heat
        self.action_translator.car = self.car

        # initialise demand ahead (2 days)
        self.batch = {}
        self.car.batch = {}
        self._initialise_batch_entries()

        if not load_data or (load_data and self.add_noise):
            self._initialise_new_data(passive=passive, evaluation=evaluation)

        for i in range(2):
            self._load_next_day(i_load=i, evaluation=evaluation)

        self.car.add_batch(self.batch)
        if self.prm['syst']['n_homes' + self.ext] > 0:
            self.batch = self.car.compute_battery_demand_aggregated_at_start_of_trip(self.batch)
        self.tests.loads_test()
        self.batch_flex = copy.deepcopy(self.batch['flex'])

        return self.save_file, self.batch

    def set_passive_active(self, passive: bool = False, evaluation: bool = False):
        """Update environment properties for passive or active case."""
        passive_ext = 'P' if passive else ''
        self.evaluation = evaluation
        test_ext = \
            '_test' if evaluation \
            and self.prm['syst']['test_different_to_train'] \
            else ''
        self.ext = passive_ext + test_ext
        self.n_homes = self.prm['syst']['n_homes' + self.ext]
        self.action_translator.n_homes = self.n_homes
        self.homes = range(self.n_homes)
        setattr(self.action_translator, 'n_homes', getattr(self, 'n_homes'))
        setattr(self.spaces, 'n_homes', getattr(self, 'n_homes'))
        self.T_air = [
            self.prm['heat']['T_req' + self.ext][home][0]
            for home in self.homes
        ]
        self.car.set_passive_active(self.ext, self.prm)

    def update_date(self, i0_costs: int, date0: datetime = None, evaluation=False):
        """Update new date for new day."""
        self.i0_costs = i0_costs
        self.update_i0_costs(evaluation)
        if date0 is not None:
            self.date0 = date0
            self.hedge.date = date0 - timedelta(days=1)
            spaces = self.get_current_spaces(evaluation)
            spaces.current_date0 = self.date0
            self.action_translator.date0 = self.date0
            self.date_end = date0 + timedelta(hours=self.N * self.dt)
            self.car.date0 = self.date0
            self.car.date_end = self.date_end

    def fix_data_a(self, homes, file_id, evaluation, its=0):
        """Recompute data for home a that is infeasible."""
        self._seed(self.envseed[0] + its)
        self.dloaded = 0
        for i in range(2):
            self._load_next_day(homes=homes, i_load=i, evaluation=evaluation)
        self.car.add_batch(self.batch)
        self.batch = self.car.compute_battery_demand_aggregated_at_start_of_trip(self.batch)
        np.save(self.res_path / f"batch{file_id}", self.batch)

    def update_flex(
            self,
            flex_cons: list,
            opts: list = None
    ) -> np.ndarray:
        """Given step flexible consumption, update remaining flexibility."""
        if opts is None:
            time_step = self._get_time_step()
            n_homes = self.n_homes
            batch_flex = self.batch_flex
        else:
            time_step, batch_flex, max_delay, n_homes = opts

        self.tests.check_shape_batch_flex(batch_flex)

        new_batch_flex = copy.deepcopy(batch_flex[:, time_step: time_step + 2])

        for home in range(n_homes):
            remaining_cons = max(flex_cons[home], 0)

            # remove what has been consumed
            for i_flex in range(1, self.max_delay + 1):
                delta_cons = min(new_batch_flex[home, 0, i_flex], remaining_cons)
                remaining_cons -= delta_cons
                new_batch_flex[home, 0, i_flex] -= delta_cons

            self.tests.check_flex_and_remaining_cons_after_update_home(
                flex_cons, batch_flex, remaining_cons, home, time_step
            )

            # move what has not been consumed to one step more urgent
            self._prep_next_flex_step(batch_flex, new_batch_flex, time_step, home)

        return new_batch_flex

    def step(
        self,
        action: list,
        implement: bool = True,
        record: bool = False,
        netp_storeout: bool = False,
        E_req_only: bool = False,
        evaluation=True,
    ) -> list:
        """Compute environment updates and reward from selected action."""
        time_step = self._get_time_step()
        homes = self.homes
        self.tests.batch_tests(time_step)
        # update batch if needed
        daynumber = (self.date - self.date0).days
        if time_step == 1 and self.time_step > 1 \
                and self.dloaded < daynumber + 2 == 0 \
                and not self.slid_day:
            for key in self.batch:
                self.batch[key][:, 0: self.N] = self.batch[key][:, self.N: self.N * 2]
            self._load_next_day(i_load=1, evaluation=evaluation)
            self.slid_day = True
        self.car.add_batch(self.batch)

        if time_step == 2:
            self.slid_day = False
        [
            home_vars, loads, hourly_line_losses, voltage_squared,
            q_ext_grid, constraint_ok, q_car, q_house
        ] = self.policy_to_rewardvar(action, E_req_only=E_req_only, implement=implement)

        netp0 = self.prm['loads']['netp0'][:, time_step]
        if not constraint_ok:
            print('constraint false not returning to original values')
            return [None, None, None, None, None, constraint_ok, None]

        else:
            reward, break_down_rewards = self.get_reward(
                netp=home_vars['netp'],
                hourly_line_losses=hourly_line_losses,
                voltage_squared=voltage_squared,
                q_ext_grid=q_ext_grid,
                evaluation=evaluation,
            )

            # ----- update environment variables and state
            new_batch_flex = self.update_flex(loads['flex_cons'])
            next_date = self.date + timedelta(hours=self.dt)
            next_done = next_date == self.date_end
            inputs_next_state = [
                self.time_step + 1, next_date, next_done, new_batch_flex, self.car.store
            ]
            next_state = self.get_state_vals(inputs=inputs_next_state, evaluation=evaluation) \
                if not self.done \
                else [None for _ in homes]
            T = self.heat.T.copy()

            if implement:
                for home in homes:
                    self.batch_flex[home][time_step: time_step + 2] = new_batch_flex[home]
                self.tot_cons_loads.append(loads['tot_cons_loads'])
                self.time_step += 1
                self.tests.test_flex_cons(self.time_step, self.batch_flex)
                self.date = next_date
                self.idt = 0 if self.date.weekday() < 5 else 1
                self.done = next_done
                self.heat.update_step()
                self.car.update_step(time_step=self.time_step)

            if record:
                loads_flex = np.zeros(self.n_homes) if next_done \
                    else [sum(self.batch_flex[home][time_step][1:]) for home in homes]
                if (
                    self.prm['grd']['manage_voltage']
                    or self.prm['grd']['simulate_panda_power_only']
                ):
                    loaded_buses, sgen_buses = self.network.loaded_buses, self.network.sgen_buses
                else:
                    loaded_buses, sgen_buses = None, None

                record_output = [
                    home_vars['netp'],
                    netp0,
                    self.car.discharge,
                    self.car.store,
                    home_vars['tot_cons'].copy(),
                    self.heat.tot_E.copy(),
                    T,
                    self.heat.T_air.copy(),
                    voltage_squared,
                    hourly_line_losses,
                    action, reward, loads['flex_cons'].copy(),
                    loads_flex, loads['l_fixed'].copy(),
                    loads['tot_cons_loads'].copy(),
                    self.prm['grd'][f'C{test_str(evaluation)}'][self.time_step].copy(),
                    self.__dict__[f"wholesale{test_str(evaluation)}"][self.time_step].copy(),
                    self.__dict__[f"cintensity{test_str(evaluation)}"][self.time_step].copy(),
                    break_down_rewards,
                    loaded_buses, sgen_buses,
                    q_ext_grid,
                    q_car, q_house
                ]

                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok, record_output]
            elif netp_storeout:
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok,
                        [home_vars['netp'], self.car.discharge_tot,
                         self.car.charge]]
            else:
                return [next_state, self.done, reward, break_down_rewards,
                        home_vars['bool_flex'], constraint_ok, None]

    def get_passive_vars(self, time_step):
        passive_vars = [
            self.prm["loads"][e][:, time_step]
            for e in ["netp0", "discharge_tot0", "charge0"]
        ]
        return passive_vars

    def get_reward(
            self,
            netp: list,
            discharge_tot: list = None,
            charge: list = None,
            passive_vars: list = None,
            time_step: int = None,
            voltage_squared: list = None,
            hourly_line_losses: int = 0,
            q_ext_grid: int = 0,
            evaluation: bool = False,
    ) -> Tuple[list, list]:
        """Compute reward from netp and battery charge at time step."""
        if passive_vars is not None:
            netp0, discharge_tot0, charge0 = passive_vars
        elif self.ext == 'P':
            netp0, discharge_tot0, charge0 = [
                np.zeros(self.prm['syst']['n_homesP']) for _ in range(3)
            ]
        else:
            seconds_per_interval = 3600 * 24 / self.prm['syst']['H']
            time_step = int((self.date - self.date0).seconds / seconds_per_interval)
            netp0, discharge_tot0, charge0 = self.get_passive_vars(time_step)

        time_step = self.time_step if time_step is None else time_step
        if discharge_tot is None:
            discharge_tot = self.car.discharge_tot
        charge = self.car.charge if charge is None else charge
        grdCt, wholesalet, cintensityt = [
            self.prm['grd'][f'C{test_str(evaluation)}'][time_step],
            self.__dict__[f"wholesale{test_str(evaluation)}"][time_step],
            self.__dict__[f"cintensity{test_str(evaluation)}"][time_step]
        ]

        # negative netp is selling, positive buying, losses in kWh
        grid = sum(netp) + sum(netp0) + hourly_line_losses

        # import and export limits
        if self.prm['grd']['manage_agg_power'] or self.prm['grd']['simulate_panda_power_only']:
            import_export_costs, _, _ = utils.compute_import_export_costs(
                grid, self.prm['grd'], self.prm['syst']['n_int_per_hr']
            )
        else:
            import_export_costs = 0
        if self.prm['grd']['manage_voltage']:
            voltage_costs = utils.compute_voltage_costs(
                voltage_squared, self.prm['grd']
            )
        else:
            voltage_costs = 0
        if self.prm['grd']['manage_voltage'] or self.prm['grd']['simulate_panda_power_only']:
            if time_step < self.N - 1:
                mean_voltage_deviation, mean_voltage_violation, max_voltage_violation, \
                    n_voltage_violation_bus, n_voltage_violation_hour =  \
                    utils.mean_max_hourly_voltage_deviations(
                        voltage_squared,
                        self.prm['grd']['max_voltage'],
                        self.prm['grd']['min_voltage']
                    )
            else:
                mean_voltage_deviation = 0
                mean_voltage_violation = 0
                max_voltage_violation = 0
                n_voltage_violation_bus = 0
                n_voltage_violation_hour = 0
        else:
            mean_voltage_deviation = 0
            mean_voltage_violation = 0
            max_voltage_violation = 0
            n_voltage_violation_bus = 0
            n_voltage_violation_hour = 0

        if self.prm['grd']['charge_type'] == 0:
            sum_netp_export = sum(self.netp_to_exports(netp))
            sum_netp0_export = sum(self.netp_to_exports(netp0))
            netp_export = sum_netp_export + sum_netp0_export
            distribution_network_export_costs = self.prm['grd']['export_C'] * netp_export
        else:
            netpvar = sum([netp[home] ** 2 for home in self.homes]) \
                + sum([netp0[home] ** 2 for home in range(len(netp0))])
            distribution_network_export_costs = self.prm['grd']['export_C'] * netpvar
        if not self.prm['grd']['penalise_individual_exports']:
            distribution_network_export_costs = 0
        grid_energy_costs = grdCt * (grid + self.prm['grd']['loss'] * grid ** 2)
        cost_distribution_network_losses = grdCt * hourly_line_losses
        indiv_grid_energy_costs = [wholesalet * netp[home] for home in self.homes]
        battery_degradation_costs = self.prm['car']['C'] \
            * (sum(discharge_tot[home] + charge[home]
                   for home in self.homes)
                + sum(discharge_tot0[home] + charge0[home]
                      for home in range(len(discharge_tot0))))
        indiv_battery_degradation_costs = [
            self.prm['car']['C'] * (discharge_tot[home] + charge[home]) for home in self.homes
        ]
        indiv_grid_battery_costs = [
            g + s for g, s in zip(indiv_grid_energy_costs, indiv_battery_degradation_costs)
        ]
        network_costs = self.prm['grd']['weight_network_costs'] * (
            import_export_costs + voltage_costs
        )
        total_reward = - (
            battery_degradation_costs + distribution_network_export_costs + grid_energy_costs
            + network_costs
        )
        if self.prm['RL']['competitive']:
            reward = - np.array(indiv_grid_battery_costs)
        else:
            reward = total_reward
        costs_wholesale = wholesalet * (sum(netp) + sum(netp0))
        costs_upstream_losses = wholesalet * self.prm['grd']['loss'] * grid ** 2
        total_costs = - total_reward
        emissions = cintensityt * (grid + self.prm['grd']['loss'] * grid ** 2)
        emissions_from_grid = cintensityt * grid
        emissions_from_loss = cintensityt * self.prm['grd']['loss'] * grid ** 2
        break_down_rewards = [
            grid_energy_costs, battery_degradation_costs, distribution_network_export_costs,
            import_export_costs, voltage_costs, hourly_line_losses,
            cost_distribution_network_losses, costs_wholesale, costs_upstream_losses, emissions,
            emissions_from_grid, emissions_from_loss, total_costs,
            indiv_grid_energy_costs, indiv_battery_degradation_costs, indiv_grid_battery_costs,
            mean_voltage_deviation, mean_voltage_violation, max_voltage_violation,
            n_voltage_violation_bus, n_voltage_violation_hour
        ]
        self.tests.check_no_values_issing_break_down_rewards(break_down_rewards)

        reward += self.delta_reward
        if self.offset_reward and reward < 0:
            print(f"reward {reward - self.delta_reward} < {- self.delta_reward}")
        reward /= self.rl['normalisation_reward']

        return reward, break_down_rewards

    def netp_to_exports(self, netp):
        netp = np.array(netp)
        return - np.where(netp < 0, netp, 0)

    def get_loads_fixed_flex_gen(self, date, time_step):
        loads, home_vars = {}, {}
        if date == self.date_end - timedelta(hours=self.dt):
            loads['l_flex'] = np.zeros(self.n_homes)
            loads['l_fixed'] = np.sum(self.batch_flex[:, time_step], axis=1)
        else:
            loads['l_flex'] = np.sum(self.batch_flex[:, time_step, 1:], axis=1)
            loads['l_fixed'] = np.array(self.batch_flex[:, time_step, 0])

        home_vars['gen'] = self.batch['gen'][:, time_step]

        return loads, home_vars

    def policy_to_rewardvar(
            self,
            action: list,
            other_input: list = None,
            E_req_only: bool = False,
            implement: bool = True,
    ):
        """Given selected action, obtain results of the step."""
        if other_input is None:
            date = self.date
            time_step = self._get_time_step()
            loads, home_vars = self.get_loads_fixed_flex_gen(date, time_step)
            self.heat.current_temperature_bounds(time_step)
        else:
            date, action, gens, loads = other_input
            gens = np.array(gens)
            self.date = date
            time_step = self._get_time_step()
            home_vars = {'gen': gens}
        self.heat.E_heat_min_max(time_step)
        last_step = True \
            if date == self.date_end - timedelta(hours=self.dt) \
            else False
        bool_penalty = self.car.min_max_charge_t(time_step, date)
        self.heat.potential_E_flex()

        #  ----------- meet consumption + check constraints ---------------
        if action is None:
            for info in ['flex_cons', 'tot_cons_loads']:
                loads[info] = np.zeros(self.n_homes)
            home_vars['bool_flex'] = False
            for info in ['netp', 'tot_cons']:
                home_vars[info] = np.zeros(self.n_homes)
        else:
            loads, home_vars, bool_penalty, flexible_q_car = self.action_translator.actions_to_env_vars(
                    loads, home_vars, action, date, time_step
            )
        self.tests.check_no_flex_left_unmet(home_vars, loads, time_step)

        self.heat.next_T(update=True)
        self._check_constraints(
            bool_penalty, date, loads, E_req_only, time_step, last_step, home_vars
        )

        if self.prm['syst']['n_homesP'] > 0:
            netp0 = self.prm['loads']['netp0'][:, time_step]
        else:
            netp0 = []
        if self.prm['grd']['manage_voltage'] or self.prm['grd']['simulate_panda_power_only']:
            if (
                not self.prm['grd']['reactive_power_for_voltage_control']
                or self.prm['grd']['simulate_panda_power_only']
            ):
                # retrieve info from battery if not a decision variable
                q_car_flex = self.car.p_car_flex * self.prm['grd']['active_to_reactive_flex']
            else:
                # if agents decide on reactive power of battery
                q_car_flex = flexible_q_car
            # run pandapower simulation
            voltage_squared, hourly_line_losses, q_ext_grid, netq_flex = \
                self.network.power_flow_res_with_pandapower(
                    home_vars, netp0, q_car_flex, passive=self.ext == 'P', implement=implement
                )
            q_house = netq_flex - q_car_flex
        else:
            voltage_squared = None
            hourly_line_losses = 0
            q_ext_grid = 0
            q_car_flex = 0
            q_house = 0

        constraint_ok = not sum(bool_penalty) > 0

        return (home_vars, loads, hourly_line_losses, voltage_squared,
                q_ext_grid, constraint_ok, q_car_flex, q_house)

    def get_state_vals(
            self,
            descriptors: list = None,
            inputs: list = None,
            evaluation: bool = False
    ) -> np.ndarray:
        """
        Get values corresponding to array of descriptors inputted.

        (before translation into index)
        """
        if inputs is None:
            time_step, date, done, store = [
                self.time_step, self.date, self.done, self.car.store
            ]
            batch_flex_h = self.batch_flex[:, time_step]
            inputs = [time_step, date, done, batch_flex_h, store]
        else:
            time_step, date = inputs[0:2]

        # inputs_ = [time_step, date, done, batch_flex_h, store] if inputs is None else inputs

        idt = 0 if date.weekday() < 5 else 1
        descriptors = descriptors if descriptors is not None \
            else self.spaces.descriptors['state']
        vals = np.zeros((self.n_homes, len(descriptors)))
        # time_step = self._get_time_step(date)
        flexibility_state = any(
            descriptor in self.rl['flexibility_states'] for descriptor in descriptors
        )
        if flexibility_state and not self.rl['trajectory']:
            self.car.update_step(time_step=time_step)
            self.car.min_max_charge_t(time_step, date)
            self.tets.check_time_car_and_env_match()
            self.heat.E_heat_min_max(time_step)
            loads, home_vars = self.get_loads_fixed_flex_gen(date, time_step)
            self.action_translator.initial_processing(loads, home_vars)
        for home in self.homes:
            for i, descriptor in enumerate(descriptors):
                vals[home, i] = self._descriptor_to_val(
                    descriptor, inputs, idt, home, evaluation
                )
        if flexibility_state and not self.rl['trajectory']:
            self.car.revert_last_update_step()

        return vals

    def _seed(self, seed=None):
        if seed is not None and not isinstance(seed, integer_types):
            seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _p_trans_label(self, transition_type, data_type):
        if transition_type == "wd2wd" and "wd2wd" not in self.p_trans[data_type]:
            transition_type_ = "wd"
        elif transition_type == "we2we" and "we2we" not in self.p_trans[data_type]:
            transition_type_ = "we"
        else:
            transition_type_ = transition_type

        return transition_type_

    def _file_id(self):
        ext_opt_res_file = '_test' if self.ext[0:len('_test')] == '_test' else ''
        opt_res_file = self.prm['paths']['opt_res_file_no' + ext_opt_res_file]

        return f"_{self.no_name_file}{self.ext}_{opt_res_file}"

    def _ps_rand_to_choice(self, ps: list, rand: float) -> int:
        """Given probability of each choice and a random number, select."""
        p_intervals = [sum(ps[0:i]) for i in range(len(ps))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]
        return choice

    def _get_next_clusters(self, transition_type, homes):
        for home in homes:
            for data_type in self.behaviour_types:
                # get next cluster
                transition_type_ = \
                    transition_type[0:2] \
                    if transition_type not in self.p_trans[data_type] \
                    else transition_type
                clus_a = self.clus[data_type + self.ext][home]
                ps = self.p_trans[data_type][transition_type_][clus_a]
                cump = [sum(ps[0:i]) for i in range(1, len(ps))] + [1]
                rdn = self.np_random.rand()
                self.clus[data_type + self.ext][home] = \
                    [c > rdn for c in cump].index(True)
                self.cluss[home][data_type].append(self.clus[data_type + self.ext][home])

    def _load_next_day(self, homes: list = [], i_load=0, evaluation=False):
        """
        Load next day of data.

        Either it is not presaved and needs to be generated,
        or it can just be loaded
        """
        if not self.load_data or len(homes) > 0:
            homes = homes if len(homes) > 0 else self.homes
            if len(homes) > 0:
                if self.ext == '':
                    day = self.hedge.make_next_day(homes)
                elif self.ext == 'P':
                    day = self.passive_hedge.make_next_day(homes)
                elif self.ext == '_test':
                    day = self.test_hedge.make_next_day(homes)

                for e in day.keys():
                    self.batch[e][homes, i_load * self.N: (i_load + 1) * self.N] = day[e]
                self._loads_to_flex(homes, i_load=i_load)
            self.dloaded += 1
        else:
            for info in ['batch', 'i0_costs']:
                if (self.res_path / f"{info}{self._file_id()}").is_file():
                    setattr(
                        self,
                        info,
                        np.load(
                            self.res_path / f"{info}{self._file_id()}",
                            allow_pickle=True
                        ).item()
                    )
            self.update_i0_costs(evaluation)
            self.dloaded += self.prm['syst']['D']

    def _loads_to_flex(self, homes: list = None, i_load: int = 0):
        """Apply share of flexible loads to new day loads data."""
        homes = self.homes if len(homes) == 0 else homes
        for home in homes:
            dayflex_a = np.zeros((self.N, self.max_delay + 1))
            for time_step in range(self.N):
                loads_t = self.batch["loads"][home, i_load * self.N + time_step]
                dayflex_a[time_step, 0] = (1 - self.share_flexs[home]) * loads_t
                dayflex_a[time_step, self.max_delay] = self.share_flexs[home] * loads_t
            self.batch['flex'][home, i_load * self.N: (i_load + 1) * self.N] = dayflex_a

    def _get_time_step(self, date: datetime = None) -> int:
        """Given date, obtain time step."""
        date = self.date if date is None else date
        time_elapsed = date - self.date0
        time_step = int(
            time_elapsed.days * self.prm["syst"]["H"]
            + time_elapsed.seconds / (60 * 60) * self.n_int_per_hr
        )

        return time_step

    def _check_loads(
        self,
        home: int,
        date: datetime,
        time_step: int,
        loads: dict,
        bool_penalty: List[bool]
    ) -> List[bool]:
        """Check load-related constraints for given home after step."""
        flex_cons, l_fixed = [loads[e] for e in ['flex_cons', 'l_fixed']]

        if date == self.date_end - timedelta(hours=self.dt) \
                and flex_cons[home] > 1e-2:
            print(f"home = {home}, flex_cons[home] = {flex_cons[home]}")
            bool_penalty[home] = True

        if loads['l_flex'][home] > 1e2:
            print(f"time_step = {time_step}, home = {home}, l_flex[home] = {loads['l_flex'][home]}")
            bool_penalty[home] = True

        return bool_penalty

    @property
    def share_flexs(self):
        return self.prm['loads']['share_flexs' + self.ext]

    def _check_constraints(
            self,
            bool_penalty: List[bool],
            date: datetime,
            loads: dict,
            E_req_only: bool,
            time_step: int,
            last_step: bool,
            home_vars: dict
    ) -> List[bool]:
        """Given result of the step action, check environment constraints."""
        for home in [home for home, bool in enumerate(bool_penalty) if not bool]:
            self.car.check_constraints(home, date, time_step)
            self.heat.check_constraints(home, time_step, E_req_only)
            bool_penalty = self._check_loads(home, date, time_step, loads, bool_penalty)
            # prosumer balance
            prosumer_balance_sum = \
                abs(home_vars['netp'][home]
                    - (self.car.loss_ch[home] + self.car.charge[home])
                    + self.car.discharge[home]
                    + home_vars['gen'][home]
                    - home_vars['tot_cons'][home])
            if prosumer_balance_sum > 1e-2:
                print(f"home {home} prosumer_balance_sum = {prosumer_balance_sum}")
                print(f"self.car.loss_ch[{home}] = {self.car.loss_ch[home]}")
                print(f"self.car.charge[{home}] = {self.car.charge[home]}")
                print(f"self.car.discharge[{home}] = {self.car.discharge[home]}")
                print(f"home = {home}, loads = {loads}")
                np.save('action_translator_d', self.action_translator.d)
                np.save('action_translator_mu', self.action_translator.action_intervals)
                np.save('action_translator_k', self.action_translator.k)
                bool_penalty[home] = True

            # check tot cons
            if home_vars['tot_cons'][home] < - 1e-2:
                print(f"negative tot_cons {home_vars['tot_cons'][home]} home = {home}")
                bool_penalty[home] = True

            if last_step \
                    and home_vars['tot_cons'][home] < \
                    self.batch['loads'][home, time_step] * (1 - self.share_flexs[home]):
                print(f"home = {home}, no flex cons at last time step")
                bool_penalty[home] = True
        # self.car.revert_last_update_step()

        return bool_penalty

    def _compute_min_voltage(self, time_step):
        if time_step == 0:
            return 1
        else:
            return np.min(self.network.voltage)

    def _compute_dT_next(self, home, time_step):
        T_req = self.prm['heat']['T_req' + self.ext][home]
        t_change_T_req = [
            time_step for time_step in range(time_step + 1, self.N)
            if T_req[time_step] != T_req[time_step]
        ]
        if len(t_change_T_req) > 0:
            current_T_req = T_req[time_step]
            next_T_req = T_req[t_change_T_req[0]]
            val = (next_T_req - current_T_req) / (t_change_T_req[0] - time_step)
        else:
            val = 0

        return val

    def _get_grdC_level(self, inputs, evaluation):
        spaces = self.get_current_spaces(evaluation)

        return spaces._get_grdC_level(inputs)

    def _descriptor_to_val(
            self,
            descriptor: str,
            inputs: list,
            idt: int,
            home: int,
            evaluation: bool
    ):
        """Given state of action space descriptor, get value."""
        time_step, date, done, batch_flex_h, store = inputs
        dict_vals = {
            "None": 0,
            "time_step_day": time_step % self.prm["syst"]["H"],
            "bat_dem_agg": self.batch["bat_dem_agg"][home, time_step],
            "store0": store[home],
            "grdC": self.prm['grd'][f'C{test_str(evaluation)}'][time_step],
            "day_type": idt,
            "dT": self.prm["heat"]["T_req" + self.ext][home][time_step] - self.T_air[home],
            "grdC_level": self._get_grdC_level(
                [time_step, None, None, None, self.prm], evaluation
            )
        }
        dict_functions_home = {
            "bool_flex": self.action_translator.aggregate_action_bool_flex,
            "store_bool_flex": self.action_translator.get_store_bool_flex,
            "flexibility": self.action_translator.get_flexibility
        }

        if descriptor in dict_vals:
            val = dict_vals[descriptor]
        elif descriptor in dict_functions_home:
            val = dict_functions_home[descriptor]()[home]
        elif descriptor[0: len('grdC_t')] == 'grdC_t':
            t_ = int(descriptor[len('grdC_t'):])
            val = self.prm['grd'][f'C{test_str(evaluation)}'][time_step + t_] \
                if time_step + t_ < self.N \
                else self.prm['grd'][f'C{test_str(evaluation)}'][self.N - 1]
        elif len(descriptor) >= 4 and descriptor[0:4] == 'grdC':
            val = self.__dict__[f'normalised_grdC{test_str(evaluation)}'][time_step]
        elif descriptor == 'dT_next':
            val = self._compute_dT_next(home, time_step)
        elif descriptor == 'min_voltage':
            val = self._compute_min_voltage(time_step)
        elif descriptor == 'car_tau':
            val = self.car.car_tau(time_step, date, home, store[home])
        elif (
                len(descriptor) > 9
                and (descriptor[-9:-5] == 'fact' or descriptor[-9:-5] == 'clus')
        ):
            # scaling factors / profile clusters for the whole day
            val = self._get_factor_or_cluster_state(descriptor, home)
        else:  # select current or previous time step - step or prev
            time_step = self._get_time_step() if descriptor[-4:] == 'step' \
                else self._get_time_step() - 1
            if len(descriptor) > 8 and descriptor[0: len('avail_car')] == 'avail_car':
                val = self.batch['avail_car'][home, time_step]
            elif descriptor[0:5] == 'loads':
                val = np.sum(batch_flex_h[home][1])
            else:
                # gen_prod_step / prev and car_cons_step / prev
                batch_type = 'gen' if descriptor[0:3] == 'gen' else 'loads_car'
                val = self.batch[batch_type][home, time_step]
        spaces = self.get_current_spaces(evaluation)
        val = spaces.normalise_state(descriptor, val, home)

        return val

    def get_current_spaces(self, evaluation):
        return self.spaces_test \
            if evaluation and self.prm['syst']['test_different_to_train'] \
            else self.spaces

    def _prep_next_flex_step(self, batch_flex, new_batch_flex, time_step, home):
        self.tests.check_flex_not_too_large(new_batch_flex, batch_flex, home, time_step)
        for i_flex in range(self.max_delay):
            loads_next_flex = new_batch_flex[home][0][i_flex + 1]
            self.tests.check_share_flex_makes_sense_with_fixed_flex_total(
                i_flex, new_batch_flex, loads_next_flex, batch_flex, home, time_step
            )
            new_batch_flex[home][0][i_flex + 1] -= loads_next_flex
            new_batch_flex[home][1][i_flex] += loads_next_flex

    def set_i0_costs(self, i0_costs):
        if i0_costs is not None:
            self.i0_costs = i0_costs

    def update_i0_costs(self, evaluation, i0_costs=None):
        self.set_i0_costs(i0_costs)
        i_start, i_end = self.i0_costs, self.i0_costs + self.N + 1
        test_str_ = test_str(evaluation)
        self.prm['grd'][f'C{test_str_}'] = self.prm['grd'][f'Call{test_str_}'][i_start: i_end]
        setattr(
            self,
            f'wholesale{test_str_}',
            self.prm['grd'][f'wholesale_all{test_str_}'][i_start: i_end]
        )
        setattr(
            self,
            f'cintensity{test_str_}',
            self.prm['grd'][f'cintensity_all{test_str_}'][i_start: i_end]
        )
        i_grdC_level = [
            i for i in range(len(self.spaces.descriptors['state']))
            if self.spaces.descriptors['state'][i] == 'grdC_level'
        ]
        if len(i_grdC_level) > 0:
            min_grdC = min(self.prm['grd'][f'C{test_str_}'][0: self.N])
            max_grdC = max(self.prm['grd'][f'C{test_str_}'][0: self.N])
            self.__dict__[f'normalised_grdC{test_str_}'] = [
                (grid_energy_costs - min_grdC) / (max_grdC - min_grdC)
                for grid_energy_costs in self.prm['grd'][f'C{test_str_}'][0: self.N + 1]
            ]

            if not self.spaces.type_env == "continuous":
                spaces = self.get_current_spaces(evaluation)
                spaces.brackets['state'][i_grdC_level[0]] = [
                    [
                        np.percentile(
                            self.__dict__[f'normalised_grdC{test_str_}'],
                            i * 100 / self.n_grdC_level
                        )
                        for i in range(self.n_grdC_level)
                    ] + [1]
                    for _ in self.homes
                ]
        if 'heat' in self.__dict__:
            self.heat.update_i0_costs(self.prm, self.i0_costs, evaluation)

    def _initialise_new_data(self, passive: bool = False, evaluation: bool = False):
        # we have not loaded data from file -> save new data

        # date_end is not max date end but date end based on
        # current date0 and duration as specified in learning.py
        for i in range(2):
            self._load_next_day(i_load=i, evaluation=evaluation)

        if not passive and self.n_homes > 0:
            self.batch = self.car.compute_battery_demand_aggregated_at_start_of_trip(self.batch)

        for e in ['batch', 'i0_costs']:
            file_id = f"{e}{self._file_id()}"
            np.save(self.res_path / file_id, getattr(self, e))
        np.save('outputs/opt_res/files_list.npy', self.prm['paths']['files_list'])

        self._initialise_batch_entries()

        self.batch_file = self.save_file
        self.load_data = True
        self.dloaded = 0
        self.add_noise = False

    def _initialise_batch_entries(self):
        self.batch = {entry: np.zeros((self.n_homes, 2 * self.N)) for entry in self.batch_entries}
        self.car.batch = {
            entry: np.zeros((self.n_homes, 2 * self.N)) for entry in self.car.batch_entries
        }
        self.batch['flex'] = np.zeros((self.n_homes, self.N * 2, self.max_delay + 1))
        self.car.batch['flex'] = np.zeros((self.n_homes, self.N * 2, self.max_delay + 1))

    def _get_factor_or_cluster_state(self, descriptor, home):
        module = descriptor.split('_')[0]  # car, loads or gen
        if descriptor.split('_')[-1] == 'prev':
            prev_data = self.hedge.list_factors if descriptor[-9:-5] == 'fact' \
                else self.hedge.list_clusters
            val = prev_data[module][home][-1] if len(prev_data[module][home]) == 1 \
                else prev_data[module][home][-2]
        else:  # step
            step_data = self.hedge.factors if descriptor[-9:-5] == 'fact' \
                else self.hedge.clusters
            val = step_data[module][home]

        return val
