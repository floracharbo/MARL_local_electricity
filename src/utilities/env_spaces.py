"""This file containts the EnvSpaces class."""

from datetime import timedelta

import numpy as np
import pandas as pd
import torch as th


def _actions_to_unit_box(actions, rl):
    if isinstance(actions, np.ndarray):
        return rl["actions2unit_coef_numpy"] * actions \
            + rl["actions_min_numpy"]
    elif actions.is_cuda:
        return rl["actions2unit_coef"] * actions + rl["actions_min"]
    else:
        return rl["actions2unit_coef_cpu"] * actions \
            + rl["actions_min_cpu"]


def _actions_from_unit_box(actions, rl):
    if isinstance(actions, np.ndarray):
        return th.div((actions - rl["actions_min_numpy"]),
                      rl["actions2unit_coef_numpy"])
    elif actions.is_cuda:
        return th.div((actions - rl["actions_min"]),
                      rl["actions2unit_coef"])
    else:
        return th.div((actions - rl["actions_min_cpu"]),
                      rl["actions2unit_coef_cpu"])


def granularity_to_multipliers(granularity):
    """
    Get multipliers for each indicator index to get unique number.

    Given the granularity of a list of indicators;
    by how much to multiply each of the indexes
    to get a unique integer identifier.
    """
    # check that i am not going to encounter
    # RuntimeWarning: overflow encountered in long scalars
    # granular spaces should only be used if their size is manageable
    for i in range(1, len(granularity)):
        assert np.prod(granularity[-i:]) < 1e9, \
            "the global space is too large for granular representation"
    multipliers = []
    for i in range(len(granularity) - 1):
        multipliers.append(np.prod(granularity[i + 1:]))
    multipliers.append(1)

    return multipliers


def compute_max_car_cons_gen_values(env, state_space):
    """Get the maximum possible values for car consumption, household consumption and generation."""
    max_car_cons, max_normcons, max_normgen, max_bat_dem_agg = [-1 for _ in range(4)]
    day_types = env.prm["syst"]["day_types"]

    if any(descriptor[0: len("bat_cons_")] == "bat_cons_" for descriptor in state_space):
        max_car_cons = np.max(
            [[env.prof["car"]["cons"][dt][c] for dt in day_types] for c in range(env.n_clus["car"])]
        )
    if any(descriptor[0: len("loads_cons_")] == "loads_cons_" for descriptor in state_space):
        max_normcons = np.max(
            [
                [np.max(env.prof["loads"][dt][c]) for dt in day_types]
                for c in range(env.n_clus["loads"])
            ]
        )
    if any(descriptor[0: len("gen_prod_")] == "gen_prod_" for descriptor in state_space):
        max_normgen = np.max([env.prof["gen"][m] for m in range(12)])

    if any(descriptor == "bat_dem_agg" for descriptor in state_space):
        max_bat_dem_agg = np.max(
            [
                [sum(env.hedge.profs["car"]["cons"][dt][c]) for dt in day_types]
                for c in range(env.n_clus["car"])
            ]
        )

    return max_car_cons, max_normcons, max_normgen, max_bat_dem_agg


class EnvSpaces():
    """Manage operations for environment states and actions spaces."""

    def __init__(self, env):
        """Initialise EnvSpaces class, add properties."""
        self.n_homes = env.n_homes
        self.current_date0 = env.prm['syst']['date0_dtm']
        self.get_state_vals = env.get_state_vals
        self.c_max = env.prm["car"]["c_max"]
        self.N = env.N
        self.i0_costs = env.i0_costs
        self.car = env.car
        self.action_translator = env.action_translator
        self._get_space_info(env)
        prm = env.prm
        self._init_factors_profiles_parameters(prm)
        for e in [
            "dim_actions", "aggregate_actions", "type_env",
            "n_discrete_actions", "evaluation_methods", "flexibility_states",
        ]:
            self.__dict__[e] = prm["RL"][e]
        self.normalise_states_bool = prm["RL"]["normalise_states"]
        self.i0_costs = env.i0_costs
        self.state_funcs = {
            "store0": self._get_store,
            "grdC_level": self._get_grdC_level,
            "dT_next": self._get_dT_next,
            "car_tau": self._get_car_tau,
            "bat_dem_agg": self._get_bat_dem_agg,
            "bool_flex": self.get_bool_flex,
            "flexibility": self.get_flexibility,
            "store_bool_flex": self.get_store_bool_flex
        }
        self.cluss, self.factors = env.hedge.clusters, env.hedge.factors

    def _init_factors_profiles_parameters(self, prm):
        self.perc = {}
        for e in ["loads", "gen", "car", "grd"]:
            if "perc" in prm[e]:
                self.perc[e] = prm[e]["perc"]

    def _get_space_info(self, env):
        """Initialise information on action and state spaces."""
        prm = env.prm
        # info on state and action spaces
        max_car_cons, max_normcons, max_normgen, max_bat_dem_agg \
            = compute_max_car_cons_gen_values(env, prm["RL"]["state_space"])
        if self.i0_costs == 12 * 24:
            np.save("max_bat_dem_agg", max_bat_dem_agg)
            print("save max_bat_dem_agg")
        columns = ["name", "min", "max", "n", "discrete"]
        rl = prm["RL"]
        i_month = env.date.month - 1 if 'date' in env.__dict__.keys() else 0
        n_other_states = rl["n_other_states"]
        f_min, f_max = env.hedge.f_min, env.hedge.f_max
        max_flexibility = prm['car']['c_max'] + prm['car']['d_max'] + 1
        n_clus = prm['n_clus']
        info = [
            ["None", 0, 0, 1, 1],
            ["hour", 0, prm['syst']['N'], n_other_states, 0],
            ["store0", 0, prm["car"]["cap"], n_other_states, 0],
            ["grdC", min(prm["grd"]["Call"]), max(prm["grd"]["Call"]), n_other_states, 0],
            ["grdC_level", 0, 1, rl["n_grdC_level"], 0],
            [
                "dT",
                - (prm["heat"]["Tc"] - prm["heat"]["Ts"]),
                prm["heat"]["Tc"] - prm["heat"]["Ts"],
                n_other_states,
                0
            ],
            [
                "dT_next",
                - (prm["heat"]["Tc"] - prm["heat"]["Ts"]),
                prm["heat"]["Tc"] - prm["heat"]["Ts"],
                n_other_states,
                0
            ],
            ["day_type", 0, 1, 2, 1],
            ["store_bool_flex", 0, 1, 2, 1],
            ["flexibility", 0, max_flexibility, n_other_states, 0],
            ["bool_flex", 0, 1, 2, 1],
            ["avail_car_step", 0, 1, 2, 1],
            ["avail_car_prev", 0, 1, 2, 1],
            ["car_tau", 0, prm["car"]["c_max"], 3, 0],
            # clusters - for whole day
            ["loads_clus_step", 0, n_clus['loads'] - 1, n_clus['loads'], 1],
            ["loads_clus_prev", 0, n_clus["loads"] - 1, n_clus["loads"], 1],
            ["bat_cbat_clus_step", 0, n_clus["car"] - 1, n_clus["car"], 1],
            ["bat_clus_prev", 0, n_clus["car"] - 1, n_clus["car"], 1],
            # scaling factors - for whole day
            ["loads_fact_step", f_min["loads"], f_max["loads"], n_other_states, 0],
            ["loads_fact_prev", f_min["loads"], f_max["loads"], n_other_states, 0],
            ["gen_fact_step", f_min["gen"], f_max["gen"], n_other_states, 0],
            ["gen_fact_prev", f_min["gen"], f_max["gen"], n_other_states, 0],
            ["bat_fact_step", f_min["car"], f_max["car"], n_other_states, 0],
            ["bat_fact_prev", f_min["car"], f_max["car"], n_other_states, 0],
            # absolute value at time step / hour
            ["loads_cons_step", 0, max_normcons * f_max["loads"], n_other_states, 0],
            ["loads_cons_prev", 0, max_normcons * f_max["loads"], n_other_states, 0],
            ["gen_prod_step", 0, max_normgen * f_max["gen"][i_month], n_other_states, 0],
            ["gen_prod_prev", 0, max_normgen * f_max["gen"][i_month], n_other_states, 0],
            ["bat_cons_step", 0, max_car_cons, n_other_states, 0],
            ["bat_cons_prev", 0, max_car_cons, n_other_states, 0],
            ["bat_dem_agg", 0, max_bat_dem_agg, n_other_states, 0],

            # action
            ["action", 0, 1, rl["n_discrete_actions"], 0],
            ["flexible_cons_action", 0, 1, rl["n_discrete_actions"], 0],
            ["flexible_heat_action", 0, 1, rl["n_discrete_actions"], 0],
            ["battery_action", -1, 1, rl["n_discrete_actions"], 0]
        ]

        self.space_info = pd.DataFrame(info, columns=columns)

    def descriptor_for_info_lookup(self, descriptor):
        return 'grdC' if descriptor[0: len('grdC_t')] == 'grdC_t' else descriptor

    def new_state_space(self, state_space):
        """Initialise current indicators info for state and action spaces."""
        [self.descriptors, self.granularity, self.maxval, self.minval,
         self.multipliers, self.global_multipliers, self.n, self.discrete,
         self.possible] = [{} for _ in range(9)]
        action_space = ["action"] if self.aggregate_actions \
            else ["flexible_cons_action", "flexible_heat_action", "battery_action"]
        for space, descriptors in zip(["state", "action"],
                                      [state_space, action_space]):
            # looping through state and action spaces
            self.descriptors[space] = descriptors
            descriptors = ["None"] if descriptors == [None] else descriptors
            descriptors_idx = [
                self.space_info["name"] == self.descriptor_for_info_lookup(descriptor)
                for descriptor in descriptors
            ]
            subtable = [self.space_info.loc[i] for i in descriptors_idx]
            [self.granularity[space], self.minval[space],
             self.maxval[space], self.discrete[space]] = [
                [row[field].values.item() for row in subtable]
                for field in ["n", "min", "max", "discrete"]]

            if self.type_env == "discrete":
                self.n[space] = np.prod(self.granularity[space])
                # initialise multipliers
                self.multipliers[space] = granularity_to_multipliers(
                    self.granularity[space])
                self.possible[space] = np.linspace(
                    0, self.n[space] - 1, num=self.n[space])
                # initialise global multipliers for going to agent
                # states and actions to global states and actions
                if any(method[-2] == 'C' for method in self.evaluation_methods):
                    self.global_multipliers[space] = \
                        granularity_to_multipliers(
                            [self.n[space] for _ in range(self.n_homes)])

        # need to first define descriptors to define brackets
        self.brackets = self._init_brackets()

    def global_to_indiv_index(self, typev, global_ind, multipliers=None):
        """From global discretised space index, get individual indexes."""
        if global_ind is None and typev == "global_action":
            indexes = None
        else:
            if multipliers is None:
                if typev == "global_state":
                    granularity = [self.n["state"]
                                   for _ in range(self.n_homes)]
                    multipliers = granularity_to_multipliers(granularity)

                elif typev == "global_action":
                    granularity = [self.n["action"]
                                   for _ in range(self.n_homes)]
                    multipliers = granularity_to_multipliers(granularity)
                else:
                    multipliers = self.multipliers[typev]
            n = len(multipliers)
            indexes = [[] for _ in range(len(multipliers))]
            remaining = global_ind
            for i in range(n):
                indexes[i] = int((remaining - remaining % multipliers[i])
                                 / multipliers[i])
                remaining -= indexes[i] * multipliers[i]

        return indexes

    def indiv_to_global_index(self, type_descriptor, indexes=None,
                              multipliers=None, done=False):
        """From discrete space indexes, get global combined index."""
        if indexes is None and type_descriptor == "state":
            if done:
                indexes = [None for _ in range(self.n_homes)]
            else:
                indexes = self.get_space_indexes(
                    done=done, all_vals=self.get_state_vals())
        elif indexes is None and type_descriptor == "action":
            print("need to input action indexes for indiv_to_global_index")
        if multipliers is None:
            multipliers = self.global_multipliers[type_descriptor]

        sum_None = sum(1 for i in range(len(indexes))
                       if indexes[i] is None or multipliers[i] is None)
        if sum_None > 0:
            global_index = None
        else:
            global_index = sum(a * b for a, b in zip(indexes, multipliers))

        return global_index

    def index_to_val(self, index, typev="state"):
        """From state/action discretised index, get value."""
        val = []
        for s in range(len(index)):
            if self.discrete[typev][s] == 1:
                val.append(index[s])
            else:
                brackets_s = self.brackets[typev][s] + [self.maxval[typev][s]]
                if typev == "action" and index[s] == 0:
                    val.append(0)
                elif typev == "action" and index[s] == self.n_discrete_actions - 1:
                    val.append(1)
                else:
                    val.append((brackets_s[int(index[s])]
                                + brackets_s[int(index[s] + 1)]) / 2)

        return val

    def get_space_indexes(self, done=False, all_vals=None,
                          type_="state", indiv_indexes=False):
        """
        Return array of indexes of current agents' states/actions.

        Inputs:
                all_vals : all_vals[home][descriptor] =
                env.get_state_vals() / or values directly for action or
                if values inputted are not that of the current environment
                info : optional -
                input for action or if not testing current environment
                type_ : "action" or "state" - default "state"
        """
        t = type_
        type_ = "state" if type_ in ["state", "next_state"] else "action"

        if t == "next_state" and done:
            # if the sequence is over, return None
            return [None for _ in range(self.n_homes)]
        if type_ == "state" and self.descriptors["state"] == [None]:
            # if the state space is None, return 0
            return [0 for _ in range(self.n_homes)]

        # translate values into indexes
        index = []  # one global index per agent
        for home in range(self.n_homes):
            vals_home = all_vals[home]

            indexes = []  # one index per value - for current agent
            for v in range(len(vals_home)):
                if vals_home[v] is None or np.isnan(vals_home[v]):
                    indexes.append(0)
                elif self.discrete[type_][v] == 1:
                    indexes.append(int(vals_home[v]))
                else:
                    # correct if value is smaller than smallest bracket
                    if vals_home[v] is None:
                        indexes.append(None)
                    else:
                        brackets = self.brackets[type_][v]
                        brackets_v = brackets \
                            if len(np.shape(brackets)) == 1 \
                            else brackets[home]
                        assert vals_home[v] > brackets_v[0] - 1e-2, \
                            f"brackets_v0 = {brackets_v[0]} " \
                            f"vals_home[v] = {vals_home[v]}"
                        if vals_home[v] < brackets_v[0]:
                            vals_home[v] = 0
                        mask = vals_home[v] >= np.array(brackets_v[:-1])
                        interval = np.where(mask)[0][-1]
                        indexes.append(interval)

            if indiv_indexes:
                index.append(indexes)
            else:
                # global index for all values of current agent home
                index.append(
                    self.indiv_to_global_index(
                        type_, indexes=indexes,
                        multipliers=self.multipliers[type_]
                    )
                )
                assert not (
                    index[-1] is not None and index[-1] >= self.n[type_]
                ), f"index larger than total size of space agent {home}"

        return index

    def get_global_ind(self, current_state, state, action, done, method):
        """Given state/action values list, get global space index."""
        global_ind = {}
        for label, type_ind, x in zip(["state", "next_state", "action"],
                                      ["state", "state", "action"],
                                      [current_state, state, action]):
            if not (label == "next_state" and done):
                ind_x = self.get_space_indexes(
                    done=False, all_vals=x, type_=type_ind)
                if method[-2] == 'C':
                    global_ind[label] = [
                        self.indiv_to_global_index(
                            type_ind,
                            indexes=ind_x,
                            multipliers=self.global_multipliers[type_ind],
                            done=done
                        )
                    ]
            else:
                global_ind[label] = None

        return global_ind

    def _init_brackets(self):
        """Initialise intervals to convert state/action values into indexes."""
        brackets = {}
        for typev in ["state", "action"]:
            if (
                    self.type_env == "continuous"
                    or (typev == "state"
                        and self.descriptors["state"] == [None])
            ):
                brackets[typev] = None
                continue
            perc_dict = {
                'loads_cons_step': 'loads',
                'gen_prod_step': 'gen',
                'car_cons_step': 'car',
                'grdC': 'grd'
            }
            brackets[typev] = []
            for s in range(len(self.descriptors[typev])):
                ind_str = self.descriptors[typev][s]
                n_bins = self.granularity[typev][s]

                if self.discrete[typev][s] == 1:
                    brackets[typev].append([0])
                elif ind_str in perc_dict:
                    i_perc = [
                        int(1 / n_bins * 100 * i)
                        for i in range(n_bins + 1)
                    ]
                    brackets[typev].append(
                        [self.perc[perc_dict[ind_str]][i] for i in i_perc]
                    )
                elif ind_str == "car_tau":
                    brackets[typev].append([-75, 0, 10, self.c_max])
                elif type(self.maxval[typev][s]) in [int, float]:
                    brackets[typev].append(
                        [self.minval[typev][s]
                         + (self.maxval[typev][s] - self.minval[typev][s])
                         / n_bins * i
                         for i in range(n_bins + 1)])
                else:
                    if isinstance(self.maxval[typev][s], list):
                        brackets[typev].append(
                            [[self.minval[typev][s]
                              + (self.maxval[typev][s][home]
                                 - self.minval[typev][s]) / n_bins * i
                              for i in range(n_bins + 1)]
                             for home in range(self.n_homes)])
                    else:
                        brackets[typev].append(
                            [[self.minval[typev][s]
                              + (self.maxval[typev][s]
                                 - self.minval[typev][s]) / n_bins * i
                              for i in range(n_bins + 1)]
                             for _ in range(self.n_homes)])

        return brackets

    def get_ind_global_state_action(self, step_vals_i):
        """Get the global index for a given states or actions combination."""
        action = step_vals_i["action"]
        if (
                self.type_env == "discrete"
                and any(method[-2] == 'C' for method in self.evaluation_methods)
        ):
            ind_state = self.get_space_indexes(
                all_vals=step_vals_i["state"])
            step_vals_i["ind_global_state"] = \
                [self.indiv_to_global_index(
                    "state", indexes=ind_state,
                    multipliers=self.global_multipliers["state"])]
            ind_action = self.get_space_indexes(
                all_vals=action, type_="action")
            for home in range(self.n_homes):
                assert not (ind_action is None and action[home] is not None), \
                    f"action[{home}] {step_vals_i['action'][home]} " \
                    f"is none whereas action {action[home]} is not"
            step_vals_i["ind_global_action"] = \
                [self.indiv_to_global_index(
                    "action", indexes=ind_action,
                    multipliers=self.global_multipliers["action"])]
        else:
            step_vals_i["ind_global_state"] = None
            step_vals_i["ind_global_action"] = None

        return step_vals_i

    def _initial_processing_bool_flex_computation(self, time_step, date, loads, home_vars):
        """Prepare variables for flexibility assessment."""
        if any(
            descriptor in self.flexibility_states for descriptor in self.descriptors['state']
        ):
            self.car.update_step(time_step=time_step)
            self.car.min_max_charge_t(time_step, date)
            self.E_heat_min_max(time_step)
            self.action_translator.initial_processing(loads, home_vars)

    def _revert_changes_bool_flex_computation(self):
        """Undo variable changes made for flexibility assessment."""
        if any(
            descriptor in self.flexibility_states for descriptor in self.descriptors['state']
        ):
            self.car.revert_last_update_step()

    def opt_step_to_state(
            self,
            prm: dict,
            res: dict,
            time_step: int,
            loads_prev: list,
            loads_step: list,
            batch_avail_car: np.ndarray,
            loads: dict,
            home_vars: dict
    ) -> list:
        """
        Get state descriptor values.

        Get values corresponding to state descriptors specified,
        based on optimisation results.
        """
        n_homes = len(res["T_air"])
        vals = []
        date = self.current_date0 + timedelta(hours=time_step)
        self._initial_processing_bool_flex_computation(time_step, date, loads, home_vars)

        for home in range(n_homes):
            vals_home = []
            state_vals = {
                "None": 0,
                "hour": time_step % 24,
                "grdC": prm["grd"]["Call"][self.i0_costs + time_step],
                "day_type": 0 if date.weekday() < 5 else 1,
                "loads_cons_step": loads_step[home],
                "loads_cons_prev": loads_prev[home],
                "dT": prm["heat"]["T_req"][home][time_step]
                - res["T_air"][home][min(time_step, len(res["T_air"][home]) - 1)]
            }

            for descriptor in self.descriptors["state"]:
                if descriptor in state_vals:
                    val = state_vals[descriptor][home] \
                        if isinstance(state_vals[descriptor], list) \
                        else state_vals[descriptor]
                elif descriptor in self.state_funcs:
                    inputs = time_step, res, home, date, prm
                    val = self.state_funcs[descriptor](inputs)
                elif descriptor[0: len('grdC_t')] == 'grdC_t':
                    t_ = int(descriptor[len('grdC_t'):])
                    val = prm["grd"]["Call"][self.i0_costs + time_step + t_] \
                        if time_step + t_ < self.N \
                        else prm["grd"]["Call"][self.i0_costs + self.N - 1]

                elif len(descriptor) > 9 \
                        and (descriptor[-9:-5] == "fact"
                             or descriptor[-9:-5] == "clus"):
                    # scaling factors / profile clusters for the whole day
                    day = (date - prm["syst"]["current_date0_dtm"]).days
                    module = descriptor.split("_")[0]  # car, loads or gen
                    index_day = day - \
                        1 if descriptor.split("_")[-1] == "prev" else day
                    index_day = max(index_day, 0)
                    data = self.factors if descriptor[-9:-5] == "fact" else self.cluss
                    val = data[home][module][index_day]
                else:  # select current or previous hour - step or prev
                    time_step_val = time_step if descriptor[-4:] == "step" else time_step - 1
                    time_step_val = np.max(time_step_val, 0)
                    if len(descriptor) > 8 and descriptor[0: len('avail_car')] == "avail_car":
                        if time_step_val < len(batch_avail_car[0]):
                            val = batch_avail_car[home][time_step_val]
                        else:
                            val = 1
                    elif descriptor[0:3] == "gen":
                        val = prm["grd"]["gen"][home][time_step_val]
                    else:  # remaining are car_cons_step / prev
                        val = prm["car"]["batch_loads_car"][home][time_step]
                val = self.normalise_state(descriptor, val, home)

                vals_home.append(val)
            vals.append(vals_home)

        self._revert_changes_bool_flex_computation()

        assert np.shape(vals) \
               == (self.n_homes, len(self.descriptors["state"])), \
               f"np.shape(vals) {np.shape(vals)} " \
               f"self.n_homes {self.n_homes} " \
               f"len descriptors['state'] {len(self.descriptors['state'])}"

        return vals

    def _get_dT_next(self, inputs):
        """Get temperature requirements parameter at current time step."""
        time_step, _, home, _, prm = inputs
        T_req = prm["heat"]["T_req"][home]
        t_next = [time for time in range(time_step + 1, self.N)
                  if T_req[time] != T_req[time_step]]
        if not t_next:
            val = 0
        else:
            val = (T_req[t_next[0]] - T_req[time_step]) \
                / (t_next[0] - time_step)

        return val

    def get_bool_flex(self, inputs):
        """Get general bool flex (if any of the three bool flex is True then True)"""
        home = inputs[2]
        return self.action_translator.aggregate_action_bool_flex(home)

    def get_flexibility(self, inputs):
        """Get the flexibility (energy different between min/max import/export) for time step."""
        home = inputs[2]
        flexibility = self.action_translator.get_flexibility(home)
        assert flexibility >= 0, f"flexibility {flexibility}"
        return flexibility

    def get_store_bool_flex(self, inputs, home=None):
        """Whether there is flexibility in the battery operation"""
        home = inputs[2]
        return self.action_translator.store_bool_flex(home)

    def _get_car_tau(self, inputs):
        """Get the battery requirement tau parameter for the current time step."""

        time_step, res, home, date, _ = inputs

        loads_T, deltaT, _ = \
            self.car.next_trip_details(time_step, date, home)

        if loads_T is not None and deltaT > 0:
            val = ((loads_T - res["store"][home][time_step]) / deltaT)
        else:
            val = - 1

        return val

    def _get_store(self, inputs):
        """Get the battery storage level for the current time step."""
        time_step, res, home, _, prm = inputs
        if time_step < len(res["store"][home]):
            val = res["store"][home][time_step]
        else:
            val = prm["car"]["store0"][home]

        return val

    def _get_grdC_level(self, inputs):
        """Get the grdC level for the current time step."""
        time_step = inputs[0]
        prm = inputs[-1]
        costs = prm["grd"]["Call"][
            self.i0_costs: self.i0_costs + self.N + 1
        ]
        val = (costs[time_step] - min(costs)) \
            / (max(costs) - min(costs))

        return val

    def _get_bat_dem_agg(self, inputs):
        """Get the aggregated battery demand at current time step."""
        time_step, _, home, _, prm = inputs
        val = prm["car"]["bat_dem_agg"][home][time_step]

        return val

    def normalise_state(self, descriptor, val, home):
        """Normalise state value between 0 and 1."""
        if self.normalise_states_bool:
            descriptor_info = self.space_info.loc[
                self.space_info['name'] == self.descriptor_for_info_lookup(descriptor)
            ]
            max_home = descriptor_info['max'].values.item()[home] \
                if isinstance(descriptor_info['max'].values.item(), list) \
                else descriptor_info['max'].values.item()
            min_val = descriptor_info['min'].values.item()
            if (val - min_val) / (max_home - min_val) > 1:
                print()
            normalised_val = (val - min_val) / (max_home - min_val)
            if abs(normalised_val) < 1e-5:
                normalised_val = 0
            assert 0 <= normalised_val <= 1, \
                f"val {normalised_val} max_home {max_home.values.item()} descriptor {descriptor}"
        else:
            normalised_val = val

        return normalised_val
