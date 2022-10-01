"""This file containts the EnvSpaces class."""

import traceback

import numpy as np
import pandas as pd

from utils.userdeftools import granularity_to_multipliers


class EnvSpaces():
    """Manage indexes operations for environment states and actions."""

    def __init__(self, env):
        """Initialise EnvSpaces class, add properties."""
        self.n_agents = env.n_agents
        self.n_actions = env.prm["RL"]["n_discrete_actions"] \
            if "n_discrete_actions" in env.prm["RL"].keys() \
            else env.prm["RL"]["n_actions"]
        self.list_factors = {}
        for e, obj in \
                zip(["gen", "loads", "bat"],
                    [env.prm["gen"], env.prm["loads"], env.prm["bat"]]):
            self.list_factors[e] = obj["listfactors"]
        self.get_state_vals = env.get_state_vals

        # info on state and action spaces
        maxEVcons, max_normcons_hour, max_normgen_hour = [-1 for _ in range(3)]
        for dt in ["wd", "we"]:
            for c in range(env.n_clus["bat"]):
                if np.max(env.prof["bat"]["cons"][dt][c]) > maxEVcons:
                    maxEVcons = np.max(env.prof["bat"]["cons"][dt][c])
            for c in range(env.n_clus["loads"]):
                if np.max(env.prof["loads"][dt][c]) > max_normcons_hour:
                    max_normcons_hour = np.max(env.prof["loads"][dt][c])
            for m in range(12):
                if len(env.prof["gen"][m]) > 0 \
                        and np.max(env.prof["gen"][m]) > max_normgen_hour:
                    max_normgen_hour = np.max(env.prof["gen"][m])

        self.c_max = env.prm["bat"]["c_max"]
        self.type_eval = env.prm["RL"]["type_eval"]
        prm = env.prm

        columns = ["name", "min", "max", "n", "discrete"]
        rl = prm["RL"]
        info = [["none", None, None, 1, 0],
                ["houre0dC", 0, 24, rl["n_other_states"], 0],
                ["store0dC", 0, prm["bat"]["cap"], rl["n_other_states"], 0],
                ["grdC", min(prm["grd"]["Call"]), max(prm["grd"]["Call"]),
                 rl["n_other_states"], 0],
                ["grdC_level", 0, 1, rl["n_grdC_level"], 0],
                ["dT", - (prm["heat"]["Tc"] - prm["heat"]["Ts"]),
                 prm["heat"]["Tc"] - prm["heat"]["Ts"],
                 rl["n_other_states"], 0],
                ["dT_next", - (prm["heat"]["Tc"] - prm["heat"]["Ts"]),
                 prm["heat"]["Tc"] - prm["heat"]["Ts"],
                 rl["n_other_states"], 0],
                ["day_type", 0, 1, 2, 1],
                ["avail_EV_step", 0, 1, 2, 1],
                ["avail_EV_prev", 0, 1, 2, 1],
                ["EV_tau", 0, prm["bat"]["c_max"], 3, 0],
                # clusters - for whole day
                ["loads_clus_step", 0, env.n_clus["loads"] - 1,
                 env.n_clus["loads"], 1],
                ["loads_clus_prev", 0, env.n_clus["loads"] - 1,
                 env.n_clus["loads"], 1],
                ["bat_cbat_clus_step", 0, env.n_clus["bat"] - 1,
                 env.n_clus["bat"], 1],
                ["bat_clus_prev", 0, env.n_clus["bat"] - 1,
                 env.n_clus["bat"], 1],
                # scaling factors - for whole day
                ["loads_fact_step", env.minf["loads"], env.maxf["loads"],
                 rl["n_other_states"], 0],
                ["loads_fact_prev", env.minf["loads"], env.maxf["loads"],
                 rl["n_other_states"], 0],
                ["gen_fact_step", env.minf["gen"], env.maxf["gen"],
                 rl["n_other_states"], 0],
                ["gen_fact_prev", env.minf["gen"], env.maxf["gen"],
                 rl["n_other_states"], 0],
                ["bat_fact_step", env.minf["bat"], env.maxf["bat"],
                 rl["n_other_states"], 0],
                ["bat_fact_prev", env.minf["bat"], env.maxf["bat"],
                 rl["n_other_states"], 0],
                # absolute value at time step / hour
                ["loads_cons_step", 0, max_normcons_hour * env.maxf["loads"],
                 rl["n_other_states"], 0],
                ["loads_cons_prev", 0, max_normcons_hour * env.maxf["loads"],
                 rl["n_other_states"], 0],
                ["gen_prod_step", 0, max_normgen_hour * env.maxf["gen"],
                 rl["n_other_states"], 0],
                ["gen_prod_prev", 0, max_normgen_hour * env.maxf["gen"],
                 rl["n_other_states"], 0],
                ["bat_cons_step", 0, maxEVcons, rl["n_other_states"], 0],
                ["bat_cons_prev", 0, maxEVcons, rl["n_other_states"], 0],
                ["bat_dem_agg", 0, maxEVcons, rl["n_other_states"], 0],

                # action
                ["mu", 0, 1, rl["n_discrete_actions"], 0],
                ["mu_flex_cons", 0, 1, rl["n_discrete_actions"], 0],
                ["mu_flex_heat", 0, 1, rl["n_discrete_actions"], 0],
                ["mu_charge", -1, 1, rl["n_discrete_actions"], 0]]

        self.space_info = pd.DataFrame(info, columns=columns)
        self.perc = {}
        for e in ["loads", "gen", "bat", "grd"]:
            self.perc[e] = prm[e]["perc"]

        for e in ["dim_actions", "aggregate_actions", "type_env"]:
            self.__dict__[e] = rl[e]

    def new_state_space(self, state_space):
        """Initialise current indicators info for state and action spaces."""
        [self.descriptors, self.granularity, self.maxval, self.minval,
         self.multipliers, self.global_multipliers, self.n, self.discrete,
         self.possible] = [{} for _ in range(9)]
        action_space = ["mu"] if self.aggregate_actions \
            else ["mu_flex_cons", "mu_flex_heat", "mu_charge"]
        for space, descriptors in zip(["state", "action"],
                                      [state_space, action_space]):
            # looping through state and action spaces
            self.descriptors[space] = descriptors
            descriptors = ["none"] if descriptors == [None] else descriptors
            descriptors_idx = [self.space_info["name"] == descriptor_
                               for descriptor_ in descriptors]
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
                if any(t[-2] == 'C' for t in self.type_eval):
                    self.global_multipliers[space] = \
                        granularity_to_multipliers(
                            [self.n[space] for _ in range(self.n_agents)])

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
                                   for _ in range(self.n_agents)]
                    multipliers = granularity_to_multipliers(granularity)

                elif typev == "global_action":
                    granularity = [self.n["action"]
                                   for _ in range(self.n_agents)]
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
                indexes = [None for a in range(self.n_agents)]
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
                try:
                    if typev == "action" and index[s] == 0:
                        val.append(0)
                    elif typev == "action" and index[s] == self.n_actions - 1:
                        val.append(1)
                    else:
                        val.append((brackets_s[int(index[s])]
                                    + brackets_s[int(index[s] + 1)]) / 2)
                except Exception as ex:
                    print(ex)

        return val

    def get_space_indexes(self, done=False, all_vals=None, info=None,
                          type_="state", indiv_indexes=False):
        """
        Return array of indexes of current agents' states/actions.

        Inputs:
                all_vals : all_vals[a][descriptor] =
                env.get_state_vals() / or values directly for action or
                if values inputted are not that of the current environment
                info : optional -
                input for action or if not testing current environment
                type_ : "action" or "state" - default "state"
        """
        t = type_
        type_ = "state" if type_ in ["state", "next_state"] else "action"

        # if info  not inputted, first obtain them
        if info is not None:
            discrete, brackets, n_agents, descriptors, multipliers = info
        else:
            [discrete, brackets, n_agents, descriptors, multipliers] = \
                [self.discrete[type_], self.brackets[type_],
                 self.n_agents, self.descriptors[type_],
                 self.multipliers[type_]]
        if type_ == "state":
            if t == "next_state" and done:
                # if the sequence is over, return None
                return [None for a in range(self.n_agents)]
            if self.descriptors["state"] == [None]:
                # if the state space is None, return 0
                return [0 for a in range(self.n_agents)]
            if all_vals is None:
                print("input all_vals = self.get_state_vals() from env")

        elif type_ == "action":  # for action mu, input values
            mu_action = all_vals

        # translate values into indexes
        index = []  # one global index per agent
        for a in range(n_agents):
            if type_ == "state":
                vals_a = all_vals[a]
                for descriptor, val in zip(descriptors, all_vals[a]):
                    # correct negative values
                    if descriptor == "gen_prod_prev" and val < 0:
                        print("gen = {}, correct to 0".format(val))
                        val = 0
            elif type_ == "action":
                vals_a = mu_action[a]

            indexes = []  # one index per value - for current agent
            try:
                for v in range(len(vals_a)):
                    if discrete[v] == 1:
                        try:
                            indexes.append(int(vals_a[v]))
                        except Exception as ex:
                            print(f"ex {ex} vals_a {vals_a} v {v}")

                    else:
                        # correct if value is smaller than smallest bracket
                        if vals_a[v] is None:
                            indexes.append(None)
                        else:
                            brackets_v = brackets[v] \
                                if len(np.shape(brackets[v])) == 1 \
                                else brackets[v][a]
                            try:
                                if vals_a[v] < brackets_v[0]:
                                    if abs(vals_a[v] - brackets_v[0]) > 1e-2:
                                        print(f"brackets_v0 = {brackets_v[0]}"
                                              f"vals_a[v] = {vals_a[v]}")
                                    vals_a[v] = brackets_v[0]
                            except Exception as ex:
                                print(f"ex {ex}")
                                print(traceback.format_exc())
                            interval = [i for i in range(len(brackets_v) - 1)
                                        if vals_a[v] >= brackets_v[i]][-1]
                            indexes.append(interval)
            except Exception as ex:
                print(f"ex {ex}")
                print(traceback.format_exc())
            if len(indexes) == 1 and indexes[0] is None and type_ == "action":
                index.append(None)
            else:
                if indiv_indexes:
                    index.append(indexes)
                else:
                    try:
                        # global index for all values of current agent a
                        index_a = sum([a * b for a, b in
                                       zip(indexes, multipliers)])
                    except Exception as ex:
                        print(f"ex {ex}")
                        print(traceback.format_exc())
                    if index_a >= self.n[type_]:
                        print("index larger than total size of space")
                        return 0
                    else:
                        index.append(index_a)
        return index

    def get_global_ind(self, current_state, state, action, done, t):
        """Given state/action values list, get global space index."""
        global_ind = {}
        for label, type_ind, x in zip(["state", "next_state", "action"],
                                      ["state", "state", "action"],
                                      [current_state, state, action]):
            if t != "tryopt" and not (label == "next_state" and done):
                ind_x = self.get_space_indexes(
                    done=False, all_vals=x, info=None, type_=type_ind)
                if t[-2] == 'C':
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
        brackets = {}
        for typev in ["state", "action"]:
            if self.type_env == "continuous":
                brackets[typev] = None
                continue
            if typev == "state" and self.descriptors["state"] == [None]:
                brackets[typev] = None
            else:
                brackets[typev] = []
                for s in range(len(self.descriptors[typev])):
                    ind_str = self.descriptors[typev][s]
                    n_bins = self.granularity[typev][s]
                    if self.discrete[typev][s] == 1:
                        brackets[typev].append([0])
                    elif ind_str[-1] == "f":
                        brackets[typev].append(
                            [np.percentile(
                                self.list_factors[ind_str[0:3]],
                                1 / n_bins * 100 * i) for i in range(n_bins)])
                    elif ind_str == "loads_cons_step":
                        brackets[typev].append(
                            [self.perc["loads"][int(1 / n_bins * 100 * i)]
                             for i in range(n_bins + 1)])
                    elif ind_str == "gen_prod_step":
                        brackets[typev].append(
                            [self.perc["gen"][int(1 / n_bins * 100 * i)]
                             for i in range(n_bins + 1)])
                    elif ind_str == "EV_cons_step":
                        brackets[typev].append(
                            [self.perc["bat"][int(1 / n_bins * 100 * i)]
                             for i in range(n_bins + 1)])
                    elif ind_str == "grdC":
                        brackets[typev].append(
                            [self.perc["grd"][int(1 / n_bins * 100 * i)]
                             for i in range(n_bins + 1)])
                    elif ind_str == "EV_tau":
                        brackets[typev].append([-75, 0, 10, self.c_max])
                    elif type(self.maxval[typev][s]) in [int, float]:
                        brackets[typev].append(
                            [self.minval[typev][s]
                             + (self.maxval[typev][s] - self.minval[typev][s])
                             / n_bins * i
                             for i in range(n_bins + 1)])
                    else:
                        if type(self.maxval[typev][s]) is list:
                            brackets[typev].append(
                                [[self.minval[typev][s]
                                  + (self.maxval[typev][s][a]
                                     - self.minval[typev][s]) / n_bins * i
                                  for i in range(n_bins + 1)]
                                 for a in range(self.n_agents)])
                        else:
                            brackets[typev].append(
                                [[self.minval[typev][s]
                                  + (self.maxval[typev][s]
                                     - self.minval[typev][s]) / n_bins * i
                                  for i in range(n_bins + 1)]
                                 for _ in range(self.n_agents)])
        return brackets
