"""
Home energy data generator (HEDGE).

Generates subsequent days of car, PV generation and home electricity data
for a given number of homes.

The main method is 'make_next_day', which generates new day of data
(car, loads, gen profiles), calling other methods as needed.
"""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import yaml
from scipy.stats import norm

from src.utilities.userdeftools import initialise_dict


class HEDGE:
    """
    Home energy data generator (HEDGE).

    Generates subsequent days of car, gen and loads data for
    a given number of homes.
    """

    # pylint: disable=too-many-instance-attributes, disable=no-member

    def __init__(
        self,
        n_homes: int,
        factors0: dict = None,
        clusters0: dict = None,
        prm: dict = None,
        passive: bool = False,
    ):
        """Initialise HEDGE object and initial properties."""
        # update object properties
        self.passive_ext = 'P' if passive else ''
        self.labels_day = ["wd", "we"]
        self.n_homes = n_homes
        self.homes = range(self.n_homes)

        # load input data
        self._load_input_data(prm, factors0, clusters0)

        self.save_day_path = Path(prm["paths"]["record_folder"])

    def _load_input_data(self, prm, factors0, clusters0):
        prm = self._load_inputs(prm)
        self._init_factors(factors0)
        self._init_clusters(clusters0)
        self.profs = self._load_profiles(prm)

        # number of time steps per day
        if "loads" in self.profs:
            self.n_steps = len(self.profs["loads"]["wd"][0][0])
        elif "car" in self.profs:
            self.n_steps = len(self.profs["car"]["cons"]["wd"][0][0])
        else:
            self.n_steps = len(self.profs["gen"][0][0])

    def make_next_day(
            self,
            homes: Optional[list] = None,
            plotting: bool = False
    ) -> dict:
        """Generate new day of data (car, gen, loads profiles)."""
        homes = self.homes if homes is None else homes
        self.date += timedelta(days=1)
        day_type, transition = self._transition_type()
        prev_clusters = self.clusters.copy()

        # obtain scaling factors
        factors, interval_f_ev = self._next_factors(transition, prev_clusters)

        # obtain clusters
        clusters = self._next_clusters(transition, prev_clusters)

        # obtain profile indexes
        i_month = self.date.month - 1
        i_profiles = {}
        for data_type in self.data_types:
            i_profiles[data_type] = self._select_profiles(
                data_type, day_type, clusters=clusters, i_month=i_month
            )

        # obtain days
        day = {}
        if "loads" in self.data_types:
            day["loads"] = [
                [p * factors["loads"][home]
                 for p in self.profs["loads"][day_type][
                     clusters["loads"][home]][i_profiles["loads"][home]]]
                for home in homes]
        if "gen" in self.data_types:
            gen_profs = self.profs["gen"][i_month]
            day["gen"] = [
                [
                    p * factors["gen"][home]
                    for p in gen_profs[i_profiles["gen"][home]]
                ]
                for home in homes
            ]
        if "car" in self.data_types:
            day["loads_car"] = np.array([
                [p * factors["car"][home]
                 for p in self.profs["car"]["cons"][day_type][
                     clusters["car"][home]][i_profiles["car"][home]]]
                for home in homes
            ])

            # check loads car are consistent with maximum battery load
            interval_f_ev, factors, day, i_profiles["car"] \
                = self._adjust_max_ev_loads(
                day, interval_f_ev, factors, transition, clusters,
                day_type, i_profiles["car"], homes
            )

            day["avail_car"] = np.array([
                self.profs["car"]["avail"][day_type][
                    clusters["car"][home]][i_profiles["car"][home]]
                for home in homes
            ])

            for i_home, home in enumerate(homes):
                day["avail_car"][i_home] = np.where(
                    day["loads_car"][i_home] > 0, 0, day["avail_car"][i_home]
                )

        self.factors = factors
        self.clusters = clusters

        # save factors and clusters
        for home in homes:
            for data in self.data_types:
                self.list_factors[data][home].append(self.factors[data][home])
            for data in self.behaviour_types:
                self.list_clusters[data][home].append(self.clusters[data][home])

        self._plotting_profiles(day, plotting)

        return day

    def _import_dem(self, prm):
        # possible types of transition between week day types (week/weekend)
        day_trans = []
        for prev_day in prm["syst"]["day_types"]:
            for next_day in prm["syst"]["day_types"]:
                day_trans.append(f"{prev_day}2{next_day}")

        for transition in day_trans:
            if self.residual_distribution_prms["loads"][transition] is None:
                continue
            self.residual_distribution_prms["loads"][transition] \
                = list(self.residual_distribution_prms["loads"][transition])
            self.residual_distribution_prms["loads"][transition][1] *= prm["syst"]["f_std_share"]
        self.select_cdfs["loads"] = {}
        for day_type in prm["syst"]["day_types"]:
            self.select_cdfs["loads"][day_type] = [
                min_cdf + prm["syst"]["clust_dist_share"] * (max_cdf - min_cdf)
                for min_cdf, max_cdf in zip(
                    self.min_cdfs["loads"][day_type],
                    self.max_cdfs["loads"][day_type]
                )
            ]

    def _load_inputs(self, prm):
        # load inputs
        if prm is None:
            with open("inputs/parameters.yaml", "rb") as file:
                prm = yaml.safe_load(file)

        prm = self._init_params(prm)

        # general inputs with all data types
        factors_path = prm["paths"]["hedge_inputs"] / "factors"
        for property_ in ["f_min", "f_max", "f_mean"]:
            path = factors_path / f"{property_}.pickle"
            with open(path, "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        for property_ in ["mean_residual", "residual_distribution_prms"]:
            with open(factors_path / f"{property_}.pickle", "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        clusters_path = prm["paths"]["hedge_inputs"] / "clusters"
        for property_ in ["p_clus", "p_trans", "min_cdfs", "max_cdfs"]:
            path = clusters_path / f"{property_}.pickle"
            with open(str(path), "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        with open(clusters_path / "n_clus.pickle", "rb") as file:
            prm["n_clus"] = pickle.load(file)
        self.n_all_clusters_ev = prm["n_clus"]["car"] + 1

        self.select_cdfs = {}
        # household demand-specific inputs
        if "loads" in self.data_types:
            self._import_dem(prm)

        # car-specific inputs
        if "car" in self.data_types:
            for property_ in ["p_pos", "p_zero2pos", "fs_brackets", "mid_fs_brackets"]:
                path = factors_path / f"car_{property_}.pickle"
                with open(path, "rb") as file:
                    self.__dict__[property_] = pickle.load(file)

        # PV generation-specific inputs
        if "gen" in self.data_types:
            self.residual_distribution_prms["gen"] = list(self.residual_distribution_prms["gen"])
            self.residual_distribution_prms["gen"][1] *= prm["syst"]["f_std_share"]

            self.select_cdfs["gen"] = [
                min_cdf + prm["syst"]["clust_dist_share"] * (max_cdf - min_cdf)
                for min_cdf, max_cdf in zip(self.min_cdfs["gen"], self.max_cdfs["gen"])
            ]

        return prm

    def _init_factors(self, factors0):
        _, transition = self._transition_type()
        self.factors = {}
        if factors0 is None:
            if "loads" in self.data_types:
                self.factors["loads"] = [
                    self.f_mean["loads"]
                    + norm.ppf(
                        np.random.rand(),
                        *self.residual_distribution_prms["loads"][transition])
                    for _ in self.homes
                ]

            if "gen" in self.data_types:
                self.factors["gen"] = [
                    self.f_mean["gen"][self.date.month - 1]
                    + norm.ppf(
                        np.random.rand(),
                        *self.residual_distribution_prms["gen"])
                    for _ in self.homes]

            if "car" in self.data_types:
                randoms = np.random.rand(self.n_homes)
                self.factors["car"] = [
                    self._ps_rand_to_choice(
                        self.p_zero2pos[transition],
                        randoms[home]
                    )
                    for home in self.homes
                ]
        else:
            for data in self.data_types:
                if isinstance(factors0[data], int):
                    self.factors[data] = [factors0[data] for _ in self.homes]

        self.list_factors = initialise_dict(self.data_types, second_level_entries=self.homes)
        for home in self.homes:
            for data in self.data_types:
                self.list_factors[data][home] = [self.factors[data][home]]

    def _init_clusters(self, clusters0):
        day_type, transition = self._transition_type()
        self.clusters = {}

        if clusters0 is None:
            for data in self.behaviour_types:
                self.clusters[data] \
                    = [self._ps_rand_to_choice(
                        self.p_clus[data][day_type], np.random.rand())
                        for _ in self.homes]
        else:
            for data in self.behaviour_types:
                if isinstance(clusters0[data], int):
                    self.clusters[data] = [clusters0[data] for _ in self.homes]

        self.list_clusters = initialise_dict(self.behaviour_types, second_level_entries=self.homes)
        for home in self.homes:
            for data in self.behaviour_types:
                self.list_clusters[data][home] = [self.clusters[data][home]]

    def _next_factors(self, transition, prev_clusters):
        prev_factors = self.factors.copy()
        factors = initialise_dict(self.data_types)
        random_f = {}
        for data_type in self.data_types:
            random_f[data_type] = [np.random.rand() for _ in self.homes]
        interval_f_ev = []

        for home in self.homes:
            if "car" in self.data_types:
                current_interval \
                    = [i for i in range(len(self.fs_brackets[transition]) - 1)
                       if self.fs_brackets[transition][i]
                       <= prev_factors["car"][home]][-1]
                if prev_clusters["car"][home] == self.n_all_clusters_ev - 1:
                    # no trip day
                    probabilities = self.p_zero2pos[transition]
                else:
                    probabilities = self.p_pos[transition][current_interval]
                interval_f_ev.append(
                    self._ps_rand_to_choice(
                        probabilities,
                        random_f["car"][home],
                    )
                )
                factors["car"].append(
                    self.mid_fs_brackets[transition][int(interval_f_ev[home])]
                )

            if "gen" in self.data_types:
                i_month = self.date.month - 1
                # factor for generation
                # without differentiation between day types
                delta_f = norm.ppf(
                    random_f["gen"][home],
                    *self.residual_distribution_prms["gen"]
                )
                factors["gen"].append(
                    prev_factors["gen"][home]
                    + delta_f
                    - self.mean_residual["gen"]
                )
                factors["gen"][home] = min(
                    max(self.f_min["gen"][i_month], factors["gen"][home]),
                    self.f_max["gen"][i_month]
                )

            if "loads" in self.data_types:
                # factor for demand - differentiate between day types
                delta_f = norm.ppf(
                    random_f["loads"][home],
                    *list(self.residual_distribution_prms["loads"][transition])
                )
                factors["loads"].append(
                    prev_factors["loads"][home]
                    + delta_f
                    - self.mean_residual["loads"][transition]
                )

            for data_type in self.behaviour_types:
                factors[data_type][home] = min(
                    max(self.f_min[data_type], factors[data_type][home]),
                    self.f_max[data_type]
                )

        return factors, interval_f_ev

    def _next_clusters(self, transition, prev_clusters):
        clusters = initialise_dict(self.behaviour_types)

        random_clus = [
            [np.random.rand() for _ in self.homes]
            for _ in self.behaviour_types
        ]
        for home in self.homes:
            for it, data in enumerate(self.behaviour_types):
                prev_cluster = prev_clusters[data][home]
                probs = self.p_trans[data][transition][prev_cluster]
                cum_p = [sum(probs[0:i]) for i in range(1, len(probs))] + [1]
                clusters[data].append(
                    [c > random_clus[it][home] for c in cum_p].index(True)
                )

        return clusters

    def _transition_type(self):
        day_type = "wd" if self.date.weekday() < 5 else "we"
        prev_day_type \
            = "wd" if (self.date - timedelta(days=1)).weekday() < 5 \
            else "we"
        transition = f"{prev_day_type}2{day_type}"

        return day_type, transition

    def _adjust_max_ev_loads(self, day, interval_f_ev, factors,
                             transition, clusters, day_type, i_ev, homes):
        for i_home, home in enumerate(homes):
            it = 0
            cap = self.car["cap" + self.passive_ext]
            while np.max(day["loads_car"][i_home]) > cap[home] and it < 100:
                if it == 99:
                    print("100 iterations _adjust_max_ev_loads")
                if factors["car"][home] > 0 and interval_f_ev[home] > 0:
                    interval_f_ev[home] -= 1
                    factors["car"][home] = self.mid_fs_brackets[transition][
                        int(interval_f_ev[home])]
                    ev_cons = self.profs["car"]["cons"][day_type][
                        clusters["car"][home]][i_ev[home]]
                    assert sum(ev_cons) == 0 or abs(sum(ev_cons) - 1) < 1e-3, \
                        f"ev_cons {ev_cons}"
                    day["loads_car"][i_home] = ev_cons * factors["car"][home]
                else:
                    i_ev[home] = np.random.choice(np.arange(
                        self.n_prof["car"][day_type][clusters["car"][home]]))
                it += 1

        return interval_f_ev, factors, day, i_ev

    def _compute_number_of_available_profiles(self, data, day_type, i_month):
        if data in self.behaviour_types:
            n_profs0 = [
                self.n_prof[data][day_type][cluster]
                for cluster in range(len(self.n_prof[data][day_type]))
            ]
            if data == 'car':
                for cluster in range(len(self.n_prof[data][day_type])):
                    assert self.n_prof[data][day_type][cluster] \
                           == len(self.profs[data]["cons"][day_type][cluster]), \
                           f"self.n_prof[{data}][{day_type}][{cluster}] " \
                           f"{self.n_prof[data][day_type][cluster]}"

        else:
            n_profs0 = self.n_prof[data][i_month]

        return n_profs0

    def _select_profiles(
            self,
            data: str,
            day_type: str = None,
            i_month: int = 0,
            clusters: List[int] = None
    ) -> List[int]:
        """Randomly generate index of profile to select for given data."""
        i_profs = []
        n_profs0 = self._compute_number_of_available_profiles(data, day_type, i_month)
        n_profs = n_profs0
        for home in self.homes:
            if data in self.behaviour_types:
                n_profs_ = n_profs[clusters[data][home]]
                if n_profs_ > 1:
                    n_profs[clusters[data][home]] -= 1
            else:
                n_profs_ = n_profs
                n_profs -= 1
            i_prof = round(np.random.rand() * (n_profs_ - 1))
            for previous_i_prof in sorted(i_profs):
                if previous_i_prof <= i_prof < n_profs_ - 1 and n_profs_ > 1:
                    i_prof += 1
            i_profs.append(i_prof)
            if data == "car":
                assert i_prof < len(self.profs["car"]["cons"][day_type][clusters[data][home]]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"
            elif data == "loads":
                assert i_prof < len(self.profs[data][day_type][clusters[data][home]]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"
            else:
                assert i_prof < len(self.profs[data][i_month]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"

        return i_profs

    def _ps_rand_to_choice(self, probs: List[float], rand: float) -> int:
        """Given list of probabilities, select index."""
        p_intervals = [sum(probs[0:i]) for i in range(len(probs))]
        choice = [ip for ip in range(len(p_intervals))
                  if rand > p_intervals[ip]][-1]

        return choice

    def _load_ev_profiles(self, input_dir, profiles):
        labels_day = self.labels_day

        for data in ["cons", "avail"]:
            profiles["car"][data] = initialise_dict(labels_day)
            for day_type in labels_day:
                profiles["car"][data][day_type] = initialise_dict(
                    range(self.n_all_clusters_ev))
        self.n_prof["car"] = initialise_dict(labels_day)

        for data, label in zip(["cons", "avail"], ["norm_car", "car_avail"]):
            path = input_dir / "profiles" / label
            files = os.listdir(path)
            for file in files:
                if file[0] != ".":
                    cluster = int(file[1])
                    data_type = file[3: 5]
                    profiles_ = np.load(path / file, mmap_mode="r")
                    # mmap_mode = 'r': not loaded, but elements accessible

                    prof_shape = np.shape(profiles_)
                    if len(prof_shape) == 1:
                        profiles_ = np.reshape(
                            prof_shape, (1, len(prof_shape))
                        )
                    profiles["car"][data][data_type][cluster] = profiles_

        for day_type in labels_day:
            self.n_prof["car"][day_type] = [
                len(profiles["car"]["cons"][day_type][clus])
                for clus in range(self.n_all_clusters_ev)
            ]

        return profiles

    def _load_dem_profiles(self, profiles, prm):
        profiles["loads"] = initialise_dict(prm["syst"]["day_types"])

        self.n_prof["loads"] = {}
        clusters = [
            int(file[1])
            for file in os.listdir(prm['profiles_path'] / "norm_loads")
        ]

        n_dem_clus = max(clusters) + 1

        path = prm["paths"]["hedge_inputs"] / "profiles" / "norm_loads"
        for day_type in prm["syst"]["day_types"]:
            profiles["loads"][day_type] = [
                np.load(path / f"c{cluster}_{day_type}.npy", mmap_mode="r")
                for cluster in range(n_dem_clus)
            ]
            self.n_prof["loads"][day_type] = [
                len(profiles["loads"][day_type][clus])
                for clus in range(n_dem_clus)
            ]

        return profiles

    def _load_gen_profiles(self, inputs_path, profiles):
        path = inputs_path / "profiles" / "norm_gen"

        profiles["gen"] = [np.load(
            path / f"i_month{i_month}.npy", mmap_mode="r")
            for i_month in range(12)
        ]

        self.n_prof["gen"] = [len(profiles["gen"][m]) for m in range(12)]

        return profiles

    def _load_profiles(self, prm: dict) -> dict:
        """Load banks of profiles from files."""
        profiles: Dict[str, Any] = {"car": {}}
        prm['profiles_path'] = prm["paths"]["hedge_inputs"] / "profiles"
        self.n_prof: dict = {}

        # car profiles
        if "car" in self.data_types:
            profiles \
                = self._load_ev_profiles(prm["paths"]["hedge_inputs"], profiles)

        # loads profiles
        if "loads" in self.data_types:
            profiles = self._load_dem_profiles(profiles, prm)

        # PV generation bank and month
        if "gen" in self.data_types:
            profiles = self._load_gen_profiles(prm["paths"]["hedge_inputs"], profiles)

        return profiles

    def _check_feasibility(self, day: dict) -> List[bool]:
        """Given profiles generated, check feasibility."""
        feasible = np.ones(self.n_homes, dtype=bool)
        if self.max_discharge is not None:
            for home in self.homes:
                if self.max_discharge < np.max(day["loads_car"][home]):
                    feasible[home] = False
                    for time in range(len(day["loads_car"][home])):
                        if day["loads_car"][home][time] > self.max_discharge:
                            day["loads_car"][home][time] = self.max_discharge

        for home in self.homes:
            if feasible[home]:
                feasible[home] = self._check_charge(home, day)

        return feasible

    def _check_charge(
            self,
            home: int,
            day: dict
    ) -> bool:
        """Given profiles generated, check feasibility of battery charge."""
        t = 0
        feasible = True

        while feasible:
            # regular initial minimum charge
            min_charge_t_0 = (
                self.store0 * day["avail_car"][home][t]
                if t == self.n_steps - 1
                else self.car["min_charge"] * day["avail_car"][home][t]
            )
            # min_charge if need to charge up ahead of last step
            if day["avail_car"][home][t]:  # if you are currently in garage
                # obtain all future trips
                trip_loads: List[float] = []
                dt_to_trips: List[int] = []

                end = False
                t_trip = t
                while not end:
                    trip_load, dt_to_trip, t_end_trip \
                        = self._next_trip_details(t_trip, home, day)
                    if trip_load is None:
                        end = True
                    else:
                        feasible = self._check_trip_load(
                            feasible, trip_load, dt_to_trip,
                            t, day["avail_car"][home])
                        trip_loads.append(trip_load)
                        dt_to_trips.append(dt_to_trip)
                        t_trip = t_end_trip

                charge_req = self._get_charge_req(
                    trip_loads, dt_to_trips,
                    t_end_trip, day["avail_car"][home])

            else:
                charge_req = 0
            min_charge_t = [
                max(min_charge_t_0[home], charge_req[home])
                for home in self.homes
            ]

            # determine whether you need to charge ahead for next car trip
            # check if opportunity to charge before trip > 37.5
            if feasible and t == 0 and day["avail_car"][home][0] == 0:
                feasible = self._ev_unavailable_start(
                    t, home, day)

            # check if any hourly load is larger than d_max
            if sum(1 for t in range(self.n_steps)
                   if day["loads_car"][home][t] > self.car["d_max"] + 1e-2)\
                    > 0:
                # would have to break constraints to meet demand
                feasible = False

            if feasible:
                feasible = self._check_min_charge_t(
                    min_charge_t, day, home, t)

            t += 1

        return feasible

    def _get_charge_req(self,
                        trip_loads: List[float],
                        dt_to_trips: List[int],
                        t_end_trip: int,
                        avail_ev: List[bool]
                        ) -> float:
        # obtain required charge before each trip, starting with end
        n_avail_until_end = sum(
            avail_ev[t] for t in range(t_end_trip, self.n_steps)
        )
        # this is the required charge for the current step
        # if there is no trip
        # or this is what is needed coming out of the last trip
        if len(trip_loads) == 0:
            n_avail_until_end -= 1
        charge_req \
            = max(0, self.store0 - self.car["c_max"] * n_avail_until_end)
        for it in range(len(trip_loads)):
            trip_load = trip_loads[- (it + 1)]
            dt_to_trip = dt_to_trips[- (it + 1)]
            if it == len(trip_loads) - 1:
                dt_to_trip -= 1
            # this is the required charge at the current step
            # if this is the most recent trip,
            # or right after the previous trip
            charge_req = max(
                0,
                charge_req + trip_load - dt_to_trip * self.car["c_max"]
            )
        return charge_req

    def _check_trip_load(
            self,
            feasible: bool,
            trip_load: float,
            dt_to_trip: int,
            t: int,
            avail_ev_: list
    ) -> bool:
        if trip_load > self.car["cap"] + 1e-2:
            # load during trip larger than whole
            feasible = False
        elif (
                dt_to_trip > 0
                and sum(avail_ev_[0: t]) == 0
                and trip_load / dt_to_trip > self.store0 + self.car["c_max"]
        ):
            feasible = False

        return feasible

    def _ev_unavailable_start(self, t, home, day):
        feasible = True
        trip_load, dt_to_trip, _ \
            = self._next_trip_details(t, home, day)
        if trip_load > self.store0:
            # trip larger than initial charge
            # and straight away not available
            feasible = False
        if sum(day["avail_car"][home][0:23]) == 0 \
                and sum(day["loads_car"][home][0:23]) \
                > self.car["c_max"] + 1e-2:
            feasible = False
        trip_load_next, next_dt_to_trip, _ \
            = self._next_trip_details(dt_to_trip, home, day)
        if next_dt_to_trip > 0 \
            and trip_load_next - (self.car["store0"] - trip_load) \
                < self.car["c_max"] / next_dt_to_trip:
            feasible = False

        return feasible

    def _check_min_charge_t(self,
                            min_charge_t: float,
                            day: dict,
                            home: int,
                            t: int,
                            ) -> bool:
        feasible = True
        if min_charge_t > self.car["cap"] + 1e-2:
            feasible = False  # min_charge_t larger than total cap
        if min_charge_t > self.car["store0"] \
                - sum(day["loads_car"][home][0: t]) \
                + (sum(day["loads_car"][home][0: t]) + 1) * self.car["c_max"] \
                + 1e-3:
            feasible = False

        if t > 0 and sum(day["avail_car"][home][0:t]) == 0:
            # the car has not been available at home to recharge until now
            store_t_a = self.store0 - sum(day["loads_car"][home][0:t])
            if min_charge_t > store_t_a + self.car["c_max"] + 1e-3:
                feasible = False

        return feasible

    def _next_trip_details(
            self,
            start_t: int,
            home: int,
            day: dict) \
            -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Identify the next trip time and requirements for given time step."""
        # next time the car is on a trip
        ts_trips = [
            i for i in range(len(day["avail_car"][home][start_t:]))
            if day["avail_car"][home][start_t + i] == 0
        ]
        if len(ts_trips) > 0 and start_t + ts_trips[0] < self.n_steps:
            # future trip that starts before end
            t_trip = int(start_t + ts_trips[0])

            # next time the car is back from the trip to the garage
            ts_back = [t_trip + i
                       for i in range(len(day["avail_car"][home][t_trip:]))
                       if day["avail_car"][home][t_trip + i] == 1]
            t_back = int(ts_back[0]) if len(ts_back) > 0 \
                else len(day["avail_car"][home])
            dt_to_trip = t_trip - start_t  # time until trip
            t_end_trip = int(min(t_back, self.n_steps))

            # car load while on trip
            trip_load = np.sum(day["loads_car"][home][t_trip: t_end_trip])

            return trip_load, dt_to_trip, t_end_trip

        return None, None, None

    def _plot_ev_avail(self, day, hr_per_t, hours, home):
        bands_ev = []
        non_avail = [
            i for i in range(self.n_steps)
            if day["avail_car"][home][i] == 0
        ]
        if len(non_avail) > 0:
            current_band = [non_avail[0] * hr_per_t]
            if len(non_avail) > 1:
                for i in range(1, len(non_avail)):
                    if non_avail[i] != non_avail[i - 1] + 1:
                        current_band.append(
                            (non_avail[i - 1] + 0.99) * hr_per_t
                        )
                        bands_ev.append(current_band)
                        current_band = [non_avail[i] * hr_per_t]
            current_band.append(
                (non_avail[-1] + 0.999) * hr_per_t
            )
            bands_ev.append(current_band)

        fig, ax = plt.subplots()
        ax.step(
            hours[0: self.n_steps],
            day["loads_car"][home][0: self.n_steps],
            color='k',
            where='post',
            lw=3
        )
        for band in bands_ev:
            ax.axvspan(
                band[0], band[1], alpha=0.3, color='grey'
            )
        grey_patch = matplotlib.patches.Patch(
            alpha=0.3, color='grey', label='car unavailable')
        ax.legend(handles=[grey_patch], fancybox=True)
        plt.xlabel("Time [hours]")
        plt.ylabel("car loads and at-home availability")
        fig.tight_layout()
        fig.savefig(self.save_day_path / f"avail_car_home{home}")
        plt.close("all")

    def _plotting_profiles(self, day, plotting):
        if not plotting:
            return
        if not os.path.exists(self.save_day_path):
            os.mkdir(self.save_day_path)
        y_labels = {
            "car": "Electric vehicle loads",
            "gen": "PV generation",
            "loads": "Household loads"
        }
        font = {'size': 22}
        matplotlib.rc('font', **font)
        hr_per_t = 24 / self.n_steps
        hours = [i * hr_per_t for i in range(self.n_steps)]
        for data_type in self.data_types:
            key = "loads_car" if data_type == "car" else data_type
            for home in self.homes:
                fig = plt.figure()
                print(f"len(hours) {len(hours)} len(day[{key}][{home}]) {len(day[key][home])}")
                plt.plot(hours, day[key][home], color="blue", lw=3)
                plt.xlabel("Time [hours]")
                plt.ylabel(f"{y_labels[data_type]} [kWh]")
                y_fmt = tick.FormatStrFormatter('%.1f')
                plt.gca().yaxis.set_major_formatter(y_fmt)
                plt.tight_layout()
                fig.savefig(self.save_day_path / f"{data_type}_a{home}")
                plt.close("all")

                if data_type == "car":
                    self._plot_ev_avail(day, hr_per_t, hours, home)

    def _init_params(self, prm):
        # add relevant parameters to object properties
        self.data_types = prm["syst"]["data_types"]
        self.behaviour_types = [
            data_type for data_type in self.data_types if data_type != "gen"
        ]

        self.car = prm["car"]
        self.store0 = self.car["SoC0"] * np.array(self.car["cap" + self.passive_ext])
        # update date and time information
        self.date = datetime(*prm["syst"]["date0"])

        return prm
