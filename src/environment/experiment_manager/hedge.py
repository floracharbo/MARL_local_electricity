"""
Home energy data generator (HEDGE).

Generates subsequent days of car, PV generation and home electricity data
for a given number of homes.

The main method is 'make_next_day', which generates new day of data
(car, loads, gen profiles), calling other methods as needed.
"""

import copy
import os
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import torch as th
import yaml
from scipy.stats import norm

from src.environment.utilities.userdeftools import (f_to_interval,
                                                    initialise_dict,
                                                    list_potential_paths)


def check_file_start(name, file):
    return file[0: len(name)] == name

def car_loads_to_availability(car_loads, tol=1e-2):
    """Tag car availability based on trips' origins and destinations."""
    car_loads[car_loads < tol] = 0
    i_start_trips = np.where((car_loads[:-1] == 0) & (car_loads[1:] > 0))[0] + 1
    if car_loads[0] > 0:
        i_start_trips = np.insert(i_start_trips, 0, 0)
    ev_avail = np.where(car_loads > 0, 0, 1)
    for i in range(0, len(i_start_trips), 2):
        i_start = i_start_trips[i]
        if i + 1 < len(i_start_trips):
            i_start_next = i_start_trips[i + 1]
            ev_avail[i_start: i_start_next] = 0

    return ev_avail, car_loads


class HEDGE:
    """
    Home energy data generator (HEDGE).

    Generates subsequent days of car, gen and loads data for
    a given number of homes.
    """
    def __init__(
        self,
        n_homes: int,
        n_steps: int = 24,
        factors0: Optional[dict] = None,
        clusters0: Optional[dict] = None,
        prm: Optional[dict] = None,
        other_prm: Optional[dict] = None,
        n_consecutive_days: Optional[int] = 2,
        brackets_definition: Optional[str] = 'percentile',
        ext='',
    ):
        """Initialise HEDGE object and initial properties."""
        # update object properties
        self.labels_day = ["wd", "we"]
        self.n_homes = n_homes
        self.n_steps = n_steps
        self.n_consecutive_days = n_consecutive_days
        self.brackets_definition = brackets_definition
        self.ext = ext

        self.homes = range(self.n_homes)
        self.it_plot = 0
        self._load_input_data(prm, other_prm, factors0, clusters0, brackets_definition)

    def _load_input_data(self, prm, other_prm, factors0, clusters0, brackets_definition):
        prm = self._load_inputs(prm, brackets_definition)
        prm = self._replace_car_prm(prm, other_prm)
        self._init_factors(factors0)
        self._init_clusters(clusters0)
        self.profile_generator = self._load_profile_generators(prm)

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

        factors, interval_f = self._next_factors(transition, prev_clusters, homes)
        clusters = self._next_clusters(transition, prev_clusters)

        # obtain days
        day = {
            info: np.zeros((len(homes), self.n_steps))
            for info in ['gen', 'loads', 'loads_car', 'avail_car']
        }
        for data_type in self.data_types:
            day_type_ = '' if data_type == 'gen' else day_type
            data_type_ = 'loads_car' if data_type == 'car' else data_type
            for i_home, home in enumerate(homes):
                cluster = clusters[data_type][home] if data_type in self.behaviour_types \
                    else self.date.month - 1
                generated_profile = self._generate_profile(data_type, day_type_, cluster)
                day[data_type_][i_home] = generated_profile * factors[data_type][i_home]

        if 'car' in self.data_types:
            # check loads car are consistent with maximum battery load
            interval_f['car'], factors['car'], day = self._adjust_max_ev_loads(
                day, interval_f['car'], factors['car'], transition, clusters,
                day_type, homes
            )
            for i_home, home in enumerate(homes):
                day["avail_car"][i_home], day['loads_car'][i_home] \
                    = car_loads_to_availability(day['loads_car'][i_home])

        for data_type in self.data_types:
            self.factors[data_type][homes] = factors[data_type]
        if any(factor > self.car['max_daily_energy_cutoff'] for factor in factors['car']):
            print('Warning: max daily energy cutoff exceeded')
        self.clusters = clusters

        # save factors and clusters
        for data_type in self.data_types:
            self.list_factors[data_type] = np.hstack(
                (
                    self.list_factors[data_type],
                    np.reshape(self.factors[data_type], (self.n_homes, 1))
                )
            )

        for data_type in self.behaviour_types:
            self.list_clusters[data_type] = np.hstack(
                (
                    self.list_clusters[data_type],
                    np.reshape(self.clusters[data_type], (self.n_homes, 1))
                )
            )

        self._plotting_profiles(day, plotting)

        return day

    def _import_cdfs(self, prm):
        self.select_cdfs = {}
        for data_type in self.behaviour_types:
            self.select_cdfs[data_type] = {}
            for day_type in prm["syst"]["weekday_types"]:
                self.select_cdfs[data_type][day_type] = [
                    min_cdf + prm["syst"]["clus_dist_share"] * (max_cdf - min_cdf)
                    for min_cdf, max_cdf in zip(
                        self.min_cdfs[data_type][day_type],
                        self.max_cdfs[data_type][day_type]
                    )
                ]

    def _load_inputs(self, prm, brackets_definition):
        # load inputs
        if prm is None:
            with open("config_parameters/hedge_config.yaml", "rb") as file:
                prm = yaml.safe_load(file)
            with open("config_parameters/fixed_parameters.yaml", "rb") as file:
                syst = yaml.safe_load(file)
            for info in syst:
                prm["syst"][info] = syst[info]

        self._init_params(prm)

        # general inputs with all data types
        if not (Path(prm["paths"]["hedge_inputs"])).exists():
            prm["paths"]["hedge_inputs"] += "_sample"
            print("Using sample data. See README for how to prepare the total input data.")

        factors_path = prm["paths"]["hedge_inputs"] / "factors"

        properties = ["f_min", "f_max", "f_mean", "residual_distribution_prms"]
        for property_ in properties:
            path = factors_path / f"{property_}.pickle"
            with open(path, "rb") as file:
                self.__dict__[property_] = pickle.load(file)

        properties = ["p_pos", "p_zero2pos", "mid_fs_brackets", "fs_brackets"]
        for property_ in properties:
            path = factors_path \
                / f"{property_}_n_consecutive_days{self.n_consecutive_days}_" \
                f"brackets_definition_{brackets_definition}.pickle"
            with open(path, "rb") as file:
                setattr(self, property_, pickle.load(file))

        for brackets in ['fs_brackets', 'mid_fs_brackets']:
            self.__dict__[brackets]['gen']['all'] = np.array(self.__dict__[brackets]['gen']['all'])
            self.__dict__[brackets]['gen']['all'] *= prm['syst']['f0']['gen'] / self.f_mean['gen'][0]
            for info in ['car', 'loads']:
                for transition in prm['syst']['day_trans'] + ['all']:
                    if not (info == 'car' and transition == 'we2wd'):
                        self.__dict__[brackets][info][transition] = np.array(self.__dict__[brackets][info][transition])
                        self.__dict__[brackets][info][transition] *= prm['syst']['f0'][info] / self.f_mean[info]

        clusters_path = prm["paths"]["hedge_inputs"] / "clusters"
        for property_ in [
            "p_clus", "p_trans", "min_cdfs", "max_cdfs", "clus_dist_bin_edges",
            "clus_dist_cdfs", "fitted_kmeans_obj", "fitted_scalers"
        ]:
            potential_paths = list_potential_paths(
                prm, data_types=self.data_types,
                root_path=prm['paths']["input_folder"],
                data_folder='hedge_inputs', sub_data_folder='clusters'
            )
            file_found = False
            for potential_path in potential_paths:
                path = potential_path / f"{property_}.pickle"
                try:
                    with open(str(path), "rb") as file:
                        setattr(self, property_, pickle.load(file))
                    file_found = True
                    break
                except FileNotFoundError:
                    pass

            if not file_found:
                print(f"no file found for {property_}")

        with open(clusters_path / "n_clus.pickle", "rb") as file:
            prm["n_clus"] = pickle.load(file)
        self.n_all_clusters = {
            data_type: prm['n_clus'][data_type] + 1 if data_type == 'car'
            else prm['n_clus'][data_type]
            for data_type in self.behaviour_types
        }
        self.n_all_clusters['gen'] = 12

        # household demand-specific inputs
        self._import_cdfs(prm)

        # PV generation-specific inputs
        if "gen" in self.data_types:
            self.residual_distribution_prms["gen"] = list(self.residual_distribution_prms["gen"])
            self.residual_distribution_prms["gen"][1] *= prm["syst"]["f_std_share"]
            self.select_cdfs["gen"] = [
                min_cdf + prm["syst"]["clus_dist_share"] * (max_cdf - min_cdf)
                for min_cdf, max_cdf in zip(self.min_cdfs["gen"], self.max_cdfs["gen"])
            ]

        return prm

    def _replace_car_prm(self, prm, other_prm):
        prm = copy.deepcopy(prm)
        if other_prm is not None:
            for key, val in other_prm.items():
                for subkey, subval in val.items():
                    prm[key][subkey] = subval
        # add relevant parameters to object properties
        self.car = prm["car"]
        self.store0 = self.car["SoC0"] * np.array(self.car['caps' + self.ext])
        if 'own_car' not in self.car:
            self.car['own_car'] = np.ones(self.n_homes, dtype=bool)

        return prm

    def _init_factors(self, factors0):
        self.factors0 = factors0
        _, transition = self._transition_type()
        self.factors = {}
        if self.factors0 is None:
            if "loads" in self.data_types:
                self.factors["loads"] = [
                    self.f_mean["loads"]
                    + norm.ppf(
                        np.random.rand(),
                        *self.residual_distribution_prms["loads"][transition][1:]
                    )
                    for _ in self.homes
                ]
            if "gen" in self.data_types:
                self.factors["gen"] = [
                    self.f_mean["gen"][self.date.month - 1]
                    + norm.ppf(
                        np.random.rand(),
                        *self.residual_distribution_prms["gen"][1:])
                    for _ in self.homes
                ]
                i_month = self.date.month - 1
                self.factors["gen"] = np.minimum(
                    np.maximum(self.f_min['gen'][i_month], np.array(self.factors['gen'])),
                    self.f_max["gen"][i_month]
                )

            if "car" in self.data_types:
                randoms = np.random.rand(self.n_homes)
                self.factors["car"] = [
                    self._ps_rand_to_choice(
                        self.p_zero2pos['car'][transition],
                        randoms[home]
                    )
                    for home in self.homes
                ]
        else:
            for data_type in self.data_types:
                if isinstance(self.factors0[data_type], (int, float)):
                    self.factors[data_type] = np.full(
                        self.n_homes, self.factors0[data_type], dtype=float
                    )
                else:
                    self.factors[data_type] = self.factors0[data_type]
        for data_type in self.behaviour_types:
            self.factors[data_type] = np.minimum(
                np.maximum(self.f_min[data_type], self.factors[data_type]),
                self.f_max[data_type]
            )
        self.list_factors = {
            data_type: np.zeros((self.n_homes, self.n_consecutive_days - 1))
            for data_type in self.data_types
        }
        for data_type in self.data_types:
            for home in self.homes:
                self.list_factors[data_type][home] = np.full(
                    self.n_consecutive_days - 1, self.factors[data_type][home]
                )

    def _init_clusters(self, clusters0):
        self.clusters0 = clusters0
        day_type, transition = self._transition_type()
        self.clusters = {}

        if self.clusters0 is None:
            for data_type in self.behaviour_types:
                self.clusters[data_type] \
                    = [self._ps_rand_to_choice(
                        self.p_clus[data_type][day_type], np.random.rand())
                        for _ in self.homes]
        else:
            for data_type in self.behaviour_types:
                if isinstance(self.clusters0[data_type], int):
                    self.clusters[data_type] = [self.clusters0[data_type] for _ in self.homes]
                else:
                    self.clusters[data_type] = self.clusters0[data_type]

        self.list_clusters = {
            data_type: np.zeros((self.n_homes, self.n_consecutive_days - 1))
            for data_type in self.behaviour_types
        }
        for data_type in self.behaviour_types:
            for home in self.homes:
                self.list_clusters[data_type][home] = np.full(
                    self.n_consecutive_days - 1, self.clusters[data_type][home]
                )

    def _next_factors(self, transition, prev_clusters, homes):
        prev_factors = {
            data_type: self.list_factors[data_type][:, - (self.n_consecutive_days - 1):]
            for data_type in self.data_types
        }
        interval_f = {data_type: np.zeros(len(homes), dtype=int) for data_type in self.data_types}
        factors = {data_type: np.zeros(len(homes)) for data_type in self.data_types}
        random_f = {data_type: np.random.rand(len(homes)) for data_type in self.data_types}
        for i_home, home in enumerate(homes):
            for data_type in self.data_types:
                if self.own_der[data_type][home]:
                    if data_type == 'gen' or (data_type == 'car' and transition == 'we2wd'):
                        transition_ = 'all'
                    else:
                        transition_ = transition
                    previous_intervals = tuple(
                        f_to_interval(
                            prev_factors[data_type][home][- (self.n_consecutive_days - 1 - d)],
                            self.fs_brackets[data_type][transition_]
                        )
                        for d in range(self.n_consecutive_days - 1)
                    )
                    if (
                            data_type == 'car'
                            and prev_clusters[data_type][home] == self.n_all_clusters[data_type] - 1
                    ):
                        # no trip day
                        probabilities = self.p_zero2pos[data_type][transition_]
                    else:
                        probabilities = self.p_pos[data_type][transition_][previous_intervals]
                    interval_f[data_type][i_home] = self._ps_rand_to_choice(
                        probabilities,
                        random_f[data_type][i_home],
                    )
                    prev_factor = self.list_factors[data_type][home][-1]
                    factor = self.mid_fs_brackets[data_type][transition_][interval_f[data_type][i_home]]
                    factor = prev_factor + self.clus_dist_share * (factor - prev_factor)
                    factors[data_type][i_home] = factor
                else:
                    factors[data_type][i_home] = 0

        return factors, interval_f

    def _next_clusters(self, transition, prev_clusters):
        clusters = initialise_dict(self.behaviour_types)

        random_clus = [
            [np.random.rand() for _ in self.homes]
            for _ in self.behaviour_types
        ]
        for home in self.homes:
            for it, data_type in enumerate(self.behaviour_types):
                prev_cluster = prev_clusters[data_type][home]
                probs = self.p_trans[data_type][transition][prev_cluster]
                # cum_p = [sum(probs[0:i]) for i in range(1, len(probs))] + [1]
                # clusters[data_type].append(
                #     [c > random_clus[it][home] for c in cum_p].index(True)
                # )
                clusters[data_type].append(0)
        return clusters

    def _transition_type(self):
        day_type = "wd" if self.date.weekday() < 5 else "we"
        prev_day_type \
            = "wd" if (self.date - timedelta(days=1)).weekday() < 5 \
            else "we"
        transition = f"{prev_day_type}2{day_type}"

        return day_type, transition

    def _adjust_max_ev_loads(
        self, day, interval_f_car, factors, transition, clusters, day_type, homes
    ):
        transition_ = 'all' if transition == 'we2wd' else transition
        homes = [home for home in homes if self.car['own_car'][home]]
        for i_home, home in enumerate(homes):
            it = 0
            max_loads_car = np.max(day["loads_car"][i_home])
            car_cap = self.car['caps' + self.ext][home]
            factor = factors[i_home]
            while (
                    np.max(day["loads_car"][i_home]) > self.car['caps' + self.ext][home]
                    or factors[i_home] > self.car['max_daily_energy_cutoff']
            ) and it < 100:
                if it == 99:
                    print("100 iterations _adjust_max_ev_loads")
                if factors[i_home] > 0 and interval_f_car[i_home] > 0:
                    factor0 = factors[i_home].copy()
                    interval_f_car[i_home] -= 1
                    mid_fs_brackets = self.mid_fs_brackets['car'][transition_]
                    factors[i_home] = mid_fs_brackets[interval_f_car[i_home]]
                    day['loads_car'][i_home] *= factors[i_home] / factor0

                else:
                    profile = self._generate_profile('car', day_type, clusters['car'][home])
                    day['loads_car'][i_home] = profile * factors[i_home]
                    day['avail_car'][i_home], day['loads_car'][i_home] = car_loads_to_availability(
                        day['loads_car'][i_home]
                    )

                it += 1

        return interval_f_car, factors, day

    def forward(self, generator, data_type, n_steps_nonzero, min=None, max=None):
        output = generator(th.randn(1, 1))
        size_output = n_steps_nonzero * self.n_items
        if data_type != 'car':
            output = output.reshape(-1, n_steps_nonzero)
            output = th.div(output, th.sum(output, dim=1).reshape(-1, 1)).reshape(-1, size_output)
            output = th.clamp(output, min=min, max=max)
        else:
            output = th.clamp(output, min=min)
        output = output.detach().numpy()

        return output

    def _generate_profile(self, data_type, day_type, cluster):
        # day type can also be month index
        its = 0
        if data_type == 'car' and cluster == self.n_all_clusters['car'] - 1:
            return np.zeros(self.n_steps)
        profile_validated = False
        cluster_ = 12 if data_type == 'gen' else cluster
        fitted_kmeans_id = self.month0 - 1 if data_type == 'gen' else day_type
        generator = self.profile_generator[f"{data_type}_{day_type}_{cluster_}"]
        if data_type == 'gen':
            clus_dist_cdfs = self.clus_dist_cdfs[data_type][fitted_kmeans_id]
            select_cdfs = self.select_cdfs[data_type][fitted_kmeans_id]
            clus_dist_bin_edges = self.clus_dist_bin_edges[data_type][fitted_kmeans_id]
        else:
            clus_dist_cdfs = self.clus_dist_cdfs[data_type][fitted_kmeans_id][cluster]
            select_cdfs = self.select_cdfs[data_type][fitted_kmeans_id][cluster]
            clus_dist_bin_edges = self.clus_dist_bin_edges[data_type][fitted_kmeans_id][cluster]
        max_dist = clus_dist_bin_edges[np.where(select_cdfs > clus_dist_cdfs)[0][-1]]
        zero_values = self.zero_values[f"{data_type}_{day_type}_{cluster_}"]
        n_steps_nonzero = sum(~zero_values)
        while not profile_validated and its < 1000:
            if its == 999:
                print("1000 iterations _generate_profile")
            if its % self.n_items == 0:
                generated_profiles = self.forward(
                    generator, data_type, n_steps_nonzero,
                    min=self.min[f"{data_type}_{day_type}_{cluster_}"],
                    max=self.max[f"{data_type}_{day_type}_{cluster_}"]
                )
            it_i_profile = 0
            found_non_nan_profile = False
            while it_i_profile < 1000 and not found_non_nan_profile:
                i_profile = random.randint(0, self.n_items - 1)
                idx = n_steps_nonzero * i_profile
                profile_nonzero = generated_profiles[0, idx: idx + n_steps_nonzero]
                profile = np.zeros(self.n_steps)
                profile[~zero_values] = profile_nonzero
                if np.sum(np.isnan(profile)) == 0:
                    found_non_nan_profile = True
                else:
                    print(f"nans in profile {data_type} fitted_kmeans_id {fitted_kmeans_id} cluster_ {cluster_}")
                if it_i_profile == 999:
                    print("1000 iterations _generate_profile")
            if self.clus_dist_share < 1:
                transformed_features = self._get_transformed_features(
                    profile, data_type, fitted_kmeans_id
                )
                fitted_kmeans_obj = self.fitted_kmeans_obj[data_type][fitted_kmeans_id]
                idx = 0 if data_type == 'gen' else cluster_
                cluster_distance = fitted_kmeans_obj.transform(transformed_features)[0, idx]
                profile_validated = cluster_distance < max_dist
            else:
                profile_validated = True

            if abs(np.sum(profile) - 1) > 1:
                if abs(np.sum(profile) - 1) > 1.5:
                    profile_validated = False
                else:
                    profile = profile / np.sum(profile)

            if its > 1 / self.clus_dist_share * 100 and not profile_validated:
                print(
                    f"{its} iterations _generate_profile {data_type} "
                    f"np.sum(profile) {np.sum(profile)} "
                )
                if self.clus_dist_share < 1:
                    print(f"cluster_distance {cluster_distance} (max_dist {max_dist})")
                break

            profile /= np.sum(profile)
            its += 1

        return profile

    def _compute_number_of_available_profiles(self, data_type, day_type, i_month):
        if data_type in self.behaviour_types:
            n_profs0 = [
                self.n_prof[data_type][day_type][cluster]
                for cluster in range(len(self.n_prof[data_type][day_type]))
            ]
            if data_type == 'car':
                for cluster in range(len(self.n_prof[data_type][day_type])):
                    assert self.n_prof[data_type][day_type][cluster] \
                           == len(self.profs[data_type]["cons"][day_type][cluster]), \
                           f"self.n_prof[{data_type}][{day_type}][{cluster}] " \
                           f"{self.n_prof[data_type][day_type][cluster]}"

        else:
            n_profs0 = self.n_prof[data_type][i_month]

        return n_profs0

    def _select_profiles(
            self,
            data_type: str,
            day_type: str = None,
            i_month: int = 0,
            clusters: List[int] = None
    ) -> List[int]:
        """Randomly generate index of profile to select for given data."""
        i_profs = []
        n_profs0 = self._compute_number_of_available_profiles(data_type, day_type, i_month)
        n_profs = n_profs0
        for home in self.homes:
            if data_type in self.behaviour_types:
                n_profs_ = n_profs[clusters[data_type][home]]
                if n_profs_ > 1:
                    n_profs[clusters[data_type][home]] -= 1
            else:
                n_profs_ = n_profs
                n_profs -= 1
            i_prof = round(np.random.rand() * (n_profs_ - 1))
            for previous_i_prof in sorted(i_profs):
                if previous_i_prof <= i_prof < n_profs_ - 1 and n_profs_ > 1:
                    i_prof += 1
            i_profs.append(i_prof)
            profs = self.profs[data_type]
            if data_type == "car":
                assert i_prof < len(profs["cons"][day_type][clusters[data_type][home]]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"
            elif data_type == "loads":
                assert i_prof < len(profs[day_type][clusters[data_type][home]]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"
            else:
                assert i_prof < len(profs[i_month]), \
                    f"i_profs {i_profs} i_prof {i_prof} " \
                    f"n_profs_ {n_profs_} n_profs {n_profs} n_profs0 {n_profs0}"

        return i_profs

    def _ps_rand_to_choice(self, probs: List[float], rand: float) -> int:
        """Given list of probabilities, select index."""
        p_intervals = np.cumsum(probs)
        choice = np.where(rand <= p_intervals)[0][0]

        return choice

    def _load_dem_profiles(self, profiles, prm):
        profiles["loads"] = initialise_dict(prm["syst"]["weekday_type"])

        self.n_prof["loads"] = {}
        clusters = [
            int(file[1])
            for file in os.listdir(prm['profiles_path'] / "norm_loads")
        ]

        n_dem_clus = max(clusters) + 1

        path = prm["paths"]["hedge_inputs"] / "profiles" / "norm_loads"
        for day_type in prm["syst"]["weekday_type"]:
            profiles["loads"][day_type] = [
                np.load(path / f"c{cluster}_{day_type}.npy", mmap_mode="r")
                for cluster in range(n_dem_clus)
            ]
            self.n_prof["loads"][day_type] = [
                len(profiles["loads"][day_type][clus])
                for clus in range(n_dem_clus)
            ]

        return profiles

    def _load_profile_generators(self, prm: dict) -> dict:
        """Load profile generators."""
        prm['profiles_path'] = prm["paths"]["hedge_inputs"] / "profiles"
        profile_generator = {}
        self.zero_values = {}
        self.min = {}
        self.max = {}
        for data_type in self.data_types:
            path = prm['profiles_path'] / f"norm_{data_type}"
            files = os.listdir(path)
            for file in files:
                if any(check_file_start(name, file) for name in ['generator', 'zerovalues', 'min', 'max']):
                    day_type = file.split('_')[2]
                    cluster = file.split('_')[3].split('.')[0]
                    if check_file_start("generator", file):
                        profile_generator[f"{data_type}_{day_type}_{cluster}"] = th.load(path / file)
                    if check_file_start("zerovalues", file):
                        self.zero_values[f"{data_type}_{day_type}_{cluster}"] = np.load(path / file)
                    if check_file_start("min", file):
                        self.min[f"{data_type}_{day_type}_{cluster}"] = th.Tensor(np.load(path / file))
                    if check_file_start("max", file):
                        self.max[f"{data_type}_{day_type}_{cluster}"] = th.Tensor(np.load(path / file))

        return profile_generator

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
        time_step = 0
        feasible = True

        while feasible:
            # regular initial minimum charge
            min_charge_t_0 = (
                self.store0 * day["avail_car"][home, time_step]
                if time_step == self.n_steps - 1
                else self.car["min_charge"] * day["avail_car"][home, time_step]
            )
            # min_charge if need to charge up ahead of last step
            if day["avail_car"][home, time_step]:  # if car is currently in garage
                # obtain all future trips
                trip_loads: List[float] = []
                dt_to_trips: List[int] = []

                end = False
                t_trip = time_step
                while not end:
                    trip_load, dt_to_trip, t_end_trip \
                        = self._next_trip_details(t_trip, home, day)
                    if trip_load is None:
                        end = True
                    else:
                        feasible = self._check_trip_load(
                            feasible, trip_load, dt_to_trip,
                            time_step, day["avail_car"][home])
                        trip_loads.append(trip_load)
                        dt_to_trips.append(dt_to_trip)
                        t_trip = t_end_trip

                charge_req = self._get_charge_req(
                    trip_loads, dt_to_trips, t_end_trip, day["avail_car"][home]
                )

            else:
                charge_req = 0
            min_charge_t = np.max(min_charge_t_0, charge_req)
            # determine whether you need to charge ahead for next car trip
            # check if opportunity to charge before trip > 37.5
            if feasible and time_step == 0 and day["avail_car"][home][0] == 0:
                feasible = self._ev_unavailable_start(time_step, home, day)

            # check if any hourly load is larger than d_max
            if sum(1 for time_step in range(self.n_steps)
                   if day["loads_car"][home, time_step] > self.car["d_max"] + 1e-2)\
                    > 0:
                # would have to break constraints to meet demand
                feasible = False

            if feasible:
                feasible = self._check_min_charge_t(min_charge_t, day, home, time_step)

            time_step += 1

        return feasible

    def _get_charge_req(self,
                        trip_loads: List[float],
                        dt_to_trips: List[int],
                        t_end_trip: int,
                        avail_car: List[bool]
                        ) -> float:
        # obtain required charge before each trip, starting with end
        n_avail_until_end = sum(avail_car[t_end_trip: self.n_steps])
        # this is the required charge for the current step
        # if there is no trip
        # or this is what is needed coming out of the last trip
        if len(trip_loads) == 0:
            n_avail_until_end -= 1
        charge_req = max(0, self.store0 - self.car["c_max"] * n_avail_until_end)
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
            time_step: int,
            avail_car: list
    ) -> bool:
        if trip_load > self.car["caps" + self.ext] + 1e-2:
            # load during trip larger than whole
            feasible = False
        elif (
                dt_to_trip > 0
                and sum(avail_car[0: time_step]) == 0
                and trip_load / dt_to_trip > self.store0 + self.car["c_max"]
        ):
            feasible = False

        return feasible

    def _ev_unavailable_start(self, time_step, home, day):
        feasible = True
        trip_load, dt_to_trip, _ \
            = self._next_trip_details(time_step, home, day)
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
                            time_step: int,
                            ) -> bool:
        feasible = True
        if min_charge_t > self.car["caps" + self.ext][home] + 1e-2:
            feasible = False  # min_charge_t larger than total cap
        if min_charge_t > self.car["store0"] \
                - sum(day["loads_car"][home][0: time_step]) \
                + (sum(day["loads_car"][home][0: time_step]) + 1) * self.car["c_max"] \
                + 1e-3:
            feasible = False

        if time_step > 0 and sum(day["avail_car"][home][0: time_step]) == 0:
            # the car has not been available at home to recharge until now
            store_t_a = self.store0 - sum(day["loads_car"][home][0: time_step])
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

    def _plot_ev_avail(self, day_plot, avail_car_plot, hr_per_t, hours, home, cumulative_plot):
        bands_ev = []
        n_steps = self.n_steps * self.it_plot if cumulative_plot else self.n_steps
        non_avail = [
            i for i in range(n_steps)
            if avail_car_plot[i] == 0
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
            hours[0: n_steps],
            day_plot[0: n_steps],
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
        plt.ylabel("Car loads [kWh]")
        fig.tight_layout()
        title = \
            f"avail_car_home{home}_n_consecutive_days{self.n_consecutive_days}_" \
            f"brackets_definition_{self.brackets_definition}_" \
            f"clus_dist_share{self.clus_dist_share}".replace('.', '_')
        if cumulative_plot:
            title += "_cumulative"
            for i in range(self.it_plot):
                plt.vlines(i * self.n_steps, 0, max(day_plot), ls='--', color='k')
        fig.savefig(self.save_day_path / title)
        plt.close("all")

    def _plotting_profiles(self, day, plotting):
        if not plotting:
            return
        if not os.path.exists(self.save_day_path):
            os.mkdir(self.save_day_path)
        np.save(self.save_day_path / f"day_{self.it_plot}", day)
        np.save(
            self.save_day_path
            / f"list_factors_{self.it_plot}_brackets_definition_{self.brackets_definition}",
            self.list_factors
        )
        np.save(
            self.save_day_path
            / f"list_clusters_{self.it_plot}_n_consecutive_days{self.n_consecutive_days}",
            self.list_clusters
        )
        self.it_plot += 1
        y_labels = {
            "car": "Electric vehicle loads",
            "gen": "PV generation",
            "loads": "Household loads"
        }
        font = {'size': 22}
        matplotlib.rc('font', **font)
        hr_per_t = 24 / self.n_steps

        for cumulative_plot in [False, True]:
            if cumulative_plot and plotting:
                hours = [i * hr_per_t for i in range(self.n_steps * self.it_plot)]
                for info in ['factors', 'clusters']:
                    list_info = getattr(self, f"list_{info}")
                    for data_type in list_info:
                        fig = plt.figure()
                        for home in range(self.n_homes):
                            plt.plot(list_info[data_type][home])
                        plt.xlabel("Day")
                        plt.ylabel(f"{info} {data_type}")
                        fig.tight_layout()
                        fig.savefig(
                            self.save_day_path
                            / f"{info}_{data_type}_n_consecutive_days{self.n_consecutive_days}_"
                            f"cumulative_brackets_definition_{self.brackets_definition}"
                        )
            else:
                hours = [i * hr_per_t for i in range(self.n_steps)]

            for data_type in self.data_types:
                key = "loads_car" if data_type == "car" else data_type
                for home in self.homes:
                    if cumulative_plot:
                        day_plot = np.array([])
                        if data_type == "car":
                            avail_car_plot = np.array([])
                        for i in range(self.it_plot):
                            day_i = day if i == self.it_plot - 1 else np.load(
                                self.save_day_path / f"day_{i}.npy",
                                allow_pickle=True
                            ).item()
                            day_plot = np.concatenate((day_plot, day_i[key][home]))
                            if data_type == "car":
                                avail_car_plot = np.concatenate(
                                    (avail_car_plot, day_i["avail_car"][home])
                                )

                    else:
                        day_plot = day[key][home]
                        if 'car' in self.data_types:
                            avail_car_plot = day["avail_car"][home]
                    fig = plt.figure()
                    plt.plot(hours, day_plot, color="k", lw=3)
                    plt.xlabel("Time [hours]")
                    plt.ylabel(f"{y_labels[data_type]} [kWh]")
                    y_fmt = tick.FormatStrFormatter('%.1f')
                    plt.gca().yaxis.set_major_formatter(y_fmt)
                    plt.tight_layout()
                    title = \
                        f"{data_type}_a{home}_n_consecutive_days{self.n_consecutive_days}_" \
                        f"brackets_definition_{self.brackets_definition}_" \
                        f"clus_dist_share{self.clus_dist_share}".replace('.', '_')
                    if cumulative_plot:
                        title += "_cumulative"
                        for i in range(self.it_plot):
                            plt.vlines(i * self.n_steps, 0, max(day_plot), ls='--', color='k')
                    fig.savefig(self.save_day_path / title)
                    plt.close("all")

                    if data_type == "car":
                        self._plot_ev_avail(
                            day_plot, avail_car_plot, hr_per_t, hours, home, cumulative_plot
                        )

        for data_type in self.list_factors:
            fig = plt.figure()
            for home in self.homes:
                plt.plot(self.list_factors[data_type][home])
            plt.xlabel("Day")
            plt.ylabel(f"{data_type} factor")
            plt.tight_layout()
            fig.savefig(self.save_day_path / f"{data_type}_factors")
            plt.close("all")

        for data_type in self.list_clusters:
            fig = plt.figure()
            for home in self.homes:
                plt.plot(self.list_clusters[data_type][home])
            plt.xlabel("Day")
            plt.ylabel(f"{data_type} cluster")
            plt.tight_layout()
            fig.savefig(self.save_day_path / f"{data_type}_clusters")
            plt.close("all")

    def _init_params(self, prm):
        # add relevant parameters to object properties
        for info in ['data_types', 'n_items', 'n_items_gen', 'clus_dist_share', 'dem_intervals', 'month0']:
            setattr(self, info, prm['syst'][info])
        self.own_loads = prm['loads']['own_loads' + self.ext]
        self.own_PV = prm['gen']['own_PV' + self.ext]
        self.own_car = prm['car']['own_car' + self.ext]
        self.own_der = {
            'car': self.own_car,
            'loads': self.own_loads,
            'gen': self.own_PV,
        }
        self.behaviour_types = [
            data_type for data_type in self.data_types if data_type != "gen"
        ]
        self.car = prm["car"]
        if 'caps' not in self.car and isinstance(self.car['cap'], int):
            self.car['caps' + self.ext] = np.full(self.n_homes, self.car['cap'])
        self.store0 = self.car["SoC0"] * np.array(self.car['cap'])
        # update date and time information
        self.date = datetime(*prm["syst"]["date0"])
        self.save_day_path = Path(prm["paths"]["record_folder"]) / "hedge_days"

    def _get_transformed_features(self, profile, data_type, cluster_id):
        features = []
        if data_type == "loads":
            peak = np.max(profile)
            t_peak = np.argmax(profile)
            values = [
                np.mean(
                    profile[
                        int(interval[0] * self.n_steps / 24): int(interval[1] * self.n_steps / 24)
                    ]
                )
                for interval in self.dem_intervals
            ]
            features = [peak, t_peak] + values

        elif data_type in ["car", "gen"]:
            features = profile[
                int(6 * 24 / self.n_steps): int(22 * 24 / self.n_steps)
            ]
        transformed_features = self.fitted_scalers[data_type][cluster_id].transform(
            np.reshape(features, (1, -1))
        )

        return transformed_features
