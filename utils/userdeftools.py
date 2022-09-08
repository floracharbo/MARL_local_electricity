#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:56:31 2020.

@author: floracharbonnier
"""

import random

import numpy as np
import pandas as pd
import torch as th


def _empty(data):
    """Check if data is empty."""
    if data == '' \
            or data == ' ' \
            or data == [] \
            or data is None \
            or np.isnan(data):
        return True
    else:
        return False


def comb(dims):
    """Obtain all possible combinations between variables."""
    obj = []
    if len(dims) == 2:
        for i in range(dims[0]):
            for j in range(dims[1]):
                obj.append((i, j))
    elif len(dims) == 3:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    obj.append((i, j, k))

    return obj


def str_to_float(listarr):
    """Convert string to float."""
    if isinstance(listarr, str):
        obj = float(listarr) if not _empty(listarr) else []
    elif isinstance(listarr, pd.core.series.Series):
        obj = [float(s) if not _empty(s) else [] for s in listarr]
    elif isinstance(listarr, np.float64):
        obj = float(listarr)
    elif isinstance(listarr[0], list):
        obj = [float(s[0]) if not _empty(s) else [] for s in listarr]
    else:
        obj = [float(s) if not _empty(s) else [] for s in listarr]

    return obj


def str_to_int(listarr):
    """Convert strings to integers."""
    if isinstance(listarr, str):
        obj = int(listarr) if not _empty(listarr) else []
    elif isinstance(listarr, pd.core.series.Series):
        obj = [int(s) if not _empty(s) else [] for s in listarr]
    elif isinstance(listarr, np.float64):
        obj = int(listarr)
    elif isinstance(listarr[0], list):
        obj = [int(s[0]) if not _empty(s) else [] for s in listarr]
    else:
        obj = [int(s) if not _empty(s) else [] for s in listarr]

    return obj


def initialise_dict(entries, type_obj='empty_list', n=1):
    """Initialise a dictionary with keys 'entries'."""
    obj = {}
    for e in entries:
        if type_obj == 'empty_list':
            obj[e] = []
        elif type_obj == 'zeros':
            obj[e] = np.zeros(n)
        elif type_obj == 'empty_dict':
            obj[e] = {}
        elif type_obj == 'zero':
            obj[e] = 0
        elif type_obj == 'Nones':
            obj[e] = [None] * n

    return obj


def play_sound():
    """Play a sound to alert user."""
    import simpleaudio as sa

    filename = '/Users/floracharbonnier/OneDrive - Nexus365/' \
               'DPhil/Python/beep-24.wav'
    wave_obj = sa.WaveObject.from_wave_file(filename)
    wave_obj.play()


def get_moving_average(array, n_window, Nones=True):
    """Get moving average of array over window n_window."""
    x = max(int(n_window / 2 - 0.5), 1)
    n = len(array)
    mova = []
    for j in range(x):
        if Nones:
            mova.append(None)
        else:
            if j == 0:
                mova.append(array[j])
            else:
                if sum(a is None for a in array[0: j * 2 + 1]) == 0:
                    mova.append(np.mean(array[0: j * 2 + 1]))
                else:
                    mova.append(None)
    for j in range(x, n - x):
        if sum(a is None for a in array[j - x: j + x]) == 0:
            mova.append(np.mean(array[j - x: j + x]))
        else:
            mova.append(None)
    for j in range(n - x, n):
        if Nones:
            mova.append(None)
        else:
            if j == len(array) - 1:
                mova.append(array[j])
            else:
                n_to_end = len(array) - j
                if sum(a is None for a in array[j - n_to_end: -1]) == 0:
                    mova.append(np.mean(array[j - n_to_end: -1]))
                else:
                    mova.append(None)

    return mova


def set_seeds_rdn(seed):
    """Seed the random generators."""
    if not isinstance(seed, int):
        seed = int(seed)
    np.random.seed(seed), random.seed(seed)
    th.manual_seed(seed)


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
