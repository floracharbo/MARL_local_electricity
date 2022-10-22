#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:31:23 2021.

@author: floracharbonnier

take saved MARL runs and compare what inputs are different
"""

# packages
import os

import numpy as np

# numbers of runs to compare
# RUN_A = 578 # q learning
# RUN_B = 571 # DQN
RUN_A = 342  # facmac good
# RUN_B = 564 # DDPG
RUN_B = 628  # facmac not good
runs = [RUN_A, RUN_B]

# path
PATH = 'outputs/results/'

prm, labels = [], []
for i in range(2):
    RUN = runs[i]
    print(f'RUN {RUN}')
    if 'prm.npy' in os.listdir(PATH + f'run{RUN}/inputData'):
        prm.append(np.load(PATH + f'run{RUN}/inputData/prm.npy',
                           allow_pickle=True).item())
        labels.append('prm')
    else:
        prm.append(np.load(PATH + f'run{RUN}/inputData/syst.npy',
                           allow_pickle=True).item())
        prm[i]['RL'] = np.load(PATH + f'run{RUN}/inputData/lp.npy',
                               allow_pickle=True).item()
        labels.append('syst_lp')

list_objs = prm


def _find_interchangeable_val(current_val):
    interchangeable = [['n_loads_clus', 'nldsclus'],
                       ['cmax', 'c_max'],
                       ['dmax', 'd_max'],
                       ['etach', 'eta_ch'],
                       ['prm', 'syst'],
                       ['wd', 'wd2wd'],
                       ['we', 'we2we'],
                       ['lds', 'loads'],
                       ['n_actions', 'n_discrete_actions'],
                       ['repeats', 'n_repeats'],
                       ['chargetype', 'charge_type'],
                       ['bat', 'car']
                       ]

    i_find_val = [i for i, it in enumerate(interchangeable)
                  if current_val in it]
    if len(i_find_val) > 0:
        it = interchangeable[i_find_val[0]]
        idx_current_val = it.index(current_val)
        idx_other_val = 1 if idx_current_val == 0 else 0
        replacement_val = it[idx_other_val]
    else:
        replacement_val = None

    return replacement_val


def _check_shape_array_vals(obj_a, obj_b, label_a, label_b):
    """Check if obj_a and obj_b are the same (shape + values)."""
    try:
        if isinstance(obj_a, type) and isinstance(obj_b, type):
            if not obj_a == obj_b:
                print(f"{label_a} {obj_a} != {label_b} {obj_b}")
        elif len(np.shape(obj_a)) > 0:
            same_shape = np.shape(obj_a) == np.shape(obj_b)
            if not same_shape:
                print(
                    f"np.shape({label_a}) = {np.shape(obj_a)}, "
                    f"np.shape({label_b}) = {np.shape(obj_b)}")
            elif not (np.array(obj_a) == np.array(obj_b)).all():
                print(f"{label_a} {obj_a} != {label_b} {obj_b}")
        elif obj_a != obj_b:
            print(f"{label_a} {obj_a} != {label_b} {obj_b}")
    except Exception as ex:
        print(ex)


def replace_outdated_labels(k1, objs):
    """Replace dictionary keys with more recent labels."""
    k1_ = k1
    if k1_ not in objs[1]:
        k1_ = _find_interchangeable_val(k1_)
        if k1_ is None:
            print(f"{k1} not in objs[1] = {objs[1].keys()}")

    return k1_


def check_if_two_dicts_match_for_individual_key(objs, k1, k1_, label):
    """Check that values under k1/k1_ are the same for both dictionaries."""
    for k2 in objs[0][k1].keys():
        k2_ = k2
        if k2_ not in objs[1][k1_]:
            replacement_val = _find_interchangeable_val(k2_)
            if replacement_val is None:
                print(f"{k2} not in {label[1]}[{k1_}] "
                      f"= {objs[1][k1_].keys()}")
            else:
                k2_ = replacement_val

        elif isinstance(objs[0][k1][k2], dict):
            for k3 in objs[0][k1][k2].keys():
                k3_ = k3
                if k3_ not in objs[1][k1_][k2_]:
                    k3_ = _find_interchangeable_val(k3_)
                    if k3_ is None:
                        print(f"{k3} not in "
                              f"objs[1][{k1_}][{k2_}] "
                              f"= {objs[1][k1_][k2_].keys()}")
                if k3_ is not None:
                    _check_shape_array_vals(
                        objs[0][k1][k2][k3],
                        objs[1][k1_][k2_][k3_],
                        f'{RUN_A}_{label}_{k1}_{k2}_{k3}',
                        f'{RUN_B}_{label}_{k1_}_{k2}_{k3_}')

        else:
            _check_shape_array_vals(
                objs[0][k1][k2],
                objs[1][k1_][k2_],
                f'{RUN_A}_{label}_{k1}_{k2}',
                f'{RUN_B}_{label}_{k1_}_{k2_}')


for objs, label in zip([list_objs], [labels]):
    for k1 in objs[0].keys():
        k1_ = replace_outdated_labels(k1, objs)

        if k1_ is not None:
            if isinstance(objs[0][k1], dict):
                check_if_two_dicts_match_for_individual_key(objs, k1, k1_, label)
            else:
                _check_shape_array_vals(
                    objs[0][k1],
                    objs[1][k1_],
                    f'{RUN_A}_{label}_{k1}',
                    f'{RUN_B}_{label}_{k1_}')
