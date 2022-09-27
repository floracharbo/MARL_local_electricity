#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:48:27 2021.

@author: Flora Charbonnier
"""

# import python packages
import os
from pathlib import Path
from typing import Tuple

# to turn the input data into an usable format
from config.initialise_prm import initialise
# import user-defined functions and modules
# scripts where input data is stored
from config.input_data import input_params
# for trying different protocols for learning
from simulations.record import Record


def initialise_objects(
        prm: dict,
        settings: dict = None,
        no_run: int = None,
        initialise_record: bool = True
) -> Tuple[dict, object, dict]:
    """
    Set up parameters dictionary, load data, initialise recording object.

    inputs:
    prm:
        dictionary of run parameters;
        has attributes bat, grd, loads, ntw, prm, gen (from inputs files)
    settings:
        in main_rl.py, defaults settings may be overrun
    no_run:
        no of current run for folder naming/recording purposes
    initialise_record:
        boolean for whether we need to
        initialise the record object
        e.g. not needed from plot_summary_no_agents where
        we only need high-level data

    output:
    prm:
        dictionary of parameters; with settings updated
        and intermediate variables computed
    record:
        the object to keep a record of the data and experiments
    profiles:
        the battery, generation and loads profiles
        to input to the environment
        (not in prm so as not to carry those large datasets around)
    """
    # general input paths and system parameters are in inputs
    # where
    prm = input_params(prm, settings)
    for folder in ['results', 'opt_res']:
        if not Path(folder).exists():
            os.mkdir(folder)

    if no_run is None:
        prev_runs = \
            [r for r in os.listdir('results')
             if r[0:3] == 'run']
        no_prev_runs = [int(r[3:]) for r in prev_runs]
        no_run = max(no_prev_runs + [0]) + 1

    # turn into an usable format
    prm, profiles = initialise(prm, no_run, initialise_all=initialise_record)

    if initialise_record:
        # initialise recording of progress - rewards, counters, etc.
        record = Record(prm, no_run=no_run)
    else:
        record = False
    return prm, record, profiles
