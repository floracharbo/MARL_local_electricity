#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 2023.

@author: julie-vienne
"""

import numpy as np
import os



class ComputationalBurdenAnalyzer:
    """
    Performs and saves calculations on computational burden of
    solving the optimiaztion or running a pandapower simulation

    """

    def __init__(self, prm):
        """
        Initialise ComputationalBurdenAnalyzer object.

        inputs:
        prm:
            input parameters
        """

        # subset line losses
        for info in [
            'subset_line_losses_modelled'
            ]:
            setattr(self, info, prm['grd'][info])

    def _save_computational_burden_opti(
        self, time_to_solve_opti, number_opti_constraints):
        """Save computational burden results to file."""
        if os.path.exists(f"{self.paths['record_folder']}/computational_res.npz"):
            computational_res = np.load(f"{self.paths['record_folder']}/computational_res.npz")
            opti_timer = np.append(
                computational_res['opti_timer'], time_to_solve_opti)
            n_constraints = np.append(
                computational_res['n_constraints'], number_opti_constraints)
            np.savez_compressed(f"{self.paths['record_folder']}/computational_res.npz",
                opti_timer=opti_timer, n_constraints=n_constraints)
        else:
            np.savez_compressed(f"{self.paths['record_folder']}/computational_res.npz",
                opti_timer=time_to_solve_opti, n_constraints=number_opti_constraints)



