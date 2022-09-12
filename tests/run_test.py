import numpy as np
import pytest
from unittest import mock

from config.generate_colors import generate_colors, _check_color_diffs


def random_True_False(colors, color, min_diffs=None):
    return np.random.random() > 0.5

def test_generate_colors(mocker):
    colours_only = True
    all_type_eval = [
        'env_r_c', 'env_r_d', 'env_A_c', 'env_A_d', 'env_d_c', 'env_d_d',
        'opt_r_c', 'opt_r_d', 'opt_A_c', 'opt_A_d', 'opt_d_c', 'opt_d_d',
        'opt_n_c', 'opt_n_d', 'facmac'
    ]
    save, prm = None, None

    patched_check_color_diffs = mocker.patch("config.generate_colors._check_color_diffs", side_effect=random_True_False)

    colors = generate_colors(save, prm, all_type_eval, colours_only)

    assert len(colors) == len(all_type_eval), \
        f"generate colours lengths do not match {len(colors)} != {len(all_type_eval)}"