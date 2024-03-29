"""
The generate_colours function generates a list of colours for plotting results.

Different methods should use
colours that are clearly visible and easily differentiated
The colours generated need to be the same at every run for easy
comparison of results.
"""

import random

import matplotlib.pyplot as plt
from matplotlib import colors

from src.environment.utilities.userdeftools import distr_learning


def _check_colour_diffs(colours, new_colour, min_diffs=None):
    """
    Check colours are different from each other, black, white, and extremes.

    inputs:
    colours:
        the list of current colours
    new_colour:
        the 3 values describing the new candidate colour
    min_cols:
        how different the colours need to be different from each other
    min_white:
        how different the colours need to be different from white
    min_black:
        how different the colours need to be different from black
    max_1_vals:
        how many of the 3 rgb values can be equal to 1

    output:
    enough_diff:
        whether the new colour passes the test of being different enough
    """
    min_cols, min_white, min_black, max_1_vals = min_diffs
    diff_colours = 1 if len(colours) == 0 else min(
        [sum((new_colour[k] - colour[k]) ** 2 for k in range(3))
         for colour in colours])
    diff_white = sum((new_colour[k] - 1) ** 2 for k in range(3))
    diff_black = sum((new_colour[k]) ** 2 for k in range(3))
    n_1 = sum(1 for k in range(3) if new_colour[k] == 1)
    enough_diff = all([diff_colours > min_cols, diff_white > min_white,
                       diff_black > min_black, n_1 < max_1_vals])

    return enough_diff


def _colours_to_prm(save, prm, colours, colours0, all_evaluation_methods):
    """
    Add initial list of colours 'colours0' + the generated list of colours.

    Colours corresponding to the types of evaluations 'all_evaluation_methods'
    to the prm dictionary.
    """
    if save is None:
        save = {}
        prm['save'] = save

    save['colours'] = colours + colours0

    allocate_1 = [
        evaluation_method for evaluation_method in all_evaluation_methods
        if evaluation_method not in ['opt', 'baseline']
        and distr_learning(evaluation_method) == 'd'
        and evaluation_method not in save['colourse'].keys()
    ]
    print(f"allocate_1 {allocate_1}")
    for i, evaluation_method in enumerate(allocate_1):
        save['colourse'][evaluation_method] = save['colours'][i + 1]

    # then allocate to the other environment-based learning
    allocate_2 = [
        evaluation_method for evaluation_method in all_evaluation_methods
        if not (
            evaluation_method in ['opt', 'baseline']
            or distr_learning(evaluation_method) == 'd'
        )
        and evaluation_method not in save['colourse'].keys()
    ]
    for i, evaluation_method in enumerate(allocate_2):
        save['colourse'][evaluation_method] = save['colours'][i + 1]

    return save, prm


def list_all_evaluation_methods(entries):
    """List all possible evaluation methods - not just in this run."""
    if entries is None:
        all_evaluation_methods = []
        reward_structure_combs = \
            ['r_c', 'r_d', 'A_Cc', 'A_c', 'A_d', 'd_Cc', 'd_c', 'd_d']
        for experience_source in ['env_', 'opt_']:
            all_evaluation_methods += [
                experience_source + rs_comb
                for rs_comb in reward_structure_combs
            ]
        all_evaluation_methods += ['opt', 'opt_n_c', 'opt_n_d']
    else:
        all_evaluation_methods = entries

    return all_evaluation_methods


def add_colours_from_candidate_palettes(n_colours, colours0, candidate_colours):
    """Add colours from candidate palettes to the list if they are different enough."""
    palettes = [
        'Set1', 'Set2', 'Set3', 'viridis', 'plasma', 'inferno',
        'magma', 'cividis', 'Pastel1', 'Pastel2', 'Paired',
        'Accent', 'Dark2', 'tab10', 'tab20', 'tab20b', 'tab20c'
    ]
    colours = []
    for palette in palettes:
        # the colour map we select additional colours from
        colour_map = plt.get_cmap(palette)
        colours_palette = [
            colour_map(j) for j in range(len(colour_map.__dict__['colors']))
        ]
        candidate_colours += colours_palette

    n_add = n_colours - len(colours)
    n_added = 0
    it = 0
    while n_added < n_add and it < 1000:
        colour = candidate_colours[it]
        # loop through colours in the palette
        # for as long as we still need colours,
        # check the proposed colours
        # are different enough
        enough_diff = _check_colour_diffs(
            colours0 + colours, colour, min_diffs=[0.21, 0.6, 0.5, 1]
        )
        if len(colours) == 0 or enough_diff:
            # if different enough, add to the list of colours
            colours += [colour]
            n_added += 1

        it += 1

    return colours, n_added


def add_random_colours(colours, colours0, n_colours):
    """Add random colours to the list if they are different enough."""
    iteration = 0
    random.seed(0)
    while len(colours + colours0) < n_colours and iteration < 1000:
        # if we still need more colours, just try random colours
        colour = (random.random(), random.random(), random.random(), 1)
        enough_diff = _check_colour_diffs(
            colours + colours0, colour, min_diffs=[0, 0, 0, 3]
        )
        if len(colours) == 0 or enough_diff:
            colours += [colour]
        iteration += 1

    return colours


def generate_colours(save, prm, colours_only=False, entries=None):
    """
    Generate a list of colours for plotting the results.

    Different methods using colours need to be clearly visible
    and easily differentiated.

    The colours generated need to be the same at every run
    for easy comparison of results.

    Inputs
    save:
        dictionary of parameters relative to saving results
        (may be None at this point)
    prm:
        dictionary of all parameters for current run.
        Later on prm['save'] = save
    all_types_eval:
        the methods evaluated in this run.
        Need an allocated colour for plotting

    Outputs
    save:
        the updated/intialised saving parameters, containing the colours
    prm:
        the updated dictionary of parameters
    """
    # list all possible for consistent colours ordering
    all_evaluation_methods = list_all_evaluation_methods(entries)
    n_colours = len(all_evaluation_methods)
    for i, initial_evaluation_method in enumerate(all_evaluation_methods):
        if initial_evaluation_method[0: 3] == "env" \
                and initial_evaluation_method not in prm["RL"]["evaluation_methods"]:
            method_replacement = [
                evaluation_method_start_opt
                for evaluation_method_start_opt in prm["RL"]["evaluation_methods"]
                if initial_evaluation_method in evaluation_method_start_opt
            ]
            if len(method_replacement) > 0:
                all_evaluation_methods[i] = method_replacement[0]

    # first, user defined colours
    colourse = {}
    colourse['baseline'] = (0, 0, 0)  # the baseline is always black

    grey = colors.to_rgba('#999999')
    blue = colors.to_rgba('#377eb8')  # colorblind blue
    red = (192 / 255, 0, 0)
    for c in ['d', 'c']:
        for explo in ['env', 'opt']:
            colourse[f'{explo}_r_{c}'] = red
            colourse[f'{explo}_d_{c}'] = blue
            colourse[f'{explo}_A_{c}'] = colors.to_rgba('forestgreen')

        colourse[f'opt_n_{c}'] = colors.to_rgba('darkorange')
    colourse['opt'] = grey

    # then, loop through candidate colours and colour palettes
    colours0 = list(colourse.values())
    candidate_colours = [
        colors.to_rgba(colour_name) for colour_name in
        ['darkviolet', 'deepskyblue', 'mediumblue', 'lawngreen']
    ]

    colours, n_added = add_colours_from_candidate_palettes(n_colours, colours0, candidate_colours)
    colours = add_random_colours(colours, colours0, n_colours)

    if colours_only:
        return colours0 + colours

    save['colourse'] = colourse

    # save the list to the parameters
    save, prm = _colours_to_prm(save, prm, colours, colours0, all_evaluation_methods)

    return save, prm
