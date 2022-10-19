"""
The generate_colours function generates a list of colours for plotting results.

Different methods should use
colours that are clearly visible and easily differentiated
The colours generated need to be the same at every run for easy
comparison of results.
"""

import random

import matplotlib.pyplot as plt

from src.utilities.userdeftools import distr_learning


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


def _colours_to_prm(save, prm, colours0, colours, all_evaluation_methods):
    """
    Add initial list of colours 'colours0' + the generated list of colours.

    Colours corresponding to the types of evaluations 'all_evaluation_methods'
    to the prm dictionary.
    """
    if save is None:
        save = {}
        prm['save'] = save

    save['colours'] = colours0 + colours
    save['colourse'] = {}
    save['colourse']['baseline'] = 'k'  # the baseline is always black
    save['colourse']['random'] = 'b'

    save['colourse']['opt'] = save['colours'][0]

    # first allocate the colours to decentralised environment-based learning
    allocate_1 = [evaluation_method for evaluation_method in all_evaluation_methods
                  if evaluation_method not in ['opt', 'baseline']
                  and distr_learning(evaluation_method) == 'd']
    for i, evaluation_method in enumerate(allocate_1):
        save['colourse'][evaluation_method] = save['colours'][i + 1]

    # then allocate to the other environment-based learning
    allocate_2 = [evaluation_method for evaluation_method in all_evaluation_methods
                  if not (evaluation_method in ['opt', 'baseline']
                          or distr_learning(evaluation_method) == 'd')]
    for i, evaluation_method in enumerate(allocate_2):
        save['colourse'][evaluation_method] = save['colours'][i + 1]

    return save, prm


def list_all_evaluation_methods(entries):
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
    n_added = 0
    colours = []

    # first, loop through candidate colours and colour palettes
    colours0 = ['red', 'darkorange', 'grey', 'forestgreen',
               'deepskyblue', 'mediumblue', 'darkviolet', 'lawngreen']
    palettes = ['Set1', 'Set2', 'Set3', 'viridis', 'plasma', 'inferno',
                'magma', 'cividis', 'Pastel1', 'Pastel2', 'Paired',
                'Accent', 'Dark2', 'tab10', 'tab20', 'tab20b', 'tab20c']

    for palette in palettes:
        # the colour map we select additional colours from
        colour_map = plt.get_cmap(palette)
        colours_palette = [colour_map(j)
                          for j in range(len(colour_map.__dict__['colors']))]
        n_add = min(n_colours - n_added - len(colours0), len(colours_palette))
        added_i = 0
        for colour in colours_palette:
            # loop through colours in the palette
            if added_i < n_add:
                # for as long as we still need colours,
                # check the proposed colours
                # are different enough
                enough_diff = _check_colour_diffs(
                    colours, colour, min_diffs=[0.21, 0.6, 0.5, 1]
                )
                if len(colours) == 0 or enough_diff:
                    # if different enough, add to the list of colours
                    colours += [colour]
                    added_i += 1
        n_added += added_i

    iteration = 0
    random.seed(0)
    while n_added < n_colours - len(colours0) and iteration < 1000:
        # if we still need more colours, just try random colours
        colour = (random.random(), random.random(), random.random(), 1)
        enough_diff = _check_colour_diffs(
            colours, colour, min_diffs=[0, 0, 0, 3]
        )
        if len(colours) == 0 or enough_diff:
            colours += [colour]
            n_added += 1
        iteration += 1

    if colours_only:
        return colours0 + colours

    # save the list to the parameters
    save, prm = _colours_to_prm(save, prm, colours0, colours, all_evaluation_methods)
    return save, prm
