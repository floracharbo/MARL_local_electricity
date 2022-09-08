"""
The generate_colors function generates a list of colors for plotting results.

Different methods should use
colours that are clearly visible and easily differentiated
The colours generated need to be the same at every run for easy
comparison of results.
"""

import random

import matplotlib.pyplot as plt


def _check_color_diffs(colors, new_color, min_diffs=None):
    """
    Check colors are different from each other, black, white, and extremes.

    inputs:
    colors:
        the list of current colours
    new_color:
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
    diff_colors = 1 if len(colors) == 0 else min(
        [sum((new_color[k] - color[k]) ** 2 for k in range(3))
         for color in colors])
    diff_white = sum((new_color[k] - 1) ** 2 for k in range(3))
    diff_black = sum((new_color[k]) ** 2 for k in range(3))
    n_1 = sum(1 for k in range(3) if new_color[k] == 1)
    enough_diff = all([diff_colors > min_cols, diff_white > min_white,
                       diff_black > min_black, n_1 < max_1_vals])

    return enough_diff


def _colors_to_prm(save, prm, colors0, colors, all_type_eval):
    """
    Add initial list of colors 'colors0' + the generated list of colors.

    Colours corresponding to the types of evaluations 'all_type_eval'
    to the prm dictionary.
    """
    if save is None:
        save = {}
        prm['save'] = save

    save['colors'] = colors0 + colors
    save['colorse'] = {}
    save['colorse']['baseline'] = 'k'  # the baseline is always black
    save['colorse']['random'] = 'b'

    save['colorse']['opt'] = save['colors'][0]

    # first allocate the colours to decentralised environment-based learning
    allocate_1 = [type_eval for type_eval in all_type_eval
                  if type_eval not in ['opt', 'baseline']
                  and type_eval.split('_')[2] == 'd']
    for i, type_eval in enumerate(allocate_1):
        save['colorse'][type_eval] = save['colors'][i + 1]

    # then allocate to the other environment-based learning
    allocate_2 = [type_eval for type_eval in all_type_eval
                  if not (type_eval in ['opt', 'baseline']
                          or type_eval.split('_')[2] == 'd')]
    for i, type_eval in enumerate(allocate_2):
        save['colorse'][type_eval] = save['colors'][i + 1]

    return save, prm


def generate_colors(save, prm, all_type_eval):
    """
    Generate a list of colors for plotting the results.

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
    n_colors = len(all_type_eval)
    n_added = 0
    colors = []

    # first, loop through candidate colours and colour palettes
    colors0 = ['red', 'darkorange', 'grey', 'forestgreen',
               'deepskyblue', 'mediumblue', 'darkviolet', 'lawngreen']
    palettes = ['Set1', 'Set2', 'Set3', 'viridis', 'plasma', 'inferno',
                'magma', 'cividis', 'Pastel1', 'Pastel2', 'Paired',
                'Accent', 'Dark2', 'tab10', 'tab20', 'tab20b', 'tab20c']

    for palette in palettes:
        # the colour map we select additional colours from
        color_map = plt.get_cmap(palette)
        colors_palette = [color_map(j)
                          for j in range(len(color_map.__dict__['colors']))]
        n_add = min(n_colors - n_added - len(colors0), len(colors_palette))
        added_i = 0
        for color in colors_palette:
            # loop through colours in the palette
            if added_i < n_add:
                # for as long as we still need colours,
                # check the proposed colours
                # are different enough
                enough_diff = _check_color_diffs(
                    colors, color, min_diffs=[0.21, 0.6, 0.5, 1]
                )
                if len(colors) == 0 or enough_diff:
                    # if different enough, add to the list of colours
                    colors += [color]
                    added_i += 1
        n_added += added_i

    iteration = 0
    random.seed(0)
    while n_added < n_colors - len(colors0) and iteration < 1000:
        # if we still need more colours, just try random colours
        color = (random.random(), random.random(), random.random(), 1)
        enough_diff = _check_color_diffs(
            colors, color, min_diffs=[0, 0, 0, 3]
        )
        if len(colors) == 0 or enough_diff:
            colors += [color]
            n_added += 1
        iteration += 1

    # save the list to the parameters
    save, prm = _colors_to_prm(save, prm, colors0, colors, all_type_eval)

    return save, prm
