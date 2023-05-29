import matplotlib.pyplot as plt
import numpy as np


def title_and_save(
        title, fig, prm, ax=None,
        display_title=True, title_display=None
):
    if title_display is None:
        if 'theta' in title:
            ind = title.index('theta')
            title_display = title[0:ind] + r'$\theta$' + title[ind + 5:]
        elif '_air' in title:
            ind = title.index('_air')
            title_display = title[0:ind] + r'$_air$' + title[ind + 4:]
        else:
            title_display = title
    if display_title:
        if ax is not None:
            ax.set_title(title_display)
        else:
            plt.title(title_display)
    if prm['save']['save_run']:
        fig_folder = prm['paths']['fig_folder']
        if prm['save']['high_res']:
            fig.savefig(fig_folder / f"{title.replace(' ', '_')}.pdf",
                        bbox_inches='tight', format='pdf', dpi=1200)
        else:
            fig.savefig(fig_folder / title.replace(' ', '_'),
                        bbox_inches='tight')
    plt.close('all')

    return fig


def formatting_figure(
        prm, fig=None, title=None, pos_leg='right', anchor_pos=None,
        ncol_leg=None, legend=True, display_title=True,
        title_display=None
):

    if anchor_pos is None:
        if pos_leg == 'right':
            anchor_pos = (1.25, 0.5)
        elif pos_leg == 'upper center':
            anchor_pos = (0.5, 1.2)

    if ncol_leg is None:
        if pos_leg == 'right':
            ncol_leg = 1
        elif pos_leg == 'upper center':
            ncol_leg = 3
    if legend:
        fig.legend(loc=pos_leg, bbox_to_anchor=anchor_pos,
                   ncol=ncol_leg, fancybox=True)
    fig.tight_layout()
    title_and_save(
        title, fig, prm,
        display_title=display_title, title_display=title_display
    )


def formatting_ticks(
        ax, fig, prm, xs=None, ys=None, title=None, im=None,
        grid=True, display_title=True
):
    if ys is not None:
        ax.set_xticks(np.arange(len(ys)))
        ax.set_xticklabels(ys)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    if xs is not None:
        ax.set_yticks(np.arange(len(xs)))
        ax.set_yticklabels(xs)
    if im is not None:
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if grid is True:
        plt.grid()
    title_and_save(
        title, fig, prm, ax=ax, display_title=display_title
    )
