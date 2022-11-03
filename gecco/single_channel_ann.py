import os
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
from constrain import V_MW, X_KD, get_cmb_limit, get_gamma_ray_limits
from hazma.cmb import vx_cmb
from hazma.single_channel import SingleChannelAnn
from rich.progress import Progress
from utils import (
    ALPHA_EXISTING,
    COLOR_DICT,
    configure_ticks,
    make_legend_ax,
    nested_dict_to_string,
)

MX_RANGES = {"e e": (0.51, 1e4), "g g": (1e-1, 1e2), "mu mu": (106, 1e4)}
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
DEFAULT_LIM_PATH = os.path.join(SCRIPT_PATH, "outputs", "single-channel-ann.npz")
DEFAULT_FIG_PATH = os.path.join(SCRIPT_PATH, "figures", "single-channel-ann.pdf")
FS_TO_TITLE = {
    "e e": r"$e^+ e^-$",
    "g g": r"$\gamma \gamma$",
    "mu mu": r"$\mu^+ \mu^-$",
}
FS_TO_YLIM = {"e e": (1e-33, 1e-23), "g g": (1e-37, 1e-27), "mu mu": (1e-30, 1e-22)}


@click.group()
def cli():
    ...


@cli.command()
@click.option("--n-mxs", default=100)
@click.option("--save-path", default=DEFAULT_LIM_PATH)
@click.option("--print-output/--no-print-output", default=False)
def constrain(n_mxs, save_path, print_output):
    lims = {}
    with Progress() as progress:
        for fs, mx_range in MX_RANGES.items():
            lims[fs] = {}
            model = SingleChannelAnn(1.0, fs, 1.0)
            lims[fs]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
            lims[fs]["CMB"] = get_cmb_limit(model, lims[fs]["mxs"], progress=progress)
            lims[fs]["gecco"], lims[fs]["existing"] = get_gamma_ray_limits(
                model, lims[fs]["mxs"], progress=progress
            )

    if print_output:
        print(nested_dict_to_string(lims))

    np.savez(save_path, **lims)


def plot_lims(ax, mxs, lims):
    for label, ls in lims["gecco"].items():
        ax.loglog(mxs, ls, label=label, color=COLOR_DICT[label])

    for label, ls in lims["existing"].items():
        ax.fill_between(
            mxs,
            ls,
            1e100,
            alpha=ALPHA_EXISTING,
            label=label,
            color=COLOR_DICT[label],
        )

    v_ratios = np.array([(V_MW / vx_cmb(mx, X_KD)) ** 2 for mx in mxs])
    ax.plot(mxs, v_ratios * lims["CMB"], ls="--", lw=1, c="k", label=r"CMB ($p$-wave)")
    ax.plot(mxs, lims["CMB"], ls="-.", c="k", lw=1, label=r"CMB ($s$-wave)")

    return ax


@cli.command()
@click.option("--lim-path", default=DEFAULT_LIM_PATH)
@click.option("--save-path", default=DEFAULT_FIG_PATH)
def plot(lim_path, save_path):
    try:
        lims = {k: v.item() for k, v in np.load(lim_path, allow_pickle=True).items()}
    except FileNotFoundError:
        print(f"the constraint file '{lim_path}' has not yet been generated")
        return

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    for ax, (fs, ls) in zip(axes.flatten(), lims.items()):
        mxs = ls["mxs"]
        plot_lims(ax, mxs, ls)
        configure_ticks(ax)
        ax.grid(linewidth=0.3)
        ax.set_xlabel(r"$m_\chi$ [MeV]")
        ax.set_ylabel(r"$\langle \sigma v \rangle_{\bar{\chi}\chi,0}$ [cm$^3$/s]")
        ax.set_xlim(mxs.min(), mxs.max())
        ax.set_ylim(*FS_TO_YLIM[fs])
        ax.set_title(FS_TO_TITLE[fs])

    # Can use any final state to get GECCO and existing target names
    make_legend_ax(axes[1, 1], lims["e e"]["gecco"], lims["e e"]["existing"])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    cli()
