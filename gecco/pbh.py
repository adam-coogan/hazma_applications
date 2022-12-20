import os
import pathlib

import click
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from constrain import get_gamma_ray_limits
from hazma.parameters import MeV_to_g, g_to_MeV
from hazma.pbh import PBH
from rich.progress import Progress
from utils import configure_ticks, make_legend_ax, nested_dict_to_string, plot_lims, curve_outline

YLIM = (1e-7, 1.0)
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()


@click.group()
def cli():
    ...


@cli.command()
@click.option("--save-path", default=None)
@click.option("--print-output/--no-print-output", default=False)
def constrain(save_path, print_output):
    lims = {}
    model = PBH(1e15 * g_to_MeV)
    mxs = model._mxs
    with Progress() as progress:
        lims["mxs"] = mxs
        lims["gecco"], lims["existing"] = get_gamma_ray_limits(
            model, mxs, progress=progress
        )

    if print_output:
        print(nested_dict_to_string(lims))

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "outputs", "pbh.npz")
    np.savez(save_path, **lims)
    print(f"saved limits to {save_path}")


@cli.command()
@click.option("--lim-path", default=None)
@click.option("--save-path", default=None)
def plot(lim_path, save_path):
    if lim_path is None:
        lim_path = os.path.join(SCRIPT_PATH, "outputs", "pbh.npz")
        print(f"loading limits from default file, {lim_path}")

    try:
        lims = {}
        for k, v in np.load(lim_path, allow_pickle=True).items():
            if v.shape == ():
                lims[k] = v.item()
            else:
                # v is a numpy array
                lims[k] = v
    except FileNotFoundError:
        print(f"the constraint file '{lim_path}' has not yet been generated")
        return

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "figures", "pbh.pdf")

    fig, axs = plt.subplots(
        1, 2, figsize=(7, 3.5), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    lims["existing"] = {r"$\gamma$-ray telescopes": curve_outline(lims["existing"])}

    ax = axs[0]
    mxs = lims["mxs"] * MeV_to_g
    plot_lims(ax, mxs, lims, "ann")
    configure_ticks(ax)
    ax.grid(linewidth=0.3)
    ax.set_xlabel(r"$m_\mathrm{PBH}$ [g]")
    ax.set_ylabel(r"$f_\mathrm{PBH}$")
    ax.set_xlim(1e15, 1e18)
    ax.set_ylim(YLIM)

    # Can use any final state to get GECCO and existing target names
    # make_legend_ax(ax, lims["gecco"].keys(), lims["existing"].keys())
    ax = axs[1]
    make_legend_ax(ax, lims["gecco"].keys(), lims["existing"].keys(), "")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"saved figure to {save_path}")


if __name__ == "__main__":
    cli()
