import os
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
from constrain import get_cmb_limit, get_gamma_ray_limits
from hazma.single_channel import SingleChannelAnn, SingleChannelDec
from rich.progress import Progress
from utils import configure_ticks, make_legend_ax, nested_dict_to_string, plot_lims

MX_RANGES = {"e e": (1.5, 1e4), "g g": (1e-1, 1e2), "mu mu": (220, 1e4)}
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
FS_TO_TITLE = {
    "e e": r"$e^+ e^-$",
    "g g": r"$\gamma \gamma$",
    "mu mu": r"$\mu^+ \mu^-$",
}
FS_TO_YLIM = {
    "ann": {"e e": (1e-33, 1e-23), "g g": (1e-37, 1e-27), "mu mu": (1e-30, 1e-22)},
    "dec": {"e e": (1e23, 1e27), "g g": (1e25, 1e31), "mu mu": (1e23, 1e26)},
}


@click.group()
def cli():
    ...


@cli.command()
@click.option("--ann", "kind", flag_value="ann")
@click.option("--dec", "kind", flag_value="dec")
@click.option("--n-mxs", default=100)
@click.option("--save-path", default=None)
@click.option("--print-output/--no-print-output", default=False)
def constrain(kind, n_mxs, save_path, print_output):
    lims = {}
    with Progress() as progress:
        for fs, mx_range in MX_RANGES.items():
            if kind == "ann":
                model = SingleChannelAnn(1.0, fs, 1.0)
            else:
                model = SingleChannelDec(1.0, fs, 1.0)

            lims[fs] = {}
            lims[fs]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
            lims[fs]["gecco"], lims[fs]["existing"] = get_gamma_ray_limits(
                model, lims[fs]["mxs"], progress=progress
            )
            if kind == "ann":
                lims[fs]["CMB"] = get_cmb_limit(
                    model, lims[fs]["mxs"], progress=progress
                )
            else:
                # Load limits
                if fs == "e e":
                    lim_cmb = np.genfromtxt(
                        os.path.join(SCRIPT_PATH, "data", "cmb_epem.csv"), delimiter=","
                    )
                elif fs == "g g":
                    lim_cmb = np.genfromtxt(
                        os.path.join(SCRIPT_PATH, "data", "cmb_gamma_gamma.csv"),
                        delimiter=",",
                    )
                else:
                    continue

                mxs_cmb = lim_cmb[:, 0] * 1e3  # GeV -> MeV
                lim_cmb = lim_cmb[:, 1]
                lims[fs]["CMB"] = np.interp(lims[fs]["mxs"], mxs_cmb, lim_cmb)

    if print_output:
        print(nested_dict_to_string(lims))

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "outputs", f"single-channel-{kind}.npz")
    np.savez(save_path, **lims)
    print(f"saved limits to {save_path}")


@cli.command()
@click.option("--ann", "kind", flag_value="ann")
@click.option("--dec", "kind", flag_value="dec")
@click.option("--lim-path", default=None)
@click.option("--save-path", default=None)
def plot(kind, lim_path, save_path):
    if lim_path is None:
        lim_path = os.path.join(SCRIPT_PATH, "outputs", f"single-channel-{kind}.npz")
        print(f"loading limits from default file, {lim_path}")

    try:
        lims = {k: v.item() for k, v in np.load(lim_path, allow_pickle=True).items()}
    except FileNotFoundError:
        print(f"the constraint file '{lim_path}' has not yet been generated")
        return

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "figures", f"single-channel-{kind}.pdf")

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))

    for ax, (fs, ls) in zip(axes.flatten(), lims.items()):
        mxs = ls["mxs"]
        plot_lims(ax, mxs, ls, kind)
        configure_ticks(ax)
        ax.grid(linewidth=0.3)
        ax.set_xlabel(r"$m_\chi$ [MeV]")

        if kind == "ann":
            ax.set_ylabel(r"$\langle \sigma v \rangle_{\bar{\chi}\chi,0}$ [cm$^3$/s]")
        else:
            # Plot CMB limits
            ax.set_ylabel(r"$\tau$ [s]")

            if fs == "e e":
                mxs_cmb, svs_cmb = np.loadtxt(
                    os.path.join(SCRIPT_PATH, "data", "single-channel-dec-ee-cmb.dat"),
                    unpack=True,
                )
                mxs_cmb *= 1e3  # GeV -> MeV
                ax.plot(mxs_cmb, svs_cmb, ls="--", lw=1, c="k", label="CMB")

        ax.set_xlim(mxs.min(), mxs.max())
        ax.set_ylim(*FS_TO_YLIM[kind][fs])
        ax.set_title(FS_TO_TITLE[fs])

    # Can use any final state to get GECCO and existing target names
    make_legend_ax(
        axes[1, 1], lims["e e"]["gecco"].keys(), lims["e e"]["existing"].keys(), "dec"
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"saved figure to {save_path}")


if __name__ == "__main__":
    cli()
