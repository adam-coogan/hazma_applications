import os
import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
from constrain import get_gamma_ray_limits
from hazma.parameters import s_to_inv_MeV
from hazma.rh_neutrino import RHNeutrino
from rich.progress import Progress
from utils import (
    configure_ticks,
    curve_outline,
    make_legend_ax,
    nested_dict_to_string,
    plot_lims,
)

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
MX_RANGES = {"e": (0.1, 100.0), "mu": (0.1, 100.0)}
TITLES = {"e": r"$\ell = e$", "mu": r"$\ell = \mu$"}
YLIMS = {"e": (1e21, 1e29), "mu": (1e21, 1e29)}
FONTSIZE = 10


@click.group()
def cli():
    ...


# @cli.command()
# @click.option("--n-mxs", default=3)
# def relic(n_mxs):
#     with Progress() as progress:
#         model = KineticMixing(mx=1.0, mv=1.0, gvxx=1.0, eps=1.0)
#
#         lims = {}
#         lims["mv_over_mx"] = MV_OVER_MX
#         mxs = lims["mxs"] = np.geomspace(*MX_RANGE, num=n_mxs)
#         # NOTE: stheta must be set to 1 here!
#         lims["relic_density"] = get_relic_density_limit(
#             model, "eps", mxs, update_model=update_model, progress=progress
#         )
#
#     print(nested_dict_to_string(lims))


@cli.command()
@click.option("--n-mxs", default=100)
@click.option("--save-path", default=None)
@click.option("--print-output/--no-print-output", default=False)
def constrain(n_mxs, save_path, print_output):
    lims = {}
    for lepton, mx_range in MX_RANGES.items():
        print(f"decay to {lepton}")
        with Progress() as progress:
            model = RHNeutrino(1.0, theta=1.0, flavor=lepton)

            lims[lepton] = {}
            mxs = lims[lepton]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
            lims[lepton]["gecco"], lims[lepton]["existing"] = get_gamma_ray_limits(
                model, mxs, progress=progress
            )

    if print_output:
        print(nested_dict_to_string(lims))

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "outputs", f"rh_neutrino.npz")
    np.savez(save_path, **lims)
    print(f"saved limits to {save_path}")


def add_theta_contour(ax, mxs, theta, lepton):
    rhn = RHNeutrino(1, theta, lepton)
    taus = np.zeros_like(mxs)
    for i, m in enumerate(mxs):
        rhn.mx = m
        taus[i] = 1 / s_to_inv_MeV * 1 / rhn.decay_widths()["total"]
    ax.plot(mxs, taus, ls="-.", color="k", alpha=0.5, lw=1)



def add_theta_label(ax, x, y, power):
    ax.text(
        x,
        y,
        r"$\theta=10^{" + str(power) + r"}$",
        fontsize=7,
        rotation=-55,
        color="k"
    )


@cli.command()
@click.option("--lim-path", default=None)
@click.option("--save-path", default=None)
def plot(lim_path, save_path):
    # Load limits
    if lim_path is None:
        lim_path = os.path.join(SCRIPT_PATH, "outputs", "rh_neutrino.npz")
        print(
            f"loading limits for annihilation to mediator from default file, {lim_path}"
        )

    try:
        lims = {k: v.item() for k, v in np.load(lim_path, allow_pickle=True).items()}
    except FileNotFoundError:
        print(f"the constraint file '{lim_path}' has not yet been generated")
        return

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "figures", "rh_neutrino.pdf")

    fig, axs = plt.subplots(
        1, 3, figsize=(10, 3), gridspec_kw={"width_ratios": [1.5, 1.5, 1]}
    )

    text_xs = [42, 7, 1.1, 0.18, 0.13]
    text_ys = [1e27, 1e27, 1e27, 1e27, 5e22]
    powers = [-16, -14, -12, -10, -8]
    for (lepton, ls), ax in zip(lims.items(), axs):
        mxs = ls["mxs"]

        # Envelope of existing constraints
        ls["existing"] = {r"$\gamma$-ray telescopes": curve_outline(ls["existing"])}
        # Plot limits
        plot_lims(ax, mxs, ls, "dec")
        # Coupling contours
        for x, y, power in zip(text_xs, text_ys, powers):
            add_theta_contour(ax, mxs, 10**power, lepton)
            add_theta_label(ax, x, y, power)

        configure_ticks(ax)
        ax.grid(linewidth=0.3)
        ax.set_xlabel(r"$m_N$ [MeV]")
        ax.set_xlim(mxs.min(), mxs.max())
        ax.set_ylim(*YLIMS[lepton])
        ax.set_title(TITLES[lepton])

    ax = axs[0]
    ax.set_ylabel(r"$\tau$ [s]")

    # Can use any final state to get GECCO and existing target names
    # HACK to get pheno patch in legend
    ax = axs[2]
    make_legend_ax(
        axs[2],
        lims["e"]["gecco"].keys(),
        lims["e"]["existing"].keys(),
        fontsize=FONTSIZE,
    )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"saved figure to {save_path}")


if __name__ == "__main__":
    cli()
