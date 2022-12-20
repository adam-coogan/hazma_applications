import os
import pathlib
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
from constants import V_MW
from constrain import get_cmb_limit, get_gamma_ray_limits, get_relic_density_limit
from hazma.parameters import electron_mass as m_e
from hazma.vector_mediator import KineticMixing
from rich.progress import Progress
from scipy.interpolate import interp1d
from utils import (
    ALPHA_EXISTING,
    configure_ticks,
    curve_outline,
    get_progress_update,
    make_legend_ax,
    nested_dict_to_string,
    plot_lims,
    sigmav,
)

SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
MX_RANGE = (1.12 * m_e, 250.0)
MV_OVER_MX = 3.0
TITLE = r"$m_V = %g m_\chi$" % MV_OVER_MX
YLIM = (1e-32, 1e-25)
FONTSIZE = 10


@click.group()
def cli():
    ...


def update_model(model, mx):
    model.mx = mx
    model.mv = MV_OVER_MX * mx


def get_pheno_limit(
    mxs, model, alpha_d=0.5, vx=V_MW, outline=True, progress: Optional[Progress] = None
) -> dict[str, np.ndarray]:
    data = {
        "babar": np.genfromtxt(
            os.path.join(SCRIPT_PATH, "data", "babar.csv"), delimiter=","
        ),
        "lsnd": np.genfromtxt(
            os.path.join(SCRIPT_PATH, "data", "lsnd.csv"), delimiter=","
        ),
        "e137": np.genfromtxt(
            os.path.join(SCRIPT_PATH, "data", "e137.csv"), delimiter=","
        ),
    }
    interps = {
        ex: interp1d(datum.T[0], datum.T[1], fill_value=np.inf, bounds_error=False)
        for ex, datum in data.items()
    }
    progress_update = get_progress_update(progress, "Pheno", len(data))

    gvxx = np.sqrt(4 * np.pi * alpha_d)
    sigma_vs = np.zeros((len(mxs), len(data)))
    for j, interp in enumerate(interps.values()):
        for i, mx in enumerate(mxs):
            update_model(model, mx)
            y = interp(mx * 1e-3)
            if y < np.inf:
                eps = np.sqrt((model.mx / model.mv) ** 4 * y / alpha_d)
                model.gvxx = gvxx
                model.eps = eps
                sigma_vs[i, j] = sigmav(model, vx)
            else:
                sigma_vs[i, j] = np.inf
            progress_update()

    if outline:
        return {"Pheno": sigma_vs.min(1)}
    else:
        return {ex: sigma_vs[:, j] for j, ex in enumerate(data)}


@cli.command()
@click.option("--n-mxs", default=3)
def relic(n_mxs):
    with Progress() as progress:
        model = KineticMixing(mx=1.0, mv=1.0, gvxx=1.0, eps=1.0)

        lims = {}
        lims["mv_over_mx"] = MV_OVER_MX
        mxs = lims["mxs"] = np.geomspace(*MX_RANGE, num=n_mxs)
        # NOTE: stheta must be set to 1 here!
        lims["relic_density"] = get_relic_density_limit(
            model, "eps", mxs, update_model=update_model, progress=progress
        )

    print(nested_dict_to_string(lims))


@cli.command()
@click.option("--n-mxs", default=100)
@click.option("--save-path", default=None)
@click.option("--print-output/--no-print-output", default=False)
def constrain(n_mxs, save_path, print_output):
    with Progress() as progress:
        model = KineticMixing(mx=1.0, mv=1.0, gvxx=1.0, eps=1.0)

        lims = {}
        lims["mv_over_mx"] = MV_OVER_MX
        mxs = lims["mxs"] = np.geomspace(*MX_RANGE, num=n_mxs)

        lims["relic_density"] = get_relic_density_limit(
            model,
            "eps",
            mxs,
            param_range=(1e-10, 1.0),
            update_model=update_model,
            progress=progress,
        )
        lims.update(get_pheno_limit(mxs, model, progress=progress))

        lims["gecco"], lims["existing"] = get_gamma_ray_limits(
            model, mxs, update_model, progress
        )
        lims["CMB"] = get_cmb_limit(
            model, mxs, update_model=update_model, progress=progress
        )

    if print_output:
        print(nested_dict_to_string(lims))

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "outputs", f"km.npz")
    np.savez(save_path, **lims)
    print(f"saved limits to {save_path}")


# TODO: fix
@cli.command()
@click.option("--lim-path", default=None)
@click.option("--save-path", default=None)
def plot(lim_path, save_path):
    # Load limits
    if lim_path is None:
        lim_path = os.path.join(SCRIPT_PATH, "outputs", "km.npz")
        print(
            f"loading limits for annihilation to mediator from default file, {lim_path}"
        )

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

    # Replace gamma-ray constraints with their envelope
    lims["existing"] = {r"$\gamma$-ray telescopes": curve_outline(lims["existing"])}

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "figures", "km.pdf")

    fig, axs = plt.subplots(
        1, 2, figsize=(7, 3.5), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    ax = axs[0]
    mxs = lims["mxs"]
    plot_lims(ax, mxs, lims, "ann", "s-wave")
    ax.plot(mxs, lims["relic_density"], ":k")
    ax.fill_between(
        mxs,
        lims["Pheno"],
        curve_outline(lims["existing"]),
        color="y",
        alpha=ALPHA_EXISTING,
    )

    configure_ticks(ax)
    ax.grid(linewidth=0.3)
    ax.set_xlabel(r"$m_\chi$ [MeV]")
    ax.set_xlim(mxs.min(), mxs.max())
    ax.set_ylim(*YLIM)
    ax.set_title(TITLE)
    ax.set_ylabel(r"$\langle \sigma v \rangle_{\bar{\chi}\chi,0}$ [cm$^3$/s]")

    # Can use any final state to get GECCO and existing target names
    # HACK to get pheno patch in legend
    ax = axs[1]
    make_legend_ax(
        axs[1],
        lims["gecco"].keys(),
        list(lims["existing"].keys()),  # + ["Pheno"],
        "p-wave",
        True,
        FONTSIZE,
    )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"saved figure to {save_path}")


if __name__ == "__main__":
    cli()
