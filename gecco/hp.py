import os
import pathlib
import warnings
from functools import partial
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from constrain import V_MW, get_cmb_limit, get_gamma_ray_limits, get_progress_update
from hazma.parameters import omega_h2_cdm
from hazma.relic_density import relic_density
from hazma.scalar_mediator import HiggsPortal
from rich.progress import Progress
from scipy.optimize import root_scalar
from utils import (
    ALPHA_EXISTING,
    configure_ticks,
    curve_outline,
    make_legend_ax,
    nested_dict_to_string,
    plot_lims,
    sigmav,
)

MX_RANGES = {"med": (1e-1, 1e3), "sm": (1e-1, 250)}
MS_OVER_MX = {"med": 0.5, "sm": 1.5}
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
TITLES = {
    "med": r"$m_S = %g m_\chi$" % MS_OVER_MX["med"],
    "sm": r"$m_S = %g m_\chi$" % MS_OVER_MX["sm"],
}
YLIMS = {"med": (1e-37, 1e-23), "sm": (1e-37, 1e-23)}
FONTSIZE = 10


@click.group()
def cli():
    ...


def _update_model(model, mx, ann_to):
    model.mx = mx
    model.ms = MS_OVER_MX[ann_to] * mx


def get_pheno_limit_helper(hp, sv, vx=V_MW):
    """
    Checks whether *some* value of stheta is consistent with pheno constraints
    at a given (mx, <sigma v>) point. If the returned value is negative, it is
    inconsistent.
    """
    if hp.ms <= hp.mx:
        raise ValueError("ms must be larger than mx so the DM annihilates into SM")
    gsxx_max = 4 * np.pi
    # <sigma v> with couplings set to 1
    hp.gsxx = 1
    hp.stheta = 1
    sv_1 = sigmav(hp, vx)

    # Find smallest stheta compatible with <sigma v> for the given
    # gsxx_max
    stheta_min = np.sqrt(sv / sv_1) / gsxx_max
    if stheta_min > 0.999:
        return -1e100

    stheta_grid = np.geomspace(stheta_min, 0.999, 50)
    constr_mins = np.full_like(stheta_grid, np.inf)
    gsxxs = np.zeros_like(stheta_grid)
    for i, stheta in enumerate(stheta_grid):
        hp.stheta = stheta
        hp.gsxx = np.sqrt(sv / sv_1) / hp.stheta
        gsxxs[i] = hp.gsxx
        # Figure out strongest constraint
        constr_mins[i] = np.min([fn() for fn in hp.constraints().values()])

    # Check if (mx, ms, sv) point is allowed for some (gsxx, stheta) combination
    return constr_mins.max()


def get_pheno_limit(
    hp,
    mxs,
    ms_mx_ratio,
    sv_range: Tuple[float, float] = YLIMS["sm"],
    n_svs: int = 95,
    progress: Optional[Progress] = None,
    vx=V_MW,
):
    progress_update = get_progress_update(progress, "Pheno", len(mxs))
    svs = np.geomspace(*sv_range, num=n_svs)
    lims = np.zeros(len(mxs))
    for i, mx in enumerate(mxs):
        hp.mx = mx
        hp.ms = ms_mx_ratio * mx
        vals = np.vectorize(partial(get_pheno_limit_helper, hp, vx=vx))(svs)
        lims[i] = svs[np.argmin(vals > 0)]
        progress_update()
    return lims


def get_relic_density_limit_helper(model, gsxx_range=(1e-5, 4 * np.pi), vx=V_MW):
    lb = np.log10(gsxx_range[0])
    ub = np.log10(gsxx_range[1])

    def f(log10_gsxx):
        model.gsxx = 10**log10_gsxx
        return relic_density(model, semi_analytic=True) - omega_h2_cdm

    try:
        root = root_scalar(f, bracket=[lb, ub], method="brentq")
        if not root.converged:
            warnings.warn(f"root_scalar did not converge. Flag: {root.flag}")
        model.gsxx = 10**root.root
        return sigmav(model, vx)
    except ValueError as e:
        warnings.warn(f"Error encountered: {e}. Returning nan", RuntimeWarning)
        return np.nan


def get_relic_density_limit(
    model,
    params,
    param_range: Tuple[float, float] = (1e-5, 4 * np.pi),
    vx=V_MW,
    update_model=lambda model, mx: setattr(model, "mx", mx),
    progress: Optional[Progress] = None,
):
    progress_update = get_progress_update(progress, "Relic density", len(params))
    lims = np.zeros(len(params))
    for i, param in enumerate(params):
        update_model(model, param)
        lims[i] = get_relic_density_limit_helper(model, param_range, vx)
        progress_update()
    return lims


@cli.command()
@click.option("--n-mxs", default=3)
def relic(n_mxs):
    lims = {}
    with Progress() as progress:
        for ann_to, mx_range in MX_RANGES.items():
            model = HiggsPortal(1.0, 1.0, 1.0, 1.0)
            update_model = partial(_update_model, ann_to=ann_to)

            lims[ann_to] = {}
            lims[ann_to]["ms_over_mx"] = MS_OVER_MX[ann_to]
            mxs = lims[ann_to]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
            # NOTE: stheta must be set to 1 here!
            lims[ann_to]["relic_density"] = get_relic_density_limit(
                model, mxs, update_model=update_model, progress=progress
            )

    print(nested_dict_to_string(lims))


# @cli.command()
# @click.option("--n-mxs", default=3)
# @click.option("--n-svs", default=2)
# def pheno(n_mxs, n_svs):
#     lims = {}
#     with Progress() as progress:
#         for ann_to, mx_range in MX_RANGES.items():
#             if ann_to == "sm":
#                 model = HiggsPortal(1.0, 1.0, 1.0, 1.0)
#                 lims[ann_to] = {}
#                 lims[ann_to]["ms_over_mx"] = MS_OVER_MX[ann_to]
#                 mxs = lims[ann_to]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
#                 lims[ann_to]["pheno"] = get_pheno_limit(
#                     model, mxs, MS_OVER_MX[ann_to], n_svs=n_svs, progress=progress
#                 )
#
#     print(nested_dict_to_string(lims))


@cli.command()
@click.option("--n-mxs", default=100)
@click.option("--save-path", default=None)
@click.option("--print-output/--no-print-output", default=False)
@click.option("--n-svs", default=95)
def constrain(n_mxs, save_path, print_output, n_svs):
    lims = {}
    with Progress() as progress:
        for ann_to, mx_range in MX_RANGES.items():
            model = HiggsPortal(1.0, 1.0, 1.0, 1.0)
            update_model = partial(_update_model, ann_to=ann_to)

            lims[ann_to] = {}
            lims[ann_to]["ms_over_mx"] = MS_OVER_MX[ann_to]
            mxs = lims[ann_to]["mxs"] = np.geomspace(*mx_range, num=n_mxs)
            lims[ann_to]["gecco"], lims[ann_to]["existing"] = get_gamma_ray_limits(
                model, mxs, update_model, progress
            )
            lims[ann_to]["CMB"] = get_cmb_limit(
                model, mxs, update_model=update_model, progress=progress
            )
            lims[ann_to]["relic_density"] = get_relic_density_limit(
                model, mxs, update_model=update_model, progress=progress
            )

            if ann_to == "sm":
                lims[ann_to]["pheno"] = get_pheno_limit(
                    model, mxs, MS_OVER_MX[ann_to], n_svs=n_svs, progress=progress
                )

    if print_output:
        print(nested_dict_to_string(lims))

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "outputs", f"hp.npz")
    np.savez(save_path, **lims)
    print(f"saved limits to {save_path}")


def add_gsxx_contour(ax, mxs, ms_mx_ratio, gsxx, stheta, vx=V_MW):
    hp = HiggsPortal(mx=1, ms=1, gsxx=gsxx, stheta=stheta)
    svs = np.zeros_like(mxs)
    for i, m in enumerate(mxs):
        hp.mx = m
        hp.ms = m * ms_mx_ratio
        svs[i] = sigmav(hp, vx)
    ax.plot(mxs, svs, ls=":", color="r", lw=1)


def add_gsxx_label(ax, x, y, gsxx, stheta=None, rotation=0):
    if stheta is not None:
        label = r"$(g_{S\chi},s_{\theta}) = $(" + gsxx + "," + stheta + ")"
        ax.text(x, y, label, fontsize=8, rotation=rotation, color="r")
    else:
        label = r"$g_{S\chi}$ = " + gsxx
        ax.text(x, y, label, fontsize=8, rotation=rotation, color="r")


@cli.command()
@click.option("--lim-path", default=None)
@click.option("--save-path", default=None)
def plot(lim_path, save_path):
    # Load limits
    if lim_path is None:
        lim_path = os.path.join(SCRIPT_PATH, "outputs", "hp.npz")
        print(
            f"loading limits for annihilation to mediator from default file, {lim_path}"
        )

    try:
        lims = {k: v.item() for k, v in np.load(lim_path, allow_pickle=True).items()}
    except FileNotFoundError:
        print(f"the constraint file '{lim_path}' has not yet been generated")
        return

    if save_path is None:
        save_path = os.path.join(SCRIPT_PATH, "figures", "hp.pdf")

    fig, axs = plt.subplots(
        1, 3, figsize=(10, 3), gridspec_kw={"width_ratios": [1.5, 1.5, 1]}
    )

    for (ann_to, ls), ax in zip(lims.items(), axs):
        mxs = ls["mxs"]
        plot_lims(ax, mxs, ls, "ann", "p-wave")
        ax.plot(mxs, ls["relic_density"], ":k")
        if ann_to == "sm":
            ax.fill_between(
                mxs,
                ls["pheno"],
                curve_outline(ls["existing"]),
                color="y",
                alpha=ALPHA_EXISTING,
            )

        configure_ticks(ax)
        ax.grid(linewidth=0.3)
        ax.set_xlabel(r"$m_\chi$ [MeV]")
        ax.set_xlim(mxs.min(), mxs.max())
        ax.set_ylim(*YLIMS[ann_to])
        ax.set_title(TITLES[ann_to])

    ax = axs[0]
    add_gsxx_contour(ax, lims["med"]["mxs"], MS_OVER_MX["med"], 1e-2, 1)
    add_gsxx_label(ax, 100, 2e-32, r"$10^{-2}$", rotation=-25)
    add_gsxx_contour(ax, lims["med"]["mxs"], MS_OVER_MX["med"], 1e-4, 1)
    add_gsxx_label(ax, 2, 5e-37, r"$10^{-4}$", rotation=-25)
    ax.set_ylabel(r"$\langle \sigma v \rangle_{\bar{\chi}\chi,0}$ [cm$^3$/s]")

    ax = axs[1]
    add_gsxx_contour(ax, lims["med"]["mxs"], MS_OVER_MX["sm"], 4 * np.pi, 1)
    add_gsxx_label(ax, 9, 2e-33, r"$4\pi$", r"$1$")

    # Can use any final state to get GECCO and existing target names
    # HACK to get pheno patch in legend
    ax = axs[2]
    make_legend_ax(
        axs[2],
        lims["sm"]["gecco"].keys(),
        list(lims["sm"]["existing"].keys()) + ["Pheno"],
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
