from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from constants import V_MW, X_KD
from hazma.cmb import vx_cmb
from hazma.parameters import sv_inv_MeV_to_cm3_per_s
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rich.progress import Progress

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
ALPHA_EXISTING = 0.3

BLUE = "#1f77b4"
DARK_BLUE = "#2f3194"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
BROWN = "#8c564b"
PINK = "#e377c2"
GREY = "#7f7f7f"
YELLOW_GREEN = "#bcbd22"
TEAL = "#17becf"
BLACK = "k"
COLOR_DICT = {
    r"$\gamma$-ray telescopes": DARK_BLUE,
    "Thermal relic": "k",
    "Pheno": "y",
    "COMPTEL": BLUE,
    "EGRET": ORANGE,
    "Fermi": GREEN,
    "INTEGRAL": RED,
    "pheno": ORANGE,
    "GECCO (GC 1', NFW)": BROWN,
    "GECCO (GC 1', Einasto)": PURPLE,
    "GECCO (M31 1')": PINK,
    "GECCO (Draco 1')": GREY,
    r"GECCO (GC 5$^\circ$, NFW)": BROWN,
    r"GECCO (GC 5$^\circ$, Einasto)": PURPLE,
    r"GECCO (Draco $5^\circ$)": PINK,
    r"GECCO (M31 $5^\circ$)": GREY,
    "CMB": BLACK,
    r"CMB ($s$-wave)": BLACK,
    r"CMB ($p$-wave)": BLACK,
    "gc_nfw_5_deg": BROWN,
    "gc_ein_5_deg_optimistic": PURPLE,
    "m31_nfw_5_deg": PINK,
    "draco_nfw_5_deg": GREY,
}


def nested_dict_to_string(d):
    def helper(v, k=None, prefix=""):
        if isinstance(v, dict):
            # Don't indent the first level
            additional_spacing = "  " if k is not None else ""

            ret_str = f"{prefix}{k}\n"
            for key, value in v.items():
                ret_str += helper(value, key, prefix + additional_spacing)
            return ret_str
        else:
            return f"{prefix}{k}: {v}\n"

    return helper(d)


def make_legend_ax(
    ax,
    gecco_targets,
    existing_measurements,
    cmb_kind="both",
    thermal_relic=False,
    fontsize=12,
):
    ax.axis("off")
    handles = []

    for label in gecco_targets:
        handles += [
            Line2D(
                [0],
                [0],
                color=COLOR_DICT[label],
                label=label,
            )
        ]

    for label in existing_measurements:
        handles += [Patch(color=COLOR_DICT[label], label=label, alpha=ALPHA_EXISTING)]

    if cmb_kind in ["s-wave", "both"]:
        label = r"CMB ($s$-wave)"
        handles += [
            Line2D([0], [0], linestyle="-.", color=COLOR_DICT[label], label=label)
        ]

    if cmb_kind in ["p-wave", "both"]:
        label = r"CMB ($p$-wave)"
        handles += [
            Line2D([0], [0], linestyle="--", color=COLOR_DICT[label], label=label)
        ]

    if thermal_relic:
        label = "Thermal relic"
        handles += [
            Line2D([0], [0], linestyle=":", color=COLOR_DICT[label], label=label)
        ]

    ax.legend(handles=handles, loc="center", fontsize=fontsize)
    return ax


def configure_ticks(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    return ax


def plot_lims(ax, mxs, lims, kind: str, cmb_kind: str = "both"):
    if "gecco" in lims:
        for label, ls in lims["gecco"].items():
            ax.loglog(
                mxs,
                ls if kind == "ann" else 1 / ls,
                label=label,
                color=COLOR_DICT[label],
            )

    if "existing" in lims:
        for label, ls in lims["existing"].items():
            ax.fill_between(
                mxs,
                ls if kind == "ann" else 1 / ls,
                1e100 if kind == "ann" else -1e100,
                alpha=ALPHA_EXISTING,
                label=label,
                color=COLOR_DICT[label],
            )

    if "CMB" in lims:
        if kind == "ann":
            if cmb_kind in ["p-wave", "both"]:
                v_ratios = np.array([(V_MW / vx_cmb(mx, X_KD)) ** 2 for mx in mxs])
                ax.plot(
                    mxs,
                    v_ratios * lims["CMB"],
                    ls="--",
                    lw=1,
                    c="k",
                    label=r"CMB ($p$-wave)",
                )
            if cmb_kind in ["s-wave", "both"]:
                ax.plot(mxs, lims["CMB"], ls="-.", c="k", lw=1, label=r"CMB ($s$-wave)")
        else:
            if "CMB" in lims:
                ax.plot(mxs, 1 / lims["CMB"], ls="--", c="k", lw=1, label="CMB")

    return ax


def curve_outline(lims_existing, decay=False):
    outline = np.stack(list(lims_existing.values())).min(0)
    if decay:
        return 1 / outline
    return outline


def get_progress_update(progress: Optional[Progress], desc: str, total: int):
    if progress is None:
        return lambda: None
    task = progress.add_task(desc, total=total)
    return lambda: progress.update(task, advance=1, refresh=True)


def sigmav(model, vx=V_MW):
    """Compute (sigma v) for the given model."""
    cme = 2 * model.mx * (1.0 + 0.5 * vx**2)
    sig = model.annihilation_cross_sections(cme)["total"]
    return sig * vx * sv_inv_MeV_to_cm3_per_s
