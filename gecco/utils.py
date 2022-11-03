from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rich.progress import Progress

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
ALPHA_EXISTING = 0.3

BLUE = "#1f77b4"
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
    "COMPTEL": BLUE,
    "EGRET": ORANGE,
    "Fermi": GREEN,
    "INTEGRAL": RED,
    "pheno": ORANGE,
    "GECCO (GC 1', NFW)": BROWN,
    "GECCO (GC 1', Einasto)": PURPLE,
    "GECCO (M31 1')": PINK,
    "GECCO (Draco 1')": GREY,
    "CMB": BLACK,
    r"CMB ($s$-wave)": BLACK,
    r"CMB ($p$-wave)": BLACK,
    "gc_nfw_5_deg": BROWN,
    "gc_ein_5_deg_optimistic": PURPLE,
    "m31_nfw_5_deg": PINK,
    "draco_nfw_5_deg": GREY,
}


def get_progress_update(progress: Optional[Progress], desc: str, total: int):
    if progress is None:
        return lambda: None
    task = progress.add_task(desc, total=total)
    return lambda: progress.update(task, advance=1, refresh=True)


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
            return (f"{prefix}{k}: {v}\n")

    return helper(d)


def make_legend_ax(ax, gecco_targets, existing_measurements):
    ax.axis("off")
    handles = []

    for label in gecco_targets.keys():
        handles += [
            Line2D(
                [0],
                [0],
                color=COLOR_DICT[label],
                label=label,
            )
        ]

    for label in existing_measurements.keys():
        handles += [Patch(color=COLOR_DICT[label], label=label, alpha=ALPHA_EXISTING)]

    label = r"CMB ($s$-wave)"
    handles += [Line2D([0], [0], linestyle="-.", color=COLOR_DICT[label], label=label)]
    label = r"CMB ($p$-wave)"
    handles += [Line2D([0], [0], linestyle="--", color=COLOR_DICT[label], label=label)]

    ax.legend(handles=handles, loc="center", fontsize=12)
    return ax


def configure_ticks(ax):
    ax.tick_params(axis="both", which="both", direction="in")
    return ax
