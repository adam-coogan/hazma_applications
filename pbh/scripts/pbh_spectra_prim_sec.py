"""
Script for generating plots of the gamma-ray spectra from PBHs
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os

GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0

# path to the directory containing data from collected blackhawk results.
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)

# path to the directory containing data from collected blackhawk results.
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "figures"
)

if __name__ == "__main__":
    exponents = [15.0, 16.0, 17.0, 18.0]
    columns = [str(10 ** exponent) for exponent in exponents]
    colors = ["steelblue", "firebrick", "goldenrod", "mediumorchid"]

    # Load the data
    df_primary = pd.read_csv(
        os.path.join(RESULTS_DIR, "dnde_photon_primary.csv")
    )
    df_secondary = pd.read_csv(
        os.path.join(RESULTS_DIR, "dnde_photon_secondary.csv")
    )

    plt.figure(dpi=150, figsize=(7, 7 / GOLD_RATIO))

    for (column, color) in zip(columns, colors):
        energies = df_secondary["energies"]
        spec = df_secondary[column]
        plt.plot(energies, energies ** 2 * spec, lw=2, c=color)

        energies = df_primary["energies"]
        spec = df_primary[column]
        plt.plot(energies, energies ** 2 * spec, lw=1, ls="--", c=color)

    lines = [
        Line2D([], [], ls="-", lw=2, c=colors[0]),
        Line2D([], [], ls="-", lw=2, c=colors[1]),
        Line2D([], [], ls="-", lw=2, c=colors[2]),
        Line2D([], [], ls="-", lw=2, c=colors[3]),
        Line2D([], [], ls="-", lw=2, c="k"),
        Line2D([], [], ls="--", lw=2, c="k"),
    ]
    labels = [
        r"$M_{\mathrm{PBH}}=10^{15} \ \mathrm{g}$",
        r"$M_{\mathrm{PBH}}=10^{16} \ \mathrm{g}$",
        r"$M_{\mathrm{PBH}}=10^{17} \ \mathrm{g}$",
        r"$M_{\mathrm{PBH}}=10^{18} \ \mathrm{g}$",
        r"$\mathrm{Primary}$",
        r"$\mathrm{Secondary}$",
    ]
    plt.legend(lines, labels)

    plt.xlim([1e-6, 1])
    plt.ylim([1e7, 1e20])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(
        r"$E_{\gamma}^2 \frac{dN_{\gamma}}{dE_{\gamma}dt} \ (\mathrm{GeV}\mathrm{s}^{-1})$",
        fontsize=16,
    )
    plt.xlabel(r"$E_{\gamma} \ (\mathrm{GeV})$", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "PBH_spectra_prim_sec.pdf"))
