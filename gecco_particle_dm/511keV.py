#!/usr/bin/python

"""
Script for generating a plot of GECCO's capabilities of detecting a 511 keV
line.
"""

import matplotlib.pyplot as plt
import numpy as np

GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0

if __name__ == "__main__":
    gecco_bc = 7.4e-8
    gecco_wc = 3.2e-7

    phi_511 = 3e-4
    N_MSP = (2.7e3, 0.9e3)

    ds = np.logspace(-2, 2, 100)
    num_bc = phi_511 / gecco_bc * (8.5 / ds) ** 2
    num_wc = phi_511 / gecco_wc * (8.5 / ds) ** 2

    FIG_WIDTH = 7
    plt.figure(dpi=150, figsize=(FIG_WIDTH, FIG_WIDTH / GOLD_RATIO))

    Y_MIN = 1e2
    Y_MAX = 1e8

    plt.fill_between(ds, num_bc, num_wc, alpha=0.5, color="steelblue")
    plt.fill_between(ds, num_wc, alpha=0.5, color="firebrick")
    plt.text(0.5, 6e4, "GECCO (best-case)", rotation=-24, fontsize=12)
    plt.text(0.5, 1e4, "GECCO (conservative)", rotation=-24, fontsize=12)

    plt.plot(ds, [N_MSP[0] for _ in ds], ls="-", lw=2, c="k")
    plt.fill_between(
        ds,
        [N_MSP[0] + N_MSP[1] for _ in ds],
        [N_MSP[0] - N_MSP[1] for _ in ds],
        # color="mediumorchid",
        color="teal",
        alpha=0.5,
    )

    plt.text(1, 1e3, "MSP", fontsize=12)

    # Wolf-Rayet
    D_WR = 0.350
    plt.vlines(D_WR, Y_MIN, Y_MAX, colors="k", linestyles="--")
    plt.text(D_WR * 0.9, 4.5e3, "Wolf-Rayet", rotation=90, fontsize=9, c="k")
    # LMXB 4U 1700+24
    D_LMXB = 0.42
    plt.vlines(D_LMXB, Y_MIN, Y_MAX, colors="k", linestyles="--")
    plt.text(
        D_LMXB * 0.9, 4.5e3, "LMXB 4U 1700+24", rotation=90, fontsize=9, c="k"
    )
    # MSP J0427-4715
    D_MSP = 0.16
    plt.vlines(D_MSP, Y_MIN, Y_MAX, colors="k", linestyles="--")
    plt.text(
        D_MSP * 0.9, 4.5e3, "MSP J0427-4715", rotation=90, fontsize=9, c="k"
    )

    plt.xlim([0.1, 20])
    plt.ylim(Y_MIN, Y_MAX)

    plt.yscale("log")
    plt.xscale("log")

    plt.ylabel(r"$N_{\mathrm{src}}$", fontsize=16)
    plt.xlabel(r"$d_{\mathrm{src}} \ (\mathrm{kpc})$", fontsize=16)

    plt.tight_layout()
    plt.savefig("/home/logan/Projects/GECCO/figures/511.pdf")
