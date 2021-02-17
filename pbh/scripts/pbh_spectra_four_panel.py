"""
Script for generating plots of the gamma-ray spectra from PBHs including
FSR from electrons, muon and charged-pions and decay spectra from muons,
neutral-pions and charged-pions.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, LogLocator
import numpy as np
import os

from utils import (
    compute_charged_pion_spectrum,
    compute_electron_spectrum,
    compute_muon_spectrum,
    compute_neutral_pion_spectrum,
    spectrum_geometic_approximation,
    temperature_to_mass,
    BlackHawk,
    MASS_CONVERSION,
    FIGURES_DIR,
)


GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0


if __name__ == "__main__":
    temperatures = [20e-3, 3e-3, 0.3e-3, 0.06e-3]
    pbh_masses = [temperature_to_mass(T) for T in temperatures]
    pbh_masses_str = ["{:e}".format(m / MASS_CONVERSION) for m in pbh_masses]
    dec = [round(float(m.split("e")[0]), 1) for m in pbh_masses_str]
    exponents = [m.split("e+")[1] for m in pbh_masses_str]

    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row")
    fig.dpi = 150

    blackhawk = BlackHawk(pbh_masses[0] / MASS_CONVERSION)
    axes_idxs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    titles = [
        r"$T_{{H}} = 20 \ \mathrm{{MeV}}, M = {dec}\times 10^{{{exp}}}\mathrm{{g}}$".format(
            dec=dec[0], exp=exponents[0],
        ),
        r"$T_{{H}} = 3 \ \mathrm{{MeV}}, M = {dec}\times 10^{{{exp}}}\mathrm{{g}}$".format(
            dec=dec[1], exp=exponents[1],
        ),
        r"$T_{{H}} = 0.3 \ \mathrm{{MeV}}, M = {dec}\times 10^{{{exp}}}\mathrm{{g}}$".format(
            dec=dec[2], exp=exponents[2],
        ),
        r"$T_{{H}} = 0.06 \ \mathrm{{MeV}}, M = {dec}\times 10^{{{exp}}}\mathrm{{g}}$".format(
            dec=dec[3], exp=exponents[3],
        ),
    ]

    for i in range(4):
        mpbh_gev = pbh_masses[i]

        blackhawk.mpbh = mpbh_gev / MASS_CONVERSION
        print("{:e}".format(blackhawk.mpbh))
        blackhawk.run()

        primary_energies = blackhawk.primary["energies"]
        primary_photon = blackhawk.primary["photon"]
        primary_electron = blackhawk.primary["electron"]
        primary_muon = blackhawk.primary["muon"]

        geom_photon = 2 * spectrum_geometic_approximation(
            primary_energies, mpbh_gev, 2
        )
        geom_electron = 4 * spectrum_geometic_approximation(
            primary_energies, mpbh_gev, 1
        )

        dnde_electron = compute_electron_spectrum(
            primary_energies, primary_energies, primary_electron
        )
        dnde_electron_geom = compute_electron_spectrum(
            primary_energies, primary_energies, geom_electron
        )
        dnde_muon = compute_muon_spectrum(
            primary_energies, primary_energies, primary_muon
        )
        dnde_neutral_pion = compute_neutral_pion_spectrum(
            primary_energies, mpbh_gev
        )
        dnde_charged_pion = compute_charged_pion_spectrum(
            primary_energies, mpbh_gev
        )
        dnde_tot = (
            dnde_electron
            + dnde_muon
            + dnde_charged_pion
            + dnde_neutral_pion
            + primary_photon
        )

        idxs = axes_idxs[i]
        ax = axes[idxs[0]][idxs[1]]

        ax.plot(
            primary_energies * 1e3,
            primary_photon * 1e-3,
            ls="--",
            lw=1,
            c="firebrick",
            label=r"$\mathrm{BH}_{\mathrm{prim}}$",
        )
        ax.plot(
            blackhawk.secondary["energies"] * 1e3,
            blackhawk.secondary["photon"] * 1e-3,
            ls="--",
            lw=2,
            c="firebrick",
            label=r"$\mathrm{BH}_{\mathrm{sec}}$",
        )
        ax.plot(
            primary_energies * 1e3,
            dnde_tot * 1e-3,
            ls="-",
            lw=2,
            c="steelblue",
            label=r"$\mathrm{This} \ \mathrm{study},\\ \mathrm{all} \ \mathrm{contributions}$",
        )
        ax.plot(
            primary_energies * 1e3,
            dnde_electron * 1e-3,
            ls="-.",
            label=r"$e^{\pm}$",
            c="goldenrod",
            lw=1,
        )
        ax.plot(
            primary_energies * 1e3,
            dnde_muon * 1e-3,
            ls="-.",
            label=r"$\mu^{\pm}$",
            c="mediumorchid",
            lw=1,
        )
        ax.plot(
            primary_energies * 1e3,
            dnde_neutral_pion * 1e-3,
            ls=":",
            label=r"$\pi^{0}$",
            c="Peru",
            lw=1,
        )
        ax.plot(
            primary_energies * 1e3,
            dnde_charged_pion * 1e-3,
            ls=":",
            label=r"$\pi^{\pm}$",
            c="teal",
            lw=1,
        )
        if i == 2 or i == 3:
            ax.plot(
                primary_energies * 1e3,
                (dnde_electron_geom + geom_photon) * 1e-3,
                ls="--",
                c="k",
                lw=1,
            )

        X_LIMS = np.array([1e-6, blackhawk.max_primary_eng])

        ax.set_xlim(X_LIMS * 1e3)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(titles[i])
        ax.set_xticks(np.geomspace(1e-6, 1, 7) * 1e3)

        if i == 0 or i == 1:
            Y_LIMS = np.array([1e16, 1e21])
            ax.set_ylim(Y_LIMS)
            # ax.set_yticks(np.geomspace(Y_LIMS[0], Y_LIMS[1], 6))
        else:
            Y_LIMS = np.array([1e15, 1e20])
            ax.set_ylim(Y_LIMS)
            # ax.set_yticks(np.geomspace(Y_LIMS[0], Y_LIMS[1], 6))

        if i == 0 or i == 2:
            ax.set_ylabel(
                r"$\frac{dN_{\gamma}}{dE_{\gamma}dt} \ (\mathrm{MeV}\mathrm{s}^{-1})$",
                fontsize=16,
            )
        if i == 2 or i == 3:
            ax.set_xlabel(r"$E_{\gamma} \ (\mathrm{MeV})$", fontsize=16)

        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.yaxis.set_minor_locator(
            LogLocator(
                base=10.0,
                subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                numticks=20,
            )
        )
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)

    #        ax.tick_params(
    #            bottom=True,
    #            top=True,
    #            left=True,
    #            right=True,
    #            labeltop=False,
    #            labelright=False,
    #            labelleft=True,
    #            labelbottom=True,
    #        )

    ax3 = axes[1][0]
    ax4 = axes[1][1]

    lab = r"\begin{tabular}{cc}"
    lab += r"\quad & BlackHawk\\ \hline "
    lab += r"\ \ & Primary\\"
    lab += r"\ \ & Secondary\\ \hline"
    lab += r"\quad  & This Study\\ \hline "
    lab += r"\ \ & All\\"
    lab += r"\ \ & $e^{\pm}$\\"
    lab += r"\ \ & $\mu^{\pm}$\\"
    lab += r"\ \ & $\pi^{0}$\\"
    lab += r"\ \ & $\pi^{\pm}$\\"
    lab += r"\hline"
    lab += r"\end{tabular}"

    xs = np.array([2e-3, 1e-2]) * 1e3
    ys = np.array([1e21, 3e20, 4e19, 1.3e19, 4e18, 1.5e18, 5e17, 3e19]) * 1e-2

    ax4.text(xs[0], ys[-1], lab, fontsize=9)

    ax4.plot(xs, [ys[0]] * 2, ls="--", lw=1, c="firebrick")
    ax4.plot(xs, [ys[1]] * 2, ls="--", lw=2, c="firebrick")
    ax4.plot(xs, [ys[2]] * 2, ls="-", lw=2, c="steelblue")
    ax4.plot(xs, [ys[3]] * 2, ls="-.", lw=1, c="goldenrod")
    ax4.plot(xs, [ys[4]] * 2, ls="-.", lw=1, c="mediumorchid")
    ax4.plot(xs, [ys[5]] * 2, ls=":", lw=1, c="Peru")
    ax4.plot(xs, [ys[6]] * 2, ls=":", lw=1, c="teal")

    ax4.text(1.5e-3, 4e19, "Geometric Optics Approx.", fontsize=8)
    ax4.annotate(
        " ",
        xytext=(2e-3, 4e19),
        xy=(2e-3, 1e18),
        fontsize=9,
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    ax3.text(1e-1, 2e19, "Geometric Optics Approx.", fontsize=8)
    ax3.annotate(
        " ",
        xytext=(1e1, 1.5e19),
        xy=(1.5e0, 3e18),
        fontsize=9,
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )

    # ax4.text(1.5e-3, 2e19, "Geometric", fontsize=9)

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(FIGURES_DIR, "PBH_spectra_four_panel.pdf"))

