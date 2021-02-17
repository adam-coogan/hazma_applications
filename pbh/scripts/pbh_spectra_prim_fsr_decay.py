"""
Script for generating plots of the gamma-ray spectra from PBHs including
FSR from electrons, muon and charged-pions and decay spectra from muons,
neutral-pions and charged-pions.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

from utils import (
    compute_charged_pion_spectrum,
    compute_electron_spectrum,
    compute_muon_spectrum,
    compute_neutral_pion_spectrum,
    BlackHawk,
    mass_conversion,
    FIGURES_DIR,
)


GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0

# File to use


if __name__ == "__main__":
    mpbh_str = "1e15"
    mpbh = float(mpbh_str)
    mpbh_gev = mpbh * mass_conversion
    blackhawk = BlackHawk(mpbh)
    blackhawk.max_primary_eng = 20.0
    blackhawk.run()

    # car_es, car_spec = np.genfromtxt(
    #    os.path.join(RESULTS_DIR, "pbh_spec_egrb.csv"), delimiter=","
    # ).T

    plt.figure(dpi=150)

    primary_energies = blackhawk.primary["energies"]
    primary_photon = blackhawk.primary["photon"]
    primary_electron = blackhawk.primary["electron"]
    primary_muon = blackhawk.primary["muon"]

    dnde_electron = compute_electron_spectrum(
        primary_energies, primary_energies, primary_electron
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

    plt.plot(
        primary_energies,
        primary_photon,
        ls="--",
        lw=1,
        c="steelblue",
        label=r"$\mathrm{BH}_{\mathrm{prim}}$",
    )
    plt.plot(
        blackhawk.secondary["energies"],
        blackhawk.secondary["photon"],
        ls="--",
        lw=2,
        c="steelblue",
        label=r"$\mathrm{BH}_{\mathrm{sec}}$",
    )
    plt.plot(
        primary_energies,
        dnde_tot,
        ls="-",
        lw=2,
        c="steelblue",
        label=r"$\mathrm{BH}_{\mathrm{prim}}+\mathrm{Decay}+\mathrm{FSR}$",
    )
    plt.plot(
        primary_energies,
        dnde_muon,
        ls="-.",
        label=r"$\mu^{\pm}$",
        c="goldenrod",
    )
    plt.plot(
        primary_energies,
        dnde_neutral_pion,
        ls=":",
        label=r"$\pi^{0}$",
        c="mediumorchid",
    )
    plt.plot(
        primary_energies,
        dnde_charged_pion,
        ls=":",
        label=r"$\pi^{\pm}$",
        c="teal",
    )
    # plt.plot(
    #    car_es, car_spec, ls=":", lw=2, c="firebrick", label=r"Car et. al.",
    # )

    plt.xlim([1e-3, blackhawk.max_primary_eng])
    plt.ylim([1e19, 1e26])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(
        r"$\frac{dN_{\gamma}}{dE_{\gamma}dt} \ (\mathrm{GeV}\mathrm{s}^{-1})$",
        fontsize=16,
    )
    plt.xlabel(r"$E_{\gamma} \ (\mathrm{GeV})$", fontsize=16)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(
        os.path.join(
            FIGURES_DIR, "PBH_spectra_prim_fsr_decay_" + mpbh_str + "g.pdf"
        )
    )

