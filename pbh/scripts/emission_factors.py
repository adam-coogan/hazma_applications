"""
Script for computing the emission factors of a black hole into various
particles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import (
    BlackHawk,
    MASS_CONVERSION,
    temperature_to_mass,
    mass_to_temperature,
    FIGURES_DIR,
    RESULTS_DIR,
)


def key_to_label(key):
    if key == "photon":
        return r"$\gamma$"
    elif key == "electron":
        return r"$e^{\pm}$"
    elif key == "muon":
        return r"$\mu^{\pm}$"
    elif key == "tau":
        return r"$\tau^{\pm}$"
    elif key == "neutrino":
        return r"$\nu$"
    elif key == "gluon":
        return r"$g$"
    elif key == "graviton":
        return r"$h_{\mu\nu}$"
    elif key == "up":
        return r"$u$"
    elif key == "charm":
        return r"$c$"
    elif key == "top":
        return r"$t$"
    elif key == "down":
        return r"$d$"
    elif key == "strange":
        return r"$s$"
    elif key == "bottom":
        return r"$b$"
    elif key == "W":
        return r"$W^{\pm}$"
    elif key == "Z":
        return r"$Z^{0}$"
    elif key == "higgs":
        return r"$H$"


def full_scan():
    mpbhs_grams = np.geomspace(1e13, 1e18, 100)

    blackhawk = BlackHawk(mpbhs_grams[0])
    phis = {
        key: []
        for key, spec in blackhawk.primary.items()
        if not key == "energies"
    }

    for mpbh in mpbhs_grams:
        blackhawk = BlackHawk(mpbh)
        blackhawk.run()
        energies = blackhawk.primary["energies"]

        for key in phis.keys():
            spec = blackhawk.primary[key]
            phi = np.trapz(energies * spec, energies)
            phis[key].append(phi)

        # Normalize
        norm = sum(val[-1] for val in phis.values())
        for key in phis.keys():
            phis[key][-1] = phis[key][-1] / norm

    plt.figure(dpi=150)
    ylims = [1e-3, 1]

    for key in phis.keys():
        phi = phis[key]
        # Only add to plot if it will show up given ylims
        if np.max(phi) > ylims[0]:
            plt.plot(mpbhs_grams, phi, label=key_to_label(key))

    plt.xlim([np.min(mpbhs_grams), np.max(mpbhs_grams)])
    plt.ylim(ylims)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.xlabel(r"$M_{\mathrm{PBH}} \ (\mathrm{g})$", fontsize=16)
    plt.ylabel(r"$\phi$", fontsize=16)
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(FIGURES_DIR, "emission_factors.pdf"))


def partial_scan():
    mpbhs_grams = [1e15, 1e16, 1e17, 1e18]
    outfiles = [
        "primary_1e15g.csv",
        "primary_1e16g.csv",
        "primary_1e17g.csv",
        "primary_1e18g.csv",
    ]

    for mpbh, file in zip(mpbhs_grams, outfiles):
        blackhawk = BlackHawk(mpbh)
        blackhawk.run()

        df = pd.DataFrame(blackhawk.primary)
        df.to_csv(os.path.join(RESULTS_DIR, file), index=False)


if __name__ == "__main__":
    partial_scan()
    full_scan()
