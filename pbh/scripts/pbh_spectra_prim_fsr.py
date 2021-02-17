"""
Script for generating plots of the gamma-ray spectra from PBHs
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import (
    RESULTS_DIR,
    FIGURES_DIR,
    compute_electron_spectrum,
)


def generate_data(df_photon, df_electron):
    """
    Generate data for for black-hole spectrum + electron FSR given data
    containing primary photon and electron spectra from the evaporation of
    black-holes with various masses. The primary photon and electron must
    have the same energies and same black-hole masses.

    Parameters
    ----------
    df_photon: pandas.DataFrame
        DataFrame containing the data for the primary photon spectra.
    df_electron: pandas.DataFrame
        DataFrame containing the data for the primary electron spectra.

    Returns
    -------
    df_new: pandas.DataFrame
        New pandas DataFrame containing the primary photon spectrum + FSR
        spectrum off electron.
    """
    df_new = df_photon.copy()

    energies = np.array(df_photon["energies"])
    primary_photon_spectra = df_photon.drop(["energies"], axis=1)
    primary_electron_spectra = df_electron.drop(["energies"], axis=1)

    columns = primary_photon_spectra.columns

    for i in tqdm(range(len(columns))):
        col = columns[i]
        dnde_e = np.array(primary_electron_spectra[col])
        dnde_g = np.array(primary_photon_spectra[col])
        dnde_fsr = compute_electron_spectrum(energies, energies, dnde_e)
        dnde_tot = dnde_g + dnde_fsr
        df_new[columns[i]] = dnde_tot

    return df_new


def plot_data(df):
    """
    Plot all of the spectra.
    """
    plt.figure(dpi=100)

    energies = np.array(df["energies"])
    df_spectra = df.drop(["energies"], axis=1)
    print(df_spectra.columns)
    for col in df_spectra.columns:
        spec = np.array(df_spectra[col])
        plt.plot(energies, energies ** 2 * spec)

    plt.xlim([1e-6, 1])
    plt.ylim([1e1, 1e20])
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    df_photon = pd.read_csv(
        os.path.join(RESULTS_DIR, "dnde_photon_primary.csv")
    )
    df_electron = pd.read_csv(
        os.path.join(RESULTS_DIR, "dnde_electron_primary.csv")
    )

    df_tot = generate_data(df_photon, df_electron)
    df_tot.to_csv(
        os.path.join(RESULTS_DIR, "dnde_photon_primary_with_fsr.csv")
    )
    plot_data(df_tot)

