"""
Various utilities for generating gamma-ray spectra from evaporating
black-holes using `BlackHawk`.
"""

import numpy as np
import os
import subprocess
from typing import List
from hazma.parameters import (
    neutral_pion_mass as _mpi0,
    charged_pion_mass as _mpi,
    electron_mass as _me,
    muon_mass as _mmu,
)
from hazma.decay import (
    neutral_pion as _decay_pi0,
    charged_pion as _decay_pi,
    muon as _decay_muon,
)

# Convert grams to GeV
MASS_CONVERSION = 5.60958884e23
# Convert seconds to GeV^-1
TIME_CONVERSION = 1.519267407e24
# Convert cm to GeV^-1
LENGTH_CONVERSION = 5.06773058e13
# Newton constant in GeV
G_NEWTON = (
    6.67408e-11
    * (LENGTH_CONVERSION * 100.0) ** 3
    / (MASS_CONVERSION * 1000.0)
    / TIME_CONVERSION ** 2
)
# Planck mass in GeV
M_PLANK = 1.221e19
# Fine structure constant
ALPHA_EM = 1.0 / 137.0

M_ELECTRON = _me * 1e-3
M_MUON = _mmu * 1e-3
M_NEUTRAL_PION = _mpi0 * 1e-3
M_CHARGED_PION = _mpi * 1e-3


# path to the directory containing data from collected blackhawk results.
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)

# path to the directory containing figures
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "figures"
)


def temperature_to_mass(temperature: float) -> float:
    """
    Convert the temperature of a black-hole to its mass.

    Parameters
    ----------
    temperature: float
        Temperature of the black-hole in GeV.

    Returns
    -------
    mass: float
        Mass of the black-hole in GeV.
    """
    return 1.0 / (8.0 * np.pi * G_NEWTON * temperature)


def mass_to_temperature(mass: float) -> float:
    """
    Convert the mass of a black-hole to its temperature.

    Parameters
    ----------
    mass: float
        Mass of the black-hole in GeV.

    Returns
    -------
    temperature: float
        Temperature of the black-hole in GeV.
    """
    return 1.0 / (8.0 * np.pi * G_NEWTON * mass)


def ap_spec(
    photon_energy: float, energy: float, mass: float, spin2: int
) -> float:
    """
    Compute the Altarelli-Parisi spectrum from a charged particle.

    Parameters
    ----------
    photon_energy: float
        Energy of the photon.
    energy: float
        Energy of the radiating particle.
    mass: float
        Mass of the radiating particle.
    spin2: int
        Twice the spin of the particle.

    Returns
    -------
    dnde: float
        Photon spectrum at `photon_energy`.
    """
    Q = 2.0 * energy
    x = 2 * photon_energy / Q
    mu = mass / Q
    if 0.0 < x < 1.0:
        if spin2 == 0:
            split = 2.0 * (1 - x) / x
        elif spin2 == 1:
            split = (1.0 + (1.0 - x) ** 2) / x
        else:
            raise ValueError("Invalid value for spin2. Use 0 or 1.")
        log = np.log((1.0 - x) / mu ** 2) - 1.0
        if log < 0.0:
            return 0.0
        return ALPHA_EM / np.pi * split * log / Q
    return 0.0


def get_greybody_factors(spin2: int, mpbh: float):
    """
    Get the greybody factors for a particle with a spin equal to `spin2/2`
    evaporating from a black-hole with mass mpbh (in GeV).

    Parameters
    ----------
    spin2: int
        Twice the spin of the particle.
    mpbh: float
        Black-hole mass is GeV.

    Returns
    -------
    energies: np.array
        Energies of the evaporating particle in GeV.
    greybodies: np.array
        Greybody factors corresponding to the particle energy in `energies`.
    """
    if spin2 == 0:
        fname = os.path.join(RESULTS_DIR, "greybody_spin_0.txt")
    elif spin2 == 1:
        fname = os.path.join(RESULTS_DIR, "greybody_spin_0.5.txt")
    elif spin2 == 2:
        fname = os.path.join(RESULTS_DIR, "greybody_spin_1.txt")
    elif spin2 == 4:
        fname = os.path.join(RESULTS_DIR, "greybody_spin_2.txt")
    else:
        raise ValueError("Only 2*spin = 0, 1, 2 and 4 are available.")

    with open(fname, "r") as f:
        xline = f.readline().split("   ")[5:]
        xline[-1] = xline[-1].split("\n")[0]
        xs = np.array([float(x) for x in xline])
        gbline = f.readline().split("   ")[2:]
        gbline[-1] = gbline[-1].split("\n")[0]
        gbs = np.array([float(gb) for gb in gbline])

    return xs / (2.0 * G_NEWTON * mpbh), gbs


def __dnde_neutral_pion(egam: float, epi: float) -> float:
    """
    Compute the spectrum from the decay of a neutral pion.
    """
    if epi < M_NEUTRAL_PION:
        return 0.0
    return _decay_pi0(egam * 1e3, epi * 1e3) * 1e3


def __dnde_charged_pion(egam: float, epi: float) -> float:
    """
    Compute the spectrum from the decay of a neutral pion.
    """
    if epi < M_CHARGED_PION:
        return 0.0
    return _decay_pi(egam * 1e3, epi * 1e3) * 1e3 + ap_spec(
        egam, epi, M_CHARGED_PION, 0
    )


def convolve_spectrum(
    photon_energies: List[float],
    particle_energies: List[float],
    particle_dist: List[float],
    particle_dnde: List[float],
) -> List[float]:
    """
    Compute the convolved gamma-ray spectrum of a particle evaporating from
    a black-hole.

    Parameters
    ----------
    photon_energies: array-like
        Energies of the photon where the spectrum should be computed.
    particle_energies: array-like, (n,)
        Energies of the evaporated particles.
    particle_dist: array-like, (n,)
        Distribution dN/dEdt of the evaporated particles corresponding to the
        energies `particle_energies`.
    particle_dnde: callable
        Function returning the photon spectrum from the particle. Signature
        should be `dnde(photon_energy, particle_energy)`.

    Returns
    -------
    convolved: array-like
        Convolved photon spectrum.
    """
    integrand = np.array(
        [
            [
                dnde_p * particle_dnde(egam, ep)
                for (ep, dnde_p) in zip(particle_energies, particle_dist)
            ]
            for egam in photon_energies
        ]
    )
    return np.trapz(integrand, particle_energies)


def compute_neutral_pion_spectrum(
    photon_energies: List[float], mpbh: float
) -> List[float]:
    """
    Compute the secondary gamma-ray spectrum produced from a black-hole
    evaporating into neutral pions.

    Parameters
    ----------
    photon_energies: array-like
        Energies of the photon where the spectrum should be computed.
    mpbh: float
        Mass of the black-hole in GeV.
    """
    pion_energies, greybodies = get_greybody_factors(0, mpbh)

    # Pion spectrum is:
    #   dN/dEdt = Gamma / (2pi) / (e^E/T - 1)
    # and 'greybodies' is
    #   Gamma / (e^E/T - 1),
    # with all units in GeV. So we divide by 2pi and convert GeV to time to get
    # units of (GeV^-1 s^-1)
    dnde_pis = greybodies / (2.0 * np.pi) * TIME_CONVERSION

    return convolve_spectrum(
        photon_energies, pion_energies, dnde_pis, __dnde_neutral_pion
    )


def compute_charged_pion_spectrum(
    photon_energies: List[float], mpbh: float
) -> List[float]:
    """
    Compute the secondary gamma-ray spectrum produced from a black-hole
    evaporating into charged pions.

    Parameters
    ----------
    photon_energies: array-like
        Energies of the photon where the spectrum should be computed.
    mpbh: float
        Mass of the black-hole in GeV.
    """
    pion_energies, greybodies = get_greybody_factors(0, mpbh)

    # Pion spectrum is:
    #   dN/dEdt = 2 * Gamma / (2pi) / (e^E/T - 1)
    # (2 for 2 d.o.f.) and 'greybodies' is
    #   Gamma / (e^E/T - 1),
    # with all units in GeV. So we divide by 2pi and convert GeV to time to get
    # units of (GeV^-1 s^-1)
    dnde_pis = 2.0 * greybodies / (2.0 * np.pi) * TIME_CONVERSION

    return convolve_spectrum(
        photon_energies, pion_energies, dnde_pis, __dnde_charged_pion
    )


def compute_electron_spectrum(
    photon_energies: List[float],
    electron_energies: List[float],
    electron_dist: List[float],
) -> List[float]:
    """
    Compute the FSR spectrum off electron evaporated from a PBH.
    """

    def dnde_electron(photon_energy, electron_energy):
        return ap_spec(photon_energy, electron_energy, M_ELECTRON, 1)

    return convolve_spectrum(
        photon_energies, electron_energies, electron_dist, dnde_electron
    )


def compute_muon_spectrum(
    photon_energies: List[float],
    muon_energies: List[float],
    muon_dist: List[float],
) -> List[float]:
    """
    Compute the FSR + decay spectrum off muon evaporated from a PBH.
    """

    def dnde_muon(photon_energy, muon_energy):
        return (
            ap_spec(photon_energy, muon_energy, M_MUON, 1)
            + _decay_muon(photon_energy * 1e3, muon_energy * 1e3) * 1e3
        )

    return convolve_spectrum(
        photon_energies, muon_energies, muon_dist, dnde_muon
    )


def spectrum_geometic_approximation(
    energies: List[float], mpbh: float, spin2: int
) -> List[float]:
    """
    Compute the spectrum from a PBH using the geometric approximation.

    Parameters
    ----------
    energies: array-like
        Energies where the spectrum should be computed.
    mpbh: float
        Mass of the black hole in GeV.
    spin2:
        Twice the spin of the emitted particle.

    Returns
    -------
    spec: array-like
        Spectrum evaluated at `energies`.
    """
    eta = 1.0 if spin2 % 2 == 0 else -1.0
    T = mass_to_temperature(mpbh)
    greybodies = 27.0 * mpbh ** 2 * G_NEWTON ** 2 * energies ** 2
    boltz = 1.0 / (np.exp(energies / T) - eta)

    return greybodies * boltz / (2.0 * np.pi) * TIME_CONVERSION


# ==============================
# ---- Black Hawk Functions ----
# ==============================


class BlackHawk:
    """
    Class for running the BlackHawk code.
    """

    def __init__(self, mpbh):
        """
        Initialize a BlackHawk runner object.

        Parameters
        ----------
        mpbh: float
            Mass of the black-hole in grams.
        """
        self.mpbh = mpbh
        self.path_to_blackhawk = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "blackhawk_v1.2"
        )
        self.num_primary_eng = 1000
        self.min_primary_eng = 5e-7  # 0.5 keV
        self.max_primary_eng = 1.0  # 1.0 GeV

        # Dictionaries of arrays storing the results
        self.primary = {
            "energies": None,
            "photon": None,
            "gluon": None,
            "higgs": None,
            "W": None,
            "Z": None,
            "neutrino": None,
            "electron": None,
            "muon": None,
            "tau": None,
            "up": None,
            "down": None,
            "charm": None,
            "strange": None,
            "top": None,
            "bottom": None,
            "graviton": None,
        }
        self.secondary = {
            "energies": None,
            "photon": None,
            "electron": None,
            "nu_e": None,
            "nu_mu": None,
            "nu_tau": None,
            "proton": None,
        }

    def __clean_dirs(self):
        """
        Clean up the directories/files that were created by BlackHawk
        and by `run`.
        """
        results_files = [
            "BH_spectrum.txt",
            "instantaneous_primary_spectra.txt",
            "instantaneous_secondary_spectra.txt",
            "temp.txt",
        ]
        os.remove(os.path.join(self.path_to_blackhawk, "temp.txt"))
        for file in results_files:
            os.remove(
                os.path.join(self.path_to_blackhawk, "results", "TEMP", file)
            )
        os.rmdir(os.path.join(self.path_to_blackhawk, "results", "TEMP"))

    def __write_parameter_file(self):
        """
        Write a temporary parameter file for BlackHawk.
        """
        par_file = os.path.join(self.path_to_blackhawk, "temp.txt")
        with open(par_file, "w") as file:
            file.writelines(
                [
                    "destination_folder = TEMP\n",
                    "full_output = 1\n",
                    "interpolation_method = 0\n",
                    "BHnumber = 1\n",
                    f"Mmin = {self.mpbh}\n",
                    f"Mmax = {self.mpbh}\n",
                    "anumber = 1\n",
                    "amin = 0\n",
                    "amax = 0.5\n",
                    "spectrum_choice = 0\n",
                    "spectrum_choice_a = 0\n",
                    "amplitude_lognormal = 1.0\n",
                    "amplitude_lognormal2 = 1.0\n",
                    "stand_dev_lognormal = 1.0\n",
                    "crit_mass_lognormal = 1.0\n",
                    "amplitude_powerlaw = 1.0\n",
                    "eqstate_powerlaw = 0.3333\n",
                    "amplitude_critical_collapse = 1.0\n",
                    "crit_mass_critical_collapse = 1.0\n",
                    "amplitude_uniform = 1.0\n",
                    "stand_dev_a_gaussian = 1.0\n",
                    "mean_a_gaussian = 0.5\n",
                    "table = table.txt\n",
                    "tmin = 1e-30\n",
                    "limit = 5000\n",
                    f"Enumber = {self.num_primary_eng}\n",
                    f"Emin = {self.min_primary_eng}\n",
                    f"Emax = {self.max_primary_eng}\n",
                    "particle_number = 15\n",
                    "grav = 1\n",
                    "primary_only = 0\n",
                    "hadronization_choice = 2\n",
                ]
            )

    def __read_results(self):
        """
        Read in and store the data produced from BlackHawk.
        """
        # Collect the primary spectra

        print(self.path_to_blackhawk)
        data_primary = np.genfromtxt(
            os.path.join(
                self.path_to_blackhawk,
                "results",
                "TEMP",
                "instantaneous_primary_spectra.txt",
            ),
            skip_header=2,
        ).T
        for i, key in enumerate(self.primary.keys()):
            self.primary[key] = data_primary[i]

        # Collect the secondary spectra
        data_secondary = np.genfromtxt(
            os.path.join(
                self.path_to_blackhawk,
                "results",
                "TEMP",
                "instantaneous_secondary_spectra.txt",
            ),
            skip_header=2,
        ).T
        for i, key in enumerate(self.secondary.keys()):
            self.secondary[key] = data_secondary[i]

    def run(self):
        """
        Write the BlackHawk parameters file given the exponent of the
        black-hole mass.

        Parameters
        ----------
        exponent: float
            Exponent of the black-hole mass, i.e. M_PBH = 10**exponent g.
        directory: str
            Directory name of the results relative to BlackHawk results
            directory.
        """

        # Save the current directory
        cur_dir = os.path.abspath(os.getcwd())

        # Move into the BlackHawk directory
        os.chdir(self.path_to_blackhawk)

        self.__write_parameter_file()

        # Run BlackHawk. The two 'y' are to answer questions from
        # BlackHawk.
        subprocess.run(
            "echo 'y' 'y' | ./BlackHawk_inst.x temp.txt",
            shell=True,
            stdout=subprocess.PIPE,
        )

        self.__read_results()
        self.__clean_dirs()

        # Return to original directory
        os.chdir(cur_dir)
