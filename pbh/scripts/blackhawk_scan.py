# Script for scanning over PBH masses between 1e15 and 1e18 and running
# BlackHawk to generate the gamma-ray spectra.

from utils import BlackHawk
import numpy as np
import os
import pandas as pd
import tqdm

# Path to current directory
CUR_DIR = os.path.abspath(os.getcwd())

# Path to the directory containing BlackHawk code. On my machine, this is
# is the project root directory. Modify according to where you have blackhawk.
BLACKHAWK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "blackhawk_v1.2"
)

# Path to BlackHawk results directory
BLACKHAWK_RESULTS_DIR = os.path.join(BLACKHAWK_DIR, "results")

# Path to BlackHawk parameters file
PAR_FILE = os.path.join(BLACKHAWK_DIR, "parameters.txt")

# Path to where we will store the data from BlackHawk results
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)


if __name__ == "__main__":
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )

    primary_photon = {"energies": None}
    primary_electron = {"energies": None}
    secondary_photon = {"energies": None}
    secondary_electron = {"energies": None}

    masses = np.geomspace(1e15, 6e18, 100)
    blackhawk = BlackHawk(masses[0])
    for i in tqdm.tqdm(range(len(masses))):
        mass = masses[i]
        blackhawk.mpbh = mass
        blackhawk.run()

        if i == 0:
            primary_photon["energies"] = blackhawk.primary["energies"]
            primary_electron["energies"] = blackhawk.primary["energies"]
            secondary_photon["energies"] = blackhawk.secondary["energies"]
            secondary_electron["energies"] = blackhawk.secondary["energies"]

        primary_photon[str(mass)] = blackhawk.primary["photon"]
        primary_electron[str(mass)] = blackhawk.primary["electron"]
        secondary_photon[str(mass)] = blackhawk.secondary["photon"]
        secondary_electron[str(mass)] = blackhawk.secondary["electron"]

    pd.DataFrame(primary_photon).to_csv(
        os.path.join(out_dir, "dnde_photon_primary.csv"), index=False
    )
    pd.DataFrame(secondary_photon).to_csv(
        os.path.join(out_dir, "dnde_photon_secondary.csv"), index=False
    )

    pd.DataFrame(primary_electron).to_csv(
        os.path.join(out_dir, "dnde_electron_primary.csv"), index=False
    )
    pd.DataFrame(secondary_electron).to_csv(
        os.path.join(out_dir, "dnde_electron_secondary.csv"), index=False
    )
