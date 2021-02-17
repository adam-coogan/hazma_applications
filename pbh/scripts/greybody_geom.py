from utils import (
    G_NEWTON,
    BlackHawk,
    temperature_to_mass,
    MASS_CONVERSION,
)
import matplotlib.pylot as plt
import numpy as np


def spec_geom(E, T, M):
    gam = 27 * G_NEWTON ** 2 * M ** 2 * E ** 2
    boltz = 1.0 / (np.exp(E / T) - 1.0)
    return gam * boltz / (2.0 * np.pi)


if __name__ == "__main__":
    mpbh_gram = 1.5e18
    mpbh_gev = mpbh_gram * MASS_CONVERSION
    blackhawk = BlackHawk(mpbh_gram)
    blackhawk.run()

    engs = blackhawk.primary["energies"]
    spec = blackhawk.primary["photon"]

    T = temperature_to_mass(mpbh_gev)
    spec2 = spec_geom(engs, T, mpbh_gev)

    plt.figure(dpi=150)
    plt.plot(engs, spec)
    plt.plot(engs, spec2)
    plt.show()

