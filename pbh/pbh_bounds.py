from os.path import join

import numpy as np
from hazma.parameters import Msun_to_g
from scipy.interpolate import interp1d

default_base_dir = "PBHbounds/bounds/"


class PBHBounds:
    """
    Container for PBH bounds.
    """

    def __init__(self, bound_names, base_dir=default_base_dir):
        self._base_dir = base_dir
        self.bound_names = bound_names

    # Reload bounds when `bound_names` or `_base_dir` change
    @property
    def base_dir(self):
        return self._base_dir

    @base_dir.setter
    def base_dir(self, base_dir):
        self._base_dir = base_dir
        self._reload_bounds()

    @property
    def bound_names(self):
        return self._bound_names

    @bound_names.setter
    def bound_names(self, bound_names):
        self._bound_names = bound_names
        self._reload_bounds()

    def _reload_bounds(self):
        self._bounds = {}
        self._bound_interps = {}
        for bn in self.bound_names:
            # Our bound is not yet in PBHbounds
            if bn == "COMPTEL":
                m_pbhs, f_pbhs = np.loadtxt(join("data", f"{bn}.txt"), unpack=True)
            else:
                m_pbhs, f_pbhs = np.loadtxt(
                    join(self.base_dir, f"{bn}.txt"), unpack=True
                )
            m_pbhs *= Msun_to_g
            self._bounds[bn] = m_pbhs, f_pbhs
            self._bound_interps[bn] = interp1d(
                m_pbhs, f_pbhs, bounds_error=False, fill_value=np.inf
            )

    @property
    def bounds(self):  # read-only
        return self._bounds

    @property
    def bound_interps(self):  # read-only
        return self._bound_interps

    # Wrap `bounds`
    def __getitem__(self, name):
        return self.bounds[name]

    def __len__(self):
        return len(self.bounds)

    def keys(self):
        return self.bounds.keys()

    def values(self):
        return self.bounds.values()

    def items(self):
        return self.bounds.items()

    def bound_envelope(self):
        f_interps = []
        m_min = min([interp.x.min() for interp in self.bound_interps.values()])
        m_max = max([interp.x.max() for interp in self.bound_interps.values()])
        m_pbhs = np.geomspace(m_min, m_max, 500)
        f_pbhs = np.stack([f(m_pbhs) for f in self.bound_interps.values()]).min(0)
        return (m_pbhs, f_pbhs)
