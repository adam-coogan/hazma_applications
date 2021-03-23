"""
Module for generating constraints on the Hazma kinetic-mixing model.
"""

import warnings
from math import pi

from tqdm.auto import tqdm
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional, Dict
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d

from hazma.flux_measurement import FluxMeasurement
from hazma.parameters import (
    omega_h2_cdm,
    sv_inv_MeV_to_cm3_per_s,
    dimensionless_hubble_constant,
)
from hazma.vector_mediator import KineticMixing
from hazma.relic_density import relic_density
from hazma.gamma_ray_parameters import (
    TargetParams,
    # background
    BackgroundModel,
    default_bg_model,
    # targets
    gc_targets_optimistic,
    # effective areas
    effective_area_adept,
    effective_area_amego,
    effective_area_all_sky_astrogam,
    effective_area_e_astrogam,
    effective_area_gecco,
    effective_area_grams,
    effective_area_mast,
    effective_area_pangu,
    # energy resolutions
    energy_res_adept,
    energy_res_amego,
    energy_res_all_sky_astrogam,
    energy_res_e_astrogam,
    energy_res_gecco,
    energy_res_grams,
    energy_res_mast,
    energy_res_pangu,
    # diffuse
    comptel_diffuse,
    comptel_diffuse_targets_optimistic,
    fermi_diffuse,
    fermi_diffuse_targets_optimistic,
    integral_diffuse,
    integral_diffuse_targets_optimistic,
    egret_diffuse,
    egret_diffuse_targets_optimistic,
)

EFFECTIVE_AREAS = {
    "adept": effective_area_adept,
    "amego": effective_area_amego,
    "all-sky astrogam": effective_area_all_sky_astrogam,
    "e-astrogam": effective_area_e_astrogam,
    "gecco": effective_area_gecco,
    "grams": effective_area_grams,
    "mast": effective_area_mast,
    "pangu": effective_area_pangu,
}

ENERGY_RESOLUTIONS = {
    "adept": energy_res_adept,
    "amego": energy_res_amego,
    "all-sky astrogam": energy_res_all_sky_astrogam,
    "e-astrogam": energy_res_e_astrogam,
    "gecco": energy_res_gecco,
    "grams": energy_res_grams,
    "mast": energy_res_mast,
    "pangu": energy_res_pangu,
}

PROFILE = "ein"

babar_data = np.genfromtxt("data/BaBar.dat", delimiter=",")
lsnd_data = np.genfromtxt("data/LSND.dat", delimiter=",")
e137_data = np.genfromtxt("data/E137.dat", delimiter=",")

babar_interp = interp1d(babar_data.T[0], babar_data.T[1])
lsnd_interp = interp1d(lsnd_data.T[0], lsnd_data.T[1])
e137_interp = interp1d(e137_data.T[0], e137_data.T[1])


class Constraints(KineticMixing):
    """
    Class for computing the gamma-ray constraints from the KineticMixing
    model.
    """

    def __init__(self, mx, mv, gvxx, eps):
        super().__init__(mx, mv, gvxx, eps)

        # The baseline observation time for new telescopes
        self._observation_time = 3 * 365.25 * 24 * 60 * 60  # 3 yr
        # Baseline existing telescopes
        self._existing_telescopes = {
            "comptel": comptel_diffuse,
            "egret": egret_diffuse,
            "fermi": fermi_diffuse,
            "integral": integral_diffuse,
        }
        # Swap in optimistic targets
        self._existing_telescopes[
            "comptel"
        ].target = comptel_diffuse_targets_optimistic[PROFILE]
        self._existing_telescopes["egret"].target = egret_diffuse_targets_optimistic[
            PROFILE
        ]
        self._existing_telescopes["fermi"].target = fermi_diffuse_targets_optimistic[
            PROFILE
        ]
        self._existing_telescopes[
            "integral"
        ].target = integral_diffuse_targets_optimistic[PROFILE]
        # Baseline new telescopes
        self._new_telescopes = [
            "adept",
            "amego",
            "all-sky astrogam",
            "e-astrogam",
            "gecco",
            "grams",
            "mast",
            "pangu",
        ]
        # Baseline target (Galactic center)
        self._target_params = gc_targets_optimistic[PROFILE]["10x10 deg box"]
        # Baseline background model
        self._bg_model = BackgroundModel(
            [0, 1e5], lambda e: 7 * default_bg_model.dPhi_dEdOmega(e)
        )

    @property
    def observation_time(self) -> float:
        """
        The observation time for new telescopes in seconds.
        """
        return self._observation_time

    @property
    def existing_telescopes(self) -> Dict[str, FluxMeasurement]:
        """
        The existing telescopes using for constraining.
        """
        return self._existing_telescopes

    @property
    def new_telescopes(self) -> List[str]:
        """
        The new telescopes using for constraining.
        """
        return self._new_telescopes

    @property
    def target_params(self) -> TargetParams:
        """
        The parameters of the target being used for constraints.
        """
        return self._target_params

    @property
    def bg_model(self) -> BackgroundModel:
        """
        The background model being use for constraints.
        """
        return self._bg_model

    def compute_telescope_constraints(
        self,
        mxs: ArrayLike,
        mvs: ArrayLike,
        existing_telescopes: Optional[Dict[str, FluxMeasurement]] = None,
        new_telescopes: Optional[List[str]] = None,
    ) -> Dict[str, ArrayLike]:
        """
        Compute the constraints on <sigma*v> from gamma-ray telescopes.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mvs: ArrayLike
            Mediator masses.
        existing_telescopes: Dict[str, FluxMeasurement], optional
            List of existing telescopes. If None, comptel, egret, fermi and
            integral are used.
        new_telescopes: List[str], optional
            List of existing telescopes. If None, amego, adept, all-sky
            astrogam, e-astrogam, gecco, grams, mast, and pangu are used.

        Returns
        -------
        constraints: Dict[str, ArrayLike]
            Dictionary containing the constraints on <sigma*v>.
        """
        constraints = {}

        warnings.filterwarnings("ignore")

        if existing_telescopes is None:
            existing_telescopes = self.existing_telescopes

        if new_telescopes is None:
            new_telescopes = self.new_telescopes

        # Compute the constraints from existing telescopes
        for name, diffuse in existing_telescopes.items():
            constraints[name] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="{:11}".format(name))):
                self.mx = mx
                self.mv = mvs[i]
                constraints[name][i] = self.binned_limit(diffuse)

        # Compute constraints for new telescopes
        for name in new_telescopes:
            constraints[name] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="{:11}".format(name))):
                self.mx = mx
                self.mv = mvs[i]
                constraints[name][i] = self.unbinned_limit(
                    EFFECTIVE_AREAS[name],
                    ENERGY_RESOLUTIONS[name],
                    self.observation_time,
                    self.target_params,
                    self.bg_model,
                )

        return constraints

    def compute_cmb_constraints(
        self, mxs: ArrayLike, mvs: ArrayLike, x_kd: float = 1e-6
    ):
        """
        Compute the constraints on <sigma*v> from CMB.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mvs: ArrayLike
            Mediator masses.
        x_kd: float
            Value of m/T at kinetic decoupling.

        Returns
        -------
        constraints: ArrayLike
            Array of the constraints on <sigma*v> from CMB.
        """
        constraints: np.ndarray = np.zeros_like(mxs)
        for i, mx in enumerate(tqdm(mxs, desc="cmb")):
            self.mx = mx
            self.mv = mvs[i]
            constraints[i] = self.cmb_limit(x_kd=x_kd)

        return constraints

    def compute_relic_density_contours(
        self,
        mxs: ArrayLike,
        mvs: ArrayLike,
        semi_analytic: bool = True,
        log_eps_min: float = -30.0,
        log_eps_max: float = 0.0,
        vx: float = 1e-3,
    ):
        """
        Compute the values of epsilon such that the dark matter relic density
        is the observed value.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mvs: ArrayLike
            Mediator masses. Must be same shape as mx.
        semi_analytic: bool
            If true, a appoximate scheme is used to compute the relic density.
            Otherwise, the Boltzmann equation is solved.
        semi_analytic: bool
            If true, a appoximate scheme is used to compute the relic density.
            Otherwise, the Boltzmann equation is solved.
        log_eps_min: float
            Log of the minimum value of epsilon used to search for the relic
            density.
        log_eps_max: float
            Log of the maximum value of epsilon used to search for the relic
            density.

        Returns
        -------
        svs: ArrayLike
            Values of the annihilation cross section such that the dark matter
            relic density is the observed value.
        """
        svs: np.ndarray = np.zeros_like(mxs)

        def residual(log_eps):
            """
            Compute the difference between the relic density and the observed
            dark matter relic density for use in root solving.
            """
            self.eps = 10 ** log_eps
            return (
                relic_density(self, semi_analytic=semi_analytic)
                - omega_h2_cdm / dimensionless_hubble_constant ** 2
            )

        for i, mx in enumerate(tqdm(mxs, desc="relic-density")):
            self.mx = mx
            self.mv = mvs[i]
            try:
                root = root_scalar(residual, bracket=[log_eps_min, log_eps_max])
                self.eps = 10 ** (root.root)
                svs[i] = (
                    vx
                    * sv_inv_MeV_to_cm3_per_s
                    * self.annihilation_cross_sections(
                        2 * self.mx * (1 + 0.5 * vx ** 2)
                    )["total"]
                )

            except ValueError:
                svs[i] = None

        return svs

    def __babar_constraint(self, alphad: float, vx: float):
        """
        Compute the value of <sigma*v> constrained by BaBar.

        Parameters
        ----------
        alphad: float
            Alpha for the dark U(1).
        vx: float
            Velocity of the DM.

        Returns
        -------
        sigma_v: float
            The constraint on <sigma*v> from BaBar.
        """
        gvxx = self.gvxx
        eps = self.eps
        if self.mx * 1e-3 > babar_data.T[0][-1]:
            return np.inf
        if self.mx * 1e-3 < babar_data.T[0][0]:
            y = babar_data.T[1][0]
        else:
            y = babar_interp(self.mx * 1e-3)
        self.gvxx = np.sqrt(4 * pi * alphad)
        # y = eps**2 * alphad / 3**4
        self.eps = np.sqrt((self.mx / self.mv) ** 4 * y / alphad)
        sv = (
            self.annihilation_cross_sections(2 * self.mx * (1 + 0.5 * vx ** 2))["total"]
            * vx
        )
        self.eps = eps
        self.gvxx = gvxx
        return sv * sv_inv_MeV_to_cm3_per_s

    def __lsnd_constraint(self, alphad: float, vx: float):
        """
        Compute the value of <sigma*v> constrained by LSND.

        Parameters
        ----------
        alphad:float
            Alpha for the dark U(1).
        vx: float
            Velocity of the DM.

        Returns
        -------
        sigma_v: float
            The constraint on <sigma*v> from LSND.
        """
        gvxx = self.gvxx
        eps = self.eps
        if self.mx * 1e-3 > lsnd_data.T[0][-1]:
            return np.inf
        if self.mx * 1e-3 < lsnd_data.T[0][0]:
            y = lsnd_data.T[1][0]
        else:
            y = lsnd_interp(self.mx * 1e-3)
        self.gvxx = np.sqrt(4 * pi * alphad)
        # y = eps**2 * alphad / 3**4
        self.eps = np.sqrt((self.mx / self.mv) ** 4 * y / alphad)
        sv = (
            self.annihilation_cross_sections(2 * self.mx * (1 + 0.5 * vx ** 2))["total"]
            * vx
        )
        self.eps = eps
        self.gvxx = gvxx

        return sv * sv_inv_MeV_to_cm3_per_s

    def __e137_constraint(self, alphad: float, vx: float):
        """
        Compute the value of <sigma*v> constrained by E137.

        Parameters
        ----------
        alphad:float
            Alpha for the dark U(1).
        vx: float
            Velocity of the DM.

        Returns
        -------
        sigma_v: float
            The constraint on <sigma*v> from E137.
        """
        gvxx = self.gvxx
        eps = self.eps
        if self.mx * 1e-3 > e137_data.T[0][-1]:
            return np.inf
        if self.mx * 1e-3 < e137_data.T[0][0]:
            y = e137_data.T[1][0]
        else:
            y = e137_interp(self.mx * 1e-3)
        self.gvxx = np.sqrt(4 * pi * alphad)
        # y = eps**2 * alphad / 3**4
        self.eps = np.sqrt((self.mx / self.mv) ** 4 * y / alphad)
        sv = (
            self.annihilation_cross_sections(2 * self.mx * (1 + 0.5 * vx ** 2))["total"]
            * vx
        )
        self.eps = eps
        self.gvxx = gvxx
        return sv * sv_inv_MeV_to_cm3_per_s

    def compute_pheno_constraints(
        self,
        mxs: ArrayLike,
        mvs: ArrayLike,
        experiments: Optional[List[str]] = None,
        alphad: float = 0.5,
        vx: float = 1e-3,
    ):
        """
        Compute the constraints from non-telescope experiments.

        Parameters
        ----------
        mxs: array-like
            Dark matter masses.
        mvs: array-like
            Mediator masses. Must be same shape as mx.
        experiments: List[str], optional
            List of the non-telescope experiments to use in computing
            constraints. Default is ['babar', 'lsnd', 'e137'].
        alphad: float, optional
            Value of alpha for the dark U(1). Default is 0.5.
        vx: float, optional
            Dark matter velocity.

        Returns
        -------
        constraints: Dict[str, ArrayLike]
            Dictionary containing the constraints on <sigma*v> from
            non-telescope experiments.

        """
        if experiments is None:
            experiments = ["babar", "lsnd", "e137"]

        constraints = {}

        if "babar" in experiments:
            constraints["babar"] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="babar")):
                self.mx = mx
                self.mv = mvs[i]
                constraints["babar"][i] = self.__babar_constraint(alphad, vx)

        if "lsnd" in experiments:
            constraints["lsnd"] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="lsnd ")):
                self.mx = mx
                self.mv = mvs[i]
                constraints["lsnd"][i] = self.__lsnd_constraint(alphad, vx)

        if "e137" in experiments:
            constraints["e137"] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="e137 ")):
                self.mx = mx
                self.mv = mvs[i]
                constraints["e137"][i] = self.__e137_constraint(alphad, vx)

        return constraints
