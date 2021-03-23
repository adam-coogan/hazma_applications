"""
Module for generating constraints on the Hazma kinetic-mixing model.
"""

import warnings
from math import pi, sqrt

from tqdm.auto import tqdm
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional, Dict
from scipy.optimize import root_scalar

from hazma.flux_measurement import FluxMeasurement
from hazma.parameters import (
    omega_h2_cdm,
    sv_inv_MeV_to_cm3_per_s,
    dimensionless_hubble_constant,
)
from hazma.scalar_mediator import HiggsPortal
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


class Constraints(HiggsPortal):
    """
    Class for computing the gamma-ray constraints from the HiggsPortal model.
    """

    def __init__(self, mx, ms, gsxx, stheta):
        super().__init__(mx, ms, gsxx, stheta)

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
        mss: ArrayLike,
        existing_telescopes: Optional[Dict[str, FluxMeasurement]] = None,
        new_telescopes: Optional[List[str]] = None,
    ) -> Dict[str, ArrayLike]:
        """
        Compute the constraints on <sigma*v> from gamma-ray telescopes.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mss: ArrayLike
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
                self.ms = mss[i]
                constraints[name][i] = self.binned_limit(diffuse)

        # Compute constraints for new telescopes
        for name in new_telescopes:
            constraints[name] = np.zeros_like(mxs)
            for i, mx in enumerate(tqdm(mxs, desc="{:11}".format(name))):
                self.mx = mx
                self.ms = mss[i]
                constraints[name][i] = self.unbinned_limit(
                    EFFECTIVE_AREAS[name],
                    ENERGY_RESOLUTIONS[name],
                    self.observation_time,
                    self.target_params,
                    self.bg_model,
                )

        return constraints

    def compute_cmb_constraints(
        self, mxs: ArrayLike, mss: ArrayLike, x_kd: float = 1e-6
    ):
        """
        Compute the constraints on <sigma*v> from CMB.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mss: ArrayLike
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
            self.ms = mss[i]
            constraints[i] = self.cmb_limit(x_kd=x_kd)

        return constraints

    def compute_relic_density_contours(
        self,
        mxs: ArrayLike,
        mss: ArrayLike,
        semi_analytic: bool = True,
        stheta: float = 1,
        vx: float = 1e-3,
    ) -> ArrayLike:
        """
        Compute the values of stheta such that the dark matter relic density is
        the observed value.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mss: ArrayLike
            Mediator masses. Must be same shape as mx.
        semi_analytic: bool
            If true, a appoximate scheme is used to compute the relic density.
            Otherwise, the Boltzmann equation is solved.
        stheta: float = 1
            Value to which stheta is fixed.
            TODO: when would this matter?
        vx: float, optional
            Dark matter velocity.

        Returns
        -------
        svs: ArrayLike
            Values of the annihilation cross section such that the dark matter
            relic density is the observed value.
        """
        svs: np.ndarray = np.zeros_like(mxs)

        for i, mx in enumerate(tqdm(mxs, desc="relic-density")):
            self.stheta = stheta
            self.mx = mx
            self.ms = mss[i]

            def residual(log10_gsxx):
                """
                Compute the difference between the relic density and the observed
                dark matter relic density for use in root solving.
                """
                self.gsxx = 10 ** log10_gsxx
                return (
                    relic_density(self, semi_analytic=semi_analytic)
                    - omega_h2_cdm / dimensionless_hubble_constant ** 2
                )

            try:
                # Chosen by trial and error
                if self.ms < self.mx:
                    bracket = [-5, np.log10(4 * pi)]
                else:
                    bracket = [-4, 8]

                root = root_scalar(residual, bracket=bracket, xtol=1e-100, rtol=1e-3)
                self.gsxx = 10 ** (root.root)
                svs[i] = (
                    self.annihilation_cross_sections(2 * self.mx * (1 + 0.5 * vx ** 2))[
                        "total"
                    ]
                    * vx
                    * sv_inv_MeV_to_cm3_per_s
                )
            except ValueError:
                svs[i] = None

        return svs

    def consistent_with_pheno_constraints(
        self,
        mx_mg: ArrayLike,
        ms_mg: ArrayLike,
        sv_mg: ArrayLike,
        experiments: Optional[List[str]] = None,
        vx: float = 1e-3,
        gsxx_max: float = 4 * pi,
    ) -> ArrayLike:
        """
        Checks whether a set of (mx, ms, <sigma * v>) points are consistent
        with pheno constraints. This is done by establishing the minimum and
        maximum stheta value consistent with the cross section and determining
        whether any points in this range are allowed by pheno constraints.

        Parameters
        ----------
        mx_mg: array-like
            2D array of dark matter masses.
        ms_mg: array-like
            2D array of mediator masses.
        sv_mg: array-like
            2D array of <sigma * v> values at which to check consistency [cm^3
            / s].
        experiments: List[str], optional
            List of the non-telescope experiments to use in computing
            constraints. Default is all available through `self.constraints()`.
        vx: float, optional
            Dark matter velocity.
        gsxx_max: float, optional
            Maximum value of gsxx for which to check compatibility.

        Returns
        -------
        constraints: ArrayLike
            2D boolean array indicating which points are consistent with pheno
            constraints.

        """
        if experiments is None:
            experiments = list(self.constraints().keys())

        def _helper(mx, ms, sv):
            assert ms > mx
            self.mx, self.ms = mx, ms

            # Reference cross section
            self.gsxx, self.stheta = 1, 1
            sv_1 = (
                self.annihilation_cross_sections(2 * mx * (1 + 0.5 * vx ** 2))["total"]
                * vx
                * sv_inv_MeV_to_cm3_per_s
            )

            # Find smallest stheta compatible with <sigma * v> for given
            # gsxx_max. This assumes sv ~ (gsxx * stheta)**2, which is the case
            # when ms > mx.
            stheta_min = sqrt(sv / sv_1) / gsxx_max

            if stheta_min > 0.999:
                # No viable stheta values
                return -1e100

            # Otherwise find the weakest constraint over the compatible stheta
            # range
            stheta_grid = np.geomspace(stheta_min, 0.999, 20)
            constr_mins = np.full(stheta_grid.shape, np.inf)
            for i, stheta in enumerate(stheta_grid):
                self.stheta = stheta
                self.gsxx = np.sqrt(sv / sv_1) / self.stheta
                # Constraint comes from whichever experiment provides the
                # strongest constraint
                constr_mins[i] = np.min(
                    [
                        fn()
                        for name, fn in self.constraints().items()
                        if name in experiments
                    ]
                )

            return constr_mins.max()

        return np.vectorize(_helper)(mx_mg, ms_mg, sv_mg)

    def sv_max(
        self,
        mxs: ArrayLike,
        mss: ArrayLike,
        vx: float = 1e-3,
        gsxx_max: float = 4 * pi,
    ) -> ArrayLike:
        """
        Computes largest possible value of <sigma * v>.

        Parameters
        ----------
        mxs: ArrayLike
            Dark matter masses.
        mss: ArrayLike
            Mediator masses.
        vx: float, optional
            Dark matter velocity.
        gsxx_max: float, optional
            Value of gsxx.

        Returns
        -------
        svs: ArrayLike
            Largest <sigma * v> values at each dark matter and mediator mass
            [cm^3 / s].
        """
        self.gsxx, self.stheta = gsxx_max, 1

        svs: np.ndarray = np.zeros_like(mxs)
        for i, mx in enumerate(mxs):
            self.mx = mx
            self.ms = mss[i]
            svs[i] = (
                self.annihilation_cross_sections(2 * mx * (1 + 0.5 * vx ** 2))["total"]
                * vx
                * sv_inv_MeV_to_cm3_per_s
            )

        return svs
