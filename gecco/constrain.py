import warnings
from typing import Any, Callable, Optional, Tuple

import hazma.gamma_ray_parameters as grp
import numpy as np
from constants import TOBS, V_MW, X_KD
from hazma.parameters import omega_h2_cdm
from hazma.relic_density import relic_density
from hazma.theory import TheoryCMB, TheoryGammaRayLimits, TheoryAnn, TheoryDec
from rich.progress import Progress
from scipy.optimize import root_scalar
from utils import get_progress_update, sigmav

GECCO_BG_MODELS = {
    "GECCO (GC 1', NFW)": grp.GalacticCenterBackgroundModel(),
    "GECCO (GC 1', Einasto)": grp.GalacticCenterBackgroundModel(),
    "GECCO (Draco 1')": grp.GeccoBackgroundModel(),
    "GECCO (M31 1')": grp.GeccoBackgroundModel(),
    r"GECCO (GC 5$^\circ$, NFW)": grp.GalacticCenterBackgroundModel(),
    r"GECCO (GC 5$^\circ$, Einasto)": grp.GalacticCenterBackgroundModel(),
    r"GECCO (Draco $5^\circ$)": grp.GeccoBackgroundModel(),
    r"GECCO (M31 $5^\circ$)": grp.GeccoBackgroundModel(),
}

GECCO_TARGETS_ANN = {
    "GECCO (GC 1', NFW)": grp.gc_targets["nfw"]["1 arcmin cone"],
    "GECCO (GC 1', Einasto)": grp.gc_targets_optimistic["ein"]["1 arcmin cone"],
    "GECCO (Draco 1')": grp.draco_targets["nfw"]["1 arcmin cone"],
    "GECCO (M31 1')": grp.m31_targets["nfw"]["1 arcmin cone"],
}
GECCO_TARGETS_DEC = {
    r"GECCO (GC 5$^\circ$, NFW)": grp.gc_targets["nfw"]["5 deg cone"],
    r"GECCO (GC 5$^\circ$, Einasto)": grp.gc_targets_optimistic["ein"]["5 deg cone"],
    r"GECCO (Draco $5^\circ$)": grp.draco_targets["nfw"]["5 deg cone"],
    r"GECCO (M31 $5^\circ$)": grp.m31_targets["nfw"]["5 deg cone"],
}
EXISTING_MEASUREMENTS = {
    "COMPTEL": grp.comptel_diffuse,
    "EGRET": grp.egret_diffuse,
    "Fermi": grp.fermi_diffuse,
    "INTEGRAL": grp.integral_diffuse,
}


def limit_gecco(
    model: TheoryGammaRayLimits,
    params,
    target,
    tobs,
    bg_model,
    callback: Callable[[], None] = lambda: None,
    update_model: Callable[
        [TheoryGammaRayLimits, Any], None
    ] = lambda model, mx: setattr(model, "mx", mx),
    method="fisher",
):
    """
    Computes projected GECCO limits.
    """
    lims = np.zeros(len(params))
    for i, param in enumerate(params):
        update_model(model, param)
        if method == "fisher":
            lims[i] = model.fisher_limit(
                grp.effective_area_gecco,
                grp.energy_res_gecco,
                target,
                bg_model,
                tobs,
            )[0]
        elif method == "old":
            lims[i] = model.unbinned_limit(
                grp.effective_area_gecco,
                grp.energy_res_gecco,
                tobs,
                target,
                bg_model,
            )
        else:
            raise ValueError("invalid method: must be 'fisher' or 'old'")
        callback()

    return lims


def binned_limit(
    model: TheoryGammaRayLimits,
    params,
    measurement,
    method="chi2",
    callback: Callable[[], None] = lambda: None,
    update_model: Callable[
        [TheoryGammaRayLimits, Any], None
    ] = lambda model, mx: setattr(model, "mx", mx),
):
    lims = np.zeros(len(params))
    for i, param in enumerate(params):
        update_model(model, param)
        lims[i] = model.binned_limit(measurement, method=method)
        callback()
    return lims


def get_gamma_ray_limits(
    model,
    params,
    update_model: Callable[
        [TheoryGammaRayLimits, Any], None
    ] = lambda model, mx: setattr(model, "mx", mx),
    progress: Optional[Progress] = None,
):
    if isinstance(model, TheoryAnn):
        gecco_targets = GECCO_TARGETS_ANN
    elif isinstance(model, TheoryDec):
        gecco_targets = GECCO_TARGETS_DEC
    else:
        raise ValueError("invalid model class")

    lims_existing = {}
    for name, measurement in EXISTING_MEASUREMENTS.items():
        progress_update = get_progress_update(progress, name, len(params))
        lims_existing[name] = binned_limit(
            model,
            params,
            measurement,
            callback=progress_update,
            update_model=update_model,
        )

    lims_gecco = {}

    for name, target in gecco_targets.items():
        progress_update = get_progress_update(progress, name, len(params))
        lims_gecco[name] = limit_gecco(
            model,
            params,
            target,
            TOBS,
            GECCO_BG_MODELS[name],
            callback=progress_update,
            update_model=update_model,
        )

    return lims_gecco, lims_existing


def get_cmb_limit(
    model: TheoryCMB,
    params,
    x_kd=X_KD,
    update_model: Callable[[TheoryCMB, Any], None] = lambda model, mx: setattr(
        model, "mx", mx
    ),
    progress: Optional[Progress] = None,
):
    """
    Computes constraint on model at epoch of CMB formation.
    """
    progress_update = get_progress_update(progress, "CMB", len(params))
    lims = np.zeros(len(params))
    for i, param in enumerate(params):
        update_model(model, param)
        lims[i] = model.cmb_limit(x_kd)
        progress_update()
    return lims


def get_relic_density_limit_helper(
    model, param_name, param_range=(1e-5, 4 * np.pi), vx=V_MW
):
    lb = np.log10(param_range[0])
    ub = np.log10(param_range[1])

    def f(log10_gsxx):
        setattr(model, param_name, 10**log10_gsxx)
        return relic_density(model, semi_analytic=True) - omega_h2_cdm

    try:
        root = root_scalar(f, bracket=[lb, ub], method="brentq")
        if not root.converged:
            warnings.warn(f"root_scalar did not converge. Flag: {root.flag}")
        model.gsxx = 10**root.root
        return sigmav(model, vx)
    except ValueError as e:
        warnings.warn(f"Error encountered: {e}. Returning nan", RuntimeWarning)
        return np.nan


def get_relic_density_limit(
    model,
    param_name,
    other_param_grid,
    param_range: Tuple[float, float] = (1e-5, 4 * np.pi),
    vx=V_MW,
    update_model=lambda model, mx: setattr(model, "mx", mx),
    progress: Optional[Progress] = None,
):
    progress_update = get_progress_update(
        progress, "Relic density", len(other_param_grid)
    )
    lims = np.zeros(len(other_param_grid))
    for i, param in enumerate(other_param_grid):
        update_model(model, param)
        lims[i] = get_relic_density_limit_helper(model, param_name, param_range, vx)
        progress_update()
    return lims
