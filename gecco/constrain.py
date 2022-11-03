from typing import Any, Callable, Optional

import hazma.gamma_ray_parameters as grp
import numpy as np
from hazma.theory import TheoryCMB, TheoryGammaRayLimits
from rich.progress import Progress

X_KD = 1e-6
V_MW = 1e-3
N_MXS = 100
TOBS = 1e6  # s
GECCO_BG_MODELS = {
    "GECCO (GC 1', NFW)": grp.GalacticCenterBackgroundModel(),
    "GECCO (GC 1', Einasto)": grp.GalacticCenterBackgroundModel(),
    "GECCO (Draco 1')": grp.GeccoBackgroundModel(),
    "GECCO (M31 1')": grp.GeccoBackgroundModel(),
}

GECCO_TARGETS = {
    "GECCO (GC 1', NFW)": grp.gc_targets["nfw"]["1 arcmin cone"],
    "GECCO (GC 1', Einasto)": grp.gc_targets_optimistic["ein"]["1 arcmin cone"],
    "GECCO (Draco 1')": grp.draco_targets["nfw"]["1 arcmin cone"],
    "GECCO (M31 1')": grp.m31_targets["nfw"]["1 arcmin cone"],
}
EXISTING_MEASUREMENTS = {
    "COMPTEL": grp.comptel_diffuse,
    "EGRET": grp.egret_diffuse,
    "Fermi": grp.fermi_diffuse,
    "INTEGRAL": grp.integral_diffuse,
}


def get_progress_update(progress: Optional[Progress], desc: str, total: int):
    if progress is None:
        return lambda: None
    task = progress.add_task(desc, total=total)
    return lambda: progress.update(task, advance=1, refresh=True)


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
    for name, target in GECCO_TARGETS.items():
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
