"""
Utilities for running simulations
Some functions from https://github.com/beckermr/pizza-cutter-sims
"""

import copy

import yaml
import numpy as np

import fitsio
import galsim
import ngmix
import metadetect


ORMASK_CUTS = [True, False]
S2N_CUTS = [7, 8, 9, 10, 15, 20]
MFRAC_CUTS = [0, 1, 2, 5, 8, 10, 20, 50, 80, 100]


def _get_dtype():
    fkeys = ["g1p", "g1m", "g1", "g2p", "g2m", "g2"]
    ikeys = ["s2n_cut", "ormask_cut", "mfrac_cut"]
    dtype = []
    for key in fkeys:
        dtype.append((key, "f8"))
    for key in ikeys:
        dtype.append((key, "i4"))

    dtype.append(("weight", "f8"))

    return dtype


def _get_size(n_sims):
    return n_sims * len(S2N_CUTS) * (len(ORMASK_CUTS) + len(MFRAC_CUTS))


def _make_res_arrays(n_sims):
    dt = _get_dtype()
    n = _get_size(n_sims)
    return np.zeros(n, dtype=dt), np.zeros(n, dtype=dt)


def generate_arguments(config, galsim_config, rng, n, memmap_dict, logger):
    """
    Generate arguments for the measurement builder.
    """
    i = 0
    while i < n:
        seed = rng.integers(low=1, high=2**29)
        arg_dict = {
            "config": copy.deepcopy(config),
            "galsim_config": copy.deepcopy(galsim_config),
            "rng": np.random.default_rng(seed),
            "memmap_dict": memmap_dict,
            "idx": i,
            "logger": logger,
        }
        yield arg_dict
        i += 1


def observation_builder(config, rng, logger):
    """
    Build an ngmix Observation from a GalSim config dictionary.
    """
    image = galsim.config.BuildImage(config, logger=None)
    # TODO: alternatively...
    # image = galsim.config.BuildImage(config, logger=None)
    # psf = galsim.config.GetFinalExtraOutput("psf", config, logger=None)[0]
    # weight = galsim.config.GetFinalExtraOutput("weight", config, logger=None)[0]
    # badpix = galsim.config.GetFinalExtraOutput("badpix", config, logger=None)[0]

    # build the GalSim object representation of the PSF
    psf_obj = galsim.config.BuildGSObject(config, "psf", logger=None)[0]
    psf = psf_obj.drawImage()

    # build the weight array
    # TODO: we should get the weight image from GalSim as an extra output
    #       since we don't know how to do this without writing the output
    #       to a file, we will do this hack for now
    weight = np.full(image.array.shape, 1 / config["image"]["noise"]["sigma"] ** 2)

    # build the noise array
    # TODO: we should get the noise image from GalSim as an extra output
    #       since we don't know how to do this without writing the output
    #       to a file, we will do this hack for now
    noise = rng.normal(0, config["image"]["noise"]["sigma"], image.array.shape)

    # build the bmask array
    # TODO: what is this anyways
    bmask = np.full(image.array.shape, int(0))

    # build the ormask array
    # TODO: what is this anyways
    ormask = np.full(image.array.shape, int(0))

    # construct the WCS
    # TODO: verify that this is the correct WCS to be using
    wcs = galsim.AffineTransform(
        config["pixel_scale"],
        0.0,
        0.0,
        config["pixel_scale"],
        origin=image.center,
    )

    # build the ngmix Observation of the PSF
    psf_cen = (psf.array.shape[0] - 1) / 2
    psf_jac = ngmix.jacobian.Jacobian(
        x=psf_cen,
        y=psf_cen,
        wcs=wcs,
    )
    psf_obs = ngmix.Observation(
        psf.array.copy(),
        jacobian=psf_jac,
    )

    # build the ngmix Observation of the image
    im_cen = (image.array.shape[0] - 1) / 2
    im_jac = ngmix.jacobian.Jacobian(
        x=im_cen,
        y=im_cen,
        wcs=wcs,
    )
    obs = ngmix.Observation(
        image.array.copy(),
        weight=weight.copy(),
        bmask=bmask.copy(),
        ormask=ormask.copy(),
        jacobian=im_jac,
        psf=psf_obs,
        noise=noise.copy(),
    )

    return obs


def make_pair_config(config, g=0.02):
    """
    Create a pair of configs to simulate scenes sheared with equal and opposite
    shears for the noise bias cancellation algorithm.
    """
    config_p = copy.deepcopy(config)
    config_m = copy.deepcopy(config)

    config_p["stamp"]["shear"]["g1"] = g
    config_m["stamp"]["shear"]["g1"] = -g

    return config_p, config_m


def measurement_builder(config, galsim_config, rng, memmap_dict, idx, logger):
    """
    Build measurements of simulations and write to a memmap
    """
    cosmic_shear = config["shear"]["g"]
    config_p, config_m = make_pair_config(galsim_config, cosmic_shear)

    # TODO: multithread the p/m pieces in parallel?
    obs_p = observation_builder(config_p, rng, logger)
    obs_m = observation_builder(config_m, rng, logger)
    obslist_p = ngmix.ObsList()
    obslist_p.append(obs_p)
    obslist_m = ngmix.ObsList()
    obslist_m.append(obs_m)

    # TODO: this should really be grabbing an ObsList for each band
    mbobs_p = ngmix.MultiBandObsList()
    mbobs_p.append(obslist_p)
    mbobs_m = ngmix.MultiBandObsList()
    mbobs_m.append(obslist_m)

    # TODO: how do we handle all of the RNGs? when are they shared?
    mdet_seed = rng.integers(low=1, high=2**29)
    mdet_rng = np.random.default_rng(mdet_seed)

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng,
    )

    measurements = measure_pairs(config, res_p, res_m)

    # return measurements

    # TODO: a bit of a hack but works for now
    idx_start = idx * _get_size(1)
    idx_stop = (idx + 1) * _get_size(1)
    memmap = np.memmap(**memmap_dict)
    memmap[idx_start:idx_stop] = np.stack(measurements[0], axis=-1)
    memmap.flush()  # TODO: I'm not sure if this call is necessary

    return


def compute_responsivity(meas, calibration_shear):
    """
    Compute the 11 element of the responsivity matrix.
    """
    return (np.mean(meas["g1p"]) - np.mean(meas["g1m"])) / (2 * calibration_shear)


def estimate_shear(meas, responsivity):
    """
    Compute the responsivity-calibrated shear.
    """
    return (1 / responsivity) * np.mean(meas["g1"])


def estimate_multiplicative_bias(meas_p, meas_m, calibration_shear, cosmic_shear):
    """
    Estimate the multiplicative bias with noise bias cancellation.
    """
    # TODO: we want to average estimate_shear over many simulations (?)
    responsivity_p = compute_responsivity(meas_p, calibration_shear)
    responsivity_m = compute_responsivity(meas_m, calibration_shear)
    return (
        estimate_shear(meas_p, responsivity_p) - estimate_shear(meas_m, responsivity_m)
    ) / (2.0 * np.abs(cosmic_shear)) - 1.0


def estimate_additive_bias(meas_p, meas_m, calibration_shear):
    """
    Estimate the additive bias with noise bias cancellation.
    """
    # TODO: we want to average estimate_shear over many simulations (?)
    responsivity_p = compute_responsivity(meas_p, calibration_shear)
    responsivity_m = compute_responsivity(meas_m, calibration_shear)

    return (
        estimate_shear(meas_p, responsivity_p) + estimate_shear(meas_m, responsivity_m)
    ) / 2.0


def estimate_biases(meas_p, meas_m, calibration_shear, cosmic_shear):
    """
    Estimate both additive and multiplicative biases with noise bias
    cancellation.
    """
    responsivity_p = compute_responsivity(meas_p, calibration_shear)
    responsivity_m = compute_responsivity(meas_m, calibration_shear)

    m = (
        estimate_shear(meas_p, responsivity_p) - estimate_shear(meas_m, responsivity_m)
    ) / (2.0 * np.abs(cosmic_shear)) - 1.0
    c = (
        estimate_shear(meas_p, responsivity_p) + estimate_shear(meas_m, responsivity_m)
    ) / 2.0

    return m, c


def measure_shear_metadetect(
    res, *, s2n_cut, t_ratio_cut, ormask_cut, mfrac_cut, model
):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    ormask_cut : bool
        If True, cut on the `ormask` flags.
    mfrac_cut : float or None
        If not None, cut objects with a masked fraction higher than this
        value.
    model : str
        The model kind (e.g. wmom).

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metadetect measurement.
    g1m : float
        The mean 1-component shape for the minus metadetect measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metadetect measurement.
    g2p : float
        The mean 2-component shape for the plus metadetect measurement.
    g2m : float
        The mean 2-component shape for the minus metadetect measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metadetect measurement.
    """

    def _mask(data):
        if "flags" in data.dtype.names:
            flag_col = "flags"
        else:
            flag_col = model + "_flags"

        _cut_msk = (
            (data[flag_col] == 0)
            & (data[model + "_s2n"] > s2n_cut)
            & (data[model + "_T_ratio"] > t_ratio_cut)
        )
        if ormask_cut:
            _cut_msk = _cut_msk & (data["ormask"] == 0)
        if mfrac_cut is not None:
            _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
        return _cut_msk

    op = res["1p"]
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op[model + "_g"][q, 0]

    om = res["1m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om[model + "_g"][q, 0]

    o = res["noshear"]
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o[model + "_g"][q, 0]
    g2 = o[model + "_g"][q, 1]

    op = res["2p"]
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op[model + "_g"][q, 1]

    om = res["2m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om[model + "_g"][q, 1]

    return (
        np.mean(g1p),
        np.mean(g1m),
        np.mean(g1),
        np.mean(g2p),
        np.mean(g2m),
        np.mean(g2),
    )


def measure_pairs(config, res_p, res_m):
    model = config["metadetect"]["model"]
    if model == "wmom":
        tcut = 1.2
    else:
        tcut = 0.5

    if len(res_p) > 0:
        wgt = len(res_p)

        # TODO: stack datap and datam into a single array of depth 2?
        dtype = _get_dtype()
        datap = []
        datam = []
        for ormask_cut in ORMASK_CUTS:
            for s2n_cut in S2N_CUTS:
                pgm = measure_shear_metadetect(
                    res_p,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                )
                mgm = measure_shear_metadetect(
                    res_m,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                )
                if pgm is None or mgm is None:
                    continue

                datap.append(
                    tuple(list(pgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
                )
                datam.append(
                    tuple(list(mgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
                )

        for mfrac_cut in MFRAC_CUTS:
            for s2n_cut in S2N_CUTS:
                pgm = measure_shear_metadetect(
                    res_p,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut / 100,
                    model=model,
                )
                mgm = measure_shear_metadetect(
                    res_m,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut / 100,
                    model=model,
                )
                if pgm is None or mgm is None:
                    continue

                datap.append(tuple(list(pgm) + [s2n_cut, -1, mfrac_cut, wgt]))
                datam.append(tuple(list(mgm) + [s2n_cut, -1, mfrac_cut, wgt]))

        return [(np.array(datap, dtype=dtype), np.array(datam, dtype=dtype))]
    else:
        return [(None, None)]
