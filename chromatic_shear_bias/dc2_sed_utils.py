import logging
import re

import fitsio
import numpy as np

from galsim import LookupTable
from galsim.catalog import Catalog
from galsim.config.bandpass import BuildBandpass
from galsim.config.input import RegisterInputType, RegisterInputConnectedType, GetInputObj, InputLoader
from galsim.config.sed import SEDBuilder, RegisterSEDType
from galsim.config.util import LoggerWrapper
from galsim.config.value import GetAllParams, SetDefaultIndex, RegisterValueType
from galsim.errors import GalSimConfigError
from galsim.sed import SED

class DC2_SEDCatalog(Catalog):
    """A class storing the data from a DC2 SED input catalog
    We primarily inherit from Catalog but redefine the readFits method
    and define new getRow and getSED methods.
    """
    def readFits(self):
        # Here we explicitly do _not_ read the entire catalog into memory; this is because
        # the DC2 SED catalogs are large and we only need one row at a time. However, we do
        # set certain properties of the table
        with fitsio.FITS(self.file_name) as fits:
            self._data = None  # Data is only ingested when the getRow method is called
            self.names = fits[1].get_colnames()
            self.nobjects = fits[1].get_nrows()
            self._ncols = fits[1].get_nrows()
            self.isfits = True

    def getRow(self, index):
        with fitsio.FITS(self.file_name) as fits:
            data = fits[1].read(rows=[index])
        return data

    def getSED(self, index):
        # DC2 SEDs are in units of 4.4659e13 W/Hz defined over bins in units of Angstrom
        # cf. https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/SCHEMA.md#extragalactic-catalogs
        wave_type = "Ang"
        flux_type = "fnu"  # we convert to this flux type below

        # From https://github.com/LSSTDESC/imSim/blob/main/imsim/instcat.py
        # Using area-weighted effective aperture over FOV
        # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
        _rubin_area = 0.25 * np.pi * 649**2  # cm^2

        sed_array = self.getRow(index)
        sed_bins = [_q for _q in sed_array.dtype.names if re.match(r"sed_\d+_\d+$", _q)]
        redshift = float(sed_array["redshift"])

        # DC2 SEDs are tophats; the file has columns corresponding to the value of the tophat
        # for bins in wavelength. We want to parse the bin edges from the column names and
        # repeat the tophat values for each bin edge.
        sed_bins_array = np.asarray(
            [np.cumsum(np.asarray(bin.split("_")[1:], dtype="float")) for bin in sed_bins]
        )
        sed_values_array = np.asarray(
            [np.repeat(sed_array[bin], 2) for bin in sed_bins]
        ) * 4.4659e13 * 1e7 / _rubin_area  # convert from 4.4659e13 W/Hz to erg/Hz/cm^2/s (i.e., nu)

        # We sort the SED tophats according to the bins, narrow by a small
        # factor to avoid overlapping bin edges; ravel to a 1-D array.
        _idx = np.argsort(sed_bins_array, axis=0)
        sed_bins_array = np.take_along_axis(sed_bins_array * np.asarray([1, 1 - 1e-9]), _idx, axis=0).ravel()
        sed_values_array = np.take_along_axis(sed_values_array, _idx, axis=0).ravel()

        # Set the SED response to 0 at the bluest wavelengths
        sed_bins_array = np.concatenate([np.array([1e-9, sed_bins_array[0] * (1 - 1e-9)], dtype="float"), sed_bins_array])
        sed_values_array = np.concatenate([np.array([0, 0], dtype="float"), sed_values_array])

        tophat_lookuptable = LookupTable(
            x=sed_bins_array,
            f=sed_values_array,
            interpolant="linear",
        )  # Linearly interpolate between the tophats
        sed = SED(tophat_lookuptable, wave_type, flux_type, redshift=redshift)
        # sed = sed.withFluxDensity(1., 500)  # TODO use appropriate units rather than rescaled flux density
        # sed = sed.atRedshift(redshift)
        sed = sed.withMagnitude(float(sed_array["mag_r_lsst"]), bps_zp[2])  # use the provided magnitudes to set overall normalization

        return sed


class DC2_SEDBuilder(SEDBuilder):
    """A class for defining an SED from a set of tophats

    DC2_SED expects the following parameters:

        index (required)    The index into the DC2 SED catalog for which to build the SED
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        req = {"index": int}
        opt = {}
        ignore = {}

        dc2_sed_cat = GetInputObj("dc2_sed_catalog", config, base, "DC2_SED_Catalog")

        SetDefaultIndex(config, dc2_sed_cat.getNObjects())

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        index = kwargs["index"]
        sed = dc2_sed_cat.getSED(index)
        logger.debug("obj %d: sed = %s", base.get("obj_num", 0), sed)

        return sed, safe


RegisterInputType("dc2_sed_catalog", InputLoader(DC2_SEDCatalog, has_nobj=True))
RegisterSEDType("dc2_sed", DC2_SEDBuilder())
