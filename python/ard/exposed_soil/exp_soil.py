import numpy as np

"""
>>> Copyright (C) 2022, Max Gerhards, Henning Buddenbaum, David Frantz
"""

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    return bandnames


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16), nrows & ncols always 1
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads (always 1)
    Write results into outarray.
    """

    # remove nodata
    inarray = inarray[:, :, 0, 0]
    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band
    if len(valid) == 0:
        return

    # subset
    vals = inarray[valid,:]

    # band indices
    green = np.argwhere(bandnames == b'GREEN')[0][0]
    red   = np.argwhere(bandnames == b'RED')[0][0]
    nir   = np.argwhere(bandnames == b'NIR')[0][0]
    swir1 = np.argwhere(bandnames == b'SWIR1')[0][0]

    # remove "high" NDVI
    ndvi = (vals[:,nir]-vals[:,red]) / (vals[:,nir]+vals[:,red])
    valid = np.where(ndvi < 0.3)[0]
    if len(valid) == 0:
        return

    # subset again
    vals = vals[valid,:]

    # Dry Bare Soil Index, Rasul et al. 2018
    inds  = ((vals[:,swir1]-vals[:,green]) / (vals[:,swir1]+vals[:,green])) - ((vals[:,nir]-vals[:,red]) / (vals[:,nir]+vals[:,red]))

    if np.all(inds == 0):
        return

    # weighted average
    # there still is an error when weights are 0. Need to fix
    outarray[:] = np.average(vals, weights = inds, axis = 0)
