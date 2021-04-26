import numpy as np

"""
>>> Dynamic Habitat Indices
>>> Copyright (C) 2021 Andreas Rabe
"""

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    return ['cumulative', 'minimum', 'variation']


def forcepy_block(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16)
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads
    Write results into outarray.
    """

    # prepare data
    inarray = inarray[:, 0].astype(np.float32) # cast to float ...
    inarray[inarray == nodata] = np.nan        # ... and inject NaN to enable np.nan*-functions

    # calculate DHI
    cumulative = np.nansum(inarray, axis=0) / 1e2
    minimum    = np.nanmin(inarray, axis=0)
    variation  = np.nanstd(inarray, axis=0) / np.nanmean(inarray, axis=0) * 1e4

    # store results
    for arr, outarr in zip([cumulative, minimum, variation], outarray):
        valid = np.isfinite(arr)
        outarr[valid] = arr[valid]
