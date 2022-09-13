import numpy as np

"""
>>> Copyright (C) 2021-2022, Franz Schug, David Frantz
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

    # NDVI
    red   = np.argwhere(bandnames == b'RED')[0][0]
    nir   = np.argwhere(bandnames == b'NIR')[0][0]
    ndvi = (vals[:,nir]-vals[:,red]) / (vals[:,nir]+vals[:,red])
    
    outarray[:] = vals[np.argmax(ndvi),:]
