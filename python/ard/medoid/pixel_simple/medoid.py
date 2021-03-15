import numpy as np
from scipy.spatial.distance import squareform, pdist

"""
>>> Medoid
>>> Copyright (C) 2021 Andreas Rabe
"""

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    
    print(dates)
    print(sensors)
    print(bandnames)

    return bandnames


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """
    inarray:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16)
    outarray:  numpy.ndarray[nOutBands](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads (always 1)
    Write results into outarray.
    """

    inarray = inarray[:, :, 0, 0]
    valid = np.where(inarray[:, 0] != nodata)[0]  # skip no data; just check first band
    if len(valid) == 0:
        return
    pairwiseDistancesSparse = pdist(inarray[valid], 'euclidean')
    pairwiseDistances = squareform(pairwiseDistancesSparse)
    cumulativDistance = np.sum(pairwiseDistances, axis=0)
    argMedoid = valid[np.argmin(cumulativDistance)]
    medoid = inarray[argMedoid, :]
    outarray[:] = medoid
