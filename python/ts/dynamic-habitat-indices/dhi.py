import numpy as np
from scipy.spatial.distance import squareform, pdist
## please check imports

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

    return ("cum", "min", "var")
# please check whether return is python-conform


# this function will get interpolated VI as 4d array (bands == 1) for exactly one year
# only three simple equations to be implemented: http://silvis.forest.wisc.edu/data/dhis/ (next to the image)
# give back array of 3 images
# the code below is from your medoid function (pixel-based, attention, this here should be a block function)
def forcepy_block(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
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
