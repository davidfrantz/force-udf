import numpy as np
from scipy.spatial.distance import squareform, pdist


def forcepy_init(dates, sensors, bandNames):
    return bandNames


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """Calculate medoid."""

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
