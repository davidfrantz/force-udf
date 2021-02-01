import numpy as np
from scipy.spatial.distance import squareform, pdist


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """Calculate medoid."""

    valid = inarray[:, 0] != nodata  # skip no data; just check first band
    pairwiseDistancesSparse = pdist(inarray[valid], 'euclidean')
    pairwiseDistances = squareform(pairwiseDistancesSparse)
    cumulativDistance = np.sum(pairwiseDistances, axis=0)
    argMedoid = np.argmin(cumulativDistance)
    medoid = inarray[argMedoid, :]
    outarray[:] = medoid


def testPixelProcessing():
    nodata = -9999
    inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
    inarray = np.concatenate([inarray, inarray * 0 + nodata, inarray * 0 + nodata])  # add a lot of no data pixel
    outarray = np.array([nodata, nodata], dtype=np.int16)
    forcepy_pixel(inarray, outarray, None, None, None, nodata, 1)
    print('Done', outarray)
    assert np.all(outarray == (5, 5))


if __name__ == '__main__':
    testPixelProcessing()
