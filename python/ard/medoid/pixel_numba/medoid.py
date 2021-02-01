import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, parallel=True)
def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    """Calculate medoid."""

    nDates, nBands, nRows, nColumns = inarray.shape
    inarray = inarray.reshape((nDates, nBands))  # skip empty dims
    cumulativDistance = np.zeros(shape=(nDates,), dtype=np.float32)
    for i in range(nDates):
        if inarray[i, 0] == nodata:
            cumulativDistance[i] = np.inf
            continue
        for j in range(i + 1, nDates):
            if inarray[j, 0] == nodata:
                continue
            distance = np.sum((inarray[i] - inarray[j]) ** 2)
            cumulativDistance[i] += distance
            cumulativDistance[j] += distance
    argMedoid = np.argmin(cumulativDistance)
    medoid = inarray[argMedoid, :]
    outarray[:] = medoid

def testPixelProcessing():
    nodata = -9999
    inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
    inarray = np.concatenate([inarray, inarray * 0 + nodata, inarray * 0 + nodata])  # add a lot of no data pixel
    inarray = np.expand_dims(np.expand_dims(inarray, 2), 3)
    outarray = np.array([nodata, nodata], dtype=np.int16)
    forcepy_pixel(inarray, outarray, None, None, None, nodata, 0)
    print('Done', outarray)
    assert np.all(outarray == (5, 5))


if __name__ == '__main__':
    testPixelProcessing()
