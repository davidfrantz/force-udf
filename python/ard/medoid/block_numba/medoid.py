import numpy as np
from numba import jit, prange, set_num_threads


@jit(nopython=True, nogil=True, parallel=True)
def forcepy_block(inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    """Calculate medoid."""

    set_num_threads(nproc)

    nDates, nBands, nY, nX = inblock.shape
    for iYX in prange(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        inarray = inblock[:, :, iY, iX]
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
        outblock[:, iY, iX] = medoid


def testBlockProcessing():
    nodata = -9999
    inarray = np.array([[0, 0], [0, 10], [10, 0], [10, 10], [5, 5]], dtype=np.int16)
    inarray = np.concatenate([inarray * 0 + nodata, inarray, inarray * 0 + nodata])  # add no data pixel
    nDates, nBands = inarray.shape
    nY, nX = 300, 300
    inarray = inarray.reshape((nDates, nBands, 1, 1)) * np.ones((nDates, nBands, nY, nX), dtype=np.int16)
    outarray = np.full((nBands, nY, nX), nodata, dtype=np.int16)
    nproc = 4
    forcepy_block(inarray, outarray, None, None, None, nodata, nproc)
    print('Done', outarray)
    assert np.all(outarray == np.full_like(outarray, 5))


if __name__ == '__main__':
    testBlockProcessing()
