import numpy as np
from numba import jit, prange, set_num_threads

def forcepy_init(dates, sensors, bandNames):
    return bandNames

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
