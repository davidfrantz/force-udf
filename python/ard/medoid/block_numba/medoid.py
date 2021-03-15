import numpy as np
from numba import jit, prange, set_num_threads

"""
>>> Medoid
>>> Copyright (C) 2021 Andreas Rabe
"""

def forcepy_init(dates, sensors, bandNames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    return bandNames


@jit(nopython=True, nogil=True, parallel=True)
def forcepy_block(inblock, outblock, dates, sensors, bandnames, nodata, nproc):
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
