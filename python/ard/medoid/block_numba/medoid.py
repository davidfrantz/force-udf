import numpy as np
from numba import jit, prange, set_num_threads

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

    return bandnames


def forcepy_block(inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    """
    This is a wrapper that runs the actual UDF in a parallel process.
    Maybe this solves a problem with Numba when calling the UDF from C.
    """
    from multiprocessing import Pool
    pool = Pool(1)
    result = pool.map(forcepy_block_2, [(inblock, outblock, dates, sensors, bandnames, nodata, nproc)])[0]
    outblock[:] = result

def forcepy_block_2(args):
    """
    We need this extra wrapper, because we can not call a @jit function directly with Pool.
    """
    inblock, outblock, dates, sensors, bandnames, nodata, nproc = args
    forcepy_block_3(inblock, outblock, dates, sensors, bandnames, nodata, nproc)
    return outblock

@jit(nopython=True, nogil=True, parallel=True)
def forcepy_block_3(inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    """
    inblock:   numpy.ndarray[nDates, nBands, nrows, ncols](Int16)
    outblock:  numpy.ndarray[nOutBands, nrows, ncols](Int16) initialized with no data values
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    nodata:    int
    nproc:     number of allowed processes/threads
    Write results into outblock.
    """
    set_num_threads(nproc)

    nDates, nBands, nY, nX = inblock.shape
    for iYX in prange(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        #print(iX, iY)
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
