import numpy as np
from numba import jit, prange, set_num_threads

def forcepy_init(dates, sensors, bandNames):
    assert len(bandNames) == 1
    outBandNames = list()
    for date, sensor in zip(dates, sensors):
        datestamp = str(np.datetime64('1970-01-01') + np.timedelta64(date, 'D')).replace('-', '')
        for name in ['COS-1', 'COS0', 'COS+1', 'COS-1_DTIME', 'COS1_DTIME']:
            outBandNames.append(f'{name}_{bandNames[0]}_{datestamp}_{sensor}')
    return outBandNames


@jit(nopython=True, nogil=True, parallel=True)
def forcepy_block(inblock, outblock, dates, sensors, bandnames, nodata, nproc):
    """Calculate clear observation sequences."""

    set_num_threads(nproc)

    nDates, nBands, nY, nX = inblock.shape
    for iYX in prange(nY * nX):
        iX = iYX % nX
        iY = iYX // nX
        inarray = inblock[:, :, iY, iX]
        outarray = outblock[:, iY, iX]
        valid = (inarray != nodata).ravel()
        for i in range(len(dates)):
            off = i * 5
            if not valid[i]:  # skip nodata
                outarray[off: off + 3] = nodata
            else:
                k1 = k2 = 0  # need to init variables because of numba type detection

                for k1 in range(i - 1, -1, -1):  # search backward for AD-1
                    if valid[k1]:
                        outarray[off] = inarray[k1, 0]
                        break

                outarray[off + 1] = inarray[i, 0]  # just copy center AD

                for k2 in range(i + 1, nDates):  # search forward AD+1
                    if valid[k2]:
                        outarray[off + 2] = inarray[k2, 0]
                        break

                if outarray[off] != nodata:  # calc timedelta for AD-1
                    days = dates[k1] - dates[i]
                    outarray[off + 3] = days

                if outarray[off + 2] != nodata:  # calc timedelta for AD+1
                    days = dates[k2] - dates[i]
                    outarray[off + 4] = days


def testPixelProcessing():
    nodata = -9999

    # create test timeseries (this is the test pixel from Benjamins screenshot)
    inarray = np.expand_dims(np.array([159, 167, nodata, 183, nodata, nodata, 207, 215, nodata], dtype=np.int16), 1)
    adMinusOne = [nodata, 159, nodata, 167, nodata, nodata, 183, 207, nodata]
    ad = [159, 167, nodata, 183, nodata, nodata, 207, 215, nodata]
    adPlusOne = [167, 183, nodata, 207, nodata, nodata, 215, nodata, nodata]
    timedelta = [nodata, 25, nodata, 41, nodata, nodata, 33, nodata, nodata]
    gold = np.array(adMinusOne + ad + adPlusOne + timedelta, dtype=np.int16).reshape((4, -1)).T.ravel()

    # calculate COS
    dates = np.array([np.datetime64('2000-01-01') + np.timedelta64(doy-1,'D')
                      for doy in [159, 167, 175, 183, 191, 199, 207, 215, 223]])
    outarray = np.full_like(gold, nodata, dtype=np.int16)
    forcepy_block(inarray, outarray, dates, None, None, nodata, 1)

    # check result
    print(list(outarray[3::4]))
    print(timedelta)

    assert np.all(outarray == gold)


if __name__ == '__main__':
    testPixelProcessing()
