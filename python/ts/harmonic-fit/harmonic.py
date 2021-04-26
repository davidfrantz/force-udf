import numpy as np
from PyQt5.QtCore import QDate
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform, pdist

## please check imports

"""
>>> Harmonic time series fit
>>> Copyright (C) 2021 Andreas Rabe
"""

# some global config variables
date_start = 14727  # days since epoch (1970-01-01)
date_end   = 15631  # days since epoch (1970-01-01)
step       = 16  # days

def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    bandnames = [QDate(1970, 1, 1).addDays(days).toString('yyyy-MM-dd') + ' sin-interpolation'
                 for days in range(date_start, date_end, step)]
    return bandnames


def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
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

    # prepare dataset
    profile = inarray.flatten()
    valid = profile != nodata
    if len(valid) == 0:
        return
    xtrain = dates[valid]
    ytrain = profile[valid]

    # regressor
    def objective(x, a, b):
        return a * np.sin(2 * np.pi / 365 * x) + b

    # fit
    popt, _ = curve_fit(objective, xtrain, ytrain)

    # predict
    xtest = np.array(range(date_start, date_end, step))
    ytest = objective(xtest, *popt)

    outarray[:, 0, 0] = ytest


def test():  # pure python test
    from matplotlib.pylab import plot, show
    from test_udf.utils import Utils
    boaArray, dates, sensors = Utils.boaBlock(dateMin='20100101', dateMax='20130101')  # 3 years of data
    ndviArray = Utils.ndviBlock(boaArray)
    profile = ndviArray[:, 0, 10, 10].astype(float)
    valid = profile != -9999
    xtrain = dates[valid]
    ytrain = profile[valid]

    # regressor
    def objective(x, a, b):
        return a * np.sin(2 * np.pi / 365 * x) + b

    # fit
    popt, _ = curve_fit(objective, xtrain, ytrain)
    print(min(xtrain), max(xtrain))
    # predict
    xtest = np.linspace(min(xtrain), max(xtrain), 100)
    ytest = objective(xtest, *popt)

    plot(xtrain, ytrain, '*')
    plot(xtest, ytest, '-')
    show()


def test2():  # forcepy test
    from matplotlib.pylab import plot, show
    from test_udf.utils import Utils
    boaArray, dates, sensors = Utils.boaBlock(dateMin='20100101', dateMax='20130101')  # 3 years of data
    ndviArray = Utils.ndviBlock(boaArray)
    inarray = ndviArray[:, :, 10:11, 10:11]

    # doit
    bandnames = forcepy_init(dates, sensors, Utils.BOA_NAMES1)
    outarray = np.full(shape=(len(bandnames), 1, 1), fill_value=-9999)
    forcepy_pixel(inarray, outarray, dates, sensors, bandnames, -9999, 1)

    # plot result
    profile = inarray.flatten()
    valid = profile != -9999
    xtrain = dates[valid]
    ytrain = profile[valid]
    xtest = list(range(date_start, date_end, step))
    ytest = outarray.flatten()
    plot(xtrain, ytrain, '*')
    plot(xtest, ytest, '-')
    show()

if __name__ == '__main__':
    test2()
