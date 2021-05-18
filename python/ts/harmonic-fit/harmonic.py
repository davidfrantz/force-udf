from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import curve_fit

"""
>>> Harmonic time series fit
>>> Copyright (C) 2021 Andreas Rabe
"""

# some global config variables
date_start = 14727  # days since epoch (1970-01-01)
date_end = 15631  # days since epoch (1970-01-01)
step = 16  # days


def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """

    bandnames = [(datetime(1970, 1, 1) + timedelta(days=days)).strftime('%Y-%m-%d') + ' sin-interpolation'
                 for days in range(date_start, date_end, step)]
    print(bandnames)
    return bandnames


# regressor
# - define all three models from the paper
def objective_simple(x, a0, a1, b1, c1):
    return a0 + a1 * np.cos(2 * np.pi / 365 * x) + b1 * np.sin(2 * np.pi / 365 * x) + c1 * x


def objective_advanced(x, a0, a1, b1, c1, a2, b2):
    return objective_simple(x, a0, a1, b1, c1) + a2 * np.cos(4 * np.pi / 365 * x) + b2 * np.sin(4 * np.pi / 365 * x)


def objective_full(x, a0, a1, b1, c1, a2, b2, a3, b3):
    return objective_advanced(x, a0, a1, b1, c1, a2, b2) + a3 * np.cos(6 * np.pi / 365 * x) + b3 * np.sin(
        6 * np.pi / 365 * x)


# - choose which model to use
objective = objective_advanced


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

    # fit
    popt, _ = curve_fit(objective, xtrain, ytrain)

    # predict
    xtest = np.array(range(date_start, date_end, step))
    ytest = objective(xtest, *popt)

    outarray[:] = ytest
