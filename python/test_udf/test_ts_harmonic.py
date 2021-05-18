# todo this is just a saved version - clean this up!

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
