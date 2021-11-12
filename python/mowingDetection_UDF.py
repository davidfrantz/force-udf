from scipy import interpolate
from datetime import datetime, timedelta
import time
import numpy as np

"""
>>> Mowing detection
>>> Copyright (C) 2021 Marcel Schwieder and Max Wesemeyer
"""


def get_cso(x, y, nodata=-9999, verbose=False, SoS=2018.2, EOS=2018.85):
    # if no gap is found it will return 5 days as gap
    # in case the last potential observation misses the function calculates the gap to the EOS
    if np.all(y == nodata):
        nodata_ratio = 0
        return nodata_ratio, (x[-1] - x[0])*365, nodata
    nodata_sum = np.sum(np.where(y==nodata, True, False))

    nodata_ratio = 1-(nodata_sum/len(y))
    data_gap = 0
    data_gap_indeces = []
    data_gap_dates_list = []
    for index, value in enumerate(y):
        if value == nodata:
            if index < 1:
                continue
            data_gap += 1
            if data_gap == 0:
                data_gap_indeces.append(index-1)
            data_gap_indeces.append(index)
        else:
            if len(x[data_gap_indeces]) >= 1:
                data_gap_indeces.append(index)
                gap_dates = x[data_gap_indeces]
                gap_days = (gap_dates[-1]-gap_dates[0])*365
                data_gap_dates_list.append(gap_days)
            else:
                data_gap_dates_list.append(0)
            data_gap = 0
            data_gap_indeces = []
    #########################
    # calculating gap to EOS
    index_to_end_save = -1
    for indeces_to_end in range(1, len(y)):
        if y[-indeces_to_end] == nodata:
            index_to_end_save = -(indeces_to_end + 1)
            continue
        else:
            break
    gap_to_EOS = (EOS - x[index_to_end_save]) * 365
    data_gap_dates_list.append(gap_to_EOS)
    #########################
    # calculating gap to SOS
    index_to_start_save = 0
    for indeces_to_start in range(len(y)):
        if y[indeces_to_start] == nodata:
            index_to_start_save = (indeces_to_start + 1)
            continue
        else:
            break
    gap_to_SOS = (x[index_to_start_save]-SoS) * 365
    data_gap_dates_list.append(gap_to_SOS)
    #########################
    if int(max(data_gap_dates_list)) == 0:
        data_gap_dates_list.append(5)
    if verbose:
        print(max(data_gap_dates_list), 'MAX GAP')
        print(x, y)
    return nodata_ratio, max(data_gap_dates_list), len(y)-nodata_sum


def toYearFraction(date):

    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def detectMow_S2_new(xs, ys,  clearWd, yr, type='ConHull', nOrder=3, model='linear'):
    another_thrs = 0.15

    Y = np.asarray(ys)/10000
    X = np.asarray(xs)

    Season_min_frac = yr + GLstart
    Season_max_frac = yr + GLend
    Start_frac = yr + PSstart
    End_frac = yr + PSend

    if type == 'ConHull':
        validIndex = Y < 1
        Y = Y[validIndex]

        # Y_Blu = Y_Blu[validIndex]
        X = X[validIndex]
        validIndex_2 = Y > 0
        Y = Y[validIndex_2]
        # Y_Blu = Y_Blu[validIndex_2]
        X = X[validIndex_2]

        ##############################################
        # averages duplicates in the time series
        records_array = X
        vals, inverse, count = np.unique(records_array, return_inverse=True, return_counts=True)

        idx_vals_repeated = np.where(count > 1)[0]

        vals_repeated = vals[idx_vals_repeated]

        new_x_ = np.unique(X)
        new_y_ = np.zeros(shape=vals.shape)
        for repeated_value in vals_repeated:
            where = np.where(X == repeated_value)
            new_y_[np.where(new_x_ == repeated_value)] = np.mean(Y[where])

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])

        mask = np.ones(shape=X.shape, dtype=bool)
        mask[cols] = False
        result = Y[mask]

        mask = np.ones(new_x_.shape, dtype=bool)
        mask[idx_vals_repeated] = False
        new_y_[mask] = result
        Y = new_y_
        X = new_x_

        ##############################################

        SoGLSdiff = np.abs(X - Season_min_frac)

        SoGLS = np.where(SoGLSdiff == np.nanmin(SoGLSdiff))

        if np.nanmin(SoGLSdiff) < Season_min_frac:
            SoGLS = SoGLS[0] + 1
        #    print(SoGLS)

        EoGLS = np.abs(X - Season_max_frac)
        EoGLS = np.where(EoGLS == np.nanmin(EoGLS))

        # Y = np.asarray(Y[SoGLS[0][0]:EoGLS[0][0]])
        # X = np.asarray(X[SoGLS[0][0]:EoGLS[0][0]])
        Y = np.asarray(Y[SoGLS[0]:EoGLS[0][0]])
        # Y_Blu = np.asarray(Y_Blu[SoGLS[0]:EoGLS[0][0]])
        X = np.asarray(X[SoGLS[0]:EoGLS[0][0]])

        # calculate NDVI difference (t1) - (t-1)
        yT1 = np.asarray(Y[1:])
        yT2 = np.asarray(Y[:-1])

        YDiffzero = [0]
        YDiff = yT1 - yT2
        YDiff = np.append(YDiffzero, YDiff)

        EVI_STD = np.nanstd(Y)
        EVI_mean = np.nanmean(Y)
        EVI_obs = sum(~np.isnan(Y))
        EVI_obs_pot = EVI_obs / len(Y)

        LoS = int(X[len(X) - 1] * 365 - X[0] * 365)
        EVI_obs_potII = EVI_obs / (LoS / 5)

        # identify first peak somewhere around the "mid" of the season
        # DOY 120
        MoSStart = np.abs(X - Start_frac)
        MoSStart = np.where(MoSStart == np.min(MoSStart))

        # DOY 240
        MoSEnd = np.abs(X - End_frac)
        MoSEnd = np.where(MoSEnd == np.min(MoSEnd))

        YPeakSub = Y[MoSStart[0][0]:MoSEnd[0][0]]

        if len(YPeakSub) == 0:
            return

        MoSPeak = np.nanmax(YPeakSub)
        MoSIndex = np.where(YPeakSub == MoSPeak)[0][0]
        IndexDiff = len(X[0:MoSStart[0][0]])
        MoSIndex = MoSIndex + IndexDiff

        earlyIndex2 = []
        lateIndex2 = []

        # todo check if early and late peak equals Y0
        Y0 = np.argwhere(np.isfinite(Y))
        Y0 = np.min(Y0)

        if MoSIndex <= 2:
            earlyPeak1 = np.nanmax(Y[0:MoSIndex])
            earlyIndex1 = np.min(np.where(Y == earlyPeak1))
        else:
            searchInd = np.argwhere(X <= X[MoSIndex] - clearWd * 0.00273973)
            if np.any(searchInd):
                searchInd = searchInd.max()
                earlyPeak1 = np.nanmax(Y[0:searchInd])
                earlyIndex1 = np.min(np.where(Y == earlyPeak1))
            else:
                earlyIndex1 = 0

        if MoSIndex + 2 == len(X):
            latePeak1 = np.nanmax(Y[MoSIndex + 1:len(X)])
            lateIndex1 = np.max(np.where(Y == latePeak1))
        else:
            searchInd2 = np.argwhere(X >= X[MoSIndex] + clearWd * 0.00273973)
            if np.any(searchInd2):
                searchInd2 = searchInd2.min()
                if searchInd2 != len(X)-1:
                    latePeak1 = np.nanmax(Y[searchInd2:len(X)-1])
                    lateIndex1 = np.max(np.where(Y == latePeak1))
                else:
                    lateIndex1 = 0
            else:
                lateIndex1 = 0

        if (earlyIndex1 != 0) and (earlyIndex1 - 2) > 0 and np.any(Y[0:earlyIndex1 - 2]):
            searchInd3 = np.argwhere(X <= X[earlyIndex1] - clearWd * 0.00273973)
            if np.any(searchInd3):
                searchInd3 = searchInd3.max()
                earlyPeak2 = np.nanmax(Y[0:searchInd3])
                earlyIndex2 = np.min(np.where(Y == earlyPeak2))

        if (lateIndex1 != 0) and lateIndex1 + 2 <= len(X) and np.any(Y[lateIndex1 + 2:len(X)]):
            searchInd4 = np.argwhere(X >= X[lateIndex1] + clearWd * 0.00273973)
            if np.any(searchInd4):
                searchInd4 = searchInd4.min()
                latePeak2 = np.nanmax(Y[searchInd4:len(X)])
                lateIndex2 = np.max(np.where(Y == latePeak2))

        Xarr = [X[Y0], X[earlyIndex1], X[MoSIndex], X[lateIndex1], X[len(X) - 1]]
        Yarr = [Y[Y0], Y[earlyIndex1], Y[MoSIndex], Y[lateIndex1], Y[len(Y) - 1]]

        if earlyIndex2:
            Xarr = [X[Y0], X[earlyIndex2], X[earlyIndex1], X[MoSIndex], X[lateIndex1], X[len(X) - 1]]
            Yarr = [Y[Y0], Y[earlyIndex2], Y[earlyIndex1], Y[MoSIndex], Y[lateIndex1], Y[len(Y) - 1]]
            if lateIndex2:
                Xarr = [X[Y0], X[earlyIndex2], X[earlyIndex1], X[MoSIndex], X[lateIndex1], X[lateIndex2], X[len(X) - 1]]
                Yarr = [Y[Y0], Y[earlyIndex2], Y[earlyIndex1], Y[MoSIndex], Y[lateIndex1], Y[lateIndex2], Y[len(Y) - 1]]

        if lateIndex2:
            Xarr = [X[Y0], X[earlyIndex1], X[MoSIndex], X[lateIndex1], X[lateIndex2], X[len(X) - 1]]
            Yarr = [Y[Y0], Y[earlyIndex1], Y[MoSIndex], Y[lateIndex1], Y[lateIndex2], Y[len(Y) - 1]]
            if earlyIndex2:
                Xarr = [X[Y0], X[earlyIndex2], X[earlyIndex1], X[MoSIndex], X[lateIndex1], X[lateIndex2], X[len(X) - 1]]
                Yarr = [Y[Y0], Y[earlyIndex2], Y[earlyIndex1], Y[MoSIndex], Y[lateIndex1], Y[lateIndex2], Y[len(Y) - 1]]

    if model == 'linear':
        # model and fit spline
        polyVal = np.interp(X, xp=Xarr, fp=Yarr)

    if model == 'poly':
        # model and fit polynom of n-th order
        poly = np.polyfit(Xarr, Yarr, nOrder)
        polyVal = np.polyval(poly, X)

    if model == 'spline':
        tck = interpolate.splrep(x=Xarr, y=Yarr, s=0)

        #  predict values with spline and write to array
        polyVal = interpolate.splev(X, tck, der=0)

    # difference between polynom and values
    diff = np.abs(polyVal - Y)
    diff_sum = np.nansum(diff)
    #diff_std = np.nanstd(diff)
    diff_mean = np.nanmean(diff)
    #diff_med = np.nanmedian(diff)
    #testVal = diff_sum * EVI_obs_pot
    testVal = diff_sum * EVI_obs_potII

    thresh = diff_mean
    NDVIthresh = -EVI_STD
    NDVIthresh_list = list(np.random.normal(NDVIthresh, GFstd, 100))

    # create empty array for neighborhood indices
    clearWidth = []

    mow_date_index = []
    mowingEvents = []
    mowingDoy = []

    if len(diff) > 0:
        i = 1
        for evIndex, ev in enumerate(diff):
            ndvi_diff_check = False
            NDV_Check_list = [YDiff[evIndex]] * 100
            # print(NDVIthresh_list)
            result = [a for a, b in zip(NDV_Check_list, NDVIthresh_list) if a < b]

            if len(result) >= posEval:
                ndvi_diff_check = True
            else:
                continue
                #ndvi_diff_check = False

            eventDate = X[evIndex]

            if evIndex == len(X)-1:
                eventDate_next = X[evIndex] + 1
            else:
                eventDate_next = X[evIndex + 1]

            if i == 1:
                if ev > thresh:
                    # check NDVI difference and compare to threshold
                    if ndvi_diff_check:
                        # check next observation
                        if eventDate_next - eventDate <= 6 * 0.00273973:
                            if YDiff[evIndex + 1] > another_thrs:
                                continue
                        # get julian date
                        doy = ((eventDate - yr) * 365) + 1
                        if doy > 305:
                            continue
                        else:
                            dt = datetime(yr, 1, 1)
                            dtdelta = timedelta(days=doy)
                            dates = str(dt + dtdelta)
                            date = dates[0:10]
                            mowingEvents.append(date)
                            mowingDoy.append(np.int(doy))
                            mow_date_index.append(evIndex)
                            i = i + 1
            else:
                if ev > thresh:
                    dec_date_preceding = X[np.array(mow_date_index)[-1]]
                    dec_date_current_iter = X[evIndex]
                    # delta days in decimal format
                    delta_days = dec_date_current_iter - dec_date_preceding
                    # clearwd (days) divided by 365 = minimum distance from preceding mowing event as decimal number
                    clearWd_days = clearWd / 365
                    if delta_days > clearWd_days:
                        # if evIndex not in clearWidth:
                        # date of event when threshold was crossed
                        eventDate = X[evIndex]
                        if ndvi_diff_check:
                            if eventDate_next - eventDate <= 6 * 0.00273973:
                                if YDiff[evIndex + 1] > another_thrs:
                                    continue
                            # get julian date
                            doy = ((eventDate - yr) * 365) + 1
                            if doy > 305:
                                continue
                            else:
                                #############################
                                # check if there is one observation that is higher than the preceding between
                                # two mowing events
                                time_mask = np.where((X >= X[mow_date_index[-1]]) & (X <= eventDate), True, False)
                                any_preced_lower = np.any(np.ediff1d(Y[time_mask]) > 0)
                                # in case there is no increase in EVI values between two mowing events
                                # "any_preced_lower" will be False
                                # print('Any observation higher than preceding between DOY ', mowingDoy[-1], 'and ',
                                #      int(doy), '?', any_preced_lower)
                                #############################
                                if any_preced_lower:
                                    dt = datetime(yr, 1, 1)
                                    dtdelta = timedelta(days=doy)
                                    dates = str(dt + dtdelta)
                                    date = dates[0:10]
                                    mowingEvents.append(date)
                                    mowingDoy.append(np.int(doy))
                                    mow_date_index.append(evIndex)
                                    i = i + 1
                    else:
                        None

    return mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, testVal

# new version
def forcepy_init(dates, sensors, bandnames):
    """
    dates:     numpy.ndarray[nDates](int) days since epoch (1970-01-01)
    sensors:   numpy.ndarray[nDates](str)
    bandnames: numpy.ndarray[nBands](str)
    """
    bandnames = ['mowingEvents', 'max_gap_days', 'CSO_ABS', 'Data_Ratio',
                 'Mow_1', 'Mow_2', 'Mow_3', 'Mow_4', 'Mow_5', 'Mow_6', 'Mow_7', 'Mean', 'Median', 'SD', 'diff_sum',
                 'diff_sum_dataavail', 'Error']

    return bandnames


def serial_date_to_string(srl_no):
    new_date = datetime(1970,1,1,0,0) + timedelta(int(srl_no) - 1)
    return new_date


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
    global GLstart, GLend, GLendII, PSstart, PSend, GFstd, posEval, clrwd

    ################# user defined parameters #################
    # define the approximate length of grassland season in which you expect the main mowing activity; in decimal years = DOY / 365; make sure too include a temporal buffer --> here end of December
    GLstart = 0.2  # DOY 73
    GLend = 1  # DOY 365

    # define end of grassland season
    GLendII = 0.85  # DOY

    # define the approximate length of the main vegetation season; i.e., time of the year in which you expect at least one peak
    PSstart = 0.33  # DOY 120
    PSend = 0.66  # DOY 240

    # adjust sensitivity of thresholds; i.e., width of gaussian function and number of positive evaluations needed
    GFstd = 0.02
    posEval = 40

    # define minimum distance between two consecutive mowing eventsin days
    clrwd = 15
    ###########################################################

    np.seterr(all='ignore')
    ts = inarray.squeeze()
    dateList = []

    for imgDate in dates:
        dateList.append(serial_date_to_string(imgDate))

    date = np.array(dateList)

    nodata = nodata

    all_no_data = np.all(ts == nodata)
    all_zero = np.all(ts == 0)

    if all_no_data:
        return
    elif all_zero:
        return
    else:
        try:
            # x = date
            x = np.array(list(map(toYearFraction, date)))
            yr = int(str(x[0])[:4])
            #################################
            # get sd mean median
            Season_min_frac = yr + GLstart
            Season_max_frac = yr + GLendII
            subsetter = np.where((Season_min_frac < x) & (x < Season_max_frac), True, False)

            Y = np.array(ts[subsetter])
            X = x[subsetter]
            nodata_ratio, max_gap_days, cso_abs = get_cso(X, Y, nodata=nodata, verbose=False, SoS=Season_min_frac, EOS=Season_max_frac)
            Y = np.array(ts[subsetter], dtype=np.float)
            Y[Y == nodata] = np.nan
            mean = np.nanmean(Y)
            median = np.nanmedian(Y)
            sd = np.nanstd(Y)

            Season_min_frac = yr + GLstart
            Season_max_frac = yr + GLend
            subsetter = np.where((Season_min_frac < x) & (x < Season_max_frac), True, False)
            X = x[subsetter]
            Y = ts[subsetter]

            mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, diff_sum_dataavail = detectMow_S2_new(X, Y,
                                                                                                           clearWd=clrwd,
                                                                                                           yr=yr,
                                                                                                           type='ConHull',                                                                                           nOrder=3,
                                                                                                           model='linear')
            #print(mowingEvents)

            mowing_doy_out = [0] * 7

            for index, doys in enumerate(mowing_doy_out):
                try:
                    mowing_doy_out[index] = mowingDoy[index]
                except:
                    break
            outarray[:] = [int(len(mowingEvents)), int(max_gap_days), int(cso_abs), int(nodata_ratio * 100),
                           mowing_doy_out[0],
                           mowing_doy_out[1], mowing_doy_out[2], mowing_doy_out[3], mowing_doy_out[4],
                           mowing_doy_out[5], mowing_doy_out[6], mean, median, sd,
                           int(diff_sum * 100),
                           int(diff_sum_dataavail * 100), 0]
            print(outarray)

            #return [int(len(mowingEvents)), int(max_gap_days), int(cso_abs), int(nodata_ratio * 100), mowing_doy_out[0],
            #        mowing_doy_out[1], mowing_doy_out[2], mowing_doy_out[3], mowing_doy_out[4],
            #        mowing_doy_out[5], mowing_doy_out[6], mean, median, sd, int(diff_sum * 100),
            #       int(diff_sum_dataavail * 100), 0]
        except:
            print('ERROR')
            outarray[-1] = 1



'''
# test data set
if __name__ == '__main__':
    args = np.array([-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,997,-9999,-9999
,1566,-9999,2242,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999
,-9999,3913,5102,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999
,4649,-9999,-9999,5352,-9999,-9999,4712,-9999,-9999,4291,-9999,-9999
,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999
,-9999,-9999,-9999,-9999,-9999,-9999,-9999,2490,-9999,-9999,-9999,-9999
,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,3550,-9999,-9999
,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999
,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999] ),np.array([2018.013698630137, 2018.0191780821917, 2018.0191780821917, 2018.0684931506848, 2018.1013698630136, 2018.1205479452055, 2018.1260273972603, 2018.1698630136987, 2018.1698630136987, 2018.2109589041097, 2018.213698630137, 2018.2601598173517, 2018.2656392694064, 2018.2738584474887, 2018.2930365296804, 2018.2985159817351, 2018.3012557077625, 2018.3012557077625, 2018.3067351598174, 2018.3067351598174, 2018.3149543378995, 2018.3341324200912, 2018.3396118721462, 2018.3423515981735, 2018.345091324201, 2018.3478310502283, 2018.3505707762556, 2018.3560502283106, 2018.3615296803653, 2018.36700913242, 2018.3697488584476, 2018.3697488584476, 2018.3752283105023, 2018.380707762557, 2018.3889269406393, 2018.3889269406393, 2018.394406392694, 2018.394406392694, 2018.3971461187214, 2018.4026255707763, 2018.408105022831, 2018.4108447488584, 2018.4135844748857, 2018.4163242009133, 2018.4245433789954, 2018.4300228310503, 2018.4327625570777, 2018.4519406392694, 2018.4574200913241, 2018.4574200913241, 2018.462899543379, 2018.4848173515982, 2018.4902968036529, 2018.4930365296805, 2018.4985159817352, 2018.5012557077625, 2018.5122146118722, 2018.5259132420092, 2018.5259132420092, 2018.5341324200913, 2018.5396118721462, 2018.545091324201, 2018.545091324201, 2018.5478310502283, 2018.553310502283, 2018.5615296803653, 2018.5642694063927, 2018.56700913242, 2018.5697488584474, 2018.580707762557, 2018.586187214612, 2018.5889269406393, 2018.6163242009131, 2018.635502283105, 2018.6437214611872, 2018.649200913242, 2018.6546803652968, 2018.6574200913242, 2018.6574200913242, 2018.662899543379, 2018.6683789954338, 2018.676598173516, 2018.676598173516, 2018.6985159817352, 2018.7122146118722, 2018.731392694064, 2018.739611872146, 2018.745091324201, 2018.753310502283, 2018.7724885844748, 2018.7834474885844, 2018.7861872146118, 2018.794406392694, 2018.7998858447488, 2018.8053652968038, 2018.8190639269405, 2018.8218036529681, 2018.8520547945207, 2018.8712328767124, 2018.8739726027397, 2018.876712328767, 2018.876712328767, 2018.9013698630138, 2018.9232876712329, 2018.9397260273972, 2018.9424657534246]), -9999
    #for i in range(1,len(args[0]),1):
    #    print(args[1][i],  args[0][i],',')


    result = forcepy_tsi(args)
    print('Done:', result)

#[1000, 2719, 3259, 7000, 6, 1000, -9999, -9999, 5000, 1840, -9999, -9999, 1300, 5000, 8000, 10000]
'''
