from scipy import interpolate
from datetime import datetime, timedelta
import time
import numpy as np
import warnings

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
    warnings.simplefilter('ignore')
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
        X = X[validIndex]
        validIndex_2 = Y > 0
        Y = Y[validIndex_2]
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

        # filter time series to season (check if needed or a code legacy)
        SoGLSdiff = np.abs(X - Season_min_frac)
        SoGLS = np.where(SoGLSdiff == np.nanmin(SoGLSdiff))
        EoGLS = np.abs(X - Season_max_frac)
        EoGLS = np.where(EoGLS == np.nanmin(EoGLS))
        Y = np.asarray(Y[SoGLS[0][0]:EoGLS[0][0]])
        X = np.asarray(X[SoGLS[0][0]:EoGLS[0][0]])

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
            if MoSIndex == 0:
                earlyPeak1 = Y[0]
            else:
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
    diff_mean = np.nanmean(diff)
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
                        
    if profileAnalytics:
        return mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, testVal, Xarr, Yarr, X, polyVal
    else:
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
    global GLstart, GLend, GLendII, PSstart, PSend, GFstd, posEval, clrwd, profileAnalytics

    ################# user defined parameters #################
    # define if you want to run the UDF in FORCE or display the result of the algorithm per pixel using QGIS-Plugin Profile Analytics
    # see details: https://enmap-box.readthedocs.io/en/latest/usr_section/usr_manual/eo4q.html?highlight=profile#profile-analytics
    # make sure to append an environmental variable in QGIS following this example:
    # Settings --> Options --> System --> Environment: Apply: Append | Variable: PYTHONPATH | Value: PATH\TO\mowingDetection_UDF.py
    
    profileAnalytics = False
    
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
    
    nodata = nodata

    all_no_data = np.all(ts == nodata)
    all_zero = np.all(ts == 0)
    
    if all_no_data:
        return
    elif all_zero:
        return
    else:
    
        dateList = []

        if profileAnalytics:
            for imgDate in dates:
                dateList.append(imgDate)
        else:
            for imgDate in dates:
                dateList.append(serial_date_to_string(imgDate))

        date = np.array(dateList)
    
        try:
            if profileAnalytics:
                x = date
            else:
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
            
            if profileAnalytics:
                mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, diff_sum_dataavail, xPeak, yPeak, xPol, yPol = detectMow_S2_new(
                    X, Y, clearWd=clrwd, yr=yr, type='ConHull', nOrder=3, model='linear'
                )
            else:
                mowingEvents, mowingDoy, diff_sum, EVI_obs, EVI_obs_pot, diff_sum_dataavail = detectMow_S2_new(
                    X, Y, clearWd=clrwd, yr=yr, type='ConHull', nOrder=3, model='linear'
                )

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
            if profileAnalytics:
                return mowingEvents, mowing_doy_out, xPeak, yPeak, xPol, yPol            
        except:
            #print('ERROR')
            outarray[-1] = 1



'''
# test data set
if __name__ == '__main__':
    profileAnalytics = True

    bandnames = forcepy_init(None, None, None)
    sensors = None

    text = '2021.0 7681.0, 2021.013698630137 7813.0, 2021.027397260274 7842.0, 2021.041095890411 7823.0, 2021.054794520548 7670.0, 2021.0684931506848 7567.0, 2021.0821917808219 7237.0, 2021.0958904109589 7237.0, 2021.109589041096 6993.0, 2021.123287671233 6916.0, 2021.13698630137 6863.0, 2021.150684931507 6853.0, 2021.164383561644 6937.0, 2021.1780821917807 7011.0, 2021.1917808219177 7022.0, 2021.2054794520548 7292.0, 2021.2191780821918 7541.0, 2021.2328767123288 7722.0, 2021.2465753424658 7667.0, 2021.2602739726028 7544.0, 2021.2739726027398 7145.0, 2021.2876712328766 7010.0, 2021.3013698630136 7457.0, 2021.3150684931506 7894.0, 2021.3287671232877 7927.0, 2021.3424657534247 7779.0, 2021.3561643835617 7655.0, 2021.3698630136987 7879.0, 2021.3835616438357 7926.0, 2021.3972602739725 8093.0, 2021.4109589041095 7964.0, 2021.4246575342465 7666.0, 2021.4383561643835 7035.0, 2021.4520547945206 7176.0, 2021.4657534246576 7406.0, 2021.4794520547946 7606.0, 2021.4931506849316 7740.0, 2021.5068493150684 7410.0, 2021.5205479452054 7269.0, 2021.5342465753424 7127.0, 2021.5479452054794 7101.0, 2021.5616438356165 7049.0, 2021.5753424657535 6826.0, 2021.5890410958905 6723.0, 2021.6027397260275 6510.0, 2021.6164383561643 6122.0, 2021.6301369863013 5919.0, 2021.6438356164383 6295.0, 2021.6575342465753 6443.0, 2021.6712328767123 7090.0, 2021.6849315068494 6990.0, 2021.6986301369864 6767.0, 2021.7123287671234 6507.0, 2021.7260273972602 6385.0, 2021.7397260273972 6284.0, 2021.7534246575342 6277.0, 2021.7671232876712 6243.0, 2021.7808219178082 6193.0, 2021.7945205479452 5828.0, 2021.8082191780823 5633.0, 2021.8219178082193 5479.0, 2021.835616438356 5426.0, 2021.849315068493 5425.0, 2021.86301369863 5554.0, 2021.876712328767 6390.0, 2021.890410958904 6638.0, 2021.9041095890411 6879.0, 2021.9178082191781 6934.0, 2021.9315068493152 7222.0, 2021.945205479452 7267.0, 2021.958904109589 7528.0, 2021.972602739726 7370.0, 2021.986301369863 7179.0'
    text = '2018.035616438356 2983.0, 2018.0849315068492 3342.0, 2018.0986301369862 3106.0, 2018.1041095890412 3160.0, 2018.1178082191782 3011.0, 2018.1178082191782 -9999, 2018.13698630137 2731.0, 2018.145205479452 2857.0, 2018.1616438356164 2782.0, 2018.1671232876713 2572.0, 2018.2054794520548 -9999, 2018.2082191780821 2436.0, 2018.2246575342465 2881.0, 2018.227397260274 -9999, 2018.2493150684932 2825.0, 2018.2493150684932 2890.0, 2018.2630136986302 -9999, 2018.268493150685 3965.0, 2018.268493150685 3975.0, 2018.2904109589042 5382.0, 2018.2931506849316 5290.0, 2018.295890410959 5898.0, 2018.304109589041 -9999, 2018.317808219178 -9999, 2018.323287671233 7505.0, 2018.33698630137 7889.0, 2018.33698630137 8057.0, 2018.345205479452 8228.0, 2018.3506849315067 8488.0, 2018.3643835616438 9036.0, 2018.3780821917808 -9999, 2018.3808219178081 9042.0, 2018.386301369863 9182.0, 2018.3917808219178 -9999, 2018.4 -9999, 2018.4054794520548 9255.0, 2018.4136986301369 -9999, 2018.427397260274 8679.0, 2018.4328767123288 8533.0, 2018.441095890411 8628.0, 2018.495890410959 5672.0, 2018.5013698630137 -9999, 2018.5123287671233 5107.0, 2018.5287671232877 5261.0, 2018.531506849315 6430.0, 2018.5369863013698 6234.0, 2018.5424657534247 6375.0, 2018.5506849315068 -9999, 2018.5561643835617 -9999, 2018.5561643835617 -9999, 2018.5643835616438 6787.0, 2018.5698630136985 7416.0, 2018.5753424657535 7059.0, 2018.5780821917808 7079.0, 2018.5972602739726 7322.0, 2018.6 7888.0, 2018.6109589041096 -9999, 2018.6383561643836 7313.0, 2018.6657534246576 -9999, 2018.6739726027397 -9999, 2018.6794520547944 7208.0, 2018.6876712328767 5541.0, 2018.6876712328767 4451.0, 2018.7150684931507 6746.0, 2018.731506849315 7893.0, 2018.7616438356165 2303.0, 2018.7753424657535 3070.0, 2018.7753424657535 3107.0, 2018.7835616438356 3265.0, 2018.7890410958903 3461.0, 2018.8027397260273 3743.0, 2018.8301369863013 -9999, 2018.8438356164384 -9999, 2018.8794520547945 2259.0, 2018.9068493150685 2873.0, 2018.9068493150685 2686.0, 2018.9260273972602 2832.0, 2018.9260273972602 2874.0'
    text = text.replace(', ', ' ').split(' ')
    data = np.array(text, float).reshape(-1, 2)
    dates = data[:, 0]
    inarray = data[:, 1]

    nodata = -9999
    nproc = 1
    outarray = np.ones(len(bandnames))
    result = forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc)
    print('Done:', result)
'''
