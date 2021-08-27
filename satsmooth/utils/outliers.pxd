# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from . cimport common
from . cimport percentiles


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, unsigned int, unsigned int, double) nogil


cdef inline double _poly_func(double xv, double beta0, double beta1, double alpha) nogil:
    return common.clip(beta0*xv + beta1*common.pow2(xv) + alpha, 0.0, 1.0)


cdef inline double _get_mean(double[:, ::1] data_array,
                             Py_ssize_t row_index,
                             unsigned int col_start,
                             unsigned int col_end,
                             double no_data) nogil:

    cdef:
        Py_ssize_t col_index
        double valid_total
        double ymean = 0.0

    valid_total = 0.0
    for col_index in range(col_start, col_end):

        if (data_array[row_index, col_index] != no_data) and (data_array[row_index, col_index] != -999.0):
            ymean += data_array[row_index, col_index]
            valid_total += 1.0

    return ymean / valid_total


cdef inline bint _is_vertical_center(vector[double] ydatav, Py_ssize_t cidx, unsigned int valid_samples) nogil:

    cdef:
        Py_ssize_t q1g, q2g, q1s, q2s
        bint is_center_greenup = True
        bint is_center_senescence = True

    # Greenup
    for q1g in range(0, cidx):
        if ydatav[q1g] >= ydatav[cidx]:
            is_center_greenup = False
            break

    if is_center_greenup:

        for q2g in range(cidx+1, valid_samples):
            if ydatav[q2g] <= ydatav[cidx]:
                is_center_greenup = False
                break

    # Senescence
    for q1s in range(0, cidx):
        if ydatav[q1s] <= ydatav[cidx]:
            is_center_senescence = False
            break

    if is_center_senescence:

        for q2s in range(cidx+1, valid_samples):
            if ydatav[q2s] >= ydatav[cidx]:
                is_center_senescence = False
                break

    if is_center_greenup and is_center_senescence:
        return True
    else:
        return False


#cdef inline double _get_vector_min(vector[double] vdata, unsigned int nsamples) nogil:
#
#    cdef:
#        Py_ssize_t q
#        double min_value = vdata[0]
#
#    for q in range(1, nsamples):
#        if (vdata[q] < min_value) and (vdata[q] != -999.0):
#            min_value = vdata[q]
#
#    return min_value


cdef inline void _check_deviation(vector[int] indexesv,
                                  vector[double] xdatav,
                                  vector[double] ydatav,
                                  double[:, ::1] yarray_,
                                  unsigned int ii,
                                  unsigned int ncols,
                                  Py_ssize_t start_,
                                  Py_ssize_t end_,
                                  double[:, ::1] mbf,
                                  double dev_thresh,
                                  double no_data,
                                  double baseline,
                                  double low_values,
                                  unsigned int valid_samples,
                                  bint predict_linear,
                                  unsigned int max_days,
                                  double base_perc_thresh,
                                  double min_samples,
                                  long[:, :, ::1] index_positions_fill) nogil:

    """
    Checks each sample's deviation from the window linear fit 
    """

    cdef:
        Py_ssize_t hw, q, hw_offset
        double yhat, dev, w, w1, w2, dmax
        int dhalf
        Py_ssize_t outlier_pos
        unsigned int outlier_count
        double xvalue, yvalue
        double max_dist, scaled_days
        int outlier_index

#    dhalf = <int>(start_ + ((end_ - start_) / 2.0))
#    dmax = common.fabs(<double>(dhalf - start_))

    hw = <int>((valid_samples-0.5) / 2.0)

    if predict_linear:

        outlier_count = 0

        xvalue = xdatav[1]
        yvalue = ydatav[1]

        #if (low_values < yvalue <= baseline) and (low_values < ydatav[0] <= baseline) and (low_values < ydatav[2] <= baseline):
        if (yvalue == no_data) or (yvalue == 2.0):
            outlier_count = 0
        elif (ydatav[0] < low_values) or (ydatav[2] < low_values):
            outlier_count = 0
        else:

            # yhat = common.clip(mbf[ii, 0] * xvalue + mbf[ii, 1], 0.0, 1.0)

            yhat = common.linear_adjustment(<int>xdatav[0], <int>xdatav[1], <int>xdatav[2], ydatav[0], ydatav[2])

#            if (ydatav[0] > yvalue < ydatav[2]) and (yvalue < low_values):
#                outlier_pos = indexesv[1]
#                outlier_count = 1
#            else:

            if (low_values < yvalue <= 0.4) and (common.max_value(ydatav[0], ydatav[2]) >= 0.45) and (ydatav[0] > yvalue < ydatav[2]):
                # Protect crop cycles
                w1 = 1.0 - common.sqrt(common.max_value(ydatav[0], ydatav[2]))
            elif (yvalue >= 0.45) and (common.min_value(ydatav[0], ydatav[2]) <= baseline):
                # Protect high outliers relative to baseline
                w1 = 0.1
            elif ((low_values < yvalue <= baseline) and (low_values <= ydatav[0] <= baseline)) or ((low_values <= yvalue <= baseline) and (low_values <= ydatav[2] <= baseline)):
                # Protect baseline values
                w1 = 0.1
            else:
                # Protect high values
                w1 = 1.0 if yhat >= yvalue else 0.33

            # Weight the distance
            max_dist = common.max_value(common.fabs(xdatav[1] - xdatav[0]), common.fabs(xdatav[1] - xdatav[2]))
            scaled_days = 1.0 - common.scale_min_max(max_dist, 0.1, 0.9, 0.0, max_days)

            w2 = common.scale_min_max(common.logistic_func(scaled_days, 0.5, 10.0), 0.1, 1.0, 0.0, 1.0)

            w = w1 * w2

            # Get the weighted deviation
            dev = common.fabs(common.prop_diff(yvalue, yhat)) * w

            if not common.npy_isinf(dev) and not common.npy_isnan(dev):

                # Remove the outlier
                if dev > dev_thresh:

                    # This check avoids large dropoffs from senescence to valleys
                    if (common.prop_diff(ydatav[0], ydatav[1]) < base_perc_thresh) and (common.prop_diff(ydatav[0], ydatav[2]) < base_perc_thresh):
                        outlier_count = 0

                    # This check avoids large increases from valleys to green-up
                    elif (common.prop_diff(ydatav[2], ydatav[1]) < base_perc_thresh) and (common.prop_diff(ydatav[2], ydatav[0]) < base_perc_thresh):
                        outlier_count = 0
                    else:

                        if ydatav[0] < yvalue > ydatav[2]:
                            # Local spike
                            outlier_pos = indexesv[1]
                            outlier_count = 1
                        elif ydatav[0] > yvalue < ydatav[2]:
                            outlier_pos = indexesv[1]
                            outlier_count = 1

        # Restrict outlier removal to one value
        if outlier_count > 0:
            yarray_[ii, outlier_pos] = yhat

    else:

        dhalf = indexesv[hw]

        # Maximum days between vector center value and the end
        dmax = common.max_value(common.fabs(<double>(dhalf - indexesv[0])),
                                common.fabs(<double>(dhalf - indexesv[valid_samples-1])))

        # Setup offsets to cycle over
        #center_offset = <int>((valid_samples - min_samples) / 2.0)

        #for q in range(-1, 2):

        q = 0
        outlier_count = 0
        hw_offset = hw + q

        yvalue = ydatav[hw_offset]

        if (yvalue != no_data) and (yvalue != 2.0):

            xvalue = xdatav[hw_offset]

            # Sample prediction, clipped to 0-1
            yhat = _poly_func(xvalue, mbf[ii, 0], mbf[ii, 1], mbf[ii, 2])

    #        if low_values < yvalue <= baseline:
    #            pass
            # One of the values is < lower limits AND the center value is not the low value
    #        if (_get_vector_min(ydatav, valid_samples) < low_values) and (yvalue > low_values):
    #            pass
            if yhat < low_values:
                pass
            elif _is_vertical_center(ydatav, hw_offset, valid_samples):
                pass
            else:

                # Give a lower weight for values above prediction
                w1 = 1.5 if yhat >= yvalue else 0.5

                # Give a lower weight for values at prediction ends
                #w2 = 1.0 - common.scale_min_max(common.fabs(<double>dhalf - <double>indexesv[hw_offset]), 0.1, 1.0, 0.0, dmax)
                w2 = 1.0 - yvalue

                # Combine the weights
                w = w1 * w2

                # Get the weighted deviation
                dev = common.fabs(common.prop_diff(yvalue, yhat)) * w

                if not common.npy_isinf(dev) and not common.npy_isnan(dev):

                    # Remove the outlier
                    if dev > dev_thresh:

                        # This check avoids large dropoffs from senescence to valleys
                        if (common.prop_diff(ydatav[0], ydatav[hw_offset]) < base_perc_thresh) and (common.prop_diff(ydatav[0], ydatav[hw_offset+1]) < base_perc_thresh):
                            outlier_count = 0

                        # This check avoids large increases from valleys to green-up
                        elif (common.prop_diff(ydatav[valid_samples-1], ydatav[hw_offset]) < base_perc_thresh) and (common.prop_diff(ydatav[valid_samples-1], ydatav[hw_offset-1]) < base_perc_thresh):
                            outlier_count = 0
                        else:

                            outlier_pos = indexesv[hw_offset]
                            outlier_count = 1

            # Restrict outlier removal to one value
            if outlier_count > 0:

                outlier_index = index_positions_fill[0, ii, ncols]

                # The last column position is the count of outliers
                # The second to last column is the fill value
                if outlier_index == 0:

                    index_positions_fill[0, ii, outlier_index] = outlier_pos
                    index_positions_fill[1, ii, outlier_index] = <int>(yhat * 10000.0)
                    index_positions_fill[0, ii, ncols] = outlier_index + 1

                else:

                    if index_positions_fill[0, ii, outlier_index-1] != outlier_pos:

                        index_positions_fill[0, ii, outlier_index] = outlier_pos
                        index_positions_fill[1, ii, outlier_index] = <int>(yhat * 10000.0)
                        index_positions_fill[0, ii, ncols] = outlier_index + 1

                #yarray_[ii, outlier_pos] = 2.0


cdef inline void remove_outliers_linear(double[::1] xsparse,
                                        double[:, ::1] array_,
                                        double[:, ::1] mbz,
                                        unsigned int ii,
                                        unsigned int ncols,
                                        unsigned int min_samples,
                                        unsigned int max_days,
                                        double dev_thresh,
                                        double no_data,
                                        long[:, :, ::1] index_positions_) nogil:

    """
    Removes outliers by checking deviation from a local linear regression window
    
    Equation:
        y = b*X + a
    """

    cdef:
        Py_ssize_t start, middle1, middle2, end1, end2, end3, nct, jidx
        unsigned int valid_samples1, valid_samples2, valid_samples3
        double baseline, low_values

        double alpha, beta
        double x_dev, y_dev
        double x_var, xy_cov
        double x_mean

        vector[int] indexes
        vector[double] xdata, ydata

        double max_dist1, max_dist2
        double base_perc_thresh

        bint increase_valid

    # Set the lower baseline as the 10th percentile
    baseline = 0.15

    # Set low values as the 2nd percentile
    low_values = 0.03

    base_perc_thresh = -0.6

    for start in range(0, ncols-min_samples):

        ######################################
        # Check for large gaps with low values
        ######################################

        valid_samples1 = 0
        middle1 = 0
        max_dist1 = 0.0

        for end1 in range(start, ncols - 4):

            if (array_[ii, end1] != no_data) and (array_[ii, end1] != 2.0):

                increase_valid = False

                if valid_samples1 == 0:

                    if array_[ii, end1] >= baseline + 0.1:
                        max_dist1 = common.max_value(xsparse[end1] - xsparse[start], max_dist1)
                        middle1 = end1
                        increase_valid = True

                if valid_samples1 == 1:

                    if array_[ii, end1] <= baseline:
                        max_dist1 = common.max_value(xsparse[end1] - xsparse[middle1], max_dist1)
                        middle1 = end1
                        increase_valid = True
                    else:
                        valid_samples1 = 0

                if valid_samples1 == 2:

                    if array_[ii, end1] <= baseline:
                        max_dist1 = common.max_value(xsparse[end1] - xsparse[middle1], max_dist1)
                        increase_valid = True
                        middle1 = end1
                    else:
                        valid_samples1 = 0

                if valid_samples1 == 3:

                    max_dist1 = common.max_value(xsparse[end1] - xsparse[middle1], max_dist1)

                    if max_dist1 > max_days:

                        if array_[ii, end1] >= baseline + 0.1:
                            array_[ii, middle1] = no_data

                    break

                if increase_valid:
                    valid_samples1 += 1

        ##############################################
        # Check for large gaps with a single low value
        ##############################################

        valid_samples3 = 0
        middle2 = start
        max_dist2 = 0.0

        for end3 in range(start, ncols - 3):

            if (array_[ii, end3] != no_data) and (array_[ii, end3] != 2.0):

                increase_valid = False

                if valid_samples3 == 0:

                    if array_[ii, end3] >= 0.35:
                        max_dist2 = common.max_value(xsparse[end3] - xsparse[start], max_dist2)
                        middle2 = end3
                        increase_valid = True

                if valid_samples3 == 1:

                    if array_[ii, end3] <= baseline:
                        max_dist2 = common.max_value(xsparse[end3] - xsparse[middle2], max_dist2)
                        increase_valid = True
                        middle2 = end3
                    else:
                        valid_samples3 = 0

                if valid_samples3 == 2:

                    max_dist2 = common.max_value(xsparse[end3] - xsparse[middle2], max_dist2)

                    if max_dist2 > max_days:

                        if array_[ii, end3] >= 0.35:
                            array_[ii, middle2] = no_data

                    break

                if increase_valid:
                    valid_samples3 += 1

        ####################
        # Check for outliers
        ####################

        valid_samples2 = 0

        x_mean = 0.0
        x_var = 0.0
        xy_cov = 0.0

        end2 = start

        while True:

            if end2 + 2 >= ncols:
                break

            if xsparse[end2] - xsparse[start] > max_days:
                break

            if valid_samples2 >= min_samples:
                break

            if (array_[ii, end2] != no_data) and (array_[ii, end2] != 2.0):

                x_mean += xsparse[end2]

                valid_samples2 += 1

            end2 += 1

        if valid_samples2 >= min_samples:

#            x_mean /= <double>valid_samples2

            # Use the median instead of the mean
            #   to avoid fitting to potential outliers
#            y_med = _get_mean(array_, ii, start, end2, no_data)
            #y_med = percentiles.get_perc(array_, ii, start, end2, no_data, 50.0)

            for jidx in range(start, end2):

                if (array_[ii, jidx] != no_data) and (array_[ii, jidx] != 2.0):

#                    x_dev = xsparse[jidx] - x_mean
#                    y_dev = array_[ii, jidx] - y_med
#
#                    x_var += common.pow2(x_dev)
#                    xy_cov += x_dev * y_dev

                    indexes.push_back(jidx)
                    xdata.push_back(xsparse[jidx])
                    ydata.push_back(array_[ii, jidx])

            # least squares, 1 variable
            #   beta = slope = sum(x * y) / sum(x^2)
            #   alpha = intercept = y_mu - beta * x_mu
#            beta = xy_cov / x_var
#            alpha = y_med - beta * x_mean
#
#            mbz[ii, 0] = beta
#            mbz[ii, 1] = alpha

            # Check if values deviate from the prediction
            _check_deviation(indexes,
                             xdata,
                             ydata,
                             array_,
                             ii,
                             ncols,
                             start,
                             end2,
                             mbz,
                             dev_thresh,
                             no_data,
                             baseline,
                             low_values,
                             valid_samples2,
                             True,
                             max_days,
                             base_perc_thresh,
                             min_samples,
                             index_positions_)

            indexes.clear()
            xdata.clear()
            ydata.clear()

            # if start + 2 <= ncols:
            #
            #     # Large gap
            #     if (xsparse[start+1] - xsparse[start] > max_days) and (xsparse[start+2] - xsparse[start+1] > max_days):
            #         if (array_[ii, start] >= 0.35) and (array_[ii, start+2] >= 0.35) and (array_[ii, start+1] <= baseline):
            #             array_[ii, start+1] = no_data


cdef inline void remove_outliers_polynomial(double[::1] xsparse,
                                            double[:, ::1] array_,
                                            double[:, ::1] mbz,
                                            unsigned int ii,
                                            unsigned int ncols,
                                            unsigned int min_samples,
                                            unsigned int max_days,
                                            double dev_thresh,
                                            double no_data,
                                            long[:, :, ::1] index_positions_) nogil:

    """
    Removes outliers by checking deviation from a local polynomial regression window
    
    Equation:
        y = b1*X + b2*X^2 + a
        
    Reference:
        http://faculty.cas.usf.edu/mbrannick/regression/Part3/Reg2.html    
    """

    cdef:
        Py_ssize_t start, end, nct, jidx, f
        unsigned int valid_samples
        double baseline, low_values

        double alpha, beta1, beta2
        double x1_dev, x2_dev
        double y_dev
        double x1_var, x1y_cov, x2_var, x2y_cov
        double x1_mean, x2_mean
        double xx_cov
        double y_mn
        double numerator
        double base_perc_thresh

        vector[int] indexes
        vector[double] xdata, ydata

        metric_ptr ptr_func

    ptr_func = &_get_mean

    # Set the lower baseline as the 10th percentile
    baseline = 0.2

    # Set low values as the 2nd percentile
    low_values = 0.03

    base_perc_thresh = -0.6

    for start in range(0, ncols-min_samples+1):

        valid_samples = 0

        x1_mean = 0.0
        x1_var = 0.0
        x1y_cov = 0.0

        x2_mean = 0.0
        x2_var = 0.0
        x2y_cov = 0.0

        xx_cov = 0.0

        end = start

        if (array_[ii, start] != no_data) and (array_[ii, start] < low_values):
            array_[ii, start] = low_values

        while True:

            if xsparse[end] - xsparse[start] > max_days:
                break

            if (array_[ii, end] != no_data) and (array_[ii, end] != 2.0):

                x1_mean += xsparse[end]
                x2_mean += common.pow2(xsparse[end])

                valid_samples += 1

            if end + 1 >= ncols:
                break

            end += 1

        end += 1

        if valid_samples >= min_samples:

            x1_mean /= <double>valid_samples
            x2_mean /= <double>valid_samples

            # Use the median instead of the mean
            #   to avoid fitting to potential outliers
            y_mn = ptr_func(array_, ii, start, end, no_data)
            #y_mn = percentiles.get_perc(array_, ii, start, end, no_data, 50.0)

            for jidx in range(start, end):

                if (array_[ii, jidx] != no_data) and (array_[ii, jidx] != 2.0):

                    x1_dev = xsparse[jidx] - x1_mean
                    x2_dev = common.pow2(xsparse[jidx]) - x2_mean
                    y_dev = array_[ii, jidx] - y_mn

                    x1_var += common.pow2(x1_dev)
                    x1y_cov += (x1_dev * y_dev)

                    x2_var += common.pow2(x2_dev)
                    x2y_cov += (x2_dev * y_dev)

                    xx_cov += (x1_dev * x2_dev)

                    indexes.push_back(jidx)
                    xdata.push_back(xsparse[jidx])
                    ydata.push_back(array_[ii, jidx])

            numerator = (x1_var * x2_var) - common.pow2(xx_cov)

            # least squares, 2 variables
            beta1 = ((x2_var * x1y_cov) - (xx_cov * x2y_cov)) / numerator
            beta2 = ((x1_var * x2y_cov) - (xx_cov * x1y_cov)) / numerator

            alpha = y_mn - (beta1 * x1_mean) - (beta2 * x2_mean)

            mbz[ii, 0] = beta1
            mbz[ii, 1] = beta2
            mbz[ii, 2] = alpha

            # Check if values deviate from the prediction
            _check_deviation(indexes,
                             xdata,
                             ydata,
                             array_,
                             ii,
                             ncols,
                             start,
                             end,
                             mbz,
                             dev_thresh,
                             no_data,
                             baseline,
                             low_values,
                             valid_samples,
                             False,
                             max_days,
                             base_perc_thresh,
                             min_samples,
                             index_positions_)

            indexes.clear()
            xdata.clear()
            ydata.clear()

    # Fill the outliers
    for f in range(0, index_positions_[0, ii, ncols]):
        array_[ii, index_positions_[0, ii, f]] = <double>index_positions_[1, ii, f] * 0.0001

    # Reset
    array_[ii, ncols-1] = 0
