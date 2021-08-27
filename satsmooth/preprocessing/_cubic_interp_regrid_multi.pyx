# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

from cython.parallel import prange
from cython.parallel import parallel

import numpy as np
cimport numpy as np

from ..pylu cimport dgesv
from ..utils cimport common
from ..utils cimport outliers
from ..utils cimport percentiles
from ..utils cimport harmonics

from libcpp.vector cimport vector

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, Py_ssize_t, Py_ssize_t) nogil


cdef double _get_min(double[:, ::1] in_row,
                     Py_ssize_t a,
                     unsigned int start,
                     unsigned int end) nogil:

    cdef:
        Py_ssize_t b
        double m = in_row[a, start]

    for b in range(start+1, end):

        if in_row[a, b] < m:
            m = in_row[a, b]

    return m


cdef inline double _get_max(double[:, ::1] in_row,
                            Py_ssize_t a,
                            unsigned int start,
                            unsigned int end) nogil:

    """
    Gets the maximum value in a 1-d array
    
    Args:
        in_row (2d array): The input array.
        a (int): The row position.
        start (int): The starting column position.
        end (int): The ending column position.
        
    Returns:
        The array maximum value (double)
    """

    cdef:
        Py_ssize_t b
        double m = in_row[a, start]

    for b in range(start+1, end):

        if in_row[a, b] > m:
            m = in_row[a, b]

    return m


cdef inline int _get_x1(unsigned int x2,
                        double[:, ::1] in_row,
                        Py_ssize_t ii,
                        double no_data) nogil:

    cdef:
        Py_ssize_t x1_iter
        int x1 = -9999

    for x1_iter in range(1, x2+1):

        if in_row[ii, x2-x1_iter] != no_data:
            x1 = x2 - x1_iter

            break

    return x1


cdef inline int _get_x3(unsigned int x2,
                        double[:, ::1] in_row,
                        Py_ssize_t ii,
                        unsigned int dims,
                        double no_data) nogil:

    cdef:
        Py_ssize_t x3_iter
        int x3 = -9999
        int x3_range = dims - x2

    for x3_iter in range(1, x3_range):

        if in_row[ii, x2+x3_iter] != no_data:
            x3 = x2 + x3_iter

            break

    return x3


cdef inline void _find_indices(double[:, ::1] in_row,
                                Py_ssize_t ii,
                                unsigned long[::1] index_positions_,
                                unsigned int dims,
                                double no_data,
                                double[::1] xsparse_,
                                unsigned long[:, ::1] gap_length_array__,
                                bint record_gap_lengths) nogil:

    cdef:
        Py_ssize_t fi, fj
        Py_ssize_t counter = 0
        unsigned int gap_length

    for fi in range(0, dims):

        if in_row[ii, fi] == no_data:

            index_positions_[counter] = fi

            counter += 1

        # Store the gap lengths
        if record_gap_lengths and (fi+1 < dims):

            if in_row[ii, fi] != no_data:

                for fj in range(fi+1, dims):

                    if in_row[ii, fj] != no_data:
                        break

                gap_length = <int>(xsparse_[fj] - xsparse_[fi])

                gap_length_array__[ii, fi] = gap_length

    # Store the count in the last index position
    index_positions_[dims] = counter


cdef inline void _fill_ends(double[:, ::1] array_to_fill_,
                            Py_ssize_t rowidx,
                            unsigned int ncol,
                            unsigned int whalf) nogil:

    cdef:
        Py_ssize_t m
        double fill_value_start = array_to_fill_[rowidx, whalf]
        double fill_value_end = array_to_fill_[rowidx, ncol-whalf-1]

    for m in range(0, whalf):

        array_to_fill_[rowidx, m] = fill_value_start
        array_to_fill_[rowidx, m+ncol-whalf] = fill_value_end


cdef inline void _fill_no_data_ends(double[:, ::1] in_row_,
                                    Py_ssize_t ii,
                                    unsigned int dims,
                                    double no_data) nogil:

    """
    Fills 1-d array endpoints
    
    Args:
        in_row_ (2d array)
        ii (int)
        dims (int)
        no_data (double)
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t ib, ic
        double valid_data

    if in_row_[ii, 0] == no_data:

        # Find the first valid data point
        for ib in range(1, dims):

            if in_row_[ii, ib] != no_data:
                valid_data = in_row_[ii, ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(0, ib):
            in_row_[ii, ic] = valid_data

    if in_row_[ii, dims-1] == no_data:

        # Find the first non-zero data
        for ib in range(dims-2, 0, -1):

            if in_row_[ii, ib] != no_data:
                valid_data = in_row_[ii, ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(ib+1, dims):
            in_row_[ii, ic] = valid_data


cdef inline void _fill_no_data(double[:, ::1] in_data_,
                               Py_ssize_t ii,
                               unsigned long[::1] index_positions,
                               unsigned int dims,
                               double no_data,
                               bint check_ifdata,
                               bint interp_step,
                               double[::1] xsparse_,
                               unsigned long[:, ::1] gap_length_array_,
                               bint record_gap_lengths) nogil:

    """
    Fills 'no data' values by linear interpolation between valid data
    
    Args:
        in_data_ (2d array)
        ii (int)
        index_positions (1d array)
        dims (int)
        no_data (double)
        check_ifdata (bool)
        interp_step (bool)
        
    Returns:
        Filled array (1d array)
    """

    cdef:
        Py_ssize_t x2_idx
        unsigned int len_idx
        int x1, x2, x3
        double interp_value
        double sample_min, sample_max, high_diff, low_diff

    _find_indices(in_data_,
                  ii,
                  index_positions,
                  dims,
                  no_data,
                  xsparse_,
                  gap_length_array_,
                  record_gap_lengths)

    len_idx = <int>(index_positions[dims])

    if len_idx != 0:

        if check_ifdata:

            if _get_max(in_data_, ii, 0, dims) > 0:

                if interp_step:

                    sample_min = percentiles.get_perc(in_data_, ii, 0, dims-1, no_data, 5.0)
                    sample_max = percentiles.get_perc(in_data_, ii, 0, dims-1, no_data, 95.0)

                for x2_idx in range(0, len_idx):

                    x2 = <int>(index_positions[x2_idx])

                    if in_data_[ii, x2] == no_data:

                        # get x1
                        x1 = _get_x1(x2, in_data_, ii, no_data)

                        # get x3
                        x3 = _get_x3(x2, in_data_, ii, dims, no_data)

                        if (x1 != -9999) and (x3 != -9999):

                            interp_value = common.linear_adjustment(x1, x2, x3, in_data_[ii, x1], in_data_[ii, x3])

                            if interp_step:

                                if in_data_[ii, x1] > in_data_[ii, x2]:

                                    # Decreasing slope
                                    high_diff = common.fabs(sample_max - in_data_[ii, x1])
                                    low_diff = common.fabs(in_data_[ii, x3] - sample_min)

                                    if high_diff < low_diff:
                                        # Step up
                                        interp_value = interp_value + ((in_data_[ii, x1] - interp_value) * 0.33)
                                    else:
                                        # Step down
                                        interp_value = interp_value - ((in_data_[ii, x1] - interp_value) * 0.33)

                                else:

                                    # Increasing slope
                                    high_diff = common.fabs(sample_max - in_data_[ii, x3])
                                    low_diff = common.fabs(in_data_[ii, x1] - sample_min)

                                    if high_diff < low_diff:
                                        # Step up
                                        interp_value = interp_value + ((interp_value - in_data_[ii, x1]) * 0.33)
                                    else:
                                        # Step down
                                        interp_value = interp_value - ((interp_value - in_data_[ii, x1]) * 0.33)

                            in_data_[ii, x2] = interp_value

        else:

            for x2_idx in range(0, len_idx):

                x2 = <int>(index_positions[x2_idx])

                if in_data_[ii, x2] == no_data:

                    # get x1
                    x1 = _get_x1(x2, in_data_, ii, no_data)

                    # get x3
                    x3 = _get_x3(x2, in_data_, ii, dims, no_data)

                    if (x1 != -9999) and (x3 != -9999):

                        interp_value = common.linear_adjustment(x1, x2, x3, in_data_[ii, x1], in_data_[ii, x3])

                        if interp_step:

                            if in_data_[ii, x1] > in_data_[ii, x2]:
                                interp_value = interp_value + ((in_data_[ii, x1] - interp_value) * 0.33)
                            else:
                                interp_value = interp_value - ((interp_value - in_data_[ii, x1]) * 0.33)

                        in_data_[ii, x2] = interp_value

        _fill_no_data_ends(in_data_, ii, dims, no_data)


# cdef double _get_array_mean(double[:, ::1] y_array_slice,
#                             Py_ssize_t iii,
#                             Py_ssize_t jjj,
#                             Py_ssize_t t,
#                             int ignore_j=-999) nogil:
#
#     """
#     Calculates the gaussian-weighted mean of a 1-d array
#
#     Args:
#         y_array_slice (2d array): The array.
#         iii (int): The row index position.
#         jjj (int): The starting column index position.
#         t (int): The ending column index position.
#
#     Returns:
#         Mean (double)
#     """
#
#     cdef:
#         Py_ssize_t jc
#         double w
#         double array_sum = 0.0
#         double weights_sum = 0.0
#
#     for jc in range(jjj, t):
#
#         if ignore_j != -999:
#             if jc == ignore_j:
#                 continue
#
#         if y_array_slice[iii, jc] > 0:
#
#             w = common.gaussian_func(common.scale_min_max(float(jc), -1.0, 1.0, 0.0, float(t)), 0.5)
#             array_sum += y_array_slice[iii, jc] * w
#             weights_sum += w
#
#     return array_sum / weights_sum


cdef double _get_array_std(double[:, ::1] y_array_slice,
                           Py_ssize_t iii,
                           Py_ssize_t jjj,
                           Py_ssize_t t) nogil:

    """
    Calculates the standard deviation of a 1-d array
    
    Args:
        y_array_slice (2d array): The array.
        iii (int): The row index position.
        jjj (int): The starting column index position.
        t (int): The ending column index position.
        
    Returns:
        Standard deviation (double)
    """

    cdef:
        Py_ssize_t jca, jcb
        double array_sum = 0.0
        double yvalue_, array_mean
        Py_ssize_t array_count = 0
        double sq_dev, sqn
        double sum_sq = 0.0

    for jca in range(jjj, t):

        yvalue_ = y_array_slice[iii, jca]

        if yvalue_ > 0:

            array_sum += yvalue_
            array_count += 1

    array_mean = array_sum / <double>array_count

    for jcb in range(jjj, t):

        yvalue_ = y_array_slice[iii, jcb]

        if yvalue_ > 0:

            sq_dev = common.squared_diff(yvalue_, array_mean)
            sum_sq += sq_dev

    # sqn = <double>array_count - 1.0

    return common.sqrt(sum_sq / <double>array_count)


# cdef double _get_array_cv(double[:, ::1] y_array_slice,
#                          Py_ssize_t iii,
#                          Py_ssize_t jjj,
#                          Py_ssize_t t) nogil:
#
#     """
#     Calculates the gaussian-weighted CV of a 1-d array
#
#     Args:
#         y_array_slice (2d array): The array.
#         iii (int): The row index position.
#         jjj (int): The starting column index position.
#         t (int): The ending column index position.
#
#     Returns:
#         Coefficient of variation (double)
#     """
#
#     cdef:
#         Py_ssize_t jc
#         double array_sum = 0.0
#         Py_ssize_t array_count = 0
#         double sq_dev, sqn, array_std_dev, cv
#         double sum_sq = 0.0
#
#     for jc in range(jjj, t):
#
#         if y_array_slice[iii, jc] > 0:
#
#             array_sum += y_array_slice[iii, jc]
#             array_count += 1
#
#     array_mean = array_sum / float(array_count)
#
#     if array_mean < 0.3:
#         return array_mean
#
#     for jc in range(jjj, t):
#
#         if y_array_slice[iii, jc] > 0:
#
#             sq_dev = common.squared_diff(y_array_slice[iii, jc], array_mean)
#             sum_sq += sq_dev
#
#     sqn = float(array_count) - 1.0
#
#     # Standard deviation
#     array_std_dev = common.sqrt(sum_sq / sqn)
#
#     # Coefficient of variation
#     cv = array_std_dev / array_mean
#
#     # Cap the CV
#     if cv > 0.3:
#         cv = 0.3
#
#     # Return the CV, scaled to 0-1
#     return common.scale_min_max(cv, 0.0, 1.0, 0.0, 0.3)


cdef inline int _check_data(double[:, ::1] array,
                            Py_ssize_t i,
                            unsigned int start,
                            unsigned int end,
                            double no_data_value,
                            unsigned int min_data_required) nogil:

    """
    Checks if there are sufficient data points in the time series
    
    Args:
        array (2d array)
        i (int)
        start (int)
        end (int)
        no_data_value (double)
        min_data_required (int)
        
    Returns:
        (int)
    """

    cdef:
        Py_ssize_t ci
        int data_count_ = 0

    for ci in range(start, end):

        if array[i, ci] != no_data_value:
            data_count_ += 1

        if data_count_ >= min_data_required:
            break

    return data_count_


cdef inline void _replace_lower_envelope(double[:, ::1] y_array_sm,
                                         double[:, ::1] out_array_sm,
                                         Py_ssize_t ii,
                                         unsigned int col_count,
                                         Py_ssize_t smj,
                                         unsigned int t,
                                         unsigned int t_half) nogil:

    """
    Replaces values with the lower envelope
    """

    cdef:
        # Py_ssize_t smj
        double yval, smval

    # for smj in range(0, col_count-t-1):

    # Original value
    yval = y_array_sm[ii, smj+t_half]

    # Smoothed value
    smval = out_array_sm[ii, smj+t_half]

    if yval > smval:
        y_array_sm[ii, smj+t_half] = (smval + yval*0.5) / 1.5
    else:
        y_array_sm[ii, smj+t_half] = (smval*0.5 + yval) / 1.5


cdef inline void _replace_upper_envelope(double[:, ::1] y_array_sm,
                                         double[:, ::1] out_array_sm,
                                         Py_ssize_t ii,
                                         unsigned int col_count,
                                         Py_ssize_t smj,
                                         unsigned int t,
                                         unsigned int t_half,
                                         unsigned int t_diff,
                                         unsigned int t_adjust) nogil:

    """
    Replaces values with the upper envelope
    """

    cdef:
        double yval, smval, smvalb, smvalf, smvaly, smvalz
        double w1, w2, w

    # Original value
    yval = y_array_sm[ii, smj+t_half]

    # Smoothed value
    smval = out_array_sm[ii, smj+t_half]

    # Original window ends
    smvalb = out_array_sm[ii, smj]
    smvalf = out_array_sm[ii, smj+t-1]

    # Adjusted window ends
    smvaly = out_array_sm[ii, smj+t_diff]
    smvalz = out_array_sm[ii, smj+t_diff+t_adjust-1]

    if (smvalb > smval < smvalf) or (smvaly > smval < smvalz):

        w1 = 0.25
        w2 = 1.25

    elif (smvalb < smval > smvalf) or (smvaly < smval > smvalz):

        w1 = 0.25
        w2 = 1.25

    else:

        w1 = 0.5
        w2 = 1.0

    w = w1 + w2

    y_array_sm[ii, smj+t_half] = (smval*w1 + yval*w2) / w


cdef inline void _dts(double[:, ::1] y_array,
                      double[:, ::1] out_array_,
                      double[:, ::1] out_array_harmonics_,
                      Py_ssize_t ii,
                      unsigned int n_cols,
                      unsigned int t,
                      unsigned int t_half,
                      unsigned int min_window,
                      double mid_g,
                      double r_g,
                      double mid_k,
                      double r_k,
                      double mid_t,
                      double r_t,
                      double [::1] color_weights,
                      unsigned int n_iters,
                      unsigned int envelope,
                      unsigned long[:, ::1] gap_length_array_dense_) nogil:

    """
    Smooths the data with a dynamic temporal smoother
    
    Args:
        y_array (2d array): The input array.
        out_array_ (2d array): The output array.
        ii (int): The row position.
        n_cols (int): The number of columns.
        t (int): The temporal window size.
        t_half (int): The temporal window half-size.
        mid_g (double): The temporal alpha parameter.
        r_g (double): The temporal beta parameter.
        mid_k (double): The spatial alpha parameter.
        r_k (double): The spatial beta parameter.
        mid_t (double): The tail dampening beta parameter.
        r_t (double): The tail dampening beta parameter.
        color_weights (1d array): The color gaussian weights.
        n_iters (int): The number of smoothing iterations.
        envelope (int): The envelope to adjust to (if `n_iters` > 1).
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t j, zz, ni, jh
        double weighted_sum, weights_sum, base_adjust, adjusted_value
        double slice_std, bc, vc, tw, gb, gi, w, vc_scaled, r_tadj
        unsigned int t_adjust, t_diff, t_diff_half
        metric_ptr ptr_func
        double sample_min, sample_std, bcw, bcw_mu
        double max_win_diff
        double hmin, hmax, dmin, dmax, hscale
        double harmonic_weight

    ptr_func = &_get_array_std

    sample_min = percentiles.get_perc(y_array, ii, 0, n_cols, 0.0, 5.0)

    # Fit Fourier harmonics
    # Pre-fill the output array with harmonics
    harmonics.fourier(y_array, out_array_harmonics_, ii, n_cols)

    for ni in range(0, n_iters):

        # Get the time series minimum
        # sample_min = _get_min(y_array, ii, t_half, n_cols-t_half-1)

        # sample_min = percentiles.get_perc(y_array, ii, 0, n_cols, 0.0, 5.0)
        # sample_quarter = percentiles.get_perc(y_array, ii, 0, n_cols-1, 0.0, 40.0)

        # Iterate over the array.
        for j in range(0, n_cols-t+1):

            # Get the current center value
            bc = y_array[ii, j+t_half]

            # Get the window standard deviation
            sample_std = common.scale_min_max(common.clip_high(ptr_func(y_array, ii, j, j+t), 0.05), 0.0, 1.0, 0.0, 0.05)

            # Get the percentage difference between the baseline and the center value
            bcw = common.clip_high(common.fabs(common.perc_diff(sample_min, bc)) * 0.001, 1.0)

            # Weight the standard deviation higher
            #   than the signal value.
            # bcw_mu = (sample_std + bcw) * 0.5
            bcw_mu = (sample_std + bcw) * 0.5

            # Get the std of the current time slice.
            # slice_std = _func(y_array, ii, j, j+t)

            # Adjust the window size
            t_adjust = <int>(common.scale_min_max(common.logistic_func(bcw_mu, mid_k, r_k),
                                                  min_window, t, 0.0, 1.0))

            # Adjust the gaussian sigma based on the difference to the signal baseline and the standard deviation
            # Low values will have a higher adjusted alpha
            base_adjust = common.logistic_func(bcw_mu, mid_g, r_g)

            # Enforce odd-sized window
            t_adjust = common.adjust_window_up(t_adjust)

            # Get the difference between window sizes
            if t == t_adjust:
                t_diff = 0
                t_diff_half = t_half
            else:

                # Half of the difference in the user window and adjusted window
                t_diff = common.window_diff(t, t_adjust)

                # Enforce odd-sized window
                t_diff = common.clip_low_int(common.adjust_window_down(t_diff), 0)

                t_diff_half = <int>(float(t_adjust) * 0.5) + t_diff

            weighted_sum = 0.0
            weights_sum = 0.0

            max_win_diff = common.fabs(<double>t_diff_half - <double>t_diff)

            # Iterate over the window.
            for zz in range(t_diff, t_diff+t_adjust):

                # Safety check
                if j+zz >= n_cols:
                    break

                # Get the current window value.
                vc = common.clip(y_array[ii, j+zz], 0.0, 1.0)

                if vc > 0:

                    # Lower values are closer to the window center.
                    #
                    # Values close to 0 will be given:
                    #   ... a low sigmoid weight for window heads.
                    #   ... a high sigmoid weight for window tails.
                    #   ... a high gaussian weight for window sigma_adjust centers.
                    # vc_scaled = <double>(t_diff_half - zz)
                    # x -1 --> flips the window so that first-half windows values are (-) and second-half values are (+)
                    vc_scaled = common.clip(-1.0 * common.scale_min_max(<double>t_diff_half - <double>zz, -1.0, 1.0, -1.0 * max_win_diff, max_win_diff), -1.0, 1.0)

                    # `bcw` should be low for baseline values

                    if common.fabs(common.perc_diff(y_array[ii, j+t_diff], y_array[ii, j+t_diff+t_adjust-1])) >= 20:

                        if y_array[ii, j+t_diff+t_adjust-1] > y_array[ii, j+t_diff]:
                            vc_scaled = common.scale_min_max(-1.0 * vc_scaled, 0.1, 1.0, -1.0, 1.0)
                            r_tadj = -r_t
                        else:
                            vc_scaled = common.scale_min_max(vc_scaled, 0.1, 1.0, -1.0, 1.0)
                            r_tadj = r_t

                        tw = common.scale_min_max(common.logistic_func(vc_scaled, mid_t, r_tadj), 0.1, 1.0, 0.0, 1.0)

                    else:
                        tw = common.scale_min_max(common.gaussian_func(common.fabs(vc_scaled), base_adjust), 0.1, 1.0, 0.0, 1.0)

                    # Reduce the weight of lower values.
                    #
                    # Values less than the window's center value
                    #   will have a weight of 0.75, otherwise 1.0.
                    gb = common.bright_weight(bc, vc)

                    # Get the color distance
                    #
                    # Small color differences will have
                    #   a large gaussian weight.
                    gi = color_weights[common.abs_round(vc, bc)]

                    # Sum the weights
                    w = common.set_weight(tw, gb, gi)

                    # TODO: gap length threshold
                    if gap_length_array_dense_[ii, j+zz] >= 90:
                        harmonic_weight = 10.0
                    elif 60 <+ gap_length_array_dense_[ii, j+zz] < 90:
                        harmonic_weight = 10.0
                    elif 30 <+ gap_length_array_dense_[ii, j+zz] < 60:
                        harmonic_weight = 0.5
                    else:
                        harmonic_weight = 0.1

                    #harmonic_weight = 0.5  #common.clip(common.scale_min_max(out_array_harmonics_[ii, j+zz], 0.01, 0.5, 0.03, 0.8), 0.01, 0.5)
                    weighted_sum += (vc*w + out_array_harmonics_[ii, j+zz]*harmonic_weight)
                    weights_sum += (w+harmonic_weight)

            if weighted_sum > 0:

                adjusted_value = weighted_sum / weights_sum
                out_array_[ii, j+t_half] = adjusted_value

            if n_iters > 1:

                if envelope == 1:

                    # Replace ``y_array`` with the upper envelope of ``out_array_``
                    _replace_upper_envelope(y_array, out_array_, ii, n_cols, j, t, t_half, t_diff, t_adjust)

                else:

                    # Replace ``y_array`` with the lower envelope of ``out_array_``
                    _replace_lower_envelope(y_array, out_array_, ii, n_cols, j, t, t_half)

        _fill_ends(out_array_, ii, n_cols, t_half)

    #! For testing purposes only
#    for j in range(0, n_cols):
#        out_array_[ii, j] = out_array_harmonics_[ii, j]


cdef inline void _set_indexed_output(double[:, ::1] out_array_dense_view_,
                                     double[:, ::1] out_array_indexed_view_,
                                     Py_ssize_t iidx,
                                     unsigned long[::1] indices,
                                     unsigned int n_indices) nogil:

    cdef:
        Py_ssize_t jidx

    for jidx in range(0, n_indices):
        out_array_indexed_view_[iidx, jidx] = common.clip(out_array_dense_view_[iidx, indices[jidx]], 0.0, 1.0)


cdef inline double _lower_bound(unsigned int begin, unsigned int end, double[::1] x, double val):

    """
    Finds the lower bound, equivalent to C `lower_bound`
    
    Args:
        begin (int)
        end (int)
        x (1d array)
        val (double)
        
    Returns:
        The lower bound (double)
    """

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] >= val:
            break

    return <double>i


cdef inline double _upper_bound(unsigned int begin, unsigned int end, double[::1] x, double val):

    """
    Finds the upper bound, equivalent to C `upper_bound`
    
    Args:
        begin (int)
        end (int)
        x (1d array)
        val (double)
    
    Returns:
        The upper bound (double)
    """

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] > val:
            break

    return <double>i


cdef inline void _replace_array(double[:, ::1] y_array,
                                double[:, ::1] out_array_,
                                Py_ssize_t ii,
                                unsigned int cols) nogil:

    cdef:
        Py_ssize_t j

    for j in range(0, cols):
        out_array_[ii, j] = y_array[ii, j]


cdef inline void _regrid(double[:, ::1] data_array,
                         double[:, ::1] out_array_view_,
                         Py_ssize_t ii,
                         unsigned int n,
                         vector[double] avect,
                         vector[double] bvect,
                         vector[double] lower,
                         vector[double] upper,
                         double nodata,
                         long[::1] any_nans_,
                         unsigned int ncols,
                         double[::1] xsparse_,
                         unsigned long[:, ::1] gap_length_array_,
                         unsigned long[:, ::1] gap_length_array_dense_) nogil:

    cdef:
        Py_ssize_t k, p
        double interp_value

    # Interpolation to a new grid
    for k in range(0, n):

        interp_value = data_array[ii, <int>lower[k]] * avect[k] + data_array[ii, <int>upper[k]] * bvect[k]

        if common.npy_isnan(interp_value) or common.npy_isinf(interp_value):

            interp_value = nodata
            any_nans_[ii] = 1

        out_array_view_[ii, k] = interp_value

        for p in range(0, ncols-1):

            # k=0, xsparse_[0]=1000
            if k+xsparse_[0] < xsparse_[p+1]:
                break

        gap_length_array_dense_[ii, k] = gap_length_array_[ii, p]


cdef double[::1] _calc_B(double[:, ::1] y_array,
                         Py_ssize_t i,
                         unsigned int nx,
                         double[:, ::1] B,
                         double[::1] h) nogil:

    cdef:
        Py_ssize_t j

    for j in range(0, nx-2):
        B[i, j+1] = 3.0 * (y_array[i, j+2] - y_array[i, j+1]) / h[j+1] - 3.0 * (y_array[i, j+1] - y_array[i, j]) / h[j]

    return B[i, :]


cdef unsigned int _bisect_left(double[::1] a, Py_ssize_t x, unsigned int lo, unsigned int hi) nogil:
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    cdef:
        unsigned int mid

    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    while lo < hi:

        mid = <int>((lo + hi) / 2.0)

        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid

    return lo


cdef void _solve(double[:, ::1] y_array,
                 double[:, ::1] yout_,
                 double[::1] xsparse,
                 double[::1] xdense,
                 Py_ssize_t i,
                 unsigned int n,
                 unsigned int nx,
                 double[:, ::1] A,
                 double[:, ::1] B,
                 double[::1] h,
                 double[::1] b,
                 double[::1] d,
                 double[::1] solution__) nogil:

    """
    # Make a local copy of zeros
    for s in range(0, self.nx):
        solution_[s] = self.solution[s]
    solve(y_array, yout, i, solution_)
    """

    cdef:
        Py_ssize_t j, t
        #int n = A.shape[0]
        double[::1] B_ = _calc_B(y_array, i, nx, B, h)
        double tb
        double dx
        double result
        bint success
        unsigned int w, st, ed
        double v

    with gil:
        print('B', np.array(B_))

    #success = dgesv.solve_c(&A[0,0], &B_[0], &solution__[0], n)
    success = dgesv.solve_c(A, &B_[0], &solution__[0], nx)

    with gil:
        print('Solution', np.array(solution__))

    # calc spline coefficient b and d
    for j in range(0, nx-1):

        d[j] = (solution__[j+1] - solution__[j]) / (3.0 * h[j])

        tb = (y_array[i, j+1] - y_array[i, j]) / h[j] - h[j] * (solution__[j+1] + 2.0 * solution__[j]) / 3.0

        b[j] = tb

    with gil:
        print('b coeff', np.array(b))

    st = <int>(xsparse[0])
    ed = <int>(xsparse[nx-1])

    for t in range(0, n):

        v = xdense[t]

        if v < xsparse[0]:
            break
        elif v > xsparse[nx-1]:
            break

        w = _bisect_left(xsparse, v, 0, nx)
        dx = <double>(v - xsparse[w])

        result = y_array[i, w] + b[w] * dx + solution__[w] * dx**2.0 + d[w] * dx**3.0

        yout_[i, t] = result


cdef class CinterpMulti(object):

    cdef:
        unsigned int nx, n
        double[::1] xsparse
        double[::1] h
        double[:, ::1] A

    """
    A class to linearly interpolate data to a new grid

    Reference:
        https://github.com/AtsushiSakai/pycubicspline/blob/master/pycubicspline.py

    Args:
        x1 (1d array): The old grid to interpolate.
        x2 (1d array): The new grid to interpolate to.
    """

    def __init__(self, double[::1] x1, double[::1] x2):

        self.nx = x1.shape[0]
        self.n = x2.shape[0]
        self.xsparse = x1

        self.h = np.zeros(self.nx-1, dtype='float64')
        self.A = np.zeros((self.nx, self.nx), dtype='float64')

        self._setup_cinterp_obj(x1)

    cdef _diff(self, double[::1] x):

        cdef:
            Py_ssize_t k

        for k in range(0, self.nx-1):
            self.h[k] = x[k+1] - x[k]

    cdef _calc_a(self):

        """
        Initiated without y
        """

        cdef:
            Py_ssize_t i

        self.A[0, 0] = 1.0

        for i in range(0, self.nx-1):

            if i != (self.nx-2):
                self.A[i+1, i+1] = 2.0 * (self.h[i] + self.h[i+1])

            self.A[i+1, i] = self.h[i]
            self.A[i, i+1] = self.h[i]

        self.A[0, 1] = 0.0

        self.A[self.nx-1, self.nx-2] = 0.0
        self.A[self.nx-1, self.nx-1] = 1.0

    cdef _setup_cinterp_obj(self, double[::1] x):

        # First differences
        self._diff(x)

        # Calculate coefficients
        self._calc_a()

    def interpolate(self,
                    double[:, ::1] array not None,
                    bint interp_step=False,
                    bint fill_no_data=False,
                    double no_data_value=0.0,
                    bint return_indexed=False,
                    unsigned long[::1] indices=None,
                    bint remove_outliers=False,
                    unsigned int max_outlier_days1=45,
                    unsigned int max_outlier_days2=80,
                    unsigned int min_outlier_values=5,
                    unsigned int outlier_iters=1,
                    double dev_thresh1=0.2,
                    double dev_thresh2=0.2,
                    int n_jobs=1,
                    int chunksize=1):

        """
        Linearly interpolates values to a new grid

        Args:
            array (2d array): The data to interpolate. The shape should be (M,).
            interp_step (Optional[bool]): Whether to interpolate step-wise.
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[double]): The 'no data' value to fill if `fill_no_data`=True.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            remove_outliers (Optional[bool]): Whether to locate and remove outliers prior to smoothing.
            max_outlier_days1 (Optional[int]): The maximum spread, in days, to search for outliers.
            max_outlier_days2 (Optional[int]): The maximum spread, in days, to search for outliers.
            min_outlier_values (Optional[int]): The minimum number of outlier samples.
            outlier_iters (Optional[int]): The number of iterations to check for outliers.
            dev_thresh1 (Optional[double]): The deviation threshold for outliers.
            dev_thresh2 (Optional[double]): The deviation threshold for outliers.
            n_jobs (Optional[int]): The number of parallel jobs.
            chunksize (Optional[int]): The parallel thread chunksize.

        Example:
            >>> import satsmooth
            >>>
            >>> interpolator = satsmooth.LinterpMulti(x1, x2)
            >>>
            >>> array = interpolator.interp2d(array)

        Returns:
            Interpolated values (2d array)
        """

        cdef:
            Py_ssize_t i, k, j, oidx, s

            unsigned int rows = array.shape[0]
            unsigned int cols = array.shape[1]

            unsigned int n_indices

            # Views of output
            np.ndarray[DTYPE_float64_t, ndim=2] out_array = np.empty((rows, self.n), dtype='float64')
            double[:, ::1] out_array_view = out_array

            np.ndarray[DTYPE_float64_t, ndim=2] out_array_indices
            double[:, ::1] out_array_indices_view

            unsigned long[::1] index_positions = np.empty(cols+1, dtype='uint64')
            unsigned long[::1] index_positions_interp = np.empty(self.n+1, dtype='uint64')

            long[::1] any_nans = np.zeros(rows, dtype='int64')
            double interp_value

            # y = alpha + beta * X
            #   where, X = x, x^2
            #
            # mb = r_1, r_2, alpha
            double[:, ::1] mb = np.zeros((rows, 3), dtype='float64')

            long[:, :, ::1] index_positions_fill_dummy = np.zeros((2, 1, 1), dtype='int64')
            long[:, :, ::1] index_positions_fill = np.zeros((2, rows, cols+1), dtype='int64')

            unsigned long[:, ::1] gap_length_array = np.zeros((rows, cols), dtype='uint64')
            unsigned long[:, ::1] gap_length_array_dense = np.zeros((rows, self.n), dtype='uint64')

            double[::1] xsparse_ = np.empty(self.nx, dtype='float64')
            double[::1] h_ = np.zeros(self.nx, dtype='float64')
            double[:, ::1] A_ = np.zeros((self.nx, self.nx), dtype='float64')

            double[:, ::1] B_ = np.zeros((rows, self.nx), dtype='float64')
            double[::1] b_ = np.zeros(self.nx, dtype='float64')
            double[::1] d_ = np.zeros(self.nx, dtype='float64')
            double[::1] solution = np.zeros(self.nx, dtype='float64')
            double[::1] solution_ = np.zeros(self.nx, dtype='float64')

        xsparse_[...] = self.xsparse
        h_[...] = self.h
        A_[...] = self.A

        if return_indexed:

            n_indices = indices.shape[0]

            out_array_indices = np.empty((rows, n_indices), dtype='float64')
            out_array_indices_view = out_array_indices

        else:

            # Dummy objects to avoid Python checks
            out_array_indices = np.empty((1, 1), dtype='float64')
            out_array_indices_view = out_array_indices

        with nogil, parallel(num_threads=n_jobs):

            for i in prange(0, rows, schedule='static', chunksize=chunksize):

                if remove_outliers:

                    for oidx in range(0, outlier_iters):

                        # Local linear regression over 3 samples
                        outliers.remove_outliers_linear(xsparse_,
                                                        array,
                                                        mb,
                                                        i,
                                                        cols,
                                                        3,
                                                        max_outlier_days1,
                                                        dev_thresh1,
                                                        no_data_value,
                                                        index_positions_fill_dummy)

                        # Local polynomial regression over user-defined samples
                        outliers.remove_outliers_polynomial(xsparse_,
                                                            array,
                                                            mb,
                                                            i,
                                                            cols,
                                                            min_outlier_values,
                                                            max_outlier_days2,
                                                            dev_thresh2,
                                                            no_data_value,
                                                            index_positions_fill)

                if fill_no_data:

                    _fill_no_data(array,
                                  i,
                                  index_positions,
                                  cols,
                                  no_data_value,
                                  True,
                                  interp_step,
                                  xsparse_,
                                  gap_length_array,
                                  True)

                # Make a local copy of zeros
                for s in range(0, self.nx):
                    solution_[s] = solution[s]

                _solve(array,
                       out_array_view,
                       xsparse_,
                       i,
                       self.n,
                       self.nx,
                       A_,
                       B_,
                       h_,
                       b_,
                       d_,
                       solution_)

                # Interpolation to a new grid
#                _regrid(array,
#                        out_array_view,
#                        i,
#                        self.n,
#                        self.a,
#                        self.b,
#                        self.lower,
#                        self.upper,
#                        no_data_value,
#                        any_nans,
#                        cols,
#                        xsparse_,
#                        gap_length_array,
#                        gap_length_array_dense)

                if any_nans[i] == 1:

                    # Re-fill between nans
                    _fill_no_data(out_array_view,
                                  i,
                                  index_positions_interp,
                                  self.n,
                                  no_data_value,
                                  True,
                                  interp_step,
                                  xsparse_,
                                  gap_length_array,
                                  False)

                if return_indexed:

                    _set_indexed_output(out_array_view,
                                        out_array_indices_view,
                                        i,
                                        indices,
                                        n_indices)

        if return_indexed:
            return out_array_indices
        else:
            return out_array

    def interpolate_smooth(self,
                           double[:, ::1] array not None,
                           bint interp_step=False,
                           bint fill_no_data=False,
                           double no_data_value=0.0,
                           bint remove_outliers=False,
                           double dev_thresh1=0.3,
                           double dev_thresh2=0.3,
                           bint return_indexed=False,
                           unsigned long[::1] indices=None,
                           unsigned int t=21,
                           unsigned int min_window=7,
                           unsigned int max_outlier_days1=45,
                           unsigned int max_outlier_days2=80,
                           unsigned int min_outlier_values=5,
                           unsigned int outlier_iters=1,
                           unsigned int min_data_count=5,
                           double mid_g=0.3,
                           double r_g=-15.0,
                           double mid_k=0.3,
                           double r_k=-15.0,
                           double mid_t=0.5,
                           double r_t=10.0,
                           double sigma_color=0.1,
                           unsigned int n_iters=1,
                           unsigned int envelope=1,
                           double h=5.0,
                           str method='dynamic',
                           int n_jobs=1,
                           int chunksize=1):

        """
        Linearly interpolates values to a new grid, with optional smoothing

        Args:
            array (2d array): The data to interpolate. The shape should be (M,).
            interp_step (Optional[bool]): Whether to interpolate step-wise.
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[double]): The 'no data' value to fill if `fill_no_data`=True.
            remove_outliers (Optional[bool]): Whether to locate and remove outliers prior to smoothing.
            dev_thresh1 (Optional[double]): The deviation threshold for outliers.
            dev_thresh2 (Optional[double]): The deviation threshold for outliers.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            t (Optional[int]): The maximum window size.
            min_window (Optional[int]): The minimum window size.
            max_outlier_days1 (Optional[int]): The maximum spread, in days, to search for outliers.
            max_outlier_days2 (Optional[int]): The maximum spread, in days, to search for outliers.
            min_outlier_values (Optional[int]): The minimum number of outlier samples.
            outlier_iters (Optional[int]): The number of iterations to check for outliers.
            min_data_count (Optional[int]): The minimum allowed data count.
            mid_g (Optional[double]): The sigmoid alpha parameter (sigmoid location) to adjust the Gaussian sigma.
                Controls the sigmoid central location.
            r_g (Optional[double]): The sigmoid beta parameter (sigmoid scale) to adjust the Gaussian sigma.
                Controls the sigmoid steepness. Decreasing `r_g` creates a more gradual change in parameters.
            mid_k (Optional[double]): The sigmoid alpha parameter (sigmoid location) to adjust the window size.

                - Controls the sigmoid central location.
                - Decreasing ``mid_k`` skews the curve to the left (i.e., toward lower values)

            r_k (Optional[double]): The sigmoid beta parameter (sigmoid scale) to adjust the window size.
                Controls the sigmoid steepness. Decreasing `r_k` creates a steeper curve.
            mid_t (Optional[double]): The sigmoid alpha parameter (sigmoid location) to dampen the window tails.
            r_t (Optional[double]): The sigmoid beta parameter (sigmoid scale) to dampen the window tails.
            sigma_color (Optional[double]): The sigma value for Gaussian color weights.
            n_iters (Optional[int]): The number of bilateral iterations, with each iteration fitting to the
                lower|upper envelope.
            envelope (Optional[int]): The envelope to adjust to (if `n_iters` > 1). 1='upper', 0='lower'.
            h (Optional[double]): The kernel regression bandwidth.
            method (Optional[str]): The smoothing method. Choices are ['dynamic', 'lowess', 'kernel'].
            n_jobs (Optional[int]): The number of parallel jobs.
            chunksize (Optional[int]): The parallel thread chunksize.

        Example:
            >>> import satsmooth as ssm
            >>>
            >>> interpolator = ssm.LinterpMulti(x1, x2)
            >>> array = interpolator.interp2dx(array, t=21, min_window=7)

        Returns:
            Interpolated values (2d array)
        """

        cdef:
            Py_ssize_t i, k, oidx, s

            unsigned int rows = array.shape[0]
            unsigned int cols = array.shape[1]
            unsigned int n_indices

            # View of output on dense grid
            double[:, ::1] out_array_dense_view_temp = np.empty((rows, self.n), dtype='float64')
            np.ndarray[DTYPE_float64_t, ndim=2] out_array_dense = np.empty((rows, self.n), dtype='float64')
            double[:, ::1] out_array_dense_view = out_array_dense

            double[:, ::1] out_array_harmonics_view = np.empty((rows, self.n), dtype='float64')

            # View of output on input, sparse grid
            np.ndarray[DTYPE_float64_t, ndim=2] out_array_sparse
            # double[:, ::1] out_array_sparse_view

            # View of output on index grid
            np.ndarray[DTYPE_float64_t, ndim=2] out_array_indexed
            double[:, ::1] out_array_indexed_view

            unsigned long[::1] index_positions = np.empty(cols+1, dtype='uint64')
            unsigned long[::1] index_positions_interp = np.empty(self.n+1, dtype='uint64')

            unsigned int t_half = <int>(t / 2.0)

            Py_ssize_t ct

            long[::1] any_nans = np.zeros(rows, dtype='int64')

            double[::1] color_weights = np.empty(101, dtype='float64')

            int data_count

            double interp_value

            # y = alpha + beta * X
            #   where, X = x, x^2
            #
            # mb = r_1, r_2, alpha
            double[:, ::1] mb = np.zeros((rows, 3), dtype='float64')

            long[:, :, ::1] index_positions_fill_dummy = np.zeros((2, 1, 1), dtype='int64')
            long[:, :, ::1] index_positions_fill = np.zeros((2, rows, cols+1), dtype='int64')

            unsigned long[:, ::1] gap_length_array = np.zeros((rows, cols), dtype='uint64')
            unsigned long[:, ::1] gap_length_array_dense = np.zeros((rows, self.n), dtype='uint64')

            double[::1] xsparse_ = np.empty(self.nx, dtype='float64')
            double[::1] h_ = np.zeros(self.nx-1, dtype='float64')
            double[:, ::1] A_ = np.zeros((self.nx, self.nx), dtype='float64')

            double[:, ::1] B_ = np.zeros((rows, self.nx), dtype='float64')
            double[::1] b_ = np.zeros(self.nx-1, dtype='float64')
            double[::1] d_ = np.zeros(self.nx-1, dtype='float64')
            double[::1] solution = np.zeros(self.nx, dtype='float64')
            double[::1] solution_ = np.zeros(self.nx, dtype='float64')

            unsigned int _method

        if method == 'dynamic':
            _method = 1
        elif method == 'lowess':
            _method = 2
        elif method == 'kernel':
            _method = 3
        else:
            raise NameError('The method is not supported.')

        xsparse_[...] = self.xsparse
        h_[...] = self.h
        A_[...] = self.A

        # if remove_outliers:
        #     out_array_sparse_view = np.empty((rows, cols), dtype='float64')
        # else:
        #     out_array_sparse_view = np.empty((1, 1), dtype='float64')

        if return_indexed:

            n_indices = indices.shape[0]

            out_array_indexed = np.empty((rows, n_indices), dtype='float64')
            out_array_indexed_view = out_array_indexed

        else:

            out_array_indexed = np.empty((1, 1), dtype='float64')
            out_array_indexed_view = out_array_indexed

        with nogil, parallel(num_threads=n_jobs):

            # Set the color weights.
            for ct in range(0, 101):
                color_weights[ct] = common.gaussian_func(float(ct) * 0.01, sigma_color)

            for i in prange(0, rows, schedule='static', chunksize=chunksize):

                # Check if there is enough data to smooth
                data_count = _check_data(array, i, 0, cols, no_data_value, min_data_count)

                if remove_outliers:

                    for oidx in range(0, outlier_iters):

                        # Local linear regression over 3 samples
                        outliers.remove_outliers_linear(xsparse_,
                                                        array,
                                                        mb,
                                                        i,
                                                        cols,
                                                        3,
                                                        max_outlier_days1,
                                                        dev_thresh1,
                                                        no_data_value,
                                                        index_positions_fill_dummy)

                        # Local polynomial regression over user-defined samples
                        outliers.remove_outliers_polynomial(xsparse_,
                                                            array,
                                                            mb,
                                                            i,
                                                            cols,
                                                            min_outlier_values,
                                                            max_outlier_days2,
                                                            dev_thresh2,
                                                            no_data_value,
                                                            index_positions_fill)

                    _fill_no_data(array,
                                  i,
                                  index_positions,
                                  cols,
                                  no_data_value,
                                  False,
                                  interp_step,
                                  xsparse_,
                                  gap_length_array,
                                  True)

                else:

                    if fill_no_data:

                        if data_count >= min_data_count:

                            # Interpolate between 'no data' points and
                            #   update the input array.
                            _fill_no_data(array,
                                          i,
                                          index_positions,
                                          cols,
                                          no_data_value,
                                          False,
                                          interp_step,
                                          xsparse_,
                                          gap_length_array,
                                          False)

                data_count = _check_data(array, i, 0, cols, no_data_value, min_data_count)

                # Make a local copy of zeros
                for s in range(0, self.nx):
                    solution_[s] = solution[s]

                _solve(array,
                       out_array_dense_view_temp,
                       xsparse_,
                       i,
                       self.nx,
                       A_,
                       B_,
                       h_,
                       b_,
                       d_,
                       solution_)

                # Interpolation to a new grid
#                _regrid(array,
#                        out_array_dense_view_temp,
#                        i,
#                        self.n,
#                        self.a,
#                        self.b,
#                        self.lower,
#                        self.upper,
#                        no_data_value,
#                        any_nans,
#                        cols,
#                        xsparse_,
#                        gap_length_array,
#                        gap_length_array_dense)

                if any_nans[i] == 1:

                    # Re-fill between nans
                    _fill_no_data(out_array_dense_view_temp,
                                  i,
                                  index_positions_interp,
                                  self.n,
                                  no_data_value,
                                  True,
                                  interp_step,
                                  xsparse_,
                                  gap_length_array,
                                  False)

                if data_count >= min_data_count:

                    # if _method == 1:

                    # Smooth the dense, interpolated grid and modify it as the output
                    _dts(out_array_dense_view_temp,
                         out_array_dense_view,
                         out_array_harmonics_view,
                         i,
                         self.n,
                         t,
                         t_half,
                         min_window,
                         mid_g,
                         r_g,
                         mid_k,
                         r_k,
                         mid_t,
                         r_t,
                         color_weights,
                         n_iters,
                         envelope,
                         gap_length_array_dense)

                    # elif _method == 2:
                    #
                    #     _lowess(out_array_dense_view_temp,
                    #             out_array_dense_view,
                    #             i,
                    #             self.n,
                    #             t,
                    #             t_half,
                    #             min_window,
                    #             mid_k,
                    #             r_k,
                    #             no_data_value,
                    #             n_iters,
                    #             envelope)
                    #
                    # elif _method == 3:
                    #
                    #     _kernel_regression(out_array_dense_view_temp,
                    #                        out_array_dense_view,
                    #                        i,
                    #                        self.n,
                    #                        h)

                else:

                    _replace_array(out_array_dense_view_temp,
                                   out_array_dense_view,
                                   i,
                                   self.n)

                if return_indexed:

                    _set_indexed_output(out_array_dense_view,
                                        out_array_indexed_view,
                                        i,
                                        indices,
                                        n_indices)

        if return_indexed:
            return out_array_indexed
        else:
            return out_array_dense
