# distutils: language = c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

"""
@author: Jordan Graesser
"""

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free, qsort

from ..utils cimport common

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


ctypedef double (*metric_ptr)(double[::1], Py_ssize_t, Py_ssize_t) nogil


cdef int _cmp(const void * pa, const void * pb) nogil:

    cdef double a = (<double *>pa)[0]
    cdef double b = (<double *>pb)[0]

    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


cdef double _get_perc(double[::1] input_view,
                      unsigned int start,
                      unsigned int end,
                      double nodata,
                      double perc) nogil:

    cdef:
        Py_ssize_t b, bidx
        int perc_index
        double* perc_buffer #= <double *>malloc(w * sizeof(double))
        double perc_result
        unsigned int nvalid = 0

    for b in range(start, end):
        if input_view[b] != nodata:
            nvalid += 1

    perc_buffer = <double *>malloc(nvalid * sizeof(double))

    bidx = 0
    for b in range(start, end):

        if input_view[b] != nodata:

            perc_buffer[bidx] = input_view[b]
            bidx += 1

    # Sort the buffer
    qsort(perc_buffer, nvalid, sizeof(double), _cmp)

    # Get the percentile
    perc_index = <int>(<double>nvalid * perc / 100.0)

    if perc_index - 1 < 0:
        perc_result = perc_buffer[0]
    else:
        perc_result = perc_buffer[perc_index-1]

    # Deallocate the buffer
    free(perc_buffer)

    return perc_result


cdef double _bounds_check(double value, double lower, double upper) nogil:

    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


cdef void _fill_ends(double[::1] array_to_fill_,
                     unsigned int cols,
                     unsigned int whalf) nogil:

    cdef:
        Py_ssize_t m
        double fill_value_start = array_to_fill_[whalf]
        double fill_value_end = array_to_fill_[cols-whalf-1]

    for m in range(0, whalf):

        array_to_fill_[m] = fill_value_start
        array_to_fill_[m+cols-whalf] = fill_value_end


cdef double _get_min(double[::1] in_row,
                    unsigned int start,
                    unsigned int end) nogil:

    cdef:
        Py_ssize_t b
        double m = in_row[start]

    for b in range(start, end):

        if in_row[b] < m:
            m = in_row[b]

    return m


cdef double _get_array_mean(double[::1] y_array_slice,
                           Py_ssize_t jjj,
                           Py_ssize_t t) nogil:

    """
    Calculates the gaussian-weighted mean of a 1-d array
    
    Args:
        y_array_slice (1d array): The array.
        jjj (int): The starting column index position.
        t (int): The ending column index position.
    
    Returns:
        Mean (double)
    """

    cdef:
        Py_ssize_t jc
        double w
        double array_sum = 0.0
        double weights_sum = 0.0

    for jc in range(jjj, t):

        if y_array_slice[jc] > 0:

            w = common.gaussian_func(common.scale_min_max(<double>jc, -1.0, 1.0, 0.0, <double>t), 0.5)
            array_sum += y_array_slice[jc] * w
            weights_sum += w

    return array_sum / weights_sum


cdef double _get_array_std(double[::1] y_array_slice,
                          Py_ssize_t jjj,
                          Py_ssize_t t) nogil:

    """
    Calculates the standard deviation of a 1-d array
    
    Args:
        y_array_slice (2d array): The array.
        jjj (int): The starting column index position.
        t (int): The ending column index position.
        
    Returns:
        Standard deviation (double)
    """

    cdef:
        Py_ssize_t jc
        double array_sum = 0.0
        Py_ssize_t array_count = 0
        double sq_dev, sqn, array_std_dev
        double sum_sq = 0.0

    for jc in range(jjj, t):

        if y_array_slice[jc] > 0:

            array_sum += y_array_slice[jc]
            array_count += 1

    array_mean = array_sum / <double>array_count

    if array_mean < 0.3:
        return array_mean

    for jc in range(jjj, t):

        if y_array_slice[jc] > 0:

            sq_dev = common.squared_diff(y_array_slice[jc], array_mean)
            sum_sq += sq_dev

    sqn = <double>array_count - 1.0

    # Standard deviation
    array_std_dev = common.sqrt(sum_sq / sqn)

    if array_std_dev < 0.1:
        array_std_dev = 0.1
    elif array_std_dev > 0.2:
        array_std_dev = 0.2

    return array_std_dev


cdef void _replace_lower_envelope(double[::1] y_array_sm,
                                  double[::1] out_array_sm,
                                  unsigned int col_count,
                                  unsigned int t,
                                  unsigned int t_half) nogil:

    """
    Replaces values with the lower envelope
    """

    cdef:
        Py_ssize_t smj
        double yval, oval

    for smj in range(0, col_count-t-1):

        yval = y_array_sm[smj+t_half]
        oval = out_array_sm[smj+t_half]

        if yval > oval:
            y_array_sm[smj+t_half] = oval


cdef void _replace_upper_envelope(double[::1] y_array_sm,
                                  double[::1] out_array_sm,
                                  unsigned int col_count,
                                  unsigned int t,
                                  unsigned int t_half) nogil:

    """
    Replaces values with the upper envelope
    """

    cdef:
        Py_ssize_t smj
        double yval, oval

    for smj in range(0, col_count-t-1):

        # Original value
        yval = y_array_sm[smj+t_half]

        # Smoothed value
        oval = out_array_sm[smj+t_half]

        if yval < oval:
            y_array_sm[smj+t_half] = oval


cdef void bilateral_1d(double[::1] y_array,
                       double[::1] out_array_,
                       unsigned int n_cols,
                       unsigned int t,
                       unsigned int t_half,
                       unsigned int min_window,
                       double alpha_t,
                       double beta_t,
                       double alpha_w,
                       double beta_w,
                       double tail_beta,
                       double[::1] color_weights,
                       unsigned int n_iters,
                       unsigned int envelope) nogil:

    """
    Smooths the data with an adaptive gaussian smoother
    
    Args:
        y_array (1d array): The input array.
        out_array_ (1d array): The output array.
        n_cols (int): The number of columns.
        t (int): The temporal window size.
        t_half (int): The temporal window half-size.
        alpha_t (double): The temporal alpha parameter.
        beta_t (double): The temporal beta parameter.
        alpha_w (double): The spatial alpha parameter.
        beta_w (double): The spatial beta parameter.
        tail_beta (double): The tail dampening beta parameter.
        color_weights (1d array): The color gaussian weights.
        n_iters (int): The number of smoothing iterations.
        envelope (int): The envelope to adjust to (if `n_iters` > 1).
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t j, zz, ni
        double weighted_sum, weights_sum, sigmoid_alpha_adjust, adjusted_value
        double slice_std, bc, vc, tw, gb, gi, w, vc_scaled
        unsigned int t_adjust, t_diff, t_diff_half
        metric_ptr ptr_func
        double sample_min, sample_std, bcw, bcw_mu

    ptr_func = &_get_array_std

    for ni in range(0, n_iters):

        # Get the time series minimum
        # sample_min = _get_min(y_array, t_half, n_cols-t_half-1)

        sample_min = _get_perc(y_array, 0, n_cols - 1, 0.0, 5.0)
        sample_quarter = _get_perc(y_array, 0, n_cols - 1, 0.0, 25.0)

        # Iterate over the array.
        for j in range(0, n_cols-t+1):

            # Get the current center value
            bc = y_array[j+t_half]

            # Get the window standard deviation
            sample_std = common.scale_min_max(common.clip_high(ptr_func(y_array, j, j+t), 0.2), 0.0, 1.0, 0.0, 0.2)

            # Get the percentage difference between the baseline and the center value
            bcw = common.clip_high(common.perc_diff(sample_min, bc) * 0.001, 1.0)

            bcw_mu = (sample_std + bcw) * 0.5

            # Get the std of the current time slice.
            # slice_std = _func(y_array, ii, j, j+t)

            # Adjust the window size
            t_adjust = <int>(common.scale_min_max(1.0 - common.logistic_func(bcw_mu, alpha_w, beta_w),
                                                  min_window, t, 0.0, 1.0))

            # Adjust the gaussian sigma based on the difference to the signal baseline and the standard deviation
            # Low values will have a higher adjusted alpha
            sigmoid_alpha_adjust = 1.0 - common.logistic_func(bcw_mu, alpha_t, beta_t)

            # Enforce odd-sized window
            t_adjust = common.adjust_window_up(t_adjust)

            # Get the difference between window sizes
            if t == t_adjust:
                t_diff = 0
                t_diff_half = t_half
            else:

                t_diff = common.window_diff(t, t_adjust)

                # Enforce odd-sized window
                t_diff = common.clip_low_int(common.adjust_window_down(t_diff), 0)

                t_diff_half = <int>(<double>t_adjust * 0.5) + t_diff

            weighted_sum = 0.0
            weights_sum = 0.0

            # Iterate over the window.
            for zz in range(t_diff, t_diff+t_adjust):

                # Safety check
                if j+zz >= n_cols:
                    break

                # Get the current window value.
                vc = _bounds_check(y_array[j+zz], 0.0, 1.0)

                if vc > 0:

                    # Lower values are closer to the window center.
                    #
                    # Values close to 0 will be given:
                    #   ... a low sigmoid weight for window heads.
                    #   ... a high sigmoid weight for window tails.
                    #   ... a high gaussian weight for window centers.
                    vc_scaled = common.scale_min_max(<double>(t_diff_half-zz), -1.0, 1.0, -<double>(t_diff_half-t_diff), <double>(t_diff_half-t_diff))

                    if bcw < 0.005:

                        # Apply more weight to values near window center
                        tw = ((1.0 - common.fabs(vc_scaled)) + sigmoid_alpha_adjust) / 2.0

                    else:

                        # Increasing segment
                        if y_array[t_diff + t_adjust] > y_array[t_diff]:

                            if vc < sample_quarter:

                                # Apply more weight to the window start
                                # favoring values near the signal baseline
                                vc_scaled = common.scale_min_max(common.clip(vc_scaled, 0.0, 1.0), 0.1, 1.0, 0.0, 1.0)

                            else:

                                # Apply more weight to the window end
                                # favoring values away from the signal baseline
                                vc_scaled = common.clip(-1.0 * vc_scaled, 0.0, 1.0)

                        else:

                            if vc >= sample_quarter:

                                # Apply more weight to the window start
                                # favoring values away from the signal baseline
                                vc_scaled = common.scale_min_max(common.clip(vc_scaled, 0.0, 1.0), 0.1, 1.0, 0.0, 1.0)

                            else:

                                # Apply more weight to the window end
                                # favoring values near the signal baseline
                                vc_scaled = common.clip(-1.0 * vc_scaled, 0.0, 1.0)

                        tw = common.scale_min_max(common.logistic_func(vc_scaled, 0.5, tail_beta), 0.1, 1.0, 0.0, 1.0)

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

                    weighted_sum += vc*w
                    weights_sum += w

            if weighted_sum > 0:

                adjusted_value = weighted_sum / weights_sum

                out_array_[j+t_half] = adjusted_value

        _fill_ends(out_array_, n_cols, t_half)

        if n_iters > 1:

            if envelope == 1:

                # Replace the output array with the upper envelope
                _replace_upper_envelope(y_array, out_array_, n_cols, t, t_half)

            else:

                # Replace the output array with the lower envelope
                _replace_lower_envelope(y_array, out_array_, n_cols, t, t_half)


def adaptive_bilateral(double[::1] array not None,
                       np.ndarray[DTYPE_float64_t, ndim=1] out_array not None,
                       unsigned int t=21,
                       unsigned int min_window=7,
                       double alpha_t=0.4,
                       double beta_t=10.0,
                       double alpha_w=0.4,
                       double beta_w=10.0,
                       double tail_beta=5.0,
                       double sigma_color=0.1,
                       unsigned int n_iters=1,
                       unsigned int envelope=1):

    """
    Smooths a 1d signal with an adaptive bilateral smoothing using Gaussian weights

    Args:
        array (1d array): The data to smooth, scaled 0-1.
        out_array (1d array): The array to write outputs to. (Same shape as `array`).
        t (Optional[int]): The maximum window size.
        min_window (Optional[int]): The minimum window size.
        alpha_t (Optional[double]): The sigmoid alpha parameter (sigmoid location) to adjust the Gaussian sigma.
            Decreasing `alpha_t` decreases the sigma weight on high `array` values.
        beta_t (Optional[double]): The sigmoid beta parameter (sigmoid scale) to adjust the Gaussian sigma.
            Decreasing `beta_t` creates a more gradual change in parameters.
        alpha_w (Optional[double]): The sigmoid alpha parameter (sigmoid location) to adjust the window size.
        beta_w (Optional[double]): The sigmoid beta parameter (sigmoid scale) to adjust the window size.
        tail_beta (double): The tail dampening beta parameter.
        sigma_color (Optional[double]): The sigma value for Gaussian color weights.
        n_iters (Optional[int]): The number of iterations to fit to the lower|upper envelope.
        envelope (Optional[int]): The envelope to adjust to (if `n_iters` > 1). 1='upper', 0='lower'.

    Returns:
        Smoothed signal (1d array)
    """

    cdef:
        unsigned int t_half = <int>(t / 2.0)
        unsigned int cols = array.shape[0]

        Py_ssize_t ct, iter_

        # Memory view of the output
        double[::1] out_array_view = out_array

        # View of the input view (only required in case of multiple iterations)
        double[::1] array_copy = array.copy()

        double[::1] color_weights = np.empty(101, dtype='float64')

    with nogil:

        # Set the color weights.
        for ct in range(0, 101):
            color_weights[ct] = common.gaussian_func(<double>ct * 0.01, sigma_color)

        bilateral_1d(array_copy,
                     out_array_view,
                     cols,
                     t,
                     t_half,
                     min_window,
                     alpha_t,
                     beta_t,
                     alpha_w,
                     beta_w,
                     tail_beta,
                     color_weights,
                     n_iters,
                     envelope)

    return out_array
