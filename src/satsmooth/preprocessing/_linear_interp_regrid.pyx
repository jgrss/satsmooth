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
from libcpp.vector cimport vector

from ..utils cimport common

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t


ctypedef float (*metric_ptr)(DTYPE_float32_t[::1], Py_ssize_t, Py_ssize_t) nogil


cdef float _get_max(DTYPE_float32_t[::1] in_row, unsigned int start, unsigned int end) nogil:

    cdef:
        Py_ssize_t a
        float m = in_row[start]

    for a in range(start+1, end):

        if in_row[a] > m:
            m = in_row[a]

    return m


cdef int _get_x1(unsigned int x2,
                 DTYPE_float32_t[::1] in_row,
                 float no_data) nogil:

    cdef:
        Py_ssize_t x1_iter
        int x1 = -9999

    for x1_iter in range(1, x2+1):

        if in_row[x2-x1_iter] != no_data:
            x1 = x2 - x1_iter

            break

    return x1


cdef int _get_x3(unsigned int x2,
                 DTYPE_float32_t[::1] in_row,
                 unsigned int dims,
                 float no_data) nogil:

    cdef:
        Py_ssize_t x3_iter
        int x3 = -9999
        int x3_range = dims - x2

    for x3_iter in range(1, x3_range):

        if in_row[x2+x3_iter] != no_data:
            x3 = x2 + x3_iter

            break

    return x3


cdef void _find_indices(DTYPE_float32_t[::1] in_row,
                        DTYPE_uint64_t[::1] index_positions_,
                        unsigned int dims,
                        float no_data) nogil:

    cdef:
        Py_ssize_t fi
        Py_ssize_t counter = 0

    for fi in range(0, dims):

        if in_row[fi] == no_data:
            index_positions_[counter] = fi

            counter += 1

    # Store the count in the last index position
    index_positions_[dims] = counter


cdef void _fill_ends(DTYPE_float32_t[::1] array_to_fill_,
                     unsigned int ncol,
                     unsigned int whalf) nogil:

    cdef:
        Py_ssize_t m
        float fill_value_start = array_to_fill_[whalf]
        float fill_value_end = array_to_fill_[ncol-whalf-1]

    for m in range(0, whalf):

        array_to_fill_[m] = fill_value_start
        array_to_fill_[m+ncol-whalf] = fill_value_end


cdef void _fill_no_data_ends(DTYPE_float32_t[::1] in_row_,
                             unsigned int dims,
                             float no_data) nogil:

    """
    Fills 1-d array endpoints
    
    Args:
        in_row_ (1d array)
        dims (int)
        no_data (float)
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t ib, ic
        float valid_data

    if in_row_[0] == no_data:

        # Find the first valid data point
        for ib in range(1, dims):

            if in_row_[ib] != no_data:
                valid_data = in_row_[ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(0, ib):
            in_row_[ic] = valid_data

    if in_row_[dims-1] == no_data:

        # Find the first non-zero data
        for ib in range(dims-2, 0, -1):

            if in_row_[ib] != no_data:
                valid_data = in_row_[ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(ib+1, dims):
            in_row_[ic] = valid_data


cdef void _fill_no_data(DTYPE_float32_t[::1] in_data_,
                        DTYPE_uint64_t[::1] index_positions,
                        unsigned int dims,
                        float no_data,
                        bint check_ifdata) nogil:

    """
    Fills 'no data' values by linear interpolation between valid data
    
    Args:
        in_data_ (1d array)
        index_positions (1d array)
        dims (int)
        no_data (float)
        check_ifdata (bool)
        
    Returns:
        Filled array (1d array)
    """

    cdef:
        Py_ssize_t x2_idx
        unsigned int len_idx
        int x1, x2, x3

    _find_indices(in_data_,
                  index_positions,
                  dims,
                  no_data)

    len_idx = <int>(index_positions[dims])

    if len_idx != 0:

        if check_ifdata:

            if _get_max(in_data_, 0, dims) > 0:

                for x2_idx in range(0, len_idx):

                    x2 = <int>(index_positions[x2_idx])

                    # get x1
                    x1 = _get_x1(x2, in_data_, no_data)

                    # get x3
                    x3 = _get_x3(x2, in_data_, dims, no_data)

                    if (x1 != -9999) and (x3 != -9999):
                        in_data_[x2] = common.linear_adjustment(x1, x2, x3, in_data_[x1], in_data_[x3])

        else:

            for x2_idx in range(0, len_idx):

                x2 = < int > (index_positions[x2_idx])

                # get x1
                x1 = _get_x1(x2, in_data_, no_data)

                # get x3
                x3 = _get_x3(x2, in_data_, dims, no_data)

                if (x1 != -9999) and (x3 != -9999):
                    in_data_[x2] = common.linear_adjustment(x1, x2, x3, in_data_[x1], in_data_[x3])

        _fill_no_data_ends(in_data_, dims, no_data)


cdef float _get_array_stats(DTYPE_float32_t[::1] y_array_slice, unsigned int t) nogil:

    """
    Gaussian-weighted mean
    """

    cdef:
        Py_ssize_t jc
        float w
        float array_sum = 0.0
        float weights_sum = 0.0

    for jc in range(0, t):

        if y_array_slice[jc] > 0:

            w = common.gaussian_func(common.scale_min_max(float(jc), -1.0, 1.0, 0.0, float(t)), 0.5)
            array_sum += y_array_slice[jc] * w
            weights_sum += w

    return array_sum / weights_sum


cdef float _get_min(DTYPE_float32_t[::1] in_row,
                    unsigned int start,
                    unsigned int end) nogil:

    cdef:
        Py_ssize_t b
        float m = in_row[start]

    for b in range(start+1, end):

        if in_row[b] < m:
            m = in_row[b]

    return m


cdef float _get_array_mean(DTYPE_float32_t[::1] y_array_slice,
                           Py_ssize_t jjj,
                           Py_ssize_t t,
                           int ignore_j=-999) nogil:

    """
    Calculates the gaussian-weighted mean of a 1-d array
    
    Args:
        y_array_slice (1d array): The array.
        jjj (int): The starting column index position.
        t (int): The ending column index position.
        
    Returns:
        Mean (float)
    """

    cdef:
        Py_ssize_t jc
        float w
        float array_sum = 0.0
        float weights_sum = 0.0

    for jc in range(jjj, t):

        if ignore_j != -999:
            if jc == ignore_j:
                continue

        if y_array_slice[jc] > 0:

            w = common.gaussian_func(common.scale_min_max(float(jc), -1.0, 1.0, 0.0, float(t)), 0.5)
            array_sum += y_array_slice[jc] * w
            weights_sum += w

    return array_sum / weights_sum


cdef void _remove_outliers(DTYPE_float32_t[::1] array_,
                           double[::1] xsparse,
                           DTYPE_float32_t[::1] smooth_sparse_view_,
                           unsigned int cols,
                           unsigned int max_outlier_window,
                           unsigned int max_outlier_days,
                           unsigned int min_outlier_values,
                           float no_data_value,
                           float low_dev_thresh,
                           float high_dev_thresh,
                           DTYPE_uint64_t[::1] index_positions,
                           int remove_type) nogil:

    """
    Locates and removes outliers
    
    Args:
        array_ (1d array): The array with potential outliers.
        smooth_sparse_view_ (1d array): The smoothed array.
        ii (int): The row position.
        cols (int): The number of columns.
        no_data_value (float): The 'no data' value.
        dev_thresh (float): The deviation threshold.
        index_positions (1d array)
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t j, j2a, j2b
        unsigned int time_len_a, time_len_b
        unsigned int data_count_
        float yhat
        bint no_data_checker = False
        unsigned int wh = <int>(max_outlier_window / 2.0)

    for j in range(0, cols-max_outlier_window+1):

        j2a = wh-1
        j2b = wh+1

        while True:

            if j2a < 0:
                j2a = 0
                break

            # Time, in days, of the window spread
            time_len_a = <int>(xsparse[j+wh] - xsparse[j+j2a])

            if time_len_a >= max_outlier_days:
                break

            j2a -= 1

        while True:

            if j2b >= max_outlier_window:
                j2b = max_outlier_window
                break

            # Time, in days, of the window spread
            time_len_b = <int>(xsparse[j+j2b] - xsparse[j+wh])

            if time_len_b >= max_outlier_days:
                break

            j2b += 1

        data_count_ = _check_data(array_, j+j2a, j+j2b, no_data_value, min_outlier_values)

        if data_count_ >= min_outlier_values:

            if remove_type < 0:

                if array_[j+wh] < smooth_sparse_view_[j+wh]:

                    # |(Original - Smoothed) / original|
                    yhat = common.fabs(common.prop_diff(smooth_sparse_view_[j+wh], array_[j+wh]))

                    if yhat >= low_dev_thresh:

                        # The outlier > deviation threshold AND
                        #   the outlier value < the smoothed value.
                        array_[j+wh] = 0.0
                        no_data_checker = True

            else:

                if array_[j+wh] > smooth_sparse_view_[j+wh]:

                    # |(Original - Smoothed) / original|
                    yhat = common.fabs(common.prop_diff(smooth_sparse_view_[j+wh], array_[j+wh]))

                    if yhat >= high_dev_thresh:

                        array_[j+wh] = 0.0
                        no_data_checker = True

    if no_data_checker:

        # Interpolate between 'no data' points after removing outliers
        _fill_no_data(array_,
                      index_positions,
                      cols,
                      no_data_value,
                      True)


cdef float _get_array_std(DTYPE_float32_t[::1] y_array_slice,
                          Py_ssize_t jjj,
                          Py_ssize_t t) nogil:

    """
    Calculates the standard deviation of a 1-d array
    
    Args:
        y_array_slice (2d array): The array.
        jjj (int): The starting column index position.
        t (int): The ending column index position.
        
    Returns:
        Standard deviation (float)
    """

    cdef:
        Py_ssize_t jc
        float array_sum = y_array_slice[jjj]
        float yvalue_, array_mean
        Py_ssize_t array_count = 1
        float sq_dev, sqn
        float sum_sq = 0.0

    for jc in range(jjj+1, t):

        yvalue_ = y_array_slice[jc]

        if yvalue_ > 0:

            array_sum += yvalue_
            array_count += 1

    array_mean = array_sum / float(array_count)

    for jc in range(jjj, t):

        yvalue_ = y_array_slice[jc]

        if yvalue_ > 0:

            sq_dev = common.squared_diff(yvalue_, array_mean)
            sum_sq += sq_dev

    sqn = float(array_count) - 1.0

    return common.sqrt(sum_sq / sqn)


cdef void _replace_lower_envelope(DTYPE_float32_t[::1] y_array_sm,
                                  DTYPE_float32_t[::1] out_array_sm,
                                  unsigned int col_count,
                                  unsigned int t,
                                  unsigned int t_half) nogil:

    """
    Replaces values with the lower envelope
    """

    cdef:
        Py_ssize_t smj
        float yval, oval

    for smj in range(0, col_count-t-1):

        yval = y_array_sm[smj+t_half]
        oval = out_array_sm[smj+t_half]

        if yval > oval:
            y_array_sm[smj+t_half] = oval


cdef void _replace_upper_envelope(DTYPE_float32_t[::1] y_array_sm,
                                  DTYPE_float32_t[::1] out_array_sm,
                                  unsigned int col_count,
                                  unsigned int t,
                                  unsigned int t_half) nogil:

    """
    Replaces values with the upper envelope
    """

    cdef:
        Py_ssize_t smj
        float yval, oval

    for smj in range(0, col_count-t-1):

        # Original value
        yval = y_array_sm[smj+t_half]

        # Smoothed value
        oval = out_array_sm[smj+t_half]

        if yval < oval:
            y_array_sm[smj+t_half] = oval


cdef void _bilateral(DTYPE_float32_t[::1] y_array,
                     DTYPE_float32_t[::1] out_array_,
                     unsigned int n_cols,
                     unsigned int t,
                     unsigned int t_half,
                     unsigned int min_window,
                     float alpha_t,
                     float beta_t,
                     float alpha_w,
                     float beta_w,
                     DTYPE_float32_t[::1] color_weights,
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
        alpha_t (float): The temporal alpha parameter.
        beta_t (float): The temporal beta parameter.
        alpha_w (float): The spatial alpha parameter.
        beta_w (float): The spatial beta parameter.
        color_weights (1d array): The color gaussian weights.
        n_iters (int): The number of smoothing iterations.
        envelope (int): The envelope to adjust to (if `n_iters` > 1).
        
    Returns:
        None
    """

    cdef:
        Py_ssize_t j, zz, ni
        float weighted_sum, weights_sum, sigma_adjust, adjusted_value
        float slice_std, bc, vc, tw, gb, gi, w, vc_scaled
        unsigned int t_adjust, t_diff, t_diff_half
        metric_ptr ptr_func
        float sample_min, sample_std, bcw, bcw_mu

    ptr_func = &_get_array_std

    for ni in range(0, n_iters):

        # Get the time series minimum
        sample_min = _get_min(y_array, t_half, n_cols-t_half-1)

        # Iterate over the array.
        for j in range(0, n_cols-t+1):

            # Get the current center value
            bc = y_array[j+t_half]

            sample_std = common.scale_min_max(common.clip_high(ptr_func(y_array, j, j+t), 0.2),
                                              0.0, 1.0, 0.0, 0.2)

            # Get the percentage difference between the baseline and the center value
            bcw = common.clip_high(common.perc_diff(sample_min, bc) * 0.001, 1.0)

            # Weight the standard deviation higher
            #   than the signal value.
            bcw_mu = (sample_std + bcw) * 0.5

            # Get the std of the current time slice.
            # slice_std = _func(y_array, ii, j, j+t)

            # Adjust the window size
            t_adjust = <int>(common.scale_min_max(1.0 - common.logistic_func(bcw_mu, alpha_w, beta_w),
                                                  min_window, t, 0.0, 1.0))

            # Adjust the gaussian sigma
            sigma_adjust = common.scale_min_max(common.logistic_func(bcw_mu, alpha_t, beta_t), 0.3, 0.6, 0.0, 1.0)

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

                t_diff_half = <int>(float(t_adjust) * 0.5) + t_diff

            weighted_sum = 0.0
            weights_sum = 0.0

            # Iterate over the window.
            for zz in range(t_diff, t_diff+t_adjust):

                # Safety check
                if j+zz >= n_cols:
                    break

                # Get the current window value.
                vc = common.clip(y_array[j+zz], 0.0, 1.0)

                if vc > 0:

                    # Lower values are closer to the window center.
                    #
                    # Values close to 0 will be given:
                    #   ... a low sigmoid weight for window heads.
                    #   ... a high sigmoid weight for window tails.
                    #   ... a high gaussian weight for window centers.
                    vc_scaled = common.scale_min_max(float(common.abs(t_diff_half-zz)), 0.0, 1.0, 0.0, float(t_diff_half-t_diff))

                    # Before the window center
                    if zz < t_diff+t_diff_half:

                        # Increasing segment
                        if y_array[t_diff+t_adjust-1] > y_array[t_diff]:

                            # Apply more weight to the window start
                            # favoring values near the signal baseline
                            tw = common.logistic_func(vc_scaled, sigma_adjust, 10.0) * bcw

                        else:

                            # Apply more weight to the window center
                            # favoring values away from the signal baseline
                            tw = common.logistic_func(vc_scaled, sigma_adjust, -10.0) * (1.0 * bcw)

                    else:

                        # Increasing segment
                        if y_array[t_diff + t_adjust - 1] > y_array[t_diff]:

                            # Apply more weight to the window center
                            # favoring values away from the signal baseline
                            tw = common.logistic_func(vc_scaled, sigma_adjust, -10.0) * (1.0 * bcw)

                        else:

                            # Apply more weight to the window end
                            # favoring values near the signal baseline
                            tw = common.logistic_func(vc_scaled, sigma_adjust, 10.0) * bcw

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


cdef double _lower_bound(int begin, int end, double[::1] x, double val):

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] >= val:
            break

    return float(i)


cdef double _upper_bound(int begin, int end, double[::1] x, double val):

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] > val:
            break

    return float(i)


cdef unsigned int _check_data(DTYPE_float32_t[::1] array,
                              unsigned int start,
                              unsigned int end,
                              float no_data_value,
                              unsigned int min_data_required) nogil:

    """
    Checks if there are sufficient data points in the time series
    
    Args:
        array (1d array)
        start (int)
        end (int)
        no_data_value (float)
        min_data_required (float)
        
    Returns:
        (int)
    """

    cdef:
        Py_ssize_t ci
        unsigned int data_count_ = 0

    for ci in range(start, end):

        if array[ci] != no_data_value:
            data_count_ += 1

        if data_count_ >= min_data_required:
            break

    return data_count_


cdef void _set_indexed_output(DTYPE_float32_t[::1] out_array_dense_view_,
                              DTYPE_float32_t[::1] out_array_indexed_view_,
                              DTYPE_uint64_t[::1] indices,
                              unsigned int n_indices) nogil:

    cdef:
        Py_ssize_t jidx

    for jidx in range(0, n_indices):
        out_array_indexed_view_[jidx] = common.clip(out_array_dense_view_[indices[jidx]], 0.0, 1.0)


cdef void _replace_array(DTYPE_float32_t[::1] y_array,
                         DTYPE_float32_t[::1] out_array_,
                         unsigned int cols) nogil:

    cdef:
        Py_ssize_t j

    for j in range(0, cols):
        out_array_[j] = y_array[j]


cdef class Linterp(object):

    cdef:
        unsigned int n
        double[::1] xsparse
        vector[double] lower
        vector[double] upper
        vector[double] a
        vector[double] b

    """
    A class to interpolate to a new grid

    Args:
        x1 (1d array): The old grid to interpolate.
        x2 (1d array): The new grid to interpolate to.
    """

    def __init__(self, double[::1] x1, double[::1] x2):

        self.n = x2.shape[0]
        self.xsparse = x1
        self._setup_interp_obj(x1, x2)

    cdef _setup_interp_obj(self, double[::1] x1, double[::1] x2):

        cdef:
            unsigned int m = x1.shape[0]

            Py_ssize_t k
            double low, high

        for k in range(0, self.n):

            low = _lower_bound(0, m, x1, x2[k])
            high = _upper_bound(0, m, x1, x2[k])

            # Index of upper bound for current point
            self.upper.push_back(low)

            # Index of lower bound for current point
            self.lower.push_back(high - 1.0)

            if (self.lower[k] == -1) and (self.upper[k] == 0):

                self.lower[k] = 0.0
                self.upper[k] = 1.0

            elif (self.lower[k] == m-1) and (self.upper[k] == m):

                self.lower[k] = m - 2.0
                self.upper[k] = m - 1.0

            if self.lower[k] == self.upper[k]:
                self.b.push_back(0.5)
            else:
                self.b.push_back((x2[k] - x1[<int>self.lower[k]]) / (x1[<int>self.upper[k]] - x1[<int>self.lower[k]]))

            self.a.push_back(1.0 - self.b[k])

    def interp1d(self,
                 DTYPE_float32_t[::1] array not None,
                 np.ndarray[DTYPE_float32_t, ndim=1] out_array not None,
                 bint fill_no_data=False,
                 float no_data_value=0.0,
                 bint return_indexed=False,
                 DTYPE_uint64_t[::1] indices=None,
                 np.ndarray[DTYPE_float32_t, ndim=1] out_array_indices=None):

        """
        Linearly interpolates values to a new grid

        Args:
            array (1d array): The data to interpolate. The shape should be (M,).
            out_array (1d array): The array to write outputs to. (Same shape as `array`). Initialized as
                optional, but must be declared if `out_array_indices` is not given. This may be less
                convenient to the user, but provides an easier interface to iterate over many arrays
                without re-declaring the output on every iteration.
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[float]): The 'no data' value to fill if `fill_no_data`=True.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            out_array_indices (Optional[1d array]): The output array if `return_indexed`=True. The shape
                should match `indices`. *Overrides `out_array`.

        Example:
            >>> import satsmooth
            >>> interpolator = satsmooth.Linterp(x1, x2)
            >>>
            >>> array = interpolator.interp1d(array, out_array=array.copy())

        Returns:
            Interpolated values (1d array)
        """

        cdef:
            Py_ssize_t k, j

            # Views of output
            DTYPE_float32_t[::1] out_array_view = out_array
            DTYPE_float32_t[::1] out_array_indices_view = out_array_indices

            unsigned int dims = array.shape[0]

            DTYPE_uint64_t[::1] index_positions = np.empty(dims+1, dtype='uint64')

        with nogil:

            if fill_no_data:

                # Interpolate between 'no data' points
                _fill_no_data(array,
                              index_positions,
                              dims,
                              no_data_value,
                              True)

            # Interpolation to a new grid
            for k in range(0, self.n):
                out_array_view[k] = array[<int>self.lower[k]] * self.a[k] + array[<int>self.upper[k]] * self.b[k]

            if return_indexed:

                for j in range(0, indices.shape[0]):
                    out_array_indices_view[j] = out_array_view[indices[j]]

        if return_indexed:
            return out_array_indices
        else:
            return out_array

    def interp1dx(self,
                  DTYPE_float32_t[::1] array not None,
                  bint fill_no_data=False,
                  float no_data_value=0.0,
                  bint remove_outliers=False,
                  unsigned int outlier_iters=1,
                  float low_dev_thresh=0.2,
                  float high_dev_thresh=0.2,
                  bint return_indexed=False,
                  DTYPE_uint64_t[::1] indices=None,
                  unsigned int t=21,
                  unsigned int min_window=7,
                  unsigned int low_outlier_window=15,
                  unsigned int high_outlier_window=15,
                  unsigned int max_outlier_days=31,
                  unsigned int min_outlier_values=5,
                  unsigned int min_data_count=5,
                  float alpha_t=0.4,
                  float beta_t=10.0,
                  float alpha_w=0.4,
                  float beta_w=10.0,
                  float sigma_color=0.1,
                  unsigned int n_iters=1,
                  unsigned int envelope=1):

        """
        Linearly interpolates values to a new grid, with optional smoothing

        Args:
            array (1d array): The data to interpolate. The shape should be (M,).
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[float]): The 'no data' value to fill if `fill_no_data`=True.
            remove_outliers (Optional[bool]): Whether to locate and remove outliers prior to smoothing.
            outlier_iters (Optional[int]): The number of outlier iterations.
            low_dev_thresh (Optional[float]): The deviation threshold for low outliers.
            high_dev_thresh (Optional[float]): The deviation threshold for high outliers.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            t (Optional[int]): The maximum window size.
            min_window (Optional[int]): The minimum window size.
            low_outlier_window (Optional[int]): The maximum window size to search for outliers.
            high_outlier_window (Optional[int]): The maximum window size to search for outliers.
            max_outlier_days (Optional[int]): The maximum spread, in days, to search for outliers.
            min_outlier_values (Optional[int]): The minimum number of outlier samples.
            min_data_count (Optional[int]): The minimum allowed data count.
            alpha_t (Optional[float]): The sigmoid alpha parameter (sigmoid location) to adjust the Gaussian sigma.
                Decreasing `alpha_t` decreases the sigma weight on high `array` values.
            beta_t (Optional[float]): The sigmoid beta parameter (sigmoid scale) to adjust the Gaussian sigma.
                Decreasing `beta_t` creates a more gradual change in parameters.
            alpha_w (Optional[float]): The sigmoid alpha parameter (sigmoid location) to adjust the window size.
            beta_w (Optional[float]): The sigmoid beta parameter (sigmoid scale) to adjust the window size.
            sigma_color (Optional[float]): The sigma value for Gaussian color weights.
            n_iters (Optional[int]): The number of bilateral iterations, with each iteration fitting to the
                lower|upper envelope.
            envelope (Optional[int]): The envelope to adjust to (if `n_iters` > 1). 1='upper', 0='lower'.

        Example:
            >>> import satsmooth
            >>> interpolator = satsmooth.Linterp(x1, x2)
            >>>
            >>> array = interpolator.interp1dx(array, array.copy(), t=5)

        Returns:
            Interpolated values (1d array)
        """

        cdef:
            Py_ssize_t i, k, oiter

            unsigned int cols = array.shape[0]
            unsigned int n_indices

            # View of output on dense grid
            DTYPE_float32_t[::1] out_array_dense_view_temp = np.empty(self.n, dtype='float32')
            np.ndarray[DTYPE_float32_t, ndim=1] out_array_dense = np.empty(self.n, dtype='float32')
            DTYPE_float32_t[::1] out_array_dense_view = out_array_dense

            # View of output on input, sparse grid
            np.ndarray[DTYPE_float32_t, ndim=1] out_array_sparse
            DTYPE_float32_t[::1] out_array_sparse_view

            # View of output on index grid
            np.ndarray[DTYPE_float32_t, ndim=1] out_array_indexed
            DTYPE_float32_t[::1] out_array_indexed_view

            DTYPE_uint64_t[::1] index_positions = np.empty(cols+1, dtype='uint64')
            DTYPE_uint64_t[::1] index_positions_interp = np.empty(self.n+1, dtype='uint64')

            unsigned int t_half = <int>(t / 2.0)

            Py_ssize_t ct

            DTYPE_float32_t[::1] color_weights = np.empty(101, dtype='float32')

            unsigned int data_count

            bint any_nans
            float interp_value

            double[::1] xsparse_ = np.empty(cols, dtype='float64')

        xsparse_[...] = self.xsparse

        if remove_outliers:
            out_array_sparse_view = np.empty(cols, dtype='float32')
        else:
            out_array_sparse_view = np.empty(1, dtype='float32')

        if return_indexed:

            n_indices = indices.shape[0]

            out_array_indexed = np.empty(n_indices, dtype='float32')
            out_array_indexed_view = out_array_indexed

        else:

            out_array_indexed = np.empty(1, dtype='float32')
            out_array_indexed_view = out_array_indexed

        with nogil:

            # Set the color weights.
            for ct in range(0, 101):
                color_weights[ct] = common.gaussian_func(float(ct) * 0.01, sigma_color)

            # Check if there is enough data to smooth
            if remove_outliers:
                data_count = _check_data(array, 0, cols, no_data_value, high_outlier_window)
            else:
                data_count = _check_data(array, 0, cols, no_data_value, min_data_count)

            if data_count >= high_outlier_window:

                if fill_no_data:

                    # Interpolate between 'no data' points and
                    #   update the input array.
                    _fill_no_data(array,
                                  index_positions,
                                  cols,
                                  no_data_value,
                                  False)

                if remove_outliers:

                    for oiter in range(0, outlier_iters):

                        # Smooth the filled signal, updating the sparse output.
                        _bilateral(array,
                                   out_array_sparse_view,
                                   cols,
                                   high_outlier_window,
                                   <int>(high_outlier_window / 2.0),
                                   high_outlier_window,
                                   alpha_t,
                                   beta_t,
                                   alpha_w,
                                   beta_w,
                                   color_weights,
                                   1, 0)

                        # Remove high outliers
                        _remove_outliers(array,
                                         xsparse_,
                                         out_array_sparse_view,
                                         cols,
                                         high_outlier_window,
                                         max_outlier_days,
                                         min_outlier_values,
                                         no_data_value,
                                         low_dev_thresh,
                                         high_dev_thresh,
                                         index_positions,
                                         1)

                        data_count = _check_data(array, 0, cols, no_data_value, low_outlier_window)

                        if data_count >= low_outlier_window:

                            # Smooth the filled signal, updating the sparse output.
                            _bilateral(array,
                                       out_array_sparse_view,
                                       cols,
                                       low_outlier_window,
                                       <int>(low_outlier_window / 2.0),
                                       low_outlier_window,
                                       alpha_t,
                                       beta_t,
                                       alpha_w,
                                       beta_w,
                                       color_weights,
                                       1, 0)

                            # Remove low outliers
                            _remove_outliers(array,
                                             xsparse_,
                                             out_array_sparse_view,
                                             cols,
                                             low_outlier_window,
                                             max_outlier_days,
                                             min_outlier_values,
                                             no_data_value,
                                             low_dev_thresh,
                                             high_dev_thresh,
                                             index_positions,
                                             -1)

                            data_count = _check_data(array, 0, cols, no_data_value, min_data_count)

            any_nans = False

            # Interpolation to a new grid
            for k in range(0, self.n):

                interp_value = array[<int>self.lower[k]] * self.a[k] + array[<int>self.upper[k]] * self.b[k]

                if common.npy_isnan(interp_value) or common.npy_isinf(interp_value):
                    interp_value = no_data_value
                    any_nans = True

                out_array_dense_view_temp[k] = interp_value

            if any_nans:

                # Re-fill between nans
                _fill_no_data(out_array_dense_view_temp,
                              index_positions_interp,
                              self.n,
                              no_data_value,
                              True)

            if data_count >= min_data_count:

                # Smooth the dense, interpolated grid and modify it as the output
                _bilateral(out_array_dense_view_temp,
                           out_array_dense_view,
                           self.n,
                           t,
                           t_half,
                           min_window,
                           alpha_t,
                           beta_t,
                           alpha_w,
                           beta_w,
                           color_weights,
                           n_iters,
                           envelope)

            else:

                _replace_array(out_array_dense_view_temp,
                               out_array_dense_view,
                               self.n)

            if return_indexed:

                _set_indexed_output(out_array_dense_view,
                                    out_array_indexed_view,
                                    indices,
                                    n_indices)

        if return_indexed:
            return out_array_indexed
        else:
            return out_array_dense
