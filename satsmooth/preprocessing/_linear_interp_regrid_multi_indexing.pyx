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

from ..utils cimport common

from libcpp.vector cimport vector

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t


ctypedef float (*metric_ptr)(DTYPE_float32_t[::1], Py_ssize_t, Py_ssize_t) nogil


cdef float _bounds_check(float value, float lower, float upper) nogil:

    """
    Checks for data beyond lower and upper bounds

    Args:
        value (float): The value to check.
        lower (float): The lower bound.
        upper (float): The upper bound.

    Returns:
        Checked data (float)
    """

    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


cdef float _get_max(DTYPE_float32_t[::1] in_row, unsigned int cols) nogil:

    """
    Gets the maximum value in a 1-d array

    Args:
        in_row (1d array): The intput array.
        cols (int): The number of array columns.

    Returns:
        The array maximum value (float)
    """

    cdef:
        Py_ssize_t a
        float m = in_row[0]

    for a in range(1, cols):

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


cdef void _fill_ends(DTYPE_float32_t[::1] in_row_,
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


cdef DTYPE_float32_t[::1] _fill_no_data(DTYPE_float32_t[::1] in_row,
                                        DTYPE_uint64_t[::1] index_positions,
                                        unsigned int dims,
                                        float no_data) nogil:

    """
    Fills 'no data' values by linear interpolation between valid data

    Args:
        in_row (1d array)
        index_positions (1d array)
        dims (int)
        no_data (float)

    Returns:
        Filled array (1d array)
    """

    cdef:
        Py_ssize_t x2_idx
        unsigned int len_idx
        int x1, x2, x3

    _find_indices(in_row,
                  index_positions,
                  dims,
                  no_data)

    len_idx = <int>(index_positions[dims])

    if len_idx != 0:

        # check for data
        if _get_max(in_row, dims) > 0:

            for x2_idx in range(0, len_idx):

                x2 = <int>(index_positions[x2_idx])

                # get x1
                x1 = _get_x1(x2, in_row, no_data)

                # get x3
                x3 = _get_x3(x2, in_row, dims, no_data)

                if (x1 != -9999) and (x3 != -9999):
                    in_row[x2] = common.linear_adjustment(x1, x2, x3, in_row[x1], in_row[x3])

        _fill_ends(in_row, dims, no_data)

    return in_row


cdef float _get_array_mean(DTYPE_float32_t[::1] y_array_slice,
                           Py_ssize_t jjj,
                           Py_ssize_t t) nogil:

    """
    Calculates the gaussian-weighted mean of a 1-d array

    Args:
        y_array_slice (2d array): The array.
        iii (int): The row index position.
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

        if y_array_slice[jc] > 0:

            w = common.gaussian_func(common.scale_min_max(float(jc), -1.0, 1.0, 0.0, float(t)), 0.5)
            array_sum += y_array_slice[jc] * w
            weights_sum += w

    return array_sum / weights_sum


cdef DTYPE_float32_t[::1] _remove_outliers(DTYPE_float32_t[::1] array_,
                                           DTYPE_float32_t[::1] out_array_sparse_view_,
                                           unsigned int cols,
                                           float no_data_value,
                                           float dev_thresh) nogil:

    """
    Locates and removes outliers

    Args:
        array_ (2d array): The input array.
        out_array_sparse_view_ (2d array): The output array.
        ii (int): The row position.
        cols (int): The number of columns.
        no_data_value (float): The 'no data' value.
        dev_thresh (float): The deviation threshold.

    Returns:
        None
    """

    cdef:
        Py_ssize_t j
        float yhat

    # Check for and remove outliers
    for j in range(0, cols):

        # |(Original - Smoothed) / original|
        yhat = common.fabs((array_[j] - out_array_sparse_view_[j]) / out_array_sparse_view_[j])

        # Set the original array value to 'no data'.
        if yhat >= dev_thresh:
            array_[j] = no_data_value

    return array_



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
        Coefficient of variation (float)
    """

    cdef:
        Py_ssize_t jc
        float array_sum = 0.0
        Py_ssize_t array_count = 0
        float sq_dev, sqn, array_std_dev
        float sum_sq = 0.0

    for jc in range(jjj, t):

        if y_array_slice[jc] > 0:

            array_sum += y_array_slice[jc]
            array_count += 1

    array_mean = array_sum / float(array_count)

    if array_mean < 0.3:
        return array_mean

    for jc in range(jjj, t):

        if y_array_slice[jc] > 0:

            sq_dev = common.squared_diff(y_array_slice[jc], array_mean)
            sum_sq += sq_dev

    sqn = float(array_count) - 1.0

    # Standard deviation
    array_std_dev = common.sqrt(sum_sq / sqn)

    if array_std_dev < 0.1:
        array_std_dev = 0.1
    elif array_std_dev > 0.2:
        array_std_dev = 0.2

    # Return the standard deviation, scaled to 0-1
    return common.scale_min_max(array_std_dev, 0.0, 1.0, 0.1, 0.2)


cdef DTYPE_float32_t[::1] _bilateral(DTYPE_float32_t[::1] y_array,
                                     DTYPE_float32_t[::1] out_array_,
                                     unsigned int cols,
                                     unsigned int t,
                                     unsigned int t_half,
                                     unsigned int min_window,
                                     float alpha_t,
                                     float beta_t,
                                     float alpha_w,
                                     float beta_w,
                                     DTYPE_float32_t[::1] color_weights) nogil:

    """
    Smooths the data with an adaptive gaussian smoother

    Args:
        y_array (1d array): The input array.
        out_array_ (1d array): The output array.
        ii (int): The row position.
        cols (int): The number of columns.
        t (int): The temporal window size.
        t_half (int): The temporal window half-size.
        alpha_t (float): The temporal alpha parameter.
        beta_t (float): The temporal beta parameter.
        alpha_w (float): The spatial alpha parameter.
        beta_w (float): The spatial beta parameter.
        color_weights (1d array): The color gaussian weights.

    Returns:
        None
    """

    cdef:
        Py_ssize_t wt, j, zz
        float weighted_sum, weights_sum, sigma_adjust, adjusted_value, slice_std #slice_mu
        float bc, vc, tw, gb, gi, w, vc_scaled
        unsigned int t_adjust, t_diff, t_diff_half
        unsigned int smooth_func
        metric_ptr _func

    _func = &_get_array_std

    # Iterate over the array.
    for j in range(0, cols-t+1):

        # Get the current center value
        bc = y_array[j+t_half]

        # Get the mean of the current time slice.
        # slice_mu = _get_array_mean(y_array, j, j+t)

        # Get the std of the current time slice.
        slice_std = _func(y_array, j, j+t)

        # Adjust the window size
        t_adjust = <int>(common.scale_min_max(1.0 - common.logistic_func(slice_std, alpha_w, beta_w), min_window, t, 0.0, 1.0))

        # Adjust the gaussian sigma
        sigma_adjust = common.scale_min_max(1.0 - common.logistic_func(slice_std, alpha_t, beta_t), 0.2, 1.0, 0.0, 1.0)

        # Adjust the gaussian sigma and window size
        # sigma_adjust = common.scale_min_max(1.0 - common.logistic_func(slice_mu, alpha_t, beta_t), 0.1, 1.0, 0.0, 1.0)
        # t_adjust = <int>(common.scale_min_max(1.0 - common.logistic_func(slice_mu, alpha_w, beta_w), min_window, t, 0.0, 1.0))

        # Enforce odd-sized window
        if t_adjust % 2 == 0:
            t_adjust += 1

        # Get the difference between window sizes
        if t == t_adjust:
            t_diff = 0
            t_diff_half = t_half
        else:

            t_diff = common.window_diff(t, t_adjust)

            # Enforce odd-sized window
            if t_diff % 2 == 0:
                t_diff -= 1

            if t_diff < 0:
                t_diff = 0

            t_diff_half = <int>(float(t_adjust) / 2.0) + t_diff

        # Get the difference between window sizes
        # t_diff = common.window_diff(t, t_adjust)

        # Get window direction for sigmoid curve
        if (y_array[j + t - 1] - y_array[j]) >= 0.1:
            smooth_func = 1
        elif (y_array[j + t - 1] - y_array[j]) <= -0.1:
            smooth_func = 2
        else:
            smooth_func = 3

        weighted_sum = 0.0
        weights_sum = 0.0

        # Iterate over the window.
        for zz in range(t_diff, t_diff+t_adjust):

            # Safety check
            if j+zz >= cols:
                break

            # Get the window center value.
            vc = _bounds_check(y_array[j+zz], 0.0, 1.0)

            if vc > 0:

                vc_scaled = common.scale_min_max(float(common.abs(t_diff_half-zz)), 0.0, 1.0, 0.0,
                                                 float(t_diff_half - t_diff))

                # Get the time distance weight
                #
                # Values at the adjusted window tails (i.e., scaled to -1 and 1)
                #   will have smaller gaussian weights.
                if smooth_func == 1:
                    # Apply more weight to the window start
                    tw = common.logistic_func(vc_scaled, sigma_adjust, -10.0)
                elif smooth_func == 2:
                    # Apply more weight to the window end
                    tw = common.logistic_func(vc_scaled, sigma_adjust, 10.0)
                else:
                    # Apply equal gaussian weight
                    tw = common.gaussian_func(vc_scaled, sigma_adjust)

                # Get the time distance weight
                #
                # Values at the window tails (i.e., -1 and 1) will have smaller gaussian weights
                # tw = common.gaussian_func(common.scale_min_max(float(zz), -1.0, 1.0, 0.0, float(t_adjust)), sigma_adjust)

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

    return out_array_


cdef DTYPE_float32_t[::1] _set_indexed_output(DTYPE_float32_t[::1] out_array_dense_view,
                                              DTYPE_float32_t[::1] out_array_indexed_view_,
                                              DTYPE_uint64_t[::1] indices,
                                              unsigned int n_indices,
                                              bint check_bounds) nogil:

    cdef:
        Py_ssize_t j

    # Take the requested indices
    if check_bounds:

        for j in range(0, n_indices):
            out_array_indexed_view_[j] = _bounds_check(out_array_dense_view[indices[j]], 0.0, 1.0)

    else:

        for j in range(0, n_indices):
            out_array_indexed_view_[j] = out_array_dense_view[indices[j]]

    return out_array_indexed_view_


cdef double _lower_bound(int begin, int end, double[::1] x, double val):

    """
    Finds the lower bound, equivalent to C `lower_bound`

    Args:
        begin (int)
        end (int)
        x (1d array)
        val (float)

    Returns:
        The lower bound (float)
    """

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] >= val:
            break

    return float(i)


cdef double _upper_bound(int begin, int end, double[::1] x, double val):

    """
    Finds the upper bound, equivalent to C `upper_bound`

    Args:
        begin (int)
        end (int)
        x (1d array)
        val (float)

    Returns:
        The upper bound (float)
    """

    cdef:
        Py_ssize_t i

    for i in range(begin, end):

        if x[i] > val:
            break

    return float(i)


cdef class LinterpMultiIndex(object):

    cdef:
        unsigned int n
        vector[double] lower
        vector[double] upper
        vector[double] a
        vector[double] b

    """
    A class to linearly interpolate data to a new grid

    Args:
        x1 (1d array): The old grid to interpolate.
        x2 (1d array): The new grid to interpolate to.
    """

    def __init__(self, double[::1] x1, double[::1] x2):

        self.n = x2.shape[0]
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

    def interp2d(self,
                 DTYPE_float32_t[:, ::1] array not None,
                 bint fill_no_data=False,
                 float no_data_value=0.0,
                 bint return_indexed=False,
                 DTYPE_uint64_t[::1] indices=None,
                 int n_jobs=1):

        """
        Linearly interpolates values to a new grid

        Args:
            array (2d array): The data to interpolate. The shape should be (M,).
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[float]): The 'no data' value to fill if `fill_no_data`=True.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            n_jobs (Optional[int]): The number of parallel jobs.

        Example:
            >>> import satsmooth
            >>>
            >>> interpolator = satsmooth.LinterpMulti(x1, x2)
            >>>
            >>> array = interpolator.interp2d(array, array.copy())

        Returns:
            Interpolated values (2d array)
        """

        cdef:
            Py_ssize_t i, k, j

            unsigned int rows = array.shape[0]
            unsigned int cols = array.shape[1]

            # Views of output
            np.ndarray[DTYPE_float32_t, ndim=2] out_array = np.empty((rows, self.n), dtype='float32')
            DTYPE_float32_t[:, ::1] out_array_view = out_array

            np.ndarray[DTYPE_float32_t, ndim=2] out_array_indices
            DTYPE_float32_t[:, ::1] out_array_indices_view

            DTYPE_uint64_t[::1] index_positions = np.empty(cols+1, dtype='uint64')

        if return_indexed:

            out_array_indices = np.empty((rows, cols), dtype='float32')
            out_array_indices_view = out_array_indices

        with nogil, parallel(num_threads=n_jobs):

            for i in prange(0, rows, schedule='static'):

                if fill_no_data:

                    # Interpolate between 'no data' points
                    array[i, :] = _fill_no_data(array[i, :],
                                                index_positions,
                                                cols,
                                                no_data_value)

                # Interpolation to a new grid
                for k in range(0, self.n):
                    out_array_view[i, k] = array[i, <int>self.lower[k]] * self.a[k] + array[i, <int>self.upper[k]] * self.b[k]

                if return_indexed:

                    for j in range(0, indices.shape[0]):
                        out_array_indices_view[i, j] = out_array_view[i, indices[j]]

        if return_indexed:
            return out_array_indices
        else:
            return out_array

    def interp2dx(self,
                  DTYPE_float32_t[:, ::1] array not None,
                  bint fill_no_data=False,
                  float no_data_value=0.0,
                  bint remove_outliers=False,
                  float dev_thresh=0.2,
                  bint return_indexed=False,
                  DTYPE_uint64_t[::1] indices=None,
                  bint check_bounds=False,
                  unsigned int t=21,
                  unsigned int min_window=7,
                  float alpha_t=0.4,
                  float beta_t=10.0,
                  float alpha_w=0.4,
                  float beta_w=10.0,
                  float sigma_color=0.1,
                  int n_jobs=1):

        """
        Linearly interpolates values to a new grid, with optional smoothing

        Args:
            array (2d array): The data to interpolate. The shape should be (M,).
            fill_no_data (Optional[bool]): Whether to pre-fill 'no data' values.
            no_data_value (Optional[float]): The 'no data' value to fill if `fill_no_data`=True.
            remove_outliers (Optional[bool]): Whether to locate and remove outliers prior to smoothing.
            dev_thresh (Optional[float]): The deviation threshold for outliers.
            return_indexed (Optional[bool]): Whether to return an indexed array.
            indices (Optional[1d array]): The indices to use for slicing if `return_indexed`=True.
            check_bounds (Optional[bool]): Whether to check lower (0) and upper bounds (1).
            t (Optional[int]): The maximum window size.
            min_window (Optional[int]): The minimum window size.
            alpha_t (Optional[float]): The sigmoid alpha parameter (sigmoid location) to adjust the Gaussian sigma.
                Decreasing `alpha_t` decreases the sigma weight on high `array` values.
            beta_t (Optional[float]): The sigmoid beta parameter (sigmoid scale) to adjust the Gaussian sigma.
                Decreasing `beta_t` creates a more gradual change in parameters.
            alpha_w (Optional[float]): The sigmoid alpha parameter (sigmoid location) to adjust the window size.
            beta_w (Optional[float]): The sigmoid beta parameter (sigmoid scale) to adjust the window size.
            sigma_color (Optional[float]): The sigma value for Gaussian color weights.
            n_jobs (Optional[int]): The number of parallel jobs.

        Example:
            >>> import satsmooth
            >>>
            >>> interpolator = satsmooth.LinterpMulti(x1, x2)
            >>>
            >>> array = interpolator.interp2dx(array, t=5)

        Returns:
            Interpolated values (2d array)
        """

        cdef:
            Py_ssize_t i, k, j

            unsigned int rows = array.shape[0]
            unsigned int cols = array.shape[1]
            unsigned int n_indices

            # View of output on dense grid
            DTYPE_float32_t[:, ::1] out_array_dense_view_temp = np.empty((rows, self.n), dtype='float32')
            np.ndarray[DTYPE_float32_t, ndim=2] out_array_dense = np.empty((rows, self.n), dtype='float32')
            DTYPE_float32_t[:, ::1] out_array_dense_view = out_array_dense

            # View of output on input, sparse grid
            np.ndarray[DTYPE_float32_t, ndim=2] out_array_sparse
            DTYPE_float32_t[:, ::1] out_array_sparse_view

            # View of output on index grid
            np.ndarray[DTYPE_float32_t, ndim=2] out_array_indexed
            DTYPE_float32_t[:, ::1] out_array_indexed_view

            DTYPE_uint64_t[::1] index_positions = np.empty(cols+1, dtype='uint64')

            unsigned int t_half = <int>(t / 2.0)

            Py_ssize_t ct
            # float yhat

            DTYPE_float32_t[::1] color_weights = np.empty(101, dtype='float32')

        if remove_outliers:
            out_array_sparse_view = np.empty((rows, cols), dtype='float32')

        if return_indexed:

            n_indices = indices.shape[0]

            out_array_indexed = np.empty((rows, n_indices), dtype='float32')
            out_array_indexed_view = out_array_indexed

        with nogil, parallel(num_threads=n_jobs):

            # Set the color weights.
            for ct in range(0, 101):
                color_weights[ct] = common.gaussian_func(float(ct) * 0.01, sigma_color)

            for i in prange(0, rows, schedule='static'):

                if fill_no_data:

                    # Interpolate between 'no data' points and
                    #   update the input array.
                    array[i, :] = _fill_no_data(array[i, :],
                                                index_positions,
                                                cols,
                                                no_data_value)

                if remove_outliers:

                    # Smooth the filled signal, updating the sparse output.
                    out_array_sparse_view[i, :] = _bilateral(array[i, :],
                                                             out_array_sparse_view[i, :],
                                                             cols,
                                                             t,
                                                             t_half,
                                                             min_window,
                                                             alpha_t,
                                                             beta_t,
                                                             alpha_w,
                                                             beta_w,
                                                             color_weights)

                    array[i, :] = _remove_outliers(array[i, :],
                                                   out_array_sparse_view[i, :],
                                                   cols,
                                                   no_data_value,
                                                   dev_thresh)

                    # Interpolate between 'no data' points after removing outliers
                    array[i, :] = _fill_no_data(array[i, :],
                                                index_positions,
                                                cols,
                                                no_data_value)

                # Interpolation to a new grid
                for k in range(0, self.n):
                    out_array_dense_view_temp[i, k] = array[i, <int>self.lower[k]] * self.a[k] + array[i, <int>self.upper[k]] * self.b[k]

                # Smooth the dense, interpolated grid and modify it as the output
                out_array_dense_view[i, :] = _bilateral(out_array_dense_view_temp[i, :],
                                                        out_array_dense_view[i, :],
                                                        self.n,
                                                        t,
                                                        t_half,
                                                        min_window,
                                                        alpha_t,
                                                        beta_t,
                                                        alpha_w,
                                                        beta_w,
                                                        color_weights)

                if return_indexed:

                    out_array_indexed_view[i, :] = _set_indexed_output(out_array_dense_view[i, :],
                                                                       out_array_indexed_view[i, :],
                                                                       indices,
                                                                       n_indices,
                                                                       check_bounds)

        if return_indexed:
            return out_array_indexed
        else:
            return out_array_dense
