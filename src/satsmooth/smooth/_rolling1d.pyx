# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from ..utils cimport common

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


# Define a function pointer to a metric.
ctypedef void (*metric_ptr)(double[::1], double[::1], unsigned int, unsigned int, unsigned int, double[::1], float no_data)


cdef void _fill_ends(double[::1] array_to_fill_,
                     unsigned int cols,
                     unsigned int whalf) nogil:

    cdef:
        Py_ssize_t m
        float fill_value_start = array_to_fill_[whalf]
        float fill_value_end = array_to_fill_[cols-whalf-1]

    for m in range(0, whalf):

        array_to_fill_[m] = fill_value_start
        array_to_fill_[m+cols-whalf] = fill_value_end


cdef float _calc_mean(double[::1] warray,
                      Py_ssize_t m,
                      unsigned int window_size,
                      double[::1] weights,
                      float no_data) nogil:

    cdef:
        Py_ssize_t n
        float w
        float window_sum = 0.0
        float weights_sum = 0.0

    for n in range(0, window_size):

        if warray[m+n] != no_data:

            w = weights[n]
            window_sum += warray[m+n]*w
            weights_sum += w

    if weights_sum == 0:
        return no_data
    else:
        return window_sum / weights_sum


cdef float _calc_min(double[::1] warray,
                     Py_ssize_t m,
                     unsigned int window_size,
                     float no_data) nogil:

    cdef:
        Py_ssize_t n
        float window_value
        float window_min = 1e9

    for n in range(0, window_size):

        if warray[m+n] != no_data:

            window_value = warray[m+n]

            if window_value < window_min:
                window_min = window_value

    if window_min == 1e9:
        return no_data
    else:
        return window_min


cdef float _calc_max(double[::1] warray,
                     Py_ssize_t m,
                     unsigned int window_size,
                     float no_data) nogil:

    cdef:
        Py_ssize_t n
        float window_value
        float window_max = -1000000.0

    for n in range(0, window_size):

        if warray[m+n] != no_data:

            window_value = warray[m+n]

            if window_value > window_max:
                window_max = window_value

    if window_max == -1000000.0:
        return no_data
    else:
        return window_max


cdef float _calc_std(double[::1] warray,
                     Py_ssize_t m,
                     unsigned int window_size,
                     float wmean,
                     float no_data) nogil:

    cdef:
        Py_ssize_t n
        float sq_dev
        float sum_sq = 0.0
        float sqn #= float(window_size) - 1.0
        Py_ssize_t window_count = 0

    for n in range(0, window_size):

        if warray[m+n] != no_data:

            # Calculate the squared deviation from the mean
            sq_dev = common.squared_diff(warray[m+n], wmean)

            sum_sq += sq_dev

            window_count += 1

    if window_count == 0:
        return no_data
    else:

        sqn = float(window_count) - 1.0

        return common.sqrt(sum_sq / sqn)


cdef void _calc_mean_std(double[::1] warray,
                         Py_ssize_t start,
                         Py_ssize_t end,
                         double[::1] mu_std_holder__) nogil:

    cdef:
        Py_ssize_t jc
        float array_sum = warray[start]
        float yvalue_, array_mean, array_std_dev
        Py_ssize_t array_count = 1
        float sq_dev, sqn
        float sum_sq = 0.0

    for jc in range(start+1, end):

        yvalue_ = warray[jc]

        if yvalue_ > 0:

            array_sum += yvalue_
            array_count += 1

    array_mean = array_sum / float(array_count)

    for jc in range(start+1, end):

        yvalue_ = warray[jc]

        if yvalue_ > 0:

            sq_dev = common.squared_diff(yvalue_, array_mean)
            sum_sq += sq_dev

    sqn = float(array_count) - 1.0

    array_std_dev = common.sqrt(sum_sq / sqn)

    mu_std_holder__[0] = array_mean
    mu_std_holder__[1] = array_std_dev


cdef void _min(double[::1] array,
               double[::1] out_array_,
               unsigned int window_size,
               unsigned int cols,
               unsigned int whalf,
               double[::1] window_weights,
               float no_data):

    cdef:
        Py_ssize_t j
        float window_min

    with nogil:

        for j in range(0, cols-window_size+1):

            window_min = _calc_min(array,
                                   j,
                                   window_size,
                                   no_data)

            out_array_[j+whalf] = window_min

        # Fill ends
        _fill_ends(out_array_, cols, whalf)


cdef void _max(double[::1] array,
               double[::1] out_array_,
               unsigned int window_size,
               unsigned int cols,
               unsigned int whalf,
               double[::1] window_weights,
               float no_data):

    cdef:
        Py_ssize_t j
        float window_max

    with nogil:

        for j in range(0, cols-window_size+1):

            window_max = _calc_max(array,
                                   j,
                                   window_size,
                                   no_data)

            out_array_[j+whalf] = window_max

        # Fill ends
        _fill_ends(out_array_, cols, whalf)


cdef void _mean(double[::1] array,
                double[::1] out_array_,
                unsigned int window_size,
                unsigned int cols,
                unsigned int whalf,
                double[::1] window_weights,
                float no_data):

    cdef:
        Py_ssize_t j
        float window_mean

    with nogil:

        for j in range(0, cols-window_size+1):

            window_mean = _calc_mean(array,
                                     j,
                                     window_size,
                                     window_weights,
                                     no_data)

            out_array_[j+whalf] = window_mean

        # Fill ends
        _fill_ends(out_array_, cols, whalf)


cdef void _std(double[::1] array,
               double[::1] out_array_,
               unsigned int window_size,
               unsigned int cols,
               unsigned int whalf,
               double[::1] window_weights,
               float no_data):

    cdef:
        Py_ssize_t j
        float window_mean, std

    with nogil:

        for j in range(0, cols-window_size+1):

            # Calculate the mean
            window_mean = _calc_mean(array,
                                     j,
                                     window_size,
                                     window_weights,
                                     no_data)

            std = _calc_std(array,
                            j,
                            window_size,
                            window_mean,
                            no_data)

            out_array_[j+whalf] = std

        # Fill ends
        _fill_ends(out_array_, cols, whalf)


cdef void _mean_std(double[::1] array,
                    double[::1] out_array_mean_,
                    double[::1] out_array_std_,
                    unsigned int window_size,
                    unsigned int cols,
                    unsigned int whalf,
                    double[::1] window_weights,
                    float no_data,
                    double[::1] mu_std_holder_):

    cdef:
        Py_ssize_t j

    with nogil:

        for j in range(0, cols-window_size+1):

            _calc_mean_std(array,
                           j,
                           j+window_size,
                           mu_std_holder_)

            # Calculate the mean
            # window_mean = _calc_mean(array,
            #                          j,
            #                          window_size,
            #                          window_weights,
            #                          no_data)

            # window_std = _calc_std(array,
            #                        j,
            #                        window_size,
            #                        window_mean,
            #                        no_data)

            out_array_mean_[j+whalf] = mu_std_holder_[0]
            out_array_std_[j+whalf] = mu_std_holder_[1]

        # Fill ends
        _fill_ends(out_array_mean_, cols, whalf)
        _fill_ends(out_array_std_, cols, whalf)


def rolling_mean_std1d(np.ndarray[double, ndim=1] array not None,
                       unsigned int window_size=3,
                       float no_data=0.0,
                       np.ndarray[double, ndim=1] window_weights=None):

    """
    Calculates the rolling mean and rolling standard deviation

    Args:
        array (1d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        Mean and Standard deviation (tuple)
    """

    cdef:
        unsigned int cols = array.shape[0]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=1] out_array_mean = array.copy()
        double[::1] out_array_mean_view = out_array_mean

        np.ndarray[DTYPE_float64_t, ndim=1] out_array_std = array.copy()
        double[::1] out_array_std_view = out_array_std

        double[::1] weights

        double[::1] mu_std_holder = np.empty(2, dtype='float64')

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    _mean_std(array,
              out_array_mean_view,
              out_array_std_view,
              window_size,
              cols,
              whalf,
              weights,
              no_data,
              mu_std_holder)

    return out_array_mean, out_array_std


def rolling_std1d(np.ndarray[DTYPE_float64_t, ndim=1] array not None,
                  unsigned int window_size=3,
                  float no_data=0.0,
                  np.ndarray[DTYPE_float64_t, ndim=1] window_weights=None):

    """
    Calculates the rolling standard deviation

    Args:
        array (1d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        (1d array)
    """

    cdef:
        unsigned int cols = array.shape[0]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=1] out_array = array.copy()
        double[::1] out_array_view = out_array

        double[::1] weights
        metric_ptr wfunc = &_std

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    wfunc(array,
          out_array_view,
          window_size,
          cols,
          whalf,
          weights,
          no_data)

    return out_array


def rolling_std2d(np.ndarray[DTYPE_float64_t, ndim=2] array not None,
                  unsigned int window_size=3,
                  float no_data=0.0,
                  np.ndarray[DTYPE_float64_t, ndim=1] window_weights=None):

    """
    Calculates the rolling standard deviation

    Args:
        array (2d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        (1d array)
    """

    cdef:
        Py_ssize_t i, j

        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=2] out_array = array.copy()
        double[:, ::1] out_array_view = out_array

        double[::1] in_array_view1d = np.zeros(cols, dtype='float64')
        double[::1] out_array_view1d = in_array_view1d.copy()

        double[::1] weights
        metric_ptr wfunc = &_std

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    for i in range(0, rows):

        for j in range(0, cols):
            in_array_view1d[j] = array[i, j]

        out_array_view1d[...] = in_array_view1d

        wfunc(in_array_view1d,
              out_array_view1d,
              window_size,
              cols,
              whalf,
              weights,
              no_data)

        for j in range(0, cols):
            out_array_view[i, j] = out_array_view1d[j]

    return out_array


def rolling_mean1d(np.ndarray[DTYPE_float64_t, ndim=1] array not None,
                   unsigned int window_size=3,
                   float no_data=0.0,
                   np.ndarray[DTYPE_float64_t, ndim=1] window_weights=None):

    """
    Calculates the rolling mean

    Args:
        array (1d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        (1d array)
    """

    cdef:
        unsigned int cols = array.shape[0]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=1] out_array = array.copy()
        double[::1] out_array_view = out_array

        double[::1] weights
        metric_ptr wfunc = &_mean

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    wfunc(array,
          out_array_view,
          window_size,
          cols,
          whalf,
          weights,
          no_data)

    return out_array


def rolling_min1d(np.ndarray[DTYPE_float64_t, ndim=1] array not None,
                  unsigned int window_size=3,
                  float no_data=0.0,
                  np.ndarray[DTYPE_float64_t, ndim=1] window_weights=None):

    """
    Calculates the rolling minimum

    Args:
        array (1d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        (1d array)
    """

    cdef:
        unsigned int cols = array.shape[0]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=1] out_array = array.copy()
        double[::1] out_array_view = out_array

        double[::1] weights
        metric_ptr wfunc = &_min

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    wfunc(array,
          out_array_view,
          window_size,
          cols,
          whalf,
          weights,
          no_data)

    return out_array


def rolling_max1d(np.ndarray[DTYPE_float64_t, ndim=1] array not None,
                  unsigned int window_size=3,
                  float no_data=0.0,
                  np.ndarray[DTYPE_float64_t, ndim=1] window_weights=None):

    """
    Calculates the rolling maximum

    Args:
        array (1d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[1d array]): Window weights.

    Returns:
        (1d array)
    """

    cdef:
        unsigned int cols = array.shape[0]
        unsigned int whalf = <int>(window_size / 2.0)

        np.ndarray[DTYPE_float64_t, ndim=1] out_array = array.copy()
        double[::1] out_array_view = out_array

        double[::1] weights
        metric_ptr wfunc = &_max

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones(window_size, dtype='float64')

    wfunc(array,
          out_array_view,
          window_size,
          cols,
          whalf,
          weights,
          no_data)

    return out_array
