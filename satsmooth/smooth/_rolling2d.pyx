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

from cython.parallel import prange
from cython.parallel import parallel

from ..utils cimport percentiles


# Define a function pointer to a metric.
ctypedef double (*metric_ptr)(double[:, ::1], Py_ssize_t, Py_ssize_t, unsigned int, double, double[::1]) nogil


cdef double _window_mean(double[:, ::1] array,
                         Py_ssize_t row_index,
                         Py_ssize_t col_index,
                         unsigned int w,
                         double no_data_value,
                         double[::1] weights) nogil:

    cdef:
        Py_ssize_t k
        double dvalue
        double weights_sum = 0.0
        double wsum = 0.0
        double wv

    for k in range(0, w):

        dvalue = array[row_index, col_index+k]

        if dvalue != no_data_value:

            wv = weights[k]

            wsum += (dvalue * wv)
            weights_sum += wv

    if weights_sum > 0:
        return wsum / weights_sum
    else:
        return no_data_value


cdef inline void _rolling_mean1d(double[:, ::1] array,
                                 double[:, ::1] out_array_,
                                 Py_ssize_t ii,
                                 unsigned int ncols,
                                 unsigned int w,
                                 unsigned int wh,
                                 double no_data_value,
                                 double[::1] weights) nogil:

    cdef:
        Py_ssize_t j
        metric_ptr wfunc

    wfunc = &_window_mean

    for j in range(0, ncols-w):
        out_array_[ii, j+wh] = wfunc(array, ii, j, w, no_data_value, weights)


cdef inline void _rolling_q1d(double[:, ::1] array,
                              double[:, ::1] out_array_,
                              Py_ssize_t ii,
                              unsigned int ncols,
                              unsigned int w,
                              unsigned int wh,
                              double no_data_value,
                              double q) nogil:

    cdef:
        Py_ssize_t j

    for j in range(0, ncols-w):
        out_array_[ii, j+wh] = percentiles.get_perc(array, ii, j, j+w, no_data_value, q)


def rolling_mean2d(double[:, ::1] array,
                   unsigned int w=3,
                   double no_data_value=0.0,
                   double[::1] weights=None,
                   unsigned int n_jobs=1,
                   unsigned int chunksize=10):

    """
    Calculates the rolling mean

    Args:
        array (2d array): The data to process, shaped [samples x time].
        w (Optional[int]): The window size.
        no_data_value (Optional[float | int]): The 'no data' value.
        n_jobs (Optional[int]): The number of parallel workers.
        chunksize (Optional[int]): The parallel thread chunksize.

    Returns:
        2d array
    """

    cdef:
        Py_ssize_t i
        unsigned int nrows = array.shape[0]
        unsigned int ncols = array.shape[1]
        unsigned int wh = <int>(w / 2.0)

        double[:, ::1] out_array = array.copy()

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, nrows, schedule='static', chunksize=chunksize):

            _rolling_mean1d(array,
                            out_array,
                            i,
                            ncols,
                            w,
                            wh,
                            no_data_value,
                            weights)

    return np.float64(out_array)


def rolling_quantile2d(double[:, ::1] array,
                       unsigned int w=3,
                       double q=50.0,
                       double no_data_value=0.0,
                       unsigned int n_jobs=1,
                       unsigned int chunksize=10):

    """
    Calculates the rolling quantile

    Args:
        array (2d array): The data to process, shaped [samples x time].
        w (Optional[int]): The window size.
        q (Optional[float]): The quantile [1-99].
        no_data_value (Optional[float | int]): The 'no data' value.
        n_jobs (Optional[int]): The number of parallel workers.
        chunksize (Optional[int]): The parallel thread chunksize.

    Returns:
        2d array
    """

    cdef:
        Py_ssize_t i
        unsigned int nrows = array.shape[0]
        unsigned int ncols = array.shape[0]
        unsigned int wh = <int>(w / 2.0)

        double[:, ::1] out_array = array.copy()

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, nrows, schedule='static', chunksize=chunksize):

            _rolling_q1d(array,
                         out_array,
                         i,
                         ncols,
                         w,
                         wh,
                         no_data_value,
                         q)

    return np.float64(out_array)