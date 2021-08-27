# distutils: language = c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

from . cimport common
from . cimport percentiles


cdef extern from 'math.h':
   double cos(double val) nogil


cdef extern from 'math.h':
   double sin(double val) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isnan(double value) nogil


cdef extern from 'numpy/npy_math.h':
    bint npy_isinf(double value) nogil


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef inline double _window_mean(double[:, ::1] data_array,
                                Py_ssize_t ipos,
                                Py_ssize_t jpos,
                                unsigned int k) nogil:

    cdef:
        Py_ssize_t q
        double dval
        double wmean = 0.0
        Py_ssize_t ctr = 0

    for q in range(0, k):

        dval = data_array[ipos, jpos+q]

        if dval > 0:

            wmean += dval
            ctr += 1

    return wmean / <double>ctr


cdef inline vector[double] _window_stats(double[:, ::1] data_array,
                                         Py_ssize_t ipos,
                                         Py_ssize_t jpos,
                                         unsigned int k) nogil:

    cdef:
        Py_ssize_t q
        double dval
        double wmean = 0.0
        double wmin = 1e9
        double wmax = -1e9
        Py_ssize_t ctr = 0
        vector[double] array_stats_

    for q in range(0, k):

        dval = data_array[ipos, jpos+q]

        if dval > 0:

            if dval < wmin:
                wmin = dval

            if dval > wmax:
                wmax = dval

            wmean += dval

            ctr += 1

    if wmin < 0.0:
        wmin = 0.0

    if wmax > 1.0:
        wmax = 1.0

    array_stats_.push_back(wmin)
    array_stats_.push_back(wmax)
    array_stats_.push_back(wmean / <double>ctr)

    return array_stats_


cdef inline void _fit_func(double[:, ::1] in_array,
                           double[:, ::1] out_array_harmonics___,
                           unsigned int period_int,
                           Py_ssize_t row_index_pos,
                           Py_ssize_t jindex,
                           unsigned int ncols,
                           double w,
                           double padding) nogil:

    cdef:
        Py_ssize_t j1, j2, j3, j4
        double x1_mean, x2_mean
        double x1_dev, x2_dev, y_dev
        double x1_var, x2_var
        double x1y_cov, x2y_cov
        double y_mean, x1x2_cov
        double alpha, beta0, beta1
        double numerator
        double yhat, yhat_sc
        unsigned int counter = 0
        #vector[double] array_stats_raw, array_stats_har
        double y_min, y_max, yhat_min, yhat_max, yhat_mean
        double piece_diff

    y_min = percentiles.get_perc(in_array, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1), 0.0, 10.0)
    y_max = percentiles.get_perc(in_array, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1), 0.0, 90.0)
    y_mean = percentiles.get_perc(in_array, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1), 0.0, 66.0)

    if y_max > 0:

        x1_mean = 0.0
        x1_var = 0.0
        x1y_cov = 0.0

        x2_mean = 0.0
        x2_var = 0.0
        x2y_cov = 0.0

        y_mean = 0.0
        x1x2_cov = 0.0

        # Get the data means
        for j1 in range(jindex, jindex+common.min_value_int(period_int, ncols-jindex-1)):

            x1_mean += cos(w*(<double>j1+1))
            x2_mean += sin(w*(<double>j1+1))

            counter += 1

        x1_mean /= <double>counter
        x2_mean /= <double>counter

        #array_stats_raw = _window_stats(in_array, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1))

        #y_min = array_stats_raw[0]
        #y_max = array_stats_raw[1]
        #y_mean = array_stats_raw[2]

        for j2 in range(jindex, jindex+common.min_value_int(period_int, ncols-jindex-1)):

            if in_array[row_index_pos, j2] > 0:

                x1_dev = cos(w*(<double>j2+1)) - x1_mean
                x2_dev = sin(w*(<double>j2+1)) - x2_mean

                y_dev = in_array[row_index_pos, j2] - y_mean

                x1_var += common.pow2(x1_dev)
                x1y_cov += (x1_dev * y_dev)

                x2_var += common.pow2(x2_dev)
                x2y_cov += (x2_dev * y_dev)

                x1x2_cov += (x1_dev * x2_dev)

        numerator = (x1_var * x2_var) - common.pow2(x1x2_cov)

        beta0 = ((x2_var * x1y_cov) - (x1x2_cov * x2y_cov)) / numerator
        beta1 = ((x1_var * x2y_cov) - (x1x2_cov * x1y_cov)) / numerator
        alpha = y_mean - (beta0 * x1_mean) - (beta1 * x2_mean)

        piece_diff = 0.0

        for j3 in range(jindex, jindex+common.min_value_int(period_int, ncols-jindex-1)):

            # Get the harmonic
            yhat = common.clip(beta0*cos(w*(<double>j3+1)) + beta1*sin(w*(<double>j3+1)) + alpha, 0.0, 1.0)

            if npy_isnan(yhat) or npy_isinf(yhat):
                yhat = 0.0

            if (jindex > padding) and (j3 == jindex):
                piece_diff = out_array_harmonics___[row_index_pos, jindex-1] - yhat

            out_array_harmonics___[row_index_pos, j3] = yhat + piece_diff

        #array_stats_har = _window_stats(out_array_harmonics___, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1))

        #yhat_min = array_stats_har[0]
        #yhat_max = array_stats_har[1]
        #yhat_mean = array_stats_har[2]

        #yhat_min = percentiles.get_perc(out_array_harmonics___, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1), 0.0, 10.0)
        yhat_max = percentiles.get_perc(out_array_harmonics___, row_index_pos, jindex, jindex+common.min_value_int(period_int, ncols-jindex-1), 0.0, 90.0)

        # Scale to the data
        for j4 in range(jindex, jindex+common.min_value_int(period_int, ncols-jindex-1)):

            yhat = out_array_harmonics___[row_index_pos, j4]

            yhat_sc = common.scale_min_max(yhat, y_min, y_max, y_min, yhat_max)

            if npy_isnan(yhat_sc) or npy_isinf(yhat_sc):
                yhat_sc = 0.0

            out_array_harmonics___[row_index_pos, j4] = yhat_sc


cdef inline void fourier(double[:, ::1] in_array,
                         double[:, ::1] out_array_harmonics__,
                         Py_ssize_t row_index_pos,
                         unsigned int ncols,
                         double period=365.25,
                         unsigned int padding=122) nogil:

    """
    122 days (1 March to 1 July) of padding
    """

    cdef:
        Py_ssize_t j, jl
        unsigned int k, kh
        unsigned int period_int = <int>period
        unsigned int range_len = ncols - padding
        double w = 2.0 * 3.141592653589793 / period

    # Time series padding
    _fit_func(in_array,
              out_array_harmonics__,
              period_int,
              row_index_pos,
              0,
              padding,
              w,
              padding)

    # Iterate over each annual cycle
    for j from padding <= j < ncols-padding by period_int:
    #for j in range(padding, range_len, period_int):

        _fit_func(in_array,
                  out_array_harmonics__,
                  period_int,
                  row_index_pos,
                  j,
                  ncols-padding,
                  w,
                  padding)

    # Time series padding
    _fit_func(in_array,
              out_array_harmonics__,
              period_int,
              row_index_pos,
              ncols-padding,
              ncols,
              w,
              padding)

    # Smooth the cycle transitions
    k = 61
    kh = <int>(k / 2.0)

    for j in range(0, ncols-k+1):
        out_array_harmonics__[row_index_pos, j+kh] = _window_mean(out_array_harmonics__, row_index_pos, j, k)

    for jl in range(j+1, ncols):
        out_array_harmonics__[row_index_pos, jl] = out_array_harmonics__[row_index_pos, j]
