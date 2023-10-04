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
from libc.stdlib cimport free, malloc

from ..utils cimport common


cdef struct LinCoeff:
    double y_mean
    double beta1
    double beta2
    double alpha


cdef inline LinCoeff _calc_wlsq(double *xdata,
                                double *ydata,
                                double xavg1,
                                double xavg2,
                                double yavg,
                                unsigned int count) nogil:

    """
    Calculates the least squares slope and intercept
    """

    cdef:
        Py_ssize_t s
        double xdev1, xdev2, ydev
        double xvar1, xvar2, x1y_cov, x2y_cov, xx_cov, numerator
        double beta1, beta2, alpha
        double yvar
        LinCoeff lin_coeff_

    xvar1 = 0.0
    xvar2 = 0.0
    yvar = 0.0
    xx_cov = 0.0
    x1y_cov = 0.0
    x2y_cov = 0.0

    for s in range(0, count):

        xdev1 = (xdata[s] - xavg1)
        xdev2 = (common.pow2(xdata[s]) - xavg2)
        ydev = (ydata[s] - yavg)

        # Variance
        xvar1 += common.pow2(xdev1)
        xvar2 += common.pow2(xdev2)
        yvar += common.pow2(ydev)

        x1y_cov += (xdev1 * ydev)
        x2y_cov += (xdev2 * ydev)

        xx_cov += (xdev1 * xdev2)

    numerator = (xvar1 * xvar2) - common.pow2(xx_cov)

    # Polynomial least squares
    beta1 = ((xvar2 * x1y_cov) - (xx_cov * x2y_cov)) / numerator
    beta2 = ((xvar1 * x2y_cov) - (xx_cov * x1y_cov)) / numerator
    alpha = yavg - (beta1 * xavg1) - (beta2 * xavg2)

    lin_coeff_.y_mean = yavg
    lin_coeff_.beta1 = beta1
    lin_coeff_.beta2 = beta2
    lin_coeff_.alpha = alpha

    return lin_coeff_


cdef inline double _get_array_std(double[:, ::1] y_array_slice,
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

    return common.sqrt(sum_sq / <double>array_count)


cdef inline void lowess(long[::1] ordinals,
                        double[:, ::1] y,
                        double[:, ::1] yout_,
                        unsigned int w,
                        Py_ssize_t idx,
                        unsigned int ncols) nogil:

    cdef:
        Py_ssize_t j, k
        double *xdata
        double *ydata
        double xvalue, yvalue
        double xavg1, xavg2, yavg
        LinCoeff lin_coeff
        double x0, x1
        double wf
        unsigned int wh = <int>(w * 0.5)

    wf = <double>w

    xdata = <double *>malloc(w * sizeof(double))
    ydata = <double *>malloc(w * sizeof(double))

    for j in range(0, ncols-w):

        xavg1 = 0.0
        xavg2 = 0.0
        yavg = 0.0

        for k in range(0, w):

            xvalue = ordinals[j+k]
            yvalue = y[idx, j+k]

            xdata[k] = xvalue
            ydata[k] = yvalue

            xavg1 += xvalue
            xavg2 += common.pow2(xvalue)
            yavg += yvalue

        xavg1 /= wf
        xavg2 /= wf
        yavg /= wf

        lin_coeff = _calc_wlsq(xdata,
                               ydata,
                               xavg1,
                               xavg2,
                               yavg,
                               w)

        x0 = xdata[wh]
        x1 = common.pow2(x0)

        yout_[idx, j+wh] = lin_coeff.beta1*x0 + lin_coeff.beta2*x1 + lin_coeff.alpha

    free(xdata)
    free(ydata)
