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

from cython.parallel import prange
from cython.parallel import parallel

from libc.stdlib cimport malloc, free

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t


cdef extern from 'math.h' nogil:
    double floor(double val)


cdef struct LinCoeff:
    double y_mean
    double beta1
    double beta2
    double alpha


#ctypedef LinCoeff (*metric_ptr)(double *, double *, double, double, double, unsigned int, double *) nogil


cdef inline Py_ssize_t _get_rindex(Py_ssize_t col_dims, Py_ssize_t index) nogil:
    return <int>floor(<double>index / <double>col_dims)


cdef inline Py_ssize_t _get_cindex(Py_ssize_t col_dims, Py_ssize_t index, Py_ssize_t row_index) nogil:
    return <int>(index - <double>col_dims * row_index)


cdef inline double _apply(LinCoeff results_,
                          double center_avg,
                          double nodata,
                          double dev_thresh) nogil:

    """
    Applies the least squares solution to estimate the missing value
    """

    cdef:
        double estimate

    # y = b0*x + a
    #estimate = results_.beta1*center_avg + results_.alpha

    # y = b0*x + b1*x^2 + a
    estimate = results_.beta1*center_avg + results_.beta2*center_avg**2 + results_.alpha

    if estimate > 1:
        estimate = nodata
    elif estimate < 0:
        estimate = nodata
    elif estimate > (results_.y_mean + dev_thresh*10.0):
        estimate = results_.y_mean
    elif estimate < (results_.y_mean - dev_thresh*10.0):
        estimate = results_.y_mean

    return estimate


cdef inline LinCoeff _calc_wlsq(double *xdata_,
                                double *ydata_,
                                double *weights_,
                                double xavg1,
                                double xavg2,
                                double yavg,
                                unsigned int count) nogil:

    """
    Calculates the least squares slope and intercept
    """

    cdef:
        Py_ssize_t s
        double x_, y_, w_
        double xdev1, xdev2, ydev
        double xvar1, xvar2, x1y_cov, x2y_cov, xx_cov, denom
        double beta1, beta2, alpha
        LinCoeff lin_coeff_

    xvar1 = 0.0
    xvar2 = 0.0
    xx_cov = 0.0
    x1y_cov = 0.0
    x2y_cov = 0.0

    for s in range(0, count):

        x_ = xdata_[s]
        y_ = ydata_[s]
        w_ = weights_[s]

        xdev1 = w_ * (x_ - xavg1)
        xdev2 = w_ * (x_**2 - xavg2)
        ydev = w_ * (y_ - yavg)

        # Variance
        xvar1 += xdev1**2
        xvar2 += xdev2**2

        x1y_cov += (xdev1 * ydev)
        x2y_cov += (xdev2 * ydev)

        xx_cov += (xdev1 * xdev2)

    denom = (xvar1 * xvar2) - xx_cov**2

    # Polynomial least squares
    beta1 = ((xvar2 * x1y_cov) - (xx_cov * x2y_cov)) / denom
    beta2 = ((xvar1 * x2y_cov) - (xx_cov * x1y_cov)) / denom
    alpha = yavg - (beta1 * xavg1) - (beta2 * xavg2)

    # Linear least squares
    #beta1 = x1y_cov / xvar1
    #beta2 = 0.0
    #alpha = yavg - beta1 * xavg1

    #rsquared_num = 0.0
    #rsquared_denom = 0.0

    # Calculate the r squared
    #for m in range(0, count):

    #    yhat = beta1*xdata[m] + beta2*common.pow2(xdata[m]) + alpha
    #    rsquared_num += common.pow2(yhat - yavg)
    #    rsquared_denom += common.pow2(ydata[m] - yavg)

    #rsquared = rsquared_num / rsquared_denom

    lin_coeff_.y_mean = yavg
    lin_coeff_.beta1 = beta1
    lin_coeff_.beta2 = beta2
    lin_coeff_.alpha = alpha

    return lin_coeff_


cdef inline void _get_center_mean(double[:, :, ::1] X,
                                  long[:, ::1] clusters,
                                  long[::1] lc_classes,
                                  double[:, ::1] x_cluster_,
                                  Py_ssize_t fff,
                                  Py_ssize_t b,
                                  Py_ssize_t iii,
                                  Py_ssize_t jjj,
                                  Py_ssize_t ci,
                                  unsigned int hw,
                                  double nodata) nogil:

    """
    Gets the center average of the reference data
    """

    cdef:
        Py_ssize_t m, n, cidx
        unsigned int offset = hw - <int>(ci / 2.0)
        double center_sum
        double ref_xvalue
        int ref_xclass, ref_xclass_spass
        Py_ssize_t count

    center_sum = 0.0
    count = 0

    for m in range(0, ci):
        for n in range(0, ci):

            # Reference X value
            ref_xvalue = X[b, iii+m+offset, jjj+n+offset]

            # Reference X class
            ref_xclass = clusters[iii+m+offset, jjj+n+offset]

            # Update the class count
            # [X mean, X class, mode holder, <class counts>]
            x_cluster_[fff, 3+ref_xclass] += 1.0

            if ref_xvalue != nodata:

                center_sum += ref_xvalue
                count += 1

    if center_sum > 0:

        # Iterate over the unique set of class values
        for cidx in range(0, lc_classes.shape[0]):

            # Get the class with the highest count
            # [X mean, X class, mode holder, <class counts>]
            if x_cluster_[fff, 3+cidx] > x_cluster_[fff, 2]:

                # Update the mode class value
                # [X mean, X class, mode holder, <class counts>]
                x_cluster_[fff, 1] = lc_classes[cidx]

                # Update the class mode
                # [X mean, X class, mode holder, <class counts>]
                x_cluster_[fff, 2] = x_cluster_[fff, 3+cidx]

        # [X mean, X class, mode holder, <class counts>]
        x_cluster_[fff, 0] = center_sum / <double>count

        # Update the average for the target class

        center_sum = 0.0
        count = 0

        ref_xclass = <int>x_cluster_[fff, 1]

        for m in range(0, ci):
            for n in range(0, ci):

                # Reference X value
                ref_xvalue = X[b, iii+m+offset, jjj+n+offset]

                # Reference X class
                ref_xclass_spass = clusters[iii+m+offset, jjj+n+offset]

                if (ref_xvalue != nodata) and (ref_xclass_spass == ref_xclass):

                    center_sum += ref_xvalue
                    count += 1

        if count > 0:
            x_cluster_[fff, 0] = center_sum / <double>count


cdef inline void _estimate_gap(double[:, :, ::1] X,
                               double[:, :, ::1] y,
                               long[:, ::1] clusters,
                               double[:, :, ::1] output__,
                               Py_ssize_t ffff,
                               Py_ssize_t b,
                               Py_ssize_t iiii,
                               Py_ssize_t jjjj,
                               unsigned int wmin,
                               unsigned int wmax,
                               unsigned int hw,
                               double nodata,
                               unsigned int min_thresh,
                               double[:, ::1] x_cluster__,
                               double dev_thresh,
                               double[:, ::1] spatial_weights,
                               double[:, ::1] day_weights) nogil:

    cdef:
        Py_ssize_t m, n, rzi, wi
        unsigned int offset

        double xvalue, yvalue
        Py_ssize_t count

        double xavg1, xavg2, yavg
        double sq_diff
        double estimate

        double *xdata
        double *ydata
        double *weights

        LinCoeff lin_coeff

        unsigned int max_vct_size = <int>(wmax*wmax)

        int ref_xclass
        int tar_xclass = <int>x_cluster__[ffff, 1]

        #metric_ptr wlsq_func

    #wlsq_func = &_calc_wlsq

    # Maximum possible size of the arrays
    xdata = <double *>malloc(max_vct_size * sizeof(double))
    ydata = <double *>malloc(max_vct_size * sizeof(double))
    weights = <double *>malloc(max_vct_size * sizeof(double))

    for rzi in range(0, max_vct_size):
        xdata[rzi] = 0.0
        ydata[rzi] = 0.0

    # Linear regression coefficients
    # [x_mean, x_std, beta_0, beta_1, alpha, rsquared]
    #stdv = <double *>malloc(6 * sizeof(double))

    xavg1 = 0.0
    xavg2 = 0.0
    yavg = 0.0

    count = 0

    # Search for data over increasing windows
    for wi from wmin <= wi < wmax+2 by 2:

        # Adjustment for different sized windows
        offset = hw - <int>(wi / 2.0)

        for m in range(0, wi):
            for n in range(0, wi):

                # Skip the center that has already been checked
                if wi > wmin:

                    if (m > 0) and (m < wi-1) and (n > 0) and (n < wi-1):
                        continue

                # Get the reference cluster value
                ref_xclass = clusters[iiii+m+offset, jjjj+n+offset]

                # Only take samples with the same cluster as the target
                if ref_xclass == tar_xclass:

                    # Target value
                    yvalue = y[b, iiii+m+offset, jjjj+n+offset]

                    # Reference value
                    xvalue = X[b, iiii+m+offset, jjjj+n+offset]

                    # X and y must have data
                    if (xvalue != nodata) and (yvalue != nodata):

                        # Squared difference
                        sq_diff = (xvalue - yvalue)**2

                        # X and y must have a low deviation
                        if sq_diff <= dev_thresh:

                            xdata[count] = xvalue
                            ydata[count] = yvalue
                            weights[count] = spatial_weights[m, n] * day_weights[iiii+m+offset, jjjj+n+offset]

                            xavg1 += xvalue
                            xavg2 += xvalue**2
                            yavg += yvalue

                            count += 1

        if count >= min_thresh:
            break

    if count >= min_thresh:

        countf = <double>count

        # Window average
        xavg1 /= countf
        xavg2 /= countf
        yavg /= countf

        # Std. dev. of [x, y], slope, intercept
        lin_coeff = _calc_wlsq(xdata,
                               ydata,
                               weights,
                               xavg1,
                               xavg2,
                               yavg,
                               count)

        # Calculate the least squares solution
        estimate = _apply(lin_coeff,
                          x_cluster__[ffff, 0],
                          nodata,
                          dev_thresh)

        output__[b, iiii+hw, jjjj+hw] = estimate

    free(xdata)
    free(ydata)
    free(weights)


#cdef inline void _center_mean_iter(Py_ssize_t ff,
#                                   Py_ssize_t b,
#                                   Py_ssize_t ii,
#                                   Py_ssize_t jj,
#                                   unsigned int hw,
#                                   double nodata,
#                                   double[:, :, ::1] X,
#                                   double[:, :, ::1] y,
#                                   long[:, ::1] clusters,
#                                   long[::1] lc_classes,
#                                   double[:, ::1] x_cluster_) nogil:
#
#    # Get an average of the center value
#    #for ci from 3 <= ci < 9 by 2:
#
#    # center_avg = X
#    _get_center_mean(X, clusters, lc_classes, x_cluster_, ff, b, ii, jj, 3, hw, nodata)


cdef inline void _fill_gap(Py_ssize_t ff,
                           double[:, :, ::1] X,
                           double[:, :, ::1] y,
                           long[:, ::1] clusters,
                           long[::1] lc_classes,
                           double[:, ::1] day_weights,
                           double[:, :, ::1] output_,
                           Py_ssize_t b,
                           unsigned int col_dims,
                           unsigned int wmin,
                           unsigned int wmax,
                           unsigned int hw,
                           double nodata,
                           unsigned int min_count,
                           double dev_thresh,
                           double[:, ::1] spatial_weights,
                           double[:, ::1] x_cluster) nogil:

    cdef:
        Py_ssize_t ii, jj, ci, wi, xxi

    ii = _get_rindex(col_dims, ff)
    jj = _get_cindex(col_dims, ff, ii)

    # Center target sample
    if y[b, ii+hw, jj+hw] == nodata:

        _get_center_mean(X, clusters, lc_classes, x_cluster, ff, b, ii, jj, 3, hw, nodata)

#        _center_mean_iter(ff,
#                          b,
#                          ii,
#                          jj,
#                          hw,
#                          nodata,
#                          X,
#                          y,
#                          clusters,
#                          lc_classes,
#                          x_cluster)

        if x_cluster[ff, 0] > 0:

            _estimate_gap(X,
                          y,
                          clusters,
                          output_,
                          ff,
                          b,
                          ii,
                          jj,
                          wmin,
                          wmax,
                          hw,
                          nodata,
                          min_count,
                          x_cluster,
                          dev_thresh,
                          spatial_weights,
                          day_weights)


cdef double[:, :, ::1] _fill_gaps(double[:, :, ::1] X,
                                  double[:, :, ::1] y,
                                  long[:, ::1] clusters,
                                  long[::1] lc_classes,
                                  double[:, ::1] day_weights,
                                  double[:, :, ::1] output,
                                  unsigned int wmax,
                                  unsigned int wmin,
                                  double nodata,
                                  unsigned int min_count,
                                  double dev_thresh,
                                  unsigned int n_jobs,
                                  unsigned int chunksize):

    cdef:
        Py_ssize_t b, i, j, f, mm, nn
        unsigned int bands = y.shape[0]
        unsigned int rows = y.shape[1]
        unsigned int cols = y.shape[2]
        unsigned int hw = <int>(wmax / 2.0)
        unsigned int row_dims = rows - <int>(hw*2.0)
        unsigned int col_dims = cols - <int>(hw*2.0)
        unsigned int nsamples = <int>(row_dims * col_dims)
        double[:, ::1] spatial_weights = np.zeros((wmax, wmax), dtype='float64')
        double max_edist

        # [X mean, X class, mode holder, <class counts>]
        double[:, ::1] x_cluster = np.zeros((nsamples, 3+lc_classes.shape[0]), dtype='float64')

    with nogil:

        max_edist = common.edist(0.0, 0.0, hw) + 0.001

        # Fill the spatial weights
        for mm in range(0, wmax):
            for nn in range(0, wmax):
                spatial_weights[mm, nn] = common.logistic_func(1.0 - ((common.edist(<double>nn, <double>mm, hw)+0.001) / max_edist), 0.5, 10.0)

    with nogil, parallel(num_threads=n_jobs):

        for f in prange(0, nsamples, schedule='static', chunksize=chunksize):

            for b in range(0, bands):

                _fill_gap(f,
                          X,
                          y,
                          clusters,
                          lc_classes,
                          day_weights,
                          output,
                          b,
                          col_dims,
                          wmin,
                          wmax,
                          hw,
                          nodata,
                          min_count,
                          dev_thresh,
                          spatial_weights,
                          x_cluster)

    return output


def fill_gaps(np.ndarray[DTYPE_float64_t, ndim=3] X not None,
              np.ndarray[DTYPE_float64_t, ndim=3] y not None,
              np.ndarray[DTYPE_int64_t, ndim=2] cluster_data not None,
              np.ndarray[DTYPE_float64_t, ndim=2] day_weights not None,
              wmax=25,
              wmin=9,
              nodata=0,
              min_count=20,
              dev_thresh=0.02,
              n_jobs=1,
              chunksize=1):

    """
    Fills data gaps using spatial-temporal weighted least squares linear regression

    Args:
        X (3d array): Bands x rows x columns. The reference layer.
        y (3d array): Bands x rows x columns. The target layer.
        cluster_data (2d array): Land cover clusters.
        day_weights (2d array): Day difference weights.
        wmax (Optional[int]): The maximum window size.
        wmin (Optional[int]): The minimum window size.
        nodata (Optional[int]): The 'no data' value to fill.
        min_count (Optional[int]): The minimum required number of samples to fit a window.
        dev_thresh (Optional[float]): The squared deviation threshold for similar values.
        n_jobs (Optional[int]): The number of bands to process in parallel.
        chunksize (Optional[int]): The parallel thread chunksize.

    Returns:
        3d ``numpy.ndarray`` (bands x rows x columns)
    """

    cdef:
        double[:, :, ::1] output = y.copy()
        long[::1] lc_classes = np.unique(cluster_data).astype('int64')

    return np.float64(_fill_gaps(X,
                                 y,
                                 cluster_data,
                                 lc_classes,
                                 day_weights,
                                 output,
                                 wmax,
                                 wmin,
                                 nodata,
                                 min_count,
                                 dev_thresh,
                                 n_jobs,
                                 chunksize))
