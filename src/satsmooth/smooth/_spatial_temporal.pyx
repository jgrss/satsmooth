# distutils: language=c++
# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython

cimport cython

import numpy as np
from cython.parallel import parallel, prange

cimport numpy as np

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

from ..utils cimport common

#ctypedef double (*metric_ptr)(double[:, :, ::1], Py_ssize_t, Py_ssize_t, Py_ssize_t, unsigned int, unsigned int, unsigned int) nogil


cdef inline int _get_rindex(int col_dims, Py_ssize_t index) nogil:
    return <int>common.floor(<double>index / <double>col_dims)


cdef inline int _get_cindex(int col_dims, Py_ssize_t index, int row_index) nogil:
    return <int>(index - <double>col_dims * row_index)


cdef inline int _abs_round(double v1, double v2) nogil:
    return <int>common.round(common.fabs((v1 - v2) * 100.0))


cdef inline unsigned int _add_int(unsigned int v1, unsigned int v2) nogil:
    return v1 + v2


cdef inline double _add_double(double v1, double v2) nogil:
    return v1 + v2


cdef inline double _multiply_double(double v1, double v2) nogil:
    return v1 * v2


cdef inline double _set_weight(double w1, double w2, double w3, double w4) nogil:
    return w1 + w2 + w3 + w4


cdef inline double _bright_weight(double center_dn, double edge_dn) nogil:
    return 1.0 if center_dn >= edge_dn else 0.75


cdef inline double _euclidean_distance(double x1, double y1, double x2, double y2) nogil:
    return common.sqrt(common.pow2(x1 - x2) + common.pow2(y1 - y2))


#cdef double _window_mean(double[:, :, ::1] narray,
#                         Py_ssize_t z,
#                         Py_ssize_t i,
#                         Py_ssize_t j,
#                         unsigned int k_half,
#                         unsigned int dims,
#                         unsigned int t) nogil:
#
#    cdef:
#        Py_ssize_t ww
#        double window_mean = narray[z, i+k_half, j+k_half]
#
#    for ww in range(1, common.min_value_int(dims-t-z+1, 60)):
#        window_mean += narray[z+ww, i+k_half, j+k_half]
#
#    return window_mean / <double>common.min_value_int(dims-t-z+1, 60)


cdef double _stretch(double[:, :, ::1] nd_array,
                     Py_ssize_t z,
                     Py_ssize_t i,
                     Py_ssize_t j,
                     Py_ssize_t k_half,
                     Py_ssize_t t_half,
                     unsigned int t,
                     unsigned int ncols,
                     double min_valley_weight,
                     double max_peak_weight) nogil:

    """
    Calculate the stretch value from the average gradient of 3 consecutive points
    """

    cdef:
        double bc_delta1, bc_delta2, bc_delta_avg
        double bc_delta_avg1, bc_delta_avg2, bc_delta_avg3

    # Window tails
    if (z == 0) or (z + t >= ncols):

        # Proportional difference between windows half value and tails
        bc_delta1 = common.prop_diff(nd_array[z, i+k_half, j+k_half], nd_array[z+t_half, i+k_half, j+k_half])
        bc_delta2 = common.prop_diff(nd_array[z+t-1, i+k_half, j+k_half], nd_array[z+t_half, i+k_half, j+k_half])

        # Average of the two
        bc_delta_avg = (bc_delta1 + bc_delta2) / 2.0

    else:

        bc_delta1 = common.prop_diff(nd_array[z-1, i+k_half, j+k_half], nd_array[z-1+t_half, i+k_half, j+k_half])
        bc_delta2 = common.prop_diff(nd_array[z-1+t-1, i+k_half, j+k_half], nd_array[z-1+t_half, i+k_half, j+k_half])
        bc_delta_avg1 = (bc_delta1 + bc_delta2) / 2.0

        bc_delta1 = common.prop_diff(nd_array[z, i+k_half, j+k_half], nd_array[z+t_half, i+k_half, j+k_half])
        bc_delta2 = common.prop_diff(nd_array[z+t-1, i+k_half, j+k_half], nd_array[z+t_half, i+k_half, j+k_half])
        bc_delta_avg2 = (bc_delta1 + bc_delta2) / 2.0

        bc_delta1 = common.prop_diff(nd_array[z+1, i+k_half, j+k_half], nd_array[z+1+t_half, i+k_half, j+k_half])
        bc_delta2 = common.prop_diff(nd_array[z+1+t-1, i+k_half, j+k_half], nd_array[z+1+t_half, i+k_half, j+k_half])
        bc_delta_avg3 = (bc_delta1 + bc_delta2) / 2.0

        bc_delta_avg = (bc_delta_avg1 + bc_delta_avg2 + bc_delta_avg3) / 3.0

    return common.scale_min_max(common.clip(bc_delta_avg, -0.4, 0.4), min_valley_weight, max_peak_weight, -0.4, 0.4)


cdef double[:, :, ::1] _spatial_temporal(double[:, :, ::1] nd_array,
                                         unsigned int k,
                                         unsigned int t,
                                         double sigma_time,
                                         double sigma_color,
                                         double sigma_space,
                                         unsigned int n_jobs,
                                         unsigned int chunksize,
                                         unsigned int n_iters,
                                         double max_peak_weight,
                                         double min_valley_weight):

    cdef:
        Py_ssize_t f, i, j, z, wt, ct, i_, j_, zz, ii, jj
        Py_ssize_t iter_
        unsigned int space_counter

        double gt, gi, gs, gb, bc, bv, w
        double weighted_mean, weights_sum

        unsigned int k_half = <int>(k / 2.0)
        unsigned int t_half = <int>(t / 2.0)
        unsigned int dims = nd_array.shape[0]
        unsigned int rows = nd_array.shape[1]
        unsigned int cols = nd_array.shape[2]

        unsigned int row_dims = rows - <int>(k_half*2.0)
        unsigned int col_dims = cols - <int>(k_half*2.0)

        unsigned int nsamples = <int>(row_dims * col_dims)

        double[:, :, ::1] out_array = nd_array.copy()

        double[::1] time_weights = np.float64(np.linspace(-1, 1, t))
        double[::1] space_weights = np.zeros(k*k, dtype='float64')
        double[::1] color_weights = np.float64(np.linspace(0, 1, 101))

        double[::1] linspace_view = time_weights.copy()

        double max_dist, yhat#, bce

        #metric_ptr ptr_func

    #ptr_func = &_window_mean

    with nogil:

        max_dist = _euclidean_distance(<double>(k - 1), 0.0, <double>(k_half), <double>(k_half))

        # Set the time weights.
        for wt in range(0, t):
            time_weights[wt] = common.gaussian_func(linspace_view[wt], sigma_time)

        # Set the space weights.
        space_counter = 0
        for ii in range(0, k):

            for jj in range(0, k):

                space_weights[space_counter] = common.gaussian_func(_euclidean_distance(<double>(jj),
                                                                             <double>(ii),
                                                                             <double>(k_half),
                                                                             <double>(k_half)) / max_dist,
                                                         sigma_space)

                space_counter += 1

        # Set the color weights.
        for ct in range(0, 101):
            color_weights[ct] = common.gaussian_func(color_weights[ct], sigma_color)

    for iter_ in range(0, n_iters):

        with nogil, parallel(num_threads=n_jobs):

            for f in prange(0, nsamples, schedule='static', chunksize=chunksize):

                i = _get_rindex(col_dims, f)
                j = _get_cindex(col_dims, f, i)

                for z in range(0, dims-t+1):

                    weighted_mean = 0.0
                    weights_sum = 0.0

                    # Get the current block center value
                    bc = nd_array[z+t_half, i+k_half, j+k_half]

                    if common.npy_isnan(bc) or common.npy_isinf(bc):
                        bc = 0.0

                    # Gradient between window center and start
                    #bce = _stretch(nd_array, z, i, j, k_half, t_half, t, cols, min_valley_weight, max_peak_weight)

                    for zz in range(0, t):

                        # time distance
                        gt = time_weights[zz]

                        space_counter = 0

                        for ii in range(0, k):

                            for jj in range(0, k):

                                # Current block value
                                bv = nd_array[z+zz, i+ii, j+jj]

                                # color distance
                                # current value - center value
                                gi = color_weights[_abs_round(bv, bc)]

                                # Reduce the weight of lower values.
                                gb = _bright_weight(bc, bv)

                                # space distance
                                gs = space_weights[space_counter]

                                # weight
                                # w = temporal distance + color distance + low values + spatial distance
                                w = _set_weight(gt, gi, gb, gs)

                                # Update the counter and weights.
                                space_counter = _add_int(space_counter, 1)
                                weighted_mean = _add_double(weighted_mean, bv*w)
                                weights_sum = _add_double(weights_sum, w)

                    if weighted_mean > 0:

                        yhat = weighted_mean / weights_sum

                        #if yhat*bce <= 0:
                        out_array[z+t_half, i+k_half, j+k_half] = yhat
                        #else:
                        #    out_array[z+t_half, i+k_half, j+k_half] = yhat * bce

        if n_iters > 1:

            with nogil, parallel(num_threads=n_jobs):

                for f in prange(0, nsamples, schedule='static', chunksize=chunksize):

                    i = _get_rindex(cols, f)
                    j = _get_cindex(cols, f, i)

                    for z in range(0, dims):
                        nd_array[z, i, j] = out_array[z, i, j]

    return out_array


def spatial_temporal(np.ndarray[DTYPE_float64_t, ndim=3] nd_array not None,
                     unsigned int k=3,
                     unsigned int t=7,
                     double sigma_time=0.1,
                     double sigma_color=0.1,
                     double sigma_space=0.1,
                     unsigned int n_jobs=1,
                     unsigned int chunksize=1,
                     unsigned int n_iters=1,
                     double max_peak_weight=1.2,
                     double min_valley_weight=0.8):

    """
    Smooths a time series using a spatial-temporal trilateral filter with Gaussian weights

    Args:
        nd_array (3d array): The data to smooth, shaped as [time x rows x columns] and scaled 0-1.
        k (Optional[int]): The spatial window size.
        t (Optional[int]): The temporal window size.
        sigma_time (Optional[float]): The temporal sigma.
        sigma_color (Optional[float]): The color sigma.
        sigma_space (Optional[float]): The space sigma.
        n_jobs (Optional[int]): The number of parallel jobs.
        chunksize (Optional[int]): The parallel thread chunksize.
        n_iters (Optional[int]): The number of iterations.
        max_peak_weight (Optional[float]): The peak weight.
        min_valley_weight (Optional[float]): The valley weight.

    Returns:
        Smoothed data (3d array)
    """

    return np.float64(_spatial_temporal(np.ascontiguousarray(nd_array, dtype='float64'),
                                        k,
                                        t,
                                        sigma_time,
                                        sigma_color,
                                        sigma_space,
                                        n_jobs,
                                        chunksize,
                                        n_iters,
                                        max_peak_weight,
                                        min_valley_weight))
