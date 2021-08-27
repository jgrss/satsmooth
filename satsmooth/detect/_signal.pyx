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
from cython.parallel import prange
from cython.parallel import parallel

import numpy as np
cimport numpy as np

from ..utils cimport common

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

DTYPE_float64 = np.float64
ctypedef np.float64_t DTYPE_float64_t

DTYPE_uint64 = np.uint64
ctypedef np.uint64_t DTYPE_uint64_t

DTYPE_int64 = np.int64
ctypedef np.int64_t DTYPE_int64_t


cdef extern from 'math.h':
   double ceil(double a) nogil


cdef inline double _round(double value, double factor) nogil:
    return ceil(value*factor) / factor


cdef double _get_min1d(double[::1] array, Py_ssize_t start, Py_ssize_t end) nogil:

    cdef:
        Py_ssize_t m
        double min_value = 1e9

    for m in range(start, end):

        if array[m] < min_value:
            min_value = array[m]

    return min_value


cdef double _get_min2d(double[:, ::1] array, Py_ssize_t ii, Py_ssize_t start, Py_ssize_t end) nogil:

    cdef:
        Py_ssize_t m
        double min_value = 1e9

    for m in range(start, end):

        if array[ii, m] < min_value:
            min_value = array[ii, m]

    return min_value


cdef Py_ssize_t _get_max_index1d(double[::1] array, Py_ssize_t start, Py_ssize_t end) nogil:

    cdef:
        Py_ssize_t m
        double max_value = -1e9
        Py_ssize_t max_index = start

    for m in range(start, end):

        if array[m] > max_value:
            max_value = array[m]
            max_index = m

    return max_index


cdef Py_ssize_t _get_max_index2d(double[:, ::1] array, Py_ssize_t ii, Py_ssize_t start, Py_ssize_t end) nogil:

    cdef:
        Py_ssize_t m
        double max_value = -1e9
        Py_ssize_t max_index = start

    for m in range(start, end):

        if array[ii, m] > max_value:
            max_value = array[ii, m]
            max_index = m

    return max_index


cdef void _find_peaks1d(double[::1] array,
                        double[:, ::1] out_array_sub,
                        DTYPE_uint64_t[::1] out_array_npeaks_sub,
                        DTYPE_uint64_t[::1] out_array_max_peaks_sub,
                        unsigned int cols,
                        double min_value,
                        double min_dist,
                        double min_sp_dist,
                        unsigned int order) nogil:

    cdef:
        Py_ssize_t jx, j1, j2, j3, j4, j5, j6, k, l, m, jback, minp
        double signal_end, peak_value
        unsigned int total_peaks, total_valleys
        double minv = 1e9

    # First pass
    # --Locate peaks and valleys
    j1 = order
    while True:

        if j1 >= (cols - order):
            break

        # Peaks
        if (array[j1] - array[j1-order] > 0) and (array[j1] - array[j1+order] > 0):

            # Value check
            if array[j1] >= min_value:

                peak_value = array[j1]

                # Check backward if a flat peak was missed
                for jback in range(j1-1, 0, -1):

                    if out_array_sub[0, jback] == 1:
                        break

                    if out_array_sub[1, jback] == 1:
                        break

                    if array[jback] > peak_value:
                        peak_value = array[jback]
                        j1 = jback

                signal_end = float(j1 + 1 + min_dist)
                signal_end = common.min_value(signal_end, float(cols - order))

                total_peaks = 0

                # Locate the largest peak
                for jx in range(j1 + 1, <int>signal_end):

                    # Flat peaks
                    if array[jx] == array[j1]:
                        total_peaks += 1
                    elif array[jx] > array[j1]:
                        total_peaks = 0
                        j1 = jx

                if total_peaks > 0:
                    j1 += <int>(total_peaks / 2.0)

                out_array_sub[0, j1] = 1

            j1 += order

        # Valleys
        elif (array[j1] - array[j1-order] < 0) and (array[j1] - array[j1+order] < 0):

            signal_end = float(j1 + 1 + min_dist)
            signal_end = common.min_value(signal_end, float(cols - order))

            # Locate the smallest valley
            for jx in range(j1+1, <int>signal_end):

                total_valleys = 0

                if array[jx] == array[j1]:
                    total_valleys += 1
                elif array[jx] < array[j1]:
                    total_valleys = 0
                    j1 = jx

            if total_valleys > 0:
                j1 += <int>(total_valleys / 2.0)

            out_array_sub[1, j1] = 1

            j1 += order

        else:
            j1 += 1

    # Second pass
    # --Check for small amplitudes
    for j2 in range(order, cols-order):

        # Valley?
        if out_array_sub[1, j2] == 1:

            # Find the preceding peak
            k = j2 - 1
            while True:

                if k - order <= 0:
                    break

                # Peak?
                if out_array_sub[0, k] == 1:

                    # Check if the spectral distance is sufficient

                    # Peak < Valley?
                    # or
                    # Peak/valley difference < `min_sp_dist` threshold?
                    if (array[k] < array[j2]) or ((array[k] - array[j2]) < min_sp_dist):

                        # Remove the peak/valley pair
                        out_array_sub[0, k] = 0
                        out_array_sub[1, j2] = 0

                        # Remove the following peak
                        # for l in range(j2+1, cols-order):
                        #
                        #     # Peak?
                        #     if out_array_sub[0, l] == 1:
                        #
                        #         out_array_sub[0, l] = 0
                        #         break

                k -= 1

    # Third pass
    # --Check for peak proximity
    for j3 in range(order, cols-order):

        # Peak?
        if out_array_sub[0, j3] == 1:

            # Check the distance to the next peak
            for k in range(j3+1, cols-order):

                # Peak?
                if out_array_sub[0, k] == 1:

                    # Distance < `min_dist` threshold?
                    if <int>(k - j3) < min_dist:

                        # Take the peak with the larger value
                        if array[j3] > array[k]:
                            out_array_sub[0, k] = 0
                        else:
                            out_array_sub[0, j3] = 0

    # Fourth pass
    # --Check for double valleys without a peak
    j4 = order
    while True:

        if j4 >= (cols - order):
            break

        # Valley?
        if out_array_sub[1, j4] == 1:

            # Check for another valley before a peak
            for m in range(j4+1, cols-order):

                # Peak before the valley
                if (out_array_sub[0, m] == 1) and (out_array_sub[1, m] == 0):

                    j4 = m
                    break

                # Valley before the peak
                elif (out_array_sub[0, m] == 0) and (out_array_sub[1, m] == 1):

                    # Take the valley with the lower value
                    if array[j4] > array[m]:
                        out_array_sub[1, j4] = 0
                    else:
                        out_array_sub[1, m] = 0

                    j4 = m
                    break

        j4 += 1

    # Tally the peaks
    for j5 in range(order, cols-order):

        # Update the peak count
        out_array_npeaks_sub[0] += <int>out_array_sub[0, j5]

        if out_array_sub[0, j5] == 1:

            if out_array_max_peaks_sub[0] > 0:

                if array[j5] > array[<int>out_array_max_peaks_sub[0]]:
                    out_array_max_peaks_sub[0] = j5

            else:
                out_array_max_peaks_sub[0] = j5

    # Check if valleys should be added on the ends
    for j6 in range(0, cols):

        # valley
        if out_array_sub[1, j6] == 1:

            if out_array_sub[0, j6] != 1:
                break

        # track the minimum valley value
        if out_array_sub[1, j6] < minv:

            minv = out_array_sub[1, j6]
            minp = j6

        # peak
        if out_array_sub[0, j6] == 1:

            if minv < out_array_sub[0, j6]:

                out_array_sub[1, minp] = 1
                break


cdef void _find_peaks2d(double[:, ::1] array,
                        double[:, :, ::1] out_array_sub,
                        long[::1] out_array_npeaks_sub,
                        long[::1] out_array_max_peaks_sub,
                        Py_ssize_t ii,
                        unsigned int cols,
                        double min_value,
                        double min_dist,
                        double min_sp_dist,
                        double min_prop_sp_dist,
                        unsigned int order) nogil:

    cdef:
        Py_ssize_t jx, j1, j2, j3, j4, j5, j6, k, k2, l, m, jback, minp
        double signal_end, peak_value
        unsigned int total_peaks, total_valleys
        double minv = 1e9

    # First pass
    # --Locate peaks and valleys
    j1 = order
    while True:

        if j1 >= (cols - order):
            break

        # Peaks
        if (array[ii, j1] - array[ii, j1-order] > 0) and (array[ii, j1] - array[ii, j1+order] > 0):

            # Value check
            if array[ii, j1] >= min_value:

                peak_value = array[ii, j1]

                # Check backward if a flat peak was missed
                for jback in range(j1-1, 0, -1):

                    if out_array_sub[0, ii, jback] == 1:
                        break

                    if out_array_sub[1, ii, jback] == 1:
                        break

                    if array[ii, jback] > peak_value:
                        peak_value = array[ii, jback]
                        j1 = jback

                signal_end = float(j1 + 1 + min_dist)
                signal_end = common.min_value(signal_end, float(cols - order))

                total_peaks = 0

                # Locate the largest peak
                for jx in range(j1 + 1, <int>signal_end):

                    # Flat peaks
                    if array[ii, jx] == array[ii, j1]:
                        total_peaks += 1
                    elif array[ii, jx] > array[ii, j1]:
                        total_peaks = 0
                        j1 = jx

                if total_peaks > 0:
                    j1 += <int>(total_peaks / 2.0)

                out_array_sub[0, ii, j1] = 1

            j1 += order

        # Valleys
        elif (array[ii, j1] - array[ii, j1-order] < 0) and (array[ii, j1] - array[ii, j1+order] < 0):

            signal_end = float(j1 + 1 + min_dist)
            signal_end = common.min_value(signal_end, float(cols - order))

            # Locate the smallest valley
            for jx in range(j1 + 1, <int>signal_end):

                total_valleys = 0

                if array[ii, jx] == array[ii, j1]:
                    total_valleys += 1
                elif array[ii, jx] < array[ii, j1]:
                    total_valleys = 0
                    j1 = jx

            if total_valleys > 0:
                j1 += <int>(total_valleys / 2.0)

            out_array_sub[1, ii, j1] = 1

            j1 += order

        else:
            j1 += 1

    # Second pass
    # --Check for small amplitudes
    for j2 in range(order, cols-order):

        # Valley?
        if out_array_sub[1, ii, j2] == 1:

            k = j2 - 1
            k2 = j2 + 1

            # Find the preceding peak
            while True:

                if k - order <= 0:
                    break

                # Peak?
                if out_array_sub[0, ii, k] == 1:
                    break

                k -= 1

            # Find the following peak
            while True:

                if k + order >= cols:
                    break

                # Peak?
                if out_array_sub[0, ii, k2] == 1:
                    break

                k2 += 1

            # Peak < Valley?
            if (array[ii, k] < array[ii, j2]) or (array[ii, k2] < array[ii, j2]):

                # Remove the valley
                out_array_sub[1, ii, j2] = 0

                # Remove the lower peak
                if array[ii, k] < array[ii, k2]:
                    out_array_sub[0, ii, k] = 0
                else:
                    out_array_sub[0, ii, k2] = 0

            # Peak/valley difference < `min_sp_dist` threshold?
            if (common.fabs(array[ii, k] - array[ii, j2]) < min_sp_dist) or (common.fabs(common.prop_diff(array[ii, k], array[ii, j2])) < min_prop_sp_dist):

                # Remove the peak/valley pair
                out_array_sub[0, ii, k] = 0
                out_array_sub[1, ii, j2] = 0

            # Peak/valley difference < `min_sp_dist` threshold?
            if (common.fabs(array[ii, k2] - array[ii, j2]) < min_sp_dist) or (common.fabs(common.prop_diff(array[ii, k2], array[ii, j2])) < min_prop_sp_dist):

                # Remove the peak/valley pair
                out_array_sub[0, ii, k2] = 0
                out_array_sub[1, ii, j2] = 0

#                # Remove the following valley
#                for l in range(j2+1, cols-order):
#
#                    # Valley?
#                    if out_array_sub[1, ii, l] == 1:
#
#                        out_array_sub[1, ii, l] = 0
#                        break

    # Third pass
    # --Check for peak proximity
    for j3 in range(order, cols-order):

        # Peak?
        if out_array_sub[0, ii, j3] == 1:

            # Check the distance to the next peak
            for k in range(j3+1, cols-order):

                # Peak?
                if out_array_sub[0, ii, k] == 1:

                    # Distance < `min_dist` threshold?
                    if <int>(k - j3) < min_dist:

                        # Take the peak with the larger value
                        if array[ii, j3] > array[ii, k]:
                            out_array_sub[0, ii, k] = 0
                        else:
                            out_array_sub[0, ii, j3] = 0

    # Fourth pass
    # --Check for double valleys without a peak
#    j4 = order
#    while True:
#
#        if j4 >= (cols - order):
#            break
#
#        # Valley?
#        if out_array_sub[1, ii, j4] == 1:
#
#            # Check for another valley before a peak
#            for m in range(j4+1, cols-order):
#
#                # Peak before the valley
#                if (out_array_sub[0, ii, m] == 1) and (out_array_sub[1, ii, m] == 0):
#
#                    j4 = m
#                    break
#
#                # Valley before the peak
#                elif (out_array_sub[0, ii, m] == 0) and (out_array_sub[1, ii, m] == 1):
#
#                    # Take the valley with the lower value
#                    if array[ii, j4] > array[ii, m]:
#                        out_array_sub[1, ii, j4] = 0
#                    else:
#                        out_array_sub[1, ii, m] = 0
#
#                    j4 = m
#                    break
#
#        j4 += 1

    # Tally the peaks
    for j5 in range(order, cols-order):

        # Update the peak count
        out_array_npeaks_sub[ii] += <int>out_array_sub[0, ii, j5]

        if out_array_sub[0, ii, j5] == 1:

            if out_array_max_peaks_sub[ii] > 0:

                if array[ii, j5] > array[ii, <int>out_array_max_peaks_sub[ii]]:
                    out_array_max_peaks_sub[ii] = j5

            else:
                out_array_max_peaks_sub[ii] = j5

    # Check if valleys should be added on the ends
    for j6 in range(0, cols):

        # valley
        if out_array_sub[1, ii, j6] == 1:
            if out_array_sub[0, ii, j6] != 1:
                break

        # track the minimum valley value
        if out_array_sub[1, ii, j6] < minv:

            minv = out_array_sub[1, ii, j6]
            minp = j6

        # peak
        if out_array_sub[0, ii, j6] == 1:

            if minv < out_array_sub[0, ii, j6]:

                out_array_sub[1, ii, minp] = 1
                break


cdef void _peaks_valleys2d(double[:, ::1] array,
                           double[:, :, ::1] out_array_main,
                           long[::1] out_array_npeaks_main,
                           long[::1] out_array_max_peaks_main,
                           double min_value,
                           double min_dist,
                           double min_sp_dist,
                           double min_prop_sp_dist,
                           unsigned int rows,
                           unsigned int cols,
                           unsigned int order,
                           int n_jobs):

    cdef:
        Py_ssize_t i

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, rows, schedule='static'):

            _find_peaks2d(array,
                          out_array_main,
                          out_array_npeaks_main,
                          out_array_max_peaks_main,
                          i,
                          cols,
                          min_value,
                          min_dist,
                          min_sp_dist,
                          min_prop_sp_dist,
                          order)


cdef void _peaks_valleys1d(double[::1] array,
                           double[:, ::1] out_array_main,
                           DTYPE_uint64_t[::1] out_array_npeaks_main,
                           DTYPE_uint64_t[::1] out_array_max_peaks_main,
                           double min_value,
                           double min_dist,
                           double min_sp_dist,
                           unsigned int cols,
                           unsigned int order):

    with nogil:

        _find_peaks1d(array,
                      out_array_main,
                      out_array_npeaks_main,
                      out_array_max_peaks_main,
                      cols,
                      min_value,
                      min_dist,
                      min_sp_dist,
                      order)


def peaks_valleys1d(double[::1] array not None,
                    double min_value=0.2,
                    double min_dist=5,
                    double min_sp_dist=0.05,
                    unsigned int order=1):

    """
    Detects peaks and valleys in a 1-dimensional signal

    Args:
        array (1d array): The input signal.
        min_value (Optional[float]): The minimum value allowed for a peak.
        min_dist (Optional[float]): The minimum distance allowed between two peaks.
        min_sp_dist (Optional[float]): The minimum spectral distance allowed between a peak and a valley.
        order (Optional[int]): The nth order difference.

    Returns:
        Peaks and valleys (2d array): First layer = peaks; Second layer = valleys.
        Number of peaks (int)
        Index of maximum peak (int)
    """

    cdef:
        unsigned int cols = array.shape[0]

        # View of output
        np.ndarray[DTYPE_float64_t, ndim=2] out_array = np.zeros((2, cols), dtype='float64')
        double[:, ::1] out_array_view = out_array

        np.ndarray[DTYPE_uint64_t, ndim=1] out_array_npeaks = np.zeros(1, dtype='uint64')
        DTYPE_uint64_t[::1] out_array_npeaks_view = out_array_npeaks

        np.ndarray[DTYPE_uint64_t, ndim=1] out_array_max_peaks = np.zeros(1, dtype='uint64')
        DTYPE_uint64_t[::1] out_array_max_peaks_view = out_array_max_peaks

    _peaks_valleys1d(array,
                     out_array_view,
                     out_array_npeaks_view,
                     out_array_max_peaks_view,
                     min_value,
                     min_dist,
                     min_sp_dist,
                     cols,
                     order)

    return out_array, <int>out_array_npeaks_view[0], <int>out_array_max_peaks_view[0]


cdef inline void _group(long[:, :, ::1] pvs,
                        Py_ssize_t sidx,
                        Py_ssize_t idx,
                        Py_ssize_t jidx,
                        unsigned int ncols,
                        unsigned int w) nogil:

    cdef:
        Py_ssize_t k, v, b
        unsigned int pv_total, pv_count
        bint v_end

    pv_total = 0
    pv_count = 0

    for k in range(0, w):

        if pvs[sidx, idx, jidx+k] == 1:

            # Check if the valley is an endpoint
            if sidx == 1:

                v_end = True

                for v in range(jidx+k+1, ncols):
                    if pvs[sidx, idx, v] == 1:
                        v_end = False
                        break

                v = jidx + k - 1
                while True:

                    if v < 0:
                        break

                    if pvs[sidx, idx, v] == 1:
                        v_end = False
                        break

                    v -= 1

                if v_end:
                    continue

            pv_total += k
            pv_count += 1

            pvs[sidx, idx, jidx+k] = 0

            if sidx == 0:

                # Remove false valleys
                pvs[1, idx, jidx+k] = 0

    if pv_count > 0:
        pvs[sidx, idx, jidx+<int>(pv_total / <double>pv_count)] = 1


cdef inline void _group_valleys(long[:, :, ::1] pvs,
                                double[:, ::1] y,
                                double[::1] y_means,
                                Py_ssize_t idx,
                                Py_ssize_t jidx,
                                unsigned int ncols) nogil:

    cdef:
        Py_ssize_t k

    # Valley
    if pvs[1, idx, jidx] == 1:

        # Check if the valley value is greater than the average peak value
        if y[idx, jidx] > y_means[idx]:
            pvs[1, idx, jidx] = 0
        else:

            k = jidx + 1

            # Search for a peak before the next valley
            while True:

                if k >= ncols:
                    break

                # Peak exists so exit
                if pvs[0, idx, k] == 1:
                    break

                # Valley before the next peak
                if pvs[1, idx, k] == 1:

                    # Group the valleys

                    # Valley values are far apart so take the lower
                    if common.fabs(y[idx, jidx] - y[idx, k]) > 0.2:

                        # First valley is higher than the second
                        if y[idx, jidx] > y[idx, k]:
                            pvs[1, idx, jidx] = 0
                        else:
                            pvs[1, idx, k] = 0

                    else:

                        pvs[1, idx, jidx] = 0
                        pvs[1, idx, k] = 0

                        pvs[1, idx, <int>((jidx+k)*0.5)] = 1

                k += 1


cdef inline void _group_peaks(long[:, :, ::1] pvs,
                              double[:, ::1] y,
                              Py_ssize_t idx,
                              Py_ssize_t jidx,
                              unsigned int ncols) nogil:

    cdef:
        Py_ssize_t k

    # Peak
    if pvs[0, idx, jidx] == 1:

        k = jidx + 1

        # Search for a valley before the next peak
        while True:

            if k >= ncols:
                break

            # Valley exists so exit
            if pvs[1, idx, k] == 1:
                break

            # Peak before the next valley
            if pvs[0, idx, k] == 1:

                # Group the peaks

                # Peak values are far apart so take the upper
                if common.fabs(y[idx, jidx] - y[idx, k]) > 0.2:

                    # First peak is lower than the second
                    if y[idx, jidx] < y[idx, k]:
                        pvs[0, idx, jidx] = 0
                    else:
                        pvs[0, idx, k] = 0

                else:

                    pvs[0, idx, jidx] = 0
                    pvs[0, idx, k] = 0

                    pvs[0, idx, <int>((jidx+k)*0.5)] = 1

            k += 1


cdef inline void _get_peak_mean(long[:, :, ::1] pvs,
                                double[:, ::1] y,
                                double[::1] yp_means_,
                                Py_ssize_t ii,
                                unsigned int ncols,
                                unsigned int w) nogil:

    cdef:
        Py_ssize_t j1
        Py_ssize_t yp_count = 0

    for j1 in range(0, ncols-w):

        if pvs[0, ii, j1] == 1:
            yp_means_[ii] += y[ii, j1]
            yp_count += 1

    yp_means_[ii] /= <double>yp_count


cdef inline void _pv_prop_check(long[:, :, ::1] pvs,
                                double[:, ::1] y,
                                double[::1] yp_means,
                                Py_ssize_t idx,
                                unsigned int ncols,
                                unsigned int w,
                                double min_prop_sp_dist) nogil:

    cdef:
        Py_ssize_t j, v1, v2
        unsigned int n_changes = 0

    for j in range(0, ncols):

        # Peak
        if pvs[0, idx, j] == 1:

            # Find the valleys

            v1 = j - 1
            while True:

                if v1 < 0:
                    break

                if pvs[1, idx, v1] == 1:
                    break

                v1 -= 1

            v2 = j + 1
            while True:

                if v2 >= ncols:
                    break

                if pvs[1, idx, v2] == 1:
                    break

                v2 += 1

            if (common.fabs(common.prop_diff(y[idx, v1], y[idx, j])) < min_prop_sp_dist) or (common.fabs(common.prop_diff(y[idx, v2], y[idx, j])) < min_prop_sp_dist):

                # Remove the peak
                pvs[0, idx, j] = 0

                n_changes += 1

    if n_changes > 0:

        for j in range(0, ncols-w):

            # Group valleys with a missing peak
            _group_valleys(pvs, y, yp_means, idx, j, ncols)


def group_peaks_valleys2d(long[:, :, ::1] pvs,
                          double[:, ::1] y,
                          unsigned int w=15,
                          double min_prop_sp_dist=0.2,
                          unsigned int n_jobs=1,
                          unsigned int chunksize=10):

    cdef:
        Py_ssize_t i, j2, j3
        unsigned int nrows = pvs.shape[1]
        unsigned int ncols = pvs.shape[2]
        double[::1] yp_means = np.zeros(nrows, dtype='float64')

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, nrows, schedule='static', chunksize=chunksize):

            # Get peak mean
            _get_peak_mean(pvs, y, yp_means, i, ncols, w)

            for j2 in range(0, ncols-w):

                # Group peaks
                _group(pvs, 0, i, j2, ncols, w)

                # Group valleys
                _group(pvs, 1, i, j2, ncols, w)

            for j3 in range(0, ncols-w):

                # Group valleys with a missing peak
                _group_valleys(pvs, y, yp_means, i, j3, ncols)

                # Group peaks with a missing valley
                _group_peaks(pvs, y, i, j3, ncols)

            _pv_prop_check(pvs, y, yp_means, i, ncols, w, min_prop_sp_dist)

    return np.int64(pvs)


def peaks_valleys2d(double[:, ::1] array not None,
                    double min_value=0.2,
                    double min_dist=5,
                    double min_sp_dist=0.05,
                    double min_prop_sp_dist=1.0,
                    unsigned int order=1,
                    unsigned int n_jobs=1):

    """
    Detects peaks and valleys in a 2-dimensional series of signals

    Args:
        array (2d array): The input signals, shaped [samples x time series].
        min_value (Optional[float]): The minimum value allowed for a peak.
        min_dist (Optional[float]): The minimum distance allowed between two peaks.
        min_sp_dist (Optional[float]): The minimum spectral distance allowed between a peak and a valley.
        min_prop_sp_dist (Optional[float]): The minimum proportional spectral distance allowed between a peak and a valley.
        order (Optional[int]): The nth order difference.
        n_jobs (Optional[int]): The number of parallel processes.

    Returns:
        Peaks and valleys (3d array): First layer = peaks; Second layer = valleys.
        Number of peaks (1d array)
        Index of maximum peak (1d array)
    """

    cdef:
        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]

        # View of output
        np.ndarray[DTYPE_float64_t, ndim=3] out_array = np.zeros((2, rows, cols), dtype='float64')
        double[:, :, ::1] out_array_view = out_array

        np.ndarray[DTYPE_int64_t, ndim=1] out_array_npeaks = np.zeros(rows, dtype='int64')
        long[::1] out_array_npeaks_view = out_array_npeaks

        np.ndarray[DTYPE_int64_t, ndim=1] out_array_max_peaks = np.zeros(rows, dtype='int64')
        long[::1] out_array_max_peaks_view = out_array_max_peaks

    _peaks_valleys2d(array,
                     out_array_view,
                     out_array_npeaks_view,
                     out_array_max_peaks_view,
                     min_value,
                     min_dist,
                     min_sp_dist,
                     min_prop_sp_dist,
                     rows,
                     cols,
                     order,
                     n_jobs)

    return out_array, out_array_npeaks, out_array_max_peaks


cdef void _greenup_metrics1d(double[::1] array,
                             double[::1] out_array_,
                             Py_ssize_t viter,
                             Py_ssize_t piter,
                             double sos_thresh,
                             double mos_thresh,
                             double eos_thresh,
                             Py_ssize_t peak_counter,
                             unsigned int n_metrics) nogil:

    cdef:
        Py_ssize_t jj
        double vvalue, pvalue, cvalue
        double sos_marker, mos_marker, eos_marker
        bint sosf, mosf
        double season_max, season_min
        double pv_diff

    # Ensure the maximum peak is accurate
    #piter = _get_max_index1d(array, viter, piter)

    # Scaled value at the valley and peak
    vvalue = array[viter]
    pvalue = array[piter]

    pv_diff = pvalue - vvalue

    # season_max = array[piter]
    # season_min = _get_min1d(array, viter, piter)

    sos_marker = _round(vvalue + pv_diff * sos_thresh, 1000.0)
    mos_marker = _round(vvalue + pv_diff * mos_thresh, 1000.0)
    eos_marker = _round(vvalue + pv_diff * eos_thresh, 1000.0)

    sosf = False
    mosf = False

    for jj in range(viter, piter):

        # cvalue = common.scale_min_max(_round(array[i], 1000.0), 0.0, 1.0, vvalue, pvalue)
        cvalue = array[jj]

        if not sosf and (cvalue >= sos_marker):
            out_array_[0+(7*peak_counter)] = jj
            sosf = True
        elif not mosf and (cvalue >= mos_marker):
            out_array_[1+(7*peak_counter)] = jj
            mosf = True
        elif cvalue >= eos_marker:
            out_array_[2+(7*peak_counter)] = jj
            break

    if out_array_[2+(7*peak_counter)] == 0:

        if (out_array_[0+(7*peak_counter)] > 0) and (out_array_[1+(7*peak_counter)] > 0):
            out_array_[2+(7 * peak_counter)] = _round(out_array_[1+(7*peak_counter)] + ((piter - out_array_[1+(7*peak_counter)]) / 2.0), 1.0)

    out_array_[3+(7*peak_counter)] = piter


cdef void _greenup_metrics2d(double[:, ::1] array,
                             double[:, ::1] out_array_,
                             Py_ssize_t ii,
                             Py_ssize_t viter,
                             Py_ssize_t piter,
                             double sos_thresh,
                             double mos_thresh,
                             double eos_thresh,
                             Py_ssize_t peak_counter,
                             unsigned int n_metrics) nogil:

    cdef:
        Py_ssize_t jj
        double vvalue, pvalue, cvalue
        double sos_marker, mos_marker, eos_marker
        bint sosf, mosf
        double pv_diff

    # Ensure the maximum peak is accurate
    #piter = _get_max_index2d(array, ii, viter, piter)

    # Scaled value at the valley and peak
    vvalue = array[ii, viter]
    pvalue = array[ii, piter]

    pv_diff = pvalue - vvalue

    # season_max = array[ii, piter]
    # season_min = _get_min2d(array, ii, viter, piter)

    sos_marker = _round(vvalue + pv_diff * sos_thresh, 1000.0)
    mos_marker = _round(vvalue + pv_diff * mos_thresh, 1000.0)
    eos_marker = _round(vvalue + pv_diff * eos_thresh, 1000.0)

    sosf = False
    mosf = False

    for jj in range(viter, piter):

        # cvalue = common.scale_min_max(_round(array[ii, jj], 1000.0), 0.0, 1.0, vvalue, pvalue)
        cvalue = array[ii, jj]

        if not sosf and (cvalue >= sos_marker):
            out_array_[ii, 0+(n_metrics*peak_counter)] = jj
            sosf = True
        elif not mosf and (cvalue >= mos_marker):
            out_array_[ii, 1+(n_metrics*peak_counter)] = jj
            mosf = True
        elif cvalue >= eos_marker:
            out_array_[ii, 2+(n_metrics*peak_counter)] = jj
            break

    if out_array_[ii, 2 + (n_metrics * peak_counter)] == 0:

        if (out_array_[ii, (n_metrics * peak_counter)] > 0) and (out_array_[ii, 1 + (n_metrics * peak_counter)] > 0):
            out_array_[ii, 2 + (n_metrics * peak_counter)] = _round(out_array_[ii, 1 + (n_metrics * peak_counter)] + ((piter - out_array_[ii, 1 + (n_metrics * peak_counter)]) / 2.0), 1.0)

    # Mis-calculated 50% -- set to midpoint of start and end
    if out_array_[ii, 1+(n_metrics * peak_counter)] < out_array_[ii, (n_metrics * peak_counter)]:
        out_array_[ii, 1+(n_metrics * peak_counter)] = _round((out_array_[ii, (n_metrics * peak_counter)] + out_array_[ii, 2+(n_metrics * peak_counter)]) / 2.0, 1.0)

    # Peak
    out_array_[ii, 3+(n_metrics*peak_counter)] = piter


cdef void _greendown_metrics1d(double[::1] array,
                               double[::1] out_array_,
                               Py_ssize_t viter,
                               Py_ssize_t piter,
                               unsigned int ncols,
                               double sos_thresh,
                               double mos_thresh,
                               double eos_thresh,
                               Py_ssize_t peak_counter,
                               unsigned int n_metrics) nogil:

    cdef:
        Py_ssize_t i
        double vvalue, pvalue, cvalue
        double sos_marker, mos_marker, eos_marker
        bint sosf, mosf
        double pv_diff

    # Value at the peak
    vvalue = array[piter]
    pvalue = array[viter]

    pv_diff = pvalue - vvalue

    sos_marker = _round(vvalue + pv_diff * sos_thresh, 1000.0)
    mos_marker = _round(vvalue + pv_diff * mos_thresh, 1000.0)
    eos_marker = _round(vvalue + pv_diff * eos_thresh, 1000.0)

    sosf = False
    mosf = False

    for i in range(viter+1, piter):

        # cvalue = common.scale_min_max(_round(array[i], 1000.0), 0.0, 1.0, vvalue, pvalue)
        cvalue = array[i]

        if not sosf and (cvalue <= sos_marker):
            out_array_[4+(n_metrics*peak_counter)] = i
            sosf = True
        elif not mosf and (cvalue <= mos_marker):
            out_array_[5+(n_metrics*peak_counter)] = i
            mosf = True
        elif cvalue <= eos_marker:
            out_array_[6+(n_metrics*peak_counter)] = i
            break

    # Peak with no end
    if (out_array_[3 + (n_metrics * peak_counter)] > 0) and (out_array_[6 + (n_metrics * peak_counter)] == 0):

        out_array_[6 + (n_metrics * peak_counter)] = piter - 1 #out_array_[5 + (n_metrics * peak_counter)]
        out_array_[5 + (n_metrics * peak_counter)] = _round(out_array_[4 + (n_metrics * peak_counter)] + out_array_[6 + (n_metrics * peak_counter)], 1.0)


cdef void _greendown_metrics2d(double[:, ::1] array,
                               double[:, ::1] out_array_,
                               Py_ssize_t ii,
                               Py_ssize_t viter,
                               Py_ssize_t piter,
                               unsigned int ncols,
                               double sos_thresh,
                               double mos_thresh,
                               double eos_thresh,
                               Py_ssize_t peak_counter,
                               unsigned int n_metrics) nogil:

    cdef:
        Py_ssize_t jj
        double vvalue, pvalue, cvalue
        double sos_marker, mos_marker, eos_marker
        bint sosf, mosf
        double pv_diff

    # Value at the peak
    vvalue = array[ii, piter]
    pvalue = array[ii, viter]

    pv_diff = pvalue - vvalue

    sos_marker = _round(vvalue + pv_diff * sos_thresh, 1000.0)
    mos_marker = _round(vvalue + pv_diff * mos_thresh, 1000.0)
    eos_marker = _round(vvalue + pv_diff * eos_thresh, 1000.0)

    sosf = False
    mosf = False

    for jj in range(viter+1, piter):

        # cvalue = common.scale_min_max(_round(array[ii, jj], 1000.0), 0.0, 1.0, vvalue, pvalue)
        cvalue = array[ii, jj]

        if not sosf and (cvalue <= sos_marker):
            out_array_[ii, 4+(n_metrics*peak_counter)] = jj
            sosf = True
        elif not mosf and (cvalue <= mos_marker):
            out_array_[ii, 5+(n_metrics*peak_counter)] = jj
            mosf = True
        elif cvalue <= eos_marker:
            out_array_[ii, 6+(n_metrics*peak_counter)] = jj
            break

    # Peak with no end
    if (out_array_[ii, 3+(n_metrics*peak_counter)] > 0) and (out_array_[ii, 6+(n_metrics*peak_counter)] == 0):

        out_array_[ii, 6+(n_metrics*peak_counter)] = piter - 1 #out_array_[ii, 5+(n_metrics*peak_counter)]
        out_array_[ii, 5+(n_metrics*peak_counter)] = _round(out_array_[ii, 4+(n_metrics*peak_counter)] + out_array_[ii, 6+(n_metrics*peak_counter)], 1.0)

    # Mis-calculated 50% -- set to midpoint of start and end
    if out_array_[ii, 5+(n_metrics*peak_counter)] > out_array_[ii, 6+(n_metrics*peak_counter)]:
        out_array_[ii, 5+(n_metrics*peak_counter)] = _round((out_array_[ii, 4+(n_metrics*peak_counter)] + out_array_[ii, 6+(n_metrics*peak_counter)]) / 2.0, 1.0)


cdef unsigned int _count_peaks(double[:, ::1] pvs):

    cdef:
        Py_ssize_t m
        unsigned int n_samps = pvs.shape[1]
        unsigned int n_peaks = 0

    for m in range(0, n_samps):

        if pvs[0, m] == 1:
            n_peaks += 1

    return n_peaks


cdef unsigned int _count_peaks3d(double[:, :, ::1] pvs, Py_ssize_t ii) nogil:

    cdef:
        Py_ssize_t m
        unsigned int n_samps = pvs.shape[2]
        unsigned int n_peaks = 0

    for m in range(0, n_samps):

        if pvs[0, ii, m] == 1:
            n_peaks += 1

    return n_peaks


def phenometrics1d(double[::1] array not None,
                   double[:, ::1] pvs not None,
                   double low_thresh=0.1,
                   double mid_thresh=0.5,
                   double high_thresh=0.9):

    """
    Calculates phenological metrics

    Args:
        array (1d array): The input signal.
        pvs (2d array): Peaks and valleys.
        low_thresh (Optional[float]): The low season threshold.
        mid_thresh (Optional[float]): The middle season threshold.
        high_thresh (Optional[float]): The high season threshold.

    Returns:
        [greening]
        start-of-season
        middle-of-season
        end-of-season
        peak
        [browning]
        start-of-season
        middle-of-season
        end-of-season
        amplitude
        season length
    """

    cdef:
        Py_ssize_t j, pgu, pgd, v
        unsigned int cols = array.shape[0]
        bint peak_found

        unsigned int n_peaks = _count_peaks(pvs)
        unsigned int n_metrics = 9

        # View of output
        np.ndarray[double, ndim=1] out_array = np.zeros(<int>(n_metrics*n_peaks), dtype='float64')
        double[::1] out_array_view = out_array

        Py_ssize_t peak_counter

    with nogil:

        # Check if a valley should be added
        #   before the first peak.
        for j in range(0, cols):

            if pvs[1, j] == 1:
                break

            if pvs[0, j] == 1:

                if array[0] < array[j]:
                    pvs[1, 0] = 1

                break

        # Check if a valley should be added
        #   after the last peak.
        for j in range(0, cols):

            if pvs[1, cols-j-1] == 1:
                break

            if pvs[0, cols-j-1] == 1:

                if array[cols-1] < array[cols-j-1]:
                    pvs[1, cols-1] = 1

                break

        v = 0

        peak_counter = 0

        while True:

            # Valley
            if pvs[1, v] == 1:

                peak_found = False

                # Greenup period
                for pgu in range(v+1, cols):

                    # Peak
                    if pvs[0, pgu] == 1:

                        _greenup_metrics1d(array,
                                           out_array_view,
                                           v,
                                           pgu,
                                           low_thresh,
                                           mid_thresh,
                                           high_thresh,
                                           peak_counter,
                                           n_metrics)

                        peak_found = True

                        break

                if peak_found:

                    # Greendown period
                    for pgd in range(pgu+1, cols):

                        # Valley or end of series
                        if (pvs[1, pgd] == 1) or (pgd+1 == cols):

                            _greendown_metrics1d(array,
                                                 out_array_view,
                                                 pgu,
                                                 pgd,
                                                 cols,
                                                 high_thresh,
                                                 mid_thresh,
                                                 low_thresh,
                                                 peak_counter,
                                                 n_metrics)

                            peak_counter += 1

                            break

                    v = pgd

                else:
                    v = pgu

            else:
                v += 1

            if v+1 >= cols:
                break

    return out_array


cdef void _phenometrics2d(double[:, ::1] array_,
                          double[:, :, ::1] pvs,
                          Py_ssize_t ii,
                          unsigned int cols,
                          double[:, ::1] out_array_view_,
                          unsigned int max_peaks,
                          double low_thresh,
                          double mid_thresh,
                          double high_thresh,
                          unsigned int n_metrics) nogil:

    cdef:
        Py_ssize_t j, pgu, pgd, v
        unsigned int n_peaks
        bint peak_found
        Py_ssize_t peak_counter

    n_peaks = _count_peaks3d(pvs, ii)

    # Check if a valley should be added
    #   before the first peak.
    for j in range(0, cols):

        if pvs[1, ii, j] == 1:
            break

        if pvs[0, ii, j] == 1:

            if array_[ii, 0] < array_[ii, j]:
                pvs[1, ii, 0] = 1

            break

    # Check if a valley should be added
    #   after the last peak.
    for j in range(0, cols):

        if pvs[1, ii, cols-j-1] == 1:
            break

        if pvs[0, ii, cols-j-1] == 1:

            if array_[ii, cols-1] < array_[ii, cols-j-1]:
                pvs[1, ii, cols-1] = 1

            break

    v = 0

    peak_counter = 0

    while True:

        # Valley
        if pvs[1, ii, v] == 1:

            peak_found = False

            # Greening period
            for pgu in range(v+1, cols):

                # Peak
                if pvs[0, ii, pgu] == 1:

                    _greenup_metrics2d(array_,
                                       out_array_view_,
                                       ii,
                                       v,
                                       pgu,
                                       low_thresh,
                                       mid_thresh,
                                       high_thresh,
                                       peak_counter,
                                       n_metrics)

                    peak_found = True

                    # amplitude (peak value - start-of-season value)
                    out_array_view_[ii, 7+(n_metrics * peak_counter)] = array_[ii, <int>out_array_view_[ii, 3+(n_metrics * peak_counter)]] - array_[ii, <int>out_array_view_[ii, 0+(n_metrics * peak_counter)]]

                    break

            if peak_found:

                # Browning period
                for pgd in range(pgu+1, cols):

                    # Valley or end of series
                    if (pvs[1, ii, pgd] == 1) or (pgd+1 == cols):

                        _greendown_metrics2d(array_,
                                             out_array_view_,
                                             ii,
                                             pgu,
                                             pgd,
                                             cols,
                                             high_thresh,
                                             mid_thresh,
                                             low_thresh,
                                             peak_counter,
                                             n_metrics)

                        # season length (browning end-of-season - greening start-of-season)
                        out_array_view_[ii, 8+(n_metrics * peak_counter)] = out_array_view_[ii, 6+(n_metrics * peak_counter)] - out_array_view_[ii, 0+(n_metrics * peak_counter)]

                        peak_counter += 1

                        break

                v = pgd

            else:
                v = pgu

        else:
            v += 1

        if peak_counter > max_peaks:
            break

        if v+1 >= cols:
            break


def phenometrics2d(double[:, ::1] array not None,

                   double[:, :, ::1] pvs not None,
                   double low_thresh=0.1,
                   double mid_thresh=0.5,
                   double high_thresh=0.9,
                   int n_jobs=1):

    """
    Calculates phenological metrics

    Args:
        array (2d array): The input signal.
        pvs (3d array): Peaks and valleys, where the First layer = peaks; Second layer = valleys.
        low_thresh (Optional[float]): The low season threshold.
        mid_thresh (Optional[float]): The middle season threshold.
        high_thresh (Optional[float]): The high season threshold.
        n_jobs (Optional[int]): The number of parallel processes.

    Returns:
        [greening]
        1: start-of-season
        2: middle-of-season
        3: end-of-season
        4: peak
        [browning]
        5: start-of-season
        6: middle-of-season
        7: end-of-season
        8: amplitude
        9: season length
    """

    cdef:
        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]

        Py_ssize_t i

        unsigned int max_peaks = 6
        unsigned int n_metrics = 9

        # View of output
        np.ndarray[double, ndim=2] out_array = np.zeros((rows, <int>(n_metrics*max_peaks)), dtype='float64')
        double[:, ::1] out_array_view = out_array

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, rows, schedule='static'):

            _phenometrics2d(array,
                            pvs,
                            i,
                            cols,
                            out_array_view,
                            max_peaks,
                            low_thresh,
                            mid_thresh,
                            high_thresh,
                            n_metrics)

    return out_array
