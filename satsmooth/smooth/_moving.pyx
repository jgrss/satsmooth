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
ctypedef void (*metric_ptr)(double[:, ::1], double[:, ::1], Py_ssize_t, Py_ssize_t, unsigned int, unsigned int) nogil


cdef void _mean(double[:, ::1] array,
                double[:, ::1] out_array_,
                Py_ssize_t i,
                Py_ssize_t j,
                unsigned int window_size,
                unsigned int half_window) nogil:

    cdef:
        Py_ssize_t m, n
        float block_sum = 0.0

    for m in range(i, i+window_size):
        for n in range(j, j+window_size):

            block_sum += array[m, n]

    out_array_[i+half_window, j+half_window] = block_sum / float(window_size * window_size)


def moving_mean(np.ndarray[DTYPE_float64_t, ndim=2] array not None,
                unsigned int window_size=3,
                np.ndarray[DTYPE_float64_t, ndim=2] window_weights=None):

    """
    Calculates the moving mean

    Args:
        array (2d array): The data to process.
        window_size (Optional[int]): The window size.
        window_weights (Optional[2d array]): Window weights.

    Returns:
        (2d array)
    """

    cdef:
        Py_ssize_t i, j

        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]

        unsigned int half_window = <int>(window_size / 2.0)

        double[:, ::1] out_array = array.copy()

        double[:, ::1] weights

        metric_ptr wfunc = &_mean

    if isinstance(window_weights, np.ndarray):
        weights = window_weights
    else:
        weights = np.ones((window_size, window_size), dtype='float64')

    with nogil:

        for i in range(0, rows-window_size+1):
            for j in range(0, cols-window_size+1):

                wfunc(array,
                      out_array,
                      i,
                      j,
                      window_size,
                      half_window)

    return np.float64(out_array)
