# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t


ctypedef float (*metric_ptr)(DTYPE_float32_t[:, ::1], Py_ssize_t, Py_ssize_t) nogil


cdef float _set_data(DTYPE_float32_t[:, ::1] array, Py_ssize_t i, Py_ssize_t j) nogil:
    return array[i, j]


def test_no_slicing(unsigned int n_rows, unsigned int n_cols):

    cdef:
        Py_ssize_t i, j

        DTYPE_float32_t[:, ::1] array = np.random.randn(n_rows * n_cols).reshape(n_rows, n_cols).astype(np.float32)
        DTYPE_float32_t[:, ::1] output = np.empty((n_rows, n_cols), dtype='float32')

    with nogil:

        for i in range(0, n_rows):

            for j in range(0, n_cols):
                output[i, j] = _set_data(array, i, j)

    return output


def test_no_slicing_metric(unsigned int n_rows, unsigned int n_cols):

    cdef:
        Py_ssize_t i, j

        DTYPE_float32_t[:, ::1] array = np.random.randn(n_rows*n_cols).reshape(n_rows, n_cols).astype(np.float32)
        DTYPE_float32_t[:, ::1] output = np.empty((n_rows, n_cols), dtype='float32')
        # cdef float * array_ = <float *>malloc(n_cols * sizeof(int))
        metric_ptr func

    func = &_set_data

    with nogil:

        for i in range(0, n_rows):

            for j in range(0, n_cols):
                output[i, j] = func(array, i, j)

    return output
