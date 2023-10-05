# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython

from ..utils cimport common


cdef inline double _get_max(double[:, ::1] in_rows,
                            Py_ssize_t i,
                            unsigned int cols) nogil:

    cdef:
        Py_ssize_t a
        double m = in_rows[i, 0]

    for a in range(1, cols):

        if in_rows[i, a] > m:
            m = in_rows[i, a]

    return m


cdef inline int _get_x1(unsigned int x2,
                        double[:, ::1] in_rows,
                        Py_ssize_t i,
                        double no_data) nogil:

    cdef:
        Py_ssize_t x1_iter
        int x1 = -9999

    for x1_iter in range(1, x2+1):

        if in_rows[i, x2-x1_iter] != no_data:
            x1 = x2 - x1_iter

            break

    return x1


cdef inline int _get_x3(unsigned int x2,
                        double[:, ::1] in_rows,
                        Py_ssize_t i,
                        unsigned int dims,
                        double no_data) nogil:

    cdef:
        Py_ssize_t x3_iter
        int x3 = -9999
        int x3_range = dims - x2

    for x3_iter in range(1, x3_range):

        if in_rows[i, x2+x3_iter] != no_data:
            x3 = x2 + x3_iter

            break

    return x3


cdef inline void _find_indices(double[:, ::1] in_rows,
                               Py_ssize_t i,
                               long[:, ::1] index_positions_find,
                               unsigned int dims,
                               double no_data) nogil:

    cdef:
        Py_ssize_t fi
        Py_ssize_t counter = 0

    for fi in range(0, dims):

        if in_rows[i, fi] == no_data:

            index_positions_find[i, counter] = fi
            counter += 1

    # Store the count in the last index position
    index_positions_find[i, dims] = counter


cdef inline void _fill_ends_1d(double[::1] in_row_,
                               unsigned int dims,
                               double no_data) nogil:

    cdef:
        Py_ssize_t ib, ic
        double valid_data

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


cdef inline void _fill_ends(double[:, ::1] in_rows_,
                            Py_ssize_t i,
                            unsigned int dims,
                            double no_data) nogil:

    cdef:
        Py_ssize_t ib, ic
        double valid_data

    if in_rows_[i, 0] == no_data:

        # Find the first valid data point
        for ib in range(1, dims):

            if in_rows_[i, ib] != no_data:
                valid_data = in_rows_[i, ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(0, ib):
            in_rows_[i, ic] = valid_data

    if in_rows_[i, dims-1] == no_data:

        # Find the first non-zero data
        for ib in range(dims-2, 0, -1):

            if in_rows_[i, ib] != no_data:
                valid_data = in_rows_[i, ib]
                break

        # Fill the ends up to the valid data point.
        for ic in range(ib+1, dims):
            in_rows_[i, ic] = valid_data


cdef inline void _fill_no_data(double[:, ::1] in_rows,
                               Py_ssize_t i,
                               long[:, ::1] index_positions_fill,
                               unsigned int dims,
                               double no_data) nogil:

    cdef:
        Py_ssize_t x2_idx
        unsigned int len_idx
        int x1, x2, x3

    _find_indices(in_rows,
                  i,
                  index_positions_fill,
                  dims,
                  no_data)

    len_idx = <int>index_positions_fill[i, dims]

    if len_idx != 0:

        # check for data
        if _get_max(in_rows, i, dims) > 0:

            for x2_idx in range(0, len_idx):

                x2 = <int>index_positions_fill[i, x2_idx]

                # get x1
                x1 = _get_x1(x2, in_rows, i, no_data)

                # get x3
                x3 = _get_x3(x2, in_rows, i, dims, no_data)

                if (x1 != -9999) and (x3 != -9999):
                    in_rows[i, x2] = common.linear_adjustment(x1, x2, x3, in_rows[i, x1], in_rows[i, x3])

        _fill_ends(in_rows, i, dims, no_data)


cdef inline void interpolate(double[:, ::1] input_array,
                             long[:, ::1] index_positions,
                             Py_ssize_t i,
                             double no_data_value) nogil:

    """
    Linearly interpolates between 'no data' points
    """

    cdef:
        unsigned int dims = input_array.shape[1]

    _fill_no_data(input_array,
                  i,
                  index_positions,
                  dims,
                  no_data_value)
