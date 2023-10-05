# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

from libc.stdlib cimport free, malloc

from ..utils cimport common


cdef inline void btwo_points(double t, Py_ssize_t i2, double *points_x, double *points_y) nogil:

    cdef:
        double p1a, p1b, p2a, p2b

    p1a = points_x[i2]
    p1b = points_y[i2]
    p2a = points_x[i2+1]
    p2b = points_y[i2+1]

    points_x[i2] = (1.0 - t) * p1a + t * p2a
    points_y[i2] = (1.0 - t) * p1b + t * p2b


cdef inline void bpoints(double t, double *points_x, double *points_y, unsigned int pcount) nogil:

    cdef:
        Py_ssize_t i1

    for i1 in range(0, pcount):
        btwo_points(t, i1, points_x, points_y)


cdef inline void bpoint(double t, double *points_x, double *points_y, Py_ssize_t kcount) nogil:

    cdef:
        unsigned int pcount = kcount - 1

    while True:

        bpoints(t, points_x, points_y, pcount)

        pcount -= 1

        if pcount < 2:
            break


cdef inline void bcurve(double[::1] t_values,
                        double[:, ::1] yarray,
                        unsigned long[:, ::1] gap_array,
                        Py_ssize_t idx,
                        unsigned int start,
                        unsigned int n,
                        unsigned int k=31,
                        unsigned int s=5) nogil:

    """
    Args:
        t_values (1d array): [0-1]
        y (2d array): The values.
        i (int): The row position of `y`.
        n (int): The series length.
        k (int): The window size.
        s (int): The smoothing parameter.

    Reference:
        https://github.com/torresjrjr/Bezier.py/blob/master/Bezier.py
    """

    cdef:
        unsigned int k_half = <int>(k / 2.0)
        Py_ssize_t a, b, j, t
        Py_ssize_t kidx
        double *points_x
        double *points_y

    points_x = <double *>malloc(k * sizeof(double))
    points_y = <double *>malloc(k * sizeof(double))

    # Iterate over the series
    for a in range(start, n-k):

        #s = <int>common.clip(common.scale_min_max(<double>gap_array[idx, a+k_half], 1.0, 5.0, 10.0, 60.0), 1.0, 5.0)

        b = a + k

        kidx = 0

        # Iterate over the current window
        for j from a <= j < b by s:

            points_x[kidx] = <double>j
            points_y[kidx] = yarray[idx, j]

            kidx += 1

        # Get the center points
        # The t_values length is always 100
        bpoint(t_values[49], points_x, points_y, kidx)

        # Update the center value
        yarray[idx, a+k_half] = points_y[0]

    free(points_x)
    free(points_y)
