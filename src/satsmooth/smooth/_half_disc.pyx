# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

"""
@author: Jordan Graesser
"""

import cython

cimport cython

import numpy as np

cimport numpy as np

DTYPE_uint8 = np.uint8
ctypedef np.uint8_t DTYPE_uint8_t

DTYPE_float32 = np.float32
ctypedef np.float32_t DTYPE_float32_t

import pymorph
from scipy.ndimage import rotate


cdef inline DTYPE_float32_t _pow(DTYPE_float32_t a) nogil:
    return a*a


cdef inline DTYPE_float32_t _squared_diff(DTYPE_float32_t a, DTYPE_float32_t b) nogil:
    return _pow(a - b)


cdef DTYPE_float32_t _calc_chi2(DTYPE_float32_t[::1] h1_,
                                DTYPE_float32_t[::1] h2_,
                                unsigned int n_bins) nogil:

    """ 
    Calculates the Chi-square statistic
    """

    cdef:
        DTYPE_float32_t ss = 0.0
        DTYPE_float32_t ss_, tt_
        Py_ssize_t p
        DTYPE_float32_t g, h

    for p in range(0, n_bins):

        g = h1_[p]
        h = h2_[p]

        ss_ = _squared_diff(g, h)
        tt_ = g + h

        if (ss_ > 0) and (tt_ > 0):
            ss += ss_ / tt_

    return ss / 2.0


cdef DTYPE_float32_t _get_hist(DTYPE_uint8_t[:, ::1] array_sub,
                               DTYPE_uint8_t[:, ::1] disc,
                               unsigned int w,
                               DTYPE_float32_t[::1] h1_,
                               DTYPE_float32_t[::1] h2_,
                               unsigned int n_bins) nogil:

    """
    Gets the histograms for the half discs
    """

    cdef:
        Py_ssize_t i, j
        unsigned int bin_

    for i in range(0, w):

        for j in range(0, w):

            if disc[i, j] != 0:

                bin_ = array_sub[i, j]

                if disc[i, j] == 1:
                    h1_[bin_] += 1

                elif disc[i, j] == 2:
                    h2_[bin_] += 1

    return _calc_chi2(h1_, h2_, n_bins)


def moving_half_disc(DTYPE_uint8_t[:, :, ::1] array not None,
                     unsigned int w,
                     DTYPE_float32_t[:] rotations=None,
                     unsigned int n_bins=32):

    """
    Args:
        array (4d array): [dimensions x rows x columns]
        w (int): The disc window size.
        rotations (Optional[list])
    """

    cdef:
        unsigned int dims = array.shape[0]
        unsigned int rows = array.shape[1]
        unsigned int cols = array.shape[2]

        # DTYPE_float32_t[::1] mnmx = np.zeros(2, dtype='float32')

        DTYPE_float32_t[::1] h1 = np.zeros(n_bins, dtype='float32')
        DTYPE_float32_t[::1] h2 = h1.copy()
        DTYPE_float32_t[::1] hz = h1.copy()

        unsigned int radius = <int>(w / 2.0)

        DTYPE_uint8_t[:, ::1] disc = np.ascontiguousarray(np.uint8(pymorph.sedisk(r=radius,
                                                                                  dim=2,
                                                                                  metric='euclidean',
                                                                                  flat=True,
                                                                                  h=0)))

        DTYPE_uint8_t[:, ::1] zs = disc.copy()

        unsigned int n_rotations = rotations.shape[0]
        Py_ssize_t ridx, d, i, j, iz, jz
        DTYPE_float32_t rot

        DTYPE_uint8_t[:, ::1] array_sub

        DTYPE_float32_t[:, :, ::1] out_array = np.zeros((dims, rows, cols), dtype='float32')

        DTYPE_float32_t grad_mag, curr_val

    disc[:radius, :] = 2

    for iz in range(0, w):
        for jz in range(0, w):
            if zs[iz, jz] == 0:
                disc[iz, jz] = 0

    for ridx in range(0, n_rotations):

        rot = rotations[ridx]

        if rot == 90:

            disc = np.ascontiguousarray(rotate(disc, 90, reshape=False, order=3, mode='constant'))
            disc = np.ascontiguousarray(np.fliplr(disc))

        elif rot == 180:
            disc = np.ascontiguousarray(np.flipud(disc))
        else:
            disc = np.ascontiguousarray(rotate(disc, rot, reshape=False, order=3, mode='constant'))

        with nogil:

            for i in range(0, rows-w+1):

                for j in range(0, cols-w+1):

                    for d in range(0, dims):

                        array_sub = array[d, i:i+w, j:j+w]

                        # The gradient magnitude is the Chi^2 value
                        #   of the half disc histogram comparison.
                        grad_mag = _get_hist(array_sub, disc, w, h1, h2, n_bins)

                        # Reset the histograms to zeros.
                        h1[...] = hz
                        h2[...] = hz

                        # The first values will be zeros.
                        curr_val = out_array[d, i+radius, j+radius]

                        # Get the maximum response over all
                        #   of the rotations and scales.
                        if grad_mag > curr_val:
                            out_array[d, i+radius, j+radius] = grad_mag

    return np.float32(out_array)
