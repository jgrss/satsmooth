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

from cython.parallel import prange
from cython.parallel import parallel

from .lowess cimport lowess


def lowess_smooth(long[::1] ordinals=None,
                  double[:, ::1] y=None,
                  unsigned int w=61,
                  unsigned int n_jobs=1,
                  unsigned int chunksize=10):

    cdef:
        Py_ssize_t i
        unsigned int nsamps = y.shape[0]
        unsigned int ntime = y.shape[1]

        double[:, ::1] yout = np.zeros((nsamps, ntime), dtype='float64')

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, nsamps, schedule='static', chunksize=chunksize):

            lowess(ordinals,
                   y,
                   yout,
                   w,
                   i,
                   ntime)

    return np.float64(yout)
