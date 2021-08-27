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
cimport numpy as np

from cython.parallel import prange
from cython.parallel import parallel

from ..utils cimport outliers


def remove_outliers(double[::1] x not None,
                    double[:, ::1] array not None,
                    double no_data_value=0.0,
                    unsigned int max_outlier_days1=120,
                    unsigned int max_outlier_days2=120,
                    unsigned int min_outlier_values=7,
                    unsigned int outlier_iters=1,
                    double dev_thresh1=0.2,
                    double dev_thresh2=0.2,
                    int n_jobs=1,
                    int chunksize=1):

    """
    Removes outliers

    Args:
        x (1d array): The time intervals.
        array (2d array): The data to smooth, scaled 0-1.
        no_data_value (Optional[double]): The 'no data' value to fill if `fill_no_data`=True.
        max_outlier_days1 (Optional[int]): The maximum spread, in days, to search for outliers.
        max_outlier_days2 (Optional[int]): The maximum spread, in days, to search for outliers.
        min_outlier_values (Optional[int]): The minimum number of outlier samples.
        outlier_iters (Optional[int]): The number of iterations to check for outliers.
        dev_thresh1 (Optional[double]): The deviation threshold for outliers.
        dev_thresh2 (Optional[double]): The deviation threshold for outliers.
        n_jobs (Optional[int]): The number of parallel jobs.
        chunksize (Optional[int]): The parallel thread chunksize.

    Returns:
        Signals with outliers removed (2d array)
    """

    cdef:
        Py_ssize_t i, oidx

        unsigned int rows = array.shape[0]
        unsigned int cols = array.shape[1]

        # y = alpha + beta * X
        #   where, X = x, x^2
        #
        # mb = beta_1, beta_2, alpha
        double[:, ::1] mb = np.zeros((rows, 3), dtype='float64')

        long[:, :, ::1] index_positions_fill_dummy = np.zeros((2, 1, 1), dtype='int64')
        long[:, :, ::1] index_positions_fill = np.zeros((2, rows, cols+1), dtype='int64')

    with nogil, parallel(num_threads=n_jobs):

        for oidx in range(0, outlier_iters):

            for i in prange(0, rows, schedule='static', chunksize=chunksize):

                # Local linear regression over 3 samples
                outliers.remove_outliers_linear(x,
                                                array,
                                                mb,
                                                i,
                                                cols,
                                                3,
                                                max_outlier_days1,
                                                dev_thresh1,
                                                no_data_value,
                                                index_positions_fill_dummy)

                # Local polynomial regression over user-defined samples
                outliers.remove_outliers_polynomial(x,
                                                    array,
                                                    mb,
                                                    i,
                                                    cols,
                                                    min_outlier_values,
                                                    max_outlier_days2,
                                                    dev_thresh2,
                                                    no_data_value,
                                                    index_positions_fill)

    return np.float64(array)
