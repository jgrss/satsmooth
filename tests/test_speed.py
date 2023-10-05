import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from .. import LinterpMulti, LinterpMultiIndex
from .data import create_1d
from .timings import time_func


def test_multi(n_rows=500, n_cols=150, t=25, min_window=7, n_jobs=1):

    x1, x2, y_raw = create_1d(100, 5000, 0.2, 0.1, n_cols)

    indices = np.uint64(np.linspace(x2[0], x2[-1], 100))

    y = np.tile(y_raw.copy(), (n_rows, 1))

    sci_interpolator = interp1d(x1, y[0])

    with time_func('SciPy interp1d'):

        for i in range(0, y.shape[0]):
            yo_ = sci_interpolator(x2)

    interpolator = LinterpMulti(x1, x2)
    interpolator_index = LinterpMultiIndex(x1, x2)

    with time_func('Warm-up'):

        yo_ = interpolator.interp2d(
            y[:2].copy(), fill_no_data=True, no_data_value=0.0, n_jobs=2
        )

        yo_ = interpolator.interp2dx(
            y[:2].copy(),
            fill_no_data=True,
            no_data_value=0.0,
            remove_outliers=True,
            dev_thresh=0.2,
            return_indexed=True,
            indices=indices,
            t=t,
            min_window=min_window,
            n_jobs=2,
        )

        # yo_ = interpolator_index.interp2dx(y[:2].copy(),
        #                                    fill_no_data=True,
        #                                    no_data_value=0.0,
        #                                    remove_outliers=True,
        #                                    dev_thresh=0.2,
        #                                    return_indexed=True,
        #                                    indices=indices,
        #                                    t=t,
        #                                    min_window=min_window,
        #                                    n_jobs=2)

    with time_func('Interpolation'):

        yo = interpolator.interp2d(
            y.copy(), fill_no_data=True, no_data_value=0.0, n_jobs=n_jobs
        )

    with time_func('Interpolation with smoothing'):

        yos = interpolator.interp2dx(
            y.copy(),
            fill_no_data=True,
            no_data_value=0.0,
            remove_outliers=True,
            dev_thresh=0.2,
            return_indexed=False,
            t=t,
            min_window=min_window,
            n_jobs=n_jobs,
        )

    with time_func('Interpolation with smoothing and indexed'):

        yoi = interpolator.interp2dx(
            y.copy(),
            fill_no_data=True,
            no_data_value=0.0,
            remove_outliers=True,
            dev_thresh=0.2,
            return_indexed=True,
            indices=indices,
            t=t,
            min_window=min_window,
            n_jobs=n_jobs,
        )

    # print(y.shape, y.dtype, yo.shape, yo.dtype)

    # plt.scatter(x1, y_raw, c='k', s=5, label='Raw')
    # plt.plot(x2, yo[1], c='purple', lw=0.5, label='Linear interp')
    # plt.plot(x2, yos[1], c='cyan', lw=0.5, label='Smoothed')
    # # plt.plot(x2, yid[1], c='magenta', lw=0.5, label='Smoothed (indexing)')
    # plt.plot(
    #     indices, yoi[1], c='g', ls='--', lw=0.5, label='Smoothed and indexed'
    # )

    # plt.show()
