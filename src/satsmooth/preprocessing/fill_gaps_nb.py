import numba as nb
import numpy as np
from numba import float64, int64, int_, jit, prange


@jit('int_(int_, int_)', nopython=True, nogil=True, fastmath=True, cache=True)
def _get_rindex(col_dims, index):
    return int_(np.floor(float64(index) / float64(col_dims)))


@jit(
    'int_(int_, int_, int_)',
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def _get_cindex(col_dims, index, row_index):
    return int_(index - float64(col_dims) * row_index)


@jit(
    'float64(float64, float64, float64)',
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def edist(xloc, yloc, hw):
    return ((xloc - hw) ** 2 + (yloc - hw) ** 2) ** 0.5


@jit(
    'float64(float64, float64, float64)',
    nopython=True,
    nogil=True,
    fastmath=True,
    cache=True,
)
def logistic_func(x, x0, r):
    denom = 1.0 + np.exp(-r * (x - x0))
    return 1.0 / denom if denom != 0 else 0.0


@jit(
    'float64(int_, ' 'float64[:, :], ' 'float64, ' 'float64, ' 'float64)',
    nopython=True,
    nogil=True,
    boundscheck=False,
    cache=True,
)
def _apply(f__, results_, center_avg, nodata, dev_thresh):

    """Applies the least squares solution to estimate the missing value."""

    y_mean = results_[f__, 0]
    beta1 = results_[f__, 0]
    beta2 = results_[f__, 0]
    alpha = results_[f__, 0]

    # y = b0*x + b1*x^2 + a
    estimate = beta1 * center_avg + beta2 * center_avg**2 + alpha

    if estimate > 1:
        estimate = nodata
    elif estimate < 0:
        estimate = nodata
    elif estimate > (y_mean + dev_thresh * 10.0):
        estimate = y_mean
    elif estimate < (y_mean - dev_thresh * 10.0):
        estimate = y_mean

    return estimate


@jit(
    'void(int_, '
    'float64[:, :], '
    'float64[:, :], '
    'float64, '
    'float64, '
    'float64, '
    'int_, '
    'float64[:, :], '
    'float64[:, :])',
    nopython=True,
    nogil=True,
    boundscheck=False,
    cache=True,
)
def _calc_wlsq(
    f_, xdata, ydata, xavg1, xavg2, yavg, count, weights, lin_coeffs_
):

    """Calculates the least squares slope and intercept."""

    xvar1 = 0.0
    xvar2 = 0.0
    yvar = 0.0
    xx_cov = 0.0
    x1y_cov = 0.0
    x2y_cov = 0.0

    for s in range(0, count):

        xdev1 = weights[f_, s] * (xdata[f_, s] - xavg1)
        xdev2 = weights[f_, s] * (xdata[f_, s] ** 2 - xavg2)
        ydev = weights[f_, s] * (ydata[f_, s] - yavg)

        # Variance
        xvar1 += xdev1**2
        xvar2 += xdev2**2
        yvar += ydev**2

        x1y_cov += xdev1 * ydev
        x2y_cov += xdev2 * ydev

        xx_cov += xdev1 * xdev2

    denom = (xvar1 * xvar2) - xx_cov**2

    # Polynomial least squares
    beta1 = (
        ((xvar2 * x1y_cov) - (xx_cov * x2y_cov)) / denom if denom != 0 else 0.0
    )
    beta2 = (
        ((xvar1 * x2y_cov) - (xx_cov * x1y_cov)) / denom if denom != 0 else 0.0
    )
    alpha = yavg - (beta1 * xavg1) - (beta2 * xavg2)

    lin_coeffs_[f_, 0] = yavg
    lin_coeffs_[f_, 1] = beta1
    lin_coeffs_[f_, 2] = beta2
    lin_coeffs_[f_, 3] = alpha


@jit(
    'void(float64[:, :, :, :], '
    'int64[:, :], '
    'int64[:], '
    'float64[:, :], '
    'int_, int_, int_, int_, int_, int_, int_, '
    'float64,'
    'float64[:, :])',
    nopython=True,
    nogil=True,
    boundscheck=False,
    cache=True,
)
def _get_center_mean(
    indata,
    clusters,
    lc_classes,
    x_cluster_,
    b,
    fff,
    iii,
    jjj,
    dims,
    ci,
    hw,
    nodata,
    count_vector,
):

    """Gets the center average of the reference data."""

    offset = hw - int64(ci / 2.0)

    # sum
    count_vector[fff, 0] = 0.0

    # average
    count_vector[fff, 1] = 0.0

    for d in range(0, dims):

        for m in range(0, ci):
            for n in range(0, ci):

                # Reference X value
                ref_xvalue = indata[d, b, iii + m + offset, jjj + n + offset]

                # Reference X class
                ref_xclass = clusters[iii + m + offset, jjj + n + offset]

                # Update the class count
                # [X mean, X class, <class counts>]
                x_cluster_[fff, 2 + ref_xclass] += 1.0

                if ref_xvalue != nodata:

                    count_vector[fff, 0] += ref_xvalue
                    count_vector[fff, 1] += 1

        # Break if there is data
        if count_vector[fff, 0] > 0:
            break

    if count_vector[fff, 0] > 0:

        ref_xclass = lc_classes[0]
        ref_xclass_mode = x_cluster_[fff, 2]

        # Get the class with the highest count
        for cidx in range(3, lc_classes.shape[0]):

            if x_cluster_[fff, cidx] > ref_xclass_mode:
                ref_xclass = lc_classes[cidx - 2]
                ref_xclass_mode = x_cluster_[fff, cidx]

        x_cluster_[fff, 0] = count_vector[fff, 0] / count_vector[fff, 1]
        x_cluster_[fff, 1] = float64(ref_xclass)

        # Get the updated average for the target class

        count_vector[fff, 0] = 0.0
        count_vector[fff, 1] = 0.0

        for d in range(0, dims):
            for m in range(0, ci):
                for n in range(0, ci):

                    # Reference X value
                    ref_xvalue = indata[
                        d, b, iii + m + offset, jjj + n + offset
                    ]

                    # Reference X class
                    ref_xclass_spass = clusters[
                        iii + m + offset, jjj + n + offset
                    ]

                    if (ref_xvalue != nodata) and (
                        ref_xclass_spass == ref_xclass
                    ):

                        count_vector[fff, 0] += ref_xvalue
                        count_vector[fff, 1] += 1

            # Break if there is data
            if count_vector[fff, 0] > 0:
                break

        if count_vector[fff, 1] > 0:
            x_cluster_[fff, 0] = count_vector[fff, 0] / count_vector[fff, 1]


@jit(
    'void(float64[:, :, :, :], '
    'int64[:, :], '
    'float64[:, :, :], '
    'int_, int_, int_, int_, int_, int_, int_, int_, int_, '
    'float64, '
    'float64, '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :])',
    nopython=True,
    nogil=True,
    boundscheck=False,
    cache=True,
)
def _estimate_gap(
    indata,
    clusters,
    output__,
    b,
    ffff,
    iiii,
    jjjj,
    dims,
    wmin,
    wmax,
    hw,
    min_thresh,
    nodata,
    dev_thresh,
    x_cluster__,
    xdata,
    ydata,
    weights,
    spatial_weights,
    lin_coeffs,
):

    count = 0
    yavg = 0.0

    max_vct_size = int64(wmax * wmax * dims)

    tar_xclass = int64(x_cluster__[ffff, 1])

    for rzi in range(0, max_vct_size):
        xdata[ffff, rzi] = 0.0
        ydata[ffff, rzi] = 0.0

    xavg1 = 0.0
    xavg2 = 0.0

    # Search for data over increasing windows
    for wi in range(wmin, wmax + 2, 2):

        # Adjustment for different sized windows
        offset = hw - int64(wi / 2.0)

        # Iterate over each reference file to fill the window
        for d in range(1, dims):

            for m in range(0, wi):
                for n in range(0, wi):

                    # Skip the center that has already been checked
                    if wi > wmin:

                        if (
                            (m > 0)
                            and (m < wi - 1)
                            and (n > 0)
                            and (n < wi - 1)
                        ):
                            continue

                    # Get the reference cluster value
                    ref_xclass = clusters[iiii + m + offset, jjjj + n + offset]

                    # Only take samples with the same cluster as the target
                    if ref_xclass == tar_xclass:

                        # Target value
                        yvalue = indata[
                            0, b, iiii + m + offset, jjjj + n + offset
                        ]

                        # Reference value
                        xvalue = indata[
                            d, b, iiii + m + offset, jjjj + n + offset
                        ]

                        # X and y must have data
                        if (xvalue != nodata) and (yvalue != nodata):

                            # Squared difference
                            sq_diff = (xvalue - yvalue) ** 2

                            # X and y must have a low deviation
                            if sq_diff <= dev_thresh:

                                xdata[ffff, count] = xvalue
                                ydata[ffff, count] = yvalue
                                weights[ffff, count] = spatial_weights[m, n]

                                xavg1 += xvalue
                                xavg2 += xvalue**2
                                yavg += yvalue

                                count += 1

            if count >= min_thresh:
                break

        if count >= min_thresh:
            break

    if count >= min_thresh:

        countf = float64(count)

        # Window average
        xavg1 /= countf
        xavg2 /= countf
        yavg /= countf

        # Std. dev. of [x, y], slope, intercept
        _calc_wlsq(
            ffff, xdata, ydata, xavg1, xavg2, yavg, count, weights, lin_coeffs
        )

        # Calculate the least squares solution
        estimate = _apply(
            ffff, lin_coeffs, x_cluster__[ffff, 0], nodata, dev_thresh
        )

        output__[b, iiii + hw, jjjj + hw] = estimate


@jit(
    'void(int_, '
    'float64[:, :, :, :], '
    'int64[:, :], '
    'int64[:], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :, :], '
    'int_, int_, int_, int_, int_, int_, int_, '
    'float64, '
    'float64, '
    'float64[:, :], '
    'float64[:, :], '
    'float64[:, :])',
    nopython=True,
    nogil=True,
    boundscheck=False,
    cache=True,
)
def _fill_gap(
    ff,
    indata,
    clusters,
    lc_classes,
    x_cluster,
    xdata,
    ydata,
    weights,
    output_,
    b,
    col_dims,
    wmin,
    wmax,
    hw,
    dims,
    min_count,
    nodata,
    dev_thresh,
    spatial_weights,
    count_vector,
    lin_coeffs,
):

    ii = _get_rindex(col_dims, ff)
    jj = _get_cindex(col_dims, ff, ii)

    # Center target sample
    if indata[0, b, ii + hw, jj + hw] == nodata:

        for xxi in range(0, 2 + lc_classes.shape[0]):
            x_cluster[ff, xxi] = 0.0

        # Get an average of the center value
        for ci in range(3, 9, 2):

            # center_avg = X
            _get_center_mean(
                indata,
                clusters,
                lc_classes,
                x_cluster,
                b,
                ff,
                ii,
                jj,
                dims,
                ci,
                hw,
                nodata,
                count_vector,
            )

            if x_cluster[ff, 0] > 0:
                break

        if x_cluster[ff, 0] > 0:

            _estimate_gap(
                indata,
                clusters,
                output_,
                b,
                ff,
                ii,
                jj,
                dims,
                wmin,
                wmax,
                hw,
                min_count,
                nodata,
                dev_thresh,
                x_cluster,
                xdata,
                ydata,
                weights,
                spatial_weights,
                lin_coeffs,
            )


@jit(
    'float64[:, :, :](float64[:, :, :, :], '
    'int64[:, :], '
    'int_, '
    'int_, '
    'float64, '
    'int_, '
    'float64, '
    'int_)',
    parallel=True,
    cache=True,
)
def _fill_gaps(
    indata, clusters, wmax, wmin, nodata, min_count, dev_thresh, n_jobs
):

    eps = 1e-08

    dims = indata.shape[0]
    bands = indata.shape[1]
    rows = indata.shape[2]
    cols = indata.shape[3]
    hw = int64(wmax / 2.0)
    row_dims = rows - int64(hw * 2.0)
    col_dims = cols - int64(hw * 2.0)
    nsamples = int64(row_dims * col_dims)

    num_threads = nb.set_num_threads(n_jobs)

    # Get the maximum distance for the spatial weights
    max_edist = edist(0.0, 0.0, hw) + eps

    max_vct_size = int64(wmax * wmax * dims)

    # Initialize arrays
    output = indata[0].copy()
    lc_classes = np.int64(np.unique(clusters))
    x_cluster = np.zeros((nsamples, 2 + lc_classes.shape[0]), dtype='float64')
    xdata = np.zeros((nsamples, max_vct_size), dtype='float64')
    ydata = np.zeros((nsamples, max_vct_size), dtype='float64')
    weights = np.zeros((nsamples, max_vct_size), dtype='float64')
    lin_coeffs = np.zeros((nsamples, 4), dtype='float64')
    spatial_weights = np.zeros((wmax, wmax), dtype='float64')
    count_vector = np.zeros((nsamples, 2), dtype='float64')

    # Fill the spatial weights
    for mm in range(0, wmax):
        for nn in range(0, wmax):
            spatial_weights[mm, nn] = logistic_func(
                1.0
                - ((edist(float64(nn), float64(mm), hw) + eps) / max_edist),
                0.5,
                10.0,
            )

    for f in prange(0, nsamples):

        for b in range(0, bands):

            _fill_gap(
                f,
                indata,
                clusters,
                lc_classes,
                x_cluster,
                xdata,
                ydata,
                weights,
                output,
                b,
                col_dims,
                wmin,
                wmax,
                hw,
                dims,
                min_count,
                nodata,
                dev_thresh,
                spatial_weights,
                count_vector,
                lin_coeffs,
            )

    return output


@jit(
    'float64[:, :, :](float64[:, :, :, :], '
    'int64[:, :], '
    'float64[:], '
    'int_, '
    'int_, '
    'float64, '
    'int_, '
    'float64, '
    'int_, '
    'int_)',
    cache=True,
)
def fill_gaps(
    gap_data,
    cluster_data,
    days,
    wmax=25,
    wmin=9,
    nodata=0,
    min_count=20,
    dev_thresh=0.02,
    n_jobs=1,
    chunksize=1,
):

    """Fills data gaps using spatial-temporal weighted least squares linear
    regression.

    Args:
        gap_data (4d array): Layers x bands x rows x columns. The first layer is the target and the remaining layers
            are the references. The reference layers should be sorted from the date closest to the target to the
            date furthest from the target date.
        cluster_data (2d array): Land cover clusters.
        wmax (Optional[int]): The maximum window size.
        wmin (Optional[int]): The minimum window size.
        nodata (Optional[int]): The 'no data' value to fill.
        min_count (Optional[int]): The minimum required number of samples to fit a window.
        dev_thresh (Optional[float]): The squared deviation threshold for similar values.
        n_jobs (Optional[int]): The number of bands to process in parallel.
        chunksize (Optional[int]): The parallel thread chunksize.

    Returns:
        3d ``numpy.ndarray`` (bands x rows x columns)
    """

    return np.float64(
        _fill_gaps(
            gap_data,
            cluster_data,
            wmax,
            wmin,
            float64(nodata),
            min_count,
            dev_thresh,
            n_jobs,
        )
    )
