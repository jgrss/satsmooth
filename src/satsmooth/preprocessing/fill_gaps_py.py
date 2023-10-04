import functools
import time

import numpy as np
from numba import jit

np_nanmean = np.nanmean
np_nansum = np.nansum


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):

        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic

        print(f"Elapsed time for {func.__name__}: {elapsed_time:0.4f} s")

        return value

    return wrapper_timer


def scale_min_max(x, min_in, max_in, min_out, max_out):
    """Scales linearly."""
    return np.clip(
        (((max_out - min_out) * (x - min_in)) / (max_in - min_in)) + min_out,
        min_out,
        max_out,
    )


def edist(xloc, yloc, hw):
    """Calculates the euclidean distance."""
    return ((xloc - hw) ** 2 + (yloc - hw) ** 2) ** 0.5


def logistic_func(x, x0, r):
    """Calculates the logistic function."""
    return 1.0 / (1.0 + np.exp(-r * (x - x0)))


def fill_x(xdates, xdata, nodata):

    """Fills X reference data.

    Args:
        xdates (list): A list of ``datetime`` objects.
        xdata (4d array): A stack of reference data, shaped [time x bands x rows x columns].
        nodata (float): The 'no data' value.
    """

    nlyrs, nbands, nrows, ncols = xdata.shape

    # Fill values
    X = xdata[0].copy()

    # Weights
    w = np.where(X != nodata, 1, 0)

    min_day_diff = 0.0
    max_day_diff = float(abs(int((xdates[-1] - xdates[0]).days)))

    for k in range(1, nlyrs):

        # Update the fill values
        X = np.where((X == nodata) & (xdata[k] != nodata), xdata[k], nodata)

        # Update the weights
        day_diff = float(abs(int((xdates[k] - xdates[0]).days)))
        # Scale nearest days -> [0,1]
        day_weight = 1.0 - scale_min_max(
            day_diff, min_day_diff, max_day_diff, 0.0, 1.0
        )
        day_weight = logistic_func(day_weight, 0.25, 7)
        w = np.where((w == 0) & (X != nodata), day_weight, 0)

        if X.min() > nodata:
            break

    return X, w.squeeze()


def get_spatial_weights(w, hw):

    """Calculates spatial weights.

    Args:
        w (int): The window size.
        hw (int): The half window size.
    """

    spatial_weights = np.zeros((w, w), dtype='float64')

    max_edist = edist(0.0, 0.0, hw) + 0.001

    # Fill the spatial weights
    for m in range(0, w):
        for n in range(0, w):
            spatial_weights[m, n] = logistic_func(
                1.0 - ((edist(float(n), float(m), hw) + 0.001) / max_edist),
                0.5,
                10.0,
            )

    return spatial_weights[np.newaxis, np.newaxis]


def get_strides(array, w):

    height, width = array.shape

    strided_array = np.lib.stride_tricks.as_strided(
        array,
        shape=[height - w + 1, width - w + 1, w, w],
        strides=array.strides + array.strides,
    )

    return strided_array


@jit(nopython=True, nogil=True)
def get_masks(cluster_strides, cluster_centers):

    h, w, r, c = cluster_strides.shape

    mask = np.zeros((h, w, r, c), dtype='uint8')

    for index, r in np.ndenumerate(np.empty((h, w))):

        sl = (
            slice(index[0], index[0] + 1),
            slice(index[1], index[1] + 1),
            slice(0, None),
            slice(0, None),
        )

        mask[sl] = np.where(
            cluster_strides[sl] == cluster_centers[index[0], index[1]], 1, 0
        )

    return mask


def lstsq(
    X_strides=None, y_strides=None, w_strides=None, X_centers=None, nodata=None
):

    """Ordinary least squares regression."""

    yavg = np_nanmean(y_strides, axis=(2, 3))
    xavg1 = np_nanmean(X_strides, axis=(2, 3))
    xavg2 = np_nanmean(X_strides**2, axis=(2, 3))

    ydev = w_strides * (y_strides - yavg[:, :, np.newaxis, np.newaxis])
    xdev1 = w_strides * (X_strides - xavg1[:, :, np.newaxis, np.newaxis])
    xdev2 = w_strides * (X_strides**2 - xavg2[:, :, np.newaxis, np.newaxis])

    xvar1 = np_nansum(xdev1**2, axis=(2, 3))
    xvar2 = np_nansum(xdev2**2, axis=(2, 3))

    x1y_cov = np_nansum(xdev1 * ydev, axis=(2, 3))
    x2y_cov = np_nansum(xdev2 * ydev, axis=(2, 3))
    xx_cov = np_nansum(xdev1 * xdev2, axis=(2, 3))

    # Polynomial least squares
    denom = ((xvar1 * xvar2) - xx_cov**2) + 1e-9
    beta1 = ((xvar2 * x1y_cov) - (xx_cov * x2y_cov)) / denom
    beta2 = ((xvar1 * x2y_cov) - (xx_cov * x1y_cov)) / denom
    alpha = yavg - (beta1 * xavg1) - (beta2 * xavg2)

    est = np.nan_to_num(
        beta1 * X_centers + beta2 * X_centers**2 + alpha,
        nan=nodata,
        posinf=nodata,
        neginf=nodata,
    )

    return est


def fill_gaps(data, clusters, w, nodata=0, dev_thresh=0.1):

    """
    Args:
        data (4d array): [time x bands x rows x columns], where
            t0 = target layer and
            t1..tn = predictor layers
        clusters (2d array): The clusters.
        w (int): The 2d window size.
        nodata (Optional[int | float]): The 'no data' value.
        dev_thresh (Optional[float]): The deviation threshold.
    """

    data[data == nodata] = np.nan

    hw = int(w / 2.0)

    # Target layer (y)
    y = data[0]

    # Fill layers (X)
    # TODO: use decay weight
    X = fill_x(None, data[1:], nodata)

    # Spatial weights window
    spatial_weights = get_spatial_weights(w, hw)

    # K-means clusters
    cluster_strides = get_strides(clusters, w)
    cluster_centers = cluster_strides[(slice(0, None), slice(0, None), hw, hw)]
    # Numba warmup
    __ = get_masks(cluster_strides[:2, :2, :, :], cluster_centers[:2, :2])
    cluster_masks = get_masks(cluster_strides, cluster_centers)

    s1, s2, s3, s4 = cluster_strides.shape

    # Spatial weights as strides
    w_strides = np.repeat(np.repeat(spatial_weights, s1, axis=0), s2, axis=1)

    # Output estimates array
    estimates = np.zeros((y.shape[0], s1, s2), dtype=data.dtype)

    # Iterate over each band
    for b in range(0, y.shape[0]):

        # Get the y strides
        y_strides = get_strides(y[b], w)

        # Get the existing center values
        y_centers = y_strides[(slice(0, None), slice(0, None), hw, hw)]

        # Get the X strides
        X_strides = get_strides(X[b], w)

        # Mask for the center clusters
        X_strides = X_strides * cluster_masks

        # 3x3 average (ignoring 'no data') around the center value
        X_centers = np_nanmean(
            X_strides[
                (
                    slice(0, None),
                    slice(0, None),
                    slice(hw - 1, hw + 2),
                    slice(hw - 1, hw + 2),
                )
            ],
            axis=(2, 3),
        )

        # Squared deviation = (x - y)^2
        sq_diff = (X_strides - y_strides) ** 2

        # Mask where data is valid in both and the deviation is low
        valid_mask = np.where(
            ~np.isnan(y_strides)
            & ~np.isnan(X_strides)
            & (X_strides != 0)
            & (sq_diff <= dev_thresh),
            1,
            0,
        )

        sq_diff = None

        if valid_mask.max() == 1:

            # Mask valid data
            y_strides = y_strides * valid_mask
            X_strides = X_strides * valid_mask

            y_strides[y_strides == 0] = np.nan
            X_strides[X_strides == 0] = np.nan

            # Least squares regression
            estimates[b] = np.where(
                estimates[b] != 0,
                lstsq(
                    X_strides=X_strides,
                    y_strides=y_strides,
                    w_strides=w_strides,
                    X_centers=X_centers,
                    nodata=nodata,
                ),
                y_centers,
            )

    return estimates
