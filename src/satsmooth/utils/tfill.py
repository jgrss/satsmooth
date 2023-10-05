import calendar
import concurrent.futures
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numba import jit
from sklearn.impute import IterativeImputer
from tqdm.auto import tqdm, trange

from ..preprocessing import fill_gaps, interp2d
from ..preprocessing.fill_gaps_py import fill_x
from . import nd_to_columns, prepare_x

ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor


@jit(nopython=True, nogil=True)
def calc_gap_lengths(ordinals, darray):

    """Calculates the length of gaps.

    Args:
        ordinals (1d array)
        darray (2d array)
    """

    nr, nc = darray.shape

    gap_counts = np.zeros((nr, nc), dtype='int64')

    for i in range(0, nr):

        j = 0

        while True:

            if darray[i, j] == 0:

                gap_total = 0

                # Find all gaps
                for g in range(j, nc):

                    if darray[i, g] == 0:
                        gap_total += ordinals[g]
                    else:
                        break

                for q in range(j, g):
                    gap_counts[i, q] = gap_total

                j = g + 1

            else:
                j += 1

            if j >= nc:
                break

    return gap_counts


def spatial_mean(yarray, w=5):

    """Spatial mean over rows and columns."""

    wh = int(w / 2.0)

    yarray = np.pad(yarray.copy(), ((0, 0), (w, w), (w, w)), mode='reflect')

    ymean = np.zeros(yarray.shape, dtype='float64')
    ymask = np.where(yarray > 0, 1, 0).astype('uint8')
    ycount = np.zeros(yarray.shape, dtype='uint8')

    offsets = np.linspace(-wh, wh, w).astype(int).tolist()

    for yoff in offsets:
        for xoff in offsets:
            ymean += np.roll(yarray, (0, yoff, xoff), (0, 1, 2))
            ycount += np.roll(ymask, (0, yoff, xoff), (0, 1, 2))

    ymean /= ycount.astype('float64')

    return ymean[:, w:-w, w:-w]


def tsmooth(yarray, q, method='linear', w=7):

    """Temporal smoothing over time dimension."""

    return (
        pd.DataFrame(data=yarray.copy())
        .interpolate(method=method, axis=1)
        .rolling(w, center=True, axis=1)
        .quantile(q)
        .bfill(axis=1)
        .ffill(axis=1)
    ).values


def kmeans(yarray, n_classes=5, max_iters=20, n_attempts=20, n_jobs=1):

    """Clusters data by k-means.

    Args:
        yarray (2d array): The data to cluster.
        n_classes (Optional[int]): The number of clusters.
        max_iters (Optional[int]): The maximum number of iterations.
        n_attempts (Optional[int]): The number of attempts.
        n_jobs (Optional[int]): The number of concurrent threads for interpolation.
    """

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iters,
        1.0,
    )

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    labels = cv2.kmeans(
        np.float32(interp2d(yarray.copy(), no_data_value=0.0, n_jobs=n_jobs)),
        n_classes,
        None,
        criteria,
        n_attempts,
        flags,
    )[1].flatten()

    return labels


class FillMixin(object):
    @staticmethod
    def get_gap_lengths(X, y):

        ordinals = np.array(
            [
                X[i + 1].toordinal() - X[i].toordinal()
                for i in range(0, X.shape[0] - 1)
            ]
            + [0],
            dtype='int64',
        )

        return calc_gap_lengths(ordinals, y)

    @staticmethod
    def check_day_dist(dta, dtb, max_days):

        """Checks if two dates fall within a day range.

        Args:
            dta (object): The first ``datetime.datetime`` object.
            dtb (object): The second ``datetime.datetime`` object.
            max_days (int): The maximum number of days.

        Returns:
            ``bool``
        """

        # Get the maximum number of days in the current month
        max_month_days = calendar.monthrange(dta.year, dtb.month)[1]
        month_day = min(dtb.day, max_month_days)

        dtc = datetime.strptime(
            f'{dta.year}-{dtb.month}-{month_day}', '%Y-%m-%d'
        )

        if abs(dta - dtc).days <= max_days:
            return True

        # Get the maximum number of days in the current month
        max_month_days = calendar.monthrange(dta.year - 1, dtb.month)[1]
        month_day = min(dtb.day, max_month_days)

        dtc = datetime.strptime(
            f'{dta.year-1}-{dtb.month}-{month_day}', '%Y-%m-%d'
        )

        if abs(dta - dtc).days <= max_days:
            return True

        # Get the maximum number of days in the current month
        max_month_days = calendar.monthrange(dta.year + 1, dtb.month)[1]
        month_day = min(dtb.day, max_month_days)

        dtc = datetime.strptime(
            f'{dta.year+1}-{dtb.month}-{month_day}', '%Y-%m-%d'
        )

        if abs(dta - dtc).days <= max_days:
            return True

        return False


def check_day_dist(dta, dtb, max_days):

    """Checks if two dates fall within a day range.

    Args:
        dta (object): The first ``datetime.datetime`` object.
        dtb (object): The second ``datetime.datetime`` object.
        max_days (int): The maximum number of days.

    Returns:
        ``bool``
    """

    # Get the maximum number of days in the current month
    max_month_days = calendar.monthrange(dta.year, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(f'{dta.year}-{dtb.month}-{month_day}', '%Y-%m-%d')

    if abs(dta - dtc).days <= max_days:
        return True

    # Get the maximum number of days in the current month
    max_month_days = calendar.monthrange(dta.year - 1, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(
        f'{dta.year-1}-{dtb.month}-{month_day}', '%Y-%m-%d'
    )

    if abs(dta - dtc).days <= max_days:
        return True

    # Get the maximum number of days in the current month
    max_month_days = calendar.monthrange(dta.year + 1, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(
        f'{dta.year+1}-{dtb.month}-{month_day}', '%Y-%m-%d'
    )

    if abs(dta - dtc).days <= max_days:
        return True

    return False


def sort_by_date(xinfo, data, time_index, max_days=30, max_years=2):

    """Sorts images by nearest date to reference.

    Args:
        xinfo (object): The data object.
        data (3d array): The data to slice and sort.
        time_index (int): The current time reference.
        max_days (Optional[int]): The maximum number of days difference.
        max_years (Optional[int]): The maximum number of years difference.

    Returns:
        3d array
    """

    # filter references by date
    try:
        target_date = xinfo.dates[time_index]
    except IndexError:
        return None, np.array([0])

    ref_idx = np.array(
        [
            i
            for i in range(0, xinfo.dates.shape[0])
            if (abs(target_date.year - xinfo.dates[i].year) <= max_years)
            and check_day_dist(target_date, xinfo.dates[i], max_days)
        ],
        dtype='int64',
    )

    if ref_idx.shape[0] == 0:
        return None, np.array([0])
    else:

        slice_idx = ref_idx[np.argsort(np.abs(ref_idx - time_index))]

        # Sort by nearest to reference ``time_index``
        return xinfo.dates[slice_idx], data[slice_idx]


def fill_layer(
    xinfo,
    y,
    didx,
    nrows,
    ncols,
    wmax,
    wmin,
    min_count,
    dev_thresh,
    max_days,
    max_years,
    nodata,
    num_threads,
    chunksize,
):

    """Fills gaps."""

    Results = namedtuple('Results', 'index array')

    # Sort the arrays in ascending order to the target date
    gap_dates, gap_array = sort_by_date(
        xinfo, y, didx, max_days=max_days, max_years=max_years
    )

    if (gap_array.shape[1] == 1) or (gap_array.max() == nodata):
        return Results(index=didx, array=y[didx])

    # The target date needs some data
    if gap_array[0].max() > 0:

        gdims, grows, gcols = gap_array.shape

        # Get the cluster labels
        cluster_labels = kmeans(
            nd_to_columns(gap_array, gdims, grows, gcols),
            n_classes=5,
            max_iters=20,
            n_attempts=20,
            n_jobs=num_threads,
        ).reshape(grows, gcols)

        if len(y.shape) > 3:
            gap_array = np.float64(gap_array)
        else:
            gap_array = np.float64(gap_array.reshape(gdims, 1, nrows, ncols))

        prefill_wmax_half = int(wmax / 2.0)

        gap_array = np.pad(
            gap_array,
            (
                (0, 0),
                (0, 0),
                (prefill_wmax_half, prefill_wmax_half),
                (prefill_wmax_half, prefill_wmax_half),
            ),
            mode='reflect',
        )

        cluster_labels = np.pad(
            cluster_labels,
            (
                (prefill_wmax_half, prefill_wmax_half),
                (prefill_wmax_half, prefill_wmax_half),
            ),
            mode='reflect',
        )

        # Fill gaps at time ``didx``
        # y[didx] = np.squeeze(fill_gaps(gap_array,
        #                                np.ascontiguousarray(cluster_labels, dtype='int64'),
        #                                np.array([1], dtype='float64'),  # not currently used
        #                                wmax=self.wmax,
        #                                wmin=self.wmin,
        #                                nodata=self.nodata,
        #                                min_count=self.min_count,
        #                                dev_thresh=self.dev_thresh,
        #                                n_jobs=self.num_threads,
        #                                chunksize=self.chunksize))[prefill_wmax_half:-prefill_wmax_half,
        #                                                           prefill_wmax_half:-prefill_wmax_half]

        ygap = gap_array[0]
        X, day_weights = fill_x(gap_dates, gap_array[1:], nodata)

        if X.max() == nodata:
            return Results(index=didx, array=y[didx])

        return Results(
            index=didx,
            array=np.squeeze(
                fill_gaps(
                    X,
                    ygap,
                    np.ascontiguousarray(cluster_labels, dtype='int64'),
                    np.ascontiguousarray(day_weights, dtype='float64'),
                    wmax=wmax,
                    wmin=wmin,
                    nodata=nodata,
                    min_count=min_count,
                    dev_thresh=dev_thresh,
                    n_jobs=num_threads,
                    chunksize=chunksize,
                )
            )[
                prefill_wmax_half:-prefill_wmax_half,
                prefill_wmax_half:-prefill_wmax_half,
            ],
        )

        # return Results(index=didx, array=np.squeeze(fill_gaps(gap_array,
        #                                                       np.ascontiguousarray(cluster_labels, dtype='int64'),
        #                                                       w=wmax,
        #                                                       nodata=nodata,
        #                                                       dev_thresh=dev_thresh)))

    else:
        return Results(index=didx, array=y[didx])


class SFill(FillMixin):
    def __init__(
        self,
        start='2000-07-01',
        end='2020-07-01',
        rule='D',
        skip='WMS',
        wmax=25,
        wmin=5,
        nodata=0,
        n_iters=1,
        max_days=30,
        max_years=0,
        min_count=20,
        dev_thresh=0.03,
        num_threads=1,
        chunksize=10,
    ):

        self.start = start
        self.end = end
        self.rule = rule
        self.skip = skip
        self.wmax = wmax
        self.wmin = wmin
        self.nodata = nodata
        self.n_iters = n_iters
        self.max_days = max_days
        self.max_years = max_years
        self.min_count = min_count
        self.dev_thresh = dev_thresh
        self.num_threads = num_threads
        self.chunksize = chunksize

    def impute(self, X, y):

        """
        Args:
            X (1d array)
            y (3d array)
        """

        xinfo = prepare_x(
            X, self.start, self.end, rule=self.rule, skip=self.skip
        )

        ndims, nrows, ncols = y.shape

        y = np.nan_to_num(
            y, nan=self.nodata, posinf=self.nodata, neginf=self.nodata
        )

        yout = y.copy()

        with ThreadPoolExecutor(
            max_workers=int(self.num_threads / 2.0)
        ) as executor:

            for iter_ in range(0, self.n_iters):

                # for didx in range(0, ndims):
                #
                #     res = fill_layer(xinfo, y, didx,
                #                      self.wmax,
                #                      self.dev_thresh,
                #                      self.max_days,
                #                      self.max_years,
                #                      self.nodata,
                #                      int(self.num_threads / 2.0))

                # CYTHON
                data_gen = (
                    (
                        xinfo,
                        y,
                        didx,
                        nrows,
                        ncols,
                        self.wmax,
                        self.wmin,
                        self.min_count,
                        self.dev_thresh,
                        self.max_days,
                        self.max_years,
                        self.nodata,
                        2,
                        self.chunksize,
                    )
                    for didx in range(0, ndims)
                )

                # STRIDES
                # data_gen = ((xinfo,
                #              y,
                #              didx,
                #              nrows,
                #              ncols,
                #              self.wmax,
                #              self.dev_thresh,
                #              self.max_days,
                #              self.max_years,
                #              self.nodata,
                #              2) for didx in range(0, ndims))

                futures = [
                    executor.submit(fill_layer, *args) for args in data_gen
                ]

                for f in tqdm(
                    concurrent.futures.as_completed(futures), total=ndims
                ):

                    results = f.result()
                    yout[results.index] = results.array

                y = np.nan_to_num(
                    yout.copy(),
                    nan=self.nodata,
                    posinf=self.nodata,
                    neginf=self.nodata,
                )

                futures = None

        return np.nan_to_num(
            yout, nan=self.nodata, posinf=self.nodata, neginf=self.nodata
        )

    def sort_by_date(self, xinfo, data, time_index, max_days=30, max_years=2):

        """Sorts images by nearest date to reference.

        Args:
            xinfo (object): The data object.
            data (3d array): The data to slice and sort.
            time_index (int): The current time reference.
            max_days (Optional[int]): The maximum number of days difference.
            max_years (Optional[int]): The maximum number of years difference.

        Returns:
            3d array
        """

        # filter references by date
        try:
            target_date = xinfo.dates[time_index]
        except IndexError:
            return None, np.array([0])

        ref_idx = np.array(
            [
                i
                for i in range(0, xinfo.dates.shape[0])
                if (abs(target_date.year - xinfo.dates[i].year) <= max_years)
                and self.check_day_dist(target_date, xinfo.dates[i], max_days)
            ],
            dtype='int64',
        )

        if ref_idx.shape[0] == 0:
            return None, np.array([0])
        else:

            # dates = xinfo.dates[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]
            # days = [abs((dt - dates[0]).days) for dt in dates]

            # Sort by nearest to reference ``time_index``
            return data[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]


class TFill(FillMixin):

    """
    Args:
        month (Optional[int]): The start and end month.
        min_gap_length (Optional[int]): The minimum allowed gap length to not be imputed.

    Example:
        >>> tf = TFill()
        >>> tf.impute(X, y)
    """

    def __init__(
        self,
        start_year=2000,
        end_year=2020,
        month=7,
        min_gap_length=30,
        batch_size=5,
        num_threads=1,
    ):

        self.start_year = start_year
        self.end_year = end_year
        self.month = month
        self.min_gap_length = min_gap_length
        self.batch_size = batch_size
        self.num_threads = num_threads

        self.half_batch = int(self.batch_size / 2.0)
        self.df = None

    def impute(self, X, y):

        """
        Args:
            X (1d array)
            y (3d array)
        """

        ndims, nrows, ncols = y.shape

        # 5x5 spatial average
        # yfill = spatial_mean(y, w=5)

        y = nd_to_columns(y, ndims, nrows, ncols)
        yfill = y.copy()

        self.df = pd.DataFrame(
            data=range(0, X.shape[0]), columns=['name'], index=X
        )

        gap_lengths = self.get_gap_lengths(X, y)

        for batch_year in trange(self.start_year, self.end_year + 1):

            #########################
            # Center the batch window
            #########################

            # Time series start
            if batch_year - self.half_batch < self.start_year:

                batch_start = batch_year
                batch_end = batch_year + self.batch_size

            # Time series end
            elif self.end_year - batch_year < self.batch_size:

                batch_start = self.end_year - self.batch_size
                batch_end = self.end_year

            else:

                batch_start = batch_year - self.half_batch
                batch_end = batch_year - self.half_batch + self.batch_size

            dates_batch = self.df.loc[
                f'{batch_start}-{self.month:02d}-01':f'{batch_end}-{self.month:02d}-01'
            ].index.to_pydatetime()

            if dates_batch.shape[0] > 0:

                date_list = self.get_unique_dates(dates_batch)

                if date_list.shape[0] > 2:

                    yh_med, yh_lwr, yh_upr = self.stack(
                        date_list, y, batch_year, batch_year + self.batch_size
                    )

                    yfill = self.fill(
                        date_list,
                        gap_lengths,
                        y,
                        yh_med,
                        yh_lwr,
                        yh_upr,
                        yfill,
                        batch_year,
                    )

        return yfill

    def fill(
        self, X_batch, gap_lengths, y, yh_med, yh_lwr, yh_upr, yfill, year_0
    ):

        # Get a 12-month slice
        frame_slice = self.df.loc[
            f'{year_0}-{self.month:02d}-01':f'{year_0+1}-{self.month:02d}-01'
        ]

        year_slice = frame_slice.index.to_pydatetime().tolist()

        # Get all dates for the current year slice
        date_slice_list = self._adjust_dates(
            year_0, year_slice, base_year=year_0
        )

        # Get the indices that fit into the full array
        idx_full = frame_slice.name.values

        idx, put_indices = self.get_indices(X_batch, idx_full, date_slice_list)

        # Impute with the rolling median
        yfill[:, idx] = np.where(
            y[:, idx] == 0,
            np.where(
                (yh_med[:, put_indices] <= yh_upr[:, put_indices])
                & (yh_med[:, put_indices] >= yh_lwr[:, put_indices])
                & (gap_lengths[:, idx] > self.min_gap_length),
                yh_med[:, put_indices],
                y[:, idx],
            ),
            y[:, idx],
        )

        return yfill

    @staticmethod
    def get_indices(X_batch, idx_full, date_slice_list):

        put_indices = []
        idx = []

        for pix_idx, dta in zip(idx_full, date_slice_list):

            pix_put_idx = np.where(X_batch == dta)[0]

            if pix_put_idx.shape[0] > 0:

                idx.append(pix_idx)
                put_indices.append(int(pix_put_idx[0]))

        return idx, put_indices

    def stack(self, X_batch, y, year_0, year_n):

        # Create the array holder
        yh = np.zeros((y.shape[0], X_batch.shape[0]), dtype='float64')

        # TODO: iterate in batches
        for year in range(year_0, year_n + 1):

            # Get a 12-month slice
            frame_slice = self.df.loc[
                f'{year}-{self.month:02d}-01':f'{year+1}-{self.month:02d}-01'
            ]

            year_slice = frame_slice.index.to_pydatetime().tolist()

            # Get all dates for the current year slice
            date_slice_list = self._adjust_dates(
                year, year_slice, base_year=year_0, check_monotonic=True
            )

            # Get the indices that fit into the full array
            idx_full = frame_slice.name.values

            idx, put_indices = self.get_indices(
                X_batch, idx_full, date_slice_list
            )

            yh[:, put_indices] = np.where(
                yh[:, put_indices] == 0,
                y[:, idx],
                np.where(y[:, idx] > 0.01, y[:, idx], yh[:, put_indices]),
            )

        # Get the cluster labels
        cluster_labels = kmeans(
            yh,
            n_classes=5,
            max_iters=20,
            n_attempts=20,
            n_jobs=self.num_threads,
        )

        yh[yh == 0] = np.nan
        yh_impute = yh.copy()
        yh_impute[np.isnan(yh_impute)] = 0
        yh_lwr = np.zeros(yh_impute.shape, dtype='float64')
        yh_upr = np.zeros(yh_impute.shape, dtype='float64')

        imp = IterativeImputer(
            missing_values=np.nan,
            sample_posterior=False,
            n_nearest_features=10,
            max_iter=10,
            initial_strategy='mean',
            min_value=0,
            max_value=1,
        )

        # Impute for each cluster
        for clab in np.unique(cluster_labels):

            # Get the current cluster
            clab_idx = np.where(cluster_labels == clab)[0]

            yh_clab = yh[clab_idx]
            yh_impute_clab = yh_impute[clab_idx]

            # Get the columns with data
            valid_idx = np.where(yh_impute_clab.max(axis=0) > 0)[0]

            # Impute the current cluster
            yh_impute_clab[:, valid_idx] = imp.fit_transform(yh_clab)

            # MICE
            # mice_iters = 5
            # for __ in range(0, mice_iters):
            #     yh_impute[:, valid_idx] += imp.fit_transform(yh)
            #
            # yh_impute[:, valid_idx] /= float(mice_iters)

            # yh_impute = tsmooth(yh, 0.5)
            yh_lwr_clab = tsmooth(yh_clab, 0.1, w=15)
            yh_upr_clab = tsmooth(yh_clab, 0.9, w=15)

            # Update the cluster
            yh_impute[clab_idx] = yh_impute_clab
            yh_lwr[clab_idx] = yh_lwr_clab
            yh_upr[clab_idx] = yh_upr_clab

        return yh_impute, yh_lwr, yh_upr

    def get_unique_dates(self, dates_batch):

        """
        Args:
            dates_batch (list): A list of datetime objects.
        """

        year_0 = dates_batch[0].year
        year_n = dates_batch[-1].year

        date_list = []

        for year in range(year_0, year_n):
            date_list = self._adjust_dates(
                year, dates_batch, adj_date_list=date_list, base_year=year_0
            )

        return np.sort(np.unique(date_list))

    def _adjust_dates(
        self,
        year,
        dt_objs,
        adj_date_list=None,
        base_year=2000,
        check_monotonic=False,
    ):

        if not adj_date_list:
            adj_date_list = []

        for dt in dt_objs:

            # Add 6 months to set July as fake January for sorting
            month = (
                (dt + pd.DateOffset(months=12 - self.month + 1))
                .to_pydatetime()
                .month
            )

            if dt.year in [year, year + 1]:

                if dt.day > calendar.monthrange(base_year + 1, month)[1]:
                    adj_dt = datetime.strptime(
                        f'{base_year+1}{month:02d}{calendar.monthrange(base_year+1, month)[1]:02d}',
                        '%Y%m%d',
                    )
                else:
                    adj_dt = datetime.strptime(
                        f'{base_year+1}{month:02d}{dt.day:02d}', '%Y%m%d'
                    )

            else:
                adj_dt = None

            if adj_dt is not None:
                adj_date_list.append(adj_dt)

        if check_monotonic:

            mono_increase = [True] + [
                True if adj_date_list[i] > adj_date_list[i - 1] else False
                for i in range(1, len(adj_date_list))
            ]

            if not all(mono_increase):

                for mi, mono_bool in enumerate(mono_increase):

                    if not mono_bool:

                        new_dt = adj_date_list[mi] - timedelta(days=1)
                        adj_date_list[mi] = datetime.strptime(
                            f'{base_year+1}{new_dt.month:02d}{new_dt.day:02d}',
                            '%Y%m%d',
                        )

        return adj_date_list
