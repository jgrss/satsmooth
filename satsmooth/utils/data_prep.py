import numpy as np
from collections import namedtuple
from datetime import datetime

import pandas as pd


def get_time_slice(dates_array, start_dt, end_dt, skip):

    start_idx = np.argmin(np.abs(dates_array - start_dt))
    end_idx = np.argmin(np.abs(dates_array - end_dt))
    dates_slice = dates_array[start_idx:end_idx]

    if skip == 'WMS':

        # Get the indices for the time series interval
        skip_idx = np.array([dtidx for dtidx in range(0, dates_slice.shape[0])
                             if (dates_slice[dtidx].timetuple().tm_mday == 1) or
                             (dates_slice[dtidx].timetuple().tm_mday % 7 == 0)],
                            dtype='uint64')

    else:
        skip_idx = np.arange(dates_slice.shape[0])

    return dates_slice, start_idx, end_idx, skip_idx


class ResampleDates(object):

    def __init__(self, indices, datetimes, rule='D'):

        self.indices_raw = np.ascontiguousarray(indices, dtype='float64')

        self.series_raw = pd.Series(self.indices_raw, index=pd.DatetimeIndex(datetimes))

        # Remove duplicate rows (these are grouped in the satellite data)
        self.series_raw = self.series_raw[~self.series_raw.index.duplicated()]

        # Get dates as `datetime` objects
        self.dates = self.series_raw.index.to_pydatetime()

        # Resample the dates
        if rule.startswith('D') and (len(rule) > 1):

            self.series_resamp = self.series_raw.resample('D').interpolate('linear')

            skip = int(rule[1:])

            # Get the indices for the time series interval
            skip_idx = np.array([dtidx for dtidx in range(0, self.series_resamp.shape[0])
                                 if (self.series_resamp.index[dtidx].to_pydatetime().timetuple().tm_mday < 30) and
                                 ((self.series_resamp.index[dtidx].to_pydatetime().timetuple().tm_mday == 1) or
                                  (self.series_resamp.index[dtidx].to_pydatetime().timetuple().tm_mday % skip == 0))],
                                dtype='uint64')

            self.series_resamp = self.series_resamp.iloc[skip_idx]

        else:
            self.series_resamp = self.series_raw.resample(rule).interpolate('linear')

        # Get the smooth values
        self.indices_resamp = np.ascontiguousarray(self.series_resamp.values, dtype='float64')
        self.dates_resamp = self.series_resamp.index.to_pydatetime()


def prepare_x(X, start, end, rule='D', skip='WMS', write_skip=10):

    """
    Prepares the X data for modelling

    Args:
        X (list): A list of datetime objects.
        start (str): The desired times series start date.
        end (str): The desired times series end date.
        rule (Optional[str]): The resampling rule.
        skip (Optional[str]): The time series skip interval. Choices are ['N', 'WMS'].
        
            N: No difference in indices
            WMS: Weekly month start
            
        write_skip (Optional[int]): The write skip interval.

    Returns:
        X information (namedtuple)
    """

    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    xd = [1000]
    dist = 1000
    for i in range(1, len(X)):
        dist += (X[i] - X[i-1]).days
        xd.append(dist)

    rd = ResampleDates(xd, X, rule=rule)

    ##############################################################################

    start_orig_idx = np.argmin(np.abs(rd.dates - start_dt))
    end_orig_idx = np.argmin(np.abs(rd.dates - end_dt))

    skip_orig_idx = np.array([i for i in range(0, rd.series_raw.shape[0])
                              if start_dt <= rd.series_raw.index[i].to_pydatetime() <= end_dt],
                             dtype='uint64')

    py_dates_slice, start_idx, end_idx, skip_idx = get_time_slice(rd.dates_resamp, start_dt, end_dt, skip)

    xd_interp = rd.indices_resamp[skip_idx+start_idx]

    skip_slice = py_dates_slice[skip_idx]
    
    # Indices for writing
    write_skip_idx = np.array([dtidx for dtidx in range(0, skip_slice.shape[0])
                               if (skip_slice[dtidx].timetuple().tm_mday < 30) and
                               ((skip_slice[dtidx].timetuple().tm_mday == 1) or
                                (skip_slice[dtidx].timetuple().tm_mday % write_skip == 0))],
                              dtype='int64')

    XInfo = namedtuple('XInfo', 'xd xd_smooth xd_interp dates dates_smooth start_orig_idx end_orig_idx skip_orig_idx start_idx end_idx skip_idx skip_orig_slice skip_slice write_skip_idx')

    return XInfo(xd=rd.indices_raw,
                 xd_smooth=rd.indices_resamp,
                 xd_interp=xd_interp,
                 dates=rd.dates,
                 dates_smooth=rd.dates_resamp,
                 start_orig_idx=start_orig_idx,
                 end_orig_idx=end_orig_idx,
                 skip_orig_idx=skip_orig_idx,
                 start_idx=start_idx,
                 end_idx=end_idx,
                 skip_idx=skip_idx,
                 skip_orig_slice=rd.dates[skip_orig_idx],
                 skip_slice=skip_slice,
                 write_skip_idx=write_skip_idx)


def sort_by_date(xinfo, data, time_index, max_days=30, max_years=2):

    """
    Sorts images by nearest date to reference

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
    except:
        return None, np.array([0])

    ref_idx = np.array([i for i in range(0, xinfo.dates.shape[0]) if
                        (abs(target_date.year - xinfo.dates[i].year) < max_years) and
                        (abs(target_date.timetuple().tm_yday - xinfo.dates[i].timetuple().tm_yday) <= max_days)], dtype='int64')

    if ref_idx.shape[0] == 0:
        return None, np.array([0])
    else:

        dates = xinfo.dates[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]

        days = [(dt - dates[0]).days for dt in dates]

        # Sort by nearest to reference ``time_index``
        return days, data[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]


def nd_to_rgb(data):

    """
    Reshapes an array from nd layout to RGB
    """

    if len(data.shape) != 3:
        raise AttributeError('The array must be 3 dimensions.')

    if data.shape[0] != 3:
        raise AttributeError('The array must be 3 bands.')

    return np.ascontiguousarray(data.transpose(1, 2, 0))


def rgb_to_nd(data):

    """
    Reshapes an array RGB layout to nd layout
    """

    if len(data.shape) != 3:
        raise AttributeError('The array must be 3 dimensions.')

    if data.shape[2] != 3:
        raise AttributeError('The array must be 3 bands.')

    return np.ascontiguousarray(data.transpose(2, 0, 1))


def nd_to_columns(data, layers, rows, columns):

    """
    Reshapes an array from nd layout to [samples (rows*columns) x dimensions]
    """

    if layers == 1:
        return np.ascontiguousarray(data.flatten()[:, np.newaxis])
    else:
        return np.ascontiguousarray(data.transpose(1, 2, 0).reshape(rows*columns, layers))


def columns_to_nd(data, layers, rows, columns):

    """
    Reshapes an array from columns layout to [layers x rows x columns]
    """

    if layers == 1:
        return np.ascontiguousarray(data.reshape(columns, rows).T)
    else:
        return np.ascontiguousarray(data.T.reshape(layers, rows, columns))


def scale_min_max(xv, mni, mxi, mno, mxo):
    return ((((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno).clip(mno, mxo)
