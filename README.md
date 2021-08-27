[](#mit-license)[](#python-3.6)[](#package-version)

[![MIT license](https://img.shields.io/badge/License-MIT-black.svg)](https://lbesson.mit-license.org/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-black.svg)](https://www.python.org/downloads/release/python-360/)
![Package version](https://img.shields.io/badge/version-1.5.11-blue.svg?cacheSeconds=2592000)

Satellite signal interpolation and smoothing
---

## Dynamic temporal smoothing (DTS)

> Graesser, Jordan and Stanimirova, Radost, and Friedl, Mark. (2021) Reconstruction of satellite time series with a dynamic smoother. _EarthArXiv_.

```text
@article{graesser_stanimirova_friedl_2021,
  title={Reconstruction of satellite time series with a dynamic smoother},
  author={Graesser, Jordan and Stanimirova, Radost and Friedl, Mark A},
  year={2021},
  publisher={EarthArXiv}
}
```

![](data/param_diagram.png)

### Imports

```python
import satsmooth as sm
import numpy as np
```

### Prepare the dates 

```python
time_names = [datetime.datetime(), ..., datetime.datetime()]

# Prepare the dates
start = '2010-3-01'
end = '2011-11-01'
skip = 7
xinfo = sm.prepare_x(time_names, start, end, skip)
```

Reshape the data

```python
# Load data into a numpy array shaped (time x rows x columns)
dims, nrows, ncols = y.shape

# Reshape from (time x rows x columns) to (samples x time)
y = sm.utils.nd_to_columns(y, dims, nrows, ncols)
```

Setup the interpolater

```python
# Initiate a linear interpolater for a sparse-->daily transform
interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)

# Setup indices to return a sparse, regularly gridded output
indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype='uint64')
```

Smooth the data

```python
# This function applies interpolation, outlier detection, regridding, 
# and smoothing in one parallel iteration
y = interpolator.interpolate_smooth(np.ascontiguousarray(y, dtype='float64'),
                           fill_no_data=True,       # fill 'no data' by linear interpolation
                           no_data_value=nodata,    # set as 0
                           remove_outliers=True,    # search for outliers first
                           max_outlier_days1=120,   # linear
                           max_outlier_days2=120,   # polynomial
                           min_outlier_values=7,
                           outlier_iters=1,
                           dev_thresh1=0.2,
                           dev_thresh2=0.2,
                           return_indexed=True,     # return the ~weekly data rather than daily
                           indices=indices,         # the indices to return
                           max_window=61,
                           min_window=21,
                           mid_g=0.5, 
                           r_g=-10.0, 
                           mid_k=0.5,
                           r_k=-10.0,
                           mid_t=0.5,
                           r_t=15.0,
                           sigma_color=0.1,
                           n_iters=2,
                           n_jobs=8)
```