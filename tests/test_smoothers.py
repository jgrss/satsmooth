import unittest
from datetime import datetime, timedelta

import numpy as np
from skimage.exposure import rescale_intensity

import satsmooth as sm
from satsmooth.anc import AncSmoothers
from satsmooth.utils import columns_to_nd, nd_to_columns
from satsmooth.utils.data_prep import prepare_x

sparse_days = 10
x = np.arange(0, 1005, sparse_days).astype(float)
x_smooth = np.arange(0, 1000).astype(float)

dts = [datetime.strptime('2000001', '%Y%j')]
for i in range(0, x.shape[0] - 1):
    dts.append(dts[-1] + timedelta(days=sparse_days))

xinfo = prepare_x(dts, '2000-1-1', datetime.strftime(dts[-1], '%Y-%m-%d'), 7)

interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)

# np.random.seed(50)


def f(xdata):
    return rescale_intensity(
        np.sin(xdata), out_range=(0.2, 0.7)
    ) + np.random.uniform(
        low=-0.05, high=0.05, size=xdata.shape[0]
    )  # np.random.normal(scale=0.05, size=xdata.shape[0])


# y_in = np.random.random((2, xinfo.xd.shape[0]))
y_in_orig = np.array(
    [
        rescale_intensity(np.sin(x), out_range=(0.2, 0.7)).tolist()
        for i in range(0, 2)
    ]
).clip(0, 1)
y_in = np.array([f(x).tolist() for i in range(0, 2)]).clip(0, 1)

nodata = 0

for i in range(0, y_in.shape[0]):
    idx = np.random.choice(
        range(0, y_in.shape[1]), size=int(0.3 * y_in.shape[1]), replace=False
    )
    y_in[i, idx] = nodata

indices = np.ascontiguousarray(
    xinfo.skip_idx + xinfo.start_idx, dtype='uint64'
)

y = interpolator.interpolate_smooth(
    np.ascontiguousarray(y_in.copy(), dtype='float64'),
    fill_no_data=True,
    no_data_value=nodata,
    remove_outliers=True,
    max_outlier_days1=120,
    max_outlier_days2=90,
    min_outlier_values=5,
    outlier_iters=1,
    dev_thresh1=0.2,
    dev_thresh2=0.2,
    return_indexed=True,
    indices=indices,
    t=51,
    min_window=7,
    mid_g=0.5,
    r_g=-0.5,
    mid_k=0.5,
    r_k=-0.5,
    mid_t=0.5,
    r_t=15.0,
    sigma_color=0.1,
    n_iters=1,
    envelope=1,
    method='dynamic',
    n_jobs=1,
)

smt = AncSmoothers(
    xinfo,
    columns_to_nd(
        np.ascontiguousarray(y_in.copy(), dtype='float64'),
        xinfo.xd.shape[0],
        1,
        2,
    ),
    pad=81,
    n_jobs=1,
)


def apply_cspline():
    def reshape(data):
        return nd_to_columns(data, *data.shape)

    return reshape(smt.cspline(s=0.1))


# y_cspline = apply_cspline()
#
# import matplotlib.pyplot as plt
# plt.plot(x, y_in_orig[0], color='grey')
# plt.scatter(x, y_in[0], color='k')
# plt.plot(indices, y[0], color='orange', lw=2, label='DTS')
# plt.plot(indices, y_cspline[0], color='purple', lw=2, label='Cubic spline')
# plt.legend()
# plt.ylim(0, 1)
# plt.show()


class TestSmoothers(unittest.TestCase):
    def test_out_shape(self):
        self.assertEqual(y.shape[1], indices.shape[0])


if __name__ == '__main__':
    unittest.main()
