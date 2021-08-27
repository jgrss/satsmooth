import itertools
import multiprocessing as multi

from ._dl import gd
from ._lowess_smooth import lowess_smooth
from .. import remove_outliers, interp2d, LinterpMulti
from ..utils import nd_to_columns, columns_to_nd, scale_min_max
from .. import rolling_mean2d, peaks_valleys2d, group_peaks_valleys2d
# from ._bspline import bezier_curve

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from skimage.segmentation import active_contour
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
# from numba import jit, prange, set_num_threads


def _cspline_func(*args):

    xd, xd_smooth, y_, s, optimize = list(itertools.chain(*args))

    w = np.where(y_ > 0, 1, 0.1).astype('float64')

    if optimize:

        min_gcv = 1e9
        eps = 0.00001

        for s_ in s:

            spl_ = UnivariateSpline(xd, y_, w=w, s=s_)

            # Degrees of freedom (degree = 3)
            dof = spl_.get_knots().shape[0] + 3 + 1
            n = spl_.get_coeffs().shape[0]

            # Weighted sum of squared errors
            sse = spl_.get_residual()

            # Roughness penalty
            denom = n - dof*s_ + eps
            gcv = (n / denom) / ((sse / denom) + eps)

            if gcv < min_gcv:
                min_gcv = gcv
                spl = spl_

    else:
        spl = UnivariateSpline(xd, y_, w=w, s=s, k=3)

    return spl(xd_smooth)


# @jit(parallel=True)
def cspline_func(xinfo, indices, yarray, s, optimize, n_jobs, chunksize):

    data_gen = ((xinfo.xd, xinfo.xd_smooth, yarray[i], s, optimize) for i in range(0, yarray.shape[0]))

    with multi.Pool(n_jobs) as executor:
        yout = [result[indices] for result in executor.imap(_cspline_func, data_gen, chunksize=chunksize)]

    # yout = np.empty((yarray.shape[0], indices.shape[0]), dtype='float64')
    #
    # for i in range(0, yarray.shape[0]):
    #     yout[i] = _cspline_func(xinfo, yarray[i], s, optimize)[indices]

    return np.array(yout)


def speyediff(N, d, format='csc'):

    """
    (utility function)
    Construct a d-th order sparse difference matrix based on
    an initial N x N identity matrix

    Final matrix (N-d) x N
    """

    assert not (d < 0), "d must be non negative"

    shape = (N-d, N)
    diagonals = np.zeros(2*d + 1)
    diagonals[d] = 1.

    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff

    offsets = np.arange(d+1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)

    return spmat


def _whittaker_func(*args):

    coefmat, y_ = list(itertools.chain(*args))

    return splu(coefmat).solve(y_)


def whittaker_func(yarray, s, order, n_jobs, chunksize):

    """
    Source:
        https://github.com/mhvwerts/whittaker-eilers-smoother/blob/master/whittaker_smooth.py
    """

    m = yarray.shape[1]
    E = sparse.eye(m, format='csc')
    D = speyediff(m, order, format='csc')
    coefmat = E + s * D.conj().T.dot(D)

    data_gen = ((coefmat, yarray[i]) for i in range(0, yarray.shape[0]))

    with multi.Pool(n_jobs) as executor:
        yout = [result for result in executor.imap(_whittaker_func, data_gen, chunksize=chunksize)]

    return np.array(yout)


def _gpr_func(*args):

    X, y_ = list(itertools.chain(*args))

    gp_kernel = ExpSineSquared(length_scale=1.0, periodicity=365.25/2.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)

    gpr = GaussianProcessRegressor(kernel=gp_kernel)
    gpr.fit(X, y_)

    return gpr.predict(X, return_std=False)


def gpr_func(xinfo, indices, yarray, n_jobs, chunksize):

    X = xinfo.xd_smooth[:, np.newaxis]

    # yout = np.empty((yarray.shape[0], indices.shape[0]), dtype='float64')

    data_gen = ((X, yarray[i]) for i in range(0, yarray.shape[0]))

    with multi.Pool(n_jobs) as executor:
        yout = [result[indices] for result in executor.imap(_gpr_func, data_gen, chunksize=chunksize)]

    # for i in range(0, yarray.shape[0]):
    #     yout[i] = _gpr_func(X, yarray[i])[indices]

    return np.array(yout)


def _snake_contour(*args):

    X, y, pad, gain = list(itertools.chain(*args))

    data_idx = np.arange(y.shape[0])
    n = X.shape[0]

    ymax = int(y.max() / gain)
    c = np.zeros((ymax, n), dtype='float64') - 1

    ridx = (ymax-np.uint8(y / gain)).clip(0, ymax-1).astype('int64')
    ridx[0] = ridx[1]
    ridx[-1] = ridx[-2]
    idx = (ridx, data_idx)

    # Fill the array with the time series
    c[idx] = y

    min_mse = 1e9

    for alpha, beta, gamma in itertools.product([0.001, 0.01, 0.1], [0.1, 0.5, 1.0], [0.1]):

        snake = active_contour(c,
                               np.array(idx).T,
                               boundary_condition='fixed',
                               alpha=alpha,
                               beta=beta,
                               w_line=1,
                               w_edge=0,
                               gamma=gamma,
                               max_iterations=100,
                               convergence=1)

        ysnake_ = ((ymax - snake[:, 0]) * gain).clip(0, 1)

        roughness = scale_min_max(np.abs(np.diff(ysnake_, n=4).mean()), 1e-6, 1e-5, 0.01, 1)
        weights = np.where(y > ysnake_, 1, 0.5)
        weights[:pad] = 0
        weights[-pad:] = 0
        se = (ysnake_ - y)**2
        mse = np.average(se, weights=weights) * roughness

        if mse < min_mse:

            min_mse = mse
            ysnake = ysnake_.copy()

    return ysnake[pad:-pad]


def snake_contour(X, yarray, pad=10, n_jobs=1, chunksize=10):

    X = np.pad(X, (pad, pad), mode='linear_ramp')
    yarray = np.pad(yarray, ((0, 0), (pad, pad)), mode='reflect')

    data_gen = ((X, yarray[i], pad, 0.01) for i in range(0, yarray.shape[0]))

    with multi.Pool(n_jobs) as executor:
        yout = [result for result in executor.imap(_snake_contour, data_gen, chunksize=chunksize)]

    return np.array(yout)


class Fourier(object):

    def __init__(self, period=365.25, poly_order=1, harmonic_order=1):

        self.period = period
        self.poly_order = poly_order
        self.harmonic_order = harmonic_order
        self.coef_matrix = None

    def _matrix(self, dates):

        w = 2.0 * np.pi / self.period

        if self.poly_order == 0:
            self.coef_matrix = np.ones(shape=(len(dates), 1), order='F')
        else:

            self.coef_matrix = np.zeros(shape=(len(dates), self.harmonic_order*2+self.poly_order),
                                        order='F')

        w12 = w * dates

        for p_order in range(1, self.poly_order+1):
            self.coef_matrix[:, p_order-1] = dates**p_order

        if self.harmonic_order > 0:

            for h_order in range(1, self.harmonic_order+1):

                self.coef_matrix[:, h_order+self.poly_order-1] = np.cos(w12*h_order)
                self.coef_matrix[:, h_order+1+self.poly_order-1] = np.sin(w12*h_order)

        if self.poly_order > 0:
            self.coef_matrix = np.c_[self.coef_matrix, np.ones(self.coef_matrix.shape[0], dtype='float64')[:, np.newaxis]]

    def fit_predict(self, X, yarray, indices=None):

        self._matrix(X)

        if isinstance(indices, np.ndarray):
            est = np.zeros((yarray.shape[0], indices.shape[0]), dtype='float64')
        else:
            est = np.zeros(yarray.shape, dtype='float64')

        # popt coefficients -> n coeffs x n samples
        popt = np.linalg.lstsq(self.coef_matrix, yarray.T, rcond=None)[0]

        def _pred_func(a):
            return (a*popt).sum(axis=0)

        for t in range(0, self.coef_matrix[indices].shape[0]):
            est[:, t] = _pred_func(self.coef_matrix[indices][t][:, np.newaxis])

        return est


class SmoothersMixin(object):

    @property
    def shape_in(self):
        return self._shape_in

    @shape_in.setter
    def shape_in(self, shape):
        self._shape_in = shape

    @property
    def shape_out(self):
        return self.ndims_out, self.nrows, self.ncols

    @property
    def ndims_in(self):
        return self._ndims_in

    @ndims_in.setter
    def ndims_in(self, ndims):
        self._ndims_in = ndims

    @property
    def ndims_out(self):
        return self.indices.shape[0]

    @property
    def nrows(self):
        return self._nrows

    @nrows.setter
    def nrows(self, nrows):
        self._nrows = nrows

    @property
    def ncols(self):
        return self._ncols

    @ncols.setter
    def ncols(self, ncols):
        self._ncols = ncols

    def _nd_to_columns(self):
        return nd_to_columns(self.data, *self.shape_in)


def pre_remove_outliers(xinfo, yarray, n_jobs, **kwargs):

    ytest = remove_outliers(np.ascontiguousarray(xinfo.xd, dtype='float64'),
                            np.ascontiguousarray(yarray, dtype='float64'),
                            **kwargs)

    return interp2d(ytest, no_data_value=0.0, n_jobs=n_jobs)


def _dbl_pvs(y):

    pvs = np.zeros((2, *y.shape), dtype='float64')

    def gaussian_func(x, sigma):
        return np.exp(-pow(x, 2) / (2.0 * pow(sigma, 2)))

    for k in [15, 21, 28]:

        weights = gaussian_func(np.linspace(-1, 1, k), 0.5)

        ymean = rolling_mean2d(np.pad(y, ((0, 0), (k, k)), mode='reflect'), w=k, weights=weights, n_jobs=1)[:, k:-k]
        pvs += peaks_valleys2d(np.ascontiguousarray(ymean, dtype='float64'), min_value=0.05, min_dist=5, min_sp_dist=0.1, min_prop_sp_dist=0.2, order=k, n_jobs=1)[0]

    pvs[pvs > 1] = 1

    return group_peaks_valleys2d(np.int64(pvs), y, w=21, min_prop_sp_dist=0.001, n_jobs=1)


class AncSmoothers(SmoothersMixin):

    def __init__(self,
                 xinfo,
                 data,
                 pad=50,
                 index_by_indices=False,
                 remove_outliers=True,
                 max_outlier_days1=120,
                 max_outlier_days2=120,
                 min_outlier_values=7,
                 dev_thresh1=0.2,
                 dev_thresh2=0.2,
                 n_jobs=1):

        self.xinfo = xinfo
        self.data = data.copy()
        self.pad = pad
        self.index_by_indices = index_by_indices
        self.remove_outliers = remove_outliers
        self.max_outlier_days1 = max_outlier_days1
        self.max_outlier_days2 = max_outlier_days2
        self.min_outlier_values = min_outlier_values
        self.dev_thresh1 = dev_thresh1
        self.dev_thresh2 = dev_thresh2
        self.n_jobs = n_jobs

        # set_num_threads(n_jobs)

        self.shape_in = self.data.shape
        self.ndims, self.nrows, self.ncols = self.shape_in

        if self.index_by_indices:
            self.indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype='uint64')
        else:
            self.indices = np.ascontiguousarray(np.arange(0, self.ndims), dtype='uint64')

        self._preprocess()

    def _preprocess(self):

        if not self.remove_outliers:
            self.data = interp2d(np.float64(self._nd_to_columns()), no_data_value=0.0, n_jobs=self.n_jobs)
        else:

            self.data = pre_remove_outliers(self.xinfo,
                                            self._nd_to_columns(),
                                            max_outlier_days1=self.max_outlier_days1,
                                            max_outlier_days2=self.max_outlier_days2,
                                            min_outlier_values=self.min_outlier_values,
                                            dev_thresh1=self.dev_thresh1,
                                            dev_thresh2=self.dev_thresh2,
                                            n_jobs=self.n_jobs)

    def csp(self, s=0.1, optimize=False, chunksize=10):

        return columns_to_nd(cspline_func(self.xinfo,
                                          self.indices,
                                          self.data,
                                          s,
                                          optimize,
                                          self.n_jobs,
                                          chunksize), *self.shape_out)

    def bsp(self, window_size=50, mfactor=3, max_iter=1):

        return bezier_curve(self.data.copy(),
                            window_size=window_size,
                            mfactor=mfactor,
                            max_iters=max_iter)

        # interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)
        #
        # return columns_to_nd(interp.interpolate(bezier_curve(self.data.copy(),
        #                                                      window_size=window_size,
        #                                                      mfactor=mfactor,
        #                                                      max_iters=max_iter))[:, self.indices],
        #                      *self.shape_out)

    def dbl(self, 
            lr=None, 
            max_iters=1000, 
            reltol=1e-08, 
            init_params=None, 
            beta1=0.9, 
            beta2=0.99, 
            chunksize=10):

        pv_array = _dbl_pvs(self.data)

        if lr is None:
            lr = np.array([0.01, 0.1, 1.0, 0.5, 1.0, 0.5, 0.001], dtype='float64')

        if init_params is None:
            init_params = np.ascontiguousarray([0.03, 0.6, 75, 20.0, 300, 20.0, 0.0001], dtype='float64')

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(interp.interpolate(gd(ordinals=np.ascontiguousarray(self.xinfo.xd, dtype='int64'),
                                                   y=self.data,
                                                   pv_array=np.ascontiguousarray(pv_array, dtype='int64'),
                                                   lr=lr,
                                                   max_iters=max_iters,
                                                   reltol=reltol,
                                                   init_params=init_params,
                                                   constraints=np.array([[0.0, 0.2],
                                                                         [0.2, 2.0],
                                                                         [0.0, 185.0],
                                                                         [5.0, 40.0],
                                                                         [185.0, 367.0],
                                                                         [5.0, 40.0],
                                                                         [1e-8, 0.01]]),
                                                   beta1=beta1,
                                                   beta2=beta2,
                                                   n_jobs=self.n_jobs,
                                                   chunksize=chunksize),
                                                fill_no_data=True,
                                                no_data_value=0,
                                                n_jobs=self.n_jobs)[:, self.indices], *self.shape_out)

    def harm(self, period=365.25, poly_order=1, harmonic_order=1):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        clf = Fourier(period=period,
                      poly_order=poly_order,
                      harmonic_order=harmonic_order)

        ypred = clf.fit_predict(self.xinfo.xd_smooth[self.xinfo.start_idx:self.xinfo.end_idx],
                                interp.interpolate(self.data,
                                                   fill_no_data=True,
                                                   no_data_value=0,
                                                   n_jobs=self.n_jobs)[:, self.xinfo.start_idx:self.xinfo.end_idx],
                                indices=self.xinfo.skip_idx)

        return columns_to_nd(ypred, ypred.shape[1], self.nrows, self.ncols)

    def sg(self, w=7, p=3):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(interp.interpolate(savgol_filter(self.data, w, p),
                                                fill_no_data=True,
                                                no_data_value=0,
                                                n_jobs=self.n_jobs)[:, self.indices],
                             *self.shape_out)

    def wh(self, s=1.0, order=2, chunksize=10):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(interp.interpolate(whittaker_func(self.data, s, order, self.n_jobs, chunksize),
                                                fill_no_data=True,
                                                no_data_value=0,
                                                n_jobs=self.n_jobs)[:, self.indices],
                             *self.shape_out)

    def lw(self, w=31, chunksize=10):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(interp.interpolate(lowess_smooth(ordinals=np.ascontiguousarray(self.xinfo.xd, dtype='int64'),
                                                              y=self.data,
                                                              w=w,
                                                              n_jobs=self.n_jobs,
                                                              chunksize=chunksize),
                                                fill_no_data=True,
                                                no_data_value=0,
                                                n_jobs=self.n_jobs)[:, self.indices],
                             *self.shape_out)

    def gpr(self, chunksize=10):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(gpr_func(self.xinfo,
                                      self.indices,
                                      interp.interpolate(self.data,
                                                         fill_no_data=True,
                                                         no_data_value=0,
                                                         n_jobs=self.n_jobs),
                                      self.n_jobs,
                                      chunksize), *self.shape_out)

    def ac(self, pad=10, chunksize=10):

        interp = LinterpMulti(self.xinfo.xd, self.xinfo.xd_smooth)

        return columns_to_nd(rolling_mean2d(np.ascontiguousarray(interp.interpolate(snake_contour(self.xinfo.xd,
                                                                                                  self.data,
                                                                                                  pad=pad,
                                                                                                  n_jobs=self.n_jobs,
                                                                                                  chunksize=chunksize),
                                                                                    fill_no_data=True,
                                                                                    no_data_value=0,
                                                                                    n_jobs=self.n_jobs)[:, self.indices], dtype='float64'),
                                            w=21,
                                            no_data_value=0,
                                            weights=np.ones(21, dtype='float64'),
                                            n_jobs=self.n_jobs),
                             *self.shape_out)
