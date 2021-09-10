# distutils: language = c++
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

from .bspline cimport bcurve
from ..utils cimport common

from libc.stdlib cimport malloc, free


cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        void push_back(T&) nogil
        size_t size() nogil
        T& operator[](size_t) nogil
        void clear() nogil


cdef extern from 'stdlib.h' nogil:
    double exp(double val)


cdef extern from 'stdlib.h' nogil:
    double fabs(double val)


cdef extern from 'stdlib.h' nogil:
    int abs(int val)


cdef extern from 'numpy/npy_math.h' nogil:
    bint npy_isnan(double x)


cdef inline double _pow2(double x) nogil:
    return x*x


cdef inline double _sqrt(double x) nogil:
    return x**0.5


cdef double _dbl_logistic(double t,
                          double *params) nogil:

    """
    Double logistic function
    """

    cdef:
        double m1, m2, m3, m4, m5, m6, m7

    m1 = params[0]
    m2 = params[1]
    m3 = params[2]
    m4 = params[3]
    m5 = params[4]
    m6 = params[5]
    m7 = params[6]

    return m1 + (m2 - m7 * t) * ((1.0 / (1.0 + exp((m3 - t) / m4))) - (1.0 / (1.0 + exp((m5 - t) / m6))))


cdef double loss_mse(double[:, ::1] h,
                     double[:, ::1] y,
                     Py_ssize_t ii,
                     Py_ssize_t start,
                     Py_ssize_t end,
                     unsigned int ntime) nogil:

    """
    Mean squared error loss
    """

    cdef:
        Py_ssize_t q
        double cost = 0.0
        Py_ssize_t count = 0
        double dev, w

    for q in range(start, end):

        if not npy_isnan(y[ii, q]) and (q < ntime):

            dev = h[ii, q] - y[ii, q]

            # Emphasize loss toward estimates above original values
            w = 10.0 if dev > 0 else 0.1

            cost += (w * dev**2)
            count += 1

    if count > 0:
        return cost / <double>count
    else:
        return 0.0


cdef inline void _adam(double *params__,
                       double[::1] lr,
                       double *m_vec__,
                       double *v_vec__,
                       double gm,
                       Py_ssize_t m_idx,
                       Py_ssize_t e,
                       double beta1,
                       double beta2) nogil:

    cdef:
        double mhat, vhat
        double eps = 1e-8

    m_vec__[m_idx] = beta1 * m_vec__[m_idx] + (1.0 - beta1) * gm
    v_vec__[m_idx] = beta2 * v_vec__[m_idx] + (1.0 - beta2) * _pow2(gm)

    mhat = m_vec__[m_idx] / (1.0 - beta1**(e+1))
    vhat = v_vec__[m_idx] / (1.0 - beta2**(e+1))

    # Update coefficients
    params__[m_idx] -= (lr[m_idx] * mhat / (_sqrt(vhat) + eps))


cdef void _gradient(long[::1] t,
                    double[:, ::1] y,
                    double[::1] lr,
                    unsigned int ntime,
                    double *params_,
                    Py_ssize_t iii,
                    int sos,
                    int eos,
                    unsigned int nparams,
                    double[:, ::1] constraints,
                    unsigned int e,
                    double *m_vec_,
                    double *v_vec_,
                    double beta1,
                    double beta2) nogil:

    cdef:
        Py_ssize_t j, v
        double yhat, yv, ydev
        double pm1, pm2, pm3, pm4, pm5, pm6, pm7
        double gm1, gm2, gm3, gm4, gm5, gm6, gm7
        double m2_deriv, m3_deriv, m4_deriv, m5_deriv, m6_deriv, m7_deriv
        double m1, m2, m3, m4, m5, m6, m7
        unsigned int nsamps = eos - sos
        int ts

    m1 = params_[0]
    m2 = params_[1]
    m3 = params_[2]
    m4 = params_[3]
    m5 = params_[4]
    m6 = params_[5]
    m7 = params_[6]

    # Partial sums
    pm1, pm2, pm3, pm4, pm5, pm6, pm7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for j in range(sos, eos):

        if j < ntime:

            # Start-of-season adjustment
            ts = t[j] - t[sos]

            # Original value
            yv = y[iii, j]

            # Make an estimate
            yhat = _dbl_logistic(ts, params_)

            # (-1.0/(math.exp((m5 - t)/m6) + 1.0) + 1.0/(math.exp((m3 - t)/m4) + 1.0))
            m2_deriv = -1.0 / (exp((m5 - ts) / m6) + 1.0) + 1.0 / (exp((m3 - ts) / m4) + 1.0)

            # (-1.0*(m2 - m7*t)*math.exp((m3 - t)/m4)/(m4*(math.exp((m3 - t)/m4) + 1.0)**2))
            m3_deriv = -1.0 * (m2 - m7*ts) * exp((m3 - ts) / m4) / (m4 * (exp((m3 - ts) / m4) + 1.0)**2)

            # (1.0*(m2 - m7*t)*(m3 - t)*math.exp((m3 - t)/m4)/(m4**2*(math.exp((m3 - t)/m4) + 1.0)**2))
            m4_deriv = 1.0 * (m2 - m7*ts) * (m3 - ts) * exp((m3 - ts) / m4) / (m4**2 * (exp((m3 - ts) / m4) + 1.0)**2)

            # (1.0*(m2 - m7*t)*math.exp((m5 - t)/m6)/(m6*(math.exp((m5 - t)/m6) + 1.0)**2))
            m5_deriv = 1.0 * (m2 - m7*ts) * exp((m5 - ts) / m6) / (m6 * (exp((m5 - ts) / m6) + 1.0)**2)

            # (-1.0*(m2 - m7*t)*(m5 - t)*math.exp((m5 - t)/m6)/(m6**2*(math.exp((m5 - t)/m6) + 1.0)**2))
            m6_deriv = -1.0 * (m2 - m7*ts) * (m5 - ts) * exp((m5 - ts) / m6) / (m6**2 * (exp((m5 - ts) / m6) + 1.0)**2)

            # (-t*(-1.0/(math.exp((m5 - t)/m6) + 1.0) + 1.0/(math.exp((m3 - t)/m4) + 1.0)))
            m7_deriv = -ts * (-1.0 / (exp((m5 - ts) / m6) + 1.0) + 1.0 / (exp((m3 - ts) / m4) + 1.0))

            # Deviation
            ydev = yv - yhat

            # Partial derivatives
            pm1 += ydev
            pm2 += (m2_deriv * ydev)
            pm3 += (m3_deriv * ydev)
            pm4 += (m4_deriv * ydev)
            pm5 += (m5_deriv * ydev)
            pm6 += (m6_deriv * ydev)
            pm7 += (m7_deriv * ydev)

    # Gradients
    gm1 = (-2.0 * pm1 / nsamps)
    gm2 = (-2.0 * pm2 / nsamps)
    gm3 = (-2.0 * pm3 / nsamps)
    gm4 = (-2.0 * pm4 / nsamps)
    gm5 = (-2.0 * pm5 / nsamps)
    gm6 = (-2.0 * pm6 / nsamps)
    gm7 = (-2.0 * pm7 / nsamps)

    _adam(params_, lr, m_vec_, v_vec_, gm1, 0, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm2, 1, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm3, 2, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm4, 3, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm5, 4, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm6, 5, e, beta1, beta2)
    _adam(params_, lr, m_vec_, v_vec_, gm7, 6, e, beta1, beta2)

    # Update coefficients
#     params_[0] -= (lr[0] * gm1)
#     params_[1] -= (lr[1] * gm2)
#     params_[2] -= (lr[2] * gm3)
#     params_[3] -= (lr[3] * gm4)
#     params_[4] -= (lr[4] * gm5)
#     params_[5] -= (lr[5] * gm6)
#     params_[6] -= (lr[6] * gm7)

    # Constraints
    for v in range(0, nparams):

        if params_[v] < constraints[v, 0]:
            params_[v] = constraints[v, 0]

        if params_[v] > constraints[v, 1]:
            params_[v] = constraints[v, 1]


cdef struct Gaussian:

    double mean
    double var


cdef Gaussian _set_gaussian(double mean, double var) nogil:

    cdef:
        Gaussian g

    g.mean = mean
    g.var = var

    return g


cdef Gaussian _gaussian_multiply(Gaussian g1, Gaussian g2) nogil:

    cdef:
        double mean, var

    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    var = (g1.var * g2.var) / (g1.var + g2.var)

    return _set_gaussian(mean, var)


cdef Gaussian _update(Gaussian prior, Gaussian likelihood) nogil:

    cdef:
        Gaussian posterior

    posterior = _gaussian_multiply(likelihood, prior)

    return posterior


cdef void _update_params(double *priors_, double *likelihoods_) nogil:

    cdef:
        Gaussian m1_pr, m2_pr, m4_pr, m6_pr, m7_pr
        Gaussian m1_l, m2_l, m4_l, m6_l, m7_l
        Gaussian m1_p, m2_p, m4_p, m6_p, m7_p
        double m1, m2, m4, m6, m7

    m1_pr = _set_gaussian(priors_[0], 0.01)
    m1_l = _set_gaussian(likelihoods_[0], 0.02)
    m1_p = _update(m1_pr, m1_l)
    m1 = m1_p.mean

    m2_pr = _set_gaussian(priors_[1], 0.2)
    m2_l = _set_gaussian(likelihoods_[1], 0.1)
    m2_p = _update(m2_pr, m2_l)
    m2 = m2_p.mean

    #m3_pr = _set_gaussian(priors_[2], 10.0)
    #m3_l = _set_gaussian(likelihoods_[2], 10.0)
    #m3_p = _update(m3_pr, m3_l)
    #m3 = m3_p.mean

    m4_pr = _set_gaussian(priors_[3], 1.0)
    m4_l = _set_gaussian(likelihoods_[3], 2.0)
    m4_p = _update(m4_pr, m4_l)
    m4 = m4_p.mean

    #m5_pr = _set_gaussian(priors_[4], 10.0)
    #m5_l = _set_gaussian(likelihoods_[4], 10.0)
    #m5_p = _update(m5_pr, m5_l)
    #m5 = m5_p.mean

    m6_pr = _set_gaussian(priors_[5], 1.0)
    m6_l = _set_gaussian(likelihoods_[5], 2.0)
    m6_p = _update(m6_pr, m6_l)
    m6 = m6_p.mean

    m7_pr = _set_gaussian(priors_[6], 0.001)
    m7_l = _set_gaussian(likelihoods_[6], 0.002)
    m7_p = _update(m7_pr, m7_l)
    m7 = m7_p.mean

    priors_[0] = m1
    priors_[1] = m2
    priors_[2] = likelihoods_[2]
    priors_[3] = m4
    priors_[4] = likelihoods_[4]
    priors_[5] = m6
    priors_[6] = m7


cdef void _curve_fit(long[::1] ordinals,
                     double[:, ::1] y,
                     long[:, :, ::1] pv_array,
                     double[:, ::1] yout_,
                     Py_ssize_t ii,
                     unsigned int ntime,
                     double[::1] init_params,
                     unsigned int nparams,
                     unsigned int max_iters,
                     double[::1] lr,
                     double[:, ::1] constraints,
                     double reltol,
                     double beta1,
                     double beta2,
                     double[::1] bspline_t_values) nogil:

    cdef:
        Py_ssize_t pvidx, p, z, j
        int sos, pos, eos
        double season_max
        double *likelihoods
        double *priors
        double *m_vec
        double *v_vec
        double loss, last_loss
        vector[int] peak_vct
        vector[int] valley_vct
        unsigned int n_peaks

        unsigned int pw_i, pw_n

    # Get the peaks and valleys
    for pvidx in range(0, ntime):

        # Valleys
        if pv_array[1, ii, pvidx] == 1:
            valley_vct.push_back(pvidx)

        # Peaks
        if pv_array[0, ii, pvidx] == 1:
            peak_vct.push_back(pvidx)

    n_peaks = peak_vct.size()

    # Create holders for the parameters
    likelihoods = <double *>malloc(nparams * sizeof(double))
    priors = <double *>malloc(nparams * sizeof(double))
    m_vec = <double *>malloc(nparams * sizeof(double))
    v_vec = <double *>malloc(nparams * sizeof(double))

    # Fill the priors
    for p in range(0, nparams):
        priors[p] = init_params[p]

    # Iterate over each peak
    #for year_id from 0 <= year_id < n_peaks-2 by 2:
    for pvidx in range(0, n_peaks):

        # Start and end of season
        sos = valley_vct[pvidx]
        pos = peak_vct[pvidx]
        eos = valley_vct[pvidx+1]
        season_max = y[ii, pos] + 0.2

        # Update parameters

        # [0.03, 0.55, 75, 30.0, 300, 20.0, 0.0001]
        likelihoods[0] = init_params[0]
        likelihoods[1] = season_max
        likelihoods[2] = ordinals[sos + <int>((pos-sos)*0.2)] - ordinals[sos]
        likelihoods[3] = init_params[3]
        likelihoods[4] = ordinals[eos - <int>((eos-pos)*0.2)] - ordinals[sos]
        likelihoods[5] = init_params[5]
        likelihoods[6] = init_params[6]

        # Update the parameters
        _update_params(priors, likelihoods)

        last_loss = -1e9

        for z in range(0, max_iters):

            # Calculate the gradients and update the parameters
            _gradient(ordinals, y, lr, ntime, priors, ii, sos, eos, nparams, constraints, z, m_vec, v_vec, beta1, beta2)

            # Update the estimates for the season
            for j in range(sos, eos):
                if j < ntime:
                    yout_[ii, j] = _dbl_logistic(ordinals[j]-ordinals[sos], priors)

            loss = loss_mse(yout_, y, ii, sos, eos, ntime)

            if last_loss > -1e9:

                if fabs(loss - last_loss) < reltol:
                    break

            last_loss = loss

    peak_vct.clear()
    valley_vct.clear()

    free(likelihoods)
    free(priors)
    free(m_vec)
    free(v_vec)

#    for year_id from 0 <= year_id < pvs_size-2 by 2:
#
#        eos = abs(pvs[year_id+2])
#
#        pw_i = eos - 15
#        pw_n = eos + 16
#
#        if pw_i < 0:
#            break
#
#        if pw_n >= ntime:
#            break
#
#        bcurve(bspline_t_values,
#               yout_,
#               ii,
#               pw_i,
#               pw_n,
#               k=7,
#               s=1)


def gd(long[::1] ordinals=None,
       double[:, ::1] y=None,
       long[:, :, ::1] pv_array=None,
       double[::1] lr=None,
       unsigned int max_iters=100,
       double reltol=1e-06,
       double[::1] init_params=None,
       double[:, ::1] constraints=None,
       double beta1=0.9,
       double beta2=0.99,
       unsigned int n_jobs=1,
       unsigned int chunksize=10):

    """
    Curve fitting with gradient descent using Adam optimization

    Args:
        ordinals (1d array): The date ordinals.
        y (2d array): The response variables.
        pv_array (3d array): Peaks/valleys x samples x time.
        lr (Optional[1d array]): The learning rates.
        max_iters (Optional[float]): The maximum number of iterations.
        reltol (Optional[float]): The relative tolerance to stop optimization.
        init_params (Optional[1d array]): The initial model parameters.
        constraints (Optional[2d array]): The parameter constraints.
        beta1 (Optional[float]): Adam beta 1.
        beta2 (Optional[float]): Adam beta 2.

    Returns:
        2d array
    """

    cdef:
        Py_ssize_t i
        unsigned int nsamps = y.shape[0]
        unsigned int ntime = ordinals.shape[0]
        unsigned int nparams = 7

        double[:, ::1] yout = np.zeros((nsamps, ntime), dtype='float64')
        double[::1] bspline_t_values = np.arange(0, 1, 0.01)

    with nogil, parallel(num_threads=n_jobs):

        for i in prange(0, nsamps, schedule='static', chunksize=chunksize):

            _curve_fit(ordinals,
                       y,
                       pv_array,
                       yout,
                       i,
                       ntime,
                       init_params,
                       nparams,
                       max_iters,
                       lr,
                       constraints,
                       reltol,
                       beta1,
                       beta2,
                       bspline_t_values)

    return np.float64(yout)
