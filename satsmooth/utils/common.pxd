# cython: language_level=3
# cython: profile=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import cython
cimport cython


cdef extern from 'stdlib.h' nogil:
    double exp(double val)


cdef extern from 'stdlib.h' nogil:
    double fabs(double val)


cdef extern from 'stdlib.h' nogil:
    int abs(int val)


cdef extern from 'math.h' nogil:
    double round(double val)


cdef extern from 'math.h' nogil:
    double floor(double val)


cdef extern from 'math.h' nogil:
   double sqrt(double val)


cdef extern from 'numpy/npy_math.h' nogil:
    bint npy_isnan(double x)


cdef extern from 'numpy/npy_math.h' nogil:
    bint npy_isinf(double x)


cdef inline int min_value_int(int v1, int v2) nogil:
    return v1 if v1 < v2 else v2


cdef inline double min_value(double v1, double v2) nogil:
    return v1 if v1 < v2 else v2


cdef inline double max_value(double v1, double v2) nogil:
    return v1 if v1 > v2 else v2


cdef inline int abs_round(double v1, double v2) nogil:
    return <int>round(fabs((v1 - v2) * 100.0))


cdef inline unsigned int clip_low_int(unsigned int v, unsigned int l) nogil:
    return l if v < l else v


cdef inline unsigned int clip_high_int(unsigned int v, unsigned int h) nogil:
    return h if v > h else v


cdef inline double clip_low(double v, double l) nogil:
    return l if v < l else v


cdef inline double clip_high(double v, double h) nogil:
    return h if v > h else v


cdef inline double clip(double v, double l, double h) nogil:
    return l if v < l else (h if v > h else v)


cdef inline double bright_weight(double center_dn, double edge_dn) nogil:
    return 1.0 if center_dn >= edge_dn else 0.75


cdef inline double gradient(double a, double b) nogil:
    return (b - a) / 2.0


cdef inline double prop_diff(double a, double b) nogil:
    return (b - a+1e-8) / (a+1e-8)


cdef inline double perc_diff(double a, double b) nogil:
    return prop_diff(a, b) * 100.0


cdef inline double pow2(double val) nogil:
    return val**2.0


cdef inline double pow3(double val) nogil:
    return val**3.0


cdef inline double gaussian_func(double x, double sigma) nogil:
    return exp(-pow2(x) / (2.0 * pow2(sigma)))


cdef inline double linear_adjustment(int x1, int x2, int x3, double y1, double y3) nogil:
    return ((<double>(x2 - x1) * (y3 - y1)) / <double>(x3 - x1)) + y1


cdef inline double set_weight(double w1, double w2, double w3) nogil:
    return w1 + w2 + w3


cdef inline double scale_min_max(double xv, double mno, double mxo, double mni, double mxi) nogil:
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno


cdef inline double logistic_func(double x, double x0, double r) nogil:
    return 1.0 / (1.0 + exp(-r * (x-x0)))


cdef inline double squared_diff(double a, double b) nogil:
    return pow2(a - b)


cdef inline double edist(double xloc, double yloc, double hw) nogil:
    return sqrt(pow2(xloc - hw) + pow2(yloc - hw))


cdef inline unsigned int window_diff(unsigned int t, unsigned int t_adjust) nogil:
    return <int>((t - t_adjust) * 0.5)


cdef inline unsigned int adjust_window_up(unsigned int tval) nogil:
    return tval + 1 if tval % 2 == 0 else tval


cdef inline unsigned int adjust_window_down(unsigned int tval) nogil:
    return tval - 1 if tval % 2 == 0 else tval
