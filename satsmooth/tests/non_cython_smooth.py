from collections import namedtuple

import numpy as np


def dummy():

    fn = 'chile_forest_1999-05-08.dill'
    START_YEAR = '1999'
    END_YEAR = '2004'  #str(int(START_YEAR) + 1)

    with open(dfn, 'rb') as dill_file:
        groups, args, X, y = dill.load(dill_file)

    d1 = X[0]
    diffs = [(X[i+1] - X[i]).days for i in range(0, len(X)-1)]

    xinfo = prepare_x(X, args.start, args.end, args.skip)
    interp = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)
    ndims, nrows, ncols = y.shape

    # smooth
    indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype='uint64')
    ys = interp.interpolate_smooth(nd_to_columns(y.astype('float64'), ndims, nrows, ncols).copy(), indices=indices, **smoothers.SMOOTH_KWARGS)
    jys = columns_to_nd(ys, ys.shape[1], nrows, ncols) * 10000.0

    # diagram
    xtest = np.int64(xinfo.xd.copy())

    # Raw series
    yseries = y[:, 1, 1].copy().flatten()[np.newaxis, :].astype('float64')

    xoff = 0
    xlen = 1500

    # interp = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)

    # Remove outliers and fill gaps
    ytest = interp.interpolate(yseries.copy(),
                           interp_step=smoothers.SMOOTH_KWARGS['interp_step'],
                           fill_no_data=smoothers.SMOOTH_KWARGS['fill_no_data'],
                           no_data_value=smoothers.SMOOTH_KWARGS['no_data_value'],
                           remove_outliers=smoothers.SMOOTH_KWARGS['remove_outliers'],
                           max_outlier_days1=smoothers.SMOOTH_KWARGS['max_outlier_days1'],
                           max_outlier_days2=smoothers.SMOOTH_KWARGS['max_outlier_days2'],
                           min_outlier_values=smoothers.SMOOTH_KWARGS['min_outlier_values'],
                           outlier_iters=smoothers.SMOOTH_KWARGS['outlier_iters'],
                           dev_thresh1=smoothers.SMOOTH_KWARGS['dev_thresh1'],
                           dev_thresh2=smoothers.SMOOTH_KWARGS['dev_thresh2'],
                           n_jobs=1)

    ytest = ytest[0, xoff:xoff+xlen].flatten()
    ytest_orig = ytest.copy()

    sm_iters_plot, xplot, bc_plot, bcw_plot, t_adjust_plot, w_plot, twx_plot, tw_plot, t_half = utils.smooth_data(xinfo, xoff, ytest_orig, ytest,
                                                                                                              t=smoothers.SMOOTH_KWARGS['t'],
                                                                                                              min_window=smoothers.SMOOTH_KWARGS['min_window'],
                                                                                                              r_g=smoothers.SMOOTH_KWARGS['r_g'],
                                                                                                              r_k=smoothers.SMOOTH_KWARGS['r_k'],
                                                                                                              smooth_iters=5)


def bezier_point(xp, yp, t):

    """
    Args:
        xp (1d array): Control points.
        yp (1d array): Control points.
        t: (float): Smoothing
    """
    
    xb = (1 - t)**2 * xp[0] + 2 * t * (1 - t) * xp[1] + t**2 * xp[2]
    yb = (1 - t)**2 * yp[0] + 2 * t * (1 - t) * yp[1] + t**2 * yp[2]
    
    return xb, yb


def bezier_curves(xp, yp):

    curves_t = np.zeros((int(xp[-1])-int(xp[0]), 2), dtype='float64')
    
    k = 0
    for t in np.linspace(0, 1, curves_t.shape[0]):
        
        xo, yo = bezier_point(xp, yp, t)
        
        curves_t[k, 0] = xo
        curves_t[k, 1] = yo
        
        k += 1

    return curves_t


def bspline(x, y, i):
    
    """
    Args:
        x (1d array): Control points.
        y (2d array): Control points.
        i (int): The row position.        
    """
    
    nsamps = x.shape[0]
    
    curves = []

    for j in range(0, nsamps-3+1):
        curves.append(bezier_curves(x[j:j+3], y[i, j:j+3]))

    return curves


def multicolor_ylabel(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kw):
    """
    Reference:
        https://stackoverflow.com/questions/33159134/matplotlib-y-axis-label-with-multiple-colors
        
    this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn
    """
    
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='center', va='bottom', rotation=90, **kw)) 
                 for text,color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=1)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.06, 0.32), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
        
    return ax


def scale_min_max(xv, mno, mxo, mni, mxi):
    return (((mxo - mno) * (xv - mni)) / (mxi - mni)) + mno

def clip_high(v, h):
    return h if v > h else v

def perc_diff(a, b):
    return ((b - a) / a) * 100.0

def logistic_func(xv, x0, r):
    return 1.0 / (1.0 + np.exp(-r * (xv-x0)))

def adjust_window_up(tval):
    return tval + 1 if tval % 2 == 0 else tval

def adjust_window_down(tval):
    return tval - 1 if tval % 2 == 0 else tval

def window_diff(t, t_adjust):
    return int((t - t_adjust) * 0.5)

def clip_low_int(v, l):
    return l if v < l else v

def clip(v, l, h):
    return l if v < l else (h if v > h else v)

def abs_round(v1, v2):
    return int(round(abs((v1 - v2) * 100.0)))

def gaussian_func(x, sigma):
    return np.exp(-x**2 / (2.0 * sigma**2))

def bright_weight(center_dn, edge_dn):
    return 1.0 if center_dn >= edge_dn else 0.75

def pow2(val):
    return val*val

def squared_diff(a, b):
    return pow2(a - b)

def linear_adjustment(x1, x2, x3, y1, y3):
    return ((float(x2 - x1) * (y3 - y1)) / float(x3 - x1)) + y1
    
def prop_diff(a, b):
    return (b - a) / a

    
def replace_upper_envelope(y_array_sm,
                           out_array_sm,
                           col_count,
                           smj,
                           t,
                           t_half,
                           t_diff,
                           t_adjust):

    """
    Replaces values with the upper envelope
    """

    #for smj in range(0, col_count-t-1):

    # Original value
    yval = y_array_sm[smj+t_half]

    # Smoothed value
    smval = out_array_sm[smj+t_half]

    # Original window ends
    smvalb = out_array_sm[smj]
    smvalf = out_array_sm[smj+t-1]

    # Adjusted window ends
    smvaly = out_array_sm[smj+t_diff]
    smvalz = out_array_sm[smj+t_diff+t_adjust-1]    
    
    if (smvalb > smval < smvalf) or (smvaly > smval < smvalz):
        w1 = 0.25
        w2 = 1.25
    elif (smvalb < smval > smvalf) or (smvaly < smval > smvalz):
        w1 = 0.25
        w2 = 1.25
    else:
        
        w1 = 0.5
        w2 = 1.0
        
#         if smval < yval:
#             w1 = 0.5
#             w2 = 1.0
#         else:
#             w1 = 1.0
#             w2 = 0.75
    
    w = w1 + w2
    
    y_array_sm[smj+t_half] = (smval*w1 + yval*w2) / w

    return y_array_sm


def _get_array_std(y_array_slice, jjj, t):

    """
    Calculates the standard deviation of a 1-d array

    Args:
        y_array_slice (2d array): The array.
        jjj (int): The starting column index position.
        t (int): The ending column index position.

    Returns:
        Standard deviation (double)
    """

    array_sum = 0.0
    array_count = 0
    sum_sq = 0.0

    for jca in range(jjj, t):

        yvalue_ = y_array_slice[jca]

        if yvalue_ > 0:

            array_sum += yvalue_
            array_count += 1

    array_mean = array_sum / float(array_count)

    for jcb in range(jjj, t):

        yvalue_ = y_array_slice[jcb]

        if yvalue_ > 0:

            sq_dev = squared_diff(yvalue_, array_mean)
            sum_sq += sq_dev

    return np.sqrt(sum_sq / float(array_count))


def smooth_data(xinfo,
                xoff,
                ytest_orig,
                ytest,
                t=81, 
                min_window=21,
                mid_g=0.5,
                r_g=-5.0,
                mid_k=0.5,
                r_k=-5.0,
                mid_t=0.5,
                r_t=15.0,
                sigma_color=0.1,
                smooth_iters=5):

    xplot = []
    bc_plot = []
    bcw_plot = []
    t_adjust_plot = []
    w_plot = []
    twx_plot = []
    tw_plot = []
    sm_iters_plot = []

    out_array = ytest_orig.copy()
    ytest_copy = ytest_orig.copy()

    t_half = int(t / 2)

    # ytest[(ytest == 0) | (ytest == -999)] = np.nan
    # sample_min = np.nanpercentile(ytest, 5)
    # ytest[np.isnan(ytest)] = 0
    sample_min = 0.1

    color_weights = np.empty(101, dtype='float64')

    for ct in range(0, 101):
        color_weights[ct] = gaussian_func(float(ct) * 0.01, sigma_color)    
    
    for iter_ in range(0, smooth_iters):

        for j in range(0, ytest.shape[0]-t+1):

            bc = ytest[j+t_half]

            #sample_std = scale_min_max(clip_high(ytest[j:j+t].std(), 0.05), 0.0, 1.0, 0.0, 0.05)
            sample_std = scale_min_max(clip_high(_get_array_std(ytest, j, j+t), 0.05), 0.0, 1.0, 0.0, 0.05)
            bcw = clip_high(abs(perc_diff(sample_min, bc)) * 0.001, 1.0)
            bcw_mu = (sample_std + bcw) * 0.5

            t_adjust = int(scale_min_max(logistic_func(bcw_mu, mid_k, r_k), min_window, t, 0.0, 1.0))
            t_adjust = adjust_window_up(t_adjust)

            base_adjust = logistic_func(bcw_mu, mid_g, r_g)

            if t == t_adjust:
                t_diff = 0
                t_diff_half = t_half
            else:

                # Half of the difference in the user window and adjusted window
                t_diff = window_diff(t, t_adjust)

                # Enforce odd-sized window
                t_diff = clip_low_int(adjust_window_down(t_diff), 0)

                t_diff_half = int(float(t_adjust) * 0.5) + t_diff   

            max_win_diff = float(abs(t_diff_half - t_diff))

            w_plot_w = []
            twx_plot_w = []
            tw_plot_w = []

            weighted_sum = 0.0
            weights_sum = 0.0

            for zz in range(t_diff, t_diff+t_adjust):

                vc = np.clip(ytest[j+zz], 0.0, 1.0)

                if vc > 0:

                    vc_scaled = np.clip(-1.0 * scale_min_max(float(t_diff_half) - float(zz), -1.0, 1.0, -1.0 * max_win_diff, max_win_diff), -1.0, 1.0)

                    if abs(perc_diff(ytest[j+t_diff], ytest[j+t_diff+t_adjust-1])) >= 20:

                        # Increasing segment
                        if ytest[j+t_diff+t_adjust-1] > ytest[j+t_diff]:
                            vc_scaled = scale_min_max(-1.0*vc_scaled, 0.1, 1.0, -1.0, 1.0)
                            r_tadj = -r_t
                        else:
                            vc_scaled = scale_min_max(vc_scaled, 0.1, 1.0, -1.0, 1.0) 
                            r_tadj = r_t

                        tw = scale_min_max(logistic_func(vc_scaled, mid_t, r_tadj), 0.1, 1.0, 0.0, 1.0)

                    else:
                        tw = scale_min_max(gaussian_func(abs(vc_scaled), base_adjust), 0.1, 1.0, 0.0, 1.0) 

                    gb = bright_weight(bc, vc)
                    gi = color_weights[abs_round(vc, bc)]
                    w = tw + gb + gi

                    weighted_sum += vc*w
                    weights_sum += w

                if iter_ == 0:

                    if j % 100 == 0:

                        if zz == t_diff:
                            #print(perc_diff(ytest[j+t_diff], ytest[j+t_diff+t_adjust-1]))
                            w_plot_w.append((xinfo.dates_smooth[xoff+j+zz], xinfo.dates_smooth[xoff+j+zz+t_adjust-1]))

                        twx_plot_w.append(xinfo.dates_smooth[xoff+j+zz])
                        tw_plot_w.append(tw)

            if weighted_sum > 0:
                adjusted_value = weighted_sum / weights_sum
                out_array[j+t_half] = adjusted_value                    

            if iter_ == 0:

                xplot.append(xinfo.dates_smooth[xoff+j+t_half])
                bc_plot.append(bc)
                bcw_plot.append(bcw)
                t_adjust_plot.append(t_adjust)
                if w_plot_w:
                    w_plot.append(w_plot_w)
                twx_plot.append(twx_plot_w)
                tw_plot.append(tw_plot_w)

            ytest = replace_upper_envelope(ytest, out_array, ytest.shape[0], j, t, t_half, t_diff, t_adjust)

        sm_iters_plot.append(out_array.copy())
        
    return sm_iters_plot, xplot, bc_plot, bcw_plot, t_adjust_plot, w_plot, twx_plot, tw_plot, t_half


def _check_deviation(indexesv,
                     xdatav,
                     ydatav,
                     yarray_,
                     ii,
                     start_,
                     end_,
                     mbf,
                     dev_thresh,
                     no_data,
                     baseline,
                     low_values,
                     valid_samples,
                     predict_linear,
                     max_days,
                     outlier_info):

    """
    Checks each sample's deviation from the window linear fit 
    """
    
    OutlierData = namedtuple('OutlierData', 'col_pos xdata ydata alpha beta1 beta2')

    max_dev = -1e9
    outlier_count = 0

    dhalf = int(start_ + ((end_ - start_) / 2.0))
    dmax = abs(float(dhalf - start_))

    if predict_linear:

        beta1 = mbf[ii, 0]
        beta2 = 0.0
        alpha = mbf[ii, 1]
        
        xvalue = xdatav[1]
        yvalue = ydatav[1]

        if (low_values < yvalue <= baseline) and (low_values < ydatav[0] <= baseline) and (low_values < ydatav[2] <= baseline):
            outlier_count = 0
        else:

            yhat = linear_adjustment(int(xdatav[0]), int(xdatav[1]), int(xdatav[2]), ydatav[0], ydatav[2])

            if (low_values < yvalue <= 0.4) and (max(ydatav[0], ydatav[2]) >= 0.45) and (ydatav[0] > yvalue < ydatav[2]):
                # Protect crop cycles
                w1 = 1.0 - (max(ydatav[0], ydatav[2]))**0.5
            elif (yvalue >= 0.45) and (min(ydatav[0], ydatav[2]) <= baseline):
                # Protect high outliers relative to baseline
                w1 = 0.1
            elif ((low_values < yvalue <= baseline) and (low_values <= ydatav[0] <= baseline)) or ((low_values <= yvalue <= baseline) and (low_values <= ydatav[2] <= baseline)):
                # Protect baseline values
                w1 = 0.1
            else:
                # Protect high values
                w1 = 1.0 if yhat >= yvalue else 0.33

            # Weight the distance
            max_dist = max(abs(xdatav[1] - xdatav[0]), abs(xdatav[1] - xdatav[2]))
            scaled_days = 1.0 - scale_min_max(max_dist, 0.1, 0.9, 0.0, max_days)

            w2 = scale_min_max(logistic_func(scaled_days, 0.5, 10.0), 0.1, 1.0, 0.0, 1.0)

            w = w1 * w2

            # Get the weighted deviation
            dev = abs(prop_diff(yvalue, yhat)) * w

            if not np.isinf(dev) and not np.isnan(dev):

                # Remove the outlier
                if dev > dev_thresh:

                    # This check avoids large dropoffs from senescence to valleys
                    if (prop_diff(ydatav[0], ydatav[1]) < -0.9) and (prop_diff(ydatav[0], ydatav[2]) < -0.9):
                        outlier_count = 0

                    # This check avoids large increases from valleys to green-up
                    elif (prop_diff(ydatav[2], ydatav[1]) < -0.9) and (prop_diff(ydatav[2], ydatav[0]) < -0.9):
                        outlier_count = 0
                    else:

                        if ydatav[0] < yvalue > ydatav[2]:
                            # Local spike
                            max_outlier = indexesv[1]
                            outlier_count = 1
                        elif ydatav[0] > yvalue < ydatav[2]:
                            max_outlier = indexesv[1]
                            outlier_count = 1

    else:

        for jj in range(1, valid_samples-1):

            xvalue = xdatav[jj]
            yvalue = ydatav[jj]

            # Sample prediction, clipped to 0-1
            yhat = clip(mbf[ii, 0] * xvalue + mbf[ii, 1] * pow(xvalue, 2) + mbf[ii, 2], 0.0, 1.0)

            # Give a lower weight for low values
            # if yarray_[ii, jj] <= low_values:
            #     w2 = 1.0
            if low_values < yvalue <= baseline:
                continue

            # Give a lower weight for values above prediction
            w1 = 1.0 if yhat >= yvalue else 0.25

            # Give a lower weight for values at prediction ends
            w2 = 1.0 - scale_min_max(abs(float(dhalf) - float(indexesv[jj])), 0.1, 1.0, 0.0, dmax)

            # Combine the weights
            w = w1 * w2

            # Get the weighted deviation
            dev = abs(prop_diff(yvalue, yhat)) * w

            if not np.isinf(dev) and not np.isnan(dev):

                # Remove the outlier
                if dev > dev_thresh:

                    # This check avoids large dropoffs from senescence to valleys
                    if (prop_diff(ydatav[0], ydatav[2]) < -0.9) and (prop_diff(ydatav[0], ydatav[3]) < -0.9):
                        outlier_count += 0

                    # This check avoids large increases from valleys to green-up
                    elif (prop_diff(ydatav[4], ydatav[2]) < -0.9) and (prop_diff(ydatav[4], ydatav[1]) < -0.9):
                        outlier_count += 0
                    else:
                        
                        import ipdb;ipdb.set_trace()

                        if dev > max_dev:

                            max_dev = dev
                            max_outlier = indexesv[jj]

                        outlier_count += 1
                        
        beta1 = mbf[ii, 0]
        beta2 = mbf[ii, 1]
        alpha = mbf[ii, 2]     
        
        xdatav = xdatav[1:4]
        ydatav = ydatav[1:4]
        
    # Restrict outlier removal to one value
    if outlier_count > 0:

        outlier_data = OutlierData(col_pos=start_, xdata=xdatav, ydata=ydatav, alpha=alpha, beta1=beta1, beta2=beta2)
        outlier_info.append(outlier_data)
        yarray_[ii, max_outlier] = yhat
        
    return outlier_info, yarray_
    
        
def remove_outliers_linear(xsparse,
                           array_,
                           mbz,
                           ii,
                           ncols,
                           min_samples,
                           max_days,
                           dev_thresh,
                           no_data):
    
    outlier_info = []
    indexes = []
    xdata = []
    ydata = []
    
    # Set the lower baseline as the 10th percentile
    baseline = 0.15

    # Set low values as the 2nd percentile
    low_values = 0.03

    for start in range(0, ncols-min_samples):

        ######################################
        # Check for large gaps with low values
        ######################################

        valid_samples1 = 0
        middle1 = 0
        max_dist1 = 0.0

        for end1 in range(start, ncols - 4):

            if array_[ii, end1] != no_data:

                increase_valid = False

                if valid_samples1 == 0:

                    if array_[ii, end1] >= baseline + 0.1:
                        max_dist1 = max(xsparse[end1] - xsparse[start], max_dist1)
                        middle1 = end1
                        increase_valid = True

                if valid_samples1 == 1:

                    if array_[ii, end1] <= baseline:
                        max_dist1 = max(xsparse[end1] - xsparse[middle1], max_dist1)
                        middle1 = end1
                        increase_valid = True
                    else:
                        valid_samples1 = 0

                if valid_samples1 == 2:

                    if array_[ii, end1] <= baseline:
                        max_dist1 = max(xsparse[end1] - xsparse[middle1], max_dist1)
                        increase_valid = True
                        middle1 = end1
                    else:
                        valid_samples1 = 0

                if valid_samples1 == 3:

                    max_dist1 = max(xsparse[end1] - xsparse[middle1], max_dist1)

                    if max_dist1 > max_days:

                        if array_[ii, end1] >= baseline + 0.1:
                            array_[ii, middle1] = no_data

                    break

                if increase_valid:
                    valid_samples1 += 1   
                    
        ##############################################
        # Check for large gaps with a single low value
        ##############################################

        valid_samples3 = 0
        middle2 = start
        max_dist2 = 0.0

        for end3 in range(start, ncols-3):

            if array_[ii, end3] != no_data:

                increase_valid = False

                if valid_samples3 == 0:

                    if array_[ii, end3] >= 0.35:
                        max_dist2 = max(xsparse[end3] - xsparse[start], max_dist2)
                        middle2 = end3
                        increase_valid = True

                if valid_samples3 == 1:

                    if array_[ii, end3] <= baseline:
                        max_dist2 = max(xsparse[end3] - xsparse[middle2], max_dist2)
                        increase_valid = True
                        middle2 = end3
                    else:
                        valid_samples3 = 0

                if valid_samples3 == 2:

                    max_dist2 = max(xsparse[end3] - xsparse[middle2], max_dist2)

                    if max_dist2 > max_days:

                        if array_[ii, end3] >= 0.35:
                            array_[ii, middle2] = no_data

                    break

                if increase_valid:
                    valid_samples3 += 1           
                    
        ####################
        # Check for outliers
        ####################

        valid_samples2 = 0

        x_mean = 0.0
        x_var = 0.0
        xy_cov = 0.0

        end2 = start

        while True:

            if end2 + 2 >= ncols:
                break

            if xsparse[end2] - xsparse[start] > max_days:
                break

            if valid_samples2 >= min_samples:
                break

            if array_[ii, end2] != no_data:

                x_mean += xsparse[end2]

                valid_samples2 += 1

            end2 += 1

        if valid_samples2 >= min_samples:

            x_mean /= float(valid_samples2)

            n_valid = 0
            for nct in range(start, end2):
                if array_[ii, nct] != no_data:
                    n_valid += 1

            # Use the median instead of the mean
            #   to avoid fitting to potential outliers
#             array_[array_ == no_data] = np.nan
#             y_med = np.nanmean(array_[ii, start:start+end2])
#             array_[np.isnan(array_)] = no_data
            
            for jidx in range(start, end2):

                if array_[ii, jidx] != no_data:

#                     x_dev = xsparse[jidx] - x_mean
#                     y_dev = array_[ii, jidx] - y_med

#                     x_var += pow(x_dev, 2)
#                     xy_cov += x_dev * y_dev

                    indexes.append(jidx)
                    xdata.append(xsparse[jidx])
                    ydata.append(array_[ii, jidx])

            # least squares, 1 variable
            #   beta = slope = sum(x * y) / sum(x^2)
            #   alpha = intercept = y_mu - beta * x_mu
#             beta = xy_cov / x_var
#             alpha = y_med - beta * x_mean

#             mbz[ii, 0] = beta
#             mbz[ii, 1] = alpha

            # Check if values deviate from the prediction
            outlier_info, array_ = _check_deviation(indexes,
                             xdata,
                             ydata,
                             array_,
                             ii,
                             start,
                             end2,
                             mbz,
                             dev_thresh,
                             no_data,
                             baseline,
                             low_values,
                             valid_samples2,
                             True,
                             max_days,
                             outlier_info)

            indexes = []
            xdata = []
            ydata = []
            
    return outlier_info, array_[0]


def remove_outliers_polynomial(xsparse,
                               array_,
                               mbz,
                               ii,
                               ncols,
                               min_samples,
                               max_days,
                               dev_thresh,
                               no_data):

    outlier_info = []
    indexes = []
    xdata = []
    ydata = []

    # Set the lower baseline as the 10th percentile
    baseline = 0.2

    # Set low values as the 2nd percentile
    low_values = 0.03

    for start in range(0, ncols-min_samples):

        valid_samples = 0

        x1_mean = 0.0
        x1_var = 0.0
        x1y_cov = 0.0

        x2_mean = 0.0
        x2_var = 0.0
        x2y_cov = 0.0

        # y_mn = 0.0

        xx_cov = 0.0

        end = start

        while True:

            if end + 2 >= ncols:
                break

            if xsparse[end] - xsparse[start] > max_days:
                break

            if valid_samples >= min_samples:
                break

            if array_[ii, end] != no_data:

                x1_mean += xsparse[end]
                x2_mean += pow(xsparse[end], 2)

                valid_samples += 1

            end += 1

        if valid_samples >= min_samples:

            x1_mean /= float(valid_samples)
            x2_mean /= float(valid_samples)

            # Use the median instead of the mean
            #   to avoid fitting to potential outliers
            array_[array_ == no_data] = np.nan
            y_mn = np.nanmean(array_[ii, start:start+end])
            array_[np.isnan(array_)] = no_data

            for jidx in range(start, end):

                if array_[ii, jidx] != no_data:

                    x1_dev = xsparse[jidx] - x1_mean
                    x2_dev = pow(xsparse[jidx], 2) - x2_mean

                    y_dev = array_[ii, jidx] - y_mn

                    x1_var += pow(x1_dev, 2)
                    x1y_cov += (x1_dev * y_dev)

                    x2_var += pow(x2_dev, 2)
                    x2y_cov += (x2_dev * y_dev)

                    xx_cov += (x1_dev * x2_dev)

                    indexes.append(jidx)
                    xdata.append(xsparse[jidx])
                    ydata.append(array_[ii, jidx])

            numerator = (x1_var * x2_var) - pow(xx_cov, 2)

            # least squares, 2 variables
            beta1 = ((x2_var * x1y_cov) - (xx_cov * x2y_cov)) / numerator
            beta2 = ((x1_var * x2y_cov) - (xx_cov * x1y_cov)) / numerator

            alpha = y_mn - (beta1 * x1_mean) - (beta2 * x2_mean)

            mbz[ii, 0] = beta1
            mbz[ii, 1] = beta2
            mbz[ii, 2] = alpha

            # Check if values deviate from the prediction
            outlier_info, array_ = _check_deviation(indexes,
                                                    xdata,
                                                    ydata,
                                                    array_,
                                                    ii,
                                                    start,
                                                    end,
                                                    mbz,
                                                    dev_thresh,
                                                    no_data,
                                                    baseline,
                                                    low_values,
                                                    valid_samples,
                                                    False,
                                                    max_days,
                                                    outlier_info)

            indexes = []
            xdata = []
            ydata = []
            
    return outlier_info


def remove_outliers(xinfo, 
                    yseries,
                    no_data_value=0,
                    max_outlier_days1=120,
                    max_outlier_days2=90,
                    min_outlier_values=7,
                    dev_thresh1=0.2,
                    dev_thresh2=0.1,
                    n_jobs=1):

    outlier_info1, yseries_no_outliers = remove_outliers_linear(xinfo.xd,
                                 np.ascontiguousarray(np.c_[yseries.flatten().copy(), yseries.flatten().copy()].T, dtype='float64'), 
                                 np.zeros((2, 3), dtype='float64'),
                                 0,
                                 xinfo.xd.shape[0],
                                 3,
                                 max_outlier_days1,
                                 dev_thresh1,
                                 no_data_value)

    outlier_info2 = remove_outliers_polynomial(xinfo.xd,
                                 np.ascontiguousarray(np.c_[yseries_no_outliers.flatten().copy(), yseries_no_outliers.flatten().copy()].T, dtype='float64'), 
                                 np.zeros((2, 3), dtype='float64'),
                                 0,
                                 xinfo.xd.shape[0],
                                 min_outlier_values,
                                 max_outlier_days2,
                                 dev_thresh2,
                                 no_data_value)
    
    return outlier_info1 + outlier_info2


def harmonics(in_array):
    
    ncols = in_array.shape[0]
    out_array_harmonics__ = np.zeros(ncols, dtype='float64')

    w = 2.0 * 3.141592653589793 / 365.25

    # Iterate over each annual cycle
    for j in range(0, ncols, 365):

        x1_mean = 0.0
        x1_var = 0.0
        x1y_cov = 0.0

        x2_mean = 0.0
        x2_var = 0.0
        x2y_cov = 0.0

        y_mean = 0.0
        x1x2_cov = 0.0

        # Get the data means
        for j1 in range(j, j+min(365, ncols-j)):

            x1_mean += np.cos(w*(float(j1)+1))
            x2_mean += np.sin(w*(float(j1)+1))

        x1_mean /= float(ncols)
        x2_mean /= float(ncols)

        y_mean = np.median(in_array[j:j+min(365, ncols-j)])

        for j2 in range(j, j+min(365, ncols-j)):

            x1_dev = np.cos(w*(float(j2)+1)) - x1_mean
            x2_dev = np.sin(w*(float(j2)+1)) - x2_mean

            y_dev = in_array[j2] - y_mean

            x1_var += pow(x1_dev, 2)
            x1y_cov += (x1_dev * y_dev)

            x2_var += pow(x2_dev, 2)
            x2y_cov += (x2_dev * y_dev)

            x1x2_cov += (x1_dev * x2_dev)

        numerator = (x1_var * x2_var) - pow(x1x2_cov, 2)

        beta0 = ((x2_var * x1y_cov) - (x1x2_cov * x2y_cov)) / numerator
        beta1 = ((x1_var * x2y_cov) - (x1x2_cov * x1y_cov)) / numerator
        alpha = y_mean - (beta0 * x1_mean) - (beta1 * x2_mean)

        for j3 in range(j, j+min(365, ncols-j)):

            yhat = np.clip(beta0*np.cos(w*(float(j3)+1)) + beta1*np.sin(w*(float(j3)+1)) + alpha, 0.0, 1.0)
            out_array_harmonics__[j3] = yhat
            
    return out_array_harmonics__
