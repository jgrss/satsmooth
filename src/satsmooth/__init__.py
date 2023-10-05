__path__: str = __import__('pkgutil').extend_path(__path__, __name__)
__version__ = '1.5.16'

from .detect import (
    group_peaks_valleys2d,
    peaks_valleys1d,
    peaks_valleys2d,
    phenometrics1d,
    phenometrics2d,
)
from .preprocessing import (
    Linterp,
    LinterpMulti,
    LinterpMultiIndex,
    fill_gaps,
    interp1d,
    interp2d,
    remove_outliers,
)
from .smooth import (
    adaptive_bilateral,
    rolling_max1d,
    rolling_mean1d,
    rolling_mean2d,
    rolling_mean_std1d,
    rolling_min1d,
    rolling_quantile2d,
    rolling_std1d,
    spatial_temporal,
)

__all__ = [
    'interp1d',
    'interp2d',
    'Linterp',
    'LinterpMulti',
    'LinterpMultiIndex',
    'fill_gaps',
    'remove_outliers',
    'adaptive_bilateral',
    'peaks_valleys1d',
    'peaks_valleys2d',
    'group_peaks_valleys2d',
    'phenometrics1d',
    'phenometrics2d',
    'rolling_mean1d',
    'rolling_mean_std1d',
    'rolling_min1d',
    'rolling_max1d',
    'rolling_std1d',
    'rolling_mean2d',
    'rolling_quantile2d',
    'spatial_temporal',
    '__version__',
]
