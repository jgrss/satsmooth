from ._linear_interp import interp1d, interp2d
# from ._cubic_interp_regrid_multi import CinterpMulti
from ._linear_interp_regrid import Linterp
from ._linear_interp_regrid_multi import LinterpMulti
from ._linear_interp_regrid_multi_indexing import LinterpMultiIndex
from ._fill_gaps import fill_gaps
from ._outlier_removal import remove_outliers

__all__ = ['interp1d',
           'interp2d',
           'Linterp',
           'LinterpMulti',
           'LinterpMultiIndex',
           'fill_gaps',
           'remove_outliers']
