import platform

try:
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError('Cython must be installed.')

import numpy as np

compile_args = ['-fopenmp']
link_args = ['-fopenmp']

if platform.system().lower() == 'darwin':
    compile_args.insert(0, '-Xpreprocessor')
    link_args = ['-lomp']


def get_extensions():
    return [
        Extension(
            '*',
            sources=['src/satsmooth/detect/_signal.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/preprocessing/_linear_interp.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/preprocessing/_linear_interp_regrid.pyx'],
            language='c++',
        ),
        Extension(
            '*',
            sources=[
                'src/satsmooth/preprocessing/_linear_interp_regrid_multi.pyx'
            ],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language='c++',
        ),
        Extension(
            '*',
            sources=[
                'src/satsmooth/preprocessing/_linear_interp_regrid_multi_indexing.pyx'
            ],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/preprocessing/_fill_gaps.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/preprocessing/_outlier_removal.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/anc/_dl.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/anc/_lowess_smooth.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language='c++',
        ),
        Extension(
            '*',
            sources=['src/satsmooth/smooth/_adaptive_bilateral.pyx'],
            language='c++',
        ),
        Extension('*', sources=['src/satsmooth/smooth/_rolling1d.pyx']),
        Extension(
            '*',
            sources=['src/satsmooth/smooth/_rolling2d.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        ),
        Extension(
            '*',
            sources=['src/satsmooth/smooth/_spatial_temporal.pyx'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            language='c++',
        ),
    ]


def setup_package():
    metadata = dict(
        ext_modules=cythonize(get_extensions()),
        include_dirs=[np.get_include()],
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
