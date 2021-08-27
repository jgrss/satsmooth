import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

try:
    from Cython.Distutils import build_ext
except:
    from distutils.command import build_ext

import numpy as np


# Parse the version from the module.
# Source: https://github.com/mapbox/rasterio/blob/master/setup.py
with open('satsmooth/version.py') as f:

    for line in f:

        if line.find("__version__") >= 0:

            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")

            continue

pkg_name = 'satsmooth'
maintainer = 'Jordan Graesser'
maintainer_email = ''
description = 'Satellite n-dimensional signal smoothing'
git_url = 'http://github.com/jgrss/satsmooth.git'

with open('README.md') as f:
    long_description = f.read()

with open('LICENSE.txt') as f:
    license_file = f.read()

required_packages = ['cython',
                     'numpy']


def get_packages():
    return setuptools.find_packages()


def get_package_data():

    return {'': ['*.md', '*.txt'],
            'satsmooth': ['detect/*.so',
                          'smooth/*.so',
                          'preprocessing/*.so',
                          'testing/*.so',
                          'utils/*.pxd']}


def get_extensions():

    # extra_compile_args=['-O3', '-ffast-math', '-march=native', '-fopenmp']

    return [Extension('*',
                      sources=['satsmooth/detect/_signal.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_linear_interp.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_linear_interp_regrid.pyx']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_linear_interp_regrid_multi.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_linear_interp_regrid_multi_indexing.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_fill_gaps.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/preprocessing/_outlier_removal.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/anc/_lowess_smooth.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/smooth/_adaptive_bilateral.pyx']),
            Extension('*',
                      sources=['satsmooth/smooth/_rolling1d.pyx']),
            Extension('*',
                      sources=['satsmooth/smooth/_rolling2d.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp']),
            Extension('*',
                      sources=['satsmooth/smooth/_spatial_temporal.pyx'],
                      extra_compile_args=['-fopenmp'],
                      extra_link_args=['-fopenmp'])]


def setup_package():

    include_dirs = [np.get_include()]

    metadata = dict(name=pkg_name,
                    maintainer=maintainer,
                    maintainer_email=maintainer_email,
                    description=description,
                    license=license_file,
                    version=version,
                    long_description=long_description,
                    packages=get_packages(),
                    package_data=get_package_data(),
                    ext_modules=cythonize(get_extensions()),
                    cmdclass=dict(build_ext=build_ext),
                    download_url=git_url,
                    install_requires=required_packages,
                    include_dirs=include_dirs)

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
