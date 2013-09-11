
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension("a_q_c", ["a_q_c.pyx"],
include_dirs=["/opt/local/include",numpy.get_include()],
library_dirs=["/opt/local/lib"],
libraries=["gsl","gslcblas"]
)

ext_int = Extension("sparse_integrator", ["integrator.pyx","sparse_grid_hw.c"],
include_dirs=[numpy.get_include()]
)


setup(
    name = 'a_q',
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)

setup(
    name = 'sparse_integrator',
    ext_modules=[ext_int],
    cmdclass = {'build_ext': build_ext}
)