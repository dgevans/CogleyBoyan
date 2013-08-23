
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension("a_q_c", ["a_q_c.pyx"],
include_dirs=["/opt/local/include",numpy.get_include()],
library_dirs=["/opt/local/lib"],
libraries=["gsl"]
)


setup(
    name = 'a_q',
    ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
)