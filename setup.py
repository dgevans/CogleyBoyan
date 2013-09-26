
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext = Extension('LearningProblems.a_q_c', ['LearningProblems/a_q_c.pyx'],
include_dirs=[numpy.get_include()]
)

ext_int = Extension('LearningProblems.sparse_integrator', ['LearningProblems/integrator.pyx','LearningProblems/sparse_grid_hw.c'],
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