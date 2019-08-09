
# TODO: Checking the version of python interpreter. This code only works with python3.

from distutils.core import setup, Extension
import numpy

extension_mod = Extension("lilcom", ["lilcommodule.c","lilcom.c"],include_dirs=[numpy.get_include()])

setup(name = "lilcom", ext_modules=[extension_mod])


