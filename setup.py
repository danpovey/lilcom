
# TODO: Checking the version of python interpreter. This code only works with python2.

from distutils.core import setup, Extension
import numpy

extension_mod = Extension("lilcomlib", ["lilcommodule.c","lilcom.h"],include_dirs=[numpy.get_include()])

setup(name = "lilcomlib", ext_modules=[extension_mod])


