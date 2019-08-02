
# TODO: Checking the version of python interpreter. This code only works with python2.

# from distutils.core import setup, Extension

from setuptools import setup, Extension
import numpy

extension = Extension("lilcom", ["lilcommodule.c", "lilcom.c"],include_dirs=[numpy.get_include()])

setup(name = "lilcom",
      version = "1.0",
      ext_modules=[extension])




