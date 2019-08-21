
# TODO: Checking the version of python interpreter. This code only works with python3.

#from distutils.core import setup, Extension
from setuptools import setup, Extension
import os
import numpy


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extension_mod = Extension("lilcom",
                          sources=["lilcom/lilcom_c_extension.c","lilcom/lilcom.c"],
                          include_dirs=[numpy.get_include()])

setup(
    name = "lilcom",
    version = "0.0.0",
    author = "Daniel Povey, Soroush Zargar, Mahsa Yarmohammadi",
    author_email = "dpovey@gmail.com",
    description = ("Small compression utility for sequence data in NumPy"),
    license = "BSD",
    keywords = "compression numpy",
    url = "http://packages.python.org/an_example_pypi_project",
    ext_modules=[extension_mod],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)



