# Note to self on how to build and upload this:
#  rm -r dist;  python3 ./setup.py sdist;   twine upload  dist/*
# and then input my PyPi username and password.
# I have to bump the version before doing this, or it won't allow the
# upload.

# Check python version: If python 3 was not found then it returns 1 and does
#   not do anything
from platform import python_version, system
primer_version = python_version().split(".")
if int(primer_version[0]) != 3:
    print ("This module only works with python3")
    print ("To setup the module simply run `python3 setup.py install`")
    exit(1)


# Checking the version of python interpreter. This code only works with python3.
import sys
if sys.version_info < (3,5):
        sys.exit('Python < 3.5 is not supported')


#from distutils.core import setup, Extension
from setuptools import setup, Extension
import os


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()

extension_mod = Extension("lilcom.lilcom_extension",
                          sources=["lilcom/lilcom_extension.cc",
                                   "lilcom/compression.cc"],
                          # Actually it turns out that the optimization level
                          # and debugging code makes very little difference to
                          # the speed, so we're using options designed to
                          # catch errors.  -ftrapv detects overflow in
                          # signed integer arithmetic (which technically
                          # leads to undefined behavior).
                          extra_compile_args=["-g", "-Wall", "-UNDEBUG", "-Wno-c++11-compat-deprecated-writable-strings"] if system() != "Windows" else [], #, "-ftrapv"],
                          include_dirs=[get_numpy_include()])

setup(
    name = "lilcom",
    python_requires='>=3.5',
    version = "1.1.1",
    author = "Daniel Povey, Meixu Song, Soroush Zargar, Mahsa Yarmohammadi, Jian Wu",
    author_email = "dpovey@gmail.com",
    description = ("Lossy-compression utility for sequence data in NumPy"),
    license = "MIT",
    keywords = "compression numpy",
    packages=['lilcom'],
    url = "https://github.com/danpovey/lilcom",
    ext_modules=[extension_mod],
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    setup_requires=["numpy"],
    install_requires=['numpy'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: System :: Archiving :: Compression",
        "License :: OSI Approved :: MIT License",
    ],
)

exit(0)
