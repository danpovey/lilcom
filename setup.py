# Note to self on how to build and upload this:
#  rm -r dist;  python3 ./setup.py sdist;   twine upload  dist/*
# and then input my PyPi username and password.
# I have to bump the version before doing this, or it won't allow the
# upload.

# Check python version: If python 3 was not found then it returns 1 and does
#   not do anything

import re
import sys

import setuptools

from cmake.cmake_extension import BuildExtension, bdist_wheel, cmake_extension

if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported by k2.")
    sys.exit(-1)

if sys.version_info < (3, 6):
    print("Python < 3.6 is not supported")
    print("lilcom works only with python >= 3.6")
    sys.exit(-1)


def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme


def get_package_version():
    with open("CMakeLists.txt") as f:
        content = f.read()

    match = re.search(r"set\(LILCOM_VERSION (.*)\)", content)
    latest_version = match.group(1).strip('"')
    return latest_version


package_name = "lilcom"


setuptools.setup(
    name=package_name,
    python_requires=">=3.6",
    version=get_package_version(),
    author="Daniel Povey, Meixu Song, Soroush Zargar, Mahsa Yarmohammadi, Jian Wu",
    author_email="dpovey@gmail.com",
    description=("Lossy-compression utility for sequence data in NumPy"),
    license="MIT",
    keywords="compression numpy",
    packages=["lilcom"],
    install_requires=["numpy"],
    url="https://github.com/danpovey/lilcom",
    ext_modules=[cmake_extension("lilcom_extension")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: System :: Archiving :: Compression",
        "License :: OSI Approved :: MIT License",
    ],
)
