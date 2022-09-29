# lilcom


This package lossily compresses floating-point NumPy arrays
into byte strings, with an accuracy specified by the user.
The main anticipated use is in machine learning applications, for
storing things like training data and models.

This package requires Python 3 and is not compatible with Python 2.

## Installation with PyPi

From PyPi you can install this with just
```
pip3 install lilcom
```
## Installation with conda

```bash
conda install -c lilcom lilcom
```

### How to use

The most common usage pattern will be as follows (showing Python code):
```
import numpy as np
import lilcom

a = np.random.randn(300,500)
a_compressed = lilcom.compress(a)
# a_compressed is of type `bytes`, a byte string.
# In this case it will use about 1.3 bytes per element.

# decompress a
a_decompressed = lilcom.decompress(a_compressed)
```
The compression is lossy so `a_decompressed` will not be exactly the same
as `a`.  The amount of error (absolute, not relative!)  is determined by the
optional `tick_power` argument to lilcom.compress() (default: -8), which is the
power of 2 used for the step size between discretized values.  The maximum error
per element is 2**(tick_power-1), e.g.  for tick_power=-8, it is 1/512.



### Installation from Github

To install lilcom from github, first clone the repository;
```
git clone https://github.com/danpovey/lilcom.git
```
then run setup with `install` argument.
```
python3 setup.py install
```
(you may need to add the `--user` flag if you don't have system privileges).
You need to make sure a C++ compiler is installed, e.g. g++ or clang.
To test it, you can then cd to `test` and run:

```
python3 test_lilcom.py
```


## Technical details

The algorithm regresses each element on the previous element (for a 1-d array)
or, for general n-d arrays, it regresses on the previous elements along each of
the axes, i.e.  we regress element `a[i,j]` on `a[i-1,j]` and `a[i,j-1]`.  The
regression coefficients are global and written as part of the header in the
string.

The elements are then integerized and the integers are compressed using
an algorithm that gives good compression when successive elements tend to
have about the same magnitude (the number of bits we're transmitting
varies dynamically acccording to the magnitudes of the elements).

The core parts of the code are implemented in C++.


