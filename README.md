# lilcom


This package lossily compresses floating-point NumPy arrays
into byte strings, with an accuracy specified by the user.
The main anticipated use is in machine learning applications, for
storing things like training data and models.

This package requires Python 3 and is not compatible with Python 2.

## Installation

### Using PyPi

From PyPi you can install this with just
```
pip3 install lilcom
```


### Using Github Repository
To install lilcom first clone the repository;
```

git clone git@github.com:danpovey/lilcom.git
```

then run setup with `install` argument.
```
python3 setup.py install
```
(you may need to add the `--user` flag if you don't have system privileges).
To test it, you can then cd to `test` and run:

```
python3 test_interface.py
```

### How to use this compression method

The most common usage pattern will be as follows (showing Python code):
```
# Let a be a NumPy ndarray with type np.float32 or np.float64

# compress a.

import lilcom
a_compressed = lilcom.compress(a)

# decompress a
a_decompressed = lilcom.decompress(a_compressed)
```
Note: the compression is lossy so `a_decompressed` will not be
exactly the same as `a`.  The amount of error is determined
by the optional `tick_power` argument to lilcom.compress();
the maximum error per element is 2**(tick_power-1), e.g.
for tick_power=8, the maximum error per element is 1/512.



## Technical details

The algorithm regresses each element on the previous element (for a 1-d array)
or, for general n-d arrays, all the previous elements along all the axes, i.e.
we regress element a[i,j] on a[i-1,j] and a[i,j-1].  The regression coefficients
are global and written as part of the header in the string.

The elements are then integerized and the integers are compressed using
an algorithm that gives good compression when successive elements tend to
have about the same magnitude.

The core parts of the code are implemented in C++.
