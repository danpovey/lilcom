# lilcom


Package for compression and decompression of sequence data (specially audio files). This package includes:
* Compression from 16-bit integer stream to 8-bit integer stream
* Compression from floating point stream to 8-bit integer stream
* Decompression from 8-bit integer stream to 16-bit integer stream
=======
This package requires Python 3 and is not compatible with Python 2.

## Installation

### Using Github Repository
To install lilcom first clone the repository;
```

git clone git@github.com:danpovey/lilcom.git
```

then run setup with `install` argument.
```
python setup.py install
```

and then for test, run:

```
python3 test/routine1-one-dimensional-integer.py
python3 test/routine2-one-dimensional-float.py
```

