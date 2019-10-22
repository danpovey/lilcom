# lilcom


Package for compression and decompression of sequence data (especially audio
files), compatible with NumPy arrays.  The main anticipated use is in
machine learning applications.

This package lossily compresses 16-bit integer or floating-point
NumPy arrays into NumPy arrays of characters, using
between 4 and 8 bits per sample (this is selected by the user).

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
python3 setup.py install
```

and then for test, cd to `test` and run:

```
python3 test_interface.py
```

### How to use this compression method

The most common usage pattern will be as follows (showing Python code):
```
# Let a be a NumPy ndarray with type np.int16, np.float32 or np.float64

# compress a.
a_compressed = lilcom.compress(a, axis=1,
                               lpc_order=4,
                               bits_per_sample=8)
# decompress a
a_decompressed = lilcom.decompress(a_compressed, dtype=a.dtype)
```
Note: the compression is lossy so `a_decompressed` will not be
exactly the same as `a`.  The chosen `bits_per_sample` will depend on
the application; 8 is normally suitable, but 5 or 6 should suffice
for audio data that's sampled at a high rate like 44.1kHz.

The argument `axis=1` specifies which axis which will be treated as the "time"
axis.  This should be the axis along which the user expects successive amples to
be the most highly correlated, and also one that has reasonably long sequences;
a 4-byte header is created for each sequence in that axis direction.



## Technical details

The algorithm is based on LPC prediction: LPC coefficients are estimated and it
is the residual from the LPC prediction that is coded.  The LPC coefficients are
not transmitted; they are worked out from the past samples.  The LPC order may
be chosen by the user in the range 0 to 14; the default is 4.  The residual is
coded with an an exponent and a mantissa, like floating point numbers.  Only 1
bit per sample is used to encode the exponent; the reason this is feasible is
that it is the *difference* in the exponent from sample to sample that is
actually encoded.  The algorithm works out the lowest codable sequence of
exponents such that the mantissas are in the codable range.

Because the LPC coefficients are estimated from past samples, this algorithm
is very vulnerable to transmission errors: even a single bit error can
make the entire file unreadable.  This is acceptable in the kinds of
applications we have in mind (mainly machine learning).

The algorithm requires an exact bitwise correspondence between the LPC
computations when compressing and decompressing, so all computations are done in
integer arithmetic and great care is taken to ensure that all arithmetic
operations produce results that are fully defined by the C standard (this means
that we need to avoid signed integer overflow and signed right-shift).

The compression quality is very respectable; at the same bit-rate as MP3 we get
better PSNR, i.e. less compression noise.  (However, bear in mind that MP3 is
optimized for perceptual quality and not PSNR).

