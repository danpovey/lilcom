import numpy as np

from . import lilcom_extension


def test_compression_header():
  dict = {'lpc.lpc_order':8}
  h = lilcom_extension.create_compressor_config(42100, 2, 3, 4, **dict)
  print("header is: {} = {}".format(h, lilcom_extension.compressor_config_to_str(h)))


def test_compression():
  num_channels = 2

  h = lilcom_extension.create_compressor_config(42100, num_channels, 3, 4)

  print("header is: {} = {}".format(h, lilcom_extension.compressor_config_to_str(h)))

  a = ((np.random.rand(num_channels, 15) - 0.5) * 65535).astype(np.int16)

  print("array is: {}, dtype={}", a, a.dtype)

  bytes = lilcom_extension.compress_int16(a, h)

  print("ans is ", bytes)

  x = lilcom_extension.init_decompression_int16(bytes)
  print("Ans from init-decompression is ", x)
  (cf, num_channels, num_samples, sample_rate) = x

  out_array = np.empty((num_channels, num_samples), dtype=np.int16)

  ret = lilcom_extension.decompress_int16(cf, out_array)

  print("Ret from decompress_int16 is ", ret)

  print("Array after decompress_int16 is ", out_array)



def compress(input, 
             tick_power=-8
             do_regression=True, 
             check_aliasing=True):
  """
  Compresses a NumPy array lossily
  
  Args:
    input:   A numpy.ndarray.  May be of type np.float or np.double.
    tick_power:  Determines the accuracy; the input will be compressed to integer 
             multiples of 2^tick_power.
  """
  n_dim = len(input.shape)

  if not (n_dim > 0 and n_dim < 16):
    raise ValueError("Expected number of axes to be in [1,15], got: ",
                     n_dim)

  input = input.astype(np.float32)

  input, coeffs = regress_array(input, do_regression)

  # int_coeffs will be in [-65536, 65536]
  int_coeffs = [ round(x * 65536) for x in coeffs ]
  
  meta = [ tick_power ] + int_coeffs

  ans = lilcom_extension.compress_float(input, meta)
  if not isinstance(ans, bytes):
    raise RuntimeError("Something went wrong in compression, return value was ",
                       ans);
  return ans;



def regress_array(input, regression):
  """
  Works out coefficients for linear regression on the previous sample, for 
  each axis of the array. 

     @param [in] input   The array to be compressed; must contain floats or doubles.
     @param [in] regression  True if we are doing regression; if false,
                         coefficients will be all zero.

  Returns a list of size len(input.shape), with either zero or regression
  coefficients in the range [-1..1] for each corresponding axis.  Each axis's 
  regression coefficient will be zero if regression == False, or the input's 
  size on that axis was 1, or of the estimated regression coefficient had 
  absolute value less than 0.2.
  """

  input = input.copy() # we'll change the array inside this function.
  coeffs = [ 0.0 ] * len(input.shape)
  if not regression:
    return coeffs
  # the + 1.0e-20 is for
  tot_sumsq = np.dot(input.reshape(-1), input.reshape(-1)) + 1.0e-20
  for axis in range(len(input.shape)):
    print("axis is ", axis)
    if input.shape[axis] == 1:
      continue  # the size is 1 so we can't do regression

    # swap axes, so we can work on axis 0 for finding the coefficient.
    input = np.swapaxes(input, 0, axis)
    coeff = (input[:-1] * input[1:]).sum() / ((input[:-1] ** 2).sum() + 1.0e-20)
    print("Coeff is ", coeff)
    if abs(coeff) < 0.02:
      coeff = 0.0
    elif coeff < -1.0:
      coeff = -1.0
    elif coeff > 1.0:
      coeff = 1.0
    coeffs[axis] = coeff
    print("Sumsq is ", (input**2).sum())
    input[1:] -= input[:-1] * coeff
    print("Sumsq is now ", (input**2).sum())
    # swap the axes back
    input = np.swapaxes(input, 0, axis)
  return coeffs
  


def compress(input, sample_rate, loss_level=0, compression_level=5,
             debug=False, **kwargs):
  """
  Compresses `input` and returns it as a byte string.

     Args:
        input:          A numpy.ndarray, which must have dtype=np.int16
                      and 2 axes, whose shape will be interpreted as
                      (num_channels, num_samples).  It can have any
                      stride.
        sample_rate:   The sampling rate of the data, as an integer.
                      E.g. 16000.
        loss_level:   Loss level between 0 (least lossy) and 5 (most lossy)
      compression_level:  Compression level between 0 (fastest, least compression
                      and 5 (slowest, most compression)
       kwargs:        Additonal args that can be used to specify other
                      configuration values, e.g. chunk_size=1024,
                      lpc_lpc_order=4, truncation_num_significant_bits=5...
                      "lpc_" and "truncation_" are special prefixes that
                      point to members `lpc` and `truncation` of class
                      CompressionConfig in compression.h.
  """
  if len(input.shape) != 2:
    raise ValueError("Expected input to have 2 axes, got {}".format(input.shape))
  (num_channels, num_samples) = input.shape
  if num_channels > 5 * num_samples or num_samples == 0:
    # hard to believe that this could be what was intended
    raise ValueError("""It looks like you have your axes mixed up, should be
                     (num_channels, num_samples), shape was {}""".format(input.shape))
  if input.dtype != np.int16:
    raise ValueError('Expected dtype of input to be int16, got {}'.format(input.dtype))
  if not isinstance(sample_rate, int):
    raise TypeError('Expected sample rate to be int, got {}'.format(sample_rate))
  if not (loss_level in range(0,6) and compression_level in range(0,6)):
    raise ValueError('Expected loss-level and compression-level to be in [0..5], got: '
                     '{},{}'.format(loss_level, compression_level))


  config = lilcom_extension.create_compressor_config(sample_rate, num_channels,
                                                     loss_level, compression_level,
                                                     **kwargs)

  if debug:
    print("config is: {}".format(lilcom_extension.compressor_config_to_str(config)))

  ans = lilcom_extension.compress_int16(input, config)
  if not isinstance(ans, bytes):
    raise RuntimeError("Something went wrong in compression, return value was ",
                       ans);
  return ans;



def decompress(input, channel_major=True):
  """
   Decompresses audio data compressed by compress().

   Args:
       input:      A bytes object as returned by compress()
    channel_major: If true the decompressed data will be
                   in the format (num_channels, num_samples);
                   otherwise it will be as (num_samples, num_channels);
                   in either case the data will be contiguous.
   Return:
       On success returns a tuple (decompressed_data, sampling_rate);
       the decompressed data will have dtype=np.int16.  On failure
       raises an exception.
     """
  if not isinstance(input, bytes):
    raise TypeError("Expected input to be of type `bytes`, got {}".format(type(input)))


  ret = lilcom_extension.init_decompression_int16(input)
  if not isinstance(ret, tuple):
    raise RuntimeError("""Something went wrong in decompression (corrupted or
                       invalid data?  Got {}""".format(ret))
  (cf, num_channels, num_samples, sample_rate) = ret
  if channel_major:
    ans = np.empty((num_channels, num_samples), dtype=np.int16)
    to_decompress = ans
  else:
    ans = np.empty((num_samples, num_channels), dtype=np.int16)
    to_decompress = ans.transpose()

  ret = lilcom_extension.decompress_int16(cf, to_decompress)
  if ret is not True:
    raise RuntimeError("Something went wrong in decompression, decompress_int16 returned ",
                       ret);
  return ans;



def test_compression_wrappers():
  num_channels = 2

  h = lilcom_extension.create_compressor_config(42100, num_channels, 3, 4)

  orig = ((np.random.rand(num_channels, 13455) - 0.5) * 65536).astype(np.int16)

  sample_rate = 42100
  compressed = compress(orig, 42100)

  decompressed = decompress(compressed)

  assert np.array_equal(decompressed, orig);

