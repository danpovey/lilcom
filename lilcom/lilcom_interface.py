import numpy as np

import lilcom_extension



def compress(input,
             tick_power=-8,
             do_regression=True):
  """
  Compresses a NumPy array lossily

  Args:
    input:   A numpy.ndarray.  May be of type np.float or np.double.
    tick_power:  Determines the accuracy; the input will be compressed to integer
             multiples of 2^tick_power.
    do_regression:  If true, use regression on previous elements in the array
             (one regression coefficient per axis) to reduce the magnitudes of
             the values to compress.
  """
  n_dim = len(input.shape)

  if not (n_dim > 0 and n_dim < 16):
    raise ValueError("Expected number of axes to be in [1,15], got: ",
                     n_dim)

  input = np.ascontiguousarray(input, np.float32)

  coeffs = regress_array(input, do_regression)

  # int_coeffs will be in [-256, 256]
  int_coeffs = [ round(x * 256) for x in coeffs ]

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
    if input.shape[axis] == 1:
      continue  # the size is 1 so we can't do regression

    # swap axes, so we can work on axis 0 for finding the coefficient.
    input = np.swapaxes(input, 0, axis)
    coeff = (input[:-1] * input[1:]).sum() / ((input[:-1] ** 2).sum() + 1.0e-20)
    if abs(coeff) < 0.02:
      coeff = 0.0
    elif coeff < -1.0:
      coeff = -1.0
    elif coeff > 1.0:
      coeff = 1.0
    coeffs[axis] = coeff
    input[1:] -= input[:-1] * coeff
    # swap the axes back
    input = np.swapaxes(input, 0, axis)
  return coeffs


def decompress(byte_string):
  """
   Decompresses audio data compressed by compress().

   Args:
       input:    A bytes object as returned by compress()
   Return:
       On success returns a NumPy array of float; on failure
       raises an exception.
     """
  if not isinstance(byte_string, bytes):
    raise TypeError("Expected input to be of type `bytes`, got {}".format(type(byte_string)))

  shape = lilcom_extension.get_float_matrix_shape(byte_string)

  if shape is None:
    raise ValueError("Could not work out shape of array from input: "
                     "is not really compressed data?")

  ans = np.empty(shape, dtype=np.float32)

  ret = lilcom_extension.decompress_float(byte_string, ans)

  if ret is None or ret != 0:
    raise ValueError("Something went wrong in decompression (likely bad data): "
                     "decompress_float returned {}".format(ret))
  else:
    return ans



def get_shape(byte_string):
  """
  Decompresses audio data compressed by compress() just enough to get the
  shape it would be when decompressed.
   Args:
       input:    A bytes object as returned by compress()
   Return:
       On success returns a tuple representing the shape; on failure
       raises an exception.
  """

  shape = lilcom_extension.get_float_matrix_shape(byte_string)

  if shape is None:
    raise ValueError("Could not work out shape of array from input: "
                     "is not really compressed data?")
  return shape
