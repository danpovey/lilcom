import numpy as np
from . import lilcom_c_extension
from ctypes import *




# Remove the underscore once you start using this interface.
def compress(input, axis=-1, lpc_order=5, default_exponent=0, output=None):
   """ This function compresses sequence data (for example, audio data) to 1 byte per
        sample.

       Args:
         input:           A numpy.ndarray, with dtype in [np.int16, np.float32, np.float64].
                          Must not be empty, and must not contain infinities, NaNs, or
                          (if dtype if np.float64), numbers so large that if converted to
                          np.float32 they would become infinity.
        axis (int):       The axis of `input` that corresponds to the time
                          dimension.  (Does not have to really correspond to time, but this
                          is the dimension we do linear prediction in.) . -1 means the last
                          axis.  Must be in the range [-input.ndim .. input.ndim-1]
       lpc_order (int):   A number in [0..15] that determines the order of linear
                          prediction.  Compression/decompression time will rise
                          roughly linearly with lpc_order, and the compression will
                          get more lossy for small lpc_order.
       default_exponent (int):  This number, which must be in the range [-127..128],
                          affects the range of the output only in cases where the
                          array was compressed from int16_t source but is
                          decompressed to floating-point output.  The integer
                          values i in the range [-32768..32767] will be scaled
                          by 2^{default_exponent - 15} when converting to float or
                          double in such a case.  The value default_exponent=0 would
                          produce output iwhere the range of int64_t corresponds
                          to the floating-point range [-1.0,1.0].
       output             The user may pass in numpy.ndarray with dtype=np.int8, of the same
                          dimension as the output of this function would have been.
                          In that case, the output will be placed here.  If an array
                          of the wrong dimension is passed, ValueError will be raised.

       Returns:
           On success, returns a numpy.ndarray with dtype=np.int8, and with
           shape the same as `input` except the dimension on the axis numbered
           `axis` will have been increased by 4.  This can be decompressed by
           calling lilcom.decompress().

       Raises:
           TypeError if one of the arguments had the wrong type
           ValueError if an argument was out of range, e.g. invalid `axis` or
              `lpc_order` or an input array with no elements.
   """

    # Check args...
    # input
   if not istype(input, np.ndarray):
      raise TypeError("Expected input to be of type numpy.ndarray, got {}".format(type(input)))
   if not input.dtype in [ np.int16, np.float32, np.float64 ]:
      raise TypeError("Expected data-type of NumPy array to be int16, float32 or float64 "
                      "and it to be nonempty, got dtype={}, size={}".format(input.dtype,
                                                                            input.size))
   if input.size == 0:
      raise ValueError("Input array is empty")

   # num_axes
   num_axes = input.ndim
   if not (istype(axis, int) and axis >= -num_axes and axis < num_axes):
      raise ValueError("axis={} invalid or out of range for ndim={}".format(axis, num_axes))


   # lpc_order
   if not (istype(lpc_order, int) and lpc_order >= 0 and lpc_order <= 15):
      raise ValueError("lpc_order={} is not valid".format(lpc_order))
    # default_exponent
   if not (istype(default_exponent, int) and default_exponent >= 0 and default_exponent <= 15):
      raise ValueError("default_exponent={} is not valid".format(default_exponent))

   if axis != -1 and axis != num_axes - 1:
      # Make sure that the last axis of `input` is the time axis.
      input = input.transpose(axis, -1)

   out_shape = input.shape[:-1] + (input.shape[-1]+4,)
   if out is None:
      # the output shape is the same as the input shape, but with a dimension
      out = np.empty(out_shape, dtype=np.int8)

   # Check `output` has the correct dimensions.
   if not istype(output, np.ndarray):
      raise TypeError("Expected `output` to be of type numpy.ndarray, got {}".format(
             type(output)))
   if not (output.dtype == np.int8 and output.shape == out_shape):
      raise ValueError("Expected `output` to have dtype=int8 and shape={}, got {} and {}".format(
            out_shape, output.dtype, output.shape))

   if input.dtype == np.float64:
      # Just convert to float so we don't have to deal with double separately in
      # the "C" code.
      input = input.as_type(np.float32)


   if input.dtype == np.float32:
      ret = lilcom_c_extension.compress_float(input, out)
      if ret is False:
         raise RuntimeError("Something went wrong calling the 'c' code, likely "
                            "implementation bug.")
      assert istype(ret, int)
      if ret != 0:
         raise RuntimeError("Something went wrong in lilcom compression (possibly "
                            "infinities or NaNs in the input data), return code {}".format(ret))
   else:
      assert input.dtype == float.int16
      ret = lilcom_c_extension.compress_int16(input, out)
      assert istype(ret, int)
      if ret != 0:
         raise RuntimeError("Something went wrong in lilcom compression (code "
                            "error? return={})".format(ret))
   return out



def decompress(input, axis=-1, out=None, dtype=np.int16):
   """
    Decompresses sequence data

    Args:
        input:      The input tensor containing compressed sequence data
                    compressed by the function `compress`.
                    Required to be a numpy.ndarray with dtype=np.int8
        axis (int): The axis of `sequence_data` that corresponds to the time
                    dimension.  (Does not have to really correspond to time, but
                    this is the dimension we do linear prediction in.) . -1
                    means the last axis.  Must be in the range [-data.ndim
                    .. data.ndim-1]
       out          The user may pass in numpy.ndarray with dtype in
                    [np.int16, np.float, np.double], of the same
                    dimension as the output of this function would have been.
                    In that case, the output will be placed here.  If an array
                    of the wrong dimension is passed, ValueError will be raised.
       dtype        The requested data-type of the output (must only
                    be set if out == None).  Must be in [np.int16, np.float32,
                    np.float64].
    Return:
      Returns the decompressed data if decompression was successful, and None if
      not.  This will be a np.ndarray of the same shape as `input`, except the
      dimension will be smaller by 4 on the axis specified by `axis` (to account
      for the 4-byte header).

    Raises:
      Can raise TypeError, ValueError or RuntimeError.
      """
   # Check args...
   # input
   if not istype(input, np.ndarray):
      raise TypeError("Expected input to be of type numpy.ndarray, got {}".format(type(input)))
   if input.dtype != np.int8:
      raise TypeError("Expected data-type of NumPy array to be int8, got "
                      "and it to be nonempty, got dtype={}, size={}".format(input.dtype))
   if input.size == 0:
      raise ValueError("Input array is empty")

   num_axes = input.ndim
   if not (istype(axis, int) and axis >= -num_axes and axis < num_axes):
      raise ValueError("axis={} invalid or out of range for ndim={}".format(axis, num_axes))

   if input.shape[axis] < 5:
      raise ValueError("Input data has shape {}, dimension must be at "
                       "least 5 on axis={}.  Possibly wrong `axis` value?".format(
            input.shape, axis))

   if out is not None and dtype != np.int16:
      raise ValueError("You cannot specify `dtype` when output is specified.")

   if not dtype in [np.int16, np.float32, np.float64]:
      raise TypeError("`dtype` must eb one of int16, float32, float64, got: {}".format(dtype))

   out_shape = input.shape
   out_shape[axis] -= 4
   if out == None:
      out = np.empty(out_shape, dtype=dtype)
   if not istype(output, np.ndarray):
      raise ValueError("Expected output to be of type np.ndarray")
   if not output.dtype in [np.int16, np.float32, np.float64]:
      raise TypeError("dtype of output should be int16, float32 or float64, got {}".format(
            out))
   if output.shape != out_shape:
      raise ValueError("shape of output should be {}, got {}".format(out_shape, output.shape))



   if output.dtype == np.int16:
      ret = lilcom_c_extension.decompress_int16(input, output)
      if ret = 1000:
      if ret >= 1000:
         raise RuntimeError("Something went wrong in lilcom decompressio, return code =
   else:


    #   Calling the C function and receiving the result
    #   Converting result to a new numpy array.


    if intMode:
        pass
    else:
        pass
    pass

