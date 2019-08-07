import numpy
from ctypes import *




# Remove the underscore once you start using this interface.
def _compress(data, axis=-1, lpc_order=5, default_exponent=0, out=None):
   """ This function compresses sequence data (for example, audio data) to 1 byte per
        sample.

       Args:
         data:            A numpy.ndarray, with dtype in [np.int16, np.float32, np.float64].
                          Must not be empty, and must not contain infinities, NaNs, or
                          (if dtype if np.float64), numbers so large that if converted to
                          np.float32 they would become infinity.
             axis (int):  The axis of `sequence_data` that corresponds to the time
                          dimension.  (Does not have to really correspond to time, but this
                          is the dimension we do linear prediction in.) . -1 mean the last
                          axis.
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
       out                The user may pass in numpy.ndarray with dtype=np.int8, of the same
                          dimension as the output of this function would have been.
                          In that case, the output will be placed here.  If an array
                          of the wrong dimension is passed, ValueError will be raised.

       Returns:
           On success, returns a numpy.ndarray with dtype=np.int8, and with
           shape the same as `data` except the dimension on the axis numbered
           `axis` will have been increased by 4.  This can be decompressed by
           calling lilcom.decompress().

       Raises:
           TypeError if one of the arguments had the wrong type
           ValueError if an argument was out of range, e.g. invalid `axis` or
              `lpc_order` or an input array with no elements.
    """
    pass


def compress(inputSignal):
    """
    Lossily compresses 'inputSignal', samples of sequence data;
    @param [in] inputSignal The given sequence data given in form of
                    a numpy array
    @param [out] A numpy array storing the compressed sequence.

    @exception: Drops an exception in case that the dimension of given numpy
                    array is not correct.

    """

    # Steps to do so:
    #   Determining whether the numpy array is made of integer for floats
    intMode = True
    if inputSignal.dtype in ['float64', 'float32', 'float16']:
        intMode = False
    #   Vectorizing the numpy array and finding appropriate strides
    inputSignal_vectorized = inputSignal.flatten()
    numSamples = inputSignal.shape[0]
    try:
        inputStride = inputSignal.shape[1]
    except:
        inputStride = 1

    #   Calling the C function and receiving the result
    #   Converting result to a new numpy array.



    if intMode:
        pass
    else:
        pass



    pass


def decompress(inputSignal):
    """
    Uncompresses 'inputSignal', samples of sequence data;
    @param [in] inputSignal The given sequence data given in form of
                    a numpy array
    @param [out] A numpy array storing the compressed sequence.

    @exception: Drops an exception in case that the dimension of given numpy
                    array is not correct.
    """


    # Steps to do so:
    #   Determining whether the numpy array is made of integer for floats
    intMode = True
    if inputSignal.dtype in ['float64', 'float32', 'float16']:
        intMode = False
    #   Vectorizing the numpy array and finding appropriate strides
    inputSignal_vectorized = inputSignal.flatten()
    numSamples = inputSignal.shape[0]
    try:
        inputStride = inputSignal.shape[1]
    except:
        inputStride = 1

    #   Calling the C function and receiving the result
    #   Converting result to a new numpy array.


    if intMode:
        pass
    else:
        pass
    pass

