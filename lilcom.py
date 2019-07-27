import numpy

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
    #   Vectorizing the numpy array and finding appropriate strides
    #   Calling the C function and receiving the result
    #   Converting result to a new numpy array.

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
    #   Vectorizing the numpy array and finding appropriate strides
    #   Calling the C function and receiving the result
    #   Converting result to a new numpy array.

    pass

