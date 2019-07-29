import numpy
from ctypes import *


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

