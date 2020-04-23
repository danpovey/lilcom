#ifndef __LILCOM_COMPRESSION_H__
#define __LILCOM_COMPRESSION_H__ 1

#include <stdint.h>
#include <sys/types.h>
#include "int_stream.h"


/**
   This header provides a C++ interface to lilom's audio-compression algorithm.
   We'll further wrap this in plain "C" for ease of Python-wrapping.
*/


/*
  Implementation of lossy compression of a possibly multi-dimensional
  array of floats.

    @param [in] tick_power  Power that we'll take 2 to to give the discretization 
                   interval; no error due discretization will exceed half of this 
		   value (unless the absolute value of an input element exceeds
		   2^(32 - tick_power), in which case we'll get more error
		   due to range limitation.  Must be in range [-20, 20]
    @param [in] data   Pointer to start of the input data.  CAUTION: this
                   data is changed by being replaced by the compressed version;
		   view it as being destructively consumed.  (This is necessary
		   because of how the regression works; we regress on the
    		   compressed versions).
    @param [in] num_axes  The number of axes in `data`; must be >0.
    @param [in] dims   The dimension of each axis i is given by dim[i].
    @param [in] strides  The stride on each axis i is given by strides[i];
                   these are strides in float elements, not bytes.
    @param [in] regression_coeffs  Integerized regression coefficients, 
                   one per axis, such that (using the 3-axis case as
                   an example, and assuming all indexes are nonzero,
                   and using Python-style indexing as if `data` were a
                   NumPy array), we are writing the elements of the
                   array as:
                      data[i,j,k] = data[i-1,j,k]*regression_coeffs[0]*a + 
                                    data[i,j-1,k]*regression_coeffs[1]*a +
                                    data[i,j,k-1]*regression_coeffs[2]*a +
                                    offset
                   where a == (1/256) to convert integer into fractional
                   values, and `offset` is what we encode.  The idea is that
                   the regression coefficients are estimated to minimize
                   the sum-of-squares of these offset values.  We always
                   regressed on the previously compressed version of the data.
		   Out of range values (for i, j or k == 0) are treated
		   as zero.

    @return  Returns a vector of bytes representing the compressed data.  Certain
            error conditions (generally: code errors in calling code)
            will cause it to return an empty vector.
 */
std::vector<char> CompressFloat(int tick_power,
                                float *data, 
                                int num_axes, 
                                const int *dims, 
                                const int *strides,
                                const int *regression_coeffs);
			  

/*
  This function gets the shape of an array that has been compressed by
  CompressFloat().
     @param [in] data   Start of the compressed data
     @param [in] num_bytes  The number of bytes in the array `data`
     @param [out] meta   Pointer to an array where some meta-information
                         will be stored (the size of the array must be at
                         least 17).  On successful exit it will contain:
  			 { num_axes, dim1, dim2, dim3, ... }
      @return   Returns true on success, false on error (e.g. data did not seem
               to be valid).
 */
bool GetCompressedDataShape(const char *data,
                            int num_bytes,
                            int *meta);

/*
  Decompresses data that was compressed by CompressFloat().
      @param [in] src     Start of the compressed data
      @param [in] num_bytes   Number of bytes in the compressed data (note:
                         must exactly match the length of the string
                         originally returned by CompressFloat()
      @param [out] array  Start of the array to which we are writing
      @param [in] num_axes Number of axes of `array`; must
                          be in range [1..16].
      @param [in] dims    Dimensions of each axis of `array`; must
                          match the dimensions returned by
                          GetCompressedDataSize() on `src`.
      @param [in] strides Strides of each axis of `array`, in
                          floats (not bytes).
      @return       Returns zero on success, otherwise various
                    nonzero error codes (see code for meanings)
 */
int DecompressFloat(const char *src,
		    int num_bytes,
		    float *data, 
		    int num_axes, 
		    const int *dims, 
		    const int *strides);




#endif /* __LILCOM_COMPRESSION_H__ */

