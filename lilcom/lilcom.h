#include <stdint.h>


/**
   Lossily compresses 'num_samples' samples of int16 sequence data (e.g. audio
   data) into 'num_samples + 4' bytes.

      @param [in] num_samples  The number of samples of sequence
                      data.  Mus be greater than zero.
      @param [in] input   The 16-bit input sequence data: a pointer
                      to an array of size at least `num_samples`
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [out] output   The 8-bit compresed data:  a pointer
                      to an array of size at least `num_samples + 4`,
                      where the extra 4 bytes form a header.  Note:
                      the header does not contain the length of the
                      sequence, that is assumed to be known
                      externally (e.g. from the file length or
                      the dimension of the matrix).
      @param [in] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] lpc_order  The order of linear prediction to use.
                      Must be in [0..15] (see MAX_LPC_ORDER in lilcom.c).
                      Larger values will give higher fidelity (especially
                      for audio data) but the compression and decompression
                      will be slower.
      @param [in] conversion_exponent  A user-specified number which
                      is required to be in the range [-127, 128] and which will
                      be returned to the user by lilcom_decompress().  This
                      affects the scaling of the output if we decompress to
                      float, and is the mechanism by which the
                      lilcom_compress_float() and lilcom_decompress_float() get
                      the data to the right dynamic range.  When we convert to
                      float, we will cast the int16_t to float (or double, if
                      necessary due to potential overflow or underflow) and then
                      multiply by 2 to the power (conversion_exponent - 15).  If
                      you plan to convert back to int16_t you can choose any value: for
                      instance, a value of conversion_exponent = 0 will put the
                      data in the range [-1,1] if you ever do want to convert to
                      float; this value might be suitable for audio data.

      @return  Returns:
                 0 on success
                 1 on failure (only possible failure is if one of the
                    caller-specified parameters had a disallowed value,
                    e.g. lpc_order or conversion_exponent out of range,
                    or num_samples <= 0.

   This process can (approximately) be reversed by calling `lilcom_decompress`.
*/
int lilcom_compress(int64_t num_samples,
                    const int16_t *input, int input_stride,
                    int8_t *output, int output_stride,
                    int lpc_order, int conversion_exponent);

/**
   Lossily compresses 'num_samples' samples of floating-point sequence data
   (e.g. audio data) into 'num_samples + 4' bytes of int8_t.  Internally it
   converts the data into int16_t using an appropriate power-of-two scaling
   factor and then calls lilcom_compress().  Note: this is mostly provided for
   people who use this as a "C" library; there may be faster or better ways to
   implement this from Python if we are calling this from NumPy.

      @param [in] num_samples  The number of samples of floating-point
                      data.  Must be greater than zero.
      @param [in] input   The floating-point input sequence data: a pointer
                      to an array with at least `num_samples` elements
                      and with stride `input_stride`.  If it contains
                      infinities, an error return code will be generated and the
                      data won't be compressed.  If it contains NaN's,
                      the behavior is undefined and this code may crash.
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [out] output   The 8-bit compresed data:  a pointer
                      to an array with at least `num_samples + 4`
                      elements and stride `output_stride`.  Note:
                      the header does not contain the length of the sequence;
                      that is assumed to be known externally (e.g. from the file
                      length or the dimension of the matrix).
      @param [in] output_stride  The offset from one output element to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] lpc_order  The order of linear prediction to use.
                      Must be in [0..15] (see MAX_LPC_ORDER in lilcom.c).
                      Larger values will give higher fidelity (especially
                      for audio data) but the compression and decompression
                      will be slower.
      @param [in] temp_space  A pointer to a temporary array of int16_t that
                      can be used inside this function.  It must have size
                      at least `num_samples`.  If NULL is provided, this
                      function will allocate one (you can provide one in
                      order to avoid allocations and deallocations).

      @return         Returns:
                        0  on success
                        1  if it failed because num_samples, input_stride,
                           output_stride or lpc_order had an invalid value.
                        2  if there were infinitites or NaN's in the input data.
                        3  if it failed to allocate a temporary array (only
                           possible if you did not provide one).

   This process can (approximately) be reversed by calling
   `lilcom_decompress_float` or `lilcom_decompress_double`.
 */
int lilcom_compress_float(int64_t num_samples,
                          const float *input, int input_stride,
                          int8_t *output, int output_stride,
                          int lpc_order, int16_t *temp_space);



/**
   Uncompress 'num_samples' samples of sequence data that was previously
   compressed by lilcom_compress().

      @param [in] input   The 8-bit compresed data:  a pointer
                      to an array of size at least `num_samples + 4`,
                      where the extra 4 bytes form a header.  Note:
                      the header does not contain the length of the
                      sequence, that is assumed to be known
                      externally (e.g. from the file length or
                      the dimension of the matrix).
      @param [in] input_stride  The offset from one input sample
                      to the next; may have any nonzero value.
      @param [out] output     The decompressed data will be
                      written to here, on success; on failure, the
                      contents are undefined.
      @param [in] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.
      @param [out] conversion_exponent
                      This will be set to the value in the range [-125,120]
                      which was passed into the original call to
                      lilcom_compress() via its 'conversion_exponent' parameter.
                      It is for use when we are actually compressing a sequence of
                      floating-point numbers, to set the appropriate scale for
                      integerization.
      @return      Returns:
                      0 on success
                      1 on failure
                        Failure modes include invalid num_samples, input_stride
                        or output_stride, or that the input data was not
                        generated by lilcom_compress, or that it was corrupted,
                        or-- we hope not!-- a bug in the code.
*/
int lilcom_decompress(int64_t num_samples,
                      const int8_t *input, int input_stride,
                      int16_t *output, int output_stride,
                      int *conversion_exponent);

/**
   Uncompress 'num_samples' samples of sequence data that was previously
   compressed by lilcom_compress_float() or lilcom_compress_double().

      @param [in] input   The flo input sequence data: a pointer
                      to an array of size at least `num_samples`
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.
      @param [in] input   The 8-bit compresed data:  a pointer
                      to an array of size at least `num_samples + 4`,
                      where the extra 4 bytes form a header.  Note:
                      the header does not contain the length of the
                      sequence, that is assumed to be known
                      externally (e.g. from the file length or
                      the dimension of the matrix).
      @param [in] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.

      @return    Returns
                     0 on success
                     1 on failure
                        Failure modes include invalid num_samples, input_stride
                        or output_stride, or that the input data was not
                        generated by lilcom_compress, or that it was corrupted,
                        or-- we hope not!-- a bug in the code.
*/
int lilcom_decompress_float(int64_t num_samples,
                            const int8_t *input, int input_stride,
                            float *output, int output_stride);


