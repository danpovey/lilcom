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
                      is required to be in the range [-128, 127] and
                      which will be returned to the user by
                      lilcom_decompress().  This affects the scaling
                      of the output if we decompress to float, and is
                      the mechanism by which the lilcom_compress_float()
                      and lilcom_decompress_float() get the data to
                      the right dynamic range.  When we convert to
                      float, we will cast the int16_t to float
                      and then multiply by 2^conversion_exponent.
                      If you plan to convert back to int16_t you can
                      choose any value: for instance, -15 will
                      put the data in the range [-1,1] if you ever
                      do want to convert to float, so this might be
                      suitable for audio data.

      @return  Returns 0 on success, 1 on failure.  The only failure mode
                     is if one of the caller-specified parameters had a
                     disallowed value.

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
                      and with stride `input_stride`.   Must not have
                      NaNs or infinities (if it does, an error return
                      code will be generated and it won't be compressed).
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
      @param [out] output_stride  The offset from one output element to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @return         Returns 0 on success, 1 on failure.  The only
                      failure mode is if there are infinitites or NaN's
                      in the input data.

   This process can (approximately) be reversed by calling
   `lilcom_decompress_float` or `lilcom_decompress_double`.
 */
int lilcom_compress_float(int64_t num_samples,
                          const float *input, int input_stride,
                          int8_t *output, int output_stride);

/** This is like lilcom_compress_float, but for double-precision input.  Note:
    you can decompress either as float or as double (or as int16 if you like,
    but you might want to remember the conversion_exponent.)

    This function returns 1 on failure like lilcom_compress_float, but it has an
    additional failure mode, which can happen when there were input samples
    which were finite but with absolute value greater than or equal to 2^143.
 */
int lilcom_compress_double(int64_t num_samples,
                           const double *input, int input_stride,
                           int8_t *output, int output_stride);


/**
   Uncompress 'num_samples' samples of sequence data that was previously
   compressed by lilcom_compress().

      @param [in] input   The 8-bit compressed data: a pointer to
                      an array of size `num_samples


      16-bit input sequence data: a pointer
                      to an array of size at least `num_samples`
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] input   The 8-bit compresed data:  a pointer
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
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] input   The 8-bit compresed data:  a pointer
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
*/

int lilcom_decompress_float(int64_t num_samples,
                            const int8_t *input, int input_stride,
                            float *output, int output_stride);


