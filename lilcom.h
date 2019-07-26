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
      @return  Returns 0 on success, 1 on failure.  The only failure mode
                     is if one of the caller-specified parameters had a
                     disallowed value.

   This process can (approximately) be reversed by calling `lilcom_decompress`.
*/
int lilcom_compress(int64_t num_samples,
                    const int16_t *input, int input_stride,
                    int8_t *output, int output_stride,
                    int lpc_order);

/**
   Lossily compresses 'num_samples' samples of floating-point sequence data
   (e.g. audio data) into 'num_samples + 12' bytes.  (A 12-byte header
   consisting of two floats, then the same header as `lilcom_compress` uses,
   will be created).  It will first turn this data into
   int16's using a linear encoding going from the minimum to the maximum
   values seen in the sequence, and then compress that sequence using
   `lilcom_compress()`.

      @param [in] num_samples  The number of samples of floating-point
                      data.  Must be greater than zero.
      @param [in] input   The floating-point input sequence data: a pointer
                      to an array of size at least `num_samples`.  Must be
                      free of NaN or infinity.
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] input   The 8-bit compresed data:  a pointer
                      to an array of size at least `num_samples + 12`,
                      where the extra 12 bytes form a header.  Note:
                      the header does not contain the length of the
                      sequence, that is assumed to be known
                      externally (e.g. from the file length or
                      the dimension of the matrix).
      @param [out] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.

   This process can (approximately) be reversed by calling
   `lilcom_decompress_float`.
 */
void lilcom_compress_float(int64_t num_samples,
                           const float *input, int input_stride,
                           int8_t *output, int output_stride);


/**
   Uncompress 'num_samples' samples of sequence data that was previously
   compressed by lilcom_compress().

      @param [in] input   The 16-bit input sequence data: a pointer
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
      @param [out] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
*/
int lilcom_decompress(int64_t num_samples,
                      const int8_t *input, int input_stride,
                      int16_t *output, int output_stride);


int lilcom_decompress_float(int64_t num_samples,
                            const int8_t *input, int input_stride,
                            int16_t *output, int output_stride);


