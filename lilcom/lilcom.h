#include <stdint.h>
#include <sys/types.h>


typedef void* LilcomConfigT;

/**
   Returns an opaque type which defines a set of configuration values.  The type
   will contain other values other than the ones explicitly provided, but for
   now these will be set to some appropriate default values and are not user
   specifiable.
*/
LilcomConfigT* lilcom_get_config(int samp_rate,
                                 int num_channels,
                                 int lpc_order,
                                 int max_bits_per_sample,
                                 int block_size);

void lilcom_delete_config(LilcomConfigT *config);



/**
   Lossily compresses 'num_samples' samples of int16 sequence data (e.g. audio
   data) into an array of bytes

      @param [in] input   The 16-bit input sequence data: a pointer
                      to an array of size at least `num_samples`.
      @param [in] num_samples  The number of samples of sequence
                      data to compress; must be greater than zero.
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.
      @param [out] output   The 8-bit compressed data:  a pointer
                      to an array with num_bytes elements.
                      Note: the header will not contain the length of the
                      sequence; the length of the byte sequence will be
                      required to work out the original `num_samples`.
      @param [in] output_stride  The offset from one output sample to
                      the next.  Would normally be 1, but may have any nonzero
                      value.
      @param [in] lpc_order  The order of linear prediction to use.
                      Must be in [0..14] (c.f. MAX_LPC_ORDER in lilcom.c).
                      Larger values will give higher fidelity (especially
                      for audio data) but the compression and decompression
                      will be slower.  For audio we recommend 4, since the gains
                      in fidelity for values above that will be small.
      @param [in] bits_per_sample  The number of bits per compressed sample;
                      must be in [4..8].  We normally recommend 8.
      @param [in] conversion_exponent  A user-specified number which
                      is required to be in the range [-127, 128] and which will
                      be returned to the user by lilcom_decompress().
                      Just set this to 0 if you will be dealing only with
                      integers.  This affects the scaling of the output if we
                      decompress to float, and is the mechanism by which the
                      lilcom_compress_float() and lilcom_decompress_float() get
                      the data to the right dynamic range.  When we convert to
                      float, we will cast the int16_t to float (or double, if
                      necessary due to potential overflow or underflow) and then
                      multiply by 2 to the power (conversion_exponent - 15).  If
                      you plan to convert back to int16_t you can choose any
                      value.  Setting conversion_exponent = 0 will put the data
                      in the range [-1,1] if you ever do want to convert to
                      float; this value might be suitable for audio data.

      @return  Returns:
                 0 on success
                 1 on failure (only possible failure is if one of the
                    caller-specified parameters had a disallowed value,
                    e.g. lpc_order or conversion_exponent out of range,
                    or num_samples <= 0.

   This process can (approximately) be reversed by calling `lilcom_decompress`.
*/



int lilcom_compress(const int16_t *input, ssize_t num_samples, int input_stride,
                    int8_t *output, ssize_t num_bytes, int output_stride,
                    int lpc_order, int bits_per_sample,
                    int conversion_exponent);
/**
int8_t *lilcom_compress_int16(const int16_t *input,
                              LilcomConfigT *config,
                              ssize_t num_samples,
                              int sample_stride,
                              int channel_stride,
                              int8_t **output); */

/**
   Lossily compresses 'num_samples' samples of floating-point sequence data
   (e.g. audio data) into an array of int8_t.  Internally it converts the data
   into int16_t using an appropriate power-of-two scaling factor and then calls
   lilcom_compress().

      @param [in] input   The floating-point input sequence data: a pointer
                      to an array with at least `num_samples` elements
                      and with stride `input_stride`.  If it contains
                      infinities, an error return code will be generated and the
                      data won't be compressed.  If it contains NaN's,
                      the behavior is undefined and this code may crash.
      @param [in] num_samples  The number of samples of floating-point
                      data.  Must be greater than zero.
      @param [in] input_stride  The offset from one input sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [out] output   The 8-bit compresed data:
                      a pointer to an array with `num_bytes` elements.
      @param [in] num_bytes  The number of bytes in the output compressed
                      data; this is purely as a sanity check, since it is
                      required to equal `lilcom_get_num_bytes(num_samples,
                      bits_per_sample)`.
      @param [in] output_stride  The offset from one output element to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value, but this might not
                      be checked.
      @param [in] lpc_order  The order of linear prediction to use.
                      Must be in [0..15] (see MAX_LPC_ORDER in lilcom.c).
                      Larger values will give higher fidelity (especially
                      for audio data) but the compression and decompression
                      will be slower.
      @param [in] bits_per_sample  The number of bits per sample; must be
                      in [4..8].  We normally recommend 8.
      @param [in] temp_space  A pointer to a temporary array of `num_samples`
                      int16_t's that can be used inside this function.  If NULL
                      is provided, this function will allocate one (you can
                      provide one in order to avoid allocations and
                      deallocations).

      @return         Returns:
                        0  on success
                        1  if it failed because num_samples, input_stride,
                           output_stride, num_bytes or lpc_order had an
                           invalid value.
                        2  if there were infinitites or NaN's in the input data.
                        3  if it failed to allocate a temporary array (only
                           possible if you did not provide one).

   This process can (approximately) be reversed by calling
   `lilcom_decompress_float` or `lilcom_decompress_double`.
 */
int lilcom_compress_float(
    const float *input, ssize_t num_samples, int input_stride,
    int8_t *output, ssize_t num_bytes, int output_stride,
    int lpc_order, int bits_per_sample, int16_t *temp_space);

/**
   Returns the number of samples in the signal that was compressed
   to this sequence of bytes.
      @param [in] input  Pointer to the start of the compressed data;
                      would correspond to the `output` argument to
                      lilcom_compress() or lilcom_compress_float().
      @param [in] num_bytes  Length in bytes
                      of the compressed input (note: this is necessary to find
                      out the number of samples, as the number of samples is not
                      directly stored in the header.)
      @param [in] input_stride  Stride of the input array (would
                      normally be 1.)

      @return  Returns -1 if, from the information provided,
             this cannot correspond to lilcom-compressed
             data (e.g. num_bytes is less than 5 or the header is
             invalid).
               Otherwise returns the number of samples that the
             data that was compressed contained, corresponding
             to the num_samples argument to lilcom_compress or
             lilcom_compress_float.
 */
ssize_t lilcom_get_num_samples(const int8_t *input,
                               ssize_t num_bytes,
                               int input_stride);

/**
   Uncompress a sequence of data that was previously compressed by
   lilcom_compress().

      @param [in] input   The 8-bit compressed data:  a pointer
                      to an array of size, let's say,
                      length_of_input_array, which is required to equal
                      lilcom_get_num_bytes(num_samples, bits_per_sample)
                      with the bits_per_sample obtained from the header
                      information.  If not it is an error and this
                      function will return 1.
      @param [in] num_bytes  The number of bytes in the compressed
                     stream
      @param [in] input_stride  The offset from one input sample
                      to the next; may have any nonzero value.
      @param [out] output   An array of size num_samples.
                      The decompressed data will be written to here, on success;
                      on failure, the contents are undefined.
      @param [in] num_samples  The number of samples in the `output` array.
                      If this is not equal to
                      `lilcom_get_num_samples(num_bytes, input, input_stride)`,
                      it is an error and this function will return error
                      status (1).
      @param [in] output_stride  The offset from one output sample to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.
      @param [out] conversion_exponent
                      This will be set to the value in the range [-125,120]
                      which was passed into the original call to
                      lilcom_compress() via its 'conversion_exponent' parameter
                      (it is obtained from the header).  It is useful in cases
                      when we are actually compressing a sequence of
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
int lilcom_decompress(const int8_t *input, ssize_t num_bytes, int input_stride,
                      int16_t *output, ssize_t num_samples, int output_stride,
                      int *conversion_exponent);

/**
   Uncompress a compressed stream to float.  If this was previously compressed
   by lilcom_compress_float(), the output will be a close approximation to the
   original.  If this stream was previously compressed by licom_compress() from
   16-bit input, the output range will depend on the `conversion_exponent`
   argument; the default of 0 will give output scaled such that [-32768,32768]
   maps to [-1,1].

      @param [in] input   The compressed data: a pointer
                      to an array of size `num_bytes`.
      @param [in] num_bytes   The number of bytes in the input compressed
                     sequence
      @param [in] input_stride  The offset from one input byte to
                      the next, in elements.  Would normally be 1.
                      May have any nonzero value.
      @param [out] output  The start of the array where the output data
                      will be placed, with `num_samples` elements.
      @param [in] num_samples  The number of samples in the `output` array.
                      If this is not equal to
                      `lilcom_get_num_samples(num_bytes, input, input_stride)`,
                      it is an error and this function will return error
                      status (1).
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
int lilcom_decompress_float(
    const int8_t *input, ssize_t num_bytes, int input_stride,
    float *output, ssize_t num_samples, int output_stride);


