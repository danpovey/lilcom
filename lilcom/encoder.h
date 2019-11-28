#ifndef LILCOM_ENCODER_H_INCLUDED_
#define LILCOM_ENCODER_H_INCLUDED_

#include <stdint.h>
#include <sys/types.h>
#include <assert.h>
#include <stdio.h>
#include "lilcom_common.h"
#include "bit_packer.h"


/**
   This header contains declarations for some compression code used in
   lilcom; it is an attempt to partition away some of the complexity.
 */
struct BacktrackingEncoder {
  /*
    If this were a class it would have members:
     backtracking_encoder_init() == constructor
     backtracking_encoder_encode()
   See documentation of those functions for more information.
   Also see struct Decoder, which is the reverse of this
   (allows you to extract the approximately-encoded residuals.)
  */


  struct BitPacker bit_packer;

  /* bits_per_sample is a user-supplied configuration value in [4..8]. */
  int bits_per_sample;

  /* The exponent for t == -1 will be written to here by this
     object whenever needed. */
  int8_t *nbits_m1;

  /* nbits is a rolling buffer of the number of bits present
     in the mantissa (before any truncation by bits_per_sample - 1).
     We use it for a couple of different purposes.
     For t < num_samples_success, it contains the number of bits
     used; for num_samples_success <= t <= most_recent_attempt,
     it contains the minimum usable exponents for those times.
     (would be present during backtracking.)
  */
  int nbits[NBITS_BUFFER_SIZE];
  /* `most_recent_attempt` records the most recent t value for which the user
     has so far called `backtracking_encoder_encode()`.
  */
  ssize_t most_recent_attempt;
  /*
     next_sample_to_encode is the next sample for which the user is
     required to call backtracking_encoder_encode().  This may go
     backwards as well as forwards, but it will never be the case
     that
       most_recent_attempt - next_sample_to_encode > (2*MAX_POSSIBLE_NBITS + 1 == 31).
  */
  ssize_t next_sample_to_encode;

  /** num_backtracks is only used in debug mode. */
  ssize_t num_backtracks;
};




/**
   Initializes the backtracking-encoder object.  After this you will want to
   repeatedly call either backtracking_encoder_encode() or
   backtracking_encoder_encode_limited(), with the data for index
   encoder->next_sample_to_encode, until encoder->next_sample_to_encode equals
   encoder->num_samples_to_write.

     @param [in] num_samples_to_write  The number of samples that
                  will be written to this stream.  (This is just
                  used for checks.)
     @param [in] compressed_code_start  Pointer to where the
                  first bit of the compressed code will be written.
     @param [in] compressed_code_stride  Spacing between elements of
                   compressed code; will normally be 1.  Must be nonzero.
     @param [out] encoder  The encoder object to be initialized

 */
void backtracking_encoder_init(ssize_t num_samples_to_write,
                               int8_t *compressed_code_start,
                               int compressed_code_stride,
                               struct BacktrackingEncoder *encoder);


/**
   Flushes remaining samples from the bit-packer object owned by
   BacktrackingEncoder; to be called once you have called it for all samples.
   Assumes you have called backtracking_encoder_encode() until its
   next_sample_to_encode equals the number of samples you were going to write.

    @param [out] avg_bits_per_sample  The average number of bits written
                            per sample will be written to here.
    @param [out] next_free_byte  Points to one past the last element
                            written to (taking into account the stride,
                            of course.)
 */

void backtracking_encoder_finish(struct BacktrackingEncoder *encoder,
                                  float *avg_bits_per_sample,
                                  int8_t **next_free_byte);



/**
   Attempts to lossily compress `value` (which may be any int32_t).

      @param [in] value    The signed number to be encoded.  (Will be lossless
                           as long as it can be encoded (including a bit for the
                           sign) in `max_bits_in_sample` bits.
      @param [in] max_bits_in_sample  The maximum number of bits that the
                           encoder is allowed to use to encode this sample,
                           INCLUDING THE EXPONENT BIT.  Must be at least 4.  The
                           max_bits_in_sample does not have to be the same from
                           sample to sample, but the sequence of
                           max_bits_in_sample values this is called with must be
                           the same when decoding as when encoding.

      @param [out] next_value  On success, the input `predicted` plus
                           the approximated residual will be written to here.
      @param [out] code    On success, the code will be written to
                           here.  (Only the lowest-order encoder->bits_per_sample
                           bits will be relevant).  Note: later on we
                           might extend this codebase to support writing
                           fewer than the specified number of bits where
                           doing so would not affect the accuracy, by allowing
                           negative excursions of the exponent.  That
                           would require interface changes, though.
      @param [in,out] encoder  The encoder object.  Will be modified by
                           this call.

      Requires that encoder->next_sample_to_encode >= 0.

      @return  Returns 0 on success, 1 on failure.  Success means
             a code was created, encoder->next_sample_to_encode
             was increased by 1, and the code was written to `packer`.  On
             failure, encoder->next_sample_to_encode will be decreased (while
             still satisfying encoder->most_recent_attempt -
             encoder->next_sample_to_encode < (2*MAX_POSSIBLE_NBITS + 1)).
             [Note: this type of failure happens in the normal course of
             compression, it is part of backtracking.]

     See also decoder_decode(), which you can think of as the reverse
     of this.
 */
static inline int backtracking_encoder_encode(int max_bits_in_sample,
                                              int32_t value,
                                              struct BacktrackingEncoder *encoder);

/**
   Attempts to lossily compress `residual` (which is allowed to be in the range
   [-(2**16-1) .. 2**16-1]) to an 8-bit code.  The encoding scheme uses an
   exponent and a mantissa, and the exponent is stored as a single bit
   `delta_exponent` by dint of only storing the changes in the exponent,
   according to the formula:
       exponent(t) := exponent(t-1) + (t % 2) + delta_exponent(t)
   where delta_exponent in {0,1} is the only thing we encode.  When we
   discover that we can't encode a large enough exponent to encode a particular
   sample, we have to go back to previous samples and use a larger exponent for
   them.

      @param [in] max_bits_in_sample  The maximum number of bits that the
                           encoder is allowed to use to encode this sample,
                           INCLUDING THE EXPONENT BIT.  Must be at least 4.  The
                           max_bits_in_sample does not have to be the same from
                           sample to sample, but the sequence of
                           max_bits_in_sample values this is called with must be
                           the same when decoding as when encoding.

                           TODO: find upper limit in max_bits_in_sample.

      @param [in] residual The value to be encoded for time
                           encoder->next_sample_to_encode.  Required to be
                           in the range [-(2**16-1) .. 2**16-1].
      @param [in] value    The predicted value from which this residual is
                           an offset.  This is needed in order to ensure that
                           (predicted + *approx_residual) does not
                           overflow the range of int16_t.
      @param [out] next_value  On success, the input `predicted` plus
                           the approximated residual will be written to here.
      @param [out] code    On success, the code will be written to
                           here.  (Only the lowest-order encoder->bits_per_sample
                           bits will be relevant).  Note: later on we
                           might extend this codebase to support writing
                           fewer than the specified number of bits where
                           doing so would not affect the accuracy, by allowing
                           negative excursions of the exponent.  That
                           would require interface changes, though.
      @param [in,out] encoder  The encoder object.  Will be modified by
                           this call.

      Requires that encoder->next_sample_to_encode >= 0.

      @return  Returns 0 on success, 1 on failure.  Success means
             a code was created, encoder->next_sample_to_encode
             was increased by 1, and the code was written to `packer`.  On
             failure, encoder->next_sample_to_encode will be decreased (while
             still satisfying encoder->most_recent_attempt -
             encoder->next_sample_to_encode < (2*MAX_POSSIBLE_NBITS + 1)).
             [Note: this type of failure happens in the normal course of
             compression, it is part of backtracking.]

     See also decoder_decode(), which you can think of as the reverse
     of this; and backtracking_encoder_encode(), which is a simpler
     version of this without the logic to keep the result within
     int16_t.
 */
static inline int32_t backtracking_encoder_encode_limited(int max_bits_in_sample,
                                                          int32_t residual,
                                                          int16_t predicted,
                                                          int16_t *next_value,
                                                          struct BacktrackingEncoder *encoder);


/* View this as the reverse of struct BacktrackingDecoder.
   It interprets the encoded exponents and mantissas as 32-bit
   numbers.  (It's very simple, actually; it just needs to
   keep track of the exponent.)

   See functions decoder_init(), decoder_decode(), and
   decoder_finish().
*/
struct Decoder {
  struct BitUnpacker bit_unpacker;

  int num_bits;  /* num_bits is the number of bits in the signed integer for
                    the current sample, excluding the exponent bit. */
};


/**
   Initialize the decoder object.

        @param [out] decoder  The object to be initialized.
 */
void decoder_init(ssize_t num_samples_to_read,
                  const int8_t *compressed_code,
                  int compressed_code_stride,
                  struct Decoder *decoder);

/**
   Finishes use of the decoder object.
      @param [in] decoder   Decoder object that we are done with
      @param [out] next_compressed_code   The compressed-code point
              that's one past the end of the stream.  (Should mostly
              be needed for checking.)
 */
void decoder_finish(const struct Decoder *decoder,
                    const int8_t **next_compressed_code);


/**
   Converts one sample's code into a signed int32 value (which will typically represent
   a residual).  Must be called exactly in sequence for t = 0, t = 1 and so on.

           @param [in] t  The time index we are decoding (its value modulo
                         2 helps determine the exponent).
           @param [in] max_encoded_mantissa_bits  The maximum number of bits
                         that we will use for this sample (excluding the 1-bit
                         exponent).
           @param [in,out] decoder  The decoder object
           @param [out]  value  The decoded value will be written here.

           @return     Returns 0 on success, 1 on failure.  (Failure
                       can happen if the exponent goes out of range
                       or the decoded value exceeds the range of int16_t, and
                       would normally indicate data corruption or an error
                       in lilcom code.)
 */
static inline int decoder_decode(ssize_t t,
                                 int max_encoded_mantissa_bits,
                                 struct Decoder *decoder,
                                 int32_t *value);

#ifdef LILCOM_TEST
void lilcom_test_extract_mantissa();
void lilcom_test_encode_decode_signed();
void lilcom_test_encode_residual();
#endif


#endif  /* LILCOM_ENCODER_H_INCLUDED_ */
