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
   Initializes the backtracking-encoder object.    After this you will
   want to repeatedly call backtracking_encoder_encode().

     @param [in] bits_per_sample   User-specified configuration value
                  in the range [LILCOM_MIN_BPS .. LILCOM_MAX_BPS],
                  currently [4..16]
     @param [in,out] nbits_m1   Address to write the exponent
                  for time t == -1 (this allows it to initialize
                  the sequence of exponents).  This address is
                  stored inside the encoder and its contents will be
                  modified as needed.
     @param [out] encoder  The encoder to be initialized

 */
void backtracking_encoder_init(int8_t *nbits_m1,
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
                           encoder is allowed to use to encode this sample.
                           Must be at least 4.  The max_bits_in_sample does
                           not have to be the same from sample to sample,
                           but the sequence of max_bits_in_sample values
                           this is called with must be the same as in
                           training time.
      @param [in] residual The value to be encoded for time
                           encoder->next_sample_to_encode.  Required to be
                           in the range [-(2**16-1) .. 2**16-1].
      @param [in] predicted The predicted value from which this residual is
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
      @param [in,out] encoder  The encoder object.  May be modified by
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
                                              int32_t residual,
                                              int16_t predicted,
                                              int16_t *next_value,
                                              struct BacktrackingEncoder *encoder,
                                              struct BitPacker *packer);


struct Decoder {
  /* View this as the reverse of struct BacktrackingDecoder.
     It interprets the encoded exponents and mantissas as 32-bit
     numbers.  (It's very simple, actually; it just needs to
     keep track of the exponent.)
     See functions decoder_init() and decoder_decode().
   */
  int bits_per_sample;
  int exponent;
};


/**
   Initialize the decoder object.
        @param [in] bits_per_sample   The number of bits per sample
                 (user-specified but stored in the compressed header).
                 Must be in [4..8].
        @param [in] nbits_m1  The exponent for time t == -1,
                 as stored in the compressed file (you could view
                 this as part of the header).
        @param [out] decoder  The object to be initialized.
 */
void decoder_init(int bits_per_sample,
                  int nbits_m1,
                  struct Decoder *decoder);


/**
   Converts one sample's code into a signed int32 value (which will typically represent
   a residual).  Must be called exactly in sequence for t = 0, t = 1 and so on.

           @param [in] unpacker  The BitUnpacker object from which to
                       read codes.  (Actually we read exponents and mantissas
                       separately because we need the exponent to know the
                       size of the mantissa.)

           code  The encoded value; only the lower-order
                          `decoder->bits_per_sample` bits will be inspected.
           @param [in,out] decoder  The decoder object
           @param [out]  value  The decoded value will be written here.
           @return     Returns 0 on success, 1 on failure.  (Failure
                       can happen if the exponent goes out of range, and
                       would normally indicate data corruption or an error
                       in lilcom code.)
 */
static inline int decoder_decode(ssize_t t,
                                 struct BitUnpacker *unpacker,
                                 struct Decoder *decoder,
                                 int32_t *value);

#ifdef LILCOM_TEST
void lilcom_test_extract_mantissa();
void lilcom_test_encode_decode_signed();
void lilcom_test_encode_residual();
#endif


#endif  /* LILCOM_ENCODER_H_INCLUDED_ */
