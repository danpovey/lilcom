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

  /* The width for t == -1 will be written to here by this
     object whenever needed. */
  int8_t *width_m1;

  /* width is a rolling buffer of the number of bits present
     in the mantissa (before any truncation by bits_per_sample - 1).
     We use it for a couple of different purposes.
     For t < num_samples_success, it contains the number of bits
     used; for num_samples_success <= t <= most_recent_attempt,
     it contains the minimum usable widths for those times.
     (would be present during backtracking.)
  */
  int width[WIDTH_BUFFER_SIZE];
  /* `most_recent_attempt` records the most recent t value for which the user
     has so far called `backtracking_encoder_encode()`.
  */
  ssize_t most_recent_attempt;
  /*
     next_sample_to_encode is the next sample-index for which the user is
     required to call backtracking_encoder_encode().  This may go
     backwards as well as forwards, but it will never be the case
     that
       most_recent_attempt - next_sample_to_encode > (2*MAX_POSSIBLE_WIDTH + 1 == 31).
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
   Attempts to lossily compress `value` (which may be any int32_t).  [If a
   signed integer with max_bits_in_sample-1 bits is large enough to encode
   `value`, though, it will be exact].

      @param [in] max_bits_in_sample  The maximum number of bits that the
                           encoder is allowed to use to encode this sample,
                           INCLUDING THE WIDTH BIT.  Must be in [4,32].  The
                           max_bits_in_sample does not have to be the same from
                           sample to sample, but the sequence of
                           max_bits_in_sample values this is called with must be
                           the same when decoding as when encoding.
      @param [in] value    The value to be encoded.  May have any value that
                           fits in a 31-bit int, i.e. in [-2^30 .. 2^30 - 1. ]
      @param [out] decoded_value  On success (i.e. if this function returns 0),
                           the encoded-then-decoded version of `value` will be
                           written to here (it may differ, due to compression).
                           Otherwise its value at exit is undefined.

      This function requires that encoder->next_sample_to_encode >= 0.
      (which it always would be.)

      @return  Returns 0 on success, 1 on failure.  Success means
             a code was created and encoder->next_sample_to_encode
             was increased by 1, and the code was written to the underlying
             bit-packer.  On failure, encoder->next_sample_to_encode will be
             decreased by up to max_bits_in_sample + 1.  [Note: this type of
             failure happens in the normal course of compression, it is part of
             backtracking.]

     See also decoder_decode(), which you can think of as the reverse
     of this.
 */
static inline int backtracking_encoder_encode(int max_bits_in_sample,
                                              int32_t value,
                                              struct BacktrackingEncoder *encoder,
                                              int32_t *decoded_value);

/*
  This function takes the code point at time t = encoder->next_sample_to_encode
  - 1, regresses it by one towards zero, and puts the modified decoded output
  in *decoded_value.  What we mean by `regresses it toward zero` is, if the
  mantissa (i.e. all but the exponent bit) is m, doing

     m += (m >= 0 ? -1 : 1).

  This is done in cases where we are encoding int16's and the predicted value
  plus the rounded residual would exceed the range of int16_t, so we need
  to make it closer to zero.
  This function requires that the sample for this time was encoded with at
  least 2 bits (and the calling code does, naturally, respect this limit).

  This function does not change encoder->next_sample_to_encode.

    @param [in,out] encoder   The encoder object in which we are
                  changing the most recently written sample.
    @param [out] decoded_value  The new encoded-and-then-decoded value,
               as in the `decoded_value` output of `backtracking_encoder_encode()`.
 */
static inline void backtracking_encoder_regress_most_recent_t(
    struct BacktrackingEncoder *encoder,
    int32_t *decoded_value);



/* View this as the reverse of struct BacktrackingDecoder.
   It interprets the encoded widths and mantissas as 32-bit
   numbers.  (It's very simple, actually; it just needs to
   keep track of the width.)

   See functions decoder_init(), decoder_decode(), and
   decoder_finish().
*/
struct Decoder {
  struct BitUnpacker bit_unpacker;

  int num_bits;  /* num_bits is the number of bits in the signed integer for
                    the current sample, excluding the width bit. */
};


/**
   Initialize the decoder object.

        @param [in]  num_samples_to_read  The number of samples that
                     will be read from this decoder object (used only for
                     checking)
        @param [in]  compressed_code  Pointer to the first byte of the
                     compressed code to be decoded
        @param [in]  compressed_code_stride  Stride between elements of
                     `compressed_code`.  [TODO: this will alter be removed.]
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
                         2 helps determine the width).
           @param [in] max_encoded_mantissa_bits  The maximum number of bits
                         that we will use for this sample (excluding the 1-bit
                         width).
           @param [in,out] decoder  The decoder object
           @param [out]  value  The decoded value will be written here.

           @return     Returns 0 on success, 1 on failure.  (Failure
                       can happen if the width goes out of range
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
#endif


#endif  /* LILCOM_ENCODER_H_INCLUDED_ */
