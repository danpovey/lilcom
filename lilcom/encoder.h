#include <stdint.h>
#include <sys/types.h>
#include "lilcom_common.h"



/**
   This macro is added mainly for documentation purposes.  It clarifies what the
   possible exponents are for time t given that we know the exponent for time
   t-1.

       @param [in] t   The time for which we want to know the possible
                       exponent range.  t >= -1.
       @param [in] exponent_tm1   The exponent used for time t-1:
                       a value in [0..15], otherwise it is an error.
       @return min_exponent  The minimum exponent that time t
                       will be able to use, given exponent_tm1.
                       This equals exponent_tm1 - ((t+exponent_tm1)&1),
                       where &1 is a way of computing mod 2.
                       It forms a checkerboard pattern.  If you draw
                       the possible trajectories of exponents on
                       a grid you'll see that this choice has nice
                       properties that allows exponents to change
                       quite fast when needed.  The return value is a value
                       in [-1..15], but currently going to exponent
                       -1 is "forbidden", meaning the encoded bit for exponent
                       change is bound to be 1 in this case so the
                       exponent for time t would be zero.  We may later
                       make excursions to negative exponents allowed,
                       which would allow us to save and then re-use
                       bits, e.g. for on-disk formats.

   Note: the maximum exponent is just min_exponent + 1 where min_exponent is
   what this macro returns, because we use 1 bit for the change in exponent; a
   bit value of 1 means we choose the max, zero means we choose the min.
 */
#define LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, exponent_tm1) (exponent_tm1 - ((((int)t)+exponent_tm1)&1))

/**
   This macro is defined mainly for documentation purposes.  It returns the
   smallest value the exponent for time t-1 could be, given the constraint that
   we require the exponent for time t to be at least a specified value.

        @param [in]  t   The current time, t >= 0.  We want to compute an exponent-floor for
                         the preceding time (t-1).
        @param [in] exponent_t  An exponent for time t, where the caller says
                         they want an exponent at least this large.  Must be
                         in the range [0..MAX_POSSIBLE_NBITS].
        @return  Returns the smallest value that the exponent could have on
                         time t-1 such that the exponent on time t will be
                         at least exponent_t.  This may be -1 if exponent_t
                         was 0, but this macro should never be called in that
                         case.  So this function returns the maximum k such that
                         LILCOM_COMPUTE_MIN_CODABLE_COMPONENT(t, k) >= exponent_t.
                         The answer will obviously be either exponent_t or
                         exponent_t - 1.  We work out the formula as follows:

                         (a) First, assume (t+exponent_t) is even, so
                         ((((int)t)+exponent_t)&1) is zero.  Now we ask,
                         if exponent_tm1 = exponent_t - 1, what is
                         the greatest exponent that we can have on time t?
                         The minimum exponent we can have on time t is given by
                         LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, (exponent_t - 1))
                         and in that case we get an odd number in the modulus
                         so there is a -1 in the formula.  That would imply the
                         minimum exponent on time t is (exponent_t - 1) - 1 =
                         exponent_t - 2, so the maximum is exponent_t - 1
                         (since the bit adds one).  That's a no-go, i.e.
                         for even ((((int)t)+exponent_t)&1) the answer is
                         just exponent_t; for odd it's exponent_t - 1.
                         This happens to be exactly the same formula as
                         LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, exponent_t).
 */
#define LILCOM_COMPUTE_MIN_PRECEDING_EXPONENT(t, exponent_t) (exponent_t - ((((int)t)+exponent_t)&1))



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
   want to do
   backtracking_encoder_set_exponent_tm1() and then (repeatedly)
   backtracking_encoder_encode(), and possibly further calls to
   backtracking_encoder_set_exponent_tm1() if required.

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
static
void backtracking_encoder_init(int8_t *nbits_m1,
                               struct BacktrackingEncoder *encoder) {
  encoder->nbits_m1 = nbits_m1;
  encoder->nbits[NBITS_BUFFER_SIZE - 1] = (*nbits_m1 = 0);
  encoder->most_recent_attempt = -1;
  encoder->next_sample_to_encode = 0;
#ifndef NDEBUG
  encoder->num_backtracks = 0;
#endif
}



/*
  least_bits(residual, min_exponent):

  Returns least n >= min_exponent such that:
    -2^n <= value * 2 < 2^n

  Note: the "* 2" part of the formulation may seem odd, but:
  (1) It means that the number of bits can include the bit
  needed for the sign (2s complement encoding..), and
  (2) It allows us to save on even using the sign bit in
  the case where value == 0, i.e. that equation works
  with n == 0 in that case.

  @param [in] value  Value to be encoded, there is no limitation
                  except that it has to fit in int32_t.
  @param [in] min_bits  A value in [0, 30] that will be a lower limit
                  on the returned value.
*/
static inline int least_bits(int32_t value,
                             int min_bits) {
  int bits = min_bits;
  int limit = 1 << bits;
  while (value * 2 < -limit || value * 2 >= limit) {
    limit <<= 1;
    bits++;
  }
  return bits;
}

/**
   Does the central part of encoding a value (`value` would typically be
   a residual).
      @param [in] value      The value to be encoded
      @param [in] min_bits   The minimum number of bits that we are
                     allowed to use to encode `value`... suppose the
                     power-of-two is n, that means our encoding will be capable
                     of encoding values s.t. -2^n <= v*2 < 2^n.
                     [Note: this num-bits does not include the bit for the
                     exponent.]  min_bits will be determined by constraints
                     on how we encode the exponent.
      @param [in] max_bits_encoded   A user-specified value that says the
                     maximum number of bits we'll use in `encoded_value`..
                     this means something quite different from `min_bits`,
                     it means that once the num-bits exceeds this value,
                     we will not encode the least significant bits.
                     It must be >1, but it is allowed to be less than
                     min_bits.
      @param [out] num_bits   A number >= min_bits that will
                    dictate the encoding.  Note: this may be > max_bits,
                    but if so the lower-order bits won't be included,
                    see `num_bits_encoded`
      @param [out] num_bits_encoded  This will just equal
                    min(*num_bits, max_bits_encoded).
      @param [out] encoded_value  The value which will get written
                   to the bit stream; only the lowest-order
                   `*num_bits_encoded` of this are valid.
                   but the higher-order-than-that bits will be
                   all 0 or 1 according to the sign.
                   encoded_value << (*num_bits - *num_bits_encoded)
                   will be a close approximation to `value`, and
                   it will be exact if num_bits == num_bits_encoded.

   See also "decode_signed_value" which reverses this process.
 */
static inline void encode_signed_value(int32_t value,
                                       int min_bits,
                                       int max_bits_encoded,
                                       int *num_bits,
                                       int *num_bits_encoded,
                                       int32_t *encoded_value) {
  assert(min_bits >= 0 && max_bits_encoded > 1);
  *num_bits = least_bits(value, min_bits);
  if (*num_bits <= max_bits_encoded) {
    *num_bits_encoded = *num_bits;
    *encoded_value = value;  /* Only the lowest-order *num_bits_encoded are to
                              * be used */
  } else {
    *num_bits_encoded = max_bits_encoded;
    int rshift = *num_bits - max_bits_encoded;
    *encoded_value = value >> rshift;
  }
}

/**
   This function does the (approximate) reverse of what `encode_signed_value()` does.

     @param [in] code   The code, as retrieved from the bit stream.
                        Only the lowest-order `num_bits_encoded` bits are
                        inspected by this function; the rest may have
                        any value.
     @param [in] num_bits  The number of bits used in the code prior
                        to discarding the lower-order bits
     @param [in] num_bits_encoded  This is equal to
                        min(num_bits, max_bits_encoded), where max_bits_encoded
                        will ultimately be user specified (e.g. bits-per-sample - 1),
                        where bits-per-sample is user-specified.
 */
static inline int32_t decode_signed_value(int32_t code,
                                          int num_bits,
                                          int num_bits_encoded) {
  if (num_bits == 0) {
    return 0;
  } else {
    int32_t full_code = code << (num_bits - num_bits_encoded);

    /**
       non_sign_part is the lowest-order `num_bits - 1` bits of the code.
       sign_part is the `num_bits - 1`'th bit, repeated at that
       and higher-order positions by multiplying by -1.
     */
    int32_t non_sign_part = (full_code & ((1 << (num_bits - 1)) - 1)),
        sign_part = (((int32_t) -1)) * (full_code & ((1 << (num_bits - 1)))),
        rounding_part = (1 << num_bits) >> (num_bits_encoded + 1);
    return non_sign_part | sign_part | rounding_part;
  }
}


/**
   Encodes the residual, returning the least number of bits required to do so
   subject to a caller-specified floor.

      @param [in] residual  The residual that we are trying to encode,
                     meaning: the observed value minus the value that was
                     predicted by the linear prediction.  This is a
                     difference of int16_t's, but such differences cannot
                     always be represented as int16_t, so it's represented
                     as an int32_t.
      @param [in] predicted   The predicted sample (so the residual
                     is the observed sample minus this).  The only reason
                     this needs to be specified is to detect situations
                     where, due to quantization effects, the next decompressed
                     sample would exceed the range of int16_t; in those
                     cases, we need to reduce the magnitude of the encoded value to
                     stay within the allowed range of int16_t (this avoids us
                     having to implement extra checks in the decoder).
       @param [in] min_bits   A caller-supplied floor on the number of bits
                     we will use to encode the residual (the max of this
                     and what least_bits() returns will be used.)
                     It must be in the range [0, 17].
       @param [in] max_bits_encoded  This is a user-specified maximum
                     on the number of bits to use (it will be the
                     bits-per-sample minus one, since the exponent takes one
                     bit.)  It must be >= 2.  Note: It is allowed to be less
                     than min_bits; if the number of bits needed exceeds
                     max_bits_encoded, we discard lower-order bits.
       @param [out]  num_bits  The number of bits used to encode this
                     value before truncation.  Will be the smallest
                     integer n such that n >= min_bits and
                     -2^n <= 2 * residual < 2^n.
       @param [out] num_bits_encoded  Will contain min(num_bits,
                     max_bits_encoded) at exit.
       @param [out]  encoded_value   A signed integer which can be
                     encoded in `num_bits_encoded` bits.
       @param [out] next_decompressed_value  Will contain
                     residual + decode_signed_value(*encoded_value,
                                  *num_bits, *num_bits_encoded).
                    It just happens to be convenient to compute it
                    here.
*/
static inline void encode_residual(int32_t residual,
                                   int16_t predicted,
                                   int min_bits,
                                   int max_bits_encoded,
                                   int *num_bits,
                                   int *num_bits_encoded,
                                   int32_t *encoded_value,
                                   int16_t *next_decompressed_value) {
  assert(min_bits >= 0 && min_bits <= 17 && max_bits_encoded >= 2);

  encode_signed_value(residual, min_bits, max_bits_encoded,
                      num_bits, num_bits_encoded, encoded_value);

  int32_t decoded_residual = decode_signed_value(*encoded_value, *num_bits,
                                                 *num_bits_encoded),
      next_value = (int32_t) predicted + decoded_residual;

  {
    int32_t max_error = (((int32_t) 1) << *num_bits) >> (*num_bits_encoded + 1);
    assert(lilcom_abs(decoded_residual - residual) <= max_error);
  }

  if (((int16_t) next_value) != next_value) {
    /* This should be very rare; it means that we have exceeded the range of
       int16_t due to rounding effects, so we have to decrease the magnitude of
       *encoded_value by 1.  Note: there is a reason why it's >= 0 and not > 0.
       *encoded_value of zero will, if *num_bits > num_bits_encoded, be decoded as
       a positive number (search for `rounding_part` in decode_signed_value()),
       so if this is overshooting we have to turn it into -1.
    */
    *encoded_value += (*encoded_value >= 0 ? -1 : 1);
    int fixed_decoded_residual = decode_signed_value(*encoded_value, *num_bits,
                                                     *num_bits_encoded);
    decoded_residual = fixed_decoded_residual;
    next_value = (int32_t) predicted + decoded_residual;
    assert(((int16_t) next_value) == next_value);
  }
  *next_decompressed_value = next_value;
}


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
inline static int backtracking_encoder_encode(int max_bits_in_sample,
                                              int32_t residual,
                                              int16_t predicted,
                                              int16_t *next_value,
                                              struct BacktrackingEncoder *encoder,
                                              struct BitPacker *packer) {
  ssize_t t = encoder->next_sample_to_encode,
      t_most_recent = encoder->most_recent_attempt;
  size_t t_mod = (t & (NBITS_BUFFER_SIZE - 1)), /* t % buffer_size */
  /* t1_mod is (t-1) % buffer_size using mathematical modulus, not C modulus,
     so -1 % buffer_size is positive.
   */
      t1_mod = ((t - 1) & (NBITS_BUFFER_SIZE - 1));

  int exponent_t1 = encoder->nbits[t1_mod],
      min_codable_exponent =
      LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, exponent_t1),
      exponent_floor = (min_codable_exponent < 0 ? 0 : min_codable_exponent);
  if (t <= t_most_recent) {
    /* We are backtracking, so there will be limitations on the
       exponent. (encoder->nbits[t_mod] represents a floor
       on the exponent). */
    if (encoder->nbits[t_mod] > min_codable_exponent) {
      exponent_floor = encoder->nbits[t_mod];
      /* If the following assert fails it means the logic of the
         backtracking_encoder's routines has failed somewhere. */
      assert(exponent_floor == min_codable_exponent + 1);
    }
  } else {
    encoder->most_recent_attempt = t;
  }

  int exponent, /* note: exponent is not any more the exponent, it is the
                 * num-bits; must rename. */
      num_bits_encoded, mantissa;
  encode_residual(residual, predicted,
                  exponent_floor,  /* min_bits */
                  max_bits_in_sample - 1,
                  &exponent, &num_bits_encoded,
                  &mantissa, next_value);
  if (exponent <= min_codable_exponent + 1) {
    /* Success. */
    int exponent_delta = exponent - min_codable_exponent;
    /* The following is a checks (non-exhaustively) that exponent_delta is 0 or 1. */
    assert((exponent_delta & 254) == 0);
    /** Success; we can represent this.  This is (hopefully) the normal code
        path. */
    encoder->nbits[t_mod] = exponent;
    encoder->next_sample_to_encode = t + 1;

    /* Actually write the code.  Note: this is as if we had written
       first the exponent bit and then the mantissa.  We do it in
       one call, which actually matters because of the way
       the bit-packer object deals with backtracking.
    */
    bit_packer_write_code(t, ((mantissa << 1) + exponent_delta),
                          num_bits_encoded + 1,
                          packer);
    return 0;
  } else {
    /* Failure: the exponent required to most accurately
       approximate this residual is too large; we will need
       to backtrack. */
    encoder->nbits[t_mod] = exponent;
    while (1) {
      exponent = LILCOM_COMPUTE_MIN_PRECEDING_EXPONENT(t, exponent);
      t--;
#ifndef NDEBUG
      encoder->num_backtracks++;
#endif
      /* Now `exponent` is the minimum exponent we'll allow for time t-1. */
      t_mod = t & (NBITS_BUFFER_SIZE - 1);
      if (encoder->nbits[t_mod] >= exponent) {
        /* The exponent value we used for this time satsifes our limit, so we will
           next be encoding the value at time t + 1.  */
        encoder->next_sample_to_encode = t + 1;
        return 1;
      } else {
        /* The following will set a floor on the allowed
           exponent, which this function will inspect when
           it is asked to encode the sample at t-1.. */
        encoder->nbits[t_mod] = exponent;
        /* TODO: maybe change the code so the following if-statement is no
         * longer necessary? */
        if (t < 0) {  /* t == -1 */
          *(encoder->nbits_m1) = exponent;
          encoder->next_sample_to_encode = 0;
          return 1;
        }
      }
    }
    return 1;
  }
}

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
                  struct Decoder *decoder) {
  decoder->bits_per_sample = bits_per_sample;
  decoder->exponent = nbits_m1;
}

/**  This function returns `code`, interpreted as a 2s-complement
     signed integer with `num_bits` bits, with the format changed
     to a 2s-complement signed integer that fits in type `int`.

     Specifically this means that its bits [0..num_bits - 2] are unchanged and
     the bit numbered [num_bits - 1] is duplicated at all higher-ordered
     positions.  (All bit numberings here are from lowest-to-highest order, so
     position 0 is the ones bit.)

       @param [in] code  The integer code to be manipulated.
                         Note: only its bits 0 through num_bits - 1
                         will affect the return value.
       @param [in] num_bits  The number of bits in the code.
                         The code is interpreted as a 2s-complement
                         signed integer with `num_bits` bits.
    @return  Returns the code extended to be a signed int.
*/
static inline int extend_sign_bit(int code, int num_bits) {
  /*
     The first term in the outer-level 'or' is all the `num_bits` bits
     the mantissa, which will be correct for positive mantissa but for
     negative mantissa we need all the higher-order-than-bits_per_sample bits
     to be set as well.  That's what the second term in the or is for.  The
     big number is 2^31 - 1, which means that we duplicate that bit (think of
     it as the sign bit), at its current position and at all positions to its
     left.  */
  if (num_bits == 0) {
    return 0;
  } else {
    return (((unsigned int) code) & ((1 << (num_bits)) - 1)) |
        ((((unsigned int) code) & (1 << (num_bits - 1))) * 2147483647);
  }
}

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
                                 int32_t *value) {
  int exponent_bit = 1 & bit_unpacker_read_next_code(1, unpacker),
      min_codable_exponent = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(
      t, decoder->exponent),
      exponent = min_codable_exponent + exponent_bit,
      num_bits_encoded = lilcom_min(decoder->bits_per_sample - 1,
                                    exponent),
      mantissa = extend_sign_bit(
      bit_unpacker_read_next_code(num_bits_encoded,
                                  unpacker),
      num_bits_encoded);

  decoder->exponent = exponent;

  if (((unsigned int) exponent) > MAX_POSSIBLE_NBITS) {
    debug_fprintf(stderr, "Decompression failed, bad exponent %d at t=%d\n",
                  (int) exponent, (int) t);
    return 1;
  } else {
    /* TODO: don't need to extract the mantissa. */
    *value = decode_signed_value(mantissa, exponent,
                                 num_bits_encoded);
    return 0;
  }
}

