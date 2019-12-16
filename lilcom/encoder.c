#include "encoder.h"

/**
   CAUTION: although this has the .c suffix, it is actually included from
   lilcom.c as if it were a header.  This is so that we can do "static inline"
   and avoid the possibility of multiply defined symbols.  Regular "inline" or
   "extern inline" seems not to give as much speed improvement.

   But aside from inlining issues, interface-wise this does behave like
   a .c file that you could compile independently.
 */




/**
   This macro is added mainly for documentation purposes.  It clarifies what the
   possible bit-widths are for time t given that we know the bit-width for time
   t-1.

       @param [in] t   The time for which we want to know the possible
                       bit-width range.  t >= -1.
       @param [in] width_tm1   The width of the number stored at time t-1:
                       a value in [0..31], otherwise it is an error.
       @return min_width  The minimum width that time t
                       will be able to use, given width_tm1.
                       This equals width_tm1 - ((t+width_tm1)&1),
                       where &1 is a way of computing mod 2.
                       It forms a checkerboard pattern.  If you draw
                       the possible trajectories of widths on
                       a grid you'll see that this choice has nice
                       properties that allows widths to change
                       quite fast when needed.  The return value is a value
                       in [-1..31], but currently going to width
                       -1 is "forbidden", meaning the encoded bit for width
                       change is bound to be 1 in this case so the
                       width for time t would be zero.  We may later
                       make excursions to negative widths allowed,
                       which would allow us to save and then re-use
                       bits, e.g. for on-disk formats.

   Note: the maximum width is just min_width + 1 where min_width is
   what this macro returns, because we use 1 bit for the change in width; a
   bit value of 1 means we choose the max, zero means we choose the min.
 */
#define LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, width_tm1) (width_tm1 - ((((int)t)+width_tm1)&1))

/**
   This macro is defined mainly for documentation purposes.  It returns the
   smallest value the width for time t-1 could be, given the constraint that
   we require the width for time t to be at least a specified value.

        @param [in]  t   The current time, t >= 0.  We want to compute an width-floor for
                         the preceding time (t-1).
        @param [in] width_t  An width for time t, where the caller says
                         they want an width at least this large.  Must be
                         in the range [0..MAX_POSSIBLE_WIDTH].
        @return  Returns the smallest value that the width could have on
                         time t-1 such that the width on time t will be
                         at least width_t.  This may be -1 if width_t
                         was 0, but this macro should never be called in that
                         case.  So this function returns the maximum k such that
                         LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, k) >= width_t.
                         The answer will obviously be either width_t or
                         width_t - 1.  We work out the formula as follows:

                         (a) First, assume (t+width_t) is even, so
                         ((((int)t)+width_t)&1) is zero.  Now we ask,
                         if width_tm1 = width_t - 1, what is
                         the greatest width that we can have on time t?
                         The minimum width we can have on time t is given by
                         LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, (width_t - 1))
                         and in that case we get an odd number in the modulus
                         so there is a -1 in the formula.  That would imply the
                         minimum width on time t is (width_t - 1) - 1 =
                         width_t - 2, so the maximum is width_t - 1
                         (since the bit adds one).  That's a no-go, i.e.
                         for even ((((int)t)+width_t)&1) the answer is
                         just width_t; for odd it's width_t - 1.
                         This happens to be exactly the same formula as
                         LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, width_t).
 */
#define LILCOM_COMPUTE_MIN_PRECEDING_WIDTH(t, width_t) (width_t - ((((int)t)+width_t)&1))




/*
  least_bits(value, min_width):

  For -2^30 <= value < 2^30 and min_width >= 0, returns the
  least n >= min_width such that:

      -2^n <= value * 2 < 2^n

  For `value` and `n` outside this range, we do not define the return value (the
  user should not call this function with such values).  The returned value is
  the smallest number of bits we would need (as a signed 2s-complement integer)
  to represent the value.

  Note: the "* 2" part of the formulation may seem odd, but:

  (1) It means that the number of bits can include the bit needed for the sign
  (2s complement encoding..), and

  (2) It allows us to save on even using the sign bit in the case where value ==
  0, i.e. that equation works with n == 0 in that case.

  @param [in] value  Value to be encoded.  Must satisfy
              -(1<<30) <= value < (1<<30), or the output is meaningless.
  @param [in] min_bits  A value in [0, 31] that will act as a lower limit
              on the returned value, i.e. the search will start from
              this point.
  @return     For `value` satisfying -(1<<30) <= value < (1<<30),
              returns the smallest n >= min_bits which satisfies
              -2^n <= value * 2 < 2^n.  Otherwise, returns a
              meaningless value.
*/
static inline int least_bits(int32_t value,
                             int min_bits) {
  int bits = min_bits;
  int limit = -1 << bits;
  while (value * 2 < limit || value * -2 <= limit) {
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
                     width.]  min_bits will be determined by constraints
                     on how we encode the width.
      @param [in] max_bits_encoded   A user-specified value in [2, 31]
                     that says the maximum number of bits we'll use in
                     `encoded_value`.  This means something quite different
                     from `min_bits`.
                     it means that once the num-bits exceeds max_bits_encoded,
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
                     bits-per-sample minus one, since the width takes one
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



/* See documentation in header */
static inline int backtracking_encoder_encode_limited(int max_bits_in_sample,
                                                      int32_t residual,
                                                      int16_t predicted,
                                                      int16_t *next_value,
                                                      struct BacktrackingEncoder *encoder) {
  ssize_t t = encoder->next_sample_to_encode,
      t_most_recent = encoder->most_recent_attempt;
  size_t t_mod = (t & (WIDTH_BUFFER_SIZE - 1)), /* t % buffer_size */
  /* t1_mod is (t-1) % buffer_size using mathematical modulus, not C modulus,
     so -1 % buffer_size is positive.
   */
      t1_mod = ((t - 1) & (WIDTH_BUFFER_SIZE - 1));

  int width_t1 = encoder->width[t1_mod],
      min_codable_width =
      LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, width_t1),
      width_floor = (min_codable_width < 0 ? 0 : min_codable_width);
  if (t <= t_most_recent) {
    /* We are backtracking, so there will be limitations on the
       width. (encoder->width[t_mod] represents a floor
       on the width). */
    if (encoder->width[t_mod] > min_codable_width) {
      width_floor = encoder->width[t_mod];
      /* If the following assert fails it means the logic of the
         backtracking_encoder's routines has failed somewhere. */
      assert(width_floor == min_codable_width + 1);
    }
  } else {
    encoder->most_recent_attempt = t;
  }

  int width, /* note: width is not any more the width, it is the
                 * num-bits; must rename. */
      num_bits_encoded, mantissa;
  encode_residual(residual, predicted,
                  width_floor,  /* min_bits */
                  max_bits_in_sample - 1,
                  &width, &num_bits_encoded,
                  &mantissa, next_value);
  if (width <= min_codable_width + 1) {
    /* Success. */
    int width_delta = width - min_codable_width;
    /* The following is a checks (non-exhaustively) that width_delta is 0 or 1. */
    assert((width_delta & 254) == 0);
    /** Success; we can represent this.  This is (hopefully) the normal code
        path. */
    encoder->width[t_mod] = width;
    encoder->next_sample_to_encode = t + 1;

    /* Actually write the code.  Note: this is as if we had written
       first the width bit and then the mantissa.  We do it in
       one call, which actually matters because of the way
       the bit-packer object deals with backtracking.
    */
    if (min_codable_width >= 0) {
      bit_packer_write_code(t, ((mantissa << 1) + width_delta),
                            num_bits_encoded + 1,
                            &encoder->bit_packer);
    } else {
      /* If the min codable width was -1, then 'width_delta' would have to be
         1, so there is no point writing it; we save a bit. */
      bit_packer_write_code(t, mantissa,
                            num_bits_encoded,
                            &encoder->bit_packer);

    }
    return 0;
  } else {
    /* Failure: the width required to most accurately
       approximate this residual is too large; we will need
       to backtrack. */
    encoder->width[t_mod] = width;
    while (1) {
      width = LILCOM_COMPUTE_MIN_PRECEDING_WIDTH(t, width);
      t--;
#ifndef NDEBUG
      encoder->num_backtracks++;
#endif
      /* Now `width` is the minimum width we'll allow for time t-1. */
      t_mod = t & (WIDTH_BUFFER_SIZE - 1);
      if (encoder->width[t_mod] >= width) {
        /* The width value we used for this time satsifes our limit, so we will
           next be encoding the value at time t + 1.  */
        encoder->next_sample_to_encode = t + 1;
        return 1;
      } else {
        /* The following will set a floor on the allowed
           width, which this function will inspect when
           it is asked to encode the sample at t-1.. */
        encoder->width[t_mod] = width;
        /* TODO: maybe change the code so the following if-statement is no
         * longer necessary? */
        if (t < 0) {  /* t == -1 */
          *(encoder->width_m1) = width;
          encoder->next_sample_to_encode = 0;
          return 1;
        }
      }
    }
    return 1;
  }
}



/* See documentation in header */
static inline int backtracking_encoder_encode(int max_bits_in_sample,
                                              int32_t residual,
                                              struct BacktrackingEncoder *encoder) {
  ssize_t t = encoder->next_sample_to_encode,
      t_most_recent = encoder->most_recent_attempt;
  size_t t_mod = (t & (WIDTH_BUFFER_SIZE - 1)), /* t % buffer_size */
  /* t1_mod is (t-1) % buffer_size using mathematical modulus, not C modulus,
     so -1 % buffer_size is positive.
   */
      t1_mod = ((t - 1) & (WIDTH_BUFFER_SIZE - 1));

  int width_t1 = encoder->width[t1_mod],
      min_codable_width =
      LILCOM_COMPUTE_MIN_CODABLE_WIDTH(t, width_t1),
      width_floor = (min_codable_width < 0 ? 0 : min_codable_width);
  if (t <= t_most_recent) {
    /* We are backtracking, so there will be limitations on the
       width. (encoder->width[t_mod] represents a floor
       on the width). */
    if (encoder->width[t_mod] > min_codable_width) {
      width_floor = encoder->width[t_mod];
      /* If the following assert fails it means the logic of the
         backtracking_encoder's routines has failed somewhere. */
      assert(width_floor == min_codable_width + 1);
    }
  } else {
    encoder->most_recent_attempt = t;
  }

  int width, /* note: width is not any more the width, it is the
                 * num-bits; must rename. */
      num_bits_encoded, mantissa;
  encode_signed_value(residual,
                      width_floor,  /* min_bits */
                      max_bits_in_sample - 1,
                      &width, &num_bits_encoded,
                      &mantissa);
  if (width <= min_codable_width + 1) {
    /* Success. */
    int width_delta = width - min_codable_width;
    /* The following is a checks (non-exhaustively) that width_delta is 0 or 1. */
    assert((width_delta & 254) == 0);
    /** Success; we can represent this.  This is (hopefully) the normal code
        path. */
    encoder->width[t_mod] = width;
    encoder->next_sample_to_encode = t + 1;

    /* Actually write the code.  Note: this is as if we had written
       first the width bit and then the mantissa.  We do it in
       one call, which actually matters because of the way
       the bit-packer object deals with backtracking.
    */
    if (min_codable_width >= 0) {
      bit_packer_write_code(t, ((mantissa << 1) + width_delta),
                            num_bits_encoded + 1,
                            &encoder->bit_packer);
    } else {
      /* If the min codable width was -1, then 'width_delta' would have to be
         1, so there is no point writing it; we save a bit. */
      bit_packer_write_code(t, mantissa,
                            num_bits_encoded,
                            &encoder->bit_packer);

    }
    return 0;
  } else {
    /* Failure: the width required to most accurately
       approximate this residual is too large; we will need
       to backtrack. */
    encoder->width[t_mod] = width;
    while (1) {
      width = LILCOM_COMPUTE_MIN_PRECEDING_WIDTH(t, width);
      t--;
#ifndef NDEBUG
      encoder->num_backtracks++;
#endif
      /* Now `width` is the minimum width we'll allow for time t-1. */
      t_mod = t & (WIDTH_BUFFER_SIZE - 1);
      if (encoder->width[t_mod] >= width) {
        /* The width value we used for this time satsifes our limit, so we will
           next be encoding the value at time t + 1.  */
        encoder->next_sample_to_encode = t + 1;
        return 1;
      } else {
        /* The following will set a floor on the allowed
           width, which this function will inspect when
           it is asked to encode the sample at t-1.. */
        encoder->width[t_mod] = width;
        /* TODO: maybe change the code so the following if-statement is no
         * longer necessary? */
        if (t < 0) {  /* t == -1 */
          *(encoder->width_m1) = width;
          encoder->next_sample_to_encode = 0;
          return 1;
        }
      }
    }
    return 1;
  }
}




void backtracking_encoder_init(ssize_t num_samples_to_write,
                               int8_t *compressed_code_start,
                               int compressed_code_stride,
                               struct BacktrackingEncoder *encoder) {
  encoder->width_m1 = compressed_code_start;
  encoder->width[WIDTH_BUFFER_SIZE - 1] = (encoder->width_m1[0] = 0);
  encoder->most_recent_attempt = -1;
  encoder->next_sample_to_encode = 0;
#ifndef NDEBUG
  encoder->num_backtracks = 0;
#endif
  bit_packer_init(num_samples_to_write,
                  compressed_code_start + (1 * compressed_code_stride),
                  compressed_code_stride,
                  &encoder->bit_packer);
}


void backtracking_encoder_finish(struct BacktrackingEncoder *encoder,
                                 float *avg_bits_per_sample,
                                 int8_t **next_free_byte) {
  bit_packer_finish(&encoder->bit_packer,
                    avg_bits_per_sample,
                    next_free_byte);
}



void decoder_init(ssize_t num_samples_to_read,
                  const int8_t *compressed_code,
                  int compressed_code_stride,
                  struct Decoder *decoder) {
  int width_m1 = *compressed_code;  /* num-bits for t = -1. */
  decoder->num_bits = width_m1;
  compressed_code += compressed_code_stride;
  /* we give num_samples_to_read * 2 to bit_unpacker_init because
     we will be extracting the width bit and the mantissa separately.
     (Necessary because the size of the width determines the num-bits
     in the mantissa.) */
  bit_unpacker_init(num_samples_to_read * 2,
                    compressed_code, compressed_code_stride,
                    &(decoder->bit_unpacker));
}

void decoder_finish(const struct Decoder *decoder,
                    const int8_t **next_compressed_code) {
  bit_unpacker_finish(&decoder->bit_unpacker,
                      next_compressed_code);
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
       @param [in] num_bits  The number of bits in the code
                         (excluding the width bit).
                         Must satisfy 0 <= n <= 31.  The code is
                         interpreted as a 2s-complement signed integer with
                         `num_bits` bits.

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

/* See documentation in header */
static inline int decoder_decode(ssize_t t,
                                 int max_encoded_mantissa_bits,
                                 struct Decoder *decoder,
                                 int32_t *value) {
  /*
    The expression for `width_bit` below is a little complex...  If
      min_codable_width >= 0, then we read the 1-bit code.  Otherwise
      (min_codable_width == -1), the `width_bit` is bound to be 1 so we don't
      need to read it.  However we want to read a 0-bit code to keep the
      num-samples-read consistent (this is mostly for checking code).  We do
      that, and then discard its value using the "comma" operator, using instead
      the value 1.
  */
  int min_codable_width = LILCOM_COMPUTE_MIN_CODABLE_WIDTH(
          t, decoder->num_bits),
      width_bit = (min_codable_width >= 0  ?
                   1 & bit_unpacker_read_next_code(1, &decoder->bit_unpacker) :
                   (bit_unpacker_read_next_code(0, &decoder->bit_unpacker), 1)),
      width = min_codable_width + width_bit,
      num_bits_encoded = lilcom_min(max_encoded_mantissa_bits,
                                    width),
      mantissa = extend_sign_bit(
          bit_unpacker_read_next_code(num_bits_encoded,
                                      &decoder->bit_unpacker),
      num_bits_encoded);

  decoder->num_bits = width;

  if (width < 0) {
    debug_fprintf(stderr, "Decompression failed, negative width %d at t=%d\n",
                  (int) width, (int) t);
    return 1;
  } else {
    /* TODO: don't need to extract the mantissa. */
    *value = decode_signed_value(mantissa, width,
                                 num_bits_encoded);
    return 0;
  }
}

#ifdef LILCOM_TEST
void lilcom_test_extract_mantissa() {
  for (int bits_per_sample = 4;
       bits_per_sample <= 32; bits_per_sample++) {
    for (int i = 0; i < 1000; i++) {
      int32_t mantissa = -(1<<(bits_per_sample-2));
      if (i == 999)
        mantissa = 1<<(bits_per_sample-2);
      if (mantissa >= (1<<(bits_per_sample-2)))
        continue;
      for (int width_bit = 0; width_bit < 1; width_bit++) {
        for (int random = -3; random <= 3; random++) {
          int32_t code = (((((uint64_t)mantissa) << 1) + width_bit) & ((((uint64_t)1)<<bits_per_sample)-1)) +
              (((uint64_t)random) << bits_per_sample);
          assert((code & 1) == width_bit);
          code >>= 1;
          assert(extend_sign_bit(code, bits_per_sample - 1) == mantissa);
        }
      }
    }
  }
}



void lilcom_test_encode_decode_signed() {
  for (int value = -1000; value <= 1000; value++) {
    for (int min_bits = 0; min_bits < 4; min_bits++) {
      for (int max_bits_encoded = 2; max_bits_encoded <= 32; max_bits_encoded++) {
        int num_bits, num_bits_encoded;
        int32_t encoded;
        encode_signed_value(value, min_bits, max_bits_encoded,
                            &num_bits, &num_bits_encoded, &encoded);

        int32_t decoded = decode_signed_value(encoded, num_bits,
                                              num_bits_encoded);
        /*
        printf("Value=%d, min-bits=%d, max-bits=%d, num-bits{,enc}=%d,%d, encoded=%d, decoded=%d\n",
               (int)value, (int)min_bits, (int)max_bits_encoded,
               (int)num_bits,
               (int)num_bits_encoded, (int)encoded, (int)decoded); */
        int max_error = (num_bits == num_bits_encoded ? 0 :
                         1 << (num_bits - num_bits_encoded - 1));
        assert(lilcom_abs(decoded - value) <= max_error);

      }
    }
  }
}

void lilcom_test_encode_residual() {
  for (int source = 0; source < 1000; source++)  {
    for (int min_bits = 0; min_bits < 4; min_bits++) {
      for (int max_bits_encoded = 2; max_bits_encoded <= 6; max_bits_encoded++) {
        int num_bits, num_bits_encoded;
        int32_t encoded;

        int16_t predicted = (source * 23456) % 65535,
            next_value = (source * 12345) % 65535;
        int32_t residual = next_value - (int32_t)predicted;
        int16_t next_decompressed_value;
        encode_residual(residual, predicted,
                        min_bits, max_bits_encoded,
                        &num_bits, &num_bits_encoded,
                        &encoded,
                        &next_decompressed_value);
        int32_t decoded = decode_signed_value(encoded,
                                              num_bits,
                                              num_bits_encoded),
            decompressed_check = predicted + decoded;
        assert(next_decompressed_value == decompressed_check);
        /*
        printf("residual = %d, predicted = %d, next-value = %d, max-bits=%d, "
               "num-bits{,enc}=%d,%d encoded = %d,  next{,decompressed}=%d,%d\n",
               residual, predicted, next_value, max_bits_encoded, num_bits,
               num_bits_encoded, encoded, next_value, next_decompressed_value);*/
      }
    }
  }
}


void lilcom_test_backtracking_encoder() {
  for (int i = 0; i < 2; i++) {
    for (int max_bits = 3; max_bits < 32; max_bits++) {
      if (max_bits > 16 && i == 1)
        continue;
      /* max_bits includes the width bit. */
      int prime = 111;
      int32_t to_encode[500],
          decoded[500];
      int8_t encoded[5000];
      int stride = 1;
      int t;
      uint64_t rand = extend_sign_bit(1245684, max_bits - 1);
      struct BacktrackingEncoder encoder;
      backtracking_encoder_init(500, encoded, stride, &encoder);
      int8_t *next_free_byte;

      for (t = 0; t < 500; t++) {
        /* rand can be viewed as a signed integer that fits (with the
           sign bit) into `max_bits` bits. */
        rand =  extend_sign_bit(rand * prime, max_bits - 1);
        fprintf(stderr, "to-encode[%d] = %d\n", (int)t, (int)rand);
        to_encode[t] = rand;
      }
      while (encoder.next_sample_to_encode < 500) {
        int16_t dontcare;
        if (i == 1)  {  /* this happens only for max_bits <= 16 */
          backtracking_encoder_encode_limited(max_bits, to_encode[encoder.next_sample_to_encode],
                                              0, &dontcare, &encoder);
        } else {
          backtracking_encoder_encode(max_bits, to_encode[encoder.next_sample_to_encode],
                                      &encoder);
        }
      }
      float avg_bits_per_sample;
      backtracking_encoder_finish(&encoder, &avg_bits_per_sample,
                                  &next_free_byte);

      struct Decoder decoder;
      decoder_init(500, encoded + 0, stride, &decoder);
      for (t = 0; t < 500; t++) {
        int ans = decoder_decode(t, max_bits - 1, &decoder, decoded + t);
        assert(decoded[t] == to_encode[t]);
        assert(!ans);
      }
      const int8_t *next_compressed_code;
      decoder_finish(&decoder, &next_compressed_code);
      assert(next_compressed_code == next_free_byte);
    }
  }
}

#endif /* LILCOM_TEST */
