#include <assert.h>
#include <stdlib.h>  /* for malloc */
#include <math.h>  /* for frexp and frexpf and pow, used in floating point compression. */

/*#undef NDEBUG  */

#ifndef NDEBUG
#include <stdio.h>  /* print statements are only made if NDEBUG is not defined. */
#endif
#include <float.h>  /* for FLT_MAX */

#include "lilcom.h"
#include "lilcom_common.h"
#include "bit_packer.h"
#include "bit_packer.c"
#include "encoder.h"
#include "encoder.c"

/**
   This contains information related to the computation of the linear prediction
   coefficients (LPC).
 */
struct LpcComputation {
  /**
     'autocorr[i]' contains a weighted sum of (s[j] * s[j - i]) << AUTOCORR_LEFT_SHIFT; we use a
     'forgetting factor' that depends on AUTOCORR_DECAY_EXPONENT, i.e. each time we process a
     block (of size AUTOCORR_BLOCK_SIZE) we multiply by (1 +
     2^-{AUTOCORR_DECAY_EXPONENT})^2.  (It's squared because the factor inside
     the parentheses affects the signal, and the autocorrelation is a product of
     two signals.)

     The elements of the sum cannot exceed
     (2^15)^2 = 2^30, shifted left AUTOCORR_LEFT_SHIFT bits, so the elements of
     the sum are <= 2^53.  The sum will be no
     greater than this times around AUTOCORR_BLOCK_SIZE *
     2^(AUTOCORR_DECAY_EXPONENT+1) (the approximate number of terms added together)
     = 2^8, so the sum would be <= 2^(50+8) = 2^58.  The most this could be
     without overflow is around 2^61.  */
  int64_t autocorr[MAX_LPC_ORDER + 1];

  /**
     autocorr_to_remove contains some terms which are *in* autocorr but which
     need to be removed every time we update the autocorrelation coefficients.
     Let the 'baseline' autocorrelation be that of a signal that is scaled
     down exponentially as you go further back in time.
   */
  int64_t autocorr_to_remove[MAX_LPC_ORDER + 1];

  /*
     max_exponent is the smallest number >= 1 such that autocorr[0] >>
     max_exponent == 0.  (Note: autocorr[0] is nonnegative).
     This implies that autocorr[0] < 2^{max_exponent}.
     It is used in the fixed-point arithmetic used in lilcom_compute_lpc(), to
     shift the coefficients to a 'nice' range.

     Note: it is guaranteed by the nature of the autocorrelation of a signal
     that for 1 <= i <= lpc_order, abs(autocorr[i]) <= autocorr[0], so this
     also gives a bound on the magnitudes of the other autocorrelation
     coefficients.
  */
  int32_t max_exponent;
  /*
     Contains the LPC coefficients times 2^LPC_APPLY_LEFT_SHIFT.  These are
     initialized to [1 0 0 0.. ] at the start of the utterance (so the
     prediction is just the previous sample).  Of course that's represented as
     [1<<LPC_APPLY_LEFT_SHIFT 0 0 0 ...]

     If we have processed at least one block of AUTOCORR_BLOCK_SIZE samples and
     have called lilcom_compute_lpc, this will contain the coefficients
     estimated from there.  Only elements 0 through lpc_order - 1 will be valid.
     (lpc_order is passed directly into functions dealing with this object).
  */
  int32_t lpc_coeffs[MAX_LPC_ORDER];
};

/**
   Initialize an LpcComputation object with the default parameter that we have
   at the start of the file.  This corresponds to all-zero autocorrelation
   stats, max_exponent = 0, and lpc_coeffs that correspond to [1.0 0 0 0].
*/
static void lilcom_init_lpc(struct LpcComputation *lpc,
                            int lpc_order) {
  for (int i = 0; i <= lpc_order; i++) {
    lpc->autocorr[i] = 0;
    lpc->autocorr_to_remove[i] = 0;
  }
  lpc->max_exponent = 1;
  /* The LPC coefficientss are stored shifted left by LPC_APPLY_LEFT_SHIFT, so this
     means the 1st coeff is 1.0 and the rest are zero-- meaning, we start
     prediction from the previous sample.  */
  lpc->lpc_coeffs[0] = 1 << LPC_APPLY_LEFT_SHIFT;
  for (int i = 1; i < lpc_order; i++) {
    lpc->lpc_coeffs[i] = 0;
  }
  /** Note: some code which is called  after calling this, will set
      lpc->lpc_coeffs[lpc_order] to 0 also, if lpc_order is odd.
      It's needed because of some loop unrolling we do (search for
      "sum2").
  */

}

/**
   Updates the autocorrelation stats in 'coeffs', by scaling down the previously
   computed stats slightly and adding one block's worth of new autocorrelation
   data.  This is called every AUTOCORR_BLOCK_SIZE samples.

     @param [in,out]  autocorr   The statistics to be updated (i.e. scaled down
                           and then added to).
     @param [in] lpc_order   The LPC order, must be in [0..MAX_LPC_ORDER]
     @param [in] compute_lpc   compute_lpc will be nonzero if the LPC
                           coefficients will be recomputed after this block.
                           It's needed only because we can avoid some work
                           if we know that we won't be computing LPC
                           after this block.
     @param [in] signal    Pointer to the signal at the start of the block
                           from which we are accumulating statistics.
                           Note: the LPC coefficients used for the
                           i'th block depend on the autocorr stats for
                           block i-1, so 'signal' will point to the
                           start of block i-1.
                           CAUTION:
                           In all cases we will access values signal[-lpc_order]
                           through signal[AUTOCORR_BLOCK_SIZE-1], so 'signal'
                           cannot point to the start of an array.
*/
static inline void lilcom_update_autocorrelation(
    struct LpcComputation *lpc, int lpc_order, int compute_lpc,
    const int16_t *signal) {
  /** 'temp_autocorr' will contain the raw autocorrelation stats without the
      shifting left by AUTOCORR_LEFT_SHIFT; we'll do the left-shifting at the
      end, to save an instruction in the inner loop).
      The dimension is MAX_LPC_ORDER + 2 instead of just plus one, because
      of a loop unrolling trick we do below (another, unused element may
      be written.)
  */
  int64_t temp_autocorr[MAX_LPC_ORDER + 2];
  int i;

  /** Remove the temporary term in autocorr_to_remove (that arises from
      reflecting the signal).  Then scale down the current data slightly.  This
      is to form an exponentially decaying sum of the autocorrelation stats (to
      preserve freshness), but done at the block level not the sample level.
      Also zero `temp_autocorr`.  */
  for (i = 0; i <= lpc_order; i++) {
    lpc->autocorr[i] -= lpc->autocorr_to_remove[i];
    lpc->autocorr_to_remove[i] = 0;

    /**
      Now we scale down / decay the previously computed autocorrelation coefficients
      to account for the exponential windowing function.
       What we really want to do is, if a is the autocorrelation coefficient
       and d == AUTOCORR_DECAY_EXPONENT, to scale it by
         a *=  (1 - 2^-d)^2  =  (1 - 2^{-(d-1)} + 2^{-2d})
       (it's squared because autocorrelations are products of 2 signal values.)
       So assuming right-shift was guaranteed to be arithmetic right-shift
       (which, unfortunately, C doesn't seem to guarantee), we'd want to do:
          a = a  - a>>(d-1)  + a>>(2*d)
       The divisions below is a 'safer' way of right-shifting by those
       amounts; technically, right-shift gives implementation-defined results
       for negative input.
    */
    lpc->autocorr[i] = lpc->autocorr[i] -
        (lpc->autocorr[i] / (1 << (AUTOCORR_DECAY_EXPONENT - 1))) +
        (lpc->autocorr[i] / (1 << (2 * AUTOCORR_DECAY_EXPONENT)));
    temp_autocorr[i] = 0;
  }

  /** HISTORY SCALING
      Process any terms involving the history samples that are prior to the
      start of the block.

      The samples (for left-context) that come from the previous block need to
      be scaled down slightly, in order to be able to guarantee that no element
      of lpc->autocorr is greater than the zeroth element.  We can then view
      the autocorrelation as a simple sum on a much longer (but
      decreasing-with-time) sequence of data, which means we don't have to
      reason about what happens when we interpolate autocorrelation stats.

      See the comment where AUTOCORR_DECAY_EXPONENT is defined for more details.
      Notice that we write some terms directly to lpc->autocorr instead of to
      temp_autocorr, to make the scaling-down possible.  */
  for (i = 0; i < lpc_order; i++) {
    int64_t signal_i = signal[i];
    int j;
    for (j = 0; j <= i; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
    for (j = i + 1; j <= lpc_order; j++)
      lpc->autocorr[j] += (signal[i - j] * signal_i) *
          ((1 << AUTOCORR_LEFT_SHIFT) -
              (1 << (AUTOCORR_LEFT_SHIFT - AUTOCORR_DECAY_EXPONENT)));
  }

  /** OK, now we handle the samples that aren't close to the boundary.
      currently, i == lpc_order. */
  for (; i < AUTOCORR_BLOCK_SIZE; i++) {
    /* signal_i only needs to be int64_t to handle the case where signal[i - j] and signal_i are
       both -32768, so their product can't be represented as int32_t.  We rely on the
       product below being automatically cast to int64_t. */
    int64_t signal_i = signal[i];

    /* The unrolled loop below will write an extra, useless element to
       temp_autocorr[lpc_order+1] if lpc_order is even.  The unrolling
       should make pipelined execution faster. */
    for (int j = 0; j <= lpc_order; j += 2) {
      temp_autocorr[j] += signal[i - j] * signal_i;
      temp_autocorr[j + 1] += signal[i - j - 1] * signal_i;
    }
  }

  /** Copy from the temporary buffer to struct lpc, shifting left
      appropriately.  */
  for (int j = 0; j <= lpc_order; j++)
    lpc->autocorr[j] += temp_autocorr[j] << AUTOCORR_LEFT_SHIFT;


  /** The next block adds some 'temporary' terms which arise from (conceptually):
      (a) reflecting the signal at the most-recent-time + 1/2, so all
          terms appear twice, but we have a few extra terms due to edge effects
      (b) dividing by 2.
     So what we get that's extra is the extra terms from edge effects,
     divided by two.

     We only do this if `compute_lpc` is true (i.e., if we will be re-estimating
     the LPC coefficients after this block), because if we won't be, what's
     inside this block is a "don't-care".. we'll just be subtracting it on the
     next block.
   */
  if (compute_lpc) {
    /* Note: signal_edge must only be indexed with negative coeffs.  Imagine a
       symmetric virtual signal v[t], where for t < 0, v[t] := signal_edge[t],
       and for t >= 0, v[t] := signal_edge[-1-t].  [It's symmetric around
       t=0.5.]

       We are accumulating the LPC coeffs that 'cross the boundary', i.e.
       involve products of v[t]'s where one index has t < 0 and the other has t
       >= 0.  The "i" below is the index with t >= 0.
    */
    const int16_t *signal_edge = signal + AUTOCORR_BLOCK_SIZE;

    for (i = 0; i < lpc_order; i++) {
      /* the -1 in the exponent below is the factor of 0.5 mentioned as (b)
         above. */
      int64_t signal_i = ((int64_t) signal_edge[-1 - i]) << (AUTOCORR_LEFT_SHIFT - 1);
      for (int j = i + 1; j <= lpc_order; j++) {  /* j is the lag */
        lpc->autocorr_to_remove[j] += signal_i * signal_edge[i - j];
      }
    }
    for (i = 0; i <= lpc_order; i++)
      lpc->autocorr[i] += lpc->autocorr_to_remove[i];
  }


  /* The next statement takes care of the smoothing to make sure that the
     autocorr[0] is nonzero, and adds extra noise proportional to the signal
    energy, which is determined by AUTOCORR_EXTRA_VARIANCE_EXPONENT.  This will
    allow us to put a bound on the value of the LPC coefficients so we don't
    need to worry about integer overflow.
    (Search for: "NOTE ON BOUNDS ON LPC COEFFICIENTS")  */
  lpc->autocorr[0] +=
      ((int64_t) ((AUTOCORR_BLOCK_SIZE * AUTOCORR_EXTRA_VARIANCE) << AUTOCORR_LEFT_SHIFT)) +
          (temp_autocorr[0] << (AUTOCORR_LEFT_SHIFT - AUTOCORR_EXTRA_VARIANCE_EXPONENT));

  /* We will have copied the max_exponent from the previous LpcComputation
     object, and it will usually already have the correct value.  Return
     immediately if so.  */
  int exponent = lpc->max_exponent;
  /* Make autocorr_0 unsigned to make it clear that right-shift is well defined... actually
     it can never be negative as it is a sum of squares. */
  uint64_t autocorr_0 = lpc->autocorr[0];
  assert(autocorr_0 != 0 && exponent > 0);
  if ((autocorr_0 >> (exponent - 1)) == 1) {
    /*  max_exponent has the correct value.  This is the normal code path. */
    return;
  }
  while ((autocorr_0 >> (exponent - 1)) == 0)
    exponent--;
  while ((autocorr_0 >> (exponent - 1)) > 1)
    exponent++;
  /** We can assert that exponent > 0 because we know that lpc->autocorr[0] is
      at this point comfortably greater than 1; see above, the term
      (AUTOCORR_BLOCK_SIZE*AUTOCORR_EXTRA_VARIANCE)<<AUTOCORR_LEFT_SHIFT)).
      The fact that exponent > 0 is necessary to stop right-shifting
      by (exponent-1) from generating an error.
  */
  assert((autocorr_0 >> (exponent - 1)) == 1 &&
      (autocorr_0 >> exponent) == 0 && exponent > 0);
  lpc->max_exponent = exponent;
}

/*
    *NOTE ON BOUNDS ON LPC COEFFICIENTS*

    Suppose the newly added part of autocorrelation stats at time zero (i.e. the
    sum of squares) is S.  We add to the autocorrelation at time zero, S *
    2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} (currently 2^16), so the autocorrelation
    stats for time t=0 are always greater than their "real" values by a factor of
    1 + 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT}.

    This can provide a bound on the magnitude of the LPC coefficients, which
    is useful in reasoning about fixed-point computations.

    For each LPC coefficient, say c_i, the variance in our estimate
    of the current sample would be increased by
    c_i^2 * S * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT}, due to this
    extra term.  (Remember that this raw-variance applies to history
    samples as well as the current sample, so imagine the past
    sample comes with some built-in noise).

    Now, if the LPC coeffs were all zero, the prediction variance we'd get would
    be exactly S (in expectation, over the stats we have.. note, we're ignoring
    scaling factors or normalization, as all this is invariant to them).

    We know that the extra noise from term mentioned above must not exceed S, or
    we'd be doing worse than just using zero LPC coeffs, which would imply that
    something had gone wrong in our LPC estimation.  So that means:a

       c_i^2 * S * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} < S
     so
       c_i^2 * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} < 1

    (don't worry about the case where S = 0; see AUTOCORR_EXTRA_VARIANCE).
    That implies that c_i <= 2^(AUTOCORR_EXTRA_VARIANCE_EXPONENT / 2),
    i.e. abs(c_i) < 2^8.
    (the factor of 2 comes from taking the square root of both sides).
    With current values of constants, this means that the LPC coeffs
    cannot exceed 2^8 (which is certainly not the tightest bound we
    could get).  So when stored as integers with LPC_EST_LEFT_SHIFT (currently
    23) as the exponent, they cannot exceed 2^31.  This is used
    during the LPC estimation.

    SUMMARY: the absolute value of the LPC coefficients (their real values, not
    the integer representation) must be less than 2^8.  =
    2^(AUTOCORR_EXTRA_VARIANCE_EXPONENT / 2).
 */


/**
   struct CompressionState contains the state that we need to maintain
   while compressing a signal; it is passed around by functions
   that are compressing the signal.
 */
struct CompressionState {

  /** The LPC order, a value in [0..MAX_LPC_ORDER].  User-specified.  */
  int lpc_order;

  /** The user-specified bits per sample, in the range [4..8]. */
  int bits_per_sample;

  /**
     'lpc_computations' is to be viewed as a circular buffer of size 2,
     containing the autocorrelation and LPC coefficients.  Define the
     block-index b(t), a function of t, as:

         b(t) = t / AUTOCORR_BLOCK_SIZE,

     rounding down, of course.  The LPC coefficients we use to
     predict sample t will be those taken from lpc_computations[b(t) % 2].

     Whenever we process a t value that's a multiple of AUTOCORR_BLOCK_SIZE and
     which is nonzero, we will use the preceding block of AUTOCORR_BLOCK_SIZE
     coefficients of the signal (and also, the `lpc_order` samples of context
     preceding that) to update the autocorrelation coefficients and possibly
     also the LPC coefficients (see docs for LPC_COMPUTE_INTERVAL).
  */
  struct LpcComputation lpc_computations[LPC_ROLLING_BUFFER_SIZE];

  /**
     `encoder` handles the logic of exponents, mantissas and backtracking,
     and gives us codes to be converted to bytes by struct `BitPacker`.
   */
  struct BacktrackingEncoder encoder;

  /**
    `decompressed_signal` is the compressed-and-then-decompressed version of the
     input signal.  We need to keep track of this because the decompressed
     version is used to compute the LPC coefficients (this ensures that we can
     compute them in exactly the same way when we decompress, so we don't have
     to transmit them).

     The signal at time t is located at

       decompressed_signal[MAX_LPC_ORDER + (t % SIGNAL_BUFFER_SIZE)]

     The reason for the extra MAX_LPC_ORDER elements is so that we can ensure,
     when we roll around to the beginning of the buffer, that we have a place to
     put the recent history (the previous `lpc_order` samples).  This keeps the
     code of lilcom_update_autocorrelation and lilcom_compute_predicted_value
     from having to loop around.
  */
  int16_t decompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE];

  /** The following object has responsibility for writing out the codes
      (each with between 4 and 8 bits). */
  struct BitPacker packer;

  /**  The input signal that we are compressing  */
  const int16_t *input_signal;

  /**  The stride associated with `input_signal`; normally 1 */
  int input_signal_stride;

  /** header_start is the start of the header in the compressed code. */
  int8_t *header_start;
  /** stride of the header, equals packer.compressed_code_stride
      but repeated for clarity. */
  int header_stride;

  /** num_backtracks is only used in debug mode. */
  ssize_t num_backtracks;

};



/*
  HEADER FORMAT:

  The lilcom_header functions below mostly serve to clarify the format
  of the header; we could just have put the statements inline, which is
  what we hope the compiler will do.

  The format of the 4-byte header is:

    Byte 0:  Least-significant 4 bits are currently unused.
             The next 3 bits contain LILCOM_VERSION (currently 1).
             The highest-order bit is always set (this helps work out the
             time axis when decompressing, together with it never being
             set for byte 2.
    Byte 1:  The lowest 7 bits contain the LPC order which must be
             in 0..MAX_LPC_ORDER (currently 14.)
             The highest-order bit is set if the number of samples
             stored is odd; this is used to disambiguate the number of
             samples.
    Byte 2:  Contains the bits-per-sample which must
             be in [LILCOM_MIN_BPS..LILCOM_MAX_BPS].
             We use the fact that this can never be negative or exceed 127 to
             note that the highest-order bit is never set; this is
             used to work out the time axis of compressed data, in conjunction
             with the highest-order bit of byte 0 always being set.
    Byte 3:  The negative of the conversion exponent c, as int8_t.  (We store
             as the negative because the allowed range of c is [-127..128], and
             an int8_t can store [-128..127].  The conversion exponent is only
             relevant when uncompressing to floating point types.  After
             converting to int16, we will cast to float and then multiply by
             2^(c-15).  When compressing, we would have multiplied by 2^{15-c}
             and then rounded to int16, with c chosen to avoid overflow.  This
             will normally be set (by calling code) to 0 if the data was
             originally int16; this will mean that when converting to float,
             we'll remain in the range [-1, 1]
    Byte 4:  The exponent for time t == -1, as a signed integer.
 */


#define LILCOM_HEADER_NBITS_M1_OFFSET 4

/** Set the conversion_exponent in the header.
        @param [in] header  Pointer to start of the header
        @param [in] stride  Stride of the output int8_t data, normally 1
        @param [in] conversion_exponent   Search above for 'Byte 3:' for an
                            explanation of what this is.  It's user-supplied,
                            as far as this part of the code is concerned.
                            Required to be in the range [-128, 127], and
                            we fail with assert error if not (the code that
                            enforces this is inside this module, so it would
                            be our code error if this fails).
*/
static inline void lilcom_header_set_conversion_exponent(
    int8_t *header, int stride, int conversion_exponent) {
  assert(conversion_exponent >= -127 && conversion_exponent <= 128);
  header[3 * stride] = (int8_t) (-conversion_exponent);
}

static inline int lilcom_header_get_conversion_exponent(
    const int8_t *header, int stride) {
  return -((int) (header[3 * stride]));
}

/** Set the LPC order, the bits-per-sample and the bit saying
    whether the num-samples was odd, in the header.
         @param [out] header  Pointer to start of header
         @param [in] stride  Stride of header
         @param [in] lpc_order  lpc order in [0..14].
         @param [in] bits_per_sample  bits_per_sample in
                    [LPC_MIN_BPS..LPC_MAX_BPS]
         @param [in] num_samples_odd  1 if num-samples was odd, else 0.
   All this goes in byte 1 of the header, i.e. the 2nd byte.
*/
static inline void lilcom_header_set_user_configs(
    int8_t *header, int stride, int lpc_order,
    int bits_per_sample, int num_samples_odd) {
  assert(lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER &&
      bits_per_sample >= LILCOM_MIN_BPS &&
      bits_per_sample <= LILCOM_MAX_BPS &&
      num_samples_odd <= 1);
  header[0 * stride] = (int8_t) ((LILCOM_VERSION << 4) + 128);
  header[1 * stride] = (int8_t) lpc_order + (num_samples_odd << 7);
  header[2 * stride] = bits_per_sample;
}

/** Return the LPC order from the header.  Does no range checking!  */
static inline int lilcom_header_get_lpc_order(const int8_t *header, int stride) {
  return (int) (header[1 * stride] & 127);
}

/** Returns bits_per_sample from the header; result will be in [4..8].  */
static inline int lilcom_header_get_bits_per_sample(const int8_t *header, int stride) {
  return (int) header[2 * stride];
}

/** Returns the parity of the original num-samples from the header,
    i.e. 0 if it was even, 1 if it was odd. */
static inline int lilcom_header_get_num_samples_parity(const int8_t *header, int stride) {
  return ((int) (header[1 * stride] & 128)) != 0;
}

/**  Check that this is plausibly a lilcom header.  */
static inline int lilcom_header_plausible(const int8_t *header,
                                          int stride) {
  /** Currently only one version number is supported. */
  int byte0 = header[0 * stride],
      byte2 = header[2 * stride],
      bps = lilcom_header_get_bits_per_sample(header, stride),
      lpc_order = lilcom_header_get_lpc_order(header, stride);
  int ans = (byte0 & 0xF0) == ((LILCOM_VERSION << 4) + 128) &&
      (byte2 & 128) == 0 &&
      bps >= LILCOM_MIN_BPS && bps <= LILCOM_MAX_BPS &&
      lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER;
  assert(ans);
  return ans;
}

/**
   Computes the LPC coefficients via an integerized version of the Durbin
   computation (The Durbin computation computes linear prediction coefficients
   from autocorrelation coefficients).

     `lpc_order` is the linear prediction order, a user-specified number
              in [0..MAX_LPC_ORDER.  It is the number of taps of the IIR filter
              / the number of past samples we predict on.
     `lpc` is a struct containing the autocorrelation coefficients and
              LPC coefficients (to be written).

   This code was adapted to integer arithmetic from the 'Durbin' function in
   Kaldi's feat/mel-computations.cc, which in turn was originaly derived from
   HTK.  (We had a change couple of signs, though; for some reason the original
   code was computing the negative of the LPC coefficients.)  Anyway, Durbin's
   algorithm is very well known.  The reason for doing this in integer
   arithmetic is mostly to ensure 100% reproducibility without having to get
   into system-dependent methods of setting floating point rounding modes.
   Since the LPC coefficients are not transmitted, but are estimated from the
   preceding data, in order for the decompression to work we need to ensure that
   the coefficients are exactly the same as they were when we were compressing
   the data.

*/
void lilcom_compute_lpc(int lpc_order,
                        struct LpcComputation *lpc) {
  /**
     autocorr is just a copy of lpc->autocorr, but shifted to ensure that the
     absolute value of all elements is less than 1<<LPC_EST_LEFT_SHIFT (but as large
     as possible given that constraint).  We don't need to know how much it was
     shifted right, because a scale on the autocorrelation doesn't affect the
     LPC coefficients.  */
  int32_t autocorr[MAX_LPC_ORDER + 1];

  { /** This block just sets up the `autocorr` array with appropriately
        shifted copies of lpc->autocorr.  */
    int max_exponent = lpc->max_exponent;
    assert(max_exponent >= AUTOCORR_LEFT_SHIFT &&
        (((uint64_t) (lpc->autocorr[0])) >> (max_exponent - 1) == 1));
    if (max_exponent > LPC_EST_LEFT_SHIFT) {
      /* shift right (the normal case).  We do it by division, because
         technically the result of shifting a negative number right
         is undefined (even though it normally does what we want,
         which is to duplicate the left-most -1) */
      int right_shift = max_exponent - LPC_EST_LEFT_SHIFT;
      for (int i = 0; i <= lpc_order; i++)
        autocorr[i] = (int32_t) (lpc->autocorr[i] / ((int64_t) 1 << right_shift));
    } else {
      int left_shift = LPC_EST_LEFT_SHIFT - max_exponent;
      for (int i = 0; i <= lpc_order; i++)
        autocorr[i] = (int32_t) (lpc->autocorr[i] << left_shift);
    }
    assert((((uint64_t) autocorr[0]) >> (LPC_EST_LEFT_SHIFT - 1)) == 1);
    for (int i = 1; i <= lpc_order; i++) {  /* TODO: remove this loop. */
      assert(lilcom_abs(autocorr[i]) <= autocorr[0]);
    }
  }

  /**
   'temp' is a temporary array used in the LPC computation, stored with
   exponent of LPC_EST_LEFT_SHIFT (i.e. shifted left by that amount).  In Kaldi's
   mel-computations.cc is is called pTmp.  It seems to temporarily store the
   next iteration's LPC coefficients.  Note: the floating-point value of these
   coefficients is strictly less than 2^8 (see BOUNDS ON LPC COEFFICIENTS
   above).  The absolute values of the elements of 'temp' will thus be less than
   2^(LPC_EST_LEFT_SHIFT+8) = 2^31.  i.e. it is guaranteed to fit into an int32.
  */
  int64_t temp[MAX_LPC_ORDER];

  int32_t E = autocorr[0];
  int j;
  for (int i = 0; i < lpc_order; i++) {
    /** ki will eventually be the next reflection coefficient, a value in [-1, 1], but
        shifted left by LPC_EST_LEFT_SHIFT to represent it in fixed point.
        But after the following line it will represent a floating point
        number times 2 to the power 2*LPC_EST_LEFT_SHIFT,
        so currently abs(ki) < 2^(LPC_EST_LEFT_SHIFT+LPC_EST_LEFT_SHIFT) = 2^46.
        We do this because we'll later divide by E.
      Original code: "float ki = autocorr[i + 1];"  */
    int64_t ki = ((int64_t) autocorr[i + 1]) << LPC_EST_LEFT_SHIFT;

    for (j = 0; j < i; j++) {
      /** max magnitude of the terms added below is 2^(LPC_EST_LEFT_SHIFT*2 + 8) = 2^54, i.e.
          the abs value of the added term is less than 2^54.
          ki still represents a floating-point number times 2 to the
          power 2*LPC_EST_LEFT_SHIFT.
        The original floating-point code looked the same as the next line
        (not: ki has double the left-shift of the terms on the right).
      */
      ki -= lpc->lpc_coeffs[j] * (int64_t) autocorr[i - j];
    }
    /** RE the current magnitude of ki:
        ki is a summation of terms, so add LPC_ORDER_BITS=4 to the magnitude of
        2^54 computed above, it's now bounded by 2^59 (this would bound its value
        at any point in the summation above).
        original code: "ki = ki / E;".   */
    ki = ki / E;
    /** At this point, ki is mathematically in the range [-1,1], since it's a
        reflection coefficient; and it is stored times 2 to the power LPC_EST_LEFT_SHIFT, so its
        magnitude as an integer is <= 2^LPC_EST_LEFT_SHIFT.  */

    /**  Original code: "float c = 1 - ki * ki;"  Note: ki*ki is nonnegative,
         so shifting right corresponds to division. */
    int64_t c = (((int64_t) 1) << LPC_EST_LEFT_SHIFT) -
        (((uint64_t) (ki * ki)) >> LPC_EST_LEFT_SHIFT);

    /** c is the factor by which the residual has been reduced; mathematically
        it is always >= 0, but here it must be > 0 because of our smoothing of
        the variance via AUTOCORR_EXTRA_VARIANCE_EXPONENT and
        AUTOCORR_EXTRA_VARIANCE which means the residual can never get to
        zero.
    */

    /** The original code did: E *= c;
        Note: the product is int64_t because c is int64_t, which is important
        to avoid overflow.  Also note: unsigned right-shift only corresponds to division
        because the result (still an energy E) is nonnegative; in fact,
        E is guaranteed to be positive here because we smoothed the
        0th autocorrelation coefficient (search for variable names containing
        EXTRA_VARIANCE)
    */
    E = (int32_t) (((uint64_t) (E * c)) >> LPC_EST_LEFT_SHIFT);
    /**
        If E < = 0, it means that something has gone wrong.  We've never
        actually encountered this case, but we want to make sure the algorithm
        is robust and doesn't crash, even if we have made a mistake in our logic
        somewhere.
    */
    if (E <= 0)
      goto panic;

    /** compute the new LP coefficients
        Original code did: pTmp[i] = -ki;
        Note: abs(temp[i]) <= 2^LPC_EST_LEFT_SHIFT, since ki is in the range [-1,1]
        when viewed as a real number. */
    temp[i] = (int32_t) ki;
    for (j = 0; j < i; j++) {
      /** The original code did:
          pTmp[j] = pLP[j] - ki * pLP[i - j - 1]
         These are actual LPC coefficients (computed for LPC order i + 1), so
         their magnitude is less than 2^(LPC_EST_LEFT_SHIFT+8) = 2^31.

         The term on the RHS that we cast to int32_t is also bounded by
         2^31, because it's (conceptually) an LPC coefficient multiplied by a
         reflection coefficient ki with a value <= 1.
         We divide by (1 << LPC_EST_LEFT_SHIFT) rather than just shifting
         right because the C standard says that the effect of right-shifting
         negative signed integers is undefined.  Hopefully the compiler can
         optimize that expression.
      */
      temp[j] = lpc->lpc_coeffs[j] -
          (int32_t) ((ki * lpc->lpc_coeffs[i - j - 1]) / (1 << LPC_EST_LEFT_SHIFT));
    }
    for (j = 0; j <= i; j++) {
      assert(lilcom_abs(temp[j]) < ((int64_t) 1 << (LPC_EST_LEFT_SHIFT + 8)));
      lpc->lpc_coeffs[j] = (int32_t) temp[j];
    }
  }
  /** E > 0 because we added fake extra variance via
     AUTOCORR_EXTRA_VARIANCE_EXPONENT and AUTOCORR_EXTRA_VARIANCE, so according
     to these stats the sample should never be fully predictable.  We'd like to
     assert that E <= autocorr[0] because even if the data is totally
     uncorrelated, we should never be increasing the predicted error vs. having
     no LPC at all.  But I account for the possibility that in pathological
     cases, rounding errors might make this untrue.
  */
  assert(E > 0 && E <= (autocorr[0] + ((int64_t) (((uint64_t) autocorr[0]) >> 10))));


  /**
     We want to shift the LPC coefficients right by LPC_EST_LEFT_SHIFT -
     LPC_APPLY_LEFT_SHIFT, because we store and apply them with lower precision
     than we estimate them.  We divide rather than shift because technically
     the result of right-shifting a negative number is implementation-defined.
  */
  for (int i = 0; i < lpc_order; i++) {
    lpc->lpc_coeffs[i] /= (1 << (LPC_EST_LEFT_SHIFT - LPC_APPLY_LEFT_SHIFT));
  }
  return;

  panic:
  debug_fprintf(stderr, "Lilcom: warning: panic code reached.\n");
  lpc->lpc_coeffs[0] = (1 << LPC_APPLY_LEFT_SHIFT);
  for (int i = 0; i < lpc_order; i++)
    lpc->lpc_coeffs[0] = 0;
  assert(0);  /** when compiled with -NDEBUG this won't actually crash. */
}

/**
   Compute predicted signal value based on LPC coeffiients for time t and the
   preceding state->lpc_order samples (treated as zero for negative time
   indexes).  Note: the preceding samples are taken from the buffer
   state->decompressed_signal; we need to use the compressed-then-decompressed
   values, not the original values, so that the LPC coefficients will
   be exactly the same when we decode.

         @param [in] state   Object containing variables for the compression
                             computation
         @param [in] t       Time index t >= 0 for which we need the predicted
                             value.
         @return             Returns the predicted value as an int16_t.
 */
static inline int16_t lilcom_compute_predicted_value(
    struct CompressionState *state,
    ssize_t t) {
  uint32_t lpc_index =
      ((uint32_t) (((size_t) t) >> LOG_AUTOCORR_BLOCK_SIZE)) % LPC_ROLLING_BUFFER_SIZE;
  struct LpcComputation *lpc = &(state->lpc_computations[lpc_index]);

  int lpc_order = state->lpc_order;

  /** Get the pointer to the t'th signal in the circular buffer
      'state->decompressed_signal'.  The buffer has an extra MAX_LPC_ORDER
      samples at the start, to provide needed context when we roll around,
      to keep this code simple. */
  int16_t *decompressed_signal_t =
      &(state->decompressed_signal[MAX_LPC_ORDER + (t & (SIGNAL_BUFFER_SIZE - 1))]);


  /**
     sum1 and sum2 correspond to two terms in the LPC-prediction calculation,
     which we break into two for pipelining reasons.  They are stored as
     unsigned to ensure that overflow behavior is architecture-independent
     (search for `wildly` below for more information).

     The initial value of sum1 can be thought of a being:

       - 0.5 (converted to this fixed-point representation), which
         is to ensure we `round-to-closest` rather than round down;

         plus ..

       - A big number (2^16) that is designed to keep the sum positive, so that
         when, later, we right shift, the behavior is well defined.  Later we'll
         remove this by having it overflow.  It's important that this
         number be >= 16 because that's the smallest power of 2 that, when
         cast to int16_t, disappears.  Also this number plus
         LILCOM_APPLY_LEFT_SHIFT must be less than 31, to avoid overflowing
         int32_t.
  */
  uint32_t sum1 = (1 << (LPC_APPLY_LEFT_SHIFT - 1)) +
      (1 << (LPC_APPLY_LEFT_SHIFT + 16)),
      sum2 = 0;

  /** The following is an optimization of a simple loop for i = 0 .. lpc_order -
      1.  It may access a one-past-the-end element if lpc_order is odd, but
      this is OK.  We made sure in lilcom_init_lpc() that unused
      elements of lpc_coeffs are zeroed. */
  int i;
  const int32_t *lpc_coeffs = &(lpc->lpc_coeffs[0]);
  for (i = 0; i < lpc_order; i += 2) {
    /* Cast them to uint32_t before multiplying, to avoid a crash when
       the compiler option -ftrapv is used. */
    sum1 += (uint32_t) lpc_coeffs[i] * (uint32_t) decompressed_signal_t[-1 - i];
    sum2 += (uint32_t) lpc_coeffs[i + 1] * (uint32_t) decompressed_signal_t[-2 - i];
  }

  /** The lpc_coeffs were stored times 2^LPC_APPLY_LEFT_SHIFT.  Divide by this
      to get the integer prediction `predicted`.  We do the shift in
      uint32_t to make sure it's well defined.

      Define the `true` predicted value as:

    true_predicted =
     (sum_{i=0}^{lpc_order-1}  lpc_coeffs[i] * decompressed_signal_t[-1-i]) / 2^LPC_APPLY_LEFT_SHIFT
      computed exactly as a mathematical expression.

      If true_predicted is in the range [-(1<<16), 1<<16] (which is
      twice the range of signed int16_t), then we can show that
      the expression for `predicted` below is the same as
      (int(true_predicted) + 1<<16), where int() rounds to
      the closest integer but towards +infinty in case of ties.

      If true_predicted is out of that range, that does not always hold, and we
      say that this expression `does what it does`.  I.e. we don't worry
      about the accuracy of the prediction in this highly-pathological
      case, but just make sure that all expressions are well defined so
      it's all full deterministic (very important for this algorithm).
  */
  int32_t predicted = (int32_t) ((sum1 + sum2) >> LPC_APPLY_LEFT_SHIFT);

#ifdef LILCOM_TEST
  if (1) {  /* This block tests the logic in the comment above "predicted". */
    int64_t true_predicted = 0;
    for (i = 0; i < lpc_order; i ++)
      true_predicted += lpc_coeffs[i] * decompressed_signal_t[-1-i];
    /* Note we haven't yet divided by 2^LPC_APPLY_LEFT_SHIFT. */
    if (true_predicted >= -(1<<(16+LPC_APPLY_LEFT_SHIFT)) &&
        true_predicted <= (1<<(16+LPC_APPLY_LEFT_SHIFT))) {
      /* The following will ensure true_predicted is nonnegative
         so that right shift is well defined.. */
      true_predicted += (1<<(16+LPC_APPLY_LEFT_SHIFT));
      /* The following is like adding 0.5, which will ensure
         rounding to the closest integer (but up in case of
         ties). */
      true_predicted += (1<<(LPC_APPLY_LEFT_SHIFT - 1));
      true_predicted >>= LPC_APPLY_LEFT_SHIFT;
      assert(predicted == true_predicted);
    }
  }
#endif


  /** Caution: at this point, `predicted` contains the predicted value
      plus 2^16 = 65536.  Recall that we initialized the sum
      with a term of 1 << (LPC_APPLY_LEFT_SHIFT + 16). */

  /** Now we deal with the case where the predicted value was outside
      the range [-32768,32764].  Right now `predicted` has 65536 added
      to it so we need to account for that when making the comparisons.
      The offset 65536 will naturally disappear when casting to int16_t.
  */

  /** The following if-statement is a one-shot way of testing "is the variable
      `predicted` outside the range [65536-32768, 65536+32767]?"  If it is
      outside that range, that implies that the `real` predicted value (given by
      predicted - 65536) was outside the range [-32768,32767], which would mean
      we need to do some truncation to avoid predicting a value not
      representable as int16_t (allowing the prediction to wrap around would
      degrade fidelity).
      The above condition is equivalent to (predicted-32768) being in
      [0..65535], which is what we test first because it makes the fast path
      faster if there is just one conditional.
   */
  if (((predicted - 32768) & ~(int32_t) 65535) != 0) {
    if (predicted > 32767 + 65536)
      predicted = 65536 + 32767;
    else if (predicted < -32768 + 65536)
      predicted = 65536 - 32768;
  }
  assert(predicted >= -32768 + 65536 && predicted <= 32767 + 65536);
  return (int16_t) predicted;
}

/** Copies the final state->lpc_order samples from the end of the
    decompressed_signal buffer to the beginning in order to provide required
    context when we roll around.  This function is expected to be called only
    when t is a nonzero multiple of SIGNAL_BUFFER_SIZE. */
static inline void lilcom_copy_to_buffer_start(
    struct CompressionState *state) {
  for (int i = 1; i <= state->lpc_order; i++)
    state->decompressed_signal[MAX_LPC_ORDER - i] =
        state->decompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE - i];
}

/**
   This function updates the autocorrelation statistics and, if relevant, the
   LPC coefficients.
      @param [in] t    Sample index of the sample that we are about to compress.
                   Required to be a multiple of AUTOCORR_BLOCK_SIZE and >= 0.  The data we're
                   going to use to update the autocorrelation statistics
                   are those from the *previous* block, from at t -
                   AUTOCORR_BLOCK_SIZE to t-1.  (If t == 0 we will do nothing and rely
                   on the initialization previously done in lilcom_init_lpc).

      @param [in,out]  state   Struct containing the computation state.
                   We are modifying one of the `lpc_computations` elements.
 */
void lilcom_update_autocorrelation_and_lpc(
    ssize_t t, struct CompressionState *state) {
  assert(t % AUTOCORR_BLOCK_SIZE == 0 && state->lpc_order > 0 && t >= 0);
  /** We'll compute the LPC coeffs if t is a multiple of LPC_COMPUTE_INTERVAL or
      if t is a nonzero value less than LPC_COMPUTE_INTERVAL (for LPC freshness
      at the start).  */
  int compute_lpc = ((t & (LPC_COMPUTE_INTERVAL - 1)) == 0);
  if (t < LPC_COMPUTE_INTERVAL) {
    if (t == 0) {
      /* For time t = 0, there is nothing to do because there
         is no previous block to get stats from.  We rely on
         lilcom_init_lpc having previously been called. */
      return;
    }
    /** For 0 < t < LPC_COMPUTE_INTERVAL we recompute the LPC coefficients
       every AUTOCORR_BLOCK_SIZE, to improve the LPC estimates
       for the first few samples. */
    compute_lpc = 1;
  }
  /** The expressions below are defined because the args to '%' are unsigned;
      modulus for negative args is implementation-defined according to the C
      standard.  */
  uint32_t lpc_index = ((uint32_t) (((size_t) t) >> LOG_AUTOCORR_BLOCK_SIZE)) % LPC_ROLLING_BUFFER_SIZE,
      prev_lpc_index = (lpc_index + LPC_ROLLING_BUFFER_SIZE - 1) % LPC_ROLLING_BUFFER_SIZE;

  struct LpcComputation *this_lpc = &(state->lpc_computations[lpc_index]);
  struct LpcComputation *prev_lpc = &(state->lpc_computations[prev_lpc_index]);

  /** Copy the previous autocorrelation coefficients and max_exponent.  We'll
      either re-estimate or copy the LPC coefficients below. */
  int lpc_order = state->lpc_order;
  for (int i = 0; i <= lpc_order; i++) {
    this_lpc->autocorr[i] = prev_lpc->autocorr[i];
    this_lpc->autocorr_to_remove[i] = prev_lpc->autocorr_to_remove[i];
  }
  this_lpc->max_exponent = prev_lpc->max_exponent;

  ssize_t prev_block_start_t = t - AUTOCORR_BLOCK_SIZE;
  assert(prev_block_start_t >= 0);
  /**  buffer_index = prev_block_start_t % SIGNAL_BUFFER_SIZE */
  int32_t buffer_index = prev_block_start_t & (SIGNAL_BUFFER_SIZE - 1);
  /*  We view the buffer as 'really' starting from the element numbered
      MAX_LPC_ORDER; those extra elements are for left context at the start of
      the buffer (c.f. lilcom_copy_to_buffer_start).  */
  const int16_t *signal_pointer =
      &(state->decompressed_signal[MAX_LPC_ORDER + buffer_index]);
  /** Update the autocorrelation stats (decay the old ones, add
      new terms */
  lilcom_update_autocorrelation(this_lpc, lpc_order,
                                compute_lpc, signal_pointer);
  if (compute_lpc) {
    /** Recompute the LPC based on the new autocorrelation
        statistics  */
    lilcom_compute_lpc(state->lpc_order, this_lpc);
  } else {
    /* Copy the previous block's LPC coefficients  */
    for (int i = 0; i < lpc_order; i++)
      this_lpc->lpc_coeffs[i] = prev_lpc->lpc_coeffs[i];
  }
}

/**
   Initializes a newly created CompressionState struct, setting fields and doing
   the compression for time t = 0 which is a special case.

   Does not check its arguments; that is assumed to have already been done
   in calling code.
 */
static inline void lilcom_init_compression(
    ssize_t num_samples,
    const int16_t *input, int input_stride,
    int8_t *output, int output_stride,
    int lpc_order, int bits_per_sample,
    int conversion_exponent,
    struct CompressionState *state) {

  bit_packer_init(num_samples,
                  output + LILCOM_HEADER_BYTES * output_stride,
                  output_stride,
                  &state->packer);

  state->bits_per_sample = bits_per_sample;
  state->lpc_order = lpc_order;

  lilcom_init_lpc(&(state->lpc_computations[0]), lpc_order);
  if (lpc_order % 2 == 1) {
    /** The following is necessary because of some loop unrolling we do while
        applying lpc; search for "sum2". */
    for (int i = 0; i < LPC_ROLLING_BUFFER_SIZE; i++)
      state->lpc_computations[i].lpc_coeffs[lpc_order] = 0;
  }

  backtracking_encoder_init(
      output + (LILCOM_HEADER_NBITS_M1_OFFSET * output_stride),
      &(state->encoder));

  state->input_signal = input;
  state->input_signal_stride = input_stride;
  state->header_start = output;
  state->header_stride = output_stride;

  /** Put it in an assert so it doesn't happen in non-debug mode. */
#ifndef NDEBUG
  state->num_backtracks = 0;
#endif

  for (int i = 0; i < MAX_LPC_ORDER; i++)
    state->decompressed_signal[i] = 0;

  lilcom_header_set_conversion_exponent(output, output_stride,
                                        conversion_exponent);
  lilcom_header_set_user_configs(output, output_stride,
                                 lpc_order, bits_per_sample,
                                 num_samples % 2);
  /* TODO: Remove the following. */
  assert(lilcom_header_get_lpc_order(output, output_stride) == lpc_order &&
      lilcom_header_get_bits_per_sample(output, output_stride) == bits_per_sample &&
      lilcom_header_get_num_samples_parity(output, output_stride) == num_samples % 2);
}

/*  See documentation in lilcom.h.  */
ssize_t lilcom_get_num_bytes(ssize_t num_samples,
                             int bits_per_sample) {
  if (!(num_samples > 0 && bits_per_sample >= LILCOM_MIN_BPS &&
      bits_per_sample <= LILCOM_MAX_BPS))
    return -1;
  else
    return LILCOM_HEADER_BYTES + (bits_per_sample * num_samples + 7) / 8;
}

/*  See documentation in lilcom.h  */
int lilcom_compress(
    const int16_t *input, ssize_t num_samples, int input_stride,
    int8_t *output, ssize_t num_bytes, int output_stride,
    int lpc_order, int bits_per_sample, int conversion_exponent) {
  if (num_samples <= 0 || input_stride == 0 || output_stride == 0 ||
      lpc_order < 0 || lpc_order > MAX_LPC_ORDER ||
      bits_per_sample < LILCOM_MIN_BPS || bits_per_sample > LILCOM_MAX_BPS ||
      conversion_exponent < -127 || conversion_exponent > 128 ||
      num_bytes != lilcom_get_num_bytes(num_samples, bits_per_sample)) {
    debug_fprintf(stderr, "[lilcom] failure in compression: something "
                          "wrong with args.");
    return 1;  /* error */
  }

  struct CompressionState state;
  lilcom_init_compression(num_samples, input, input_stride,
                          output, output_stride, lpc_order,
                          bits_per_sample, conversion_exponent,
                          &state);

  while (state.encoder.next_sample_to_encode < num_samples) {
    ssize_t t = state.encoder.next_sample_to_encode;
    if ((t & (AUTOCORR_BLOCK_SIZE - 1)) == 0 && state.lpc_order != 0 && t != 0) {
      if ((t & (SIGNAL_BUFFER_SIZE - 1)) == 0) {
        /**  If this is the start of the uncompressed_signal buffer we need to
             make sure that the required left context is copied appropriately. */
        lilcom_copy_to_buffer_start(&state);
      }
      /** Update the autocorrelation coefficients and possibly the LPC
          coefficients. */
      lilcom_update_autocorrelation_and_lpc(t, &state);
    }
    int16_t predicted_value = lilcom_compute_predicted_value(&state, t),
        observed_value = state.input_signal[t * state.input_signal_stride];
    /** cast to int32 when computing the residual because a difference of int16's may
        not fit in int16. */
    int32_t residual = ((int32_t) observed_value) - ((int32_t) predicted_value);
    int16_t *next_value =
        &(state.decompressed_signal[MAX_LPC_ORDER + (t & (SIGNAL_BUFFER_SIZE - 1))]);

    backtracking_encoder_encode(bits_per_sample, residual,
                                predicted_value, next_value,
                                &state.encoder, &state.packer);
    /* We are actually ignoring the return status of backtracking_encoder_encode. */
  }

  float bits_written_per_sample;
  int8_t *next_free_byte;  /* Unused currently. */
  bit_packer_flush(&state.packer,
                   &bits_written_per_sample,
                   &next_free_byte);
  debug_fprintf(stderr,
                "Avg bits-per-sample is %.2f bits vs. max of %d bits\n",
                (float) bits_written_per_sample, (int) bits_per_sample);

  debug_fprintf(stderr, "Backtracked %f%% of the time\n",
                ((state.num_backtracks * 100.0) / num_samples));

  return 0;
}

/**
   This function does the core part of the decompression of one sample
   (excluding the part about updating the autocorrelation statistics and
   updating the LPC coefficients; that is done externally.

      @param [in] t    The current time index, cast to int; we actually
                     only need its lowest-order bit.
      @param [in] bits_per_sample  The number of bits per sample,
                     in [4..8].
      @param [in] lpc_order  The order of the LPC computation,
                     a number in [0..LPC_MAX_ORDER] obtained from
                     the header.
      @param [in] lpc_coeffs  The LPC coefficients, multiplied
                     by 2^23 and represented as integers.
      @param [in] input_code  The code for the sample that
                      we are about to decompress.  Its lower-order
                      `bits_per_sample` bits correspond to the code;
                      its higher-order bit values are undefined.
      @param [in,out] output_sample  A pointer to the output sample
                     for time t.  CAUTION: this function assumes that
                     `output_sample` this is a pointer to an array with stride 1
                     and that the preceding `lpc_order` samples exist, i.e. that
                     we can read from output_sample[-lpc_order] through
                     output_sample[-1] and write to output_sample[0].
      @param [in,out] exponent  At entry, this will be set to the
                     exponent used to encode the previous sample,
                     which must be in [0..12] else this function will
                     fail (see return status).  At exit, it will be the
                     exponent used to encode the current frame.
      @return  Returns 0 on success, 1 on failure.  Failure would normally
                     mean data corruption or possily a code error.
                     This function will fail if the input exponent is not
                     in the range [0,12] or the signal left the bounds
                     of int16_t.

 */
static inline int lilcom_decompress_one_sample(
    ssize_t t,
    int lpc_order,
    const int32_t *lpc_coeffs,
    int32_t residual,
    int16_t *output_sample) {

  int16_t predicted_sample;

  { /** This block computes the predicted value.  For explanation please
        look for similar statements in `lilcom_compute_predicted_value()`,
        which is well documented. */
    uint32_t sum1 = (1 << (LPC_APPLY_LEFT_SHIFT - 1)) +
        (1 << (LPC_APPLY_LEFT_SHIFT + 16)), sum2 = 0;

    for (int i = 0; i < lpc_order; i += 2) {
      sum1 += (uint32_t) lpc_coeffs[i] * (uint32_t) output_sample[-1 - i];
      sum2 += (uint32_t) lpc_coeffs[i + 1] * (uint32_t) output_sample[-2 - i];
    }
    int32_t predicted = (int32_t) ((sum1 + sum2) >> LPC_APPLY_LEFT_SHIFT);
    if (((predicted - 32768) & ~65535) != 0) {
      if (predicted > 32767 + 65536)
        predicted = 65536 + 32767;
      else if (predicted < -32768 + 65536)
        predicted = 65536 - 32768;
    }
    assert(predicted >= 65536 - 32768 && predicted <= 65536 + 32767);
    predicted_sample = (int16_t) predicted;
  }

  int32_t new_sample = (int32_t) predicted_sample + residual;

  if (((new_sample + 32768) & ~(int32_t) 65535) != 0) {
    /** If `new_sample` is outside the range [-32768 .. 32767], it
        is an error; we should not be generating such samples. */
    debug_fprintf(stderr, "lilcom: decompression failure (corruption?), t = %d\n",
                  (int) t);
    return 1;
  }
  output_sample[0] = new_sample;
  return 0;  /** Success */
}

ssize_t lilcom_get_num_samples(const int8_t *input,
                               ssize_t input_length,
                               int input_stride) {
  if (input_length <= 5 || input_stride == 0 ||
      !lilcom_header_plausible(input, input_stride))
    return -1;  /** Error */
  int bits_per_sample = lilcom_header_get_bits_per_sample(input, input_stride),
      parity = lilcom_header_get_num_samples_parity(input, input_stride);
  /* num_samples is set below to the maximum number of samples that could be
     encoded by `input_length` bytes.  There may be some ambiguity because we
     had to round up to a multiple of 8 bits when compressing, (i.e. the
     original number of samples might have been one less), so we use 'parity' to
     disambiguate.
  */
  ssize_t num_samples = ((input_length - LILCOM_HEADER_BYTES) * 8) / bits_per_sample;

  if (num_samples % 2 != parity)
    num_samples--;
  return num_samples;
}

int lilcom_decompress(const int8_t *input, ssize_t num_bytes, int input_stride,
                      int16_t *output, ssize_t num_samples, int output_stride,
                      int *conversion_exponent) {
  if (num_samples <= 0 || input_stride == 0 || output_stride == 0 ||
      !lilcom_header_plausible(input, input_stride) ||
      num_samples != lilcom_get_num_samples(input, num_bytes, input_stride)) {
    debug_fprintf(stderr,
                  "lilcom: Warning: bad header, num-bytes=%d, num-samples=%d, "
                  "input-stride=%d, plausible=%d\n",
                  (int) num_bytes, (int) num_samples, (int) input_stride,
                  (int) lilcom_header_plausible(input, input_stride));
    return 1;  /** Error */
  }

  int lpc_order = lilcom_header_get_lpc_order(input, input_stride),
      bits_per_sample = lilcom_header_get_bits_per_sample(input, input_stride);

  *conversion_exponent = lilcom_header_get_conversion_exponent(
      input, input_stride);

  struct BitUnpacker unpacker;
  /*  The reason we set the num-samples to num_samples * 2 is that
     the code actually reads the exponent bit and the mantissa separately.
   */
  bit_unpacker_init(num_samples * 2,
                    input + (input_stride * LILCOM_HEADER_BYTES), input_stride,
                    &unpacker);

  struct LpcComputation lpc;
  lilcom_init_lpc(&lpc, lpc_order);
  /** The following is necessary because of some loop unrolling we do while
      applying lpc; search for "sum2". */
  if (lpc_order % 2 == 1)
    lpc.lpc_coeffs[lpc_order] = 0;

  /** The first LPC_MAX_ORDER samples are for left-context; view the array as
      starting from the element with index LPC_MAX_ORDER.
      In the case where output_stride is 1, we'll only be using the
      first LPC_MAX_ORDER + AUTOCORR_BLOCK_SIZE elements of this
      array.
  */
  int16_t output_buffer[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE];
  int i;
  for (i = 0; i < MAX_LPC_ORDER; i++)
    output_buffer[i] = 0;

  struct Decoder decoder;
  decoder_init(bits_per_sample,
               input[LILCOM_HEADER_NBITS_M1_OFFSET * input_stride],
               &decoder);

  output_buffer[MAX_LPC_ORDER] = output[0];
  int t;
  for (t = 0; t < AUTOCORR_BLOCK_SIZE && t < num_samples; t++) {
    int32_t residual;
    if (decoder_decode(t, &unpacker, &decoder, &residual) != 0 ||
        lilcom_decompress_one_sample(t, lpc_order,
                                     lpc.lpc_coeffs, residual,
                                     &(output_buffer[MAX_LPC_ORDER + t])) != 0) {
      debug_fprintf(stderr, "lilcom: decompression failure for t=%d\n", (int) t);
      return 1;  /** Error */
    }
    output[t * output_stride] = output_buffer[MAX_LPC_ORDER + t];
  }
  if (t >= num_samples) {
    bit_unpacker_finish(&unpacker);
    return 0;  /** Success */
  }

  if (output_stride == 1) {
    /** Update the autocorrelation with stats from the 1st block (it's
        a special case, as we need to use `output_buffer` so the
        left-context will work right. */
    lilcom_update_autocorrelation(&lpc, lpc_order, 1,
                                  output_buffer + MAX_LPC_ORDER);
    /** Recompute the LPC.  Even though time t = AUTOCORR_BLOCK_SIZE
        is not a multiple of LPC_COMPUTE_INTERVAL, we update it every
        AUTOCORR_BLOCK_SIZE for t < LPC_COMPUTE_INTERVAL, for
        freshness at the start of the signal. */
    lilcom_compute_lpc(lpc_order, &lpc);

    /** From this point forward we don't need a separate buffer; `output`
        has stride 1, so we can use that as the buffer for autocorrelation
        and lpc. */
    while (t < num_samples) {
      /** Every `AUTOCORR_BLOCK_SIZE` samples we need to accumulate
          autocorrelation statistics and possibly recompute the LPC
          coefficients.  If t == AUTOCORR_BLOCK_SIZE we don't do this, since in
          that case we already did it a few lines above (using the temporary
          buffer, which has the left-context for t < 0 appropriately set to
          zero). */
      assert((t & (AUTOCORR_BLOCK_SIZE - 1)) == 0);

      if (t != AUTOCORR_BLOCK_SIZE) {
        int compute_lpc = (t & (LPC_COMPUTE_INTERVAL - 1)) == 0 ||
            (t < LPC_COMPUTE_INTERVAL);

        lilcom_update_autocorrelation(&lpc, lpc_order,
                                      compute_lpc, output + t - AUTOCORR_BLOCK_SIZE);
        if (compute_lpc)
          lilcom_compute_lpc(lpc_order, &lpc);
      }
      ssize_t local_max_t = (t + AUTOCORR_BLOCK_SIZE < num_samples ?
                             t + AUTOCORR_BLOCK_SIZE : num_samples);
      for (; t < local_max_t; t++) {
        int32_t residual;
        if (decoder_decode(t, &unpacker, &decoder, &residual) != 0 ||
            lilcom_decompress_one_sample(t, lpc_order, lpc.lpc_coeffs, residual,
                                         output + t) != 0) {
          debug_fprintf(stderr, "lilcom: decompression failure for t=%d\n",
                        (int) t);
          return 1;  /** Error */
        }
      }
    }
    return 0;  /** Success */
  } else {
    /** output_stride != 1, so we need to continue to use `output_buffer` and
        copy it to `output` when done. */

    while (t < num_samples) {
      /** Every `AUTOCORR_BLOCK_SIZE` samples we need to accumulate
          autocorrelation statistics and possibly recompute the LPC
          coefficients.  If t == AUTOCORR_BLOCK_SIZE we don't do this, since in
          that case we already did it a few lines above (using the temporary
          buffer, which has the left-context for t < 0 appropriately set to
          zero). */
      assert((t & (AUTOCORR_BLOCK_SIZE - 1)) == 0);

      if ((t & (SIGNAL_BUFFER_SIZE - 1)) == 0) {
        /** A multiple of SIGNAL_BUFFER_SIZE.  We need to copy the context to
            before the beginning of the buffer.  */
        for (i = 1; i <= lpc_order; i++)
          output_buffer[MAX_LPC_ORDER - i] =
              output_buffer[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE - i];
      }

      /** buffer_start_t is the t value at which we'd want to give
          `lilcom_update_autocorrelation` a pointer to the output.
          We'll later have to take this modulo SIGNAL_BUFFER_SIZE
          and then add MAX_LPC_ORDER to find the right position in
          `output_buffer`. */
      ssize_t buffer_start_t = t - AUTOCORR_BLOCK_SIZE;
      int compute_lpc = (t & (LPC_COMPUTE_INTERVAL - 1)) == 0 ||
          (t < LPC_COMPUTE_INTERVAL);
      lilcom_update_autocorrelation(&lpc, lpc_order, compute_lpc,
                                    output_buffer + MAX_LPC_ORDER + buffer_start_t % SIGNAL_BUFFER_SIZE);
      /** If t is a multiple of LPC_COMPUTE_INTERVAL or < LPC_COMPUTE_INTERVAL.. */
      if (compute_lpc)
        lilcom_compute_lpc(lpc_order, &lpc);

      ssize_t local_max_t = (t + AUTOCORR_BLOCK_SIZE < num_samples ?
                             t + AUTOCORR_BLOCK_SIZE : num_samples);
      for (; t < local_max_t; t++) {
        int32_t residual;
        if (decoder_decode(t, &unpacker, &decoder, &residual) != 0 ||
            lilcom_decompress_one_sample(
                t, lpc_order, lpc.lpc_coeffs, residual,
                output_buffer + MAX_LPC_ORDER + (t & (SIGNAL_BUFFER_SIZE - 1))) != 0) {
          debug_fprintf(stderr, "lilcom: decompression failure for t=%d\n",
                        (int) t);
          return 1;  /** Error */
        }

        output[t * output_stride] =
            output_buffer[MAX_LPC_ORDER + (t & (SIGNAL_BUFFER_SIZE - 1))];
      }
    }
    bit_unpacker_finish(&unpacker);
    return 0;  /** Success */
  }
}

/**
   Returns the maximum absolute value of any element of the array 'f'.  If there
   is any NaN in the array, return NaN if possible Note: at higher compiler
   optimization levels we cannot always rely on this behavior so officially
   the behavior with inputs with NaN's is undefined.

      @param [in] input            The input array
      @param [in] num_samples      The number of elements in the array.
      @param [in] stride           The stride between array elements;
                                 usually 1.
 */
float max_abs_float_value(const float *input, ssize_t num_samples, int stride) {
  ssize_t t;

  float max_abs_value = 0.0;
  for (t = 0; t + 4 <= num_samples; t += 4) {
    float f1 = lilcom_abs(input[t * stride]),
        f2 = lilcom_abs(input[(t + 1) * stride]),
        f3 = lilcom_abs(input[(t + 2) * stride]),
        f4 = lilcom_abs(input[(t + 3) * stride]);
    /**
       The reason we have this big "and" statement is that computing a bitwise
       "and" of integers doesn't incur any branches (whereas && might)...
       there is only one branch when processing 4 values, which is good
       good for pipelining.

       Note: we'll very rarely go inside this if-statement.  The reason we use
       <= in the comparisons is because we want any NaN's we encounter to take
       us inside the if-statement.  But we can't fully rely on this behavior,
       since compiler optimization may change it. */
    if (!((f1 <= max_abs_value) &
        (f2 <= max_abs_value) &
        (f3 <= max_abs_value) &
        (f4 <= max_abs_value))) {
      if (!(f1 <= max_abs_value)) {
        max_abs_value = f1;
        if (max_abs_value != max_abs_value)
          return max_abs_value;  /** NaN. */
      }
      if (!(f2 <= max_abs_value)) {
        max_abs_value = f2;
        if (max_abs_value != max_abs_value)
          return max_abs_value;  /** NaN. */
      }
      if (!(f3 <= max_abs_value)) {
        max_abs_value = f3;
        if (max_abs_value != max_abs_value)
          return max_abs_value;  /** NaN. */
      }
      if (!(f4 <= max_abs_value)) {
        max_abs_value = f4;
        if (max_abs_value != max_abs_value)
          return max_abs_value;  /** NaN. */
      }
    }
  }
  for (t = 0; t < num_samples; t++) {
    float f = lilcom_abs(input[t * stride]);
    if (!(f <= max_abs_value)) {
      max_abs_value = f;
      if (max_abs_value != max_abs_value)
        return max_abs_value;  /** NaN. */
    }
  }
  return max_abs_value;
}

/**
   This function returns the conversion exponent we will use for a floating-point
   input sequence with a maximum-absolute-value equal to 'max_abs_value'.

   If max_abs_value is infinity or NaN, it will return -256, which means
   "invalid".  Otherwise the result will be in the range [-127..128].

   If max_abs_value is zero, this function will return zero.  (The exponent
   won't matter in this case).

   Let the canonical answer be the most negative i satisfying:

        max_abs_value < 32767.5 * 2^(i-15)

    (this inequality ensures that max_abs_value will be rounded to numbers with
    absolute value <= 32767 when quantizing, if we use this exponent, thus
    avoiding overflowing int16_t range).
    ... so the canonical answer is the most negative i satisfying:

        max_abs_value < (65535.0/65536) * 2^i

   If the canonical answer would be outside the range [-127, 128], then the
   returned value truncated to that range.  Note: if we truncated to 128 this
   means we need to be extra-careful with float-to-int conversion, because
   +FLT_MAX ~ 2^128 would be multiplied by 2^(15 - 128) before conversion, so
   we'd be converting a number just under 2^15 = +32768 to int.  That would be
   rounded to +32768, which if we don't truncate, will be interpreted as -32768
   in int16.  We are extra careful in the float-to-int conversion if we detect
   that conversion_exponent is 127.

   Note: the return i, if not equal to -256, will always satisfy
          max_abs_value < 2^i
 */
int compute_conversion_exponent(float max_abs_value) {
  if (!(max_abs_value - max_abs_value == 0))
    return -256;  /* Inf or nan -> error. */
  if (max_abs_value == 0)
    return 0;

  /** The next few lines just get a reasonable initial guess at the exponent.
      We add 1 because floating point numbers are, in most cases, greater
      than 2^exponent (since the mantissa m satisfies 2 < m <= 1).
  */
  int i;
  frexpf(max_abs_value, &i);
  i += 1;

  /**
     The range of base-2 exponents allowed for single-precision floating point
     numbers is [-126..127].  If we get to close to the edges of this range, we
     have to worry about special cases like what happens if pow(2.0, i) is not
     representable as float or 32767 * powf(2.0, i) is not representable as
     float.  So we do the computation in double below (implicitly, via
     upcasting).  It's slower, but this doesn't dominate at all.

     Neither of the loops below should go more than once, I don't think.  The
     reason for having two loops, instead of one, is so that we don't have to
     reason too carefully about this and worry about correctness.

     The reason for the i < 1000 and i > -1000 is so that if something totally
     weird happened (like the compiler optimized too aggressively and
     max_abs_value was infinity), we don't loop forever.
  */

  while (max_abs_value >= (65535.0 / 65536) * pow(2.0, i) && i < 1000)
    i++;
  while (max_abs_value < (65535.0 / 65536) * pow(2.0, i - 1) && i > -1000)
    i--;

  if (i == 1000 || i == -1000) {
    /** This point should never be reached. */
    debug_fprintf(stderr, "lilcom: warning: something went wrong while "
                          "finding the exponent: i=%d\n", i);
    return -256;
  }

  if (i < -127)
    i = -127;
  if (i > 128)
    i = 128;
  return i;
}

int lilcom_compress_float(
    const float *input, ssize_t num_samples, int input_stride,
    int8_t *output, ssize_t num_bytes, int output_stride,
    int lpc_order, int bits_per_sample, int16_t *temp_space) {

  if (num_samples <= 0 || input_stride == 0 || output_stride == 0 ||
      lpc_order < 0 || lpc_order > MAX_LPC_ORDER ||
      bits_per_sample < LILCOM_MIN_BPS || bits_per_sample > LILCOM_MAX_BPS ||
      num_bytes != lilcom_get_num_bytes(num_samples, bits_per_sample))
    return 1;  /* error */

  if (temp_space == NULL) {
    /* Allocate temp array and recurse. */
    int16_t *temp_array = (int16_t *) malloc(sizeof(int16_t) * num_samples);
    if (temp_array == NULL)
      return 2;  /* Special error code for this situation. */
    int ans = lilcom_compress_float(input, num_samples, input_stride,
                                    output, num_samples, output_stride,
                                    lpc_order, bits_per_sample,
                                    temp_array);
    free(temp_array);
    return ans;
  }

  float max_abs_value = max_abs_float_value(input, num_samples, input_stride);
  if (max_abs_value - max_abs_value != 0) {
    debug_fprintf(stderr, "[lilcom] Detected inf or NaN (1)\n");
    return 1;  /* Inf's or Nan's detected */
  }
  int conversion_exponent = compute_conversion_exponent(max_abs_value);

  /* -256 is the error code when compute_conversion_exponent detects infinities
      or NaN's. */
  if (conversion_exponent == -256) {
    debug_fprintf(stderr, "[lilcom] Detected inf or NaN (2)\n");
    return 2;  /* This is the error code meaning we detected inf or NaN. */
  }
  assert(conversion_exponent >= -127 && conversion_exponent <= 128);

  int adjusted_exponent = 15 - conversion_exponent;

  if (adjusted_exponent > 127) {
    /** A special case.  If adjusted_exponent > 127, then 2**adjusted_exponent
        won't be representable as single-precision float: we need to do the
        multiplication in double.  [Note: just messing with the floating-point
        mantissa manually would be easier, but it's probably harder to make
        hardware-independent.]. */
    double scale = pow(2.0, adjusted_exponent);

    for (ssize_t k = 0; k < num_samples; k++) {
      double f = input[k * input_stride];
      int32_t i = (int32_t) (f * scale);
      assert(i == (int16_t) i);
      temp_space[k] = i;
    }
  } else if (conversion_exponent == 128) {
    /** adjusted_exponent will be representable, but we have a different risk
        here: conversion_exponent might have been truncated from 129.  We need
        to truncate to the range of int16_t when doing the conversion, otherwise
        there is a danger that FLT_MAX and numbers very close to it could become
        negative after conversion to int, since they'd be rounded to 32768.
    */
    float scale = pow(2.0, adjusted_exponent);
    for (ssize_t k = 0; k < num_samples; k++) {
      float f = input[k * input_stride];
      int32_t i = (int32_t) (f * scale);
      assert(i >= -32768 && i <= 32768);
      if (i >= 32768)
        i = 32767;
      temp_space[k] = i;
    }
  } else {
    /** The normal case; we should be here in 99.9% of cases. */
    float scale = pow(2.0, adjusted_exponent);
    for (ssize_t k = 0; k < num_samples; k++) {
      float f = input[k * input_stride];
      int32_t i = (int32_t) (f * scale);
      assert(i == (int16_t) i);
      temp_space[k] = i;
    }
  }

  int ret = lilcom_compress(temp_space, num_samples, 1,
                            output, num_bytes, output_stride,
                            lpc_order, bits_per_sample,
                            conversion_exponent);
  return ret;  /* 0 for success, 1 for failure, e.g. if lpc_order out of
                  range. */
}

int lilcom_decompress_float(
    const int8_t *input, ssize_t num_bytes, int input_stride,
    float *output, ssize_t num_samples, int output_stride) {
  if (num_bytes < 5 || input_stride == 0 || output_stride == 0 ||
      num_samples != lilcom_get_num_samples(input, num_bytes, input_stride)) {
    debug_fprintf(stderr, "[lilcom] Error in header, decompressing float\n");
    return 1;  /* Error */
  }
  /* Note: we re-use the output as the temporary int16_t array */
  int16_t *temp_array = (int16_t *) output;
  int temp_array_stride;
  if (output_stride == 1) {
    temp_array_stride = 1;
  } else {
    temp_array_stride = output_stride * (sizeof(float) / sizeof(int16_t));
  }
  int conversion_exponent;
  int ans = lilcom_decompress(input, num_bytes, input_stride,
                              temp_array, num_samples, temp_array_stride,
                              &conversion_exponent);
  if (ans != 0)
    return ans;  /* the only other possible value is 1, actually. */

  assert(conversion_exponent >= -127 && conversion_exponent <= 128);

  int adjusted_exponent = conversion_exponent - 15;

  if (adjusted_exponent < -126 || conversion_exponent >= 128) {
    /** Either adjusted_exponent itself is outside the range representable in
        single precision, or there is danger of overflowing single-precision
        range after multiplying by the integer values, so we do the conversion
        in double.  We also check for things that exceed the range representable
        as float, although this is only necessary for the case when
        conversion_exponent == 128. */
    double scale = pow(2.0, adjusted_exponent);

    for (ssize_t k = num_samples - 1; k >= 0; k--) {
      int16_t i = temp_array[k * temp_array_stride];
      double d = i * scale;
      if (lilcom_abs(d) > FLT_MAX) {
        if (d > 0) d = FLT_MAX;
        else d = -FLT_MAX;
      }
      output[k * output_stride] = (float) d;
    }
  } else {
    float scale = pow(2.0, adjusted_exponent);
    for (ssize_t k = num_samples - 1; k >= 0; k--) {
      int16_t i = temp_array[k * temp_array_stride];
      output[k * output_stride] = i * scale;
    }
  }
  return 0;  /* Success */
}

#ifdef LILCOM_TEST

#include <math.h>
#include <stdio.h>

/** This function does nothing; it only exists to check that
    various relationships between the #defined constants are satisfied.
*/
static inline int lilcom_check_constants() {
  /* At some point we use 1 << (LPC_APPLY_LEFT_SHIFT + 16) as an int32 and we
     want to preclude overflow. */
  assert(LPC_APPLY_LEFT_SHIFT + 16 < 31);
  assert(STAGING_BLOCK_SIZE > MAX_POSSIBLE_NBITS+1);
  assert((STAGING_BLOCK_SIZE & (STAGING_BLOCK_SIZE - 1)) == 0);
  assert(NBITS_BUFFER_SIZE > MAX_POSSIBLE_NBITS+1);
  assert((LPC_ROLLING_BUFFER_SIZE - 1) * AUTOCORR_BLOCK_SIZE > 24);
  assert(MAX_LPC_ORDER >> LPC_ORDER_BITS == 0);
  assert(MAX_LPC_ORDER % 2 == 0);
  assert(LPC_EST_LEFT_SHIFT + (AUTOCORR_EXTRA_VARIANCE_EXPONENT+1)/2 <= 31);
  assert(LPC_APPLY_LEFT_SHIFT + 16 < 31);
  assert((30 + AUTOCORR_LEFT_SHIFT + LOG_AUTOCORR_BLOCK_SIZE + (AUTOCORR_DECAY_EXPONENT-1))
         < 61);
  assert((AUTOCORR_BLOCK_SIZE & (AUTOCORR_BLOCK_SIZE-1)) == 0);  /** Power of 2. */
  assert(AUTOCORR_BLOCK_SIZE > MAX_LPC_ORDER);
  assert(LPC_COMPUTE_INTERVAL % AUTOCORR_BLOCK_SIZE == 0);
  assert(AUTOCORR_BLOCK_SIZE >> LOG_AUTOCORR_BLOCK_SIZE == 1);


  /** the following inequalitites are assumed in some code where we left shift
   * by the difference.*/
  assert(AUTOCORR_LEFT_SHIFT >= AUTOCORR_EXTRA_VARIANCE_EXPONENT);
  assert(AUTOCORR_LEFT_SHIFT >= AUTOCORR_DECAY_EXPONENT);

  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  /* The y < x / 2 below just means "y is much less than x". */
  assert(LPC_COMPUTE_INTERVAL < (AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT) / 2);
  assert((NBITS_BUFFER_SIZE-1)*2 > 12);
  assert((NBITS_BUFFER_SIZE & (NBITS_BUFFER_SIZE-1)) == 0);  /* Power of 2. */
  assert(NBITS_BUFFER_SIZE > (12/2) + 1); /* should exceed maximum range of exponents,
                                                divided by  because that's how much they can
                                                change from time to time.  The + 1 is
                                                in case I missed something. */

  assert((NBITS_BUFFER_SIZE & (NBITS_BUFFER_SIZE-1)) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE % AUTOCORR_BLOCK_SIZE == 0);
  assert((SIGNAL_BUFFER_SIZE & (SIGNAL_BUFFER_SIZE - 1)) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE > AUTOCORR_BLOCK_SIZE + NBITS_BUFFER_SIZE + MAX_LPC_ORDER);

  return 1;
}

/**
   Computes the SNR, as a ratio, where a is the signal and (b-a) is the noise.
 */
float lilcom_compute_snr(ssize_t num_samples,
                         int16_t *signal_a, int stride_a,
                         int16_t *signal_b, int stride_b) {
  int64_t signal_sumsq = 0, noise_sumsq = 0;
  for (ssize_t i = 0; i < num_samples; i++) {
    int64_t a = signal_a[stride_a * i], b = signal_b[stride_b * i];
    signal_sumsq += a * a;
    noise_sumsq += (b-a)*(b-a);
  }
  /** return the answer in decibels.  */
  if (signal_sumsq == 0.0 && noise_sumsq == 0.0)
    return -10.0 * log10(0.0); /* log(0) = -inf. */
  else
    return -10.0 * log10((noise_sumsq * 1.0) / signal_sumsq);
}

/**
   Computes the SNR, as a ratio, where a is the signal and (b-a) is the noise.
 */
float lilcom_compute_snr_float(ssize_t num_samples,
                               float *signal_a, int stride_a,
                               float *signal_b, int stride_b) {
  double signal_sumsq = 0, noise_sumsq = 0;
  for (ssize_t i = 0; i < num_samples; i++) {
    double a = signal_a[stride_a * i], b = signal_b[stride_b * i];
    signal_sumsq += a * a;
    noise_sumsq += (b-a)*(b-a);
  }
  /** return the answer in decibels.  */
  if (signal_sumsq == 0.0 && noise_sumsq == 0.0)
    return -10.0 * log10(0.0); /* log(0) = -inf. */
  else
    return -10.0 * log10((noise_sumsq * 1.0) / signal_sumsq);
}

void lilcom_test_compress_sine() {
  int16_t buffer[999];
  int lpc_order = 10;
  for (int i = 0; i < 999; i++)
    buffer[i] = 700 * sin(i * 0.01);

  /* TODO: use LILCOM_MAX_BPS */
  for (int bits_per_sample = LILCOM_MAX_BPS; bits_per_sample >= LILCOM_MIN_BPS; bits_per_sample--) {
    printf("Bits per sample = %d\n", bits_per_sample);
    int exponent = -15, exponent2;
    ssize_t num_bytes = lilcom_get_num_bytes(999, bits_per_sample);
    int8_t *compressed = (int8_t*)malloc(num_bytes);
    int ret = lilcom_compress(buffer, 999, 1,
                              compressed, num_bytes, 1,
                              lpc_order, bits_per_sample, exponent);
    assert(!ret);
    int sum = 0;
    for (int32_t i = 0; i < num_bytes; i++)
      sum += compressed[i];
    printf("hash = %d\n", sum);
    int16_t decompressed[999];
    if (lilcom_decompress(compressed, num_bytes, 1,
                          decompressed, 999, 1,
                          &exponent2) != 0) {
      debug_fprintf(stderr, "Decompression failed\n");
    }
    assert(exponent2 == exponent);
    fprintf(stderr, "Bits-per-sample=%d, sine snr (dB) = %f\n",
            bits_per_sample,
            lilcom_compute_snr(999 , buffer, 1, decompressed, 1));
    free(compressed);
  }
}


void lilcom_test_compress_maximal() {
  /** this is mostly to check for overflow when computing autocorrelation. */

  for (int bits_per_sample = 4; bits_per_sample <= 16; bits_per_sample++) {
    for (int lpc_order = 0; lpc_order <= MAX_LPC_ORDER; lpc_order++) {
      if (lpc_order > 4 && lpc_order % 2 == 1)
        continue;
      for (int n = 0; n < 2; n++) {
        int16_t buffer[4096];
        ssize_t num_samples = 4096;
        int stride = 1 + (bits_per_sample % 2);
        num_samples /= stride;

        if (n == 0) {
          for (int i = 0; i < num_samples; i++) {
            if (i < 3000 || i % 100 < 50)
              buffer[i*stride] = -32768;
            else
              buffer[i*stride] = 32767;
          }
        } else {
          for (int i = 0; i < num_samples; i++) {
            if (i < 3000 || i % 100 < 50)
              buffer[i*stride] = 32767;
            else
              buffer[i*stride] = -32768;
          }
        }

        /* Stride is either 1 or 2. */
        int byte_stride = 1 + (bits_per_sample + lpc_order) % 2;

        ssize_t num_bytes = lilcom_get_num_bytes(num_samples, bits_per_sample);
        printf("num_samples=%d, bits_per_sample=%d, num_bytes = %d\n",
               (int)num_samples, bits_per_sample, (int)num_bytes);
        int8_t *compressed = (int8_t*)malloc(num_bytes * byte_stride);
        int exponent = -15, exponent2;
        lilcom_compress(buffer, num_samples, stride,
                        compressed, num_bytes, byte_stride,
                        lpc_order, bits_per_sample, exponent);
        int decompressed_stride = 1 + (bits_per_sample % 3);
        int16_t *decompressed = (int16_t*)malloc(num_samples * decompressed_stride *
                                                 sizeof(int16_t));
        if (lilcom_decompress(compressed, num_bytes, byte_stride,
                              decompressed, num_samples, decompressed_stride,
                              &exponent2) != 0) {
          fprintf(stderr, "Decompression failed\n");
        }
        assert(exponent2 == exponent);
        fprintf(stderr, "Minimal snr (dB) = %f\n",
                lilcom_compute_snr(num_samples, buffer, stride,
                                   decompressed, decompressed_stride));
        free(compressed);
        free(decompressed);
      }
    }
  }
}


void lilcom_test_compress_sine_overflow() {
  for (int bits_per_sample = 4;
       bits_per_sample <= 16; bits_per_sample += 2) {
    ssize_t num_samples = 1001 + bits_per_sample;
    int16_t *buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));
    int lpc_order = 10;
    for (int i = 0; i < num_samples; i++)
      buffer[i] = 65535 * sin(i * 0.01);

    int exponent = -15, exponent2;
    ssize_t num_bytes = lilcom_get_num_bytes(num_samples, bits_per_sample);
    int8_t *compressed = (int8_t*)malloc(num_bytes);
    lilcom_compress(buffer, num_samples, 1,
                    compressed, num_bytes, 1,
                    lpc_order, bits_per_sample, exponent);
    int16_t *decompressed = (int16_t*) malloc(num_samples * sizeof(int16_t));
    if (lilcom_decompress(compressed, num_bytes, 1,
                          decompressed, num_samples, 1,
                          &exponent2) != 0) {
      fprintf(stderr, "Decompression failed\n");
    }
    assert(exponent2 == exponent);
    fprintf(stderr, "Sine-overflow snr (dB) at bps=%d is %f\n",
            bits_per_sample,
            lilcom_compute_snr(num_samples , buffer, 1, decompressed, 1));
    free(buffer);
    free(compressed);
    free(decompressed);
  }
}


/**
   Note on SNR's.  A absolute (and not-tight) limit on how negative the SNR
   for a signal that originally came from floats can be is around -90: this
   is what you get from the quantization to int32.  For a signal in the range
   [-1,1] quantized by multiplying by 32768 and rounding to int, the quantization error
   will exceed 0.5 after multiplication, which would be 2^-16 in the original floating
   point range.

   The largest energy of original signal could have is 1 per frame, and the average
   energy quantization would be... well the largest it could be is 2^{-32}, but
   on average it would be 2^{-34}: the extra factor of 1/4 comes from an integral
   that I haven't done all the details of (distribution of errors.)

   Converting to decibels:
     perl -e 'print 10.0 * log(2**(-34)) / log(10);'
   -102.35019852575

   So around -102 dB.
*/

void lilcom_test_compress_float() {
  float buffer[1000];

  int lpc_order = 5;

  for (int bits_per_sample = 4; bits_per_sample <= 16; bits_per_sample++) {
    int stride = 1 + (bits_per_sample % 3);
    int num_samples = 1000 / stride;
    for (int exponent = -140; exponent <= 131; exponent++) {
      /* Note: 130 and 131 are special cases, for NaN and FLT_MAX.. */
      double scale = pow(2.0, exponent); /* Caution: scale may be inf. */
      if (exponent == 130)
        scale = scale - scale;  /* NaN. */

      for (int i = 0; i < num_samples; i++) {
        buffer[i*stride] = 0.5 * sin(i * 0.01) + 0.2 * sin(i * 0.1) + 0.1 * sin(i * 0.25);
        buffer[i*stride] *= scale;
      }

      if (exponent == 131) {
        /* Replace infinities with FLT_MAX. */
        for (int i = 0; i < num_samples; i++) {
          if (buffer[i*stride] - buffer[i*stride] != 0) {
            buffer[i*stride] = (buffer[i*stride] > 0 ? FLT_MAX : -FLT_MAX);
          }
        }
      }

      ssize_t num_bytes = lilcom_get_num_bytes(num_samples, bits_per_sample);
      int byte_stride = 1 + (bits_per_sample + lilcom_abs(exponent)) % 3;
      int8_t *compressed = (int8_t*)malloc(num_bytes * byte_stride);
      int16_t temp_space[num_samples];
      int ret = lilcom_compress_float(buffer, num_samples, stride,
                                      compressed, num_bytes, byte_stride,
                                      lpc_order, bits_per_sample,
                                      temp_space);
      if (ret) {
        fprintf(stderr, "float compression failed for exponent = %d (this may be expected), max abs float value = %f\n",
                exponent,  max_abs_float_value(buffer, num_samples, stride));
        free(compressed);
        continue;
      }

      float decompressed[num_samples];
      if (lilcom_decompress_float(compressed, num_bytes, byte_stride,
                                  decompressed, num_samples, 1) != 0) {
        fprintf(stderr, "Decompression failed.  This is not expected if compression worked.\n");
        exit(1);
      }

      fprintf(stderr, "For data-generation exponent=%d, bits-per-sample=%d, {input,output} max-abs-float-value={%f,%f}, floating-point 3-sine snr = %fdB\n",
              exponent, bits_per_sample,
              max_abs_float_value(buffer, num_samples, stride),
              max_abs_float_value(decompressed, num_samples, 1),
              lilcom_compute_snr_float(num_samples, buffer, stride,
                                       decompressed, 1));
      free(compressed);
    }
  }
}


void lilcom_test_compute_conversion_exponent() {
  for (int i = 5; i < 100; i++) {
    float mantissa = i / 100.0;
    for (int j = -150; j <= 133; j++) {
      float exponent_factor = pow(2.0, j); /* May cause inf due to overflow. */
      if (j == 130) {
        exponent_factor = exponent_factor - exponent_factor;  /* Should cause NaN. */
      } else if (j == 131) {
        mantissa = 0.0;
      }
      float product = mantissa * exponent_factor;
      if (j == 132)
        product = FLT_MAX;

      int exponent = compute_conversion_exponent(product);
      if (product - product != 0) {  /* inf or NaN */
        assert(exponent == -256);
      } else if (product == 0.0) {
        assert(exponent == 0);
      } else {
        assert(exponent >= -127 && exponent <= 128);
        /* Note: the comparison below would be done in double since pow returns
         * double. */
        if (exponent < 128) {
          assert(product < (65535.0 / 65536) * pow(2.0, exponent));
        } else {
          assert(product < pow(2.0, exponent));
        }
        if (exponent > -127) {
          /* check that a exponent one smaller wouldn't have worked. */
          assert(product >  (65535.0 / 65536) * pow(2.0, exponent - 1));
        }  /* if -127 it might have been truncated. */
      }
    }
  }
  assert(compute_conversion_exponent(FLT_MAX) == 128);
  assert(compute_conversion_exponent(0) == 0);
  /** Note: FLT_MIN is not really the smallest float, it's the smallest
      normalized number, so multiplying by 1/8 is meaningful.  */
  assert(compute_conversion_exponent(FLT_MIN * 0.125) == -127);
}

void lilcom_test_get_max_abs_float_value() {
  float array[101];  /* only use values 0..99 as part of array. */
  for (int i = 0; i < 100; i++)
    array[i] = 0.0;

  array[50] = -100.0;
  assert(max_abs_float_value(array, 10, 10) == lilcom_abs(array[50]));

  array[50] = pow(2.0, 129);  /* should generate infinity. */

  assert(max_abs_float_value(array, 100, 1) == lilcom_abs(array[50]));

  array[50] = array[50] - array[50];  /* should generate nan. */

  /** The following checks that the return value is NaN. */
  float ans = max_abs_float_value(array, 100, 1);
  assert(ans != ans);  /* Check for NaN. */

  /** Positions near the end are special, need to test separately. */
  array[97] = array[50];
  array[50] = 0.0;
  ans = max_abs_float_value(array, 99, 1);
  assert(ans != ans);  /* Check for NaN. */
  array[97] = 0.0;

  array[50] = 5;
  array[51] = -6;

  assert(max_abs_float_value(array, 100, 1) == lilcom_abs(array[51]));

  array[99] = 500.0;
  array[100] = 1000.0;  /* not part of the real array. */

  assert(max_abs_float_value(array, 100, 1) == lilcom_abs(array[99]));
}

int main() {
  lilcom_check_constants();
  lilcom_test_extract_mantissa();
  lilcom_test_compress_sine();
  lilcom_test_compress_maximal();
  lilcom_test_compress_sine_overflow();
  lilcom_test_compress_float();
  lilcom_test_compute_conversion_exponent();
  lilcom_test_get_max_abs_float_value();
  lilcom_test_encode_decode_signed();
  lilcom_test_encode_residual();
}
#endif
