#include "lilcom.h"
#include <assert>
#include <math.h>


/**
   The version number of the format.   Note: you can't change the various
   constants below without changing LILCOM_VERSION, because it will mess up the
   decompression.  That is: the decompression method requires exact
   compatibility with the compression method, w.r.t. how the LPC coefficients
   are computed.
*/
#define LILCOM_VERSION 1

/**
   Number of bytes in the header we use when compressing 16-bit to 8-bit ints.
   I don't expect this to ever be changed.
 */
#define LILCOM_HEADER_BYTES 4


/**
   max_lpc_order is the maximum allowed order of the linear prediction
   computation (used to set various array sizes).  Defines a limit on how large
   the user-specified LPC order may be.  May not be >= AUTOCORR_BLOCK_SIZE, or
   else we'd need to do extra work to handle edge cases near the beginning of
   the signal.
**/
#define MAX_LPC_ORDER 15


/** a value >= log_2 of MAX_LPC_ORDER+1.  Used in explanations,
    not currently in any code */
#define LPC_ORDER_BITS 4


/**
   The amount by which we shift left the LPC coefficients for the fixed-point
   representation; corresponds approximately to the precision (for LPC
   coefficients with magnitude close to 1).
*/
#define LPC_LEFT_SHIFT 23

/*
   An amount that we add to the zeroth autocorrelation coefficient to avoid
   divisions by zero, etc.  1 corresponds to a variation of +-1 sample, it's
   very small, just to deal with that zero issue.  (We also have another
   variance-smoothing method determined by AUTOCORR_EXTRA_VARIANCE_EXPONENT,
   which ensures we can bound the LPC coefficients).
*/
#define AUTOCORR_EXTRA_VARIANCE 1


/**
   Defines an extra amount that we add to autocorr[0], that's relative
   to its observed value.  Basically we multiply autocorr[0] by
     (1 + 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT})
   This helps give us bounds on the value of the LPC coefficients.

   Caution: if you increase this, some changes to assertions in the code,
   especially assertions in lilcom_compute_lpc, would be required (since we
   hardcode 8 as half of this value), and LPC_LEFT_SHIFT would have to be
   reduced by half of the amount by which you increased this.  Reducing this is
   safe, though.
 */
#define AUTOCORR_EXTRA_VARIANCE_EXPONENT 16


/**
   The amount by which we shift left the autocorrelation stats, to avoid
   roundoff error due to the decay.  Larger -> more accurate, but,
   as explained in the comment for LpcComputation::autocorr,
   we require

   (30 + AUTOCORR_LEFT_SHIFT + log(AUTOCORR_BLOCK_SIZE) + AUTOCORR_DECAY_EXPONENT)
     to not exceed 61.
   Currently that equals: 30 + 20 + 5 + 3 = 58  <= 61.  I'm giving
   it a little wiggle room there in case other parameters get changed
   while tuning.
 */
#define AUTOCORR_LEFT_SHIFT 20

/**
   We update the autocorrelation coefficients every this-many samples.  For times
   t < LPC_COMPUTE_INTERVAL we will also recompute the LPC coefficients every
   this-many samples (this makes the LPC coefficients converge to a reasonable
   value faster at the beginning of the signal.)

   Requirements:
      - The code in lilcom_update_autocorrelation() assumes that this is greater
        than MAX_LPC_ORDER.
      - Must be a power of 2, as  we may use bitwise tricks to compute modulus
        with it.
      - Must divide LPC_COMPUTE_INTERVAL exactly.
 */
#define AUTOCORR_BLOCK_SIZE 32
/**
   LOG_AUTOCORR_BLOCK_SIZE must be the log-base-2 of AUTOCORR_BLOCK_SIZE.
   Used in bitwise tricks for division.
 */
#define LOG_AUTOCORR_BLOCK_SIZE 5

/**
   AUTOCORR_DECAY_EXPONENT, together with AUTOCORR_BLOCK_SIZE, defines
   how fast we decay the autocorrelation stats; small values mean the
   autocorrelation coefficients move faster, large values mean
   they move slower.

   The approximate number of samples that we `remember` in the autocorrelation`
   computation is approximately AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT,
   which is currently 256.  Of course it's a decaying average, so we remember
   all samples to some extent.  This may be important to tune, for
   performance.
 */
#define AUTOCORR_DECAY_EXPONENT 3



/**
   Determines how frequently we update the LPC coefficients for large t.  We
   update the LPC coefficients every this-many samples for t >=
   LPC_COMPUTE_INTERVAL.  For t < LPC_COMPUTE_INTERVAL we update them every
   AUTOCORR_BLOCK_SIZE, to make sure the accuracy doesn't suffer too much for

   the first few samples of the signal.

   Must be a multiple of AUTOCORR_BLOCK_SIZE, and should be substantially
   smaller than  AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT to ensure
   freshness of the LPC coefficients.

   Smaller values will give 'fresher' LPC coefficients but will be slower,
   both in compression and decompression.  But the amount of work is not
   that great; every LPC_COMPUTE_INTERVAL samples we do work equivalent
   to about `lpc_order` samples, where lpc_order is a user-specified value
   in the range [1, MAX_LPC_ORDER].
 */
#define LPC_COMPUTE_INTERVAL 64


/**
   This rolling-buffer size determines How far back in time we keep the
   exponents we used to compress the signal.  It determines how far it's
   possible for us to backtrack when we encounter out-of-range values and need
   to increase the exponent.  8 is more than enough, since the most we'd ever
   need to add to the exponent is +12 (the exponent can never exceed 12); the
   exponent can increase by at most 2 for each sample; and (8-1) * 2 > 12 (if we
   have 8 in the buffer, the furthest we can go back in time is t-7, which is
   why we have (8-1) above).

   It must be a power of 2 due to a trick used to compute a modulus.
 */
#define EXPONENT_BUFFER_SIZE 8

/**
   SIGNAL_BUFFER_SIZE determines the size of a rolling buffer containing
   the history of the compressed version of the signal, to be used while
   compressing.  It must be a multiple of AUTOCORR_BLOCK_SIZE.  It must
   also satisfy:

    SIGNAL_BUFFER_SIZE >
       AUTOCORR_BLOCK_SIZE + EXPONENT_BUFFER_SIZE + MAX_LPC_ORDER

    (currently: 128 > 32 + 8 + 15).

   That is: it needs to store a whole block's worth of data, plus the farthest
   we might backtrack, plus enough context to process the first sample of the
   block.  This backtracking is because there may be situations where we need to
   recompute the autocorrelation coefficients of a block if backtracking
   causes us to revisit it.

   It must also be a power of 2, due to tricks we use when computing modulus.

   Every time we get to the beginning of the buffer we need to do a little extra
   work to place the history before the start of the buffer, that's why we don't
   just make this 64 (i.e. two blocks' worth).  The reason for not using a huge
   value is to save memory and be kind to the cache.
 */
#define SIGNAL_BUFFER_SIZE 128



/**
   This contains information related to the computation of the linear prediction
   coefficients (LPC).
 */
struct LpcComputation {
  /**
     'autocorr[i]' contains a weighted sum of (s[j] * s[j - i]) << AUTOCORR_LEFT_SHIFT; we use a
     'forgetting factor' of AUTOCORR_DECAY_EXPONENT, i.e. each time we process a
     block (of size AUTOCORR_BLOCK_SIZE) we multiply by (1 +
     2^-{AUTOCORR_DECAY_EXPONENT}).  The elements of the sum cannot exceed
     (2^15)^2 = 2^30, shifted left AUTOCORR_LEFT_SHIFT bits, so the elements of
     the sum are <= 2^53.  The sum will be no
     greater than this times around AUTOCORR_BLOCK_SIZE *
     2^AUTOCORR_DECAY_EXPONENT (the approximate number of terms added together)
     = 2^8, so the sum would be <= 2^(50+8) = 2^58.  The most this could be
     without overflow is around 2^61.  */
  int64_t autocorr[MAX_LPC_ORDER + 1];
  /*
     max_exponent is the smallest number >= 0 such that autocorr[0] >>
     max_exponent == 0.  (Note: autocorr[0] is nonnegative).  This implies that autocorr[0] < 2^{max_exponent}.
     It is used in the fixed-point arithmetic used in lilcom_compute_lpc(), to
     shift the coefficients to a 'nice' range.

     Note: is is guaranteed that for no 0 < i <= lpc_order,
     abs(autocorr[i]) > autocorr[0], so this also gives a bound on the magnitudes
     of the other autocorrelation coefficients.  */
  int32_t max_exponent;
  /*
     Contains the LPC coefficients times 2^LPC_LEFT_SHIFT.  These are
     initialized to [1 0 0 0.. ] at the start of the utterance (so the
     prediction is just the previous sample).  Of course that's represented as
     [1<<LPC_LEFT_SHIFT 0 0 0 ...]

     If we have processed at least one block of AUTOCORR_BLOCK_SIZE samples and
     have called lilcom_compute_lpc, this will contain the coefficients
     estimated from there.  Only elements 0 through lpc_order - 1 will be valid.
     (lpc_order is passed directly into functions dealing with this object).  */
  int32_t lpc_coeffs[MAX_LPC_ORDER];
};


/**
   Initialize an LpcComputation object with the default parameter that we have
   at the start of the file.  This corresponds to all-zero autocorrelation
   stats, max_exponent = 0, and lpc_coeffs that correspond to [1.0 0 0 0].
*/
static void lilcom_init_lpc(LpcComputation *lpc) {
  for (int i = 0; i <= MAX_LPC_ORDER; i++)
    coeffs->autocorr[i] = 0;
  /* The LPC coefficientss are stored shifted left by LPC_LEFT_SHIFT, so this
     means the 1st coeff is 1.0 and the rest are zero-- meaning, we start
     prediction from the previous sample.  */
  coeffs->lpc_coeffs[0] = 1 << LPC_LEFT_SHIFT;
  for (int i = 1; i < MAX_LPC_ORDER; i++)
    coeffs->lpc_coeffs[i] = 0;

  coeffs->max_exponent = 0;
}

/**
   Updates the autocorrelation stats in 'coeffs', by scaling down the previously
   computed stats slightly and adding one block's worth of new autocorrelation
   data.  This is called every AUTOCORR_BLOCK_SIZE samples.

     @param [in,out]  autocorr   The statistics to be updated (i.e. scaled down
                           and then added to).
     @param [in] lpc_order   The LPC order, must be in [1..MAX_LPC_ORDER]
     @param [in] is_first_block  Nonzero if this is the first block of
                           signal (meaning: `signal` points to time t=0;a
                           this would mean that we are computing the
                           autocorrelation coefficients for block 1).
                           This affects what happens for t earlier than the
                           start of the block (we'll ignore preceding
                           values if is_first_block != 0).
     @param [in] signal    Pointer to the signal at the start of the block
                           (Note: the LPC coefficients used at the
                           i'th block depend on the autocorr stats for
                           block i-1, so 'signal' will point to the
                           start of block i-1.
                           In all cases we will access values signal[0]
                           through signal[AUTOCORR_BLOCK_SIZE-1].  If
                           is_first_block == 0 (not the first block),
                           we will also look at elements signal[-MAX_LPC_ORDER]
                           through signal[-1].
*/
inline static void lilcom_update_autocorrelation(
    LpcComputation *lpc, int lpc_order, int is_first_block,
    const int16_t *signal) {

  // 'temp_autocorr' will contain the raw autocorrelation stats without the
  // shifting left by AUTOCORR_LEFT_SHIFT (we'll do that at the end, to save an
  // instruction in the inner loop).
  int64_t temp_autocorr[MAX_LPC_ORDER + 1];
  int i;

  if (!is_first_block) {
    // Process any terms involving the history samples that are prior to the
    // start of the block.
    //
    // The samples that come from the previous block need to be scaled down
    // slightly, in order to be able to guarantee that no element of
    // lpc->autocorr is greater than the zeroth element.  The way we do this is
    // to process them before we scale down the elements of lpc->autocorr.
    // Note: we add these terms directly to lpc->autocorr without using the
    // temporary storage of `temp_autocorr`.
    for (i = 0; i < lpc_order; i++) {
      int64_t signal_i = signal[i];
      int j;
      for (j = i + 1; j <= lpc_order; j++) {
        lpc->autocorr[j] += (signal[i - j] * signal_i) << AUTOCORR_LEFT_SHIFT;
      }
    }
  }

  // Scale down the current data slightly.  This is an exponentially decaying
  // sum, but done at the block level not the sample level.
  // Also zero `temp_autocorr`.

  for (i = 0; i <= lpc_order; i++) {
    lpc->autocorr[i] -= (lpc->autocorr[i] >> AUTOCORR_DECAY_EXPONENT);
    temp_autocorr[i] = 0;
  }

  // This first loop handles the first few samples, that have edge effects.  We
  // need to exclude negative-numbered samples from the sum.  In the first block
  // they don't exist (conceptually, they are zero); in subsequent blocks we
  // need to scale them down according to AUTOCORR_DECAY_EXPONENT in order to be
  // able to guarantee that the zeroth coefficient is the largest one; and this
  // has been handled above; search for `if (!is_first_block)`.  [actually it's
  // not clear that we need this property, but it makes certain things easier to
  // reason about.]
  for (i = 0; i < lpc_order; i++) {
    int32_t signal_i = signal[i];
    int j;
    for (j = 0; j <= i; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
  }
  // OK, now we handle the samples that aren't close to the boundary.
  // currently, i == lpc_order.
  for (; i < AUTOCORR_BLOCK_SIZE; i++) {
    int32_t signal_i = signal[i];
    for (int j = 0; j <= lpc_order; j++) {
      temp_autocorr[j] += signal[i - j] * signal_i;
    }
  }
  for (int j = 0; j <= lpc_order; j++) {
    lpc->autocorr[j] += temp_autocorr[j] << AUTOCORR_LEFT_SHIFT;
  }

  // The next statement takes care of the smoothing to make sure that the
  // autocorr[0] is nonzero, and adds extra noise determined by
  // AUTOCORR_EXTRA_VARIANCE_EXPONENT and the signal energy, which will allow us
  // to put a bound on the value of the LPC coefficients so we don't need to
  // worry about integer overflow.
  // (Search for: "NOTE ON BOUNDS ON LPC COEFFICIENTS")
  lpc->autocorr[0] +=
      ((int64_t)(AUTOCORR_BLOCK_SIZE*AUTOCORR_EXTRA_VARIANCE)<<AUTOCORR_LEFT_SHIFT) +
      temp_autocorr[0] << (AUTOCORR_LEFT_SHIFT - AUTOCORR_EXTRA_VARIANCE_EXPONENT);


  // We will have copied the max_exponent from the previous LpcComputation
  // object, and it will usually already have the correct value.  Return
  // immediately if so.
  int exponent = lpc->max_exponent;
  int64_t abs_autocorr_0 =
      (lpc->autocorr[0] > 0 ? lpc->autocorr[0] : -lpc->autocorr[0]);
  assert(abs_autocorr_0 != 0);
  if ((abs_autocorr_0 >> exponent) == 1) {
    // max_exponent has the correct value.
    return;
  }
  while ((abs_autocorr_0 >> exponent) == 0)
    exponent--;
  while ((abs_autocorr_0 >> exponent) > 1)
    exponent++;
  assert((abs_autocorr >> exponent) == 1);
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
    could get).  So when stored as integers with LPC_LEFT_SHIFT (currently
    23) as the exponent, they cannot exceed 2^31.  That's good because
    we store the LPC coefficients as int32_t, so 31 is the maximum
    we can allow.

    SUMMARY: the absolute value of the LPC coefficients must be less than 2^8.
 */



/**
   struct CompressionState contains the state that we need to maintain
   while compressing a signal; it is passed around by functions
   that are compressing the signal.
 */
struct CompressionState {

  /** The LPC order, a value in [1..MAX_LPC_ORDER].  User-specified.  */
  int32_t lpc_order;

  /**
     'lpc_computations' is to be viewed as a circular buffer of size 2,
     containing the autocorrelation and LPC coefficients.  Define the
     block-index b(t), a function of t, as:

         b(t) = t / AUTOCORR_BLOCK_SIZE,

     rounding down, of course.  The LPC coefficients we use to
     predict sample t will be those taken from lpc_computations[b(t) % 2].

     Whenever we process a t value that's a multiple of AUTOCORR_BLOCK_SIZE and
     which is nonzero, we will use the preceding block of AUTOCORR_BLOCK_SIZE
     coefficients of the signal (and, if present, `lpc_order` samples of context
     preceding that) to update the autocorrelation coefficients and possibly
     also the LPC coefficients (see docs for LPC_COMPUTE_INTERVAL).
  */
  LpcComputation lpc_computations[2];


  /**
     `exponents` is a circular buffer of the exponent values used to compress
     the signal; the exponent used at time t is stored in exponents[t %
     EXPONENT_BUFFER_SIZE].  Note: the only the differences between the
     exponents are stored in the code; define the difference in the exponent
     at time t as:

            d(t) = e(t) - e(t-1),

     where d(t) is in the range [-1..2].  We store (d(t) + 1) in
     the lowest-order two bits of the compressed code.  This buffer is
     needed primarily to deal with backtracking (which is what happens
     when the limitation of d(t) to the range [-1..2] means we can't
     use an exponent large enough to encode the signal).
  */
  int exponents[EXPONENT_BUFFER_SIZE];

  /**
    `uncompressed_signal` is the compressed-and-then-uncompressed version of the
     input signal.  We need to keep track of this because the uncompressed
     version of the is used to compute the LPC coefficients (this ensures that
     we can compute them in exactly the same way when we decompress, so we don't
     have to transmit them).

     The signal at time t is located at

       uncompressed_signal[MAX_LPC_ORDER + (t % SIGNAL_BUFFER_SIZE)]

     The reason for the extra MAX_LPC_ORDER elements is so that we can ensure,
     when we roll around to the beginning of the buffer, that we have a place to
     put the recent history (the previous `lpc_order` samples).  This keeps the
     code of lilcom_update_autocorrelation from having to loop around
  */
  int16_t uncompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE];


  /**  The input signal that we are compressing  */
  const int16_t *input_signal;
  /**  The stride associated with `input_signal`; normally 1 */
  int input_signal_stride;

  /** The compressed code that we are generating, one byte per sample.  This
      pointer does *not* point to the start of the header (it has been shifted
      forward by 4); the code for the t'th signal value is located at
      compressed_code[t].  */
  int8_t *compressed_code;
  /**  The stride associated with `compressed_code`; normally 1 */
  int compressed_code_stride;
};



// returns the sign of 'val', i.e. +1 if is is positive, -1 if
// it is negative, and 0 if it is zero.
inline int lilcom_sgn(int val) {
  return (0 < val) - (val < 0);
}

#define lilcom_abs(a) ((a) > 0 ? (a) : -(a))

/**
   Computes the least exponent (subject to a caller-specified floor) which
   is sufficient to encode (an approximation of) this residual; also
   computes the associated mantissa.

      @param [in] residual  The residual that we are trying to encode,
                     meaning: the observed value minus the value that was
                     predicted by the linear prediction.  This is a
                     difference of int16_t's, but such differences cannot
                     always be represented as int16_t, so it's represented
                     as an int32_t.
      @param [in] predicted   The predicted sample (so the residual
                     is the observed sample minus this).  The only reason
                     this needs to be specified is to detect situations
                     where, due to quantization effects, we would exceed
                     the range of int16_t; in those cases, we need to
                     reduce the magnitude of the mantissa to stay within
                     the allowed range.
       @param [in] min_exponent  A caller-supplied floor on the exponent;
                     must be in the range [0, 12].  This function will
                     never return a value less than this.  min_exponent
                     will normally be the previous sample's exponent
                     minus 1, but may be more than that if we are
                     backtracking.
        @param [out] mantissa  This function will write an integer in
                     the range [-32, 31] to here, such that
                     (mantissa << exponent) is a close approximation
                     of `residual` and satisfies the property that
                     `predicted + (mantissa << exponent)` does not
                     exceed the range of int16_t.
        @param [out] next_compressed_value  The next compressed value
                     will be written to here; this will be an approxmation
                     to `predicted_value + residual`.

        @return  Returns the value in the range [min_exponent..12] which,
                 together with `mantissa`, provides a close approximation
                 to `residual` while not allowing the next sample
                 to exceed the range of int16_t.

                 The intention of this function is to return the exponent in the
                 range [min_exponent..12] which gives the closest approximation
                 to `residual`, while choosing the lower exponent in case of
                 ties.  This is largely what it does, although it may not always
                 do so in situations where we needed to modify the mantissa to
                 not exceed the range of int16_t.  The details of
                 how the integer mantissa is chosen (especially w.r.t. rounding
                 and ties) is explained in a comment inside the function.


   The following explains some of how this function operates internally.

   Define the exact mantissa m(e), which is a function of the exponent e,
   as:
            m(e) =   (y_uncomp - y_pred_int) / (2^e),
   viewed as an exact mathematical expresion, not an integer.
   We want to return a value of e such that

     -33.0 <= m(e) <= 31.5.

   This inequality ensures that there will be no loss of accuracy by choosing e
   instead of e+1 as the exopnent.  (If we had a larger exponent, the closest
   points we'd be able to reach would be equivalent to m(e) = -34 or +32; and if
   m(e) satisfies the inequality above we'd have no loss of precision by using e
   rather than e+1 as the exponent.  We can express the above in integer math as:

     (-66 << e) <= (y_uncomp - y_pred_int) * 2 <= (63 << e)

   and notice that we have multiplied by 2 so that we don't have fractional
   expressions (the 31.5); we can't shift left by e-1 because that might
   be negative.

   [Note on the range of the exponent, and why it could be as large
   as 12.  The largest-magnitude residual is 65535, which is
   13767 - (-13768).  This could be most closely represented by
   +65536, which would be represented as 16 << 12.]

   Define the actual mantissa am(e) as just m(e), rounded to the closest
   integer, rounding towards zero in case of ties (and, as a special case,
   if m(e) is -33, letting am(e) be -32).
*/
inline static int least_exponent(int32_t residual,
                                 int16_t predicted,
                                 int min_exponent,
                                 int *mantissa,
                                 int16_t *next_compressed_value) {
  assert (((uint32_t)min_exponent) <= 10);
  int exponent = min_exponent,
      residual2 = residual * 2,
      minimum = -66 << exponent,
      maximum = 63 << exponent;
  while (residual2 < minimum || residual2 > maximum) {
    minimum *= 2;
    maximum *= 2;
    exponent++;
  }
  assert(exponent <= 12);

  {
    // This code block computes 'mantissa', the integer mantissa which we call
    // m(e) in the math above, and which should be a value in the range
    // [-32, 31].
    //
    // m(e) is the result of rounding (residual / (float)2^exponent)
    // to the nearest integer, rounding towards zero in case of ties; and
    // then, if the result is -33, changing it to -32.
    //
    // What we'd like to do, approximately, is to compute
    //
    //     mantissa = residual >> exponent
    //
    // where >> can be interpreted as division by 2^exponent; but we want some
    // control of the rounding behavior (that expression rounds down).  In
    // general we want to round to the closest, like round() for floating
    // point expressions; but we want some control of what happens in case of
    // ties.  Always rounding towards zero might possibly generate ringing
    // depending on the LPC coefficients, in certain circumstances, so
    // we want to round in a random direction (up or down).
    //
    // We'll use (predicted%2) as the source of randomness.  This will be
    // sufficiently random for loud signals (varying by more than about 1 or 2
    // from sample to sample); and for very quiet signals (magnitude close to 1)
    // we'll be exactly coding it anyway so it won't matter.
    //
    //
    //
    // Consider the expression:
    //
    //    mantissa = (residual*2 + offset) >> (exponent + 1)
    //
    //  and consider two possibilities for `offset`.
    // (a)  offset = (1<<exponent)
    // (b)  offset = ((1<<exponent) - 1)
    //
    // In case (a) it rounds up in case of ties (e.g. if the residual is 6 and
    // exponent is 2 so we're rounding to a multiple of 4).  In case (b)
    // it rounds down.  By using
    //  offset = ((1<<exponent) - (predicted&1))
    // we are randomly choosing (a) or (b) based on randomness in the
    // least significant bit of `predicted`.
    //
    // Note: the reason why we are using residual * 2, and exponent + 1
    // in these expressions, not just `residual` and `exponent`, is so that
    // if exponent == 0 it won't generate an unwanted shift.
    int offset = ((1 << exponent) - (predicted&1)),
        local_mantissa = (residual2 + offset) >> (exponent + 1);

    assert(local_mantissa >= -33 && local_mantissa <= 31);

    // We can't actually represent -33 in 6 bits, but we choose to retain this
    // exponent, in this case, because -33 is as close to -32 (which is
    // representable) as it is to -34 (which is the next closest thing
    // we'd get if we used a one-larger exponent).
    if (local_mantissa == -33)
      local_mantissa = -32;

    int32_t next_signal_value =
        ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);

    {
      // Just a check.  I will remove this block after it's debugged.
      // Checking that the error is in the expected range.
      int16_t error = (int16_t)(next_signal_value - residual);
      if (error < 0) error = -error;
      assert(error <= (1 << exponent) >> 1);
    }


    if (next_signal_value != (int16_t)next_signal_value) {
      // The next signal exceeds the range of int16_t; this can in principle
      // happen if the predicted signal was close to the edge of the range
      // [-32768..32767] and quantization effects took us over.  We need to
      // reduce the magnitude of the mantissa by one in this case.
      local_mantissa -= lilcom_sgn(local_mantissa);
      next_signal_value =
          ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);
      assert(next_signal_value == (int16_t)next_signal_value);
    }

    *next_compressed_value = next_signal_value;
    *mantissa = local_mantissa;
  }
  return exponent;
}


/**
   Computes the LPC coefficients via an Integerized version of the Durbin
   computation (that computes linear prediction coefficients given
   autocorrelation coefficients).

     `lpc` is a struct containing the autocorrelation coefficients and
              LPC coefficients (to be written).
     `lpc_order` is the linear prediction order, a number
              in [1..MAX_LPC_ORDER.  It is the number of taps of the IIR filter
              / the number of past samples we predict on.

   This code was adapted to integer arithmetic from the 'Durbin' function in
   Kaldi's feat/mel-computations.cc, which in turn was originaly derived from
   HTK.  Anyway, Durbin's algorithm is very well known.  The reason for doing
   this in integer arithmetic is mostly to ensure 100% reproducibility without
   having to get into system-dependent methods of setting floating point
   rounding modes.  Since the LPC coefficients are not transmitted, but are
   estimated from the preceding data, in order for the decompression to work we
   need to ensure that the coefficients are exactly the same as they were when
   we were compressing the data.
*/
void lilcom_compute_lpc(int lpc_order,
                        LpcComputation *lpc) {

  /**
     autocorr is just a copy of lpc->autocorr, but shifted to ensure that the
     absolute value of all elements is less than 1<<LPC_LEFT_SHIFT (but as large
     as possible given that constraint).  We don't need to know how much it was
     shifted right, because a scale on the autocorrelation doesn't affect the
     LPC coefficients.  */
  int32_t autocorr[MAX_LPC_ORDER];

  {
    int max_exponent = autocorr_in->max_exponent;
    assert(max_exponent >= AUTOCORR_LEFT_SHIFT &&
           (autocorr_in->autocorr[0]) >> (max_exponent - 1) == 1);
    if (max_exponent > LPC_LEFT_SHIFT) {
      /* shift right (the normal case) */
      int right_shift = max_exponent - LPC_LEFT_SHIFT;
      for (i = 0; i <= lpc_order; i++)
        autocorr[i] = autocorr_in->autocorr[i] >> right_shift;
    } else {
      int left_shift = LPC_LEFT_SHIFT - max_exponent;
      for (i = 0; i <= lpc_order; i++)
        autocorr[i] = autocorr_in->autocorr[i] << left_shift;
    }
    assert((autocorr[0] >> (LPC_LEFT_SHIFT - 1)) == 1);
    for (i = 1; i <= lpc_order; i++) {  /* TODO: remove this loop. */
      assert(lilcom_abs(autocorr[i]) <= autocorr[0]);
    }
  }

  /**
   'temp' is a temporary array used in the LPC computation, stored with
   exponent of LPC_LEFT_SHIFT (i.e. shifted left by that amount).  In Kaldi's
   mel-computations.cc is is called pTmp.  It seems to temporarily store the
   next iteration's LPC coefficients.  Note: the floating-point value of these
   coefficients is strictly less than 2^8 (see BOUNDS ON LPC COEFFICIENTS
   above).  The absolute values of the elements of 'temp' will thus be less than
   2^(LPC_LEFT_SHIFT+8) = 2^31.  i.e. it is guaranteed to fit into an int32.
  */
  int32_t temp[MAX_LPC_ORDER];


  int32_t E = autocorr[0];

  int j;
  for (int i = 0; i < lpc_order; i++) {
    /** ki will eventually be the next reflection coefficient, a value in [-1, 1], but
        shifted left by LPC_LEFT_SHIFT to represent it in fixed point.
        But after the following line it will represent a floating point
        number times 2 to the power 2*LPC_LEFT_SHIFT,
        so currently abs(ki) < 2^(LPC_LEFT_SHIFT+LPC_LEFT_SHIFT) = 2^46
      Original code: "float ki = autocorr[i + 1];"  */
    int64_t ki = autocorr[i + 1] << LPC_LEFT_SHIFT;

    for (j = 0; j < i; j++) {
      /** max magnitude of the terms added below is 2^(LPC_LEFT_SHIFT*2 + 8) = 2^54, i.e.
          the abs value of the added term is less than 2^54.
          ki still represents a floating-point number times 2 to the
          power 2*LPC_LEFT_SHIFT.
        The original floating-point code looked the same as the next line
        (not: ki has double the left-shift of the terms on the right).
      */
      ki += lpc->lpc_coeffs[j] * autocorr[i - j];
    }
    /** RE the current magnitude of ki:
        ki is a summation of terms, so add LPC_ORDER_BITS=4 to the magnitude of
        2^54 computed above, it's now bounded by 2^58 (this would bound its value
        at any point in the summation above).
        original code: "ki = ki / E;".   */
    ki = ki / E;
    /** At this point, ki is mathematically in the range [-1,1], since it's a
        reflection coefficient; and it is stored times 2 to the power LPC_LEFT_SHIFT, so its
        magnitude as an integer is <= 2^LPC_LEFT_SHIFT.  Check that it's less
        than 2^LPC_LEFT_SHIFT plus a margin to account for rounding errors.  */
    assert(lilcom_abs(ki) < (1<<LPC_LEFT_SHIFT + 1<<(LPC_LEFT_SHIFT - 8)));

    /**  Original code: "float c = 1 - ki * ki;"  */
    int64_t c = (((int64_t)1) << LPC_LEFT_SHIFT) - ((ki*ki) >> LPC_LEFT_SHIFT);

    /** c is the factor by which the residual has been reduced; mathematically
        it is always >= 0, but here it must be > 0 because of our smoothing of
        the variance via AUTOCORR_EXTRA_VARIANCE_EXPONENT and
        AUTOCORR_EXTRA_VARIANCE which means the residual can never get to
        zero.*/
    assert(c > 0);
    /** The original code did: E *= c;
        Note: the product is int64_t because c is int64_t, which is important
        to avoid overflow.
    */
    E = (int32_t)((E * c) >> LPC_LEFT_SHIFT);

    /** compute the new LP coefficients
        Original code did: pTmp[i] = -ki;
        Note: abs(temp[i]) <= 2^LPC_LEFT_SHIFT, since ki is in the range [-1,1]
        when viewed as a real number. */
    temp[i] = -((int32_t)ki);
    for (j = 0; j < i; j++) {
      /** The original code did:
          pTmp[j] = pLP[j] - ki * pLP[i - j - 1]
         These are actual LPC coefficients (computed for LPC order i + 1), so
         their magnitude is less than 2^(LPC_LEFT_SHIFT+8) = 2^31.

         The term on the RHS that we cast to int32_t is also bounded by
         2^31, because it's (conceptually) an LPC coefficient multiplied by a
         reflection coefficient ki with a value <= 1.  */
      temp[j] = lpc->lpc_coeffs[j] -
          (int32_t)((ki * lpc->lpc_coeffs[i - j - 1]) >> LPC_LEFT_SHIFT);
    }
    for (j = 0; j <= i; j++) {
      assert(lilcom_abs(temp[j]) < ((int64_t)1<<(LPC_LEFT_SHIFT + 8)));
      lpc->lpc_coeffs[j] = temp[j];  // magnitude less than 2^(LPC_LEFT_SHIFT+8) = 2^31.
    }
  }
  /** E > 0 because we added fake extra variance via
     AUTOCORR_EXTRA_VARIANCE_EXPONENT ahnd AUTOCORR_EXTRA_VARIANCE, so according
     to these stats the sample should never be fully predictable.  We'd like
     to assert that E <= autocorr[0] because even if the data is totally uncorrelated, we should
     never be increasing the predicted error vs. having no LPC at all.
     But I account for the possibility that in pathological cases, rounding
     errors might make this untrue.  */
  assert(E > 0 && E <= (autocorr[0] + autocorr[0] >> 10));
}

/**
   Compute predicted signal value based on LPC coeffiients for time t and the
   preceding state->lpc_order samples (treated as zero for negative time
   indexes).  Note: the preceding sample are taken from the buffer
   state->uncompressed_signal; we need to use the compressed-then-uncompressed
   values, not the original values, so that the LPC coefficients will
   be exactly the same when we decode.

         @param [in] state   Object containing variables for the compression
                             computation
         @param [in] t       Time index t >= 0 for which we need the predicted
                             value.
         @return             Returns the predicted value as an int16_t.
 */
inline int16_t lilcom_compute_predicted_value(
    CompressionState *state,
    int64_t t) {
  int64_t start_t = t - state->lpc_order;
  int32_t block_index = ((int32_t)t) >> LOG_AUTOCORR_BLOCK_SIZE;
  LpcComputation *lpc = &(state->lpc_computations[block_index % 2]);

  if (start_t < 0)
    start_t = 0;

  int32_t lpc_order = state->lpc_order;
  if (lpc_order > t)  // We should later make sure this if-statement doesn't
                      // have to happen.
    lpc_order = t;
  int64_t sum = 0;
  for (int32_t i = 0; i < lpc_order; i++) {
    // Note: ((int64_t)(t - 1 - i) & (SIGNAL_BUFFER_SIZE-1)) is just
    // (t-1-i) % SIGNAL_BUFFER_SIZE.  We know, incidentally, that (t-1-i)
    // is nonnegative, because i < lpc_order and lpc_order <= t., so i < t.
    sum += ((int64_t)lpc->lpc_coeffs[i]) * ((int64_t)(t - 1 - i) & (SIGNAL_BUFFER_SIZE-1));
  }
  // the lpc_coeffs were stored shifted left by LPC_LEFT_SHIFT; shift the sum
  // back.  The following expression will round down.  This doesn't really
  // matter; the rounding could only make a significant difference for very
  // small signals, and in those cases we could usually encode the signal
  // without loss.
  sum = sum >> LPC_LEFT_SHIFT;

  // We need to truncate sum to fit within the range that int16_t can
  // represent (note: in principle we could just let it wrap around and
  // accept that the prediction will be terrible; that would worsen
  // compression but increase speed).
  if (sum > 32767)
    return (int16_t)32767;
  else if (sum < -32768)
    return (int16_t)-32768;
  else
    return sum;
}


/**
   This function updates the autocorrelation statistics and LPC coefficients for
 */

void lilcom_update_autocorrelation_and_lpc(
    int64_t t, CompressionState *state) {
  assert(t % AUTOCORR_BLOCK_SIZE == 0);
  int32_t block_index = ((int32_t)t) >> LOG_AUTOCORR_BLOCK_SIZE;
  int compute_lpc = (t & (LPC_COMPUTE_INTERVAL - 1) == 0);
  if (t < LPC_COMPUTE_INTERVAL) {
    if (t == 0) {
      // For time t = 0, there is nothing to do because there
      // is no previous block to get stats from.  We rely on
      // lilcom_init_lpc having previously been called.
      return;
    }
    // For t < LPC_COMPUTE_INTERVAL we recompute the LPC coefficients
    // every AUTOCORR_BLOCK_SIZE, to improve the LPC estimates
    // for the first few samples.
    compute_lpc = 1;
  }
  // Read block_index & 1 as block_index % 2 (valid for nonnegative
  // block_index, which it is).
  LpcComputation *this_lpc = &(state->lpc_computations[block_index & 1]);
  // prev_lpc is the 'other' LPC object in the circular buffer of size 2.
  LpcComputation *prev_lpc = &(state->lpc_computations[!(block_index & 1)]);
  assert(prev_lpc != this_lpc);  // TODO: remove this.

  // Copy the previous autocorrelation coefficients and
  // max_exponent.
  for (int i = 0; i <= lpc_order; i++)
    this_lpc->autocorr[i] = prev_lpc->autocorr[i];
  this_lpc->max_exponent = prev_lpc->max_exponent;


  int64_t prev_block_start_t = t - AUTOCORR_BLOCK_SIZE;
  // buffer_index = prev_block_start % SIGNAL_BUFFER_SIZE
  int32_t buffer_index = prev_block_start_t & (SIGNAL_BUFFER_SIZE - 1);

  if (buffer_index == 0) {
    // If this is the start of the buffer we need to make sure that the
    // required left context is copied appropriately.
    if (prev_block_start_t == 0) {
      for (int i = 1; i <= state->lpc_order; i++)
        state->uncompressed_signal[MAX_LPC_ORDER - i] = 0;
      for (int i = 1; i <= state->lpc_order; i++)
        state->uncompressed_signal[MAX_LPC_ORDER - i] =
            state->uncompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE - i];
    }
  }

  int is_first_block = (prev_block_start_t == 0);
  // We view the buffer as 'really' starting from the element
  // numbered MAX_LPC_ORDER; those extra elements are for left
  // context at the start of the buffer.
  const int16_t *signal_pointer =
      &(state->uncompressed_signal[MAX_LPC_ORDER + buffer_index]);
  lilcom_update_autocorrelation(this_lpc, lpc_order, is_first_block,
                                signal_pointer);
  if (compute_lpc) {
    lilcom_compute_lpc(state->lpc_order, this_lpc);
  } else {
    // Copy the previous block's LPC coefficients
    for (int i = 0; i < lpc_order; i++)
      this_lpc->lpc_coeffs[i] = prev_lpc->lpc_coeffs[i];
  }
}


/**
   lilcom_compress_for_time_internal attempts to compress the signal for time t;
   on success, it will write to
   state->compressed_code[t*state->compressed_code_stride].

      @param [in] t     The time that we are requested to compress the signal.
                        We require t > 0.
      @param [in] prev_exponent   The exponent value that was used to compress the
                        previous sample (if t > 0), or the "starting" exponent
                        value present in the header if t == 0.  Must be in
                        the range [0, 12].
      @param [in] exponent_floor  A value in the range [0..12] which puts
                        a lower limit on the exponent used for this sample,
                        It is required, in addition to being in [0..12],
                        to be in the range
                        [prev_exponent-1 .. prev_exponent+2].
      @param [in,out] state  Contains the computation state and pointers to the
                       input and output data.

   On success (i.e. if it was able to do the compression) it returns the
   exponent used, which is a number >= 0.

   On failure, which can happen if the exponent required to compress this value
   was greater than prev_exponent + 2, it returns the negative of the exponent
   that would have required to compress this sample (a number greater than
   prev_exponent + 2); this will cause us to enter backtracking code to inrease
   the exponent used on the previous sample.
*/
inline int lilcom_compress_for_time_internal(
    int64_t t,
    int prev_exponent,
    int exponent_floor,
    CompressionState *state) {
  if (t % AUTOCORR_BLOCK_SIZE == 0) {
    // The start of a block.  We need to update the autocorrelation
    // coefficients and LPC coefficients.
    lilcom_update_autocorrelation_and_lpc(t, state);
  }

  assert(t > 0 && prev_exponent >= 0 && prev_exponent <= 12 &&
         exponent_floor >= 0 && exponent_floor <= 12 &&
         exponent_floor >= prev_exponent - 1 &&
         exponent_floor <= prev_exponent + 2);

  int16_t predicted_value = lilcom_compute_predicted_value(state, t),
      observed_value = input_signal[t * state->input_signal_stride];
  // cast to int32 because difference of int16's may not fit in int32.
  int32_t residual = ((int32_t)observed_value) - ((int32_t)predicted_value);

  int mantissa,
      exponent = least_exponent(residual, predicted_value,
                                exponent_floor, &mantissa,
                                &(state->uncompressed_signal[MAX_LPC_ORDER+t]));

  if (exponent <= prev_exponent + 2) {
    // Success; we can represent the difference of exponents in the range
    // [-1..2].  This is the normal code path.
    int exponent_code = (exponent - prev_exponent + 1);
    assert(exponent_code >= 0 && exponent_code < 4 &&
           mantissa >= -32 && mantissa < 32);
    state->compressed_code[t*state->compressed_code_stride] =
        (int8_t)(mantissa << 2 + exponent_code);
    state->exponents[t & (EXPONENT_BUFFER_SIZE - 1)] = exponent;
    return exponent;  // Success.
  } else {
    return -exponent;  // Failure.  The calling code will backtrack, increase the
                       // previous exponent to at least this value minus 2, and
                       // try again.
  }
}

/*
  This function is a special case of compressing a single sample, for t == 0.
  Time zero is a little different because of initialization effects (the header
  contains an exponent and a mantissa for t == -1, which gives us a good
  starting point).

    @param [in] min_exponent  A number in the range [0, 12]; the caller
                  requires the exponent for time t = 0 to be >= min_exponent.
    @param [in,out] state  Stores shared state and the input and output
                  sequences.  The primary output is to
                  state->compressed_code[0]; the header is also
                  modified (to store the exponent and mantissa for
                  phantom sample -1).
*/
void lilcom_compress_for_time_zero(
    int min_exponent,
    CompressionState *state) {
  int16_t first_signal_value = state->input_signal[0];
  assert(min_exponent >= 0 && min_exponent <= 12);
  // m1 refers to -1 (sample-index minus one).
  int sample_m1_min_exponent =
      (min_exponent >= 2 ? 0 : min_exponent - 2);
  int16_t signal_m1,  // compressed signal for "phantom sample" -1.
      predicted_m1 = 0, residual_m1 = first_signal_value;
  int mantissa_m1,
      exponent_m1 = least_exponent(residual_m1,
                                   predicted_m1,
                                   sample_m1_min_exponent,
                                   &mantissa_m1, &signal_m1);
  assert(exponent_m1 > 0);  // TODO: remove this.
  int c_stride = state->compressed_code_stride;

  // Fill in the exponent and mantissa for frame -1, which form bytes
  // 2 and 3 of the 4-byte header.
  state->compressed_code[(-LILCOM_HEADER_BYTES + 2) * c_stride] = exponent_m1;
  state->compressed_code[(-LILCOM_HEADER_BYTES + 3) * c_stride] = mantissa_m1;

  // store the initial exponent, at sample -1.  Probably won't be
  // accessed, actually.  [TODO: remove this?]
  state->exponents[EXPONENT_BUFFER_SIZE - 1] = exponent_m1;

  if (exponent_m1 - 1 > min_exponent)
    min_exponent = exponent_m1 - 1;

  // The autocorrelation parameters for the first block say "simply copy the
  // previous sample".  We do this manually here rather than accessing
  // the LPC coefficients.
  int32_t predicted_0 = signal_m1,
      residual_0 = first_signal_value - predicted_0;

  int mantissa_0,
      exponent_0 = least_exponent(residual_0,
                                  predicted_0,
                                  min_exponent,
                                  &mantissa_0,
                                  &(state->uncompressed_signal[MAX_LPC_ORDER + 0]));
  int delta_exponent = exponent_0 - exponent_m1;
  // The residual cannot be greater in magnitude than first_value, since we
  // already encoded first_signal_value and we are now just dealing with the
  // remaining part of it, so whatever exponent we used for sample -1 would
  // be sufficiently large for sample 0; that's how we can guarantee
  // delta_exponent <= 2.
  assert(delta_exponent >= -1 && delta_exponent <= 2 &&
         exponent_0 >= min_exponent && mantissa_0 >= -32
         && mantissa0 <= 31);

  state->compressed_code[0 * c_stride] =
      (int16_t)((mantissa_0 << 2) + (delta_exponent + 1));



  for (int i = 0; i < MAX_LPC_ORDER; i++) {
    // All samples prior to t=0 are assumed to have zero value for purposes of
    // computing LPC coefficients.  This may not actually matter right now,
    // doing it just in case it's an issue in case of future code changes.
    state->uncompressed_signal[i] = 0;
  }
  // We already set state->uncompressed_signal for time t=0
  // (which lives at state->uncompressed_signal[MAX_LPC_ORDER]).
  state->exponents[0] = exponent_0;
}

/**
   This is a version of lilcom_compress_for_time that is called when the
   exponent was too small, and we have to backtrack to increase the exponent for
   previous samples.  Basically, it handles the hard cases that
   lilcom_compress_for_time cannot directly handle.  The main purpose of this
   function is to compress the signal for time t, but to do that it may have to
   go back to previous sample and re-compress those in order to get an exponent
   large enough (we only store the differences in the exponents, and the
   differences are limited to the range [-1..2]).

     @param [in] t   The time for which we want to compress the signal;
                   t >= 0.
     @param [in] min_exponent  The caller requires that we compress the
                  signal for time t with an exponent not less than
                  `min_exponent`, even if it was possible to compress it
                  with a smaller exponent.  We require min_exponent >= 0.
     @param [out] state  The compression state; this is modified by
                  this function.  The primary output of this function is
                  state->compressed_code[t*state->compressed_code_stride], but
                  it may also modify state->compressed_code for time values
                  less than t.  This function will update other elements
                  of `state` as needed (exponents, LPC info).
*/
void lilcom_compress_for_time_backtracking(
    int64_t t,
    int min_exponent,
    CompressionState *state) {
  assert(t >= 0 && min_exponent >= 0);
  if (t > 0) {
    int prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)];
    if (prev_exponent < min_exponent - 2) {
      // We need to revisit the exponent for sample t-1, as we're not
      // able to encode differences greater than +2.
      lilcom_compres_for_time_backtracking(t - 1, min_exponent - 2, state);
      prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)];
      assert(prev_exponent >= min_exponent - 2);
    }
    if (min_exponent < prev_exponent - 1) {
      // lilcom_compress_for_time requires min_exponent to be be in the range
      // [prev_component-1..prev_component+2], so we need to increase
      // min_exponent.  (Decreasing it would break our contract with the
      // caller; increasing it is OK).
      min_exponent = prev_component - 1;
    }
    int exponent = lilcom_compress_for_time_internal(
        t, prev_exponent, min_exponent, state);
    if (exponent < 0) {
      // Now make `exponent` positive. It was negated as a signal that there was
      // a failure: specifically, that exponent required to encode this sample
      // was greater than prev_exponent + 2.  [This path is super unlikely, as
      // we've already backtracked, but it theoretically could happen].  We can
      // deal with it via recursion.
      assert(exponent > prev_exponent + 2 && exponent > min_exponent);
      lilcom_compress_for_time_backtracking(t, exponent, state);
    }
  } else {
    // time t=0.
    lilcom_compress_for_time_zero(min_exponent, state);
  }
}

/**
   Compress the signal for time t
       @param [in] t   The time we are asked to compress.  We require t > 0.
       @param [in,out] state    Struct that stores the state associated with
                         the compression, and inputs and outputs.
*/
inline void lilcom_compress_for_time(
    int64_t t,
    CompressionState *state) {
  assert(t > 0);

  int prev_exponent =
      state->exponents[(t - 1) & (EXPONENT_BUFFER_SIZE - 1)],
      exponent_floor = (prev_exponent == 0 ? 0 : prev_exponent - 1);

  int exponent = lilcom_compress_for_time_internal(
      t, prev_exponent, exponent_floor, state);
  if (exponent >= 0) {
    // lilcom_compress_for_time_internal succeeded; we are done.
    return;
  } else {
    // The returned exponent is negative; it's the negative of the exponent that
    // was needed to compress sample t.  The following call will handle this
    // harder case.
    lilcom_compress_for_time_backtracking(t, -exponent, state);
  }
}

static inline void lilcom_init_compression(
    int64_t num_samples,
    const int16_t *input, int input_stride,
    int8_t *output, int output_stride,
    int lpc_order, CompressionState *state) {
  state->lpc_order = lpc_order;
  lilcom_init_lpc(&(state->lpc_computations[0]));
  lilcom_init_lpc(&(state->lpc_computations[1]));
  state->input_signal = input_signal;
  state->input_signal_stride = input_stride;
  state->compressed_code =
      output_signal + (LILCOM_HEADER_BYTES * output_stride);
  state->compressed_code_stride = output_stride;

  // Element zero of the header is the magic byte 'l'.
  output_signal[0*output_stride] = (int_8)'l';
  // Element one of the header is the LPC order.
  output_signal[1*output_stride] = (int_8)lpc_order;
  // The second and third bytes of the header will be set in
  // `lilcom_compress_for_time_zero` below.
  int min_exponent = 0;
  lilcom_compress_for_time_zero(min_exponent, state);
}

/*  See documentation in lilcom.h  */
int lilcom_compress(int64_t num_samples,
                     const int16_t *input, int input_stride,
                     int8_t *output, int output_stride,
                     int lpc_order) {
  if (!num_samples > 0 && input_stride != 0 && output_stride != 0 &&
      lpc_order >= 1 && lpc_order <= MAX_LPC_ORDER) {
    return 1;  // error
  }
  CompressionState state;
  lilcom_init_compression(num_samples, input, input_stride,
                          output, output_stride, lpc_order,
                          &state);
  for (int64_t t = 1; t < num_samples; t++)
    lilcom_compress_for_time(t, state);

  return 0;
}








/**
   Note on integer arithmetic.  We use only integer arithmetic for exact reproducibility,
   so we don't have to worry about slight differences in roundoff behavior leading
   to unpredictable results.

   We view all the 16-bit signal values from -32768 .. 32767 as values between 0 and 1,
   by dividing by 32767.

   We have another encoding for M.  We'll add 128 to all residuals in the 16-bit encoding
   to make sure M^{-1} can't get huge.    This means that the smallest eigenvalue of
   M would be, in real numbers, (128/32768)^2 = 2^{-16}.  The largest possible eigenvalue
   of M would be 2.0.  This means that the possible range of eigenvalues of M^{-1}
   would be 0.5 to 2^16.

   We store elements of M scaled by 2^14 to use the full range of 32-bit ints.
   For z_t's

   x_t are already stored as integers (times 2^15).


 */



/**
   Linear regression...
     Predicting x_t as a function of x_{t-1}, x_{t-2}, ...

     Maximizing p(x_t)

...
  2nd order predictor based on last n samples' stats?



 */


void lilcom_uncompress(u_int64_t size, int8_t *src, int16_t *dest) {


}
