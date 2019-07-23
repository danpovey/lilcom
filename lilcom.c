#include "lilcom.h"
#include <assert>
#include <math.h>


#define LILCOM_VERSION 1

#define LILCOM_HEADER_BYTES 4


/**
   Note: you can't change the various constants below without changing
   LILCOM_VERSION, because it will mess up the decompression.  That is:
   the decompression method requires exact compatibility with the
   compression method, w.r.t. how the LPC coefficients are computed.
*/


/**
   max_lpc_order is the maximum allowed order of the linear prediction
   computation.  Defines a limit on how large the user-specified LPC order may
   be.  May not be >= LPC_COMPUTE_INTERVAL, or else we'd need to do extra work
   to handle edge cases near the beginning of the signal.
**/
#define MAX_LPC_ORDER 15


/** a value >= log_2 of MAX_LPC_ORDER+1.  Used in computing certain bounds (?not needed?) */
#define LPC_ORDER_BITS 4

/*
   An amount that we add to the zeroth autocorrelation coefficient to avoid
   divisions by zero, etc.  1 corresponds to a variation of +-1 sample,
   it's very small, just to deal with that zero issue.
*/
#define AUTOCORR_EXTRA_VARIANCE 1


/**
   Defines an extra amount that we add to autocorr[0], that's relative
   to its observed value.  Basically we multiply autocorr[0] by
     (1.0 + 1.0 / 1<<AUTOCORR_EXTRA_VARIANCE_EXPONENT).
   This helps give us bounds on the value of the LPC coefficients.
 */
#define AUTOCORR_EXTRA_VARIANCE_EXPONENT 16


/**
   We recompute the LPC coefficients every this-many frames.  (Larger values are
   more efficient but may produce not-quite-as-good compression).  Caution: the
   code in lilcom_update_autocorrelation() assumes that this is greater than
   MAX_LPC_ORDER.
 */
#define LPC_COMPUTE_INTERVAL 32

/**
   LPC_COMPUTE_INTERVAL_LOG2 is required to be a nonnegative integer that does
   not exceed log_2(LPC_COMPUTE_INTERVAL).  It's used to optimize the
   code where we work out the highest-order bit set in the autocorrelation
   coefficients.
 */
#define LPC_COMPUTE_INTERVAL_LOG2 5

/**
   AUTOCORR_DECAY_EXPONENT, together with LPC_COMPUTE_INTERVAL, defines
   how fast we decay the autocorrelation stats; small values mean the
   autocorrelation coefficients move faster, large values mean
   they move slower.
 */
#define AUTOCORR_DECAY_EXPONENT 4

/**
   This rolling-buffer size determines How far back in time we keep the
   exponents we used to compress the signal.  It determines how far it's
   possible for us to backtrack when we encounter out-of-range values and need
   to increase the exponent.  8 is more than enough, since the most we'd ever
   need to add to the exponent is +11 (the exponent can never exceed 11); the
   exponent can increase by at most 2 for each sample; and (8-1) * 2 > 11 (if we
   have 8 in the buffer, the furthest we can go back in time is t-7, which is
   why we have (8-1) above).

   It must be a power of 2 due to a trick used to compute a modulus.
 */
#define EXPONENT_BUFFER_SIZE 8

/**
   SIGNAL_BUFFER_SIZE determines the size of a rolling buffer containing
   the history of the compressed version of the signal, to be used while
   compressing.  It must be a multiple of LPC_COMPUTE_INTERVAL.  It must
   be at least as great as

       LPC_COMPUTE_INTERVAL + EXPONENT_BUFFER_SIZE + MAX_LPC_ORDER

   (i.e. it needs to store a whole block's worth of data, plus the
   farthest we might backtrack, plus enough context to process the
   first sample of the block.  This backtracking stuff is because there
   may be situations where we need to recompute the autocorrelation
   coefficients of a block after backtracking, because the compressed
   signal changed.

   It must also be a power of 2, due to tricks we use when computing
   modulus.

   Every time we get to the beginning of the buffer we need to do a little extra
   work to place the history before the start of the buffer, that's why we don't
   just make this 64 (i.e. two blocks' worth).  The reason for not using a huge
   value is to save memory and be kind to the cache.
 */
#define SIGNAL_BUFFER_SIZE 128


/**
   We are predicting a signal s_t based on s_{t-1}, s_{t-2}, ... s_{t}.

   Define y_t = s_t and x_t = s_{t-1} through s_{t-k}; we're predicting
   y as a function of x.

   Our estimate will be y_t = p x_t + noise, where p is the vector of
   regression coefficients.  We want to minimize
    error =  (y_t - p x_t)^2

          = (1/T) \sum_{t=1}^T   y_t^2 - 2 p x_t y_t  +  p^t x_t x_t^T p_t
    which we can write as:
      error =  \sum_t y_t^2 - 2 p a + p^t M p.
             = const(p) - 2 p a + m^t M p
   where:
              a =  (1/T) \sum_{t=1}^T y_t x_t
              M =  (1/T) \sum_{t=1}^T  x_t x_t^T

   The derivative of the error derivative w.r.t. p is:
      d(error)/dp =   0 =   - 2 a + 2 M p
   so the solution for p is:
     p = M^{-1} a

   In fact we'll be updating a and M^{-1} online, and instead of a simple sum,
   we'll let a and M be decaying sums with a "forgetting factor".

   Let \alpha < 1 (e.g. \alpha = 0.99) be a number that determines how must of the stats we
   keep each time-- a forgetting factor.  So the basic recurrence will be:

     M_t := \alpha M_{t-1} + (1-\alpha) x_t x_t^T
     a_t := \alpha a_{t-1} + (1-\alpha) y_t x_t

  Because we want to keep M_t^{-1} updated, we actually use the Sherman-Morrison formula:

  First write:
      M_t      = \alpha (M_{t-1} + (1-\alpha)/\alpha x_t x_t^T)
 then
    M_t^{-1}  = (1/\alpha) (M_{t-1} + (1-\alpha)/\alpha x_t x_t^T)^{-1}
      Define z_t := M_{t-1}^{-1} x_t.
 Then expanding using the Sherman-Morrison formula,

    M_t^{-1}   = (1/\alpha) (M_{t-1}^{-1}  -  \frac{  (1-\alpha)/\alpha z_t z_t^T  }
                                                   {  1 + (1-\alpha)/\alpha z_t^T x_t } )
    M_t^{-1}   = (1/\alpha) (M_{t-1}^{-1}  -  \frac{   z_t z_t^T  }
                                                   {  \alpha/(1-\alpha) +  z_t^T x_t }
               = (1/\alpha) M_{t-1}^{-1}  - 1/(\alpha^2/(1-\alpha) + \alpha z_t^T x_t)  z_t z_t^T

  The above is the basic update.  But we are concerned about M_t^{-1} getting very large
  or infinite if we have a string of zeros, or other linearly dependent data.  To
  prevent that, conceptually, we do, after the recurrence above,
     M_t := M_t + SMOOTH_M_AMOUNT * I.
  That is:
      M_t = \alpha (M_{t-1} + (1-\alpha)/\alpha x_t x_t^T)
   followed by
     M_t := M_t + (1-\alpha) * SMOOTH_M_AMOUNT * I
  Because that's going to be an O(n^3) operation, instead of doing the above we do:
    if (t % SMOOTH_M_INTERVAL == 0) {
      M_t := M_t + (1-\alpha) * SMOOTH_M_AMOUNT * SMOOTH_M_INTERVAL * I
    }
  However, that would be a potentially expensive operation, involving matrix inversion,
  and I don't want to have to implement it (and certainly don't want to add a huge
  library dependency).  We can approximate it fairly easily though.

  Consider that d/dx (1/x) = -1/x^2.  We can apply the same approach to the
  eigenvalues of M, so that for a symmetric matrix S,
      S := S + \beta I
  can be approximated in the inverse space by:
      S^{-1} := S^{-1} - 2 \beta S^{-1} S^{-1}, which in our case becomes:

    M_t^{-1} := M_t^{-1} - 2*(1-\alpha)*SMOOTH_M_AMOUNT*SMOOTH_M_INTERVAL * M_t^{-1}*M_t^{-1}.  [eqn:a]

  (caution: this logic is a little approximate): if we are applying this
  smoothing and it's working, the eigenvalues of M_t should never be less
  than SMOOTH_M_AMOUNT, hence the eigenvalues of M_t^{-1} should never be
  greater than 1/SMOOTH_M_AMOUNT, hence the eigenvalues of the negated term
  of [eqn:a] may be, at most,
         2*(1-\alpha)*SMOOTH_M_INTERVAL / SMOOTH_M_AMOUNT.
  Now, the l.h.s. term eigenvalues must be >= 1/SMOOTH_M_AMOUNT, so for
  [eqn:a] to be stable / not overshoot, we basically need to have:
         2*(1-\alpha)*SMOOTH_M_INTERVAL < 1
  i.e.
        SMOOTH_M_INTERVAL < 1 / (2(1-\alpha))
  and since the logic above is, as I noted, a little approximate, to
  get a proper bound we need a little wiggle room.  To be safe let's
  make this a factor of 4 (probaly more like 2 would suffice), so we'll
  require:
        SMOOTH_M_INTERVAL < 1 / (8(1-\alpha))







 .. which, physically, would be realized as follows, via a special case of the
 Sherman-Morrison-Woodbury formula:
    (M + \alpha I)^-1 = M^{-1} - M^{-1} (1/\alpha I + M^{-1}

 {\displaystyle \left(A+UCV\right)^{-1}=A^{-1}-A^{-1} U \left(C^{-1}+VA^{-1}U\right)^{-1}VA^{-1},}

*/



/**
   This contains information related to the computation of the linear prediction
   coefficients (LPC).
 */
struct LpcComputation {
  // 'autocorr[i]' contains a weighted sum of (s[j] * s[j - i]) << 23;
  // we use a 'forgetting factor' of AUTOCORR_DECAY_EXPONENT, i.e. each
  // time we multiply by (1 + 2^-{AUTOCORR_DECAY_EXPONENT}).
  // The elements of the sum cannot exceed (2^15)^2 = 2^30, shifted
  // left 23 bits, so <= 2^53.
  // The sum will be no greater than this times 2^AUTOCORR_DECAY_EXPONENT,
  // so the sum would be <= 2^(53+8) = 2^61.
  int64 autocorr[MAX_LPC_ORDER + 1];

  // highest_bit_set is the index of the highest bit set in abs(autocorr[0]);
  // or if autocorr[0] == 0, it is 0.  (Note: if autocorr[0] is nonzero,
  // highest bit set will always exceed 23).
  // So highest_bit_set is the largest number >= 0 such that
  // abs(autocorr[0]) >= (1 << highest_bit_set), or zero if that does not
  // exist.
  // Also note: highest_bit_set will always be >= 23 if we have
  // processed at least one block.
  int32 highest_bit_set;

  // The LPC coefficients times 2^23.  These are initialized to
  // (1 0 0 0.. ) at the start of the utterance (so the prediction
  // is just based on the previous sample).  Of course that's
  // represented as (1<<23 0 0 0 ...).
  //
  // If we have processed at least one block of LPC_COMPUTE_INTERVAL samples and
  // have called lilcom_durbin, this will contain the coefficients estimated
  // from there.
  // Only elements 0 through lpc_order - 1 will be valid.
  // (lpc_order is passed directly into functions dealing with this).
  int32 lpc_coeffs[MAX_LPC_ORDER + 1];
};


void lilcom_init_autocorrelation(AutocorrCoeffs *coeffs) {
  for (int i = 0; i <= MAX_LPC_ORDER; i++) {
    coeffs->autocorr[i] = 0;
    coeffs->lpc_coeffs[i] = 0;
  }
  coeffs->highest_bit_set = 0;
  // the LPC coeffs are stored shifted left by 23, so this means the 1st coeff
  // is 1 and the rest are zero-- meaning, we start prediction from the previous
  // sample.
  coeffs->lpc_coeffs[0] = 1 << 23;
}


/**
   Updates the autocorrelation coefficients in 'coeffs', adding one block's
   worth of autocorrelation data.  This is called every LPC_COMPUTE_COEFFS
   samples.

     @param [in,out]  autocorr   The coefficients to be updated.  Assumed
                            to have been previously computed if
                            is_first_block == 0 (and to be updated);
                            must have its coefficients zeroed if
                            is_first_block != 0.
     @param [in] lpc_order   The LPC order, must be in [1..MAX_LPC_ORDER]
     @param [in] is_first_block  Nonzero if this is the first block of
                           signal

     @param [in] signal     Pointer to the signal we are processing.
                           In all cases we will access values signal[0]
                           through signal[LPC_COMPUTE_INTERVAL-1].  In
                           is_first_block == 0, we will also look at
                           elements signal[-MAX_LPC_ORDER] through
                           signal[-1].
*/
inline static void lilcom_update_autocorrelation(
    AutocorrCoeffs *autocorr, int lpc_order, int is_first_block,
    int16_t *signal) {

  // 'temp_autocorr' will contain the raw autocorrelation stats
  // without the shifting left by 23 (we'll do that at the end,
  // to save an instruction in the inner loop).
  int64_t temp_autocorr[MAX_LPC_ORDER + 1];

  if (!is_first_block) {
    // The samples that come from the previous block need to be scaled down
    // slightly, in order to be able to guarantee that no element of
    // autocorr->autocorr is greater than the zeroth element.  The way we do
    // this is to process them before we scale down the elements of
    // autocorr->autocorr.  We add these directly to autocorr->autocorr without
    // using the temporary storage of `temp_autocorr`.
    for (i = 0; i < lpc_order; i++) {
      int64 signal_i = signal[i];
      for (int j = i + 1; j <= lpc_order; j++) {
        autocorr->autocorr[j] += (signal[i - j] * signal_i) << 23;
      }
    }
  }

  // Scale down the current data slightly.  This is an exponentially decaying
  // sum, but done at the block level not the sample level.
  for (int i = 0; i <= lpc_order; i++) {
    autocorr->autocorr[i] -= (autocorr->autocorr[i] >> AUTOCORR_DECAY_EXPONENT);
    temp_autocorr[i] = 0;
  }

  int i = 0;

  // We need to exclude negative-numbered samples from the sum.  In the first
  // block they don't exist (conceptually, they are zero); in subsequent blocks
  // we need to scale them down according to AUTOCORR_DECAY_EXPONENT in order to
  // be able to guarantee that the zeroth coefficient is the largest one; and
  // this has been handled above; search for `if (!is_first_block)`.
  // [actually it's not clear that we need this property, but it makes certain
  // things easier to reason about.]
  for (; i < lpc_order; i++) {
    int32 signal_i = signal[i];
    for (int j = 0; j <= i; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
  }
  // OK, now we handle the samples that aren't close to the boundary.
  // currently, i == lpc_order.
  for (; i < LPC_COMPUTE_INTERVAL; i++) {
    int32 signal_i = signal[i];
    for (int j = 0; j <= lpc_order; j++) {
      temp_autocorr[j] += signal[i - j] * signal_i;
    }
  }
  for (int j = 0; j <= lpc_order; j++) {
    autocorr->autocorr[j] += temp_autocorr[j] << 23;
  }

  // This takes care of the smoothing to make sure that the autocorr[0] is nonzero,
  // and to add extra noise determined by AUTOCORR_EXTRA_VARIANCE_EXPONENT and the
  // signal energy, which will allow us to put a bound on the value of the LPC
  // coefficients so we don't need to worry about integer overflow.
  autocorr->autocorr[0] += ((int32_t)(LPC_COMPUTE_INTERVAL*AUTOCORR_EXTRA_VARIANCE)<<23) +
      temp_autocorr[0] << (23 - AUTOCORR_EXTRA_VARIANCE_EXPONENT);


  int exponent = LPC_COMPUTE_INTERVAL_LOG2 + 23;
  int64_t abs_autocorr_0 =
      (autocorr->autocorr[0] > 0 ? autocorr->autocorr[0] : -autocorr->autocorr[0])
      >> exponent;
  while (abs_autocorr_0 >= 8) {
    abs_autocorr_0 >>= 4;
    exponent += 4;
  }
  if (abs_autocorr_0 >= 2) {
    abs_autocorr_0 >>= 2;
    exponent += 2;
  }
  if (abs_autocorr_0 >= 1) {
    exponent += 1;
    assert(abs_autocorr_0 == 1);
  }
  assert(abs_autocor_0 >= (1 << exponent) &&
         abs_autocor_0 < (2 << exponent) &&
         exponent >= 23);
  autocorr->highest_bit_set = exponent;
}

/*
      *NOTE ON BOUNDS ON LPC COEFFICIENTS*

     Suppose the autocorrelation at time zero (i.e. the sum of
     squares) is S.  We add to the autocorrelation at time zero,
     S * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} (currently 2^16).

     This can provide a bound on the LPC coefficients.
     For each LPC coefficient alpha, the variance in our estimate
     of the current sample would be increased by
     alpha^2 * S * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT}.  Now, if
     the LPC coeffs were all zero, the prediction variance we'd get
     would be about S.  We know that the extra noise from term
     mentioned above must not exceed S, or we'd be doing worse
     than zero LPC coeffs.  So

       alpha^2 * S * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} <= S
     so
       alpha^2 * 2^{-AUTOCORR_EXTRA_VARIANCE_EXPONENT} <= 1
    (don't worry about S = 0; see AUTOCORR_EXTRA_VARIANCE).1
    That implies alpha <= 2^(AUTOCORR_EXTRA_VARIANCE_EXPONENT / 2),
    i.e. currently the LPC coeffs cannot exceed 2^8 (which is
    a very-worst case).  So when stored as integers with 23 as
    the exponent, they cannot exceed 2^31.
 */



/**
   struct CompressionState contains the state that we need to maintain
   while compressing a signal; it is passed around by functions
   that are compressing the signal.
 */
struct CompressionState {

  /**
     The LPC order, a value in [1..MAX_LPC_ORDER].  May be user-specified.
   */
  int32_t lpc_order;

  /**
     'lpc_computations' is to be viewed as a rolling buffer of size
     2, containing the autocorrelation and LPC coefficients.
     Define the block-index, as a function of t,
         b(t) = t / LPC_COMPUTE_INTERVAL,
     rounding down, of course.  The LPC coefficients we use to
     predict sample t will be obtained from lpc_computations[b(t) % 2].

     Whenever we process a t value that's a multiple of LPC_COMPUTE_INTERVAL and
     which is nonzero, we will use the preceding block of LPC_COMPUTE_INTERVAL
     coefficients of the signal (and, if present, `lpc_order` samples context
     preceding that) to update the autocorrelation coefficients and the
     resulting LPC coefficients.
  */
  LpcComputation lpc_computations[2];


  /**
     A rolling buffer of the exponent values used to compress the signal.
     Note: the compressed code only contains the delta of the exponents,
     so this history can be quite useful.
   */
  int exponent[EXPONENT_BUFFER_SIZE];

  /**
     The compressed-and-then-compressed version of the input signal.  We need to
     keep track of this because it's the uncompressed version of the signal
     that's used to compute the LPC coefficients (this ensures that we can
     compute them in the same way when we decompress).

     The signal at time t is located at
       uncompressed_signal[(t % SIGNAL_BUFFER_SIZE) + MAX_LPC_ORDER]

     The reason for the extra MAX_LPC_ORDER elements is so that we can ensure,
     when we roll around to the beginning of the buffer, that we have a place to
     put the recent history (the previous `lpc_order` samples).  This keeps the
     code of lilcom_update_autocorrelation simple.
  */
  int16_t uncompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE];


  /**
     The input signal that we are compressing.  We put it here so it can easily
     be passed around.
   */
  const int16_t *input_signal;

  /**
     The compressed code that we are generating, one byte per time frame.  This
     pointer *not* point to the start of the header (the pointer has been
     shifted forward by 4), so the t'th value is located at compressed_code[t].
   */
  int8_t *compressed_code;

};



// returns the sign of 'val', i.e. +1 if is is positive, -1 if
// it is negative, and 0 if it is zero.
inline int sgn(int val) {
  return (0 < val) - (val < 0);
}

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
                     must be in the range [0, 11].  This function will
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

        @return  Returns the value in the range [min_exponent..11] which,
                 together with `mantissa`, provides a close approximation
                 to `residual` while not allowing the next sample
                 to exceed the range of int16_t.

                 The intention of this function is to return the exponent in the
                 range [min_exponent..11] which gives the closest approximation
                 to `residual`, while choosing the lower exponent and
                 lower-magnitude mantissa in case of ties.  This is largely what
                 it does, although it may not always do so in situations where
                 we needed to modify the mantissa to not exceed the range of
                 int16_t.


   The following explains the internals of how this function operates.

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
   as the exponent.  We can express the above in integer math as:

      -66 * 2^e <= (y_uncomp - y_pred_int) * 2 <= 63 * 2^e

   and notice that we have multiplied by 2 so that we don't have fractional
   expressions.

   Define the actual mantissa am(e) as just m(e), rounded to the closest
   integer, rounding towards zero in case of ties (and, as a special case,
   if m(e) is -33, letting am(e) be -32).
*/
inline static int least_exponent(int32_t residual,
                                 int16_t predicted,
                                 int min_exponent,
                                 int *mantissa) {
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
  assert(exponent <= 11);

  {
    // This code block computes 'mantissa', the integer mantissa which we call
    // m(e) in the math above, and which should be a value in the range
    // [-32, 31].
    //
    // m(e) is the result of rounding (residual / (float)2^exponent)
    // to the nearest integer, rounding towards zero in case of ties; and
    // then, if the result is -33, changing it to -32.
    //
    // What we'd like to do is compute:
    //
    //     mantissa = residual / (1<<exponent)
    //
    // and have the "/" expression round to the closest integer, rounding
    // towards zero in case of ties; but that's now how it works in integer
    // division in C (it removes any fractional part).  By writing
    //
    //    mantissa = (residual*2 + offset) / (2<<exponent)
    //
    // using "C" integer division, we can get what we want, where 'offset' is a
    // number that controls the rounding behavior, defined as
    //
    //    offset = (1<<exponent - 1) * sign(residual).
    //
    // Of course, the correct behavior of this relies on the fact that
    // C integer division truncates any fractional part towards zero.
    int offset = ((1 << exponent) - 1) * sgn(residual),
        n2 = 2 << exponent,
        local_mantissa = (residual2 + offset) / n2;


    // maybe the expression below would be as fast; I should test.
    // The 0.999 is to ensure we round towards zero in case of ties.
    assert(local_mantissa == round((residual / (float)(1<<exponent)) * 0.999) &&
           local_mantissa >= -33 && local_mantissa <= 31);
    if (local_mantissa == -33)
      local_mantissa = -32;

    int32_t next_signal_value =
        ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);
    if (next_signal_value != (int16_t)next_signal_value) {
      // The next signal exceeds the range of int16_t; this can in principle
      // happen if the predicted signal was close to the edge of the range
      // [-32768..32767] and quantization effects took us over.  We need to
      // reduce the magnitude of the mantissa by one in this case.
      local_mantissa -= sgn(local_mantissa);
      next_signal_value =
          ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);
      assert(next_signal_value == (int16_t)next_signal_value);
    }
    *mantissa = local_mantissa;
  }
  return exponent;
}

void lilcom_init_computation_state(int exponent,
                                   ComputationState *computation_state) {
  for (int i = 0; i < MAX_LPC_ORDER; i++) {
    computation_state->x[i] = 0.0;
    computation_state->a[i] = 0.0;
    computation_state->p[i] = 0.0;
    for (int j = 0; j < MAX_LPC_ORDER; j++) {
      if (i != j) computation_state->M_inv[i][j] = 0.0;
      else computation_state->M_inv[i][j] = 1.0 / SMOOTH_M_AMOUNT;
    }
  }
  computation_state->exponent = exponent;
  computation_state->y = 0.0;

  // the following values won't actually matter.
  computation_state->y_pred = 0.0;
  computation_state->y_pred_in = 0.0;
  computation_state->mantissa = 0;
  computation_state->y_int = 0;
}



// Durbin's recursion - converts autocorrelation coefficients to the LPC
// pTmp - temporal place [n]
// pAC - autocorrelation coefficients [n + 1]
// pLP - linear prediction coefficients [n] (predicted_sn = sum_1^P{a[i-1] * s[n-i]}})
//       F(z) = 1 / (1 - A(z)), 1 is not stored in the demoninator


/**
   Integerized Durbin computation (computes linear prediction coefficients
   given autocorrelation coefficients).


     `coeffs` is a struct containing the autocorrelation coefficients;
              see its documentation.  Only entries 0 through lpc_order
              will be defined.
     `lpc_order` is the linear prediction order, a number at least 1 and
              not to exceed MAX_LPC_ORDER.  It is the number
              of taps of the IIR filter / the number of past samples
              we predict on.
     `autocorr` is the autocorrelation coefficients, an array of size
              lpc_order.  These are actually un-normalized autocorrelation
              coefficients, i.e. we haven't divided by the total number of
              points in the sum, because doing so would make no difference to
              the result.  (It would be equivalent to a scaling of the signal).


     `lpc_coeffs` (an output) is the linear prediction coefficients (imagine
             they were floats) times 2^23.  At exit, values 0 through lpc_order - 1
             will be set.
             The prediction of s[n] would be:
              s_n_predicted =
                 (sum(i=0..lpc_order-1): (s[n-1-i] * coeff[i])) >> 23

    This code was adapted to integer arithmetic from the 'Durbin' function in
    Kaldi's feat/mel-computations.cc, which in turn was originaly derived
    from HTK.  Any way durbin's algorithm is very well known.
*/
void lilcom_compute_lpc(int32 lpc_order,
                        LpcComputation *c) {
  // autocorr_local and temp are both to be viewed as floating point numbers
  // with an exponent of -23.  That is, to interpret them as floating point
  // numbers you'd cast to float and multiply by 2^-23.  Bear in mind that the
  // 'normal range' of the data in `autocorr` and `temp` is about -1 to 1
  // (independent of the signal scale), which is why it's OK to used a fixed
  // exponent.  The number 23 was chosen to exceed the precision of regular
  // floating point, while giving plenty of overhead so that when we
  // multiply two of these together, even if they are significantly larger
  // than 1.

  /**
     autocorr is just a copy of c->autocorr, but shifted to ensure that the
     magnitude of all elements is < 2^23 (but as large as possible given that
     constraint).  We don't need to now how much it was shifted right, because
     it won't affect the LPC coefficients.
  */
  int32 autocorr[MAX_LPC_ORDER];  // will be limited in magnitude to < 2^23,
                                  // similar to the precision of float32.
  {
    // Set autocorr to equal c->autocorr, with elements shifted
    // to the right enough to ensure that the highest bit set in abs(autocorr[0])
    // is the 22nd bit (hence its values are < 2^23, and all values in
    // autocorr are < 2^23, since we have guaranteed that autocorr[0] is the
    // largest-absolute-value element.
    assert(aucorr_in->highest_bit_set > 22);
    int32 right_shift = autocorr_in->highest_bit_set - 22;
    for (i = 0; i <= lpc_order; i++)
      autocorr[i] = autocorr_in->autocorr[i] >> right_shift;
  }


  int32 temp[MAX_LPC_ORDER];  // 'temp' is a temporary array used in the LPC
                              // computation, stored with exponent of 23.  In
                              // Kaldi's mel-computations.cc is is called pTmp.
                              // It seems to temporarily store the next
                              // iteration's LPC coefficients.  Note: the
                              // floating-point value of these coefficients
                              // would not exceed 2^8 (8 =
                              // AUTOCORR_EXTRA_VARIANCE_EXPONENT / 2); see
                              // BOUNDS ON LPC COEFFICIENTS above.  The
                              // magnitude of the elements of 'temp' will be
                              // less than 2^(23+8) = 2^31.  i.e. it fits into
                              // an int32.



  int32 E = autocorr[0];  // Note: the highest bit set in E will be the 22nd; it is
                          // less than 2^23.  The autocorr coeffs are scaled to
                          // ensure this.

  int32 j;
  for (int32 i = 0; i < lpc_order; i++) {
    // k_i will be the next reflection coefficient, a value in [-1, 1],
    // but scaled by 2^23 to represent in fixed point.
    int64 ki = autocorr[i + 1] << 23;  // currently, magnitude < 2^(23+23) = 2^46

    for (j = 0; j < i; j++) {
      // max magnitude of the terms added below is 2^(31 + 23) = 2^54, i.e.
      // the abs value of the added term is less than 2^54.
      ki += coeffs[j] * autocorr[i - j];
    }
    // ki is a summation of terms, so add LPC_ORDER_BITS=4 to the magnitude of
    // 2^54 computed above, it's now bounded by 2^58 (this would bound its value
    // at any point in the summation above).
    ki = ki / E;
    // at this point, ki is mathematically in the range [-1,1], since it's a
    // reflection coefficient; and it is stored times 2^23, so its magnitude as
    // an integer is <= 2^23.  We check that it's less than 2^23 plus a margin
    // to account for roundoff errors.
    assert(abs(ki) < (1<<23 + 1<<15));

    // float c = 1 - ki * ki;
    int64 c = (((int64)1) << 23) - ((ki*ki) >> 23);

    // The original code did as follows, but this is not
    // necessary as we handle it with AUTOCORR_EXTRA_VARIANCE
    // and AUTOCORR_EXTRA_VARIANCE_EXPONENT, so E should never
    // become zero or negative.
    // if (c < 1.0e-5)
    //   c = 1.0e-5;
    //

    // Then the original code did:
    // E *= c;
    E = (int32)((E * c) >> 23);

    // compute the new LP coefficients
    // Original code did: pTmp[i] = -ki;
    temp[i] = -ki;  // abs(temp[i]) <= 2^23, since ki is in the range [-1,1] when
                    // viewed as a real number.

    for (j = 0; j < i; j++) {
      // The original code did:
      //   pTmp[j] = pLP[j] - ki * pLP[i - j - 1]
      // These are actual LPC coefficients (computed for LPC order i + 1), so
      // their magnitude is less than 2^(23+8) = 2^31.
      //
      // The term on the RHS that we cast to int32 is also bounded by
      // 2^31, because it's (conceptually) an LPC coefficient multiplied by a
      // reflection coefficient ki with a value <= 1.
      //
      // Also coeffs[j] and temp[j] may both be interpreted as LPC coefficients,
      // just of different orders.  This implies that they are both (when viewed
      // as integers) strictly less than 2^31.
      temp[j] = coeffs[j] - (int32)((ki * coeffs[i - j - 1]) >> 23);
    }
    for (j = 0; j <= i; j++) {
      assert(abs(temp[j]) < ((int64)1<<31));
      coeffs[j] = temp[j];  // magnitude less than 2^(23+8) = 2^31.
    }
  }
  // E > 0 because we added fake extra variance via
  // AUTOCORR_EXTRA_VARIANCE_EXPONENT ahnd AUTOCORR_EXTRA_VARIANCE, so according
  // to these stats the sample should never be fully predictable.  E =
  // autocorr[0] because even if things are uncorrelated, we should never be
  // increasing the predicted error vs. no LPC at all.
  assert(E > 0 && E <= autocorr[0]);
}

inline int16_t lilcom_compute_predicted_value(
    CompressionState *state,
    int64_t t) {
  int64_t start_t = t - state->lpc_order;
  int32_t block_index = ((int32_t)t) / LPC_COMPUTE_INTERVAL;
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
  // the lpc_coeffs were stored shifted left by 23.
  sum = sum >> 23;
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
   This
 */
void lilcom_backtrack(
    int64_t t,
    int exponent_floor,
    CompressionState *state);



/**
   lilcom_compress_for_time attempts to compress the signal for time t;
   on success, it will write to state->compressed_code[t].

      @param [in] t     The time that we are requested to compress the signal.
      @param [in] prev_exponent   The exponent value that was used to compress the
                        previous frame (if t > 0), or the "starting" exponent
                        value present in the header if t == 0.
      @param [in] exponent_floor  A value in the range [0, 11] which puts
                        a lower limit on the exponent used for this frame,
                        and which is required, in addition to being >=0,
                        to be in the range
                        [prev_exponent-1 .. prev_exponent+2].
      @param [in,out] state  Contains the computation state and pointers to the
                       input and output data.

   On success (i.e. if it was able to do the compression) it returns the
   exponent used, which is a number >= 0.

   On failure, which can happen if the exponent required was greater
   than prev_exponent + 2, it returns the negative of the exponent
   that it would have required to compress this frame.
*/
inline int lilcom_compress_for_time_internal(
    int64_t t,
    int prev_exponent,
    int exponent_floor,
    CompressionState *state) {
  if (t % LPC_COMPUTE_INTERVAL == 0) {
    // The start of a block.  We need to update the autocorrelation
    // coefficients and LPC coefficients.

  }

  assert(prev_exponent >= 0 && exponent_floor >= 0 &&
         exponent_floor >= prev_exponent - 1 &&
         exponent_floor <= prev_exponent + 2);

  int16_t predicted_value = lilcom_compute_predicted_value(state, t),
      observed_value = input_signal[t];
  // cast to int32 because difference of int16's may not fit in int32.
  int32_t residual = ((int32_t)observed_value) - ((int32_t)predicted_value);

  int mantissa,
      exponent = least_exponent(residual, exponent_floor, &mantissa);

  if (exponent <= prev_exponent + 2) {
    // Success; we can represent the difference of exponents in the range
    // [-1..2].  This is the normal code path.
    int exponent_code = (exponent - prev_exponent + 1);
    assert(exponent_code >= 0 && exponent_code < 4 &&
           mantissa >= -32 && mantissa < 32);
    state->compressed_code[t] = (int8_t)(mantissa << 2 + exponent_code);
    state->exponent[t & (EXPONENT_BUFFER_SIZE - 1)] = exponent;
    return exponent;  // Success.
  } else {
    return -exponent;  // Failure.  The calling code will backtrack, increase the
                       // previous exponent to at least this value minus 2, and
                       // try again.
  }
}

/*
  This function is a special case of compressing a single sample, for t == 0.
  This is a little different because of initialization effects (the header
  contains an exponent and a mantissa for t == -1, which gives us a good
  starting point).

    @param [in] min_exponent  A number in the range [0, 11]; the caller
                  requires the exponent for time t = 0 to be no less than
                  this value.
    @param [in,out] state  Stores shared state and the input and output
                  sequences.  The primary output is to
                  state->compressed_code[0], and to
                  state->compressed_code[-LILCOM_HEADER_BYTES + 2] and
                  state->compressed_code[-LILCOM_HEADER_BYTES + 3] which
                  contain the exponent and mantissa for a phantom frame t-1.
 */
void lilcom_compress_for_time_zero(
    int min_exponent,
    CompressionState *state) {
  int16_t first_value = state->input_signal[0];

  int frame_m1_min_exponent =
      (min_exponent >= 2 ? 0 : min_exponent - 2);
  int mantissa_m1,
      exponent_m1 = least_exponent(first_value,
                                   frame_m1_min_exponent,
                                   &mantissa_m1);
  state->compressed_code[-LILCOM_HEADER_BYTES + 2] = exponent_m1;
  state->compressed_code[-LILCOM_HEADER_BYTES + 3] = mantissa_m1;

  state->exponents[EXPONENT_BUFFER_SIZE - 1] = exponent_m1;

  // autocorrelation parameters for first block simply copy the previous frame.
  int16_t frame_m1_uncompressed_value = mantissa_m1 << exponent_m1;
  int32_t residual = first_value - ((int32)frame_m1_uncompressed_value);

  if (exponent_m1 - 1 > min_exponent)
    min_exponent = exponent_m1 - 1;

  int mantissa0,
      exponent0 = least_exponent(residual,
                                 min_exponent,
                                 &mantissa0),
      delta_exponent = exponent0 - exponent_m1;
  // The residual cannot be greater in magnitude than first_value, since we
  // already encoded part of it, so whatever exponent we used for frame -1 would
  // be sufficiently large for frame 0; that's how we can guarantee
  // delta_exponent <= 2.
  assert(delta_exponent >= -1 && delta_exponent <= 2);

  state->compressed_code[0] = (int16_t)((mantissa0 << 2) + (delta_exponent + 1));

  state->uncompressed_signal[MAX_LPC_ORDER] =



      state->exponents[0] = exponent0;

      EXPONENT_BUFFER_SIZE - 1] = exponent_m1;

  assert(exponent0 >= min_exponent && mantissa0 >= -32
         && mantissa0 <= 31);


                                 (exponent == 0 ? exponent - 1 : 0),
                                     first_value, 0, &mantissa);


                  state->compressed_code[0], and to


}



void lilcom_compress_for_time_backtracking(
    int64_t t,
    int min_exponent,
    CompressionState *state) {
  if (t == 0) {
    // t == 0 is a special case: we just set the initial exponent, in the
    // header, to a particular value.
    state->compressed_code[-LILCOM_HEADER_BYTES + 2] = min_exponent;


  }

}


void lilcom_compress_for_time(
    int64_t t,
    CompressionState *state) {

  int prev_exponent =
      state->exponents[(cur_t - 1) & (EXPONENT_BUFFER_SIZE - 1)],
      exponent_floor = (prev_exponent == 0 ? 0 : prev_exponent - 1);

  int exponent = lilcom_compress_for_time_internal(
      t, prev_exponent, exponent_floor, state);
  if (exponent >= 0) {
    // lilcom_compress_for_time_internal succeeded; there is no problem.
    return;
  } else {
    // exponent is negative; it's the negative of the exponent that
    // was needed to compress frame t.
    lilcom_compress_for_time_backtracking(t, -exponent, state);
  }

  state->exponents[cur_t & (EXPONENT_BUFFER_SIZE - 1)] = exponent_floor;

  while (cur_t <= t) {
    int this_exponent_floor =
        state->exponents[cur_t & (EXPONENT_BUFFER_SIZE - 1)],
        prev_exponent =
        state->exponents[(cur_t - 1) & (EXPONENT_BUFFER_SIZE - 1)];



  }

  if (lilcom_compress_for_time_internal



}



  if (exponent_floor >= prev_exponent) {
    if (exponent_floor > prev_exponent + 2) {
      // We have a problem because the exponent can't increase by more than
      // 2 each time.  We need to backtrack to ensure that prev_exponent is
      // at lesat exponent_floor - 2.
      if (t > 0) {
        lilcom_compress_for_time(t-1, exponent_floor - 2, input_signal,
                                 compressed_code, state);
        prev_exponent = state->exponent[prev_exponent_index];
        assert(prev_exponent >= exponent_floor - 2);
      } else {
        // t == 0.
        // The statement below modifies the initial exponent which has
        // been written as the 3rd byte (index 2) in the header.
        prev_exponent = exponent_floor - 2;
        (compressed_code-LILCOM_HEADER_BYTES)[2] = prev_exponent;
        state->exponents[EXPONENT_BUFFER_SIZE - 1] = prev_exponent;
      }
    }
  } else {
    // the exponent floor is 'not active', i.e. is making no difference;
    // the minimum exponent is determined by prev_exponent in this
    // case.  Note: exponent_floor cannot be negative because
    // if prev_exponent were zero, we would not have reached this
    // point (the caller-supplied exponent_floor is always positive).
    exponent_floor = prev_exponent - 1;
  }
  assert(exponent_floor >= 0 && exponent_floor >= prev_exponent - 1);
  assert(exponent >= min_exponent && mantissa >= -32 && mantissa <= 31);
  if (exponent > prev_exponent + 2) {
    // We need to backtrack, because this exponent is too large to
    // be representable.
  }



}


void lilcom_update_computation_state(
    ComputationState *prev_state,
    ComputationState *state) {

}

/**
   This function is central to the compression method.  It attempts to compute
   the next byte of the compressed stream.  Suppose the byte we are trying
   to compute is the one for time t.
     `prev_state` is the ComputationState for time t-1.  All fields are expected
         to have been initialized.
     `state` is the ComputationState for time t, which is to be partially set
         up.  At entry, none of its fields are defined.  Upon successful
         completion (i.e. if this function returns with status zero), its fields:
             mantissa  exponent  y_int  y  byte
         will be set.
     `y_observed` is the 16-bit integer that was observed, and which we will
         attempt to compress.  (the `y_int` field of `state` will, at exit,
         be close to this value).
   Return:
      On success, returns zero.

      On failure, returns the exponent that (we estimate) prev_state would need
       to have in order for this call to succeed.  This must be a value greater
       than zero.  (Note: the "we estimate" part means we don't guarantee that
       once we backtrack, give 'prev_state' the requested exponent and recompute
       all the new prediction coefficients, the next call would succeed; it
       may be necessary, very occasionally, to backtrack more than once.
 */
int32_t lilcom_compute_next_byte(
    ComputationState *prev_state,
    ComputationState *state,
    int16_t y_observed) {
  int32_t min_exponent = prev_state->exponent - 1;
  int32_t residual = (int32_t)y_observed - (int32_t)prev_state->next_y_pred_int;
  state->exponent = least_exponent(residual, min_exponent, &(state->mantissa));
  int exponent_code = state->exponent - min_exponent;
  if (exponent_code < 4) {
    // This is the "success path".  The exponent is not so large that we need
    // to backtrack, and we can encode this directly.
    state->byte = (int16_t)(4 * state->mantissa + exponent_code);
    int32_t y_int =
        prev_state->next_y_pred_int + (state->mantissa << state->exponent);
    if (y_int < -65536) y_int = -65536;
    else if (y_int > 65535) y_int = 65535;
    state->y_int = (int16_t)y_int;
    state->y = ((float)(1.0 / 32768.0)) * y_int;
    return 0;  // Success.
  } else {
    // The exponent can increase by at most 2 each time.  We return a
    // requested minimum value for the previous state's exponent.
    // (We can't guarantee that it will work the next time, but
    // eventually it will work if you keep backtracking and trying again).
    return state->exponent - 2;
  }
}

/**
   Initializes the compression, writing the first 5 bytes of output corresponding
   to the 4-byte header and the first sample.  The 4-byte header contains the
   magic letter 'l', the version (1), the initial exponent and the LPC order
   combined into one byte, and then one zero byte.
 */
void lilcom_init_compression(
    RememberedState *r,
    int16_t first_value,
    int8_t *compressed,
    int output_stride) {
  int mantissa,
      exponent = least_exponent(first_value, 0, &mantissa);
  compressed[0] = 'l';
  compressed[1] = (int8_t)LILCOM_VERSION;
  compressed[2] = (int8_t)exponent;
  compressed[3] = (int8_t)mantissa;


  // The "previous exponent" will be found in compressed[2].
  // The + 1 below is the value for the exponent offset that
  // means "keep the exponent the same as before".  (0 would
  // mean, decrease it by 1).


  int next_exponent = least_exponent(residual,
                                     (exponent == 0 ? exponent - 1 : 0),
                                     first_value, 0, &mantissa);
  compressed[4] =

  compressed[4] = (int8_t)(mantissa * 4 + 1);

  r->t = 0;
  // The following sets the computation-state for time t = -1 to the
  // "default" value, the same as if we had seen a string of zeroes,
  // but with the exponent set to the provided value.
  lilcom_init_computation_state(exponent, &(r->state[STATE_BUFFER_SIZE - 1]));
}


/**
   Lossily compresses 'num_samples' samples of int16 audio data into  'num_samples + 4'
   bytes of data.  (The first four bytes contain version information etc., and are
   mostly added for future compatibility).

   This process can (approximately) be reversed by calling `lilcom_decompress`.
*/
void lilcom_compress(size_t num_samples,
                     int16_t *uncompressed, int input_stride,
                     int8_t *compressed, int output_stride) {

  RememberedState r;









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
