#include "lilcom.h"
#include <assert>
#include <math.h>



#define LILCOM_VERSION 1


/**
   Note: you can't change the various constants below without changing
   LILCOM_VERSION, because if we do the compression with a certain value
   of those constants, we'd need to have the same values in order to
   decompress.
*/

/* alpha is a forgetting factor used in the LPC estimation.  Want around
   0.9 but choose a number exactly representable in floating point,
   i.e. 127 / 128.  This probably doesn't matter. */
#define ALPHA 0.9921875

/** lpc_order is the order of the linear prediction computation.  **/
#define LPC_ORDER 3

/** Defines how frequently we smooth our statistics M used for LPC with
    a term intended to limit the eigenvalues.  As explained in a comment
    below, for stability we need
         SMOOTH_M_INTERVAL < 1 / (8(1-\alpha))
    which becomes
         SMOOTH_M_INTERVAL < 16
   (note: there is a factor-of-4 wiggle room baked into that formula
   already), so we'll set it to 8.
   CAUTION: if you mess with alpha you'd have to change this too.
    and
    equivalent to noise.  Ensures that our running estimate of M^{-1} is always
    bounded even if the input data is runs of zeros.  Should be more than
    1 (for speed) and less than 1/(1-ALPHA).
*/
#define SMOOTH_M_INTERVAL 8

/**
   Defines how much of the identity matrix we add to M.  So conceptually,
   M is that running-average of the data, plus this amount times the identity.
   We add an amount of the identity that's equivalent to adding a random
   value -16 or +16 to each signal value. (More precise prediction than this is
   pointless because we could use exponent = 0).  16*16 is 256, so this
   is the amount of the identity that we smooth with.
*/
#define SMOOTH_M_AMOUNT 256.0

/**
   How many ComputationState objects we keep when compressing, which defines how
   far we can backtrack when we encounter out-of-range values and need to
   increase the exponent.  7 is more than enough, since the most we'd ever need
   to add to the exponent is +11 (the exponent can never exceed 11), and
   (7-1) * 2 > 11 (if we have 7 in the buffer, the furthest we can go back in
   time is t-6, which is why we have (7-1) above).
 */
#define STATE_BUFFER_SIZE 7


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



// This struct contains the quantities described above, relating to the linear
// regression computation at time t (and our prediction for time t + 1).
struct ComputationState {

  // For the explanations below, we assume that the ComputationState for
  // the previous time (t-1) is available under the name 'prev', like:
  // ComputationState prev;

  // If this is the computation state at time t, x will contain the previous
  // signal values at time t-1, t-2, ... t-lpc_order.  These are in the
  // range [-1, 1].  Note: these are the compressed signal values (since
  // we need the computation state to be the same in encoding and decoding time).
  float x[LPC_ORDER];

  // M_inv is our current estimate of M_t^{-1}, where M_t is the decaying average of
  // the outer product of x.  This includes the current value of x.
  float M_inv[LPC_ORDER][LPC_ORDER];

  // a is our decaying average of y_t x_t.
  float a[lpc_order];

  // p is M_t^{-1} a, i.e. the LPC prediction coefficients.
  float p[lpc_order];

  // mantissa is a value in the range [-32, 31], that will be
  // stored as 6 bits of the int8-encoded value.
  // See later the formula:
  //   y_int = prev.y_pred_int + mantissa << exponent
  // Definition:
  //   If the signal we are trying to compress has y value y_uncomp,
  //   let the 'exact mantissa', m(e) which is a function of the
  //   current exponent e, be defined by the formula as follows:
  //      m(e) =   (y_uncomp - prev.y_pred_int) / (float)(1<<e),
  //  and note, this is a floating point, not integer, number.
  int32_t mantissa;

  // exponent is a value in the range [0, 10], that will be used
  // to scale the mantissa, see definition of y_int below.
  // Two bits of the int8-encoded output are used to keep
  // track of this value; this is done by the formula:
  // exponent = prev.exponent - 1 + delta_exponent,
  // where delta_exponent is in the range [0, 3], encoded by those
  // two bits, and prev is the previous ComputationState.
  //
  // In general we will choose 'exponent' to be the lowest value that is not
  // less than prev.exponent - 1, and such that the 'exact mantissa' m(e)
  // satisfies:
  //    -33.0 <= m(e) <= 31.5.
  // This is the range in which there will be no loss of accuracy by choosing
  // this exponent instead of the next largest one.  (If we had e+1 as the
  // exponent, the closest points we'd be able to reach would be equivalent to
  // -34 and +32; and with d(e) satisfying that inequality, we'd get at least as
  // good precision with e as the exponent.  [Note: -33.0 is the midpoint between
  // -34 and -32, and 31.5 is the midpoint between 31 and 32.]
  //
  // The above rule may in some circumstances mean that we want
  // an exponent that's greater than prev.exponent + 2, which of course
  // is not representable.  In such circumstances we will 'backtrack',
  // meaning we go back to a previous time, increase the exponent,
  // and re-do the computation.
  int32_t exponent;

  // y_int is the int16 value of y (after decompression).  It is obtained with
  // the formula: y_int = prev.next_y_pred_int + mantissa << exponent.  If this formula
  // would overflow the int16 range (rare, but could happen due to rounding)
  // we'll limit it to [-32768, 32767].
  int16_t y_int;

  // y is the floating point version of y_int, i.e. it's (1.0 / 32768.0) * y_int.
  // This is after decompression.  It will be in the range [-1, 1].
  float y;

  // next_y_pred is the predicted value of y for time t+1.
  // prediction coefficients.  I.e. next_y_pred equals our prediction coefficients
  // p times the next sample's "x" vector (which is our current x vector
  // shifted up, with our "y" in position zero).
  float next_y_pred;

  // next_y_pred_int is the integer predicted value of y for time t+1; it equals
  // round(y_pred * 32768), limited to the range [-32768, 32767] and cast to
  // int16.
  int16_t next_y_pred_int;

  // 'byte' contains the byte we'll write to the compressed stream (or that we
  // read from there, if we're reading).  It contains mantissa * 4 +
  //  (exponent - prev_exponent + 1)
  char byte;

};


/*
   CompressionComputationState stores a circular buffer of ComputationStates.

*/
struct RememberedState {
  // The current time (i.e. the index of the current signal value that we are
  // working on compressing)
  int t;


  ComputationState state[STATE_BUFFER_SIZE];
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
  compressed[3] = (int8_t)0;
  // The "previous exponent" will be found in compressed[2].
  // The + 1 below is the value for the exponent offset that
  // means "keep the exponent the same as before".  (0 would
  // mean, decrease it by 1).
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
