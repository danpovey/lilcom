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
   representation while estimating them; corresponds approximately to the
   precision used in that computation (for LPC coefficients with magnitude close
   to 1).
*/
#define LPC_EST_LEFT_SHIFT 23

/**
   The power of two by which the LPC coefficients are multiplied when
   stored in struct LpcComputation and while being applied during
   prediction.  This corresponds roughly to a precision.  The
   reason why we use this value is so that we can do the summation
   in int32_t.  (Note: there may be overflow while computing
   partial sums, but this will cancel out when we get the final
   prediction, see comments in the prediction code.
 */
#define LPC_APPLY_LEFT_SHIFT 15

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
   hardcode 8 as half of this value), and LPC_EST_LEFT_SHIFT would have to be
   reduced by half of the amount by which you increased this.  Reducing this is
   safe, though.
 */
#define AUTOCORR_EXTRA_VARIANCE_EXPONENT 16


/**
   The amount by which we shift left the autocorrelation stats during
   accumulation, to avoid roundoff error due to the decay.  Larger -> more
   accurate, but, as explained in the comment for LpcComputation::autocorr, we
   require

   (30 + AUTOCORR_LEFT_SHIFT + log(AUTOCORR_BLOCK_SIZE) + AUTOCORR_DECAY_EXPONENT + 1)
     to not exceed 61.
   (Note: the + 1 is required because of the variable `twice_autocorr_0` in the function
    lilcom_update_autocorrelation).
   Currently that equals: 30 + 20 + 5 + 3 + 1 = 59  <= 61.  I'm giving
   it a little wiggle room there in case other parameters get changed
   while tuning.

   Caution: you will need to recompute AUTOCORR_DECAY_SQRT if you change this
   value.
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
   CAUTION: if you change AUTOCORR_DECAY_EXPONENT or AUTOCORR_LEFT_SHIFT
   you need to recompute this.

   It represents the sqrt of
      (1 - 2^-AUTOCORR_DECAY_EXPONENT) * 2^AUTOCORR_LEFT_SHIFT
   which is needed to scale history stats when computing autocorrelation
   terms that cross block boundaries.
   (The factor of (1 - 2^-AUTOCORR_DECAY_EXPONENT) that we apply
   to the autocorrelation stats when we cross a block boundary may
   be viewed as the square root of that factor, applied to the
   raw samples of data.  The AUTOCORR_LEFT_SHIFT factor just relates
   to how we store the stats with a left shift.

   This number was computed by typing: floor(sqrt(1 - 2^-3) * 2^20)
   into wolfram alpha.
*/
#define AUTOCORR_DECAY_SQRT 980853



/**
   Determines how frequently we update the LPC coefficients for large t.  We
   update the LPC coefficients every this-many samples for t >=
   LPC_COMPUTE_INTERVAL.  For t < LPC_COMPUTE_INTERVAL we update them every
   AUTOCORR_BLOCK_SIZE, to make sure the accuracy doesn't suffer too much for

   the first few samples of the signal.

   Must be a multiple of AUTOCORR_BLOCK_SIZE,
   and should be substantially smaller than AUTOCORR_BLOCK_SIZE <<
   AUTOCORR_DECAY_EXPONENT to ensure freshness of the LPC coefficients.

   Smaller values will give 'fresher' LPC coefficients but will be slower,
   both in compression and decompression.  But the amount of work is not
   that great; every LPC_COMPUTE_INTERVAL samples we do work equivalent
   to about `lpc_order` samples, where lpc_order is a user-specified value
   in the range [0, MAX_LPC_ORDER].
 */
#define LPC_COMPUTE_INTERVAL 64


/**
   This rolling-buffer size determines how far back in time we keep the
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




/** returns the sign of 'val', i.e. +1 if is is positive, -1 if
    it is negative, and 0 if it is zero.  */
inline int lilcom_sgn(int val) {
  return (0 < val) - (val < 0);
}

#define lilcom_abs(a) ((a) > 0 ? (a) : -(a))


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
  /* The LPC coefficientss are stored shifted left by LPC_APPLY_LEFT_SHIFT, so this
     means the 1st coeff is 1.0 and the rest are zero-- meaning, we start
     prediction from the previous sample.  */
  coeffs->lpc_coeffs[0] = 1 << LPC_APPLY_LEFT_SHIFT;
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
     @param [in] lpc_order   The LPC order, must be in [0..MAX_LPC_ORDER]
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
inline static void lilcom_update_autocorrelation(
    LpcComputation *lpc, int lpc_order,
    const int16_t *signal) {
  /** 'temp_autocorr' will contain the raw autocorrelation stats without the
      shifting left by AUTOCORR_LEFT_SHIFT; we'll do the left-shifting at the
      end, to save an instruction in the inner loop).  */
  int64_t temp_autocorr[MAX_LPC_ORDER + 1];
  int i;

  /** Scale down the current data slightly.  This is to form an exponentially
      decaying sum of the autocorrelation stats (to preserve freshness), but
      done at the block level not the sample level.  Also zero
      `temp_autocorr`.  */
  for (i = 0; i <= lpc_order; i++) {
    /** The division below is a 'safer' way of right-shifting by
        AUTOCORR_DECAY_EXPONENT, which would technically give undefined results
        for negative input */
    lpc->autocorr[i] -= (lpc->autocorr[i] / (1 << AUTOCORR_DECAY_EXPONENT));
    temp_autocorr[i] = 0;
  }

  {
    /** HISTORY SCALING
       Process any terms involving the history samples that are prior to the
       start of the block.

       The samples (for left-context) that come from the previous block need to
       be scaled down slightly, in order to be able to guarantee that no element
       of lpc->autocorr is greater than the zeroth element.  We can then view
       the autocorrelation as a simple sum on a much longer (but
       decreasing-with-time) sequence of data, which means we don't have to
       reason about what happens when we interpolate autocorrelation stats.

       See the comment where AUTOCORR_DECAY_SQRT is defined for more details.
       Notice that we write this part directly to lpc->autocorr instead of to
       temp_autocorr.  */
  for (i = 0; i < lpc_order; i++) {
    int32_t signal_i = signal[i];
    int j;
    for (j = 0; j <= i; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
    for (j = i + 1; j <= lpc_order; j++)
      lpc->autocorr[j] += (signal[i - j] * signal_i) * AUTOCORR_DECAY_SQRT;
  }

  /** OK, now we handle the samples that aren't close to the boundary.
      currently, i == lpc_order. */
  for (; i < AUTOCORR_BLOCK_SIZE; i++) {
    int32_t signal_i = signal[i];
    for (int j = 0; j <= lpc_order; j++) {
      temp_autocorr[j] += signal[i - j] * signal_i;
    }
  }
  for (int j = 0; j <= lpc_order; j++) {
    lpc->autocorr[j] += temp_autocorr[j] << AUTOCORR_LEFT_SHIFT;
  }

  /* The next statement takes care of the smoothing to make sure that the
     autocorr[0] is nonzero, and adds extra noise proportional to the signal
    energy, which is determined by AUTOCORR_EXTRA_VARIANCE_EXPONENT.  This will
    allow us to put a bound on the value of the LPC coefficients so we don't
    need to worry about integer overflow.
    (Search for: "NOTE ON BOUNDS ON LPC COEFFICIENTS")  */
  lpc->autocorr[0] +=
      ((int64_t)(AUTOCORR_BLOCK_SIZE*AUTOCORR_EXTRA_VARIANCE)<<AUTOCORR_LEFT_SHIFT) +
      temp_autocorr[0] << (AUTOCORR_LEFT_SHIFT - AUTOCORR_EXTRA_VARIANCE_EXPONENT);

  /* We will have copied the max_exponent from the previous LpcComputation
     object, and it will usually already have the correct value.  Return
     immediately if so.  */
  int exponent = lpc->max_exponent;
  int64_t autocorr_0 = lpc->autocorr[0];
      twice_autocorr_0 = autocorr_0 * 2;
      assert(autocorr_0 != 0);
  if ((twice_autocorr_0 >> exponent) == 1) {
    /*  max_exponent has the correct value.  This is the normal code path. */
    return;
  }
  while ((twice_autocorr_0 >> exponent) == 0)
    exponent--;
  while ((twice_autocorr_0 >> exponent) > 1)
    exponent++;
  assert((twice_autocorr >> exponent) == 1 &&
         autocorr >> exponent == 0);
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
     coefficients of the signal (and also, the `lpc_order` samples of context
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


  /**  The input signal that we are compressing  */
  const int16_t *input_signal;

  /**  The stride associated with `input_signal`; normally 1 */
  int input_signal_stride;

  /** The compressed code that we are generating, one byte per sample.  This
      pointer does *not* point to the start of the header (it has been shifted
      forward by 4).  It points to the byte for t == 0.
      ; the code for the t'th signal value is located at
      compressed_code[t].  */
  int8_t *compressed_code;

  /**  The stride associated with `compressed_code`; normally 1 */
  int compressed_code_stride;
};




/*******************
   The lilcom_header functions below mostly serve to document the format
   of the header, we could just have put the statements inline, which is
    what we hope the compiler will do.
   **************************/

/** Set the exponent for frame -1 in the header.  This also sets the
    0x7 magic number in the 1st 4 bits.  */
inline void lilcom_header_set_exponent_m1(int8_t *header, int stride,
                                          int exponent) {
  assert(exponent >= 0 && exponent <= 12);
  header[0 * stride] = (int8_t)(exponent << 4 | 0x7);
}

/** The exponent for the phantom sample at t = -1 is located in the
    4 highest bits of the first byte.  This function returns that
    value (it does not check it is in the range (0..12)).  */
inline int lilcom_header_get_exponent_m1(const int8_t *header,
                                         int stride) {
  return ((int)(uint8_t)(header[0 * stride])) >> 4;
}

/**  Check that this is plausibly a lilcom header.  The low-order 4 bits of the
     first byte of the header are used for this; the magic number is 7.  */
inline int lilcom_header_plausible(const int8_t *header,
                                   int stride) {
  return (header[0 * stride] & 0xF == 0x7) &&
      lilcom_header_get_exponent_m1(header, stride) <= 12;
}

/**  Initialize the header.  Currently it's only necessary to zero byte 3 (which
     isn't always used; when we're compressing float data it stores an
     exponent there);
     Other bytes will be set up while the stream is compressed.  */
inline int lilcom_header_init(int8_t *header, int stride) {
  header[3 * stride] = (int8_t)0;
}

/** Set the LPC order in the header.  This goes in byte 1.  Currently it only
    uses 4 bits since the max LPC order allowed is currently 15.  */
inline void lilcom_header_set_lpc_order(int8_t *header,
                                        int stride, int lpc_order) {
  assert(lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER);
  header[1 * stride] = (int8_t)lpc_order;
}

/** Return the LPC order from the header.  Does no range checking!  */
inline int lilcom_header_get_lpc_order(const int8_t *header, int stride) {
  return (int)(header[1 * stride]);
}

/** Set the -1'th sample's mantissa in the header.  This goes in byte 2,
    the higher-order 6 bits (the lower-order 2 are currently unused).
    Using the higher-order bits is easier w.r.t signed integers.
*/
inline void lilcom_header_set_mantissa_m1(int8_t *header,
                                          int stride, int mantissa) {
  assert(mantissa >= -32 && mantissa <= 31);
  header[2 * stride] = (int8_t) (mantissa * 4);
}
/** Return the -1'th sample's mantissa from the header.  */
inline void lilcom_header_get_mantissa_m1(const int8_t *header,
                                          int stride) {
  return ((int)header[2 * stride]) / 4;
}



/**
   Computes the least exponent (subject to a caller-specified floor) which
   is sufficient to encode (an approximation of) this residual; also
   computes the associated mantissa and the next predicted value

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
                     cases, we need to reduce the magnitude of the mantissa to
                     stay within the allowed range of int16_t (this avoids us
                     having to implement extra checks in the decoder).

       @param [in] min_exponent  A caller-supplied floor on the exponent;
                     must be in the range [0, 12].  This function will
                     never return a value less than this.  min_exponent will
                     normally be the maximum of zero and the previous sample's
                     exponent minus 1, but may be more than that if we are
                     backtracking (because future samples need a larger
                     exponent).

       @param [out] mantissa  This function will write an integer in
                    the range [-32, 31] to here, such that
                    (mantissa << exponent) is a close approximation
                    of `residual` and satisfies the property that
                    `predicted + (mantissa << exponent)` does not
                    exceed the range of int16_t.

       @param [out] next_compressed_value  The next compressed value
                    will be written to here; at exit this will contain
                    `predicted + (mantissa << exponent)`.

       @return  Returns the exponent chosen, a value in the range [min_exponent..12].


   The intention of this function is to return the exponent in the range
   [min_exponent..12] which gives the closest approximation to `residual` that
   we could get with any exponent, while choosing the lower exponent in case of
   ties.  This is largely what it does, although it may not always do so in the
   corner cases where we needed to modify the mantissa to not exceed the range
   of int16_t.  The details of how the integer mantissa is chosen (especially
   w.r.t. rounding and ties) is explained in a comment inside the function.

   The following explains how this function chooses the exponent.

   Define the exact mantissa m(e), which is a function of the exponent e,
   as:
            m(e) =  residual / (2^e),
   viewed as an exact mathematical expresion, not as an integer.
   We want to return the smallest value of e such that

     -33.0 <= m(e) <= 31.5.

   This inequality ensures that there will be no loss of accuracy by choosing e
   instead of e+1 as the exponent.  (If we had a larger exponent, the closest
   points we'd be able to reach would be equivalent to m(e) = -34 or +32; and if
   m(e) satisfies the inequality above we'd have no loss of precision by using e
   rather than e + 1 as the exponent.  (Notice that -33 is the midpont of [-34,-32]
   and 31.5 is the midpoint of [31,32]).  Multiplying by two, we can express the
   above in integer math as:

     (-66 << e) <= residual * 2 <= (63 << e)

   NOTE ON THE RANGE OF THE EXPONENT, explaining why it could be as large as 12.
   The largest-magnitude residual is 65535, which is 13767 - (-13768).  Of the
   numbers representable by (mantissa in [-32,31]) << (integer exponent),
   the closest approximation of 65535 is 65536, for which the lowest exponent
   that can generate that number is 12 (65536 = 16 << 12)
   << 12.
*/
inline static int least_exponent(int32_t residual,
                                 int16_t predicted,
                                 int min_exponent,
                                 int *mantissa,
                                 int16_t *next_decompressed_value) {
  assert (min_exponent >= 0 && min_exponent <= 12); /* TODO: remove this */
  int exponent = min_exponent;
  int32_t residual2 = residual * 2,
      minimum = -66 << exponent,
      maximum = 63 << exponent;
  while (residual2 < minimum || residual2 > maximum) {
    minimum *= 2;
    maximum *= 2;
    exponent++;
  }
  assert(exponent <= 12);

  {
    /**
       This code block computes 'mantissa', the integer mantissa which we call
       which should be a value in the range [-32, 31].


       The mantissa will be the result of rounding (residual /
       (float)2^exponent) to the nearest integer (see below for the rounding
       behavior in case of ties, which we randomize); and then, if the result is
       -33 or +32, changing it to -32 or +31 respectively.

       What we'd like to do, approximately, is to compute

           mantissa = residual >> exponent

       where >> can be interpreted, roughly, as division by 2^exponent; but we
       want some control of the rounding behavior.  To maximize accuracy we want
       to round to the closest, like what round() does for floating point
       expressions; but we want some control of what happens in case of ties.
       Always rounding towards zero might possibly generate a slight ringing, in
       certain circumstances (it's a kind of bias toward the LPC prediction), so
       we want to round in a random direction (up or down).  We choose to round
       up or down, pseudo-randomly.

       We'll use (predicted%2) as the source of randomness.  This will be
       sufficiently random for loud signals (varying by more than about 1 or 2
       from sample to sample); and for very quiet signals (magnitude close to 1)
       we'll be exactly coding it anyway so it won't matter.

       Consider the expression:

          mantissa = (residual*2 + offset) >> (exponent + 1)

       (and assume >> is the same as division by a power of 2 but rounding down;
       the C standard coesn't guarantee this behavior but we'll fix that in a
       different way).

        and consider two possibilities for `offset`.
       (a)  offset = (1<<exponent)
       (b)  offset = ((1<<exponent) - 1)

       In case (a) it rounds up in case of ties (e.g. if the residual is 6 and
       exponent is 2 so we're rounding to a multiple of 4).  In case (b)
       it rounds down.  By using:
         offset = ((1<<exponent) - (predicted&1))
       we are randomly choosing (a) or (b) based on randomness in the
       least significant bit of `predicted`.

       OK, imagine the code was as follows:

    int offset = ((1 << exponent) - (predicted&1)),
        local_mantissa = (residual2 + offset) >> (exponent + 1);

       The above code would do what we want on almost all platforms (since >> on
       a signed number is normally arithmetic right-shift in practice, which is
       what we want).  But we're going to modify it to guarantee correct
       behavior for negative numbers, since the C standard says right shift of
       negative signed numbers is undefined.

       We do this by adding 512 << exponent to "offset".  This becomes 256 when
       shifting right by (exponent + 1), which will disappear when casting to
       int8_t.  The 512 also guarantees that (residual2 + offset) is positive
       (since 512 is much greater than 32), and this ensures well-defined
       rounding behavior.  Note: we will now explicitly make `offset` an int32_t because
       we don't want 512 << 12 to overflow int is int16_t (super unlikely, I know).
    */
    int32_t offset = (((int32_t)(512+1)) << exponent) - (predicted&1);
    int local_mantissa = (int)((int8_t)((residual2 + offset) >> (exponent + 1)));

    assert(local_mantissa >= -33 && local_mantissa <= 32);

    /* We can't actually represent -33 in 6 bits, but we choose to retain this
       exponent, in this case, because -33 is as close to -32 (which is
       representable) as it is to -34 (which is the next closest thing
       we'd get if we used a one-larger exponent).  */
    if (local_mantissa == -33)
      local_mantissa = -32;
    /*  The following could happen if we really wanted the mantissa to be
        exactly 31.5, and `predicted` was even so we rounded up.  It would only
        happen in case of ties, so we lose no accuracy by doing this. */
    if (local_mantissa == 32)
      local_mantissa = 31;

    int32_t next_signal_value =
        ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);

    {
      /* Just a check.  I will remove this block after it's debugged.
         Checking that the error is in the expected range.  */
      int16_t error = (int16_t)(next_signal_value - residual);
      if (error < 0) error = -error;
      if (local_mantissa > -32) {
        assert(error <= (1 << exponent) >> 1);
      } else {
        /** The error could be twice as large if the desired mantissa was -33,
            which we would have converted to -32.  (But the error is no larger
            than if we had chosen a one-larger exponent).  */
        assert(error <= (1 << exponent));
      }
    }

    if (next_signal_value != (int16_t)next_signal_value) {
      /** The next signal exceeds the range of int16_t; this can in principle
        happen if the predicted signal was close to the edge of the range
        [-32768..32767] and quantization effects took us over the edge.  We
        need to reduce the magnitude of the mantissa by one in this case. */
      local_mantissa -= lilcom_sgn(local_mantissa);
      next_signal_value =
          ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);
      assert(next_signal_value == (int16_t)next_signal_value);
    }

    *next_decompressed_value = next_signal_value;
    *mantissa = local_mantissa;
  }
  return exponent;
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
     absolute value of all elements is less than 1<<LPC_EST_LEFT_SHIFT (but as large
     as possible given that constraint).  We don't need to know how much it was
     shifted right, because a scale on the autocorrelation doesn't affect the
     LPC coefficients.  */
  int32_t autocorr[MAX_LPC_ORDER];

  {
    int max_exponent = autocorr_in->max_exponent;
    assert(max_exponent >= AUTOCORR_LEFT_SHIFT &&
           (autocorr_in->autocorr[0]) >> (max_exponent - 1) == 1);
    if (max_exponent > LPC_EST_LEFT_SHIFT) {
      /* shift right (the normal case).  We do it by division, because
         technically the result of shifting a negative number right
         is undefined (even though it normally does what we want,
         which is to duplicate the left-most -1) */
      int right_shift = max_exponent - LPC_EST_LEFT_SHIFT;
      for (i = 0; i <= lpc_order; i++)
        autocorr[i] = autocorr_in->autocorr[i] / (1 << right_shift);
    } else {
      int left_shift = LPC_EST_LEFT_SHIFT - max_exponent;
      for (i = 0; i <= lpc_order; i++)
        autocorr[i] = autocorr_in->autocorr[i] << left_shift;
    }
    assert((autocorr[0] >> (LPC_EST_LEFT_SHIFT - 1)) == 1);
    for (i = 1; i <= lpc_order; i++) {  /* TODO: remove this loop. */
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
  int32_t temp[MAX_LPC_ORDER];


  int32_t E = autocorr[0];

  int j;
  for (int i = 0; i < lpc_order; i++) {
    /** ki will eventually be the next reflection coefficient, a value in [-1, 1], but
        shifted left by LPC_EST_LEFT_SHIFT to represent it in fixed point.
        But after the following line it will represent a floating point
        number times 2 to the power 2*LPC_EST_LEFT_SHIFT,
        so currently abs(ki) < 2^(LPC_EST_LEFT_SHIFT+LPC_EST_LEFT_SHIFT) = 2^46
      Original code: "float ki = autocorr[i + 1];"  */
    int64_t ki = autocorr[i + 1] << LPC_EST_LEFT_SHIFT;

    for (j = 0; j < i; j++) {
      /** max magnitude of the terms added below is 2^(LPC_EST_LEFT_SHIFT*2 + 8) = 2^54, i.e.
          the abs value of the added term is less than 2^54.
          ki still represents a floating-point number times 2 to the
          power 2*LPC_EST_LEFT_SHIFT.
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
        reflection coefficient; and it is stored times 2 to the power LPC_EST_LEFT_SHIFT, so its
        magnitude as an integer is <= 2^LPC_EST_LEFT_SHIFT.  Check that it's less
        than 2^LPC_EST_LEFT_SHIFT plus a margin to account for rounding errors.  */
    assert(lilcom_abs(ki) < (1<<LPC_EST_LEFT_SHIFT + 1<<(LPC_EST_LEFT_SHIFT - 8)));

    /**  Original code: "float c = 1 - ki * ki;"  Note: ki*ki is nonnegative,
         so shifting right is well defined. */
    int64_t c = (((int64_t)1) << LPC_EST_LEFT_SHIFT) - ((ki*ki) >> LPC_EST_LEFT_SHIFT);

    /** c is the factor by which the residual has been reduced; mathematically
        it is always >= 0, but here it must be > 0 because of our smoothing of
        the variance via AUTOCORR_EXTRA_VARIANCE_EXPONENT and
        AUTOCORR_EXTRA_VARIANCE which means the residual can never get to
        zero.*/
    assert(c > 0);
    /** The original code did: E *= c;
        Note: the product is int64_t because c is int64_t, which is important
        to avoid overflow.  Also note: it's only well-defined to right-shift
        because E is nonnegative.
    */
    E = (int32_t)((E * c) >> LPC_EST_LEFT_SHIFT);

    /** compute the new LP coefficients
        Original code did: pTmp[i] = -ki;
        Note: abs(temp[i]) <= 2^LPC_EST_LEFT_SHIFT, since ki is in the range [-1,1]
        when viewed as a real number. */
    temp[i] = -((int32_t)ki);
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
          (int32_t)((ki * lpc->lpc_coeffs[i - j - 1]) / (1 << LPC_EST_LEFT_SHIFT));
    }
    for (j = 0; j <= i; j++) {
      assert(lilcom_abs(temp[j]) < ((int64_t)1<<(LPC_EST_LEFT_SHIFT + 8)));
      lpc->lpc_coeffs[j] = temp[j];
    }
  }
  /** E > 0 because we added fake extra variance via
     AUTOCORR_EXTRA_VARIANCE_EXPONENT and AUTOCORR_EXTRA_VARIANCE, so according
     to these stats the sample should never be fully predictable.  We'd like to
     assert that E <= autocorr[0] because even if the data is totally
     uncorrelated, we should never be increasing the predicted error vs. having
     no LPC at all.  But I account for the possibility that in pathological
     cases, rounding errors might make this untrue.  */
  assert(E > 0 && E <= (autocorr[0] + autocorr[0] >> 10));


  /**
     We want to shift the LPC coefficients right by LPC_EST_LEFT_SHIFT -
     LPC_APPLY_LEFT_SHIFT, because we store and apply them with lower precision
     than we estimate them.  We divide rather than shift because technically
     the result of right-shifting a negative number is implementation-defined.
  */
  for (i = 0; i < lpc_order; i++) {
    lpc->lpc_coeffs[i] /= (1 << (LPC_EST_LEFT_SHIFT - LPC_APPLY_LEFT_SHIFT));
  }
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
inline int16_t lilcom_compute_predicted_value(
    CompressionState *state,
    int64_t t) {
  int32_t block_index = ((int32_t)t) >> LOG_AUTOCORR_BLOCK_SIZE;
  /** block_index & 1 is just block_index % 2 (note: block_index is nonnegative)  */
  LpcComputation *lpc = &(state->lpc_computations[block_index & 1]);

  int32_t lpc_order = state->lpc_order;
  /* Initialize the sum.  The term (1 << (LPC_APPLY_LEFT_SHIFT - 1)) is
     basically 0.5 in our fixed-point representation; this is added so that
     later when we shift right we are effectively rounding to the closest value
     instead of rounding down.

     The term (1 << (32 + LPC_APPLY_LEFT_SHIFT)) is to ensure that the sum remains
     positive so that, later, when we compute (sum >> LPC_APPLY_LEFT_SHIFT), we know
     exactly what the behavior will be (right-shift on negative signed integers
     officially gives undefined behavior).  After shifting right by
     LPC_APPLY_LEFT_SHIFT this becomes 1 << 32, which will overflow and be equivalent
     to zero when cast to int32_t. */
  int64_t sum = (1 << (LPC_APPLY_LEFT_SHIFT - 1)) + (1 << (32 + LPC_LEFT_SHIFT));

  /** Get the pointer to the t'th signal in the circular buffer
      'state->decompressed_signal'.  The buffer has an extra MAX_LPC_ORDER
      samples at the start, to provide needed context when we roll around,
      to keep this code simple. */
  int16_t *decompressed_signal_t =
      &(state->decompressed_signal[MAX_LPC_ORDER + (t&(SIGNAL_BUFFER_SIZE-1))]);

  int i;
  for (i = 0; i < lpc_order; i++)
    sum += ((int64_t)lpc->lpc_coeffs[i]) * decompressed_signal_t[-1-i];

  /** The lpc_coeffs were stored shifted left by LPC_LEFT_SHIFT; shift the sum
      back to get the integer prediction.  See the initialization of `sum` for
      info on how this rounds.  (effectively: to the closest, rounding up in
      case of (very rare!) ties).  */
  int32_t predicted = (int32_t)(sum >> LPC_LEFT_SHIFT);

  /** We need to truncate `predicted` to fit within the range that int16_t can
      represent. (note: in principle we could have chosen to just let it wrap
      around and accept that the prediction will be terrible in those cases;
      that would worsen fidelity but increase speed slightly. */
  if (predicted != (int16_t)predicted) {
    if (predicted > 32767)
      predicted = 32767;
    else if (predicted < -32768)
      predicted = -32768;
  }
  return (int16_t)predicted;
}


/** Copies the final state->lpc_order samples from the end of the
    decompressed_signal buffer to the beginning in order to provide required
    context when we roll around.  This function is expected to be called only
    when t is a nonzero multiple of SIGNAL_BUFFER_SIZE. */
inline void lilcom_copy_to_buffer_start(
    CompressionState *state) {
  for (int i = 1; i <= state->lpc_order; i++)
    state->decompressed_signal[MAX_LPC_ORDER - i] =
        state->decompressed_signal[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE - i];
}


/**
   This function updates the autocorrelation statistics and, if relevant, the
   LPC coefficients.
      @param [in] t    Sample index of the sample that we are about to compress.
                   Required to be a multiple of AUTOCORR_BLOCK_SIZE.  The data we're
                   going to use to update the autocorrelation statistics
                   are those from the *previous* block, from at t -
                   AUTOCORR_BLOCK_SIZE to t-1.  (If t == 0 we will do nothing and rely
                   on the initialization previously done in lilcom_init_lpc).

      @param [in,out]  state   Struct containing the computation state.
                   We are modifying one of the `lpc_computations` elements.
 */
void lilcom_update_autocorrelation_and_lpc(
    int64_t t, CompressionState *state) {
  assert(t % AUTOCORR_BLOCK_SIZE == 0 && state->lpc_order > 0);
  int32_t block_index = ((int32_t)t) >> LOG_AUTOCORR_BLOCK_SIZE;
  /** We'll compute the LPC coeffs if t is a multiple of LPC_COMPUTE_INTERVAL or
      if t is a nonzero value less than LPC_COMPUTE_INTERVAL (for LPC freshness
      at the start).  */
  int compute_lpc = (t & (LPC_COMPUTE_INTERVAL - 1) == 0);
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
  /** Interpret `block_index & 1` as `block_index % 2` (valid for nonnegative
      block_index, which it is). */
  LpcComputation *this_lpc = &(state->lpc_computations[block_index & 1]);
  // prev_lpc is the 'other' LPC object in the circular buffer of size 2.
  LpcComputation *prev_lpc = &(state->lpc_computations[!(block_index & 1)]);
  assert(prev_lpc != this_lpc);  // TODO: remove this.

  /** Copy the previous autocorrelation coefficients and max_exponent.  We'll
      either re-estimate or copy the LPC coefficients below. */
  for (int i = 0; i <= lpc_order; i++)
    this_lpc->autocorr[i] = prev_lpc->autocorr[i];
  this_lpc->max_exponent = prev_lpc->max_exponent;

  int64_t prev_block_start_t = t - AUTOCORR_BLOCK_SIZE;
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
                                signal_pointer);
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
   lilcom_compress_for_time_internal attempts to compress the signal for time t;
   on success, it will write to
   state->compressed_code[t*state->compressed_code_stride].

      @param [in] t     The time that we are requested to compress the signal for.
                        Requires t > 0 (c.f. lilcom_comprss_for_time_zero).
      @param [in] prev_exponent   The exponent value that was used to compress the
                        previous sample.  Must be in the range [0..12].
      @param [in] min_exponent  A value in the range [0..12] which puts
                        a lower limit on the exponent that this function will
                        use for this sample.  min_exponent is required, in
                        addition to being in [0..12], to be in the range
                        [prev_exponent-1 .. prev_exponent+2].
      @param [in,out] state  Contains the computation state and pointers to the
                       input and output data.

   On success (i.e. if it was able to do the compression) it returns the
   exponent used, which is a number >= 0.

   On failure, which can happen if the exponent required to compress this value
   was greater than prev_exponent + 2, it returns the negative of the exponent
   that would have required to compress this sample.  this will cause us to
   enter backtracking code to inrease the exponent used on the previous sample.
*/
inline int lilcom_compress_for_time_internal(
    int64_t t,
    int prev_exponent,
    int min_exponent,
    CompressionState *state) {
  assert(t > 0 && prev_exponent >= 0 && prev_exponent <= 12 &&
         min_exponent >= 0 && min_exponent <= 12 &&
         min_exponent >= prev_exponent - 1 &&
         min_exponent <= prev_exponent + 2);

  if ((t & (AUTOCORR_BLOCK_SIZE - 1)) == 0 && state->lpc_order != 0) {
    if ((t & (SIGNAL_BUFFER_SIZE - 1)) == 0) {
      /**  If this is the start of the uncompressed_signal buffer we need to
           make sure that the required left context is copied appropriately. */
      lilcom_copy_to_buffer_start(t, state);
    }
    /** Update the autocorrelation coefficients and possibly the LPC
        coefficients. */
    lilcom_update_autocorrelation_and_lpc(t, state);
  }

  int16_t predicted_value = lilcom_compute_predicted_value(state, t),
      observed_value = input_signal[t * state->input_signal_stride];
  /** cast to int32 when computing the residual because a difference of int16's may
      not fit in int16. */
  int32_t residual = ((int32_t)observed_value) - ((int32_t)predicted_value);

  int mantissa,
      exponent = least_exponent(
          residual, predicted_value,
          min_exponent, &mantissa,
          &(state->decompressed_signal[MAX_LPC_ORDER+(t&(SIGNAL_BUFFER_SIZE-1))]]));

  if (exponent <= prev_exponent + 2) {
    /** Success; we can represent the difference of exponents in the range
        [-1..2].  This is the normal code path. */
    int exponent_code = (exponent - prev_exponent + 1);
    assert(exponent_code >= 0 && exponent_code < 4 &&
           mantissa >= -32 && mantissa < 32);
    state->compressed_code[t*state->compressed_code_stride] =
        (int8_t)((mantissa << 2) + exponent_code);
    state->exponents[t & (EXPONENT_BUFFER_SIZE - 1)] = exponent;
    return exponent;
  } else {
    /** Failure.  The calling code will backtrack, increase the previous
        exponent to at least exponent - 2, and try again.  */
    return -exponent;
  }
}

/*
  This function is a special case of compressing a single sample, for t == 0.
  Time zero is a little different because of initialization effects (the header
  contains an exponent and a mantissa for t == -1, which gives us a good
  starting point).

    @param [in] min_exponent  A number in the range [0, 12]; the caller
                  requires the exponent for time t = 0 to be >= min_exponent.
                  (Normally 0, but may get called with values >0 if
                  called from backtracking code.)
    @param [in,out] state  Stores shared state and the input and output
                  sequences.  The primary output is to
                  state->compressed_code[0], but the 4-byte header is also
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
      (min_exponent <= 2 ? 0 : min_exponent - 2);
  int16_t signal_m1,  /* compressed signal for "phantom sample" -1. */
      predicted_m1 = 0,
      residual_m1 = first_signal_value;
  int mantissa_m1,
      exponent_m1 = least_exponent(residual_m1,
                                   predicted_m1,
                                   sample_m1_min_exponent,
                                   &mantissa_m1, &signal_m1);
  assert(exponent_m1 >= 0);  /* TODO: remove this. */
  int c_stride = state->compressed_code_stride;

  int header_stride = state->compressed_code_stride;
  int8_t *header = state->compressed_code -
      (LILCOM_HEADER_BYTES * header_stride);


  lilcom_header_set_exponent_m1(header, header_stride, exponent_m1);
  assert(lilcom_header_get_exponent_m1(header, header_stride)
         == exponent_m1);  /* TEMP */
  lilcom_header_set_mantissa_m1(header, header_stride, mantissa_m1);
  assert(lilcom_header_get_mantissa_m1(header, header_stride)
         == mantissa_m1);  /* TEMP */

  /** store the initial exponent, at sample -1.  This probably won't be
      accessed, actually.  [TODO: remove this?] */
  state->exponents[EXPONENT_BUFFER_SIZE - 1] = exponent_m1;

  if (exponent_m1 - 1 > min_exponent)
    min_exponent = exponent_m1 - 1;

  /** The autocorrelation parameters for the first block say "simply copy the
      previous sample".  We do this manually here rather than accessing
      the LPC coefficients. */
  int32_t predicted_0 = signal_m1,
      residual_0 = first_signal_value - predicted_0;

  int mantissa_0,
      exponent_0 = least_exponent(residual_0,
                                  predicted_0,
                                  min_exponent,
                                  &mantissa_0,
                                  &(state->decompressed_signal[MAX_LPC_ORDER + 0]));
  int delta_exponent = exponent_0 - exponent_m1;
  /** The residual cannot be greater in magnitude than first_value, since we
      already encoded first_signal_value and we are now just dealing with the
      remaining part of it, so whatever exponent we used for sample -1 would
      be sufficiently large for sample 0; that's how we can guarantee
      delta_exponent <= 2.  */
  assert(delta_exponent >= -1 && delta_exponent <= 2 &&
         exponent_0 >= min_exponent && mantissa_0 >= -32
         && mantissa0 <= 31);

  state->compressed_code[0 * c_stride] =
      (int16_t)((mantissa_0 << 2) + (delta_exponent + 1));


  for (int i = 0; i < MAX_LPC_ORDER; i++) {
    /** All samples prior to t=0 are treated as having zero value for purposes of
        computing LPC coefficients.  (The phantom sample -1 is not involved
        here. */
    state->decompressed_signal[i] = 0;
  }
  state->exponents[0] = exponent_0;
}

/**
   This is a version of lilcom_compress_for_time that is called when we needed
   an exponent larger than the previous exponent plus 2, so we have to backtrack
   to increase the exponent for previous samples.  Basically, it handles the
   hard cases that lilcom_compress_for_time cannot directly handle.  The main
   purpose of this function is to compress the signal for time t, but to do that
   it may have to recursively go back to previous samples and re-compress those
   in order to get an exponent large enough.

     @param [in] t   The time for which we want to compress the signal;
                   t >= 0.
     @param [in] min_exponent  The caller requires that we compress the
                  signal for time t with an exponent not less than
                  `min_exponent`, even if it was possible to compress it
                  with a smaller exponent.  We require min_exponent >= 0.
     @param [in,out] state  The compression state (will be modified by
                  this function).  The primary output of this function is
                  state->compressed_code[t*state->compressed_code_stride], but
                  it may also modify state->compressed_code for time values
                  less than t.  This function will update other elements
                  of `state` as needed (exponents, autocorrelation and
                  LPC info, etc.).
*/
void lilcom_compress_for_time_backtracking(
    int64_t t, int min_exponent,
    CompressionState *state) {
  assert(t >= 0 && min_exponent >= 0);
  if (t > 0) {
    int prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)];
    if (prev_exponent < min_exponent - 2) {
      /** We need to revisit the exponent for sample t-1, as we're not
          able to encode differences greater than +2. */
      lilcom_compress_for_time_backtracking(t - 1, min_exponent - 2, state);
      prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)];
      assert(prev_exponent >= min_exponent - 2);
    }
    if (min_exponent < prev_exponent - 1) {
      /** lilcom_compress_for_time requires min_exponent to be be in the range
         [prev_component-1..prev_component+2], so we need to increase
         min_exponent.  (Decreasing it would break our contract with the caller;
         increasing it is OK).  */
      min_exponent = prev_exponent - 1;
    }
    int exponent = lilcom_compress_for_time_internal(
        t, prev_exponent, min_exponent, state);
    if (exponent >= 0) {
      return;  /** Normal code path: success.  */
    } else {
      /* Now make `exponent` positive. It was negated as a signal that there was
         a failure: specifically, that exponent required to encode this sample
         was greater than prev_exponent + 2.  [This path is super unlikely, as
         we've already backtracked, but it theoretically could happen].  We can
         deal with it via recursion.  */
      exponent = -exponent;
      assert(exponent > prev_exponent + 2 && exponent > min_exponent);
      lilcom_compress_for_time_backtracking(t, exponent, state);
    }
  } else {
    /* time t=0. */
    lilcom_compress_for_time_zero(min_exponent, state);
  }
}

/**
   Compress the signal for time t; this is the top-level wrapper function
   that takes care of everything for time t.
       @param [in] t   The sample index we are asked to compress.
                    We require t > 0.  (C.f. lilcom_compress_for_time_zero).
       @param [in,out] state    Struct that stores the state associated with
                         the compression, and inputs and outputs.
*/
inline void lilcom_compress_for_time(
    int64_t t,
    CompressionState *state) {
  assert(t > 0);
  int prev_exponent =
      state->exponents[(t - 1) & (EXPONENT_BUFFER_SIZE - 1)],
      min_exponent = (prev_exponent == 0 ? 0 : prev_exponent - 1);

  int exponent = lilcom_compress_for_time_internal(
      t, prev_exponent, min_exponent, state);
  if (exponent >= 0) {
    /** lilcom_compress_for_time_internal succeeded; we are done.  */
    return;
  } else {
    /** The returned exponent is negative; it's the negative of the exponent
        that was needed to compress sample t.  The following call will handle
        this more difficult case. */
    lilcom_compress_for_time_backtracking(t, -exponent, state);
  }
}

/**
   Initializes a newly created CompressionState struct, setting fields and doing
   the compression for time t = 0 which is a special case.

   Does not check its arguments; that is assumed to have already been done
   in calling code.
 */
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
      output + (LILCOM_HEADER_BYTES * output_stride);
  state->compressed_code_stride = output_stride;

  for (int i = 0; i < MAX_LPC_ORDER; i++)
    state->decompressed_signal[i] = 0;


  lilcom_header_init(output, output_stride);
  lilcom_header_set_lpc_order(header, header_stride,
                              state->lpc_order);
  assert(lilcom_header_get_lpc_order(output, output_stride) == lpc_order);  /* TEMP */

  /** The remaining parts of the header will be initialized in
      lilcom_compress_for_time_zero`. */
  int min_exponent = 0;
  lilcom_compress_for_time_zero(min_exponent, state);
}

/** This function does nothing; it only exists to check that
    various relationships between the #defined constants are satisfied.
*/
inline int lilcom_check_constants() {
  assert(MAX_LPC_ORDER >> LPC_ORDER_BITS == 0);
  assert(LPC_LEFT_SHIFT + (AUTOCORR_EXTRA_VARIANCE_EXPONENT+1)/2 <= 31);
  assert((30 + AUTOCORR_LEFT_SHIFT + log(AUTOCORR_BLOCK_SIZE) + AUTOCORR_DECAY_EXPONENT)
         <= 61);
  assert((AUTOCORR_BLOCK_SIZE & (AUTOCORR_BLOCK_SIZE-1)) == 0);  // Power of 2.
  assert(AUTOCORR_BLOCK_SIZE > MAX_LPC_ORDER);
  assert(LPC_COMPUTE_INTERVAL % AUTOCORR_BLOCK_SIZE == 0);
  assert(AUTOCORR_BLOCK_SIZE >> LOG_AUTOCORR_BLOCK_SIZE == 1);
  {
    int64_t n1 = int64_t(AUTOCORR_DECAY_SQRT);
    n1 *= n1;
    n1 >= AUTOCORR_LEFT_SHIFT;
    int64_t n2 = 1 << AUTOCORR_LEFT_SHIFT - (1 << (AUTOCORR_LEFT_SHIFT - AUTOCORR_DECAY_EXPONENT));
    assert(lilcom_abs(n1 - n2) <= 3);
  }
  /** assumed in some code where we left shift by the difference.*/
  assert(AUTOCORR_LEFT_SHIFT >= AUTOCORR_EXTRA_VARIANCE_EXPONENT);

  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  /* The y < x / 2 below just means "y is much less than x". */
  assert(LPC_COMPUTE_INTERVAL < (AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT) / 2);
  assert((EXPONENT_BUFFER_SIZE-1)*2 > 12);
  assert(EXPONENT_BUFFER_SIZE & (EXPONENT_BUFFER_SIZE-1) == 0);  /* Power of 2. */
  assert(EXPONENT_BUFFER_SIZE > (12/2) + 1); /* should exceed maximum range of exponents,
                                                divided by  because that's how much they can
                                                change from time to time.  The + 1 is
                                                in case I missed something. */

  assert(EXPONENT_BUFFER_SIZE & (EXPONENT_BUFFER_SIZE-1) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE % AUTOCORR_BLOCK_SIZE == 0);
  assert(SIGNAL_BUFFER_SIZE & (SIGNAL_BUFFER_SIZE - 1) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE > AUTOCORR_BLOCK_SIZE + EXPONENT_BUFFER_SIZE + MAX_LPC_ORDER);
  return 1;
}


/*  See documentation in lilcom.h  */
int lilcom_compress(int64_t num_samples,
                    const int16_t *input, int input_stride,
                    int8_t *output, int output_stride,
                    int lpc_order) {

  assert(lilcom_check_constants());

  if (!num_samples > 0 && input_stride != 0 && output_stride != 0 &&
      lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER) {
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
   This function does the core part of the decompression of one sample
   (excluding the part about updating the autocorrelation statistics and
   updating the LPC coefficients; that is done externally.

      @param [in] input_code  The int8_t code for the sample that
                      we are about to decompress.
      @param [in] lpc_order  The order of the LPC computation,
                     a number in [0..LPC_MAX_ORDER] obtained from
                     the header.
      @param [in] lpc_coeffs  The LPC coefficients, multiplied
                     by 2^23 and represented as integers.
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
      @param [in]

 */
inline int lilcom_decompress_for_time_internal(
    int8_t input_code,
    int lpc_order,
    const int32_t *lpc_coeffs,
    int16_t *output_sample,
    int *exponent) {


  int16_t predicted_value;


  { // This block computes the predicted value.  For explanation please
    // look for similar statements in the compression code, which
    // are better documented.
    int64_t sum = (1 << (LPC_LEFT_SHIFT - 1)) + (1 << (32 + LPC_LEFT_SHIFT));
    int i;
    for (i = 0; i < lpc_order; i++)
      sum += ((int64_t)lpc_coeffs[i] pc->lpc_coeffs[i]) * decompressed_signal_t[-1-i];



}

/*  See documentation in lilcom.h  */
int lilcom_decompress(int64_t num_samples,
                      const int8_t *input, int input_stride,
                      int16_t *output, int output_stride) {
  if (!lilcom_header_plausible(input))
    return 1;  // Error status.


}



  if (!num_samples > 0 && input_stride != 0 && output_stride != 0 &&
      lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER) {
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

define LILCOM_TEST 1

#ifdef LILCOM_TEST
int main() {
  lilcom_check_constants();
}
#endif
