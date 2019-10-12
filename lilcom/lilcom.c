#include "lilcom.h"
#include <assert.h>
#include <stdlib.h>  /* for malloc */
#include <math.h>  /* for frexp and frexpf and pow, used in floating point compression. */
#include <stdio.h>  /* TEMP-- for debugging. */
#include <float.h>  /* for FLT_MAX */


/**
   The version number of the format.   Note: you can't change the various
   constants below without changing LILCOM_VERSION, because it will mess up the
   decompression.  That is: the decompression method requires exact
   compatibility with the compression method, w.r.t. how the LPC coefficients
   are computed.

   The issue is that this algorithm is based on computing LPC coefficients from
   the previous decompressed samples, and then encoding the residual after
   linear prediction.  If there is any change, however slight, in the
   computation of the LPC coefficients it could affect a sample, which would
   further change the LPC coefficients, and it might (in principle) cause
   unbounded error.  So this compression method is totally unsuitable for
   situations in which data corruption or loss might happen within a sequence.
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

   Must be even (relates to a loop-unrolling trick used in
   lilcom_compute_predicted_value.  We use 14 not 16 because it has to be less
   than AUTOCORR_BLOCK_SIZE.
*/
#define MAX_LPC_ORDER 14


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
   in int32_t.

   CAUTION: this cannot be greater than 14 = 30 - 16, due to tricks
   used in the prediction code.
 */
#define LPC_APPLY_LEFT_SHIFT 14

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
    lilcom_update_autocorrelation; this is to avoid overflow).
   Currently the sum above equals: 30 + 20 + 5 + 3 + 1 = 59  <= 61.  I'm giving
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
#define AUTOCORR_BLOCK_SIZE 16
/**
   LOG_AUTOCORR_BLOCK_SIZE must be the log-base-2 of AUTOCORR_BLOCK_SIZE.
   Used in bitwise tricks for division.
 */
#define LOG_AUTOCORR_BLOCK_SIZE 4

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
#define AUTOCORR_DECAY_EXPONENT 6


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

   This number was computed by typing: floor(sqrt(1 - 2^-5) * 2^20)
   into wolfram alpha.
*/
#define AUTOCORR_DECAY_SQRT 1040351



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
   to increase the exponent.  The most we'd ever need to add to the exponent
   is +12, and we can only increase the exponent every other frame, so
   this needs to be at least 24 (or maybe one or two more than that, I forget
   the exact logic.)

   It must be a power of 2 due to a trick used to compute a modulus, which
   dictates 32.
 */
#define EXPONENT_BUFFER_SIZE 32


/**
   Size of rolling buffer of struct LpcComputation.  Must satisfy
   (LPC_ROLLING_BUFFER_SIZE - 1) * AUTOCORR_BLOCK_SIZE > 24
   (or maybe 22... but greater than 24 should be enough.)
   This is to ensure we can never backtrack past all of the buffers,
   to prevent us overwriting some autocorrelation coefficients we
   will need in future.

   (Note: 24 is a number greater than twice the range of possible exponents,
   which is 0..11; the "twice" is because the fastest the exponent
   can increase is every other frame.)
 */
#define LPC_ROLLING_BUFFER_SIZE 4

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
static inline int lilcom_sgn(int val) {
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


  /**
     autocorr_to_remove contains some terms which are *in* autocorr but which
     need to be removed every time we update the autocorrelation coefficients.
     Let the 'baseline' autocorrelation be that of a signal that is scaled
     down exponentially as you go further back in time.
   */
  int64_t autocorr_to_remove[MAX_LPC_ORDER + 1];

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
     (lpc_order is passed directly into functions dealing with this object).
  */
  int32_t lpc_coeffs[MAX_LPC_ORDER];
};


/**
   Initialize an LpcComputation object with the default parameter that we have
   at the start of the file.  This corresponds to all-zero autocorrelation
   stats, max_exponent = 0, and lpc_coeffs that correspond to [1.0 0 0 0].
*/
static void lilcom_init_lpc(struct LpcComputation *lpc) {
  for (int i = 0; i <= MAX_LPC_ORDER; i++) {
    lpc->autocorr[i] = 0;
    lpc->autocorr_to_remove[i] = 0;
  }
  lpc->max_exponent = 0;
  /* The LPC coefficientss are stored shifted left by LPC_APPLY_LEFT_SHIFT, so this
     means the 1st coeff is 1.0 and the rest are zero-- meaning, we start
     prediction from the previous sample.  */
  lpc->lpc_coeffs[0] = 1 << LPC_APPLY_LEFT_SHIFT;
  for (int i = 1; i < MAX_LPC_ORDER; i++) {
    /** It's important that we go beyond the `lpc_order`'th element while
        zeroing here.  If `lpc_order` is odd, we may access the one-past-the-end
        element as part of a loop unrolling used in
        lilcom_compute_predicted_value(), so we need that to be zero. */
    lpc->lpc_coeffs[i] = 0;
  }
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
static inline void lilcom_update_autocorrelation(
    struct LpcComputation *lpc, int lpc_order,
    const int16_t *signal) {
  /** 'temp_autocorr' will contain the raw autocorrelation stats without the
      shifting left by AUTOCORR_LEFT_SHIFT; we'll do the left-shifting at the
      end, to save an instruction in the inner loop).  */
  int64_t temp_autocorr[MAX_LPC_ORDER + 1];
  int i;

  /** Remove the temporary term in autocorr_to_remove (that arises from
      reflecting the signal).  Then scale down the current data slightly.  This
      is to form an exponentially decaying sum of the autocorrelation stats (to
      preserve freshness), but done at the block level not the sample level.
      Also zero `temp_autocorr`.  */
  for (i = 0; i <= lpc_order; i++) {
    lpc->autocorr[i] -= lpc->autocorr_to_remove[i];
    lpc->autocorr_to_remove[i] = 0;
    /** The division below is a 'safer' way of right-shifting by
        AUTOCORR_DECAY_EXPONENT, which would technically give undefined results
        for negative input */
    lpc->autocorr[i] -= (lpc->autocorr[i] / (1 << AUTOCORR_DECAY_EXPONENT));
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

      See the comment where AUTOCORR_DECAY_SQRT is defined for more details.
      Notice that we write this part directly to lpc->autocorr instead of to
      temp_autocorr.  */
  for (i = 0; i < lpc_order; i++) {
    int64_t signal_i = signal[i];
    int j;
    for (j = 0; j <= i; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
    for (j = i + 1; j <= lpc_order; j++)
      lpc->autocorr[j] += (signal[i - j] * signal_i) * AUTOCORR_DECAY_SQRT;
  }

  /** OK, now we handle the samples that aren't close to the boundary.
      currently, i == lpc_order. */
  for (; i < AUTOCORR_BLOCK_SIZE; i++) {
    /* signal_i only needs to be int64_t to handle the case where signal[i - j] and signal_i are
       both -32768, so their product can't be represented as int32_t.  We rely on the
       product below being automatically cast to int64_t. */
    int64_t signal_i = signal[i];
    for (int j = 0; j <= lpc_order; j++)
      temp_autocorr[j] += signal[i - j] * signal_i;
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
   */
  {
    /* Note: signal_edge must only be indexed with negative coeffs.  Imagine a
       symmetric virtual signal (suppose a pointer v, indexed as v[t]) where for
       t < 0, v[t] := signal_edge[t], and for t >= 0, v[t] := signal_edge[-1-t].
       [It's symmetric around t=0.5.]

       We are accumulating the LPC coeffs that 'cross the boundary', i.e.
       involve both t >= 0 and t < 0.  The "i" is the index with t >= 0.
    */
    const int16_t *signal_edge = signal + AUTOCORR_BLOCK_SIZE;

    for (i = 0; i < lpc_order; i++) {
      /* the -1 in the exponent below is the factor of 0.5 mentioned as (b)
         above. */
      int64_t signal_i = ((int64_t)signal_edge[-1-i]) << (AUTOCORR_LEFT_SHIFT-1);
      for (int j = i + 1; j <= lpc_order; j++) {  /* j is the lag */
        lpc->autocorr_to_remove[j] += signal_i * signal_edge[i-j];
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
      ((int64_t)((AUTOCORR_BLOCK_SIZE*AUTOCORR_EXTRA_VARIANCE)<<AUTOCORR_LEFT_SHIFT)) +
      (temp_autocorr[0] << (AUTOCORR_LEFT_SHIFT - AUTOCORR_EXTRA_VARIANCE_EXPONENT));

  /* We will have copied the max_exponent from the previous LpcComputation
     object, and it will usually already have the correct value.  Return
     immediately if so.  */
  int exponent = lpc->max_exponent;
  int64_t autocorr_0 = lpc->autocorr[0],
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
  assert((twice_autocorr_0 >> exponent) == 1 &&
         autocorr_0 >> exponent == 0);
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
  struct LpcComputation lpc_computations[LPC_ROLLING_BUFFER_SIZE];

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
      forward by 4 times the stride).  It points to the byte for t == 0.
      ; the code for the t'th signal value is located at
      compressed_code[t].  */
  int8_t *compressed_code;

  /**  The stride associated with `compressed_code`; normally 1 */
  int compressed_code_stride;

  /** This is only used in debug mode. */
  int64_t num_backtracks;
};



/*******************
  The lilcom_header functions below mostly serve to clarify the format
  of the header; we could just have put the statements inline, which is
  what we hope the compiler will do.

  The format of the 4-byte header is:

    Byte 0:  most-significant 4 bytes contain exponent for frame -1;
             least-significant 4 bytes contain LILCOM_VERSION (currently 1)
    Byte 1:  contains the LPC order (this was user-specifiable)
    Byte 2:  the mantissa of the -1'th sample.
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
 */



/** Set the exponent for frame -1 in the header.  This also sets the
    version number in the lower-order 4 bits.  */
static inline void lilcom_header_set_exponent_m1(int8_t *header, int stride,
                                                 int exponent) {
  assert(exponent >= 0 && exponent <= 11);
  header[0 * stride] = (int8_t)(exponent << 4 | LILCOM_VERSION);
}

/** The exponent for the phantom sample at t = -1 is located in the
    4 highest bits of the first byte.  This function returns that
    value (it does not check it is in the range (0..12)).  */
static inline int lilcom_header_get_exponent_m1(const int8_t *header,
                                                int stride) {
  /** For some reason uint8_t doesn't seem to be defined. */
  return ((int)((unsigned char)(header[0 * stride]))) >> 4;
}

/**  Check that this is plausibly a lilcom header.  The low-order 4 bits of the
     first byte of the header are used for this; the magic number is 7.  */
static inline int lilcom_header_plausible(const int8_t *header,
                                          int stride) {
  /** Currently only one version number is supported. */
  return ((header[0 * stride] & 0xF) == LILCOM_VERSION) &&
      lilcom_header_get_exponent_m1(header, stride) <= 11;
}

/**  Initialize the header.
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
  header[3 * stride] = (int8_t)(-conversion_exponent);
}

static inline int lilcom_header_get_conversion_exponent(
    const int8_t *header, int stride) {
  return -((int)(header[3 * stride]));
}


/** Set the LPC order in the header.  This goes in byte 1.  Currently it only
    uses 4 bits since the max LPC order allowed is currently 15.  */
static inline void lilcom_header_set_lpc_order(int8_t *header,
                                        int stride, int lpc_order) {
  assert(lpc_order >= 0 && lpc_order <= MAX_LPC_ORDER);
  header[1 * stride] = (int8_t)lpc_order;
}

/** Return the LPC order from the header.  Does no range checking!  */
static inline int lilcom_header_get_lpc_order(const int8_t *header, int stride) {
  return (int)(header[1 * stride]);
}

/** Set the -1'th sample's mantissa in the header.  This goes in byte 2. */
static inline void lilcom_header_set_mantissa_m1(int8_t *header,
                                          int stride, int mantissa) {
  assert(mantissa >= -64 && mantissa <= 63);
  header[2 * stride] = (int8_t) mantissa;
}
/** Return the -1'th sample's mantissa from the header, it's in byte 2. */
static inline int lilcom_header_get_mantissa_m1(const int8_t *header,
                                          int stride) {
  return (int)header[2 * stride];
}


/**
   This macro is added mainly for documentation purposes.  It clarifies what the
   possible exponents are for time t given that we know the exponent for time
   t-1.

       @param [in] t   The time for which we want to know the possible
                       exponent range.  t >= -1.
       @param [in] exponent_tm1   The exponent used for time t-1:
                       a value in [0..11], otherwise it is an error.
       @return min_exponent  The minimum exponent that time t
                       will be able to use, given exponent_tm1.
                       This equals exponent_tm1 - ((t+exponent_tm1)&1),
                       where &1 is a way of computing mod 2.
                       It forms a checkerboard pattern.  If you draw
                       the possible trajectories of exponents on
                       a grid you'll see that this choice has nice
                       properties that allows exponents to change
                       quite fast when needed.  The return value is a value
                       in [-1..11], but currently going to exponent
                       -1 is "forbidden", meaning the encoded bit for exponent
                       change is bound to be 1 in this case so the
                       exponent for time t would be zero.  We may later
                       make excursions to negative exponents allowed,
                       which would allow us to save and then pack
                       bits, e.g. for on-disk formats.

                       CAUTION: conceptually, even though we write the
                       formula as a function of t, you should view
                       this as pertaining to t-1, i.e. giving us the
                       lowest value to which (t-1, exponent_tm1)
                       can transition.

   Note: the maximum exponent is just min_exponent + 1, because we use 1 bit for
   the change in exponent; a bit value of 1 means we choose the max, zero means
   we choose the min.
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
                         in the range [0..11].
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
                     must be in the range [0, 11].  This function will
                     never return a value less than this.  min_exponent will
                     normally be the maximum of zero and the previous sample's
                     exponent minus 1, but may be more than that if we are
                     backtracking (because future samples need a larger
                     exponent).

       @param [out] mantissa  This function will write an integer in
                    the range [-64, 63] to here, such that
                    (mantissa << exponent) is a close approximation
                    of `residual` and satisfies the property that
                    `predicted + (mantissa << exponent)` does not
                    exceed the range of int16_t.

       @param [out] next_compressed_value  The next compressed value
                    will be written to here; at exit this will contain
                    `predicted + (mantissa << exponent)`.

    @return  Returns the exponent chosen, a value in the range [min_exponent..11].


   The intention of this function is to return the exponent in the range
   [min_exponent..11] which gives the closest approximation to `residual` that
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

     -65.0 <= m(e) <= 63.5.

   This inequality ensures that there will be no loss of accuracy by choosing e
   instead of e+1 as the exponent.  (If we had a larger exponent, the closest
   points we'd be able to reach would be equivalent to m(e) = -66 or +64; and if
   m(e) satisfies the inequality above we'd have no loss of precision by using e
   rather than e + 1 as the exponent.  (Notice that -65.0 is the midpoint of
   [-66,-64] and 63.5 is the midpoint of [63,64]).  Multiplying by two, we can
   express the above in integer math as:

     (-130 << e) <= residual * 2 <= (127 << e)

   NOTE ON THE RANGE OF THE EXPONENT, explaining why it could be as large as 11.
   The largest-magnitude residual is 65535, which is 13767 - (-13768).  Of the
   numbers representable by (mantissa in [-64,63]) << (integer exponent),
   the closest approximation of 65535 is 65536, for which the lowest exponent
   that can generate that number is 12 (65536 = 32 << 11)
*/
static inline int least_exponent(int32_t residual,
                                 int16_t predicted,
                                 int min_exponent,
                                 int *mantissa,
                                 int16_t *next_decompressed_value) {
  assert (min_exponent >= 0 && min_exponent <= 11); /* TODO: remove this */
  int exponent = min_exponent;
  int32_t residual2 = residual * 2,
      minimum = -130 << exponent,
      maximum = 127 << exponent;
  while (residual2 < minimum || residual2 > maximum) {
    minimum *= 2;
    maximum *= 2;
    exponent++;
  }
  {
    /**
       This code block computes 'mantissa', the integer mantissa which we call
       which should be a value in the range [-64, 63].

       The mantissa will be the result of rounding (residual /
       (float)2^exponent) to the nearest integer (see below for the rounding
       behavior in case of ties, which we randomize); and then, if the result is
       -65 or +64, changing it to -64 or +63 respectively.

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
       (since 512 is much greater than 64), and this ensures well-defined
       rounding behavior.  Note: we will now explicitly make `offset` an int32_t because
       we don't want 512 << 12 to overflow int is int16_t (super unlikely, I know).
    */
    int32_t offset = (((int32_t)(512+1)) << exponent) - (predicted&1);

    int local_mantissa = (int)((int8_t)((residual2 + offset) >> (exponent + 1)));

    assert(local_mantissa >= -65 && local_mantissa <= 64);

    /* We can't actually represent -65 in 7 bits, but we choose to retain this
       exponent, in this case, because -65 is as close to -64 (which is
       representable) as it is to -66 (which is the next closest thing
       we'd get if we used a one-larger exponent).  */
    if (local_mantissa == -65)
      local_mantissa = -64;
    /*  The following could happen if we really wanted the mantissa to be
        exactly 63.5, and `predicted` was even so we rounded up.  It would only
        happen in case of ties, so we lose no accuracy by doing this. */
    if (local_mantissa == 64)
      local_mantissa = 63;

    int32_t next_signal_value =
        ((int32_t)predicted) + (((int32_t)local_mantissa) << exponent);

    {
      /* Just a check.  I will remove this block after it's debugged.
         Checking that the error is in the expected range.  */
      int16_t error = (int16_t)(next_signal_value - (predicted + residual));
      if (error < 0) error = -error;
      if (local_mantissa > -64) {
        assert(error <= (1 << exponent) >> 1);
      } else {
        /** If local_mantissa is -64, the desired mantissa might have been -65,
            which is twice as large, so we need to make the assert different. */
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
  int32_t autocorr[MAX_LPC_ORDER];

  { /** This block just sets up the `autocorr` array with appropriately
        shifted copies of lpc->autocorr.  */
    int max_exponent = lpc->max_exponent;
    assert(max_exponent >= AUTOCORR_LEFT_SHIFT &&
           (lpc->autocorr[0]) >> (max_exponent - 1) == 1);
    if (max_exponent > LPC_EST_LEFT_SHIFT) {
      /* shift right (the normal case).  We do it by division, because
         technically the result of shifting a negative number right
         is undefined (even though it normally does what we want,
         which is to duplicate the left-most -1) */
      int right_shift = max_exponent - LPC_EST_LEFT_SHIFT;
      for (int i = 0; i <= lpc_order; i++)
        autocorr[i] = (int32_t)(lpc->autocorr[i] / ((int64_t)1 << right_shift));
    } else {
      int left_shift = LPC_EST_LEFT_SHIFT - max_exponent;
      for (int i = 0; i <= lpc_order; i++)
        autocorr[i] = (int32_t)(lpc->autocorr[i] << left_shift);
    }
    assert((autocorr[0] >> (LPC_EST_LEFT_SHIFT - 1)) == 1);
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
  /** int32_t temp[MAX_LPC_ORDER];
  ***TEMP*** Making it int64 while testing. */
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
    int64_t ki = ((int64_t)autocorr[i + 1]) << LPC_EST_LEFT_SHIFT;

    for (j = 0; j < i; j++) {
      /** max magnitude of the terms added below is 2^(LPC_EST_LEFT_SHIFT*2 + 8) = 2^54, i.e.
          the abs value of the added term is less than 2^54.
          ki still represents a floating-point number times 2 to the
          power 2*LPC_EST_LEFT_SHIFT.
        The original floating-point code looked the same as the next line
        (not: ki has double the left-shift of the terms on the right).
      */
      ki -= lpc->lpc_coeffs[j] * (int64_t)autocorr[i - j];
    }
    /** RE the current magnitude of ki:
        ki is a summation of terms, so add LPC_ORDER_BITS=4 to the magnitude of
        2^54 computed above, it's now bounded by 2^59 (this would bound its value
        at any point in the summation above).
        original code: "ki = ki / E;".   */
    ki = ki / E;
    /** At this point, ki is mathematically in the range [-1,1], since it's a
        reflection coefficient; and it is stored times 2 to the power LPC_EST_LEFT_SHIFT, so its
        magnitude as an integer is <= 2^LPC_EST_LEFT_SHIFT.  Check that it's less
        than 2^LPC_EST_LEFT_SHIFT plus a margin to account for rounding errors.  */
    assert(lilcom_abs(ki) < ((1<<LPC_EST_LEFT_SHIFT) + (1<<(LPC_EST_LEFT_SHIFT - 8))));

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
        because the result (still an energy E) is nonnegative.
    */
    E = (int32_t)((E * c) >> LPC_EST_LEFT_SHIFT);

    /** compute the new LP coefficients
        Original code did: pTmp[i] = -ki;
        Note: abs(temp[i]) <= 2^LPC_EST_LEFT_SHIFT, since ki is in the range [-1,1]
        when viewed as a real number. */
    temp[i] = (int32_t)ki;
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
      lpc->lpc_coeffs[j] = (int32_t)temp[j];
    }
  }
  /** E > 0 because we added fake extra variance via
     AUTOCORR_EXTRA_VARIANCE_EXPONENT and AUTOCORR_EXTRA_VARIANCE, so according
     to these stats the sample should never be fully predictable.  We'd like to
     assert that E <= autocorr[0] because even if the data is totally
     uncorrelated, we should never be increasing the predicted error vs. having
     no LPC at all.  But I account for the possibility that in pathological
     cases, rounding errors might make this untrue.  */
  assert(E > 0 && E <= (autocorr[0] + (autocorr[0] >> 10)));


  /**
     We want to shift the LPC coefficients right by LPC_EST_LEFT_SHIFT -
     LPC_APPLY_LEFT_SHIFT, because we store and apply them with lower precision
     than we estimate them.  We divide rather than shift because technically
     the result of right-shifting a negative number is implementation-defined.
  */
  for (int i = 0; i < lpc_order; i++) {
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
static inline int16_t lilcom_compute_predicted_value(
    struct CompressionState *state,
    int64_t t) {
  uint32_t lpc_index =
      ((uint32_t)(t >> LOG_AUTOCORR_BLOCK_SIZE)) % LPC_ROLLING_BUFFER_SIZE;
  struct LpcComputation *lpc = &(state->lpc_computations[lpc_index]);

  int lpc_order = state->lpc_order;

  /** Get the pointer to the t'th signal in the circular buffer
      'state->decompressed_signal'.  The buffer has an extra MAX_LPC_ORDER
      samples at the start, to provide needed context when we roll around,
      to keep this code simple. */
  int16_t *decompressed_signal_t =
      &(state->decompressed_signal[MAX_LPC_ORDER + (t&(SIGNAL_BUFFER_SIZE-1))]);


  /**
     The initial value of the sum can be thought of a being
     (in this fixed-point representation),
       - 0.5 (in our fixed point representation), so that when we
         eventually round down it has the effect of rounding-to-the-closest,
         plus ..
       - A big number (2^16) that is designed to keep the sum positive, so that
         when, later, we right shift, the behavior is well defined.  Later we'll
         remove this by having it overflow.  It's important that this
         number be >= 16 because that's the smallest power of 2 that, when
         cast to int16_t, disappears.  Also this number plus
         LILCOM_APPLY_LEFT_SHIFT must be less than 31, to avoid overflowing
         int32_t.
  */
  int32_t sum = (1 << (LPC_APPLY_LEFT_SHIFT - 1)) +
      (1 << (LPC_APPLY_LEFT_SHIFT + 16)),
      sum2 = 0;  /** Break into two sums for pipelining reasons (an
                  * optimization)  */


  /** The following is an optimization of a simple loop for i = 0 .. lpc_order -
      1.  It may access a one-past-the-end element if lpc_order is odd, but
      this is OK.  We made sure in lilcom_init_lpc() that unused
      elements of lpc_coeffs are zeroed. */
  int i;
  const int32_t *lpc_coeffs = &(lpc->lpc_coeffs[0]);
  for (i = 0; i < lpc_order; i += 2) {
    sum += lpc_coeffs[i] * decompressed_signal_t[-1-i];
    sum2 += lpc_coeffs[i+1] * decompressed_signal_t[-2-i];
  }


  /** The lpc_coeffs were stored times 2^LPC_APPLY_LEFT_SHIFT.  Divide by this
      to get the integer prediction.  The following corresponds to division by
      2^LPC_APPLY_LEFT_SHIFT as long as 'sum' is positive, which will be as long
      as the predicted value is not "wildly out of range" (defined below).
      Importantly, the expression below gives platform-independent results
      thanks to casting to unsigned int before doing the right shift. */
  int32_t predicted = (int32_t)(((uint32_t)(sum + sum2)) >> LPC_APPLY_LEFT_SHIFT);

  /** Caution: at this point, `predicted` contains the predicted value
      plus 2^16 = 65536.  Recall that we initialized the sum
      with a term of 1 << (LPC_APPLY_LEFT_SHIFT + 16). */

  /** Now we deal with the case where the predicted value was outside
      the range [-32768,32764].  Right now `predicted` has 65536 added
      to it so we need to account for that when making the comparisons.
      The offset 65536 naturally disappears when casting to int16_t.

      Define a "wildly out of range" predicted value as one that was
      so far outside the range [-32768,32767] that it generated overflow
      in the int32_t expressions above.  In these cases the sign of
      the returned value below may not be "correct", if by "correct"
      we mean the sign it would have had without overflow.

      We just define the expected behavior, in those highly pathological
      cases, to be "whatever the following code does".
  */

  /** The following if-statement is a one-shot way of testing "is the variable
      `predicted` outside the range [65536-32768, 65536+32767]?"  If it is
      outside that range, that implies that the `real` predicted value (given by
      predicted - 65536) was outside the range [-32768,32767], which would mean
      we need to do some truncation to avoid predicting a value not
      representable as int16_t (allowing the prediction to wrap around would
      degrade fidelity).
   */
  if (((predicted - 32768) & ~(int32_t)65535) != 0) {
    if (predicted > 32767 + 65536)
      predicted = 65536 + 32767;
    else if (predicted < -32768 + 65536)
      predicted = 65536 - 32768;
  }
  assert(predicted >= -32768 + 65536 && predicted <= 32767 + 65536);
  return (int16_t)predicted;
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
    int64_t t, struct CompressionState *state) {
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
  /** The expression for lpc_index is well defined because t is nonnegative.
      Note: it is essential, if we want the expression for prev_lpc_index to
      work for lpc_index == 0, that these be unsigned ints, or % would
      not behave right. */
  uint32_t lpc_index = ((uint32_t)(t >> LOG_AUTOCORR_BLOCK_SIZE)) % LPC_ROLLING_BUFFER_SIZE,
      prev_lpc_index = (lpc_index + LPC_ROLLING_BUFFER_SIZE - 1) % LPC_ROLLING_BUFFER_SIZE;


  /** Interpret `block_index & 1` as `block_index % 2` (valid for nonnegative
      block_index, which it is). */
  struct LpcComputation *this_lpc = &(state->lpc_computations[lpc_index]);
  /**  prev_lpc is the 'other' LPC object in the circular buffer of size 2. */
  struct LpcComputation *prev_lpc = &(state->lpc_computations[prev_lpc_index]);

  /** Copy the previous autocorrelation coefficients and max_exponent.  We'll
      either re-estimate or copy the LPC coefficients below. */
  int lpc_order = state->lpc_order;
  for (int i = 0; i <= lpc_order; i++) {
    this_lpc->autocorr[i] = prev_lpc->autocorr[i];
    this_lpc->autocorr_to_remove[i] = prev_lpc->autocorr_to_remove[i];
  }
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
      @param [in] min_codable_exponent  The exponent that we would end up with
                        if we chose bit zero to encode the exponent for this
                        frame.  If the previous frame's exponent is p,
                        this equals LILCOM_COMPUTE_MIN_EXPONENT(p, t),
                        which is a value in [-1, 11].
                        The other choice for the exponent is this plus
                        one.
      @param [in] min_allowed_exponent  The minimum exponent that the
                        caller will allow for this time.  If min_codable_exponent
                        is -1, this must be 0.  Otherwise it may equal
                        either min_codable_exponent or min_codable_exponent + 1.
                        The reason it might equal min_codable_exponent + 1
                        in cases where min_codable_exponent >= 0 is
                        backtracking: that is, we found that we weren't able
                        to code a future frame if the exponent for this
                        frame has the value `min_codable_exponent`.
      @param [in,out] state  Contains the computation state and pointers to the
                       input and output data.

   On success (i.e. if it was able to do the compression) it returns the
   exponent used, which is a number >= 0.

   On failure, which can happen if the exponent required to compress this value
   was greater than min_codable_exponent + 1, it returns the negative of the
   exponent that would have required to compress this sample.  this will cause
   us to enter backtracking code to inrease the exponent used on the previous
   sample.
*/
static inline int lilcom_compress_for_time_internal(
    int64_t t,
    int min_codable_exponent,
    int min_allowed_exponent,
    struct CompressionState *state) {
  assert(t > 0 && min_codable_exponent >= -1 &&
         min_codable_exponent <= 11 &&
         (min_allowed_exponent == min_codable_exponent ||
          min_allowed_exponent == min_codable_exponent + 1) &&
         min_allowed_exponent >= 0);

  if ((t & (AUTOCORR_BLOCK_SIZE - 1)) == 0 && state->lpc_order != 0) {
    if ((t & (SIGNAL_BUFFER_SIZE - 1)) == 0) {
      /**  If this is the start of the uncompressed_signal buffer we need to
           make sure that the required left context is copied appropriately. */
      lilcom_copy_to_buffer_start(state);
    }
    /** Update the autocorrelation coefficients and possibly the LPC
        coefficients. */
    lilcom_update_autocorrelation_and_lpc(t, state);
  }

  int16_t predicted_value = lilcom_compute_predicted_value(state, t),
      observed_value = state->input_signal[t * state->input_signal_stride];

  /** cast to int32 when computing the residual because a difference of int16's may
      not fit in int16. */
  int32_t residual = ((int32_t)observed_value) - ((int32_t)predicted_value);

  int mantissa,
      exponent = least_exponent(
          residual, predicted_value,
          min_allowed_exponent, &mantissa,
          &(state->decompressed_signal[MAX_LPC_ORDER+(t&(SIGNAL_BUFFER_SIZE-1))]));

  assert(exponent <= 11);

  int exponent_delta = exponent - min_codable_exponent;
  assert(exponent_delta >= 0);

  if (exponent_delta <= 1) {
    /** Success; we can represent this.  This is (hopefully) the normal code
        path. */
    assert(mantissa >= -64 && mantissa < 64);
    state->compressed_code[t*state->compressed_code_stride] =
        (int8_t)((mantissa << 1) + exponent_delta);
    state->exponents[t & (EXPONENT_BUFFER_SIZE - 1)] = exponent;
    return exponent;
  } else {
    /** Failure.  The calling code will backtrack, increase the previous
        to a value which will allow `exponent` to be at least
        the negative of the value returned, and try again.  */
    return -exponent;
  }
}


/**
  This function is a special case of compressing a single sample, for t == 0.
  Time zero is a little different because of initialization effects (the header
  contains an exponent and a mantissa for t == -1, which gives us a good
  starting point).

    @param [in] min_exponent  A number in the range [0, 11]; the caller
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
    struct CompressionState *state) {
  int c_stride = state->compressed_code_stride;
  int header_stride = state->compressed_code_stride;
  int8_t *header = state->compressed_code -
      (LILCOM_HEADER_BYTES * header_stride);

  int16_t first_signal_value = state->input_signal[0];
  assert(min_exponent >= 0 && min_exponent <= 12);
  /** m1 refers to -1 (sample-index minus one). */
  int sample_m1_min_exponent =
      LILCOM_COMPUTE_MIN_PRECEDING_EXPONENT(0, min_exponent);
  /** LILCOM_COMPUTE_MIN_PRECEDING_EXPONENT doesn't check for negatives. */
  if (sample_m1_min_exponent < 0)
    sample_m1_min_exponent = 0;
  int16_t signal_m1,  /* compressed signal for "phantom sample" -1. */
      predicted_m1 = 0,
      residual_m1 = first_signal_value;
  int mantissa_m1,
      exponent_m1 = least_exponent(residual_m1,
                                   predicted_m1,
                                   sample_m1_min_exponent,
                                   &mantissa_m1, &signal_m1);
  {  /** Set the exponent and mantissa for t == -1 in the header. */
    assert(exponent_m1 >= 0);
    lilcom_header_set_exponent_m1(header, header_stride, exponent_m1);
    assert(lilcom_header_get_exponent_m1(header, header_stride)
           == exponent_m1);
    lilcom_header_set_mantissa_m1(header, header_stride, mantissa_m1);
    assert(lilcom_header_get_mantissa_m1(header, header_stride)
           == mantissa_m1);
  }
  /** store the initial exponent in the buffer, at sample -1.  This probably won't be
      accessed, actually.  [TODO: remove this?] */
  state->exponents[EXPONENT_BUFFER_SIZE - 1] = exponent_m1;


  int min_codable_exponent0 = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(0, exponent_m1);
  assert(min_codable_exponent0 + 1 >= min_exponent);
  if (min_exponent < min_codable_exponent0)
    min_exponent = min_codable_exponent0;
  /** We already know min_exponent >= 0. */

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
  int exponent_bit = exponent_0 - min_codable_exponent0;
  /** The residual cannot be greater in magnitude than first_value, since we
      already encoded first_signal_value and we are now just dealing with the
      remaining part of it, so whatever exponent we used for sample -1 would
      be sufficiently large for sample 0; that's how we can guarantee
      delta_exponent <= 2.  */
  assert(exponent_bit >= 0 && exponent_bit <= 1 &&
         mantissa_0 >= -64 && mantissa_0 <= 63);

  state->compressed_code[0 * c_stride] =
      (int8_t)((mantissa_0 << 1) + exponent_bit);

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
    struct CompressionState *state) {
  assert(++state->num_backtracks);

  /** We can assume min_exponent > 0 because otherwise we wouldn't have
      reached this code. */
  assert(t >= 0 && min_exponent > 0);
  if (t > 0) {
    int prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)],
        prev_exponent_floor = LILCOM_COMPUTE_MIN_PRECEDING_EXPONENT(t, min_exponent);

    if (prev_exponent < prev_exponent_floor) {
      /** We need to revisit the exponent for sample t-1. */
      lilcom_compress_for_time_backtracking(t - 1, prev_exponent_floor, state);
      prev_exponent = state->exponents[(t-1)&(EXPONENT_BUFFER_SIZE-1)];
      assert(prev_exponent >= prev_exponent_floor);
    }
    int min_codable_exponent = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, prev_exponent);
    if (min_exponent < min_codable_exponent)
      min_exponent = min_codable_exponent;
    assert(min_exponent <= min_codable_exponent + 1);
    int exponent = lilcom_compress_for_time_internal(
        t, min_codable_exponent, min_exponent, state);
    if (exponent >= 0) {
      return;  /** Normal code path: success.  */
    } else {
      /* Now make `exponent` positive. It was negated as a signal that there was
         a failure: specifically, that exponent required to encode this sample
         was greater than min_codable_exponent + 1.  [This path is super
         unlikely, as we've already backtracked, but it theoretically could
         happen, as if previous samples are coded differently the LPC prediction
         would change.].  We can deal with this case via recursion.  */
      exponent = -exponent;
      /* fprintf(stderr, "Hit unusual code path\n"); */
      assert(exponent > min_codable_exponent + 1 && exponent > min_exponent);
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
static inline void lilcom_compress_for_time(
    int64_t t,
    struct CompressionState *state) {
  assert(t > 0);
  int prev_exponent =
      state->exponents[(t - 1) & (EXPONENT_BUFFER_SIZE - 1)],
      min_codable_exponent = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, prev_exponent),
      min_allowed_exponent = (min_codable_exponent < 0 ? 0 :
                              min_codable_exponent);
  int exponent = lilcom_compress_for_time_internal(
      t, min_codable_exponent, min_allowed_exponent, state);
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
    int lpc_order, int conversion_exponent,
    struct CompressionState *state) {
  state->lpc_order = lpc_order;
  for (int i = 0; i < LPC_ROLLING_BUFFER_SIZE; i++)
    lilcom_init_lpc(&(state->lpc_computations[i]));

  state->input_signal = input;
  state->input_signal_stride = input_stride;
  state->compressed_code =
      output + (LILCOM_HEADER_BYTES * output_stride);
  state->compressed_code_stride = output_stride;

  /** Put it in an assert so it doesn't happen in non-debug mode. */
  assert(((state->num_backtracks = 0) == 0));


  for (int i = 0; i < MAX_LPC_ORDER; i++)
    state->decompressed_signal[i] = 0;


  lilcom_header_set_conversion_exponent(output, output_stride,
                                        conversion_exponent);
  lilcom_header_set_lpc_order(output, output_stride,
                              state->lpc_order);

  /** The remaining parts of the header will be initialized in
      lilcom_compress_for_time_zero`. */
  int min_exponent = 0;
  lilcom_compress_for_time_zero(min_exponent, state);
}


/*  See documentation in lilcom.h  */
int lilcom_compress(int64_t num_samples,
                    const int16_t *input, int input_stride,
                    int8_t *output, int output_stride,
                    int lpc_order, int conversion_exponent) {
  if (num_samples <= 0 || input_stride == 0 || output_stride == 0 ||
      lpc_order < 0 || lpc_order > MAX_LPC_ORDER ||
      conversion_exponent < -127 || conversion_exponent > 128) {
    return 1;  /* error */
  }
  struct CompressionState state;
  lilcom_init_compression(num_samples, input, input_stride,
                          output, output_stride, lpc_order,
                          conversion_exponent, &state);

  for (int64_t t = 1; t < num_samples; t++)
    lilcom_compress_for_time(t, &state);

  assert(fprintf(stderr, "Backtracked %f%% of the time\n",
                 ((state.num_backtracks * 100.0) / num_samples)) || 1);

  return 0;
}


/**
   This function does the core part of the decompression of one sample
   (excluding the part about updating the autocorrelation statistics and
   updating the LPC coefficients; that is done externally.

      @param [in] t    The current time index, cast to int; we actually
                     only need its lowest-order bit.
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
      @return  Returns 0 on success, 1 on failure.  Failure would normally
                     mean data corruption or possily a code error.
                     This function will fail if the input exponent is not
                     in the range [0,12] or the signal left the bounds
                     of int16_t.

 */
static inline int lilcom_decompress_one_sample(
    int64_t t,
    int8_t input_code,
    int lpc_order,
    const int32_t *lpc_coeffs,
    int16_t *output_sample,
    int *exponent) {

  int16_t predicted_sample;

  { /** This block computes the predicted value.  For explanation please
        look for similar statements in `lilcom_compute_predicted_value()`,
        which is well documented. */
    int32_t sum = (1 << (LPC_APPLY_LEFT_SHIFT - 1)) +
        (1 << (LPC_APPLY_LEFT_SHIFT + 16));
    int i;
    for (i = 0; i < lpc_order; i++)
      sum += lpc_coeffs[i] * output_sample[-1-i];
    int32_t predicted = (int32_t)(((uint32_t)sum) >> LPC_APPLY_LEFT_SHIFT);
    if (((predicted - 32768) & ~65535) != 0) {
      if (predicted > 32767 + 65536)
        predicted = 65536 + 32767;
      else if (predicted < -32768 + 65536)
        predicted = 65536 - 32768;
    }
    assert(predicted >= 65536 - 32768 && predicted <= 65536 + 32767);
    predicted_sample = (int16_t)predicted;
  }

  if (((unsigned int)*exponent) > 11) {
    /** If `exponent` is not in the range [0,11], something is wrong.
        We return 1 on failure.  */
    printf("Bad exponent!\n");  // TEMP
    return 1;
  }

  /**
     Below, it would have been nice to be able to compute 'mantissa' as just
     input_code >> 1, but we can't rely on this being implemented as arithmetic
     bit-shift, the C standard does not guarantee that.  We also can't just
     divide by 4 because that division would round up for negative input,
     depending on the lower-order two bits, so we need to `and` it with
     ~((int_8)3) to get rid of those low-order two bits.  Hopefully the
     compiler can optimize this.
   */
  int exponent_bit = (input_code & 1),
      min_codable_exponent = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(t, *exponent),
      mantissa = (input_code & ~((int8_t)1)) / 2;
  *exponent = min_codable_exponent + exponent_bit;

  assert(*exponent >= 0);

  int32_t new_sample = (int32_t)predicted_sample  +  (mantissa << *exponent);

  if (((new_sample + 32768) & ~(int32_t)65535) != 0) {
    /** If `new_sample` is outside the range [-32768 .. 32767], it
        is an error; we should not be generating such samples. */
    return 1;
  }
  output_sample[0] = new_sample;
  return 0;  /** Success */
}



/*
  This function attempts to obtain the first sample (the one for t = 0).
    @param [in] header   Pointer to the beginning of the header.
    @paran [in] input_stride  Stride between elements of `header`
                        (naturally, past the header there are real samples,
                        with the same stride).
    @param [out] output   On success, the decoded sample will be written
                        to here.
    @param [out] exponent  On success, the exponent used to encode
                        time zero will be written to here.
    @return  Returns 0 on success, 1 on failure (e.g. invalid
             input).
 */
static inline int lilcom_decompress_time_zero(
    const int8_t *header, int input_stride,
    int16_t *output, int *exponent) {
  int exponent_m1 = lilcom_header_get_exponent_m1(header, input_stride),
      mantissa_m1 = lilcom_header_get_mantissa_m1(header, input_stride);
  /** Failure in the following assert would be a code error, not an error in the
      input.  We already checked the exponent range in
     `lilcom_header_plausible`.  */
  assert(mantissa_m1 >= -32 && mantissa_m1 <= 31 &&
         exponent_m1 >= 0 && exponent_m1 <= 12);
  int32_t sample_m1 = mantissa_m1 << exponent_m1;
  int8_t code_0 = header[input_stride * LILCOM_HEADER_BYTES];
  int exponent_bit = (code_0 & 1),
      mantissa = (code_0 & ~(int8_t)1) / 2;
  *exponent = LILCOM_COMPUTE_MIN_CODABLE_EXPONENT(0, exponent_m1) + exponent_bit;
  int32_t sample_0 = sample_m1 + (mantissa << *exponent);
  if (((sample_0 + 32768) & ~(int32_t)65535) != 0) {
    return 1;  /**  Out-of-range error */
  }
  *output = (int16_t)sample_0;
  return 0;

}



int lilcom_decompress(int64_t num_samples,
                      const int8_t *input, int input_stride,
                      int16_t *output, int output_stride,
                      int *conversion_exponent){
  if (num_samples <= 0 || input_stride == 0 || output_stride == 0 ||
      !lilcom_header_plausible(input, input_stride))
    return 1;  /** Error */

  int lpc_order = lilcom_header_get_lpc_order(input, input_stride);
  *conversion_exponent = lilcom_header_get_conversion_exponent(
      input, input_stride);

  int exponent;
  if (lilcom_decompress_time_zero(input, input_stride,
                                  &(output[0]),
                                  &exponent))
    return 1;  /** Error */

  /** This is called input 4 because (if stride == 1)
      it's 4 bytes past input. */
  const int8_t *input4  = input + (input_stride * LILCOM_HEADER_BYTES);

  struct LpcComputation lpc;
  lilcom_init_lpc(&lpc);

  /** The first LPC_MAX_ORDER samples are for left-context; view the array as
      starting from the element with index LPC_MAX_ORDER.
      In the case where output_stride is 1, we'll only be using the
      first LPC_MAX_ORDER + AUTOCORR_BLOCK_SIZE elements of this
      array.
  */
  int16_t output_buffer[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE];
  int i;
  for (i = 0; i  < MAX_LPC_ORDER; i++)
    output_buffer[i] = 0;
  output_buffer[MAX_LPC_ORDER] = output[0];
  int t;
  for (t = 1; t < AUTOCORR_BLOCK_SIZE && t < num_samples; t++) {
    if (lilcom_decompress_one_sample(t, input4[t * input_stride],
                                     lpc_order, lpc.lpc_coeffs,
                                     &(output_buffer[MAX_LPC_ORDER + t]),
                                     &exponent))
      return 1;  /** Error */

    output[t * output_stride] = output_buffer[MAX_LPC_ORDER + t];
  }
  if (t >= num_samples)
    return 0;  /** Success */

  if (output_stride == 1) {
    /** Update the autocorrelation with stats from the 1st block (it's
        a special case, as we need to use `output_buffer` so the
        left-context will work right. */
    lilcom_update_autocorrelation(&lpc, lpc_order,
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
        lilcom_update_autocorrelation(&lpc, lpc_order, output + t - AUTOCORR_BLOCK_SIZE);
        if ((t & (LPC_COMPUTE_INTERVAL - 1)) == 0 || t < LPC_COMPUTE_INTERVAL)
          lilcom_compute_lpc(lpc_order, &lpc);
      }
      int64_t local_max_t = (t + AUTOCORR_BLOCK_SIZE < num_samples ?
                             t + AUTOCORR_BLOCK_SIZE : num_samples);
      for (; t < local_max_t; t++)
        if (lilcom_decompress_one_sample(t,
                                         input4[t * input_stride],
                                         lpc_order, lpc.lpc_coeffs,
                                         output + t,
                                         &exponent))
          return 1;  /** Error */
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

      if ((t & (SIGNAL_BUFFER_SIZE-1)) == 0) {
        /** A multiple of SIGNAL_BUFFER_SIZE.  We need to copy the context to
            before the beginning of the buffer.  */
        for (int i = 1; i <= lpc_order; i++)
          output_buffer[MAX_LPC_ORDER - i] =
              output_buffer[MAX_LPC_ORDER + SIGNAL_BUFFER_SIZE - i];
      }

      /** buffer_start_t is the t value at which we'd want to give
          `lilcom_update_autocorrelation` a pointer to the output.
          We'll later have to take this modulo SIGNAL_BUFFER_SIZE
          and then add MAX_LPC_ORDER to find the right position in
          `output_buffer`. */
      int64_t buffer_start_t = t - AUTOCORR_BLOCK_SIZE;
      lilcom_update_autocorrelation(&lpc, lpc_order,
                                    output_buffer + MAX_LPC_ORDER + buffer_start_t % SIGNAL_BUFFER_SIZE);
      /** If t is a multiple of LPC_COMPUTE_INTERVAL or < LPC_COMPUTE_INTERVAL.. */
      if ((t & (LPC_COMPUTE_INTERVAL - 1)) == 0 || t < LPC_COMPUTE_INTERVAL)
        lilcom_compute_lpc(lpc_order, &lpc);

      int64_t local_max_t = (t + AUTOCORR_BLOCK_SIZE < num_samples ?
                             t + AUTOCORR_BLOCK_SIZE : num_samples);
      for (; t < local_max_t; t++) {
        if (lilcom_decompress_one_sample(t, input4[t * input_stride],
                                         lpc_order, lpc.lpc_coeffs,
                                         output_buffer + MAX_LPC_ORDER + (t&(SIGNAL_BUFFER_SIZE-1)),
                                         &exponent))
          return 1;  /** Error */

        output[t * output_stride] =
            output_buffer[MAX_LPC_ORDER + (t&(SIGNAL_BUFFER_SIZE-1))];
      }
    }
    return 0;  /** Success */
  }
}


/**
   Returns the maximum absolute value of any element of the array 'f'.  If there
   is any NaN in the array, return NaN if possible Note: at higher compiler
   optimization levels we cannot always rely on this behavior so officially
   the behavior with inputs with NaN's is undefined.

      @param [in] num_samples      The number of elements in the array.
      @param [in] sinput                The input array
      @param [in] stride           The stride between array elements;
                                 usually 1.
 */
float max_abs_float_value(int64_t num_samples, const float *input, int stride) {
  int64_t t;

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
  for (t = 0; t < num_samples; t ++) {
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

  while (max_abs_value >= (65535.0/65536) * pow(2.0, i) && i < 1000)
    i++;
  while (max_abs_value < (65535.0/65536) * pow(2.0, i - 1) && i > -1000)
    i--;

  if (i == 1000 || i == -1000) {
    /** This point should never be reached. */
    fprintf(stderr, "Warning: something went wrong while finding the exponent\n");
    return -256;
  }

  if (i < -127)
    i = -127;
  if (i > 128)
    i = 128;
  return i;
}

int lilcom_compress_float(int64_t num_samples,
                          const float *input, int input_stride,
                          int8_t *output, int output_stride,
                          int lpc_order, int16_t *temp_space) {
  assert(num_samples > 0);
  if (temp_space == NULL) {
    /* Allocate temp array and recurse. */
    int16_t *temp_array = malloc(sizeof(int16_t) * num_samples);
    if (temp_array == NULL)
      return 2;  /* Special error code for this situation. */
    int ans = lilcom_compress_float(num_samples, input, input_stride,
                                    output, output_stride,
                                    lpc_order, temp_array);
    free(temp_array);
    return ans;
  }

  float max_abs_value = max_abs_float_value(num_samples, input, input_stride);
  if (max_abs_value - max_abs_value != 0)
    return 1;  /* Inf's or Nan's detected */
  int conversion_exponent = compute_conversion_exponent(max_abs_value);

  /* -256 is the error code when compute_conversion_exponent detects infinities
      or NaN's. */
  if (conversion_exponent == -256)
    return 2;  /* This is the error code meaning we detected inf or NaN. */

  assert(conversion_exponent >= -127 && conversion_exponent <= 128);

  int adjusted_exponent = 15 - conversion_exponent;

  if (adjusted_exponent > 127) {
    /** A special case.  If adjusted_exponent > 127, then 2**adjusted_exponent
        won't be representable as single-precision float: we need to do the
        multiplication in double.  [Note: just messing with the floating-point
        mantissa manually would be easier, but it's probably harder to make
        hardware-independent.]. */
    double scale = pow(2.0, adjusted_exponent);

    for (int64_t k = 0; k < num_samples; k ++) {
      double f = input[k * input_stride];
      int32_t i = (int32_t)(f * scale);
      assert(i == (int16_t)i);
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
    for (int64_t k = 0; k < num_samples; k ++) {
      float f = input[k * input_stride];
      int32_t i = (int32_t)(f * scale);
      assert(i >= -32768 && i <= 32768);
      if (i >= 32768)
        i = 32767;
      temp_space[k] = i;
    }
  } else {
    /** The normal case; we should be here in 99.9% of cases. */
    float scale = pow(2.0, adjusted_exponent);
    for (int64_t k = 0; k < num_samples; k ++) {
      float f = input[k * input_stride];
      int32_t i = (int32_t)(f * scale);
      assert(i == (int16_t)i);
      temp_space[k] = i;
    }
  }

  int ret = lilcom_compress(num_samples, temp_space, 1,
                            output, output_stride,
                            lpc_order, conversion_exponent);
  return ret;  /* 0 for success, 1 for failure, e.g. if lpc_order out of
                  range. */
}


int lilcom_decompress_float(int64_t num_samples,
                            const int8_t *input, int input_stride,
                            float *output, int output_stride) {
  /* We re-use the output as the temporary int16_t array,
   */
  int16_t *temp_array = (int16_t*)output;
  int temp_array_stride;
  if (output_stride == 1) {
    temp_array_stride = 1;
  } else {
    temp_array_stride = output_stride * (sizeof(float) / sizeof(int16_t));
  }
  int conversion_exponent;
  int ans = lilcom_decompress(num_samples, input, input_stride,
                              temp_array, temp_array_stride,
                              &conversion_exponent);
  if (ans != 0)
    return ans;  /* the only other possible value is 1, actually. */

  assert(conversion_exponent >= -127 && conversion_exponent <= 128);  /* TODO: Remove this. */

  int adjusted_exponent = conversion_exponent - 15;

  if (adjusted_exponent < -126 || conversion_exponent >= 128) {
    /** Either adjusted_exponent itself is outside the range representable in
        single precision, or there is danger of overflowing single-precision
        range after multiplying by the integer values, so we do the conversion
        in double.  We also check for things that exceed the range representable
        as float, although this is only necessary for the case when
        conversion_exponent == 128. */
    double scale = pow(2.0, adjusted_exponent);

    for (int64_t k = num_samples - 1; k >= 0; k--) {
      int16_t i = temp_array[k * temp_array_stride];
      double d = i * scale;
      if (lilcom_abs(d) > FLT_MAX) {
        if (d > 0) d = FLT_MAX;
        else d = -FLT_MAX;
      }
      output[k * output_stride] = (float)d;
    }
  } else {
    float scale = pow(2.0, adjusted_exponent);
    for (int64_t k = num_samples - 1; k >= 0; k--) {
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
  assert(EXPONENT_BUFFER_SIZE > 24);
  assert((LPC_ROLLING_BUFFER_SIZE - 1) * AUTOCORR_BLOCK_SIZE > 24);
  assert(MAX_LPC_ORDER >> LPC_ORDER_BITS == 0);
  assert(MAX_LPC_ORDER % 2 == 0);
  assert(LPC_EST_LEFT_SHIFT + (AUTOCORR_EXTRA_VARIANCE_EXPONENT+1)/2 <= 31);
  assert(LPC_APPLY_LEFT_SHIFT + 16 < 31);
  assert((30 + AUTOCORR_LEFT_SHIFT + log(AUTOCORR_BLOCK_SIZE) + AUTOCORR_DECAY_EXPONENT)
         <= 61);
  assert((AUTOCORR_BLOCK_SIZE & (AUTOCORR_BLOCK_SIZE-1)) == 0);  /** Power of 2. */
  assert(AUTOCORR_BLOCK_SIZE > MAX_LPC_ORDER);
  assert(LPC_COMPUTE_INTERVAL % AUTOCORR_BLOCK_SIZE == 0);
  assert(AUTOCORR_BLOCK_SIZE >> LOG_AUTOCORR_BLOCK_SIZE == 1);
  {
    int64_t n1 = (int64_t)AUTOCORR_DECAY_SQRT;
    n1 *= n1;
    n1 >>= AUTOCORR_LEFT_SHIFT;
    int64_t n2 = (1 << AUTOCORR_LEFT_SHIFT) - (1 << (AUTOCORR_LEFT_SHIFT - AUTOCORR_DECAY_EXPONENT));
    assert(lilcom_abs(n1 - n2) <= 3);
  }
  /** assumed in some code where we left shift by the difference.*/
  assert(AUTOCORR_LEFT_SHIFT >= AUTOCORR_EXTRA_VARIANCE_EXPONENT);

  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  assert((LPC_COMPUTE_INTERVAL & (LPC_COMPUTE_INTERVAL-1)) == 0);  /* Power of 2. */
  /* The y < x / 2 below just means "y is much less than x". */
  assert(LPC_COMPUTE_INTERVAL < (AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT) / 2);
  assert((EXPONENT_BUFFER_SIZE-1)*2 > 12);
  assert((EXPONENT_BUFFER_SIZE & (EXPONENT_BUFFER_SIZE-1)) == 0);  /* Power of 2. */
  assert(EXPONENT_BUFFER_SIZE > (12/2) + 1); /* should exceed maximum range of exponents,
                                                divided by  because that's how much they can
                                                change from time to time.  The + 1 is
                                                in case I missed something. */

  assert((EXPONENT_BUFFER_SIZE & (EXPONENT_BUFFER_SIZE-1)) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE % AUTOCORR_BLOCK_SIZE == 0);
  assert((SIGNAL_BUFFER_SIZE & (SIGNAL_BUFFER_SIZE - 1)) == 0);  /* Power of 2. */
  assert(SIGNAL_BUFFER_SIZE > AUTOCORR_BLOCK_SIZE + EXPONENT_BUFFER_SIZE + MAX_LPC_ORDER);
  return 1;
}

/**
   Computes the SNR, as a ratio, where a is the signal and (b-a) is the noise.
 */
float lilcom_compute_snr(int64_t num_samples,
                         int16_t *signal_a, int stride_a,
                         int16_t *signal_b, int stride_b) {
  int64_t signal_sumsq = 0, noise_sumsq = 0;
  for (int64_t i = 0; i < num_samples; i++) {
    int64_t a = signal_a[stride_a * i], b = signal_b[stride_b * i];
    signal_sumsq += a * a;
    noise_sumsq += (b-a)*(b-a);
  }
  /** return the answer in decibels.  */
  if (signal_sumsq == 0.0 && noise_sumsq == 0.0)
    return 10.0 * log10(0.0); /* log(0) = -inf. */
  else
    return 10.0 * log10((noise_sumsq * 1.0) / signal_sumsq);
}

/**
   Computes the SNR, as a ratio, where a is the signal and (b-a) is the noise.
 */
float lilcom_compute_snr_float(int64_t num_samples,
                               float *signal_a, int stride_a,
                               float *signal_b, int stride_b) {
  double signal_sumsq = 0, noise_sumsq = 0;
  for (int64_t i = 0; i < num_samples; i++) {
    double a = signal_a[stride_a * i], b = signal_b[stride_b * i];
    signal_sumsq += a * a;
    noise_sumsq += (b-a)*(b-a);
  }
  /** return the answer in decibels.  */
  if (signal_sumsq == 0.0 && noise_sumsq == 0.0)
    return 10.0 * log10(0.0); /* log(0) = -inf. */
  else
    return 10.0 * log10((noise_sumsq * 1.0) / signal_sumsq);
}


void lilcom_test_compress_sine() {
  int16_t buffer[1000];
  int lpc_order = 10;
  for (int i = 0; i < 1000; i++)
    buffer[i] = 700 * sin(i * 0.01);

  int8_t compressed[1004];
  int exponent = -15, exponent2;
  lilcom_compress(1000, buffer, 1, compressed, 1,
                  lpc_order, exponent);
  int16_t decompressed[1000];
  if (lilcom_decompress(1000, compressed, 1, decompressed, 1, &exponent2) != 0) {
    fprintf(stderr, "Decompression failed\n");
  }
  assert(exponent2 == exponent);
  fprintf(stderr, "Sine snr (dB) = %f%%\n", lilcom_compute_snr(1000 , buffer, 1, decompressed, 1));
}

void lilcom_test_compress_sine_overflow() {
  int16_t buffer[1000];
  int lpc_order = 10;
  for (int i = 0; i < 1000; i++)
    buffer[i] = 65535 * sin(i * 0.01);

  int exponent = -15, exponent2;
  int8_t compressed[1004];
  lilcom_compress(1000, buffer, 1, compressed, 1,
                  lpc_order, exponent);
  int16_t decompressed[1000];
  if (lilcom_decompress(1000, compressed, 1, decompressed, 1, &exponent2) != 0) {
    fprintf(stderr, "Decompression failed\n");
  }
  assert(exponent2 == exponent);
  fprintf(stderr, "Sine-overflow snr (dB) = %f%%\n", lilcom_compute_snr(1000 , buffer, 1, decompressed, 1));
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

  for (int exponent = -140; exponent <= 131; exponent++) {
    /* Note: 130 and 131 are special cases, for NaN and FLT_MAX.. */
    double scale = pow(2.0, exponent); /* Caution: scale may be inf. */
    if (exponent == 130)
      scale = scale - scale;  /* NaN. */

    for (int i = 0; i < 1000; i++) {
      buffer[i] = 0.5 * sin(i * 0.01) + 0.2 * sin(i * 0.1) + 0.1 * sin(i * 0.25);
      buffer[i] *= scale;
    }

    if (exponent == 131) {
      /* Replace infinities with FLT_MAX. */
      for (int i = 0; i < 1000; i++) {
        if (buffer[i] - buffer[i] != 0) {
          buffer[i] = (buffer[i] > 0 ? FLT_MAX : -FLT_MAX);
        }
      }
    }

    int8_t compressed[1004];
    int16_t temp_space[500];
    int ret = lilcom_compress_float(500, buffer, 2, compressed, 2,
                                    lpc_order, temp_space);
    if (ret) {
      fprintf(stderr, "float compression failed for exponent = %d (this may be expected), max abs float value = %f\n",
              exponent,  max_abs_float_value(500, buffer, 2));
      continue;
    }

    float decompressed[500];
    if (lilcom_decompress_float(500, compressed, 2, decompressed, 1) != 0) {
      fprintf(stderr, "Decompression failed.  This is not expected if compression worked.\n");
      exit(1);
    }

    fprintf(stderr, "For data-generation exponent=%d, {input,output} max-abs-float-value={%f,%f}, floating-point 3-sine snr = %f%%\n",
            exponent,
            max_abs_float_value(500, buffer, 2),
            max_abs_float_value(500, decompressed, 1),
            lilcom_compute_snr_float(500, buffer, 2, decompressed, 1));
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
  assert(max_abs_float_value(10, array, 10) == lilcom_abs(array[50]));

  array[50] = pow(2.0, 129);  /* should generate infinity. */

  assert(max_abs_float_value(100, array, 1) == lilcom_abs(array[50]));

  array[50] = array[50] - array[50];  /* should generate nan. */

  /** The following checks that the return value is NaN. */
  float ans = max_abs_float_value(100, array, 1);
  assert(ans != ans);  /* Check for NaN. */

  /** Positions near the end are special, need to test separately. */
  array[97] = array[50];
  array[50] = 0.0;
  ans = max_abs_float_value(99, array, 1);
  assert(ans != ans);  /* Check for NaN. */
  array[97] = 0.0;

  array[50] = 5;
  array[51] = -6;

  assert(max_abs_float_value(100, array, 1) == lilcom_abs(array[51]));

  array[99] = 500.0;
  array[100] = 1000.0;  /* not part of the real array. */

  assert(max_abs_float_value(100, array, 1) == lilcom_abs(array[99]));
}


int main() {
  lilcom_check_constants();
  lilcom_test_compress_sine();
  lilcom_test_compress_sine_overflow();
  lilcom_test_compress_float();
  lilcom_test_compute_conversion_exponent();
  lilcom_test_get_max_abs_float_value();
}
#endif
