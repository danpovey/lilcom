#ifndef LILCOM_LILCOM_COMMON_H_INCLUDED_
#define LILCOM_LILCOM_COMMON_H_INCLUDED_

/* This header contains constants, and a few macros.
   Note: there is test code in lilcom.c that tests that these
   constants satisfy certain properties; see
   lilcom_check_constants().
 */

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
   Number of bytes in the header
*/
#define LILCOM_HEADER_BYTES 5

/*  These document the minimum and maximum allowed number of bits per
    sample.  The normal values would be 6 or 8. */
#define LILCOM_MIN_BPS 4
#define LILCOM_MAX_BPS 16

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
   The power of two by which the LPC coefficients are multiplied when stored as
   integers in struct LpcComputation and while being applied during prediction.
   The specific value is chosen so that we can do the summation in int32_t.

   CAUTION: this cannot be greater than 14 = 30 - 16, due to tricks used in the
   prediction code.
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
   accurate, but, as explained in the comment for LpcComputation::autocorr,
   in order to prevent overflow we require:

   (30 + AUTOCORR_LEFT_SHIFT + log(AUTOCORR_BLOCK_SIZE) + (AUTOCORR_DECAY_EXPONENT-1))
     to be less than 61.
   The -1 added to AUTOCORR_DECAY_EXPONENT is because the decay of the actual stats
   is around x=(1 - 0.5^(AUTOCORR_DECAY_EXPONENT-1)) since it's the square of the
   decay of the window; the sum of the series 1,x,x^2.. is 2^(AUTOCORR_DECAY_EXPONENT-1).

   Currently the sum above equals: 30 + 20 + 4 + 6  = 60  < 61.
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
   AUTOCORR_DECAY_EXPONENT, together with AUTOCORR_BLOCK_SIZE, defines how fast
   we decay the autocorrelation stats; small values mean the autocorrelation
   coefficients move faster, large values mean they move slower.  Conceptually
   we are computing the autocorrelation on a copy of the signal multiplied by a
   window that decreases by (1 - 0.5^AUTOCORR_DECAY_EXPONENT) each time we go
   AUTOCORR_BLOCK_SIZE samples into th efast.  [Note: for implementation
   reasons, this window falls in steps, not continuously.]  In addition, we
   imagine that this windowed signal was reflected around the current time
   This avoids the windowing function having a sharp discontinuity which
   would make the LPC prediction worse, while also ensuring that the data
   used to estimate the LPC prediction is fairly fresh.

   The approximate number of samples that we `remember` in the autocorrelation`
   computation is approximately AUTOCORR_BLOCK_SIZE << AUTOCORR_DECAY_EXPONENT,
   which is currently 2048.  Of course it's a decaying average, so we remember
   all samples to some extent.  This may be important to tune, for
   performance.
 */
#define AUTOCORR_DECAY_EXPONENT 7



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
   The mantissas encode prediction residuals.  The largest-magnitude residual is
   65535, which is 13767 - (-13768); and the least number of bits we'd need
   to encode this as a signed integer is 17.
*/
#define MAX_POSSIBLE_WIDTH 17

/**
   MAX_BACKTRACK is a convenient way of saying 'the maximum number of samples we
   might have to backtrack'.  The reason for the "+1" is that because of the way
   the exponents are encoded.  Think of it like a chess piece that can move
   subject to the following rules:
      - If it is currently on a black square, it may move either one square
        to the right, or one square down and to the right.
      - If it is currently on a white square, it may move either one square
        to the right, or one square up and to the right.
   Here, "right" == increasing t values; "up" == increasing exponent.

   If we require the exponent to have a certain value E on a particular time t
   (and suppose, for the worst case, that this corresponds to a black square on
   our board and E == MAX_POSSIBLE_WIDTH), then it can achieve that if at a
   particular time t-(E+1) it was zero.  The exponent is not allowed to be less
   than zero, so the maximumum number of frames we might have to go back in time
   is MAX_POSSIBLE_WIDTH+1.  (Actually the +1 might not even be needed because
   that exponent at time t-(E+1) wouldn't have to be changed.)
 */
#define MAX_BACKTRACK (MAX_POSSIBLE_WIDTH+1)

/**
   This rolling-buffer size determines how far back in time we keep the
   exponents we used to compress the signal.  It determines how far it's
   possible for us to backtrack when we encounter out-of-range values and need
   to increase the exponent.  The most we'd ever need to add to the exponent
   is +15, and we can only increase the exponent every other frame, so
   this needs to be at least 30 (or maybe one or two more than that, I forget
   the exact logic); anyway 32 is enough.

   This must be a power of 2 due to a trick used to compute a modulus, which
   dictates 32.
 */
#define WIDTH_BUFFER_SIZE 32


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
   SIGNAL_BUFFER_SIZE determines the size of a rolling buffer containing the
   history of the compressed version of the signal, to be used while
   compressing.  It must be a multiple of AUTOCORR_BLOCK_SIZE.  It must also
   satisfy:

    SIGNAL_BUFFER_SIZE >
       AUTOCORR_BLOCK_SIZE + (MAX_POSSIBLE_WIDTH*2) + MAX_LPC_ORDER

   (currently: 128 > 16 + (15*2) + 14).

   That is: it needs to store a whole autocorrelation-stats block's worth of
   data, plus the farthest we might backtrack (MAX_POSSIBLE_WIDTH * 2),
   plus enough context to process the first sample of the block
   (i.e. MAX_LPC_ORDER).  This backtracking is because there may be situations
   where we need to recompute the autocorrelation coefficients of a block if
   backtracking causes us to revisit that autocorrelation block.

   It must also be a power of 2, due to tricks we use when computing modulus.

   Every time we get to the beginning of the buffer we need to do a little extra
   work to place the history before the start of the buffer; the desire to avoid
   this extra work is why we don't just make this 64 (i.e. two blocks' worth).
   The reason for not using a huge value is to save memory and be kind to the
   cache.
 */
#define SIGNAL_BUFFER_SIZE 128


/**
   STAGING_BLOCK_SIZE is the size of a block in a rolling buffer containing the
   compressed signal.  The buffer contains the signal as bytes, one per
   time-step, which will later be consolidated if bytes_per_sample < 8.

   We require that STAGING_BLOCK_SIZE be a multiple of 8, which
   ensures that the number of bits in the staging block must be a multiple of
   8, hence a whole number of bytes.

   It should be at least twice the maximum number of time steps we might backtrack
   (which is MAX_POSSIBLE_WIDTH+1 = 18).  We choose to make it 32 bytes,
   meaning the entire staging area contains 64 bytes.

   There are 2 staging blocks in the buffer.
 */
#define STAGING_BLOCK_SIZE 32



#ifndef NDEBUG
#define debug_fprintf(...) fprintf(__VA_ARGS__)
#else
#define debug_fprintf(...) while (0)  /* does nothing and allows termination by ';' */
#endif

#ifdef LILCOM_TEST
#define paranoid_assert(x) assert(x)
#else
#define paranoid_assert(x) while(0)
#endif


#define lilcom_abs(a) ((a) > 0 ? (a) : -(a))
#define lilcom_min(a, b) ((a) > (b) ? (b) : (a))


#endif  /*  LILCOM_LILCOM_COMMON_H_INCLUDED_ */
