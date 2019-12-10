#ifndef __LILCOM__PREDICTION_MATH_H__
#define __LILCOM__PREDICTION_MATH_H__

#include "fixed_math.h"

/*
   This header contains some functions for estimating linear prediction
   parameters.
*/


/**
    Let `autocorr`, `y` and `x` be vectors of dimension N; and
    let A be an N x K Toeplitz matrix with entries A[i,j] = autocorr[abs(i-j)].

    This function solves the linear system
       A x = y,
    for x.

    The Toeplitz matrix must satisfy the usual conditions for algorithms on
    Toeplitz matrices, meaning no singular leading minor may be singular
    (i.e. not det(A[0:n,0:n])==0 for any n < N).  This will naturally be
    satisfied if A is the autocorrelation of a finite nonzero sequence (I
    believe).  (In practice we'll use some kind of smoothing to make extra sure
    that A is nonsingular.)

    This function solves for x using the Levinson-Trench-Zohar
    algorithm/recursion..  I had to look this up... in this case
    y is an arbitrary vector, so the Levinson-Durbin recursion as normally
    used in signal processing doesn't apply.

    I am looking at:
       https://core.ac.uk/download/pdf/4382193.pdf:
      "Levinson and fast Choleski algorithms for Toeplitz and almost
      Toeplitz matrices", RLE technical report no. 538 by Bruce R Muscius,
      Research Laboratory of Electronics, MIT,

    particularly equations 2.4, 2.6, 2.7, 2.8, (unnumbered formula below 2.9),
    2.10.  There is opportunity for simplification as compared with that
    write-up, because here the Toeplitz matrix is symmetric.

       @param [in] autocorr  The autocorrelation coefficients of dimension n
       @param [in] y         The y in A x = y above
       @param [out] x        The x to be solved for
       @param [in,out] temp1, temp2  Two temporary vectors of dimension n
                             should be from different regions from each
                             other.

           ALL VECTORS SHOULD BE FROM DIFFERENT REGIONS
           (except autocorr and y may share a region, since they are unchanged.)
           YOU SHOULD CALL ZeroRegion64() on the regions for x, temp1 and temp2
           before calling this function, to make sure the `size` values of the
           regions stay accurate.

       @return       Returns 0 on success, 1 on failure.  The only
                     possible failure condition is division by zero,
                     which should not happen if A was nonsingular (actually:
                     nonsingular enough that roundoff errors don't make it seem
                     singular)
 */
int ToeplitzSolve(const Vector64 *autocorr, const Vector64 *y, Vector64 *x,
                  Vector64 *temp1, Vector64 *temp2);


struct LpcStats {
  /* the allocated block of memory, size depends on the order. */
  void *allocated_block;
  /* The LPC order */
  int lpc_order;
  /* The max allowable block size (determines the sizes of certain stored
     vectors */
  int max_block_size;
  /* region and vector for the autocorr coeffs, of dimension lpc_order + 1 */
  Region64 autocorr_region;
  Vector64 autocorr;
  /* region and vector for a temporary vector of the same size as the
     autocorr coeffs.  Note: we actually swap autocorr and autocorr_tmp
     so they may sometimes point to each other's regions. */
  Region64 autocorr_tmp_region;
  Vector64 autocorr_tmp;

  /* region and vector for x_hat (scaled version of the input x), including
     `lpc_order` samples of context and the current block; size of region
     is (lpc_order + max_block_size) elements and size of vector can vary. */
  Region64 x_hat_region;
  Vector64 x_hat;

  /* In Python-like notation allowing negative indexing:
     sqrt_scale[-k], for k > 0, contains self.eta ** k.
     The dim of `sqrt_scale` is 2*max(lpc_order, max_block_size);
  */
  Region64 sqrt_scale_region;
  Vector64 sqrt_scale;

  /* In Python-like notation allowing negative indexing:
     inv_sqrt_scale[k], for k >= 0, contains self.eta ** -k.
     The dim of `inv_sqrt_scale` is 2 * lpc_order + 1.
     It's used when computing A^all.
  */
  Region64 inv_sqrt_scale_region;
  Vector64 inv_sqrt_scale;

  /*  The first `lpc_order` samples are stored here (for purposes of
      computing A^-), scaled: initial_samples[t] = x[t].
      They are just stores as integers and exponent=0 set. */
  Region64 initial_samples_region;
  Vector64 initial_samples;

  /*  Always contains the most recent 'lpc_order' samples, up to T-1;
      a rolling buffer modulo lpc_order, indexed by 't'. */
  int16_t *context_buf;


  /* eta is a forgetting factor, like 0.99 */
  Scalar64 eta;
  /* eta squared. */
  Scalar64 eta_2;
  /* eta^-2. */
  Scalar64 eta_m2;
  ssize_t T;
  /* eta_2T is eta to the power 2*(T-N); it will only
     have this value when T >= N. */
  Scalar64 eta_2TN;
};



/**
   This is like an appendage to struct LpcStats; it contains the parts that
   vary with the order of LPC stats requested (which is allowed to be less
   than the order of LPC stats that is accumulated).

   It is used when the user requests the statistics from the LpcStats
   object.

 */
struct LpcStatsAux {
  const struct LpcStats *stats;
  /* allocated_block is the region of memory we allocated */
  void *allocated_block;
  /* we require 0 < lpc_order <= stats->lpc_order */
  int lpc_order;

  /* Region and matrix for A_minus (A^- in the writeup).  Note: like the A^-
     stored in the dict in the Python version, it is a missing a factor of eta
     ** (self.T * 2) */
  Region64 A_minus_region;
  Matrix64 A_minus;

  /* Region and matrix for A_plus (A^+ in the writeup). */
  Region64 A_plus_region;
  Matrix64 A_plus;

  /* Region and matrix for A (A^all, A^+ and A^- go in here. */
  Region64 A_region;
  Matrix64 A;

  Region64 autocorr_reflected_region;
  Vector64 autocorr_reflected;

  /* x_context is a temporary buffer where past lpc_order input (`x`) samples
     are put, in reversed order; they are copied from stats->context_buf.  It's
     of dimension lpc_order. */
  Region64 x_context_region;
  Vector64 x_context;

};


/**
   Initializes the LpcStats object.  This is derived from LpcStats::__init__ in
   ../test/linear_prediction_new.py

    @param [in] lpc_order   Order of LPC prediction; must be >= 1
    @param [in] max_block_size  The caller asserts that they will never call
                            LpcStatsAcceptBlock() with a block size greater
                            than this.  (Needed for memory allocation).
                            Don't call this with a too-large block size as it
                            will affect performance (due to ResizeRegion())
    @param [in] eta         Constant 0 < eta < 1 like 0.99, that is a per-sample
                            decay rate (actually eta^2 is the decay rate of the
                            weights in the objective; eta is the decay of a
                            "virtual signal" that we use in the update.
    @param [in] eta_den_power  See documentation for eta_num
    @param [out] stats      The object to be initialized

    @return:
         Returns 0 on success; 1 if it failed as a result of failure to
         allocate memory
 */
int LpcStatsInit(int lpc_order,
                 int max_block_size,
                 const Scalar64 *eta,
                 struct LpcStats *stats);

/*  Frees memory used in the `stats` object. */
void LpcStatsDestroy(struct LpcStats *stats);


/**
   Initializes the LpcStatsAux object (this is for obtaining the statistics
   matrix A for a specific LPC order, which may be <= the lpc order
   of the actual stats).
   This just allocates memory.  You would then call LpcStatsAuxCompute()
   after accumulating stats.

    @param [in] stats       LpcStats object from which this LpcStatsAux
                            object will get its statistics.  Must be
                            initialized (i.e. must have called
                            LpcStatsInit()), but doesn't have to
                            actually have stats in it yet.
    @param [in] lpc_order   Order of LPC prediction; must be >= 1 and
                            <= stats->
    @param [out] stats_aux   The LpcStatsAux object to be initialized

    @return:
         Returns 0 on success; 1 if it failed as a result of failure to
         allocate memory
 */
int LpcStatsAuxInit(struct LpcStats *stats,
                    int lpc_order,
                    struct LpcStatsAux *stats_aux);

/**
   Computes stats_aux->A and stats_aux->autocorr_reflected, which are the stats
   needed to update the autocorrelation coefficients.
 */
void LpcStatsAuxCompute(struct LpcStatsAux *stats_aux);

/*  Frees memory used in the `stats_aux` object. */
void LpcStatsAuxDestroy(struct LpcStatsAux *stats_aux);



/*
  Gives a fresh block of data to the LpcStats object to process.
 */
void LpcStatsAcceptBlock16(int block_size,
                           int16_t *data,
                           struct LpcStats *stats);

#endif  /* ifndef __LILCOM__PREDICTION_MATH_H__ */


