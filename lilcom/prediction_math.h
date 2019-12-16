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
  /* eta_2T is eta to the power 2*(T-N); it will only have this value when T >=
     N, and is 1.0 otherwise. */
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

  /* Region and matrix for A_plus (A^+ in the writeup).  Dimension is
     lpc_order+1 by lpc_order+1 */
  Region64 A_plus_region;
  Matrix64 A_plus;

  /* Region and matrix for A (A^all, A^+ and A^- go in here. */
  Region64 A_region;
  Matrix64 A;

  /* Region and vector for autocorr_reflected.  Dimension is lpc_order+1. */
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

/*
  This convenience function seets up `A_for_solver` to point
  to (in NumPy notation) A[1:,1:] where A is aux->A,
  `b_for_solver` to point to A[0,1:], and autocorr_for_solver
  to point to autocorr_reflected[:-1] (i.e. all but the last
  element) where autocorr_reflected is aux->autocorr_reflected.

  `aux` is unaffected by this call, but we don't make it `const`
  because we want to draw attention to the fact that the zeroth element
  of autocorr_for_solver will be modified in OnlineLinearSolverStep(),
  so we will later be modifying `aux` as an indirect effect of
  calling this.
*/
void LpcStatsAuxGetInfoForSolver(struct LpcStatsAux *aux,
                                 Matrix64 *A_for_solver,
                                 Vector64 *b_for_solver,
                                 Vector64 *autocorr_for_solver);

/*
  This struct parallels class SimpleOnlineLinearSolver in
  ../test/linear_prediction.py.

  It is for solving systems of the form A x = b with A symmetric positive
  definite, where A can be reasonably closely approximated by a Toeplitz matrix
  that the user can supply, and A and b are changing with time (so it makes
  sense to start the optimization from the previous value of x).

  It uses a very simple method that's essentially gradient descent with a
  learning rate of 1, but preconditioned by the inverse of the Toeplitz matrix
  (we can multiply by this inverse using a Toeplitz solver in O(dim^2)).

  This class uses preconditioned conjugate gradient descent (CGD)
  to approximately solve for x.
 */
struct OnlineLinearSolver {
  /* block of memory to free when we are done */
  void *allocated_block;


  /* The dimension of the thing we are solving for. */
  int dim;

  /* 1.0 plus the proportionality constant diag_smoothing for smoothing
   * autocorr[0].  E.g. 1.0 + 1.0e-07. */
  Scalar64 diag_smoothing_plus_one;
  /* absolute value used when smoothing autocorr[0].  E.g. 1.0e-20. */
  Scalar64 abs_smoothing;


  /* The current estimate of x (e.g. x might be the filter coefficients).
     Dimension is `dim`. */
  Region64 x_region;
  Vector64 x;

  /* The residual (this is a quantity used internally).  Dimension is `dim`. */
  Region64 r_region;
  Vector64 r;
  /* The preconditioned residual == the update step.  Equals r times
     inv(M) where M is the Toeplitz matrix formed from the autocorr stats. */
  Region64 z_region;
  Vector64 z;

  /* temp1 and temp2 are two temporary vectors of dimension `dim` that are
     used in ToeplitzSolve(). */
  Region64 temp1_region;
  Vector64 temp1;
  Region64 temp2_region;
  Vector64 temp2;
};

/*
  Initialize OnlineLinearSolver object (allocates memory and sets 'x' to
  zero).

     @param [in] dim   Dimension of the linear problem we are solving.
                       (would be the LPC filter order)
     @param [in] diag_smoothing, abs_smoothing
                       Constants for smoothing added to diagonal
                       of the Toeplitz matrix formed by the autocorrelation
                       coefficients, equivalent to increasing the 0th
                       autocorrelation coefficient, formula is:
                         autocorr[0] += diag_smoothing * autocorr[0] + abs_smoothing
     @param [out] solver  The solver object to be initialized

     @return           Returns 0 on success, 1 if it fails due to memory allocation
                       failure.
 */
int OnlineLinearSolverInit(int dim,
                           const Scalar64 *diag_smoothing,
                           const Scalar64 *abs_smoothing,
                           struct OnlineLinearSolver *solver);

/*
  Do one step of the linear solver (we are trying to approach a solution of
  the equation A x = b, but where A and b are changing with time).

    @param [in] A       The matrix A that is the quadratic term in the objective
                        function we are minimizing (0.5 x^T A x - b).  Must be
                        of dimension solver->dim by solver->dim and must be
                        positive semidefinite.
    @param [in] b       The vector b that is the linear term in the objective
                        function we are minimizing.
    @param [in,out] autocorr  A vector of dimension `dim` that provides an
                        approximation to A.  Specifically: if we construct
                        a `dim` by `dim` matrix M with elements M(i,j) =
                        autocorr(abs(i-j)) [assuming zero-based indexing],
                        the caller asserts that (a) this has some similarity
                        to A, and (b) all its eigenvalues are strictly
                        positive.  There are some specific limitations
                        that need to be met to prevent divergence:
                        e.g. if for some direction x,
                        (x^T M x) < 0.5 (x^T A x), calling this function
                        repeatedly with the same values of A,b,autocorr
                        would lead to divergence of x.  We can't exclude
                        this in general, so this function will reset the
                        previous `x` in the online update to zero anytime
                        it detects that zero would give a better
                        objective-function value than the previous `x`.
                        (This should be rare).
                        CAUTION: autocorr is consumed destructively
                        (we modify one element by adding smoothing).
    @param [in,out] solver   The solver object; its x value will be
                        updated.
    @return             Returns 0 on success, and 1 if there was
                        some supposed-to-be-rare-or-impossible problem
                        such as needing to `revert` x or the Toeplitz
                        matrix not being invertible.  This failure
                        condition is just for diagnostic purposes;
                        this function will still leave solver->x in
                        a reasonable and well-defined state so that
                        compression can continue.
 */
int OnlineLinearSolverStep(const Matrix64 *A,
                           const Vector64 *b,
                           Vector64 *autocorr,
                           struct OnlineLinearSolver *solver);

/*
  Destroy OnlineLinearSolver object (releases memory).
 */
void OnlineLinearSolverDestroy(struct OnlineLinearSolver *solver);





/* TODO: make sure that it's impossible for it to attempt to shift right
   by more than 64.  (could happen if exponents are weird.) */

#endif  /* ifndef __LILCOM__PREDICTION_MATH_H__ */



