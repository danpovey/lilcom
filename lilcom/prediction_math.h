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
  /* region and vector for the autocorr coeffs, of dimension lpc_order + 1 */
  Region64 autocorr_region;
  Vector64 autocorr;
  /* eta is a forgetting factor, like 0.99 */
  Scalar64 eta;
  ssize_t T;

  Region64



};

/**
   Initializes the LpcStats object.  This is derived from LpcStats::__init__ in
   ../test/linear_prediction_new.py

    @param [in] lpc_order   Order of LPC prediction; must be >= 1
    @param [in] eta_num     Numerator in eta value.  eta is the decay constant
                            per sample (should be <1 but close to 1), and will
                            be of the form eta_num / 2^eta_den_power,
                            e.g. 127 / 128.  eta_num must be less than
                            (1 << eta_den_power)
    @param [in] eta_den_power  See documentation for eta_num
    @param [out] stats      The object to be initialized

    @return:
         Returns 0 on success; 1 if it failed as a result of failure to
         allocate memory
 */
int LpcStatsInit(int lpc_order,
                 int eta_num,
                 int eta_den_power,
                 LpcStats *stats);

/*

 */
void LpcStatsAcceptBlock(int num_samples,
                         int16_t *data,
                         LpcStats *stats);

#endif
