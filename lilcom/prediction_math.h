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
       @param [in,out] temp  A temporary vector of dimension

       @return       Returns 0 on success, 1 on failure.  The only
                     possible failure condition is division by zero,
                     which should not happen if A was nonsingular (actually:
                     nonsingular enough that roundoff errors don't make it seem
                     singular)
 */
int ToeplitzSolve(const Vector64 *autocorr, const Vector64 *y, Vector64 *x,
                  Vector64 *temp);
