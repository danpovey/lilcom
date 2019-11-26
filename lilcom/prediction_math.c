#include <stdio.h>
#include "prediction_math.h"


/**
Below is the working Python code that ToeplitzSolve is based on.

def toeplitz_solve(autocorr, y):
    # The technical report I am looking at deals with things of dimension (N+1),
    # so for consistency I am letting N be the dimension minus one.
    N = autocorr.shape[0] - 1
    assert y.shape[0] == N + 1
    x = np.zeros(N+1)
    b = np.zeros(N+1)
    b_temp = np.zeros(N+1)
    r = autocorr

    b[0] = 1.0
    epsilon = r[0]
    x[0] = y[0] / epsilon

    for n in range(1, N+1):

        # eqn 2.6 for \nu_n.  We are not evaluating \xi_n because it is
        # the same.  Be careful with the indexing of b.  Notice in Eq.
        # (2.3) that the elements of b are in a very strange order,
        # so you have to interpret (2.6) very carefully.
        nu_n = (-1.0 / epsilon) * sum([r[j+1] * b[j] for j in range(n)])

        # next few lines are Eq. 2.7
        b_temp[0] = 0.0
        b_temp[1:n+1] = b[:n]
        b_temp[:n] += nu_n * np.flip(b[:n])
        b[:n+1] = b_temp[:n+1]

        # Eq. 2.8
        epsilon *= (1.0 - nu_n * nu_n)
        assert abs(nu_n) < 1.0

        # The following is an unnumbered formula below Eq. 2.9
        lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])
        x[:n+1] += (lambda_n / epsilon) * b[:n+1];

    return x
 */
int ToeplitzSolve(const Vector64 *autocorr_in, const Vector64 *y_in, Vector64 *x_in,
                  Vector64 *temp1, Vector64 *temp2) {
  assert(autocorr_in->dim == y_in->dim && autocorr_in->dim == x_in->dim);
  /* All regions should be distinct, except possibly autocorr and y. */
  assert(y_in->region != x_in->region && temp1->region != x_in->region &&
      temp1->region != y_in->region && temp2->region != temp1->region &&
      temp2->region != x_in->region && temp2->region != y_in->region &&
      autocorr_in->region != x_in->region && autocorr_in->region != temp1->region
             && autocorr_in->region != temp2->region);
  /* CAUTION: this N  is not the same as the N mentioned in the header.
     For consistency with the literature, all the vectors are of size
     N+1 (this is done for reasons I don't understand) and I set
     N here accordingly.
   */


  /* Copy the objects to local copies; this may save a few registers.
     Also use slightly different names that correspond to the
     technical report I am following.*/
  Vector64 r = *autocorr_in,
      y = *y_in,
      x = *x_in,
      b = *temp1,
      b_temp = *temp2;
  int N = autocorr_in->dim - 1;

  SetVector64ElemToInt(0, 1, 1, &b);  /* b[0] = 1.0 */
  Scalar64 epsilon, x0, y0;
  CopyVectorElemToScalar64(&r, 0, &epsilon);  /* epsilon = r[0] */

  CopyVectorElemToScalar64(&y, 0, &y0); /* next 3 lines: x[0] = y[0] / epsilon. */
  DivideScalar64(&y0, &epsilon, &x0);
  CopyScalar64ToVectorElem(&x0, 0, &x);

  for (int n = 1; n <= N; n++) {  /* for n in range(1, N+1): */

    /* New few lines:
       nu_n = (-1.0 / epsilon) * sum([r[j+1] * b[j] for j in range(n)])
         ( == (-1.0 / epsilon * (np.dot(r[1:n+1], b[:n]))
    */
    Vector64 r1n1, bn;
    InitSubVector64(&r, 1, n, 1, &r1n1);
    InitSubVector64(&b, 0, n, 1, &bn);
    Scalar64 product, nu_n;
    DotVector64(&r1n1, &bn, &product);
    DivideScalar64(&product, &epsilon, &nu_n);
    NegateScalar64(&nu_n);

    SetVector64ElemToInt(0, 0, 0, &b_temp);  /* b_temp[0] = 0.0 */
    Vector64 b_temp_1n1; /* == b_temp[1:n+1] */
    InitSubVector64(&b_temp, 1, n, 1, &b_temp_1n1);
    CopyVector64(&bn, &b_temp_1n1); /* b_temp[1:n+1] = b[:n] */
    Vector64 b_temp_n; /* == b[0:n] */
    InitSubVector64(&b_temp, 0, n, 1, &b_temp_n);
    Vector64 b_n_flip; /* == np.flip(b[:n]) */
    InitSubVector64(&b, n - 1, n, -1, &b_n_flip);
    AddScalarVector64(&nu_n, &b_n_flip, &b_temp_n); /* b_temp[:n] += nu_n * np.flip(b[:n]) */

    {
      double d = Scalar64ToDouble(&nu_n);
      if (d <= -1 || d >= 1) {
        fprintf(stderr, "Error:, nu_n out of range %f\n", d);
      }
    }
    Scalar64 nu_n2,  /* nu_n^2 */
        factor;
    MulScalar64(&nu_n, &nu_n, &nu_n2);
    NegateScalar64(&nu_n2);
    InitScalar64FromInt(1, &factor);
    AddScalar64(&factor, &nu_n2, &factor);  /*
                                             */
    /*

          b[:n+1] = b_temp[:n+1]

          # Eq. 2.8
          epsilon *= (1.0 - nu_n * nu_n)
          assert abs(nu_n) < 1.0

          # The following is an unnumbered formula below Eq. 2.9
          lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])
          x[:n+1] += (lambda_n / epsilon) * b[:n+1];

      return x
  */
  }
  return 0;
}





