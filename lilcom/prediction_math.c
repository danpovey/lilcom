#include <stdio.h>
#include "fixed_math.h"
#include "prediction_math.h"
#include "lilcom_common.h"  /* for debug_fprintf */

/**
   Below is the working Python code that ToeplitzSolve is based on.
   Please see the documentation in prediction_math.h for the interface of
   this "C" function; this documentation explains its inner workings.

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


  /* Copy the objects to local copies; this may save a few registers.  Also use
     slightly different names internally to in the interface (these names
     correspond to the technical report I am following)..*/
  Vector64 r = *autocorr_in,
      y = *y_in,
      x = *x_in,
      b = *temp1,
      b_temp = *temp2;
  /* Caution: b and b_temp get shallow-swapped inside the function. */

  int N = autocorr_in->dim - 1;

  CopyIntToVector64Elem(0, 1, 1, &b);  /* b[0] = 1.0 */
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

    CopyIntToVector64Elem(0, 0, 0, &b_temp);  /* b_temp[0] = 0.0 (3rd arg is size_hint==0). */
    Vector64 b_temp_1n1; /* == b_temp[1:n+1] */
    InitSubVector64(&b_temp, 1, n, 1, &b_temp_1n1);
    CopyVector64(&bn, &b_temp_1n1); /* b_temp[1:n+1] = b[:n] */
    CheckRegion64Size(b_temp.region);
    Vector64 b_temp_n; /* == b[0:n] */
    InitSubVector64(&b_temp, 0, n, 1, &b_temp_n);
    Vector64 b_n_flip; /* == np.flip(b[:n]) */
    InitSubVector64(&b, n - 1, n, -1, &b_n_flip);
    CheckRegion64Size(b_temp.region);
    AddScalarVector64(&nu_n, &b_n_flip, &b_temp_n); /* b_temp[:n] += nu_n * np.flip(b[:n]) */

    /* Shallow-swap the vectors b and b_temp.  In the equations this is
       written as:
         b[:n+1] = b_temp[:n+1]
         but we do it via shallow swap. */
    { Vector64 temp = b; b = b_temp; b_temp = temp; }

    /* The next few lines will do:
         epsilon *= (1.0 - nu_n * nu_n) */
    Scalar64 nu_n2,  /* nu_n^2 */
        epsilon_minus_nu_n2;
    CheckScalar64Size(&nu_n);
    MulScalar64(&nu_n, &nu_n, &nu_n2); /* nu_n2 := nu_n * nu_n */
    CheckScalar64Size(&nu_n2);
    NegateScalar64(&nu_n2);   /* nu_n2 *= -1 */
    CheckScalar64Size(&nu_n2);
    MulScalar64(&epsilon, &nu_n2, &epsilon_minus_nu_n2);
    AddScalar64(&epsilon, &epsilon_minus_nu_n2, &epsilon); /* epsilon -= mu_n*mu_n*epsilon */
    if (epsilon.data <= 0) {
      debug_fprintf(stderr, "Negative or zero epilon %f in Toeplitz computation (n=%d)\n",
                    (float)Scalar64ToDouble(&epsilon), n);
      return 1;
    }

    /* we'll be computing:
       lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])
       ratio = epsilon / lambda_n
    */
    Scalar64 ratio;
    {
      Scalar64 lambda_n, y_elem_n;
      Vector64 x_n, r_1n1_flip;
      InitSubVector64(&x, 0, n, 1, &x_n);
      InitSubVector64(&r, n, n, -1, &r_1n1_flip);
      DotVector64(&x_n, &r_1n1_flip, &lambda_n);
      NegateScalar64(&lambda_n);
      /* lambda_n is now
         - sum([ r[n-j] * x[j] for j in range(n)]) == np.dot(np.flip(r[1:n+1]), j[:n]) */
      CopyVectorElemToScalar64(&y, n, &y_elem_n);
      AddScalar64(&y_elem_n, &lambda_n, &lambda_n); /* lambda_n += y[n]. */
      DivideScalar64(&lambda_n, &epsilon, &ratio); /* ratio = lambda_n / epsilon */
    }

    Vector64 x_n1, b_n1;
    InitSubVector64(&x, 0, n+1, 1, &x_n1);
    InitSubVector64(&b, 0, n+1, 1, &b_n1);
    /* next line: x[:n+1] += (lambda_n / epsilon) * b[:n+1] */
    AddScalarVector64(&ratio, &b_n1, &x_n1);
  }

  debug_fprintf(stderr, "Output x vector is: ");
  PrintVector64(&x);

  return 0;
}


#ifdef PREDICTION_MATH_TEST
#include <math.h>

/**
   Construct a matrix `mat` with elements
     mat[i,j] = vec[abs(i-j)]
   Requires that `mat` occupy its entire region (so that
   we can freely set its exponent and size).
 */
void ConstructToeplitzMatrix(Vector64 *vec,
                             Matrix64 *mat) {
  int size = mat->num_rows * mat->num_cols;
  assert(size == mat->region->dim &&
         "Matrix must occupy its whole region");
  mat->region->size = vec->region->size;
  mat->region->exponent = vec->region->exponent;
  int64_t *vec_data = vec->data,
      *mat_data = mat->data;
  int dim = vec->dim;
  assert(mat->num_rows == dim && mat->num_cols == dim);
  int vec_stride = vec->stride,
      mat_row_stride = mat->row_stride,
      mat_col_stride = mat->col_stride;
  for (int r = 0; r < dim; r++) {
    for (int c = 0; c < dim; c++) {
      int i = (r > c ? r - c : c - r);
      mat_data[r * mat_row_stride + c * mat_col_stride] = vec_data[i * vec_stride];
    }
  }
}


int main() {
  /* Note: this mirrors code in ../test/linear_prediction.py, in
   * test_toeplitz_solve_compare().  I'm using the same exact
   * numbers in testing, to make it easier to debug.
   */

  int64_t autocorr_array[4] = { 10, 5, 2, 1 },
      y_array[4]  = { 1, 2, 3, 4 },
      x_array[4] = { 0, 0, 0, 0 },
      temp1_array[4] = { 0, 0, 0, 0 },
      temp2_array[4] = { 0, 0, 0, 0 };
  Region64 r1, r2, r3, r4, r5;
  Vector64 autocorr, y, x, temp1, temp2;
  int size_hint = 2;
  InitRegionAndVector64(autocorr_array, 4, 0, size_hint,
                        &r1, &autocorr);
  InitRegionAndVector64(y_array, 4, 0, size_hint,
                        &r2, &y);
  InitRegionAndVector64(x_array, 4, 0, size_hint,
                        &r3, &x);
  InitRegionAndVector64(temp1_array, 4, 0, size_hint,
                        &r4, &temp1);
  InitRegionAndVector64(temp2_array, 4, 0, size_hint,
                        &r5, &temp2);
  ToeplitzSolve(&autocorr, &y, &x, &temp1, &temp2);


  CheckRegion64Size(temp1.region);
  CheckRegion64Size(temp2.region);



  /* Now check that the solution is correct. */
  int64_t mat_array[16];
  for (int i = 0; i < 16; i++) mat_array[i] = 0;
  Region64 mat_region;
  Matrix64 mat;
  InitRegionAndMatrix64(mat_array, 4, 4, 0, 0,
                        &mat_region, &mat);

  CheckRegion64Size(temp1.region);
  /* temp1 := mat * x. */
  SetMatrixVector64(&mat, &x, &temp1);
  CheckRegion64Size(temp1.region);

  /* temp1 -= y.   Now temp1 is the error/residual. */
  AddIntVector64(-1, &y, &temp1);
  double rel_error = DotVector64AsDouble(&temp1, &temp1) /
      DotVector64AsDouble(&y, &y);
  fprintf(stderr, "Relative error in Toeplitz inversion is %g\n", (float)sqrt(rel_error));
  assert(rel_error < pow(2.0, -20));

}

#endif


