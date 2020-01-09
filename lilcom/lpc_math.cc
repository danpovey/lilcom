#include "lpc_math.h"
#include "stdio.h"
#include "stdlib.h"
#include <assert.h>


namespace int_math {


/* See documentation in lpc_math.h */
void toeplitz_solve(const IntVec<int32_t> *autocorr,
                    const IntVec<int32_t> *y,
                    IntVec<int32_t> *temp,
                    IntVec<int32_t> *x) {

  IntVec<int32_t> b = *temp;
  zero_int_vector(&b);
  zero_int_vector(x);
  /* CAUTION with this N, it is the dimension of the vectors MINUS ONE.
     This is for compatibility with the literature on Toeplitz solvers.
   */
  int N1 = b.dim, N = b.dim - 1;  /* all dims are the same. */

  { /* b[-1] = 1.0 */
    const int b_nsb = 29;  /* initial number of significant bits in b.  Must be
                              <= 30, which is the max allowed for int32_t.  We
                              make it a bit smaller than 30 (i.e., 29) in order
                              to reduce the probability of needing to shift it
                              right later on if elements of it start getting
                              larger. */
    b.exponent = -(b_nsb - 1);
    b.data[N] = 1 << (b_nsb - 1);
    b.nrsb = 31 - b_nsb;
    b.check();
    assert(static_cast<float>(b[N]) == 1.0);
  }

  /* epsilon = r[0]. */
  IntScalar<int32_t> epsilon((*autocorr)[0]);

  {  /* x[0] = y[0] / epsilon */
    IntScalar<int32_t> x0 = (*y)[0];
    divide(&x0, &epsilon, &x0);
    set_only_nonzero_elem_to(&x0, 0, x);
  }

  /* for n in range(1, N+1): */
  for (int n = 1; n <= N; n++) {
    IntScalar<int64_t> prod;  /* np.dot(r[1:n+1], b[-n:]) */
    compute_dot_product(n, autocorr, 1, &b, N1 - n, &prod);

    /*
      we are computing
         nu_n = (-1.0 / epsilon) * np.dot(r[1:n+1], b[-n:])
      note: abs(nu_n) < 1.0.  mathematically it's <= 1.0, but we
      added a little smoothing to the zeroth autocorr element. */
    IntScalar<int32_t> nu_n;
    divide(&prod, &epsilon, &nu_n);
    negate(&nu_n);
    assert(int_math_abs(static_cast<float>(nu_n)) < 1.0);
    /* next line does b[-(n+1):-1] += nu_n * np.flip(b[-n:]) */
    special_reflection_function(n, &nu_n, &b);

    /* epsilon *= (1.0 - nu_n * nu_n)
       [Note: could have slightly less roundoff by computing
       1-nu_n*nu_n directly using special code?]
     */
    multiply(&nu_n, &nu_n, &nu_n);
    multiply(&nu_n, &epsilon, &nu_n);
    negate(&nu_n);
    add(&epsilon, &nu_n, &epsilon);
    assert(epsilon.elem > 0);

    /* lambda_n = y[n] - np.dot(np.flip(r[1:n+1]), x[:n]) */

    IntScalar<int64_t> lambda_n;
    /* next line sets lambda_n = np.dot(np.flip(r[1:n+1]), x[:n]) */
    compute_dot_product_flip(n, autocorr, 1, x, 0, &lambda_n);
    negate(&lambda_n);
    /* next two lines do lambda_n += y[n]. */
    IntScalar<int32_t> lambda_n32;
    copy(&lambda_n, &lambda_n32);
    IntScalar<int32_t> y_n = (*y)[n];
    add(&lambda_n32, &y_n, &lambda_n32);

    /* new few lines do x[:n+1] += (lambda_n / epsilon) * b[-(n+1):] */
    IntScalar<int32_t> lambda_n_over_epsilon;
    divide(&lambda_n32, &epsilon, &lambda_n_over_epsilon);
    add_scaled_special(n + 1, &lambda_n_over_epsilon,
                       &b, (N+1) - (n+1),
                       x, 0);
  }
}


void ToeplitzLpcEstimator::InitEta(int eta_inv_int) {
  int N = config_.lpc_order, B = config_.block_size;

  {  /* set eta = 1 - 1/eta_inv  =  (eta_inv - 1) / eta_inv. */
    assert(eta_inv_int > 1);
    IntScalar<int32_t> eta_inv_minus_one(eta_inv_int - 1),
        eta_inv(eta_inv_int);
    divide(&eta_inv_minus_one, &eta_inv, &eta_);
  }


  /* They'll contain
       eta_even_powers_[-n] = eta ** (2*n)
       eta_odd_powers_[-n] = eta ** (2*n + 1)
     The highest power we'll need is
       eta_odd_powers_[-(N+B)] = eta ** (2*(N+B) + 1)
  */
  int max_power = 2 * (N + B)  +  1;

  /* The p'th power of eta will be in eta_powers[p-1] */
  IntVec<int32_t> eta_powers(max_power);
  init_vec_as_powers(&eta_, &eta_powers);

  assert(eta_powers.exponent >= -31);

  eta_2B_ = eta_powers[2*B - 1];


  eta_even_powers_.resize(N + B);
  eta_odd_powers_.resize(N + B);
  for (int n = 1; n <= N + B; n++) {
    int eta_even_power = 2 * n,
        eta_odd_power = 2 * n + 1;
    eta_even_powers_.data[eta_even_powers_.dim - n] =
        eta_powers.data[eta_even_power - 1];
    eta_odd_powers_.data[eta_odd_powers_.dim - n] =
        eta_powers.data[eta_odd_power - 1];
  }
  eta_even_powers_.exponent = eta_powers.exponent;
  eta_even_powers_.nrsb = lrsb(eta_even_powers_.data[eta_even_powers_.dim - 1]);
  eta_odd_powers_.exponent = eta_powers.exponent;
  eta_odd_powers_.nrsb = lrsb(eta_odd_powers_.data[eta_odd_powers_.dim - 1]);


  reflection_coeffs_.resize(N);
  for (int k = 0; k < N; k++) {
    /* The power of eta we want is k+1; this is stored in the
       (k+1)-1 = k'th element of eta_powers. */
    reflection_coeffs_.data[k] = eta_powers.data[(k+1) - 1];
  }
  /* Make the exponent smaller by one to account for the factor of 0.5.
     We want
       reflection_coeffs_[k] = 0.5 * (eta ** (k+1)).
   */
  reflection_coeffs_.exponent = eta_powers.exponent - 1;
  reflection_coeffs_.nrsb = lrsb(reflection_coeffs_.data[0]);
}


void ToeplitzLpcEstimator::AcceptBlock(
    int parity, const int16_t *x, const int32_t *residual) {
  int lpc_order = config_.lpc_order, block_size = config_.block_size,
      other_parity = !parity,
      x_nrsb = array_lrsb(x - lpc_order, lpc_order + block_size);
  /* The following sets autocorr_[parity] */
  UpdateAutocorrStats(parity, x, x_nrsb);
  /* The following sets deriv_ */
  ComputeDeriv(parity, x, x_nrsb, residual);
  /* The following sets autocorr_final_ to the reflection term. */
  GetAutocorrReflected(x);
  add(&autocorr_[parity], &autocorr_final_);
  ApplyAutocorrSmoothing();
  /* The following does:
        temp32_b_ := toeplitz_solve(autocorr_final_, deriv_)
     In the NumPy code in ../test/linear_prediction.py the
     next few lines were:
     self.lpc_coeffs += toeplitz_solve(au, self.deriv)
  */
  toeplitz_solve(&autocorr_final_, &deriv_, &temp32_a_, &lpc_coeffs_[parity]);
  /* next line is: lpc_coeffs_[parity] += lpc_coeffs_[other_parity] */
  add(&lpc_coeffs_[other_parity], &lpc_coeffs_[parity]);


  /*
    We need to guarantee that the exponent won't be negative, to avoid having to
    introduce additional complexity into the residual computation.  The analysis
    to do that is a little complex, so we just test it and left-shift if needed.
    This should never fail (proof would involve the diag_smoothing constant, which
    of course would have to be >= -32 and probably some number strictly greater
    than -32).
  */
  if (lpc_coeffs_[parity].exponent > 0) {
    assert(!(lpc_coeffs_[parity].exponent > lpc_coeffs_[parity].nrsb) &&
           "LPC coefficients should not be able to get this large!");
    for (int i = 0; i < config_.lpc_order; i++)
      lpc_coeffs_[parity].data[i] <<= lpc_coeffs_[parity].exponent;
    lpc_coeffs_[parity].exponent = 0;
  }

}

void ToeplitzLpcEstimator::ApplyAutocorrSmoothing() {
  IntScalar<int32_t> autocorr_0 = autocorr_final_[0];
  IntScalar<int32_t> temp;
  multiply(&autocorr_0, &diag_smoothing_, &temp);
  add(&autocorr_0, &temp, &autocorr_0);
  add(&autocorr_0, &abs_smoothing_, &autocorr_0);
  set_elem_to(&autocorr_0, 0, &autocorr_final_);
}


inline const int32_t *ToeplitzLpcEstimator::GetEtaPowersStartingAt(int n) const {
  assert(n - config_.block_size > 1 && eta_odd_powers_.dim == eta_even_powers_.dim);
  int start_index = eta_odd_powers_.dim - (n / 2);
  assert(start_index >= 0);
  if (n % 2 == 1) {
    return eta_odd_powers_.data + start_index;
  } else {
    return eta_even_powers_.data + start_index;
  }
}

void ToeplitzLpcEstimator::UpdateAutocorrStats(
    int parity, const int16_t *x, int x_nrsb) {
  int other_parity = (~parity & 1);

  int N = config_.lpc_order, B = config_.block_size;
  /* Python code for updating autocorrelation could be written
     as follows (Note, block-size B == S-N in the python code)

        self.autocorr *= self.eta ** (T_diff * 2)
        for n in range(N):
           self.autocorr[n] += np.dot(x[N-n:S-n] * x[N:S], self._get_scale(B)) * self._get_eta_power(n)
        .. could be written as:
          self.autocorr[n] += np.dot(x[N-n:S-n] * x[N:S], self.eta ** (2*B + n  - 2*np.arange(B)))

       The quantity (self.eta ** (2*(B)+n  - 2*np.arange(B))) we get from
         GetEtaPowersStartingAt(2*B+n).
       Caution: our pointer "x" points to what would be element N of the "x" in the
       Python code.
  */

  /* right_shift_needed is how much we need to right-shift the products
       x[i] * x[i] * eta_power_[j]
     before summation, in order to ensure that the sum is guaranteed to be
     still representable as int64_t.

     This is worked out in some detail, and tested, in test_raw_triple_product_a_nrsb()
     in int_math_utils_test.cc; search for `right_shift_needed`
  */
  int B_extra_bits = extra_bits_from_factor_of(B),
      autocorr_right_shift_needed = B_extra_bits - 2 - (2 * x_nrsb);

  /* nrsb will be the smallest observed number of redundant sign bits in any
     element of temp64.data[]. */
  int nrsb = 64;
  if (autocorr_right_shift_needed <= 0) {
    for (int n = 0; n < N; n++) {
      int64_t sum = raw_triple_product_a(B, x - n, x,
                                         GetEtaPowersStartingAt(2*B + n));
      temp64_.data[n] = sum;
      nrsb = int_math_min(nrsb, lrsb(sum));
    }
    temp64_.exponent = -31;
    assert(nrsb >= -autocorr_right_shift_needed);
  } else {
    for (int n = 0; n < N; n++) {
      int64_t sum = raw_triple_product_a_shifted(B, x - n, x,
                                                 GetEtaPowersStartingAt(2*B + n),
                                                 autocorr_right_shift_needed);
      temp64_.data[n] = sum;
      nrsb = int_math_min(nrsb, lrsb(sum));
    }
    temp64_.exponent = -31 + autocorr_right_shift_needed;
  }
  assert(eta_odd_powers_.exponent == -31 && eta_even_powers_.exponent == -31);
  temp64_.set_nrsb(nrsb);

  copy(&temp64_, &autocorr_[parity]);
  add_scaled(&eta_2B_, &autocorr_[other_parity], &autocorr_[parity]);
}


void ToeplitzLpcEstimator::ComputeDeriv(
    int parity, const int16_t *x, int x_nrsb,
    const int32_t *residual) {
  int N = config_.lpc_order, B = config_.block_size,
      B_extra_bits = extra_bits_from_factor_of(B);

  /* Python code would be:
     for n in range(N):
       self.deriv[n] = np.dot(x[N-1-n:S-1-n] * residual,  self._get_scale(B))
     Note: our `x` pointer points to `x[N]` in the Python code.
  */

  /* We'll explain the difference of this expression vs. autocorr_right_shift_needed.
     Firstly, it contains - 1 instead of - 2 because `residual` can be as large
     as +- (2^32 - 1), so it has one more significant bit than x values.
     Secondly, we have x_nrsb instead of (2 * x_nrsb) because x only appears
     once in that expression.
     Imagine that there was a phantom "- residual_nrsb" there, but right
     now we are omitting it because we want to see whether we can get away
     without calculating residual_nrsb.
  */
  int deriv_right_shift_needed = B_extra_bits - 1 - x_nrsb;
  if (deriv_right_shift_needed > 0) {
    /* compute a more exact version of deriv_right_shift_needed, as it might
       affect the result.. */
    int residual_nrsb = array_lrsb(residual, B);
    /* The easiest way to see that the following is correct is to compare with
       the expression for autocorr_right_shift_needed in UpdateAutocorrStats(),
       with (residual_nrsb - 16) standing in for the other x_nrsb.  (You can see
       that that's right for numbers that actually do fit within int16_t, and
       extrapolate to others).
    */
    deriv_right_shift_needed = extra_bits_from_factor_of(B)
        - 2 - x_nrsb - (residual_nrsb - 16);
  }

  /* nrsb will be the smallest observed number of redundant sign bits in
     any element of temp64.data[]. */
  int nrsb = 64;

  if (deriv_right_shift_needed <= 0) {
    deriv_right_shift_needed = 0;
    for (int n = 0; n < N; n++) {
      int64_t sum = raw_triple_product_b(B, x - 1 - n, residual,
                                         GetEtaPowersStartingAt(2*B));
      temp64_.data[n] = sum;
      nrsb = int_math_min(nrsb, lrsb(sum));
    }
  } else {
    for (int n = 0; n < N; n++) {
      int64_t sum = raw_triple_product_b_shifted(B, x - 1 - n, residual,
                                                 GetEtaPowersStartingAt(2*B),
                                                 deriv_right_shift_needed);
      temp64_.data[n] = sum;
      nrsb = int_math_min(nrsb, lrsb(sum));
    }
  }
  /* The -31 comes from how the eta powers are represented, i.e.
     . */
  temp64_.exponent = eta_odd_powers_.exponent /* -31 */
      + deriv_right_shift_needed;
  temp64_.nrsb = nrsb;
  copy(&temp64_, &deriv_);
}


void ToeplitzLpcEstimator::GetAutocorrReflected(const int16_t *x) {
  /*
    Python code could be:
      for k in range(1, lpc_order):
         ans[k] = 0.5 * np.dot(self.x[-k:], np.flip(self.x[-k:])) * self._get_eta_power(k+1)
  */
  int N = config_.lpc_order, B = config_.block_size;
  temp64_.data[0] = 0;
  int nrsb = 64;
  for (int k = 1; k < N; k++) {
    /* sum is np.dot(self.x[-k:], np.flip(self.x[-k:]));
       note, the element one past the end of our array here is x + B.
     */
    int64_t sum = compute_raw_dot_product<int16_t, int16_t, int64_t, int64_t, -1>(
        x + B - k, x + B - 1, k);
    temp64_.data[k] = sum;
    nrsb = int_math_min(nrsb, lrsb(sum));
  }
  temp64_.exponent = 0;
  temp64_.set_nrsb(nrsb);
  temp64_.check();
  copy(&temp64_, &temp32_a_);
  /* elementwise multiply.  reflection_coeffs_[k] contains 0.5 * (eta **
   * (k+1)). */
  multiply(&temp32_a_, &reflection_coeffs_, &autocorr_final_);
}


}


