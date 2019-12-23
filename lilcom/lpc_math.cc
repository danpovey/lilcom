#include "lpc_math.h"
#include "stdio.h"
#include "stdlib.h"
#include <assert.h>


void ToeplitzLpcEstimator::AcceptBlock(
    int parity, const int16_t *x, const int32_t *residual) {
  UpdateAutocorrStatsAndDeriv(parity, x, residual);

}

inline const int32_t *ToeplitzLpcEstimator::GetEtaPowersStartingAt(int n) const {
  assert(n - block_size_ > 1 && eta_odd_powers_.dim == eta_evan_powers_.dim);
  int start_index = eta_odd_powers_.dim - (n / 2);
  assert(start_index >= 0);
  if (n % 2 == 1) {
    return eta_odd_powers_.data + start_index;
  } else {
    return eta_even_powers_.data + start_index;
  }
}

void ToeplitzLpcEstimator::UpdateAutocorrStatsAndDeriv(
    int parity, const int16_t *x, const int32_t *residual) {
  int other_parity = (~parity & 1);

  int N = lpc_order_, B = block_size_;
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
  int nrsb = 64;
  for (int n = 0; n < N; n++) {
    int64_t sum = raw_triple_product_a(B, x - n, x,
                                       GetEtaPowersStartingAt(2*B + n));
    temp64_.data[n] = sum;
    nrsb = min(nrsb, lrsb(sum));
  }
  assert(eta_odd_powers_.exponent == -31 && eta_even_powers_.exponent == -31);
  temp64_.exponent = -31;
  temp64_.nrsb = nrsb;
  copy(&temp64_, &autocorr_[parity]);
  add_scaled(&eta_2B_, &autocorr_[other_parity], &autocorr_[parity]);


  /* Now update deriv_ */

  /* Python code would be:
     for n in range(N):
       self.deriv[n] = np.dot(x[N-1-n:S-1-n] * residual,  self._get_scale(B))
     Note: our `x` pointer points to `x[N]` in the Python code.
  */
  int nrsb = 64;
  for (int n = 0; n < N; n++) {
    int64_t sum = raw_triple_product_b(B, x - 1 - n, residual,
                                       GetEtaPowersStartingAt(2*B));
    temp64_.data[n] = sum;
    nrsb = min(nrsb, lrsb(sum));
  }
  temp64_.exponent = -31;
  temp64_.nrsb = nrsb;
  copy(&temp64_, &deriv_);
}


void ToeplitzLpcEstimator::GetAutocorrReflected(const int16_t *x) {

  /*
    Python code could be:
      for k in range(1, lpc_order):
         ans[k] = 0.5 * np.dot(self.x[-k:], np.flip(self.x[-k:])) * self._get_eta_power(k+1)
  */


  int N = lpc_order_, B = block_size_;
  temp64_.data[0] = 0;
  int nrsb = 64;
  for (int k = 1; k < N; k++) {
    /* sum is np.dot(self.x[-k:], np.flip(self.x[-k:]));
       note, the element one past the end of our array here is x + B.
     */
    int64_t sum = compute_raw_dot_product<int16_t, int16_t, int64_t, -1>(
        k, x + B - k, x + B - 1);
    temp64_.data[k] = sum;
    nrsb = min(nrsb, lrsb(sum));
  }
  temp64_.exponent = -31;
  temp64_.nrsb = nrsb;
  copy(&temp64_, &temp32_);
  /* elementwise multiply.  reflection_coeffs_[k] contains 0.5 * (eta **
   * (k+1)). */
  multiply(&temp32_, &reflection_coeffs_, &autocorr_final_);

}




}


#ifdef LPC_MATH_TEST
#endif /* FIXED_MATH_TEST */
