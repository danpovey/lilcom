#ifndef __LILCOM__LPC_MATH_H__
#define __LILCOM__LPC_MATH_H__


#include "int_math_utils.h"
#include "int_scalar.h"
#include "int_vec.h"

namespace int_math {

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
    b.exponent = b_nsb - 1;
    b.data[N] = 1 << (b_nsb - 1);
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

    /* note: abs(nu_n) < 1.0.  mathematically it's <= 1.0, but we
       added a little smoothing to the zeroth autocorr element. */
    IntScalar<int32_t> nu_n;
    divide(&prod, &epsilon, &nu_n);
    assert(std::abs(static_cast<float>(nu_n)) < 1.0);
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

class ToeplitzLpcEstimator {
 public:
  ToeplitzLpcEstimator(int lpc_order,
                       int block_size,
                       int eta_inv,
                       int diag_smoothing_power,
                       int abs_smoothing_power):
      lpc_order_(lpc_order),
      diag_smoothing_power_(1, -diag_smoothing_power),
      abs_smoothing_power_(1, -abs_smoothing_power),
      temp_(lpc_order),
      eta_even_powers_(lpc_order + block_size),
      eta_odd_powers_(lpc_order + block_size),
      temp_deriv_(lpc_order) {
    autocorr_[0].resize(lpc_order);
    autocorr_[1].resize(lpc_order);
    lpc_coeffs_[0].resize(lpc_order);
    lpc_coeffs_[1].resize(lpc_order);
    InitEta(eta_inv);
    // TODO
  }


  /**
     The user should call this function to update the stats in this object and
     do one iteration of update.  The LPC coefficients in lpc_coeffs_[parity]
     will be updated.

         @param [in] parity   Must be zero or one.  The parity of the block index.
                       This is used to enable backtracking (so we can remember
                       the previous state if it becomes necessary to re-do the
                       estimation of a block)

         @param [in] x Pointer to start of the signal for this block (x in the math),
                       together with some preceding context.  Elements
                       x[-lpc_order_+1] through x[block_size_ - 1]
                       will be accessed.  Negative indexes should contain
                       the preceding context.

         @param [in] residual  Pointer to the start of the residual; elments
                       0 through block_size_ - 1 will be accessed.  This is
                       a difference of int16_t's so it has to be represented
                       as int32_t to cover the general case.  This is, of
                       course, expected to be the residual after applying
                       the previously estimated LPC coefficients, as
                       obtained from GetLpcCoeffsForBlock(parity).

      This code is similar to ToeplitzLpcEstimator.accept_block in
      ../test/linear_prediction.py
   */
  void AcceptBlock(int parity, const int16_t *x, const int32_t *residual);



  /*
    Gets the LPC coefficients that we need to apply to samples within a block
    with parity 'parity'.  This will have been estimated from the block with the
    *other* parity (since we always estimate LPC coeffs from past samples,
    to avoid having to transmit them).  Of course each block has different
    LPC coeffs, but we never backtrack further than one block, so storing
    two blocks' worth of LPC coefficients is sufficient.
   */
  inline const IntVec<int32_t> &GetLpcCoeffsForBlock(int parity) {
    return lpc_coeffs_[ (~parity) & 1];
  }

  ~ToeplitzLpcEstimator() { }
 private:

  void InitEtaPowers(int eta_inv) {
    // set eta_ to 1.0 - 1/eta_inv.
    assert(eta_inv > 1);
    IntScalar one(1), eta_inv(eta_inv);
    divide(&one, &eta_inv, &eta_);
    negate(&eta_);
    add(&eta_, &one, &eta_);
  }

  /*  Updates autocorr_[parity] and deriv_.  Similar to
      ToeplitzLpcEstimator._update_autocorr_and_deriv in
      ../test/linear_prediction.py
      x_nrsb is the smallest number of redudant sign bits
      for any x[-lpc_order_].. through x[block_size_ - 1].
   */
  void UpdateAutocorrStatsAndDeriv(
      int parity, const int16_t *x, int x_nrsb);

  void UpdateDeriv(
      int parity, const int16_t *x, int x_nrsb,
      const int32_t *residual);

  /* Sets autocorr_final_ to the reflection term in the autocorrelation
     coefficients.  See ../test/linear_prediction.py for more details
     (or the writeup..)

      @param [in] x  The signal; the same as the `x` pointer given to AcceptBlock().
  */
  void GetAutocorrReflected(const int16_t *x);


  /*
    Adds smoothing terms to autocorr_final_: essentially,
      autocorr_final_[0] +=
         abs_smoothing_ + diag_smoothing_ * autocorr_final_[0]
   */
  void ApplyAutocorrSmoothing();

  int lpc_order_;
  int block_size_;
  IntScalar<int32_t> diag_smoothing_;
  IntScalar<int32_t> abs_smoothing_;
  /* needed or not? */
  IntScalar<int32_t> eta_;


  /* for n > 0, and using Python-style negative indexes,
       eta_even_powers_[-n] = eta ** (2*n)
       eta_odd_powers_[-n] = eta ** (2*n + 1)
     they both have dimension block_size_ + lpc_order_.
  */
  IntVec<int32_t> eta_even_powers_;
  IntVec<int32_t> eta_odd_powers_;

  /* reflection_coeffs_ is a vector of dimension lpc_order_ whose
       i'th element contains 0.5 * (eta_ ** (i + 1)).
    It is initialized once and never changes.
    It is a scale/coefficient on each term in the
    `reflection part` of the autocorrelation stats.
    (the part that arises from adding the signal to its
    mirror-image)
   */
  IntVec<int32_t> reflection_coeffs_;


  /* temp_ is a vector of dimension lpc_order_, used for the autocorrelation
     accumulations*/
  IntVec<int64_t> temp64_;
  /* temp2_ is another temporary vector of dimension lpc_order_. */
  IntVec<int32_t> temp32_;

  /* eta_2B_ is eta to the power 2B where B == block_size_. */


  /* This function returns a pointer to the start of an array
     containing at least block_size_ powers of eta, specifically
     representing the following quantities:

        eta ** [ n, n-2, n-4, ... n-((block_size_-1)*2) ]

     They are represented as integers, the exponent is -31.
   */
  const int32_t *GetEtaPowersStartingAt(int n) const;

  /* raw autocorrelation stats, of dimension lpc_order.  Note:
     normally they would be of dimension lpc_order + 1, but
     our method is a little different, incorporating residuals,
     and we don't need that last one.

     This vector is indexed by `parity`, which is 0 or 1;
     we alternate on different blocks.  This allows us to
     easily revert to the previous block when the algorithm
     requirs it.
  */
  IntVec<int32_t> autocorr_[2];


  /* deriv_ is a function of the residual and the signal.  dim is lpc_order_. */
  IntVec<int32_t> deriv_;

  /* autocorr_final_ is autocorr_[parity] plus reflection terms and
     smoothing. dim is lpc_order_.  This is used temporarily when
     AcceptBlock() is called,
  */
  IntVec<int32_t> autocorr_final_;

  /* lpc_coeffs_[parity] is the LPC coeffs estimated for the most recent block with
     parity `parity`. dim is lpc_order_. */
  IntVec<int32_t> lpc_coeffs_[2];


  /* eta_2B_ is eta_ to the power 2 * B_. */
  IntScalar<int32_t> eta_2B_;


};


} // namespace int_math

#endif /* include guard */

