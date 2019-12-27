#ifndef __LILCOM__LPC_MATH_H__
#define __LILCOM__LPC_MATH_H__


#include "int_math_utils.h"
#include "int_scalar.h"
#include "int_vec.h"

namespace int_math {


/*
    Let y be a vector of dimension N and let
    `autocorr` be vector of dimension N representing, conceptually,
    an NxN Toeplitz matrix A with elements A(i,j) = autocorr[abs(i-j)].

    This function solves the linear system A x = y, giving x.

    We require for the Toeplitz matrix to satisfy the usual conditions for
    algorithms on Toeplitz matrices, meaning no singular leading minor may be
    singular (i.e. not det(A[0:n,0:n])==0 for any n).  This will naturally be
    satisfied if A is the autocorrelation of a finite nonzero sequence (I
    believe).

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
    2.10.  There is opportunity for simplification because this Toeplitz matrix
    is symmetric.

      @param [in] autocorr    The input autocorrelation coefficients.  All
                              args to this function must have the same
                              dimension.
      @param [in] y           The vector y in A x = y, where
                              A[i,j] = autocorr[abs(i-j)]
      @param     temp         Temporary vector used in this function
      @param [out] x          The quantity being solved for
 */
void toeplitz_solve(const IntVec<int32_t> *autocorr,
                    const IntVec<int32_t> *y,
                    IntVec<int32_t> *temp,
                    IntVec<int32_t> *x);

/* Custom exception type */
class InvalidParamsError: public std::runtime_error {
 public:
  InvalidParamsError(const std::string &str): runtime_error(str) { }
};

class ToeplitzLpcEstimator {
 public:
  /* Constructor raises InvalidParamsError if eta_inv is too small
     compared to the block size. */
  ToeplitzLpcEstimator(int lpc_order,
                       int block_size,
                       int eta_inv,
                       int diag_smoothing_power,
                       int abs_smoothing_power):
      lpc_order_(lpc_order),
      temp64_(lpc_order),
      temp32_a_(lpc_order),
      temp32_b_(lpc_order) {

    /* The following will zero these values, which is all the initialization
       we need. */
    autocorr_[0].resize(lpc_order);
    autocorr_[1].resize(lpc_order);
    lpc_coeffs_[0].resize(lpc_order);
    lpc_coeffs_[1].resize(lpc_order);

    assert(diag_smoothing_power > 0 && abs_smoothing_power > 0);
    init_as_power_of_two(-diag_smoothing_power, &diag_smoothing_);
    init_as_power_of_two(-abs_smoothing_power, &abs_smoothing_);

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


  /**
     Sets eta_ to 1.0 - 1/eta_inv, and sets up eta_even_powers_,
     eta_odd_powers_ and eta_2B_
   */
  void InitEta(int eta_inv);


  /*  Updates autocorr_[parity] and deriv_.  Similar to
      ToeplitzLpcEstimator._update_autocorr_and_deriv in
      ../test/linear_prediction.py
      x_nrsb is the smallest number of redudant sign bits
      for any x[-lpc_order_].. through x[block_size_ - 1].
   */
  void UpdateAutocorrStats(
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


  /* temp_ is a vector of dimension lpc_order_, used for the autocorrelation
     accumulations*/
  IntVec<int64_t> temp64_;
  /* temp32_a_ and temp32_b_ are two other temporary vectors of dimension
     lpc_order_. */
  IntVec<int32_t> temp32_a_;
  IntVec<int32_t> temp32_b_;

  /*
    reflection_coeffs_ is of dimension lpc_order_ and element k contains 0.5 *
    (eta ** (k+1)).  It appears in the formula for the signal-reflection part of
    the autocorrelation stats.
   */
  IntVec<int32_t> reflection_coeffs_;

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
     requires it.

     For block indexed b, autocorr_[b%2] contains block b's
     stats as the most recent stats (and previous ones
     included in its weighted sum).
  */
  IntVec<int32_t> autocorr_[2];


  /* deriv_ is a function of the residual and the signal.  dim is lpc_order_. */
  IntVec<int32_t> deriv_;

  /* autocorr_final_ is autocorr_[parity] plus reflection terms and
     smoothing. dim is lpc_order_.  This is used temporarily when
     AcceptBlock() is called,
  */
  IntVec<int32_t> autocorr_final_;

  /* lpc_coeffs_[parity] is the LPC coeffs estimated from stats including those
     of the the most recent block with parity `parity`. dim is lpc_order_.
     When doing the prediction for any given block, we always use the lpc
     stats for the block of the *other* parity, as to avoid transmitting
     the LPC coeffs we always estimate them from previous samples.
  */
  IntVec<int32_t> lpc_coeffs_[2];


  /* eta_2B_ is eta_ to the power 2 * B_. */
  IntScalar<int32_t> eta_2B_;


};


} // namespace int_math

#endif /* include guard */

