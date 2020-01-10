#ifndef __LILCOM__LPC_MATH_H__
#define __LILCOM__LPC_MATH_H__


#include "int_math_utils.h"
#include "int_scalar.h"
#include "int_vec.h"
#include <stdexcept>

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


/* Configuration class for linear prediction. */
struct LpcConfig {
  /*
         @param [in]  lpc_order   The LPC order, i.e. the number of LPC
                          coefficients we are estimating.  E.g. somewhere
                          in the range [4, 32].  Larger values will give better
                          compression but will be slower, both when compressing
                          and decompressing.
         @param [in]  block_size  The block size, determines how frequently
                          the LPC coefficients are updated.   Should probably
                          not be less than the LPC order, as that could make
                          things significantly slower.  MUST BE EVEN (this
                          enables some optimization in the internal code), and
                          must be at least half lpc_order.
         @param [in]  eta_inv  Constant that determines the per-sample
                          'forgetting factor' eta (which affects how
                          fast the LPC parameters will change).  eta will equal
                          (1-eta_inv)/eta_inv.   eta_inv should be
                          at least 3 * lpc_order.  This
                          code will fail with an assertion if eta_inv is
                          too small relative to lpc_order (although
                          the condition it checks is less strict, more
                          like 2.88 * lpc_order).
         @param [in]  diag_smoothing_power  This number, raised to a power of
                          2, will be a smoothing constant for the 0'th
                          autocorrelation coefficient (to ensure invertibility
                          of the Toeplitz matrix)...
                            autocorr[0] := autocorr[0] * 2**diag_smoothing_power +
                                           2 ** abs_smoothing_power
                          Suitable value: -23, corresponding to a coefficient
                          of around 10^-7.
         @param [in]  abs_smoothing_power  Number that determines smoothing
                          of autocorr[0], relevant only for zero signals.
                          E.g. suggest -33, corresponding to 10^-10.
  */
  LpcConfig(int lpc_order,
            int block_size,
            int eta_inv,
            int diag_smoothing_power,
            int abs_smoothing_power):
      lpc_order(lpc_order),
      block_size(block_size),
      eta_inv(eta_inv),
      diag_smoothing_power(diag_smoothing_power),
      abs_smoothing_power(abs_smoothing_power) {
  }
  LpcConfig(): lpc_order(16), block_size(32), eta_inv(128),
               diag_smoothing_power(-10), abs_smoothing_power(-33) { }

  LpcConfig(const LpcConfig &other):
      lpc_order(other.lpc_order), block_size(other.block_size),
      eta_inv(other.eta_inv),
      diag_smoothing_power(other.diag_smoothing_power),
      abs_smoothing_power(other.abs_smoothing_power) { }
  bool IsValid() {
    return (block_size % 2 == 0 &&
            eta_inv >= 3 * lpc_order &&
            diag_smoothing_power < 0 &&
            abs_smoothing_power < 0);
  }
  int lpc_order;
  int block_size;
  int eta_inv;
  int diag_smoothing_power;
  int abs_smoothing_power;
};


class ToeplitzLpcEstimator {
 public:
  ToeplitzLpcEstimator(const LpcConfig &config):
      config_(config),
      temp64_(config.lpc_order),
      temp32_a_(config.lpc_order),
      temp32_b_(config.lpc_order),
      autocorr_(config.lpc_order),
      deriv_(config.lpc_order),
      autocorr_final_(config.lpc_order),
      lpc_coeffs_(config.lpc_order) {
    std::cout << "diag_smoothing is " << config.diag_smoothing_power <<"\n";
    init_as_power_of_two(config.diag_smoothing_power, &diag_smoothing_);
    init_as_power_of_two(config.abs_smoothing_power, &abs_smoothing_);
    InitEta(config.eta_inv);
  }


  /**
     The user should call this function to update the stats in this object and
     do one iteration of update.  The LPC coefficients in lpc_coeffs_ will be
     updated, as well as the autocorrelation stats in autocorr_.  (Other members
     of this class which are updated inside here won't be read in future, so can
     be considered as temporaries).

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
                       obtained from GetLpcCoeffs();

      This code is similar to ToeplitzLpcEstimator.accept_block in
      ../test/linear_prediction.py
   */
  void AcceptBlock(const int16_t *x, const int32_t *residual);

  /*
    Gets the current LPC coefficients.

    We guarantee that the exponent `lpc_coeffs_.exponent` will not be positive.
    (This is required by the code that computes the residual, so it can avoid
    testing the sign of the exponent).
   */
  inline const IntVec<int32_t> &GetLpcCoeffs() const { return lpc_coeffs_; }

  ~ToeplitzLpcEstimator() { }

  const LpcConfig &Config() const { return config_; }

#ifndef LPC_MATH_TEST
  private:
#endif

  /**
     Sets eta_ to 1.0 - 1/eta_inv, and sets up eta_even_powers_,
     eta_odd_powers_ and eta_2B_
   */
  void InitEta(int eta_inv);


  /*  Updates autocorr_ and deriv_.  Similar to
      ToeplitzLpcEstimator._update_autocorr_and_deriv in
      ../test/linear_prediction.py
      x_nrsb is the smallest number of redudant sign bits
      for any x[-lpc_order_].. through x[block_size_ - 1].
   */
  void UpdateAutocorrStats(const int16_t *x, int x_nrsb);

  /*
    Computes the objective function derivative (the new part, from
    the new block)... the assumption is that the old part of the
    deriv is now zero because we `trust` the previous iteration's
    solver.
   */
  void ComputeDeriv(const int16_t *x, int x_nrsb,
                    const int32_t *residual);

  /* Sets autocorr_final_ to the reflection term in the autocorrelation
     coefficients.  See ../test/linear_prediction.py for more details
     (or the writeup..)

      @param [in] x  The signal; the same as the `x` pointer given to AcceptBlock().
  */
  void GetAutocorrReflected(const int16_t *x);


  /*
    Adds smoothing terms to autocorr_final_: essentially,
      autocorr_final_ +=
         abs_smoothing_ + diag_smoothing_ * autocorr_final_
   */
  void ApplyAutocorrSmoothing();

  LpcConfig config_;
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
  */
  IntVec<int32_t> autocorr_;


  /* deriv_ is a function of the residual and the signal.  dim is lpc_order_. */
  IntVec<int32_t> deriv_;

  /* autocorr_final_ is autocorr_ plus reflection terms and smoothing.  Its
     dimension is lpc_order_.  This is used temporarily when AcceptBlock() is
     called,
  */
  IntVec<int32_t> autocorr_final_;

  /* lpc_coeffs_ contains the most recently estimated LPC coefficients.  The
     dimension is config_.lpc_order.
  */
  IntVec<int32_t> lpc_coeffs_;

  /* eta_2B_ is eta_ to the power 2 * B_. */
  IntScalar<int32_t> eta_2B_;


};


} // namespace int_math

#endif /* include guard */

