#include "int_scalar.h"
#include "int_vec.h"
#define LPC_MATH_TEST
#include "lpc_math.h"
#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include <iostream>



namespace int_math {

/*
  This runs the Toeplitz solver with the same values as are used in
  test_toeplitz_solve_compare() in ../test/linear_prediction.py.
 */

void test_toeplitz_solve_compare() {
  IntVec<int32_t> autocorr(4),
      y(4),
      temp(4),
      x(4);
  autocorr.data[0] = 10;
  autocorr.data[1] = 5;
  autocorr.data[2] = 2;
  autocorr.data[3] = 1;
  autocorr.exponent = 0;
  autocorr.set_nrsb();
  y.data[0] = 1;
  y.data[1] = 2;
  y.data[2] = 3;
  y.data[3] = 4;
  y.exponent = 0;
  y.set_nrsb();
  toeplitz_solve(&autocorr, &y, &temp, &x);
  /* I'm just verifying that it gives the same numbers as the Python version.
     In Python there were some more exhaustive tests, which required generating
     autocorrelations from fake data. */
  std::cout << "x = " << x;
}


/*
  Compare this with test_new_lpc_compare() in ../test/linear_prediction.py, we
  use the same input and configuration so we can check that the output is the
  same.
 */
void test_lpc_est_compare() {
  /* first four zeros are left-padding */
  int order = 4, block_size = 6, eta_inv = 128,
      diag_smoothing_power=-5, abs_smoothing_power=-3;
  int16_t signal[] = { 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 9, 11, 13, 15, 17, 18 };
  /* only the first 5 elements of the residual are known; we
     need the LPC coeffs to compute the remaining 5 elements. */
  int32_t residual[] = { 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0 };

  LpcConfig config(order, block_size, eta_inv,
                   diag_smoothing_power, abs_smoothing_power);
  ToeplitzLpcEstimator lpc(config);

  int parity = 0;
  lpc.AcceptBlock(parity, signal + order, residual);

  std::cout << "After first block, autocorr = "
            << lpc.autocorr_[0]
            << ", autocorr_final = "
            << lpc.autocorr_final_
            << ", lpc-coeffs = "
            << lpc.lpc_coeffs_[0]
            << ", deriv = "
            << lpc.deriv_;

  parity = 1;
  std::cout << "Residual2 = ";
  for (int t = block_size; t < 2 * block_size; t++) {
    int16_t predicted = compute_lpc_prediction(signal + order + t,
                                               &lpc.GetLpcCoeffsForBlock(parity));
    int32_t residual_t = signal[order + t] - static_cast<int32_t>(predicted);
    residual[t] = residual_t;
    std::cout << residual_t << ' ';
  }
  std::cout << std::endl;
  lpc.AcceptBlock(parity, signal + order + block_size, residual + block_size);

  std::cout << "After second block, autocorr = "
            << lpc.autocorr_[parity]
            << ", autocorr_final = "
            << lpc.autocorr_final_
            << ", lpc-coeffs = "
            << lpc.lpc_coeffs_[parity]
            << ", deriv = "
            << lpc.deriv_;


}



}

int main() {
  using namespace int_math;
  test_toeplitz_solve_compare();
  test_lpc_est_compare();
}
