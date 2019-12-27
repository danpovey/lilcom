#include "int_scalar.h"
#include "int_vec.h"
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

  std::cout << "x = " << x;
}


}

int main() {
  using namespace int_math;
  test_toeplitz_solve_compare();
}
