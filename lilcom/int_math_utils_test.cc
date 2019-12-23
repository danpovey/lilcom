#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include <math.h>
#include "int_math_utils.h"

namespace int_math {

}



int main() {
  using namespace int_math;

  assert(int_math_min(2, 3) == 2);

  assert(lrsb(int16_t(1)) == 14);
  assert(lrsb(int32_t(1)) == 30);
  assert(lrsb(int32_t(0)) == 31);
  assert(lrsb(int32_t(-1)) == 32);
  assert(lrsb(int64_t(-1)) == 64);


  test_constructor<int32_t>();
  test_constructor<int64_t>();
  return 0;
}


