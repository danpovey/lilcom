#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_vec.h"
#include <math.h>


namespace int_math {

void test_constructor() {
  IntVec<int32_t> a;
  assert(a.dim == 0);
  IntVec<int32_t> b(4);
  assert(b.dim == 0 && data_is_zero(b.data, b.dim) &&
         b.exponent == kExponentOfZero && b.nrsb == 31);
  IntVec<int64_t> c(4);
  assert(c.dim == 0 && data_is_zero(c.data, c.dim) &&
         c.exponent == kExponentOfZero && c.nrsb == 60);
}

}

int main() {
  using namespace int_math;
  test_constructor();
}

