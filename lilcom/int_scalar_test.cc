#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_scalar.h"
#include <math.h>

namespace int_math {

template <typename I>
void test_constructor() {
  for (int i = 0; i < 31; i++) {
    IntScalar<I> a(1 << i),
        b(1, i);
    float f = pow(2.0, i);
    assert(f == (float)a && f == (float)b);
  }
}


}



int main() {
  using namespace int_math;
  test_constructor<int32_t>();
  test_constructor<int64_t>();
  return 0;
}


