#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_math.h"
#ifdef INT_MATH_TEST
#include <math.h>
#endif




#ifndef NDEBUG
template <typename I> void IntVec::check() {
  int recomputed_nrsb = get_nrsb(data, dim);
  assert(nrsb == recomputed_nrsb);
  if (nrsb == sizeof(I)*8 - 1 && is_zero(data, dim)) {
    assert(exponent == -1000);
  } else {
    assert(exponent != -1000);
  }
}
#endif

#ifdef INT_MATH_TEST



#endif
