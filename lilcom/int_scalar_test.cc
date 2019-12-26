#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_scalar.h"
#include <math.h>

namespace int_math {

/*
  Pseudo-randomly initializes an IntScalar.
 */
template <typename I> void init_scalar(int i,
                                       IntScalar<I> *s) {
  /* exponent is random between -50 and 50 */
  s->exponent = ((i * 33333) % 100) - 50;

  int B = sizeof(I) * 8;

  switch (i % 6) {
    case 0:
      s->elem = 0;
      break;
    case 1:
      s->elem = (i * 103451 * 100313);  /* those are two primes.. aiming to
                                         * overflow and get negatives. */
      break;
    case 2:
      s->elem = 1 << (i % B);
      break;
    case 3:
      s->elem = (1 << (i % B)) - 1;
      break;
    case 4:
      s->elem = -(1 << (i % B));
      break;
    case 5:
      s->elem = -((1 << (i % B)) - 1);
      break;
  }
}


template <typename I>
void test_constructor() {
  for (int i = 0; i < 31; i++) {
    IntScalar<I> a(1 << i),
        b(1, i);
    float f = pow(2.0, i);
    assert(f == (float)a && f == (float)b);
  }

  for (int i = -100; i < 100; i++) {
    IntScalar<I> is(-34, i);
    assert((float)is == -34 * pow(2, i));
  }
}

template <typename I>
void test_divide() {
  for (int r = 0; r < 4000; r++) {
    IntScalar<I> i, j;
    init_scalar(r, &i);
    init_scalar(r * 73079, &j);
    if ((double) j != 0.0) {
      double ratio = (double)i / (double)j;
      IntScalar<I> k;
      divide(&i, &j, &k);
      if ((double)i == 0.0) {
        assert((double)k == 0.0);
      } else {
        double rel_error = fabs(ratio - (double)k) / ratio;
        assert(rel_error < 1.0e-05);
      }
    }
  }
}

template <typename I>
void test_multiply() {
  for (int r = 0; r < 4000; r++) {
    IntScalar<I> i, j;
    init_scalar(r, &i);
    init_scalar(r * 73079, &j);
    if ((double) j != 0.0) {
      double product = (double)i * (double)j;
      IntScalar<I> k;
      multiply(&i, &j, &k);
      if ((double)i == 0.0 || (double)j == 0.0) {
        assert((double)k == 0.0);
      } else {
        double rel_error = fabs(product - (double)k) / product;
        assert(rel_error < 1.0e-05);
      }
    }
  }
}


template <typename I>
void test_add() {
  for (int r = 0; r < 4000; r++) {
    IntScalar<I> i, j;
    init_scalar(r, &i);
    init_scalar(r * 73079, &j);
    if ((double) j != 0.0) {
      IntScalar<I> k, l;
      add(&i, &j, &k);
      add(&i, &i, &l);

      double i_f = i, jf = j, kf = k, lf = l;

      printf("i_f = %f, jf = %f, kf = %f, lf = %f\n",
             (float)i_f, (float)jf, (float)kf, (float)lf);

      if (i_f != -jf) {
        double rel_error = (i_f + jf  - kf) / (i_f + jf);
        assert(rel_error < 1.0e-05);
      }
      if (i_f != 0.0) {
        double rel_error = (i_f+i_f - lf) / (i_f + i_f);
        assert(rel_error < 1.0e-05);
      }
    }
  }
}



void test_convert() {
  for (int r = 0; r < 4000; r++) {
    IntScalar<int64_t> i;
    init_scalar(r, &i);
    IntScalar<int32_t> j;
    IntScalar<int64_t> k;
    copy(&i, &j);
    copy(&j, &k);

    double d1 = i, d2 = j, d3 = k;
    if (d1 == 0) {
      assert(d2 == 0 && d3 == 0);
    } else {
      float rel_error1 = fabs((d1-d2)/d1),
          rel_error2 = fabs((d1-d3)/d1);
      assert(rel_error1 < 1.0e-05);
      assert(rel_error2 < 1.0e-05);
    }
  }
}


template <typename T>
void test_negate() {
  for (int r = 0; r < 4000; r++) {
    IntScalar<T> i;
    init_scalar(r, &i);
    IntScalar<T> j(i);
    negate(&j);
    IntScalar<T> k(j);
    negate(&k);
    if ((float)i == 0.0) {
      assert((float)j == 0.0 && (float)k == 0.0);
    } else {
      float fi = i, fj = j, fk = k;
      float rel_error1 = fabs((fi-fk)/fi),
          rel_error2 = fabs((fi+fj)/fi);
      assert(rel_error1 < 1.0e-05);
      assert(rel_error2 < 1.0e-05);
    }
  }
}


}  // namespce int_math



int main() {
  using namespace int_math;
  test_constructor<int32_t>();
  test_constructor<int64_t>();
  test_divide<int32_t>();
  test_convert();
  test_negate<int32_t>();
  test_negate<int64_t>();
  test_multiply<int32_t>();
  test_add<int32_t>();
  test_add<int64_t>();

  return 0;
}


