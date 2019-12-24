#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_vec.h"
#include <math.h>
#include <iostream>


namespace int_math {


/*
  Pseudo-randomly initializes the contents of an IntVec
 */
template <typename I> void init_vec(int i,  /* i is the source of randomness */
                                    IntVec<I> *s) {
  /* exponent is random between -50 and 50 */
  s->exponent = ((i * 33333) % 100) - 50;

  int B = sizeof(I) * 8;

  switch (i % 6) {
    case 0:
      // All zeros
      for (int j = 0; j < s->dim; j++)
        s->data[j] = 0;
      break;
    case 1:
      for (int j = 0; j < s->dim; j++) {
        s->data[j] = (i * (j+1) * 103451 * 100313);  /* those are two primes.. aiming to
                                                     * overflow and get negatives. */
      }
      break;
    case 2:
      for (int j = 0; j < s->dim; j++) {
        s->data[j] = 1 << (((i+100-j)*7) % B);
      }
      break;
    case 3:
      for (int j = 0; j < s->dim; j++) {
        s->data[j] = (1 << ((i+j) % B)) - 1;
      }
      break;
    case 4:
      for (int j = 0; j < s->dim; j++) {
        s->data[j] = -(1 << ((i+j) % B));
      }
      break;
    case 5:
      for (int j = 0; j < s->dim; j++) {
        s->data[j] = -((1 << ((i+j) % B)) - 1);
      }
      break;
  }
  s->set_nrsb();
}




void test_constructor() {
  IntVec<int32_t> a;
  assert(a.dim == 0);
  IntVec<int32_t> b(4);
  assert(b.dim == 4 && data_is_zero(b.data, b.dim) &&
         b.exponent == kExponentOfZero && b.nrsb == 31);
  IntVec<int64_t> c(5);
  assert(c.dim == 5 && data_is_zero(c.data, c.dim) &&
         c.exponent == kExponentOfZero && c.nrsb == 63);
  assert( (float)c[4] == 0);
  a.check();
  b.check();
  c.check();

  c.set_nrsb(63);
  c.set_nrsb(61);
  c.data[0] = 1;
  c.set_nrsb(63);

  c.resize(6);
  assert(c.dim == 6 && c.data[5] == 0);
  // c.check();  /* Should fail. */
}


template <typename I, typename J, typename K>
void test_compute_dot_product() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<I> v(dim);
    IntVec<J> w(dim);
    IntScalar<K> d, e;
    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        init_vec(i1, &v);
        init_vec(i2, &w);
        compute_dot_product(&v, &w, &d);
        compute_dot_product(&v, &w, &e);
        /* d and e should be identical. */
        assert(d.exponent == e.exponent && d.elem == e.elem);
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
          sum += (double)v[j] * (double)w[j];
        }
        double df = (double)d;
        /* std::cout << "Dot product of " << (std::string)v
                  << " and " << (std::string) w << " is " << df
                  << ", sum is " << sum << "... "
                  << ", nrsb are " << v.nrsb << ", " << w.nrsb
                  << ", ints are " << v.data[0] << " and " << w.data[0]
                  << ", output int is " << d.elem; */
        if (sum == 0.0) {
          assert(df == 0.0);
        } else {
          double rel_error = (df - sum) / sum;
          /* std::cout << "Rel error is " << rel_error << "\n"; */
          assert(fabs(rel_error) < 1.0e-05);
        }
      }
    }
  }
}


void test_add_scaled_special() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<I> v(dim);
    IntVec<J> w(dim);
    IntScalar<K> d, e;

    for (int sub_dim = 1; sub_dim

    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        init_vec(i1, &v);
        init_vec(i2, &w);
        compute_dot_product(&v, &w, &d);
        compute_dot_product(&v, &w, &e);
        /* d and e should be identical. */
        assert(d.exponent == e.exponent && d.elem == e.elem);
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
          sum += (double)v[j] * (double)w[j];
        }
        double df = (double)d;
        /* std::cout << "Dot product of " << (std::string)v
                  << " and " << (std::string) w << " is " << df
                  << ", sum is " << sum << "... "
                  << ", nrsb are " << v.nrsb << ", " << w.nrsb
                  << ", ints are " << v.data[0] << " and " << w.data[0]
                  << ", output int is " << d.elem; */
        if (sum == 0.0) {
          assert(df == 0.0);
        } else {
          double rel_error = (df - sum) / sum;
          /* std::cout << "Rel error is " << rel_error << "\n"; */
          assert(fabs(rel_error) < 1.0e-05);
        }
      }
    }
  }
}



}

int main() {
  using namespace int_math;
  test_constructor();
  test_compute_dot_product<int32_t, int32_t, int64_t>();
  test_compute_dot_product<int32_t, int16_t, int64_t>();
}

