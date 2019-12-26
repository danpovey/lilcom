#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include <math.h>
#include "int_math_utils.h"

namespace int_math {
void test_data_is_zero() {
  int32_t a [] = { 0, 0, 0, 1 };
  assert(data_is_zero(a, 3) && !data_is_zero(a, 4));
  set_data_zero(a, 4);
  assert(data_is_zero(a, 4));
}

void test_array_lrsb() {
  int32_t a [] = { 0, 1, 4, 8 };
  assert(array_lrsb(a, 1) == 31);
  assert(array_lrsb(a, 2) == 30);
  assert(array_lrsb(a, 3) == 28);
  assert(array_lrsb(a, 4) == 27);
}

void test_compute_raw_dot_product() {
  int16_t a [] = { 1, 0, 1, 1 },
          b [] = { 2, 3, 4, 5};
      int64_t c = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, 1>(
          a, b, 4);
      printf("c is %d\n", (int)c);
      assert(c == 11);

      c = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, 1>(
          a, b, 3);
      assert(c == 6);

      c = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, -1>(
          a, b+3, 4);
     assert(c == 10);

      c = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, -1>(
          a, b+3, 3);
     assert(c == 8);
}

void test_compute_raw_dot_product_shifted() {
  int16_t a [] = { 1, 0, 1, 1 },
          b [] = { 2, 3, 4, 5};
  int64_t c = compute_raw_dot_product_shifted<int16_t, int32_t, 1>(
      a, b, 4, 1);
  assert(c ==  (2>>1) + (4>>1) + (5>>1));

  c = compute_raw_dot_product_shifted<int16_t, int32_t, 1>(
      a, b, 3, 1);
  assert(c ==  (2>>1) + (4>>1));

  c = compute_raw_dot_product_shifted<int16_t, int32_t, -1>(
      a, b+3, 4, 1);
  printf("c is now %d\n", (int)c);
  assert(c ==  (2>>1) + (3>>1) + (5>>1));
}

void test_raw_add_product() {
  int32_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
  int32_t scale = 8;
  int32_t prod_rshift = 2,
      scale_rshifted = (8 >> 2);
  int64_t b_nrsb = raw_add_product(4, a, scale, b, prod_rshift);
  assert(b_nrsb == array_lrsb(b, 4));
  assert(b[0] == (2 + a[0] * scale_rshifted));
  assert(b[1] == (3 + a[1] * scale_rshifted));
  assert(b[2] == (4 + a[2] * scale_rshifted));
  assert(b[3] == (5 + a[3] * scale_rshifted));
}

void test_raw_add_product_and_rshift() {
  int32_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
  int32_t scale = 8;
  int32_t prod_rshift = 2, b_rshift = 1,
      scale_rshifted = (8 >> 2);
  int64_t b_nrsb = raw_add_product_and_rshift(4, a, scale, b, prod_rshift,
                                              b_rshift);
  assert(b_nrsb == array_lrsb(b, 4));
  assert(b[0] == ((2 >> b_rshift) + a[0] * scale_rshifted));
  assert(b[1] == ((3 >> b_rshift) + a[1] * scale_rshifted));
  assert(b[2] == ((4 >> b_rshift) + a[2] * scale_rshifted));
  assert(b[3] == ((5 >> b_rshift) + a[3] * scale_rshifted));
}


void test_raw_add_product_and_lshift() {
  int32_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
  int32_t scale = 8;
  int32_t prod_rshift = 2, b_lshift = 1,
      scale_rshifted = (8 >> 2);
  int64_t b_nrsb = raw_add_product_and_lshift(4, a, scale, b, prod_rshift,
                                              b_lshift);
  assert(b_nrsb == array_lrsb(b, 4));
  assert(b[0] == ((2 << b_lshift) + a[0] * scale_rshifted));
  assert(b[1] == ((3 << b_lshift) + a[1] * scale_rshifted));
  assert(b[2] == ((4 << b_lshift) + a[2] * scale_rshifted));
  assert(b[3] == ((5 << b_lshift) + a[3] * scale_rshifted));
}



void test_raw_copy_product() {
  int32_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
  int32_t scale = 8;
  int32_t prod_rshift = 2,
      scale_rshifted = (8 >> 2);
  int64_t b_nrsb = raw_copy_product(4, a, scale, b, prod_rshift);
  assert(b_nrsb == array_lrsb(b, 4));
  assert(b[0] == (a[0] * scale_rshifted));
  assert(b[1] == (a[1] * scale_rshifted));
  assert(b[2] == (a[2] * scale_rshifted));
  assert(b[3] == (a[3] * scale_rshifted));
}


void test_raw_multiply_elements() {
  int32_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5},
      c [] = { 4, 5, 6, 7 };
  int32_t prod_rshift = 1;
  int64_t c_nrsb = raw_multiply_elements(4, a, b, prod_rshift, c);
  assert(c[0] == ((a[0] * b[0]) >> prod_rshift));
  assert(c[1] == ((a[1] * b[1]) >> prod_rshift));
  assert(c[3] == ((a[3] * b[3]) >> prod_rshift));
  assert(c_nrsb == array_lrsb(c, 4));
}


void test_raw_triple_product_a_shifted() {
  int16_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
 int32_t c [] = { 4, 5, 6, 7 };

  int32_t prod_rshift = 1;
  int64_t sum = raw_triple_product_a_shifted(4, a, b, c, prod_rshift);
  assert(sum == ((a[0]*b[0]*c[0]) >> prod_rshift) +
         ((a[1]*b[1]*c[1]) >> prod_rshift) +
         ((a[2]*b[2]*c[2]) >> prod_rshift) +
         ((a[3]*b[3]*c[3]) >> prod_rshift));
}


void test_raw_triple_product_a() {
  int16_t a [] = { 2, 0, 7, 9 },
      b [] = { 2, 3, 4, 5};
 int32_t c [] = { 4, 5, 6, 7 };

  int32_t prod_rshift = 0;
  int64_t sum = raw_triple_product_a_shifted(4, a, b, c, prod_rshift);
  assert(sum == ((a[0]*b[0]*c[0]) >> prod_rshift) +
         ((a[1]*b[1]*c[1]) >> prod_rshift) +
         ((a[2]*b[2]*c[2]) >> prod_rshift) +
         ((a[3]*b[3]*c[3]) >> prod_rshift));
}


void test_raw_triple_product_a_nrsb() {
  /* tests the logic used in
     void ToeplitzLpcEstimator::UpdateAutocorrStatsAndDeriv()
     regarding the nrsb of the inputs to raw_triple_product_a
     and raw_triple_product_a_shifted().

     RE products of int16 x int16 x int32: consider some
     extremal cases where have the sources have nrsb=0.
       -2^15 * -2^15 * -2^31 = -2^61 (nrsb = 2)
       -2^15 * -2^15 *  2^30 =  2^60 (nrsb = 2)
    Making that 2^60 into (2^61 - 1) wouldn't make any difference
    to the nrsb of the product.
  */

  for (int nrsb_a = 0; nrsb_a < 15; nrsb_a++) {
    for (int dim = 2; dim < 16; dim += 2) {  /* dim must be even, currently, or will
                                                crash in raw_triple_product_a. */
      int16_t a [16];
      int32_t c [16];
      for (int i = 0; i < dim; i++) {
        /* negative power of 2 * negative power of 2 * positive number is a 'worst
           case' that can give us an extra significant bit in the product.
           e.g. -2*-2 = 4 changes the number of significant bits from 1+1 to 3.
        */
        a[i] = (-1 << 15) >> nrsb_a; /* compiler warning here is expected.
                                        this code does depend on left-shifting
                                        signed values working 'as expected'
                                        for 2s-complement integers.
                                     */

        /* The following is the most positive representable number, == 2^31 - 1. */
        c[i] = (((int32_t)1 << 30) - 1 + ((int32_t)1 << 30));

        assert(lrsb(c[i]) == 0);
        assert(lrsb(a[i]) == nrsb_a);
      }
      /*
        The product here is -2^15 * -2^15 * (2^31 - 1) < 2^61; the
        nrsb is the same as for 2^60, which is 2.
      */
      assert(lrsb(c[0] * (int64_t)a[0] * (int64_t)a[0]) == 2 + (2 * nrsb_a));

      assert(lrsb(2 * (c[0] * (int64_t)a[0] * (int64_t)a[0])) == 1 + (2 * nrsb_a));

      int dim_significant_bits = extra_bits_from_factor_of(dim);


      assert(lrsb(a[0]) == nrsb_a);
      int right_shift_needed = dim_significant_bits - 2 - (2 * nrsb_a);
      if (right_shift_needed >= 0) {
        int64_t sum = raw_triple_product_a_shifted(dim, a, a, c, right_shift_needed);
        printf("right shift is %d, lrsb-sum = %d, sum is %lld, dim is %d\n", right_shift_needed, lrsb(sum), sum, dim);
        assert(lrsb(sum) == 0);
      } else {
        int64_t sum = raw_triple_product_a(dim, a, a, c);
        assert(lrsb(sum) == -right_shift_needed);
      }
    }
  }
}


void test_raw_triple_product_b_shifted() {
  int16_t a [] = { 2, 0, 7, 9 };
  int32_t b [] = { 2, 3, 4, 5},
       c [] = { 4, 5, 6, 7 };

  int32_t prod_rshift = 1;
  int64_t sum = raw_triple_product_b_shifted(4, a, b, c, prod_rshift);
  assert(sum == ((a[0]*b[0]*c[0]) >> prod_rshift) +
         ((a[1]*b[1]*c[1]) >> prod_rshift) +
         ((a[2]*b[2]*c[2]) >> prod_rshift) +
         ((a[3]*b[3]*c[3]) >> prod_rshift));
}


void test_raw_triple_product_b() {
  int16_t a [] = { 2, 0, 7, 9 };
  int32_t b [] = { 2, 3, 4, 5},
       c [] = { 4, 5, 6, 7 };

  int32_t prod_rshift = 0;
  int64_t sum = raw_triple_product_b(4, a, b, c);
  assert(sum == ((a[0]*b[0]*c[0]) >> prod_rshift) +
         ((a[1]*b[1]*c[1]) >> prod_rshift) +
         ((a[2]*b[2]*c[2]) >> prod_rshift) +
         ((a[3]*b[3]*c[3]) >> prod_rshift));
}



void test_lrsb_of_prod() {
  for (int i = 0; i < 31; i++) {
    for (int j = 0; i + j < 31; j++) {
      int32_t a = -1 << i,
          b = -1 << j;
      int32_t c = a * b;
      assert(lrsb(c) == lrsb_of_prod<int32_t>(lrsb(a), lrsb(b)));
    }
  }
}


void test_safe_shift_by() {
  assert(safe_shift_by(1, 100) == 0);
  assert(safe_shift_by(-1, 31) == -1);
  assert(safe_shift_by(-1, 32) == 0);
  assert(safe_shift_by(1, -1) == 2);
  assert(safe_shift_by(-1, -1) == -2);
  assert(safe_shift_by(1, -30) == (1 << 30));
}


void test_safe_shift_array_by() {
  int32_t a[] = { 1, 1, 1 };
  safe_shift_array_by(a, 3, -20);
  assert(a[0] == (1 << 20) && a[2] == (1 << 20));
  safe_shift_array_by(a, 3, 10);
  assert(a[0] == (1 << 10) && a[2] == (1 << 10));
  safe_shift_array_by(a, 3, 100);
  assert(a[0] == 0 && a[2] == 0);
}

}

int main() {
  using namespace int_math;

  assert(int_math_min(2, 3) == 2);

  assert(lrsb(int16_t(1)) == 14);
  assert(lrsb(int32_t(1)) == 30);
  assert(lrsb(int32_t(0)) == 31);
  assert(lrsb(int32_t(-1)) == 31);
  assert(lrsb(int64_t(0)) == 63);
  assert(lrsb(int64_t(-1)) == 63);

  assert(num_significant_bits(0) == 0);
  assert(num_significant_bits(1) == 1);
  assert(num_significant_bits(8) == 4);
  assert(num_significant_bits(-1) == 0);

  assert(extra_bits_from_factor_of(1) == 0);
  assert(extra_bits_from_factor_of(2) == 1);
  assert(extra_bits_from_factor_of(3) == 2);
  assert(extra_bits_from_factor_of(4) == 2);
  assert(extra_bits_from_factor_of(5) == 3);
  assert(extra_bits_from_factor_of(8) == 3);

  test_data_is_zero();
  test_array_lrsb();
  test_compute_raw_dot_product();
  test_compute_raw_dot_product_shifted();

  test_raw_add_product();
  test_raw_add_product_and_rshift();
  test_raw_add_product_and_lshift();
  test_raw_copy_product();
  test_raw_multiply_elements();
  test_raw_triple_product_a();
  test_raw_triple_product_a_shifted();
  test_raw_triple_product_a_nrsb();
  test_raw_triple_product_b();
  test_raw_triple_product_b_shifted();

  test_lrsb_of_prod();
  test_safe_shift_by();
  test_safe_shift_array_by();

  return 0;
}


