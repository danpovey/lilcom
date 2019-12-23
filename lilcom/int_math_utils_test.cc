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
      c = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, -1>(
          a, b+3, 4);
     printf("c is now %d\n", (int)c);
     assert(c == 10);
}

void test_compute_raw_dot_product_shifted() {
  int16_t a [] = { 1, 0, 1, 1 },
          b [] = { 2, 3, 4, 5};
  int64_t c = compute_raw_dot_product_shifted<int16_t, int32_t, 1>(
      a, b, 4, 1);
  printf("c is %d\n", (int)c);
  assert(c ==  (2>>1) + (4>>1) + (5>>1));
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

  test_data_is_zero();
  test_array_lrsb();
  test_compute_raw_dot_product();
  test_compute_raw_dot_product_shifted();

  test_raw_add_product();
  test_raw_copy_product();
  test_raw_multiply_elements();

  return 0;
}


