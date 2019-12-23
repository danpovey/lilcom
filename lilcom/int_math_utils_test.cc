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


  test_data_is_zero();
  test_array_lrsb();
  test_compute_raw_dot_product();
  test_compute_raw_dot_product_shifted();
  return 0;
}


