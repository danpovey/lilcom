#ifndef __LILCOM__INT_MATH_UTILS_H__
#define __LILCOM__INT_MATH_UTILS_H__


#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>  /* for abs. */



namespace int_math {

template <typename I> inline I int_math_min(I a, I b) {
  return (a < b ? a : b);
}


/* This header contains some lower-level utilities for integer math, that
   are used in int_math.h/int_math.c (and, a little bit, in lpc_math.h/lpc_math.c). */

/* num_lrsb should not be called for int16, only for int32 and int64;
   for int16 you can cast to int32.

   It returns the number of leading redundant sign bits, i.e. the
   number of bits following the most significant bit that are
   identical to it.
*/
inline int native_lrsb(int i) {  return __builtin_clrsb(i); }
inline int native_lrsb(long int i) {  return __builtin_clrsb(i); }
inline int native_lrsb(long long int i) {  return __builtin_clrsb(i); }


/* returns number of leading redundant sign bits (for fixed-size types) */
inline int lrsb(int16_t i) { return native_lrsb((int32_t)i) - 16; }
inline int lrsb(int32_t i) { return native_lrsb(i); }
inline int lrsb(int64_t i) { return native_lrsb(i); }

/*
  num_significant_bits returns the bits apart from the sign bit, that
  are required to encode this number.

  TODO: see if these are needed.
 */
inline int num_significant_bits(int16_t i) {
  return 31 - lrsb((int32_t)(i));
}
inline int num_significant_bits(int32_t i) {
  return 31 - lrsb(i);
}
inline int num_significant_bits(int64_t i) {
  return 63 - lrsb(i);
}


template <typename I>
inline bool data_is_zero(I *data, int dim) {
  for (int i = 0; i < dim; i++)
    if (data[i] != 0) return false;
  return true;
       }
template <typename I>
inline void set_data_zero(I *data, int dim) {
  for (int i = 0; i < dim; i++)
    data[i] = 0;
}
template <typename I>
inline int get_nrsb(I *data, int dim) {
  int nrsb = 1000;
  for (int i = 0; i < dim; i++) {
    int n = lrsb(data[i]);
    if (n < nrsb) nrsb = n;
  }
  return nrsb;
}

/*
  This templated function computes the dot product of integer arrays
  A and B.  The stride of A is 1 but it's templated on the stride of B
  so you can choose that (at compile time).

  This one doesn't support any shifts.
*/
template <typename IA, typename IB, typename IProd, typename IRet, int B_stride>
IRet compute_raw_dot_product(const IA *array_A, const IB *array_B, int dim) {
  assert(sizeof(IProd) >= sizeof(IA) + sizeof(IB) &&
         sizeof(IRet) > sizeof(IA) + sizeof(IRet));
  IRet sum1 = 0, sum2 = 0;  /* separate sums for pipelining */
  int i;
  for (i = 0; i + 2 <= dim; i += 2) {
    sum1 += array_A[i] * static_cast<IProd>(array_B[i * B_stride]);
    sum2 += array_A[i + 1] * static_cast<IProd>(array_B[(i + 1) + B_stride]);
  }
  if (i < dim)
    sum1 += array_A[i] * static_cast<IProd>(array_B[i * B_stride]);
  return sum1 + sum2;
}


template <typename IIn, typename IOut, int B_stride>
IOut compute_raw_dot_product_shifted(const IIn *array_A, const IIn *array_B, int dim, int rshift) {
  IOut sum1 = 0, sum2 = 0;  /* separate sums for pipelining */
  int i;
  for (i = 0; i + 2 <= dim; i += 2) {
    sum1 += (array_A[i] * static_cast<IOut>(array_B[i * B_stride])) >> rshift;
    sum2 += (array_A[i + 1] * static_cast<IOut>(array_B[(i + 1) * B_stride])) >> rshift;
  }
  if (i < dim)
    return sum1 += (array_A[i] * static_cast<IOut>(array_B[i * B_stride])) >> rshift;
  return sum1 + sum2;
}

void raw_right_shift_by(int32_t *a, int dim, int rshift) {
  assert(static_cast<unsigned int>(rshift) <= 32);
#pragma unroll
  for (int i = 0; i < dim; i++)
    a[i] >>= rshift;
}

int raw_add_product(int dim, const int32_t *a, int32_t scale, int32_t *b,
                     int prod_rshift) {
  int min_nrsb = 31;
#pragma unroll
  for (int i = 0; i < dim; i++) {
    int32_t new_b = b[i] + ((a[i]*(int64_t)scale) >> prod_rshift);
    b[i] = new_b;
    int nrsb = lrsb(new_b);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}

int raw_copy_product(int dim, const int32_t *a, int32_t scale, int32_t *b,
                    int prod_rshift) {
  int min_nrsb = 31;
#pragma unroll
  for (int i = 0; i < dim; i++) {
    int32_t new_b = ((a[i]*(int64_t)scale) >> prod_rshift);
    b[i] = new_b;
    int nrsb = lrsb(new_b);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}


int raw_multiply_elements(int dim, const int32_t *a, const int32_t *b,
                          int prod_rshift, int32_t *c) {
  int min_nrsb = 31;
#pragma unroll
  for (int i = 0; i < dim; i++) {
    int32_t new_c = ((a[i]*(int64_t)b[i]) >> prod_rshift);
    c[i] = new_c;
    int nrsb = lrsb(new_c);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}


/**
   Computes sum_{i=0}^{dim-1} a[i]*b[i]*c[i].
   Requires dim % 2 == 0 (could easily get around this).
 */
int64_t raw_triple_product_a(int dim, const int16_t *a, const int16_t *b,
                             const int32_t *c) {
  assert(dim % 2 == 0);
  /* breaking the sum into two parts is supposed to aid pipelining
     by reducing dependencies.  I'm not sure if compilers are smart
     enough to do this kind of thing. */
  int64_t sum1 = 0, sum2 = 0;
  for (int i = 0; i < dim; i += 2) {
    int32_t ab1 = a[i] * (int32_t)b[i],
        ab2 = a[i+1] * (int32_t)b[i+1];
    int64_t abc1 = ab1 * (int64_t)c[i],
        abc2 = ab2 * (int64_t)c[i];
    sum1 += abc1;
    sum2 += abc2;
  }
  return sum1 + sum2;
}

/* Another version of raw_triple_product, where b is int32_t not int16_t
   and the caller asserts that the elements of `b` have absolute values
   not greater than (2^16 - 1).  THat means the absolute value of
   any a[i] * b[i] is <= (2^15) * (2^16 - 1) which is strictly less
   than 2^31, meaning it is representable as an int32.
 */
int64_t raw_triple_product_b(int dim, const int16_t *a, const int32_t *b,
                             const int32_t *c) {
  assert(dim % 2 == 0);
  int64_t sum1 = 0, sum2 = 0;
  for (int i = 0; i < dim; i += 2) {
    int32_t ab1 = a[i] * b[i],
        ab2 = a[i+1] * b[i+1];
    int64_t abc1 = ab1 * (int64_t)c[i],
        abc2 = ab2 * (int64_t)c[i];
    sum1 += abc1;
    sum2 += abc2;
  }
  return sum1 + sum2;
}



int raw_add_product_and_rshift(int dim, const int32_t *a, int32_t scale, int32_t *b,
                               int prod_rshift, int b_rshift) {
  int min_nrsb = 31;
#pragma unroll
  for (int i = 0; i < dim; i++) {
    int32_t new_b = (b[i] >> b_rshift) + ((a[i]*(int64_t)scale) >> prod_rshift);
    b[i] = new_b;
    int nrsb = lrsb(new_b);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}


/**
    Analysis of how rsb's interact with multiplication.
    (rsb == redundant sign bit).

    Define the num-significant-bits of an integer i
         nsb(i) = 31 - lrsb(i),

    (e.g. 0 or -1 would have 0 significant bits; 1 or -2 would have 1; 3 or -4
    would have 2...


    Assuming 32-bit integers, and that x and y are nonzero, and that
    there is no overflow....

    The largest absolute value a value with x significant bits can have
    is 2^x (achieved when the actual value is -2^x).  When we multiply
    two negative numbers with x and y significant bits and the largest
    possible values (2^x and 2^y), the product is +2^(x+y), which
    has x+y+1 significant bits.

    So    nsb(x*y)  <= nsb(x) + nsb(y) + 1

    Using nsb(i) = 31 - lrsb(i), this becomes:

      31 - lrsb(x*y)  <= (31 - lrsb(x)) + (31 - lrsb(y)) + 1
    which reduces to:
          lrsb(x*y) <=   lrsb(x) + lrsb(y) - 32
 */
template <typename I>   // Note: you have to explicitly use the template argument
    inline int lrsb_of_prod(int lrsb_a, int lrsb_b) {
  return lrsb_a + lrsb_b - (8 * sizeof(I));
};



/*
  safe_shift_by returns I right shifted by `shift` if `shift` is positive,
  left-shifted by `shift` if `shift` is negative.  Requires, for correctness,
  that shift > -8*sizeof(I).
 */
template <typename I>
inline I safe_shift_by(I i, int shift) {
  if (shift >= 0) {
    if (shift >= 8 * sizeof(I)) return 0;
    else return i >> shift;
  } else {
    return i << -shift;
  }
}

}  // namespace int_math

#endif /* include guard */
