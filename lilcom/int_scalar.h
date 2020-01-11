#ifndef __LILCOM__INT_SCALAR_H__
#define __LILCOM__INT_SCALAR_H__


#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>  /* for abs. */
#include "int_math_utils.h"
#ifndef NDEBUG
#include <iostream>
#endif

namespace int_math {


/*
  This type represents a floating-point number using integer math.
  The type I should be int32_t or int64_t.
 */
template <typename I>
struct IntScalar {
  I elem;
  int exponent;

  IntScalar(I elem, int exponent): elem(elem), exponent(exponent) { }
  IntScalar(I value): elem(value), exponent(0) { }
  IntScalar() { }
  IntScalar(const IntScalar<I> &other): elem(other.elem),
                                        exponent(other.exponent) { }
  operator float() const { return elem * powf(2.0, exponent); }
  operator double() const { return elem * pow(2.0, exponent); }
};

#ifndef NDEBUG
template <typename I>
inline std::ostream &operator << (std::ostream &os, IntScalar<I> &s) {
  return os << (double)s;
}
#endif

/**
   Find the ratio a / b between two int64_t scalars a and b, as an int32_t
   scalar.

   The following works out the number of significant bits of a ratio
   of integers.  In this analysis, assume a and b are integers
   (they'd actually be a->elem and b->elem).

   Define the num-significant-bits as
       nsb(a) = 63 - lrsb(a),
       nsb(b) = 63 - lrsb(b)
   Note: nsb(a) and nsb(b) are both nonzero since a != 0 and b != 0.
   Caution: we're using latex notation below, using ^ for powers.

   In the analysis below we assume that a and b are positive, but this is
   without loss of generality because we are concerned with the absolute values'
   magnitudes (and the rounding behavior of division is symmetric).

  It's possible to show that for i != 0,

      2^{nsb(i)-1}) <= i < 2^{nsb(i)}    OR
        -2^{nsb(i)} <= i < 2^{nsb(i)-1}
  [The floor at the top there is needed just to handle the case where nsb(i)==0.]
   That implies that
      2^{nsb(i)-1} <= abs(i) <= 2^{nsb(i)}.                                   (eqn. 1)

   and of course the same for b.  Let c = a / b (viewed as a mathematical
   expression with reals).  Assuming a != 0 and b != 0, We can get maximum and
   minimum limits on abs(c) as:

      2^{nsb(a) nsb(b) - 1}  <=  abs(c)  <= 2^{nsb(a) nsb(b) + 1}

   Defining d = nsb(a) - nsb(b), IF c != 0 we can write the above as:

        2^{d-1}  <=  abs(c)  <=  2^{d+1}.
   If we can show that c always satisifes:
        2^{nsb(d)-1}) <= i < 2^{nsb(d)+1}    OR
        -2^{nsb(d)+1} <= i < 2^{nsb(d)-1}
   when we can show that nsb(c) is either d or d+1.  The less-trivial part of
   this is to show that the strict equalities on the right hand side are both
   true (as opposed to being <=).  On the top side we can do it.

   The case where abs(c) == 2^{d+1} can only be reached when
    a == -2^{nsb(i)} and b == +2^{nsb(i) - 1},
   but notice that this would make c negative, so the top < is OK.
   The bottom < is not OK though-- it should be a <= -- because
   of rounding (we round towards zero).

   Bottom line:

     if a != 0, b != 0 and a / b != 0, nsb(a / b) will either be d-1, d or d+1,
         where d = nsb(a) - nsb(b).

   The largest nsb we can safely have in an int32 whose sign we do not know is
   30. (since 2^31 is not representable), so that implies that we require nsb(c)
   <= 30, hence d <= 29.  We let d be exactly 29 to get the maximum number of
   significant bits that we can.  So when computing a / b, we want nsb(a) -
   nsb(b) = 29.


   In this templates IA,IB must both be either int32_t or int64_t (they may be
   different, though).  We could very possibly optimize a little more for the
   case where a and/or b is 32 bit.

   The pointers given to this function do not have to be distinct.
 */
template <typename IA, typename IB>
void divide(IntScalar<IA> *a, IntScalar<IB> *b,
            IntScalar<int32_t> *c) {
  assert(b->elem != 0);

  int64_t a_elem = a->elem,
      b_elem = b->elem;

  int a_lrsb = lrsb(int_math_abs(a_elem)),
      b_lrsb = lrsb(int_math_abs(b_elem));

  /* we want b to be shifted to have lrsb=29.  see above, nsb(a) - nsb(b) = 29;
     we'll have (lrsb(b) = 29) - (lrsb(a) = 0) = 29, which is equivalent.  */
  int b_rshift = 29 - b_lrsb;
  if (b_rshift >= 0) {
    /* b_shifted will have lrsb=29.  We want a_shifted to have lrsb=0. */
    int64_t b_shifted = b_elem >> b_rshift;
    int a_lshift = a_lrsb;
    int64_t a_shifted = a_elem << a_lshift;
    /* c->elem will have lrsb=1 or 2, i.e.  29 or 30 significant
       bits (remember the sign bit takes up 1 spot) */
    c->elem = (int32_t)(a_shifted / b_shifted);
    c->exponent = (a->exponent - a_lshift) - (b->exponent + b_rshift);
  } else {
    /* We don't want to shift b right, and it doesn't make sense to shift it
       left (nothing would be gained).  We need a to have `a_lz_needed`
       leading zeros where alz_needed = blz - 29. */
    int a_lrsb_needed = b_lrsb - 29;
    int a_rshift = a_lrsb_needed - a_lrsb;
    if (a_rshift >= 0) {
      int64_t a_shifted = a_elem >> a_rshift;
      c->elem = (int32_t)(a_shifted / b_elem);
    } else {
      int64_t a_shifted = a_elem << -a_rshift;
      c->elem = (int32_t)(a_shifted / b_elem);
    }
    c->exponent = (a->exponent + a_rshift) - b->exponent;
  }

  { // TEST
    int c_lrsb = lrsb(c->elem);
    assert(c->elem == 0 || c_lrsb == 1 || c_lrsb == 2 || c_lrsb == 3);
  }
}

template <typename I>
inline void init_as_power_of_two(int power, IntScalar<I> *s) {
  s->exponent = power;
  s->elem = 1;
}

inline void copy(const IntScalar<int64_t> *a,
                 IntScalar<int32_t> *b) {
  int a_lrsb = lrsb(a->elem),
      rshift = 32 - a_lrsb;  /* e.g. if a_lrsb == 0, shift right by 32 and will
                              * still have lrsb == 0. */
  if (rshift > 0) {
    b->elem = a->elem >> rshift;
    b->exponent = a->exponent + rshift;
    assert(lrsb(b->elem) == 0);
  } else {
    b->elem = a->elem;
    b->exponent = a->exponent;
  }
}

inline void copy(const IntScalar<int32_t> *a,
                 IntScalar<int64_t> *b) {
  b->elem = a->elem;
  b->exponent = a->exponent;
}


template <typename T>
inline void negate(IntScalar<T> *a) {
  if (a->elem == ((T)1) << (sizeof(T)*8 - 1)) {
    /* handle the special case where a->elem is -2^31 or -2^63, which cannot be
       negated (it just becomes itself).  Set a->elem to +2^30 or +2^62 and
       increase the exponent by one. */
    a->elem = (((T)1) << (sizeof(T)*8 - 2));
    a->exponent++;
  } else {
    a->elem *= -1;
  }
}

inline void multiply(IntScalar<int32_t> *a, IntScalar<int32_t> *b,
                     IntScalar<int32_t> *c) {
  int64_t a_elem = a->elem, b_elem = b->elem,
      prod = a_elem * b_elem;
  int prod_lrsb = lrsb(prod),
      prod_rshift = 32 - prod_lrsb;
  if (prod_rshift > 0) {
    c->elem = (int32_t)(prod >> prod_rshift);
    c->exponent = a->exponent + b->exponent + prod_rshift;
  } else {
    c->elem = (int32_t)prod;
    c->exponent = a->exponent + b->exponent;
  }
}

template <typename I>
inline void add(IntScalar<I> *a, IntScalar<I> *b, IntScalar<I> *c) {
  int a_nrsb = lrsb(a->elem),
      b_nrsb = lrsb(b->elem),
      a_exponent = a->exponent,
      b_exponent = b->exponent,
      c_exponent = int_math_min(a_exponent, b_exponent);
  /* the above value for c_exponent (the min of the two) is the
     value that would avoid any rounding error.  If that turns out
     to be too low, we'l increase it below. */

  /* a_new_nrsb is the nrsb that a would have after being
     shifted to have the same exponent as c currently has. */
  int a_new_nrsb = (a_nrsb + c_exponent - a_exponent);
  if (a_new_nrsb <= 0) {
    /* we need the shifted a to have nrsb >= 1 to give room for addition */
    c_exponent += 1 - a_new_nrsb;
  }
  int b_new_nrsb = (b_nrsb + c_exponent - b_exponent);
  if (b_new_nrsb <= 0) {
    /* we need the shifted a to have nrsb >= 1 to give room for addition */
    c_exponent += 1 - b_new_nrsb;
  }
  I a_shifted = safe_shift_by(a->elem, c_exponent - a_exponent),
      b_shifted = safe_shift_by(b->elem, c_exponent - b_exponent),
      sum = a_shifted + b_shifted;
  assert(lrsb(a_shifted) > 0 && lrsb(b_shifted) > 0);
  c->elem = sum;
  c->exponent = c_exponent;
}

}

#endif /* include guard */

