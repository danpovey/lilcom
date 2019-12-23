#ifndef __LILCOM__INT_MATH_H__
#define __LILCOM__INT_MATH_H__


#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>  /* for abs. */


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
inline int lrsb(int32_t i) { return native_lrsb(i); }

/*
  num_significant_bits returns the bits apart from the sign bit, that
  are required to encode this number.
 */
inline int num_significant_bits(int16_t i) {
  return 31 - num_lrsb((int32_t)(i));
}
inline int num_significant_bits(int32_t i) {
  return 31 - num_lrsb(i);
}
inline int num_significant_bits(int64_t i) {
  return 63 - num_lrsb(i);
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


/* The type `I` should be int16_t, int32_t or int64_t.  Must not (currently) be
   instantiated for other types.  Not all operations are defined for all types or
   combinations of types; we only implemented what was needed.
*/
template <typename I>
struct IntVec {
  I *data;
  int dim;
  int nrsb;  /* nrsb = min(lrsb(data[i])) for 0 <= i < dim. */
  int exponent;  /* If the data in `data` is all zero, `exponent` must be
                    set to -1000. */
  IntScalar<I> operator [] (int i) { return IntScalar<I>(data[i], exponent); }

  inline void set_nrsb(int nrsb_in) {
    nrsb = nrsb_in;
    if (nrsb + 1 == (sizeof(I)*8) && data_is_zero(data, dim))
      exponent = -10000;
  }

#ifdef NDEBUG
  inline void check() const { }
#else
  void check() const;
#endif

  ~IntVec() { delete data; }
  IntVec(int dim):
      data(new I[dim]), dim(dim),
      exponent(-10000), nrsb(8 * sizeof(I) - 1) {
    set_data_zero(data, dim);
  }
  IntVec(int dim): data(NULL), dim(0) { }
  void resize(int d) {
    delete data;
    dim = d;
    data = new data[d];
    exponent = -10000;
    nrsb = 8 * sizeof(I) - 1;
    zero_data(data, dim);
  }
};

/* I should be int32_t or int64_t. */
template <typename I>
struct IntScalar {
  I elem;
  int exponent;
  IntScalar(I elem, int exponent): elem(elem), exponent(exponent) { }
  IntScalar(I value): elem(value), exponent(0) { }
  IntScalar() { }

  operator float() { return elem * powf(2.0, exponent); }
};



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
    sum1 += array_A[i] * static_cast<IProd>(array_B[i * b_stride]);
    sum2 += array_A[i + 1] * static_cast<IProd>(array_B[(i + 1) + b_stride]);
  }
  if (i < dim)
    sum1 += array_A[i] * static_cast<IProd>(array_B[i * b_stride]);
  return sum1 + sum2;
}


template <typename IIn, typename IOut, int B_stride>
IOut compute_raw_dot_product_shifted(const IIn *array_A, const IIn *array_B, int dim, int rshift) {
  IOut sum1 = 0, sum2 = 0;  /* separate sums for pipelining */
  for (int i = 0; i + 2 <= dim; i += 2) {
    sum1 += (array_A[i] * static_cast<IOut>(array_B[i * b_stride])) >> rshift;
    sum2 += (array_A[i + 1] * static_cast<IOut>(array_B[(i + 1) * b_stride])) >> rshift;
  }
  if (i < dim)
    return sum1 += (array_A[i] * static_cast<IOut>(array_B[i * b_stride])) >> rshift;
  return sum1 + sum2;
}

void raw_right_shift_by(int32 *a, int dim, int rshift) {
  assert(static_cast<unsigned int>(rshift) <= 32);
#pragma unroll
  for (int i = 0; i < dim; i++)
    a[i] >>= rshift;
}

int raw_add_product(int dim, const int32 *a, int32 scale, int32 *b,
                     int prod_rshift) {
#pragma unroll
  int min_nrsb = 31;
  for (int i = 0; i < dim; i++) {
    int32_t new_b = b[i] + ((a[i]*(int64_t)scale) >> prod_rshift);
    b[i] = new_b;
    int nrsb = lrsb(new_b);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}

int raw_copy_product(int dim, const int32 *a, int32 scale, int32 *b,
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


int raw_multiply_elements(int dim, const int32 *a, const int32 *b,
                          int prod_rshift, int32 *c) {
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
    int64 abc1 = ab1 * (int64_t)c[i],
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
    int64 abc1 = ab1 * (int64_t)c[i],
        abc2 = ab2 * (int64_t)c[i];
    sum1 += abc1;
    sum2 += abc2;
  }
  return sum1 + sum2;
}



int raw_add_product_and_rshift(int dim, const int32 *a, int32 scale, int32 *b,
                               int prod_rshift, int b_rshift) {
#pragma unroll
  int min_nrsb = 31;
  for (int i = 0; i < dim; i++) {
    int32_t new_b = (b[i] >> b_rshift) + ((a[i]*(int64_t)scale) >> prod_rshift);
    b[i] = new_b;
    int nrsb = lrsb(new_b);
    if (nrsb < min_nrsb)
      min_nrsb = nrsb;
  }
  return min_nrsb;
}

/*
  Computes dot product between an int32 and an int16 vector
 */
void compute_dot_product(IntVec<int32_t> *a, IntVec<int16_t> *b, IntScalar<int64_t> *out) {
  assert(a->dim == b->dim);
  out->elem = compute_raw_dot_product<int32_t, int16_t, int64_t, int64_t, 1>(
      a->data, b->data, a->dim);
  out->exponent = a->exponent + b->exponent;
}


/*
  Computes dot product between two int32 vectors.  Caution: it will right shift
  after the multiplication regardless of the nrsb of the inputs, which could
  lead to loss of precision.
 */
void compute_dot_product(IntVec<int32_t> *a, IntVec<int32_t> *b, IntScalar<int64_t> *out) {
  assert(a->dim == b->dim);
  int dim = a->dim,
      rshift = num_significant_bits(dim);
  out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, 1>(
      a->data, b->data, dim, rshift);
  out->exponent = a->exponent + b->exponent + rshift;
}

/*
  Computes the dot product that in NumPy would be
     np.dot(a[a_offset:a_offset+dim], b[b_offset:b_offset+dim])
 */
inline void compute_dot_product(int dim,
                                IntVec<int16_t> *a, int a_offset,
                                IntVec<int16_t> *b, int b_offset,
                                IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);
  out->elem = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, 1>(
      a->data + a_offset, b->data + b_offset, dim);
  out->exponent = a->exponent + b->exponent;
}


/*
  Computes the dot product that in NumPy would be
     np.dot(a[a_offset:a_offset+dim], np.flip(b[b_offset:b_offset+dim]))
  Note: it does not actually matter which of the two arguments is flipped.
 */
inline void compute_dot_product_flip(int dim,
                         IntVec<int16_t> *a, int a_offset,
                         IntVec<int16_t> *b, int b_offset,
                         IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);
  out->elem = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, -1>(
      a->data + a_offset, b->data + b_offset + dim - 1, dim);
  out->exponent = a->exponent + b->exponent;
}


inline void compute_dot_product_flip(int dim,
                         IntVec<int32_t> *a, int a_offset,
                         IntVec<int32_t> *b, int b_offset,
                         IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);

  int rshift = num_significant_bits(dim);

  out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, -1>(
      a->data + a_offset, b->data + b_offset + dim - 1, dim, rshift);
  out->exponent = a->exponent + b->exponent;
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
    inline int lrsb_of_prod(int lrsb_a, int lrsb_b) { return a + b - (8 * sizeof(I)); };



/*
  This function does what in NumPy would be:

     b[a_offset:a_offset+dim] += scalar * a[b_offset:b_offset+dim]

   a and b must be different pointers.

   Note: we assume that b already contains data, which may need to be shifted.
   CAUTION: this function makes several assumptions about the input,
   which should be studied carefully before using it.

       - We assume the elements of b that are not in the range
         b[dim:dim+b_offset] are all zero!  That way, this function can avoid
         shifting that data and can avoid needing to inspect it when setting the
         nrsb of b.
 */
inline void add_scaled_special(int dim,
                               const IntScalar<int32_t> *scalar,
                               const IntVec<int32_t> *a, int a_offset,
                               IntVec<int32_t> *b, int b_offset) {
  assert(b_offset + dim <= b->dim && a_offset + dim <= a->dim && a != b);
  int a_nrsb = a->nrsb,
      b_nrsb = b->nrsb,
      scalar_nrsb = nrsb(scalar->elem);
  a->check();
  b->check();
#ifndef NDEBUG
  for (int i = 0; i < b_offset; i++) {
    assert(b->data[i] = 0);
  }
  for (int i = b_offset + dim; i < b->dim; i++) {
    assert(b->data[i] = 0);
  }
#endif

  if (scalar->data == 0 || a->exponent == -10000) {
    /* nothing to do if scalar == 0 or a is zero (note: we set exponent=-10000 if
       an IntVec is zero); returning now avoids the possibility of unnecessarily
       shifting b. */
    return;
  }

  /* input_rshift will normally be positive.  It's the amount by which we right
     shift products of a->data[i] * scalar->data (before any shifting of the
     result and b together).
  */
  int input_rshift = b->exponent - (a->exponent + scalar->exponent);

  int b_rshift = 0;

  /* See if we need to shift b right */

  int min_nrsb = min(lrsb_of_prod<int32_t>(a_nrsb, b_nrsb) + input_rshift,
                     b->nrsb);
  /* We need the smaller of nrsb of b vs. (product) to be at least 1.
     (1 is headroom to allow for addition to work;
     the sum of two quantities that have 1 redundant sign bit will
     still be representable in that type. */
  int b_rshift = 1 - min_nrsb;

  int64_t a_elem = a->elem;

  if (input_rshift & 63 != 0) {  /* if input_rshift is >= 64 or < 0 ... */
    if (input_rshift < 0) {
      /* rather than left shifting in the loop, we can just left shift a; this
       * will never overflow thanks to the nrsb logic above. */
      input_rshift = 0;
      a_elem <<= input_rshift;
    } else {
      /* The thing we're adding is so small that it won't affect the output. */
      return;
    }
  }
  if (b_rshift & 31 == 0) {
    /* normal case. */
    b->exponent += b_rshift;
    b->set_nrsb(raw_add_product_and_rshift(dim, a->data + a_offset, scalar->elem,
                                           b->data + b_offset, input_rshift + b_rshift,
                                           b_rshift));
  } else {
    b->exponent += b_rshift;
    /* pathological case: if b_rshift is >= 32 or < 0 ... */
    if (b_rshift < 0) {
      /* As a policy, we never left-shift existing data. */
      b->set_nrsb(raw_add_product(dim, a->data + a_offset,
                                  scalar->elem,
                                  b->data + b_offset, input_rshift));
      return;  /* and b's exponent is unchanged. */
    } else {  /* b_rshift >= 32, assume b's data is discarded. */
      b->set_nrsb(raw_copy_product(dim, a->data + a_offset, scalar->elem,
                                   b->data + b_offset, input_rshift + b_rshift));
    }
  }
  b->check();
}

/* does y += a * x. */
inline void add_scaled(const IntScalar<int32_t> *a,
                       const IntVec<int32_t> *x,
                       IntVec<int32_t> *y) {
  assert(x->dim == y->dim && x != y);
  add_scaled_special(y->dim, a, x, 0, y, 0);
}

/* does y += x.  TODO: maybe optimize this. */
inline void add(const IntVec<int32_t> *x,
                IntVec<int32_t> *y) {
  IntScalar<int32_t> one(1);
  add_scaled(&one, x, y);
}


/*  does c := a * b  (elementwise). */
inline void multiply(const IntVec<int32_t> *a,
                     const IntVec<int32_t> *b,
                     IntVec<int32_t> *c) {
  assert(c != a &&  != b && a->dim == b->dim && b->dim == c->dim);
  a->check();
  b->check();
  int a_nrsb = a->nrsb,
      b_nrsb = b->nrsb,
      prod_nrsb = lrsb_of_prod<int32>(a_nrsb, b_nrsb);
  int right_shift = (prod_nrsb < 0 ? -prod_nrsb : 0);
  /* we'll almost always have to right shift; we don't attempt
     to handle the non-right-shift case specially. */
  c->exponent = a->exponent + b->exponent - right_shift;
  c->set_nrsb(raw_multiply_elements(a->dim, a->data, b->data,
                                    right_shift, c->data));
}


/*
  Scale a vector.
 NOT NEEDED?
 */
inline void mul_vec_by_scalar(int dim,
                              const IntScalar<int64_t> *scalar,
                              const IntVec<int32_t> *src,
                              IntVec<int32_t> *dest) {
  /* TODO. */
}

inline void mul_vec_by_vector(int dim,
                              const IntVec<int32_t> *a,
                              IntVec<int32_t> *b) {
  /* TODO. */
}

/*
  Initializes `out` to powers of a: out[n] = a^(n+1) for n = 0, 1, 2..
  Requires that 0 < a < 1.
 */
void init_vec_as_powers(int dim,
                        const IntScalar<int32_t> *a,
                        IntVec<int32_t> *out) {
  int dim = out->dim;

  assert(a_elem >= 0 && (float)(*a) < 1.0 &&
         a->exponent >= -31);

  int32_t a_int = safe_shift_by(a->elem, (-31) - a->exponent);
  assert(a_int > 0);
  a->nrsb = lrsb(a_int);
  out->exponent = -31;  /* will all be of the form 2^31 * a^(n+1) for n >= 0 */
  out->data[0] = a_int;

  /* note: a to the power n is located at out->data[n-1]. */
  for (int power = 2; power <= dim; power++) {
    int32_t factor1 = out->data[(power / 2) - 1],
        factor2;
    if (power % 2 == 0)  factor2 = factor1;
    else factor2 = out->data[(power / 2 - 1) - 1];
    out->data[power - 1] = (int32_t) ((factor1 * (int64_t)factor2) >> 31);
  }
}



/*
  reverses the order of the elements of a vector
 */
void flip_vec(IntVec<int32_t> *a);



/*
  This is some rather special-purpose code needed in the Toeplitz solver.
  it does what in NumPy would be:
      b[-(n+1):-1] += s * np.flip(b[-n:])
  This amounts to:
     b[-(n+1)] += s * b[-1].
     b[-(n+1):-1] + s * np.flip(b[-(n+1):-1])
  We also make use of several facts which are true in the context
  in which we need it, which are that:
    abs(s) < 1.0   [in the algorithm this is nu_n]
    b[-1] = 1.0.
    The only nonzero elements of b are those in b[-(n+1):].

 */
void special_reflection_function(int n, IntVec<int32_t> *b, IntScalar<int32_t> *s) {
  int dim = b->dim;
  assert(std::abs(static_cast<float>(*s)) < 1.0 &&
         static_cast<float>((*b)[dim-1]) == 1.0);
  // TODO.
  // [a] see if we have to rshift b; do it if so.
  // [b] compute rshift on multiplication, will be exponent of s.
}


void get_elem(IntVec<int32_t> *a, int i, IntScalar<int32_t> *value) {
  assert(static_cast<unsigned_int>(i) < static_cast<unsigned int>(a->dim));
  value->exponent = a->exponent;
  value->elem = a->data[i];
}



/**
   Find the ratio a / b between two int64_t scalars a and b, as an int32_t
   scalar.

   The following works out the number of significant bits of a ratio
   of integers.  In this analysis, assume a and b are integers
   (they'd actually be a->data and b->data).

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
  assert(b->data != 0);

  int64_t a_elem = a->elem,
      b_elem = b->elem;

  int a_lrsb = lrsb(abs(a_elem)),
      b_lrsb = lrsb(abs(b_elem));


  /* we want b to be shifted to have lrsb=29.  see above, nsb(a) - nsb(b) = 29;
     we'll have (lrsb(b) = 29) - (lrsb(a) = 0) = 29, which is equivalent.  */
  int b_rshift = 29 - b_lrsb;
  if (b_rshift >= 0) {
    /* b_shifted will have lrsb=29.  We want a_shifted to have lrsb=0. */
    int64_t b_shifted = b_elem >> b_rshift;
    int a_lshift = a_lrsb;
    int64_t a_shifted = a_elem << a_lshift;
    /* c->data will have lrsb=1 or 2, i.e.  29 or 30 significant
       bits (remember the sign bit takes up 1 spot) */
    c->data = (int32_t)(a_shifted / b_shifted);
    c->exponent = (a->exponent - a_lshift) - (b->exponent + b_rshift);
  } else {
    /* We don't want to shift b right, and it doesn't make sense to shift it
       left (nothing would be gained).  We need a to have `a_lz_needed`
       leading zeros where alz_needed = blz - 29. */
    int a_lrsb_needed = b_lrsb - 29;
    int a_rshift = a_lrsb_needed - a_lrsb;
    if (a_rshift >= 0) {
      int64_t a_shifted = a_elem >> a_rshift;
      c->data = (int32_t)(a_shifted / b_elem);
    } else {
      int64_t a_shifted = a_elem << -a_rshift;
      c->data = (int32_t)(a_shifted / b_elem);
    }
    c->exponent = (a->exponent + a_rshift) - b->exponent;
  }

  { // TEST
    int c_lrsb = lrsb(c->data);
    assert(c->data == 0 || clz == 1 || clz == 2 || clz == 3);
  }
}


inline void copy(const IntScalar<int64_t> *a,
                 IntScalar<int32_t> *b) {
  int64_t a_elem = a->elem;
  if (a_elem == 0) {
    b->
  }
  int a_lrsb = lrsb(a->elem),
      rshift = 32 - a_lrsb;  /* e.g. if a_lrsb == 0, shift right by 32 and will
                              * still have lrsb == 0. */
  if (rshift > 0) {
    b->elem = a->elem >> rshift;
    b->exponeont = a->exponent + rshift;
    b->lrsb = 32;
  } else {
    b->elem = a->elem;
    b->exponeont = a->exponent;
    b->lrsb = a->lrsb - 32;
  }
}

/* Convert a vector from int64_t to int32_t. */
inline void copy(const IntVector<int64_t> *a,
                 IntVector<int32_t> *b) {
  assert(a->dim == b->dim);
  int dim = a->dim;
  a->check();
  if (exponent = -1000) {
    b->exponent = -1000;
    b->nrsb = 31;
    set_data_zero(b->data, dim);
  } else {
    int rshift = 32 - a->lrsb;  /* e.g. if a->lrsb == 0, shift right by 32 and will
                                 * still have lrsb == 0. */
    if (rshift > 0) {
      for (int i = 0; i < dim; i++) {
        b->data[i] = (a->data[i] >> rshift);
      }
      b->lrsb = 0;
      b->exponent = a->exponent + rshift;
    } else {
      for (int i = 0; i < dim; i++)
        b->data[i] = a->data[i];
      b->lrsb = a->lrsb + 32;
      b->exponent = a->exponent;
    }
  }
}

void multiply(IntScalar<int32_t> *a, IntScalar<int32_t> *b,
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





/*
  Set the only nonzero element of the vector 'v' to the scalar 's', i.e.
   v[i] = s.
  The point of being the only nonzero element is that we are free to set the
  exponent to whatever we want.  It is the caller's job to make sure that
  no other elements of 'v' are nonzero.
 */
void set_only_nonzero_elem_to(const IntScalar<int32_t> *s, int i, IntVec<int32_t> *v) {
  assert(static_cast<unsigned int>(i) < static_cast<unsigned int>(v->dim));
  v->data[i] = s->elem;
  v->exponent = s->exponent;
  v->set_nrsb(lrsb(s->elem));
}

template <typename I>
inline void zero_int_vector(IntVec<I> *v) {
  for (int i = 0; i < v->dim; i++)
    v->data[i] = 0;
}

/*
  safe_shift_by returns I right shifted by `shift` if `shift` is positive,
  left-shifted by `shift` if `shift` is negative.  Requires, for correctness,
  that shift > -8*sizeof(I).
 */
template <typename I>
inline I safe_shift_by(I, int shift) {
  if (shift >= 0) {
    if (shift >= 8 * sizeof(I)) return 0;
    else return I >> shift;
  } else {
    return I << -shift;
  }
}


/*
  TODO: `unsafe` version of add that doesn't do so many checks?

  CAUTION: may not be safe for all forms of zero!  E.g.
  zero with a very large exponent.
 */
template <typename I>
inline void add(Scalar<I> *a, Scalar<I> *b, Scalar<I> *c) {
  int max_allowed_shift = (8 * sizeof(I)) - 1,
      a_nrsb = lrsb(a->elem),
      b_nrsb = lrsb(b->elem),
      a_exponent = a->exponent,
      b_exponent = b->exponent,
      c_exponent = min(a_exponent, b_exponent);

  int a_new_nrsb = (a_nrsb + a->exponent - c_exponent);
  if (a_new_nrsb <= 0) {
    c_exponent += 1 - a_new_nrsb;
  }
  int b_new_nrsb = (b_nrsb + b->exponent - c_exponent);
  if (b_new_nrsb <= 0) {
    c_exponent += 1 - b_new_nrsb;
  }
  I a_shifted = safe_shift_by(a->elem, c->exponent - a->exponent),
      b_shifted = safe_shift_by(b->elem, c->exponent - b->exponent),
      sum = a_shifted + b_shifted;
  c->elem = sum;
  c->exponent = c_exponent;
}


#endif /* include guard */

