#ifndef __LILCOM__INT_VEC_H__
#define __LILCOM__INT_VEC_H__


#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>  /* for abs. */
#include "int_math_utils.h"
#include "int_scalar.h"


namespace int_math {

enum {  kExponentOfZero = -10000 };

/* The type `I` should be int16_t, int32_t or int64_t.  Must not (currently) be
   instantiated for other types.  Not all operations are defined for all types or
   combinations of types; we only implemented what was needed.
*/
template <typename I>
struct IntVec {
  I *data;
  int dim;
  int nrsb;  /* nrsb = min(lrsb(data[i])) for 0 <= i < dim;
                lrsb is the number of leading redundant sign bits, i.e.
                the num-bits following the sign bit that are the same as
                the sign bit. */
  int exponent;  /* If the data in `data` is all zero, `exponent` must be
                    set to kExponentOfZero. */
  IntScalar<I> operator [] (int i) { return IntScalar<I>(data[i], exponent); }

  inline void set_nrsb(int nrsb_in) {
    nrsb = nrsb_in;
    if (nrsb + 1 == (sizeof(I)*8) && data_is_zero(data, dim))
      exponent = kExponentOfZero;
  }

#ifdef NDEBUG
  inline void check() const { }
#else
  void check() const;
#endif

  ~IntVec() { delete [] data; }
  IntVec(): data(NULL), dim(0) { }
  IntVec(int dim):
      data(new I[dim]), dim(dim),
      nrsb(8 * sizeof(I) - 1),
      exponent(kExponentOfZero) {
    set_data_zero(data, dim);
  }
  void resize(int d) {
    delete data;
    dim = d;
    data = new I[d];
    exponent = kExponentOfZero;
    nrsb = 8 * sizeof(I) - 1;
    zero_data(data, dim);
  }
};



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
      scalar_nrsb = lrsb(scalar->elem);
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

  if (scalar->elem == 0 || a->exponent == kExponentOfZero) {
    /* nothing to do if scalar == 0 or a is zero (note: we set
       exponent=kExponentOfZero if an IntVec is zero); returning now avoids the
       possibility of unnecessarily shifting b. */
    return;
  }

  /* input_rshift will normally be positive.  It's the amount by which we right
     shift products of a->data[i] * scalar->data (before any shifting of the
     result and b together).
  */
  int input_rshift = b->exponent - (a->exponent + scalar->exponent);

  /* See if we need to shift b right */

  int min_nrsb = int_math_min(lrsb_of_prod<int32_t>(a_nrsb, scalar_nrsb) + input_rshift,
                              b_nrsb);
  /* We need the smaller of nrsb of b vs. (product) to be at least 1.  (1 is
     headroom to allow for addition to work; the sum of two quantities that have
     1 redundant sign bit will still be representable in that type. */
  int b_rshift = 1 - min_nrsb;

  int64_t scalar_elem = scalar->elem;


  if ((input_rshift & 63) != 0) {  /* if input_rshift is >= 64 or < 0 ... */
    if (input_rshift < 0) {
      /* rather than left shifting in the loop, we can just left shift a; this
       * will never overflow thanks to the nrsb logic above. */
      input_rshift = 0;
      scalar_elem <<= input_rshift;
    } else {
      /* The thing we're adding is so small that it won't affect the output. */
      return;
    }
  }
  if ((b_rshift & 31) == 0) {
    /* normal case. */
    b->exponent += b_rshift;
    b->set_nrsb(raw_add_product_and_rshift(dim, a->data + a_offset, scalar_elem,
                                           b->data + b_offset, input_rshift + b_rshift,
                                           b_rshift));
  } else {
    b->exponent += b_rshift;
    /* unusual case: if b_rshift is >= 32 or < 0 ... */
    if (b_rshift < 0) {
      b->set_nrsb(raw_add_product_and_lshift(
          dim, a->data + a_offset,
          scalar_elem,  b->data + b_offset,
          input_rshift + b_rshift, -b_rshift));
      return;  /* and b's exponent is unchanged. */
    } else {
      /* b_rshift >= 32, so assume b's data is discarded by the shift. */
      b->set_nrsb(raw_copy_product(dim, a->data + a_offset, scalar_elem,
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
  assert(c != a && c != b && a->dim == b->dim && b->dim == c->dim);
  a->check();
  b->check();
  int a_nrsb = a->nrsb,
      b_nrsb = b->nrsb,
      prod_nrsb = lrsb_of_prod<int32_t>(a_nrsb, b_nrsb);
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
  /* TODO */
  assert(a->elem > 0 && (float)(*a) < 1.0 &&
         a->exponent >= -31);

  int32_t a_int = safe_shift_by(a->elem, (-31) - a->exponent);
  assert(a_int > 0);
  out->nrsb = lrsb(a_int);
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
  assert(abs(static_cast<float>(*s)) < 1.0 &&
         static_cast<float>((*b)[dim-1]) == 1.0);
  // TODO.
  // [a] see if we have to rshift b; do it if so.
  // [b] compute rshift on multiplication, will be exponent of s.
}


void get_elem(IntVec<int32_t> *a, int i, IntScalar<int32_t> *value) {
  assert(static_cast<unsigned int>(i) < static_cast<unsigned int>(a->dim));
  value->exponent = a->exponent;
  value->elem = a->data[i];
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

template <typename I> void IntVec<I>::check() const {
  int recomputed_nrsb = array_lrsb(data, dim);
  assert(nrsb == recomputed_nrsb);
  if (nrsb == sizeof(I)*8 - 1 && data_is_zero(data, dim)) {
    assert(exponent == kExponentOfZero);
  } else {
    assert(exponent != kExponentOfZero);
  }
}


}  // int_math.h

#endif /* include guard */

