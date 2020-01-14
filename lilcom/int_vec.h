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
#ifndef NDEBUG
#include <string>  /* for std::string, for debug. */
#include <sstream>  /* for std::ostringstream, for debug. */
#endif


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
  IntScalar<I> operator [] (int i) const { return IntScalar<I>(data[i], exponent); }

  inline void set_nrsb(int nrsb_in) {
    nrsb = nrsb_in;
    if (nrsb + 1 == (sizeof(I)*8) && data_is_zero(data, dim))
      exponent = kExponentOfZero;
  }

  inline operator std::string () const {  // for debug.
    std::ostringstream os;
    os << " [ ";
    for (int i = 0; i < dim; i++)
      os << (double)(*this)[i] << ' ';
    os << "] ";
    return os.str();
  }


  /* copy constructor (allocates new data as a copy of
     existing object.  Currently just used in testing.*/
  IntVec(const IntVec &other);

  inline void set_nrsb() {
    set_nrsb(array_lrsb(data, dim));
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
    delete [] data;
    dim = d;
    data = new I[d];
    exponent = kExponentOfZero;
    nrsb = 8 * sizeof(I) - 1;
    set_data_zero(data, dim);
  }
};


#ifndef NDEBUG
template <typename I>
inline std::ostream &operator << (std::ostream &os, const IntVec<I> &s) {
  return os << (std::string)s;
}
#endif



template <typename I>
inline void copy(const IntVec<I> *a, IntVec<I> *b) {
  assert(b->dim == a->dim);
  b->exponent = a->exponent;
  b->nrsb = a->nrsb;
  for (int i = 0; i < a->dim; i++)
    b->data[i] = a->data[i];
}

inline void copy(const IntVec<int64_t> *a, IntVec<int32_t> *b) {
  assert(b->dim == a->dim);
  int a_lrsb = a->nrsb,
      rshift = 32 - a_lrsb;  /* e.g. if a_lrsb == 0, shift right by 32 and will
                              * still have lrsb == 0. */

  if (rshift > 0) {
    b->exponent = a->exponent + rshift;
    b->nrsb = 0;
    int dim = a->dim;
    for (int i = 0; i < dim; i++)
      b->data[i] = static_cast<int32_t>(a->data[i] >> rshift);
  } else {
    b->exponent = a->exponent;
    b->nrsb = a_lrsb - 32;
    int dim = a->dim;
    for (int i = 0; i < dim; i++)
      b->data[i] = static_cast<int32_t>(a->data[i]);
  }
  b->check();
}



/*
  Computes dot product between an int32 and an int16 vector
 */
inline void compute_dot_product(IntVec<int32_t> *a, IntVec<int16_t> *b,
                                IntScalar<int64_t> *out) {
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
inline void compute_dot_product(IntVec<int32_t> *a, IntVec<int32_t> *b,
                                IntScalar<int64_t> *out) {
  assert(a->dim == b->dim);
  int dim = a->dim,
      rshift = extra_bits_from_factor_of(dim) - a->nrsb - b->nrsb;
  if (rshift > 0) {
    out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, 1>(
        a->data, b->data, dim, rshift);
    out->exponent = a->exponent + b->exponent + rshift;
  } else {
    out->elem = compute_raw_dot_product<int32_t, int32_t, int64_t, int64_t, 1>(
        a->data, b->data, dim);
    out->exponent = a->exponent + b->exponent;
  }
}

/*
  Computes the dot product that in NumPy would be
     np.dot(a[a_offset:a_offset+dim], b[b_offset:b_offset+dim])
  i.e. dot product with offsets.
 */
inline void compute_dot_product(int dim,
                                const IntVec<int32_t> *a, int a_offset,
                                const IntVec<int32_t> *b, int b_offset,
                                IntScalar<int64_t> *out) {
  assert(dim >= 0 && a_offset >= 0 && b_offset >= 0 &&
         dim + a_offset <= a->dim && dim + b_offset <= b->dim);
  assert(a->dim == b->dim);

  int rshift = extra_bits_from_factor_of(dim) - a->nrsb - b->nrsb;
  if (rshift > 0) {
    out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, 1>(
        a->data + a_offset, b->data + b_offset, dim, rshift);
    out->exponent = a->exponent + b->exponent + rshift;
  } else {
    out->elem = compute_raw_dot_product<int32_t, int32_t, int64_t, int64_t, 1>(
        a->data + a_offset, b->data + b_offset, dim);
    out->exponent = a->exponent + b->exponent;
  }
}



/*
  Computes the dot product that in NumPy would be
     np.dot(a[a_offset:a_offset+dim], np.flip(b[b_offset:b_offset+dim]))
  i.e. dot product with offsets and with one argument flipped.
  Which one is flipped doesn't matter.
 */
inline void compute_dot_product_flip(int dim,
                                     const IntVec<int32_t> *a, int a_offset,
                                     const IntVec<int32_t> *b, int b_offset,
                                     IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);
  assert(a->dim == b->dim);

  int rshift = extra_bits_from_factor_of(dim) - a->nrsb - b->nrsb;
  if (rshift > 0) {
    out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, -1>(
        a->data + a_offset, b->data + b_offset + dim - 1, dim, rshift);
    out->exponent = a->exponent + b->exponent + rshift;
  } else {
    out->elem = compute_raw_dot_product<int32_t, int32_t, int64_t, int64_t, -1>(
        a->data + a_offset, b->data + b_offset + dim - 1, dim);
    out->exponent = a->exponent + b->exponent;
  }
}




/*
  Computes the dot product that in NumPy would be
     np.dot(a[a_offset:a_offset+dim], np.flip(b[b_offset:b_offset+dim]))
  Note: it does not actually matter which of the two arguments is flipped.
 */
/* Not needed just now.
inline void compute_dot_product_flip(int dim,
                         IntVec<int16_t> *a, int a_offset,
                         IntVec<int16_t> *b, int b_offset,
                         IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);
  out->elem = compute_raw_dot_product<int16_t, int16_t, int32_t, int64_t, -1>(
      a->data + a_offset, b->data + b_offset + dim - 1, dim);
  out->exponent = a->exponent + b->exponent;
}
*/

/*
  Not needed just now.
inline void compute_dot_product_flip(int dim,
                         IntVec<int32_t> *a, int a_offset,
                         IntVec<int32_t> *b, int b_offset,
                         IntScalar<int64_t> *out) {
  assert(dim + a_offset <= a->dim && dim + b_offset <= b->dim);

  int rshift = num_bits_except_sign(dim);

  out->elem = compute_raw_dot_product_shifted<int32_t, int64_t, -1>(
      a->data + a_offset, b->data + b_offset + dim - 1, dim, rshift);
  out->exponent = a->exponent + b->exponent;
}

*/


/*
  This function does what in NumPy would be:

     b[a_offset:a_offset+dim] += scalar * a[b_offset:b_offset+dim]

   a and b must be different pointers.

   Note: we assume that b already contains data, which may need to be shifted.
   CAUTION: this function makes an assumption about the input,
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
  a->check();
  b->check();
#ifndef NDEBUG
  for (int i = 0; i < b_offset; i++) {
    assert(b->data[i] == 0);
  }
  for (int i = b_offset + dim; i < b->dim; i++) {
    assert(b->data[i] == 0);
  }
#endif

  if (scalar->elem == 0 || a->exponent == kExponentOfZero) {
    /* nothing to do if scalar == 0 or a is zero (note: we set
       exponent=kExponentOfZero if an IntVec is zero); returning now avoids the
       possibility of unnecessarily shifting b. */
    return;
  }

  int prod_exponent = a->exponent + scalar->exponent,
      b_exponent = b->exponent,
      prod_nrsb = lrsb_of_prod<int32_t>(a->nrsb, lrsb(scalar->elem));

  /* Aim for both nrsb's to be 1.  We'll right shift by
       rshift = (1 - nrsb)
     (if negative, would be a left shift);
     and the right shift increases the exponent.
   */

  /* out_nrsb_as_b is the highest possible nrsb of the output if we shifted
     it to have the same exponent as b.  The - 1 is because we add two
     things together, which requires another bit (so one less redundant
     bit).
   */
  int out_nrsb_as_b = int_math_min(b->nrsb,
                                   prod_nrsb + b_exponent - prod_exponent) - 1,
      out_exponent = b_exponent - out_nrsb_as_b,
      prod_rshift = out_exponent - prod_exponent,
      b_rshift = out_exponent - b_exponent;

  int64_t scalar_elem = scalar->elem;
  if (prod_rshift < 0) {
    assert(-prod_rshift <= lrsb(scalar_elem));
    scalar_elem <<= -prod_rshift;
    prod_rshift = 0;
  } else if (prod_rshift >= 63) {
    /* the output would not be affected, as we'd be adding zero. */
    return;
  }

  b->exponent = out_exponent;

  if (b_rshift == 0) {
    b->set_nrsb(raw_add_product(
        dim, a->data + a_offset,
        scalar_elem,  b->data + b_offset,
        prod_rshift));
  } else if (b_rshift > 31) {
    /* completely right-shift b away */
    b->set_nrsb(raw_copy_product(
        dim, a->data + a_offset,
        scalar_elem,  b->data + b_offset,
        prod_rshift));
  } else if (b_rshift > 0) {
    assert(prod_rshift == out_exponent - prod_exponent || prod_rshift == 0);
    assert(b_rshift == out_exponent - b_exponent);


    b->set_nrsb(raw_add_product_and_rshift(
        dim, a->data + a_offset,
        scalar_elem,  b->data + b_offset,
        prod_rshift, b_rshift));
  } else {
    /* b_rshift < 0, so left shift. the way we obtained the shift
       ensures that it cannot exceed 30. */
    assert(b_rshift >= -30);
    b->set_nrsb(raw_add_product_and_lshift(
        dim, a->data + a_offset,
        scalar_elem,  b->data + b_offset,
        prod_rshift, -b_rshift));
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


/*  does c := a * b  (elementwise).  `c` must not be the same as either of
    the other two args. */
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
  c->exponent = a->exponent + b->exponent + right_shift;
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
  (The reason we make it a^(n+1) not a^n is to avoid a 1.0 appearing
  there, so we can use a smaller exponent).

  This is not particularly efficient/optimized as it's just used
  in initializion code right now.
 */
inline void init_vec_as_powers(const IntScalar<int32_t> *a,
                               IntVec<int32_t> *out) {
  assert(a->elem > 0 && (float)(*a) < 1.0 &&
         a->exponent >= -31);

  int dim = out->dim;
  out->exponent = -31;
  int32_t a_elem = safe_shift_by(a->elem, out->exponent - a->exponent);
  out->nrsb = lrsb(a_elem);  /* probably 0. */
  out->data[0] = a_elem;

  for (int n = 2; n <= dim; n++) {
    /* n is the power of `a` */
    int prev_n1 = n / 2,
        prev_n2 = n - prev_n1;
    IntScalar<int32_t> a1 = (*out)[prev_n1 - 1],
        a2 = (*out)[prev_n2 - 1],
        prod;
    multiply(&a1, &a2, &prod);
    out->data[n - 1] = safe_shift_by(prod.elem, out->exponent - prod.exponent);
  }
}




/*
  This is some rather special-purpose code needed in the Toeplitz solver.
  it does what in NumPy would be:
      b[-(n+1):-1] += s * np.flip(b[-n:])
  (for 0 < n < b.shape[0]-1).
  This amounts to:
     b[-(n+1)] += s * b[-1].
     b[-n:-1] += s * np.flip(b[-n:-1])
  We also make use of several facts which are true in the context
  in which we need it, which are that:
   --  abs(s) < 1.0   [in the algorithm, this is nu_n]
   --  b[-1] = 1.0.  XX no we dont use this.
   --  The only nonzero elements of b at entry are those in b[-n:].

  This function is probably optimized a little more than it needs to be.
  It's really not a very difficult thing to do; this function is long
  because we try to handle various cases as efficiently as possible.
 */
inline void special_reflection_function(int n, const IntScalar<int32_t> *s,
                                        IntVec<int32_t> *b) {
  int dim = b->dim;
  assert(fabs(static_cast<float>(*s)) < 1.0);
  int32_t *bdata = b->data;


#ifndef NDEBUG
  for (int i = 0; i + n < dim; i++)
    assert(bdata[i] == 0.0);
#endif

  b->check();
  if (b->nrsb > 1 && b->exponent != kExponentOfZero) {
    /* This should rarely happen, and if it does, it will usually be
       when the used part of the vector is still quite small, so we
       pre-shift instead of shifting in the loop. */
    int lshift = b->nrsb - 1;
    b->exponent -= lshift;
    for (int i = dim - n; i < dim; i++) {
      /* We only need to left-shift this part because the user asserts that
         the remainder is zero. */
      bdata[i] <<= lshift;
    }
    b->nrsb = 1;
  }
  b->check();


  /* we shift s->elem to have a known exponent of -31 so that we can treat its
     exponent as fixed; that way, we don't have to shift by an unknown value in
     the loops. */
  assert(s->exponent - lrsb(s->elem) <= -31);  /* since abs(s) is less than one. */

  int32_t s_elem_shifted = safe_shift_by(s->elem, (-31) - s->exponent);

  if (b->nrsb == 0) {
    b->exponent++;
    /* this algorithm requires at least one spare bit, since things can increase
       by a factor that's strictly less than 2.  So we'll have to right-shift
       b by one bit as we go.  We have to right shift the products
       b->data[i] * s_elem_shifted by 31 because of s's exponent, and then
       1 extra because we're right-shifting b, so it's an even 32.
       (nice because in machine code there's no need to explicitly shift.)
    */


    /* b[-(n+1)] = s * b[-1].   [we don't need to do +=, since it was zero at entry.]
       Record the nrsb.
     */
    int nrsb = lrsb(
        (bdata[b->dim - (n+1)] = (s_elem_shifted * (int64_t)bdata[b->dim - 1]) >> 32));
    /* Shift the last element of b to account for the new exponent (it is not
       otherwise modified by this function). */
    nrsb = int_math_min(nrsb,
                        lrsb(b->data[b->dim - 1] >>= 1));
    /* Now,
       b[-n:-1] += s * np.flip(b[-n:-1])
    */
    for (int i = 0; i < (n-1) / 2; i++) {
      /* We are taking elements that have opposite positions in the
         range [dim-n .. dim-2]. */
      int j = dim - n + i,
          k = dim - 2 - i;
      int32_t bj = bdata[j], bk = bdata[k];
      nrsb = int_math_min(nrsb, lrsb(
          (bdata[k] = (bk >> 1) + ((bj * (int64_t)s_elem_shifted) >> 32))));
      nrsb = int_math_min(nrsb, lrsb(
          (bdata[j] = (bj >> 1) + ((bk * (int64_t)s_elem_shifted) >> 32))));
    }
    if (n % 2 == 0 && n > 0) {
      /* the following expression could also equivalently be written as:
         int j = dim - (n + 2) / 2  */
      int j = dim - ((n + 2) >> 1);
      int32_t bj = bdata[j];
      nrsb = int_math_min(nrsb, lrsb(
          (bdata[j] = (bj >> 1) + ((bj * (int64_t)s_elem_shifted) >> 32))));
    }
    b->set_nrsb(nrsb);
  } else {
    assert(b->exponent == kExponentOfZero || b->nrsb == 1);  /* We ensured this above. */

    /* There is no need to right-shift the elements of bdata as we go.  The most
       obvious implementation of the following would require right-shifting the
       product by 31 in the loop, but if s_elem_shifted is small enough, we can
       left-shift it by one now and then right-shift by exactly 32 in the loop
       (which is no right shift at all, since it's just a matter of selecting
       the appropriate register).
    */
    if (s_elem_shifted < (1<<30) && s_elem_shifted >= -(1<<30)) {
      s_elem_shifted = safe_shift_by(s->elem, (-32) - s->exponent);
      /* Note: the rest of the code here is like the code following
         `if (b->nrsb == 0) { ... ` but with the comments taken out and
         taking out the right-shift by one of the elements of b. */

      int nrsb = int_math_min(
          lrsb(bdata[b->dim - 1]),
          lrsb(bdata[b->dim - (n+1)] = (s_elem_shifted * (int64_t)bdata[b->dim - 1]) >> 32));

      for (int i = 0; i < (n-1) / 2; i++) {
        int j = dim - n + i,
            k = dim - 2 - i;
        int32_t bj = bdata[j], bk = bdata[k];
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[k] = bk + ((bj * (int64_t)s_elem_shifted) >> 32))));
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[j] = bj + ((bk * (int64_t)s_elem_shifted) >> 32))));
      }
      if (n % 2 == 0 && n > 0) {
        int j = dim - ((n + 2) >> 1);
        int32_t bj = bdata[j];
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[j] = bj + ((bj * (int64_t)s_elem_shifted) >> 32))));
      }
      b->set_nrsb(nrsb);
      b->check();
    } else {
      int nrsb = int_math_min(
          lrsb(bdata[b->dim - 1]),
          lrsb(bdata[b->dim - (n+1)] = (s_elem_shifted * (int64_t)bdata[b->dim - 1]) >> 31));
      for (int i = 0; i < (n-1) / 2; i++) {
        int j = dim - n + i,
            k = dim - 2 - i;
        int32_t bj = bdata[j], bk = bdata[k];
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[k] = bk + ((bj * (int64_t)s_elem_shifted) >> 31))));
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[j] = bj + ((bk * (int64_t)s_elem_shifted) >> 31))));
      }
      if (n % 2 == 0 && n > 0) {
        int j = dim - ((n + 2) >> 1);
        int32_t bj = bdata[j];
        nrsb = int_math_min(nrsb, lrsb(
            (bdata[j] = bj + ((bj * (int64_t)s_elem_shifted) >> 31))));

      }
      b->set_nrsb(nrsb);
    }
  }
  // TODO.
  // [a] see if we have to rshift b; do it if so.
  // [b] compute rshift on multiplication, will be exponent of s.
}


/*
  Set the only nonzero element of the vector 'v' to the scalar 's', i.e.
   v[i] = s.
  The point of being the only nonzero element is that we are free to set the
  exponent to whatever we want.  It is the caller's job to make sure that
  no other elements of 'v' are nonzero.
 */
inline void set_only_nonzero_elem_to(const IntScalar<int32_t> *s, int i,
                                     IntVec<int32_t> *v) {
  assert(static_cast<unsigned int>(i) < static_cast<unsigned int>(v->dim));
  /* The following is just a spot-check that other elements of the vector are zero. */
  assert(v->data[(i + 1) % v->dim] == 0);
  v->data[i] = s->elem;
  v->exponent = s->exponent;
  v->set_nrsb(lrsb(s->elem));
}


/*
  This function gets the exponent that we'd want to use when combining two
  quantities a and b (but not adding them together; just selecting elements from each)...

  It tries to avoid precision loss at all costs, but doesn't left shift by any
  more than necessary.

  This function assumes that neither a's nor b's data is zero.

*/
inline int get_exponent(int a_exponent, int a_nrsb, int b_exponent, int b_nrsb) {
  /* a_exponent0 and b_exponent0 are the exponents if their nrsb's were zero,
     i.e. if a and b were left-shifted to have no redundant sign bits.
  */
  int a_exponent0 = a_exponent - a_nrsb,
      b_exponent0 = b_exponent - b_nrsb,
      out_exponent = int_math_max(a_exponent0, b_exponent0),
      a_rshift = out_exponent - a_exponent,
      b_rshift = out_exponent - b_exponent;
  if (a_rshift < 0 && b_rshift < 0) {
    /* There is no point left-shifting both of them.. */
    out_exponent -= int_math_max(a_rshift, b_rshift);
    assert(out_exponent - a_exponent == 0 ||
           out_exponent - b_exponent == 0);
  }
  return out_exponent;
}

/* Caution: this will take time O(s->dim) if we have to recompute the nrsb.
 */
inline void set_elem_to(const IntScalar<int32_t> *s, int i, IntVec<int32_t> *v) {
  assert(static_cast<unsigned int>(i) < static_cast<unsigned int>(v->dim));
  int v_exponent = v->exponent, s_exponent = s->exponent,
      s_nrsb = lrsb(s->elem);
  int old_elem_nrsb = lrsb(v->data[i]);

  if (s->elem == 0) {
    v->data[i] = 0;
    if (v->nrsb == old_elem_nrsb)
      v->set_nrsb();
    return;
  } else if (v->exponent == kExponentOfZero) {
    set_only_nonzero_elem_to(s, i, v);
    return;
  }

  int out_exponent = get_exponent(v_exponent, v->nrsb, s_exponent, s_nrsb);

  safe_shift_array_by(v->data, v->dim, out_exponent - v_exponent);
  v->data[i] = safe_shift_by(s->elem, out_exponent - s_exponent);

  int v_nrsb_shifted = v->nrsb + out_exponent - v_exponent,
      s_nrsb_shifted = s_nrsb + out_exponent - s_exponent;
  if (s_nrsb_shifted <= v_nrsb_shifted) {
    v->nrsb = s_nrsb_shifted;  /* new elem has lowest nrsb */
    assert(s_nrsb_shifted == lrsb(v->data[i]));
  } else if (v->nrsb != old_elem_nrsb) {
    v->nrsb = v_nrsb_shifted;  /* overwritten elem did not determine the nrsb of
                                * v. */
  } else {
    v->set_nrsb();  /* we have to recompute the nrsb. */
  }
  v->exponent = out_exponent;
}


template <typename I>
inline void zero_int_vector(IntVec<I> *v) {
  for (int i = 0; i < v->dim; i++)
    v->data[i] = 0;
  v->nrsb = 8 * sizeof(I) - 1;
  v->exponent = kExponentOfZero;
  v->check();
}

#ifndef NDEBUG
template <typename I> void IntVec<I>::check() const {
  if (dim == 0) {
    assert(data == 0);
    return;
  } else {
    assert(dim > 0);
    assert(nrsb == array_lrsb(data, dim));
    if (nrsb == sizeof(I)*8 - 1 && data_is_zero(data, dim)) {
      assert(exponent == kExponentOfZero);
    } else {
      assert(exponent != kExponentOfZero);
    }
  }
}
#endif /* #ifndef NDEBUG */

template <typename I>
IntVec<I>::IntVec(const IntVec<I> &other) {
  if (other.dim == 0) {
    dim = 0;
    data = NULL;
  } else {
    dim = other.dim;
    data = new I[dim];
    nrsb = other.nrsb;
    exponent = other.exponent;
    for (int i = 0; i < dim; i++)
      data[i] = other.data[i];
  }
  check();  // TODO: remove
}

/*
  Computes and returns the LPC prediction for a single sample of the signal.

     @param [in] signal  Pointer to the sample for which we want the
                    prediction.  It is omly samples signal[-1], signal[-2], ...,
                    signal[-(lpc_coeffs->dim-1)] that are inspected.
     @return      Returns the predicted signal value, truncated to
                  fit in int16_t.
 */
inline int16_t compute_lpc_prediction(const int16_t *signal,
                                      const IntVec<int32_t> *lpc_coeffs) {
  int num_lpc_coeffs = lpc_coeffs->dim;
  int right_shift = -lpc_coeffs->exponent;
  /* round_to_nearest_offset is something we add to the signal so that
     it will do rounding-to-nearest instead of rounding-down.
     (Note: right-shift >> will have the effect of rounding even
     negative numbers down for any normal machine, although actually
     according to the C++ standard the behavior is undefined.
   */
  int64_t round_to_nearest_offset = (((int64_t)1) << (right_shift - 1));
  /* The casting to unsigned is just to avoid a compiler warning. */
  int32_t ans = static_cast<int32_t>(
      static_cast<uint64_t>(
          (round_to_nearest_offset +
          compute_raw_dot_product<int32_t, int16_t, int64_t, int64_t, -1>(
              lpc_coeffs->data, signal - 1, num_lpc_coeffs)) >> right_shift));
  if (ans == static_cast<int16_t>(ans)) {
    return ans;  /* Fits within int16_t */
  } else  {
    /* truncate */
    if (ans > 0) {
      return 32767;
    } else {
      return -32768;
    }
  }
}




}  // namespace int_math

#endif /* include guard */

