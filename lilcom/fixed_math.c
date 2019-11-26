#include "fixed_math.h"
#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#define FIXED_MATH_TEST 1

#ifdef FIXED_MATH_TEST
#include <math.h>
#endif



#define FM_TARGET_SIZE 50  /* TARGET_SIZE defines how many bits we want to be set
                         * in these int64_t's when we have a choice.  It should be
                         * less than 64 and >= 32.  Most of the time we only get
                         * 32 bits of precision with these calculations, because
                         * we shift right to fit within 32 bits prior to
                         * multiplication. (I.e. we don't do 4 separate
                         * multiplies to keep the full 64 bits of precision. */

#define FM_MAX_SIZE 60  /* TARGET_SIZE defines how many bits we are comfortable
                           going up to if doing so would save operations such
                           as shifting a region. */

#define FM_MIN_SIZE 38  /* The minimum `size` we allow if right shifting... this
                           is equivalent to the number of bits of precision we
                           keep.  This should be >= 32, because when we multiply
                           in the typical case we'll only keep 32 bits of precision. */


static int fm_num_bits[32] = {0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};


#define FM_MAX(a,b) ((a) > (b) ? (a) : (b))

/**
   Returns (shift >= 0 ? (i<<shift) :  (i>>shift)),
   i.e. interpreting `shift` as right shift if positive,
   left shift if negative.
 */
inline static int ShiftInt64(const int64_t i, int shift) {
  return (i >= 0 ? i << shift : i >> (-shift));
}


#ifndef NDEBUG
void PrintRegion64(Region64 *region) {
  fprintf(stderr, "{ Region64, dim = %d, exponent = %d, size = %d, data = [ ",
          region->dim, region->exponent, region->size);
  for (int i = 0; i < region->dim; i++)
    fprintf(stderr, "%lld ", region->data[i]);
  fprintf(stderr, "] }\n");
}

void PrintVector64(Vector64 *vec) {
  fprintf(stderr, "{ Vector64, dim = %d, stride = %d, data = [ ",
          vec->dim, vec->stride);
  float factor = pow(2.0, vec->region->exponent);
  for (int i = 0; i < vec->dim; i++) {
    float f = vec->data[i * vec->stride] * factor;
    fprintf(stderr, "%f ", f);
  }
  fprintf(stderr, "] }\n");
}

void PrintMatrix64(Matrix64 *mat) {
  fprintf(stderr, "{ Matrix64, num-rows = %d, row-stride = %d, num-cols = %d, col-stride = %d,\n"
          "  data = [ ", mat->num_rows, mat->row_stride, mat->num_cols, mat->col_stride);
  float factor = pow(2.0, mat->region->exponent);
  for (int i = 0; i < mat->num_rows; i++) {
    const int64_t *row_data = mat->data + i * mat->row_stride;
    for (int j = 0; j < mat->num_cols; j++) {
      float f = row_data[i * mat->col_stride] * factor;
      fprintf(stderr, "%f ", f);
    }
    if (i + 1 < mat->num_rows)
      fprintf(stderr, "\n    ");
  }
  fprintf(stderr, "] }\n");
}

#endif

#ifndef NDEBUG
/** Checks that the region currently has the size that it should have (or
 * greater); dies if not.. */
void CheckRegion64Size(const Region64 *r) {
  Region64 r_copy = *r;
  SetRegion64Size(r_copy.size, &r_copy);
  assert(r->size >= r_copy.size && "Region had invalid size.");
}

void CheckScalar64Size(const Scalar64 *scalar_in) {
  assert(scalar_in->size == FindSize(scalar_in->data, scalar_in->size));
}
#else
inline static void CheckRegion64Size(const Region64 *r) { }
inline static void CheckScalar64Size(const Scalar64 *r) { }
#endif


/* If shift > 0 shifts right by `rshift`; if shift < 0 shift left by -rshift. */
inline static void ShiftRegion64(int rshift, Region64 *region) {
  if (rshift > 0) ShiftRegion64Right(rshift, region);
  else if (rshift < 0) ShiftRegion64Left(-rshift, region);
}


/**
   This function, used to get right-shifts for arguments to a multiplication,
   chooses the shifts that will lose the least precision while ensuring
   that after multiplying a and b there is no overflow
   (i.e. that a_size + b_size - a_shift - b_shift < 64).
*/
inline static void GetInputShiftsFor2ArgMultiply(
    int a_size, int b_size,
    int *a_shift, int *b_shift) {
  if (a_size + b_size < 64) {
    *a_shift = 0;
    *b_shift = 0;
  } else if (a_size >= 32 && b_size >= 32) {
    // We'll definitely have to shift both of them right.  We have 63 bits
    // and we can't divide them quite evenly...
    *a_shift = a_size - 32;
    *b_shift = b_size - 31;
  } else {
    // OK, one of them is < 32, but their sum is >= 64,
    // so the other one must be > 32.  We only have to
    // shift one of them.
    int shift = a_size + b_size - 63;
    if (a_size > b_size) {
      *a_shift = shift;
      *b_shift = 0;
    } else {
      *a_shift = 0;
      *b_shift = shift;
    }
  }
}



/**
   [Caution: in most cases you will need GetShiftsFor2ArgMultiplyAndAdd()
   instead, because that takes care of situations where the destination
   either is being added to, or might be part of a larger region which
   would have an exponent we need to worry about.

   Get shifts for a 2-arg multplication where the output's exponent can be
   freely chosen without requiring extra work (typically when we are writing to
   a scalar).  It specifies an operation of the form:

    out = ((a >> a_shift) * (b >> b_shift)) >> post_shift.

   and possibly, after that, a summation over a dimension `dim` where
   dim_size == FindSize(dim, ...).  (Set dim_size to 0 if there is no summation).
   We need to keep the multiplication from overflowing, which requires:

     (a_size - a_shift) + (b_size - b_shift)  <= 63

   subject to the larger of those two target sizes being as large as possible.

   We also want to choose the smallest post_shift >= 0 so that the size of the
   product after shifting and summing is <= FM_TARGET_SIZE.


  This comes down to:
      (a_size - a_shift) + (b_size - b_shift) + dim_size - post_shift <= FM_TARGET_SIZE.

     @param [in] a_size  Size of one of the operands.   See doc for CheckSize()
                        for definition.  Must be in [0,63].
     @param [in] b_size  Size of the other operand.     See doc for CheckSize()
                        for definition.  Must be in [0,63.]
     @param [in] dim_size  If the products will immediately be summed over dimension
                        `dim`, then the result of calling FindSize(dim, ...).
                        Otherwise, zero.
     @param [out] a_shift, b_shift, post_shift
                        These are the shifts this function outputs... will
                        be used in an equation like:
                output_data = ((a_data >> a_shift) * (b_data >> b_shift)) >> pos_shift.
     @param [out] final_size_guess
                        This function will write to here a guess at the likely
                        size of the resulting summation (or single shifted product,
                        if there was no summation)
*/
inline static void GetShiftsFor2ArgMultiply(int a_size, int b_size, int dim_size,
                                            int *a_shift, int *b_shift,
                                            int *post_shift, int *final_size_guess) {
  GetInputShiftsFor2ArgMultiply(a_size, b_size,
                                a_shift, b_shift);
  int prod_size = a_size - *a_shift + b_size - *b_shift;
  assert(prod_size < 64);
  if (prod_size + dim_size > FM_TARGET_SIZE) {
    *post_shift = prod_size + dim_size - FM_TARGET_SIZE;
  } else {
    *post_shift = 0;
  }
  /* Note: dim_size / 2 is a compromise between 0 and dim_size... most of the
     time the things we're summing won't all have the same sign, and dim_size /
     2 has some mathematical justification if we assume the things being summed
     are uncorrelated. */
  *final_size_guess = a_size + b_size + (dim_size / 2) - *a_shift - *b_shift - *post_shift;
}

/*
  This function is used when you are either doing something like:
      y += x
  or just doing
      y := x
  when y might be part of a larger region (so we'd need to worry about its
  exponent).

  They key values it gives are `in_shift`, `out_shift`, so the
  actual computation you'll do is something like:

  ShiftRegion64(out_shift, out); [note: `out_shift` is interpreted as a
                                  right shift]
  if (in_shift >= 0)
    y[i] := y[i] + (x[i] >> in_shift)
  else
    y[i] := y[i] + (x[i] << -in_shift)

  [And if dim_size != 0, this means there is a summation of dimension `di` done
  on the x[i] after the shift, where dim_size == FindSize(dim, ...).]


  The output values must satisfy:
    in_exponent + in_shift == out_exponent + out_shift
  And to avoid overflow:
    max(in_size - in_shift, out_size - out_shift) < 63
  ... and also, if out_shift != 0, we require:
    max(in_size - in_shift, out_size - out_shift) <= FM_MAX_SIZE
  (we avoid shifting `out` if at all possible, since it might
  include a whole region.)
  Also, we never left-shift both args (i.e. never in_shift < 0 and
  out_shift < 0).

     @param [in] in_size  Size of input data (x above), as found by FindSize() or
                        stored in its region
     @param [in] in_exponent  Exopnent of input data
     @param [in] out_size  Size of output data (y above), as found by FindSize() or
                        stored in its region
     @param [in] out_exponent  Exponent of output data
     @param [in] dim_size   If >0, assume there is a summation over the x's over a
                           dimension `dim` with dim_size = FindSize(dim, ...).
     @param [out] in_shift  Shift to be applied to input data, positive for right shift.
     @param [out] out_shift  Shift to be applied to output data, positive for right shift.
     @param [out] final_size_guess  Guess at the size of the final output after
                            the operations mentioned above, to be used as the
                            1st arg to FindSize().
 */
inline static void GetShiftsForAdd(int in_size, int in_exponent,
                                   int out_size, int out_exponent,
                                   int dim_size,
                                   int *in_shift, int *out_shift,
                                   int *final_size_guess) {
  int possible_in_shift = out_exponent - in_exponent;
  /* sum_max_size is the maximum possible size of the sum, assuming we only
     shift the input (the + 1 accounts for it getting larger when we add the two
     things. */
  int sum_max_size = FM_MAX(out_size, in_size + dim_size - possible_in_shift) + 1,
      sum_min_size = FM_MAX(out_size, in_size +  possible_in_shift),
      sum_guess_size = FM_MAX(out_size, in_size + (dim_size/2) - possible_in_shift);

  if (sum_max_size < 64 && out_size != 0 &&
      (possible_in_shift <= 0 || sum_min_size >= FM_MIN_SIZE)) {
    /* If we can just shift the input, not shift the output, and not risk
       overflow or getting excessively low precision, then we do it.  (Shifting
       `out` might require shifting the entire region).  Reasons for the conditions:

       `sum_max_size < 64` because if it's 64 it's not safe to do this.

       `out_size != 0` because if out_size = 0 there is no reason going out of
       our way to avoid shifting `out`.  (The ShiftRegion operation would do no work).

       `(possible_in_shift <= 0 || sum_min_size >= FM_MIN_SIZE)` because...

          well, in the normal case we would never want shift the input right if
          it made the final magnitude smaller than FM_MIN_SIZE, as that would
          cause too much roundoff error (hence we might want to require
          sum_max_size >= FM_MIN_SIZE); but we use sum_min_size (which lacks the
          +dim_size term) instead because we are worried about the worst
          reasonable case for roundoff where the terms don't really add up.  (We
          can't worry about the almost-complete-cancellation case, that would
          never be workable.)  However, we can ignore this requirement if
          possible_in_shift <= 0 because in that case we are not losing any
          precision on the input.
      */
    *in_shift = possible_in_shift;
    *out_shift = 0;
    *final_size_guess = sum_guess_size;
  }

  /* Shift so that the max possible size becomes FM_TARGET_SIZE.  `shift_offset`
     (could be either sign) is the amount by which we'll right-shift both the
     input and output, versus the baseline where input is shifted by
     possible_in_shift.
  */
  int shift_offset = sum_max_size - FM_TARGET_SIZE;

  if (shift_offset < 0 && shift_offset + possible_in_shift < 0) {
    shift_offset = (possible_in_shift > 0 ? -possible_in_shift : 0);
  }


  *in_shift = possible_in_shift + shift_offset;
  *out_shift = shift_offset;
  *final_size_guess = sum_guess_size - shift_offset;
}

/**
   GetShiftsForAssign is the same as GetShiftsForAdd except a "+ 1" in the
   expression for sum_max_size is absent, because overflowing a power of 2
   cannot occur when we are just setting a value.  All comments have been
   removed from this code; please refer to the docs for GetShiftsForAdd()
 */
inline static void GetShiftsForAssign(int in_size, int in_exponent,
                                      int out_size, int out_exponent,
                                      int dim_size,
                                      int *in_shift, int *out_shift,
                                      int *final_size_guess) {
  int possible_in_shift = out_exponent - in_exponent;
  int sum_max_size = FM_MAX(out_size, in_size + dim_size - possible_in_shift),
      sum_min_size = FM_MAX(out_size, in_size +  possible_in_shift),
      sum_guess_size = FM_MAX(out_size, in_size + (dim_size/2) - possible_in_shift);

  if (sum_max_size < 64 && out_size != 0 &&
      (possible_in_shift <= 0 || sum_min_size >= FM_MIN_SIZE)) {
    *in_shift = possible_in_shift;
    *out_shift = 0;
    *final_size_guess = sum_guess_size;
  }
  int shift_offset = sum_max_size - FM_TARGET_SIZE;
  if (shift_offset < 0 && shift_offset + possible_in_shift < 0) {
    shift_offset = (possible_in_shift > 0 ? -possible_in_shift : 0);
  }
  *in_shift = possible_in_shift + shift_offset;
  *out_shift = shift_offset;
  *final_size_guess = sum_guess_size - shift_offset;
}


/**
   Get shifts for a 2-arg multiply-and-add, that is conceptually of the form:

         out += a * b

   In practice, it will be an operation of the following form (where
   a_shift, b_shift, out_shift and post_shift are the outputs of this function):

     ShiftRegion64(out_shift, &out_region);

     if (a_shift >= 0)
         out += ((a >> a_shift) * (b >> b_shift)) >> post_shift.
     else  # in this case b_shift, post_shift will be zero.
         out += ((a << -a_shift) * b)

   [Note: you could change that if-condition to
    if (a_shift > 0 || b_shift > 0 || post_shift > 0)
    and it would be a bit more efficient, at least for large input,
    by avoiding unnecessary shifts.]



  The job of this function is to compute out_shift, a_shift, b_shift, post_shift.

  There are various constraints.  Firstly to make the exponents match up, we need:

     a_exponent + b_exponent + a_shift + b_shift + post_shift == out_exponent + out_shift

  To prevent overflow in the multiplication we require:

     (a_size - a_shift) + (b_size - b_shift) < 64

  To prevent overflow after any summation (over `dim`) and adding `out` to the
  sum, we require that:

     a_size + b_size + dim_size - a_shift - b_shift - post_shift < 63
     out_size - out_shift < 63

  Where possible without introducing extra operations, we will ensure that
  the greater of those two sizes (the product vs. the original output)
  is <= FM_TARGET_SIZE.

  The above isn't an exhaustive description of the behavior of this function;
  see the code for more.

      a_shift + b_shift + post_shift - out_shift == exponent_diff

   while also ensuring that:
      out_size - out_shift < 62
   and if we do and up shiftin `out` (so we have control over its position and can
   choose the shifts), then we ensure that:

   max(out_size - out_shift, a_size + b_size - a_shift - b_shift - post_shift) <= target_size.




   We also want to choose the smallest post_shift >= 0 so that the
   size of the product after shifting is <= target_size,
   where 64 < target_size <= 32 is chosen by the caller.

  This comes down to:
      (a_size - a_shift) + (b_size - b_shift) - post_shift <= target_size

     @param [in] a_size  Size of one of the operands.   See doc for CheckSize()
                        for definition.  Must be in [0,63].  Note: this is
                        not symmetric with operand b, because a is the one
                        that may get shifted left.
     @param [in] a_exponent Exponent for operand a.
     @param [in] b_size  Size of the other operand.
     @param [in] b_exponent Exponent for operand b.
     @param [in] out_size  Size of the current output region.  Must be in [0,63].
                        This constraints how much we can shift the output.
     @param [in] out_exponent Exponent for the output region.  (This function
                        will decide whether to change it, and shift the
                        output data appropriately.)
     @param [in] dim_size  If the products will immediately be summed over dimension
                        `dim`, then the result of calling FindSize(dim, ..).
                        Otherwise, zero.
     @param [out] a_shift, b_shift, post_shift
                        These are the shifts this function outputs... will
                        be used in an equation like:
                output_data = ((a_data >> a_shift) * (b_data >> b_shift)) >> pos_shift.
     @param [out] out_shift  A shift value for `out`; positive means left shift,
                        negative means right shift (c.f. ShiftRegion64Right(),
                        ShiftRegion64Left()).  This function guarantees that
                        once you do this, the exponents will match up so you
                        won't have to set the exponent of y.
     @param [out] final_size_guess  An estimate of the size of the output after the
                        summation etc., to be used as hint to FindSize().
*/
inline static void GetShiftsFor2ArgMultiplyAndAdd(
    int a_size, int a_exponent, int b_size, int b_exponent,
    int out_size, int out_exponent, int dim_size,
    int *a_shift, int *b_shift, int *post_shift, int *out_shift,
    int *final_size_guess) {
  GetInputShiftsFor2ArgMultiply(a_size, b_size,
                                a_shift, b_shift);


  int product_size = a_size + b_size - *a_shift - *b_shift,
      product_exponent = a_exponent + b_exponent + *a_shift + *b_shift;

  GetShiftsForAdd(product_size, product_exponent,
                  out_size, out_exponent, dim_size,
                  post_shift, out_shift, final_size_guess);

  if (*post_shift < 0) {
    assert(*a_shift == 0 && *b_shift == 0);
    *a_shift = *post_shift;
    *post_shift = 0;
  }
}



/*
  This function is used for debugging; it checks that `size`
  is the smallest integer >= 0 such that value < (1 << size).

  It does some assertions but you should check the return status
  (returns 1 for success, 0 for failure.)
  Would normally be called as assert(CheckSize(value, size))
 */
inline static int CheckSize(uint64_t value, int size) {
  assert((value & ((uint64_t)1<<63)) == 0 && "CheckSize() cannot accept input >= 2^63. "
         "Inputs must be converted to positive values!");
  int ok = (value & ~((((uint64_t)1) << size) - 1)) == 0 &&
      ((value == 0 && size == 0) ||
       (value & (((uint64_t)1) << (size - 1) )) != 0);
  if (!ok) {
    fprintf(stderr, "Error: value=%llu size=%d\n", value, size);
    return 0;
  } else {
    return 1;
  }
}


/*
  Returns the smallest integer i >= 0 such that
  (1 << i) > value.

  `guess` may be any value in [0,63]; it is an error if it is outside that
  range.  If it's close to the true value it will be faster.
 */
int FindSize(uint64_t value, int guess) {
  assert(guess >= 0 && guess <= 63);
  int ans = guess;
  uint64_t neg_mask = ~((((uint64_t)1) << ans) - 1);

  if ((neg_mask & value) == 0) {
    /* It's not the case that value >= (1 << guess).*/
    if (value == 0)
      return 0;
    neg_mask >>= 1;
    while ((neg_mask & value) == 0) {
      ans--;
      neg_mask >>= 1;
    }
    assert(CheckSize(value, ans));
    if (abs(ans - guess) > 3) /* TEMP */
      fprintf(stderr, "Warning: FindSize: guess = %d, ans = %d\n", guess, ans);
    return ans;
  } else {
    /* value > (1 << guess).  Keep shifting neg_mask left till it fits. */
    neg_mask <<= 1;
    ans++;
    while ((neg_mask & value) != 0) {
      neg_mask <<= 1;
      ans++;
    }
    assert(CheckSize(value, ans));
    if (abs(ans - guess) > 3) /* TEMP */
      fprintf(stderr, "Warning: FindSize: guess = %d, ans = %d\n", guess, ans);
    return ans;
  }
}



/*
  Ensures that the size of `region` is at least as large as the size of `value`:
  for use when you have set one element or sub-part of `region` to something
  of size `value`.

    @param [in] value  An UNSIGNED value which reflects the size of some element of
                      `region` that we are setting.
    @param [in] value_size_guess  A guess at the size of `value`, must be in [0..63].
                       This function will be faster if it close to the size of
                       `value` as would be returned by FindSize().
    @param [out] region   The region whose size is to be updated.
 */
inline static void EnsureRegionAtLeastSizeOf(uint64_t value,
                                             int value_size_guess,
                                             Region64 *region) {
  if ((value >> region->size) == 0)
    return;  /* The current size is OK. */
  /* we know the size of `value` is > region->size... */
  if (value_size_guess <= region->size)
    value_size_guess = region->size + 1;
  region->size = FindSize(value, value_size_guess);
}


void CopyScalar64ToElem(const Scalar64 *scalar, Elem64 *elem){
  int in_shift, out_shift, final_size_guess;
  GetShiftsForAssign(scalar->size, scalar->exponent,
                   elem->region->size, elem->region->exponent,
                   0, &in_shift, &out_shift, &final_size_guess);
  ShiftRegion64(out_shift, elem->region);
  *(elem->data) = (in_shift > 0 ? scalar->data >> in_shift :
                   scalar->data << -in_shift);
  EnsureRegionAtLeastSizeOf(FM_ABS(*(elem->data)),
                            final_size_guess,
                            elem->region);
  CheckRegion64Size(elem->region);
}


void CopyElemToScalar64(const Elem64 *elem, Scalar64 *scalar) {
  scalar->data = *(elem->data);
  scalar->size = FindSize(FM_ABS(*(elem->data)), elem->region->size);
  scalar->exponent = elem->region->exponent;
}


void CopyVectorElemToScalar64(const Vector64 *a, int i, Scalar64 *y) {
  assert(i >= 0 && i < a->dim);
  y->exponent = a->region->exponent;
  y->data = a->data[i * a->stride];
  y->size = FindSize(y->data, a->region->size);
}

void CopyScalar64ToVectorElem(const Scalar64 *s, int i, Vector64 *a) {
  assert(i >= 0 && i < a->dim);
  int in_shift, out_shift, final_size_guess;
  GetShiftsForAssign(s->size, s->exponent,
                   a->region->size, a->region->exponent,
                   0, &in_shift, &out_shift, &final_size_guess);
  ShiftRegion64(out_shift, a->region);
  a->data[i * a->stride] = (in_shift > 0 ? s->data >> in_shift :
                            s->data << -in_shift);
  EnsureRegionAtLeastSizeOf(FM_ABS(a->data[i * a->stride]),
                            final_size_guess,
                            a->region);
  CheckRegion64Size(a->region);
}



void ZeroVector64(Vector64 *a) {
  int dim = a->dim, stride = a->stride;
  int64_t *data = a->data;
  int i;
  for (i = 0; i + 4 <= dim; i++) {
    data[i*stride] = 0;
    data[(i+1)*stride] = 0;
    data[(i+2)*stride] = 0;
    data[(i+3)*stride] = 0;
  }
  for (; i < dim; i++) {
    data[i*stride] = 0;
  }
  if (a->dim == a->region->dim) {
    /* Exponent won't matter. */
    a->region->size = 0;
  }
}

/* CAUTION: not tested. */
int VectorsOverlap(const Vector64 *vec1, const Vector64 *vec2){
  if (vec1->region != vec2->region) return 0;
  /* Please note: this has to work for negative strides. */
  int64_t *vec1_first = vec1->data,
      *vec1_last = vec1->data + (vec1->dim - 1) * vec1->stride,
      *vec2_first = vec2->data,
      *vec2_last = vec2->data + (vec2->dim - 1) * vec2->stride;
  if (vec1_first < vec2_first) {
    return !(vec1_first < vec2_last &&
             vec1_last < vec2_first &&
             vec1_last < vec2_last);
  } else {
    return !(vec1_first > vec2_first &&
             vec1_first > vec2_last &&
             vec1_last > vec2_first &&
             vec1_last > vec2_last);
  }
}

void DivideScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y) {

}

double Scalar64ToDouble(const Scalar64 *a) {
  return a->data * pow(2.0, a->exponent);
}

double Vector64ElemToDouble(int i, const Vector64 *vec) {
  assert(i >= 0 && i < vec->dim);
  return vec->data[i * vec->stride] * pow(2.0, vec->region->exponent);
}

void InitScalar64FromInt(int64_t i, Scalar64 *a) {
  a->exponent = 0;
  a->data = i;
  uint64_t i_abs = FM_ABS(i);
  a->size = FindSize(i_abs, 1);
  assert(CheckSize(i_abs, a->size));
}


void CopyIntToVector64Elem(int i, int64_t value, int size_hint, Vector64 *a) {
  assert(i >= 0 && i < a->dim);
  int size = FindSize(FM_ABS(value), size_hint),
      exponent = 0, dim_size = 0;
  int in_shift, out_shift, final_size_guess;
  GetShiftsForAssign(size, exponent, a->region->size, a->region->exponent,
                     dim_size, &in_shift, &out_shift, &final_size_guess);
  ShiftRegion64(out_shift, a->region);

  a->data[i * a->stride] = (in_shift >= 0 ?
                            value >> in_shift :
                            value << -in_shift);

  if (a->region->size < size - in_shift)
    a->region->size = size - in_shift;
  CheckRegion64Size(a->region);
}


void ShiftScalar64Right(int right_shift, Scalar64 *a) {
  a->exponent += right_shift;
  a->data >>= right_shift;
  a->size -= right_shift;
  if (a->size < 0)
    a->size = 0;
}


void ShiftScalar64Left(int left_shift, Scalar64 *a) {
  assert(left_shift >= 0);
  a->exponent -= left_shift;
  a->data <<= left_shift;
  a->size += left_shift;
  assert(a->size < 64);
}


void InitRegion64(int64_t *data, int dim, int exponent, int size_hint, Region64 *region) {
  assert(dim != 0 && size_hint >= 0 && size_hint <= 63);
  region->dim = dim;
  region->exponent = exponent;
  region->data = data;
  SetRegion64Size(size_hint, region);
}

void ShiftRegion64Right(int right_shift, Region64 *region) {
  assert(right_shift >= 0);
  region->exponent += right_shift;
  if (region->size == 0)
    return;  /* it's all zeros; nothing to shift. */
  region->size -= right_shift;
  if (region->size < 0)
    region->size = 0;
  int dim = region->dim;
  int64_t *data = region->data;
  for (int i = 0; i < dim; i++)
    data[i] >>= right_shift;
  CheckRegion64Size(region);
}


void ShiftRegion64Left(int left_shift, Region64 *region) {
  assert(left_shift >= 0);
  region->exponent -= left_shift;
  if (region->size == 0)
    return;  /* it's all zeros; nothing to shift. */
  region->size += left_shift;
  int dim = region->dim;
  int64_t *data = region->data;
  for (int i = 0; i < dim; i++)
    data[i] <<= left_shift;
  CheckRegion64Size(region);
}

void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y) {
  int a_size = a->size,
      b_size = b->size,
      dim_size = 0,
      a_shift, b_shift, post_shift, final_size_guess;
  GetShiftsFor2ArgMultiply(a_size, b_size, dim_size,
                           &a_shift, &b_shift,
                           &post_shift, &final_size_guess);
  y->data = ((a->data >> a_shift) * (b->data >> b_shift)) >> post_shift;
  y->exponent = a->exponent + b->exponent + a_shift + b_shift + post_shift;
  y->size = FindSize(FM_ABS(y->data), final_size_guess);
}


void InvertScalar64(const Scalar64 *a, Scalar64 *b) {
  int64_t a_data = a->data;
  assert(a_data != 0);
  int a_size = a->size,
      a_exponent = a->exponent;
  if (a_size >= (63 - FM_TARGET_SIZE)) {
    // -2^63 is the largest exact power of 2 we can fit in a signed int64_t.
    // We'll cancel out the minus sign later.
    int64_t negative_2_63 = (int64_t)(((uint64_t)1)<<63);
    if (a_size > 32) {
      int a_right_shift = a_size - 32;
      b->data = -(negative_2_63 / (a_data >> a_right_shift));
      b->size = FindSize(FM_ABS(b->data), 32);
      b->exponent = -63 - a_exponent - a_right_shift;
    } else {
      b->data = -(negative_2_63 / a_data);
      b->size = FindSize(FM_ABS(b->data), 64 - a_size);
      b->exponent = -63 - a_exponent;
    }
  } else {
    int p = a_size + FM_TARGET_SIZE;  /* Note: is a_size < 63 - FM_TARGET_SIZE,
                                         so p < 63. This guarantees that 1<<p is
                                         positive. */
    int64_t big_number = 1 << p;
    b->data = big_number / a_data;
    b->size = FindSize(FM_ABS(b->data), p - a_size);
    b->exponent = -p - a_exponent;
  }
}

/*
  Does y := a + b
 */
void AddScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y) {
  int a_size = a->size,
      b_size = b->size,
      a_exponent = a->exponent,
      b_exponent = b->exponent;
  if (a_exponent >= b_exponent) {
    // We may have to shift a's data left to make the exponents match.
    if (b_size < 63) { // No need to shift a right, no danger of overflow.
      y->data = b->data + (a->data << (a_exponent - b_exponent));
      y->exponent = b_exponent;
      y->size = FindSize(FM_ABS(y->data), b_size);
    } else { // This should be an extremely rare case, so just handle by recursion.
      Scalar64 b2;
      CopyScalar64(a, &b2);
      ShiftScalar64Right(b_size - FM_TARGET_SIZE, &b2);
      AddScalar64(a, &b2, y);
    }
  } else {
    // b_exponent > a_exponent.
    // We may have to shift b's data left to make the exponents match.
    if (a_size < 63) { // No need to shift a right, no danger of overflow.
      y->data = a->data + (b->data << (b_exponent - a_exponent));
      y->exponent = a_exponent;
      y->size = FindSize(FM_ABS(y->data), a_size);
    } else { // This should be an extremely rare case, so just handle by recursion.
      Scalar64 a2;
      CopyScalar64(a, &a2);
      ShiftScalar64Right(63 - FM_TARGET_SIZE, &a2);
      AddScalar64(&a2, b, y);
    }
  }
}

void CopyVector64(const Vector64 *src, Vector64 *dest) {
  assert(src->dim == dest->dim && src->region != dest->region);
  int dim = src->dim, src_stride = src->stride,
      dest_stride = dest->stride;
  int64_t *src_data = src->data, *dest_data = dest->data;
  int src_shift, dest_shift, final_size_guess;

  GetShiftsForAssign(0, src->region->exponent, src->region->size,
                     dest->region->exponent, dest->region->size,
                     &src_shift, &dest_shift,
                     &final_size_guess);
  ShiftRegion64(dest_shift, dest->region);

  if (src_shift == 0) {
    for (int i = 0; i < dim; i++)
      dest_data[i * dest_stride] = src_data[i * src_stride];
  } else if (src_shift > 0) {
    for (int i = 0; i < dim; i++)
      dest_data[i * dest_stride] = src_data[i * src_stride] >> src_shift;
  } else {
    for (int i = 0; i < dim; i++)
      dest_data[i * dest_stride] = src_data[i * src_stride] << -src_shift;
  }
  /* see CAUTION in header, RE how we are setting the size. */
  int new_size = src->region->size - src_shift;
  if (new_size > dest->region->size)
    dest->region->size = new_size;
  CheckRegion64Size(dest->region);
}


void AddScalarVector64(const Scalar64 *a, const Vector64 *x, Vector64 *y) {
  assert(x->region != y->region);
  int x_size = x->region->size, x_exponent = x->region->exponent,
      a_size = a->size, a_exponent = a->exponent,
      x_shift, a_shift, post_shift, y_shift, final_size_guess,
      dim = x->dim;
  assert(y->dim == x->dim);
  int y_size = y->region->size,
      y_exponent = y->region->exponent,
      dim_size = (dim < 32 ? fm_num_bits[dim] : FindSize(dim, 6));

  /* Note, we give a as the first arg to GetShifts... because that is
     the one that may get left-shifted if we need to left-shift,
     and doing so on the scalar is more efficient. */
  GetShiftsFor2ArgMultiplyAndAdd(a_size, a_exponent, x_size, x_exponent,
                                 y_size, y_exponent, dim_size,
                                 &a_shift, &x_shift, &post_shift,
                                 &y_shift, &final_size_guess);
  ShiftRegion64(y_shift, y->region);

  fprintf(stderr, "a_shift = %d, x_shift = %d,  post_shift = %d, final_size_guess = %d\n",
          a_shift, x_shift, post_shift, final_size_guess);

  int x_stride = x->stride, y_stride = y->stride;
  int64_t *x_data = x->data, *y_data = y->data;
  uint64_t tot_bits = 0;  /* `tot_bits` is a bit pattern keeping track of max size. */
  int i;
  if (a_shift > 0 || y_shift > 0 || post_shift > 0) {
    int64_t a_shifted = a->data >> a_shift;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift,
          prod2 = (a_shifted * (x_data[(i+1) * x_stride] >> x_shift)) >> post_shift,
          prod3 = (a_shifted * (x_data[(i+2) * x_stride] >> x_shift)) >> post_shift,
          prod4 = (a_shifted * (x_data[(i+3) * x_stride] >> x_shift)) >> post_shift;
      tot_bits |= ((FM_ABS(prod1) | FM_ABS(prod2)) | (FM_ABS(prod3)| FM_ABS(prod4)));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift;
      y_data[i * y_stride] += prod;
      tot_bits |= FM_ABS(prod);
    }
  } else {
    /* We can implement the shifting by shifting just a. */
    int64_t a_shifted = a->data << -a_shift;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = a_shifted * x_data[i * x_stride],
          prod2 = a_shifted * x_data[(i+1) * x_stride],
          prod3 = a_shifted * x_data[(i+2) * x_stride],
          prod4 = a_shifted * x_data[(i+3) * x_stride];
      tot_bits |= (uint64_t)((FM_ABS(prod1) | FM_ABS(prod2)) | (FM_ABS(prod3) | FM_ABS(prod4)));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = a_shifted * x_data[i * x_stride];
      y_data[i * y_stride] += prod;
      tot_bits |= FM_ABS(prod);
    }
  }
  if (y->dim == y->region->size) {
    y->region->size = FindSize(tot_bits, final_size_guess);
  } else {
    EnsureRegionAtLeastSizeOf(tot_bits, final_size_guess,
                              y->region);
  }
}


/* y := a * x. */
void SetScalarVector64(const Scalar64 *a, const Vector64 *x, Vector64 *y) {
  int x_size = x->region->size, x_exponent = x->region->exponent,
      a_size = a->size, a_exponent = a->exponent,
      dim = x->dim;
  assert(y->dim == x->dim);
  if (dim == y->region->dim) {
    /* we'll be overwriting all of y's region so its current values are
       a 'dont-care' */
    y->region->size = 0;
    y->region->exponent = 0;
  }
  int y_size = y->region->size,
      y_exponent = y->region->exponent,
      dim_size = (dim < 32 ? fm_num_bits[dim] : FindSize(dim, 6));

  int a_shift, x_shift, post_shift, y_shift, final_size_guess;

  /* Note, we give a as the first arg to GetShifts... because that is
     the one that may get left-shifted if we need to left-shift,
     and doing so on the scalar is more efficient. */
  GetShiftsFor2ArgMultiplyAndAdd(a_size, a_exponent, x_size, x_exponent,
                                 y_size, y_exponent, dim_size,
                                 &x_shift, &a_shift, &post_shift,
                                 &y_shift, &final_size_guess);
  ShiftRegion64(y_shift, y->region);

  int x_stride = x->stride, y_stride = y->stride;
  int64_t *x_data = x->data, *y_data = y->data;
  uint64_t tot_bits = 0;  /* `tot_bits` is a bit pattern keeping track of max size. */
  int i;
  if (a_shift > 0 | x_shift > 0 || post_shift > 0) {
    int64_t a_shifted = a->data >> a_shift;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift,
          prod2 = (a_shifted * (x_data[(i+1) * x_stride] >> x_shift)) >> post_shift,
          prod3 = (a_shifted * (x_data[(i+2) * x_stride] >> x_shift)) >> post_shift,
          prod4 = (a_shifted * (x_data[(i+3) * x_stride] >> x_shift)) >> post_shift;
      tot_bits |= ((FM_ABS(prod1) | FM_ABS(prod2)) | (FM_ABS(prod3)| FM_ABS(prod4)));
      y_data[i * y_stride] = prod1;
      y_data[(i+1) * y_stride] = prod2;
      y_data[(i+2) * y_stride] = prod3;
      y_data[(i+3) * y_stride] = prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift;
      y_data[i * y_stride] = prod;
      tot_bits |= FM_ABS(prod);
    }
  } else {
    assert(x_shift == 0 && post_shift == 0);
    /* We can implement the shifting by shifting just a. */
    int64_t a_shifted = a->data << -a_shift;
    uint64_t tot_bits = 0;  /* `tot_bits` is a bit pattern keeping track of max size. */
    int dim = x->dim, x_stride = x->stride, y_stride = y->stride;
    int64_t *x_data = x->data, *y_data = y->data;
    int i;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = a_shifted * x_data[i * x_stride],
          prod2 = a_shifted * x_data[(i+1) * x_stride],
          prod3 = a_shifted * x_data[(i+2) * x_stride],
          prod4 = a_shifted * x_data[(i+3) * x_stride];
      tot_bits |= (uint64_t)((prod1 | prod2) | (prod3 | prod4));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = a_shifted * x_data[i * x_stride];
      y_data[i * y_stride] += prod;
      tot_bits |= prod;
    }
  }

  if (y->dim == y->region->size) {
    y->region->size = FindSize(tot_bits, final_size_guess);
  } else {
    EnsureRegionAtLeastSizeOf(tot_bits, final_size_guess,
                              y->region);
  }
}


void Vector64AddScalar(const Scalar64 *a, Vector64 *y) {
  int in_shift, out_shift, final_size_guess;
  GetShiftsForAdd(a->size, a->exponent,
                  y->region->size, y->region->exponent,
                  0, &in_shift, &out_shift, &final_size_guess);
  ShiftRegion64(out_shift, y->region);
  int64_t a_data_shifted = ShiftInt64(a->data, in_shift);
  uint64_t tot_bits = 0;
  int dim = y->dim;
  for (int i = 0; i < dim; i++) {
    y->data[i] += a_data_shifted;
    tot_bits |= FM_ABS(y->data[i]);
  }
  if (y->dim == y->region->size) {
    y->region->size = FindSize(tot_bits, final_size_guess);
  } else {
    EnsureRegionAtLeastSizeOf(tot_bits, final_size_guess,
                              y->region);
  }
}


void DotVector64(const Vector64 *a, const Vector64 *b, Scalar64 *y) {
  int dim = a->dim;
  assert(a->dim == b->dim);
  /* dim_size is the number of bits by which the a sum over this
   * many terms could be greater than the elements of the sum. */
  int dim_size = (dim < 32 ? fm_num_bits[dim] : FindSize(dim, 6));


  int a_size = a->region->size, b_size = b->region->size,
      a_shift, b_shift, post_shift,
      final_size_guess;
  GetShiftsFor2ArgMultiply(a_size, b_size, dim_size,
                           &a_shift, &b_shift,
                           &post_shift, &final_size_guess);

  int a_stride = a->stride, b_stride = b->stride;
  const int64_t *a_data = a->data, *b_data = b->data;
  int64_t sum = 0;
  int i;
  for (i = 0; i + 4 <= dim; i += 4) {
    int64_t prod1 = ((a_data[i * a_stride] >> a_shift) *
                     (b_data[i * b_stride] >> b_shift)) >> post_shift,
        prod2 = ((a_data[(i+1) * a_stride] >> a_shift) *
                 (b_data[(i+1) * b_stride] >> b_shift)) >> post_shift,
        prod3 = ((a_data[(i+2) * a_stride] >> a_shift) *
                 (b_data[(i+2) * b_stride] >> b_shift)) >> post_shift,
        prod4 = ((a_data[(i+3) * a_stride] >> a_shift) *
                 (b_data[(i+3) * b_stride] >> b_shift)) >> post_shift;
    sum += (prod1 + prod2) + (prod3 + prod4);
  }
  for (; i < dim; i++) {
    int64_t prod = ((a_data[i * a_stride] >> a_shift) *
                    (b_data[i * b_stride] >> b_shift)) >> post_shift;
    sum += prod;
  }
  y->data = sum;
  y->exponent = a->region->exponent + b->region->exponent +
      (a_shift + b_shift + post_shift);
  y->size = FindSize(FM_ABS(sum), final_size_guess);
}


/* Computes matrix-vector product:
   y := M x.   Note: y and x must not be in the same region;
   and currently, y must occupy its entire region.  (We
   can relax this requirement later as needed.

   Note, this code closely mirrors the code of DotVector64.
*/
void SetMatrixVector64(const Matrix64 *m, const Vector64 *x,
                       Vector64 *y) {
  assert(y->region != x->region && y->region != m->region &&
         x->dim == m->num_cols && y->dim == m->num_rows);
  /*int num_rows = m->num_rows,*/
  int num_cols = m->num_cols;
  /* col_size is the number of bits by which the a sum over this many terms
   * could be greater than the elements of the sum. */
  int col_size = (num_cols < 32? fm_num_bits[num_cols] : FindSize(num_cols, 6));

  if (y->dim == y->region->dim)
    y->region->size = 0;  /* So `ShiftRegion64{Left,Right} know they doesn't
                             really have to shift.  y's data will be
                             overwritten.*/

  int m_size = m->region->size, m_exponent = m->region->exponent,
      x_size = x->region->size, x_exponent = x->region->exponent,
      y_size = y->region->size, y_exponent = y->region->exponent,
      m_shift, x_shift, post_shift, y_shift, final_size_guess;
  /* Note: it's "AndAdd" even though we don't add to y, because y may
     in general contain other data in its region. */
  GetShiftsFor2ArgMultiplyAndAdd(m_size, m_exponent, x_size, x_exponent,
                                 y_size, y_exponent, col_size,
                                 &m_shift, &x_shift, &post_shift,
                                 &y_shift, &final_size_guess);
  ShiftRegion64(y_shift, y->region);

  /* We haven't made much of an effort here to think about optimizations for
     memory loads . */
  uint64_t tot_bits = 0;  /* Bit pattern to compute size of output. */
  int x_dim = x->dim, x_stride = x->stride,
      y_dim = y->dim, y_stride = y->stride,
      m_row_stride = m->row_stride,
      m_col_stride = m->col_stride;
  const int64_t *mdata = m->data, *xdata = x->data;
  int64_t *ydata = y->data;
  if (m_shift > 0 || x_shift > 0 || post_shift > 0) {
    for (int i = 0; i < y_dim; i++) {
      int64_t y_i = 0;
      const int64_t *mrowdata = mdata + (i * m_row_stride);
      int j;
      for (j = 0; j + 4 <= x_dim; j += 4) {
        int64_t prod0 = ((mrowdata[j * m_col_stride] >> m_shift) * (xdata[j * x_stride] >> x_shift)) >> post_shift,
            prod1 = ((mrowdata[(j+1) * m_col_stride] >> m_shift) * (xdata[(j+1) * x_stride] >> x_shift)) >> post_shift,
            prod2 = ((mrowdata[(j+2) * m_col_stride] >> m_shift) * (xdata[(j+2) * x_stride] >> x_shift)) >> post_shift,
            prod3 = ((mrowdata[(j+3) * m_col_stride] >> m_shift) * (xdata[(j+3) * x_stride] >> x_shift)) >> post_shift;
        y_i += (prod0 + prod1) + (prod2 + prod3);
      }
      for (; j < x_dim; j++) {
        int64_t prod = ((mrowdata[j * m_col_stride] >> m_shift) * (xdata[j * x_stride] >> x_shift)) >> post_shift;
        y_i += prod;
      }
      tot_bits |= FM_ABS(y_i);
      ydata[i * y_stride] = y_i;
    }
  } else {
    for (int i = 0; i < y_dim; i++) {
      int64_t y_i = 0;
      const int64_t *mrowdata = mdata + (i * m_row_stride);
      int j;
      for (j = 0; j + 4 <= x_dim; j += 4) {
        int64_t prod0 = mrowdata[j * m_col_stride]  * xdata[j * x_stride],
            prod1 = mrowdata[(j+1) * m_col_stride] * xdata[(j+1) * x_stride],
            prod2 = mrowdata[(j+2) * m_col_stride] * xdata[(j+2) * x_stride],
            prod3 = mrowdata[(j+3) * m_col_stride] * xdata[(j+3) * x_stride];
        y_i += (prod0 + prod1) + (prod2 + prod3);
      }
      for (; j < x_dim; j++) {
        int64_t prod = mrowdata[j * m_col_stride] * xdata[j * x_stride];
        y_i += prod;
      }
      /* NOTE: the m_shift is supposed to be applied to m, but we can apply it anywhere,
         and it's more efficient to apply it after the summation. */
      y_i = y_i << -m_shift;
      tot_bits |= FM_ABS(y_i);
      ydata[i * y_stride] = y_i;
    }
  }
  if (y->dim == y->region->dim) {
    y->region->size = FindSize(tot_bits, final_size_guess);
  } else {
    EnsureRegionAtLeastSizeOf(tot_bits, final_size_guess, y->region);
  }
  CheckRegion64Size(y->region);
}



/** Computes the size for this region and sets it to the correct value. */
void SetRegion64Size(int size_hint, Region64 *r) {
  int64_t *data = r->data;
  int dim = r->dim;
  int i;

  uint64_t tot = 0;  /* tot will be the `or` of all the absolute values. */
  for (i = 0; i + 4 <= dim; i += 4) {
    uint64_t a1 = FM_ABS(data[i]),
        a2 = FM_ABS(data[i + 1]),
        a3 = FM_ABS(data[i + 2]),
        a4 = FM_ABS(data[i + 3]);
    tot |= (a1 | a2) | (a3 | a4);
  }
  for (; i < dim; i++)
    tot |= FM_ABS(data[i]);
  /* TODO: find a more efficient way to get the size. */
  int size = FindSize(tot, size_hint);
  r->size = size;
}

void ZeroRegion64(Region64 *region) {
  int dim = region->dim;
  for (int i = 0; i < dim; i++)
    region->data[i] = 0;
  region->exponent = 0;
  region->size = 0;
}


#ifdef FIXED_MATH_TEST

void TestFindSize() {
  uint64_t n = 1;
  assert(FindSize(0, 0) == 0);
  for (int i = 1; i < 63; i++) {
    for (int j = 0; j <= 63; j++) {
      assert(FindSize(n, j) == i);
    }
    n = n << 1;
  }
}

void TestSetToInt() {
  for (int i = -10; i < 10; i++) {
    Scalar64 a;
    InitScalar64FromInt(i, &a);
    double f = Scalar64ToDouble(&a);
    assert(f - i == 0);
  }
}


void TestShift() {
  for (int i = -10; i < 10; i++) {
    Scalar64 a;

    for (int j = 0; j < 50; j += 10) {
      InitScalar64FromInt(i, &a);
      ShiftScalar64Left(j, &a);
      ShiftScalar64Right((j - 2 >= 0 ? j - 2 : 0), &a);
      double f = Scalar64ToDouble(&a);
      assert(f - i == 0);
    }
  }
}


void TestAddScalar() {
  for (int i = -10; i < 10; i += 5) {
    for (int shift_i = 0; shift_i < 50; shift_i++) {
      Scalar64 ii;
      InitScalar64FromInt(i, &ii);
      ShiftScalar64Left(shift_i, &ii);
      for (int j = -10; j < 10; j += 3) {
        for (int shift_j = 0; shift_j < 50; shift_j++) {
          Scalar64 jj;
          InitScalar64FromInt(j, &jj);
          ShiftScalar64Left(shift_j, &jj);

          Scalar64 kk;
          AddScalar64(&ii, &jj, &kk);

          double kf = Scalar64ToDouble(&kk);
          assert(kf == i + j);

          AddScalar64(&kk, &kk, &kk);
          kf = Scalar64ToDouble(&kk);
          assert(kf == 2 * (i + j));
        }
      }
    }
  }
}



void TestMulScalar() {
  for (int i = -10; i < 10; i += 5) {
    for (int shift_i = 0; shift_i < 50; shift_i+=10) {
      Scalar64 ii;
      InitScalar64FromInt(i, &ii);
      ShiftScalar64Left(shift_i, &ii);
      for (int j = -10; j < 10; j += 3) {
        for (int shift_j = 0; shift_j < 50; shift_j+=10) {
          Scalar64 jj;
          InitScalar64FromInt(j, &jj);
          ShiftScalar64Left(shift_j, &jj);

          Scalar64 kk;
          MulScalar64(&ii, &jj, &kk);

          double kf = Scalar64ToDouble(&kk);
          fprintf(stderr, "i = %d, j = %d, k = %lf, if = %lf, jf= %lf\n", i, j, kf,
                 Scalar64ToDouble(&ii), Scalar64ToDouble(&jj));
          assert(kf == i * j);

        }
      }
    }
  }
}

void TestInitVector() {  /* Also tests InitSubVector64. */
  int64_t data[100];
  for (int i = 0; i < 100; i++)
    data[i] = i;
  Region64 region;
  InitRegion64(data, 100, 0, 5, &region);

  int orig_size = region.size;

  for (int i = 0; i <= 63; i++) {
    SetRegion64Size(i, &region);
    assert(region.size == orig_size);
  }
  Vector64 vec;
  InitVector64(&region, 10, 1, region.data, &vec);

  Vector64 vec2, vec3;
  InitSubVector64(&vec, 9, 10, -1, &vec2);
  InitSubVector64(&vec2, 0, 5, 2, &vec3);

  int64_t scalar = -79;
  int size_guess = 6;
  CopyIntToVector64Elem(2, scalar, size_guess, &vec2);
  assert(scalar == Vector64ElemToDouble(7, &vec));
  scalar += 1;
  CopyIntToVector64Elem(3, scalar, size_guess, &vec3);
  assert(scalar == Vector64ElemToDouble(6, &vec2));
}



void TestCopyIntToVector64Elem() {
  for (int source = 0; source < 500; source++) {
    int64_t data[100];
    for (int i = 0; i < 100; i++)
      data[i] = i;
    Region64 region;
    InitRegion64(data, 100, 0, 5, &region);

    int orig_size = region.size;

    for (int i = 0; i <= 63; i++) {
      SetRegion64Size(i, &region);
      assert(region.size == orig_size);
    }
    Vector64 vec;
    InitVector64(&region, 10, 1, region.data, &vec);

    int region_shift = source % 64;
    if (region.size + region_shift >= 64)
      continue;

    int64_t value = -100 + (source % 200),
        value_shift = (source * 7) % 64;
    if (value_shift + FindSize(FM_ABS(value), 0) >= 64)
      continue;
    int index = 5;
    value <<= value_shift;
    CopyIntToVector64Elem(index, value, value_shift, &vec);

    assert(value - Vector64ElemToDouble(index, &vec) <= FM_ABS(value) * 1.0e-10);
  }
}


void TestDotVector() {
  for (int shift = 0; shift < 63 - 6; shift++) {
    int64_t data[100];
    for (int i = 0; i < 100; i++)
      data[i] = i;
    int64_t ref_sum = 0;
    for (int i = 0; i < 10; i++)
      ref_sum += data[i] * data[30 - 2*i];

    Region64 region;
    InitRegion64(data, 100, 0, 5, &region);
    ShiftRegion64Left(shift, &region);

    Vector64 vec1;
    InitVector64(&region, 10, 1, region.data, &vec1);

    Vector64 vec2;
    InitVector64(&region, 10, -2, region.data + 30, &vec2);


    Scalar64 scalar;
    DotVector64(&vec1, &vec2, &scalar);

    double f = Scalar64ToDouble(&scalar);
    fprintf(stderr, "shift = %d, f = %f, ref_sum = %lld, size = %d, scalar exponent = %d\n",
            shift, f, ref_sum, scalar.size, scalar.exponent);
    assert(f == ref_sum);
  }
}




void TestAddVector() {
  for (int source = 0; source < 1000; source++ ) {
    /* source is a source of randomness. */
    int shift1 = source % 29,
        shift2 = source % 17,
        shift3 = source % 11;
    int64_t scalar_number = 79;
    int64_t data1[100];
    for (int i = 0; i < 100; i++)
      data1[i] = i;
    int64_t data2[100];
    for (int i = 0; i < 100; i++) {
      data2[i] = i + 4;
    }

    int64_t ref_sum = 0;
    for (int i = 0; i < 10; i++)
      ref_sum += data1[i] * (data2[30 - 2*i] + scalar_number * data1[i]);

    Region64 region1;
    InitRegion64(data1, 100, 0, 5, &region1);
    if (region1.size + shift1 >= 64)
      continue;
    ShiftRegion64Left(shift1, &region1);

    Region64 region2;
    InitRegion64(data2, 100, 0, 5, &region2);
    if (region2.size + shift2 >= 64)
      continue;
    ShiftRegion64Left(shift2, &region2);

    Scalar64 scalar;
    InitScalar64FromInt(scalar_number, &scalar);
    if (scalar.size + shift3 >= 64)
      continue;
    ShiftScalar64Left(shift3, &scalar);

    Vector64 vec1;
    InitVector64(&region1, 10, 1, region1.data, &vec1);

    Vector64 vec2;
    InitVector64(&region2, 10, -2, region2.data + 30, &vec2);

    /* do: vec2 += scalar * vec1.
       Then compute result = vec1 * vec2.
       so result = vec1 * (vec2 + scalar * vec1). */

    AddScalarVector64(&scalar, &vec1, &vec2);

    Scalar64 dot_prod;
    DotVector64(&vec1, &vec2, &dot_prod);

    double f = Scalar64ToDouble(&dot_prod);
    fprintf(stderr, "shift{1,2,3} = %d,%d->%d,%d, f = %f, ref_sum = %lld, size = %d, scalar exponent = %d\n",
            shift1, shift2, -region2.exponent, shift3, f, ref_sum, scalar.size, scalar.exponent);
    assert(f == ref_sum);
  }
}



void TestCopyVector() {
  for (int source = 0; source < 500; source++ ) {
    /* source is a source of randomness. */
    int shift1 = source % 29,
        shift2 = source % 17;
    int64_t scalar_number = 79;
    int64_t data1[10];
    for (int i = 0; i < 10; i++)
      data1[i] = i;
    int64_t data2[100];
    for (int i = 0; i < 10; i++) {
      data2[i] = i + 4;
    }

    Region64 region1;
    InitRegion64(data1, 10, 0, 5, &region1);
    if (region1.size + shift1 >= 64)
      continue;
    ShiftRegion64Left(shift1, &region1);

    Region64 region2;
    InitRegion64(data2, 10, 0, 5, &region2);
    if (region2.size + shift2 >= 64)
      continue;
    ShiftRegion64Left(shift2, &region2);

    Vector64 vec1;
    InitVector64(&region1, 5, 2, region1.data, &vec1);

    Vector64 vec2;
    InitVector64(&region2, 5, -1, region2.data + 9, &vec2);

    int index = 3, size_hint = 2;
    CopyIntToVector64Elem(index, scalar_number, size_hint, &vec1);

    assert(scalar_number == Vector64ElemToDouble(index, &vec1));

    CopyVector64(&vec1, &vec2);

    assert(scalar_number == Vector64ElemToDouble(index, &vec2));
  }
}



void TestSetMatrixVector64() {
  for (int source = 0; source < 1000; source++ ) {
    /* source is a source of randomness. */
    int shift1 = source % 29,
        shift2 = source % 17,
        shift3 = source % 11;
    int64_t data1[100];
    for (int i = 0; i < 100; i++)
      data1[i] = i;
    int64_t data2[100];
    for (int i = 0; i < 100; i++) {
      data2[i] = i + 4;
    }
    int64_t data3[10];
    for (int i = 0; i < 10; i++) {
      data3[i] = -1000000000;
    }

    int64_t data3_ref[10];
    for (int i = 0; i < 10; i++) {
      data3_ref[i] = 0;
      for (int j = 0; j < 10; j++) {
        data3_ref[i] += data1[i * 10 + j] * data2[j];
      }
    }


    Region64 region1;
    InitRegion64(data1, 100, 0, 5, &region1);
    if (region1.size + shift1 >= 64)
      continue;
    ShiftRegion64Left(shift1, &region1);

    Region64 region2;
    InitRegion64(data2, 100, 0, 5, &region2);
    if (region2.size + shift2 >= 64)
      continue;
    ShiftRegion64Left(shift2, &region2);

    Region64 region3;
    InitRegion64(data3, 10, 0, 5, &region3);
    ShiftRegion64Left(shift3, &region3);

    Region64 region4;
    InitRegion64(data3_ref, 10, 0, 5, &region4);

    Matrix64 mat;
    InitMatrix64(&region1, 10, 10, 10, 1, region1.data, &mat);

    Vector64 vec1;
    InitVector64(&region2, 10, 1, region2.data, &vec1);

    Vector64 vec2;
    InitVector64(&region3, 10, 1, region3.data, &vec2);

    Vector64 vec2_ref;
    InitVector64(&region4, 10, 1, region4.data, &vec2_ref);

    SetMatrixVector64(&mat, &vec1, &vec2);

    fprintf(stderr, "Product is below\n");
    PrintVector64(&vec2);

    fprintf(stderr, "Ref-sum = [ ");
    for (int i = 0; i < 10; i++)
      fprintf(stderr, "%f ", (float)data3_ref[i]);
    fprintf(stderr, "]\n");


    /* do: vec2 := vec2 - vec2_ref.  Should give us zero. */
    Scalar64 minus_one;
    InitScalar64FromInt(-1, &minus_one);
    AddScalarVector64(&minus_one, &vec2_ref, &vec2);

    Scalar64 diff_product;
    DotVector64(&vec2, &vec2, &diff_product);

    { /* Testing CopyVectorElemToScalar64 */
      Scalar64 temp;
      int index = 3;
      CopyVectorElemToScalar64(&vec2, index, &temp);
      assert(Scalar64ToDouble(&temp) == Vector64ElemToDouble(index, &vec2));
    }

    /* Check that this error term is zero. */
    double d = Scalar64ToDouble(&diff_product);
    assert(d == 0);

    /*
      fprintf(stderr, "shift{1,2,3} = %d,%d->%d,%d, f = %f, ref_sum = %lld, size = %d, scalar exponent = %d\n",
            shift1, shift2, -region2.exponent, shift3, f, ref_sum, scalar.size, scalar.exponent);
            assert(f == ref_sum); */
  }
}



void TestInvertScalar() {
  for (int i = -9; i < 100; i += 2) {
    for (int shift = 0; shift < 64; shift++) {
      Scalar64 a;
      InitScalar64FromInt(i, &a);
      if (a.size + shift >= 64)
        continue;
      ShiftScalar64Left(shift, &a);
      Scalar64 a_inv;
      CopyScalar64(&a, &a_inv);
      InvertScalar64(&a_inv, &a_inv);

      Scalar64 one;
      MulScalar64(&a, &a, &a);
      MulScalar64(&a_inv, &a_inv, &a_inv);
      MulScalar64(&a, &a_inv, &one);
      double f = Scalar64ToDouble(&one);
      assert(f - 1.0 < pow(2.0, -29));  /* actually this limit -29 is just emprical.
                                           I haven't done careful analysis here. */
    }
  }
}


void TestGetShiftsForTwoArgMultiply() {
  for (int i = 0; i < 2000; i++) {
    int a_size = i % 64,
        b_size = (i * 3) % 64,
        dim_size = (i * 11) % 32,
        a_shift = -1000, b_shift = -1000, post_shift = -1000,
        final_size_guess = -1000;
    GetShiftsFor2ArgMultiply(a_size, b_size, dim_size,
                             &a_shift, &b_shift,
                             &post_shift, &final_size_guess);
    int final_size2 = a_size + b_size + (dim_size/2) - a_shift - b_shift - post_shift;
    assert(final_size2 == final_size_guess);

    assert(a_size + b_size + dim_size - a_shift - b_shift - post_shift <= FM_TARGET_SIZE);
    assert(a_size + b_size - a_shift - b_shift <= 63);
    if (a_shift > 0 || b_shift > 0) {
      assert(a_size + b_size - a_shift - b_shift == 63);
      if (a_shift > 0)
        assert(a_size - a_shift >= 32);
      if (b_shift > 0)
        assert(b_size - b_shift >= 31);
    }
  }
}


void TestGetShiftsForAdd() {
  for (int i = 0; i < 2000; i++) {
    int in_size = i % 64,
        out_size = (i * 7) % 64,
        in_exponent = -100 + i % 199,
        out_exponent = -50 + i % 93,
        dim_size = (i * 11) % 32;

    int in_shift = -1000, out_shift = -1000,
        final_size_guess = -1000;
    GetShiftsForAdd(in_size, in_exponent,
                    out_size, out_exponent,
                    dim_size,
                    &in_shift, &out_shift,
                    &final_size_guess);
    assert(in_exponent + in_shift == out_exponent + out_shift);
    int max_size_out = FM_MAX(in_size + dim_size - in_shift, out_size - out_shift) + 1,
        size_guess = FM_MAX(in_size + (dim_size/2) - in_shift, out_size - out_shift),
        min_size_out =  FM_MAX(in_size - in_shift, out_size - out_shift);
    assert(max_size_out < 64);
    assert(final_size_guess == size_guess);
    assert( !(out_shift < 0 && in_shift < 0));  /* doesn't make sense to shift
                                                 * both left. */
    if (out_shift > 0 && in_shift > 0) {
      assert(final_size_guess <= FM_TARGET_SIZE);
    }

    if (out_shift > 0 || in_shift > 0) {
      assert(min_size_out >= FM_MIN_SIZE ||
             max_size_out >= FM_TARGET_SIZE);
    }
  }
}


void TestGetShiftsForAssign() {
  for (int i = 0; i < 2000; i++) {
    int in_size = i % 64,
        out_size = (i * 7) % 64,
        in_exponent = -100 + i % 199,
        out_exponent = -50 + i % 93,
        dim_size = (i * 11) % 32;

    int in_shift = -1000, out_shift = -1000,
        final_size_guess = -1000;
    GetShiftsForAssign(in_size, in_exponent,
                    out_size, out_exponent,
                    dim_size,
                    &in_shift, &out_shift,
                    &final_size_guess);
    assert(in_exponent + in_shift == out_exponent + out_shift);
    int max_size_out = FM_MAX(in_size + dim_size - in_shift, out_size - out_shift),
        size_guess = FM_MAX(in_size + (dim_size/2) - in_shift, out_size - out_shift),
        min_size_out =  FM_MAX(in_size - in_shift, out_size - out_shift);
    assert(max_size_out < 64);
    assert(final_size_guess == size_guess);
    assert( !(out_shift < 0 && in_shift < 0));  /* doesn't make sense to shift
                                                 * both left. */
    if (out_shift > 0 && in_shift > 0) {
      assert(final_size_guess <= FM_TARGET_SIZE);
    }

    if (out_shift > 0 || in_shift > 0) {
      assert(min_size_out >= FM_MIN_SIZE ||
             max_size_out >= FM_TARGET_SIZE);
    }
  }
}


void TestGetShiftsForTwoArgMultiplyAndAdd() {
  for (int i = 0; i < 2000; i++) {
    int a_size = i % 64,
        b_size = (i * 3) % 64,
        a_exponent = -100 + i % 199,
        b_exponent = -100 + i % 157,
        out_size = (i * 7) % 64,
        out_exponent = -100 + i % 167,
        dim_size = (i * 11) % 32;

    int a_shift = -1000, b_shift = -1000, post_shift = -1000,
        out_shift = -1000, final_size_guess = -1000;
    GetShiftsFor2ArgMultiplyAndAdd(a_size, a_exponent, b_size, b_exponent,
                                   out_size, out_exponent, dim_size,
                                   &a_shift, &b_shift, &post_shift, &out_shift,
                                   &final_size_guess);
    int final_size2 = FM_MAX(a_size + b_size + (dim_size/2) - a_shift - b_shift - post_shift,
                             out_size - out_shift);
    assert(b_shift >= 0);
    assert(final_size2 == final_size_guess);
    assert(a_size + b_size - a_shift - b_shift < 64);
    if (a_shift > 0 || b_shift > 0) {
      assert(a_size + b_size - a_shift - b_shift == 63);
    }
    if (a_shift < 0) {
      assert(b_shift == 0 && post_shift == 0);
    } else {
      assert(b_shift >= 0 && post_shift >= 0);
    }
    assert(a_exponent + b_exponent + a_shift + b_shift + post_shift == out_exponent + out_shift);
  }
}

void TestCopyScalarToElem64() {
  for (int i = -9; i < 100; i += 2) {
    for (int source = 0; source < 500; source++) {
      int value = -9 + (source % 100);
      int shift_a = (source * 3) % 64,
          shift_b = (source * 11) % 64;

      Scalar64 a;
      InitScalar64FromInt(i, &a);
      if (a.size + shift_a >= 64 || a.size + shift_b >= 64)
        continue;
      ShiftScalar64Left(shift_a, &a);

      int64_t data[20];
      for (int i = 0; i < 20; i++)
        data[i] = 0;
      data[2] = 1 << shift_b;
      Region64 region;
      InitRegion64(data, 10, 0, 5, &region);

      Elem64 elem;
      InitElem64(&region, region.data + 5, &elem);

      Scalar64 scalar;
      InitScalar64FromInt(value, &scalar);

      CopyScalar64ToElem(&scalar, &elem);

      Scalar64 scalar2;
      CopyElemToScalar64(&elem, &scalar2);

      double d1 = Scalar64ToDouble(&scalar),
          d2 = Scalar64ToDouble(&scalar2);
      assert(d1 == d2);


      { // test CopyScalar64ToVectorElem
        Vector64 vec;
        for (int i = 0; i < 20; i++)
          data[i] = 0;
        data[2] = 1 << shift_b;
        /* re-init region. */
        InitRegion64(data, 10, 0, 5, &region);
        InitVector64(&region, 10, 1, region.data, &vec);
        Scalar64 scalar;
        InitScalar64FromInt(value, &scalar);
        int index = 4;
        CopyScalar64ToVectorElem(&scalar, index, &vec);
        assert(value == Vector64ElemToDouble(index, &vec));
      }
    }
  }
}



int main() {
  TestGetShiftsForAdd();
  TestGetShiftsForAssign();
  TestGetShiftsForTwoArgMultiply();
  TestGetShiftsForTwoArgMultiplyAndAdd();
  TestFindSize();
  TestSetToInt();
  TestShift();
  TestAddScalar();
  TestMulScalar();
  TestInitVector();
  TestDotVector();
  TestAddVector();
  TestCopyVector();
  TestInvertScalar();
  TestSetMatrixVector64();
  TestCopyScalarToElem64();
  TestCopyIntToVector64Elem();
}
#endif /* FIXED_MATH_TEST */

