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

#define FM_MIN_SIZE 36  /* The minimum `size` we allow if right shifting... this
                           is equivalent to the number of bits of precision we
                           keep.  This should basically be >= 32, because when we multiply
                           in the typical case we'll only keep 32 bits of precision. */


#define FM_MAX(a,b) ((a) > (b) ? (a) : (b))


inline static uint64_t FM_ABS(int64_t a) {
  return (uint64_t)(a > 0 ? a : -a);
}


#ifndef NDEBUG
void PrintRegion64(Region64 *region) {
  fprintf(stderr, "{ Region64, dim = %d, exponent = %d, size = %d, data = [ ",
          region->dim, region->exponent, region->size);
  for (int i = 0; i < region->dim; i++)
    fprintf(stderr, "%lld ", region->data[i]);
  fprintf(stderr, "] }\n");
}
#endif


#ifndef NDEBUG
/** Checks that the region currently has the size that it should have; dies if not.. */
void CheckRegion64Size(const Region64 *r_in) {
  Region64 *r = (Region64*) r_in;  /* Remove const. */
  int size = r->size;
  SetRegion64Size(size, r);
  assert(size == r->size && "Region had wrong size.");
}
#endif


/**
   Get shifts for a 2-arg multplication; will be an operation of the form

   out = ((a >> a_shift) * (b >> b_shift)) >> post_shift.

   We need to keep the multiplication from overflowing, which requires:

     (a_size - a_shift) + (b_size - b_shift)  <= 63

   subject to the larger of those two target sizes being as large as possible.

   We also want to choose the smallest post_shift >= 0 so that the
   size of the product after shifting is <= target_size,
   where 64 < target_size <= 32 is chosen by the caller.

  This comes down to:
      (a_size - a_shift) + (b_size - b_shift) - post_shift <= target_size

     @param [in] half_max_size  Half the maximum size we allow for the product.
                               Would normally be 31, but if there is a summation
                              involved we decrease it.
     @param [in] a_size  Size of one of the operands.   See doc for CheckSize()
                        for definition.  Must be in [0,63].
     @param [in] b_size  Size of the other operand.     See doc for CheckSize()
                        for definition.  Must be in [0,63.]
     @param [out] a_shift, b_shift, post_shift
                        These are the shifts this function outputs... will
                        be used in an equation like:
                output_data = ((a_data >> a_shift) * (b_data >> b_shift)) >> pos_shift.
     @param [out] target_size   Maximum size of product after shifting right
                        by pos_shift; we will shift the product right as
                        needed to satisfy this limit.
*/
inline static void GetShiftsFor2ArgMultiply(int a_size, int b_size, int target_size,
                                            int *a_shift, int *b_shift,
                                            int *post_shift) {
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
  int prod_size = a_size - *a_shift + b_size - *b_shift;
  assert(prod_size < 64);
  if (prod_size > target_size) {
    *post_shift = prod_size - target_size;
  } else {
    *post_shift = 0;
  }
  assert(a_size + b_size - *a_shift - *b_shift - *post_shift <= target_size);
  assert(a_size + b_size - *a_shift - *b_shift <= 63);
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
    fprintf(stderr, "Error: value=%lld size=%d\n", value, size);
    return 0;
  } else {
    return 1;
  }
}

/*
  This is to be called when you have measured that the
  largest absolute element has size `size` (c.f. FindSize() below
  for definition.  It changes the size of the underlying region as
  appropriate.
*/
inline static void EnsureSizeAtLeast(Vector64 *vec, int size) {
  if (vec->region->dim == vec->dim ||
      vec->region->size < size)
    vec->region->size = size;
}

/*
  Returns the smallest integer i >= 0 such that
  (1 << i) > value.

  `guess` may be any value in [0,63]; it is an error if it is outside that
  range.  If it's close to the true value it will be faster.
 */
inline static int FindSize(int guess, uint64_t value) {
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
    return ans;
  }
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
    /* Setting the exponent to a very negative value interacts better with
       things like AddVector64. */
    a->region->exponent = -1000;
    a->region->size = 0;
  }
}

double Scalar64ToDouble(const Scalar64 *a) {
  return a->data * pow(2.0, a->exponent);
}

void InitScalar64FromInt(int64_t i, Scalar64 *a) {
  a->exponent = 0;
  a->data = i;
  uint64_t i_abs = FM_ABS(i);
  a->size = FindSize(1, i_abs);
  assert(CheckSize(i_abs, a->size));
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
      a_shift, b_shift, post_shift;
  GetShiftsFor2ArgMultiply(a_size, b_size, FM_TARGET_SIZE,
                           &a_shift, &b_shift,
                           &post_shift);
  y->data = ((a->data >> a_shift) * (b->data >> b_shift)) >> post_shift;
  y->exponent = a->exponent + b->exponent + a_shift + b_shift + post_shift;
  int size_guess = a_size + b_size - a_shift - b_shift - post_shift;
  y->size = FindSize(size_guess, FM_ABS(y->data));
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
      b->size = FindSize(32, FM_ABS(b->data));
      b->exponent = -63 - a_exponent - a_right_shift;
    } else {
      b->data = -(negative_2_63 / a_data);
      b->size = FindSize(64 - a_size, FM_ABS(b->data));
      b->exponent = -63 - a_exponent;
    }
  } else {
    int p = a_size + FM_TARGET_SIZE;  /* Note: is a_size < 63 - FM_TARGET_SIZE,
                                         so p < 63. This guarantees that 1<<p is
                                         positive. */
    int64_t big_number = 1 << p;
    b->data = big_number / a_data;
    b->size = FindSize(p - a_size, FM_ABS(b->data));
    b->exponent = -p - a_exponent;
  }
}

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
      y->size = FindSize(b_size, FM_ABS(y->data));
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
      y->size = FindSize(a_size, FM_ABS(y->data));
    } else { // This should be an extremely rare case, so just handle by recursion.
      Scalar64 a2;
      CopyScalar64(a, &a2);
      ShiftScalar64Right(63 - FM_TARGET_SIZE, &a2);
      AddScalar64(&a2, b, y);
    }
  }
}


void AddVector64(const Vector64 *x, const Scalar64 *a, Vector64 *y) {
  int x_size = x->region->size, a_size = a->size,
      x_shift, a_shift, post_shift;

  /* target_size == 62  ... we may modify the post-shift later. */
  GetShiftsFor2ArgMultiply(x_size, a_size, 62,
                           &x_shift, &a_shift, &post_shift);

  int prod_exponent = x->region->exponent + a->exponent +
      x_shift + a_shift + post_shift,
      y_exponent = y->region->exponent;
  int prod_size = x_size + a_size - x_shift - a_shift - post_shift,
      y_size = y->region->size;

  /* sum_size_as_y is the size of the sum if we were to use y's exponent.  the +
     1 is because the summation of whatever is already in y plus the new part
     might increase the size by at most 1. */
  int sum_size_as_y = FM_MAX(y_size, prod_size + (prod_exponent - y_exponent)) + 1;

  if (sum_size_as_y > 62) {
    /* We need to shift y right.  This should be fairly rare.  We handle this by recursion
       because if x->region == y->region, shifting y would invalidate some of the
       numbers we have computed.
       Note: the shifting doesn't really do any work if y->region->size == 0 because
       y's data is zero.
    */
    int right_shift = sum_size_as_y - FM_TARGET_SIZE;
    ShiftRegion64Right(right_shift, y->region);
    AddVector64(x, a, y);
    return;
  } else if (sum_size_as_y < FM_MIN_SIZE) {
    /* We need to shift y left.  This should be fairly rare.  We handle this by
       recursion because if x->region == y->region, shifting y would invalidate
       some of the numbers we have computed.
       Note: the shifting doesn't really do any work if y->region->size == 0 because
       y's data is zero.
    */
    int left_shift = FM_TARGET_SIZE - sum_size_as_y;
    ShiftRegion64Left(left_shift, y->region);
    AddVector64(x, a, y);
    return;
  }

  // post_shift will be right shift if positive, left if negative
  post_shift = post_shift + y_exponent - prod_exponent;

  if (post_shift >= 0) {
    int64_t a_shifted = a->data >> a_shift;
    int dim = x->dim, x_stride = x->stride, y_stride = y->stride;
    int64_t *x_data = x->data, *y_data = y->data;
    uint64_t tot = 0;  /* `tot` is a bit pattern keeping track of max size. */
    int i;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift,
          prod2 = (a_shifted * (x_data[(i+1) * x_stride] >> x_shift)) >> post_shift,
          prod3 = (a_shifted * (x_data[(i+2) * x_stride] >> x_shift)) >> post_shift,
          prod4 = (a_shifted * (x_data[(i+3) * x_stride] >> x_shift)) >> post_shift;
      tot |= ((FM_ABS(prod1) | FM_ABS(prod2)) | (FM_ABS(prod3)| FM_ABS(prod4)));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift;
      y_data[i * y_stride] += prod;
      tot |= FM_ABS(prod);
    }
    int size = FindSize(sum_size_as_y, tot);
    if (size > y->region->size || y->dim == y->region->dim)
      y->region->size = size;
  } else {
    /* If we are shifting the product left that means we had *room* to shift it
     * left, and due to our logic above the size of the product after
     * left-shifting is <= 62.  That means the size of the product
     * before-left-shifting was < 62, which means we would not have set nonzero
     * post-shift (notice we gave 62 to `target_size` arg to
     * GetShiftsFor2ArgMultiply()).
     */
    assert(x_shift == 0 && a_shift == 0);

    /* We can implement the shifting by shifting just a. */
    int64_t a_shifted = a->data << -post_shift;
    uint64_t tot = 0;  /* `tot` is a bit pattern keeping track of max size. */
    int dim = x->dim, x_stride = x->stride, y_stride = y->stride;
    int64_t *x_data = x->data, *y_data = y->data;
    int i;
    for (i = 0; i + 4 <= dim; i += 4) {
      int64_t prod1 = a_shifted * x_data[i * x_stride],
          prod2 = a_shifted * x_data[(i+1) * x_stride],
          prod3 = a_shifted * x_data[(i+2) * x_stride],
          prod4 = a_shifted * x_data[(i+3) * x_stride];
      tot |= (uint64_t)((prod1 | prod2) | (prod3 | prod4));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = a_shifted * x_data[i * x_stride];
      y_data[i * y_stride] += prod;
      tot |= prod;
    }
    int size = FindSize(sum_size_as_y, tot);
    if (size > y->region->size || y->dim == y->region->dim)
      y->region->size = size;
  }
}



void Vector64AddScalar(const Scalar64 *a, Vector64 *y) {
  Region64 *y_region = y->region;
  int a_size = a->size,
      y_size = y_region->size,
      max_size = (a_size > y_size ? a_size : y_size);
  int shift = a->exponent - y_region->exponent;
  /* `shift` is how much we'd left-shift a->data, assuming we leave
     y->exponent untouched.  If negative, implies a right shift.*/

  max_size = FM_MAX(a_size + shift, y_size);
  if (max_size >= 60) {
    /* The size of the numbers is getting dangerously close to the limit of the data type.
       Shift y right quite a bit.  We don't try to get right up to the limit of 63 bits,
       because in most cases we only get 32 bits of accuracy anyway (due to needing to
       shift right prior to multiplication).
    */
    int right_shift = FM_TARGET_SIZE - max_size;
    ShiftRegion64Right(right_shift, y_region);
    shift = a->exponent - y_region->exponent;
    y_size = y_region->size;
    if (a_size + shift > y_size) {
      /* it will probably end up being this large.  Having a good
         guess helps FixVector64Size(). */
      y_region->size = a_size + shift;
    }
  }
  int64_t a_value;
  if (shift >= 0) {
    a_value = a->data << shift;
  } else {
    // TODO: make this work if >> is not arithmetic.
    a_value = a->data >> shift;
  }
  // left-shift.
  int64_t *y_data = y->data;
  int dim = y->dim,
      stride = y->stride;
  uint64_t or_value = 0;
  for (int64_t i = 0; i < dim; i++) {
    /* TODO: vectorize this loop? */
    int64_t val = y_data[i*stride] + a_value;
    or_value |= (int64_t) FM_ABS(val);
    y_data[i*stride] = val;
  }
  int new_size = FindSize(y_region->size, or_value);
  if (new_size > y_region->size)
    y_region->size = new_size;
}

static int fm_num_bits[32] = {0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

void DotVector64(const Vector64 *a, const Vector64 *b, Scalar64 *y) {
  int dim = a->dim;
  assert(a->dim == b->dim);
  int dim_size;  /* dim_size is the number of bits by which the a sum over this
                  * many terms could be greater than the elements of the sum. */
  if (dim < 32)
    dim_size = fm_num_bits[dim];
  else
    dim_size = FindSize(6, dim);

  /* We won't let the elements of the summation be greater than target_size. */
  int target_size = FM_TARGET_SIZE - dim_size;

  int a_size = a->region->size, b_size = b->region->size,
      a_shift, b_shift, post_shift;
  GetShiftsFor2ArgMultiply(a_size, b_size, target_size,
                           &a_shift, &b_shift, &post_shift);

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
  fprintf(stderr, "a_shift = %d, b_shift = %d, prod_shift = %d\n", a_shift, b_shift, post_shift);
  y->data = sum;
  y->exponent = a->region->exponent + b->region->exponent +
      (a_shift + b_shift + post_shift);
  /* dim_size/2 is just a compromise between 0 and dim_size (correcting the
     magnitude for the summation).  the truth will probably be somewhere in
     between, but dim_size/2 has some mathematical justification if you assume
     the summands are uncorrelated and think about the variance. */
  int size_guess = a_size + b_size + dim_size/2 - (a_shift + b_shift + post_shift);
  y->size = FindSize(size_guess, FM_ABS(sum));
}


/* Computes matrix-vector product:
   y = M x.   Note: y and x must not be in the same region;
   and currently, y must occupy its entire region.  (We
   can relax this requirement later as needed.

   Note, this code closely mirrors the code of DotVector64.
*/
void MatTimesVector64(const Matrix64 *m, const Vector64 *x,
                     const Vector64 *y) {
  assert(y->region != x->region && y->region != m->region &&
         x->dim == m->num_cols && y->dim == m->num_rows);
  int num_rows = m->num_rows,
      num_cols = m->num_cols;
  int col_size;  /* col_size is the number of bits by which the a sum over this
                  * many terms could be greater than the elements of the sum. */
  if (num_cols < 32)
    col_size = fm_num_bits[num_cols];
  else
    col_size = FindSize(6, num_cols);
  assert(CheckSize(num_cols, col_size));
  /* We won't let the elements of the summation be greater than target_size. */
  int target_size = FM_TARGET_SIZE - col_size;

  int m_size = m->region->size, x_size = x->region->size,
      m_shift, x_shift, post_shift;
  GetShiftsFor2ArgMultiply(m_size, x_size, target_size,
                           &m_shift, &x_shift, &post_shift);

  /*
  int product_max_size = a_size + b_size - (a_shift + b_shift + post_shift) + dim_size,
      product_exponent = a->region->exponent + b->region->exponent +
      (a_shift + b_shift + post_shift);
  if (y->dim == y->region->dim) {
    y->region->exponent = product_exponent;
  } else {
    int y_right_shift = product_exponent - y->region->exponent;
    if (y_right_shift > 0) {
      // we need to shift y's data right or shift our data left.
      if (y_right_shift <= post_shift &&
          product_max_size + y_right_shift <= FM_MAX_SIZE) {
        // We can just decrease our post_shift to resolve this without needing
        // to shift y's region.
        post_shift -= y_right_shift;
        product_exponent -= y_right_shift;
        product_max_size += y_right_shift;
      } else {
        ShiftRegion64Right(y_right_shift, y->region);
      }
    } else if (y_right_shift < 0) {
      // We need to shift y's data left or shift our data right.
      if (product_max_size + y_right_shift >= FM_MIN_SIZE) {
        // We can just increase our post_shift to resolve this without needing
        // to shift y's region.
        post_shift -= y_right_shift;
        product_exponent -= y_right_shift;
        product_max_size += y_right_shift;
      } else {
        if (
        ShiftRegion64Left(-y_right_shift, y->region);
      }
    }


      int diff = y->region->exponent - exponent,
          max_left_shift = FM_TARGET_SIZE

          }

  int i;
  //HERE


       y_size_guess = a_size + b_size - (a_shift + b_shift + post_shift) + dim_size/2

  Vector64 *
           */


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
  fprintf(stderr, "tot = %lld\n", tot);
  /* TODO: find a more efficient way to get the size. */
  int size = FindSize(size_hint, tot);
  r->size = size;
}


#ifdef FIXED_MATH_TEST
void TestFindSize() {
  uint64_t n = 1;
  assert(FindSize(0, 0) == 0);
  for (int i = 1; i < 63; i++) {
    for (int j = 0; j <= 63; j++) {
      assert(FindSize(j, n) == i);
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

void TestInitVector() {
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
    for (int i = 0; i < 100; i++)
      data2[i] = i + 4;

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

    AddVector64(&vec1, &scalar, &vec2);


    Scalar64 dot_prod;
    DotVector64(&vec1, &vec2, &dot_prod);

    double f = Scalar64ToDouble(&dot_prod);
    fprintf(stderr, "shift{1,2,3} = %d,%d->%d,%d, f = %f, ref_sum = %lld, size = %d, scalar exponent = %d\n",
            shift1, shift2, -region2.exponent, shift3, f, ref_sum, scalar.size, scalar.exponent);
    assert(f == ref_sum);
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
      MulScalar64(&a, &a_inv, &one);
      double f = Scalar64ToDouble(&one);
      assert(f - 1.0 < pow(2.0, -31));
    }
  }
}

int main() {
  TestFindSize();
  TestSetToInt();
  TestShift();
  TestAddScalar();
  TestMulScalar();
  TestInitVector();
  TestDotVector();
  TestAddVector();
  TestInvertScalar();
}
#endif /* FIXED_MATH_TEST */

