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

#define FM_MIN_SIZE 36  /* The minimum `size` we allow if right shifting... this
                           is equivalent to the number of bits of precision we
                           keep.  This should basically be >= 32, because when we multiply
                           in the typical case we'll only keep 32 bits of precision. */


#define FM_MAX(a,b) ((a) > (b) ? (a) : (b))


inline static uint64_t FM_ABS(int64_t a) {
  return (uint64_t)(a > 0 ? a : -a);
}




/**
   Get shifts for a 2-arg multplication; will be an operation of the form

   out = ((a >> a_shift) * (b >> b_shift)) >> post_shift.

   We need to keep the multiplication from overflowing.  That requires that:

   (a_size - a_shift) + (b_size - b_shift)  < 64

   subject to the larger of those two target sizes being as large as possible...

   We also want to choose the smallest post_shift >= 0 so that
       post_shift   <= target_size
   where 64 < target_size <= 32 is chosen by the caller.
 */
inline static void GetShiftsFor2ArgMultiply(int a_size, int b_size,
                                            int *a_shift, int *b_shift,
                                            int *post_shift, int target_size) {
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
}



/*
  This function is used for debugging; it checks that `size`
  is the smallest integer >= 0 such that value < (1 << size).
 */
inline static void CheckSize(uint64_t value, int size) {
  assert((value & ((uint64_t)1<<63)) == 0 && "CheckSize() cannot accept input >= 2^63. "
         "Inputs must be converted to positive values!");
  int ok = (value & ~((((uint64_t)1) << size) - 1)) == 0 &&
      ((value == 0 && size == 0) ||
       (value & (((uint64_t)1) << (size - 1) )) != 0);
  if (!ok) {
    printf("Error: value=%lld size=%d\n", value, size);
    exit(1);
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
    CheckSize(value, ans);
    return ans;
  } else {
    /* value > (1 << guess).  Keep shifting neg_mask left till it fits. */
    neg_mask <<= 1;
    ans++;
    while ((neg_mask & value) != 0) {
      neg_mask <<= 1;
      ans++;
    }
    CheckSize(value, ans);
    return ans;
  }
}


float Scalar64ToFloat(const Scalar64 *a) {
  return a->data * pow(2.0, a->exponent);
}

void SetScalar64ToInt(int64_t i, Scalar64 *a) {
  a->exponent = 0;
  a->data = i;
  uint64_t i_abs = FM_ABS(i);
  a->size = FindSize(1, i_abs);
  CheckSize(i_abs, a->size);
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


void ShiftRegion64Right(int right_shift, Region64 *region) {
  assert(right_shift >= 0);
  region->exponent += right_shift;
  region->size -= right_shift;
  if (region->size < 0)
    region->size = 0;
  int dim = region->dim;
  int64_t *data = region->data;
  for (int i = 0; i < dim; i++)
    data[i] >>= right_shift;
}


void ShiftRegion64Left(int left_shift, Region64 *region) {
  assert(left_shift >= 0);
  region->exponent -= left_shift;
  if (region->size != 0)
    region->size += left_shift;

    region->size = 0;
  int dim = region->dim;
  int64_t *data = region->data;
  for (int i = 0; i < dim; i++)
    data[i] >>= left_shift;
}


void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y) {
  int a_size = a->size,
      b_size = b->size,
      a_shift, b_shift, post_shift;
  GetShiftsFor2ArgMultiply(a_size, b_size,
                           &a_shift, &b_shift,
                           &post_shift, FM_TARGET_SIZE);
  y->data = ((a->data >> a_shift) * (b->data >> b_shift)) >> post_shift;
  printf("a_shift = %d, b_shift = %d, post_shift = %d\n",
         a_shift, b_shift, post_shift);
  y->exponent = a->exponent + b->exponent + a_shift + b_shift + post_shift;
  int size_guess = a_size + b_size - a_shift - b_shift - post_shift;
  y->size = FindSize(size_guess, FM_ABS(y->data));
}


/* does: y := a.  Just copies all the elements of the struct. */
void CopyScalar64(const Scalar64 *a, Scalar64 *y) {
  y->size = a->size;
  y->exponent = a->exponent;
  y->data = a->data;
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


void AddVec64(const Vector64 *x, Scalar64 *a, Vector64 *y) {
  assert(x->region != y->region && x->dim == y->dim);
  int x_size = x->region->size, a_size = a->size,
      x_shift, a_shift, post_shift;
  /* target_size == 62  ... we may modify the post-shift later. */
  GetShiftsFor2ArgMultiply(x_size, a_size,
                           &x_shift, &a_shift, &post_shift,
                           62);
  int prod_exponent = x->region->exponent + a->exponent +
      x_shift + a_shift + post_shift,
      y_exponent = y->region->exponent;
  int prod_size = x_size + a_size - x_shift - a_shift - post_shift,
      y_size = y->region->size;

  /* sum_size_as_y is the size of the sum if we were to use y's exponent.  the +
     1 is because the summation might increase the size by at most 1. */
  int sum_size_as_y = FM_MAX(y_size, prod_size + (prod_exponent - y_exponent)) + 1;

  if (sum_size_as_y > 62) {
    /* We need to shift y right. */
    int right_shift = sum_size_as_y - FM_TARGET_SIZE;
    ShiftRegion64Right(right_shift, y->region);
    sum_size_as_y = FM_TARGET_SIZE;
    y_exponent = y->region->exponent;
  } else if (sum_size_as_y < FM_MIN_SIZE) {
    int left_shift = FM_TARGET_SIZE - sum_size_as_y;
    ShiftRegion64Left(left_shift, y->region);
    sum_size_as_y = FM_TARGET_SIZE;
    y_exponent = y->region->exponent;
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
      tot |= (uint64_t)((prod1 | prod2) | (prod3 | prod4));
      y_data[i * y_stride] += prod1;
      y_data[(i+1) * y_stride] += prod2;
      y_data[(i+2) * y_stride] += prod3;
      y_data[(i+3) * y_stride] += prod4;
    }
    for (; i < dim; i++) {
      int64_t prod = (a_shifted * (x_data[i * x_stride] >> x_shift)) >> post_shift;
      y_data[i * y_stride] += prod;
      tot |= (uint64_t)prod;
    }
    int size = FindSize(sum_size_as_y, tot);
    if (size > y->region->size)
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
    if (size > y->region->size)
      y->region->size = size;
  }
}



void Vec64AddScalar(const Scalar64 *a, Vector64 *y) {
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
         guess helps FixVec64Size(). */
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


void Dot64(const Vector64 *a, const Vector64 *b, Scalar64 *y) {
  assert(a->dim == b->dim);
  int dim = a->dim, a_stride = a->stride, b_stride = b->stride;
  const int64_t *a_data = a->data, *b_data = b->data;
  /* We need the product of (a[i] >> shift_a) * (b[i] >> shift_b) to
     have absolute value < 2^63 so as to not overflow signed int64_t.  We achieve this
     by making sure that (a[i] >> shift_a) have absolute value <2^32 and
     (b[i] >> shift_b) have absolute value <2^31.
     We then post-shift the result right to make it fit in 48 bits or so.
     This is more accuracy than the 32 bits we really use, but also
     substantially less than 2^64.
  */
  int a_size = a->region->size, b_size = b->region->size,
      a_shift = (a_size > 32 ? a_size - 32 : 0),
      b_shift = (b_size > 31 ? b_size - 31 : 0),
      y_size = a_size + b_size - a_shift - b_shift,
      post_shift = (y_size > FM_TARGET_SIZE ? y_size - FM_TARGET_SIZE : 0);
  y_size -= post_shift;

  int64_t tot = 0;
  for (int i = 0; i < dim; i++) {
    int64_t prod = ((a_data[i * a_stride] >> a_shift) *
                    (b_data[i * b_stride] >> b_shift)) >> post_shift;
    tot += prod;
  }
  y->data = tot;
  y->exponent = a->region->exponent + b->region->exponent -
      (a_shift + b_shift + post_shift);
  y->size = FindSize(y_size, FM_ABS(tot));
}


void Vec64Add(const Vector64 *x, Scalar64 *a, Vector64 *y) {

}


void SetRegion64Size(int size_hint, Region64 *r) {
  int64_t *data = r->data;
  int dim = r->dim;
  int i;

  int64_t tot = 0;  /* tot will be the `or` of all the absolute values. */
  for (i = 0; i + 4 <= dim; i += 4) {
    int64_t a1 = FM_ABS(data[dim]),
        a2 = FM_ABS(data[dim + 1]),
        a3 = FM_ABS(data[dim + 2]),
        a4 = FM_ABS(data[dim + 3]);
    a1 |= a2;
    a3 |= a4;
    tot = a1 | a3;
  }
  for (; i < dim; i++) {
    int64_t a = FM_ABS(data[dim]);
    tot |= a;
  }
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
    SetScalar64ToInt(i, &a);
    float f = Scalar64ToFloat(&a);
    assert(f - i == 0);
  }
}


void TestShift() {
  for (int i = -10; i < 10; i++) {
    Scalar64 a;

    for (int j = 0; j < 50; j += 10) {
      SetScalar64ToInt(i, &a);
      ShiftScalar64Left(j, &a);
      ShiftScalar64Right((j - 2 >= 0 ? j - 2 : 0), &a);
      float f = Scalar64ToFloat(&a);
      assert(f - i == 0);
    }
  }
}


void TestAddScalar() {
  for (int i = -10; i < 10; i += 5) {
    for (int shift_i = 0; shift_i < 50; shift_i++) {
      Scalar64 ii;
      SetScalar64ToInt(i, &ii);
      ShiftScalar64Left(shift_i, &ii);
      for (int j = -10; j < 10; j += 3) {
        for (int shift_j = 0; shift_j < 50; shift_j++) {
          Scalar64 jj;
          SetScalar64ToInt(j, &jj);
          ShiftScalar64Left(shift_j, &jj);

          Scalar64 kk;
          AddScalar64(&ii, &jj, &kk);

          float kf = Scalar64ToFloat(&kk);
          assert(kf == i + j);
        }
      }
    }
  }
}



void TestMulScalar() {
  for (int i = -10; i < 10; i += 5) {
    for (int shift_i = 0; shift_i < 50; shift_i++) {
      Scalar64 ii;
      SetScalar64ToInt(i, &ii);
      ShiftScalar64Left(shift_i, &ii);
      for (int j = -10; j < 10; j += 3) {
        for (int shift_j = 0; shift_j < 50; shift_j++) {
          Scalar64 jj;
          SetScalar64ToInt(j, &jj);
          ShiftScalar64Left(shift_j, &jj);

          Scalar64 kk;
          MulScalar64(&ii, &jj, &kk);

          float kf = Scalar64ToFloat(&kk);
          printf("i = %d, j = %d, k = %f, if = %f, jf= %f\n", i, j, kf,
                 Scalar64ToFloat(&ii), Scalar64ToFloat(&jj));
          assert(kf == i * j);

        }
      }
    }
  }
}



int main() {
  TestFindSize();
  TestSetToInt();
  TestShift();
  TestAddScalar();
  TestMulScalar();
}
#endif /* FIXED_MATH_TEST */

