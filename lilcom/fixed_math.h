#include <math.h>
#include <stdint.h>
#include <assert.h>



/*
  Defines a region of memory underlying a Vector64 or Matrix64.  The
  data is not owned here.

  Conceptually, the data in `data` represents a floating point number,
  where each element should be multiplied by pow(2.0, exponent) to
  get the actual value.
 */
typedef struct {
  /*  number of elements in `data` */
  int dim;
  /* exponent defines what the data means as a floating point number
     (multiply data[i] by pow(2.0, exponent) to get the float value). */
  int exponent;
  /* size is a number >= 0 such that abs(data[x] < (1 << size))
     for all 0 <= x < dim.  Bound does not have to be tight (i.e.
     it's OK if size is a little larger than needed). */
  int size;

  /*  The underlying data.  Not owned here from a memory management point of view,
      but the framework does not expect any given memory location to be
      owned by more than one Region64. */
  int64_t *data;
} Region64;

typedef struct {
  /* `region` is the top-level memory region that owns this data... it's needed
     when the exponent changes, as it all has to be kept consistent (there is
     one exponent per memory region.) */
  Region64 *region;
  /* number of elements in the vector.  Must be > 0 for the Vector64 to be
     valid.. */
  int dim;
  /* stride.  must be != 0. */
  int stride;
  /* Pointer to the zeroth element. */
  int64_t *data;
} Vector64;

typedef struct {
  /* memory region that owns this data. */
  Region64 *region;
  /* number of rows in the matrix */
  int num_rows;
  /* number of columns in the matrix */
  int num_cols;
  /* Distance in elements between rows */
  int row_stride;
  /* column stride, i.e. distance in elements between columns i and i+1.  Would
   * normally be 1. WARNING: current code will crash if this is not 1,
   * since we haven't needed it to be != 1 yet. */
  int col_stride;
  /* pointer to element [0,0]. */
  int64_t *data;
} Matrix64;


typedef struct {
  /* exponent defines what the data means as a floating point number
     (multiply data by pow(2.0, exponent) to get the float value). */
  int exponent;
  /* size is *the smallest* number >= 0 such that abs(data[x] < (1 << size)). */
  int size;
  int64_t data;
} Scalar64;

typedef struct {
  Region64 *region;
  int64_t *data;
} Elem64;  /* Elem64 is like Scalar64 but it is part of an existing region. */


/*
  Initializes a Region64.
     @param [in]  data    data underlying the region; should be an array of at least `dim` elements
     @param [in]  dim     number of elements in the region
     @param [in]  exponent   exponent with which to interpret whatever data is currently
                          in the region (i.e. it will be interpreted as that integer
                          number + 2^exponent).
     @param [in]  size_hint   caller's approximation to the `size` of the region,
                          meaning the smallest power of 2 >= 0that is greater than all
                          the absolute values of elements of the region.
     @param [out] region   region  the region object to be created
 */
void InitRegion64(int64_t *data, int dim, int exponent, int size_hint, Region64 *region);

extern inline void InitVector64(Region64 *region, int dim, int stride, int64_t *data, Vector64 *vec) {
  vec->region = region;
  vec->dim = dim;
  vec->stride = stride;
  vec->data = data;
  assert(dim > 0 && dim <= region->dim && stride != 0 &&
         data >= region->data && data < region->data + region->dim &&
         data + ((dim-1)*stride) >= region->data &&
         data + ((dim-1)*stride) < region->data + region->dim);
}


extern inline void InitMatrix64(Region64 *region,
                                int num_rows, int row_stride,
                                int num_cols, int col_stride,
                                int64_t *data, Matrix64 *mat) {
  mat->region = region;
  mat->num_rows = num_rows;
  mat->row_stride = row_stride;
  mat->num_cols = num_cols;
  mat->col_stride = col_stride;
  // WARNING: col-stride must be 1 currently.
  assert(col_stride == 1);
  mat->data = data;
  // TODO: the following assertions would need to change
  // if we choose to allow negative row-stride and col-stride != 1.
  int max_offset = (num_rows-1) * row_stride + (num_cols-1) * col_stride;
  assert(num_rows > 0 && num_cols > 0 && col_stride == 1 &&
         row_stride >= num_cols * col_stride &&
         data >= region->data && data + max_offset <  region->data + region->dim);
}

extern inline void InitElem64(Region64 *region, int64_t *data, Elem64 *elem) {
  elem->region = region;
  elem->data = data;
}


/* Shift the data elements right by this many bits and adjust the exponent so
   the number it represents is unchanged.
         @param [in] right_shift   A shift value >= 0
         @param [in,out] region   The region to be shifted
*/
void ShiftRegion64Right(int right_shift, Region64 *region);

/* Shift the data elements left by this many bits and adjust the exponent so
   the number it represents is unchanged.
         @param [in] left_shift   A shift value >= 0
         @param [in,out] region   The region to be shifted
*/
void ShiftRegion64Left(int left_shift, Region64 *region);

/* Shift the data right by this many bits, and adjust the exponent so
   the number it represents is unchanged. */
void ShiftScalar64Right(int right_shift, Scalar64 *scalar);

/* Shift the data left by this many bits, and adjust the exponent so
   the number it represents is unchanged.  This is present only
   for testing purposes. */
void ShiftScalar64Left(int left_shift, Scalar64 *scalar);

/* Copies data from `src` to `dest`; they must have the same dimension
   and be non-overlapping. */
void CopyVector64(const Vector64 *src, Vector64 *dest);

/* Updates the `size` field of `vec` to be accurate. */
void FixVector64Size(const Vector64 *vec);

/* like BLAS saxpy.  y := a*x + y.
   x and y must be from different regions.   */
void AddScalarVector64(const Scalar64 *a, const Vector64 *x, Vector64 *y);

/* Does y := a * x.   x and y must be from different regions. */
void SetScalarVector64(const Scalar64 *a, const Vector64 *x, Vector64 *y);

/* does y[i] += a for each element of y. */
void Vector64AddScalar(const Scalar64 *a, Vector64 *y);

/* does y[i] := a for each element of y. */
void Vector64SetScalar(const Scalar64 *a, Vector64 *y);


/* y := a * b.  It is OK if some of the pointer args are the same.  */
void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* y := a + b.  It is OK if some of the pointer args are the same.  */
void AddScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);


/* Sets the elements of this vector to zero.  Does touch the `size`
   or exponent.  Currently intended for use inside implementation functions.
 */
void ZeroVector64(Vector64 *a);

/* Sets the size for this region to the appropriate value.
   (See docs for the `size` member).  63 <= size_hint <= 0
   is a hint for what we think the size might be.
*/
void SetRegion64Size(int size_hint, Region64 *r);

inline void NegateScalar64(Scalar64 *a) { a->data *= -1; }

/* Sets this scalar to an integer. */
void InitScalar64FromInt(int64_t i, Scalar64 *a);

/* Computes dot product between two Vector64's:
   y := a . b */
void DotVector64(const Vector64 *a, const Vector64 *b, Scalar64 *y);

/* Computes matrix-vector product:
     y := M x.   Note: y must not be in the same region as x or m.
*/
void SetMatrixVector64(const Matrix64 *m, const Vector64 *x,
                       const Vector64 *y);



/* Copies the data in the scalar to the `elem`. */
void CopyScalarToElem64(const Scalar64 *scalar, Elem64 *elem);

/* Does Y := a. */
void CopyElemToScalar64(const Elem64 *a, Scalar64 *y);

/* Copies the i'th element of `a` to y:   y = a[i] */
void CopyToScalar64(const Vector64 *a, int i, Scalar64 *y);

/* Sets an element of a vector to a scalar:  y[i] = a. */
void CopyFromScalar64(const Scalar64 *a, int i, Vector64 *y);

/* Multiplies 64-bit scalars.  Must all be different pointers. */
void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* does: y := a.  Just copies all the elements of the struct. */
extern inline void CopyScalar64(const Scalar64 *a, Scalar64 *y) {
  y->size = a->size;
  y->exponent = a->exponent;
  y->data = a->data;
}

/* Computes the inverse of a 64-bit scalar: does b := 1.0 / a.  The pointers do
   not have to be different.  Will die with assertion or numerical exception if
   a == 0. */
void InvertScalar64(const Scalar64 *a, Scalar64 *b);

/* Adds two 64-bit scalars.
   does: y := a + b. The pointers do not have to be different. */
void AddScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);



/* Subtracts two 64-bit scalars. Must all be different pointers.
   does: y := a - b. */
void SubtractScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/*  Divides scalars:  y := a / b. */
void DivideScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* Does: y := 1.0 / a */
void InvertScalar64(const Scalar64 *a,  Scalar64 *y);

/* Convert to double- needed only for testing. */
double Scalar64ToDouble(const Scalar64 *a);

/* Returns nonzero if a and b are similar within a tolerance.  Intended mostly
   for checking code.*/
int Scalar64ApproxEqual(const Scalar64 *a, const Scalar64 *b, float tol);

