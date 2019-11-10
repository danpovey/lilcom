#include <math.h>
#include <stdint.h>



/*
  Defines a region of memory underlying a Vector64 or Matrix64.  The
  data is not owned here.

  Conceptually, the data in `data` represents a floating point number,
  where each element should be multiplied by pow(2.0, exponent) to
  get the actual value.
 */
typedef struct {
  /* exponent defines what the data means as a floating point number
     (multiply data[i] by pow(2.0, exponent) to get the float value). */
  int exponent;
  /* size is a number >= 0 such that abs(data[x] < (1 << size))
     for all 0 <= x < dim.  Bound does not have to be tight (i.e.
     it's OK if size is a little larger than needed). */
  int size;

  /*  number of elements in `data` */
  int dim;
  /*  The underlying data.  Not owned here from a memory management point of view,
      but the framework does not expect any given memory location to be
      owned by more than one Region64. */
  int64_t *data;
} Region64;

typedef struct {
  /* number of elements in the vector.  Must be > 0 for the Vector64 to be
   * valid.. */
  int dim;
  /* stride.  must be != 0. */
  int stride;
  /* Pointer to the zeroth element. */
  int64_t *data;
  /* `owner` is the top-level memory region that owns this data... it's needed
     when the exponent changes, as it all has to be kept consistent (there is
     one exponent per memory region.) */
  Region64 *region;
} Vector64;

typedef struct {
  /* number of rows in the matrix */
  int num_rows;
  /* number of columns in the matrix */
  int num_cols;
  /* row stride, i.e. distance in elements between row i and i+1. */
  int row_stride;
  /* column stride, i.e. distance in elements between columns i and i+1.  Would
   * normally be 1. */
  int col_stride;
  /* pointer to element [0,0]. */
  int64_t *data;
  /* memory region that owns this data. */
  Region64 *region;
} Matrix64;

typedef struct {
  /* exponent defines what the data means as a floating point number
     (multiply data by pow(2.0, exponent) to get the float value). */
  int exponent;
  /* size is *the smallest* number >= 0 such that abs(data[x] < (1 << size)). */
  int size;
  int64_t data;
} Scalar64;


/*
  Initializes a Region64.
 */
void InitRegion64(int64_t *data, int dim, Region64 *region);

/* Shift the data right by this many bits and adjust the exponent so the
   number it represents is unchanged.  */
void ShiftRegion64Right(int right_shift, Region64 *region);

/* Shift the data right by this many bits, and adjust the exponent so
   the number it represents is unchanged. */
void ShiftScalar64Right(int right_shift, Scalar64 *scalar);

/* Shift the data right by this many bits, and adjust the exponent so
   the number it represents is unchanged.  This is present only
   for testing purposes. */
void ShiftScalar64Left(int left_shift, Scalar64 *scalar);




/* Copies data from `src` to `dest`; they must have the same dimension
   and be non-overlapping. */
void Vec64Copy(const Vector64 *src, Vector64 *dest);

/* Updates the `size` field of `vec` to be accurate. */
void FixVec64Size(const Vector64 *vec);

/* like BLAS saxpy.  y := a*x + y.
   x and y must be from different regions.
 */
void AddVec64(const Vector64 *x, Scalar64 *a, Vector64 *y);

/* does y[i] += a for each element of y. */
void Vec64AddScalar(const Scalar64 *a, Vector64 *y);

/* y := a * b.  It is OK if some of the pointer args are the same.  */
void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* y := a + b.  It is OK if some of the pointer args are the same.  */
void AddScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);


/* Sets this vector to zero. */
void SetZero(Vector64 *a);

/* Sets the size for this region to the appropriate value.
   (See docs for the `size` member).  63 <= size_hint <= 0
   is a hint for what we think the size might be.
*/
void SetRegion64Size(int size_hint, Region64 *r);

inline void NegateScalar64(Scalar64 *a) { a->data *= -1; }

/* Sets this scalar to an integer. */
void SetScalar64ToInt(int64_t i, Scalar64 *a);

/* Computes dot product between two Vector64's */
void Dot64(const Vector64 *a, const Vector64 *b, Scalar64 *y);

/* Copies the i'th element of `a` to y:   y = a[i] */
void CopyToScalar64(const Vector64 *a, int i, Scalar64 *y);

/* Sets an element of a vector to a scalar:  y[i] = a. */
void CopyFromScalar64(const Scalar64 *a, int i, Vector64 *y);

/* Multiplies 64-bit scalars.  Must all be different pointers. */
void MulScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* does: y := a.  Just copies all the elements of the struct. */
void CopyScalar64(const Scalar64 *a, Scalar64 *y);

/* Adds two 64-bit scalars. Must all be different pointers.
   does: y := a + b. */
void AddScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* Subtracts two 64-bit scalars. Must all be different pointers.
   does: y := a - b. */
void SubtractScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/*  Divides scalars:  y := a / b. */
void DivideScalar64(const Scalar64 *a, const Scalar64 *b, Scalar64 *y);

/* Does: y := 1.0 / a */
void InvertScalar64(const Scalar64 *a,  Scalar64 *y);

/* Convert to float- needed only for testing. */
float Scalar64ToFloat(const Scalar64 *a);

/* Returns nonzero if a and b are similar within a tolerance.  Intended mostly
   for checking code.*/
int Scalar64ApproxEqual(const Scalar64 *a, const Scalar64 *b, float tol);

