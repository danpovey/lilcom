#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "unistd.h"  /* For malloc and free */
#include "numpy/arrayobject.h"

/* The core library */
#include "lilcom.h"



/**
   Recursive internal implementation of lilcom_compress_int16
   @param [in] num_axes   The number of axes in the arrays (which must be the same, and
                     must be >= 1).
   @param [in] axis  The axis that this function is to process.    If equal to ndim-1, it
                     actually does the compression (this dimension corresponds to the
                     time axis).  This function is called with axis=0 at the top
                     level, and recurses.
                     For values 0 ... ndim-2, this function will loop and recurse.
   @param [in] input_data
                    Raw pointer to the input data; will have been obtained from
                    `input` and possibly shifted by outer calls of
                    this function
   @param [out] output_data  Raw pointer to the output data; obtained from output
                    and possibly shifted, as for input_data.
   @param [in] input  Must point to a NumPy array representing the input.  Used
                    for its dimension and stride information
   @param [in] output  Must point to a NumPy array representing the output data.  Used
                    for its dimension and stride information
   @param [in] lpc_order   User-specified number in [0..15], higher means slower
                    but less lossy compression.
   @param [in] bits_per_sample  User-specified number in [4..8]
   @param [in] conversion_exponent   User specified number that affects what
                    happens when we convert to float.  Will normally be 0.

   @return   Returns 0 on success.  On failure, returns 1 if there was a failure
             in lilcom_compress and 2 if there was a dimension mismatch detected
             by this function.
*/
static int compress_int16_internal(int num_axes, int axis,
                                   const int16_t *input_data,
                                   int8_t *output_data,
                                   PyObject *input, PyObject *output,
                                   int lpc_order, int bits_per_sample,
                                   int conversion_exponent) {
  assert(axis >= 0 && axis < num_axes);
  int dim = PyArray_DIM(input, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(int16_t),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(int8_t);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    if (PyArray_DIM(output, axis) != dim)
      return 2;
    for (int i = 0; i < dim; i++) {
      int ret = compress_int16_internal(num_axes, axis + 1, input_data,
                                        output_data, input, output,
                                        lpc_order, bits_per_sample,
                                        conversion_exponent);
      if (ret != 0)
        return ret;  /** Some kind of failure */
      input_data += input_stride;
      output_data += output_stride;
    }
  } else {
    /** The last axis-- the time axis. */
    int output_dim = PyArray_DIM(output, axis);

    int ret = lilcom_compress(input_data, dim, input_stride,
                              output_data, output_dim, output_stride,
                              lpc_order, bits_per_sample,
                              conversion_exponent);
    if (ret != 0)
      return ret;  /** Failure, e.g. invalid lpc_order, dim or exponent. */
  }
  return 0;  /** Success */
}

/**
   The following will document this function as if it were a native
   Python function.

    def compress_int16(input, output, lpc_order = 5, conversion_exponent = 0):
      """

      Args:
       input:  A numpy.ndarray with dtype=int16.  Must have at least
            one element.  The last axis is assumed to be the time axis.
            Caution: the dtypes are not checked here!
       output:  A numpy array with dtype=int8; the compressed signal
            will go in here.  Must be the same shape as `input` except
            the last dimension is greater by 4 (for the header).
       lpc_order:  A user-specifiable number in the range [0..15];
            higher values are slower but less lossy.
       bits_per_sample:  User-specified number in [4..8]
       conversion_exponent: A user-specifiable number in the range
            [-127..128].  The same value will be returned by `decompress_int16`.
            You won't normally want to modify the default.  It affects what
            happens when the user requests this data to be decompressed
            as float; search for this name in lilcom.h for further
            details.
       Return:
            Returns 0 on success, 1 if a failure was encountered in the
            core lilcom_compress code (this would only happen if lpc_order
            or conversion_exponent was out of range); or 2 if the
            shapes or types of `input` and/or `output` were not as
            expected (detected in the internal implementation function),
            or 3 if there was an error detected in this function
            directly.
     """
 */

static PyObject *compress_int16(PyObject *self, PyObject *args, PyObject * keywds)
{
  PyObject *input; /* The input signal, passed as a numpy array. */
  PyObject *output; /* The output signal, passed as a numpy array. */
  int lpc_order = 4,
      bits_per_sample = 8,
      conversion_exponent = 0;

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"input", "output",
                           "lpc_order","bits_per_sample",
                           "conversion_exponent", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|iii", kwlist,
                                   &input, &output,
                                   &lpc_order, &bits_per_sample,
                                   &conversion_exponent))
    goto error_return;

  const int16_t *input_data = (const int16_t*)PyArray_DATA(input);
  int8_t *output_data = (int8_t*) PyArray_DATA(output);
  if (!input_data || !output_data)
    goto error_return;

  int num_axes = PyArray_NDIM(input);
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;


  int ret = compress_int16_internal(num_axes, 0,
                                    input_data, output_data,
                                    input, output, lpc_order,
                                    bits_per_sample,
                                    conversion_exponent);
  return PyLong_FromLong(ret);
error_return:
  return PyLong_FromLong(3);
}


/**
   Recursive internal implementation of lilcom_decompress_int16
   @param [in] num_axes   The number of axes in the arrays (which must be the same, and
                     must be >= 1).
   @param [in] axis  The axis that this function is to process.    If equal to ndim-1, it
                     actually does the compression (this dimension corresponds to the
                     time axis).  This function is called with axis=0 at the top
                     level, and recurses.
                     For values 0 ... ndim-2, this function will loop and recurse.
   @param [in] input_data
                    Raw pointer to the input data; will have been obtained from
                    `input` and possibly shifted by outer calls of
                    this function
   @param [out] output_data  Raw pointer to the output data; obtained from output
                    and possibly shifted, as for input_data.
   @param [in] input  Must point to a NumPy array representing the input.  Used
                    for its dimension and stride information
   @param [in] output  Must point to a NumPy array representing the output data.
                    Used for its dimension and stride information

   @return     On success, returns the conversion exponent that was used for
               the compression; this will be in [-127..128] but usually 0.

               On failure, returns the following codes:

               1001  if a failure was encountered in the
                     core lilcom_decompress code (would typically mean
                     corrupted or invalid data)
               1002  If the input and output dimensions (for the non-time
                     axis) did not match
               1003  if a mismatch was noticed in the conversion
                     exponents from different sequences (which would
                     probably indicate that this matrix was originally
                     compressed from a float matrix.)
*/
static int decompress_int16_internal(int num_axes, int axis,
                                     const int8_t *input_data,
                                     int16_t *output_data,
                                     PyObject *input, PyObject *output) {
  assert(axis >= 0 && axis < num_axes);

  int conversion_exponent = -1;

  int input_dim = PyArray_DIM(input, axis),
      output_dim =  PyArray_DIM(output, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(int8_t),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(int16_t);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    if (output_dim != input_dim)
      return 1002;
    for (int i = 0; i < input_dim; i++) {
      int ret = decompress_int16_internal(num_axes, axis + 1, input_data,
                                          output_data, input, output);
      if (ret >= 1000)
        return ret;  /** Some kind of failure */

      if (i == 0) conversion_exponent = ret;
      else if (ret != conversion_exponent) return 1003;

      input_data += input_stride;
      output_data += output_stride;
    }
    return conversion_exponent;
  } else {
    /** The last axis-- the time axis. */
    int conversion_exponent;
    int ret = lilcom_decompress(input_data, input_dim, input_stride,
                                output_data, output_dim, output_stride,
                                &conversion_exponent);
    if (ret != 0)
      return 1001;  /** Failure in decompression, e.g. corrupted data */
    else
      return conversion_exponent;
  }
}


/**
   The following will document this function as if it were a native
   Python function.

    def decompress_int16(input, output):
      """

      Args:
       input:  A numpy.ndarray with dtype=int8.  Must have at least
            one element.  The last axis is assumed to be the time axis.
            Caution: the dtypes are not checked here!
       output:  A numpy array with dtype=int16; the compressed signal
            will go in here.  Must be the same shape as `input` except
            the last dimension is less by 4 (for the header).
       Return:
            On success:

              Returns the conversion exponent, which will be the number in the
              range [-127, 128] that was specified by the user to
              `compress_int16`, which by default is zero.  If it's not the case
              that all sequences had the same conversion_exponent, it would be
              an error (this would indicate that you were trying to decompress
              as int16 something that had originally been compressed from
              float); see error code 1003.

            On error, returns:

               1001  if a failure was encountered in the
                     core lilcom_decompress code (would typically mean
                     corrupted or invalid data)
               1002  if a problem such as a dimension mismatch was
                     noticed in decompress_int16_internal
               1003  if a mismatch was noticed in the conversion
                     exponents from different sequences (which would
                     probably indicate that this matrix was originally
                     compressed from a float matrix.)
               1004  if a problem such as a dimension mismatch
                     was noticed in this function.
     """
 */
static PyObject *decompress_int16(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject *input; /* The input signal, passed as a numpy array. */
  PyObject *output; /* The output signal, passed as a numpy array. */

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"input", "output", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist,
                                   &input, &output))

    goto error_return;

  const int8_t *input_data = (const int8_t*)PyArray_DATA(input);
  int16_t *output_data = (int16_t*) PyArray_DATA(output);
  if (!input_data || !output_data)
    goto error_return;

  int num_axes = PyArray_NDIM(input);
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int ret = decompress_int16_internal(num_axes, 0,
                                      input_data, output_data,
                                      input, output);
  return PyLong_FromLong(ret);
error_return:
  return PyLong_FromLong(3);
}


/**
   Recursive internal implementation of lilcom_compress_float
   @param [in] num_axes   The number of axes in the arrays (which must be the same, and
                     must be >= 1).
   @param [in] axis  The axis that this function is to process.    If equal to ndim-1, it
                     actually does the compression (this dimension corresponds to the
                     time axis).  This function is called with axis=0 at the top
                     level, and recurses.
                     For values 0 ... ndim-2, this function will loop and recurse.
   @param [in] input_data
                    Raw pointer to the input data; will have been obtained from
                    `input` and possibly shifted by outer calls of
                    this function
   @param [out] output_data  Raw pointer to the output data; obtained from output
                    and possibly shifted, as for input_data.
   @param [in] input  Must point to a NumPy array representing the input.  Used
                    for its dimension and stride information
   @param [in] output  Must point to a NumPy array representing the output data.  Used
                    for its dimension and stride information
   @param [in] lpc_order   User-specified number in [0..15], higher means slower
                    but less lossy compression.
   @param [in] temp_space  Either NULL, or a pointer to an int16_t array with
                    at least output.shape[-1] elements.  (Note: using python
                    syntax there.)

   @return   Returns 0 on success.  On failure, returns error codes 1, 2 or 3 if
             there was a failure in lilcom_compress_float (those are its error
             codes), and 4 if there was a dimension mismatch detected by this
             function.  Note: it will return an error if we had trouble compressing
             any of the sequences.  This should only happen if infinities and
             NaN's are encountered.
*/
static int compress_float_internal(int num_axes, int axis,
                                   const float *input_data,
                                   int8_t *output_data,
                                   PyObject *input, PyObject *output,
                                   int lpc_order, int bits_per_sample,
                                   int16_t *temp_space) {
  assert(axis >= 0 && axis < num_axes);
  int input_dim = PyArray_DIM(input, axis),
      output_dim = PyArray_DIM(output, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(float),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(int8_t);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    if (output_dim != input_dim)
      return 4;
    for (int i = 0; i < input_dim; i++) {
      int ret = compress_float_internal(num_axes, axis + 1, input_data,
                                        output_data, input, output,
                                        lpc_order, bits_per_sample,
                                        temp_space);
      if (ret != 0)
        return ret;  /** Some kind of failure */
      input_data += input_stride;
      output_data += output_stride;
    }
  } else {
    /** The last axis-- the time axis. */
    int ret = lilcom_compress_float(input_data, input_dim, input_stride,
                                    output_data, output_dim, output_stride,
                                    lpc_order, bits_per_sample,
                                    temp_space);
    if (ret != 0)
      return ret;  /** Failure, e.g. invalid lpc_order, dim or exponent. */
  }
  return 0;  /** Success */
}


/**
   The following will document this function as if it were a native
   Python function.

    def compress_float(input, output, lpc_order = 5):
      """

      Args:
       input:  A numpy.ndarray with dtype=float32.  Must have at least
            one element.  The last axis is assumed to be the time axis.
            Caution: the dtypes are not checked here!
       output:  A numpy array with dtype=int8; the compressed signal
            will go in here.  Must be the same shape as `input` except
            the last dimension is greater by 4 (for the header).
       lpc_order:  A user-specifiable number in the range [0..15];
            higher values are slower but less lossy.
       Return:
            Returns 0 on success; nonzero error codes on failure.
            Error code meanings:
            1  if lilcom_compress_float failed because num_samples, input_stride,
               output_stride or lpc_order had an invalid value.
            2  if there were infinitites or NaN's in the input data.
            3  if it failed to allocate a temporary array (only
               possible if you did not provide one).
            4  if an error such as a dimension mismatch was discovered
               in compress_float_internal()
            5  if an error (e.g. a dimension mismatch) was discovered
               in this function.
     """
 */
static PyObject *compress_float(PyObject *self, PyObject * args, PyObject * keywds)
{
  PyObject *input; /* The input signal, passed as a numpy array. */
  PyObject *output; /* The output signal, passed as a numpy array. */
  int lpc_order = 4,
      bits_per_sample = 8;
  int16_t *temp_space = NULL;

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"input", "output",
                           "lpc_order", "bits_per_sample", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ii", kwlist,
                                   &input, &output, &lpc_order,
                                   &bits_per_sample))
    goto error_return;

  const float *input_data = (const float*)PyArray_DATA(input);
  int8_t *output_data = (int8_t*) PyArray_DATA(output);
  if (!input_data || !output_data)
    goto error_return;

  int num_axes = PyArray_NDIM(input);
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int sequence_length = PyArray_DIM(input, num_axes - 1);
  temp_space = malloc(sizeof(int16_t) * sequence_length);
  if (temp_space == NULL)
    return PyLong_FromLong(3);  /* Return error-code 3 which means: failed to
                                 * allocate memory. */

  int ret = compress_float_internal(num_axes, 0,
                                    input_data, output_data,
                                    input, output, lpc_order,
                                    bits_per_sample, temp_space);
  free(temp_space);
  return PyLong_FromLong(ret);
error_return:
  if (temp_space != NULL)
    free(temp_space);
  return PyLong_FromLong(5);
}


/**
   The following will document this function as if it were a native
   Python function.

    def get_num_bytes(num_samples, bits_per_sample):
      """

      Args:
       num_samples: an integer > 0.
       bits_per_sample: an integer in [4..8].
      Returns:
       Returns the number of bytes that lilcom would use to compress
       a sequence with this num_samples and this bits_per_sample,
       or -1 if an error was encountered.
      """
 */
static PyObject *get_num_bytes(PyObject *self, PyObject * args, PyObject * keywds) {
  int num_samples, bits_per_sample;

  static char *kwlist[] = {"num_samples", "bits_per_sample", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "ii", kwlist,
                                   &num_samples, &bits_per_sample))
    goto error_return;

  int64_t num_bytes = lilcom_get_num_bytes(num_samples, bits_per_sample);
  return PyLong_FromLong(num_bytes);
error_return:
  return PyLong_FromLong(-1);

}


/**
   The following will document this function as if it were a native
   Python function.

    def get_time_axis_info(input):
      """
      Returns information which can be used to determine the shape that this
      array will have when decompressed by lilcom.  It can work out which axis
      is the time axis from the header information, essentially by trying
      each axis in turn and seeing if it could be the header; properties
      of the header ensure that two axes can't simultaneously satisfy this
      condition.

      Args:
       input:  A NumPy array of np.int8 that is the result of
               compressing data with lilcom (see compress() in
               lilcom_interface.py).

      Returns:
        On success, returns a tuple (time_axis, num_samples) where time_axis >=
        0 is the axis that corresponds to the time axis and num_samples is the
        length of the original data on that axis.
        On failure (e.g. this data was not a lilcom-compressed array),
        returns None.
      """
 */
static PyObject *get_time_axis_info(
    PyObject *self, PyObject * args, PyObject *keywds) {
  PyObject *input; /* The compressed data, as a NumPy array of int8 */

  static char *kwlist[] = { "input", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist,
                                   &input))
    goto error_return;

  const int8_t *input_data = (const int8_t*)PyArray_DATA(input);
  if (!input_data) goto error_return;
  int num_axes = PyArray_NDIM(input);
  int time_axis = -1;
  int64_t num_samples = 0;
  for (int axis = 0; axis < num_axes; axis++) {
    int dim = PyArray_DIM(input, axis),
        stride = PyArray_STRIDE(input, axis);
    int64_t uncompressed_dim  = lilcom_get_num_samples(input_data, dim, stride);
    if (uncompressed_dim != -1) {
      assert(time_axis < 0 && "There appear to be two time axes; "
             "this shouldn't be possible.");
      num_samples = uncompressed_dim;
      time_axis = axis;
    }
  }
  if (time_axis < 0)
    goto error_return;
  return PyTuple_Pack(2,
                      PyLong_FromLong(time_axis),
                      PyLong_FromLong(num_samples));
error_return:
  Py_RETURN_NONE;

}


/**
   Internal implementation of decompress_float().

     @param [in] num_axes   Number of axes in these arrays.  Must be >= 1
     @param [in] axis       Axis we are immediately working on.  Will be
                           called with 0, and will then recurse to num_axes - 1.
                           Must be in range [0..num_axes-1].
     @param [in] input_data  Data pointer for `input` object, possibly shifted
                           by previous recursion.
     @param [out] output_data  Data pointer for `output` object, possibly shifted
                           by previous recursion.
     @param [in] input   Input NumPy object containing floats-- needed only
                           for dimension info.
     @param [in] output   Output NumPy object containing floats-- needed only
                           for dimension info.

     @return
           Returns 0 on success, 1 if there was a failure in the core
           decompression routine (e.g. data was corrupted or not lilcom-compressed
           data), and 2 if there was some kind of dimension mismatch with
           `input` and `output`.
*/
int decompress_float_internal(int num_axes, int axis,
                              const int8_t *input_data,
                              float *output_data,
                              PyObject *input, PyObject *output) {
  assert(axis >= 0 && axis < num_axes);
  int input_dim = PyArray_DIM(input, axis),
      output_dim = PyArray_DIM(output, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(int8_t),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(float);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    if (input_dim != output_dim)
      return 2;
    for (int i = 0; i < input_dim; i++) {
      int ret = decompress_float_internal(num_axes, axis + 1, input_data,
                                          output_data, input, output);
      if (ret != 0)
        return ret;  /** Some kind of failure */
      input_data += input_stride;
      output_data += output_stride;
    }
    return 0;  /** Success */
  } else {
    /** The last axis-- the time axis. */
    int ret = lilcom_decompress_float(input_data, input_dim, input_stride,
                                      output_data, output_dim, output_stride);
    return ret;
  }
}




 /**
   NOTE: the documentation below will document this function AS IF it were
   a Python function.

   def decompress_float(input, output):
   """
   This function decompresses data from int8_t to float.  The data is assumed
   to have previously been compressed by `compress_float`.

   Args:
   input     Must be of type numpy.ndarray, with dtype=int8.  The
   last axis is assumed to be the time axis, and the last
   axis must have dimension > 4.
   output    Must be of type numpy.ndarray, with dtype=int16.  Must
   be of the same shape as `input`, except the dimension on
   the last axis must be less than that of `input` by 4.

   Return:
       0 on success
       1 on failure in the decompression routine (e.g. if the data was corrupted
           or was not lilcom-compressed data)
       2 if there was some dimension mismatch between the input and output
         arrays
       3 If the inputs did not have the correct types or had different num-axes.
  """
*/
static PyObject *decompress_float(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *input; /* The input signal, passed as a numpy array. */
  PyObject *output; /* The output signal, passed as a numpy array. */

  static char *kwlist[] = {"input", "output", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist,
                                   &input, &output))
    goto error_return;

  const int8_t *input_data = (const int8_t*)PyArray_DATA(input);
  float *output_data = (float*) PyArray_DATA(output);
  if (!input_data || !output_data) goto error_return;

  int num_axes = PyArray_NDIM(input);
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int ret = decompress_float_internal(num_axes, 0,
                                      input_data, output_data,
                                      input, output);

  return PyLong_FromLong(ret);
error_return:
  return PyLong_FromLong(3);
}

static PyMethodDef LilcomMethods[] = {
  { "compress_int16", (PyCFunction)compress_int16, METH_VARARGS | METH_KEYWORDS,
    "Lossily compresses samples of int16 sequence data (e.g. audio data) int8_t."},
  { "compress_float", (PyCFunction)compress_float, METH_VARARGS | METH_KEYWORDS,
    "Lossily compresses samples of float sequence data (e.g. audio data) int8_t."},
  { "decompress_int16", (PyCFunction)decompress_int16, METH_VARARGS | METH_KEYWORDS,
    "Decompresses a compressed signal to int16"  },
  { "decompress_float", (PyCFunction)decompress_float, METH_VARARGS | METH_KEYWORDS,
    "Decompresses a compressed signal to float16"  },
  { "get_num_bytes", (PyCFunction)get_num_bytes, METH_VARARGS | METH_KEYWORDS,
    "Returns the number of bytes needed to compress a sequence" },
  { "get_time_axis_info", (PyCFunction)get_time_axis_info, METH_VARARGS | METH_KEYWORDS,
    "Returns the number of bytes needed to compress a sequence" },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef lilcom =
{
    PyModuleDef_HEAD_INIT,
    "lilcom_c_extension", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    LilcomMethods
};

PyMODINIT_FUNC PyInit_lilcom_c_extension(void)
{
    import_array();
    return PyModule_Create(&lilcom);
}
