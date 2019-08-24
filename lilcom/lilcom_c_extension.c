// Needed definitions and includes for python C interface
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "unistd.h"  /* For malloc and free */

#include "numpy/arrayobject.h"

/* The core library */
#include "./lilcom.h"



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
   @param [in] conversion_exponent   User specified number that affects what
                    happens when we convert to float.  Will normally be 0.

   @return   Returns 0 on success.  On failure, returns 1 if there was a failure
             in lilcom_compress and 2 if there was a dimension mismatch detected
             by this function.
*/
int compress_int16_internal(int num_axes, int axis,
                            const int16_t *input_data,
                            int8_t *output_data,
                            PyObject *input, PyObject *output,
                            int lpc_order,
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
                                        lpc_order, conversion_exponent);
      if (ret != 0)
        return ret;  /** Some kind of failure */
      input_data += input_stride;
      output_data += output_stride;
    }
  } else {
    /** The last axis-- the time axis. */
    if (PyArray_DIM(output, axis) != dim + 4)
      return 2;
    int ret = lilcom_compress(dim, input_data, input_stride,
                              output_data, output_stride,
                              lpc_order, conversion_exponent);
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

static PyObject *compress_int16(PyObject *self, PyObject * args, PyObject * keywds)
{
  PyObject *input; // The input signal, passed as a numpy array.
  PyObject *output; // The output signal, passed as a numpy array.
  int lpc_order; // LPC Order defined in the core function (more information -> lilcom.h)
  int conversion_exponent; // Conversion Exponent defined in the core function (more information -> lilcom.h)

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"input", "output",
                           "lpc_order", "conversion_exponent", NULL}; //definition of keywords received in the function call from python
  // Parsing Arguments: All input arguments are obligatory. Default assignment left for python wrapper.
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOii", kwlist,
                                   &input, &output,
                                   &lpc_order, &conversion_exponent))
    goto error_return;

  const int16_t *input_data = (const int16_t*)PyArray_DATA(input);
  int8_t *output_data = (int8_t*) PyArray_DATA(output);


  // Initializing shape related variables
  int num_axes = PyArray_NDIM(input); // Get the number of dimensions
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;


  // Calles the internal function which recursively calles it self until it's ready for compression
  int ret = compress_int16_internal(num_axes, 0,
                                    input_data, output_data,
                                    input, output, lpc_order,
                                    conversion_exponent);
  return Py_BuildValue("i", ret);
error_return:
  return Py_BuildValue("i", 3);
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
   @param [in] output  Must point to a NumPy array representing the output data.  Used
                    for its dimension and stride information

   @return     On success, returns the conversion exponent that was used for
               the compression; this will be in [-127..128] but usually 0.

               On failure, returns the following codes:

               1001  if a failure was encountered in the
                     core lilcom_decompress code (would typically mean
                     corrupted or invalid data)
               1002  if a problem such as a dimension mismatch was
                     noticed in this function
               1003  if a mismatch was noticed in the conversion
                     exponents from different sequences (which would
                     probably indicate that this matrix was originally
                     compressed from a float matrix.)
*/
int decompress_int16_internal(int num_axes, int axis,
                              const int8_t *input_data,
                              int16_t *output_data,
                              PyObject *input, PyObject *output) {
  assert(axis >= 0 && axis < num_axes);


  /* ISSUE: the function is supposed to be returning a conversion exponent and an error code, to do this we must make a consideration to handel it here and in the python wrapper  */
  int conversion_exponent = 0;

  int dim = PyArray_DIM(input, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(int8_t),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(int16_t);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    int conversion_exponent = -1;
    if (PyArray_DIM(output, axis) != dim)
      return 2;
    for (int i = 0; i < dim; i++) {
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
    if (PyArray_DIM(output, axis) != dim - 4)
      return 1002;
    int conversion_exponent;
    int ret = lilcom_decompress(dim - 4, input_data, input_stride,
                                output_data, output_stride,
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
static PyObject *decompress_int16(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *input; // The input signal, passed as a numpy array.
  PyObject *output; // The output signal, passed as a numpy array.

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"X", "Y", NULL};
  // Parsing Arguments

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOii", kwlist,
                                   &input, &output,
                                   &lpc_order, &conversion_exponent))

    goto error_return;

  const int8_t *input_data = (const int8_t*)PyArray_DATA(input);
  int16_t *output_data = (int16_t*) PyArray_DATA(output);


  // Initializing shape related variables
  int num_axes = PyArray_NDIM(input); // Get the number of dimensions
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int ret = decompress_int16_internal(num_axes, 0,
                                      input_data, output_data,
                                      input, output);
  return Py_BuildValue("i", ret);
error_return:
  return Py_BuildValue("i", 3);
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
                    at least input.shape[-1] elements.  (Note: using python
                    syntax there.)

   @return   Returns 0 on success.  On failure, returns error codes 1, 2 or 3 if
             there was a failure in lilcom_compress_float (those are its error
             codes), and 4 if there was a dimension mismatch detected by this
             function.  Note: it will return an error if we had trouble compressing
             any of the sequences.  This should only happen if infinities and
             NaN's are encountered.
*/
int compress_float_internal(int num_axes, int axis,
                            const float *input_data,
                            int8_t *output_data,
                            PyObject *input, PyObject *output,
                            int lpc_order, int16_t *temp_space) {
  assert(axis >= 0 && axis < num_axes);
  int dim = PyArray_DIM(input, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(float),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(int8_t);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    if (PyArray_DIM(output, axis) != dim)
      return 4;
    for (int i = 0; i < dim; i++) {
      int ret = compress_float_internal(num_axes, axis + 1, input_data,
                                        output_data, input, output,
                                        lpc_order, temp_space);
      if (ret != 0)
        return ret;  /** Some kind of failure */
      input_data += input_stride;
      output_data += output_stride;
    }
  } else {
    /** The last axis-- the time axis. */
    if (PyArray_DIM(output, axis) != dim + 4)
      return 4;
    int ret = lilcom_compress_float(dim, input_data, input_stride,
                                    output_data, output_stride,
                                    lpc_order, temp_space);
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
  PyObject *input; // The input signal, passed as a numpy array.
  PyObject *output; // The output signal, passed as a numpy array.
  int lpc_order = 5; // LPC Order defined in the core function (more information -> lilcom.h)
  int16_t *temp_space = NULL;

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"input", "output",
                           "lpc_order", NULL}; //definition of keywords received in the function call from python
  // Parsing Arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|i", kwlist,
                                   &input, &output,
                                   &lpc_order))
    goto error_return;

  const float *input_data = (const float*)PyArray_DATA(input);
  int8_t *output_data = (int8_t*) PyArray_DATA(output);



  // Initializing shape related variables
  int num_axes = PyArray_NDIM(input); // Get the number of dimensions
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int sequence_length = PyArray_DIM(input, num_axes - 1);
  temp_space = malloc(sizeof(int16_t) * sequence_length);
  if (temp_space == NULL)
    return Py_BuildValue("i", 3);  /* Error code meaning: failed to allocate memory. */


  int ret = compress_float_internal(num_axes, 0,
                                    input_data, output_data,
                                    input, output, lpc_order,
                                    temp_space);
  free(temp_space);
  return Py_BuildValue("i", ret);
error_return:
  if (temp_space != NULL)
    free(temp_space);
  return Py_BuildValue("i", 5);
}



/**
  TODO: add comments
*/
int decompress_float_internal(int num_axes, int axis,
                              const int8_t *input_data,
                              float *output_data,
                              PyObject *input, PyObject *output) {
  assert(axis >= 0 && axis < num_axes);
  int dim = PyArray_DIM(input, axis),
      input_stride = PyArray_STRIDE(input, axis) / sizeof(int8_t),
      output_stride = PyArray_STRIDE(output, axis) / sizeof(float);

  if (axis < num_axes - 1) {  /** Not the time axis. */
    int conversion_exponent = -1;
    if (PyArray_DIM(output, axis) != dim)
      return 2;
    for (int i = 0; i < dim; i++) {
      int ret = decompress_float_internal(num_axes, axis + 1, input_data,
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
    if (PyArray_DIM(output, axis) != dim - 4)
      return 1002;
    int ret = lilcom_decompress_float(dim - 4, input_data, input_stride,
                                output_data, output_stride);
    if (ret != 0)
      return 1001;  /** Failure in decompression, e.g. corrupted data */
    else
      return ret;
  }
}


/**
  TODO: add comments
*/
static PyObject *decompress_float(PyObject *self, PyObject *args, PyObject *keywds)
{
  PyObject *input; // The input signal, passed as a numpy array.
  PyObject *output; // The output signal, passed as a numpy array.

  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"X", "Y", NULL};
  // Parsing Arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist,
                                   &input, &output))
    goto error_return;

  const int8_t *input_data = (const int8_t*)PyArray_DATA(input);
  float *output_data = (float*) PyArray_DATA(output);

  // Initializing shape related variables
  int num_axes = PyArray_NDIM(input); // Get the number of dimensions
  if (PyArray_NDIM(output) != num_axes)
    goto error_return;

  int ret = decompress_float_internal(num_axes, 0,
                                      input_data, output_data,
                                      input, output);

  return Py_BuildValue("i", ret);
error_return:
  return Py_BuildValue("i", 3);
}

static PyMethodDef LilcomMethods[] = {
  { "compress_int16", (PyCFunction)compress_int16, METH_VARARGS | METH_KEYWORDS, "Lossily compresses samples of int16 sequence data (e.g. audio data) int8_t."},
  { "compress_float", (PyCFunction)compress_float, METH_VARARGS | METH_KEYWORDS, "Lossily compresses samples of float sequence data (e.g. audio data) int8_t."},
  { "decompress_int16", (PyCFunction)decompress_int16, METH_VARARGS| METH_KEYWORDS, "Decompresses a compressed signal to int16"  },
  { "decompress_float", (PyCFunction)decompress_float, METH_VARARGS| METH_KEYWORDS, "Decompresses a compressed signal to float16"  },
  { NULL, NULL, 0, NULL }
};

static struct PyModuleDef lilcom =
{
    PyModuleDef_HEAD_INIT,
    "lilcom", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    LilcomMethods
};

PyMODINIT_FUNC PyInit_lilcom(void)
{
    import_array();
    return PyModule_Create(&lilcom);
}
