// Note: this will be compiled by invoking ../setup.py, e.g. cd ..; python3 ./setup.py build

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"
#include <string.h>  // for memcpy

#define LILCOM_FORMAT_VERSION 0
#define LILCOM_HEADER_LEN 2  // Must not be changed.  Header is 'L' then
                             // LILCOM_FORMAT_VERSION.

/* The core library */
#include "compression.h"
#include <cstring>  // for memcpy


extern "C" {



/**
   The following will document this function as if it were a native
   Python function.

    def compress_float(input, config):
      """

      Args:
       input:  A numpy.ndarray with dtype=np.float32 and number of axes
           in the range [1..15].  Caution: it will be replaced with
           the approximate ersion of itself that you'll get after
           decompressing the return value.
       meta:  A list of integers containing some meta-information:
            [ tick_power, coeff1, coeff2, .. ] where tick_power (e.g. -8),
            which must be in the range [-20,20] (this decision was
            arbitrary), is the power to which we'll raise 2 to use
	    or the `tick` (distance between 2 encoded elements, dictating the
            accuracy), and the coefficients correspond to estimated
            regression coefficients, one per axis, each multiplied by
            256 and rounded to the nearest integer.  See
            documentation of `regression_coeffs` arg of
            CompressFloat(), in compression.h, for more details about
            the regression coefficients.


       Return:

            On success, returns the compressed data as a bytes object
            which can be decompressed.  On failure or if one of the
            args was not right, returns None.  On memory allocation
            failure, raises MemoryError.

            (Note: this code could also die with an assertion, which
            would indicate a code error at the C++ level).  """
 */
static PyObject *compress_float(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *input; /* The input signal, passed as a numpy array of np.float32's. */
  PyObject *meta; // List of python ints containing metadata in the form
                  // [tick_power, coeff1, coeff2.. ]

  static const char *kwlist[] = {"input", "meta", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", (char**)kwlist,
                                   (PyObject**)&input, &meta))
    Py_RETURN_NONE;

  int num_axes = PyArray_NDIM(input),
    list_size = PyList_Size(meta);
  if (num_axes <= 0 || num_axes >= 16 || list_size != num_axes + 1 ||
      !PyLong_Check(PyList_GetItem(meta, 0)))
    Py_RETURN_NONE;

  int tick_power = PyLong_AsLong(PyList_GetItem(meta, 0));

  int dims[16], strides[16];
  int regression_coeffs[16];


  for (int i = 0; i < num_axes; i++) {
    int int_coeff = PyLong_AsLong(PyList_GetItem(meta, i + 1));
    assert(int_coeff >= -256 && int_coeff <= 256);
    regression_coeffs[i] = int_coeff;
    dims[i] = PyArray_DIM(input, i);
    strides[i] = PyArray_STRIDE(input, i) / sizeof(float);
  }

  float *input_data = (float*)PyArray_DATA(input);

  try {
    std::vector<char> ans = CompressFloat(tick_power, input_data,
					  num_axes, dims, strides,
					  regression_coeffs);
    if (ans.empty()) {
      // Something went wrong.  An error message may have been printed.
      Py_RETURN_NONE;
    }
    PyObject *ret = PyBytes_FromStringAndSize(NULL, LILCOM_HEADER_LEN + ans.size());
    char* ret_data = PyBytes_AsString(ret);
    ret_data[0] = 'L';
    ret_data[1] = LILCOM_FORMAT_VERSION;
    memcpy(ret_data + 2, &(ans[0]), ans.size());
    return ret;
  } catch (std::bad_alloc) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failure to allocate memory in lilcom compression");
    return NULL;
  }
}

  /*
    Gets the bytes object as a char* pointer and length (both excluding the header);
    returns true on success, false on failure; in that case the user should return
    NULL and an exception will have been set.
   */
  bool lilcom_check_bytes_header(PyObject *bytes_in, char **bytes_array, Py_ssize_t *length) {
    if (PyBytes_AsStringAndSize(bytes_in, bytes_array, length) != 0) {
      PyErr_SetString(PyExc_ValueError, "lilcom: Expected bytes object as 1st arg");
      return false;
    } else if (*length <= LILCOM_HEADER_LEN) {
      PyErr_SetString(PyExc_ValueError, "lilcom: Length of string was too short");
      return false;
    } else if (**bytes_array != 'L') {
      PyErr_SetString(PyExc_ValueError, "lilcom: Lilcom-compressed data must begin with L");
      return false;
    } else if ((*bytes_array)[1] != LILCOM_FORMAT_VERSION) {
      PyErr_SetString(PyExc_ValueError, "lilcom: Trying to decompress data from a future format "
                      "version (use newer code)");
      return false;
    }
    // remove the header from what we return.
    *bytes_array += LILCOM_HEADER_LEN;
    *length -= LILCOM_HEADER_LEN;
    return true;
  }


  /**
     The following will document this function as if it were a native
    Python function.

       def get_float_matrix_shape(bytes_in):
         """
         Return the shape required for a matrix into which the data in this
	 bytes object can be decompressed.

         Args:
            `bytes_in` must be a bytes object
         Return:
            If `bytes_in` had the expected format (as compressed by compress_float),
            this function will return a tuple of ints:
               (dim1, dim2, dim3 ...)
	    containing the shape of the compressed matrix; if something went
            wrong with the argument-passing, e.g. wrong types, it will return
            None.

         Will throw ValueError if the argument is not of type `bytes` or does
         not seem to be a Lilcom-compressed string of the right dimension; will
         return None if the wrong number of args was given or something went
         wrong getting the size.  """
   */
  static PyObject *get_float_matrix_shape(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    char *bytes_array;
    Py_ssize_t length;
    if (nargs != 1)
      Py_RETURN_NONE;
    PyObject *bytes_in = args[0];
    if (!lilcom_check_bytes_header(bytes_in, &bytes_array, &length))
      return NULL;

    int meta[17];
    if (GetCompressedDataShape(bytes_array, length, meta)) {
      int num_axes = meta[0];
      assert(num_axes > 0 && num_axes <= 16);  // was checked in
                                               // GetCompressedDataShape()
      PyObject *ans = PyTuple_New(num_axes);
      for (int i = 0; i < num_axes; i++) {
        int dim = meta[i+1];
        assert(dim > 0);  // was checked in GetCompressedDataSize()
        PyTuple_SET_ITEM(ans, i, PyLong_FromLong(dim));
      }
      return ans;
    } else {
      Py_RETURN_NONE;
    }
  }


  /**
    The following will document this function as if it were a native Python
    function.

       def decompress_float(bytes_in, array_out)
         """
         Decompress an array of float that was compressed with compress_float()


         Args:
            byts_in: a `bytes` object that was returned from compress_float()

            array_out: must be a NumPy array with dtype numpy.float32, and
               shape equal to the result of calling get_float_matrix_shape() on
               this same bytes object.

         Return:
           Returns 0 on success, a nonzero code if there was a
           failure in the decompression.
           raises ValueError if the args appeared to have the wrong type or
           could not be decompressed.
         """

   */
  static PyObject *decompress_float(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    char *bytes_array;
    Py_ssize_t length;
    if (nargs != 2)
      Py_RETURN_NONE;
    PyObject *bytes_in = args[0];
    PyArrayObject *output = (PyArrayObject*)args[1];

    if (!lilcom_check_bytes_header(bytes_in, &bytes_array, &length))
      return NULL;

    int dims[16], strides[16];
    int num_axes = PyArray_NDIM(output);
    for (int i = 0; i < num_axes; i++) {
      dims[i] = PyArray_DIM(output, i);
      strides[i] = PyArray_STRIDE(output, i) / sizeof(float);
    }

    int ans = DecompressFloat(bytes_array, length,
                              (float*)PyArray_DATA(output),
                              num_axes, dims, strides);
    return PyLong_FromLong(ans);
  }



  static PyMethodDef LilcomExtensionMethods[] = {
    {"compress_float", (PyCFunction) compress_float, METH_VARARGS | METH_KEYWORDS,
     "Compresses the supplied data and returns compressed form as bytes object."},
    {"get_float_matrix_shape", (PyCFunction) get_float_matrix_shape, METH_FASTCALL,
     "Takes a bytes object as returned from compress_float(), and returns a "
     "tuple representing the shape of the array that was compressed, or "
     "None on error."},
    {"decompress_float", (PyCFunction) decompress_float, METH_FASTCALL,
     "Takes a bytes object and an appropriately sized NumPy array of floats, "
     "with shape as given by get_float_matrix_shape(), and decompresses the "
     "data into the array.  Returns 0 on success, and a nonzero code or None "
     "on failure."},
    {NULL, NULL, 0, NULL}
  };

  static struct PyModuleDef lilcom_extension =
  {
    PyModuleDef_HEAD_INIT,
    "lilcom_extension", /* name of module */
    "",           /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    LilcomExtensionMethods
  };

  PyMODINIT_FUNC PyInit_lilcom_extension(void) {
    import_array();
    return PyModule_Create(&lilcom_extension);
  }


}  /* extern "C" */
