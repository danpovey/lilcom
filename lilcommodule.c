// Needed definitions and includes for python C interface
#define PY_SSIZE_T_CLEAN
#include <Python.h>
 
#include "numpy/arrayobject.h"

// The core library 
#include "./lilcom.h"


/**
   Lossily compresses a given numpy array of 16-bit integer sequence data.
   It will first convert the numpy array data to a C `int16_t` array and then 
   passes the array to the core lilcom core library, finally it returns the result
   in form of a numpy array.
 */
static PyObject * compress(PyObject * self, PyObject * args, PyObject * keywds)
{
  /* Defining all variables */
  PyObject *signal_input; // The input signal, passed as a numpy array.
  PyObject *signal_output; // The output signal, passed as a numpy array.
  int n_dims = 0; // Number of dimensions of the given numpy array
  int input_stride; // The number of integers between to consecutive samples
  int output_stride; // The number of integers between to consecutive samples in output
  int n_samples = 0; // number of samples given in the numpy array
  // INPUT IS DEFINED INSIDE IF CONDITIONS
  int8_t *output; // The one dimensional vectorized output array which will be modified by the core function
  int16_t *input; // The one dimensional vectorized input array which will be given to the core function
  int lpc_order = 5; // LPC Order defined in the core function (more information -> lilcom.h)

  /* Reading and information - extracting for input data 
    From the python function there are two numpy arrays and an intger (optional) LPC_order 
    passed to this madule. Following part will parse the set of variables and store them in corresponding
    objects.
  */
  static char *kwlist[] = {"X", "Y" , "lpc_order", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|i", kwlist, &signal_input, &signal_output, &lpc_order)) 
    Py_RETURN_FALSE;
  
  // Initializing shape related variables
  n_dims = PyArray_NDIM(signal_input); // Getting the number of dimensions
  n_samples = PyArray_DIM(signal_input , 0); // Getting the first dimension
  /* In cases that numpy array is not contiguous then stride values are not equal to 1
    The stride values given in bytes and division to the size of variables gives the 
    strides as blocks.
   */
  input_stride = PyArray_STRIDE(signal_input, 0)/sizeof(int16_t);
  output_stride = PyArray_STRIDE(signal_output, 0)/sizeof(int8_t);

  /* Access to the data part of the numpy array */
  input = PyArray_DATA(signal_input);
  output = PyArray_DATA(signal_output);
  
  /* Calling the core function */
  lilcom_compress(n_samples, input, input_stride, output, output_stride, lpc_order);

  /* Making the resulting array */
  npy_intp * output_dimensions = malloc(sizeof(npy_intp)*2);
  output_dimensions[0] = n_samples;
  output_dimensions[1] = output_stride;
  PyArrayObject * output_array = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT8, (void*) output);

  // /* Returning numpy array */
  PyObject *returner = PyArray_Return(output_array);
  return returner;
}



static PyObject * decompress(PyObject * self, PyObject * args,  PyObject * keywds)
{
   /* Defining all variables */
  PyObject *signal_input; // The input signal, passed as a numpy array.
  PyObject *signal_output; // The output signal, passed as a numpy array.
  int n_dims = 0; // Number of dimensions of the given numpy array
  int input_stride; // The number of integers between to consecutive samples
  int output_stride; // The number of integers between to consecutive samples in output
  int n_samples = 0; // number of samples given in the numpy array
  // INPUT IS DEFINED INSIDE IF CONDITIONS
  int16_t *output; // The one dimensional vectorized output array which will be modified by the core function
  int8_t *input; // The one dimensional vectorized input array which will be given to the core function
  /* Reading and information - extracting for input data 
    From the python function there are two numpy arrays and an intger (optional) LPC_order 
    passed to this madule. Following part will parse the set of variables and store them in corresponding
    objects.
  */
  static char *kwlist[] = {"X", "Y", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist, &signal_input, &signal_output)) 
    Py_RETURN_FALSE;
  
  // Initializing shape related variables
  n_dims = PyArray_NDIM(signal_input); // Getting the number of dimensions
  n_samples = PyArray_DIM(signal_input , 0); // Getting the first dimension
  /* In cases that numpy array is not contiguous then stride values are not equal to 1
    The stride values given in bytes and division to the size of variables gives the 
    strides as blocks.
   */
  input_stride = PyArray_STRIDE(signal_input, 0)/sizeof(int8_t);
  output_stride = PyArray_STRIDE(signal_output, 0)/sizeof(int16_t);

  /* Access to the data part of the numpy array */
  input = PyArray_DATA(signal_input);
  output = PyArray_DATA(signal_output);
  
  /* Calling the core function */
  lilcom_decompress(n_samples, input, input_stride, output, output_stride);

  /* Making the resulting array */
  npy_intp * output_dimensions = malloc(sizeof(npy_intp)*2);
  output_dimensions[0] = n_samples;
  output_dimensions[1] = output_stride;
  PyArrayObject * output_array = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT8, (void*) output);

  // /* Returning numpy array */
  PyObject *returner = PyArray_Return(output_array);
  return returner;
}


/* Defining Functions in the Madule */
static PyMethodDef LilcomMethods[] = {
  { "compress", compress, METH_VARARGS | METH_KEYWORDS, "Compresses a recieved signal" },
  { "decompress", decompress, METH_VARARGS| METH_KEYWORDS, "Decompresses a recieved compress signal"  },
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


