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

    @param [in] input   The 16-bit input sequence data: a numpy array
                        just in dimension
  
    @param [out] output   The 8-bit compresed data:  a numpy
                        array passed to the function, which should
                        be the proper sub-array of the output to paste
                        the output array.
 */
static PyObject * compress16i_8i(PyObject * self, PyObject * args, PyObject * keywds)
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
  int conversion_exponent = 0; // Conversion Exponent defined in the core function (more information -> lilcom.h)

  /* Reading and information - extracting for input data 
    From the python function there are two numpy arrays and an intger (optional) LPC_order 
    passed to this madule. Following part will parse the set of variables and store them in corresponding
    objects.
  */
  static char *kwlist[] = {"X", "Y" , "lpc_order", "conversion_exponent", NULL}; //definition of keywords received in the function call from python
  // Parsing Arguments
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|ii", kwlist, &signal_input, &signal_output, &lpc_order, &conversion_exponent)) 
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


  /* Access to the data part of the numpy array
    PyArray_Data returns a pointer to the data stored in the memory.
   */
  input = PyArray_DATA(signal_input);
  output = PyArray_DATA(signal_output);
  
  /* Calling the core function */
  int function_state = lilcom_compress(n_samples, input, input_stride, output, output_stride, lpc_order,conversion_exponent) ;

  return Py_BuildValue("i",function_state);
}


static PyObject * compressf_8i(PyObject * self, PyObject * args, PyObject * keywds)
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
  float *input; // The one dimensional vectorized input array which will be given to the core function
  int lpc_order = 5; // LPC Order defined in the core function (more information -> lilcom.h)
  int conversion_exponent = 0; // Conversion Exponent defined in the core function (more information -> lilcom.h)


  /* Reading and information - extracting for input data 
    From the python function there are two numpy arrays and an intger (optional) LPC_order 
    passed to this madule. Following part will parse the set of variables and store them in corresponding
    objects.
  */
  static char *kwlist[] = {"X", "Y" , "lpc_order", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO|i", kwlist, &signal_input, &signal_output, &lpc_order)) 
    Py_RETURN_FALSE;
  

  printf("conversion madule = %d\n", conversion_exponent);
  // Initializing shape related variables
  n_dims = PyArray_NDIM(signal_input); // Getting the number of dimensions
  n_samples = PyArray_DIM(signal_input , 0); // Getting the first dimension
  /* In cases that numpy array is not contiguous then stride values are not equal to 1
    The stride values given in bytes and division to the size of variables gives the 
    strides as blocks.
   */
  input_stride = PyArray_STRIDE(signal_input, 0)/sizeof(float);
  output_stride = PyArray_STRIDE(signal_output, 0)/sizeof(int8_t);


  /* Access to the data part of the numpy array
    PyArray_Data returns a pointer to the data stored in the memory.
   */
  input = PyArray_DATA(signal_input);
  output = PyArray_DATA(signal_output);

  /* Calling the core function */
  int function_state = lilcom_compress_float(n_samples, input, input_stride, output, output_stride, lpc_order, NULL) ;

  return Py_BuildValue("i",function_state);
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
  int16_t *output; // The one dimensional vectorized output array which will be modified by the core function
  int8_t *input; // The one dimensional vectorized input array which will be given to the core function
  int conversion_exponent;
  /* Reading and information - extracting for input data 
    From the python function there are two numpy arrays and an intger (optional) LPC_order 
    passed to this madule. Following part will parse the set of variables and store them in corresponding
    objects.
  */
  static char *kwlist[] = {"X", "Y", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist, &signal_input, &signal_output)) 
    return Py_BuildValue("i",1);
  
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
  int function_state = lilcom_decompress(n_samples, input, input_stride, output, output_stride, &conversion_exponent);

  return Py_BuildValue("i",conversion_exponent);
}


/* Defining Functions in the Madule */
static PyMethodDef LilcomMethods[] = {
  { "compress16i_8i", compress16i_8i, METH_VARARGS | METH_KEYWORDS, "Lossily compresses samples of int16 sequence data (e.g. audio data) int8_t."},
  { "compressf_8i", compressf_8i, METH_VARARGS | METH_KEYWORDS, "Lossily compresses samples of float sequence data (e.g. audio data) int8_t."},
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


