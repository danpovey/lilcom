#include <Python.h>
#include "numpy/arrayobject.h"
#include "./lilcom.h"



/**
   Lossily compresses a given numpy array of 16-bit integer sequence data.
   It will first convert the numpy array data to a C `int16_t` array and then
   passes the array to the core lilcom library, finally it returns the result
   in form of a numpy array.
 */
static PyObject * compress(PyObject * self, PyObject * args)
{

  PyObject *signalObj; // The input signal, passed as a numpy array.
  if (!PyArg_ParseTuple(args, "O", &signalObj)) { //Parsing the input of the python code
      Py_RETURN_FALSE;
  }
  int n_dims = PyArray_NDIM(signalObj); // Number of dimensions in the numpy array
  int stride; // The number of integers between to consecutive samples.

  if (n_dims == 1){
    stride = 1;
  } // One dimensional array does't have any problem in stride.

  // Finding the shape of array
  int n_samples = (int)PyArray_DIM(signalObj, 0);
  if (n_dims != 1)
    stride = (int)PyArray_DIM(signalObj, 1);

  int **singal_stream = (int **)(PyArray_DATA(signalObj));

  // Conversion to int16_t
  int16_t *input = malloc(sizeof(int16_t) * n_samples * stride); // alocating a linear array of considered type.
  for (int i = 0 ; i < n_samples * stride; i++){
    input[i] = singal_stream[i];
  }

  int8_t * output;
  int output_stride = 1;
  int lpc_order = 1;

  // FUNCTION CALL HERE!
  printf("function call!\n");
  lilcom_compress(n_samples, input, stride, output, output_stride, lpc_order);


  Py_RETURN_TRUE;
}



static PyObject * decompress(PyObject * self, PyObject * args)
{
  PyObject *signalObj; // The input signal, passed as a numpy array.
  if (!PyArg_ParseTuple(args, "O", &signalObj)) { //Parsing the input of the python code
      Py_RETURN_FALSE;
  }
  int n_dims = PyArray_NDIM(signalObj); // Number of dimensions in the numpy array
  int stride; // The number of integers between to consecutive samples.

  if (n_dims == 1){
    stride = 1;
  } // One dimensional array does't have any problem in stride.

  // Finding the shape of array
  int n_samples = (int)PyArray_DIM(signalObj, 0);
  if (n_dims != 1)
    stride = (int)PyArray_DIM(signalObj, 1);

  int **singal_stream = (int **)(PyArray_DATA(signalObj));

  // Conversion to int16_t
  int8_t *input = malloc(sizeof(int8_t) * n_samples * stride); // Allocating a linear array of considered type.
  for (int i = 0 ; i < n_samples * stride; i++){
    input[i] = singal_stream[i];
  }

  int16_t * output;
  int output_stride = 1;
  int lpc_order = 1;

  // FUNCTION CALL HERE!
  //lilcom_compress(n_samples, input, stride, output, output_stride, lpc_order);


  Py_RETURN_TRUE;
}



static PyMethodDef LilcomMethods[] = {
  { "compress", compress, METH_VARARGS, "Compresses a recieved signal" },
  { "decompress", decompress, METH_VARARGS, "Decompresses a recieved compress signal"  },
  { NULL, NULL, 0, NULL }
};


/*
PyMODINIT_FUNC initlilcom(){
  Py_InitModule3("lilcom", LilcomMethods, "A compression decompression package");
}
*/

static struct PyModuleDef lilcom =
{
  PyModuleDef_HEAD_INIT,
  "lilcom", /* name of module */
  "usage: foo\n", /* module documentation, may be NULL */
  -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  LilcomMethods
};

PyMODINIT_FUNC PyInit_lilcom(void)
{
  return PyModule_Create(&lilcom);
}
