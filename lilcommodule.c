#include <Python.h>
 
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
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
    input[i] = (int16_t)singal_stream[i];
  }

  
  int output_stride = 1;
  int lpc_order = 1;
  int8_t * output = malloc(sizeof(int8_t)*output_stride*n_samples);

  // FUNCTION CALL HERE!
  printf("function call!\n");
  lilcom_compress(n_samples, input, stride, output, output_stride, lpc_order);
  printf("after function call!\n");
  

  // for(int ii = 0 ; ii < n_samples ; ii++){
  //   printf("On sample %d, a = %d and b = %d\n", ii , input[ii], output[ii]);
  // }

  
  printf("before dimension!\n");
  int output_dimensions[n_dims];
  output_dimensions[0] = n_samples;
  printf("after dimension!\n");

  printf("See the error is malloc array!\n");
  int * output_temp = malloc(sizeof(int) * n_samples * output_stride);
  for (int i = 0; i < n_samples*output_stride; i++){
    output_temp[i] = (int)output[i];
  }

 

  if (n_dims > 1){
    printf("stride change\n");
    output_dimensions[1] = output_stride;
  }

  import_array();
  printf("See the error is after array!\n");
  PyArrayObject * output_numpy_array = (PyArrayObject *)PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT, (void*) output_temp);
  printf("See the error is after cast!\n");
  //PyArrayObject * return_val = (PyArrayObject *) output_numpy_array; 
  printf("See the error is after return!\nArray size is %d\n", PyArray_DIM(output_numpy_array, 0));

  return PyArray_Return(output_numpy_array);
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
    input[i] = (int8_t)singal_stream[i];
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
    return PyModule_Create(&lilcom);
    import_array();
}

// PyMODINIT_FUNC initlilcom(){
//   Py_InitModule3("lilcom", LilcomMethods, "A compression decompression package");
// }

