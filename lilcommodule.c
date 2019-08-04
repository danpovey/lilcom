// Needed definitions and includes for python C interface
#define PY_SSIZE_T_CLEAN
#include <Python.h>
 
#include "numpy/arrayobject.h"

// The core library 
#include "./lilcom.h"

int integral_check (int array_type){
  if (array_type == NPY_INT8) return 1;
  if (array_type == NPY_INT16) return 1;
  if (array_type == NPY_INT32) return 1;
  if (array_type == NPY_INT64) return 1;
  if (array_type == NPY_SHORT) return 1;
  if (array_type == NPY_INT) return 1;
  if (array_type == NPY_LONG) return 1;
  if (array_type == NPY_LONGLONG) return 1;

  if (array_type == NPY_UINT8) return 1;
  if (array_type == NPY_UINT16) return 1;
  if (array_type == NPY_UINT32) return 1;
  if (array_type == NPY_UINT64) return 1;
  if (array_type == NPY_USHORT) return 1;
  if (array_type == NPY_UINT) return 1;
  if (array_type == NPY_ULONG) return 1;
  if (array_type == NPY_ULONGLONG) return 1;
  return 0;
}


/**
   Lossily compresses a given numpy array of 16-bit integer sequence data.
   It will first convert the numpy array data to a C `int16_t` array and then 
   passes the array to the core lilcom core library, finally it returns the result
   in form of a numpy array.
 */
static PyObject * compress(PyObject * self, PyObject * args, PyObject * keywds)
{
  /* Defining all variables */
  PyObject *signal_object; // The input signal, passed as a numpy array.
  int n_dims = 0; // Number of dimensions of the given numpy array
  int input_stride; // The number of integers between to consecutive samples
  int output_stride; // The number of integers between to consecutive samples in output
  int n_samples = 0; // number of samples given in the numpy array
  // INPUT IS DEFINED INSIDE IF CONDITIONS
  int8_t *output; // The one dimensional vectorized output array which will be modified by the core function
  int lpc_order = 5; // LPC Order defined in the core function (more information -> lilcom.h)
  int integral = 1; // Checks whether the array contains integer or float

  /* Reading and information - extracting for input data */
  static char *kwlist[] = {"X" , "lpc_order", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwlist, &signal_object, &lpc_order)) 
    Py_RETURN_FALSE;
  
  // Initializing shape related variables
  n_dims = PyArray_NDIM(signal_object); // Getting the number of dimensions
  n_samples = PyArray_DIM(signal_object , 0); // Getting the first dimension
  if (n_dims > 1) input_stride = PyArray_DIM(signal_object , 0); // Conditioning on second dimension
  else input_stride = 1;
  integral = integral_check(((PyArrayObject*)signal_object)->descr->type_num); // Checking whether the array has float


  /* Debug: Comment or Uncomment when on debug */
  // printf("The array is received and the lpc_order = %d\n", lpc_order);
  // printf("Array is %d dimensional\n", n_dims);
  // printf("It has %d samples and stride is %d\n", n_samples, input_stride);
  // printf("PyArrayType == int is %d\n", integral );


  output_stride = input_stride; ////// CHANGE IT IF NEEDED
  
  /* Ensuring that the array is contiguous */
  PyArrayObject *signal_object_contiguous = PyArray_GETCONTIGUOUS(signal_object); 


  void *signal_object_data = PyArray_DATA(signal_object_contiguous);

  if(integral == 1) { // Integer
    int16_t *input; // The one dimensional vectorized input array which will be given to the core function
    /* Allocating the space for input and output */
    input = malloc(sizeof(int16_t) * n_samples * input_stride);
    output = malloc(sizeof(int8_t) * n_samples * output_stride);
    
    

    /* Conversion to int16_t array */
    switch (((PyArrayObject*)signal_object)->descr->type_num){
      case NPY_INT8:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((int8_t *)signal_object_data)[i]); 
        } break;
      case NPY_INT16:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((int16_t *)signal_object_data)[i]); 
        } break;
      case NPY_INT32:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((int32_t *)signal_object_data)[i]); 
        } break;
      case NPY_INT64:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((int64_t *)signal_object_data)[i]); 
        } break;
      case NPY_UINT8:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((uint8_t *)signal_object_data)[i]); 
        } break;
      case NPY_UINT16:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((uint16_t *)signal_object_data)[i]); 
        } break;
      case NPY_UINT32:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((uint32_t *)signal_object_data)[i]);
        } break;
      case NPY_UINT64:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (int16_t)(((uint64_t *)signal_object_data)[i]); 
        } break;
    }

    /* Calling the core function */
    lilcom_compress(n_samples, input, input_stride, output, output_stride, lpc_order);

    /* Debug: Comment or Uncomment when on debug */
    // for (int i = 0; i < n_samples * input_stride ; i++){
    //   printf("for index %d a = %d and b = %d\n", i , input[i], output[i]);
    // }

    /* Making the resulting array */
    npy_intp * output_dimensions = malloc(sizeof(npy_intp)*2);
    output_dimensions[0] = n_samples;
    output_dimensions[1] = output_stride;
    PyArrayObject * output_array = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT8, (void*) output);

    /* Overcoming memory leak problem */
    free(input);
    /* Returning numpy array */
    PyObject *returner = PyArray_Return(output_array);
    return returner;
  } 
  
  else { // Float
    float *input; // The one dimensional vectorized input array which will be given to the core function
    /* Allocating the space for input and output */
    input = malloc(sizeof(float) * n_samples * input_stride);
    output = malloc(sizeof(int8_t) * n_samples * output_stride);
    

    /* Conversion to int16_t array */
    switch (((PyArrayObject*)signal_object)->descr->type_num){
      case NPY_FLOAT:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (float)(((float *)signal_object_data)[i]); 
        } break;
      case NPY_DOUBLE:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (float)(((double *)signal_object_data)[i]); 
        } break;
      case NPY_LONGDOUBLE:
        for (int i = 0 ; i < n_samples * input_stride ; i++){
          input[i] = (float)(((long double *)signal_object_data)[i]); 
        } break;
    }

    /* Calling the core function */
    //lilcom_compress_float(n_samples, input, input_stride, output, output_stride);

    /* Debug: Comment or Uncomment when on debug */
    // for (int i = 0; i < n_samples * input_stride ; i++){
    //   printf("for index %d a = %f and b = %d\n", i , input[i], output[i]);
    // }

    /* Making the resulting array */
    npy_intp * output_dimensions = malloc(sizeof(npy_intp)*2);
    output_dimensions[0] = n_samples;
    output_dimensions[1] = output_stride;
    PyArrayObject * output_array = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT8, (void*) output);

    /* Overcoming memory leak problem */
    free(input);
    /* Returning numpy array */
    PyObject *returner = PyArray_Return(output_array);
    return returner;
  }
  return NULL;
}



static PyObject * decompress(PyObject * self, PyObject * args,  PyObject * keywds)
{
  /* Defining all variables */
  PyObject *signal_object; // The input signal, passed as a numpy array.
  int n_dims = 0; // Number of dimensions of the given numpy array
  int input_stride; // The number of integers between to consecutive samples
  int output_stride; // The number of integers between to consecutive samples in output
  int n_samples = 0; // number of samples given in the numpy array
  // INPUT IS DEFINED INSIDE IF CONDITIONS
  int16_t *output; // The one dimensional vectorized output array which will be modified by the core function

  /* Reading and information - extracting for input data */
  static char *kwlist[] = {"X" , NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O", kwlist, &signal_object)) 
    Py_RETURN_FALSE;
  
  // Initializing shape related variables
  n_dims = PyArray_NDIM(signal_object); // Getting the number of dimensions
  n_samples = PyArray_DIM(signal_object , 0); // Getting the first dimension
  if (n_dims > 1) input_stride = PyArray_DIM(signal_object , 0); // Conditioning on second dimension
  else input_stride = 1;


  /* Debug: Comment or Uncomment when on debug */
  // printf("Array is %d dimensional\n", n_dims);
  // printf("It has %d samples and stride is %d\n", n_samples, input_stride);


  output_stride = input_stride; ////// CHANGE IT IF NEEDED

  /* Ensuring that the array is contiguous */
  PyArrayObject *signal_object_contiguous = PyArray_GETCONTIGUOUS(signal_object);

  void *signal_object_data = PyArray_DATA(signal_object_contiguous);


  int8_t *input; // The one dimensional vectorized input array which will be given to the core function
  /* Allocating the space for input and output */
  input = malloc(sizeof(int8_t) * n_samples * input_stride);
  output = malloc(sizeof(int16_t) * n_samples * output_stride);
  

  /* Conversion to int16_t array */
  switch (((PyArrayObject*)signal_object)->descr->type_num){
    case NPY_INT8:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((int8_t *)signal_object_data)[i]); 
      } break;
    case NPY_INT16:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((int16_t *)signal_object_data)[i]); 
      } break;
    case NPY_INT32:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((int32_t *)signal_object_data)[i]); 
      } break;
    case NPY_INT64:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((int64_t *)signal_object_data)[i]); 
      } break;
    case NPY_UINT8:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((uint8_t *)signal_object_data)[i]); 
      } break;
    case NPY_UINT16:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((uint16_t *)signal_object_data)[i]); 
      } break;
    case NPY_UINT32:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((uint32_t *)signal_object_data)[i]);
      } break;
    case NPY_UINT64:
      for (int i = 0 ; i < n_samples * input_stride ; i++){
        input[i] = (int8_t)(((uint64_t *)signal_object_data)[i]); 
      } break;
  }

  /* Calling the core function */
  lilcom_decompress(n_samples, input, input_stride, output, output_stride);

  /* Debug: Comment or Uncomment when on debug */
  // for (int i = 0; i < n_samples * input_stride ; i++){
  //   printf("for index %d a = %d and b = %d\n", i , input[i], output[i]);
  // }

  /* Making the resulting array */
  npy_intp * output_dimensions = malloc(sizeof(npy_intp)*2);
  output_dimensions[0] = n_samples;
  output_dimensions[1] = output_stride;
  PyArrayObject * output_array = (PyArrayObject *) PyArray_SimpleNewFromData(n_dims, output_dimensions, NPY_INT16, (void*) output);

  /* Overcoming memory leak problem */
  free(input);
  /* Returning numpy array */
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


