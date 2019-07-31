#include <Python.h>
#include "numpy/arrayobject.h"
#include "lilcom.h"

static PyObject * compress(PyObject * self, PyObject * args)
{
  printf("DEBUG: Compress called \n");
  
  PyObject *signalObj;
  if (!PyArg_ParseTuple(args, "O", &signalObj)) {
      Py_RETURN_FALSE;
  }
  int n_samples = (int)PyArray_DIM(signalObj, 0);
  int stride = (int)PyArray_DIM(signalObj, 1);

  // There is a bug in the stride calculation
  stride = 1;

  printf("number of samples = %d\n", n_samples);
  printf("stride = %d\n", stride);

  int **singal_stream = (int **)(PyArray_DATA(signalObj));

  // Conversion to int16_t
  int16_t *input = malloc(sizeof(int16_t)* n_samples * stride);
  for (int i = 0 ; i < n_samples * stride; i++){ 
    printf("i: %d\t", i);
    printf("%d\t",singal_stream[i]);
    input[i] = singal_stream[i];
    printf("%d\n", input[i]);
  }

  int8_t * output;
  int output_stride = 1;
  int lpc_order = 1;

  // FUNCTION CALL HERE!
  //lilcom_compress(n_samples, input, stride, output, output_stride, lpc_order);

  for (int i = 0 ; i < n_samples * stride; i++){ 
    printf("i: %d\t", i);
    printf("%d\n", output[i]);
  }
  
  printf("length = %d\n", n_samples);
  printf("Data consists of %d\n", singal_stream[5]);
  
  Py_RETURN_TRUE;
}



static PyObject * decompress(PyObject * self, PyObject * args)
{
  printf("DEBUG: Decompress called\n");

  PyObject *signal_obj;
  if (!PyArg_ParseTuple(args, "O", &signal_obj)) {
      Py_RETURN_FALSE;
  }
  int n_samples = (int)PyArray_DIM(signal_obj, 0);
  int stride = (int)PyArray_DIM(signal_obj, 1);
  
 // PyObject * signal = PyArray_FROM_OTF(signalObj, NPY_INT, NPY_IN_ARRAY);
  int ** singal_stream = (int **)(PyArray_DATA(signal_obj));

  int converted_signal;

  printf("length = %d\n", n_samples);
  printf("Data consists of %d\n", singal_stream[120]);
}




// static PyObject * lilcom_wrapper(PyObject * self, PyObject * args)
// {
//   char * input;
//   char * result;
//   PyObject * ret;

//   // parse arguments
//   if (!PyArg_ParseTuple(args, "s", &input)) {
//     return NULL;
//   }

//   // run the actual function
//   result = hello(input);

//   // build the resulting string into a Python object.
//   ret = PyString_FromString(result);
//   free(result);

//   return ret;
// }


static PyMethodDef LilcomMethods[] = {
  { "compress", compress, METH_VARARGS, "Compresses a recieved signal" },
  { "decompress", decompress, METH_VARARGS, "Decompresses a recieved compress signal"  },
  { NULL, NULL, 0, NULL }
};


PyMODINIT_FUNC initlilcomlib(){
  Py_InitModule3("lilcomlib", LilcomMethods, "A compression decompression package");
}

// DL_EXPORT(void) initlilcom(void)
// {
//   Py_InitModule("lilcom", LilcomMethods);
// }
