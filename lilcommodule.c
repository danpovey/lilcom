#include <Python.h>
//#include "lilcom.h"

char * hello(char * what) {
  return "hello";
}

static PyObject * compress(PyObject * self, PyObject * args)
{
  printf("Compress called with argument %d\n", &args[0]);
  
  Py_RETURN_NONE;
}

static PyObject * decompress(PyObject * self, PyObject * args)
{
  printf("Decompress called with argument %d\n", &args[0]);
  Py_RETURN_NONE;
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
