#include <Python.h>
#include "lilcom.h"

char * hello(char * what) {
  return "hello";
}

static PyObject * lilcom_wrapper(PyObject * self, PyObject * args)
{
  char * input;
  char * result;
  PyObject * ret;

  // parse arguments
  if (!PyArg_ParseTuple(args, "s", &input)) {
    return NULL;
  }

  // run the actual function
  result = hello(input);

  // build the resulting string into a Python object.
  ret = PyString_FromString(result);
  free(result);

  return ret;
}


static PyMethodDef LilcomMethods[] = {
  { "lilcom", lilcom_wrapper, METH_VARARGS, "Say lilcom" },
  { NULL, NULL, 0, NULL }
};


DL_EXPORT(void) initlilcom(void)
{
  Py_InitModule("lilcom", LilcomMethods);
}
