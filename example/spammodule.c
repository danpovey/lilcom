#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Define a new exception
static PyObject *SpamError;

// The C function that is called when "spam.system(string)" is run in Python
static PyObject *spam_system(PyObject *self, PyObject *args) {
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  sts = system(command);
  if (sts != 0) {
    PyErr_SetString(SpamError, "System command failed");
    return NULL;
  }
  return PyLong_FromLong(sts);
}

// List spam_system() in a “method table”
static PyMethodDef SpamMethods[] = {
  {"system",  spam_system, METH_VARARGS,
   "Execute a shell command."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

// The method table must be referenced in the module definition structure
static struct PyModuleDef spammodule = {
  PyModuleDef_HEAD_INIT,
  "spam",   /* name of module */
  NULL, /* module documentation, may be NULL */
  -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
  SpamMethods
};

// The module’s initialization function
PyMODINIT_FUNC PyInit_spam(void) {
  PyObject *m;

  m = PyModule_Create(&spammodule);
  if (m == NULL)
    return NULL;

  SpamError = PyErr_NewException("spam.error", NULL, NULL);
  Py_INCREF(SpamError);
  PyModule_AddObject(m, "error", SpamError);
  return m;
}

// When embedding Python, the PyInit_spam() function is not called automatically
// unless there’s an entry in the PyImport_Inittab table
//int main(int argc, char *argv[]) {
//  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
//  if (program == NULL) {
//    fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
//    exit(1);
//  }
//
//  /* Add a built-in module, before Py_Initialize */
//  PyImport_AppendInittab("spam", PyInit_spam);
//
//  /* Pass argv[0] to the Python interpreter */
//  Py_SetProgramName(program);
//
//  /* Initialize the Python interpreter.  Required. */
//  Py_Initialize();
//
//  /* Optionally import the module; alternatively,
//     import can be deferred until the embedded script
//     imports it. */
//  PyImport_ImportModule("spam");
//    
//  PyMem_RawFree(program);
//  return 0;
//}
