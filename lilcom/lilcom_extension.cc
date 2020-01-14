#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "unistd.h"  /* For malloc and free */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"

/* The core library */
#include "compression.h"


extern "C" {
  /* Functions declared inside here have plain "C" linkage,
     which is necessary for Python to find them.
   */



  static CompressorConfig * capsule_to_CompressorConfig(PyObject * obj) {
    return (CompressorConfig*) PyCapsule_GetPointer(obj, "CompressorConfig");
  }

/* Destructor function for class CompressorConfig */
  static void del_CompressorConfig(PyObject * obj) {
    delete (CompressorConfig*) PyCapsule_GetPointer(obj, "CompressorConfig");
  }

  /* creates a capsule from a CompressorConfig object; takes ownership of it. */
  static PyObject * CompressorConfig_to_capsule(CompressorConfig *c) {
    return PyCapsule_New(c, "CompressorConfig", del_CompressorConfig);
  }


/*
  Create a configuration object for lilcom compression.
  The following will document this function as if it were a native
  Python function.

  def create_compressor_config(sampling_rate, num_channels,
                              loss_level, compression_level):
      """ Creates and returns an opaque configuration object that
          can be used for lilcom compression, or None if there
          was an error.

          Args:
           sampling_rate  Sampling rate of the signal, in Hz.  This does not
                     affect anything, it is passed through the compression
                     and back to the user.
           num_channels  Number of channels in the signal, e.g. 1 for
                     mono, 2 for stereo (may be any number greater than zero).
           loss_level     Dictates how lossy the compression will be.
                       0 == lossless, 5 == most lossy.
           compression_level    Dictates the speed / file-size tradeoff.
                           0 == fastest, but biggest file; 5 == slowest,
                           smallest file.
         Returns:
           An opaque configuration object on success, None if one of the
           args was incorrect.
      """


 */
static PyObject *create_compressor_config(PyObject *self, PyObject *args, PyObject *keywds) {

  int sampling_rate, num_channels,
      loss_level, compression_level;


  /* Reading and information - extracting for input data
     From the python function there are two numpy arrays and an intger (optional) LPC_order
     passed to this madule. Following part will parse the set of variables and store them in corresponding
     objects.
  */
  static char *kwlist[] = {"sampling_rate", "num_channels", "loss_level",
                           "compression_level", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "iiii", kwlist,
                                   &sampling_rate, &num_channels,
                                   &loss_level, &compression_level))
    Py_RETURN_NONE;

  CompressorConfig *c = new CompressorConfig(sampling_rate, num_channels,
                                             loss_level, compression_level);
  if (!c->IsValid()) {
    delete c;
    Py_RETURN_NONE;
  }
  return CompressorConfig_to_capsule(c);

}

static PyObject *compressor_config_to_str(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
  CompressorConfig *c;
  if (nargs != 1 ||
      (c = capsule_to_CompressorConfig(args[0])) == NULL)
    Py_RETURN_NONE;
  std::string str(*c);
  return PyUnicode_FromString(str.c_str());
}



/**
   The following will document this function as if it were a native
   Python function.

    def compress_int16(input, config):
      """

      Args:
       input:  A numpy.ndarray with dtype=int16 and 2 axes, which must
            represent respectively the num-channels and the sample index.
            There is no limitation on the strides.
       config:  A configuration object generated with
             create_compressor_config()

       Return:
            On success, returns the compressed data as a bytes object.  On
            failure if one of the args was not right, returns None.
            On memory allocation failure, raises MemoryError.
            (Note: this
            code could also die with an assertion, which would indicate a
            code error).  """
 */
static PyObject *compress_int16(PyObject *self, PyObject *args, PyObject *keywds) {
  PyArrayObject *input; /* The input signal, passed as a numpy array of int16's. */
  PyObject *config; /* The configuration object */

  static char *kwlist[] = {"input", "config", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO", kwlist,
                                   (PyObject**)&input, &config))
    Py_RETURN_NONE;

  CompressorConfig *cc = capsule_to_CompressorConfig(config);
  const int16_t *input_data = (const int16_t*)PyArray_DATA(input);

  if (cc == NULL || !cc->IsValid())
    return PyLong_FromUnsignedLong(2);

  int num_axes = PyArray_NDIM(input);
  if (num_axes != 2)
    return PyLong_FromUnsignedLong(1);
  int num_channels = PyArray_DIM(input, 0),
      channel_stride = PyArray_STRIDE(input, 0) / sizeof(int16_t),
      sample_stride = PyArray_STRIDE(input, 1) / sizeof(int16_t),
      num_samples = PyArray_DIM(input, 1);
  if (num_channels != cc->num_channels ||
      num_samples == 0)
    Py_RETURN_NONE;

  try {
    CompressedFile cf(*cc, num_samples, input_data, sample_stride, channel_stride);

    char *data;
    size_t length;
    data = cf.Write(&length);
    assert(data != NULL);

    /* note: this does an unnecessary copy but I don't see an easy way to do
       this without that copy. */
    PyObject *ans = PyBytes_FromStringAndSize(data, length);
    if (ans == NULL) {
      /* Memory failure when copying the output to bytes object */
      return PyLong_FromUnsignedLong(4);
    } else {
      delete [] data;
    }
    return ans;
  } catch (std::bad_alloc) {
    PyErr_SetString(PyExc_MemoryError,
                    "Failure to allocate memory in lilcom compression");
    return NULL;
  }
}

  /* Destructor function for class CompressedFile */
  static void del_CompressedFile(PyObject * obj) {
    delete (CompressedFile*) PyCapsule_GetPointer(obj, "CompressedFile");
    /* Allow the `bytes` object to be freed if no other references to it
     * survived.  */
    Py_DECREF((PyObject*) PyCapsule_GetContext(obj));
  }

  /* creates a capsule from a CompressorConfig object; takes ownership of it.
     Also maintains a reference to `bytes` to avoid its memory being freed while
     the object `c` exists, since it contains a pointer to that data.
   */
  static PyObject * CompressedFile_to_capsule(CompressedFile *c, PyObject *bytes) {
    PyObject *ans = PyCapsule_New(c, "CompressedFile", del_CompressedFile);
    Py_INCREF(bytes);
    PyCapsule_SetContext(ans, (void*)bytes);
    return ans;
  }

  static CompressedFile * capsule_to_CompressedFile(PyObject * obj) {
    return (CompressedFile*) PyCapsule_GetPointer(obj, "CompressedFile");
  }



  /**
    The following will document this function as if it were a native
    Python function.

       def init_decompression_int16(bytes):
         """
         Initializes the decompression of a bytes array containing lilcom-compressed
         data

         Args:
            `bytes` must be a bytes object
         Return:
            If `bytes` had the expected format and was not truncated,
            this function will return a tuple:
               (decompressor, num_channels, num_samples)
            where `decompressor` is an opaque object that should later be passed
            to `decompress_part_int16`.

            Otherwise, this function will return None.
         """
   */
  static PyObject *init_decompression_int16(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    char *bytes_array;
    Py_ssize_t length;
    if (nargs != 1)
      Py_RETURN_NONE;
    PyObject *bytes = args[0];

    if (PyBytes_AsStringAndSize(bytes, &bytes_array, &length) != 0)
      return 0;  /* we are raising ValueError. */

    try {
      CompressedFile *f = new CompressedFile();
      int ret = f->InitForReading(bytes_array, bytes_array + length);
      if (ret != 0) {
        /* Right now we don't try to get too specific about passing error
           information back to the python layer. */
        delete f;
        Py_RETURN_NONE;
      } else {
        PyObject *f_capsule = CompressedFile_to_capsule(f, bytes);
        return Py_BuildValue("Oll", f_capsule, (long)f->NumChannels(), (long)f->NumSamples());
      }
    } catch (std::bad_alloc) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failure to allocate memory in lilcom compression");
      return NULL;
    }
  }


  /**
    The following will document this function as if it were a native Python
    function.

       def decompress_int16(compressed_file_object, array_out)
         """
         Decompress all of an int16 array that was previously compressed
         with `compress_int16`.

         Args:
            compressed_file_object: an opaque object that will have been returned
               by `init_decompression_int16()`
            array_out: must be a NumPy array with dtype numpy.int16, and
               dimensions (num_channels, num_samples) these values
               correspond to the values returned by init_decompression_int16().
               There are no constraints on the strides of `array_out`.


         Return:
           Returns true on success, false on failure in the decompression,
           or None if the args appeared to have the wrong type.
         """

   */
  static PyObject *decompress_int16(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2)
      Py_RETURN_NONE;
    PyObject *compressed_file = args[0];
    PyArrayObject *numpy_array = (PyArrayObject*)args[1];

    CompressedFile *cf = capsule_to_CompressedFile(compressed_file);
    int num_axes = PyArray_NDIM(numpy_array);
    if (num_axes != 2 || cf == NULL ||
        (size_t)PyArray_DIM(numpy_array, 0) != (size_t)cf->NumChannels() ||
        (size_t)PyArray_DIM(numpy_array, 1) != (size_t)cf->NumSamples())
      Py_RETURN_NONE;

    int channel_stride = PyArray_STRIDE(numpy_array, 0) / sizeof(int16_t),
        sample_stride = PyArray_STRIDE(numpy_array, 1) / sizeof(int16_t);

    try {
      if (cf->ReadAllData(sample_stride, channel_stride,
                          (int16_t*)PyArray_DATA(numpy_array)))
        Py_RETURN_TRUE;
      else
        Py_RETURN_FALSE;
    } catch (std::bad_alloc) {
      PyErr_SetString(PyExc_MemoryError,
                      "Failure to allocate memory in lilcom decompression");
      return NULL;
    }
  }



  static PyMethodDef LilcomExtensionMethods[] = {
    {"create_compressor_config", (PyCFunction) create_compressor_config, METH_VARARGS | METH_KEYWORDS,
     "Creates and returns an opaque configuration object that can be used for lilcom compression."},
    {"compressor_config_to_str", (PyCFunction) compressor_config_to_str, METH_FASTCALL,
     "Returns a string that describes the compressor-config object passed in, or None "
     "if the object had the wrong type."},
    {"compress_int16", (PyCFunction) compress_int16, METH_VARARGS | METH_KEYWORDS,
     "Compresses the supplied data and returns compressed form as bytes object."},
    {"init_decompression_int16", (PyCFunction) init_decompression_int16, METH_FASTCALL,
     "Takes a bytes object, returns a tuple (c, num_channels, num_samples) where c is an "
     "compressed-file object to be passed to decompress_int16()."},
    {"decompress_int16", (PyCFunction) decompress_int16, METH_FASTCALL,
     "Takes a compressed-file object (from init_decompression_int16) and an "
     "appropriately sized NumPy array, and returns bytes object, returns a tuple (o, num_channels, num_samples) where o is an "
     "opaque object to be passed to decompress_int16()."},
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
