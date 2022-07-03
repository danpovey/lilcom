#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#include "lilcom/compression.h"

#define LILCOM_FORMAT_VERSION 0

// Must not be changed.  Header is 'L' then
// LILCOM_FORMAT_VERSION.
#define LILCOM_HEADER_LEN 2

static constexpr const char *kCompressFloatDoc = R"doc(
Compresses the supplied data and returns compressed form as bytes object.

Args:
 input:
   A numpy.ndarray with dtype=np.float32 and number of axes in the range
   [1..15].  Caution: it will be replaced with the approximate version of
   itself that you'll get after decompressing the return value.
  meta:
    A list of integers containing some meta-information:
    [ tick_power, coeff1, coeff2, .. ] where tick_power (e.g. -8), which must
    be in the range [-20,20] (this decision was arbitrary), is the power to
    which we'll raise 2 to use or the `tick`
    (distance between 2 encoded elements, dictating the accuracy), and the
    coefficients correspond to estimated regression coefficients, one per axis,
    each multiplied by 256 and rounded to the nearest integer.  See
    documentation of `regression_coeffs` arg of CompressFloat(), in
    compression.h, for more details about the regression coefficients.

Returns:
  On success, returns the compressed data as a bytes object which can be
  decompressed.  On failure or if one of the args was not right, returns None.
  On memory allocation failure, raises MemoryError.
  (Note: this code could also die with an assertion, which would indicate a
  code error at the C++ level).
)doc";

static constexpr const char *kGetFloatMatrixShapeDoc = R"doc(
Takes a bytes object as returned from :func:`compress_float`, and returns a
tuple representing the shape of the array that was compressed, or
None on error.

Args:
  bytes_in:
    a bytes object returned by :func:`compress_float`.

Returns:
  A tuple containing the shape of the input array of :func:`compress_float`.
  Return None if the given object is not a bytes object returned by
  :func:`compress_float`.
)doc";

static constexpr const char *kDecompressFloatDoc = R"doc(
Takes a bytes object and an appropriately sized NumPy array of floats,
with shape as given by get_float_matrix_shape(), and decompresses the
data into the array.  Returns 0 on success, and a nonzero code or None
on failure.

Args:
  byts_in:
    A `bytes` object that was returned from :func:`compress_float`.

  array_out:
    Must be a NumPy array with dtype numpy.float32, and shape equal to the
    result of calling :func:`get_float_matrix_shape` on this same bytes object.

Return:
  Returns 0 on success, a nonzero code if there was a failure in the
  decompression. Raises ValueError if the args appeared to have the wrong type
  or could not be decompressed.
)doc";

static py::object CompressesFloatWrapper(py::array_t<float> input,
                                         const std::vector<int32_t> &meta) {
  int32_t num_axes = input.ndim();
  int32_t list_size = static_cast<int32_t>(meta.size());

  if (num_axes <= 0 || num_axes >= 16 || list_size != num_axes + 1)
    return py::none();

  int32_t tick_power = meta[0];

  int32_t dims[16], strides[16];
  int32_t regression_coeffs[16];

  for (int32_t i = 0; i != num_axes; ++i) {
    int32_t int_coeff = meta[i + 1];
    assert(int_coeff >= -256 && int_coeff <= 256);
    regression_coeffs[i] = int_coeff;
    dims[i] = input.shape(i);
    strides[i] = input.strides(i) / sizeof(float);
  }

  float *input_data = input.mutable_data();

  std::vector<char> ans = CompressFloat(tick_power, input_data, num_axes, dims,
                                        strides, regression_coeffs);

  if (ans.empty()) {
    // Something went wrong.  An error message may have been printed.
    return py::none();
  }
  py::bytes ret(NULL, LILCOM_HEADER_LEN + ans.size());

  char *ret_data = PyBytes_AsString(ret.ptr());
  ret_data[0] = 'L';
  ret_data[1] = LILCOM_FORMAT_VERSION;
  memcpy(ret_data + 2, &(ans[0]), ans.size());
  return ret;
}

static bool LilcomCheckBytesHeader(py::bytes bytes_in, char **bytes_array,
                                   Py_ssize_t *length) {
  if (PyBytes_AsStringAndSize(bytes_in.ptr(), bytes_array, length) != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "lilcom: Expected bytes object as 1st arg");
    throw py::error_already_set();
    return false;
  } else if (*length <= LILCOM_HEADER_LEN) {
    PyErr_SetString(PyExc_ValueError, "lilcom: Length of string was too short");
    throw py::error_already_set();
    return false;
  } else if (**bytes_array != 'L') {
    PyErr_SetString(PyExc_ValueError,
                    "lilcom: Lilcom-compressed data must begin with L");
    throw py::error_already_set();
    return false;
  } else if ((*bytes_array)[1] != LILCOM_FORMAT_VERSION) {
    PyErr_SetString(PyExc_ValueError,
                    "lilcom: Trying to decompress data from a future format "
                    "version (use newer code)");
    throw py::error_already_set();
    return false;
  }
  // remove the header from what we return.
  *bytes_array += LILCOM_HEADER_LEN;
  *length -= LILCOM_HEADER_LEN;
  return true;
}

static py::object GetFloatMatrixShape(py::bytes bytes_in) {
  char *bytes_array;
  Py_ssize_t length;

  if (!LilcomCheckBytesHeader(bytes_in, &bytes_array, &length)) {
    return py::none();
  }

  int32_t meta[17];
  if (GetCompressedDataShape(bytes_array, length, meta)) {
    int32_t num_axes = meta[0];
    assert(num_axes > 0 && num_axes <= 16); // was checked in
                                            // GetCompressedDataShape()
    py::tuple ans(num_axes);
    for (int32_t i = 0; i < num_axes; i++) {
      int32_t dim = meta[i + 1];
      assert(dim > 0); // was checked in GetCompressedDataSize()
      ans[i] = dim;
    }
    return ans;
  } else {
    return py::none();
  }
}

static py::object DecompressFloatWrapper(py::bytes bytes_in,
                                         py::array_t<float> array_out) {
  char *bytes_array;
  Py_ssize_t length;

  if (!LilcomCheckBytesHeader(bytes_in, &bytes_array, &length)) {
    return py::none();
  }

  int dims[16], strides[16];
  int num_axes = array_out.ndim();
  for (int i = 0; i != num_axes; ++i) {
    dims[i] = array_out.shape(i);
    strides[i] = array_out.strides(i) / sizeof(float);
  }

  int ans = DecompressFloat(bytes_array, length, array_out.mutable_data(),
                            num_axes, dims, strides);
  return py::int_(ans);
}

PYBIND11_MODULE(lilcom_extension, m) {
  m.doc() = "pybind11 binding of lilcom";

  m.def("compress_float", &CompressesFloatWrapper, py::arg("input"),
        py::arg("meta"), kCompressFloatDoc);

  m.def("get_float_matrix_shape", &GetFloatMatrixShape, py::arg("bytes_in"),
        kGetFloatMatrixShapeDoc);

  m.def("decompress_float", &DecompressFloatWrapper, py::arg("bytes_in"),
        py::arg("array_out"), kDecompressFloatDoc);
}
