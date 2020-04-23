#include "compression.h"
#include <iostream>
#include <cassert>
#include <cmath> 
#include <limits> 



/*
  Internal recursively called function that writes codes to `is` to compress
  this array.
      @param [in] tick  Distance between compressed values, e.g. 2^-8
      @param [in] inv_tick Inverse of `tick`
      @param [in] data  Data to compress (start of the original array)
      @param [in] num_axes  Number of axes in `data`.  Must be in the range [1..16]
      @param [in] dims  Dimensions of `data`, indexed by axis
      @param [in] strides   Strides of elements of `data`, in float elements
                       (not bytes), indexed by axis. 
      @param [in] regression_coeffs  Externally estimated regression coefficients,
                        one per axis.  See docs in compression.h for how this works
      @param [in,out] is   The codes will be written to this stream, one per data
                           element.  Meta-info is expected to have already been
                           written to here.
      @param [in] axis  The axis to iterate on; top-level call is with 0,
                        this will have values 0 <= axis < num_axes.
      @param [in] indexes  Array of indexes we're processing on axes *prior* to the
                        current axis, i.e. for axes i with 0 <= i < axis.  Must
                        have space equal to at least num_axes-1.
                        Will be set in the recursion.
 */
void CompressFloatInternal(float tick,
                           float inv_tick,
                           float *data, 
                           int num_axes,
                           const int *dims, 
                           const int *strides,
                           const float *regression_coeffs,
                           IntStream *is,
                           int axis,
                           int *indexes) {
  if (axis + 1 < num_axes) {
    for (int i = 0; i < dims[axis]; i++) {
      indexes[axis] = i;
      // Recurse
      CompressFloatInternal(tick, inv_tick, data, num_axes, dims, strides,
			    regression_coeffs, is, axis + 1, indexes);
    }
    return;
  }
  assert(axis == num_axes - 1);

  /* local_strides and local_coeffs will be the strides and
     corresponding regression coefficients for axes prior to `axis`
     whose corresponding indexes are not zero and whose regression
     coefficients are not 1. */

  int local_strides[16];
  float local_coeffs[16];
  int local_prev_axes = 0;
  float *cur_data = data;
  for (int i = 0; i < axis; i++) {
    /* make `local_data` point to the start of what we're processing here. */
    cur_data += indexes[i] * strides[i];
    if (regression_coeffs[i] != 0.0 && indexes[i] != 0) {
      local_strides[local_prev_axes] = strides[i];
      local_coeffs[local_prev_axes] = regression_coeffs[i];
      local_prev_axes++;
    }
  }

  /* The base-case, where there is 1 dimension, is a bit more optimized. */
  
  int dim = dims[axis],
    stride = strides[axis];
  float coeff = regression_coeffs[axis];

  float prev_prediction = 0.0;
  float *end = cur_data + (dim * stride);
  for (; cur_data < end; cur_data += stride) {
    float predicted = prev_prediction; /* will be prev element times coeff */
    for (int i = 0; i < local_prev_axes; i++) {
      /* add prediction from lower-numbered axes to this prediction. */
      predicted += cur_data[-(local_strides[i])] * local_coeffs[i];
    }
    float offset = *cur_data - predicted;
    int32_t code = round(offset * inv_tick);

    if (std::abs(offset - (code * tick)) > tick) {
      // Handle out-of-range data that cannot be represented; pin to
      // edges of range.  NOTE: this could be removed for speed,
      // at the expense of handling these kinds of situations less well.
      if (offset * inv_tick < std::numeric_limits<int32_t>::min()) {
        code = std::numeric_limits<int32_t>::min();
      } else if (offset * inv_tick > std::numeric_limits<int32_t>::max()) {
        code = std::numeric_limits<int32_t>::max();
      }
      // else do nothing; the difference could just be roundoff
      // error, which we can ignore.
    }
    is->Write(code);
    float compressed_data = predicted + (code * tick);
    *cur_data = compressed_data;
    prev_prediction = compressed_data * coeff;
  }
}


std::vector<char> CompressFloat(int tick_power,  /* e.g. -8 meaning tick=1.0/256.0 */
                                float *data, 
                                int num_axes, 
                                const int *dims, 
                                const int *strides,
                                const int *regression_coeffs) {
  IntStream is;
  float regression_coeffs_float[16];
  int indexes[16];

  if (num_axes <= 0 || num_axes > 16) {
    std::cerr << "lilcom: compression error: num-axes out of range "
	      << num_axes << std::endl;
    // Something is wrong here.  This is for memory safety.
    return std::vector<char>();
  }
  if (strides[num_axes - 1] != 1) {
    std::cerr << "lilcom: compression error: last stride should be 1, got "
	      << strides[num_axes - 1] << std::endl;
    return std::vector<char>();
  }
  if (tick_power < -20 || tick_power > 20) {
    std::cerr << "lilcom: tick_power out of range: " << tick_power
	      << std::endl;
    return std::vector<char>();
  }
  is.Write(num_axes);
  is.Write(tick_power);
  for (int i = 0; i < num_axes; i++) {
    regression_coeffs_float[i] = regression_coeffs[i] * (1.0 / 256.0);
    is.Write(dims[i]);
    is.Write(regression_coeffs[i]);
  }
  float tick = pow(2.0, tick_power), 
    inv_tick = pow(2.0, -tick_power);
  while (num_axes > 1 && dims[num_axes - 1] == 1)
    num_axes--;  /* will increase speed without affecting the output, in the
		    case where the last axis is useless. */
  CompressFloatInternal(tick, inv_tick, data, num_axes, dims, strides,
                        regression_coeffs_float, &is, 0, indexes);
  return is.Code();
}




bool GetCompressedDataShape(const char *data,
                            int num_bytes,
                            int *meta) {
  ReverseIntStream ris(data, data + num_bytes);
  int32_t num_axes = -100, tick_power = -100;
  if (!ris.Read(&num_axes) ||
      num_axes < 1 || num_axes > 16) {
    std::cerr << "lilcom: num_axes=" << num_axes
              << " is out of range or could not be read" << std::endl;
    return false;
  }
  if (!ris.Read(&tick_power) ||
      tick_power < -20 || tick_power > 20) {
    std::cerr << "lilcom: tick_power=" << tick_power
              << " is out of range or could not be read" << std::endl;
    return false;
  }

  meta[0] = num_axes;
  for (int i = 0; i < num_axes; i++) {
    int32_t dim = -100, _coeff;
    if (!ris.Read(&dim) || !ris.Read(&_coeff) || dim < 1) {
      std::cerr << "lilcom: dim=" << dim << " for axis="
                << i << " could not be read or is out of range.";
      return false;
    }
    meta[i + 1] = dim;
  }
  return true;
}




/*
  Internal recursively called function that reads codes from `ris` to 
  decompress this array.
      @param [in] ris   The codes will be read from this stream, one per data
                        element.  The meta-info will already have been
                        read from here.
      @param [in] tick  Distance between compressed values, e.g. 2^-8
      @param [in] data  Array to write data to
      @param [in] num_axes  Number of axes in `data`.  Must be in the range [1..16]
      @param [in] dims  Dimensions of `data`, indexed by axis
      @param [in] strides   Strides of elements of `data`, in float elements
                       (not bytes), indexed by axis.
      @param [in] regression_coeffs  Regression coefficients, one per axis, 
                        the same as used for compression (these will have been
                        read from the header).  See docs in compression.h for how 
			this works
      @param [in] axis  The axis to iterate on; top-level call is with 0,
                        this will have values 0 <= axis < num_axes.
      @param [in] indexes  Array of indexes we're processing on axes *prior* to the
                        current axis, i.e. for axes i with 0 <= i < axis.  Must
                        have space equal to at least num_axes-1.
                        Will be set in the recursion.
      @return  Returns true on success, false if we reached the end of the stream
                        before decompression was finished.
 */
bool DecompressFloatInternal(ReverseIntStream *ris,
			     float tick,
			     float *data, 
			     int num_axes,
			     const int *dims, 
			     const int *strides,
			     const float *regression_coeffs,
			     int axis,
			     int *indexes) {
  if (axis + 1 < num_axes) {
    for (int i = 0; i < dims[axis]; i++) {
      indexes[axis] = i;
      // Recurse
      if (!DecompressFloatInternal(ris, tick, data, num_axes, dims, strides, 
				   regression_coeffs, axis + 1, indexes))
        return false;
    }
    return true;
  }
  assert(axis == num_axes - 1);

  /* local_strides and local_coeffs will be the strides and
     corresponding regression coefficients for axes prior to `axis`
     whose corresponding indexes are not zero and whose regression
     coefficients are not 1. */

  int local_strides[16];
  float local_coeffs[16];
  int local_prev_axes = 0;
  float *cur_data = data;
  for (int i = 0; i < axis; i++) {
    /* make `local_data` point to the start of what we're processing here. */
    cur_data += indexes[i] * strides[i];
    if (regression_coeffs[i] != 0.0 && indexes[i] != 0) {
      local_strides[local_prev_axes] = strides[i];
      local_coeffs[local_prev_axes] = regression_coeffs[i];
      local_prev_axes++;
    }
  }

  int dim = dims[axis],
    stride = strides[axis];
  float coeff = regression_coeffs[axis];

  /* The base-case, where there is 1 dimension, is a bit more optimized. */
  float prev_prediction = 0.0;
  float *end = cur_data + (dim * stride);
  for (; cur_data < end; cur_data += stride) {
    float predicted = prev_prediction; /* will be prev element times coeff */
    int32_t code;
    if (!ris->Read(&code))
      return false;
    for (int i = 0; i < local_prev_axes; i++) {
      /* add prediction from lower-numbered axes to this prediction. */
      predicted += cur_data[-(local_strides[i])] * local_coeffs[i];
    }
    float value = predicted + code * tick;
    *cur_data = value;
    prev_prediction = value * coeff;
  }
  return true;
}


int DecompressFloat(const char *src,
		    int num_bytes,
		    float *array, 
		    int num_axes, 
		    const int *dims, 
		    const int *strides) {
  if (num_axes < 1 || num_axes > 16)
    return 1;
  ReverseIntStream ris(src, src + num_bytes);
  float regression_coeffs[16];
  int indexes[16];
  int _num_axes, tick_power;
  if (!ris.Read(&_num_axes) || _num_axes != num_axes)
    return 2;
  if (!ris.Read(&tick_power) || tick_power < -20 || 
      tick_power > 20)
    return 3;

  for (int i = 0; i < num_axes; i++) {
    int32_t dim, coeff;
    if (!ris.Read(&dim) || !ris.Read(&coeff) || dim != dims[i] || dim < 1)
      return 4;
    if (coeff < -256 || coeff > 256)
      return 5;
    regression_coeffs[i] = coeff * (1.0 / 256.0);
  }

  if (!DecompressFloatInternal(&ris, pow(2.0, tick_power), array,
			       num_axes, dims, strides, regression_coeffs,
			       0, indexes)) {
    return 6;
  }
  if (ris.NextCode() != src + num_bytes )
    return 7;
  return 0;  // Success
}




