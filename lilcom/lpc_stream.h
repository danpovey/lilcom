#ifndef __LILCOM__LPC_STREAM_H__
#define __LILCOM__LPC_STREAM_H__ 1

#include <stdint.h>
#include <sys/types.h>
#include <vector>
#include "int_stream.h"
#include "lpc_math.h"
#include <iostream>


/**
   This header contains class LpcStream which allows you to compress a stream
   of integers using linear prediction, encoding the residuals from that linear
   prediction with TruncatedIntStream or IntStream. */

/*
  class LpcPrediction is a base-class for classes LpcStream and ReverseLpcStream;
  it provides a more convenient interface for class ToeplitzLpcEstimator.
*/
class LpcPrediction: public int_math::ToeplitzLpcEstimator {
 public:
  LpcPrediction(const int_math::LpcConfig &lpc_config):
      ToeplitzLpcEstimator(lpc_config),
      t_(0),
      buffer_storage_(Config().lpc_order + Config().block_size, 0),
      buffer_start_(&(buffer_storage_[Config().lpc_order])),
      residual_(Config().block_size)
  { }

  inline int16_t GetPrediction() const {
    int offset = (t_ % Config().block_size);
    if (offset == 0)
      offset = Config().block_size;
    const int16_t *buffer_pos = buffer_start_ + offset;
    return int_math::compute_lpc_prediction(
        buffer_pos, &(GetLpcCoeffs()));
  }

  /*
    Update the state of the LPC accumulation, advancing t_ by one.
      @param [in] value   The (possibly-lossily-compressed) signal
                      value at time t_
      @param [in] residual     The resdidual, must equal
                    . GetPrediction() - value.
   */
  inline void AdvanceLpcState(int16_t value, int32_t residual) {
    raw_sumsq_ += value * value;
    residual_sumsq_ += residual * residual;
    int t_mod = t_ % Config().block_size;
    /* TODO: remove this assertion */
    assert(residual == value - (int32_t)GetPrediction());
    buffer_start_[t_mod] = value;
    residual_[t_mod] = residual;
    if (t_mod + 1 == Config().block_size) {
      ToeplitzLpcEstimator::AcceptBlock(buffer_start_,
                                        &(residual_[0]));
      /* Copy context to the `lpc_order` samples preceding buffer_start_. */
      for (int i = -Config().lpc_order; i < 0; i++) {
        /* Copy prediction context to the start of the buffer. */
        buffer_start_[i] = buffer_start_[Config().block_size + i];
      }
    }
    t_++;
  }

  ~LpcPrediction() {
    //std::cout << "residual/raw=" << (residual_sumsq_/raw_sumsq_) << "\n";
  }

 private:

  double raw_sumsq_;
  double residual_sumsq_;

  /* t_ starts at 0 and is incremented every time SetValue() is called. */
  int t_;
  /* buffer_storage_ is a vector of size
       Config().lpc_order + Config().block_size

     it's accessed via buffer_start_, which points to the element at
     Config().lpc_order.
  */
  std::vector<int16_t> buffer_storage_;

  /* buffer_start_ points to buffer_storage_[Config().lpc_order],
     which is where the buffer "really" starts (the first lpc_order samples are
     provided for left-context).
     The buffer contains the *compressed then decompressed* samples for the current
     block (block of size Config().block_size)
  */
  int16_t *buffer_start_;

  std::vector<int32_t> residual_;


};



/**
   class TruncatedLpcStream is responsible for coding 16-bit audio into a sequence
   of bytes, supporting truncation.
 */
class LpcStream: public TruncatedIntStream, LpcPrediction {
 public:
  LpcStream(const TruncationConfig &truncation_config,
            const int_math::LpcConfig &lpc_config):
      TruncatedIntStream(truncation_config),
      LpcPrediction(lpc_config) { }

  /*
    Write one sample to the stream
      @param [in] value  The value to write
      @param [out] decompressed_value_out  If non-NULL,
                  the approximated value that you'll get when decompressing the
                  stream will be written to here.
   */
  inline void Write(int16_t value,
                    int16_t *decompressed_value_out = NULL) {
    int16_t prediction = GetPrediction();
    int32_t residual = value - static_cast<int32_t>(prediction);
    int16_t decompressed_value;
    int32_t decompressed_residual;
    TruncatedIntStream::WriteLimited(residual, prediction,
                                     &decompressed_value,
                                     &decompressed_residual);
    AdvanceLpcState(decompressed_value, decompressed_residual);
    if (decompressed_value_out)
      *decompressed_value_out = decompressed_value;
  }

  /* We inherit Code() from base-class TruncatedIntStream. */

};



/**
   class ReverseTruncatedLpcStream is responsible for decoding a sequence of bytes
   into 16-bit audio.
 */
class ReverseLpcStream: public ReverseTruncatedIntStream, LpcPrediction {
 public:
  ReverseLpcStream(const TruncationConfig &truncation_config,
                   const int_math::LpcConfig &lpc_config,
                   const char *code,
                   const char *code_memory_end):
      ReverseTruncatedIntStream(truncation_config,
                                code, code_memory_end),
      LpcPrediction(lpc_config) { }

  inline bool Read(int16_t *value) {
    int32_t residual;
    if (!ReverseTruncatedIntStream::Read(&residual))
      return false;
    int16_t prediction = GetPrediction();
    int32_t next_value = prediction + residual;
    if (next_value != static_cast<int16_t>(next_value)) {
      /* overflowed int16_t range; might indicate corruption. */
      return false;
    }
    AdvanceLpcState(next_value, residual);
    *value = next_value;
    return true;
  }

  /* Inherits NextCode() from ReverseTruncatedIntStream. */
};



#endif /* __LILCOM__LPC_STREAM_H__ */

