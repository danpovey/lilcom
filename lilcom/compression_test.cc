#include <stdlib.h>
#include <iostream>
#include "compression.h"



inline bool operator == (const TruncationConfig &a, const TruncationConfig &b) {
  return a.num_significant_bits == b.num_significant_bits &&
      a.alpha == b.alpha &&
      a.block_size == b.block_size &&
      a.first_block_correction == b.first_block_correction;
}


inline bool operator == (const int_math::LpcConfig &a, const int_math::LpcConfig &b) {
  return a.lpc_order == b.lpc_order &&
      a.block_size == b.block_size &&
      a.eta_inv == b.eta_inv &&
      a.diag_smoothing_power == b.diag_smoothing_power &&
      a.abs_smoothing_power == b.abs_smoothing_power;
}


  /* Used for testing; checks that members are equal. */
inline bool operator == (const CompressorConfig &a, const CompressorConfig &b) {
  return a.format_version == b.format_version &&
      a.truncation == b.truncation &&
      a.lpc == b.lpc &&
      a.chunk_size == b.chunk_size &&
      a.sampling_rate == b.sampling_rate &&
      a.num_channels == b.num_channels;
}


void compressor_config_test() {
  int sampling_rate = 42100;
  for (int num_channels = 1; num_channels < 5; num_channels++) {
    for (int num_samples = 10; num_samples * num_channels < 500; num_samples++) {
      for (int loss_level = 0; loss_level <= 5; loss_level++) {
        for (int compression_level = 0; compression_level <= 5; compression_level++) {
          CompressorConfig config(sampling_rate, num_channels, loss_level,
                                  compression_level);
          IntStream is;
          config.Write(&is);
          ReverseIntStream ris(&(is.Code()[0]),
                               &(is.Code()[0]) + is.Code().size());
          CompressorConfig config2;
          bool ans = config2.Read(&ris);
          assert(ans && config2 == config);
        }
      }
    }
  }
}

inline double rand_uniform() {
  int64_t r = rand();
  assert(r >= 0 && r < RAND_MAX);
  float ans = (1.0 + r) / (static_cast<double>(RAND_MAX) + 2);
  assert(ans > 0.0 && ans < 1.0 && ans == ans);
  return ans;
}

inline float rand_gauss() {
  return sqrtf(-2 * logf(rand_uniform())) *
      cosf(2 * M_PI * rand_uniform());
}




void compressed_file_test() {
  int16_t input[500],
      decompressed[500];

  int sampling_rate = 42100;
  for (int num_channels = 1; num_channels < 5; num_channels++) {
    for (int num_samples = 10; num_samples * num_channels < 500; num_samples += (1 + num_samples)) {
      for (int loss_level = 0; loss_level <= 5; loss_level++) {
        for (int compression_level = 0; compression_level <= 5; compression_level++) {
          CompressorConfig config(sampling_rate, num_channels, loss_level,
                                  compression_level);
          std::cout << "Config is: " << (std::string)config << "\n";
          if (loss_level + compression_level % 2 == 0)
            config.chunk_size = 32;  /* sometimes try small chunk sizes, to
                                      * exercise more of the code */

          int gauss_stddev = 1 << ((loss_level + compression_level) % 16);

          bool channel_major = (loss_level % 2 != 0);
          int channel_stride, sample_stride;
          if (channel_major) {
            channel_stride = num_samples;
            sample_stride = 1;
          } else {
            channel_stride = 1;
            sample_stride = num_channels;
          }
          for (int i = 0; i < num_channels * num_samples; i++) {
            input[i] = round(rand_gauss() * gauss_stddev);
            decompressed[i] = std::numeric_limits<int16_t>::min();
          }
          CompressedFile cf(config, num_samples,
                            input,
                            sample_stride, channel_stride);

          if (loss_level + compression_level % 2 == 0) {
            /* test without serialization in between */
            bool ans = cf.ReadAllData(sample_stride, channel_stride,
                                      decompressed);
            assert(ans);
          } else {
            /* test with serialization. */
            size_t num_bytes;
            char *c = cf.Write(&num_bytes);
            CompressedFile cf2;
            int ret = cf2.InitForReading(c, c + num_bytes);
            assert(ret == 0);
            bool ans = cf2.ReadAllData(sample_stride, channel_stride,
                                       decompressed);
            assert(ans);
            delete c;
          }
          /* TODO: compare */
          int64_t error_sumsq = 0, data_sumsq = 0;
          for (int i = 0; i < num_channels * num_samples; i++) {
            int r = input[i],
                s = decompressed[i];
            data_sumsq += r * r;
            error_sumsq += (r - s) * (r - s);
            assert(s != std::numeric_limits<int16_t>::min()); // TEMP.
          }
          ssize_t num_ints = num_channels * num_samples;
          std::cout << " For gauss-stddev=" << gauss_stddev
                    << ", num_channels=" << num_channels
                    << ", num_samples=" << num_samples
                    << ", loss_level=" << loss_level
                    << ", compression_level=" << compression_level
                    << ": data-rms=" << (sqrt(data_sumsq * 1.0 / num_ints))
                    << ", error-rms=" << (sqrt(error_sumsq * 1.0 / num_ints))
                    << ", ratio=" << (sqrt(error_sumsq * 1.0 / data_sumsq))
                    << std::endl;
        }
      }
    }
  }
}
/*
    for (int n = 0; n < 20; n++) {
      int64_t error_sumsq = 0, data_sumsq = 0, error_sumsq_nolpc = 0;

      LpcStream ls(truncation_config, lpc_config);
      TruncatedIntStream tis(truncation_config);

      for (int i = 0; i < num_ints; i++) {
        int32_t r = round(rand_gauss() * gauss_stddev);
        input[i] = r;
        int16_t r_compressed;
        ls.Write(r, &r_compressed);
        int32_t r_compressed_nolpc;
        tis.Write(r, &r_compressed_nolpc);
        //std::cout << "r_compressed = " << r_compressed << "\n";
        compressed_input[i] = r_compressed;

        error_sumsq += (r - r_compressed) * (r - r_compressed);
        error_sumsq_nolpc += (r - r_compressed_nolpc) * (r - r_compressed_nolpc);
        data_sumsq += r * r;
      }

 //       Note: the compression error and bits-per-sample will be less for the
 //       no-lpc version because these samples are actually uncorrelated.
      std::cout << " For nsb=" << num_significant_bits
                << ", gauss-stddev=" << gauss_stddev
                << ", num_ints=" << num_ints
                << ", alpha=" << alpha
                << ", block-size=" << block_size
                << ": avg-data=" << (sqrt(data_sumsq * 1.0 / num_ints))
                << ", avg-compression-error=[" << (sqrt(error_sumsq * 1.0 / num_ints))
                << ",no-lpc=" << (sqrt(error_sumsq_nolpc * 1.0 / num_ints))
                << "], bits-per-sample=[" << (ls.Code().size() * 8.0 / num_ints)
                << ",no-lpc=" << (tis.Code().size() * 8.0 / num_ints)
                << "]" << std::endl;

      ReverseLpcStream rls(truncation_config, lpc_config,
                           &(ls.Code()[0]),
                           &(ls.Code()[0]) + ls.Code().size());


      for (int i = 0; i < num_ints; i++) {
        int16_t r;
        bool ans = rls.Read(&r);
        assert(ans);
        if (r != compressed_input[i]) {
          std::cout << "Failure, " << r << " != " << compressed_input[i] << "\n";
          exit(1);
        }
      }
      for (int i = 0; i < 8; i++) {
        int16_t r;
        rls.Read(&r);  //these should mostly fail.
      }
      int16_t r;
      assert(!rls.Read(&r));  // we have now definitely overflowed,
                              //   so this will definitely fail.
    }
  }
  }*/


int main() {
  compressor_config_test();
  compressed_file_test();
  std::cout << "Done\n";
}
