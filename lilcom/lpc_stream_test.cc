#include <stdlib.h>
#include <iostream>
#include "lpc_stream.h"



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

void lpc_stream_test() {
  int16_t input[500],
      compressed_input[500];


  for (int num_ints = 1; num_ints < 500; num_ints += (1 + num_ints / 4)) {
    int gauss_stddev = 1 << (num_ints % 16);

    int num_significant_bits = 3 + num_ints % 13,
        alpha = 3 + (num_ints % 62),
        block_size = 2 << (num_ints % 10);
    TruncationConfig truncation_config(num_significant_bits,
                                       alpha, block_size);
    int_math::LpcConfig lpc_config;

    for (int n = 0; n < 20; n++) {  /* multiple tries for each size.. */
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

      /*
        Note: the compression error and bits-per-sample will be less for the
        no-lpc version because these samples are actually uncorrelated.
       */
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
        rls.Read(&r);  /*these should mostly fail. */
      }
      int16_t r;
      assert(!rls.Read(&r));  /* we have now definitely overflowed,
                                 so this will definitely fail. */
    }
  }
}


int main() {
  lpc_stream_test();
  std::cout << "Done\n";
}
