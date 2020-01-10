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


  for (int num_ints = 1; num_ints < 500; num_ints+= (1 + num_ints / 4)) {

    int num_significant_bits = 3 + num_ints % 28,
        alpha = 3 + (num_ints % 62),
        block_size = 2 << (num_ints % 10);
    TruncationConfig truncation_config(num_significant_bits,
                                       alpha, block_size);
    int_math::LpcConfig lpc_config;

    for (int n = 0; n < 20; n++) {  /* multiple tries for each size.. */
      int64_t error_sumsq = 0, data_sumsq = 0;

      LpcStream ls(truncation_config, lpc_config);
      for (int i = 0; i < num_ints; i++) {
        int32_t r = (int)(rand_gauss() * 1000);
        input[i] = r;
        int16_t r_compressed;
        ls.Write(r, &r_compressed);
        //std::cout << "r_compressed = " << r_compressed << "\n";
        compressed_input[i] = r_compressed;

        error_sumsq += (r - r_compressed) * (r - r_compressed);
        data_sumsq += r * r;
      }
      std::cout << " For nsb=" << num_significant_bits
                << ", num_ints=" << num_ints
                << ", alpha=" << alpha
                << ", block-size=" << block_size
                << ": avg-data=" << (sqrt(data_sumsq * 1.0 / num_ints))
                << ": avg-compression-error=" << (sqrt(error_sumsq * 1.0 / num_ints))
                << std::endl;

      ls.Flush();
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


void int_stream_test_gauss() {
  int32_t buffer[10000];

  for (int stddev = 1; stddev <= 10000; stddev *= 2) {
    /* Entropy of Gaussian distribution is
          H(x) = 1/2 (1 + log(2 sigma^2 pi))
       since we are integerizing without any scaling, for large
       enough variance this is the same as the entropy of the discretized
       distribution.  (Of course, the definitions of entropy are a little
       different).
     */
    double sumsq = 0.0;
    float entropy = 0.5 * (1.0 + log(2.0 * stddev * stddev * M_PI));
    for (int i = 0; i < 10000; i++) {
      float f = rand_gauss() * stddev;
      int f_int = (int)round(f);
      buffer[i] = f_int;
      sumsq += f_int * (double)f_int;
    }

    std::cout <<  "About to compress; stddev=" << stddev << " (measured=" << (float)(sqrtf(sumsq / 10000))
              << "), theoretical entropy in base-2 (i.e. min bits per sample) is " <<
        (entropy / log(2.0)) << "\n";

    IntStream is;
    for (int i = 0; i < 10000; i++) {
      is.Write(buffer[i]);
    }
    is.Flush();


    ReverseIntStream ris(&(is.Code()[0]),
                         &(is.Code()[0]) + is.Code().size());
    for (int i = 0; i < 10000; i++) {
      int32_t r;
      bool ans = ris.Read(&r);
      assert(ans);
      assert(r == buffer[i]);
    }
    size_t num_bits = is.Code().size() * 8;
    std::cout << "Actual bits per sample was " << (num_bits * 1.0 / 10000) << "\n";
  }
}


int main() {
  lpc_stream_test();
  std::cout << "Done\n";
}
