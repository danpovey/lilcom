#ifdef _MSC_VER
// see https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=msvc-170
#define _USE_MATH_DEFINES
#endif

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include "int_stream.h"



void uint_stream_test_one() {
  {
    UintStream us;
    us.Write(1);
    /*
      OK, when we write [ 1 ], i.e. a stream with just 1 in it, we write as follows:
     -  00001 as the first_num_bits (a 1 written as 5 bits).  [search for started_ in the code]

     - 0 to indicate that the num_bits following the first num_bits is also 1.
        [i.e. unchanged.]  Search for "assert(delta_num_bits == 0);" to see where
        this happens, and for "num_bits.push_back(num_bits.back());" to see why
        the num-bits "after the end" is the same as the num-bits of the last number
        that is written.

      [then nothing; the 1-bit doesn't have to be written as we know what it is
      from the sequence of exponents.]
    */
    assert(us.Code().size() == 1 &&
           us.Code()[0] == (char)1);


    ReverseUintStream rus(&(us.Code()[0]),
                          &(us.Code()[0]) + us.Code().size());
    uint32_t i;
    bool ans = rus.Read(&i);
    assert(ans);
    assert(i == 1);
  }

  {
    UintStream us;
    us.Write(0);
    /*
      OK, when we write 0 we write as follows:
     -  00000  as the first_num_bits (0 written as 5 bits).  [search for started_ in the code]

     -  1 to indicate that the runlength of zeros was 1.  [since the stream is
            terminated.  Note: runlength of 2 would be fine too, it doesn't change anything]

    */
    assert(us.Code().size() == 1 &&
           us.Code()[0] == (char)32);


    ReverseUintStream rus(&(us.Code()[0]),
                          &(us.Code()[0]) + us.Code().size());
    uint32_t i;
    bool ans = rus.Read(&i);
    assert(ans);
    assert(i == 0);
  }


  {
    UintStream us;
    us.Write(17);

    /* The logic is similar to above, except we write 17.  We write as follows:
        - 00101 == 5 as the first_num_bits (since 17 needs 5 bits to write).
        - 0 to indicate that the num_bits following the first num_bits is also 5,
          see comments above.
        - 0001 which is the lower-order bits of 17 (the top bit doesn't need
          to be written).

       The things written first go to the lowest order bits of the first bytes.

       So: first byte = 01|0|00101, i.e. 01000101 = 69.
       Second byte = 00 i.e. 0.
    */
    assert(us.Code().size() == 2 &&
           us.Code()[0] == (char)69 &&
           us.Code()[1] == (char)0);

    ReverseUintStream rus(&(us.Code()[0]),
                          &(us.Code()[0]) + us.Code().size());
    uint32_t i;
    bool ans = rus.Read(&i);
    assert(ans);
    assert(i == 17);
  }
}

uint32_t rand_special() {
  uint32_t num_to_shift = rand() % 32,
      n = rand() % 5;
  return n << num_to_shift;
}

void int_stream_test_two() {
  unsigned int input[500];

  for (int num_ints = 1; num_ints < 500; num_ints++) {
    for (int n = 0; n < 20; n++) {  /* multiple tries for each size.. */

      UintStream us;
      for (int i = 0; i < num_ints; i++) {
        uint32_t r = rand_special();
        input[i] = r;
        us.Write(r);
      }

      ReverseUintStream rus(&(us.Code()[0]),
                            &(us.Code()[0]) + us.Code().size());
      for (int i = 0; i < num_ints; i++) {
        uint32_t r;
        bool ans = rus.Read(&r);
        assert(ans);
        if (r != input[i]) {
          std::cout << "Failure, " << r << " != " << input[i] << "\n";
          exit(1);
        }
      }
      for (int i = 0; i < 8; i++) {
        uint32_t r;
        rus.Read(&r);  /*these should mostly fail. */
      }
      uint32_t r;
      assert(!rus.Read(&r));  /* we have now definitely overflowed,
                                 so this will definitely fail. */
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

void test_truncation_config_io() {
  TruncationConfig s;
  s.num_significant_bits = 4;
  s.alpha = 3;
  s.block_size = 32;
  s.first_block_correction = 5;
  IntStream is;
  int format_version = 1;
  s.Write(&is, format_version);
  ReverseIntStream ris(&is.Code()[0],
                       &is.Code()[0] + is.Code().size());
  TruncationConfig s2;
  bool ans = s2.Read(format_version, &ris);
  assert(ans &&
         s2.num_significant_bits == s.num_significant_bits &&
         s2.alpha == s.alpha &&
         s2.block_size == s.block_size &&
         s2.first_block_correction == s.first_block_correction);
}

void truncated_int_stream_test() {
  int input[500],
      compressed_input[500];


  for (int num_ints = 1; num_ints < 500; num_ints+= (1 + num_ints / 4)) {

    int num_significant_bits = 3 + num_ints % 28,
        alpha = 3 + (num_ints % 62),
        block_size = 2 << (num_ints % 10);
    TruncationConfig config(num_significant_bits,
                            alpha, block_size);

    for (int n = 0; n < 20; n++) {  /* multiple tries for each size.. */
      int64_t error_sumsq = 0, data_sumsq = 0;

      TruncatedIntStream tis(config);
      for (int i = 0; i < num_ints; i++) {
        int32_t r = (int)(rand_gauss() * 1000);
        input[i] = r;
        int32_t r_compressed;
        tis.Write(r, &r_compressed);
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

      ReverseTruncatedIntStream rtis(config,
                                     &(tis.Code()[0]),
                                     &(tis.Code()[0]) + tis.Code().size());


      for (int i = 0; i < num_ints; i++) {
        int32_t r;
        bool ans = rtis.Read(&r);
        assert(ans);
        if (r != compressed_input[i]) {
          std::cout << "Failure, " << r << " != " << compressed_input[i] << "\n";
          exit(1);
        }
      }
      for (int i = 0; i < 8; i++) {
        int32_t r;
        rtis.Read(&r);  /*these should mostly fail. */
      }
      int32_t r;
      assert(!rtis.Read(&r));  /* we have now definitely overflowed,
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
  uint_stream_test_one();
  int_stream_test_two();
  int_stream_test_gauss();
  truncated_int_stream_test();
  test_truncation_config_io();
  std::cout << "Done\n";
}
