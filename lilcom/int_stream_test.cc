#include <stdlib.h>
#include <iostream>
#include "int_stream.h"



void int_stream_test_one() {
  {
    UintStream us;
    us.Write(1);
    us.Flush();
    /*
      OK, when we write 1 we write as follows:
     -  00001  as the first_num_bits (1 written as 5 bits).  [search for started_ in the code]

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
    us.Write(17);
    us.Flush();

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
  int input[500];

  for (int num_ints = 1; num_ints < 500; num_ints++) {
    for (int n = 0; n < 20; n++) {  /* multiple tries for each size.. */

      UintStream us;
      for (int i = 0; i < num_ints; i++) {
        uint32_t r = rand_special();
        input[i] = r;
        us.Write(r);
      }
      us.Flush();

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

void int_stream_test_gauss() {
  uint32_t buffer[4096];

  for (int stddev = 2; stddev <= 4096; stddev *= 2) {
    /* Entropy of Gaussian distribution is
          H(x) = 1/2 (1 + log(2 sigma^2 pi))
       since we are integerizing without any scaling, for large
       enough variance this is the same as the entropy of the discretized
       distribution.  (Of course, the definitions of entropy are a little
       different).
     */
    double sumsq = 0.0;
    float entropy = 0.5 * (1.0 + log(2.0 * stddev * stddev * M_PI));
    for (int i = 0; i < 4096; i++) {
      float f = rand_gauss() * stddev;
      int f_int = (int)round(f);
      buffer[i] = (f_int >= 0 ? 2 * f_int : - (2 * f_int) - 1);
      sumsq += f_int * (double)f_int;
    }

    std::cout <<  "About to compress; stddev=" << stddev << " (measured=" << (float)(sqrtf(sumsq / 4096))
              << "), theoretical entropy in base-2 (i.e. min bits per sample) is " <<
        (entropy / log(2.0)) << "\n";

    UintStream us;
    for (int i = 0; i < 4096; i++) {
      us.Write(buffer[i]);
    }
    us.Flush();
    size_t num_bits = us.Code().size() * 8;
    std::cout << "Actual bits per sample was " << (num_bits * 1.0 / 4096) << "\n";
  }
}


int main() {
  int_stream_test_one();
  int_stream_test_two();
  int_stream_test_gauss();
  printf("Done\n");
}
