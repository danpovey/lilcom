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
  int input[150];


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
        assert(r == input[i]);
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

int main() {
  int_stream_test_one();
  int_stream_test_two();
  printf("Done\n");
}
