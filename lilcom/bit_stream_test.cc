#include <stdio.h>
#include <cassert>
#include "bit_stream.h"


void bit_stream_test_one() {
  BitStream bs;
  bs.Write(8, 255);
  assert(bs.Code().size() == 1 &&
         bs.Code()[0] == (char)255);

  ReverseBitStream rbs(&(bs.Code()[0]), &(bs.Code()[0]) + bs.Code().size());
  uint32_t bits;
  bool ans = rbs.Read(8, &bits);
  assert(ans && bits == 255);
  assert(!rbs.Read(1, &bits));
}

void bit_stream_test_two() {
  int n = 5;
  uint32_t bits [] = { 0, 12, 1, 13, 2 };
  int32_t num_bits [] = { 1, 5, 2, 4, 3 };

  BitStream bs;
  for (int i = 0; i < n; i++)
    bs.Write(num_bits[i], bits[i]);
  ReverseBitStream rbs(&(bs.Code()[0]), &(bs.Code()[0]) + bs.Code().size());
  for (int i = 0; i < n; i++) {
    uint32_t this_bits;
    bool ans = rbs.Read(num_bits[i], &this_bits);
    assert(ans);
    assert(this_bits == bits[i]);
  }
  assert(rbs.NextCode() == &(bs.Code()[0]) + bs.Code().size());
}

void bit_stream_test_order() {
  BitStream bs;
  bs.Write(1, 1);
  bs.Write(1, 0);
  ReverseBitStream rbs(&(bs.Code()[0]), &(bs.Code()[0]) + bs.Code().size());
  uint32_t i;
  rbs.Read(2, &i);
  /*  the next line prints 'i is 1'.  Shows that the first bits written
      become the lower-order bits.  (This is relevant when writing and reading
      different-sized chunks).
  */
  printf("i is %d\n", (int)i);

}

int main() {
  bit_stream_test_one();
  bit_stream_test_two();
  bit_stream_test_order();
  printf("Done\n");
}
