/**
   CAUTION: although this has the .c suffix, it is actually included from
   lilcom.c as if it were a header.  This is so that we can do "static inline"
   and avoid the possibility of multiply defined symbols.  Regular "inline" or
   "extern inline" seems not to give as much speed improvement.

   But aside from inlining issues, interface-wise this does behave like
   a .c file that you could compile independently.
 */

#include <assert.h>
#include "bit_packer.h"



void bit_packer_init(ssize_t num_samples_to_write,
                     int8_t *compressed_code,
                     int compressed_code_stride,
                     struct BitPacker *packer) {
  packer->num_samples_to_write = num_samples_to_write;
  packer->num_samples_committed = 0;
  packer->num_samples_committed_mod = 0;
  packer->compressed_code_start = compressed_code;
  packer->next_compressed_code = compressed_code;
  packer->compressed_code_stride = compressed_code_stride;
  packer->remaining_bits = 0;
  packer->remaining_num_bits = 0;
}

void bit_packer_commit_block(ssize_t begin_t,
                             ssize_t end_t,
                             int flush,
                             struct BitPacker *packer) {
  assert(begin_t == packer->num_samples_committed);
  int compressed_code_stride = packer->compressed_code_stride;
  int8_t *next_compressed_code = packer->next_compressed_code;

  /* `code` and `bits_in_code` are like a little buffer of bits
     that we're going to write. */
  uint64_t code = packer->remaining_bits;
  unsigned int bits_in_code = packer->remaining_num_bits;

  int cur_index = begin_t % (STAGING_BLOCK_SIZE * 2),
      end_index = cur_index + (end_t - begin_t);
  assert(end_index <= STAGING_BLOCK_SIZE * 2);
  for (; cur_index != end_index; ++cur_index) {
    unsigned int this_num_bits = packer->staging_buffer[cur_index].num_bits;
    uint64_t this_code = packer->staging_buffer[cur_index].code,
        this_mask = ((((uint64_t) 1) << this_num_bits) - 1);
    code |= (this_code & this_mask) << bits_in_code;
    bits_in_code += this_num_bits;
    while (bits_in_code >= 8) {  /* Shift off the lowest-order byte */
      *next_compressed_code = (int8_t) code;
      next_compressed_code += compressed_code_stride;
      code >>= 8;
      bits_in_code -= 8;
    }
  }
  if (flush) {
    if (bits_in_code != 0) {
      /* Get rid of the last partial byte */
      *next_compressed_code = (int8_t) code;
      next_compressed_code += compressed_code_stride;
    }
    assert(end_t == packer->num_samples_to_write);
  } else {
    packer->remaining_bits = code;
    packer->remaining_num_bits = bits_in_code;
  }
  packer->num_samples_committed = end_t;
  packer->num_samples_committed_mod = end_t % (STAGING_BLOCK_SIZE * 2);
  packer->next_compressed_code = next_compressed_code;
}

void bit_packer_finish(struct BitPacker *packer,
                       float *avg_bits_per_sample,
                       int8_t **next_free_byte) {
  ssize_t T = packer->num_samples_to_write;
  assert(packer->num_samples_committed <= T &&
         packer->num_samples_committed >= T - (2 * STAGING_BLOCK_SIZE));
  while (packer->num_samples_committed < T) {
    ssize_t new_end = packer->num_samples_committed + STAGING_BLOCK_SIZE;
    int flush;
    if (new_end > T) {
      new_end = T;
      flush = 1;
    } else {
      flush = 0;
    }
    bit_packer_commit_block(packer->num_samples_committed, new_end,
                            flush, packer);
  }
  ssize_t num_bits_written = (8 * (packer->next_compressed_code - packer->compressed_code_start)) /
      packer->compressed_code_stride;
  float real_bits_per_sample = num_bits_written * 1.0 / packer->num_samples_to_write;
  *avg_bits_per_sample = real_bits_per_sample;
  *next_free_byte = packer->next_compressed_code;
}




void bit_unpacker_init(ssize_t num_samples_to_read,
                       const int8_t *compressed_code, int compressed_code_stride,
                       struct BitUnpacker *unpacker) {
#ifndef NDEBUG
  unpacker->num_samples_to_read = num_samples_to_read;
  unpacker->num_samples_read = 0;
#endif
  unpacker->next_compressed_code = compressed_code;
  unpacker->compressed_code_stride = compressed_code_stride;
  unpacker->remaining_bits = 0;
  unpacker->remaining_num_bits = 0;
}

void bit_unpacker_finish(const struct BitUnpacker *unpacker,
                         const int8_t **next_compressed_code) {
#ifndef NDEBUG
  assert(unpacker->num_samples_read == unpacker->num_samples_to_read);
#endif
  *next_compressed_code = unpacker->next_compressed_code;
}


/* see documentation in header. */
static inline int32_t bit_unpacker_read_next_code(int num_bits,
                                                  struct BitUnpacker *unpacker) {
#ifndef NDEBUG
  assert(unpacker->num_samples_read < unpacker->num_samples_to_read);
  unpacker->num_samples_read++;
#endif
  uint64_t remaining_bits = unpacker->remaining_bits;
  int remaining_num_bits = unpacker->remaining_num_bits;

  while (remaining_num_bits < num_bits) {
    /** We need more bits.  Put them above (i.e. higher-order-than) any bits we
        have currently. */

    /* TODO: read an int at a time. */
    unsigned char code = *unpacker->next_compressed_code;
    unpacker->next_compressed_code += unpacker->compressed_code_stride;
    remaining_bits |= (((uint64_t) ((unsigned char) (code))) << remaining_num_bits);
    remaining_num_bits += 8;
  }
  /* CAUTION: only the lowest-order `num_bits` bits of `ans` are valid; the rest
     are to be ignored by the caller. */
  int32_t ans = remaining_bits;
  unpacker->remaining_bits = remaining_bits >> num_bits;
  unpacker->remaining_num_bits = remaining_num_bits - num_bits;
  return ans;
}
