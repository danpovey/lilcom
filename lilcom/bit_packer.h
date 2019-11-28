#ifndef LILCOM_BIT_PACKER_H_
#define LILCOM_BIT_PACKER_H_ 1

#include <stdint.h>
#include <sys/types.h>
#include "lilcom_common.h"


/**
   This header contains declarations for a BitPacker object whose job is to pack
   codes containing variable numbers of bits into a byte stream.

   There is a BitUnpacker object which reverses the process.  The lengths
   of the codes are not encoded in the stream; the calling code has to
   know that.
 */


/**
   struct BitPacker is responsible for packing integers with between
   1 and 24 bits into bytes.  See functions starting with
   bit_packer_.

   The key function is bit_packer_write_code(); see its documentation.

   See also struct BitUnpacker and associated functions, for how this
   process is reversed.
 */
struct BitPacker {
  int max_bits_per_sample;
  /** num_samples_to_write is the total number of samples which we intend to
      write out. Used only for checking purposes. */
  ssize_t num_samples_to_write;
  /* compressthe byte where we started writing the code.. */
  int8_t *compressed_code_start;

  /* next_compressed_code is where we'll write the next byte of data;
      it will equal compressed_code_start +
      (compressed_code_stride * (number of bytes written)).
      byte of the sequence, where we will put the sample for t == 0.. */
  int8_t *next_compressed_code;
  /* compressed_code_stride is the number of bytes bretween elements of
     the compressed code (normally 1) */

  int compressed_code_stride;

  /* An array of an anonymous struct...  See comments above about the
     constraints on STAGING_BLOCK_SIZE. */
  struct {
    int32_t code;
    int num_bits;
  } staging_buffer[STAGING_BLOCK_SIZE * 2];

  /** num_samples_committed is the number of samples we have so far committed to
      `compressed_code`.  Will always be a multiple of STAGING_BLOCK_SIZE
      until flush() is called, after which nothing else should be done
      with this struct.*/
  ssize_t num_samples_committed;
  /** num_samples_committed_mod always equals num_samples_committed % buffer_size,
      where buffer_size == STAGING_BLOCK_SIZE*2.  */
  ssize_t num_samples_committed_mod;

  /* After we finish writing one staging block, there may be bits
     remaining that didn't fit exactly into a byte.
      0 <= remaining_num_bits < 8 will be the number of those bits, and
      the actual bits will be the lowest-order bits of `remaining_bits`. */
  uint32_t remaining_bits;
  int remaining_num_bits;

  /**
     If this were a class, the members would be:
     bit_packer_init() == constructor
     bit_packer_write_code()
     bit_packer_finish() == destructor
  */
};

/**
   Init the bit-packer object.
      @param [in] num_samples_to_write  The number of samples that will be
                         written to this buffer; each will take up
                         `bits_per_sample` bits.
      @param [in] compressed_code  Start of the place where we will write
                         the compressed code.  This will point to a buffer
                         containing (num_samples_to_write*bits_per_sample + 7)/8
                         bytes.
      @param [in] compressed_code_stride  Spacing between elements of
                          compressed code; will normally be 1.  Must be nonzero.
 */
void bit_packer_init(ssize_t num_samples_to_write,
                     int8_t *compressed_code,
                     int compressed_code_stride,
                     struct BitPacker *packer);

/* Think of this as private. */
void bit_packer_commit_block(ssize_t begin_t,
                             ssize_t end_t,
                             int flush,
                             struct BitPacker *packer);

/**
   Write one code point to the bit-packer object.  This must be called
   (at least once) for each t in 0, 1... pack->num_samples_to_write - 1.

   You're allowed to backtrack and re-write old samples, as long as you
   always satisfy:
     t-you-are-writing-now > largest-t-you-have-ever-written - STAGING_BLOCK_SIZE.

     @param [in] 0 <= t < packer->num_samples_to_write  The time
                 index for which you are writing this sample.  These do not
                 have to be in increasing order, but you are not allowed
                 to call with a 't' value that is less by more than
                 STAGING_BLOCK_SIZE than the largest 't' value you have
                 ever used.
     @param [in] code  The code to write; the lowest 'num_bits' of it
                 will be the relevant ones, and others will be ignored.
     @param [in] num_bits  The number of bits in this code; must
                 be in the range [1..16].
     @param [in,out] packer  The BitPacker object we are using to
                 write the code.
 */
static inline void bit_packer_write_code(ssize_t t,
                                         int code, int num_bits,
                                         struct BitPacker *packer) {
  ssize_t t_mod = t & (STAGING_BLOCK_SIZE * 2 - 1);
  if (t % STAGING_BLOCK_SIZE == 0) {
    /* If t is a multiple of STAGING_BLOCK_SIZE, check whether we
       need to write out a block.  We write out blocks as late as
       we can. */
    if (t_mod == packer->num_samples_committed_mod &&
        t != packer->num_samples_committed) {
      bit_packer_commit_block(
          packer->num_samples_committed,
          packer->num_samples_committed + STAGING_BLOCK_SIZE,
          0, /* no flush */
          packer);
    }
  }
  /* assert(((num_bits - 1) & ~(int) 15) == 0);   check that 0 < num_bits <= 16 */
  packer->staging_buffer[t_mod].code = code;
  packer->staging_buffer[t_mod].num_bits = num_bits;
}



/**
   Flushes remaining samples from the bit-packer object.  Assumes you have
   called write_compressed_code() for all t in 0 .. packer->num_samples_to_write

    @param [out] avg_bits_per_sample  The average number of bits written
                            per sample will be written to here.
    @param [out] next_free_byte  Points to one past the last element
                            written to (taking into account the stride,
                            of course.)
 */
void bit_packer_finish(struct BitPacker *packer,
                      float *avg_bits_per_sample,
                      int8_t **next_free_byte);





struct BitUnpacker {
#ifndef NDEBUG
  /** the number of sample to be read; only used for checks, if NDEBUG not
   * defined */
  ssize_t num_samples_to_read;
  /** The number of samples read; only used for checks, if NDEBUG not
      defined.*/
  ssize_t num_samples_read;
#endif

  /** next_compressed_code is the source of the next byte of the data */
  const int8_t *next_compressed_code;
  int compressed_code_stride;

  uint32_t remaining_bits;
  int remaining_num_bits;
  /**
     If this were a class, the members would be:
     bit_unpacker_init() == constructor
     bit_unpacker_read_next_code()
     bit_unpacker_finish() == destructor [only does checks.]
  */
};

/**
   Initialize BitUnpacker object
       @param [in] num_samples_to_read  The number of samples to be read from
                           this buffer
       @param [in] compressed_code  Pointer to the start of the compressed
                           data we are reading, i.e. it points to the
                           byte where the sample for time t == 0 starts.
       @param [out] unpacker  The unpacker object to be initialized
*/
void bit_unpacker_init(ssize_t num_samples_to_read,
                       const int8_t *compressed_code, int compressed_code_stride,
                       struct BitUnpacker *unpacker);

/**
   Say we are done with this object.  (Only contains checks).
      @param [in] unpacker  Unpacker object we are done with
      @param [out] next_compressed_code   The compressed-code point
              that's one past the end of the stream.  (Should mostly
              be needed for checking.)
 */
void bit_unpacker_finish(const struct BitUnpacker *unpacker,
                         const int8_t **next_compressed_code);


/**
   Read a single code from the bit_unpacker object.
       @param [in] num_bits  The number of bits in the code to be
                  read; must be in [1, 24].
       @param [in,out] unpacker  The unpacker object that we are
                  reading from

       @return    Returns an integer whose least-significant
                  `num_bits` bits coincide with the code that was originally
                  written; the higher order bits are undefined.
 */

static inline int32_t bit_unpacker_read_next_code(int num_bits,
                                                  struct BitUnpacker *unpacker) {
#ifndef NDEBUG
  assert(unpacker->num_samples_read < unpacker->num_samples_to_read);
  unpacker->num_samples_read++;
#endif
  uint32_t remaining_bits = unpacker->remaining_bits;
  int remaining_num_bits = unpacker->remaining_num_bits;

  while (remaining_num_bits < num_bits) {
    /** We need more bits.  Put them above (i.e. higher-order-than) any bits we
        have currently. */
    unsigned char code = *unpacker->next_compressed_code;
    unpacker->next_compressed_code += unpacker->compressed_code_stride;
    remaining_bits |= (((uint32_t) ((unsigned char) (code))) << remaining_num_bits);
    remaining_num_bits += 8;
  }
  /* CAUTION: only the lowest-order `num_bits` bits of `ans` are valid; the rest
     are to be ignored by the caller. */
  int32_t ans = remaining_bits;
  unpacker->remaining_bits = remaining_bits >> num_bits;
  unpacker->remaining_num_bits = remaining_num_bits - num_bits;
  return ans;
}

#endif /* LILCOM_BIT_PACKER_H_ */

