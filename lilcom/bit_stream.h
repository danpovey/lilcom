#ifndef LILCOM_BIT_STREAM_H_
#define LILCOM_BIT_STREAM_H_ 1

#include <stdint.h>
#include <sys/types.h>
#include <vector>


/**
   This header contains the implementation of a BitStream object
   whose job is to pack codes containing variable numbers of bits into a byte
   stream.

   There is a ReverseBitStream object which reverses the process.  The lengths
   of the codes, and the number of codes, are not encoded in the stream; the
   calling code has to know that.
 */


/**
   class BitStream is responsible for packing integers with between
   1 and 32 bits into bytes.

   See also class ReverseBitStream.
 */
class BitStream {
 public:
  /*  Constructor */
  BitStream():
      remaining_bits_(0),
      remaining_num_bits_(0),
      flushed_(false) { }

  /*
    Write the bits.  The lower-order `num_bits_in` of `bits_in` will
    be written to the stream.
      @param [in] num_bits_in   Number of bits to write; must
                       be in [0,32]
      @param [in] bits_in       Bits to write; lower-order
                       `num_bits_in` bits will be written.
                       The remaining bits MUST BE ZERO.
   */
  inline void Write(int num_bits_in, uint32_t bits_in) {
    //std::cout << "[Writing " << bits_in << " as " << num_bits_in << " bits].";
    assert(static_cast<unsigned int>(num_bits_in) <= 32);
    /* assert out-of-range bits are zero. */
    assert((bits_in & ~((1 << num_bits_in) - 1)) == 0);

    int num_bits = remaining_num_bits_;
    int64_t bits = (((uint64_t)bits_in) << num_bits) | remaining_bits_;
    num_bits += num_bits_in;
    while (num_bits >= 8) {
      code_.push_back((int8_t)bits);
      num_bits -= 8;
      bits >>= 8;
    }
    remaining_bits_ = bits;
    remaining_num_bits_ = num_bits;
  }

  /**
     Flushes out the last partial byte.  This should be called exactly once,
     after you are done calling Write(), and after that you should not
     call Write() again.  This is not checked.
  */
  inline void Flush() {
    assert(!flushed_);
    flushed_ = true;
    if (remaining_num_bits_ > 0) {
      code_.push_back((int8_t)remaining_bits_);
      remaining_num_bits_ = 0;
    }
  }

  /* Gets the code that was written.  You should not call thes
     before having called Flush(). */
  const std::vector<int8_t> &Code() const { assert(flushed_); return code_; }
 private:
  std::vector<int8_t> code_;

  /* remaining_bits contains any bits
     remaining that didn't fit exactly into a byte.

      0 <= remaining_num_bits < 8 will be the number of those bits, and
      the actual bits will be the lowest-order bits of `remaining_bits`. */
  uint32_t remaining_bits_;
  int remaining_num_bits_;
  /* flushed_ is true if Flush() was called. Helps check usage. */
  bool flushed_;
};


class ReverseBitStream {
 public:
  /*
     Constructor
         @param [in] code  First byte of the code to be read.
         @param [in] code_memory_end  Pointer to one past the
                          end of the memory region allocated for
                          `code`.  This is only to prevent
                          segmentation faults in case a
                          corrupted or invalid stream is
                          attempted to be decoded; in most cases,
                          we'll never reach there.
   */
  ReverseBitStream(const int8_t *code,
                   const int8_t *code_memory_end):
      next_code_(code),
      code_memory_end_(code_memory_end),
      remaining_bits_(0),
      remaining_num_bits_(0) {
    /* >= just to allow an empty stream. */
    assert(code_memory_end >= code);
  }

  /*
    Read some bits of the code.  If you create this object
    with the output of class BitStream() and call with the same
    sequence of num_bits, you'll get the same sequence of bits_out
    values.  If the sequence of num_bits values is different
    (e.g. write 4 bits then 4 bits, and read 8 bits), then
    the numbers which are the *first to be written become
    the lower-order bits.  E.g. if you read 3 as 4-bit then 0 as 4-bit,
    then read 8-bit, you will get 3, not 3*16.

        @param [in] num_bits  The number of bits to be read.
                          Must be in [0,32].
        @param [out] bits_out  On success the bits will be written to the
                          lowest-order bits of `*bits`; its
                          highest-order bits will be zero.
        @return   Returns true on success, false on failure
                  (the only possible failure is an attempt to
                  read past code_memory_end_).
   */
  inline bool Read(int num_bits,
                   uint32_t *bits_out) {
    int remaining_num_bits = remaining_num_bits_;
    uint64_t remaining_bits = remaining_bits_;
    while (remaining_num_bits < num_bits) {
      if (next_code_ >= code_memory_end_) {
        //std::cout << "Past stream end!\n";
        return false;
      }
      unsigned char code = *(next_code_++);
      remaining_bits |= (((uint64_t) ((unsigned char) (code))) << remaining_num_bits);
      remaining_num_bits += 8;
    }
    *bits_out = remaining_bits & (((uint64_t)1 << num_bits) - 1);
    //std::cout << "[Read " << *bits_out << " as " << num_bits << " bits.";
    remaining_num_bits_ = remaining_num_bits - num_bits;
    remaining_bits_ = remaining_bits >> num_bits;
    return true;
  }

  /*
     Returns a pointer to one past the end of the last byte read;
     may be needed, for instance, if we know another bit stream is
     directly after this one.
   */
  const int8_t *NextCode() const { return next_code_; }

 private:
      /* next_code_ is advanced each time we read a byte; it always points to
         the next byte to be read. */
  const int8_t *next_code_;
  const int8_t *code_memory_end_;

  uint64_t remaining_bits_;
  int remaining_num_bits_;

};


#endif /* LILCOM_BIT_STREAM_H_ */

