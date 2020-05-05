#ifndef __LILCOM__INT_STREAM_H_
#define __LILCOM__INT_STREAM_H_ 1

#include <stdint.h>
#include <sys/types.h>
#include <vector>
#include "int_math_utils.h"  /* for num_bits() */
#include "bit_stream.h"
#include <iostream>
#include <sstream>


/**
   This header contains declarations for UintStream and IntStream objects
   whose job it is to pack (respectively) unsigned and signed integers into
   bit-streams.  It uses an algorithm that is especially efficient when
   there are correlations between the magnitudes of successive values
   written (i.e. if small values tend to follow small values, and large values
   tend to follow large values).

   It also has ReverseUintStream and ReverseIntStream which are for decoding
   the output of UintStream and IntStream respectively.

   Also we defined TruncatedIntStream and ReverseTruncatedIntStream, which
   are for lossy compression where the user specifies the number of bits
   of precision desired.
*/


/**
   class UintStream (with the help of class BitStream) is responsible for coding
   32-bit integers into a sequence of bytes.

   See also class ReverseUintStream and class IntStream.
 */
class UintStream {
 public:

  /*  Constructor */
  UintStream(): most_recent_num_bits_(0),
                started_(false),
                flushed_(false),
                num_pending_zeros_(0) { }

  /*
    Write the bits.  The lower-order `num_bits_in` of `bits_in` will
    be written to the stream.
      @param [in] value  The number to write.

   Note: you must call Write() AT LEAST ONCE; this class does not
   support writing an empty stream.
   You cannot call Write() after calling Code().
   */
  inline void Write(uint32_t value) {
    assert(!flushed_);
    buffer_.push_back(value);
    if (buffer_.size() >= 64) {
      FlushSome(32);
    }
  }


  /* Gets the code that was written.  After calling this you must
     not call Write(), since this function flushes the stream. */
  std::vector<char> &Code() {
    if (!flushed_) Flush();
    return bit_stream_.Code();
  }

 private:

  /*
    Flush the stream.  Called from Code(); after this is called, you can't
    write any more to the stream.
   */
  void Flush() {
    assert(!flushed_);
    assert(!buffer_.empty());  /* check that data has been written. */
    flushed_ = true;
    FlushSome(buffer_.size());
    if (num_pending_zeros_)
      FlushPendingZeros();
  }

  inline void FlushPendingZeros() {
    assert(num_pending_zeros_ >= 1);
    /*
      This writes the code that dictates how many zeros are in a run of zeros.
      (this is, of course, after we already got into a situation where
      the num_bits is zero, which would have been a code '1, then 0' starting
      from num_bits=1.)

      The code is:
           1 -> 1 zero, then a 1 [or end of stream]
           01x -> 2+x zeros, then a 1 [or end of stream].  (could be 2 or 3 zeros)
           001xx -> 4+xx zeros, then a 1 [or end of stream].  (could be 4,5,6,7 zeros)
           0001xxx -> 8+xxx zeros, then a 1 [or end of stream]. ...
              and so on.
       Note:
     */

    /* -1 below because we don't need to write the most significant bit. */
    int num_bits_written = int_math::num_bits(num_pending_zeros_) - 1;

    /* think of the following as writing `num_bits_written` zeros, then
       a 1. */
    bit_stream_.Write(num_bits_written + 1, 1 << num_bits_written);

    /* Write `num_pending_zeros_`, except for the top bit.  We couldn't just
       write num_bits_written zeros, then num_pending_zeros_ in
       num_bits_written+1 bits, because they'd be in the wrong order; we need
       the `1` to be first so we can identify where the zeros end.
     */
    bit_stream_.Write(num_bits_written,
                      num_pending_zeros_ & ((1 << num_bits_written) - 1));
    num_pending_zeros_ = 0;
  }


  /* buffer_ contains pending values that we have not yet encoded. */
  std::vector<uint32_t> buffer_;

  /* most_recent_num_bits_ is 0 if we have not yet called FlushSome();
     otherwise is is the num-bits of the most recent int that was
     written to the packer_ object (i.e. the one previous to the
     int in buffer_[0]).
  */
  int most_recent_num_bits_;

  BitStream bit_stream_;

  /**
     This function outputs to the `num-bits` array the number of bits
     for each corresponding element of buffer_, increased as necessary
     to ensure that successive elements differ by no more than 1
     (and the zeroth element is no less than most_recent_num_bits_ - 1).

       `num_bits_out` must have size equal to buffer_.size().
  */
  inline void ComputeNumBits(std::vector<int> *num_bits_out) const {
    int prev_num_bits = most_recent_num_bits_;

    std::vector<uint32_t>::const_iterator buffer_iter = buffer_.begin(),
        buffer_end = buffer_.end();
    std::vector<int>::iterator num_bits_iter = num_bits_out->begin();
    /* Simplified version of the code below without end effects treated
       right is:
       for (i = 0 ... size-1):
        num_bits_[i] = max(num_bits[i-1] - 1, num_bits(buffer[i]))
    */
    for (; buffer_iter != buffer_end; ++buffer_iter, ++num_bits_iter) {
      *num_bits_iter = (prev_num_bits = int_math::int_math_max(
          int_math::num_bits(*buffer_iter),
          prev_num_bits - 1));
    }

    std::vector<int>::reverse_iterator num_bits_riter = num_bits_out->rbegin(),
        num_bits_rend = num_bits_out->rend();
    int next_num_bits = 0;
    /* Simplified version of the code below without end effects being
       treated correctly is:
       for (i = size-1, size-2, ... 0):
         num_bits[i] = max(num_bits[i+1] - 1, num_bits[i]);
    */
    for (; num_bits_riter != num_bits_rend; ++num_bits_riter) {
      int this_num_bits = *num_bits_riter;
      next_num_bits = *num_bits_riter = int_math::int_math_max(this_num_bits,
                                                               next_num_bits - 1);
    }
    /*
    size_t size = buffer_.size();
    std::cout << "size = " << size;
    for (int i = 0; i < size; i++) {
      std::cout << "  n=" << buffer_[i] << ", nbits=" << (*num_bits_out)[i];
    }
    std::cout << std::endl;
    */
  }

  /**
     This function takes care of writing the code for one integer ('i') to the
     stream.  Note: the order we write the codes is: each time, we write
     the 1 or 2 bits that determine the *next* sample's num_bits, and then
     we write the bits for *this* sample.  The reason is that we need to
     know the next-sample's num_bits to know whether the most significant
     bit of this sample is to be written.


     (This code does not handle runs of zeros; in fact,
     that is not implemented yet.  We just have a reserved space in the code.)

        @param [in] prev_num_bits  The num_bits of the previous integer...
                             see ComputeNumBits() for what this means.
        @param [in] cur_num_bits  The num_bits of the current 'integer'...
                             see ComputeNumBits() for what num_bits means.
                             It will be greater than or equal to num_bits(i).
                             (search int_math_utils.h for num_bits).
        @param [in] next_num_bits  The num_bits for the next integer
                             in the stream.
        @param [in] i        The integer that we are encoding.
                             (Must satisfy num_bits(i) <= cur_num_bits).
   */
  inline void WriteCode(int prev_num_bits,
                        int cur_num_bits,
                        int next_num_bits,
                        uint32_t i) {
    if (cur_num_bits == 0) {
      num_pending_zeros_++;
      /* Nothing is actually written in this case, until FlushPendingZeros()
         is called.  Since there are 0 bits in this integer, we don't have
         to write any to encode the actual value.
      */
      return;
    } else {
      if (num_pending_zeros_)
        FlushPendingZeros();
    }


    /*std::cout << "Called WriteCode: " << prev_num_bits << ", "
      << cur_num_bits << ", " << next_num_bits << ", " << i << std::endl;*/
    assert(int_math::num_bits(i) <= cur_num_bits);
    /* We write things out of order.. we encode the num_bits of the *next* sample
       before the actual value of this sample.  Ths is needed because of how the
       top_bit_redundant condition works (we'll need the num_bits of the next
       sample in order to encode the current sample's value). */
    int delta_num_bits = next_num_bits - cur_num_bits;

    if (delta_num_bits == 1) {
      /* Write 3 as a 2-bit number.  Think of this as writing a 1-bit, then a
       * 1-bit. */
      bit_stream_.Write(2, 3);
    } else if (delta_num_bits == -1) {
      /* Write 1 as a 2-bit number.  Think of this as writing a 1-bit, then a
       * 0-bit.  (Those written first become the lower order bits). */
      bit_stream_.Write(2, 1);
    } else {
      assert(delta_num_bits == 0);
      /* Write 0 as a 1-bit number.  We allocate half the probability space
         to the num_bits staying the same, then a quarter each to going up
         or down. */
      bit_stream_.Write(1, 0);
    }
    /* if top_bit_redundant is true then cur_num_bits will be exactly
       equal to num_bits(i), so we don't need to write out the highest-order
       bit of i (we know it's set). */
    bool top_bit_redundant = (prev_num_bits <= cur_num_bits &&
                              next_num_bits <= cur_num_bits &&
                              cur_num_bits > 0);

    if (!top_bit_redundant) {
      bit_stream_.Write(cur_num_bits, i);
    } else {
      /* top_bit_redundant is true, so we don't write the top bit.
         We have to zero it out, since BitStream::Write() requires that
         only the bits to write be nonzero.  The caret operator (^)
         takes care of that.
      */
      assert((i & (1 << (cur_num_bits - 1))) != 0);
      bit_stream_.Write(cur_num_bits - 1, i ^ (1 << (cur_num_bits - 1)));
    }
  }

  /**
     Flushes out up to `num_to_flush` ints from `buffer_` (or exactly
     `num_to_flush` ints if it equals buffer_.size().  This is called
     by Write(), and also by Flush().
  */
  inline void FlushSome(uint32_t num_to_flush) {
    size_t size = buffer_.size();
    assert(num_to_flush <= size);
    if (size == 0)
      return;  /* ? */

    /* num_bits contains an upper bound on the number of bits in each element of
       buffer_, from 0 to 32.  We choose the smallest sequence of num_bits
       such that num_bits[i] >= num_bits(buffer_[i]) and successive elements
       of num_bits differ by no more than 1.

       So basically we compute the num_bits() of each element of the buffer,
       then increase the values as necessary to ensure that the
       absolute difference between successive values is no greater than 1.
     */
    std::vector<int> num_bits(size);
    ComputeNumBits(&num_bits);
    if (num_to_flush == size) {
      /* end of stream.  we need to modify for end effects... */
      num_bits.push_back(num_bits.back());
    }

    if (!started_) {
      int first_num_bits = num_bits[0];
      /* write the num_bits of the first sample using 5 bits
         for all num_bits < 31; and if the num_bits is 31 or 32
         then specify it with one more bit.
      */
      bit_stream_.Write(5, (first_num_bits == 32 ? 31 : first_num_bits));
      if (first_num_bits >= 31)
        bit_stream_.Write(1, first_num_bits - 31);
      started_ = true;
      /* treat it as if there had been a value at time t=-1 with
         `first_num_bits` bits.. this only affects the top_bit_redundant
         condition. */
      most_recent_num_bits_ = first_num_bits;
    }
    int prev_num_bits = most_recent_num_bits_,
        cur_num_bits = num_bits[0];

    std::vector<uint32_t>::const_iterator iter = buffer_.begin();
    for (size_t i = 0; i < num_to_flush; i++,++iter) {
      int next_num_bits = num_bits[i+1];
      /* we're writing the element at buffer_[i] to the bit stream, and
         also encoding the exponent of the element following it. */
      WriteCode(prev_num_bits,
                cur_num_bits,
                next_num_bits,
                *iter);
      prev_num_bits = cur_num_bits;
      cur_num_bits = next_num_bits;
    }
    most_recent_num_bits_ = num_bits[num_to_flush - 1];
    buffer_.erase(buffer_.begin(), buffer_.begin() + num_to_flush);
  }
  /* started_ is true if we have called FlushSome() at least once. */
  bool started_;

  /* flushed_ will be set when Flush() has been called by the user. */
  bool flushed_;

  /* num_pending_zeros_ has to do with run-length encoding of sequences
     of 0's. It's the number of zeros in the sequence that we
     need to write. */
  uint32_t num_pending_zeros_;

};


class ReverseUintStream {
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
                          MUST be greater than `code`.
   */
  ReverseUintStream(const char *code,
                    const char *code_memory_end):
      bit_reader_(code, code_memory_end),
      zero_runlength_(-1) {
    assert(code_memory_end > code);
    uint32_t num_bits;
    bool ans = bit_reader_.Read(5, &num_bits);
    assert(ans);
    if (num_bits >= 31) {
      /* we need an extra bit to distinguish between 31 and 32 (the
         initial num_bits can be anywhere from 0 to 32). */
      uint32_t extra_bit;
      ans = bit_reader_.Read(1, &extra_bit);
      assert(ans);
      num_bits += extra_bit;
    }
    prev_num_bits_ = num_bits;
    cur_num_bits_ = num_bits;
  }

  /*
    Read in an integer that was encoded by class UintStream.
        @param [out] int_out  The integer that was encoded will
                        be written to here.
        @return   Returns true on success, false on failure
                  (failure can result from a corrupted or too-short
                  input stream.)
  */
  inline bool Read(uint32_t *int_out) {
    int prev_num_bits = prev_num_bits_,
        cur_num_bits = cur_num_bits_,
        next_num_bits;
    uint32_t bit1, bit2;

    /* The following big if/else statement sets next_num_bits. */

    if (cur_num_bits != 0) {  /* the normal case, cur_num_bits > 0 */
      if (!bit_reader_.Read(1, &bit1))
        return false;  /* truncated code? */
      if (bit1 == 0) {
        next_num_bits = cur_num_bits;
      } else {
        if (!bit_reader_.Read(1, &bit2))
          return false;  /* truncated code? */
        if (bit2) {
          next_num_bits = cur_num_bits + 1;
          if (next_num_bits > 32)
            return false;  /* corrupted code? */
        } else {
          next_num_bits = cur_num_bits - 1;
          if (next_num_bits < 0)
            return false;  /* corrupted code? */
        }
      }
    } else {
      /* cur_num_bits == 0; this is treated specially. */
      if (zero_runlength_ >= 0) {
        /* We have already read the code for the sequence of zeros,
           and set zero_runlength_; we are in the middle of consuming
           it. */
        if (zero_runlength_ == 0) {
          /* we came to the end of the run and now need to output next_num_bits = 1. */
          next_num_bits = 1;
        } else {
          next_num_bits = 0;
        }
        /* note: zero_runlength_ may go to -1 below, which is by design. */
        zero_runlength_--;
      } else {
        /* Assume we have just arrived at the situation where
           cur_num_bits == 0, i.e. the previous num_bits was 1 and
           we encountered the code "1, then 0".
           We have to interpret the following bits to figure out the
           run-length, i.e. the number of zeros.  The code once we reach
           the cur_num_bits=0 situation is as follows:

              1 -> 1 zero, then a 1 [or end of stream]
              01x -> 2+x zeros, then a 1 [or end of stream].  (could be 2 or 3 zeros)
              001xx -> 4+xx zeros, then a 1 [or end of stream].  (could be 4,5,6,7 zeros)
              0001xxx -> 8+xxx zeros, then a 1 [or end of stream]. ...
                and so on.

           Please note that in the explanation above, the binary digits are
           displayed in the order in which they are written, which is from least
           to most significant, rather than the way a human would normally write
           them.
        */
        int num_zeros_read = 0;
        uint32_t bit;
        while (1) {
          if (!bit_reader_.Read(1, &bit) || num_zeros_read > 31)
            return false;  /* truncated or corrupted code? */
          if (bit == 0)
            num_zeros_read++;
          else  /* the bit was 1. */
            break;
        }
        uint32_t x;
        if (!bit_reader_.Read(num_zeros_read, &x))
          return false;

        int num_zeros_in_run = (1 << num_zeros_read) + x;
        /* minus 2 because we already have cur_num_bits == 0,
           so that's the first zero; then we are about to set
           next_num_bits, so that's potentially the second zero.
           If num_zeros_in_run is 1 (the lowest possible value),
           then zero_runlength_ will be -1, which is intended.
        */
        zero_runlength_ = num_zeros_in_run - 2;
        if (num_zeros_in_run == 1) {
          /* the next num_bits is 1. */
          next_num_bits = 1;
        } else {
          next_num_bits = 0;
        }
      }
    }

    bool top_bit_redundant = (prev_num_bits <= cur_num_bits &&
                              next_num_bits <= cur_num_bits &&
                              cur_num_bits > 0);
    if (top_bit_redundant) {
      if (!bit_reader_.Read(cur_num_bits - 1, int_out))
        return false;
      /* we know the top bit is 1, so don't need to read it. */
      *int_out |= (1 << (cur_num_bits - 1));
    } else {
      if (!bit_reader_.Read(cur_num_bits, int_out))
        return false;
    }
    prev_num_bits_ = cur_num_bits;
    cur_num_bits_ = next_num_bits;
    return true;
  }



  /*
     Returns a pointer to one past the end of the last byte read;
     may be needed, for instance, if we know another bit stream is
     directly after this one.
   */
  const char *NextCode() const { return bit_reader_.NextCode(); }

 private:
  ReverseBitStream bit_reader_;

  /* prev_num_bits_ is the num-bits of the most recently read integer
     from the stream. */
  int prev_num_bits_;
  /* cur_num_bits_ is the num-bits of the integer that we are about to
     read from the stream. */
  int cur_num_bits_;

  /*  */
  std::vector<int> pending_num_bits_;
  /* the number of 0-bits we've just seen in the stream starting from
     where the num-bits first became zero (however, this gets reset
     if we have just encoded a run of zeros
  */
  int zero_runlength_;
};

/*
  This encodes signed integers (will be effective when the
  values are close to zero).
 */
class IntStream: public UintStream {
 public:
  IntStream() { }

  inline void Write(int32_t value) {
    UintStream::Write(value >= 0 ? 2 * value : -(2 * value) - 1);
  }

  /* Flush() and Code() are inherited from class UintStream. */

};

/*
  This class is for decoding data encoded by class IntStream.
 */
class ReverseIntStream: public ReverseUintStream {
 public:
  ReverseIntStream(const char *code,
                   const char *code_memory_end):
      ReverseUintStream(code, code_memory_end) { }

  inline bool Read(int32_t *value) {
    uint32_t i;
    if (!ReverseUintStream::Read(&i)) {
      return false;
    } else {
      *value = (i % 2 == 0 ? (int32_t)(i/2) : -(int32_t)(i/2) - 1);
      return true;
    }
  }
  /* Inherits NextCode() from ReverseUintStream. */
};



struct TruncationConfig {
  /*
    Constructor.

     @param [in] num_significant_bits
           This is the number of significant bits we'll use FOR QUIET SOUNDS.
           (We may use more for loud sounds, see alpha).  Specifically: we will
           only start truncating once the rms value of the input exceeds
           num_significant_bits.  Must be > 2.
     @param [in] alpha

           Alpha, which must be in the range [3..64], determines how many more
           significant bits we will allocate as the sound gets louder.
           The rule is as follows: suppose `energy` is the average energy
           per sample of the signal, and `num_bits(energy)` is the number of
           bits in it (see int_math::num_bits())... we will
           compute the number of bits to truncate (`bits_to_truncate`) as
           follows:

              # note: extra_bits is the number of extra bits in the energy above
              # what it would be if the rms value of the signal was equal to
              # `num_significant_bits`.  If we remove extra_bits/2 bits from the
              # signal, we'll get about `num_significant_bits` significant bits
              # remaining.

              extra_bits = num_bits(energy) - (2*num_significant_bits)

              bits_to_truncate = extra_bits/2 - extra_bits/alpha

           If alpha is large (e.g. 64) then extra_bits/alpha will always be
           zero and we'll have about `num_significant_bits`
           bits in the signal.  If alpha is very small (e.g. 4), the
           number of significant bits will rise rapidly as the signal
           gets louder.  You will probably want to tune alpha
           (and block_size, which is the size of the block from which
           the average energy is computed) based on perceptual
           considerations.

           Note: alpha=4 would mean that if we multiply the signal by 2^n,
           the number of bits in the truncated signal would increase by 2^(n/2).
      @param [in] block_size        Size of blocks; on each block we potentially
           change the number of bits to truncate, based on a re-estimated local
           magnitude of the integers to be coded.  On each block, the sum-squared
           statistics are divided by 2, so the block size also dictates how fast
           these stats decay.
     @param [in] first_block_correction  Number of extra bits used to encode the first
           block.  This number will be halved on each block after that, until
           it is zero.  E.g. 5 is suitable.  The idea is that we need to compensate
           for the fact that on the first block we have lpc coeffs = 0 which means
           zero predictions, and on the second block the prediction won't be as
           good as usual because we won't have enough stats available.
  */
  TruncationConfig(
      int32_t num_significant_bits,
      int32_t alpha,
      int32_t block_size,
      int32_t first_block_correction = 5):
      num_significant_bits(num_significant_bits),
      alpha(alpha),
      block_size(block_size),
      first_block_correction(first_block_correction) { }

  /* Caution: this constructor leaves the members undefined. */
  TruncationConfig() { }

  bool IsValid() const {
    return (num_significant_bits > 2 && alpha >= 3 && alpha <= 64 &&
            block_size > 1 && block_size < 10000 &&
            first_block_correction >= 0);
  }
  TruncationConfig(const TruncationConfig &other):
      num_significant_bits(other.num_significant_bits),
      alpha(other.alpha),
      block_size(other.block_size),
      first_block_correction(other.first_block_correction) { }

  /*
     Writes configuration variables to an IntStream.
        @param [in,out] s  The stream to write the configuration variables to
        @param [in] format_version   Version of the format to use;
                       defaults to the current format version (currently 1).
   */
  void Write(IntStream *s, int format_version=1) const {
    assert(format_version == 1);  /* currently only one version supported. */
    s->Write(num_significant_bits);
    s->Write(alpha);
    s->Write(block_size);
    s->Write(first_block_correction);
  }

  /*
    Attempts to read configuration variables from stream.  Returns true
    on success, false on any kind of failure.
      @param [in] format_version  Version of format to read.  Currently
                          only 1 is accepted
      @param [in] s     The stream from which to read the data
      @return  Returns true on success, false on any type of failure.
   */
  bool Read(int format_version, ReverseIntStream *s) {
    if (format_version != 1) return false;
    switch (format_version) {
      case 1:
        return s->Read(&num_significant_bits) &&
            s->Read(&alpha) &&
            s->Read(&block_size) &&
            s->Read(&first_block_correction) &&
            IsValid();
      default:
        return false;
    }
  }

  operator std::string () const {
    std::ostringstream os;
    os << "TruncationConfig{ num-significant-bits=" << num_significant_bits
       << ", alpha=" << alpha
       << ", block-size=" << block_size
       << ", first-block-correction=" << first_block_correction
       << " }";
    return os.str();
  }

  /*
    Sets configuration values by name and value.  Returns true on success, false
    if the name did not match any value.

    The user should call IsValid() after setting all configuration values, to
    make sure they are consistent.
   */
  bool SetConfig(const char *name, int32_t value) {
    if (!strcmp(name, "num_significant_bits"))
      num_significant_bits = value;
    else if (!strcmp(name, "alpha"))
      alpha = value;
    else if (!strcmp(name, "block_size"))
      block_size = value;
    else if (!strcmp(name, "first_block_correction"))
      first_block_correction = value;
    else
      return false;
    return true;
  }

  /* These types could have been int, but it's more convenient
     for the I/O code if they are fixed-size types. */
  int32_t num_significant_bits;
  int32_t alpha;
  int32_t block_size;
  int32_t first_block_correction;
};


/* class Truncation operates on a stream of int32_t and tells us how much
   to truncate them, based on a user-specified configuration.  (The idea is that
   it will tell us how many of the least significant bits to remove depending
   on the current volume of the residual).

   The number of bits to truncate is always determined by previously
   decoded samples, so it does not have to be transmitted (once the
   configuration values of this class are known).
*/
class Truncation {
 public:
  Truncation(const TruncationConfig &config):
      config_(config),
      count_(0),
      sumsq_(0),
      first_block_correction_(2 * config.first_block_correction),
      num_truncated_bits_(0) { }

  /* This function will return the current number of bits to truncate (while
     compressing) or the number of bits that have been truncated (while
     decompressing). */
  int NumTruncatedBits() const {
    return num_truncated_bits_;
  }

  /**
     This function is to be called in sequence for each integer in the stream
     (after GetNumTruncatedBits(), because you'll need that to get the input).

     It updates the state of this object, which will eventually have the
     effect of updating the number of bits to truncate.

       @param [in] i   The element of the stream AFTER TRUNCATION, i.e.
                    the value that would be written to the stream.
                    It would have to be shifted left by num_truncated_bits_
                    to get the actual signal value.
                    CAUTION: although i_truncated is of type int32_t you
                    are not allowed to use the full range of int32_t:
                    specifically, i_truncated * block_size_ must still
                    be representable as int32_t.  (This is no problem
                    for our applications, as block_size will be small,
                    e.g. 32 bits, and the elements of the stream will
                    have either max 17 bits (for residuals of a 16-bit
                    stream) or 25 bits (for residuals of a 24-bit
                    stream, which is not implemented yet).
   */
  void Step(int32_t i_truncated) {
    count_++;
    sumsq_ += i_truncated * (int64_t)i_truncated;

    if (count_ == config_.block_size)
      Update();
  }


  inline static int32_t Truncate(int32_t value, int num_truncated_bits) {
    //std::cout << " Truncating " << value << " to " << (value >> num_truncated_bits);
    return value >> num_truncated_bits;
  }
  inline static int32_t Restore(int32_t truncated_value,
                                int num_truncated_bits) {
    /* The extra term below, + (1 << (num_truncated_bits - 1)) : 0),
       is to split the difference when restoring it, i.e. to use
       the middle of the range of where the number could have been.

       [Note: this argument only applies exactly only in the limit where
       num_truncated_bits is large; for small values, e.g. num_truncated_bits=2,
       it's less exact.]
     */
    int32_t ans = (truncated_value << num_truncated_bits) +
        (num_truncated_bits - 1 > 0 ? (1 << (num_truncated_bits - 1)) : 0);

    return ans;
  }

 private:
  /*
    Updates the state of this object.  (Called when
    count_ == block_size_.
   */
  void Update() {
    assert(count_ == config_.block_size);

    /* nbits_sq is the number of bits in the variance of the signal.

       We divide by (2*block_size_), treating it as the number of points
       in the variance stats.  (We decay the stats by 1/2 each time, so
       2*block_size_ should be understood as:
       block_size_ + block_size_/2 + block_size_/4 + .... ).

       We have to add 2*num_truncated_bits_ because we actually store the
       variance of the numbers *after truncation*.  And the numbers are
       squared, hence the factor of 2.

       extra_bits is, if positive, equal to double the amount by which the
       [num-bits required to encode the signal exactly] exceeds the target
       number of bits.  This will be halved (if alpha is large, or
       decreased by more than that if alpha is small) and will become
       the number of bits to truncate.
    */
    int nbits_sq = int_math::num_bits(sumsq_ / (2*config_.block_size)) +
                 (2*num_truncated_bits_),
        extra_bits = nbits_sq - (2*config_.num_significant_bits) -
        first_block_correction_;
    /* first_block_correction will quickly become zero. */
    first_block_correction_ /= 2;


    int num_truncated_bits;  /* will be new value of num_truncated_bits_ */
    if (extra_bits <= 0) {
      num_truncated_bits = 0;
    } else {
      num_truncated_bits = extra_bits / 2 - extra_bits / config_.alpha;
    }
    /*
      To understand the following, first imagine the +1 and -1 were
      not there.   In that case, it shifts sumsq_ by twice the
      difference between num_truncated_bits and num_truncated_bits_,
      because we compute the variance after the truncation (so we
      need to keep the old part of the stats consistent); we shift
      to match the new representation of the stats.

      The +- 1 can be understood as one extra right-shift, introduced
      in order to decay the variance stats by a factor of 2 each time.
     */
    if (2*num_truncated_bits + 1 > 2*num_truncated_bits_) {
      sumsq_ >>= (2*num_truncated_bits + 1 - 2*num_truncated_bits_);
    } else {
      /* The cast to uint32_t is to suppress warnings about arithmetic
         left shift not being defined. */
      sumsq_ = ((uint32_t)sumsq_ << (2*num_truncated_bits_ - 1 - 2*num_truncated_bits));
    }
    num_truncated_bits_ = num_truncated_bits;
    count_ = 0;
    sumsq_ = 0;
  }
  TruncationConfig config_;

  /* count_ is the number of samples in this block; once is reaches block_size_,
     we recompute num_truncated_bits_, decay the sumsq_ stats, and reset the
     count to 0.
  */
  int count_;

  /*
    sumsq_ is the sum of squares of the (already-truncated) values we're encoding,
    for this block plus the previous block's sumsq_ multiplied by 1/2.
    (This is after correcting for differences in num_truncated_bits_; see the
    code in Update() for more explanation).
   */
  uint64_t sumsq_;


  /* first_block_correction_ starts out as 2 * config_.first_block_correction
     and gets halved on each block until it reaches zero.  It is to ensure we
     can encode the first block with enough accuracy, since the LPC coefficients
     will be zero for that block.  The factor of 2 is because we add it in at
     the stage when we are dealing with squares of values (sumsq_).
  */
  int first_block_correction_;

  /* number of truncated bits (output) */
  int num_truncated_bits_;
};

/*
  class TruncatedIntStream is a lossy version of IntStream, where the least
  significant bits may not be written.  The interface is slightly different
  because the prediction code needs to know what the value would look like
  after being decompressed (which may not be the same as the input).
 */
class TruncatedIntStream: public IntStream, private Truncation {
 public:
  /* Please see constructor of class Truncation for the meaning of the configuration
     values accepted by the constructor.
  */
  TruncatedIntStream(const TruncationConfig &config):
      IntStream(),
      Truncation(config) { }

  inline void Write(int32_t value, int32_t *decompressed_value) {
    int num_truncated_bits = NumTruncatedBits();
    int32_t truncated_value = Truncate(value, num_truncated_bits);
    *decompressed_value = Restore(truncated_value, num_truncated_bits);
    IntStream::Write(truncated_value);
    /* Update the truncation base-class, which keeps NumTruncatedBits() up to
       date. */
    Step(truncated_value);
  }

  /*
    WriteLimited() provides a slightly more complicated interface than Write().
    It's like Write() but it guarantees that predicted + decompressed_residual
    still fits into int16_t.  This makes decompression simpler, as we
    can avoid range checks (or turn them into assertions).

    Note: `decompressed_value_out` is not an approximation to `residual`, it is
    an approximation to `predicted + residual`.
  */
  inline void WriteLimited(int32_t residual, int16_t predicted,
                           int16_t *decompressed_value_out,
                           int32_t *decompressed_residual_out) {
    int num_truncated_bits = NumTruncatedBits();
    int32_t truncated_residual = Truncate(residual, num_truncated_bits),
        decompressed_residual = Restore(truncated_residual, num_truncated_bits),
        decompressed_value = predicted + decompressed_residual;
    if (decompressed_value != static_cast<int16_t>(decompressed_value)) {
      /* The prediction+residual exceeded the range of int16_t.  This should
         be rare. */
      if (truncated_residual >= 0)
        truncated_residual--;  /* Note: if truncated_residual == 0, decompressed_residual
                                  would in general be positive, due to the 2nd term
                                  in the expression in Restore(). */
      else
        truncated_residual++;
      decompressed_residual = Restore(truncated_residual, num_truncated_bits);
      decompressed_value = predicted + decompressed_residual;
      assert(decompressed_value == static_cast<int16_t>(decompressed_value));
    }
    *decompressed_value_out = static_cast<int16_t>(decompressed_value);
    *decompressed_residual_out = decompressed_residual;

    //std::cout << "[T: writing " << residual << "(value=" << (predicted+residual)
    // << ") truncated to " << truncated_residual << "]";

    IntStream::Write(truncated_residual);
    /* Update the truncation base-class, which keeps NumTruncatedBits() up to
       date. */
    Step(truncated_residual);
  }


  /* Flush() and Code() are inherited from class UintStream (and ultimately,
   * from IntStream). */

};

class ReverseTruncatedIntStream: public ReverseIntStream, private Truncation {
 public:
  ReverseTruncatedIntStream(const TruncationConfig &config,
                            const char *code,
                            const char *code_memory_end):
      ReverseIntStream(code, code_memory_end),
      Truncation(config) { }

  inline bool Read(int32_t *value) {
    int32_t truncated_value;
    if (! ReverseIntStream::Read(&truncated_value))
      return false;
    int num_truncated_bits = NumTruncatedBits();
    Step(truncated_value);
    *value = Restore(truncated_value, num_truncated_bits);
    return true;
  }
  /* Inherits NextCode() from ReverseIntStream. */

};


#endif /* __LILCOM___INT_STREAM_H_ */

