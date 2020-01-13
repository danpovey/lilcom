#ifndef __LILCOM_COMPRESSION_H__
#define __LILCOM_COMPRESSION_H__ 1

#include <stdint.h>
#include <sys/types.h>
#include "lpc_stream.h"


/**
   This header provides a C++ interface to lilom's audio-compression algorithm.
   We'll further wrap this in plain "C" for ease of Python-wrapping.
*/


struct CompressorConfig {
  int32_t format_version;
  TruncationConfig truncation;
  int_math::LpcConfig lpc;
  /* `chunk_size` is the number of samples per chunk.  A chunk is
     a largish piece of file, like half a second; its purpose
     is mainly to allow a part of a file to be decompressed,
     since each chunk must be entirely decompressed even if only
     part of it is needed.  Must be >= 128.
   */
  int32_t chunk_size;

  /* `sampling_rate` is the sampling rate in Hz.  Must be > 0. */
  int32_t sampling_rate;

  /* `num_channels` is the number of channels, e.g. 1 for mono, 2 for stereo.
      Must be > 0.*/
  int32_t num_channels;


  /*
    Constructor that creates reasonable default parameters

      @param [in] sampling_rate  Sampling rate in Hz
      @param [in] num_channels   Number of channels in the file, e.g. 1 or 2.
      @param [in] loss_level     Dictates how lossy the compression will be.
                                 0 == lossless, 5 == most lossy.
      @param [in] compression_level    Dictates the speed / file-size tradeoff.
                                 0 == fastest, but biggest file; 5 == slowest,
                                 smallest file.
   */
  CompressorConfig(int32_t sampling_rate, int32_t num_channels,
                   int loss_level, int compression_level);

  /* Copy constructor */
  CompressorConfig(const CompressorConfig &other);

  /* Default constructor, to be called only prior to Read(). */
  CompressorConfig() { }

  /* Writes the configuration information to an IntStream. */
  void Write(IntStream *is) const;

  /* Reads this object from a stream; returns true on success, false
     on any kind of failure */
  bool Read(ReverseIntStream *ris);

  /* Returns true if this object has valid configuration values. */
  bool IsValid() const;

};

struct CompressedChunk {
  const char *data;
  const char *end;
  CompressedChunk(const char *data, const char *end):
      data(data), end(end) { }
  CompressedChunk(std::vector<char> *code) {
    owned_data_.swap(*code);
    assert(!owned_data_.empty());
    data = &(owned_data_[0]);
    end = data + owned_data_.size();
  }
  int32_t size() { return static_cast<int32_t>(end - data); }

  /* Construct from an IntStream; steals the memory from it via Swap().
     This is used when writing meta-information. */
  CompressedChunk(IntStream *is) {
    owned_data_.swap(is->Code());
    data = reinterpret_cast<const char*>(&(owned_data_[0]));
    end = data + owned_data_.size();
  }

  /*
    Constructs this object by compressing the supplied audio data.

        @param [in] config   Configuration class (probably set by
                         the user).  MUST BE VALID (config.IsValid()
                         must return true).
        @param [in] wave_data   The data to compress will be read from here
        @param [in] num_samples   The number of samples to compress;
                             will be config.chunk_size, except if
                             this is the last chunk of its channel,
                             in which case it may be sameller.
        @param [in] data_stride  The stride (in elements) of
                             `data`, would normally be either
                             1 or equal to the number of channels,
                             depending on the output format.
     This function cannot fail except by assertion (indicating code error)
     or memory allocation failure (in which case it will throw
     std::bad_alloc).
   */
  CompressedChunk(const CompressorConfig &config,
                  const int16_t *wave_data,
                  int num_samples,
                  int data_stride);



  /*
    Attempts to decompress the data held here.

        @param [in] config   Configuration class (probably read from
                             the compressed stream)
        @param [in] num_samples  The number of samples to decompress;
                             must be <= the number of samples that
                             was in this chunk, which would be either
                             config.block_size, or a smaller number if
                             this is the last chunk of a channel.
        @param [in] output_stride  The stride (in elements) of
                             `output_data`, would normally be either
                             1 or equal to the number of channels,
                             depending on the output format.
        @param [out] output_data  The data will be decompressed to
                             here.
        @return   Returns the number of samples successfully
                  decoded.  This will be equal to `num_samples` if
                  everything worked as expected.
   */
  int Decompress(const CompressorConfig &config,
                 int num_samples,
                 int output_stride,
                 int16_t *output_data);
 private:
  std::vector<char> owned_data_;
};



class CompressedFile {
 public:
  /*
    Constructs this object, including compressing the data.
    (Later we will allow incremental determinization but for now it
    all gets compressed at once.)
          @param [in] config   Configuration class, must be valid
                            (config.IsValid())
          @param [in] num_samples  Number of samples per channel
                            (num-channels is dictated by the config),
                            must be > 0.
          @param [in] data  The input data to be compressed
          @param [in] sample_stride  The stride between successive
                           samples/time-indexes in `input_data`, would
                           be equal to config.num_channels for wav-format
                           data.
          @param [in] channel_stride  The stride between successive
                            channels in `input_data`, e.g. 1 for
                            wav-format data.
   */
  CompressedFile(const CompressorConfig &config,
                 ssize_t num_samples,
                 const int16_t *data,
                 int sample_stride,
                 int channel_stride);


  /* Default constructor */
  CompressedFile();


  /* Converts this object to linear form, as a char* pointer (i.e. array of
     char's).
        @param [out] length  The number of bytes in the returned array
                          (note: the returned array is not null-terminated,
                          and may contain nulls internally).
        @return       Returns
   */
  char *Write(size_t *length);

  /*
    Initialize the object for reading.
    Assumes this object is currently uninitialized.
    This interface assumes all the data is available at once, so doesn't
    support streaming from a file; we could work on that type of interface
    later on.  This function just reads the header and the chunk sizes;
    it does not do the bulk of the work of decompression.

       @param [in] input  Beginning of where the input data is located
       @param [in] end    End of where the input data is located,
                          i.e. one past the last byte.

       @return           0 on success
                         1 if there was a problem reading the data,
                           e.g. it was corrupted or the wrong type of
                           input data.
                         2 if the file was partially read; in this case
                           you can work out which chunks are available
                           using NumChunksAvailable().
                         3 if memory allocation failed (unlikely)

    You should call ReadData() after this to get the actual data.
  */
  int InitForReading(const char *input, const char *input_end);

  /* Returns the number of samples */
  size_t NumSamples() const { return num_samples_; }

  /* Returns the number of channels */
  int32_t NumChannels() const { return config_.num_channels; }

  /*
     Outputs some portion of the data to `data`.
        @param [in] sample_start   First sample to get (note: the first channel of
                     this will placed at data[0]).  Would be 0 if you want all the
                     data.
        @param [in] num_samples  Number of samples to get.
        @param [in] channel_start First channel to get
        @param [in] num_channels  Number of channels to get.

        @param [in] sample_stride  Stride between samples in `data`; would likely
                      equal either NumChannels() or 1.
        @param [in] channel_stride  Stride between channels in `data`; would
                     likely equal either 1 or num_samples.
        @param [out] data  The data will be output to here; elements
                    data[0] through
                    data[(num_samples-1)*sample_stride + (num_channels-1)*channel_stride]
                    will be written to.
        @returns   Returns true on success, false if some part of the data could
                   not be read, e.g. due to failure of decompression.

      Will die with assertion if the args do not make sense in some way.
  */
  bool ReadData(ssize_t sample_start, ssize_t num_samples,
                int channel_start, int num_channels,
                int sample_stride, int channel_stride,
                int16_t *data);

  bool ReadAllData(int sample_stride, int channel_stride,
                   int16_t *data) {
    return ReadData(0, NumSamples(), 0, NumChannels(),
                    sample_stride, channel_stride, data);
  }

  /* (for use when reading, esp. if Initialize() returned
     ), returns the number of chunks that
     were successfully read.
  */
  int32_t NumChunksAvailable() { return chunks_.size(); }

  ~CompressedFile();

 private:
  /* Writes the metadata of this class (but not the chunk sizes) to a stream
     which will be stored as `header_`.
  */
  void WriteHeader();

  /* Reads the header of the class; sets header_, config_, num_complete_chunks_,
       partial_chunk_size_, and num_samples_.

       @param [in] input  Pointer to where the compressed data starts
       @param [in] input_end  End of the array where the compressed data
                         is; it should be well past where the header
                         actually ends.
       @return          Returns the next unused byte in the input
                        stream (or NULL on error)
  */
  const char* ReadHeader(const char *input, const char *input_end);

  /* Creates the CompressedChunk that encodes the sizes of the actual chunks in
     chunks_ containing the data, and puts this in compressed_chunk_sizes_. */
  void WriteChunkSizes();

  /* Reads the chunk sizes from the stream and writes to compressed_chunk_sizes_
     and chunks_.

       @param [in] input  Pointer to where the compressed data corresponding
                         to `compressed_chunk_sizes_` starts
       @param [in] input_end  End of the array where the compressed data
                        is (includes all the data, not just the data in
                        compressed_chunk_sizes_).
       @return          Returns 0 on success, 1 on failure, 2 on partial
                        failure (partial failure means a truncated stream but
                        at least one chunk was readable).
  */
  int ReadChunkSizes(const char *input, const char *input_end);



  CompressorConfig config_;

  /* `num_samples` is the number of samples per channel.  It's not written
     directly, as it's potentially too large to write to IntStream; instead we
     write num_complete_chunks and partial_chunk_size. */
  ssize_t num_samples_;

  /* `num_complete_chunks` is derived from `num_samples`, as
     num_samples / config.chunk_size. */
  int32_t num_complete_chunks_;
  /* `partial_chunk_size` equals  num_samples % config.chunk_size; if nonzero,
     we'll write a final chunk containing the remaining samples. */
  int32_t partial_chunk_size_;

  /* The serialized form of the header which contains meta-information
     (basically: the class member variables above). */
  CompressedChunk* header_;

  /* This stream encodes the sizes in bytes of the chunks of wave data (used to locate their
     beginnings, for random access */
  CompressedChunk *compressed_chunk_sizes_;

  /* `chunks` contains the compressed data; it's indexed by
       [chunk_index*config_.num_channels + channel_idx].
     We would have used unique_ptr here, but C++11 doesn't
   */
  std::vector<CompressedChunk*> chunks_;


};




#endif /* __LILCOM_COMPRESSION_H__ */

