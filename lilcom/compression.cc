#include "compression.h"





CompressorConfig::CompressorConfig(int32_t sampling_rate, int32_t num_channels,
                                   int loss_level, int compression_level):
    sampling_rate(sampling_rate),
    num_channels(num_channels) {
  format_version = 1;
  /* chunk_size_ is not very critical, it only affects the tradeoff between the
     speed when you want to decompress a small part of a file, vs. the small
     overhead of storing the bytes-per-chunk. */
  chunk_size = 8192;

  if (loss_level == 0) {
    /* no truncation.  Later we can bypass the truncation code in this case,
       for speed. */
    truncation.num_significant_bits = 18;
  } else if (loss_level > 0 && loss_level <= 5) {
    /* 7, 6, 5, 4 or 3 bits minimum.  Note: because we set alpha smallish,
       more bits than this will actually be used.
     */
    truncation.num_significant_bits = 8 - compression_level;
  } else {
    /* Make the object invalid, IsValid() will fail. */
    sampling_rate = -1;
  }
  truncation.alpha = 4;
  truncation.block_size = 16;
  truncation.first_block_correction = 6;

  lpc.diag_smoothing_power = -23;
  lpc.abs_smoothing_power = -33;

  switch(compression_level) {
    case 5:  /* most compression but slowest */
      lpc.block_size = 32;
      lpc.eta_inv = 128;
      lpc.lpc_order = 32;
      break;
    case 4:
      lpc.block_size = 32;
      lpc.eta_inv = 128;
      lpc.lpc_order = 16;
      break;
    case 3:
      lpc.block_size = 32;
      lpc.eta_inv = 64;
      lpc.lpc_order = 8;
      break;
    case 2:
      lpc.block_size = 16;
      lpc.eta_inv = 64;
      lpc.lpc_order = 4;
      break;
    case 1:
      lpc.block_size = 16;
      lpc.eta_inv = 64;
      lpc.lpc_order = 2;
      break;
    default:
      /* Make the object invalid, IsValid() will fail. */
      sampling_rate = -1;
  }
}

CompressorConfig::CompressorConfig(const CompressorConfig &other):
    format_version(other.format_version),
    truncation(other.truncation),
    lpc(other.lpc),
    chunk_size(other.chunk_size),
    sampling_rate(other.sampling_rate),
    num_channels(other.num_channels) {
  assert(IsValid());
}

bool CompressorConfig::IsValid() const {
  return (format_version == 1 &&
          truncation.IsValid() &&
          lpc.IsValid() &&
          chunk_size >= 16 &&  /* actually chunk_size should be way larger, but
                                * we allow this for test purposes.. */
          sampling_rate > 0 &&
          num_channels > 0);
}

void CompressorConfig::Write(IntStream *is) const {
  assert(IsValid());
  is->Write(format_version);
  truncation.Write(is);
  lpc.Write(is);
  is->Write(chunk_size);
  is->Write(sampling_rate);
  is->Write(num_channels);
}

bool CompressorConfig::Read(ReverseIntStream *ris) {
  return ris->Read(&format_version) &&
      truncation.Read(format_version, ris) &&
      lpc.Read(format_version, ris) &&
      ris->Read(&chunk_size) &&
      ris->Read(&sampling_rate) &&
      ris->Read(&num_channels) &&
      IsValid();
}


CompressedFile::CompressedFile():
    header_(NULL),
    compressed_chunk_sizes_(NULL) { }

CompressedFile::~CompressedFile() {
  for (size_t i = 0; i < chunks_.size(); i++) delete chunks_[i];
  delete header_;
  delete compressed_chunk_sizes_;
}



int CompressedChunk::Decompress(
    const CompressorConfig &config,
    int num_samples,
    int output_stride,
    int16_t *output_data) {

  ReverseLpcStream rls(config.truncation,
                       config.lpc,
                       data, end);
  for (int i = 0; i < num_samples; i++) {
    if (!rls.Read(output_data + i * output_stride))
      return i;
  }
  if (rls.NextCode() == end)
    return num_samples;  /* success */
  else
    return 0;  /* Something went wrong.  Assume the decoded data is not valid. */
}

CompressedChunk::CompressedChunk(
    const CompressorConfig &config,
    const int16_t *wave_data,
    int num_samples,
    int data_stride) {
  assert(num_samples > 0 && data_stride != 0);

  LpcStream ls(config.truncation, config.lpc);
  for (int s = 0; s < num_samples; s++) {
    ls.Write(wave_data[s * data_stride],
             NULL);  /* Don't need the approximated value */
  }
  owned_data_.swap(ls.Code());
  data = &(owned_data_[0]);
  end = data + owned_data_.size();
}


void CompressedFile::CompressHeader() {
  IntStream header_stream;
  config_.Write(&header_stream);
  header_stream.Write(num_complete_chunks_);
  header_stream.Write(partial_chunk_size_);
  /* The following constructor steals the memory from header_stream. */
  header_ = new CompressedChunk(&header_stream);
}

const char* CompressedFile::ReadHeader(const char *input,
                                       const char *input_end) {
  /* Read the meta-info. */
  ReverseIntStream ris(input, input_end);
  if (!config_.Read(&ris) ||
      !ris.Read(&num_complete_chunks_) || num_complete_chunks_ < 0 ||
      !ris.Read(&partial_chunk_size_) || partial_chunk_size_ < 0)
    return NULL;
  num_samples_ = num_complete_chunks_ * config_.chunk_size +
      partial_chunk_size_;
  if (!(num_samples_ > 0))
    return NULL;

  delete header_;  // In case we are re-using this object..
  header_ = new CompressedChunk(input, ris.NextCode());
  return ris.NextCode();
}


CompressedFile::CompressedFile(const CompressorConfig &config,
                               ssize_t num_samples,
                               const int16_t *data,
                               int sample_stride,
                               int channel_stride):
    config_(config),
    num_samples_(num_samples),
    header_(NULL),
    compressed_chunk_sizes_(NULL) {
  try {
    assert(num_samples > 0);

    int csize = config_.chunk_size;
    num_complete_chunks_ = num_samples / csize;
    partial_chunk_size_  = num_samples % csize;

    CompressHeader();

    // Compress each chunk
    for (int32_t chunk = 0; chunk < num_complete_chunks_; chunk++) {
      for (int32_t channel = 0; channel < config_.num_channels; channel++) {
        chunks_.push_back(new CompressedChunk(
            config_,
            data + (chunk * csize * sample_stride) + (channel * channel_stride),
            config_.chunk_size,
            sample_stride));
      }
    }
    if (partial_chunk_size_ != 0) {
      /* compress the final, partial chunk */
      for (int32_t channel = 0; channel < config_.num_channels; channel++) {
        chunks_.push_back(new CompressedChunk(
            config_,
            data + (num_complete_chunks_ * csize * sample_stride) + (channel * channel_stride),
            partial_chunk_size_,
            sample_stride));
      }
    }
    CompressChunkSizes();
  } catch (std::bad_alloc) {
    /* this code is the same as the destructor's code. */
    delete header_;
    delete compressed_chunk_sizes_;
    for (size_t i = 0; i < chunks_.size(); i++)
      delete chunks_[i];
    throw;  /* Re-throw the original exception. */
  }
}

bool CompressedFile::ReadData(
    ssize_t sample_start, ssize_t num_samples,
    int channel_start, int num_channels,
    int sample_stride,
    int channel_stride,
    int16_t *data) {
  assert(sample_start >= 0 && sample_start + num_samples <= num_samples_ &&
         num_samples > 0);
  assert(channel_start >= 0 && channel_start + num_channels <= config_.num_channels);
  assert(sample_stride != 0 && channel_stride != 0);

  ssize_t sample_end = sample_start + num_samples;

  /* `this_data` is currently where sample 0 of channel `channel_start`
     would be located. */
  int16_t *this_data = data - sample_start * sample_stride;
  for (int channel = channel_start; channel < num_channels;
       channel++, this_data += channel_stride) {
    int32_t cur_chunk = sample_start / config_.chunk_size;

    /* In this while loop we can treat the final, partial chunk just like other
       chunks, because we limit the samples extracted based on the
       num_samples. */
    for (; cur_chunk < num_complete_chunks_ + (partial_chunk_size_ != 0 ? 1 : 0);
         cur_chunk++) {
      int32_t index_into_chunks = cur_chunk * config_.num_channels + channel;
      if (index_into_chunks >= (int32_t)chunks_.size()) {
        std::cout << "A\n";
        return false;  /* e.g. could happen if the whole file was not present. */
      }
      CompressedChunk *cc = chunks_[index_into_chunks];

      ssize_t start_sample_index = cur_chunk * config_.chunk_size,
          end_sample_index = start_sample_index + config_.chunk_size;
      if (sample_end <= start_sample_index)
        break;

      /* `this_chunk_data` is constructed so that this_chunk_data[0] would
         correspond to the first sample in this chunk. */
      int16_t *this_chunk_data = this_data  + start_sample_index * sample_stride;

      /* start_in_chunk are the start/end positions in the chunk that we
         want to extract.  Normally they will equal respectively 0
         and config_.chunk_size. */
      int32_t start_in_chunk = (sample_start <= start_sample_index ?
                                0 : sample_start - start_sample_index),
          end_in_chunk = (sample_end >= end_sample_index ?
                          config_.chunk_size : sample_end - start_sample_index);
      assert(start_in_chunk >= 0 && end_in_chunk > start_in_chunk &&
             end_in_chunk <= config_.chunk_size);

      ReverseLpcStream rls(config_.truncation, config_.lpc,
                           cc->data, cc->end);
      int16_t value;
      int32_t i;
      for (i = 0; i < start_in_chunk; i++) {
        if (!rls.Read(&value)) {  /* discard these */
          std::cout << "B\n";
          return false;  /* failure */
        }
      }
      for (; i < end_in_chunk; i++) {
        if (!rls.Read(this_chunk_data + (i * sample_stride))) {
          std::cout << "C\n";
          return false;  /* failure */
        }
      }
      if (end_in_chunk == config_.chunk_size &&
          rls.NextCode() != cc->end) {
        std::cout << "D\n";
        return false;  /* junk at end of chunk, indicates corruption */
      }
    }
  }
  return true;
}

char* CompressedFile::Write(size_t *length) {
  size_t total_size = header_->size() + compressed_chunk_sizes_->size();
  for (size_t i = 0; i < chunks_.size(); i++)
    total_size += chunks_[i]->size();
  *length = total_size;
  assert(total_size > 0);
  char *ans = new char[total_size],
      *cur = ans;
  memcpy(cur, header_->data, header_->size());
  cur += header_->size();
  memcpy(cur, compressed_chunk_sizes_->data, compressed_chunk_sizes_->size());
  cur += compressed_chunk_sizes_->size();
  for (size_t i = 0; i < chunks_.size(); i++) {
    memcpy(cur, chunks_[i]->data, chunks_[i]->size());
    cur += chunks_[i]->size();
  }
  assert((size_t)(cur - ans) == total_size);
  return ans;
}


int CompressedFile::InitForReading(const char *input, const char *input_end) {
  const char *cur_input = ReadHeader(input, input_end);

  if (cur_input == NULL)
    return 1;  /* error */

  /* ReadChunkSizes() returns 0 on success, 1 on total failure,
     2 on partial failure (meaning: some chunks read). */
  return ReadChunkSizes(cur_input, input_end);
}


void CompressedFile::CompressChunkSizes() {
  assert(chunks_.size() ==
         (size_t)(config_.num_channels *
                  (num_complete_chunks_ + (partial_chunk_size_ > 0 ? 1 : 1))));
  IntStream is;
  int32_t prev_size = 0;
  for (size_t i = 0; i < chunks_.size(); i++) {
    int32_t this_size = chunks_[i]->size(),
        diff = this_size - prev_size;
    is.Write(diff);
    prev_size = this_size;
  }

  delete compressed_chunk_sizes_;  // In case we are re-using this object..
  // The constructor invoked below steals the memory from `is`.
  compressed_chunk_sizes_ = new CompressedChunk(&is);
}

int CompressedFile::ReadChunkSizes(const char *input, const char *input_end) {
  int32_t num_chunks =
      config_.num_channels * (num_complete_chunks_ + (partial_chunk_size_ > 0 ? 1 : 1));
  ReverseIntStream ris(input, input_end);
  int32_t prev_chunk_size = 0;
  std::vector<int32_t> chunk_sizes;
  chunk_sizes.reserve(num_chunks);
  for (int c = 0; c < num_chunks; c++) {
    int32_t delta_chunk_size;
    if (!ris.Read(&delta_chunk_size)) {
      std::cout << "ReadChunkSizes(): failure\n";
      return 1;  /* complete failure. */
    }
    /* Note: these are the sizes of the compressed chunks in bytes, which is
       why they are irregular. */
    int32_t this_chunk_size = prev_chunk_size + delta_chunk_size;
    if (!(this_chunk_size > 0)) {
      std::cout << "ReadChunkSizes(): failure [2]\n";
      return 1;  /* possibly corruption. */
    }
    chunk_sizes.push_back(this_chunk_size);
    prev_chunk_size = this_chunk_size;
  }
  const char *cur_data = ris.NextCode();
  for (int c = 0; c < num_chunks; c++) {
    int32_t this_size = chunk_sizes[c];
    const char *next_data = cur_data + this_size;
    if (next_data > input_end) {
      return (c == 0 ? 1 : 2); /* 1 is total failure, 2 is partial failure */
    }
    chunks_.push_back(new CompressedChunk(cur_data, next_data));
    cur_data = next_data;
  }
  return 0;  /* success */
}



