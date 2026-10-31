#pragma once

#include "breeze/model.h"

#include <string>
#include <vector>

namespace breeze {

struct MimiCodec {
    BreezeModel * m = nullptr;
    void init(BreezeModel & model) { m = &model; }

    // codes frame-major [f * n_cb + cb]; n_cb 0 means the full set. fewer codebooks drops detail,
    // 1 leaves only the semantic stage
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    // audio waveform in, returns codes frame-major, sets n_frames
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);
};

// decoder state between flushes, so a flush only decodes the frames it is about to send
struct VocoderStream {
    BreezeModel * m = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> slots;
    size_t next = 0;
    int kv_len = 0;
    bool planning = false;

    VocoderStream() = default;
    ~VocoderStream() { free(); }
    VocoderStream(const VocoderStream &) = delete;
    VocoderStream & operator=(const VocoderStream &) = delete;

    void init(BreezeModel & model);
    void reset();
    void free();
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    ggml_tensor * slot(int n0, int n1, int n2 = 1);
};

namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation);
ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & prefix, ggml_tensor * x);
ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x);
ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int seq_len);
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s,
                             const std::vector<int> & codes, int n_codebooks, int seq_len);

}

}
