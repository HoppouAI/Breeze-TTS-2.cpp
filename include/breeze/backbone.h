#pragma once

#include "breeze/model.h"

#include <array>
#include <vector>

namespace breeze {

struct BackboneState {
    KVCache kv;
    int pos = 0;
    BackboneState() = default;
    ~BackboneState() { free(); }
    BackboneState(const BackboneState &) = delete;
    BackboneState & operator=(const BackboneState &) = delete;
    void init(BreezeModel & m, int max_seq);
    void reset() { kv.reset(); pos = 0; }
    void free() { kv.free(); }
};

struct StepOut {
    std::vector<float> hidden; // [hidden_size]
    std::vector<float> logits; // [audio_vocab_size + 1]
};

// sum of the 16 codebook embeddings per frame; codes laid out frame-major [f*16 + cb]
std::vector<float> audio_embed_forward(BreezeModel & m, const std::vector<int> & codes, int n_frames);

// run a chunk of inputs_embeds through the backbone, appending to the kv cache
StepOut backbone_run(BreezeModel & m, BackboneState & st, const std::vector<float> & embeds, int n_tokens);

// one audio frame shared by two CFG branches, each with its own cache and position
std::array<StepOut, 2> backbone_run_cfg(BreezeModel & m, BackboneState & cond, BackboneState & uncond,
                                       const std::vector<float> & embed);

}
