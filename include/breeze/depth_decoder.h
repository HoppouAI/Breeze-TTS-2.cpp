#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace breeze {

struct DepthStep {
    Graph graph{2048};
    ggml_tensor * audio = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * logits = nullptr;
};

// every codebook step and its sampling in one graph, so a frame is a single submit and readback
struct DepthFused {
    Graph graph{16384};
    ggml_tensor * cb0 = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * noise = nullptr;
    ggml_tensor * cfg = nullptr;
    ggml_tensor * inv_temp = nullptr;
    ggml_tensor * codes = nullptr;
    int top_k = 0;
};

// one stream's share of a batched frame, branches of a cfg stream are cond then uncond
struct DepthItem {
    const std::vector<float> * hidden[2] = { nullptr, nullptr };
    int cb0 = 0;
    float cfg_scale = 1.0f;
    float temperature = 1.0f;
    std::mt19937 * rng = nullptr;
};

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;
    std::vector<std::unique_ptr<DepthStep>> steps;
    std::map<std::pair<int, int>, std::unique_ptr<DepthFused>> fused; // by streams and top k

    ~DepthRunner() { free(); }
    void init(BreezeModel & m, int n_branches);
    void free();

    // cond hidden first; force replaces the first n_force residual codebooks
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng,
                         const SampleParams * sp = nullptr,
                         const int * force = nullptr, int n_force = 0);

    // the fused graph over several streams at once, top_k has to be one the fused graph can take
    std::vector<std::vector<int>> run_batch(BreezeModel & m, const std::vector<DepthItem> & items, int top_k);

    static SampleParams default_params(const BreezeModel & m);

    // whether these sampling settings can use the fused graph
    static bool can_fuse(const BreezeModel & m, const SampleParams & sp);
};

}
