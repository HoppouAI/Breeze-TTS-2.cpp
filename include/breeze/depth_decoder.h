#pragma once

#include "breeze/model.h"
#include "breeze/sampling.h"

#include <memory>
#include <random>
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

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;
    std::vector<std::unique_ptr<DepthStep>> steps;
    std::unique_ptr<DepthFused> fused;

    ~DepthRunner() { free(); }
    void init(BreezeModel & m, int n_branches);
    void free();

    // cond hidden first; force replaces the first n_force residual codebooks
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng,
                         const SampleParams * sp = nullptr,
                         const int * force = nullptr, int n_force = 0);

private:
    std::vector<int> run_fused(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                               int cb0, float cfg_scale, std::mt19937 & rng, const SampleParams & sp);
};

}
