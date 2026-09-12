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

// autoregressive residual decoder: predicts codebooks 1..num_codebooks-1 for one frame
struct DepthRunner {
    KVCache kv; // CFG branches share one cache, interleaved per position
    int n_branch = 1;
    std::vector<float> freq_factors;
    std::vector<std::unique_ptr<DepthStep>> steps;

    ~DepthRunner() { free(); }
    void init(BreezeModel & m, int n_branches);
    void free();

    // cond hidden first; force replaces the first n_force residual codebooks
    std::vector<int> run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                         int cb0, float cfg_scale, std::mt19937 & rng,
                         const SampleParams * sp = nullptr,
                         const int * force = nullptr, int n_force = 0);
};

}
