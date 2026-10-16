#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace breeze {

// one compute backend (any GPU ggml was built with, else CPU) shared by all graphs
struct Backend {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool is_gpu = false;

    void init(bool prefer_gpu);
    void free();
    const char * name() const;
};

// persistent per-layer key/value cache living in its own backend buffer
// several CFG branches can share one cache, interleaved as slot = pos * n_branch + branch
struct KVCache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k, v;
    int head_dim = 0, n_kv_head = 0, max_seq = 0, len = 0, n_branch = 1;

    void init(Backend & be, int n_layer, int head_dim, int n_kv_head, int max_seq, int n_branch = 1);
    void reset() { len = 0; }
    void free();
};

// inputs are uploaded after allocation; prepared graphs own their scratch buffers
struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> pending;
    std::vector<ggml_tensor *> extra_roots;

    explicit Graph(size_t n_nodes = GGML_DEFAULT_GRAPH_SIZE);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    ggml_tensor * input_i32(const std::vector<int32_t> & data, int ne0, int ne1 = 1);
    ggml_tensor * input_f32(const std::vector<float> & data, int ne0, int ne1 = 1, int ne2 = 1);
    void write(ggml_tensor * node) { extra_roots.push_back(node); }

    void compute(Backend & be, ggml_tensor * out);
    void prepare(Backend & be, ggml_tensor * out);
    void replay(Backend & be);
};

// store cur [head_dim, n_kv_head, n] into cache at position pos; returns view over [0, pos+n)
ggml_tensor * cache_append(ggml_context * ctx, Graph & g, ggml_tensor * cache, ggml_tensor * cur, int pos);

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps);
ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x);
ggml_tensor * swiglu_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gate,
                         ggml_tensor * up, ggml_tensor * down);

// q,k,v laid out as [head_dim, n_head, n_tokens]; returns [head_dim*n_head, n_q]
ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        ggml_tensor * mask, float scale, int n_head, int n_kv_head);

std::vector<float> build_causal_mask(int n_q, int n_kv, int q_offset, int sliding_window);

// causal within a branch, blind across branches, for interleaved multi branch caches
std::vector<float> build_branch_causal_mask(int n_q, int n_kv, int q_offset, int n_branch);

// encoder style mask: everything visible, optionally limited to a window on both sides
std::vector<float> build_bidirectional_mask(int n_q, int n_kv, int sliding_window);

std::vector<float> tensor_to_f32(ggml_tensor * t);

}
