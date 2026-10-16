#include "breeze/common.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace breeze {

void Backend::init(bool prefer_gpu) {
    if (prefer_gpu) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        is_gpu = backend != nullptr;
    }
    if (!backend) {
        backend = ggml_backend_cpu_init();
        is_gpu = false;
    }
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
}

const char * Backend::name() const {
    return backend ? ggml_backend_name(backend) : "none";
}

void Backend::free() {
    if (alloc) ggml_gallocr_free(alloc);
    if (backend) ggml_backend_free(backend);
    alloc = nullptr;
    backend = nullptr;
}

void KVCache::init(Backend & be, int n_layer, int hd, int nkv, int ms, int nb) {
    head_dim = hd;
    n_kv_head = nkv;
    max_seq = ms;
    n_branch = nb;
    len = 0;
    ggml_init_params p{ ggml_tensor_overhead() * n_layer * 2 + 4096, nullptr, true };
    ctx = ggml_init(p);
    k.resize(n_layer);
    v.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        k[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkv, ms * nb);
        v[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nkv, ms * nb);
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, be.backend);
}

void KVCache::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    buffer = nullptr;
    ctx = nullptr;
}

Graph::Graph(size_t n_nodes) {
    size_t mem = ggml_tensor_overhead() * n_nodes * 2 + ggml_graph_overhead_custom(n_nodes, false) + 8192;
    ggml_init_params p{ mem, nullptr, true };
    ctx = ggml_init(p);
    gf = ggml_new_graph_custom(ctx, n_nodes, false);
}

Graph::~Graph() {
    if (alloc) ggml_gallocr_free(alloc);
    if (ctx) ggml_free(ctx);
}

ggml_tensor * Graph::input_i32(const std::vector<int32_t> & data, int ne0, int ne1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne0, ne1);
    ggml_set_input(t);
    std::vector<uint8_t> bytes(data.size() * sizeof(int32_t));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    pending.emplace_back(t, std::move(bytes));
    return t;
}

ggml_tensor * Graph::input_f32(const std::vector<float> & data, int ne0, int ne1, int ne2) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_input(t);
    std::vector<uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());
    pending.emplace_back(t, std::move(bytes));
    return t;
}

void Graph::compute(Backend & be, ggml_tensor * out) {
    ggml_set_output(out);
    for (ggml_tensor * r : extra_roots) ggml_build_forward_expand(gf, r);
    ggml_build_forward_expand(gf, out);
    if (!ggml_gallocr_alloc_graph(be.alloc, gf)) {
        ggml_gallocr_free(be.alloc);
        be.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.backend));
        throw std::runtime_error("could not allocate inference graph");
    }
    for (auto & pr : pending) {
        ggml_backend_tensor_set(pr.first, pr.second.data(), 0, pr.second.size());
    }
    replay(be);
}

void Graph::prepare(Backend & be, ggml_tensor * out) {
    if (alloc) throw std::logic_error("inference graph is already prepared");
    ggml_set_output(out);
    // constant inputs must survive in-place operations between replays
    for (auto & pr : pending) ggml_set_output(pr.first);
    for (ggml_tensor * r : extra_roots) ggml_build_forward_expand(gf, r);
    ggml_build_forward_expand(gf, out);
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) throw std::runtime_error("could not allocate cached inference graph");
    for (auto & pr : pending)
        ggml_backend_tensor_set(pr.first, pr.second.data(), 0, pr.second.size());
    pending.clear();
}

void Graph::replay(Backend & be) {
    if (ggml_backend_graph_compute(be.backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("inference graph execution failed");
}

ggml_tensor * cache_append(ggml_context * ctx, Graph & g, ggml_tensor * cache, ggml_tensor * cur, int pos) {
    const int hd = (int) cache->ne[0];
    const int nkv = (int) cache->ne[1];
    const int n = (int) cur->ne[2];
    ggml_tensor * dst = ggml_view_3d(ctx, cache, hd, nkv, n, cache->nb[1], cache->nb[2], (size_t) pos * cache->nb[2]);
    g.write(ggml_cpy(ctx, cur, dst));
    return ggml_view_3d(ctx, cache, hd, nkv, pos + n, cache->nb[1], cache->nb[2], 0);
}

ggml_tensor * rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_mul(ctx, n, w), b);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    return ggml_mul_mat(ctx, w, x);
}

ggml_tensor * swiglu_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gate,
                         ggml_tensor * up, ggml_tensor * down) {
    ggml_tensor * g = ggml_mul_mat(ctx, gate, x);
    ggml_tensor * u = ggml_mul_mat(ctx, up, x);
    return ggml_mul_mat(ctx, down, ggml_swiglu_split(ctx, g, u));
}

ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        ggml_tensor * mask, float scale, int n_head, int n_kv_head) {
    (void) n_kv_head;
    const int hd = (int) q->ne[0];
    const int nq = (int) q->ne[2];
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, kqv, hd * n_head, nq);
}

std::vector<float> build_causal_mask(int n_q, int n_kv, int q_offset, int sliding_window) {
    std::vector<float> mask((size_t) n_q * n_kv, 0.0f);
    for (int q = 0; q < n_q; q++) {
        const int qpos = q_offset + q;
        for (int kk = 0; kk < n_kv; kk++) {
            bool ok = kk <= qpos;
            if (ok && sliding_window > 0 && qpos - kk >= sliding_window) ok = false;
            mask[(size_t) q * n_kv + kk] = ok ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

std::vector<float> build_branch_causal_mask(int n_q, int n_kv, int q_offset, int n_branch) {
    std::vector<float> mask((size_t) n_q * n_kv, 0.0f);
    for (int q = 0; q < n_q; q++) {
        const int qb = q % n_branch;
        const int qpos = q_offset + q / n_branch;
        for (int kk = 0; kk < n_kv; kk++) {
            const bool ok = kk % n_branch == qb && kk / n_branch <= qpos;
            mask[(size_t) q * n_kv + kk] = ok ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

std::vector<float> build_bidirectional_mask(int n_q, int n_kv, int sliding_window) {
    std::vector<float> mask((size_t) n_q * n_kv, 0.0f);
    if (sliding_window <= 0) return mask;
    const int left = (sliding_window + 1) / 2;
    const int right = sliding_window / 2 + 1;
    for (int q = 0; q < n_q; q++) {
        for (int kk = 0; kk < n_kv; kk++) {
            const int dist = q - kk;
            const bool ok = (dist >= 0 && dist < left) || (dist < 0 && -dist < right);
            mask[(size_t) q * n_kv + kk] = ok ? 0.0f : -INFINITY;
        }
    }
    return mask;
}

std::vector<float> tensor_to_f32(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    return out;
}

}
