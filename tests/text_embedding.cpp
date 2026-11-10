#include "breeze/text_encoder.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace breeze;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fixture {
    BreezeModel model;
    ggml_tensor * gpu_embedding = nullptr;
    ggml_tensor * cpu_embedding = nullptr;

    ~Fixture() { model.free(); }

    bool init() {
        auto & m = model;
        m.backend.init(true);
        if (!m.backend.is_gpu) return false;
        m.cpu.init(false);
        ggml_backend_cpu_set_n_threads(m.cpu.backend, 1);

        auto & c = m.cfg.te;
        c.hidden = 32;
        c.n_layer = 2;
        c.n_head = 4;
        c.n_kv_head = 4;
        c.head_dim = 8;
        c.ffn = 64;
        c.vocab = 23;
        c.sliding_window = 3;
        c.query_pre_attn_scalar = 64.0f;
        c.layer_is_full = { 1, 0 };

        ggml_init_params params{ggml_tensor_overhead() * 48, nullptr, true};
        m.gg.meta = ggml_init(params);
        require(m.gg.meta != nullptr, "weight context allocation failed");
        std::vector<ggml_tensor *> weights;
        const auto add_vector = [&](const std::string & name, int n) {
            auto * t = ggml_new_tensor_1d(m.gg.meta, GGML_TYPE_F32, n);
            ggml_set_name(t, name.c_str());
            m.gg.tensors.emplace(name, t);
            weights.push_back(t);
            return t;
        };
        const auto add_matrix = [&](const std::string & name, int n0, int n1) {
            const auto type = name == "te.token_embd.weight" ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
            auto * t = ggml_new_tensor_2d(m.gg.meta, type, n0, n1);
            ggml_set_name(t, name.c_str());
            m.gg.tensors.emplace(name, t);
            weights.push_back(t);
            return t;
        };

        gpu_embedding = add_matrix("te.token_embd.weight", c.hidden, c.vocab);
        for (int il = 0; il < c.n_layer; il++) {
            const std::string p = "te.blk." + std::to_string(il);
            add_vector(p + ".attn_norm.weight", c.hidden);
            add_matrix(p + ".attn_q.weight", c.hidden, c.head_dim * c.n_head);
            add_matrix(p + ".attn_k.weight", c.hidden, c.head_dim * c.n_kv_head);
            add_matrix(p + ".attn_v.weight", c.hidden, c.head_dim * c.n_kv_head);
            add_vector(p + ".attn_q_norm.weight", c.head_dim);
            add_vector(p + ".attn_k_norm.weight", c.head_dim);
            add_matrix(p + ".attn_output.weight", c.head_dim * c.n_head, c.hidden);
            add_vector(p + ".post_attn_norm.weight", c.hidden);
            add_vector(p + ".ffn_norm.weight", c.hidden);
            add_matrix(p + ".ffn_gate.weight", c.hidden, c.ffn);
            add_matrix(p + ".ffn_up.weight", c.hidden, c.ffn);
            add_matrix(p + ".ffn_down.weight", c.ffn, c.hidden);
            add_vector(p + ".post_ffn_norm.weight", c.hidden);
        }
        add_vector("te.output_norm.weight", c.hidden);
        add_matrix("te.proj.weight", c.hidden, c.hidden);

        m.gg.buffer = ggml_backend_alloc_ctx_tensors(m.gg.meta, m.backend.backend);
        require(m.gg.buffer != nullptr, "GPU weight buffer allocation failed");
        std::mt19937 rng(17);
        for (ggml_tensor * t : weights) {
            std::vector<float> data(ggml_nelements(t));
            const bool norm = std::string(t->name).find("norm.weight") != std::string::npos;
            for (float & x : data)
                x = norm ? 1.0f : (static_cast<int>(rng() % 2001) - 1000) * 0.0005f;
            if (ggml_is_quantized(t->type)) {
                std::vector<uint8_t> packed(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, data.data(), packed.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
            } else {
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
            }
        }

        ggml_init_params cpu_params{ggml_tensor_overhead() * 2, nullptr, true};
        m.gg.meta_cpu = ggml_init(cpu_params);
        require(m.gg.meta_cpu != nullptr, "CPU embedding context allocation failed");
        cpu_embedding = ggml_dup_tensor(m.gg.meta_cpu, gpu_embedding);
        ggml_set_name(cpu_embedding, "te.token_embd.weight");
        m.gg.buffer_cpu = ggml_backend_alloc_ctx_tensors(m.gg.meta_cpu, m.cpu.backend);
        require(m.gg.buffer_cpu != nullptr, "CPU embedding buffer allocation failed");
        std::vector<uint8_t> bytes(ggml_nbytes(gpu_embedding));
        ggml_backend_tensor_get(gpu_embedding, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(cpu_embedding, bytes.data(), 0, bytes.size());
        return true;
    }
};

int main() {
    try {
        Fixture fixture;
        if (!fixture.init()) {
            std::puts("no GPU backend available");
            return 77;
        }

        auto & m = fixture.model;
        const std::vector<int> tokens = { 2, 3, 7, 5, 1 };
        const std::vector<float> gpu_output = text_encoder_forward(m, tokens);
        m.gg.tensors["te.token_embd.weight"] = fixture.cpu_embedding;
        m.te_embedding_cpu = true;
        const std::vector<float> cpu_embedding_output = text_encoder_forward(m, tokens);
        require(!gpu_output.empty(), "text encoder returned no output");
        require(gpu_output.size() == cpu_embedding_output.size(), "embedding lookup changed output shape");
        require(std::memcmp(gpu_output.data(), cpu_embedding_output.data(), gpu_output.size() * sizeof(float)) == 0,
                "CPU embedding lookup changed the GPU text-encoder output");
        std::puts("CPU embedding lookup matches the GPU path bit for bit");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
