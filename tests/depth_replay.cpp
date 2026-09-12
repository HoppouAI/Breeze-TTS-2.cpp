#include "breeze/depth_decoder.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace breeze;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static void close(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "output shape changed");
    for (size_t i = 0; i < a.size(); i++)
        require(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                std::abs(a[i] - b[i]) <= 1e-5f * (1.0f + std::abs(b[i])),
                "output differs at " + std::to_string(i));
}

static void scratch_graph(Backend & be) {
    Graph g(64);
    auto * x = g.input_f32(std::vector<float>(65536, 3.0f), 65536);
    auto * out = ggml_sqr(g.ctx, x);
    g.compute(be, out);
    require(tensor_to_f32(out).front() == 9.0f, "temporary graph failed");
}

static void test_graph(Backend & be) {
    Graph g(64);
    auto * x = g.input_f32(std::vector<float>(32, 0.0f), 32);
    auto * constant = g.input_f32(std::vector<float>(32, 0.5f), 32);
    auto * out = ggml_add(g.ctx, ggml_silu(g.ctx, x), ggml_sqr(g.ctx, constant));
    g.prepare(be, out);
    for (int run = 0; run < 8; run++) {
        scratch_graph(be);
        std::vector<float> input(32), expected(32);
        for (int i = 0; i < 32; i++) {
            input[i] = (i - 16 + run) * 0.1f;
            expected[i] = input[i] / (1.0f + std::exp(-input[i])) + 0.25f;
        }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        g.replay(be);
        close(tensor_to_f32(out), expected);
    }
    bool rejected = false;
    try { g.prepare(be, out); }
    catch (const std::logic_error &) { rejected = true; }
    require(rejected, "preparing the same graph twice was accepted");
}

struct Fixture {
    BreezeModel model;
    ~Fixture() { model.free(); }

    void init(bool gpu) {
        auto & m = model;
        m.backend.init(gpu);
        if (gpu && !m.backend.is_gpu) return;
        m.cfg.hidden_size = 128;
        m.cfg.audio_vocab_size = 32;
        m.cfg.dd.hidden = 64;
        m.cfg.dd.n_layer = 2;
        m.cfg.dd.n_head = 2;
        m.cfg.dd.n_kv_head = 1;
        m.cfg.dd.head_dim = 32;
        m.cfg.dd.ffn = 128;
        ggml_init_params params{ggml_tensor_overhead() * 64, nullptr, true};
        m.gg.meta = ggml_init(params);
        require(m.gg.meta != nullptr, "weight context allocation failed");
        std::vector<ggml_tensor *> weights;
        auto add = [&](const std::string & name, int n0, int n1 = 1, int n2 = 1) {
            auto * t = ggml_new_tensor_3d(m.gg.meta, GGML_TYPE_F32, n0, n1, n2);
            ggml_set_name(t, name.c_str());
            m.gg.tensors.emplace(name, t);
            weights.push_back(t);
        };
        add("audio_embd.weight", 128, 32 * m.cfg.num_codebooks);
        add("dd.in_proj.weight", 128, 64);
        add("dd.codebooks_head.weight", 64, 32, m.cfg.num_codebooks - 1);
        add("dd.output_norm.weight", 64);
        for (int il = 0; il < 2; il++) {
            const std::string p = "dd.blk." + std::to_string(il);
            add(p + ".attn_norm.weight", 64);
            add(p + ".attn_q.weight", 64, 64);
            add(p + ".attn_k.weight", 64, 32);
            add(p + ".attn_v.weight", 64, 32);
            add(p + ".attn_output.weight", 64, 64);
            add(p + ".ffn_norm.weight", 64);
            add(p + ".ffn_gate.weight", 64, 128);
            add(p + ".ffn_up.weight", 64, 128);
            add(p + ".ffn_down.weight", 128, 64);
        }
        m.gg.buffer = ggml_backend_alloc_ctx_tensors(m.gg.meta, m.backend.backend);
        require(m.gg.buffer != nullptr, "weight buffer allocation failed");
        std::mt19937 rng(17);
        for (auto * t : weights) {
            std::vector<float> data(ggml_nelements(t));
            const bool norm = std::string(t->name).find("norm.weight") != std::string::npos;
            for (float & v : data)
                v = norm ? 1.0f : (static_cast<int>(rng() % 2001) - 1000) * 0.0001f;
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        }
    }
};

static void test_depth(BreezeModel & m) {
    DepthRunner cached;
    std::mt19937 inputs(91);
    for (int nb : {1, 2, 1}) {
        cached.init(m, nb);
        for (int frame = 0; frame < 4; frame++) {
            const int nc = m.cfg.num_codebooks;
            std::vector<std::vector<float>> hidden(nb, std::vector<float>(m.cfg.hidden_size));
            for (auto & branch : hidden)
                for (auto & x : branch) x = (static_cast<int>(inputs() % 201) - 100) * 0.01f;
            std::vector<int> forced(nc - 1);
            for (int & code : forced) code = inputs() % m.cfg.audio_vocab_size;
            const int n_force = frame == 3 ? nc - 1 : frame;
            const int cb0 = inputs() % m.cfg.audio_vocab_size;
            std::mt19937 rng_cached(frame), rng_fresh(frame);
            const auto actual = cached.run(m, hidden, cb0, 6.0f, rng_cached, nullptr, forced.data(), n_force);
            std::vector<std::vector<float>> logits;
            for (const auto & step : cached.steps) logits.push_back(tensor_to_f32(step->logits));
            scratch_graph(m.backend);
            DepthRunner fresh;
            fresh.init(m, nb);
            const auto expected = fresh.run(m, hidden, cb0, 6.0f, rng_fresh, nullptr, forced.data(), n_force);
            require(actual == expected, "cached codebooks differ from fresh graphs");
            require(rng_cached == rng_fresh, "sampling consumed a different RNG sequence");
            for (int j = 0; j < nc - 1; j++) {
                close(logits[j], tensor_to_f32(fresh.steps[j]->logits));
                if (j < n_force) require(actual[j] == forced[j], "forced codebook was ignored");
            }
        }
    }
    cached.free();
    cached.free();
    require(cached.steps.empty() && !cached.kv.buffer, "depth cache was not released");
}

int main(int argc, char ** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        Fixture fixture;
        fixture.init(gpu);
        if (gpu && !fixture.model.backend.is_gpu) {
            std::puts("no GPU backend available");
            return 77;
        }
        std::printf("testing %s\n", fixture.model.backend.name());
        test_graph(fixture.model.backend);
        test_depth(fixture.model);
        std::puts("graph replay and depth cache tests passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}