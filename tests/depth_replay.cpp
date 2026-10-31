#include "breeze/depth_decoder.h"
#include "breeze/backbone.h"

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

    void init(bool gpu, bool quantized) {
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
        m.cfg.bb.hidden = 128;
        m.cfg.bb.n_layer = 2;
        m.cfg.bb.n_head = 4;
        m.cfg.bb.n_kv_head = 2;
        m.cfg.bb.head_dim = 32;
        m.cfg.bb.ffn = 128;
        ggml_init_params params{ggml_tensor_overhead() * 64, nullptr, true};
        m.gg.meta = ggml_init(params);
        require(m.gg.meta != nullptr, "weight context allocation failed");
        std::vector<ggml_tensor *> weights;
        auto add = [&](const std::string & name, int n0, int n1 = 1, int n2 = 1) {
            const auto type = quantized && n1 > 1 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
            auto * t = ggml_new_tensor_3d(m.gg.meta, type, n0, n1, n2);
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
        add("bb.output_norm.weight", 128);
        add("bb.lm_head.weight", 128, 33);
        for (int il = 0; il < 2; il++) {
            const std::string p = "bb.blk." + std::to_string(il);
            add(p + ".attn_norm.weight", 128);
            add(p + ".attn_q.weight", 128, 128);
            add(p + ".attn_k.weight", 128, 64);
            add(p + ".attn_v.weight", 128, 64);
            add(p + ".attn_q_norm.weight", 32);
            add(p + ".attn_k_norm.weight", 32);
            add(p + ".attn_output.weight", 128, 128);
            add(p + ".ffn_norm.weight", 128);
            add(p + ".ffn_gate.weight", 128, 128);
            add(p + ".ffn_up.weight", 128, 128);
            add(p + ".ffn_down.weight", 128, 128);
        }
        m.gg.buffer = ggml_backend_alloc_ctx_tensors(m.gg.meta, m.backend.backend);
        require(m.gg.buffer != nullptr, "weight buffer allocation failed");
        std::mt19937 rng(17);
        for (auto * t : weights) {
            std::vector<float> data(ggml_nelements(t));
            const bool norm = std::string(t->name).find("norm.weight") != std::string::npos;
            for (float & v : data)
                v = norm ? 1.0f : (static_cast<int>(rng() % 2001) - 1000) * 0.0001f;
            if (ggml_is_quantized(t->type)) {
                std::vector<uint8_t> packed(ggml_nbytes(t));
                ggml_quantize_chunk(t->type, data.data(), packed.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
                ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
            } else {
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
            }
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

static void test_fused(BreezeModel & m) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    std::mt19937 inputs(7);
    for (int nb : {1, 2}) {
        DepthRunner fused, stepped;
        fused.init(m, nb);
        stepped.init(m, nb);
        for (int frame = 0; frame < 4; frame++) {
            std::vector<std::vector<float>> hidden(nb, std::vector<float>(m.cfg.hidden_size));
            for (auto & branch : hidden)
                for (auto & x : branch) x = (static_cast<int>(inputs() % 201) - 100) * 0.01f;
            const int cb0 = inputs() % vs;
            for (int k : {1, 4}) {
                SampleParams sp;
                sp.top_k = k;
                std::mt19937 rng(frame);
                const auto codes = fused.run(m, hidden, cb0, 3.0f, rng, &sp);
                require(!fused.fused.empty() && fused.steps[0] == nullptr, "fused path was not taken");
                stepped.run(m, hidden, cb0, 3.0f, rng, &sp, codes.data(), nc - 1);
                for (int j = 0; j < nc - 1; j++) {
                    const auto out = tensor_to_f32(stepped.steps[j]->logits);
                    std::vector<float> l(out.begin(), out.begin() + vs);
                    if (nb > 1)
                        for (int i = 0; i < vs; i++) l[i] = out[vs + i] + 3.0f * (out[i] - out[vs + i]);
                    int above = 0;
                    for (int i = 0; i < vs; i++) above += l[i] > l[codes[j]];
                    require(above < k, "fused code " + std::to_string(j) + " is outside the top " + std::to_string(k));
                }
            }
        }
    }
}

// a stream has to come out the same whoever else shares its batch
static void test_batched_depth(BreezeModel & m) {
    const int vs = m.cfg.audio_vocab_size;
    std::mt19937 inputs(11);
    int checked = 0, same = 0;
    for (int nb : {1, 2}) {
        DepthRunner solo, batched;
        solo.init(m, nb);
        batched.init(m, nb);
        for (int streams : {2, 3}) {
            std::vector<std::vector<std::vector<float>>> hidden(streams);
            std::vector<DepthItem> items(streams);
            std::vector<std::mt19937> a, b;
            for (int s = 0; s < streams; s++) {
                hidden[s].assign(nb, std::vector<float>(m.cfg.hidden_size));
                for (auto & branch : hidden[s])
                    for (auto & x : branch) x = (static_cast<int>(inputs() % 201) - 100) * 0.01f;
                a.emplace_back(100 + s);
                b.emplace_back(100 + s);
            }
            for (int s = 0; s < streams; s++) {
                for (int branch = 0; branch < nb; branch++) items[s].hidden[branch] = &hidden[s][branch];
                items[s].cb0 = inputs() % vs;
                items[s].cfg_scale = 2.0f + s;
                items[s].temperature = 1.0f;
            }
            for (int k : {1, 4}) {
                for (int s = 0; s < streams; s++) items[s].rng = &a[s];
                const auto together = batched.run_batch(m, items, k);
                for (int s = 0; s < streams; s++) {
                    items[s].rng = &b[s];
                    const auto alone = solo.run_batch(m, { items[s] }, k);
                    checked++;
                    if (together[s] == alone[0]) same++;
                }
            }
        }
    }
    // plain and cfg streams in one graph, each coming out the same as on its own
    {
        DepthRunner solo, mixed;
        solo.init(m, 1);
        mixed.init(m, 1);
        std::vector<std::vector<float>> hidden(5, std::vector<float>(m.cfg.hidden_size));
        for (auto & h : hidden)
            for (auto & x : h) x = (static_cast<int>(inputs() % 201) - 100) * 0.01f;
        // deliberately out of order, the batch has to hand results back the way they came in
        std::vector<DepthItem> items(3);
        items[0].hidden[0] = &hidden[0];
        items[0].hidden[1] = &hidden[1];
        items[0].cfg_scale = 3.0f;
        items[1].hidden[0] = &hidden[2];
        items[2].hidden[0] = &hidden[3];
        items[2].hidden[1] = &hidden[4];
        items[2].cfg_scale = 2.0f;
        for (int s = 0; s < 3; s++) {
            items[s].cb0 = inputs() % vs;
            items[s].temperature = 1.0f;
        }
        std::vector<std::mt19937> a, b;
        for (int s = 0; s < 3; s++) {
            a.emplace_back(200 + s);
            b.emplace_back(200 + s);
        }
        for (int s = 0; s < 3; s++) items[s].rng = &a[s];
        const auto together = mixed.run_batch(m, items, 4);
        for (int s = 0; s < 3; s++) {
            items[s].rng = &b[s];
            checked++;
            if (together[s] == solo.run_batch(m, { items[s] }, 4)[0]) same++;
        }
    }
    // gpu rounding can tip a near tie on logits this small, the cpu has no such excuse
    require(same == checked || (m.backend.is_gpu && same * 3 >= checked * 2),
            std::to_string(checked - same) + " of " + std::to_string(checked) + " streams came out different in a batch");
}

static void test_batched_backbone(BreezeModel & m) {
    std::mt19937 rng(321);
    auto embeds = [&](int n) {
        std::vector<float> values(n * m.cfg.hidden_size);
        for (auto & x : values) x = (static_cast<int>(rng() % 201) - 100) * 0.01f;
        return values;
    };
    const int lengths[3] = {3, 7, 5};
    BackboneState batched[3], separate[3];
    for (int r = 0; r < 3; r++) {
        batched[r].init(m, 32);
        separate[r].init(m, 32);
        const auto prompt = embeds(lengths[r]);
        backbone_run(m, batched[r], prompt, lengths[r]);
        backbone_run(m, separate[r], prompt, lengths[r]);
    }
    for (int step = 0; step < 4; step++) {
        const auto input = embeds(3);
        const auto actual = backbone_run_batch(m, { &batched[0], &batched[1], &batched[2] }, input);
        for (int r = 0; r < 3; r++) {
            const std::vector<float> one(input.begin() + r * m.cfg.hidden_size, input.begin() + (r + 1) * m.cfg.hidden_size);
            const auto expected = backbone_run(m, separate[r], one, 1);
            close(actual[r].hidden, expected.hidden);
            close(actual[r].logits, expected.logits);
            require(batched[r].pos == separate[r].pos, "batched position differs");
        }
    }
    for (int r = 0; r < 3; r++) { batched[r].free(); separate[r].free(); }
}

static void test_backbone(BreezeModel & m) {
    std::mt19937 rng(123);
    auto embeds = [&](int n) {
        std::vector<float> values(n * m.cfg.hidden_size);
        for (auto & x : values) x = (static_cast<int>(rng() % 201) - 100) * 0.01f;
        return values;
    };
    for (const auto & lengths : {std::array<int, 2>{3, 7}, {7, 3}, {4, 4}}) {
        BackboneState paired[2], separate[2];
        for (int b = 0; b < 2; b++) {
            paired[b].init(m, 32);
            separate[b].init(m, 32);
            const auto prompt = embeds(lengths[b]);
            backbone_run(m, paired[b], prompt, lengths[b]);
            backbone_run(m, separate[b], prompt, lengths[b]);
        }
        for (int step = 0; step < 4; step++) {
            const auto input = embeds(1);
            const auto actual = backbone_run_cfg(m, paired[0], paired[1], input);
            for (int b = 0; b < 2; b++) {
                const auto expected = backbone_run(m, separate[b], input, 1);
                close(actual[b].hidden, expected.hidden);
                close(actual[b].logits, expected.logits);
                require(paired[b].pos == separate[b].pos, "CFG position differs");
                for (int il = 0; il < m.cfg.bb.n_layer; il++) {
                    const size_t count = paired[b].pos * m.cfg.bb.head_dim * m.cfg.bb.n_kv_head;
                    for (int value = 0; value < 2; value++) {
                        auto * a = value ? paired[b].kv.v[il] : paired[b].kv.k[il];
                        auto * e = value ? separate[b].kv.v[il] : separate[b].kv.k[il];
                        std::vector<float> av(count), ev(count);
                        ggml_backend_tensor_get(a, av.data(), 0, count * sizeof(float));
                        ggml_backend_tensor_get(e, ev.data(), 0, count * sizeof(float));
                        close(av, ev);
                    }
                }
            }
        }
        for (int b = 0; b < 2; b++) { paired[b].free(); separate[b].free(); }
    }
}

static void test_swiglu(BreezeModel & m) {
    auto * gate = m.w("dd.blk.0.ffn_gate.weight");
    auto * up = m.w("dd.blk.0.ffn_up.weight");
    auto * down = m.w("dd.blk.0.ffn_down.weight");
    for (int n : {1, 2, 4}) {
        std::vector<float> input(m.cfg.dd.hidden * n);
        for (size_t i = 0; i < input.size(); i++) input[i] = (static_cast<int>(i % 127) - 63) * 0.125f;
        Graph fused(128), separate(128);
        auto * x = fused.input_f32(input, m.cfg.dd.hidden, n);
        auto * actual = swiglu_ffn(fused.ctx, x, gate, up, down);
        fused.compute(m.backend, actual);
        const auto values = tensor_to_f32(actual);
        x = separate.input_f32(input, m.cfg.dd.hidden, n);
        auto * g = ggml_silu(separate.ctx, ggml_mul_mat(separate.ctx, gate, x));
        auto * u = ggml_mul_mat(separate.ctx, up, x);
        auto * expected = ggml_mul_mat(separate.ctx, down, ggml_mul(separate.ctx, g, u));
        separate.compute(m.backend, expected);
        close(values, tensor_to_f32(expected));
    }
}

int main(int argc, char ** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        for (bool quantized : {false, true}) {
            Fixture fixture;
            fixture.init(gpu, quantized);
            if (gpu && !fixture.model.backend.is_gpu) {
                std::puts("no GPU backend available");
                return 77;
            }
            std::printf("testing %s with %s weights\n", fixture.model.backend.name(), quantized ? "Q8_0" : "F32");
            test_graph(fixture.model.backend);
            test_depth(fixture.model);
            test_fused(fixture.model);
            test_batched_depth(fixture.model);
            test_batched_backbone(fixture.model);
            test_backbone(fixture.model);
            test_swiglu(fixture.model);
        }
        std::puts("graph replay, depth cache, CFG backbone and SwiGLU tests passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}