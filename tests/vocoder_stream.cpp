#include "breeze/codec.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>

using namespace breeze;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

// tiny random weights with the real layer layout, enough to exercise every piece of state
struct Fixture {
    BreezeModel model;
    ~Fixture() { model.free(); }

    void init(bool gpu) {
        auto & m = model;
        m.backend.init(gpu);
        if (gpu && !m.backend.is_gpu) return;
        auto & v = m.cfg.voc;
        m.cfg.num_codebooks = 4;
        v.hidden = 16;
        v.n_layer = 2;
        v.n_head = 2;
        v.n_kv_head = 2;
        v.head_dim = 8;
        v.ffn = 24;
        v.sliding_window = 6;
        v.upsampling_ratios = { 2, 2 };
        v.upsample_rates = { 3, 2 };
        const int book = 8, dim = 6, latent = 10, dec = 12;

        ggml_init_params params{ggml_tensor_overhead() * 256, nullptr, true};
        m.gg.meta = ggml_init(params);
        std::vector<ggml_tensor *> weights;
        const auto add = [&](const std::string & name, int n0, int n1 = 1, int n2 = 1) {
            auto * t = ggml_new_tensor_3d(m.gg.meta, GGML_TYPE_F32, n0, n1, n2);
            ggml_set_name(t, name.c_str());
            m.gg.tensors.emplace(name, t);
            weights.push_back(t);
        };
        const auto conv = [&](const std::string & p, int K, int in, int out) {
            add(p + ".weight", K, in, out);
            add(p + ".bias", out);
        };
        add("codec.dq.first.0.embed", dim, book);
        add("codec.dq.first.out_proj.weight", dim, latent);
        for (int i = 0; i < 3; i++) add("codec.dq.rest." + std::to_string(i) + ".embed", dim, book);
        add("codec.dq.rest.out_proj.weight", dim, latent);
        conv("codec.dpre.conv", 3, latent, latent);
        add("codec.dtf.in_proj.weight", latent, v.hidden);
        add("codec.dtf.in_proj.bias", v.hidden);
        for (int il = 0; il < v.n_layer; il++) {
            const std::string p = "codec.dtf.blk." + std::to_string(il);
            add(p + ".attn_norm.weight", v.hidden);
            add(p + ".attn_q.weight", v.hidden, v.head_dim * v.n_head);
            add(p + ".attn_k.weight", v.hidden, v.head_dim * v.n_kv_head);
            add(p + ".attn_v.weight", v.hidden, v.head_dim * v.n_kv_head);
            add(p + ".attn_output.weight", v.head_dim * v.n_head, v.hidden);
            add(p + ".attn_scale", v.hidden);
            add(p + ".ffn_norm.weight", v.hidden);
            add(p + ".ffn_gate.weight", v.hidden, v.ffn);
            add(p + ".ffn_up.weight", v.hidden, v.ffn);
            add(p + ".ffn_down.weight", v.ffn, v.hidden);
            add(p + ".ffn_scale", v.hidden);
        }
        add("codec.dtf.norm.weight", v.hidden);
        add("codec.dtf.out_proj.weight", v.hidden, dec);
        add("codec.dtf.out_proj.bias", dec);
        for (int i = 0; i < 2; i++) {
            const std::string p = "codec.dup." + std::to_string(i);
            conv(p + ".up.conv", 4, dec, dec);
            add(p + ".dw.weight", dec, 7);
            add(p + ".dw.bias", dec);
            add(p + ".norm.weight", dec);
            add(p + ".norm.bias", dec);
            add(p + ".pw1.weight", dec, 20);
            add(p + ".pw1.bias", 20);
            add(p + ".pw2.weight", 20, dec);
            add(p + ".pw2.bias", dec);
            add(p + ".gamma", dec);
        }
        conv("codec.dhead.conv", 5, dec, dec);
        int ch = dec;
        for (int i = 0; i < 2; i++) {
            const std::string p = "codec.dblk." + std::to_string(i);
            const int K = v.upsample_rates[i] * 2;
            const int out = ch / 2;
            add(p + ".alpha", ch);
            add(p + ".beta", ch);
            add(p + ".up.conv.weight", K, out, ch);
            add(p + ".up.conv.bias", out);
            for (int j = 0; j < 3; j++) {
                const std::string r = p + ".res." + std::to_string(j);
                add(r + ".a1", out);
                add(r + ".b1", out);
                conv(r + ".conv1.conv", 7, out, out);
                add(r + ".a2", out);
                add(r + ".b2", out);
                conv(r + ".conv2.conv", 1, out, out);
            }
            ch = out;
        }
        add("codec.dfin.alpha", ch);
        add("codec.dfin.beta", ch);
        conv("codec.dfin.conv", 7, ch, 1);

        m.gg.buffer = ggml_backend_alloc_ctx_tensors(m.gg.meta, m.backend.backend);
        require(m.gg.buffer != nullptr, "weight buffer allocation failed");
        std::mt19937 rng(23);
        for (auto * t : weights) {
            std::vector<float> data(ggml_nelements(t));
            const bool norm = std::string(t->name).find("norm.weight") != std::string::npos;
            for (float & x : data) x = norm ? 1.0f : (static_cast<int>(rng() % 2001) - 1000) * 0.0004f;
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        }
    }
};

static std::vector<int> random_codes(int T, int nc, int book) {
    std::mt19937 rng(5);
    std::vector<int> codes((size_t) T * nc);
    for (int & c : codes) c = rng() % book;
    return codes;
}

static double snr_db(const std::vector<float> & ref, const std::vector<float> & x) {
    double sig = 0, err = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        sig += (double) ref[i] * ref[i];
        err += ((double) ref[i] - x[i]) * ((double) ref[i] - x[i]);
    }
    return err > 0 ? 10 * std::log10(sig / err) : 999.0;
}

// chunk sizes that land on every state boundary, one side of the window and the other
static double check(BreezeModel & m, int T, const std::vector<std::vector<int>> & plans, double floor_db) {
    const int nc = m.cfg.num_codebooks;
    const int book = (int) m.w("codec.dq.first.0.embed")->ne[1];
    const std::vector<int> codes = random_codes(T, nc, book);
    VocoderStream whole;
    whole.init(m);
    const std::vector<float> ref = whole.decode(codes, T);
    require(ref.size() == (size_t) T * m.cfg.samples_per_frame, "whole decode came back the wrong length");
    double worst = 999.0;
    VocoderStream s;
    s.init(m);
    for (const auto & plan : plans) {
        s.reset();
        std::vector<float> out;
        int done = 0, p = 0;
        while (done < T) {
            const int n = std::min(plan[p++ % plan.size()], T - done);
            std::vector<int> sub(codes.begin() + (size_t) done * nc, codes.begin() + (size_t) (done + n) * nc);
            const std::vector<float> part = s.decode(sub, n);
            require(part.size() == (size_t) n * m.cfg.samples_per_frame, "flush came back the wrong length");
            out.insert(out.end(), part.begin(), part.end());
            done += n;
        }
        const double db = snr_db(ref, out);
        printf("  flushes of");
        for (int n : plan) printf(" %d", n);
        printf(": %.1f dB\n", db);
        require(db >= floor_db, "streamed audio drifted from the whole decode");
        worst = std::min(worst, db);
    }
    return worst;
}

static void bench(BreezeModel & m) {
    const int nc = m.cfg.num_codebooks;
    VocoderStream s;
    s.init(m);
    for (int n : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 32, 40 }) {
        const std::vector<int> codes = random_codes(n, nc, m.cfg.codec_codebook_size);
        const auto tf = std::chrono::steady_clock::now();
        s.decode(codes, n);
        const double first = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf).count();
        double best = 1e9, total = 0;
        const int runs = 8;
        for (int r = 0; r < runs; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            s.decode(codes, n);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, ms);
            total += ms;
        }
        printf("  T %2d: first %7.1f ms, best %6.1f ms, avg %6.1f ms, %5.2f ms/frame\n", n, first, best, total / runs, total / runs / n);
    }
}

int main(int argc, char ** argv) {
    try {
        bool gpu = false, do_bench = false;
        std::string path;
        for (int i = 1; i < argc; i++) {
            const std::string a = argv[i];
            if (a == "--gpu") gpu = true;
            else if (a == "--bench") do_bench = true;
            else path = a;
        }
        const std::vector<std::vector<int>> plans = { {1}, {2}, {3}, {1, 2, 4, 8}, {5, 1}, {7}, {16} };
        if (!path.empty()) {
            BreezeModel m;
            if (!m.load(path, gpu)) throw std::runtime_error("could not load " + path);
            printf("%s, %s\n", path.c_str(), m.backend.name());
            if (do_bench) bench(m);
            else check(m, 40, plans, 15.0);
            m.free();
            return 0;
        }
        Fixture fixture;
        fixture.init(gpu);
        if (gpu && !fixture.model.backend.is_gpu) {
            std::puts("no GPU backend available");
            return 77;
        }
        auto & m = fixture.model;
        m.cfg.samples_per_frame = 2 * 2 * 3 * 2;
        printf("testing %s\n", m.backend.name());
        // conv inputs are half precision, so rounding sets the ceiling here
        check(m, 24, plans, gpu ? 30.0 : 50.0);
        VocoderStream s;
        s.init(m);
        s.free();
        s.free();
        require(!s.buffer && !s.ctx, "vocoder state was not released");
        std::puts("vocoder stream tests passed");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
