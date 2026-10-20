#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace breeze {

static std::vector<float> llama3_freq_factors(const DepthConfig & c) {
    const int half = c.head_dim / 2;
    std::vector<float> ff(half, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float low_wl = c.rope_orig_ctx / c.rope_low_freq;
    const float high_wl = c.rope_orig_ctx / c.rope_high_freq;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(c.rope_theta, -2.0f * i / c.head_dim);
        float wavelen = 2.0f * pi / freq;
        if (wavelen > low_wl) {
            ff[i] = c.rope_factor;
        } else if (wavelen < high_wl) {
            ff[i] = 1.0f;
        } else {
            float smooth = (c.rope_orig_ctx / wavelen - c.rope_low_freq) / (c.rope_high_freq - c.rope_low_freq);
            ff[i] = 1.0f / ((1.0f - smooth) / c.rope_factor + smooth);
        }
    }
    return ff;
}

static void check_shape(ggml_tensor * t, const std::string & name,
                        int64_t n0, int64_t n1, int64_t n2 = -1) {
    if (t->ne[0] == n0 && t->ne[1] == n1 && (n2 < 0 || t->ne[2] == n2)) return;
    throw std::runtime_error(
        name + " is [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) + ", " +
        std::to_string(t->ne[2]) + "], expected [" + std::to_string(n0) + ", " + std::to_string(n1) +
        ", " + (n2 < 0 ? std::string("any") : std::to_string(n2)) + "]");
}

void DepthRunner::init(BreezeModel & m, int n_branches) {
    free();
    n_branch = n_branches;
    const DepthConfig & c = m.cfg.dd;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;

    // codebook slices require the checkpoint shapes to match the config
    check_shape(m.w("dd.in_proj.weight"), "dd.in_proj.weight", m.cfg.hidden_size, c.hidden);
    check_shape(m.w("audio_embd.weight"), "audio_embd.weight", m.cfg.hidden_size, (int64_t) nc * vs);
    check_shape(m.w("dd.codebooks_head.weight"), "dd.codebooks_head.weight", c.hidden, vs, nc - 1);

    kv.init(m.backend, c.n_layer, c.head_dim, c.n_kv_head, nc + 1, n_branches);
    freq_factors = llama3_freq_factors(c);
    steps.resize(nc - 1);
}

void DepthRunner::free() {
    steps.clear();
    fused.reset();
    kv.free();
}

// kv(il, k, v, kfull, vfull) hands back the full keys and values this step attends over
template <typename KvFn>
static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, ggml_tensor * x, int il,
                              ggml_tensor * pos, ggml_tensor * ff, ggml_tensor * mask, int n, KvFn && kv) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = ggml_rope_ext(ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = nullptr, * vfull = nullptr;
    kv(il, k, v, kfull, vfull);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

static std::unique_ptr<DepthStep> build_depth_step(BreezeModel & m, DepthRunner & r, int head_idx) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const int start = head_idx == 0 ? 0 : head_idx + 1;
    const int n_pos = head_idx == 0 ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;
    auto step = std::make_unique<DepthStep>();
    Graph & g = step->graph;

    step->audio = g.input_i32(std::vector<int32_t>(nb), nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), step->audio);
    if (head_idx == 0) {
        step->hidden = g.input_f32(std::vector<float>((size_t) nb * m.cfg.hidden_size), m.cfg.hidden_size, nb);
        embed = ggml_concat(g.ctx, step->hidden, embed, 1);
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    std::vector<int32_t> pos_i(n_tok);
    for (int i = 0; i < n_tok; i++) pos_i[i] = start + i / nb;
    ggml_tensor * pos = g.input_i32(pos_i, n_tok);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    std::vector<float> mask_v = build_branch_causal_mask(n_tok, total, start, nb);
    ggml_tensor * mask = g.input_f32(mask_v, total, n_tok);

    auto kv = [&](int il, ggml_tensor * k, ggml_tensor * v, ggml_tensor *& kf, ggml_tensor *& vf) {
        kf = cache_append(g.ctx, g, r.kv.k[il], k, start * nb);
        vf = cache_append(g.ctx, g, r.kv.v[il], v, start * nb);
    };
    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(g.ctx, m, x, il, pos, ff, mask, n_tok, kv);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) head_idx * head->nb[2]);
    step->logits = ggml_mul_mat(g.ctx, hw, last);
    g.prepare(m.backend, step->logits);
    return step;
}

static std::unique_ptr<DepthFused> build_depth_fused(BreezeModel & m, DepthRunner & r, int top_k) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    const int hs = m.cfg.hidden_size;
    auto f = std::make_unique<DepthFused>();
    f->top_k = top_k;
    Graph & g = f->graph;
    ggml_context * ctx = g.ctx;
    ggml_tensor * embd = m.w("audio_embd.weight");
    ggml_tensor * head = m.w("dd.codebooks_head.weight");

    f->cb0 = g.input_i32(std::vector<int32_t>(nb), nb);
    f->hidden = g.input_f32(std::vector<float>((size_t) nb * hs), hs, nb);
    f->noise = g.input_f32(std::vector<float>((size_t) top_k * (nc - 1)), top_k, nc - 1);
    if (nb > 1) f->cfg = g.input_f32({ 1.0f }, 1);
    f->inv_temp = g.input_f32({ 1.0f }, 1);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());

    // the cache only lives inside this graph, so it just grows by concat
    std::vector<ggml_tensor *> kc(c.n_layer, nullptr), vc(c.n_layer, nullptr);
    auto kv = [&](int il, ggml_tensor * k, ggml_tensor * v, ggml_tensor *& kf, ggml_tensor *& vf) {
        kc[il] = kc[il] ? ggml_concat(ctx, kc[il], k, 2) : k;
        vc[il] = vc[il] ? ggml_concat(ctx, vc[il], v, 2) : v;
        kf = kc[il];
        vf = vc[il];
    };

    ggml_tensor * code = nullptr;
    ggml_tensor * codes = nullptr;
    for (int h = 0; h < nc - 1; h++) {
        const int start = h == 0 ? 0 : h + 1;
        const int n_pos = h == 0 ? 2 : 1;
        const int n_tok = n_pos * nb;
        const int total = (start + n_pos) * nb;

        ggml_tensor * embed;
        if (h == 0) {
            embed = ggml_concat(ctx, f->hidden, ggml_get_rows(ctx, embd, f->cb0), 1);
        } else {
            ggml_tensor * table = ggml_view_2d(ctx, embd, embd->ne[0], vs, embd->nb[1], (size_t) h * vs * embd->nb[1]);
            embed = ggml_get_rows(ctx, table, code);
            if (nb > 1) embed = ggml_repeat(ctx, embed, f->hidden);
        }
        ggml_tensor * x = linear(ctx, m.w("dd.in_proj.weight"), embed);

        std::vector<int32_t> pos_i(n_tok);
        for (int i = 0; i < n_tok; i++) pos_i[i] = start + i / nb;
        ggml_tensor * pos = g.input_i32(pos_i, n_tok);
        ggml_tensor * mask = g.input_f32(build_branch_causal_mask(n_tok, total, start, nb), total, n_tok);

        for (int il = 0; il < c.n_layer; il++)
            x = dd_layer(ctx, m, x, il, pos, ff, mask, n_tok, kv);
        x = rms_norm(ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

        ggml_tensor * last = ggml_cont(ctx, ggml_view_2d(ctx, x, c.hidden, nb, x->nb[1], (size_t) (n_tok - nb) * x->nb[1]));
        ggml_tensor * hw = ggml_view_2d(ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) h * head->nb[2]);
        ggml_tensor * logits = ggml_mul_mat(ctx, hw, last);
        if (nb > 1) {
            ggml_tensor * cond = ggml_view_1d(ctx, logits, vs, 0);
            ggml_tensor * unc = ggml_view_1d(ctx, logits, vs, logits->nb[1]);
            logits = ggml_add(ctx, unc, ggml_mul(ctx, ggml_sub(ctx, cond, unc), f->cfg));
        }
        logits = ggml_reshape_2d(ctx, ggml_mul(ctx, logits, f->inv_temp), vs, 1);

        // gumbel max over the top k is the same draw as softmax sampling over them
        ggml_tensor * top = ggml_top_k(ctx, logits, top_k);
        ggml_tensor * vals = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, vs), top);
        ggml_tensor * noisy = ggml_add(ctx, ggml_reshape_1d(ctx, vals, top_k),
                                       ggml_view_1d(ctx, f->noise, top_k, (size_t) h * top_k * sizeof(float)));
        ggml_tensor * pick = ggml_argmax(ctx, ggml_reshape_2d(ctx, noisy, top_k, 1));
        code = ggml_get_rows(ctx, ggml_reshape_2d(ctx, top, 1, top_k), pick);
        codes = codes ? ggml_concat(ctx, codes, code, 0) : code;
    }
    f->codes = codes;
    g.prepare(m.backend, codes);
    return f;
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;
    if (sp_in) sp = *sp_in;

    const int k = sp.top_k > 0 && sp.top_k < vs ? sp.top_k : vs;
    if (!force && sp.top_p >= 1.0f && k <= 256) {
        if (!fused || fused->top_k != k) fused = build_depth_fused(m, *this, k);
        return run_fused(m, hiddens, cb0, cfg_scale, rng, sp);
    }

    std::vector<int32_t> idx(n_branch);
    std::vector<int> codes = { cb0 };
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        if (!steps[head_idx]) steps[head_idx] = build_depth_step(m, *this, head_idx);
        DepthStep & step = *steps[head_idx];
        for (int b = 0; b < n_branch; b++) idx[b] = codes[head_idx] + head_idx * vs;
        ggml_backend_tensor_set(step.audio, idx.data(), 0, idx.size() * sizeof(int32_t));
        if (step.hidden) {
            for (int b = 0; b < n_branch; b++)
                ggml_backend_tensor_set(step.hidden, hiddens[b].data(),
                                        (size_t) b * m.cfg.hidden_size * sizeof(float),
                                        (size_t) m.cfg.hidden_size * sizeof(float));
        }
        step.graph.replay(m.backend);
        std::vector<float> out = tensor_to_f32(step.logits);

        const int vocab = (int) out.size() / n_branch;
        std::vector<float> logits(out.begin(), out.begin() + vocab);
        if (n_branch > 1) {
            for (int i = 0; i < vocab; i++)
                logits[i] = out[vocab + i] + cfg_scale * (out[i] - out[vocab + i]);
        }
        // forced steps still run the graph, later codebooks are conditioned on this one
        codes.push_back(j <= n_force ? force[j - 1] : sample_token(logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

std::vector<int> DepthRunner::run_fused(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                        int cb0, float cfg_scale, std::mt19937 & rng, const SampleParams & sp) {
    const int nc = m.cfg.num_codebooks;
    const int hs = m.cfg.hidden_size;
    DepthFused & f = *fused;

    std::vector<int32_t> idx(n_branch, cb0);
    ggml_backend_tensor_set(f.cb0, idx.data(), 0, idx.size() * sizeof(int32_t));
    for (int b = 0; b < n_branch; b++)
        ggml_backend_tensor_set(f.hidden, hiddens[b].data(), (size_t) b * hs * sizeof(float), (size_t) hs * sizeof(float));

    std::vector<float> noise((size_t) f.top_k * (nc - 1));
    for (float & g : noise) {
        const double u = ((double) rng() + 0.5) / 4294967296.0;
        g = (float) -std::log(-std::log(u));
    }
    ggml_backend_tensor_set(f.noise, noise.data(), 0, noise.size() * sizeof(float));
    const float inv = 1.0f / (sp.temperature > 0 ? sp.temperature : 1.0f);
    if (n_branch > 1) ggml_backend_tensor_set(f.cfg, &cfg_scale, 0, sizeof(float));
    ggml_backend_tensor_set(f.inv_temp, &inv, 0, sizeof(float));

    f.graph.replay(m.backend);
    std::vector<int32_t> out(nc - 1);
    ggml_backend_tensor_get(f.codes, out.data(), 0, out.size() * sizeof(int32_t));
    return std::vector<int>(out.begin(), out.end());
}

}
