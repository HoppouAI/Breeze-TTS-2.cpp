#include "breeze/backbone.h"

#include <cmath>
#include <string>

namespace breeze {

static ggml_tensor * bb_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, bool f32_batch) {
    if (f32_batch && ggml_is_quantized(w->type)) {
        // a row stride keeps Vulkan from quantizing batched F32 activations
        auto * padded = ggml_pad(ctx, x, 1, 0, 0, 0);
        x = ggml_view_2d(ctx, padded, x->ne[0], x->ne[1], padded->nb[1], 0);
    }
    return linear(ctx, w, x);
}

void BackboneState::init(BreezeModel & m, int max_seq) {
    free();
    kv.init(m.backend, m.cfg.bb.n_layer, m.cfg.bb.head_dim, m.cfg.bb.n_kv_head, max_seq);
    pos = 0;
}

static ggml_tensor * build_audio_embed(ggml_context * ctx, BreezeModel & m, ggml_tensor * idx, int n) {
    const int nc = m.cfg.num_codebooks;
    const int hidden = m.cfg.hidden_size;
    // get_rows only indexes along one axis, so the frames stay flat until after the lookup
    ggml_tensor * rows = ggml_get_rows(ctx, m.w("audio_embd.weight"), idx);    // [hidden, nc*n]
    rows = ggml_reshape_3d(ctx, rows, hidden, nc, n);
    ggml_tensor * perm = ggml_cont(ctx, ggml_permute(ctx, rows, 1, 0, 2, 3));  // [nc, hidden, n]
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);                           // [1, hidden, n]
    return ggml_reshape_2d(ctx, summed, hidden, n);
}

std::vector<float> audio_embed_forward(BreezeModel & m, const std::vector<int> & codes, int n) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    Graph g(256);
    std::vector<int32_t> idx((size_t) nc * n);
    for (int f = 0; f < n; f++)
        for (int cb = 0; cb < nc; cb++)
            idx[(size_t) f * nc + cb] = codes[(size_t) f * nc + cb] + cb * vs;
    ggml_tensor * t = g.input_i32(idx, nc * n);
    ggml_tensor * out = build_audio_embed(g.ctx, m, t, n);
    g.compute(m.backend, out);
    return tensor_to_f32(out);
}

static ggml_tensor * bb_layer(ggml_context * ctx, BreezeModel & m, Graph & g, BackboneState & st,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * mask, int n,
                              BackboneState * other = nullptr, ggml_tensor * other_mask = nullptr) {
    const BackboneConfig & c = m.cfg.bb;
    const std::string p = "bb.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const bool f32_batch = other && std::string(m.backend.name()).find("Vulkan") == 0;
    auto project = [&](const char * suffix, ggml_tensor * input) {
        return bb_linear(ctx, m.w(p + suffix), input, f32_batch);
    };

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);

    ggml_tensor * q = ggml_reshape_3d(ctx, project(".attn_q.weight", h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, project(".attn_k.weight", h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, project(".attn_v.weight", h), c.head_dim, c.n_kv_head, n);
    q = rms_norm(ctx, q, m.w(p + ".attn_q_norm.weight"), c.rms_eps);
    k = rms_norm(ctx, k, m.w(p + ".attn_k_norm.weight"), c.rms_eps);
    q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    auto attend = [&](BackboneState & branch, ggml_tensor * branch_mask, int index) {
        auto slice = [&](ggml_tensor * t) {
            return other ? ggml_view_3d(ctx, t, t->ne[0], t->ne[1], 1,
                                        t->nb[1], t->nb[2], index * t->nb[2]) : t;
        };
        auto * kfull = cache_append(ctx, g, branch.kv.k[il], slice(k), branch.pos);
        auto * vfull = cache_append(ctx, g, branch.kv.v[il], slice(v), branch.pos);
        return attention(ctx, slice(q), kfull, vfull, branch_mask, scale, c.n_head, c.n_kv_head);
    };
    ggml_tensor * a = attend(st, mask, 0);
    if (other) a = ggml_concat(ctx, a, attend(*other, other_mask, 1), 1);
    a = project(".attn_output.weight", a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    auto * gate = project(".ffn_gate.weight", h);
    auto * up = project(".ffn_up.weight", h);
    h = project(".ffn_down.weight", ggml_swiglu_split(ctx, gate, up));
    return ggml_add(ctx, res, h);
}

StepOut backbone_run(BreezeModel & m, BackboneState & st, const std::vector<float> & embeds, int n) {
    const BackboneConfig & c = m.cfg.bb;
    const int total = st.pos + n;
    Graph g(8192);

    ggml_tensor * x = g.input_f32(embeds, c.hidden, n);
    std::vector<int32_t> pos_i(n);
    for (int i = 0; i < n; i++) pos_i[i] = st.pos + i;
    ggml_tensor * pos = g.input_i32(pos_i, n);
    std::vector<float> mask_v = build_causal_mask(n, total, st.pos, 0);
    ggml_tensor * mask = g.input_f32(mask_v, total, n);

    for (int il = 0; il < c.n_layer; il++) x = bb_layer(g.ctx, m, g, st, x, il, pos, mask, n);
    x = rms_norm(g.ctx, x, m.w("bb.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_view_2d(g.ctx, x, c.hidden, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    last = ggml_cont(g.ctx, last);
    ggml_tensor * logits = linear(g.ctx, m.w("bb.lm_head.weight"), last);

    ggml_set_output(last);
    g.write(last);
    g.compute(m.backend, logits);

    StepOut out;
    out.hidden = tensor_to_f32(last);
    out.logits = tensor_to_f32(logits);
    st.pos += n;
    return out;
}

std::array<StepOut, 2> backbone_run_cfg(BreezeModel & m, BackboneState & cond, BackboneState & uncond,
                                       const std::vector<float> & embed) {
    const auto & c = m.cfg.bb;
    Graph g(8192);
    auto * input = g.input_f32(embed, c.hidden);
    auto * x = ggml_concat(g.ctx, input, input, 1);
    auto * pos = g.input_i32({cond.pos, uncond.pos}, 2);
    auto * mask_c = g.input_f32(std::vector<float>(cond.pos + 1, 0.0f), cond.pos + 1);
    auto * mask_u = g.input_f32(std::vector<float>(uncond.pos + 1, 0.0f), uncond.pos + 1);
    for (int il = 0; il < c.n_layer; il++)
        x = bb_layer(g.ctx, m, g, cond, x, il, pos, mask_c, 2, &uncond, mask_u);
    x = rms_norm(g.ctx, x, m.w("bb.output_norm.weight"), c.rms_eps);
    auto * logits = bb_linear(g.ctx, m.w("bb.lm_head.weight"), x,
                             std::string(m.backend.name()).find("Vulkan") == 0);
    ggml_set_output(x);
    g.write(x);
    g.compute(m.backend, logits);

    const auto hidden = tensor_to_f32(x);
    const auto scores = tensor_to_f32(logits);
    const size_t vocab = scores.size() / 2;
    std::array<StepOut, 2> out;
    for (int b = 0; b < 2; b++) {
        out[b].hidden.assign(hidden.begin() + b * c.hidden, hidden.begin() + (b + 1) * c.hidden);
        out[b].logits.assign(scores.begin() + b * vocab, scores.begin() + (b + 1) * vocab);
    }
    cond.pos++;
    uncond.pos++;
    return out;
}

}
