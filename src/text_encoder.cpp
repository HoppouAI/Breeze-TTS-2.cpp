#include "breeze/text_encoder.h"

#include <cmath>
#include <string>

namespace breeze {

static ggml_tensor * te_layer(ggml_context * ctx, BreezeModel & m, ggml_tensor * x, int il,
                              ggml_tensor * pos, ggml_tensor * mask) {
    const TextEncoderConfig & c = m.cfg.te;
    const std::string p = "te.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt(c.query_pre_attn_scalar);

    bool is_full = il < (int) c.layer_is_full.size() ? c.layer_is_full[il] != 0 : false;
    float theta = is_full ? c.rope_theta_full : c.rope_theta_sliding;
    float freq_scale = is_full ? c.rope_freq_scale_full : 1.0f;

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);

    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, h->ne[1]);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, h->ne[1]);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, h->ne[1]);
    q = rms_norm(ctx, q, m.w(p + ".attn_q_norm.weight"), c.rms_eps);
    k = rms_norm(ctx, k, m.w(p + ".attn_k_norm.weight"), c.rms_eps);
    q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * a = attention(ctx, q, k, v, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    a = rms_norm(ctx, a, m.w(p + ".post_attn_norm.weight"), c.rms_eps);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    ggml_tensor * g = ggml_gelu(ctx, linear(ctx, m.w(p + ".ffn_gate.weight"), h));
    ggml_tensor * u = linear(ctx, m.w(p + ".ffn_up.weight"), h);
    h = linear(ctx, m.w(p + ".ffn_down.weight"), ggml_mul(ctx, g, u));
    h = rms_norm(ctx, h, m.w(p + ".post_ffn_norm.weight"), c.rms_eps);
    return ggml_add(ctx, res, h);
}

std::vector<float> text_encoder_forward(BreezeModel & m, const std::vector<int> & tokens) {
    const TextEncoderConfig & c = m.cfg.te;
    const int T = (int) tokens.size();
    Graph g(4096);

    std::vector<int32_t> tok_i(tokens.begin(), tokens.end());
    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = i;

    ggml_tensor * tok = g.input_i32(tok_i, T);
    ggml_tensor * pos = g.input_i32(pos_i, T);

    std::vector<float> mask_full = build_bidirectional_mask(T, T, 0);
    std::vector<float> mask_win = build_bidirectional_mask(T, T, c.sliding_window);
    ggml_tensor * mfull = g.input_f32(mask_full, T, T);
    ggml_tensor * mwin = g.input_f32(mask_win, T, T);

    ggml_tensor * x = ggml_get_rows(g.ctx, m.w("te.token_embd.weight"), tok);
    x = ggml_scale(g.ctx, x, c.embed_scale);

    for (int il = 0; il < c.n_layer; il++) {
        bool is_full = il < (int) c.layer_is_full.size() ? c.layer_is_full[il] != 0 : false;
        x = te_layer(g.ctx, m, x, il, pos, is_full ? mfull : mwin);
    }
    x = rms_norm(g.ctx, x, m.w("te.output_norm.weight"), c.rms_eps);
    x = linear(g.ctx, m.w("te.proj.weight"), x);

    g.compute(m.te_cpu ? m.cpu : m.backend, x);
    return tensor_to_f32(x);
}

}
