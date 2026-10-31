#include "breeze/codec.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace breeze {
namespace codec_detail {

// mimi transformer over [hidden, seq_len]; pre-norm blocks with per-branch layer scale and gelu MLP
ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int T) {
    const CodecConfig & c = m.cfg.codec;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const float eps = c.layer_norm_eps;

    std::vector<int32_t> pos_i(T);
    for (int i = 0; i < T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, T);
    std::vector<float> mask_v = build_causal_mask(T, T, 0, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, T, T);

    ggml_tensor * h = x;
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = prefix + ".blk." + std::to_string(il);
        ggml_tensor * res = h;
        ggml_tensor * cur = layer_norm(ctx, h, m.w(p + ".attn_norm.weight"), m.w(p + ".attn_norm.bias"), eps);
        ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), cur), c.head_dim, c.n_head, T);
        q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(ctx, q, k, v, mask, scale, c.n_head, c.n_head);
        a = linear(ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(ctx, a, m.w(p + ".attn_scale"));
        h = ggml_add(ctx, res, a);

        res = h;
        cur = layer_norm(ctx, h, m.w(p + ".ffn_norm.weight"), m.w(p + ".ffn_norm.bias"), eps);
        cur = ggml_gelu_erf(ctx, linear(ctx, m.w(p + ".ffn_up.weight"), cur));
        cur = linear(ctx, m.w(p + ".ffn_down.weight"), cur);
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        h = ggml_add(ctx, res, cur);
    }
    return h;
}

// vocoder pre-transformer over [hidden, seq_len]; rms norm, swiglu MLP, causal sliding window
// keys stay unrotated in the state, the window slides so their positions keep changing
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s, ggml_tensor * x, int T) {
    const VocoderConfig & c = m.cfg.voc;
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const int window = c.sliding_window - 1;
    const int n_hist = s.kv_len;
    const int total = n_hist + T;
    const int keep = std::min(window, total);

    std::vector<int32_t> pos_i(total);
    for (int i = 0; i < total; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, total);
    std::vector<int32_t> pos_q(T);
    for (int i = 0; i < T; i++) pos_q[i] = n_hist + i;
    ggml_tensor * qpos = g.input_i32(pos_q, T);
    std::vector<float> mask_v = build_causal_mask(T, total, n_hist, c.sliding_window);
    ggml_tensor * mask = g.input_f32(mask_v, total, T);

    const auto extend = [&](ggml_tensor * cur) {
        ggml_tensor * state = s.slot(c.head_dim, c.n_kv_head, window);
        ggml_tensor * all = cur;
        if (n_hist > 0) {
            ggml_tensor * old = ggml_view_3d(ctx, state, c.head_dim, c.n_kv_head, n_hist, state->nb[1], state->nb[2], 0);
            all = ggml_concat(ctx, old, cur, 2);
        }
        ggml_tensor * newest = ggml_view_3d(ctx, all, c.head_dim, c.n_kv_head, keep, all->nb[1], all->nb[2],
                                            (size_t) (total - keep) * all->nb[2]);
        ggml_tensor * dst = ggml_view_3d(ctx, state, c.head_dim, c.n_kv_head, keep, state->nb[1], state->nb[2], 0);
        g.write(ggml_cpy(ctx, newest, dst));
        return all;
    };

    ggml_tensor * h = ggml_add(ctx, linear(ctx, m.w("codec.dtf.in_proj.weight"), x),
                               m.w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = h;
        ggml_tensor * cur = rms_norm(ctx, h, m.w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), cur), c.head_dim, c.n_head, T);
        ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), cur), c.head_dim, c.n_kv_head, T);
        ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), cur), c.head_dim, c.n_kv_head, T);
        k = extend(k);
        v = extend(v);
        q = ggml_rope_ext(ctx, q, qpos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(ctx, q, k, v, mask, scale, c.n_head, c.n_kv_head);
        a = linear(ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(ctx, a, m.w(p + ".attn_scale"));
        h = ggml_add(ctx, res, a);

        res = h;
        cur = rms_norm(ctx, h, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
                         m.w(p + ".ffn_down.weight"));
        cur = ggml_mul(ctx, cur, m.w(p + ".ffn_scale"));
        h = ggml_add(ctx, res, cur);
    }
    h = rms_norm(ctx, h, m.w("codec.dtf.norm.weight"), c.rms_eps);
    return ggml_add(ctx, linear(ctx, m.w("codec.dtf.out_proj.weight"), h), m.w("codec.dtf.out_proj.bias"));
}

}
}
