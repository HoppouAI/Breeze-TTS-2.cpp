#include "breeze/codec.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace breeze {

void VocoderStream::init(BreezeModel & model) {
    free();
    m = &model;
    ggml_init_params p{ ggml_tensor_overhead() * 512, nullptr, true };
    ctx = ggml_init(p);
    // the shapes come from walking the real graph once, so they cant drift from it
    planning = true;
    {
        Graph g(32768);
        const int n_cb = model.cfg.num_codebooks;
        codec_detail::vocoder_decode(g.ctx, model, g, *this, std::vector<int>(n_cb, 0), n_cb, 1);
    }
    planning = false;
    buffer = ggml_backend_alloc_ctx_tensors(ctx, model.backend.backend);
    if (!buffer) throw std::runtime_error("could not allocate vocoder state");
    reset();
}

void VocoderStream::reset() {
    ggml_backend_buffer_clear(buffer, 0);
    kv_len = 0;
}

void VocoderStream::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx) ggml_free(ctx);
    buffer = nullptr;
    ctx = nullptr;
    slots.clear();
    next = 0;
    kv_len = 0;
}

ggml_tensor * VocoderStream::slot(int n0, int n1, int n2) {
    if (planning) {
        slots.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2));
        return slots.back();
    }
    return slots.at(next++);
}

std::vector<float> VocoderStream::decode(const std::vector<int> & codes, int T, int n_cb) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    next = 0;
    Graph g(32768);
    ggml_tensor * x = codec_detail::vocoder_decode(g.ctx, *m, g, *this, codes, n_cb, T);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    g.compute(m->backend, audio);
    kv_len = std::min(kv_len + T, m->cfg.voc.sliding_window - 1);
    return tensor_to_f32(audio);
}

namespace codec_detail {

// stores the last n samples of x [L, C] into state for the next flush
static void keep_tail(ggml_context * ctx, Graph & g, ggml_tensor * x, ggml_tensor * state, int64_t offset) {
    ggml_tensor * tail = ggml_view_2d(ctx, x, state->ne[0], x->ne[1], x->nb[1], (size_t) offset * x->nb[0]);
    g.write(ggml_cpy(ctx, ggml_cont(ctx, tail), state));
}

static ggml_tensor * conv1d_stream(ggml_context * ctx, Graph & g, VocoderStream & s, ggml_tensor * w, ggml_tensor * b,
                                   ggml_tensor * x, int dilation) {
    const int hist = ((int) w->ne[0] - 1) * dilation;
    if (hist > 0) {
        ggml_tensor * state = s.slot(hist, (int) x->ne[1]);
        const int64_t L = x->ne[0];
        x = ggml_concat(ctx, state, x, 0);
        keep_tail(ctx, g, x, state, L);
    } else if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, 1, 0, dilation);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

static ggml_tensor * convtr1d_stream(ggml_context * ctx, Graph & g, VocoderStream & s, ggml_tensor * w, ggml_tensor * b,
                                     ggml_tensor * x, int stride) {
    const int K = (int) w->ne[0];
    const int C_out = (int) w->ne[1];
    const int L = (int) x->ne[0];
    const int tail = K - stride;
    const int64_t T = (int64_t) L * stride + tail;
    ggml_tensor * y = nullptr;
    if (K % stride == 0) {
        // matmul then overlap add, the cuda conv_transpose_1d kernel is a per sample loop and takes seconds
        ggml_tensor * a = ggml_cont(ctx, ggml_permute(ctx, w, 2, 1, 0, 3));
        a = ggml_reshape_2d(ctx, a, a->ne[0], (int64_t) C_out * K);
        ggml_tensor * r = ggml_mul_mat(ctx, a, ggml_cont(ctx, ggml_transpose(ctx, x)));
        const int64_t block = (int64_t) C_out * stride;
        const int n_seg = K / stride;
        for (int j = 0; j < n_seg; j++) {
            ggml_tensor * seg = ggml_cont(ctx, ggml_view_2d(ctx, r, block, L, r->nb[1], (size_t) (j * block) * r->nb[0]));
            // pad flat, cuda pad makes ne1 a grid dim and those stop at 65535
            seg = ggml_reshape_1d(ctx, seg, block * L);
            if (n_seg > 1) seg = ggml_pad_ext(ctx, seg, j * (int) block, (n_seg - 1 - j) * (int) block, 0, 0, 0, 0, 0, 0);
            y = y ? ggml_add(ctx, y, seg) : seg;
        }
        y = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, y, C_out, T)));
    } else {
        y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    }
    if (tail > 0) {
        // the overlap the last flush couldnt finish lands on the front of this one
        ggml_tensor * state = s.slot(tail, C_out);
        y = ggml_add(ctx, y, ggml_pad_ext(ctx, state, 0, (int) (T - tail), 0, 0, 0, 0, 0, 0));
        keep_tail(ctx, g, y, state, (int64_t) L * stride);
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, (int64_t) L * stride, y->ne[1], y->nb[1], 0));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

// depthwise conv as shift and multiply, ggml conv ops have no group support. x is [L, C], w is [C, K]
static ggml_tensor * depthwise1d_stream(ggml_context * ctx, Graph & g, VocoderStream & s, ggml_tensor * w, ggml_tensor * b,
                                        ggml_tensor * x, int K) {
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor * state = s.slot(K - 1, (int) C);
    ggml_tensor * xs = ggml_concat(ctx, state, x, 0);
    keep_tail(ctx, g, xs, state, L);
    ggml_tensor * acc = nullptr;
    for (int k = 0; k < K; k++) {
        ggml_tensor * seg = ggml_cont(ctx, ggml_view_2d(ctx, xs, L, C, xs->nb[1], (size_t) k * xs->nb[0]));
        ggml_tensor * wk = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_view_2d(ctx, w, C, 1, w->nb[1], (size_t) k * w->nb[1])));
        ggml_tensor * t = ggml_mul(ctx, seg, wk);
        acc = acc ? ggml_add(ctx, acc, t) : t;
    }
    if (b) acc = ggml_add(ctx, acc, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return acc;
}

// snake beta activation: x + sin(x*alpha)^2 / beta, both params stored in log space
static ggml_tensor * snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb) {
    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, lb->ne[0]);
    ggml_tensor * sn = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, sn), beta));
}

// convnext block over [L, C]
static ggml_tensor * convnext(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s, const std::string & p,
                              ggml_tensor * x) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = depthwise1d_stream(ctx, g, s, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    return ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
}

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s, const std::string & p,
                                   ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = conv1d_stream(ctx, g, s, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, dilation);
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = conv1d_stream(ctx, g, s, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1);
    return ggml_add(ctx, x, h);
}

// rebuild the continuous latent from the residual codebooks, first stage split from the rest
static ggml_tensor * quantizer_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                                      const std::vector<int> & codes, int n_cb, int T) {
    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) idx[t] = codes[(size_t) t * n_cb + cb];
        ggml_tensor * ids = g.input_i32(idx, T);
        return ggml_get_rows(ctx, m.w(name), ids);
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    if (rest) {
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        first = ggml_add(ctx, first, rest);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, first));
}

ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g, VocoderStream & s,
                             const std::vector<int> & codes, int n_cb, int T) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T);
    h = conv1d_stream(ctx, g, s, m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer(ctx, m, g, s, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = convtr1d_stream(ctx, g, s, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsampling_ratios[i]);
        h = convnext(ctx, m, g, s, p, h);
    }

    h = conv1d_stream(ctx, g, s, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = convtr1d_stream(ctx, g, s, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(ctx, m, g, s, p + ".res." + std::to_string(j), h, dilations[j]);
        }
    }

    h = snake_beta(ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = conv1d_stream(ctx, g, s, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1);
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

}
}
