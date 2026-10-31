#include "breeze/codec.h"

#include <string>
#include <vector>

namespace breeze {
namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation) {
    const int K = (int) w->ne[0];
    int pad = (K - 1) * dilation - (stride - 1);
    if (pad < 0) pad = 0;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t n_out = (x->ne[0] + pad - (int64_t) dilation * (K - 1) - 1) / stride + 1;
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dilation);
    if (y->ne[0] != n_out) {
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, n_out, y->ne[1], y->nb[1], 0));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * h = ggml_elu(ctx, x);
    h = conv1d_causal(ctx, m.w(p + ".block.1.conv.weight"), m.w(p + ".block.1.conv.bias"), h, 1, 1);
    h = ggml_elu(ctx, h);
    h = conv1d_causal(ctx, m.w(p + ".block.3.conv.weight"), m.w(p + ".block.3.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x) {
    const std::vector<int> & up = m.cfg.codec.upsampling_ratios;
    auto conv = [&](int i, ggml_tensor * in, int s) {
        const std::string p = "codec.enc." + std::to_string(i);
        return conv1d_causal(ctx, m.w(p + ".conv.weight"), m.w(p + ".conv.bias"), in, s, 1);
    };
    const int conv_idx[4] = { 3, 6, 9, 12 };
    const int res_idx[4] = { 1, 4, 7, 10 };
    const int ratios[4] = { up[3], up[2], up[1], up[0] };
    ggml_tensor * h = conv(0, x, 1);
    for (int s = 0; s < 4; s++) {
        h = resnet_block(ctx, m, "codec.enc." + std::to_string(res_idx[s]), h);
        h = ggml_elu(ctx, h);
        h = conv(conv_idx[s], h, ratios[s]);
    }
    h = ggml_elu(ctx, h);
    return conv(14, h, 1);
}

}
}
