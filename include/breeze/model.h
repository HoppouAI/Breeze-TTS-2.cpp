#pragma once

#include "breeze/config.h"
#include "breeze/common.h"
#include "breeze/gguf_loader.h"
#include "breeze/tokenizer.h"

#include <memory>
#include <string>

namespace breeze {

struct BreezeModel {
    Backend backend;
    // the text encoder (about 0.9 GB, run once per prompt) in system memory on
    // the cpu backend, leaving the gpu for the backbone, depth decoder and codec
    Backend cpu;
    bool te_cpu = false;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;

    // te_cpu keeps the text encoder in system memory on the cpu backend;
    // BREEZE_TE_CPU=1 in the environment does the same for the c api
    bool load(const std::string & path, bool prefer_gpu, bool te_cpu = false);
    void free();

    ggml_tensor * w(const std::string & name) const { return gg.get(name); }
    ggml_tensor * wopt(const std::string & name) const { return gg.find(name); }
};

}
