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
    // only used for the text encoder with te_cpu
    Backend cpu;
    bool te_cpu = false;
    GGUFModel gg;
    BreezeConfig cfg;
    Tokenizer tok;

    // BREEZE_TE_CPU=1 turns te_cpu on too, thats the only switch the c api has
    bool load(const std::string & path, bool prefer_gpu, bool te_cpu = false);
    void free();

    ggml_tensor * w(const std::string & name) const { return gg.get(name); }
    ggml_tensor * wopt(const std::string & name) const { return gg.find(name); }
};

}
