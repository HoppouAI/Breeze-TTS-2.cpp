#pragma once

#include "breeze/common.h"
#include "ggml.h"
#include "gguf.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace breeze {

// loads a GGUF file, uploads all tensor data to the backend and exposes typed KV access
struct GGUFModel {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;

    // tensors whose names start with cpu_prefix go to cpu_be instead of be
    // (BREEZE_TE_CPU: the text encoder in system memory)
    bool load(const std::string & path, Backend & be, Backend * cpu_be = nullptr, const char * cpu_prefix = nullptr);
    ggml_context * meta_cpu = nullptr;
    ggml_backend_buffer_t buffer_cpu = nullptr;
    void free();

    ggml_tensor * get(const std::string & name) const;
    ggml_tensor * find(const std::string & name) const;
    bool has(const std::string & name) const;

    int   kv_u32(const char * key, int def = 0) const;
    float kv_f32(const char * key, float def = 0.0f) const;
    bool  kv_bool(const char * key, bool def = false) const;
    std::string kv_str(const char * key, const std::string & def = "") const;
    std::vector<int> kv_i32_array(const char * key) const;
    std::vector<std::string> kv_str_array(const char * key) const;
};

}
