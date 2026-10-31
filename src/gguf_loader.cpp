#include "breeze/gguf_loader.h"

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>
#include <stdexcept>

namespace breeze {

#ifdef _WIN32
#define breeze_fseek _fseeki64
#else
#define breeze_fseek fseeko
#endif

bool GGUFModel::load(const std::string & path, Backend & be, Backend * cpu_be, const char * cpu_prefix) {
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/&meta };
    gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) return false;

    // Tensors for the CPU get twins in their own context, allocated in a CPU
    // buffer; the originals then point at the twins' memory, so the GPU
    // allocation below skips them (it only allocates tensors without data).
    if (cpu_be && cpu_be->backend && cpu_prefix) {
        const size_t plen = strlen(cpu_prefix);
        const int64_t nt = gguf_get_n_tensors(gguf);
        ggml_init_params cp{ ggml_tensor_overhead() * (size_t) (nt + 1), nullptr, true };
        meta_cpu = ggml_init(cp);
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> twins;
        for (int64_t i = 0; i < nt; i++) {
            const char * name = gguf_get_tensor_name(gguf, i);
            if (strncmp(name, cpu_prefix, plen) != 0) continue;
            ggml_tensor * t = ggml_get_tensor(meta, name);
            ggml_tensor * d = ggml_dup_tensor(meta_cpu, t);
            ggml_set_name(d, name);
            twins.emplace_back(t, d);
        }
        if (!twins.empty()) {
            buffer_cpu = ggml_backend_alloc_ctx_tensors(meta_cpu, cpu_be->backend);
            if (!buffer_cpu) return false;
            for (auto & [t, d] : twins) { t->data = d->data; t->buffer = d->buffer; }
        }
    }

    buffer = ggml_backend_alloc_ctx_tensors(meta, be.backend);
    if (!buffer) return false;

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    const size_t data_off = gguf_get_data_offset(gguf);
    const int64_t n = gguf_get_n_tensors(gguf);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const size_t off = data_off + gguf_get_tensor_offset(gguf, i);
        const size_t sz = ggml_nbytes(t);
        buf.resize(sz);
        if (breeze_fseek(f, (long long) off, SEEK_SET) != 0) { fclose(f); return false; }
        if (fread(buf.data(), 1, sz, f) != sz) { fclose(f); return false; }
        ggml_backend_tensor_set(t, buf.data(), 0, sz);
        tensors[name] = t;
    }
    fclose(f);
    return true;
}

void GGUFModel::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (buffer_cpu) ggml_backend_buffer_free(buffer_cpu);
    if (meta_cpu) ggml_free(meta_cpu);
    buffer_cpu = nullptr;
    meta_cpu = nullptr;
    if (meta) ggml_free(meta);
    if (gguf) gguf_free(gguf);
    buffer = nullptr;
    meta = nullptr;
    gguf = nullptr;
}

ggml_tensor * GGUFModel::find(const std::string & name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor * GGUFModel::get(const std::string & name) const {
    ggml_tensor * t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool GGUFModel::has(const std::string & name) const {
    return tensors.count(name) > 0;
}

int GGUFModel::kv_u32(const char * key, int def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(gguf, id)) {
        case GGUF_TYPE_UINT32: return (int) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT32:  return (int) gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT64: return (int) gguf_get_val_u64(gguf, id);
        case GGUF_TYPE_INT64:  return (int) gguf_get_val_i64(gguf, id);
        default: return def;
    }
}

float GGUFModel::kv_f32(const char * key, float def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    const gguf_type t = gguf_get_kv_type(gguf, id);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gguf, id);
    if (t == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(gguf, id);
    return def;
}

bool GGUFModel::kv_bool(const char * key, bool def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(gguf, id) == GGUF_TYPE_BOOL) return gguf_get_val_bool(gguf, id);
    return def;
}

std::string GGUFModel::kv_str(const char * key, const std::string & def) const {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(gguf, id);
}

std::vector<int> GGUFModel::kv_i32_array(const char * key) const {
    std::vector<int> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    const void * data = gguf_get_arr_data(gguf, id);
    const gguf_type at = gguf_get_arr_type(gguf, id);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (at == GGUF_TYPE_INT32) out[i] = ((const int32_t *) data)[i];
        else if (at == GGUF_TYPE_UINT32) out[i] = (int) ((const uint32_t *) data)[i];
    }
    return out;
}

std::vector<std::string> GGUFModel::kv_str_array(const char * key) const {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(gguf, id);
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.emplace_back(gguf_get_arr_str(gguf, id, i));
    return out;
}

}
