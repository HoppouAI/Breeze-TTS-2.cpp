#include "breeze/voice.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace breeze {

static const char MAGIC[4] = { 'B', 'R', 'Z', 'V' };

static bool put32(FILE * f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
static bool get32(FILE * f, uint32_t & v) { return fread(&v, 4, 1, f) == 1; }

static bool put_str(FILE * f, const std::string & s) {
    return put32(f, (uint32_t) s.size()) && (s.empty() || fwrite(s.data(), 1, s.size(), f) == s.size());
}

static bool get_str(FILE * f, std::string & s, uint32_t max_len) {
    uint32_t n = 0;
    if (!get32(f, n) || n > max_len) return false;
    s.resize(n);
    return n == 0 || fread(&s[0], 1, n, f) == n;
}

bool save_voice(const std::string & path, const Voice & v) {
    if (v.frames <= 0 || v.n_codebooks <= 0 ||
        v.codes.size() != (size_t) v.frames * v.n_codebooks) return false;

    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);

    // version 1 when there is no metadata so older builds can still read the file
    const bool v2 = !v.description.empty() || !v.meta.empty();
    // written next to the target then swapped in, so a failed edit cant eat an existing voice
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(MAGIC, 1, 4, f) == 4 &&
              put32(f, v2 ? 2 : 1) && put32(f, (uint32_t) v.sample_rate) &&
              put32(f, (uint32_t) v.n_codebooks) && put32(f, (uint32_t) v.frames) &&
              put_str(f, v.text);
    if (ok && v2) {
        ok = put_str(f, v.description) && put32(f, (uint32_t) v.meta.size());
        for (auto it = v.meta.begin(); ok && it != v.meta.end(); ++it)
            ok = put_str(f, it->first) && put_str(f, it->second);
    }
    for (size_t i = 0; ok && i < v.codes.size(); i++) ok = put32(f, (uint32_t) (int32_t) v.codes[i]);
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) std::filesystem::remove(tmp, ec);
    return ok;
}

bool load_voice(const std::string & path, Voice & v) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char magic[4];
    uint32_t version = 0, sr = 0, ncb = 0, frames = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, MAGIC, 4) == 0 &&
              get32(f, version) && (version == 1 || version == 2) &&
              get32(f, sr) && get32(f, ncb) && get32(f, frames);

    // a corrupt header should not turn into a giant allocation
    if (ok) ok = ncb > 0 && ncb <= 64 && frames > 0 && frames <= 100000 && get_str(f, v.text, 1u << 20);
    v.description.clear();
    v.meta.clear();
    if (ok && version == 2) {
        uint32_t n = 0;
        ok = get_str(f, v.description, 1u << 16) && get32(f, n) && n <= 256;
        for (uint32_t i = 0; ok && i < n; i++) {
            std::string key, value;
            ok = get_str(f, key, 256) && get_str(f, value, 1u << 16);
            if (ok) v.meta[key] = value;
        }
    }
    if (ok) {
        v.codes.resize((size_t) frames * ncb);
        for (size_t i = 0; ok && i < v.codes.size(); i++) {
            uint32_t c = 0;
            ok = get32(f, c);
            v.codes[i] = (int) (int32_t) c;
        }
    }
    fclose(f);
    if (!ok) return false;

    v.sample_rate = (int) sr;
    v.n_codebooks = (int) ncb;
    v.frames = (int) frames;
    v.name = std::filesystem::path(path).stem().string();
    return true;
}

std::vector<Voice> load_voice_dir(const std::string & dir) {
    std::vector<Voice> out;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return out;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".breeze") continue;
        Voice v;
        if (load_voice(e.path().string(), v)) out.push_back(std::move(v));
        else fprintf(stderr, "skipping %s, not a readable voice file\n", e.path().string().c_str());
    }
    return out;
}

bool valid_voice_meta_key(const std::string & key) {
    if (key.empty() || key.size() > 64 || key == "description") return false;
    for (char c : key) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!ok) return false;
    }
    return true;
}

bool valid_voice_name(const std::string & name) {
    if (name.empty() || name.size() > 64) return false;
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

} // namespace breeze
