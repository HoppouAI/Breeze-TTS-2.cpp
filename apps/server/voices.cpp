#include "voices.h"
#include "json.h"

#include "breeze/audio.h"
#include "breeze/voice.h"

#include <chrono>
#include <cstdio>
#include <algorithm>

namespace breeze {

// keyed off the clip and its transcript, so sending the same voice twice lands on the same entry
std::string voice_key(const std::string & wav, const std::string & text) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : wav) { h ^= c; h *= 1099511628211ull; }
    for (unsigned char c : text) { h ^= c; h *= 1099511628211ull; }
    char buf[32];
    snprintf(buf, sizeof buf, "v_%016llx", (unsigned long long) h);
    return buf;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof b, "\\u%04x", (unsigned) (unsigned char) c);
                    o += b;
                } else {
                    o += c;
                }
        }
    }
    return o;
}

static std::string voice_json(const std::string & id, const CachedVoice & v, int sr, int spf) {
    char buf[256];
    snprintf(buf, sizeof buf,
             "{\"id\":\"%s\",\"frames\":%d,\"seconds\":%.2f,\"encode_ms\":%.0f,\"saved\":%s,\"ref_text\":\"",
             json_escape(id).c_str(), v.frames, (double) v.frames * spf / sr, v.encode_ms,
             v.saved ? "true" : "false");
    std::string out = std::string(buf) + json_escape(v.text) + "\",\"description\":\"" +
                      json_escape(v.description) + "\",\"metadata\":{";
    bool first = true;
    for (const auto & kv : v.meta) {
        if (!first) out += ",";
        first = false;
        out += "\"" + json_escape(kv.first) + "\":\"" + json_escape(kv.second) + "\"";
    }
    return out + "}}";
}

static void fail(httplib::Response & res, int status, const std::string & msg) {
    res.status = status;
    res.set_content("{\"error\":\"" + json_escape(msg) + "\"}", "application/json");
}

// description and metadata come either as a json body or as form fields, metadata being a json string
static std::string read_meta_request(const httplib::Request & req, JVal & root) {
    root = JVal();
    if (req.get_header_value("Content-Type").find("json") != std::string::npos) {
        if (!JParser(req.body).parse(root) || root.kind != JVal::Obj) return "request body must be a json object";
        return "";
    }
    root.kind = JVal::Obj;
    if (req.has_file("description") || req.has_param("description")) {
        JVal d;
        d.kind = JVal::Str;
        d.str = field(req, "description", "");
        root.obj["description"] = d;
    }
    const std::string meta = field(req, "metadata", "");
    if (!meta.empty()) {
        JVal m;
        if (!JParser(meta).parse(m) || m.kind != JVal::Obj) return "metadata must be a json object";
        root.obj["metadata"] = m;
    }
    return "";
}

// merges onto v, null or an empty string clears a field
static std::string apply_meta(const JVal & root, CachedVoice & v) {
    auto it = root.obj.find("description");
    if (it != root.obj.end()) {
        if (it->second.kind == JVal::Null) v.description.clear();
        else if (it->second.kind == JVal::Str) v.description = it->second.str;
        else return "description must be a string";
        if (v.description.size() > max_voice_description)
            return "description is longer than " + std::to_string(max_voice_description) + " bytes";
    }
    it = root.obj.find("metadata");
    if (it != root.obj.end() && it->second.kind != JVal::Null) {
        if (it->second.kind != JVal::Obj) return "metadata must be a json object";
        for (const auto & kv : it->second.obj) {
            if (!valid_voice_meta_key(kv.first))
                return "metadata keys are 1 to 64 letters, digits, dot, dash or underscore, and not description";
            if (kv.second.kind == JVal::Null || (kv.second.kind == JVal::Str && kv.second.str.empty())) {
                v.meta.erase(kv.first);
                continue;
            }
            if (kv.second.kind != JVal::Str) return "metadata values must be strings";
            if (kv.second.str.size() > max_voice_meta_value)
                return "metadata values are limited to " + std::to_string(max_voice_meta_value) + " bytes";
            v.meta[kv.first] = kv.second.str;
        }
        if (v.meta.size() > max_voice_meta_fields)
            return "a voice can have at most " + std::to_string(max_voice_meta_fields) + " metadata fields";
    }
    return "";
}

void VoiceStore::load_dir(const std::string & dir, int n_codebooks) {
    for (Voice & v : load_voice_dir(dir)) {
        if (v.n_codebooks != n_codebooks) {
            fprintf(stderr, "voice %s was made for a different model, skipping\n", v.name.c_str());
            continue;
        }
        CachedVoice cv;
        cv.codes = std::move(v.codes);
        cv.frames = v.frames;
        cv.text = v.text;
        cv.description = v.description;
        cv.meta = v.meta;
        cv.saved = true;
        m_voices[v.name] = std::move(cv);
        m_order.push_back(v.name);
    }
    if (!m_order.empty())
        printf("loaded %d saved voices from %s\n", (int) m_order.size(), dir.c_str());
}

bool VoiceStore::take(const std::string & id, std::vector<int> & codes, int & frames,
                      std::string & text) {
    std::lock_guard<std::mutex> vg(m_mutex);
    auto it = m_voices.find(id);
    if (it == m_voices.end()) return false;
    codes = it->second.codes;
    frames = it->second.frames;
    if (text.empty()) text = it->second.text;
    return true;
}

void VoiceStore::put(const std::string & id, CachedVoice v) {
    std::lock_guard<std::mutex> vg(m_mutex);
    if (!m_voices.count(id) && m_order.size() >= max_voices) {
        // saved voices are not evicted, they were asked for by name
        for (size_t i = 0; i < m_order.size(); i++) {
            if (m_voices[m_order[i]].saved) continue;
            m_voices.erase(m_order[i]);
            m_order.erase(m_order.begin() + i);
            break;
        }
    }
    const bool fresh = !m_voices.count(id);
    if (fresh) m_order.push_back(id);
    m_voices[id] = std::move(v);
}

void VoiceStore::add_routes(httplib::Server & svr, BreezeModel & model, MimiCodec & codec,
                            std::mutex & gpu, const std::string & dir) {
    const int sr = model.cfg.sample_rate;
    const int spf = model.cfg.samples_per_frame;
    const int ncb = model.cfg.num_codebooks;

    auto write_file = [dir, sr, ncb](const std::string & name, const CachedVoice & v) {
        Voice out;
        out.name = name;
        out.text = v.text;
        out.description = v.description;
        out.meta = v.meta;
        out.codes = v.codes;
        out.frames = v.frames;
        out.n_codebooks = ncb;
        out.sample_rate = sr;
        return save_voice(dir + "/" + name + ".breeze", out);
    };

    svr.Post("/v1/voices", [&, sr, spf, write_file](const httplib::Request & req, httplib::Response & res) {
        const std::string text = field(req, "ref_text", "");
        const std::string name = field(req, "name", "");
        if (!req.has_file("ref_audio") || text.empty())
            return fail(res, 400, "ref_audio and ref_text are required");
        if (!name.empty() && !valid_voice_name(name))
            return fail(res, 400, "name can only use letters, digits, dash and underscore");
        JVal meta;
        const std::string bad = read_meta_request(req, meta);
        if (!bad.empty()) return fail(res, 400, bad);

        const auto & f = req.get_file_value("ref_audio");
        // a named voice is written to disk under that name, an unnamed one just lives in memory
        const std::string id = name.empty() ? voice_key(f.content, text) : name;
        CachedVoice v;
        {
            std::lock_guard<std::mutex> vg(m_mutex);
            auto it = m_voices.find(id);
            if (it != m_voices.end()) {
                // saving over an existing name keeps its description unless a new one was sent
                v.description = it->second.description;
                v.meta = it->second.meta;
                if (name.empty()) {
                    CachedVoice next = it->second;
                    const std::string err = apply_meta(meta, next);
                    if (!err.empty()) return fail(res, 400, err);
                    it->second = std::move(next);
                    res.set_content(voice_json(id, it->second, sr, spf), "application/json");
                    return;
                }
            }
        }
        const std::string err = apply_meta(meta, v);
        if (!err.empty()) return fail(res, 400, err);

        std::vector<float> pcm;
        read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, pcm);
        if (pcm.empty()) return fail(res, 400, "could not read ref_audio");

        // encoding runs on the same device generation does, so it waits its turn
        std::unique_lock<std::mutex> lock(gpu, std::try_to_lock);
        if (!lock) return fail(res, 409, "busy");

        v.text = text;
        const auto t0 = std::chrono::steady_clock::now();
        v.codes = codec.encode(pcm, v.frames);
        v.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (!name.empty()) {
            if (!write_file(name, v)) return fail(res, 500, "could not write the voice file");
            v.saved = true;
        }
        const std::string body = voice_json(id, v, sr, spf);
        printf("voice %s cached%s, %d frames in %.0f ms\n", id.c_str(),
               v.saved ? " and saved" : "", v.frames, v.encode_ms);
        fflush(stdout);
        put(id, std::move(v));
        res.set_content(body, "application/json");
    });

    svr.Get("/v1/voices", [&, sr, spf](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> vg(m_mutex);
        std::string out = "[";
        for (size_t i = 0; i < m_order.size(); i++) {
            if (i) out += ",";
            out += voice_json(m_order[i], m_voices[m_order[i]], sr, spf);
        }
        res.set_content(out + "]", "application/json");
    });

    svr.Get(R"(/v1/voices/(.+))", [&, sr, spf](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> vg(m_mutex);
        auto it = m_voices.find(id);
        if (it == m_voices.end()) return fail(res, 404, "unknown voice_id");
        res.set_content(voice_json(id, it->second, sr, spf), "application/json");
    });

    svr.Patch(R"(/v1/voices/(.+))", [&, sr, spf, write_file](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        JVal meta;
        std::string err = read_meta_request(req, meta);
        if (!err.empty()) return fail(res, 400, err);
        std::lock_guard<std::mutex> vg(m_mutex);
        auto it = m_voices.find(id);
        if (it == m_voices.end()) return fail(res, 404, "unknown voice_id");
        CachedVoice next = it->second;
        err = apply_meta(meta, next);
        if (!err.empty()) return fail(res, 400, err);
        if (next.saved && !write_file(id, next)) return fail(res, 500, "could not write the voice file");
        it->second = std::move(next);
        printf("voice %s metadata updated%s\n", id.c_str(), it->second.saved ? " and saved" : "");
        fflush(stdout);
        res.set_content(voice_json(id, it->second, sr, spf), "application/json");
    });

    svr.Delete(R"(/v1/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> vg(m_mutex);
        auto it = m_voices.find(id);
        if (it == m_voices.end()) return fail(res, 404, "unknown voice_id");
        // the file stays put, deleting one over http would be a nasty way to lose a voice
        const bool saved = it->second.saved;
        m_voices.erase(it);
        m_order.erase(std::remove(m_order.begin(), m_order.end(), id), m_order.end());
        res.set_content(std::string("{\"deleted\":\"") + json_escape(id) + "\",\"file_kept\":" +
                        (saved ? "true" : "false") + "}", "application/json");
    });
}

} // namespace breeze
