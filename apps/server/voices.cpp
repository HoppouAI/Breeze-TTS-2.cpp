#include "voices.h"

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
             id.c_str(), v.frames, (double) v.frames * spf / sr, v.encode_ms, v.saved ? "true" : "false");
    return std::string(buf) + json_escape(v.text) + "\"}";
}

static void fail(httplib::Response & res, int status, const char * msg) {
    res.status = status;
    res.set_content(std::string("{\"error\":\"") + msg + "\"}", "application/json");
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

    svr.Post("/v1/voices", [&, sr, spf, dir](const httplib::Request & req, httplib::Response & res) {
        const std::string text = field(req, "ref_text", "");
        const std::string name = field(req, "name", "");
        if (!req.has_file("ref_audio") || text.empty())
            return fail(res, 400, "ref_audio and ref_text are required");
        if (!name.empty() && !valid_voice_name(name))
            return fail(res, 400, "name can only use letters, digits, dash and underscore");

        const auto & f = req.get_file_value("ref_audio");
        // a named voice is written to disk under that name, an unnamed one just lives in memory
        const std::string id = name.empty() ? voice_key(f.content, text) : name;
        if (name.empty()) {
            std::lock_guard<std::mutex> vg(m_mutex);
            auto it = m_voices.find(id);
            if (it != m_voices.end()) {
                res.set_content(voice_json(id, it->second, sr, spf), "application/json");
                return;
            }
        }
        std::vector<float> pcm;
        read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, pcm);
        if (pcm.empty()) return fail(res, 400, "could not read ref_audio");

        // encoding runs on the same device generation does, so it waits its turn
        std::unique_lock<std::mutex> lock(gpu, std::try_to_lock);
        if (!lock) return fail(res, 409, "busy");

        CachedVoice v;
        v.text = text;
        const auto t0 = std::chrono::steady_clock::now();
        v.codes = codec.encode(pcm, v.frames);
        v.encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (!name.empty()) {
            Voice out;
            out.name = name;
            out.text = text;
            out.codes = v.codes;
            out.frames = v.frames;
            out.n_codebooks = model.cfg.num_codebooks;
            out.sample_rate = sr;
            if (!save_voice(dir + "/" + name + ".breeze", out))
                return fail(res, 500, "could not write the voice file");
            v.saved = true;
        }
        put(id, std::move(v));
        printf("voice %s cached%s, %d frames in %.0f ms\n", id.c_str(),
               v.saved ? " and saved" : "", v.frames, v.encode_ms);
        fflush(stdout);
        res.set_content(voice_json(id, v, sr, spf), "application/json");
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

    svr.Delete(R"(/v1/voices/(.+))", [&](const httplib::Request & req, httplib::Response & res) {
        const std::string id = req.matches[1];
        std::lock_guard<std::mutex> vg(m_mutex);
        auto it = m_voices.find(id);
        if (it == m_voices.end()) return fail(res, 404, "unknown voice_id");
        // the file stays put, deleting one over http would be a nasty way to lose a voice
        const bool saved = it->second.saved;
        m_voices.erase(it);
        m_order.erase(std::remove(m_order.begin(), m_order.end(), id), m_order.end());
        res.set_content(std::string("{\"deleted\":\"") + id + "\",\"file_kept\":" +
                        (saved ? "true" : "false") + "}", "application/json");
    });
}

} // namespace breeze
