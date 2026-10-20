#pragma once

#include "breeze/codec.h"
#include "breeze/model.h"

#include "httplib.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

// form fields arrive as either a file part or a plain param depending on the client
inline std::string field(const httplib::Request & req, const char * name, const std::string & def) {
    if (req.has_file(name)) return req.get_file_value(name).content;
    if (req.has_param(name)) return req.get_param_value(name);
    return def;
}

// hash of the clip bytes and its transcript; the id unnamed reference clips are cached under
std::string voice_key(const std::string & wav, const std::string & text);

struct CachedVoice {
    std::vector<int> codes;
    int frames = 0;
    std::string text;
    std::string description;
    std::map<std::string, std::string> meta;
    double encode_ms = 0;
    bool saved = false; // has a .breeze file behind it, so it survives a restart
};

// reference voices the server already encoded, both the ones posted at runtime and the files on disk
class VoiceStore {
public:
    void load_dir(const std::string & dir, int n_codebooks);

    // fills in a cached reference, false only when the id is unknown
    bool take(const std::string & id, std::vector<int> & codes, int & frames, std::string & text);

    // in memory insert, unnamed entries get evicted oldest first when the table is full
    void put(const std::string & id, CachedVoice v);

    void add_routes(httplib::Server & svr, BreezeModel & model, MimiCodec & codec, std::mutex & gpu,
                    const std::string & dir);

private:
    std::map<std::string, CachedVoice> m_voices;
    std::vector<std::string> m_order;
    std::mutex m_mutex;
    static const size_t max_voices = 64;
};

} // namespace breeze
