#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace breeze {

// a saved reference voice, the codes skip the encode which is most of the wait before first audio
struct Voice {
    std::string name;
    std::string text; // exact transcript of the clip the codes came from
    std::string description;
    std::map<std::string, std::string> meta;
    std::vector<int> codes;
    int frames = 0;
    int n_codebooks = 0;
    int sample_rate = 24000;
};

constexpr size_t max_voice_description = 4096;
constexpr size_t max_voice_meta_fields = 32;
constexpr size_t max_voice_meta_value = 1024;

bool save_voice(const std::string & path, const Voice & v);
bool load_voice(const std::string & path, Voice & v);

// every .breeze in dir, named after the file. missing or unreadable dirs just come back empty
std::vector<Voice> load_voice_dir(const std::string & dir);

// filenames come from requests, so keep them to something that cannot walk out of the folder
bool valid_voice_name(const std::string & name);

bool valid_voice_meta_key(const std::string & key);

} // namespace breeze
