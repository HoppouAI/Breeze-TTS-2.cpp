#pragma once

#include <memory>
#include <string>

namespace breeze {

// turns streamed float audio into one of the openai response formats, appending bytes to out
class AudioEncoder {
public:
    virtual ~AudioEncoder() = default;
    virtual void begin(std::string & out) { (void) out; }
    virtual void write(const float * s, int n, std::string & out) = 0;
    virtual void finish(std::string & out) { (void) out; }
};

// sent is the format really produced, anything we cant encode comes back as wav
std::unique_ptr<AudioEncoder> make_encoder(const std::string & format, int sr, std::string & sent,
                                           std::string & content_type);

} // namespace breeze
