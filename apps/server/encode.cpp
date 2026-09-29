#include "encode.h"

#include "breeze/audio.h"

#include <cstdint>
#include <vector>

#ifdef BREEZE_HAVE_MP3
extern "C" {
#include "shine/src/lib/layer3.h"
}
#endif

namespace breeze {

static void put16(std::string & o, uint16_t v) {
    o += (char) (v & 0xff);
    o += (char) (v >> 8);
}

static void put32(std::string & o, uint32_t v) {
    put16(o, (uint16_t) (v & 0xffff));
    put16(o, (uint16_t) (v >> 16));
}

static void append_pcm(const float * s, int n, std::string & out) {
    const std::vector<uint8_t> pcm = to_pcm16(s, n);
    out.append((const char *) pcm.data(), pcm.size());
}

class PcmEncoder : public AudioEncoder {
public:
    void write(const float * s, int n, std::string & out) override { append_pcm(s, n, out); }
};

class WavEncoder : public AudioEncoder {
public:
    explicit WavEncoder(int sr) : m_sr(sr) {}

    void begin(std::string & out) override {
        // length isnt known up front, max sizes is what streamed wav does and players read to eof
        out += "RIFF";
        put32(out, 0xffffffffu);
        out += "WAVEfmt ";
        put32(out, 16);
        put16(out, 1);
        put16(out, 1);
        put32(out, (uint32_t) m_sr);
        put32(out, (uint32_t) m_sr * 2);
        put16(out, 2);
        put16(out, 16);
        out += "data";
        put32(out, 0xffffffffu);
    }

    void write(const float * s, int n, std::string & out) override { append_pcm(s, n, out); }

private:
    int m_sr;
};

#ifdef BREEZE_HAVE_MP3
class Mp3Encoder : public AudioEncoder {
public:
    ~Mp3Encoder() override {
        if (m_enc) shine_close(m_enc);
    }

    bool init(int sr) {
        shine_config_t cfg;
        cfg.wave.channels = PCM_MONO;
        cfg.wave.samplerate = sr;
        shine_set_config_mpeg_defaults(&cfg.mpeg);
        cfg.mpeg.mode = MONO;
        cfg.mpeg.bitr = 64;
        if (shine_check_config(sr, cfg.mpeg.bitr) < 0) return false;
        m_enc = shine_initialise(&cfg);
        if (!m_enc) return false;
        m_pass = shine_samples_per_pass(m_enc);
        return true;
    }

    void write(const float * s, int n, std::string & out) override {
        for (int i = 0; i < n; i++) {
            float v = s[i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            // shine decodes 9/8 louder than it was fed at every rate, so loud peaks would clip
            m_buf.push_back((int16_t) (v * 32767.0f * (8.0f / 9.0f)));
        }
        size_t at = 0;
        while (m_buf.size() - at >= (size_t) m_pass) {
            encode(m_buf.data() + at, out);
            at += (size_t) m_pass;
        }
        m_buf.erase(m_buf.begin(), m_buf.begin() + (std::ptrdiff_t) at);
    }

    void finish(std::string & out) override {
        if (!m_buf.empty()) {
            m_buf.resize((size_t) m_pass, 0);
            encode(m_buf.data(), out);
            m_buf.clear();
        }
        int written = 0;
        const unsigned char * data = shine_flush(m_enc, &written);
        if (data && written > 0) out.append((const char *) data, (size_t) written);
    }

private:
    void encode(int16_t * pcm, std::string & out) {
        int written = 0;
        const unsigned char * data = shine_encode_buffer_interleaved(m_enc, pcm, &written);
        if (data && written > 0) out.append((const char *) data, (size_t) written);
    }

    shine_t m_enc = nullptr;
    int m_pass = 0;
    std::vector<int16_t> m_buf;
};
#endif

std::unique_ptr<AudioEncoder> make_encoder(const std::string & format, int sr, std::string & sent,
                                           std::string & content_type) {
    if (format == "pcm") {
        sent = "pcm";
        content_type = "audio/pcm";
        return std::make_unique<PcmEncoder>();
    }
#ifdef BREEZE_HAVE_MP3
    if (format == "mp3") {
        auto enc = std::make_unique<Mp3Encoder>();
        if (enc->init(sr)) {
            sent = "mp3";
            content_type = "audio/mpeg";
            return enc;
        }
    }
#endif
    sent = "wav";
    content_type = "audio/wav";
    return std::make_unique<WavEncoder>(sr);
}

} // namespace breeze
