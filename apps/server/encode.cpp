#include "encode.h"

#include "breeze/audio.h"

#include <algorithm>
#include <cmath>
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
        // chrome and edge reject shine's mpeg-2 frames as malformed, so stay at mpeg-1 rates
        m_up = 1;
        while (sr * m_up < 32000) m_up++;
        m_rate = sr * m_up;
        build_filter();
        shine_config_t cfg;
        cfg.wave.channels = PCM_MONO;
        cfg.wave.samplerate = m_rate;
        shine_set_config_mpeg_defaults(&cfg.mpeg);
        cfg.mpeg.mode = MONO;
        cfg.mpeg.bitr = BITRATE;
        if (shine_check_config(m_rate, cfg.mpeg.bitr) < 0) return false;
        m_enc = shine_initialise(&cfg);
        if (!m_enc) return false;
        m_pass = shine_samples_per_pass(m_enc);
        return true;
    }

    int sample_rate(int) const override { return m_rate; }

    void write(const float * s, int n, std::string & out) override {
        for (int i = 0; i < n; i++) push(s[i]);
        drain(out);
    }

    void finish(std::string & out) override {
        // the filter still holds the end of the clip
        for (int i = 0; i < TAPS / 2; i++) push(0.0f);
        drain(out);
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
    static constexpr int TAPS = 32;
    static constexpr int BITRATE = 80;

    // windowed sinc per output phase, phase 0 is the input sample itself
    void build_filter() {
        const double pi = 3.14159265358979323846;
        m_fir.resize((size_t) m_up * TAPS);
        m_hist.assign(TAPS, 0.0f);
        for (int p = 0; p < m_up; p++) {
            double sum = 0.0;
            for (int k = 0; k < TAPS; k++) {
                const double x = k - (TAPS / 2 - 1) - (double) p / m_up;
                const double u = x / (TAPS / 2);
                const double win = 0.42 + 0.5 * std::cos(pi * u) + 0.08 * std::cos(2.0 * pi * u);
                const double h = (x == 0.0 ? 1.0 : std::sin(pi * x) / (pi * x)) * win;
                m_fir[(size_t) p * TAPS + k] = (float) h;
                sum += h;
            }
            for (int k = 0; k < TAPS; k++) m_fir[(size_t) p * TAPS + k] /= (float) sum;
        }
    }

    void push(float in) {
        std::copy(m_hist.begin() + 1, m_hist.end(), m_hist.begin());
        m_hist[TAPS - 1] = std::max(-1.0f, std::min(1.0f, in));
        for (int p = 0; p < m_up; p++) {
            const float * h = &m_fir[(size_t) p * TAPS];
            float v = 0.0f;
            for (int k = 0; k < TAPS; k++) v += h[k] * m_hist[k];
            // shine decodes 9/8 louder than it was fed at every rate, so loud peaks would clip
            v = std::max(-1.0f, std::min(1.0f, v * (8.0f / 9.0f)));
            m_buf.push_back((int16_t) (v * 32767.0f));
        }
    }

    void drain(std::string & out) {
        size_t at = 0;
        while (m_buf.size() - at >= (size_t) m_pass) {
            encode(m_buf.data() + at, out);
            at += (size_t) m_pass;
        }
        m_buf.erase(m_buf.begin(), m_buf.begin() + (std::ptrdiff_t) at);
    }

    void encode(int16_t * pcm, std::string & out) {
        int written = 0;
        const unsigned char * data = shine_encode_buffer_interleaved(m_enc, pcm, &written);
        if (data && written > 0) out.append((const char *) data, (size_t) written);
    }

    shine_t m_enc = nullptr;
    int m_pass = 0;
    int m_up = 1;
    int m_rate = 0;
    std::vector<float> m_fir;
    std::vector<float> m_hist;
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
