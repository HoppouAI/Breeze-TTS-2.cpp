#include "breeze/audio.h"

#include <cstdio>
#include <cstring>

namespace breeze {

static uint32_t rd_u32(const uint8_t * p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t) p[3] << 24; }
static uint16_t rd_u16(const uint8_t * p) { return p[0] | p[1] << 8; }

std::vector<float> resample_linear(const std::vector<float> & in, int in_sr, int out_sr) {
    if (in_sr == out_sr || in.empty()) return in;
    const double ratio = (double) out_sr / in_sr;
    const int n = (int) (in.size() * ratio);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) {
        double src = i / ratio;
        int i0 = (int) src;
        int i1 = i0 + 1 < (int) in.size() ? i0 + 1 : i0;
        float t = (float) (src - i0);
        out[i] = in[i0] * (1.0f - t) + in[i1] * t;
    }
    return out;
}

bool read_wav(const std::string & path, int target_sr, std::vector<float> & out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(size);
    if (fread(buf.data(), 1, size, f) != (size_t) size) { fclose(f); return false; }
    fclose(f);
    return read_wav_buffer(buf.data(), buf.size(), target_sr, out);
}

bool read_wav_buffer(const uint8_t * buf, size_t size, int target_sr, std::vector<float> & out) {
    if (size < 44 || std::memcmp(buf, "RIFF", 4) != 0 || std::memcmp(buf + 8, "WAVE", 4) != 0) return false;

    bool have_fmt = false;
    int channels = 0, bits = 0, fmt = 0;
    uint32_t sr = 0;
    size_t pos = 12, data_off = 0, data_len = 0;
    while (size - pos >= 8) {
        const uint8_t * ch = &buf[pos];
        const size_t clen = rd_u32(ch + 4);
        const size_t left = size - pos - 8;
        if (std::memcmp(ch, "fmt ", 4) == 0 && clen >= 16 && left >= clen) {
            have_fmt = true;
            fmt = rd_u16(ch + 8);
            channels = rd_u16(ch + 10);
            sr = rd_u32(ch + 12);
            bits = rd_u16(ch + 22);
            // extensible keeps the real format code at the start of its guid
            if (fmt == 0xfffe && clen >= 26) fmt = rd_u16(ch + 32);
        } else if (std::memcmp(ch, "data", 4) == 0) {
            data_off = pos + 8;
            // streamed wavs say max length here, so the bytes we actually got are what count
            data_len = clen < left ? clen : left;
        }
        if (clen >= left) break;
        pos += 8 + clen + (clen & 1);
    }
    if (!data_off || !have_fmt || channels < 1) return false;
    if (sr < 1000 || sr > 768000) return false;
    const bool is_int = fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
    const bool is_flt = fmt == 3 && bits == 32;
    if (!is_int && !is_flt) return false;

    std::vector<float> mono;
    const uint8_t * d = &buf[data_off];
    const int bytes = bits / 8;
    const size_t frames = data_len / ((size_t) bytes * channels);
    mono.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        float acc = 0.0f;
        for (int c = 0; c < channels; c++) {
            const uint8_t * s = d + (i * channels + c) * bytes;
            float v = 0.0f;
            if (fmt == 3 && bits == 32) std::memcpy(&v, s, 4);
            else if (bits == 8) v = ((int) s[0] - 128) / 128.0f;
            else if (bits == 16) v = (int16_t) rd_u16(s) / 32768.0f;
            else if (bits == 24) v = (int32_t) ((uint32_t) s[0] << 8 | (uint32_t) s[1] << 16 | (uint32_t) s[2] << 24) / 2147483648.0f;
            else if (bits == 32) v = (int32_t) rd_u32(s) / 2147483648.0f;
            acc += v;
        }
        mono[i] = acc / channels;
    }
    out = resample_linear(mono, (int) sr, target_sr);
    return true;
}

std::vector<uint8_t> to_pcm16(const float * s, int n) {
    std::vector<uint8_t> out((size_t) n * 2);
    for (int i = 0; i < n; i++) {
        float v = s[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t q = (int16_t) (v * 32767.0f);
        out[i * 2] = q & 0xff;
        out[i * 2 + 1] = (q >> 8) & 0xff;
    }
    return out;
}

bool write_wav(const std::string & path, const std::vector<float> & samples, int sr) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    std::vector<uint8_t> pcm = to_pcm16(samples.data(), (int) samples.size());
    uint32_t data_len = (uint32_t) pcm.size();
    uint32_t riff = 36 + data_len;
    auto w32 = [&](uint32_t v) { uint8_t b[4] = { (uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24) }; fwrite(b, 1, 4, f); };
    auto w16 = [&](uint16_t v) { uint8_t b[2] = { (uint8_t) v, (uint8_t) (v >> 8) }; fwrite(b, 1, 2, f); };
    fwrite("RIFF", 1, 4, f); w32(riff); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32(sr); w32(sr * 2); w16(2); w16(16);
    fwrite("data", 1, 4, f); w32(data_len);
    fwrite(pcm.data(), 1, pcm.size(), f);
    fclose(f);
    return true;
}

}
