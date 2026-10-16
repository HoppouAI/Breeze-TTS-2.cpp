#include "openai.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

namespace breeze {

namespace {

struct JVal {
    enum Kind { Null, Bool, Num, Str, Obj, Arr } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::map<std::string, JVal> obj;
};

// python sends non ascii as \u escapes by default, so this has to decode them properly
class JParser {
public:
    explicit JParser(const std::string & s) : m_s(s) {}

    bool parse(JVal & v) {
        ws();
        if (!value(v, 0)) return false;
        ws();
        return m_i == m_s.size();
    }

private:
    bool more() const { return m_i < m_s.size(); }

    void ws() {
        while (more() && (m_s[m_i] == ' ' || m_s[m_i] == '\t' || m_s[m_i] == '\n' || m_s[m_i] == '\r')) m_i++;
    }

    bool eat(char c) {
        if (!more() || m_s[m_i] != c) return false;
        m_i++;
        return true;
    }

    bool lit(const char * w) {
        const size_t n = strlen(w);
        if (m_s.compare(m_i, n, w) != 0) return false;
        m_i += n;
        return true;
    }

    bool value(JVal & v, int depth) {
        if (depth > 32 || !more()) return false;
        const char c = m_s[m_i];
        if (c == '{') return object(v, depth);
        if (c == '[') return array(v, depth);
        if (c == '"') {
            v.kind = JVal::Str;
            return string(v.str);
        }
        if (lit("true")) { v.kind = JVal::Bool; v.b = true; return true; }
        if (lit("false")) { v.kind = JVal::Bool; v.b = false; return true; }
        if (lit("null")) { v.kind = JVal::Null; return true; }
        return number(v);
    }

    bool object(JVal & v, int depth) {
        v.kind = JVal::Obj;
        m_i++;
        ws();
        if (eat('}')) return true;
        for (;;) {
            ws();
            std::string key;
            if (!more() || m_s[m_i] != '"' || !string(key)) return false;
            ws();
            if (!eat(':')) return false;
            ws();
            JVal child;
            if (!value(child, depth + 1)) return false;
            v.obj[key] = std::move(child);
            ws();
            if (eat(',')) continue;
            return eat('}');
        }
    }

    bool array(JVal & v, int depth) {
        v.kind = JVal::Arr;
        m_i++;
        ws();
        if (eat(']')) return true;
        for (;;) {
            ws();
            JVal child;
            if (!value(child, depth + 1)) return false;
            ws();
            if (eat(',')) continue;
            return eat(']');
        }
    }

    bool number(JVal & v) {
        const char * start = m_s.c_str() + m_i;
        char * end = nullptr;
        v.num = strtod(start, &end);
        if (end == start) return false;
        m_i += (size_t) (end - start);
        v.kind = JVal::Num;
        return true;
    }

    bool hex4(uint32_t & cp) {
        if (m_i + 4 > m_s.size()) return false;
        cp = 0;
        for (int k = 0; k < 4; k++) {
            const char h = m_s[m_i++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (uint32_t) (h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (uint32_t) (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (uint32_t) (h - 'A' + 10);
            else return false;
        }
        return true;
    }

    static void utf8(uint32_t cp, std::string & out) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xc0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3f));
        } else if (cp < 0x10000) {
            out += (char) (0xe0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3f));
            out += (char) (0x80 | (cp & 0x3f));
        } else {
            out += (char) (0xf0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3f));
            out += (char) (0x80 | ((cp >> 6) & 0x3f));
            out += (char) (0x80 | (cp & 0x3f));
        }
    }

    bool string(std::string & out) {
        m_i++;
        while (more()) {
            const char c = m_s[m_i++];
            if (c == '"') return true;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (!more()) return false;
            const char e = m_s[m_i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xd800 && cp < 0xdc00) {
                        uint32_t lo = 0;
                        if (m_s.compare(m_i, 2, "\\u") != 0) return false;
                        m_i += 2;
                        if (!hex4(lo) || lo < 0xdc00 || lo >= 0xe000) return false;
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                    } else if (cp >= 0xdc00 && cp < 0xe000) {
                        return false;
                    }
                    utf8(cp, out);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    const std::string & m_s;
    size_t m_i = 0;
};

const JVal * get(const JVal & o, const char * key) {
    const auto it = o.obj.find(key);
    return it == o.obj.end() ? nullptr : &it->second;
}

bool str_field(const JVal & o, const char * key, std::string & out) {
    const JVal * v = get(o, key);
    if (!v || v->kind == JVal::Null) return true;
    if (v->kind != JVal::Str) return false;
    out = v->str;
    return true;
}

// loosely typed clients sometimes quote their numbers, so take either
bool num_field(const JVal & o, const char * key, double & out) {
    const JVal * v = get(o, key);
    if (!v || v->kind == JVal::Null) return true;
    if (v->kind == JVal::Num) {
        out = v->num;
        return true;
    }
    if (v->kind != JVal::Str || v->str.empty()) return false;
    char * end = nullptr;
    const double d = strtod(v->str.c_str(), &end);
    if (*end != '\0') return false;
    out = d;
    return true;
}

std::string esc(const std::string & s) {
    std::string o;
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if ((unsigned char) c < 0x20) {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", (unsigned) c);
            o += buf;
        } else {
            o += c;
        }
    }
    return o;
}

std::string base64(const std::string & in) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = ((uint32_t) (uint8_t) in[i] << 16) | ((uint32_t) (uint8_t) in[i + 1] << 8) |
                           (uint8_t) in[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = (uint32_t) (uint8_t) in[i] << 16;
        if (i + 1 < in.size()) v |= (uint32_t) (uint8_t) in[i + 1] << 8;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < in.size() ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

} // namespace

std::string parse_openai_speech(const std::string & body, VoiceStore & store, int split_chars,
                                OpenAiSpeech & out, int & status) {
    status = 400;
    JVal root;
    if (!JParser(body).parse(root) || root.kind != JVal::Obj) return "request body must be a json object";

    GenRequest & g = out.g;
    if (!str_field(root, "input", g.text)) return "input must be a string";
    if (g.text.empty()) return "input is required";

    std::string ins;
    if (!str_field(root, "instructions", ins)) return "instructions must be a string";
    if (!ins.empty()) g.instruction = ins;

    // newer clients send the voice as {"id": ...}
    const JVal * v = get(root, "voice");
    if (v && v->kind == JVal::Obj) {
        if (!str_field(*v, "id", out.voice)) return "voice id must be a string";
    } else if (!str_field(root, "voice", out.voice)) {
        return "voice must be a string";
    }
    // sillytavern splits its voice list on bare commas, so "a, b" sends " b"
    const size_t vb = out.voice.find_first_not_of(" \t"), ve = out.voice.find_last_not_of(" \t");
    out.voice = vb == std::string::npos ? "" : out.voice.substr(vb, ve - vb + 1);
    if (!out.voice.empty() && !store.take(out.voice, g.ref_codes, g.ref_frames, g.ref_text)) {
        status = 404;
        return "voice '" + out.voice + "' is not a saved voice. register one with POST /v1/voices, "
               "or leave voice out to design one from instructions";
    }

    if (!str_field(root, "response_format", out.format)) return "response_format must be a string";
    if (out.format.empty()) out.format = "mp3";
    static const char * formats[] = { "mp3", "opus", "aac", "flac", "wav", "pcm" };
    bool known = false;
    for (const char * f : formats) known = known || out.format == f;
    if (!known) return "response_format must be one of mp3, opus, aac, flac, wav or pcm";

    std::string stream;
    if (!str_field(root, "stream_format", stream)) return "stream_format must be a string";
    if (stream == "sse") out.sse = true;
    else if (!stream.empty() && stream != "audio") return "stream_format must be audio or sse";

    // the native knobs ride along as extra fields, model and speed are accepted and ignored
    double cfg = g.cfg_scale, seed = g.seed, temp = 0, top_k = 0, top_p = 0, rep = 0, max_tok = 0;
    double split = split_chars;
    const std::pair<const char *, double *> nums[] = {
        { "cfg_scale", &cfg }, { "seed", &seed }, { "temperature", &temp }, { "top_k", &top_k },
        { "top_p", &top_p }, { "repetition_penalty", &rep }, { "max_new_tokens", &max_tok },
        { "split_chars", &split },
    };
    for (const auto & n : nums) {
        if (!num_field(root, n.first, *n.second)) return std::string(n.first) + " must be a number";
    }
    g.cfg_scale = (float) cfg;
    g.seed = (int) seed;
    g.temperature = (float) temp;
    g.top_k = (int) top_k;
    g.top_p = (float) top_p;
    g.repetition_penalty = (float) rep;
    g.max_new_tokens = (int) max_tok;
    g.split_chars = (int) split;

    status = 200;
    return "";
}

std::string openai_error(const std::string & msg, const char * type) {
    return "{\"error\":{\"message\":\"" + esc(msg) + "\",\"type\":\"" + type + "\",\"param\":null,\"code\":null}}";
}

std::string sse_audio_delta(const std::string & bytes) {
    return "data: {\"type\":\"speech.audio.delta\",\"audio\":\"" + base64(bytes) + "\"}\n\n";
}

std::string sse_audio_done() {
    return "data: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":0,\"output_tokens\":0,"
           "\"total_tokens\":0}}\n\n";
}

} // namespace breeze
