#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace breeze {

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

} // namespace breeze
