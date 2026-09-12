#include "breeze/generation.h"
#include "breeze/sampling.h"
#include "breeze/text_encoder.h"

#include <algorithm>
#include <chrono>
#include <random>

namespace breeze {

struct Seg {
    bool is_text;
    std::vector<int> tokens;
    std::vector<int> codes; // frame-major, for audio segments
    int n_frames = 0;
    bool eos = false;
};

static Seg text_seg(BreezeModel & m, const std::string & s) {
    Seg seg;
    seg.is_text = true;
    seg.tokens = m.tok.encode(s, true);
    return seg;
}

static std::vector<Seg> build_segments(BreezeModel & m, const GenRequest & r, const std::string & text,
                                       bool has_ref, const std::string & ref_text,
                                       const std::vector<int> & ref_codes, int ref_T, bool cond) {
    std::vector<Seg> segs;
    const std::string spk = "[S0]";
    if (has_ref) {
        segs.push_back(text_seg(m, spk + ref_text));
        Seg a;
        a.is_text = false;
        a.codes = ref_codes;
        a.n_frames = ref_T;
        a.eos = true;
        segs.push_back(a);
    }
    std::string tail = cond ? spk + "<ins_bos>" + r.instruction + "<ins_eos>" + text : spk + text;
    segs.push_back(text_seg(m, tail));
    return segs;
}

static std::vector<float> assemble(BreezeModel & m, const std::vector<Seg> & segs, int & total) {
    const int H = m.cfg.hidden_size;
    std::vector<float> out;
    total = 0;
    for (const Seg & s : segs) {
        if (s.is_text) {
            std::vector<float> e = text_encoder_forward(m, s.tokens);
            out.insert(out.end(), e.begin(), e.end());
            total += (int) s.tokens.size();
        } else {
            std::vector<float> e = audio_embed_forward(m, s.codes, s.n_frames);
            out.insert(out.end(), e.begin(), e.end());
            total += s.n_frames;
            std::vector<int> eos_frame(m.cfg.num_codebooks, m.cfg.codebook_eos_token_id);
            std::vector<float> ee = audio_embed_forward(m, eos_frame, 1);
            out.insert(out.end(), ee.begin(), ee.end());
            total += 1;
        }
    }
    (void) H;
    return out;
}

static std::vector<float> combine_logits(const std::vector<float> & cond, const std::vector<float> & unc,
                                         bool use_cfg, float scale) {
    if (!use_cfg) return cond;
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = unc[i] + scale * (cond[i] - unc[i]);
    return out;
}

// what a finished piece leaves behind so the next one can keep the same voice
struct ChunkRef {
    std::vector<int> codes;
    int n_frames = 0;
    std::string text;
};

static bool generate_chunk(BreezeModel & m, MimiCodec & codec, const GenRequest & req,
                           const std::string & text, const ChunkRef & ref, uint32_t seed,
                           const AudioCallback & cb, GenTimings & tm,
                           std::chrono::steady_clock::time_point t_start, ChunkRef & out) {
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };

    std::mt19937 rng(seed);
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    const bool has_ref = !ref.codes.empty() && !ref.text.empty();
    const bool use_cfg = req.cfg_scale != 1.0f;

    const std::vector<int> & ref_codes = ref.codes;
    const int ref_T = ref.n_frames;

    auto t0 = clock_now();
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref_codes, ref_T, true), total_c);
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref_codes, ref_T, false), total_u);
    tm.prompt += since(t0);

    const int max_new = req.max_new_tokens > 0 ? req.max_new_tokens : m.cfg.max_new_tokens;

    BackboneState st_c, st_u;
    st_c.init(m, total_c + max_new + 8);
    if (use_cfg) st_u.init(m, total_u + max_new + 8);

    t0 = clock_now();
    StepOut o_c = backbone_run(m, st_c, emb_c, total_c);
    StepOut o_u;
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);
    tm.prefill += since(t0);

    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    SampleParams bp;
    bp.temperature = req.temperature > 0.0f ? req.temperature : m.cfg.temperature;
    bp.top_k = req.top_k > 0 ? req.top_k : m.cfg.top_k;
    bp.top_p = req.top_p > 0.0f ? req.top_p : m.cfg.top_p;
    bp.repetition_penalty = req.repetition_penalty > 0.0f ? req.repetition_penalty : m.cfg.repetition_penalty;
    std::vector<int> suppress;
    for (int t = m.cfg.codec_codebook_size; t < m.cfg.audio_vocab_size; t++) suppress.push_back(t);

    std::vector<int> hist;
    std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
    int cb0 = sample_token(comb, bp, rng, &hist, &suppress);

    std::vector<int> frames;
    int emitted = 0;
    bool stopped = false;
    // the first flush is small so audio starts early, then it grows to keep the vocoder efficient
    const int chunk_max = std::max(1, req.chunk_max);
    int chunk = std::min(std::max(1, req.chunk_first), chunk_max);
    // the transformer window plus the slack the vocoder convolutions reach back over
    const int ctx = m.cfg.voc.sliding_window + 16;
    auto flush = [&](bool final_flush) {
        const int have = (int) frames.size() / nc;
        while (have - emitted >= chunk || (final_flush && have > emitted)) {
            const int start = emitted;
            const int count = final_flush ? have - emitted : chunk;
            const int ctx_start = start > ctx ? start - ctx : 0;
            const int sub_T = start + count - ctx_start;
            std::vector<int> sub(frames.begin() + (size_t) ctx_start * nc, frames.begin() + (size_t) (start + count) * nc);
            const auto tv = clock_now();
            std::vector<float> audio = codec.decode(sub, sub_T);
            const double vtime = since(tv);
            tm.vocoder += vtime;
            tm.flushes++;
            const int skip = (start - ctx_start) * spf;
            if (!tm.first_audio) {
                tm.first_vocoder = vtime;
                tm.first_frames = sub_T;
                tm.first_audio = since(t_start);
            }
            if (!cb(audio.data() + skip, count * spf)) return false;
            emitted += count;
            chunk = std::min(chunk + chunk / 3 + 1, chunk_max);
            if (!final_flush && have - emitted < chunk) break;
        }
        return true;
    };

    for (int step = 0; step < max_new; step++) {
        if (cb0 == m.cfg.backbone_eos_token_id) break;
        std::vector<std::vector<float>> hiddens = { o_c.hidden };
        if (use_cfg) hiddens.push_back(o_u.hidden);
        auto td = clock_now();
        std::vector<int> depth_codes = depth.run(m, hiddens, cb0, req.cfg_scale, rng);
        tm.depth += since(td);
        std::vector<int> frame = { cb0 };
        frame.insert(frame.end(), depth_codes.begin(), depth_codes.end());

        bool pad = true;
        for (int c : frame) if (c != m.cfg.codebook_pad_token_id) { pad = false; break; }
        if (!pad) {
            frames.insert(frames.end(), frame.begin(), frame.end());
            tm.frames++;
            if (!flush(false)) { stopped = true; break; }
        }
        hist.push_back(cb0);

        auto tb = clock_now();
        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        if (use_cfg) {
            auto pair = backbone_run_cfg(m, st_c, st_u, ae);
            o_c = std::move(pair[0]);
            o_u = std::move(pair[1]);
        } else {
            o_c = backbone_run(m, st_c, ae, 1);
        }
        tm.backbone += since(tb);
        comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
        cb0 = sample_token(comb, bp, rng, &hist, &suppress);
    }
    if (!stopped) stopped = !flush(true);

    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();

    out.codes = std::move(frames);
    out.n_frames = (int) out.codes.size() / nc;
    out.text = text;
    return !stopped;
}

void GenSession::begin(BreezeModel & m, MimiCodec & codec, const GenRequest & req, GenTimings * tm) {
    m_model = &m;
    m_codec = &codec;
    m_req = req;
    m_piece = 0;
    m_start = std::chrono::steady_clock::now();
    m_codes.clear();
    m_frames = 0;
    m_text.clear();

    if (!req.ref_codes.empty() && req.ref_frames > 0 && !req.ref_text.empty()) {
        m_codes = req.ref_codes;
        m_frames = req.ref_frames;
        m_text = req.ref_text;
    } else if (!req.ref_audio.empty() && !req.ref_text.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        m_codes = codec.encode(req.ref_audio, m_frames);
        m_text = req.ref_text;
        if (tm) tm->encode_ref =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
}

bool GenSession::speak(const std::string & text, const AudioCallback & cb, GenTimings * tm) {
    if (!m_model || !m_codec) return false;
    GenTimings sink;
    GenTimings & t = tm ? *tm : sink;

    ChunkRef anchor;
    anchor.codes = m_codes;
    anchor.n_frames = m_frames;
    anchor.text = m_text;

    ChunkRef made;
    const bool ok = generate_chunk(*m_model, *m_codec, m_req, text, anchor,
                                   (uint32_t) m_req.seed + m_piece, cb, t, m_start, made);
    m_piece++;
    // the opening piece stands in as the reference when there was no clip to clone
    if (ok && m_codes.empty() && made.n_frames > 0) {
        m_codes = std::move(made.codes);
        m_frames = made.n_frames;
        m_text = made.text;
    }
    return ok;
}

// long text is generated piece by piece, each one carrying the same reference so the voice does
// not change at the seams
void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings) {
    GenTimings sink;
    GenTimings & tm = timings ? *timings : sink;

    GenSession s;
    s.begin(m, codec, req, &tm);

    // a half minute of reference makes the model skip whole sentences of whatever comes next, so
    // when the first piece has to double as the reference it stays near the usual clip length
    const int anchor_chars = 200;
    const std::vector<std::string> parts =
        split_text(req.text, req.split_chars, s.needs_anchor() ? anchor_chars : 0);

    for (const std::string & part : parts)
        if (!s.speak(part, cb, &tm)) return;
}

// keeps the source's semantic codes and rebuilds the acoustic ones in the reference voice. the words
// and their timing survive, the pitch contour does not, it gets replaced by the reference's own
// the backbone goes degenerate with nothing to read, and forcing codes against that state comes out
// mumbled. what the filler says does not matter, only that there is roughly a clip's worth of it
static std::string filler_text(double secs) {
    static const char * lines[] = {
        "This is a recording of ordinary speech made in a quiet room. ",
        "The words themselves do not matter very much at all here. ",
        "It simply carries on for a little while longer than that. ",
        "Nothing in particular is being described at this point. ",
    };
    const size_t want = (size_t) (secs * 17.0) + 16;
    std::string s;
    for (int i = 0; s.size() < want; i++) s += lines[i % 4];
    return s;
}

std::vector<float> convert_voice(BreezeModel & m, MimiCodec & codec, const std::vector<int> & src_codes,
                                 int src_T, const std::vector<float> & ref_audio,
                                 const std::string & ref_text, const ConvertOptions & opt) {
    const int nc = m.cfg.num_codebooks;
    const bool use_cfg = opt.cfg_scale != 1.0f;
    std::mt19937 rng((uint32_t) opt.seed);

    SampleParams sp;
    sp.temperature = opt.temperature;
    sp.top_k = opt.top_k;

    int ref_T = 0;
    std::vector<int> ref_codes;
    if (!opt.ref_codes.empty() && opt.ref_frames > 0) {
        ref_codes = opt.ref_codes;
        ref_T = opt.ref_frames;
    } else {
        ref_codes = codec.encode(ref_audio, ref_T);
    }

    const std::string text =
        opt.src_text.empty()
            ? filler_text(src_T * (double) m.cfg.samples_per_frame / m.cfg.sample_rate)
            : opt.src_text;
    GenRequest req;
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c =
        assemble(m, build_segments(m, req, text, true, ref_text, ref_codes, ref_T, false), total_c);
    // the negative branch drops the reference, so guidance pushes toward the target voice
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, text, false, "", {}, 0, false), total_u);

    BackboneState st_c, st_u;
    st_c.init(m, total_c + src_T + 8);
    if (use_cfg) st_u.init(m, total_u + src_T + 8);
    StepOut o_c = backbone_run(m, st_c, emb_c, total_c);
    StepOut o_u;
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);

    DepthRunner depth;
    depth.init(m, use_cfg ? 2 : 1);

    std::vector<int> out((size_t) src_T * nc);
    const int keep = opt.keep_acoustic < nc - 1 ? opt.keep_acoustic : nc - 1;
    for (int t = 0; t < src_T; t++) {
        const int cb0 = src_codes[(size_t) t * nc];
        std::vector<std::vector<float>> hiddens = { o_c.hidden };
        if (use_cfg) hiddens.push_back(o_u.hidden);
        std::vector<int> rest =
            depth.run(m, hiddens, cb0, opt.cfg_scale, rng, &sp,
                      keep > 0 ? &src_codes[(size_t) t * nc + 1] : nullptr, keep);
        out[(size_t) t * nc] = cb0;
        for (int c = 1; c < nc; c++) out[(size_t) t * nc + c] = rest[c - 1];

        const int * from = opt.feed_source ? &src_codes[(size_t) t * nc] : &out[(size_t) t * nc];
        std::vector<int> frame(from, from + nc);
        std::vector<float> ae = audio_embed_forward(m, frame, 1);
        if (use_cfg) {
            auto pair = backbone_run_cfg(m, st_c, st_u, ae);
            o_c = std::move(pair[0]);
            o_u = std::move(pair[1]);
        } else {
            o_c = backbone_run(m, st_c, ae, 1);
        }
        if (t % 25 == 0) { printf("\rconverting %d/%d frames", t, src_T); fflush(stdout); }
    }
    printf("\rconverted %d frames        \n", src_T);

    st_c.free();
    if (use_cfg) st_u.free();
    depth.free();

    // the vocoder upsamples 1920x, so decoding a long clip in one graph asks for gigabytes at once.
    // walk it in windows with enough left context for the convolutions to reach back over
    const int spf = m.cfg.samples_per_frame;
    const int ctx = m.cfg.voc.sliding_window + 16;
    const int step = 40;
    std::vector<float> audio;
    audio.reserve((size_t) src_T * spf);
    for (int start = 0; start < src_T; start += step) {
        const int count = std::min(step, src_T - start);
        const int cs = start > ctx ? start - ctx : 0;
        std::vector<int> sub(out.begin() + (size_t) cs * nc, out.begin() + (size_t) (start + count) * nc);
        std::vector<float> part = codec.decode(sub, start + count - cs);
        const size_t skip = (size_t) (start - cs) * spf;
        const size_t want = (size_t) count * spf;
        if (part.size() < skip + want) break;
        audio.insert(audio.end(), part.begin() + skip, part.begin() + skip + want);
    }
    return audio;
}

}
