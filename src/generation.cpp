#include "breeze/generation.h"
#include "breeze/sampling.h"
#include "breeze/scheduler.h"
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

DepthRunner & GenPool::runner(BreezeModel & m, int branches) {
    DepthRunner & r = depth[branches - 1];
    if (r.n_branch != branches || r.steps.empty()) r.init(m, branches);
    return r;
}

std::unique_ptr<VocoderStream> GenPool::take_voc(BreezeModel & m) {
    std::unique_ptr<VocoderStream> v;
    if (!voc.empty()) {
        v = std::move(voc.back());
        voc.pop_back();
        v->reset();
    } else {
        v = std::make_unique<VocoderStream>();
        v->init(m);
    }
    return v;
}

void GenPool::free() {
    depth[0].free();
    depth[1].free();
    voc.clear();
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

SpeechJob::SpeechJob(BreezeModel & model, GenPool & p, const GenRequest & r, const std::string & t, ChunkRef reference,
                     uint32_t seed, AudioCallback callback, GenTimings & timings,
                     std::chrono::steady_clock::time_point start_time, PlayClock & play)
    : m(model), pool(p), req(r), text(t), ref(std::move(reference)), cb(std::move(callback)), tm(timings),
      t_start(start_time), clock(play), rng(seed) {}

// enough audio up front that the next chunk lands before it runs out, at the speed frames really take
static int first_chunk(const GenPool & pool, int rows, double frame_audio_ms) {
    const auto it = pool.step_ms.find(rows);
    if (it == pool.step_ms.end()) return 1;
    const double headroom = frame_audio_ms / it->second;
    return headroom >= 1.9 ? 1 : headroom >= 1.5 ? 2 : headroom >= 1.15 ? 4 : 8;
}

void SpeechJob::start(int other_rows) {
    const bool has_ref = !ref.codes.empty() && !ref.text.empty();
    use_cfg = req.cfg_scale != 1.0f;

    auto t0 = std::chrono::steady_clock::now();
    int total_c = 0, total_u = 0;
    std::vector<float> emb_c = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref.codes, ref.n_frames, true), total_c);
    std::vector<float> emb_u;
    if (use_cfg) emb_u = assemble(m, build_segments(m, req, text, has_ref, ref.text, ref.codes, ref.n_frames, false), total_u);
    tm.prompt += ms_since(t0);

    max_new = req.max_new_tokens > 0 ? req.max_new_tokens : m.cfg.max_new_tokens;
    st_c.init(m, total_c + max_new + 8);
    if (use_cfg) st_u.init(m, total_u + max_new + 8);

    t0 = std::chrono::steady_clock::now();
    o_c = backbone_run(m, st_c, emb_c, total_c);
    if (use_cfg) o_u = backbone_run(m, st_u, emb_u, total_u);
    tm.prefill += ms_since(t0);

    bp.temperature = req.temperature > 0.0f ? req.temperature : m.cfg.temperature;
    bp.top_k = req.top_k > 0 ? req.top_k : m.cfg.top_k;
    bp.top_p = req.top_p > 0.0f ? req.top_p : m.cfg.top_p;
    bp.repetition_penalty = req.repetition_penalty > 0.0f ? req.repetition_penalty : m.cfg.repetition_penalty;
    for (int t = m.cfg.codec_codebook_size; t < m.cfg.audio_vocab_size; t++) suppress.push_back(t);

    std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
    cb0 = sample_token(comb, bp, rng, &hist, &suppress);
    eos = cb0 == m.cfg.backbone_eos_token_id;

    // the first flush is small so audio starts early, then it grows to keep the vocoder efficient
    chunk_max = std::max(1, req.chunk_max);
    frame_audio_ms = 1000.0 * m.cfg.samples_per_frame / m.cfg.sample_rate;
    chunk = std::max(1, req.chunk_first);
    if (req.pace) chunk = std::max(chunk, first_chunk(pool, other_rows + branches(), frame_audio_ms));
    chunk = std::min(chunk, chunk_max);
    voc = pool.take_voc(m);
}

// a flush has to start while the listener still has enough left to cover decoding it
bool SpeechJob::due_early(int pending) const {
    if (!req.pace || !clock.started || clock.voc_ms <= 0) return false;
    // generating and decoding it has to take well under what it plays, or small flushes spiral
    if (pending * clock.frame_ms + 1.2 * clock.voc_ms * pending >= pending * frame_audio_ms) return false;
    const double left = std::chrono::duration<double, std::milli>(clock.runs_out - std::chrono::steady_clock::now()).count();
    return left < 1.25 * (clock.frame_ms + clock.voc_ms * (pending + 1)) + 100.0;
}

bool SpeechJob::flush(bool final_flush) {
    const int nc = m.cfg.num_codebooks;
    const int have = (int) frames.size() / nc;
    const auto ema = [](double & avg, double v) { avg = avg > 0 ? avg * 0.7 + v * 0.3 : v; };
    while (have > emitted) {
        const bool full = final_flush || have - emitted >= chunk;
        const bool early = !full && due_early(have - emitted);
        if (!full && !early) break;
        const int start = emitted;
        const int count = full && !final_flush ? chunk : have - emitted;
        std::vector<int> sub(frames.begin() + (size_t) start * nc, frames.begin() + (size_t) (start + count) * nc);
        const auto tv = std::chrono::steady_clock::now();
        std::vector<float> audio = voc->decode(sub, count);
        const double vtime = ms_since(tv);
        tm.vocoder += vtime;
        voc_step_ms += vtime;
        ema(clock.voc_ms, vtime / count);
        tm.flushes++;
        if (!tm.first_audio) {
            tm.first_vocoder = vtime;
            tm.first_frames = count;
            tm.first_audio = ms_since(t_start);
        }
        if (!cb(audio.data(), count * m.cfg.samples_per_frame)) return false;
        const auto now = std::chrono::steady_clock::now();
        if (!clock.started || clock.runs_out < now) clock.runs_out = now;
        clock.runs_out += std::chrono::microseconds((long long) (count * frame_audio_ms * 1000.0));
        clock.started = true;
        emitted += count;
        // an early flush means the buffer is thin, so the chunk holds its size until it recovers
        if (!early) chunk = std::min(chunk + chunk / 3 + 1, chunk_max);
    }
    return true;
}

void SpeechJob::after_depth(const std::vector<int> & depth_codes) {
    frame.clear();
    frame.push_back(cb0);
    frame.insert(frame.end(), depth_codes.begin(), depth_codes.end());

    bool pad = true;
    for (int c : frame) if (c != m.cfg.codebook_pad_token_id) { pad = false; break; }
    if (!pad) {
        frames.insert(frames.end(), frame.begin(), frame.end());
        tm.frames++;
        if (!flush(false)) { stopped = true; return; }
    }
    hist.push_back(cb0);
}

void SpeechJob::after_backbone(StepOut c, StepOut u) {
    o_c = std::move(c);
    if (use_cfg) o_u = std::move(u);
    std::vector<float> comb = combine_logits(o_c.logits, o_u.logits, use_cfg, req.cfg_scale);
    cb0 = sample_token(comb, bp, rng, &hist, &suppress);
    eos = cb0 == m.cfg.backbone_eos_token_id;
    step++;
}

void SpeechJob::abandon() {
    st_c.free();
    st_u.free();
    if (voc) pool.voc.push_back(std::move(voc));
}

void SpeechJob::finish() {
    if (!stopped) stopped = !flush(true);
    abandon();
    out.codes = std::move(frames);
    out.n_frames = (int) out.codes.size() / m.cfg.num_codebooks;
    out.text = text;
}

void step_jobs(BreezeModel & m, GenPool & pool, const std::vector<SpeechJob *> & jobs) {
    const auto t_step = std::chrono::steady_clock::now();
    int step_rows = 0;
    for (SpeechJob * j : jobs) {
        j->voc_step_ms = 0;
        step_rows += j->branches();
    }
    const SampleParams sp = DepthRunner::default_params(m);
    const int vs = m.cfg.audio_vocab_size;
    const int top_k = sp.top_k > 0 && sp.top_k < vs ? sp.top_k : vs;
    const auto td = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> codes;
    if (DepthRunner::can_fuse(m, sp)) {
        std::vector<DepthItem> items(jobs.size());
        for (size_t i = 0; i < jobs.size(); i++) {
            items[i].hidden[0] = &jobs[i]->o_c.hidden;
            items[i].hidden[1] = jobs[i]->use_cfg ? &jobs[i]->o_u.hidden : nullptr;
            items[i].cb0 = jobs[i]->cb0;
            items[i].cfg_scale = jobs[i]->req.cfg_scale;
            items[i].temperature = sp.temperature;
            items[i].rng = &jobs[i]->rng;
        }
        codes = pool.runner(m, 1).run_batch(m, items, top_k);
    } else {
        for (SpeechJob * j : jobs) {
            std::vector<std::vector<float>> hiddens = { j->o_c.hidden };
            if (j->use_cfg) hiddens.push_back(j->o_u.hidden);
            codes.push_back(pool.runner(m, j->branches()).run(m, hiddens, j->cb0, j->req.cfg_scale, j->rng));
        }
    }
    const double depth_ms = ms_since(td);
    for (size_t i = 0; i < jobs.size(); i++) {
        jobs[i]->tm.depth += depth_ms;
        jobs[i]->after_depth(codes[i]);
    }

    std::vector<SpeechJob *> live;
    std::vector<int> all_frames;
    for (SpeechJob * j : jobs) {
        if (j->stopped) continue;
        live.push_back(j);
        all_frames.insert(all_frames.end(), j->frame.begin(), j->frame.end());
    }
    if (live.empty()) return;

    const auto tb = std::chrono::steady_clock::now();
    const int hs = m.cfg.hidden_size;
    std::vector<float> ae = audio_embed_forward(m, all_frames, (int) live.size());
    std::vector<BackboneState *> states;
    std::vector<float> rows;
    for (size_t i = 0; i < live.size(); i++) {
        const auto row = ae.begin() + (ptrdiff_t) i * hs;
        states.push_back(&live[i]->st_c);
        rows.insert(rows.end(), row, row + hs);
        if (live[i]->use_cfg) {
            states.push_back(&live[i]->st_u);
            rows.insert(rows.end(), row, row + hs);
        }
    }
    std::vector<StepOut> outs;
    if (states.size() == 1) outs.push_back(backbone_run(m, *states[0], rows, 1));
    else outs = backbone_run_batch(m, states, rows);
    const double backbone_ms = ms_since(tb);

    size_t r = 0;
    for (SpeechJob * j : live) {
        StepOut c = std::move(outs[r++]);
        StepOut u;
        if (j->use_cfg) u = std::move(outs[r++]);
        j->tm.backbone += backbone_ms;
        j->after_backbone(std::move(c), std::move(u));
    }

    const double wall = ms_since(t_step);
    const auto ema = [](double & avg, double v) { avg = avg > 0 ? avg * 0.7 + v * 0.3 : v; };
    for (SpeechJob * j : live) ema(j->clock.frame_ms, wall - j->voc_step_ms);
    ema(pool.step_ms[step_rows], wall);
}

void GenSession::begin(BreezeModel & m, MimiCodec & codec, const GenRequest & req, GenTimings * tm,
                       GenScheduler * sched) {
    m_model = &m;
    m_sched = sched;
    m_codec = &codec;
    m_req = req;
    m_piece = 0;
    m_start = std::chrono::steady_clock::now();
    m_clock = PlayClock();
    m_codes.clear();
    m_frames = 0;
    m_text.clear();

    if (!req.ref_codes.empty() && req.ref_frames > 0 && !req.ref_text.empty()) {
        m_codes = req.ref_codes;
        m_frames = req.ref_frames;
        m_text = req.ref_text;
    } else if (!req.ref_audio.empty() && !req.ref_text.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto encode = [&] { m_codes = codec.encode(req.ref_audio, m_frames); };
        if (m_sched) m_sched->exec(encode);
        else encode();
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

    SpeechJob job(*m_model, m_sched ? m_sched->pool() : m_pool, m_req, text, std::move(anchor),
                  (uint32_t) m_req.seed + m_piece, cb, t, m_start, m_clock);
    bool ok;
    if (m_sched) {
        ok = m_sched->run(job);
    } else {
        job.start();
        const std::vector<SpeechJob *> one = { &job };
        while (job.wants_frame()) step_jobs(*m_model, m_pool, one);
        job.finish();
        ok = !job.stopped;
    }
    m_piece++;
    // the opening piece stands in as the reference when there was no clip to clone
    if (ok && m_codes.empty() && job.out.n_frames > 0) {
        m_codes = std::move(job.out.codes);
        m_frames = job.out.n_frames;
        m_text = job.out.text;
    }
    return ok;
}

// long text is generated piece by piece, each one carrying the same reference so the voice does
// not change at the seams
void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings, GenScheduler * sched) {
    GenTimings sink;
    GenTimings & tm = timings ? *timings : sink;

    GenSession s;
    s.begin(m, codec, req, &tm, sched);

    // a half minute of reference makes the model skip whole sentences of whatever comes next, so
    // when the first piece has to double as the reference it stays near the usual clip length
    const int anchor_chars = 200;
    const std::vector<std::string> parts =
        split_text(req.text, req.split_chars, s.needs_anchor() ? anchor_chars : 0);

    for (const std::string & part : parts)
        if (!s.speak(part, cb, &tm)) return;
}

const char * warm_up_text() {
    return "Warming up the model with a sentence about as long as a normal request would be, so that the text "
           "encoder and the backbone have already allocated what they need before anybody is waiting. This second "
           "sentence is only here to take it up to the length of a typical paragraph of speech.";
}

void warm_up(BreezeModel & m, MimiCodec & codec, const GenRequest & base) {
    GenRequest req = base;
    req.text = warm_up_text();
    req.instruction = GenRequest().instruction;
    req.max_new_tokens = 24;
    req.pace = false;
    const AudioCallback drop = [](const float *, int) { return true; };
    generate(m, codec, req, drop);
    req.cfg_scale = 2.0f;
    req.max_new_tokens = 8;
    generate(m, codec, req, drop);
    // vulkan picks matmul pipelines by size class, so touch the spread of chunk lengths a stream uses
    const int nc = m.cfg.num_codebooks;
    VocoderStream voc;
    voc.init(m);
    for (int T : { 1, 2, 4, 8, 16, 32, 64, std::max(1, base.chunk_max) }) {
        std::vector<int> codes((size_t) T * nc, 0);
        voc.decode(codes, T);
    }
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

    // the vocoder upsamples 1920x, so decoding a long clip in one graph asks for gigabytes at once
    const int step = 40;
    VocoderStream voc;
    voc.init(m);
    std::vector<float> audio;
    audio.reserve((size_t) src_T * m.cfg.samples_per_frame);
    for (int start = 0; start < src_T; start += step) {
        const int count = std::min(step, src_T - start);
        std::vector<int> sub(out.begin() + (size_t) start * nc, out.begin() + (size_t) (start + count) * nc);
        std::vector<float> part = voc.decode(sub, count);
        audio.insert(audio.end(), part.begin(), part.end());
    }
    return audio;
}

}
