#pragma once

#include "breeze/backbone.h"
#include "breeze/codec.h"
#include "breeze/depth_decoder.h"
#include "breeze/model.h"

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace breeze {

struct GenRequest {
    std::string text;
    std::string instruction = "Speak clearly and naturally.";
    std::string ref_text;
    std::vector<float> ref_audio; // mono 24 kHz, empty for voice design
    // already encoded reference. takes priority over ref_audio so a repeated voice skips the codec,
    // which is most of the wait before the first audio comes out
    std::vector<int> ref_codes;
    int ref_frames = 0;
    float cfg_scale = 1.0f;
    int seed = 42;
    // sampling. zero on any of these keeps whatever the gguf was built with
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 0.0f;
    float repetition_penalty = 0.0f;
    int max_new_tokens = 0; // 0 uses the model default
    // frames per streamed chunk, ramping from first to max. set both the same for a fixed size.
    // first is a floor, a busy gpu needs more audio up front to stay ahead of the listener
    int chunk_first = 1;
    int chunk_max = 25;
    // the model loses track of the text past about a minute of audio, 0 keeps it in one pass
    int split_chars = 600;
    // send a short chunk early when the listener is about to run dry, pointless when writing a file
    bool pace = true;
};

// when the listener's audio runs out, assuming playback starts the moment a chunk lands
struct PlayClock {
    bool started = false;
    std::chrono::steady_clock::time_point runs_out;
    double frame_ms = 0; // wall time per frame, whatever else shared the gpu included
    double voc_ms = 0;   // vocoder per decoded frame
};

// break text on sentence boundaries into pieces worth roughly budget characters, cjk counted heavier.
// first_budget sizes the opening piece only, 0 to treat it like the rest
std::vector<std::string> split_text(const std::string & text, int budget, int first_budget = 0);

// rough spoken length in seconds, only good enough to drive a progress bar
double estimate_seconds(const std::string & text);

// how faithfully the acoustic codes are resynthesised. codec language models are known to go
// unstable on random sampling, so conversion leans much harder on the likeliest codes than tts does
struct ConvertOptions {
    std::string src_text;   // transcript of the source, empty generates filler
    float temperature = 0.3f;
    int top_k = 1;
    float cfg_scale = 1.0f;
    // acoustic codebooks taken straight from the source instead of being regenerated. pitch lives in
    // the low ones, so this trades voice identity back for the source's melody and intonation
    int keep_acoustic = 0;
    bool feed_source = false;
    int seed = 42;
    // already encoded reference, used instead of the ref_audio argument when set
    std::vector<int> ref_codes;
    int ref_frames = 0;
};

// respeak already encoded audio in the reference voice. words and frame timing come from the source,
// pitch and timbre from the reference, so intonation is not carried over.
std::vector<float> convert_voice(BreezeModel & m, MimiCodec & codec, const std::vector<int> & src_codes,
                                 int src_T, const std::vector<float> & ref_audio,
                                 const std::string & ref_text, const ConvertOptions & opt = {});

// called with each decoded audio chunk; return false to stop generation early
using AudioCallback = std::function<bool(const float * samples, int n)>;

// per stage wall clock milliseconds, optionally filled by generate
struct GenTimings {
    double encode_ref = 0, prompt = 0, prefill = 0, backbone = 0, depth = 0, vocoder = 0;
    double first_audio = 0, first_vocoder = 0;
    int frames = 0, flushes = 0, first_frames = 0;
};

class GenScheduler;

// a stream runs inline when no scheduler is passed
void generate(BreezeModel & m, MimiCodec & codec, const GenRequest & req, const AudioCallback & cb,
              GenTimings * timings = nullptr, GenScheduler * sched = nullptr);

// long enough that a normal request doesnt grow the graph allocator again
const char * warm_up_text();

// runs a throwaway generation so the first real request doesnt pay for pipeline setup
void warm_up(BreezeModel & m, MimiCodec & codec, const GenRequest & base);

// what a finished piece leaves behind so the next one can keep the same voice
struct ChunkRef {
    std::vector<int> codes;
    int n_frames = 0;
    std::string text;
};

// gpu objects worth keeping between pieces
struct GenPool {
    DepthRunner depth[2];
    std::vector<std::unique_ptr<VocoderStream>> voc;
    std::map<int, double> step_ms; // recent wall time of a frame, by how many rows shared it

    DepthRunner & runner(BreezeModel & m, int branches);
    std::unique_ptr<VocoderStream> take_voc(BreezeModel & m);
    void free();
};

// one piece of text being generated, stepped from outside so several can share a batch
struct SpeechJob {
    SpeechJob(BreezeModel & m, GenPool & pool, const GenRequest & req, const std::string & text, ChunkRef ref,
              uint32_t seed, AudioCallback cb, GenTimings & tm,
              std::chrono::steady_clock::time_point t_start, PlayClock & clock);

    // other_rows is what the gpu is already busy with, it decides how big the first chunk has to be
    void start(int other_rows = 0);
    bool wants_frame() const { return !stopped && !eos && step < max_new; }
    void finish();
    // gives the gpu objects back without flushing, for after an error
    void abandon();

    int branches() const { return use_cfg ? 2 : 1; }
    void after_depth(const std::vector<int> & depth_codes);
    void after_backbone(StepOut c, StepOut u);

    BreezeModel & m;
    GenPool & pool;
    const GenRequest & req;
    std::string text;
    ChunkRef ref;
    AudioCallback cb;
    GenTimings & tm;
    std::chrono::steady_clock::time_point t_start;
    PlayClock & clock;
    ChunkRef out;

    std::mt19937 rng;
    bool use_cfg = false;
    bool stopped = false;
    bool eos = false;
    int step = 0;
    int max_new = 0;
    BackboneState st_c, st_u;
    StepOut o_c, o_u;
    SampleParams bp;
    std::vector<int> suppress;
    std::vector<int> hist;
    int cb0 = 0;
    std::vector<int> frame; // the one the backbone reads next
    double voc_step_ms = 0; // this job's vocoder time in the current frame

    std::vector<int> frames;
    int emitted = 0;
    int chunk = 1;
    int chunk_max = 1;
    double frame_audio_ms = 0;
    std::unique_ptr<VocoderStream> voc;

private:
    bool flush(bool final_flush);
    bool due_early(int pending) const;
};

// one frame for every job, in as few gpu passes as they can share
void step_jobs(BreezeModel & m, GenPool & pool, const std::vector<SpeechJob *> & jobs);

// text that arrives a bit at a time, the reference and play clock carry over between pieces
class GenSession {
public:
    void begin(BreezeModel & m, MimiCodec & codec, const GenRequest & req, GenTimings * tm = nullptr,
               GenScheduler * sched = nullptr);

    // speaks one piece. false means the callback asked to stop
    bool speak(const std::string & text, const AudioCallback & cb, GenTimings * tm = nullptr);

    void set_instruction(const std::string & s) { m_req.instruction = s; }

    // with no clip to clone the first piece becomes the reference, so it wants to stay short
    bool needs_anchor() const { return m_codes.empty(); }

private:
    BreezeModel * m_model = nullptr;
    MimiCodec * m_codec = nullptr;
    GenRequest m_req;
    std::vector<int> m_codes;
    std::string m_text;
    int m_frames = 0;
    uint32_t m_piece = 0;
    std::chrono::steady_clock::time_point m_start;
    PlayClock m_clock;
    GenScheduler * m_sched = nullptr;
    GenPool m_pool;
};

}
