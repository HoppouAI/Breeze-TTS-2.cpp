#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"
#include "breeze/voice.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace breeze;

static const char * arg(int argc, char ** argv, int & i, const char * name) {
    if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", name);
        return nullptr;
    }
    return argv[++i];
}

static void usage() {
    printf("usage: breeze-cli <model.gguf> --text <text> [options]\n"
           "  --instruction <s>   voice description or direction\n"
           "  --ref-audio <wav>   reference audio for voice clone/direction\n"
           "  --ref-text <s>      exact transcript of the reference audio\n"
           "  --voice <name>      use a saved voice instead of --ref-audio\n"
           "  --save-voice <name> encode --ref-audio and save it as a reusable voice, then exit\n"
           "  --description <s>   optional note stored with --save-voice\n"
           "  --list-voices       print the saved voices and exit\n"
           "  --voices-dir <path> where saved voices live (default voices)\n"
           "  --cfg-scale <f>     classifier free guidance scale (default 1.0)\n"
           "  --seed <n>          random seed (default 42)\n"
           "  --temp <f>          sampling temperature, 0 keeps the model default\n"
           "  --top-k <n>         sampling top-k, 0 keeps the model default\n"
           "  --top-p <f>         sampling top-p, 0 keeps the model default\n"
           "  --rep-penalty <f>   repetition penalty, 0 keeps the model default\n"
           "  --max-new <n>       max frames to generate\n"
           "  --output <wav>      output path (default output.wav)\n"
           "  --chunk-first <n>   frames in the first streamed chunk (default 4)\n"
           "  --chunk-max <n>     frames the chunk ramps up to (default 25)\n"
           "  --timings           print a stage by stage latency breakdown\n"
           "  --cpu               force CPU backend\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string model_path = argv[1];
    GenRequest req;
    req.instruction = "Speak clearly and naturally.";
    std::string ref_audio_path, output = "output.wav";
    std::string voice_name, save_voice_name, voice_description, voices_dir = "voices";
    bool list_voices = false;
    bool use_gpu = true;
    bool show_timings = false;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--text") req.text = arg(argc, argv, i, "--text");
        else if (a == "--instruction") req.instruction = arg(argc, argv, i, "--instruction");
        else if (a == "--ref-audio") ref_audio_path = arg(argc, argv, i, "--ref-audio");
        else if (a == "--ref-text") req.ref_text = arg(argc, argv, i, "--ref-text");
        else if (a == "--voice") voice_name = arg(argc, argv, i, "--voice");
        else if (a == "--save-voice") save_voice_name = arg(argc, argv, i, "--save-voice");
        else if (a == "--description") voice_description = arg(argc, argv, i, "--description");
        else if (a == "--list-voices") list_voices = true;
        else if (a == "--voices-dir") voices_dir = arg(argc, argv, i, "--voices-dir");
        else if (a == "--cfg-scale") req.cfg_scale = (float) atof(arg(argc, argv, i, "--cfg-scale"));
        else if (a == "--seed") req.seed = atoi(arg(argc, argv, i, "--seed"));
        else if (a == "--temp") req.temperature = (float) atof(arg(argc, argv, i, "--temp"));
        else if (a == "--top-k") req.top_k = atoi(arg(argc, argv, i, "--top-k"));
        else if (a == "--top-p") req.top_p = (float) atof(arg(argc, argv, i, "--top-p"));
        else if (a == "--rep-penalty") req.repetition_penalty = (float) atof(arg(argc, argv, i, "--rep-penalty"));
        else if (a == "--max-new") req.max_new_tokens = atoi(arg(argc, argv, i, "--max-new"));
        else if (a == "--split-chars") req.split_chars = atoi(arg(argc, argv, i, "--split-chars"));
        else if (a == "--chunk-first") req.chunk_first = atoi(arg(argc, argv, i, "--chunk-first"));
        else if (a == "--chunk-max") req.chunk_max = atoi(arg(argc, argv, i, "--chunk-max"));
        else if (a == "--output") output = arg(argc, argv, i, "--output");
        else if (a == "--timings") show_timings = true;
        else if (a == "--cpu") use_gpu = false;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    if (list_voices) {
        const std::vector<Voice> found = load_voice_dir(voices_dir);
        if (found.empty()) { printf("no voices in %s\n", voices_dir.c_str()); return 0; }
        for (const Voice & v : found) {
            printf("%-24s %6.2f s  %s\n", v.name.c_str(),
                   (double) v.frames * 1920 / v.sample_rate, v.text.c_str());
            if (!v.description.empty()) printf("%-24s           %s\n", "", v.description.c_str());
        }
        return 0;
    }

    Voice loaded;
    if (!voice_name.empty()) {
        if (!load_voice(voices_dir + "/" + voice_name + ".breeze", loaded)) {
            fprintf(stderr, "no voice called %s in %s\n", voice_name.c_str(), voices_dir.c_str());
            return 1;
        }
        req.ref_codes = loaded.codes;
        req.ref_frames = loaded.frames;
        if (req.ref_text.empty()) req.ref_text = loaded.text;
    }

    if (req.text.empty() && save_voice_name.empty()) { fprintf(stderr, "--text is required\n"); return 1; }
    if (!save_voice_name.empty()) {
        if (!valid_voice_name(save_voice_name)) {
            fprintf(stderr, "voice names can only use letters, digits, dash and underscore\n");
            return 1;
        }
        if (ref_audio_path.empty() || req.ref_text.empty()) {
            fprintf(stderr, "--save-voice needs --ref-audio and --ref-text\n");
            return 1;
        }
        if (voice_description.size() > max_voice_description) {
            fprintf(stderr, "--description is limited to %d bytes\n", (int) max_voice_description);
            return 1;
        }
    }

    BreezeModel model;
    printf("loading %s ...\n", model_path.c_str());
    if (!model.load(model_path, use_gpu)) { fprintf(stderr, "failed to load model\n"); return 1; }
    printf("backend: %s, sample rate: %d\n", model.backend.name(), model.cfg.sample_rate);

    if (!ref_audio_path.empty()) {
        if (!read_wav(ref_audio_path, model.cfg.sample_rate, req.ref_audio)) {
            fprintf(stderr, "failed to read reference audio\n");
            return 1;
        }
    }

    MimiCodec codec;
    codec.init(model);

    if (!save_voice_name.empty()) {
        Voice v;
        v.name = save_voice_name;
        v.text = req.ref_text;
        v.description = voice_description;
        v.sample_rate = model.cfg.sample_rate;
        v.n_codebooks = model.cfg.num_codebooks;
        v.codes = codec.encode(req.ref_audio, v.frames);
        const std::string path = voices_dir + "/" + save_voice_name + ".breeze";
        if (!save_voice(path, v)) { fprintf(stderr, "failed to write %s\n", path.c_str()); return 1; }
        printf("wrote %s (%d frames, %.2f s)\n", path.c_str(), v.frames,
               (double) v.frames * model.cfg.samples_per_frame / v.sample_rate);
        model.free();
        return 0;
    }

    req.pace = false;
    std::vector<float> audio;
    int frames = 0;
    GenTimings tm;
    generate(model, codec, req, [&](const float * s, int n) {
        audio.insert(audio.end(), s, s + n);
        frames += n;
        printf("\rgenerated %.2f s", (float) frames / model.cfg.sample_rate);
        fflush(stdout);
        return true;
    }, &tm);
    printf("\n");

    if (show_timings) {
        const double secs = (double) audio.size() / model.cfg.sample_rate;
        printf("time to first audio %.0f ms over %d flushes\n", tm.first_audio, tm.flushes);
        printf("  reference encode  %8.1f ms\n", tm.encode_ref);
        printf("  prompt build      %8.1f ms\n", tm.prompt);
        printf("  backbone prefill  %8.1f ms\n", tm.prefill);
        printf("  first vocoder     %8.1f ms  (%d frames)\n", tm.first_vocoder, tm.first_frames);
        printf("  backbone decode   %8.1f ms  (%.2f ms/frame)\n", tm.backbone, tm.backbone / tm.frames);
        printf("  depth decode      %8.1f ms  (%.2f ms/frame)\n", tm.depth, tm.depth / tm.frames);
        printf("  vocoder           %8.1f ms  (%.2f ms/frame)\n", tm.vocoder, tm.vocoder / tm.frames);
        printf("  %d frames, %.2f s audio\n", tm.frames, secs);
    }

    if (!write_wav(output, audio, model.cfg.sample_rate)) { fprintf(stderr, "failed to write %s\n", output.c_str()); return 1; }
    printf("wrote %s (%.2f s)\n", output.c_str(), (float) audio.size() / model.cfg.sample_rate);
    model.free();
    return 0;
}
