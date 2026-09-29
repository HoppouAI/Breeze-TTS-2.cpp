#include "server.h"
#include "encode.h"
#include "openai.h"
#include "voices.h"
#include "ws.h"
#include "ws_api.h"

#include "breeze/audio.h"
#include "breeze/generation.h"
#include "breeze/model.h"

#include "httplib.h"
#include "breeze_webui_assets.h"

#ifdef _WIN32
#include <windows.h> // after httplib, it pulls in winsock2 first
#endif

#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace breeze {

static std::string mmss(double s) {
    if (!(s >= 0) || s > 86399) s = 0;
    char buf[16];
    snprintf(buf, sizeof buf, "%02d:%02d", (int) s / 60, (int) s % 60);
    return buf;
}

static std::string bar(double frac, int width) {
    static const char * part[] = { " ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589" };
    if (!(frac > 0)) frac = 0;
    if (frac > 1) frac = 1;
    const double filled = frac * width;
    const int full = (int) filled;
    std::string s;
    for (int i = 0; i < full; i++) s += "\u2588";
    if (full < width) {
        s += part[(int) ((filled - full) * 8)];
        for (int i = full + 1; i < width; i++) s += " ";
    }
    return s;
}

// an origin thats not on the allowlist gets an empty string, so no header goes out
static std::string cors_origin(const std::string & allowed, const httplib::Request & req) {
    if (allowed == "*") return "*";
    const std::string origin = req.get_header_value("Origin");
    if (origin.empty()) return "";
    size_t start = 0;
    while (start <= allowed.size()) {
        size_t end = allowed.find(',', start);
        if (end == std::string::npos) end = allowed.size();
        if (allowed.compare(start, end - start, origin) == 0) return origin;
        start = end + 1;
    }
    return "";
}

// one generation with the console progress bar, emit gets each decoded chunk and false stops it
static void run_generation(BreezeModel & model, MimiCodec & codec, const GenRequest & g, int sr, bool verbose,
                           const AudioCallback & emit) {
    GenTimings tm;
    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    // the model decides when to stop, so the total is only ever an estimate
    const double est = estimate_seconds(g.text);
    size_t total = 0;
    try {
        generate(model, codec, g, [&](const float * s, int n) {
            if (!emit(s, n)) return false;
            total += (size_t) n;
            const double secs = (double) total / sr, wall = elapsed();
            const double rate = wall > 0 ? secs / wall : 0;
            const double frac = est > 0 ? secs / est : 0;
            const double eta = rate > 0 ? (est > secs ? (est - secs) / rate : 0) : 0;
            printf("\r%3.0f%%|%s| %.1f/%.1fs [%s<%s, %.1f fps, %.2fx]  ",
                   (frac < 1 ? frac : 1) * 100, bar(frac, 24).c_str(), secs, est,
                   mmss(wall).c_str(), mmss(eta).c_str(), secs * 12.5 / wall, rate);
            fflush(stdout);
            return true;
        }, &tm);
    } catch (const std::exception & e) {
        fprintf(stderr, "\ngeneration error: %s\n", e.what());
    }
    const double secs = (double) total / sr, wall = elapsed();
    printf("\r%3.0f%%|%s| %.1f/%.1fs [%s, %.1f fps, %.2fx]        \n",
           100.0, bar(1, 24).c_str(), secs, secs, mmss(wall).c_str(),
           wall > 0 ? secs * 12.5 / wall : 0.0, wall > 0 ? secs / wall : 0.0);
    printf("     %d frames in %d flushes, first audio %.0f ms\n",
           tm.frames, tm.flushes, tm.first_audio);
    if (verbose && tm.frames > 0) {
        printf("     ref %.0f  prompt %.0f  prefill %.0f ms | per frame: backbone %.2f"
               "  depth %.2f  vocoder %.2f ms\n",
               tm.encode_ref, tm.prompt, tm.prefill, tm.backbone / tm.frames,
               tm.depth / tm.frames, tm.vocoder / tm.frames);
    }
    fflush(stdout);
}

int run_server(const ServerOptions & opts) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // otherwise the bar glyphs and any chinese text come out as mojibake
#endif
    BreezeModel model;
    printf("loading %s ...\n", opts.model.c_str());
    if (!model.load(opts.model, opts.use_gpu)) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    printf("backend: %s, sample rate: %d\n", model.backend.name(), model.cfg.sample_rate);
    MimiCodec codec;
    codec.init(model);

    httplib::Server svr;
    auto mutex = std::make_shared<std::mutex>();
    const int sr = model.cfg.sample_rate;

    VoiceStore store;
    store.load_dir(opts.voices_dir, model.cfg.num_codebooks);
    store.add_routes(svr, model, codec, *mutex, opts.voices_dir);

    WsServer ws;
    const int ws_port = opts.ws_port == 0 ? opts.port + 1 : opts.ws_port;
    if (ws_port > 0) {
        const bool up = ws.start(opts.host, ws_port, [&](WsConn & c) {
            ws_connection(c, model, codec, store, *mutex, opts.chunk_first, opts.chunk_max, opts.split_chars);
        });
        if (up) printf("websocket on ws://%s:%d\n", opts.host.c_str(), ws_port);
        else fprintf(stderr, "could not open the websocket port %d\n", ws_port);
    }

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\",\"sample_rate\":" + std::to_string(sr) +
                        ",\"ws_port\":" + std::to_string(ws_port > 0 ? ws_port : 0) + "}",
                        "application/json");
    });

    // openai clients post json to the same path, the form fields keep the native behaviour
    const auto speak_openai = [&](const httplib::Request & req, httplib::Response & res,
                                  std::shared_ptr<std::unique_lock<std::mutex>> lock) {
        OpenAiSpeech o;
        int status = 200;
        const std::string err = parse_openai_speech(req.body, store, opts.split_chars, o, status);
        if (!err.empty()) {
            res.status = status;
            res.set_content(openai_error(err), "application/json");
            return;
        }
        o.g.chunk_first = opts.chunk_first;
        o.g.chunk_max = opts.chunk_max;
        std::string sent, ctype;
        std::shared_ptr<AudioEncoder> enc = make_encoder(o.format, sr, sent, ctype);

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("Cache-Control", "no-store");
        if (o.sse) ctype = "text/event-stream";

        const char * mode = o.g.ref_frames > 0 ? (o.g.instruction != GenRequest().instruction ? "direction" : "clone")
                                               : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d, openai %s%s%s\n", mode, (int) o.g.text.size(),
               o.g.cfg_scale, o.g.seed, sent.c_str(), sent != o.format ? " for " : "",
               sent != o.format ? o.format.c_str() : "");
        fflush(stdout);

        res.set_chunked_content_provider(
            ctype, [&model, &codec, g = o.g, sse = o.sse, enc, lock, sr, verbose = opts.verbose](
                       size_t, httplib::DataSink & sink) {
                std::string buf;
                const auto flush = [&] {
                    if (buf.empty()) return true;
                    const std::string out = sse ? sse_audio_delta(buf) : buf;
                    buf.clear();
                    return sink.write(out.data(), out.size());
                };
                enc->begin(buf);
                if (flush()) {
                    bool alive = true;
                    run_generation(model, codec, g, sr, verbose, [&](const float * s, int n) {
                        enc->write(s, n, buf);
                        return alive = flush();
                    });
                    if (alive) {
                        enc->finish(buf);
                        if (flush() && sse) {
                            const std::string done = sse_audio_done();
                            sink.write(done.data(), done.size());
                        }
                    }
                }
                sink.done();
                return true;
            });
    };

    svr.Post("/v1/audio/speech", [&, mutex](const httplib::Request & req, httplib::Response & res) {
        const bool json = req.get_header_value("Content-Type").find("json") != std::string::npos;
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(*mutex, std::try_to_lock);
        if (!*lock) {
            res.status = 409;
            res.set_content(json ? openai_error("busy, another generation is running", "server_error")
                                 : "{\"error\":\"busy\"}",
                            "application/json");
            return;
        }
        if (json) {
            speak_openai(req, res, lock);
            return;
        }
        GenRequest g;
        g.text = field(req, "text", "");
        g.instruction = field(req, "instruction", "Speak clearly and naturally.");
        g.ref_text = field(req, "ref_text", "");
        g.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
        g.seed = atoi(field(req, "seed", "42").c_str());
        g.temperature = (float) atof(field(req, "temperature", "0").c_str());
        g.top_k = atoi(field(req, "top_k", "0").c_str());
        g.top_p = (float) atof(field(req, "top_p", "0").c_str());
        g.repetition_penalty = (float) atof(field(req, "repetition_penalty", "0").c_str());
        g.max_new_tokens = atoi(field(req, "max_new_tokens", "0").c_str());
        g.split_chars = atoi(field(req, "split_chars", std::to_string(opts.split_chars)).c_str());
        g.chunk_first = opts.chunk_first;
        g.chunk_max = opts.chunk_max;
        if (req.has_file("ref_audio")) {
            const auto & f = req.get_file_value("ref_audio");
            if (!f.content.empty())
                read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, g.ref_audio);
        }
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty() && !store.take(vid, g.ref_codes, g.ref_frames, g.ref_text)) {
            res.status = 404;
            res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
            return;
        }
        if (g.text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"text is required\"}", "application/json");
            return;
        }

        res.set_header("X-Sample-Rate", std::to_string(sr));
        res.set_header("X-Sample-Format", "s16le");
        res.set_header("Cache-Control", "no-store");

        const bool sent_ins = req.has_file("instruction") || req.has_param("instruction");
        const char * mode = !g.ref_audio.empty() ? (sent_ins ? "direction" : "clone") : "design";
        printf("gen  %s, %d chars, cfg %.1f, seed %d\n", mode, (int) g.text.size(), g.cfg_scale, g.seed);
        fflush(stdout);

        res.set_chunked_content_provider(
            "audio/pcm",
            [&model, &codec, g, lock, sr, verbose = opts.verbose](size_t, httplib::DataSink & sink) {
                run_generation(model, codec, g, sr, verbose, [&](const float * s, int n) {
                    const std::vector<uint8_t> pcm = to_pcm16(s, n);
                    return sink.write((const char *) pcm.data(), pcm.size());
                });
                sink.done();
                return true;
            });
    });

    svr.Post("/v1/audio/convert", [&, mutex](const httplib::Request & req, httplib::Response & res) {
        std::unique_lock<std::mutex> lock(*mutex, std::try_to_lock);
        if (!lock) {
            res.status = 409;
            res.set_content("{\"error\":\"busy\"}", "application/json");
            return;
        }
        std::vector<float> src, ref;
        const auto load = [&](const char * name, std::vector<float> & out) {
            if (!req.has_file(name)) return false;
            const auto & f = req.get_file_value(name);
            return !f.content.empty() &&
                   read_wav_buffer((const uint8_t *) f.content.data(), f.content.size(), sr, out);
        };
        if (!load("source", src)) {
            res.status = 400;
            res.set_content("{\"error\":\"a source wav file is required\"}", "application/json");
            return;
        }
        std::string ref_text = field(req, "ref_text", "");
        std::vector<int> vcodes;
        int vframes = 0;
        const std::string vid = field(req, "voice_id", "");
        if (!vid.empty()) {
            if (!store.take(vid, vcodes, vframes, ref_text)) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown voice_id\"}", "application/json");
                return;
            }
        } else if (!load("ref_audio", ref)) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_audio or voice_id is required\"}", "application/json");
            return;
        }
        if (ref_text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"ref_text is required\"}", "application/json");
            return;
        }

        printf("conv %.2f s source, %s\n", (double) src.size() / sr,
               vframes > 0 ? "cached reference" : "uploaded reference");
        fflush(stdout);
        try {
            int T = 0;
            std::vector<int> codes = codec.encode(src, T);
            ConvertOptions copt;
            copt.src_text = field(req, "text", "");
            copt.temperature = (float) atof(field(req, "temperature", "0.3").c_str());
            copt.top_k = atoi(field(req, "top_k", "1").c_str());
            copt.cfg_scale = (float) atof(field(req, "cfg_scale", "1.0").c_str());
            copt.keep_acoustic = atoi(field(req, "keep_acoustic", "0").c_str());
            copt.seed = atoi(field(req, "seed", "42").c_str());
            copt.ref_codes = vcodes;
            copt.ref_frames = vframes;
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> audio = convert_voice(model, codec, codes, T, ref, ref_text, copt);
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double secs = (double) audio.size() / sr;
            printf("     %.2f s in %.2f s, %.2fx rt\n", secs, wall, wall > 0 ? secs / wall : 0.0);
            fflush(stdout);
            std::vector<uint8_t> pcm = to_pcm16(audio.data(), (int) audio.size());
            res.set_header("X-Sample-Rate", std::to_string(sr));
            res.set_header("X-Sample-Format", "s16le");
            res.set_content((const char *) pcm.data(), pcm.size(), "audio/pcm");
        } catch (const std::exception & e) {
            fprintf(stderr, "conversion error: %s\n", e.what());
            res.status = 500;
            res.set_content("{\"error\":\"conversion failed\"}", "application/json");
        }
    });

    if (opts.webui) {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::index_html, "text/html");
        });
        svr.Get("/style.css", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::style_css, "text/css");
        });
        svr.Get("/app.js", [](const httplib::Request &, httplib::Response & res) {
            res.set_content(breeze_webui::app_js, "application/javascript");
        });
        printf("web ui: http://%s:%d/\n", opts.host.c_str(), opts.port);
    }

    if (!opts.cors.empty()) {
        // post routing hits every response, errors and the chunked stream included
        svr.set_post_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
            const std::string origin = cors_origin(opts.cors, req);
            if (origin.empty()) return;
            res.set_header("Access-Control-Allow-Origin", origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Sample-Format");
            if (origin != "*") res.set_header("Vary", "Origin");
        });
        // an origin off the list still gets a 204, just without the header, so the browser refuses it
        svr.Options(".*", [](const httplib::Request & req, httplib::Response & res) {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            res.set_header("Access-Control-Max-Age", "86400");
            const std::string want = req.get_header_value("Access-Control-Request-Headers");
            if (!want.empty()) res.set_header("Access-Control-Allow-Headers", want);
            res.status = 204;
        });
        printf("cors: %s\n", opts.cors == "*" ? "any origin" : opts.cors.c_str());
    }

    printf("listening on http://%s:%d\n", opts.host.c_str(), opts.port);
    if (!svr.listen(opts.host, opts.port)) {
        fprintf(stderr, "failed to bind %s:%d\n", opts.host.c_str(), opts.port);
        return 1;
    }
    model.free();
    return 0;
}

}
