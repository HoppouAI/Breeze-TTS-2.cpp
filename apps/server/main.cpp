#include "server.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace breeze;

int main(int argc, char ** argv) {
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        printf("usage: breeze-server <model.gguf> [--host H] [--port P] [--webui] [--cpu]\n");
        printf("                     [--chunk-first N] [--chunk-max N] [--verbose]\n");
        printf("                     [--voices-dir PATH] [--ws-port P] [--split-chars N]\n");
        printf("\n");
        printf("  --chunk-first  frames in the first streamed chunk, lower starts sooner (default 4)\n");
        printf("  --chunk-max    frames the chunk ramps up to, higher is more efficient (default 25)\n");
        printf("                 set both the same to stream a fixed chunk size\n");
        printf("  --split-chars  default length long text is broken up at (default 600), 0 sends the\n");
        printf("                 whole thing through in one pass. a request can still override it\n");
        printf("  --verbose      add a per stage timing breakdown to each request\n");
        printf("  --voices-dir   folder of saved .breeze voices to load at startup (default voices)\n");
        printf("  --ws-port      websocket port for streaming sessions, default is the http port + 1,\n");
        printf("                 -1 turns it off\n");
        return argc < 2 ? 1 : 0;
    }
    ServerOptions opts;
    opts.model = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--host" && i + 1 < argc) opts.host = argv[++i];
        else if (a == "--port" && i + 1 < argc) opts.port = atoi(argv[++i]);
        else if (a == "--webui") opts.webui = true;
        else if (a == "--verbose" || a == "-v") opts.verbose = true;
        else if (a == "--voices-dir" && i + 1 < argc) opts.voices_dir = argv[++i];
        else if (a == "--ws-port" && i + 1 < argc) opts.ws_port = atoi(argv[++i]);
        else if (a == "--cpu") opts.use_gpu = false;
        else if (a == "--chunk-first" && i + 1 < argc) opts.chunk_first = atoi(argv[++i]);
        else if (a == "--chunk-max" && i + 1 < argc) opts.chunk_max = atoi(argv[++i]);
        else if (a == "--split-chars" && i + 1 < argc) opts.split_chars = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (opts.split_chars < 0) opts.split_chars = 0;
    if (opts.chunk_first < 1 || opts.chunk_max < 1) {
        fprintf(stderr, "chunk sizes must be at least 1\n");
        return 1;
    }
    return run_server(opts);
}
