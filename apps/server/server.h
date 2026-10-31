#pragma once

#include <string>

namespace breeze {

struct ServerOptions {
    std::string model;
    std::string host = "127.0.0.1";
    int port = 8080;
    bool use_gpu = true;
    bool te_cpu = false;
    bool webui = false;
    bool verbose = false;
    std::string voices_dir = "voices";
    int ws_port = 0; // 0 puts it on port + 1, negative turns it off
    int chunk_first = 1;
    int chunk_max = 25;
    int split_chars = 600; // 0 keeps long text in a single pass
    int parallel = 1; // streams generating at once, more than this get told the server is busy
    bool warmup = true;
    std::string cors; // empty turns cors off, "*" allows any origin, otherwise a comma separated allowlist
};

int run_server(const ServerOptions & opts);

}
