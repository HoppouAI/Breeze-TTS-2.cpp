#include "breeze/model.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace breeze {

bool BreezeModel::load(const std::string & path, bool prefer_gpu, bool want_te_cpu) {
    backend.init(prefer_gpu);
    const char * te = std::getenv("BREEZE_TE_CPU");
    if (te && *te && strcmp(te, "0") != 0) want_te_cpu = true;
    te_cpu = backend.is_gpu && want_te_cpu;
    if (te_cpu) {
        cpu.init(false);
        const char * nt = std::getenv("BREEZE_TE_THREADS");
        int n = nt ? std::atoi(nt) : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
        ggml_backend_cpu_set_n_threads(cpu.backend, std::max(1, n));
        printf("text encoder: cpu, %d threads\n", std::max(1, n));
    }
    if (!gg.load(path, backend, te_cpu ? &cpu : nullptr, te_cpu ? "te." : nullptr)) return false;
    cfg = parse_config(gg);
    if (!tok.load(gg)) return false;
    return true;
}

void BreezeModel::free() {
    gg.free();
    if (te_cpu) cpu.free();
    backend.free();
}

}
